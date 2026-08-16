#!/usr/bin/env python3
"""Backfill daily S&P 500 MA200 breadth from local constituent prices.

The existing ``breadth_daily.csv`` remains authoritative from its first date
onward.  This script reconstructs the earlier gap using point-in-time S&P 500
membership and the unadjusted ``Close`` histories under
``SPY/stock_prices/prices``.  A one-year Yahoo cache is used only to warm the
200-session moving averages at the beginning of 1996.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SPY_DATA_DIR = ROOT / "SPY" / "stock_prices"
PRICE_DIR = SPY_DATA_DIR / "prices"
WARMUP_DIR = SPY_DATA_DIR / "ma200_warmup_prices"
MEMBERSHIP_FILE = SPY_DATA_DIR / "membership_snapshots.csv"
WARMUP_STATUS_FILE = SPY_DATA_DIR / "ma200_warmup_status.csv"
MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

BREADTH_FILE = ROOT / "breadth_daily.csv"
BACKUP_FILE = ROOT / "breadth_daily_pre_spy_backfill.csv"
RECONSTRUCTION_FILE = ROOT / "breadth_spy_reconstruction.csv"
VALIDATION_FILE = ROOT / "breadth_backfill_validation.csv"
SUMMARY_FILE = ROOT / "breadth_backfill_validation_summary.json"
S5TH_FILE = ROOT / "S5TH.csv"

START_DATE = pd.Timestamp("1996-01-01")
WARMUP_START = pd.Timestamp("1995-01-01")
MA_WINDOW = 200
SOURCE_NAME = "SPY-constituents-MA200"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-membership", action="store_true")
    parser.add_argument("--refresh-warmup", action="store_true")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--no-update", action="store_true")
    return parser.parse_args()


def yfinance_symbol(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def load_membership(refresh: bool = False) -> pd.DataFrame:
    if refresh or not MEMBERSHIP_FILE.exists():
        with urlopen(MEMBERSHIP_URL, timeout=60) as response:  # noqa: S310
            MEMBERSHIP_FILE.write_bytes(response.read())
    membership = pd.read_csv(MEMBERSHIP_FILE, parse_dates=["date"])
    if membership.empty or membership["date"].duplicated().any():
        raise ValueError("Membership snapshots are empty or contain duplicate dates")
    return membership.sort_values("date").reset_index(drop=True)


def warmup_candidates() -> list[str]:
    """Return locally available symbols with observations at the 1996 boundary."""
    candidates = []
    for path in sorted(PRICE_DIR.glob("*.csv")):
        try:
            first = pd.read_csv(path, usecols=["Date"], nrows=1)["Date"].iloc[0]
        except (OSError, ValueError, IndexError, pd.errors.EmptyDataError):
            continue
        if pd.Timestamp(first) <= START_DATE + pd.Timedelta(days=7):
            candidates.append(path.stem)
    return candidates


def _ticker_frame(download: pd.DataFrame, symbol: str, batch_size: int) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if isinstance(download.columns, pd.MultiIndex):
        if symbol not in download.columns.get_level_values(0):
            return pd.DataFrame()
        frame = download[symbol].copy()
    elif batch_size == 1:
        frame = download.copy()
    else:
        return pd.DataFrame()
    frame = frame.dropna(how="all")
    if frame.empty or "Close" not in frame:
        return pd.DataFrame()
    return frame[["Close"]].reset_index()


def download_warmup(refresh: bool = False, batch_size: int = 40) -> pd.DataFrame:
    """Cache the 1995 prices needed for valid MA200 values in early 1996."""
    import yfinance as yf

    WARMUP_DIR.mkdir(parents=True, exist_ok=True)
    candidates = warmup_candidates()
    pending = [
        symbol
        for symbol in candidates
        if refresh or not (WARMUP_DIR / f"{symbol}.csv").exists()
    ]
    status_rows: list[dict[str, object]] = []

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        print(
            f"Warm-up batch {offset // batch_size + 1}/"
            f"{(len(pending) + batch_size - 1) // batch_size}: {len(batch)} symbols",
            flush=True,
        )
        try:
            downloaded = yf.download(
                batch,
                start=WARMUP_START.date().isoformat(),
                end=(START_DATE + pd.Timedelta(days=2)).date().isoformat(),
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                threads=True,
                progress=False,
                timeout=30,
            )
            batch_error = ""
        except Exception as exc:  # yfinance exposes mixed transport exceptions
            downloaded = pd.DataFrame()
            batch_error = f"{type(exc).__name__}: {exc}"

        for symbol in batch:
            frame = _ticker_frame(downloaded, symbol, len(batch))
            if frame.empty:
                status_rows.append(
                    {"ticker": symbol, "status": "unavailable", "rows": 0,
                     "error": batch_error or "No warm-up data returned"}
                )
                continue
            frame.to_csv(WARMUP_DIR / f"{symbol}.csv", index=False)
            status_rows.append(
                {"ticker": symbol, "status": "downloaded", "rows": len(frame),
                 "error": ""}
            )

    existing = []
    for symbol in candidates:
        path = WARMUP_DIR / f"{symbol}.csv"
        if path.exists():
            existing.append(
                {"ticker": symbol, "status": "existing", "rows": len(pd.read_csv(path)),
                 "error": ""}
            )
    status = pd.DataFrame(status_rows + existing).drop_duplicates("ticker", keep="first")
    if not status.empty:
        status.sort_values("ticker").to_csv(WARMUP_STATUS_FILE, index=False)
    return status


def load_reference() -> pd.DataFrame:
    frame = pd.read_csv(BREADTH_FILE)
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    if frame["Date"].duplicated().any():
        raise ValueError("breadth_daily.csv contains duplicate dates")
    return frame.sort_values("Date").reset_index(drop=True)


def market_dates(reference: pd.DataFrame) -> pd.DatetimeIndex:
    anchor = pd.read_csv(PRICE_DIR / "AAPL.csv", usecols=["Date"], parse_dates=["Date"])
    last_date = reference["Date"].max()
    dates = pd.DatetimeIndex(anchor["Date"])
    return dates[(dates >= START_DATE) & (dates <= last_date)]


def load_price_ratios(dates: pd.DatetimeIndex) -> pd.DataFrame:
    series = []
    for path in sorted(PRICE_DIR.glob("*.csv")):
        try:
            price = pd.read_csv(path, usecols=["Date", "Close"], parse_dates=["Date"])
        except (OSError, ValueError, pd.errors.EmptyDataError):
            continue
        warmup_path = WARMUP_DIR / path.name
        if warmup_path.exists():
            warmup = pd.read_csv(warmup_path, usecols=["Date", "Close"], parse_dates=["Date"])
            price = pd.concat([warmup, price], ignore_index=True)
        close = (
            price.dropna(subset=["Date", "Close"])
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .set_index("Date")["Close"]
        )
        close = close.where(close > 0)
        ratio = close / close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
        series.append(ratio.reindex(dates).rename(path.stem))
    if not series:
        raise ValueError(f"No usable price files found in {PRICE_DIR}")
    return pd.concat(series, axis=1)


def reconstruct_breadth(
    ratios: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    snapshot_dates = membership["date"].to_numpy(dtype="datetime64[ns]")
    rows = []
    for date, values in ratios.iterrows():
        position = np.searchsorted(snapshot_dates, np.datetime64(date), side="right") - 1
        if position < 0:
            continue
        snapshot = membership.iloc[position]
        members = [yfinance_symbol(ticker) for ticker in snapshot["tickers"].split(",")]
        available_symbols = [ticker for ticker in members if ticker in ratios.columns]
        available = values.reindex(available_symbols).dropna()
        count = len(available)
        rows.append(
            {
                "Date": date,
                "breadth": 100.0 * float((available > 1.0).mean()) if count else np.nan,
                "available_count": count,
                "constituent_count": len(members),
                "coverage": count / len(members),
                "membership_snapshot_date": snapshot["date"],
            }
        )
    return pd.DataFrame(rows)


def validation_rows(reconstructed: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    regular = reconstructed.merge(
        reference.rename(columns={"breadth": "reference_breadth", "source": "reference_source"}),
        on="Date",
        how="inner",
    )

    s5th = pd.read_csv(S5TH_FILE, encoding="utf-8-sig")
    s5th["Date"] = pd.to_datetime(s5th["Date"], format="%m/%d/%Y")
    s5th["reference_breadth"] = pd.to_numeric(
        s5th["Price"].astype(str).str.replace(",", "", regex=False)
    )
    first_reference = reference.loc[
        ~reference["source"].eq(SOURCE_NAME), "Date"
    ].min()
    sparse = reconstructed[reconstructed["Date"] < first_reference].merge(
        s5th[["Date", "reference_breadth"]], on="Date", how="inner"
    )
    sparse["reference_source"] = "S5TH-sparse-checkpoint"

    validation = pd.concat([regular, sparse], ignore_index=True)
    validation["error"] = validation["breadth"] - validation["reference_breadth"]
    return validation.sort_values(["reference_source", "Date"]).reset_index(drop=True)


def metrics(frame: pd.DataFrame) -> dict[str, object]:
    usable = frame.dropna(subset=["breadth", "reference_breadth"])
    error = usable["error"]
    return {
        "rows": int(len(usable)),
        "start": usable["Date"].min().date().isoformat() if len(usable) else None,
        "end": usable["Date"].max().date().isoformat() if len(usable) else None,
        "correlation": round(float(usable["breadth"].corr(usable["reference_breadth"])), 6),
        "mae_points": round(float(error.abs().mean()), 6),
        "rmse_points": round(float(np.sqrt((error**2).mean())), 6),
        "bias_points": round(float(error.mean()), 6),
        "median_coverage": round(float(usable["coverage"].median()), 6),
    }


def splice_backfill(reference: pd.DataFrame, reconstructed: pd.DataFrame) -> pd.DataFrame:
    authoritative = reference[~reference["source"].eq(SOURCE_NAME)].copy()
    first_reference = authoritative["Date"].min()
    gap = reconstructed[
        (reconstructed["Date"] >= START_DATE)
        & (reconstructed["Date"] < first_reference)
        & reconstructed["breadth"].notna()
    ][["Date", "breadth"]].copy()
    gap["breadth"] = gap["breadth"].round(2)
    gap["source"] = SOURCE_NAME
    return (
        pd.concat([gap, authoritative], ignore_index=True)
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def write_outputs(
    reference: pd.DataFrame,
    reconstructed: pd.DataFrame,
    validation: pd.DataFrame,
    update: bool,
) -> None:
    reconstruction_out = reconstructed.copy()
    reconstruction_out["breadth"] = reconstruction_out["breadth"].round(6)
    reconstruction_out["coverage"] = reconstruction_out["coverage"].round(6)
    reconstruction_out["Date"] = reconstruction_out["Date"].dt.strftime("%m/%d/%Y")
    reconstruction_out["membership_snapshot_date"] = reconstruction_out[
        "membership_snapshot_date"
    ].dt.strftime("%m/%d/%Y")
    reconstruction_out.to_csv(RECONSTRUCTION_FILE, index=False)

    validation_out = validation.copy()
    for column in ["breadth", "reference_breadth", "error", "coverage"]:
        validation_out[column] = validation_out[column].round(6)
    validation_out["Date"] = validation_out["Date"].dt.strftime("%m/%d/%Y")
    validation_out["membership_snapshot_date"] = validation_out[
        "membership_snapshot_date"
    ].dt.strftime("%m/%d/%Y")
    validation_out.to_csv(VALIDATION_FILE, index=False)

    actual = validation[validation["reference_source"].eq("S5TH")]
    mapped_proxy = validation[validation["reference_source"].eq("MMTH-mapped")]
    sparse = validation[validation["reference_source"].eq("S5TH-sparse-checkpoint")]
    combined = splice_backfill(reference, reconstructed)
    backfilled = combined[combined["source"].eq(SOURCE_NAME)]
    summary = {
        "method": "point-in-time S&P 500 members above unadjusted Close 200-session SMA",
        "ma_window_sessions": MA_WINDOW,
        "backfill": {
            "rows": int(len(backfilled)),
            "start": backfilled["Date"].min().date().isoformat(),
            "end": backfilled["Date"].max().date().isoformat(),
            "minimum_coverage": round(float(
                reconstructed.loc[
                    reconstructed["Date"].isin(backfilled["Date"]), "coverage"
                ].min()
            ), 6),
            "median_coverage": round(float(
                reconstructed.loc[
                    reconstructed["Date"].isin(backfilled["Date"]), "coverage"
                ].median()
            ), 6),
        },
        "validation": {
            "actual_s5th_2007_plus": metrics(actual),
            "existing_mmth_proxy_2002_2006": metrics(mapped_proxy),
            "sparse_s5th_gap_checkpoints": metrics(sparse),
        },
        "quality_gate": {
            "actual_s5th_correlation_at_least_0_98": bool(
                actual["breadth"].corr(actual["reference_breadth"]) >= 0.98
            ),
            "actual_s5th_mae_at_most_5_points": bool(actual["error"].abs().mean() <= 5),
            "sparse_gap_correlation_at_least_0_85": bool(
                sparse["breadth"].corr(sparse["reference_breadth"]) >= 0.85
            ),
            "sparse_gap_mae_at_most_10_points": bool(sparse["error"].abs().mean() <= 10),
        },
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n")

    if not all(summary["quality_gate"].values()):
        raise RuntimeError("Validation quality gate failed; breadth_daily.csv was not updated")
    if update:
        if not BACKUP_FILE.exists():
            shutil.copy2(BREADTH_FILE, BACKUP_FILE)
        output = combined.copy()
        output["Date"] = output["Date"].dt.strftime("%m/%d/%Y")
        output.to_csv(BREADTH_FILE, index=False)


def main() -> None:
    args = parse_args()
    membership = load_membership(args.refresh_membership)
    download_warmup(args.refresh_warmup, args.batch_size)
    reference = load_reference()
    dates = market_dates(reference)
    ratios = load_price_ratios(dates)
    reconstructed = reconstruct_breadth(ratios, membership)
    validation = validation_rows(reconstructed, reference)
    write_outputs(reference, reconstructed, validation, not args.no_update)
    summary = json.loads(SUMMARY_FILE.read_text())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
