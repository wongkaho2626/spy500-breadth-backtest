#!/usr/bin/env python3
"""Build historical SPY ticker manifests and download Yahoo price histories.

The SEC N-30D reports identify issuers but generally do not include exchange
tickers.  Report dates are taken from the locally downloaded SEC filings and
resolved to point-in-time S&P 500 ticker snapshots.  Prices are then requested
with yfinance, while unavailable/delisted symbols remain visible in the output
manifests.
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup


MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
REPORT_DATE_PATTERN = re.compile(
    r"SCHEDULE\s+OF\s+INVESTMENTS.{0,700}?"
    r"((?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|"
    r"OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},\s+\d{4})",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sec-dir", type=Path, default=Path("SPY"))
    parser.add_argument("--output-dir", type=Path, default=Path("stock_prices"))
    parser.add_argument("--start", default="1996-01-01")
    parser.add_argument(
        "--end",
        default=(date.today() + timedelta(days=1)).isoformat(),
        help="Exclusive yfinance end date (default: tomorrow)",
    )
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--manifests-only", action="store_true")
    return parser.parse_args()


def extract_report_date(path: Path) -> pd.Timestamp:
    raw = path.read_text(errors="ignore")
    if path.suffix.lower() in {".htm", ".html"}:
        text = BeautifulSoup(raw, "lxml").get_text("\n")
    else:
        text = re.sub(r"<[^>]+>", "\n", raw)
    match = REPORT_DATE_PATTERN.search(text)
    if not match:
        raise ValueError(f"Could not find Schedule of Investments date in {path}")
    normalized = re.sub(r"\s+", " ", match.group(1).replace("\xa0", " ")).strip()
    return pd.to_datetime(normalized, format="%B %d, %Y")


def discover_reports(sec_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in sec_dir.iterdir() if p.is_file()):
        filing_date = pd.to_datetime(path.name[:10], errors="coerce")
        if pd.isna(filing_date):
            continue
        rows.append(
            {
                "filing_date": filing_date,
                "report_date": extract_report_date(path),
                "source_filing": str(path),
            }
        )
    reports = pd.DataFrame(rows).sort_values(["report_date", "filing_date"])
    if reports.empty:
        raise ValueError(f"No dated SEC filings found in {sec_dir}")
    return reports.reset_index(drop=True)


def yfinance_symbol(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def build_holdings(
    reports: pd.DataFrame, membership: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    membership = membership.copy()
    membership["date"] = pd.to_datetime(membership["date"])
    membership = membership.sort_values("date").reset_index(drop=True)

    rows = []
    for report in reports.itertuples(index=False):
        position = membership["date"].searchsorted(report.report_date, side="right") - 1
        resolution = "on_or_before"
        if position < 0:
            position = 0
            resolution = "earliest_available"
        snapshot = membership.iloc[position]
        for ticker in str(snapshot["tickers"]).split(","):
            rows.append(
                {
                    "report_date": report.report_date,
                    "filing_date": report.filing_date,
                    "ticker": ticker,
                    "yfinance_ticker": yfinance_symbol(ticker),
                    "membership_snapshot_date": snapshot["date"],
                    "snapshot_resolution": resolution,
                    "source_filing": report.source_filing,
                }
            )

    holdings = pd.DataFrame(rows).drop_duplicates(["report_date", "ticker"])
    holdings = holdings.sort_values(["report_date", "ticker"]).reset_index(drop=True)

    report_stats = (
        holdings.groupby(["ticker", "yfinance_ticker"], as_index=False)
        .agg(
            first_report_date=("report_date", "min"),
            last_report_date=("report_date", "max"),
            reports_held=("report_date", "nunique"),
        )
    )

    period_membership = membership[
        membership["date"].between("1996-01-01", "2026-12-31")
    ]
    membership_stats: dict[str, dict[str, object]] = {}
    for snapshot in period_membership.itertuples(index=False):
        for ticker in str(snapshot.tickers).split(","):
            stats = membership_stats.setdefault(
                ticker,
                {
                    "ticker": ticker,
                    "yfinance_ticker": yfinance_symbol(ticker),
                    "first_membership_date": snapshot.date,
                    "last_membership_date": snapshot.date,
                    "membership_snapshots": 0,
                },
            )
            stats["last_membership_date"] = snapshot.date
            stats["membership_snapshots"] = int(stats["membership_snapshots"]) + 1

    tickers = pd.DataFrame(membership_stats.values()).merge(
        report_stats, on=["ticker", "yfinance_ticker"], how="left"
    )
    tickers["reports_held"] = tickers["reports_held"].fillna(0).astype(int)
    tickers = tickers.sort_values("ticker").reset_index(drop=True)
    return holdings, tickers


def read_existing_price(path: Path) -> dict[str, object] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(path, usecols=["Date"])
    except (OSError, ValueError):
        return None
    if frame.empty:
        return None
    return {
        "status": "existing",
        "rows": len(frame),
        "first_price_date": frame["Date"].iloc[0],
        "last_price_date": frame["Date"].iloc[-1],
        "output_file": str(path),
        "error": "",
    }


def ticker_frame(download: pd.DataFrame, symbol: str, batch_size: int) -> pd.DataFrame:
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
    if frame.empty:
        return frame
    frame.index.name = "Date"
    return frame.reset_index()


def download_prices(
    tickers: pd.DataFrame,
    output_dir: Path,
    start: str,
    end: str,
    batch_size: int,
    pause: float,
    refresh: bool,
) -> pd.DataFrame:
    prices_dir = output_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    symbols = tickers["yfinance_ticker"].drop_duplicates().tolist()
    statuses: dict[str, dict[str, object]] = {}
    pending = []
    for symbol in symbols:
        output_path = prices_dir / f"{symbol}.csv"
        existing = None if refresh else read_existing_price(output_path)
        if existing:
            statuses[symbol] = existing
        else:
            pending.append(symbol)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        print(
            f"Downloading batch {offset // batch_size + 1}/"
            f"{(len(pending) + batch_size - 1) // batch_size}: {len(batch)} symbols",
            flush=True,
        )
        try:
            downloaded = yf.download(
                batch,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=False,
                actions=True,
                threads=True,
                progress=False,
                timeout=30,
            )
            batch_error = ""
        except Exception as exc:  # yfinance raises mixed transport/data errors
            downloaded = pd.DataFrame()
            batch_error = f"{type(exc).__name__}: {exc}"

        for symbol in batch:
            frame = ticker_frame(downloaded, symbol, len(batch))
            output_path = prices_dir / f"{symbol}.csv"
            if frame.empty:
                statuses[symbol] = {
                    "status": "unavailable",
                    "rows": 0,
                    "first_price_date": "",
                    "last_price_date": "",
                    "output_file": "",
                    "error": batch_error or "No price data returned by Yahoo Finance",
                }
                continue
            frame.to_csv(output_path, index=False)
            statuses[symbol] = {
                "status": "downloaded",
                "rows": len(frame),
                "first_price_date": frame["Date"].iloc[0].date().isoformat(),
                "last_price_date": frame["Date"].iloc[-1].date().isoformat(),
                "output_file": str(output_path),
                "error": "",
            }
        if pause:
            time.sleep(pause)

    status = pd.DataFrame(
        [{"yfinance_ticker": symbol, **statuses[symbol]} for symbol in symbols]
    )
    return tickers.merge(status, on="yfinance_ticker", how="left")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = discover_reports(args.sec_dir)
    membership = pd.read_csv(MEMBERSHIP_URL)
    holdings, tickers = build_holdings(reports, membership)

    reports.to_csv(args.output_dir / "report_manifest.csv", index=False)
    holdings.to_csv(args.output_dir / "holdings_by_report.csv", index=False)
    tickers.to_csv(args.output_dir / "all_tickers_1996_2026.csv", index=False)
    (args.output_dir / "all_tickers_1996_2026.txt").write_text(
        "\n".join(tickers["ticker"]) + "\n"
    )
    print(
        f"Built {len(holdings):,} report holdings across {len(reports)} reports "
        f"and {len(tickers):,} unique tickers.",
        flush=True,
    )

    if args.manifests_only:
        return

    status = download_prices(
        tickers=tickers,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        batch_size=args.batch_size,
        pause=args.pause,
        refresh=args.refresh,
    )
    status.to_csv(args.output_dir / "download_status.csv", index=False)
    unavailable = status[status["status"] == "unavailable"]
    unavailable.to_csv(args.output_dir / "unavailable_tickers.csv", index=False)
    print(
        f"Price files available: {status['output_file'].ne('').sum():,}; "
        f"unavailable from Yahoo: {len(unavailable):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
