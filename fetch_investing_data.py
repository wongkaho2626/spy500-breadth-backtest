"""
Fetch latest historical data and update local CSV files.

Index prices come from Yahoo Finance (^NDX and ^GSPC). S5TH comes only
from Investing.com's published daily S5TH historical observations. Never
substitute breadth computed from today's constituent list for this series.
If the publisher blocks automated access, keep the existing data and report
the failure. An exported daily S5TH CSV can be imported with --s5th-csv.
Overlapping published rows replace previous values, including old estimates.

Instruments updated by fetch_all_updates(): all three, followed by rebuilding
breadth_daily.csv from the refreshed S5TH.csv.
Instruments updated by fetch_spy_updates(): SPX.csv + S5TH.csv, followed by the
same breadth_daily.csv rebuild.
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
from pathlib import Path

import pandas as pd
import yfinance as yf
from curl_cffi import requests
from curl_cffi.requests.exceptions import RequestException

from build_breadth_daily import build_breadth_daily

DATA_DIR = Path(__file__).parent

INSTRUMENTS = [
    {
        "name": "NASDAQ 100",
        "source": "yfinance",
        "ticker": "^NDX",
        "csv_file": DATA_DIR / "NASDAQ100.csv",
    },
    {
        "name": "S&P 500",
        "source": "yfinance",
        "ticker": "^GSPC",
        "csv_file": DATA_DIR / "SPX.csv",
    },
    {
        "name": "S&P 500 Above 200-Day MA",
        "source": "s5th",
        "url": "https://www.investing.com/indices/sp-500-stocks-above-200-day-average-historical-data",
        "csv_file": DATA_DIR / "S5TH.csv",
    },
]

# Subset used by spy_backtest.py (no NASDAQ 100)
SPY_INSTRUMENTS = [i for i in INSTRUMENTS if i["name"] != "NASDAQ 100"]

CSV_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]



def _read_existing(csv_file: Path) -> pd.DataFrame:
    if not csv_file.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(csv_file, encoding="utf-8-sig")  # utf-8-sig strips BOM
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df


def _merge_and_save(new_df: pd.DataFrame, existing: pd.DataFrame, csv_file: Path) -> None:
    if not existing.empty:
        existing = existing.copy()
        existing["Date"] = existing["Date"].dt.strftime("%m/%d/%Y")
        combined = pd.concat([new_df, existing], ignore_index=True)
    else:
        combined = new_df

    # Guard against duplicate dates: if a fetch re-returns the latest row (the
    # cutoff comparison can be off by a day), concat would append a duplicate,
    # and duplicate Date labels break reindex/.loc in every downstream backtest.
    # new_df is first, so keep="first" retains the freshly fetched row.
    combined = combined.drop_duplicates(subset="Date", keep="first")

    combined.to_csv(csv_file, index=False, quoting=1, encoding="utf-8-sig")


def _fmt_price(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_volume(value: float) -> str:
    """Mimic investing.com's volume style: 166.47M / 1.23B, empty when absent."""
    if pd.isna(value) or value <= 0:
        return ""
    for divisor, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if value >= divisor:
            return f"{value / divisor:.2f}{suffix}"
    return f"{value:.0f}"


def _fmt_change(pct: float) -> str:
    return "" if pd.isna(pct) else f"{pct:+.2f}%"


# ---------------------------------------------------------------------------
# Index prices (^NDX, ^GSPC)
# ---------------------------------------------------------------------------

def _fetch_yfinance_instrument(instrument: dict, verbose: bool) -> int:
    name = instrument["name"]
    ticker = instrument["ticker"]
    csv_file = instrument["csv_file"]

    existing = _read_existing(csv_file)
    has_dates = not existing.empty and "Date" in existing.columns and existing["Date"].notna().any()
    cutoff = existing["Date"].max() if has_dates else None

    if verbose:
        cutoff_str = cutoff.strftime("%m/%d/%Y") if cutoff is not None else "none"
        print(f"  {name}: latest in CSV = {cutoff_str}")

    try:
        if cutoff is not None:
            # Start a few days before the cutoff so pct_change has a prior
            # close for the first new row.
            start = (cutoff - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            hist = yf.download(ticker, start=start, auto_adjust=False, progress=False)
        else:
            hist = yf.download(ticker, period="max", auto_adjust=False, progress=False)
    except Exception as exc:
        print(f"  {name}: yfinance download failed ({exc}), skipping")
        return 0

    if hist is None or hist.empty:
        print(f"  {name}: yfinance returned no data, skipping")
        return 0

    # yf.download returns MultiIndex columns for a single ticker; flatten them.
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.sort_index()
    hist["ChangePct"] = hist["Close"].pct_change() * 100

    if cutoff is not None:
        hist = hist[hist.index > cutoff]
    if hist.empty:
        if verbose:
            print(f"  {name}: no new rows found")
        return 0

    rows = [
        {
            "Date": date.strftime("%m/%d/%Y"),
            "Price": _fmt_price(row["Close"]),
            "Open": _fmt_price(row["Open"]),
            "High": _fmt_price(row["High"]),
            "Low": _fmt_price(row["Low"]),
            "Vol.": _fmt_volume(row["Volume"]),
            "Change %": _fmt_change(row["ChangePct"]),
        }
        # Newest first, matching the historical CSV layout
        for date, row in hist.sort_index(ascending=False).iterrows()
    ]

    new_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    _merge_and_save(new_df, existing, csv_file)
    if verbose:
        print(f"  {name}: added {len(rows)} new row(s)")
    return len(rows)


# ---------------------------------------------------------------------------
# Actual published daily S5TH (no constituent-derived fallback)
# ---------------------------------------------------------------------------


def _validate_s5th_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the publisher's daily OHLC rows, rejecting corrupt input."""
    required = {"Date", "Price", "Open", "High", "Low"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("daily S5TH table must contain Date, Price, Open, High and Low")
    rows = frame.copy()
    dates = pd.to_datetime(rows["Date"], format="mixed", errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("daily S5TH dates must be present and unique")
    prices = rows[["Price", "Open", "High", "Low"]].apply(
        lambda column: pd.to_numeric(
            column.astype(str).str.replace(",", "", regex=False), errors="raise"
        )
    )
    if not ((prices >= 0) & (prices <= 100)).all().all():
        raise ValueError("S5TH OHLC values must be finite percentages between 0 and 100")
    if (prices["High"] < prices.max(axis=1)).any() or (
        prices["Low"] > prices.min(axis=1)
    ).any():
        raise ValueError("S5TH High/Low do not contain the published Open/Price")
    rows["Date"] = dates.dt.strftime("%m/%d/%Y")
    for column in prices:
        rows[column] = prices[column].map(_fmt_price)
    for column in ["Vol.", "Change %"]:
        if column not in rows:
            rows[column] = ""
        rows[column] = rows[column].fillna("").astype(str)
    return rows.loc[dates.sort_values(ascending=False).index, CSV_COLUMNS].reset_index(drop=True)


def _parse_s5th_html(html: str) -> pd.DataFrame:
    """Read the actual S5TH daily history table, not a quote or related index."""
    if not re.search(r"\bS5TH\b", html) or not re.search(r"\bDaily\b", html):
        raise ValueError("page does not identify S5TH daily historical data")
    for table in pd.read_html(io.StringIO(html)):
        if {"Date", "Price", "Open", "High", "Low"}.issubset(table.columns):
            return _validate_s5th_rows(table)
    raise ValueError("published daily S5TH history table not found")


def _save_s5th_rows(rows: pd.DataFrame, csv_file: Path, verbose: bool) -> int:
    existing = _read_existing(csv_file)
    old_dates = set(existing["Date"].dt.strftime("%m/%d/%Y")) if not existing.empty else set()
    new_count = len(set(rows["Date"]) - old_dates)
    # Refresh the entire available overlap: append-only would retain estimates
    # that earlier versions of this updater incorrectly saved as actual S5TH.
    _merge_and_save(rows, existing, csv_file)
    if verbose:
        print(f"  Actual daily S5TH: {new_count} new row(s); "
              f"{len(rows) - new_count} overlapping row(s) refreshed; "
              f"latest published date = {rows['Date'].iloc[0]}")
    return new_count


def _fetch_s5th_instrument(instrument: dict, verbose: bool) -> int:
    """Fetch published S5TH or explicitly retain the last saved observations."""
    try:
        response = requests.get(
            # Reuse yfinance's HTTP client and its matching browser TLS/headers.
            # A User-Agent header alone receives 403 from the publisher.
            instrument["url"], impersonate="chrome", timeout=30
        )
        response.raise_for_status()
        rows = _parse_s5th_html(response.text)
    except (RequestException, ValueError, KeyError) as exc:
        existing = _read_existing(instrument["csv_file"])
        latest = existing["Date"].max().strftime("%m/%d/%Y") if not existing.empty else "none"
        print(f"  WARNING: actual daily S5TH unavailable ({exc}). "
              f"Keeping saved data through {latest}; no calculated substitute. "
              "Import a published daily export with "
              "python3 fetch_investing_data.py --s5th-csv PATH.")
        return 0
    return _save_s5th_rows(rows, instrument["csv_file"], verbose)


def import_s5th_csv(csv_file: Path, verbose: bool = True) -> int:
    """Import an actual daily S5TH export obtained from the publisher."""
    rows = _validate_s5th_rows(pd.read_csv(csv_file, encoding="utf-8-sig"))
    return _save_s5th_rows(rows, INSTRUMENTS[2]["csv_file"], verbose)


def _publish_breadth_files() -> None:
    """Keep the website's static breadth files in sync with the Python input."""
    destination = DATA_DIR / "webapp" / "nextjs" / "public" / "data"
    if destination.is_dir():
        for name in ["S5TH.csv", "breadth_daily.csv"]:
            shutil.copyfile(DATA_DIR / name, destination / name)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_FETCHERS = {
    "yfinance": _fetch_yfinance_instrument,
    "s5th": _fetch_s5th_instrument,
}


def _fetch_instruments(instruments: list[dict], verbose: bool) -> None:
    total = 0
    for instrument in instruments:
        total += _FETCHERS[instrument["source"]](instrument, verbose)
    if verbose:
        print(f"Done. Total new rows added: {total}\n")


def fetch_all_updates(verbose: bool = True) -> None:
    if verbose:
        print("Fetching index prices from Yahoo Finance and actual daily S5TH from Investing.com...")
    _fetch_instruments(INSTRUMENTS, verbose)
    build_breadth_daily(verbose=verbose)
    _publish_breadth_files()


def fetch_spy_updates(verbose: bool = True) -> None:
    """Fetch SPX + S&P 500 breadth only (no NASDAQ 100)."""
    if verbose:
        print("Fetching S&P 500 prices from Yahoo Finance and actual daily S5TH from Investing.com...")
    _fetch_instruments(SPY_INSTRUMENTS, verbose)
    build_breadth_daily(verbose=verbose)
    _publish_breadth_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s5th-csv", type=Path, help="Import a published DAILY S5TH CSV export only")
    args = parser.parse_args()
    if args.s5th_csv:
        import_s5th_csv(args.s5th_csv)
        build_breadth_daily(verbose=True)
        _publish_breadth_files()
    else:
        fetch_all_updates(verbose=True)
