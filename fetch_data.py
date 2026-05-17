#!/usr/bin/env python3
"""
fetch_data.py — Update CSV data files with the latest prices from Yahoo Finance.

Usage:
    python fetch_data.py

Automatically updates (date,close format):
    SPY.csv, QQQ.csv, SOXX.csv, NASDAQ100.csv, S&P500.csv, Russell3000.csv

Automatically updates (Investing.com reverse-chronological format):
    S&P 500 Historical Data.csv
    CBOE Volatility Index Historical Data.csv

Manual download required from Investing.com:
    S&P 500 Stocks Above 200-Day Average Historical Data.csv
    S&P 500 Stocks Above 50-Day Average Historical Data.csv

Manual download required from other sources:
    ShillerPE.csv       — monthly CAPE (multpl.com)
    S&P500ForwardPE.csv — weekly forward PE
"""

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parent

# Ticker → filename for files stored as: date,close  (chronological)
SIMPLE_FILES = {
    "SPY":   "SPY.csv",
    "QQQ":   "QQQ.csv",
    "SOXX":  "SOXX.csv",
    "^NDX":  "NASDAQ100.csv",
    "^GSPC": "S&P500.csv",
    "^RUA":  "Russell3000.csv",
}

# Ticker → filename for Investing.com-style files (reverse-chronological, quoted fields)
INVESTING_FILES = {
    "^GSPC": "S&P 500 Historical Data.csv",
    "^VIX":  "CBOE Volatility Index Historical Data.csv",
}


def _download(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download adjusted close (and OHLC) from Yahoo Finance, flatten multi-index."""
    kwargs = dict(start=start, auto_adjust=True, progress=False)
    if end:
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    # yfinance may return MultiIndex columns like ("Close", "SPY"); flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def update_simple(ticker: str, fname: str) -> None:
    path = DATA_DIR / fname
    existing = pd.read_csv(path, parse_dates=["date"], index_col="date")
    last_date = existing.index.max()
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    new_data = _download(ticker, start)
    if new_data.empty:
        print(f"  {fname}: up to date ({last_date.date()})")
        return

    rows = new_data[["Close"]].rename(columns={"Close": "close"})
    rows.index.name = "date"
    rows.to_csv(path, mode="a", header=False)
    print(f"  {fname}: added {len(rows)} row(s), now through {rows.index.max().date()}")


def update_investing(ticker: str, fname: str) -> None:
    """Update a reverse-chronological Investing.com CSV (newest row first after header)."""
    path = DATA_DIR / fname

    with open(path, "r", encoding="utf-8-sig") as fh:
        lines = fh.readlines()

    # Line 1 (index 1) is the most recent data row
    recent_str = lines[1].split(",")[0].strip().strip('"')
    last_date = pd.to_datetime(recent_str, format="%m/%d/%Y")
    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    new_data = _download(ticker, start)
    if new_data.empty:
        print(f"  {fname}: up to date ({last_date.date()})")
        return

    # Also pull the day before the gap so we can compute the first change %.
    prev_data = _download(
        ticker,
        (last_date - timedelta(days=7)).strftime("%Y-%m-%d"),
        (last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    def _fmt(v: float, is_large: bool) -> str:
        return f'"{v:,.2f}"' if is_large else f'"{v:.2f}"'

    new_lines: list[str] = []
    for dt in sorted(new_data.index, reverse=True):
        row   = new_data.loc[dt]
        close = float(row["Close"])
        open_ = float(row["Open"])
        high  = float(row["High"])
        low   = float(row["Low"])

        prev_candidates = [d for d in prev_data.index if d < dt]
        if prev_candidates:
            prev_close = float(prev_data.loc[max(prev_candidates), "Close"])
            chg = (close - prev_close) / prev_close * 100
            chg_str = f'"{chg:+.2f}%"'
        else:
            chg_str = '""'

        is_large = close >= 100
        date_str = dt.strftime("%m/%d/%Y")
        line = (
            f'"{date_str}",'
            f'{_fmt(close, is_large)},'
            f'{_fmt(open_, is_large)},'
            f'{_fmt(high,  is_large)},'
            f'{_fmt(low,   is_large)},'
            f'"",{chg_str}\n'
        )
        new_lines.append(line)

    # Prepend new rows immediately after the header
    with open(path, "w", encoding="utf-8-sig") as fh:
        fh.writelines(lines[:1] + new_lines + lines[1:])

    latest = sorted(new_data.index)[-1].strftime("%m/%d/%Y")
    print(f"  {fname}: added {len(new_lines)} row(s), now through {latest}")


def main() -> None:
    print("Updating date,close files...")
    for ticker, fname in SIMPLE_FILES.items():
        try:
            update_simple(ticker, fname)
        except Exception as exc:
            print(f"  ERROR {fname}: {exc}", file=sys.stderr)

    print("\nUpdating Investing.com format files...")
    for ticker, fname in INVESTING_FILES.items():
        try:
            update_investing(ticker, fname)
        except Exception as exc:
            print(f"  ERROR {fname}: {exc}", file=sys.stderr)

    print("\nDone.")
    print(
        "\nManual download still required from Investing.com:\n"
        "  S&P 500 Stocks Above 200-Day Average Historical Data.csv\n"
        "  S&P 500 Stocks Above 50-Day Average Historical Data.csv\n"
        "  S&P500ForwardPE.csv\n"
        "  ShillerPE.csv (multpl.com)"
    )


if __name__ == "__main__":
    main()
