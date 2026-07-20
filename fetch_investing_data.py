"""
Fetch latest historical data and update local CSV files.

All data now comes from Yahoo Finance (yfinance):
  - NASDAQ100.csv  — ^NDX index prices
  - SPX.csv        — ^GSPC index prices
  - S5TH.csv       — % of S&P 500 stocks above their 200-day MA, computed
                     from the constituents' prices (the investing.com page
                     this used to scrape is now behind a Cloudflare
                     challenge; computed values match the scraped series to
                     within ~0.3 pts on overlap days)

The S&P 500 constituent list is read from Wikipedia. Note the S5TH update
needs an existing S5TH.csv — it extends the series incrementally and does
not rebuild deep history (a full rebuild would need 200 trading days of
constituent prices before every historical date).

Instruments updated by fetch_all_updates(): all three.
Instruments updated by fetch_spy_updates(): SPX.csv + S5TH.csv.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

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
        "source": "breadth-computed",
        "csv_file": DATA_DIR / "S5TH.csv",
    },
]

# Subset used by spy_backtest.py (no NASDAQ 100)
SPY_INSTRUMENTS = [i for i in INSTRUMENTS if i["name"] != "NASDAQ 100"]

CSV_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]

SP500_CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MA_WINDOW = 200
# Calendar days of price history to download so every new date has a full
# 200-trading-day window behind it (~1.5 trading days per calendar day + buffer).
BREADTH_LOOKBACK_DAYS = 400
# Skip days where fewer than this many constituents have a valid 200-day MA
# (guards against half-downloaded data producing a bogus percentage).
MIN_VALID_CONSTITUENTS = 400


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
# Computed breadth (% of S&P 500 above 200-day MA)
# ---------------------------------------------------------------------------

def _sp500_tickers() -> list[str]:
    """Current S&P 500 constituents from Wikipedia, in Yahoo symbol format."""
    html = requests.get(
        SP500_CONSTITUENTS_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    ).text
    symbols = pd.read_html(io.StringIO(html))[0]["Symbol"].tolist()
    # Yahoo uses "-" where the official symbol has "." (BRK.B -> BRK-B)
    return [s.replace(".", "-") for s in symbols]


def _fetch_breadth_instrument(instrument: dict, verbose: bool) -> int:
    name = instrument["name"]
    csv_file = instrument["csv_file"]

    existing = _read_existing(csv_file)
    has_dates = not existing.empty and "Date" in existing.columns and existing["Date"].notna().any()
    cutoff = existing["Date"].max() if has_dates else None

    if cutoff is None:
        print(f"  {name}: no existing {csv_file.name}; incremental breadth "
              "computation needs a seed series, skipping")
        return 0

    if verbose:
        print(f"  {name}: latest in CSV = {cutoff.strftime('%m/%d/%Y')}")

    try:
        tickers = _sp500_tickers()
    except Exception as exc:
        print(f"  {name}: failed to fetch constituent list ({exc}), skipping")
        return 0
    if len(tickers) < MIN_VALID_CONSTITUENTS:
        print(f"  {name}: constituent list looks wrong ({len(tickers)} tickers), skipping")
        return 0

    start = (cutoff - pd.Timedelta(days=BREADTH_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    try:
        data = yf.download(tickers, start=start, auto_adjust=False,
                           progress=False, threads=True)
    except Exception as exc:
        print(f"  {name}: yfinance download failed ({exc}), skipping")
        return 0
    if data is None or data.empty:
        print(f"  {name}: yfinance returned no data, skipping")
        return 0

    close = data["Close"].sort_index()
    ma = close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    valid = ma.notna() & close.notna()
    valid_counts = valid.sum(axis=1)
    pct = (close.gt(ma) & valid).sum(axis=1) / valid_counts * 100
    pct = pct[valid_counts >= MIN_VALID_CONSTITUENTS]
    change = pct.pct_change() * 100

    new_dates = pct.index[pct.index > cutoff]
    if len(new_dates) == 0:
        if verbose:
            print(f"  {name}: no new rows found")
        return 0

    # Only the closing value is computable (no intraday breadth path), so
    # Open/High/Low repeat it; downstream scripts read Price only.
    rows = [
        {
            "Date": date.strftime("%m/%d/%Y"),
            "Price": f"{pct[date]:.2f}",
            "Open": f"{pct[date]:.2f}",
            "High": f"{pct[date]:.2f}",
            "Low": f"{pct[date]:.2f}",
            "Vol.": "",
            "Change %": _fmt_change(change[date]),
        }
        for date in sorted(new_dates, reverse=True)
    ]

    new_df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    _merge_and_save(new_df, existing, csv_file)
    if verbose:
        print(f"  {name}: added {len(rows)} new row(s)")
    return len(rows)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_FETCHERS = {
    "yfinance": _fetch_yfinance_instrument,
    "breadth-computed": _fetch_breadth_instrument,
}


def _fetch_instruments(instruments: list[dict], verbose: bool) -> None:
    total = 0
    for instrument in instruments:
        total += _FETCHERS[instrument["source"]](instrument, verbose)
    if verbose:
        print(f"Done. Total new rows added: {total}\n")


def fetch_all_updates(verbose: bool = True) -> None:
    if verbose:
        print("Fetching latest data from Yahoo Finance...")
    _fetch_instruments(INSTRUMENTS, verbose)


def fetch_spy_updates(verbose: bool = True) -> None:
    """Fetch SPX + S&P 500 breadth only (no NASDAQ 100)."""
    if verbose:
        print("Fetching latest S&P 500 data from Yahoo Finance...")
    _fetch_instruments(SPY_INSTRUMENTS, verbose)


if __name__ == "__main__":
    fetch_all_updates(verbose=True)
