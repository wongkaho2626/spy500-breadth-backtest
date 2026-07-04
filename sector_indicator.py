"""
S&P 500 Sector Indicator

Loads the 11 SPDR sector ETFs (XLK, XLF, XLV, XLY, XLP, XLE, XLI, XLB, XLRE, XLU, XLC)
plus SPX.csv and reports:
  1. Trailing performance ranking per sector (1M/3M/6M/YTD/1Y) vs SPX
  2. Relative strength (sector / SPX, rebased to 100) over time — rotation chart
  3. Sector breadth — % of the 11 sectors trading above their own 200-day MA

Run `python fetch_sector_data.py` first to (re)download sector ETF history.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent
SPX_FILE = DATA_DIR / "SPX.csv"

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}

MA_WINDOW = 200
TRAILING_WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _load_price_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()
    return _parse_price(df["Price"])


def load_data() -> pd.DataFrame:
    prices = {"SPX": _load_price_csv(SPX_FILE)}
    for symbol in SECTOR_ETFS:
        prices[symbol] = _load_price_csv(DATA_DIR / f"{symbol}.csv")

    df = pd.DataFrame(prices).sort_index()
    df = df.dropna(how="all")
    return df


def trailing_returns_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, name in SECTOR_ETFS.items():
        series = df[symbol].dropna()
        if series.empty:
            continue
        row = {"Symbol": symbol, "Sector": name}
        for label, days in TRAILING_WINDOWS.items():
            if len(series) > days:
                row[label] = (series.iloc[-1] / series.iloc[-days - 1] - 1) * 100
            else:
                row[label] = float("nan")
        ytd_start = series[series.index.year == series.index[-1].year]
        row["YTD"] = (series.iloc[-1] / ytd_start.iloc[0] - 1) * 100 if not ytd_start.empty else float("nan")
        rows.append(row)

    spx = df["SPX"].dropna()
    spx_row = {"Symbol": "SPX", "Sector": "S&P 500 (benchmark)"}
    for label, days in TRAILING_WINDOWS.items():
        spx_row[label] = (spx.iloc[-1] / spx.iloc[-days - 1] - 1) * 100 if len(spx) > days else float("nan")
    ytd_start = spx[spx.index.year == spx.index[-1].year]
    spx_row["YTD"] = (spx.iloc[-1] / ytd_start.iloc[0] - 1) * 100 if not ytd_start.empty else float("nan")
    rows.append(spx_row)

    table = pd.DataFrame(rows).set_index("Symbol")
    return table.sort_values("3M", ascending=False)


def sector_breadth(df: pd.DataFrame) -> pd.Series:
    above_ma = pd.DataFrame(index=df.index)
    for symbol in SECTOR_ETFS:
        ma200 = df[symbol].rolling(MA_WINDOW).mean()
        above_ma[symbol] = df[symbol] > ma200
    # only count sectors that have live data yet (skip pre-inception ETFs, e.g. XLC/XLRE)
    has_data = df[list(SECTOR_ETFS)].notna()
    counts = has_data.sum(axis=1)
    valid = counts > 0
    breadth = above_ma.astype(float).where(has_data)[valid].sum(axis=1) / counts[valid] * 100
    return breadth


def relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    rs = pd.DataFrame(index=df.index)
    for symbol in SECTOR_ETFS:
        ratio = df[symbol] / df["SPX"]
        rs[symbol] = ratio / ratio.dropna().iloc[0] * 100
    return rs


def print_table(table: pd.DataFrame) -> None:
    print("\n=== Sector Trailing Performance (%) ===")
    fmt_cols = ["1M", "3M", "6M", "YTD", "1Y"]
    print(table[["Sector"] + fmt_cols].to_string(float_format=lambda x: f"{x:+.2f}"))


def plot_results(df: pd.DataFrame, rs: pd.DataFrame, breadth: pd.Series) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for symbol in SECTOR_ETFS:
        ax1.plot(rs.index, rs[symbol], label=symbol, linewidth=1.2)
    ax1.axhline(100, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Sector Relative Strength vs SPX (rebased to 100 at start)")
    ax1.legend(ncol=6, fontsize=8, loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(breadth.index, breadth, color="tab:blue", linewidth=1.2)
    ax2.axhline(50, color="black", linewidth=0.8, linestyle="--")
    ax2.fill_between(breadth.index, breadth, 50, where=(breadth >= 50), color="tab:green", alpha=0.2)
    ax2.fill_between(breadth.index, breadth, 50, where=(breadth < 50), color="tab:red", alpha=0.2)
    ax2.set_title(f"Sector Breadth — % of 11 sectors above their {MA_WINDOW}-day MA")
    ax2.set_ylim(0, 100)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.tight_layout()
    out_path = DATA_DIR / "sector_indicator.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nChart saved to {out_path}")


def main() -> None:
    df = load_data()
    table = trailing_returns_table(df)
    print_table(table)

    breadth = sector_breadth(df)
    rs = relative_strength(df)

    print(f"\nCurrent sector breadth: {breadth.iloc[-1]:.1f}% of sectors above {MA_WINDOW}-day MA")
    plot_results(df, rs, breadth)


if __name__ == "__main__":
    main()
