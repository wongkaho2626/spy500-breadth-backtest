"""
NDX Top-1 Breadth Strategy — buy the single #1 NDX holding each year.

Same buy/sell/divergence signals as qqq_backtest.py, but on BUY we
allocate 100% of capital to the current year's #1 NDX holding
(from nasdaq100_top_holdings.csv).  Annual rebalancing at each year start
during an open trade.  Compared against the plain QQQ index strategy.

BUY  (while OUT): breadth200 < 26%
                  AND at least 1 of 2 vote:
                    • VIX > 30  (fear spike / panic bottom)
                    • price > MA200  (uptrend pullback)
SELL (while IN):  Bearish divergence (unchanged)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR      = Path(__file__).parent
NDX_FILE      = DATA_DIR / "NASDAQ100.csv"
BREADTH_FILE  = DATA_DIR / "S5TH.csv"
# Continuous daily breadth (2002+) built by build_breadth_daily.py.
# S5TH.csv alone is only daily from 2007 — before that it is bimonthly, which
# corrupts row-based lookback windows (a "60-day" window spans ~10 years).
BREADTH_DAILY_FILE = DATA_DIR / "breadth_daily.csv"
BREADTH_DAILY_MIN  = "2007-01-01"  # fallback cutoff when daily file is absent
VIX_FILE      = DATA_DIR / "VIX.csv"
HOLDINGS_FILE = DATA_DIR / "NASDAQ100" / "nasdaq100_top_holdings.csv"
PRICES_DIR    = DATA_DIR / "NASDAQ100" / "stock_prices"

# ── Signals ───────────────────────────────────────────────────────────────────
BUY_B200_THRESH         = 26.0
VIX_BUY_THRESH          = 30.0   # VIX vote: fear spike
MA200_WINDOW            = 200     # MA200 vote: price above 200-day MA
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0
INITIAL_CAPITAL         = 10_000.0
COMMISSION              = 1.0
SLIPPAGE                = 0.0005

# ── Company name → ticker ─────────────────────────────────────────────────────
NAME_TO_TICKER = {
    "Cisco Systems Inc.":    "CSCO",
    "Microsoft Corporation": "MSFT",
    "Microsoft Corp.":       "MSFT",
    "Intel Corporation":     "INTC",
    "QUALCOMM Inc.":         "QCOM",
    "eBay Inc.":             "EBAY",
    "Apple Computer Inc.":   "AAPL",
    "Apple Inc.":            "AAPL",
    "Google Inc. Class A":   "GOOGL",
    "Google Inc. Class C":   "GOOGL",   # use GOOGL as proxy for Class C
    "Alphabet Inc.":         "GOOGL",
    "Amazon.com Inc.":       "AMZN",
    "NVIDIA Corp.":          "NVDA",
}


def _load_breadth() -> pd.DataFrame:
    """Prefer the continuous daily series (breadth_daily.csv, 2002+); S5TH.csv
    alone is bimonthly before 2007, which corrupts row-based windows."""
    if BREADTH_DAILY_FILE.exists():
        b200 = pd.read_csv(BREADTH_DAILY_FILE)
        b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
        b200.set_index("Date", inplace=True)
        return b200.rename(columns={"breadth": "Price"})
    b200 = pd.read_csv(BREADTH_FILE)
    b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
    b200.set_index("Date", inplace=True)
    b200["Price"] = _parse_price(b200["Price"])
    # S5TH is bimonthly before 2007 — drop the sparse era
    return b200[b200.index >= BREADTH_DAILY_MIN]


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fmt_vol(vol: float) -> str:
    if pd.isna(vol) or vol == 0:
        return ""
    if vol >= 1e9:
        return f"{vol/1e9:.2f}B"
    if vol >= 1e6:
        return f"{vol/1e6:.2f}M"
    return f"{vol/1e3:.2f}K"


def _update_investing_csv(path: Path, yf_ticker: str, yf) -> None:
    """Append new rows (newest-first) to an Investing.com-style CSV."""
    df_ex = pd.read_csv(path)
    df_ex["_date"] = pd.to_datetime(
        df_ex["Date"].astype(str).str.strip('"'), format="%m/%d/%Y"
    )
    last_date   = df_ex["_date"].max()
    fetch_start = last_date + pd.Timedelta(days=1)
    today       = pd.Timestamp.today().normalize()
    if fetch_start > today:
        print(f"[fetch] {path.name} already up to date ({last_date.date()}).")
        return

    print(f"[fetch] Fetching {yf_ticker} from {fetch_start.date()} …")
    raw = yf.Ticker(yf_ticker).history(start=fetch_start, auto_adjust=True)
    if raw.empty:
        print(f"[fetch] No new data for {yf_ticker}.")
        return

    rows = []
    for date, row in raw.sort_index(ascending=False).iterrows():
        rows.append(
            f'"{date.strftime("%m/%d/%Y")}"'
            f',"{row["Close"]:,.2f}"'
            f',"{row["Open"]:,.2f}"'
            f',"{row["High"]:,.2f}"'
            f',"{row["Low"]:,.2f}"'
            f',"{_fmt_vol(row["Volume"])}"'
            f',""'
        )

    existing = path.read_text(encoding="utf-8-sig")
    lines    = existing.splitlines(keepends=True)
    path.write_text(lines[0] + "\n".join(rows) + "\n" + "".join(lines[1:]), encoding="utf-8-sig")
    print(f"[fetch] {path.name}: added {len(rows)} row(s) → latest {raw.index.max().date()}.")


def _update_stock_price_csvs(yf) -> None:
    """Append new OHLCV rows to each stock CSV in PRICES_DIR."""
    csvs = [f for f in PRICES_DIR.glob("*.csv") if f.stem != "_download_summary"]
    if not csvs:
        return
    for path in sorted(csvs):
        ticker = path.stem
        df_ex = pd.read_csv(path, index_col=0)
        idx   = pd.to_datetime(df_ex.index, format="mixed", utc=True)
        last  = idx.max().normalize().replace(tzinfo=None)
        start  = last + pd.Timedelta(days=1)
        today  = pd.Timestamp.today().normalize()
        if start > today:
            continue
        raw = yf.Ticker(ticker).history(start=start, auto_adjust=True)
        if raw.empty:
            continue
        raw.index = raw.index.normalize()
        new_rows = raw[["Close", "High", "Low", "Open", "Volume"]].copy()
        new_rows.index.name = "Date"
        new_rows.to_csv(path, mode="a", header=False)
        print(f"[fetch] {ticker}.csv: added {len(new_rows)} row(s) → latest {new_rows.index.max().date()}.")


def fetch_latest_data() -> None:
    """Pull the latest market data into local CSVs via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        print("[fetch] yfinance not installed — skipping. Run: pip install yfinance")
        return

    _update_investing_csv(NDX_FILE, "^NDX", yf)
    _update_stock_price_csvs(yf)

    # S5TH (% of S&P 500 above 200-day MA) is a Bloomberg/Investing.com
    # proprietary series — not available via free APIs.  Download manually from
    # https://www.investing.com/indices/s-p-500-above-200-dma-historical-data
    # and replace S5TH.csv.
    df_b = pd.read_csv(BREADTH_FILE)
    df_b["_date"] = pd.to_datetime(
        df_b["Date"].astype(str).str.strip('"'), format="%m/%d/%Y"
    )
    last_b = df_b["_date"].max()
    lag    = (pd.Timestamp.today().normalize() - last_b).days
    if lag > 5:
        print(
            f"[fetch] WARNING: S5TH.csv is {lag} days old (last: {last_b.date()}).\n"
            "        Download the latest data from Investing.com and replace S5TH.csv."
        )
    else:
        print(f"[fetch] S5TH.csv is up to date ({last_b.date()}).")


# ── Data loading ─────────────────────────────────────────────────────────────

def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_ndx_breadth() -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx["price"] = _parse_price(ndx["Price"])

    b200 = _load_breadth()

    vix = pd.read_csv(VIX_FILE)
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix.set_index("Date", inplace=True)
    vix["vix"] = _parse_price(vix["Price"])

    merged = ndx[["price"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged = merged.join(vix[["vix"]], how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]

    merged["vix"]   = merged["vix"].ffill()
    merged["ma200"] = merged["price"].rolling(MA200_WINDOW).mean()

    # Vote gate: at least 1 of [VIX > 30, price > MA200] must be True
    # NaN → True (don't restrict when data is missing)
    merged["vix_vote"]   = merged["vix"].apply(
        lambda v: True if pd.isna(v) else v > VIX_BUY_THRESH)
    merged["ma200_vote"] = merged.apply(
        lambda r: True if pd.isna(r["ma200"]) else r["price"] > r["ma200"], axis=1)
    merged["vote_gate"]  = merged["vix_vote"] | merged["ma200_vote"]

    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)
    return merged


def load_holdings(top_n: int = 2) -> dict[int, list[tuple[str, float]]]:
    """Return {year: [(ticker, weight), ...]} value-weighted, using top N holdings."""
    df = pd.read_csv(HOLDINGS_FILE)
    holdings: dict[int, list[tuple[str, float]]] = {}
    for _, row in df.iterrows():
        year = int(row["Year"])
        pairs = []
        for i in [str(n) for n in range(1, top_n + 1)]:
            name   = str(row.get(f"#{i} Holding", "")).strip()
            val    = float(row.get(f"#{i} Value ($B)", 0) or 0)
            ticker = NAME_TO_TICKER.get(name)
            if ticker and val > 0:
                pairs.append((ticker, val))
        if pairs:
            total = sum(v for _, v in pairs)
            holdings[year] = [(t, v / total) for t, v in pairs]
    return holdings


def load_stock_prices(tickers: set[str]) -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = PRICES_DIR / f"{ticker}.csv"
        if not path.exists():
            print(f"  [WARNING] No price file for {ticker}, skipping.")
            continue
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, format='ISO8601', utc=True).tz_localize(None)
        col = "Close" if "Close" in df.columns else df.columns[0]
        prices[ticker] = df[col].dropna()
    return prices


# ── Basket helpers ────────────────────────────────────────────────────────────

def get_price(prices: dict[str, pd.Series], ticker: str, date: pd.Timestamp) -> float | None:
    s = prices.get(ticker)
    if s is None or s.empty:
        return None
    # nearest available price on or before date
    idx = s.index[s.index <= date]
    if idx.empty:
        return None
    return float(s.loc[idx[-1]])


def build_basket(
    cash: float,
    composition: list[tuple[str, float]],
    prices: dict[str, pd.Series],
    date: pd.Timestamp,
) -> dict[str, float]:
    """Allocate `cash` value-weighted; return {ticker: shares}."""
    basket: dict[str, float] = {}
    # Filter to tickers with available prices
    available = [(t, w) for t, w in composition if get_price(prices, t, date) is not None]
    if not available:
        return basket
    total_w = sum(w for _, w in available)
    for ticker, weight in available:
        alloc = cash * (weight / total_w)
        price = get_price(prices, ticker, date) * (1 + SLIPPAGE)
        basket[ticker] = alloc / price
    return basket


def basket_value(basket: dict[str, float], prices: dict[str, pd.Series], date: pd.Timestamp) -> float:
    total = 0.0
    for ticker, shares in basket.items():
        p = get_price(prices, ticker, date)
        if p is not None:
            total += shares * p
    return total


# ── Strategy ──────────────────────────────────────────────────────────────────

def _days_str(days: int) -> str:
    years, rem = divmod(days, 365)
    months = rem // 30
    if years and months:
        return f"{years}y {months}m"
    if years:
        return f"{years}y"
    if months:
        return f"{months}m"
    return f"{days}d"


def run_strategy(
    df: pd.DataFrame,
    holdings: dict[int, list[tuple[str, float]]],
    prices: dict[str, pd.Series],
) -> tuple[pd.Series, list[dict], dict | None]:

    position           = "OUT"
    basket: dict[str, float] = {}
    entry_date         = None
    original_entry_val = 0.0   # basket value at trade open — never overwritten
    trade_min_val      = 0.0   # lowest basket value seen during the trade (for MAE)
    portfolio          = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}
    entry_holdings: list[str] = []

    prev_year: int | None = None

    for date, row in df.iterrows():
        ndx_price    = row["price"]
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])
        year         = date.year

        # ── Annual rebalance during an open trade ──────────────────────────
        if position == "IN" and prev_year is not None and year != prev_year:
            new_comp = holdings.get(year, holdings.get(year - 1, []))
            current_val = basket_value(basket, prices, date)
            if current_val > 0 and new_comp:
                after_sell = current_val * (1 - SLIPPAGE) - COMMISSION
                basket     = build_basket(after_sell, new_comp, prices, date)
                # original_entry_val and trade_min_val intentionally NOT reset here

        prev_year = year

        # ── State machine ──────────────────────────────────────────────────
        vote_gate = bool(row["vote_gate"])
        if position == "OUT":
            do_buy = not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate
            if do_buy:
                comp = holdings.get(year, [])
                if not comp:
                    values[date] = portfolio
                    continue
                cash_to_invest     = portfolio - COMMISSION
                basket             = build_basket(cash_to_invest, comp, prices, date)
                original_entry_val = basket_value(basket, prices, date)
                trade_min_val      = original_entry_val
                entry_holdings     = list(basket.keys())
                entry_date         = date
                position           = "IN"
                portfolio          = 0.0

        elif position == "IN":
            current_val   = basket_value(basket, prices, date)
            trade_min_val = min(trade_min_val, current_val)

            bearish_div = price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP
            if bearish_div:
                eff_exit_val = current_val * (1 - SLIPPAGE) - COMMISSION
                gross_ret    = (eff_exit_val - original_entry_val) / original_entry_val if original_entry_val else 0
                mae          = (trade_min_val - original_entry_val) / original_entry_val * 100
                portfolio    = eff_exit_val
                trades.append({
                    "entry_date":       entry_date,
                    "exit_date":        date,
                    "entry_basket_val": original_entry_val,
                    "exit_basket_val":  eff_exit_val,
                    "return_pct":       gross_ret * 100,
                    "max_drawdown_pct": mae,
                    "accumulated":      eff_exit_val,
                    "sell_reason":      "bearish-divergence",
                    "entry_holdings":   entry_holdings,
                    "exit_holdings":    list(basket.keys()),
                })
                basket   = {}
                position = "OUT"

        # ── Mark-to-market ─────────────────────────────────────────────────
        if position == "IN":
            values[date] = basket_value(basket, prices, date)
        else:
            values[date] = portfolio

    # ── Open trade ─────────────────────────────────────────────────────────
    open_trade = None
    if position == "IN":
        last_date = df.index[-1]
        last_val  = basket_value(basket, prices, last_date)
        eff_last  = last_val * (1 - SLIPPAGE)
        gross_ret = (eff_last - original_entry_val) / original_entry_val if original_entry_val else 0
        trade_min_val = min(trade_min_val, last_val)
        mae       = (trade_min_val - original_entry_val) / original_entry_val * 100
        open_trade = {
            "entry_date":         entry_date,
            "entry_basket_val":   original_entry_val,
            "current_date":       last_date,
            "current_basket_val": last_val,
            "return_pct":         gross_ret * 100,
            "max_drawdown_pct":   mae,
            "accumulated":        eff_last,
            "entry_holdings":     entry_holdings,
            "exit_holdings":      list(basket.keys()),
        }

    return pd.Series(values, name="strategy"), trades, open_trade


def run_benchmark(df: pd.DataFrame) -> pd.Series:
    first = df["price"].iloc[0]
    return (INITIAL_CAPITAL * df["price"] / first).rename("benchmark")


# ── Metrics / printing ────────────────────────────────────────────────────────

def compute_metrics(values: pd.Series, trades: list[dict] | None = None) -> dict:
    dr    = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    tr    = (values.iloc[-1] / values.iloc[0]) - 1
    cagr  = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    mdd   = ((values - values.cummax()) / values.cummax()).min()
    std   = dr.std()
    sh    = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    m = {
        "Total Return": f"{tr:.1%}",
        "CAGR":         f"{cagr:.1%}",
        "Max Drawdown": f"{mdd:.1%}",
        "Sharpe Ratio": f"{sh:.2f}",
        "Final Value":  f"${values.iloc[-1]:,.0f}",
    }
    if trades is not None:
        n       = len(trades)
        wins    = sum(1 for t in trades if t["return_pct"] > 0)
        in_days = sum((t["exit_date"] - t["entry_date"]).days for t in trades)
        tot     = (values.index[-1] - values.index[0]).days
        m.update({
            "# Trades":       str(n),
            "Win Rate":       f"{wins/n:.1%}" if n else "—",
            "Time in Market": f"{in_days/tot:.1%}" if tot else "—",
        })
    return m


def print_metrics(strat: dict, bench: dict) -> None:
    keys = list(dict.fromkeys(list(strat) + list(bench)))
    col  = 16
    hdr  = f"{'Metric':<22}{'Strategy':>{col}}{'Buy&Hold NDX':>{col}}"
    sep  = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        print(f"  {k:<20}{strat.get(k, '—'):>{col}}{bench.get(k, '—'):>{col}}")
    print(sep)


def print_trades(trades: list[dict], open_trade: dict | None = None) -> None:
    if not trades and not open_trade:
        print("\nNo completed trades.")
        return
    hdr = (
        f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}"
        f"  {'Return':>8}  {'MaxDD':>7}"
        f"  {'Entry Port':>12}  {'MaxDD Port':>12}  {'Exit Port':>12}"
        f"  Holdings"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        days     = (t["exit_date"] - t["entry_date"]).days
        ep       = t["entry_basket_val"]
        mdd_port = ep * (1 + t["max_drawdown_pct"] / 100)
        eh       = "+".join(t.get("entry_holdings", []))
        xh       = "+".join(t.get("exit_holdings", []))
        basket   = eh if eh == xh else f"{eh} → {xh}"
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+6.1f}%"
            f"  ${ep:>11,.0f}  ${mdd_port:>11,.0f}  ${t['accumulated']:>11,.0f}"
            f"  {basket}"
        )
    if open_trade:
        days     = (open_trade["current_date"] - open_trade["entry_date"]).days
        ep       = open_trade["entry_basket_val"]
        mdd_port = ep * (1 + open_trade["max_drawdown_pct"] / 100)
        eh       = "+".join(open_trade.get("entry_holdings", []))
        xh       = "+".join(open_trade.get("exit_holdings", []))
        basket   = eh if eh == xh else f"{eh} → {xh}"
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+6.1f}%"
            f"  ${ep:>11,.0f}  ${mdd_port:>11,.0f}  ${open_trade['accumulated']:>11,.0f}"
            f"  {basket}  (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def plot_results(df, strategy, benchmark, trades, open_trade) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 0.8]}
    )
    ax1, ax2, ax3 = axes

    fig.suptitle(
        "NASDAQ 100 Breadth Strategy — NDX Top-1 Holding  (+Voting Gate)\n"
        f"BUY: breadth200 < {BUY_B200_THRESH}%  AND  (VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW})  [≥1 of 2]  |  "
        f"SELL: price ≥{DIVERGENCE_PRICE_RISE}% over {DIVERGENCE_WINDOW}d "
        f"AND breadth fell ≥{DIVERGENCE_BREADTH_FALL}pts AND breadth < {DIVERGENCE_BREADTH_CAP}%\n"
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}  |  Annual rebalancing to year's top-1 holding",
        fontsize=9, fontweight="bold"
    )

    ax1.plot(benchmark.index, benchmark, label="Buy & Hold NDX", color="#2196F3", linewidth=1.5)
    ax1.plot(strategy.index,  strategy,  label="Top-2 Basket Strategy", color="#FF5722", linewidth=1.5)

    all_entries = [t["entry_date"] for t in trades] + ([open_trade["entry_date"]] if open_trade else [])
    all_exits   = [t["exit_date"] for t in trades]
    if all_entries:
        ax1.scatter(all_entries, strategy.reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=80, zorder=5, label="Buy")
    if all_exits:
        ax1.scatter(all_exits, strategy.reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=80, zorder=5, label="Sell")

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df["breadth"], color="#7B1FA2", linewidth=1.0,
             label="% Above 200-Day MA (S&P 500)")
    ax2.axhline(BUY_B200_THRESH, color="green", linestyle="--", linewidth=1.0,
                label=f"Buy gate: <{BUY_B200_THRESH}%")
    ax2.axhline(DIVERGENCE_BREADTH_CAP, color="red", linestyle="--", linewidth=0.9,
                label=f"Sell cap: <{DIVERGENCE_BREADTH_CAP}%")
    ax2.fill_between(df.index, df["breadth"], BUY_B200_THRESH,
                     where=df["breadth"] < BUY_B200_THRESH, color="green", alpha=0.12)
    if all_entries:
        ax2.scatter(all_entries, df["breadth"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=60, zorder=5)
    if all_exits:
        ax2.scatter(all_exits, df["breadth"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=60, zorder=5)
    ax2.set_ylabel("Breadth (%)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    ax3.plot(df.index, df["price"], color="#546E7A", linewidth=1.0, label="NASDAQ 100")
    if all_entries:
        ax3.scatter(all_entries, df["price"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=50, zorder=5)
    if all_exits:
        ax3.scatter(all_exits, df["price"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=50, zorder=5)
    ax3.set_ylabel("NDX")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out = DATA_DIR / "nasdaq100_top2_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def run_qqq_strategy(df: pd.DataFrame) -> tuple[pd.Series, list[dict], dict | None]:
    """Replicate qqq_backtest.py signal logic (no trailing stop) on the NDX price series."""
    position   = "OUT"
    eff_entry  = raw_entry = 0.0
    entry_date = None
    trade_low  = 0.0
    portfolio  = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        price        = row["price"]
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])

        vote_gate = bool(row["vote_gate"])
        if position == "OUT":
            if not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate:
                portfolio -= COMMISSION
                eff_entry  = price * (1 + SLIPPAGE)
                raw_entry  = price
                entry_date = date
                trade_low  = price
                position   = "IN"
        elif position == "IN":
            trade_low = min(trade_low, price)
            if price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                trades.append({
                    "entry_date":       entry_date,
                    "exit_date":        date,
                    "entry_price":      raw_entry,
                    "exit_price":       price,
                    "return_pct":       gross_ret * 100,
                    "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
                    "accumulated":      portfolio,
                    "sell_reason":      "bearish-divergence",
                })
                position = "OUT"

        values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry) if position == "IN" else portfolio

    open_trade = None
    if position == "IN":
        last_price = df["price"].iloc[-1]
        last_date  = df.index[-1]
        eff_last   = last_price * (1 - SLIPPAGE)
        open_trade = {
            "entry_date":    entry_date,
            "entry_price":   raw_entry,
            "current_date":  last_date,
            "current_price": last_price,
            "return_pct":    (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "accumulated":   portfolio * (eff_last / eff_entry),
        }

    return pd.Series(values, name="qqq_strategy"), trades, open_trade


def clip_and_normalize(series: pd.Series, start: pd.Timestamp) -> pd.Series:
    """Clip series to start date and rescale so the first value = INITIAL_CAPITAL."""
    s = series[series.index >= start]
    return s / s.iloc[0] * INITIAL_CAPITAL


def compute_metrics_clipped(
    series: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    start: pd.Timestamp,
) -> dict:
    """Metrics computed only from `start` onward; trades/open_trade filtered to same window."""
    s = clip_and_normalize(series, start)
    clipped_trades = [t for t in trades if t["entry_date"] >= start]
    if open_trade and open_trade["entry_date"] >= start:
        clipped_trades_with_open = clipped_trades  # open trade included in time-in-market
        n_open_days = (open_trade["current_date"] - open_trade["entry_date"]).days
    else:
        n_open_days = 0

    dr    = s.pct_change().dropna()
    years = (s.index[-1] - s.index[0]).days / 365.25
    tr    = (s.iloc[-1] / s.iloc[0]) - 1
    cagr  = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    mdd   = ((s - s.cummax()) / s.cummax()).min()
    std   = dr.std()
    sh    = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    n       = len(clipped_trades)
    wins    = sum(1 for t in clipped_trades if t["return_pct"] > 0)
    in_days = sum((t["exit_date"] - t["entry_date"]).days for t in clipped_trades) + n_open_days
    tot     = (s.index[-1] - s.index[0]).days

    return {
        "Total Return":   f"{tr:.1%}",
        "CAGR":           f"{cagr:.1%}",
        "Max Drawdown":   f"{mdd:.1%}",
        "Sharpe Ratio":   f"{sh:.2f}",
        "Final Value":    f"${s.iloc[-1]:,.0f}",
        "# Trades":       str(n),
        "Win Rate":       f"{wins/n:.1%}" if n else "—",
        "Time in Market": f"{in_days/tot:.1%}" if tot else "—",
    }


def print_qqq_trades(
    trades: list[dict],
    open_trade: dict | None = None,
    scale: float = 1.0,
) -> None:
    """Print QQQ trade log with Entry Port / MaxDD Port / Exit Port columns.

    `scale` normalises the original accumulated values so the first trade's
    entry portfolio equals INITIAL_CAPITAL.
    """
    if not trades and not open_trade:
        print("\nNo completed trades.")
        return
    hdr = (
        f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}"
        f"  {'Return':>8}  {'MaxDD':>7}"
        f"  {'Entry Port':>12}  {'MaxDD Port':>12}  {'Exit Port':>12}"
    )
    print(hdr)
    print("-" * len(hdr))

    # Reconstruct entry portfolio for each trade (scaled)
    entry_ports = [INITIAL_CAPITAL]
    for t in trades[:-1]:
        entry_ports.append(t["accumulated"] * scale)

    for i, (t, ep) in enumerate(zip(trades, entry_ports), 1):
        days     = (t["exit_date"] - t["entry_date"]).days
        mdd_port = ep * (1 + t["max_drawdown_pct"] / 100)
        exit_p   = t["accumulated"] * scale
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+6.1f}%"
            f"  ${ep:>11,.0f}  ${mdd_port:>11,.0f}  ${exit_p:>11,.0f}"
        )

    if open_trade:
        ep       = trades[-1]["accumulated"] * scale if trades else INITIAL_CAPITAL
        days     = (open_trade["current_date"] - open_trade["entry_date"]).days
        mdd_port = ep * (1 + open_trade["max_drawdown_pct"] / 100)
        exit_p   = open_trade["accumulated"] * scale
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+6.1f}%"
            f"  ${ep:>11,.0f}  ${mdd_port:>11,.0f}  ${exit_p:>11,.0f}"
            f"  (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def print_metrics_3col(qqq: dict, top1: dict, bench: dict) -> None:
    keys = list(dict.fromkeys(list(qqq) + list(top1) + list(bench)))
    col  = 16
    hdr  = (f"{'Metric':<22}{'QQQ Strategy':>{col}}{'NDX Top-1':>{col}}"
            f"{'Buy&Hold NDX':>{col}}")
    sep  = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        print(f"  {k:<20}{qqq.get(k,'—'):>{col}}{top1.get(k,'—'):>{col}}"
              f"{bench.get(k,'—'):>{col}}")
    print(sep)


def main() -> None:
    fetch_latest_data()
    print("Loading NDX + breadth data...")
    df = load_ndx_breadth()

    all_tickers = {t for h in [load_holdings(1), load_holdings(2)]
                   for comps in h.values() for t, _ in comps}
    print("Loading individual stock prices...")
    prices = load_stock_prices(all_tickers)

    # ── Run strategies ────────────────────────────────────────────────────
    h1                     = load_holdings(top_n=1)
    strat1, trades1, open1 = run_strategy(df, h1, prices)
    qqq, trades_q, open_q  = run_qqq_strategy(df)
    benchmark              = run_benchmark(df)

    # ── Common start = first NDX Top-1 trade date ─────────────────────────
    common_start = strat1[strat1 != INITIAL_CAPITAL].index[0] if (strat1 != INITIAL_CAPITAL).any() else strat1.index[0]
    print(f"\nFull date range : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Common start    : {common_start.date()} (first trade)\n")

    mq = compute_metrics_clipped(qqq,       trades_q, open_q, common_start)
    m1 = compute_metrics_clipped(strat1,    trades1,  open1,  common_start)
    mb = compute_metrics_clipped(benchmark, [],       None,   common_start)

    print_metrics_3col(mq, m1, mb)

    # Scale QQQ portfolio to same INITIAL_CAPITAL at common_start
    qqq_at_start       = qqq[qqq.index >= common_start].iloc[0]
    qqq_scale          = INITIAL_CAPITAL / qqq_at_start
    qqq_trades_clipped = [t for t in trades_q if t["entry_date"] >= common_start]
    qqq_open_clipped   = open_q if (open_q and open_q["entry_date"] >= common_start) else None

    print(f"\n══ QQQ Strategy — trade log (normalised to ${INITIAL_CAPITAL:,.0f} at {common_start.date()}) ══")
    print_qqq_trades(qqq_trades_clipped, qqq_open_clipped, scale=qqq_scale)

    print(f"\n══ NDX Top-1 Strategy — trade log (starting ${INITIAL_CAPITAL:,.0f} at {common_start.date()}) ══")
    print_trades(trades1, open1)


if __name__ == "__main__":
    main()
