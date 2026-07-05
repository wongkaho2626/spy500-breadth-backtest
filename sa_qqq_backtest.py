"""
Seeking Alpha Annual Picks — QQQ-Timed Backtest (2022-2026)

Applies QQQ breadth-timing signals to the SA annual stock picks.
Four strategies on 10 SA picks per year:
  A – Always-in baseline   : Jan 1 entry, Dec 31 exit
  B – QQQ-timed entry      : wait for buy signal, Dec 31 exit
  C – QQQ-timed entry+exit : buy signal entry, bearish-div exit (30d min hold)
  D – Full QQQ timing      : buy signal entry, bearish-div OR trailing-stop exit

Buy signal (two tiers):
  Strong : breadth200 < 26%  (deep oversold — enter immediately)
  Moderate: breadth200 < 50% AND RSI14 < 45 AND (VIX > 20 OR NDX > MA200)
QQQ sell signal : NDX price rose >=3% over 60d  AND  breadth fell >=20pts  AND  breadth < 60%
                  (minimum 30 trading-day hold before this can fire)
Trailing stop   : portfolio drops >=25% from peak (Strategy D only)
Fallback entry  : if no buy signal by Apr 30, enter Jan 1 of that year
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ── QQQ signal parameters (grid-search optimised — sa_qqq_optimize.py) ───────
# Strong buy (deep oversold — same as qqq_backtest.py)
BUY_STRONG_THRESH       = 26.0
# Moderate buy: grid-optimised → breadth<40 + RSI<35 + VIX>15 (or above MA200)
BUY_MOD_THRESH          = 40.0   # optimised (was 50)
BUY_MOD_RSI_CAP         = 35.0   # optimised (was 45) — requires clearly oversold RSI
BUY_MOD_VIX_THRESH      = 15.0   # optimised (was 20)
VIX_BUY_THRESH          = 30.0   # VIX vote for strong entry
MA200_WINDOW            = 200
RSI_WINDOW              = 14

DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 25.0   # grid-optimised (was 20) — stricter, avoids premature exit
DIVERGENCE_BREADTH_CAP  = 55.0   # grid-optimised (was 60) — stricter
MIN_HOLD_DAYS           = 30

TRAILING_STOP_PCT  = 25.0
FALLBACK_MONTH_END = 3    # optimised (was 4)
INITIAL_CAPITAL    = 10_000.0

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day data; the earliest tradeable fill is the NEXT
# session. Default: a signal on day t fills at day t+1's OPEN of the SA picks.
# Set EXECUTION_LAG=0 and FILL_PRICE="close" for the legacy same-day-close fill.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

SA_PICKS = {
    2022: ["FNF", "BAC", "LYG", "CI", "BNTX", "CVX", "XOM", "AMD", "QCOM", "F"],
    2023: ["ASC", "ENGIY", "HDSN", "JXN", "MNSO", "MOD", "PDD", "SMCI", "VLO", "VRNA"],
    2024: ["APP", "CLS", "MOD", "RYCEY", "ANF", "META", "ISNPY", "GCT", "MHO", "LPG"],
    2025: ["CRDO", "INTA", "CLS", "DXPE", "AGX", "URBN", "LRN", "EAT", "PYPL", "OPFI"],
    2026: ["CLS", "MU", "AMD", "CIEN", "COHR", "ALL", "INCY", "B", "WLDN", "ATI"],
}


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _load_breadth() -> pd.DataFrame:
    """Prefer the continuous daily series (breadth_daily.csv, 2002+); S5TH.csv
    alone is bimonthly before 2007, which corrupts row-based windows."""
    daily = DATA_DIR / "breadth_daily.csv"
    if daily.exists():
        b200 = pd.read_csv(daily)
        b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
        b200.set_index("Date", inplace=True)
        return b200
    b200 = pd.read_csv(DATA_DIR / "S5TH.csv")
    b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
    b200.set_index("Date", inplace=True)
    b200["breadth"] = _parse_price(b200["Price"])
    # S5TH is bimonthly before 2007 — drop the sparse era
    return b200[b200.index >= "2007-01-01"]


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_market_data() -> pd.DataFrame:
    ndx = pd.read_csv(DATA_DIR / "NASDAQ100.csv")
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx["price"] = _parse_price(ndx["Price"])

    b200 = _load_breadth()

    vix = pd.read_csv(DATA_DIR / "VIX.csv")
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix.set_index("Date", inplace=True)
    vix["vix"] = _parse_price(vix["Price"])

    df = ndx[["price"]].join(b200[["breadth"]], how="left")
    df = df.join(vix[["vix"]], how="left")
    df.sort_index(inplace=True)
    df = df[df["breadth"].notna()]

    df["vix"]   = df["vix"].ffill()
    df["ma200"] = df["price"].rolling(MA200_WINDOW).mean()
    df["rsi14"] = _rsi(df["price"], RSI_WINDOW)

    # Strong-buy vote gate: VIX>30 or above MA200 (original QQQ rule)
    df["vix_strong"]  = df["vix"].apply(lambda v: True if pd.isna(v) else v > VIX_BUY_THRESH)
    df["ma200_vote"]  = df.apply(
        lambda r: True if pd.isna(r["ma200"]) else r["price"] > r["ma200"], axis=1)
    df["strong_gate"] = df["vix_strong"] | df["ma200_vote"]

    # Moderate-buy vote gate: VIX>20 or above MA200
    df["vix_mod"]    = df["vix"].apply(lambda v: True if pd.isna(v) else v > BUY_MOD_VIX_THRESH)
    df["mod_gate"]   = df["vix_mod"] | df["ma200_vote"]

    df["strong_buy"] = df["breadth"] < BUY_STRONG_THRESH
    df["mod_buy"]    = (
        (df["breadth"] < BUY_MOD_THRESH)
        & (df["rsi14"] < BUY_MOD_RSI_CAP)
        & df["mod_gate"]
    )
    df["any_buy"] = df["strong_buy"] | df["mod_buy"]

    pp = df["price"].shift(DIVERGENCE_WINDOW)
    bp = df["breadth"].shift(DIVERGENCE_WINDOW)
    df["price_rose"]  = ((df["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    df["breadth_fell"] = ((bp - df["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)
    df["bearish_div"]  = (
        df["price_rose"] & df["breadth_fell"] & (df["breadth"] < DIVERGENCE_BREADTH_CAP)
    )

    return df


def fetch_stock_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close AND open prices; silently skip tickers with no
    data. Returns (closes, opens) — opens are the fill prices under next-day-open
    execution; closes drive signals and mark-to-market."""
    data, odata = {}, {}
    for t in tickers:
        try:
            hist = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
            if not hist.empty and "Close" in hist.columns:
                s = hist["Close"].squeeze()
                if not s.empty:
                    data[t] = s
                    if "Open" in hist.columns:
                        o = hist["Open"].squeeze()
                        if not o.empty:
                            odata[t] = o
        except Exception:
            pass
    if not data:
        return pd.DataFrame(), pd.DataFrame()
    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    opens = pd.DataFrame(odata)
    if not opens.empty:
        opens.index = pd.to_datetime(opens.index).tz_localize(None)
        opens = opens.sort_index()
    return prices.sort_index(), opens


def _find_entry_date(
    mkt: pd.DataFrame,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
    timed: bool,
) -> tuple[pd.Timestamp, str]:
    """
    Return (entry_date, signal_type) for a year.
    signal_type: 'strong_buy' | 'mod_buy' | 'fallback'
    """
    if not timed:
        candidates = mkt.loc[year_start:year_end]
        first = candidates.index[0] if len(candidates) else year_start
        return first, "fallback"

    fallback_end = pd.Timestamp(year_start.year, FALLBACK_MONTH_END, 30)
    window = mkt.loc[year_start:fallback_end]

    for date, row in window.iterrows():
        if bool(row["strong_buy"]):
            return date, "strong_buy"
        if bool(row["mod_buy"]):
            return date, "mod_buy"

    year_window = mkt.loc[year_start:year_end]
    first = year_window.index[0] if len(year_window) else year_start
    return first, "fallback"


def simulate_year(
    year: int,
    capital: float,
    mkt: pd.DataFrame,
    strategy: str,
) -> tuple[float, dict]:
    """
    Run one year of a strategy. Returns (ending_capital, trade_record).
    """
    year_start = pd.Timestamp(year, 1, 1)
    year_end   = pd.Timestamp(year, 12, 31)
    today      = pd.Timestamp.today().normalize()
    year_end   = min(year_end, today)

    timed                 = strategy in ("B", "C", "D")
    entry_date, sig_type  = _find_entry_date(mkt, year_start, year_end, timed)

    fetch_start = (entry_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    fetch_end   = (year_end + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    prices, opens = fetch_stock_prices(SA_PICKS[year], fetch_start, fetch_end)

    if prices.empty:
        return capital, _empty_record(year, strategy, entry_date, year_end, capital)

    trading_days = mkt.loc[entry_date:year_end].index
    prices = prices.reindex(trading_days, method="ffill").dropna(how="all")
    if not opens.empty:
        opens = opens.reindex(prices.index)

    if prices.empty:
        return capital, _empty_record(year, strategy, entry_date, year_end, capital)

    # The signal is known at entry_date's close; the actual purchase happens
    # EXECUTION_LAG sessions later (next trading day by default).
    entry_fill_idx = min(EXECUTION_LAG, len(prices.index) - 1)
    entry_date     = prices.index[entry_fill_idx]
    prices         = prices.iloc[entry_fill_idx:]
    if not opens.empty:
        opens = opens.reindex(prices.index)

    def _fill_row(date: pd.Timestamp) -> pd.Series:
        """Fill prices for `date`: each ticker's open when available, else close."""
        row = prices.loc[date].copy()
        if FILL_PRICE == "open" and not opens.empty and date in opens.index:
            orow = opens.loc[date]
            for t in row.index:
                if t in orow.index and not pd.isna(orow[t]):
                    row[t] = orow[t]
        return row

    entry_row = _fill_row(prices.index[0])
    available = [t for t in entry_row.index if not pd.isna(entry_row[t])]
    if not available:
        return capital, _empty_record(year, strategy, entry_date, year_end, capital)

    n               = len(available)
    alloc_per_stock = capital / n
    entry_prices    = entry_row[available]
    shares          = alloc_per_stock / entry_prices

    # Per-stock peak tracking for max-gain calculation
    stock_peaks = entry_prices.copy().astype(float)
    stock_troughs = entry_prices.copy().astype(float)

    peak_portfolio = capital
    exit_date      = prices.index[-1]
    exit_reason    = "year-end"
    min_hold_idx   = MIN_HOLD_DAYS
    exit_signal_idx: int | None = None

    for idx, date in enumerate(prices.index):
        current_prices = prices.loc[date, available].ffill()
        port_value     = float((shares * current_prices).sum())
        stock_peaks    = pd.concat([stock_peaks, current_prices]).groupby(level=0).max()
        stock_troughs  = pd.concat([stock_troughs, current_prices]).groupby(level=0).min()

        past_min_hold = idx >= min_hold_idx

        if strategy == "D" and past_min_hold:
            peak_portfolio = max(peak_portfolio, port_value)
            if port_value < peak_portfolio * (1 - TRAILING_STOP_PCT / 100):
                exit_signal_idx = idx
                exit_reason = f"trailing-stop ({TRAILING_STOP_PCT:.0f}%)"
                break

        if strategy in ("C", "D") and past_min_hold:
            if date in mkt.index and bool(mkt.loc[date, "bearish_div"]):
                exit_signal_idx = idx
                exit_reason = "bearish-divergence"
                break

    if exit_signal_idx is not None:
        # Signal known at that close → sell EXECUTION_LAG sessions later.
        fill_i    = min(exit_signal_idx + EXECUTION_LAG, len(prices.index) - 1)
        exit_date = prices.index[fill_i]
        final_px  = _fill_row(exit_date)[available].ffill()
        capital   = float((shares * final_px).sum())
    else:
        final_px = prices.loc[exit_date, available].ffill()
        capital  = float((shares * final_px).sum())

    # Build per-stock records
    stock_trades = []
    for t in available:
        ep  = float(entry_prices[t])
        xp  = float(final_px[t])
        pk  = float(stock_peaks[t])
        tr  = float(stock_troughs[t])
        ret = (xp - ep) / ep * 100
        stock_trades.append({
            "ticker":       t,
            "entry_price":  ep,
            "exit_price":   xp,
            "peak_price":   pk,
            "trough_price": tr,
            "return_pct":   ret,
            "max_gain_pct": (pk - ep) / ep * 100,
            "max_loss_pct": (tr - ep) / ep * 100,
        })

    ret_pct = (capital / (alloc_per_stock * n) - 1) * 100
    return capital, {
        "year":          year,
        "strategy":      strategy,
        "entry_date":    entry_date.strftime("%Y-%m-%d"),
        "exit_date":     exit_date.strftime("%Y-%m-%d"),
        "exit_reason":   exit_reason,
        "sig_type":      sig_type,
        "stocks_used":   n,
        "return_pct":    ret_pct,
        "final_capital": capital,
        "stock_trades":  stock_trades,
    }


def _empty_record(year, strategy, entry_date, exit_date, capital) -> dict:
    return {
        "year":          year,
        "strategy":      strategy,
        "entry_date":    entry_date.strftime("%Y-%m-%d"),
        "exit_date":     exit_date.strftime("%Y-%m-%d"),
        "exit_reason":   "no-data",
        "sig_type":      "fallback",
        "stocks_used":   0,
        "return_pct":    0.0,
        "final_capital": capital,
        "stock_trades":  [],
    }


def run_all_strategies(mkt: pd.DataFrame) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {s: [] for s in ("A", "B", "C", "D")}
    capital: dict[str, float]      = {s: INITIAL_CAPITAL for s in ("A", "B", "C", "D")}

    for year in sorted(SA_PICKS):
        print(f"\n  Year {year}:")
        for strat in ("A", "B", "C", "D"):
            cap, rec = simulate_year(year, capital[strat], mkt, strat)
            capital[strat] = cap
            results[strat].append(rec)
            print(
                f"    [{strat}] entry={rec['entry_date']} ({rec['sig_type']:12s})"
                f"  exit={rec['exit_date']}  ({rec['exit_reason']:25s})"
                f"  stocks={rec['stocks_used']:2d}/{len(SA_PICKS[year])}"
                f"  return={rec['return_pct']:+6.1f}%  capital=${cap:,.0f}"
            )
    return results


def print_summary(results: dict[str, list[dict]]) -> None:
    strat_labels = {
        "A": "Always-in (baseline)",
        "B": "QQQ-timed entry",
        "C": "QQQ entry + div-exit",
        "D": "QQQ entry + div + stop",
    }
    years = sorted(SA_PICKS)

    print("\n" + "=" * 90)
    header = f"  {'Strategy':<30}"
    for y in years:
        header += f" {y:>8}"
    header += f"  {'Total':>9}  {'Final $':>10}"
    print(header)
    print("=" * 90)

    for strat, records in results.items():
        row = f"  [{strat}] {strat_labels[strat]:<27}"
        for r in records:
            row += f"  {r['return_pct']:>+6.1f}%"
        final_cap = records[-1]["final_capital"]
        total_ret = (final_cap / INITIAL_CAPITAL - 1) * 100
        row += f"  {total_ret:>+8.1f}%  ${final_cap:>10,.0f}"
        print(row)

    print("=" * 90)


def print_entry_signals(mkt: pd.DataFrame) -> None:
    """Show the QQQ signal status at each year's timed entry date."""
    print("\n── QQQ Signal Status at Entry Points (Strategies B/C/D) ──")
    print(f"  {'Year':>4}  {'Entry Date':12}  {'Breadth':>8}  {'VIX':>6}  "
          f"{'RSI14':>6}  {'Signal'}")
    print("  " + "─" * 72)
    for year in sorted(SA_PICKS):
        year_start = pd.Timestamp(year, 1, 1)
        year_end   = min(pd.Timestamp(year, 12, 31), pd.Timestamp.today().normalize())
        entry, sig_type = _find_entry_date(mkt, year_start, year_end, timed=True)
        if entry not in mkt.index:
            continue
        row  = mkt.loc[entry]
        b200 = row["breadth"]
        vix  = row["vix"]
        rsi  = row["rsi14"]
        sig_labels = {
            "strong_buy": f"STRONG BUY (breadth<{BUY_STRONG_THRESH}%)",
            "mod_buy":    f"MODERATE BUY (breadth<{BUY_MOD_THRESH}%+RSI<{BUY_MOD_RSI_CAP})",
            "fallback":   "fallback (Jan 1 / no signal by Apr)",
        }
        print(f"  {year:>4}  {entry.strftime('%Y-%m-%d'):12}  {b200:>7.1f}%  "
              f"{vix:>5.1f}  {rsi:>6.1f}  {sig_labels.get(sig_type, sig_type)}")


def plot_results(results: dict[str, list[dict]], mkt: pd.DataFrame) -> None:
    strat_colors = {"A": "#2196F3", "B": "#FF9800", "C": "#4CAF50", "D": "#E91E63"}
    strat_labels = {
        "A": "Always-in (baseline)",
        "B": "QQQ-timed entry",
        "C": "QQQ entry + bearish-div exit",
        "D": "QQQ entry + div + trailing stop",
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    for strat, records in results.items():
        caps  = [INITIAL_CAPITAL] + [r["final_capital"] for r in records]
        dates = (
            [pd.Timestamp(min(SA_PICKS) - 1, 12, 31)]
            + [pd.Timestamp(r["exit_date"]) for r in records]
        )
        ax1.plot(dates, caps, marker="o", color=strat_colors[strat],
                 label=f"[{strat}] {strat_labels[strat]}", linewidth=2, markersize=5)

    ax1.set_title(
        "Seeking Alpha Annual Picks — QQQ-Timed Strategies (2022-2026)\n"
        f"Strong buy: breadth<{BUY_STRONG_THRESH}%   "
        f"Moderate buy: breadth<{BUY_MOD_THRESH}%+RSI14<{BUY_MOD_RSI_CAP}+(VIX>{BUY_MOD_VIX_THRESH} or above MA200)\n"
        f"Sell (C/D): price +{DIVERGENCE_PRICE_RISE}%/{DIVERGENCE_WINDOW}d + breadth"
        f" -{DIVERGENCE_BREADTH_FALL}pts + breadth<{DIVERGENCE_BREADTH_CAP}%  (min {MIN_HOLD_DAYS}d hold)"
        f"   |   Trailing stop: -{TRAILING_STOP_PCT:.0f}% (D only)",
        fontsize=8.5, fontweight="bold",
    )
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(INITIAL_CAPITAL, color="grey", linestyle=":", linewidth=0.8)

    mkt_plot = mkt.loc["2021":"2026"]
    ax2.plot(mkt_plot.index, mkt_plot["breadth"], color="#7B1FA2", linewidth=0.9,
             label="S&P500 % above 200d MA")
    ax2.axhline(BUY_STRONG_THRESH, color="darkgreen", linestyle="--", linewidth=1.0,
                label=f"Strong buy <{BUY_STRONG_THRESH}%")
    ax2.axhline(BUY_MOD_THRESH, color="green", linestyle=":", linewidth=1.0,
                label=f"Moderate buy <{BUY_MOD_THRESH}%")
    ax2.axhline(DIVERGENCE_BREADTH_CAP, color="red", linestyle="--", linewidth=0.8,
                label=f"Sell cap <{DIVERGENCE_BREADTH_CAP}%")
    ax2.fill_between(mkt_plot.index, mkt_plot["breadth"], BUY_MOD_THRESH,
                     where=mkt_plot["breadth"] < BUY_MOD_THRESH, color="green", alpha=0.10)
    ax2.fill_between(mkt_plot.index, mkt_plot["breadth"], BUY_STRONG_THRESH,
                     where=mkt_plot["breadth"] < BUY_STRONG_THRESH, color="darkgreen", alpha=0.25)
    ax2.set_ylabel("Breadth (%)")
    ax2.legend(fontsize=7, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    out = DATA_DIR / "sa_qqq_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def print_stock_trades(results: dict[str, list[dict]], focus_strategy: str = "B") -> None:
    """Per-stock trade log for the chosen strategy, grouped by year."""
    records = results[focus_strategy]
    strat_label = {
        "A": "Always-in (baseline)",
        "B": "QQQ-timed entry",
        "C": "QQQ entry + div-exit",
        "D": "QQQ entry + div + stop",
    }
    print(f"\n{'='*108}")
    print(f"  Per-Stock Trade Log — Strategy [{focus_strategy}] {strat_label[focus_strategy]}")
    print(f"{'='*108}")

    for rec in records:
        year     = rec["year"]
        entry_dt = rec["entry_date"]
        exit_dt  = rec["exit_date"]
        sig      = rec["sig_type"]
        reason   = rec["exit_reason"]
        port_ret = rec["return_pct"]

        print(f"\n  ── {year}  entry={entry_dt} ({sig})  exit={exit_dt} ({reason})"
              f"  portfolio={port_ret:+.1f}% ──")
        print(f"  {'Ticker':>7}  {'Entry $':>9}  {'Exit $':>9}  {'Return':>8}"
              f"  {'Max Gain':>9}  {'Max Loss':>9}  {'Peak $':>9}  {'Trough $':>9}")
        print("  " + "─" * 84)

        stocks = sorted(rec["stock_trades"], key=lambda s: s["return_pct"], reverse=True)
        for s in stocks:
            win_marker = "▲" if s["return_pct"] > 0 else "▼"
            print(
                f"  {s['ticker']:>7}  {s['entry_price']:>9.2f}  {s['exit_price']:>9.2f}"
                f"  {s['return_pct']:>+7.1f}% {win_marker}"
                f"  {s['max_gain_pct']:>+8.1f}%  {s['max_loss_pct']:>+8.1f}%"
                f"  {s['peak_price']:>9.2f}  {s['trough_price']:>9.2f}"
            )

        wins   = sum(1 for s in stocks if s["return_pct"] > 0)
        losses = len(stocks) - wins
        avg    = sum(s["return_pct"] for s in stocks) / len(stocks) if stocks else 0
        best   = max(stocks, key=lambda s: s["return_pct"]) if stocks else None
        worst  = min(stocks, key=lambda s: s["return_pct"]) if stocks else None
        print(f"  {'─'*84}")
        print(f"  Wins: {wins}  Losses: {losses}  Avg return: {avg:+.1f}%", end="")
        if best:
            print(f"  Best: {best['ticker']} ({best['return_pct']:+.1f}%)", end="")
        if worst:
            print(f"  Worst: {worst['ticker']} ({worst['return_pct']:+.1f}%)", end="")
        print()

    print(f"{'='*108}")


def main() -> None:
    print("Loading market data...")
    mkt = load_market_data()
    print(f"Market data : {mkt.index[0].date()} -> {mkt.index[-1].date()}")

    print_entry_signals(mkt)

    print("\nRunning strategies (fetching stock prices via yfinance)...")
    results = run_all_strategies(mkt)

    print_summary(results)

    print("\nPortfolio-level trade log:")
    print(f"  {'Year':>4}  {'Strat':>5}  {'Entry':12}  {'Signal':14}  {'Exit':12}  "
          f"{'Exit reason':25}  {'Stks':>4}  {'Return':>8}  {'Capital':>12}")
    print("  " + "─" * 114)
    for strat in ("A", "B", "C", "D"):
        for r in results[strat]:
            print(
                f"  {r['year']:>4}  [{r['strategy']}]  {r['entry_date']:12}  "
                f"{r['sig_type']:14}  {r['exit_date']:12}  {r['exit_reason']:25}  "
                f"{r['stocks_used']:>4}  {r['return_pct']:>+7.1f}%  "
                f"${r['final_capital']:>11,.0f}"
            )
        print()

    # Per-stock logs for Strategy A (baseline) and B (best)
    print_stock_trades(results, focus_strategy="A")
    print_stock_trades(results, focus_strategy="B")

    plot_results(results, mkt)


if __name__ == "__main__":
    main()
