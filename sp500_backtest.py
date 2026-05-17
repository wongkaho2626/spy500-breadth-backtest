"""
S&P 500 Hybrid Backtest 1987 – present

Phase 1 (1987–2006): CAPE-only strategy — buy when CAPE is cheap,
  sell when CAPE recovers to overvalued territory.
Phase 2 (2007+): Full breadth + CAPE strategy (breadth 200-day & 50-day MA
  with 2-tier CAPE-adjusted divergence cap).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR     = Path(__file__).parent
SP500_FILE   = DATA_DIR / "S&P500.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"
B50_FILE     = DATA_DIR / "S&P 500 Stocks Above 50-Day Average Historical Data.csv"
CAPE_FILE    = DATA_DIR / "ShillerPE.csv"

# ── Phase 1: CAPE-only (pre-2007) ────────────────────────────────────────────
CAPE_BUY_ABS  = 24.0
CAPE_SELL_ABS = 30.0

# ── Phase 2: Breadth + CAPE (2007+) ──────────────────────────────────────────
BUY_THRESHOLD           = 18.0
BUY_50_THRESHOLD        = 25.0
BUY_THRESH_HI_CAPE      = 12.0   # tighter when CAPE is elevated
CAPE_BUY_HIGH           = 30.0   # CAPE above this → use BUY_THRESH_HI_CAPE
DIVERGENCE_WINDOW       = 100
DIVERGENCE_PRICE_RISE   = 1.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 55.0   # sell cap when CAPE < CAPE_EXPENSIVE
CAPE_EXPENSIVE          = 30.0   # above this → tighter sell cap
CAP_EXPENSIVE           = 45.0   # sell cap when CAPE >= CAPE_EXPENSIVE

# ── Shared ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
BREADTH_START   = pd.Timestamp("2007-01-02")


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    sp = pd.read_csv(SP500_FILE)
    sp["Date"] = pd.to_datetime(sp["date"], format="%Y-%m-%d")
    sp.set_index("Date", inplace=True)
    sp = sp.rename(columns={"close": "price"})

    b200 = pd.read_csv(BREADTH_FILE)
    b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
    b200.set_index("Date", inplace=True)
    b200["Price"] = _parse_price(b200["Price"])

    b50 = pd.read_csv(B50_FILE)
    b50["Date"] = pd.to_datetime(b50["Date"], format="%m/%d/%Y")
    b50.set_index("Date", inplace=True)
    b50["Price"] = _parse_price(b50["Price"])

    cape = pd.read_csv(CAPE_FILE)
    cape["Date"] = pd.to_datetime(cape["date"], format="%Y-%m-%d")
    cape.set_index("Date", inplace=True)
    cape = cape.rename(columns={"close": "cape"})

    merged = sp[["price"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged = merged.join(b50[["Price"]].rename(columns={"Price": "b50"}), how="left")
    merged = merged.join(cape[["cape"]], how="left")
    merged["cape"] = merged["cape"].ffill()
    merged.sort_index(inplace=True)

    div_cap = merged["cape"].apply(
        lambda c: CAP_EXPENSIVE if c >= CAPE_EXPENSIVE else DIVERGENCE_BREADTH_CAP
    )
    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["bearish_div"] = (
        ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE) &
        ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL) &
        (merged["breadth"] < div_cap)
    ).fillna(False)

    return merged


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


def run_strategy(df: pd.DataFrame) -> tuple[pd.Series, list[dict], dict | None]:
    position   = "OUT"
    eff_entry  = raw_entry = 0.0
    entry_date = None
    trade_low  = 0.0
    portfolio  = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        price       = row["price"]
        cape        = row["cape"]
        breadth     = row["breadth"]
        b50         = row["b50"]
        bearish_div = bool(row["bearish_div"])
        has_breadth = date >= BREADTH_START and not pd.isna(breadth)

        in_phase2 = date >= BREADTH_START

        if position == "OUT":
            if has_breadth:
                active_buy = BUY_THRESH_HI_CAPE if cape > CAPE_BUY_HIGH else BUY_THRESHOLD
                do_buy = breadth < active_buy and b50 < BUY_50_THRESHOLD
            elif not in_phase2:
                do_buy = cape < CAPE_BUY_ABS
            else:
                do_buy = False  # in phase 2 but breadth missing — wait

            if do_buy:
                portfolio -= COMMISSION
                eff_entry  = price * (1 + SLIPPAGE)
                raw_entry  = price
                entry_date = date
                trade_low  = price
                position   = "IN"

        elif position == "IN":
            trade_low = min(trade_low, price)

            if has_breadth:
                do_sell = bearish_div
                reason  = "bearish-divergence"
            elif not in_phase2:
                do_sell = cape > CAPE_SELL_ABS
                reason  = "cape-overvalued"
            else:
                do_sell = False  # in phase 2 but breadth missing — hold

            if do_sell:
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
                    "sell_reason":      reason,
                    "phase":            "breadth+CAPE" if has_breadth else "CAPE-only",
                })
                position = "OUT"

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    open_trade = None
    if position == "IN":
        last_price = df["price"].iloc[-1]
        last_date  = df.index[-1]
        eff_last   = last_price * (1 - SLIPPAGE)
        open_trade = {
            "entry_date":       entry_date,
            "entry_price":      raw_entry,
            "current_date":     last_date,
            "current_price":    last_price,
            "return_pct":       (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "accumulated":      portfolio * (eff_last / eff_entry),
            "phase":            "breadth+CAPE" if last_date >= BREADTH_START else "CAPE-only",
        }

    return pd.Series(values, name="strategy"), trades, open_trade


def run_benchmark(df: pd.DataFrame) -> pd.Series:
    first = df["price"].iloc[0]
    return (INITIAL_CAPITAL * df["price"] / first).rename("benchmark")


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
        n = len(trades)
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
    hdr  = f"{'Metric':<22}{'Strategy':>{col}}{'Buy & Hold':>{col}}"
    sep  = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        print(f"  {k:<20}{strat.get(k,'—'):>{col}}{bench.get(k,'—'):>{col}}")
    print(sep)


def print_trades(trades: list[dict], open_trade: dict | None = None) -> None:
    if not trades and not open_trade:
        print("\nNo completed trades.")
        return
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Entry $':>9}  {'Exit $':>9}"
           f"  {'Return':>8}  {'Drawdown':>9}  {'Portfolio':>12}  {'Phase':14}  Reason")
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+8.1f}%  "
            f"${t['accumulated']:>11,.0f}  {t.get('phase',''):14}  {t.get('sell_reason','—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['entry_price']:>9.2f}  {open_trade['current_price']:>9.2f}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+8.1f}%  "
            f"${open_trade['accumulated']:>11,.0f}  {open_trade.get('phase',''):14}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def plot_results(df, strategy, benchmark, trades, open_trade) -> None:
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 14), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 0.8, 0.7]}
    )
    ax1, ax2, ax3, ax4 = axes

    fig.suptitle(
        "S&P 500 Hybrid Backtest 1987–present\n"
        f"Pre-2007: CAPE-only (buy CAPE<{CAPE_BUY_ABS}, sell CAPE>{CAPE_SELL_ABS})  |  "
        f"2007+: Breadth+CAPE (buy breadth<{BUY_THRESHOLD}/{BUY_THRESH_HI_CAPE} when CAPE>{CAPE_BUY_HIGH}, "
        f"sell cap {DIVERGENCE_BREADTH_CAP}/{CAP_EXPENSIVE} when CAPE>={CAPE_EXPENSIVE})\n"
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=9, fontweight="bold"
    )

    ax1.plot(benchmark.index, benchmark, label="Buy & Hold S&P 500",
             color="#2196F3", linewidth=1.5)
    ax1.plot(strategy.index, strategy, label="Hybrid Strategy",
             color="#FF5722", linewidth=1.5)
    ax1.axvline(BREADTH_START, color="gray", linestyle=":", linewidth=1.2,
                label="Breadth data start (2007)")

    all_entries = [t["entry_date"] for t in trades] + (
        [open_trade["entry_date"]] if open_trade else [])
    all_exits = [t["exit_date"] for t in trades]
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

    b_df = df[df.index >= BREADTH_START]
    ax2.plot(b_df.index, b_df["breadth"], color="#7B1FA2", linewidth=1.0,
             label="% Above 200-Day MA")
    ax2.plot(b_df.index, b_df["b50"], color="#1565C0", linewidth=0.8,
             linestyle="--", alpha=0.7, label="% Above 50-Day MA")
    ax2.axhline(BUY_THRESHOLD, color="green", linestyle="--", linewidth=1.0,
                label=f"Buy: <{BUY_THRESHOLD}")
    ax2.fill_between(b_df.index, b_df["breadth"], BUY_THRESHOLD,
                     where=b_df["breadth"] < BUY_THRESHOLD, color="green", alpha=0.15)
    post_entries = [d for d in all_entries if d >= BREADTH_START]
    post_exits   = [d for d in all_exits   if d >= BREADTH_START]
    if post_entries:
        ax2.scatter(post_entries,
                    b_df["breadth"].reindex(post_entries, method="nearest"),
                    marker="^", color="green", s=60, zorder=5)
    if post_exits:
        ax2.scatter(post_exits,
                    b_df["breadth"].reindex(post_exits, method="nearest"),
                    marker="v", color="red", s=60, zorder=5)
    ax2.set_ylabel("Breadth (2007+)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    ax3.plot(df.index, df["cape"], color="#E65100", linewidth=1.2,
             label="Shiller PE (CAPE)")
    ax3.axhline(CAPE_BUY_ABS,   color="green",  linestyle="--", linewidth=1.0,
                label=f"Buy (pre-2007): <{CAPE_BUY_ABS}")
    ax3.axhline(CAPE_SELL_ABS,  color="orange", linestyle="--", linewidth=1.0,
                label=f"Sell (pre-2007): >{CAPE_SELL_ABS}")
    ax3.axhline(CAPE_EXPENSIVE, color="red",    linestyle=":",  linewidth=1.0,
                label=f"Expensive: >={CAPE_EXPENSIVE}")
    ax3.axvline(BREADTH_START, color="gray", linestyle=":", linewidth=1.2)
    ax3.fill_between(df.index, df["cape"], CAPE_EXPENSIVE,
                     where=df["cape"] >= CAPE_EXPENSIVE, color="red", alpha=0.10)
    ax3.set_ylabel("CAPE")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)

    ax4.plot(df.index, df["price"], color="#546E7A", linewidth=1.0,
             label="S&P 500")
    if all_entries:
        ax4.scatter(all_entries, df["price"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=50, zorder=5)
    if all_exits:
        ax4.scatter(all_exits, df["price"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=50, zorder=5)
    ax4.set_ylabel("S&P 500")
    ax4.set_xlabel("Date")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.xaxis.set_major_locator(mdates.YearLocator(4))
    fig.autofmt_xdate()

    out = DATA_DIR / "sp500_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"Date range  : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")

    current_cape = df["cape"].iloc[-1]
    cape_tier    = f"expensive (>={CAPE_EXPENSIVE})" if current_cape >= CAPE_EXPENSIVE else f"normal (<{CAPE_EXPENSIVE})"
    active_buy   = BUY_THRESH_HI_CAPE if current_cape > CAPE_BUY_HIGH else BUY_THRESHOLD
    active_cap   = CAP_EXPENSIVE if current_cape >= CAPE_EXPENSIVE else DIVERGENCE_BREADTH_CAP
    print(f"Phase 1     : CAPE-only 1987–2006 "
          f"(buy CAPE<{CAPE_BUY_ABS}, sell CAPE>{CAPE_SELL_ABS})")
    print(f"Phase 2     : Breadth+CAPE 2007+ "
          f"(buy breadth<{active_buy}, sell cap<{active_cap})")
    print(f"Valuation   : CAPE={current_cape:.1f} [{cape_tier}]  → sell cap={active_cap}")
    print(f"Costs       : ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")

    strategy, trades, open_trade = run_strategy(df)
    benchmark = run_benchmark(df)

    print_metrics(compute_metrics(strategy, trades), compute_metrics(benchmark))
    print_trades(trades, open_trade)
    plot_results(df, strategy, benchmark, trades, open_trade)


if __name__ == "__main__":
    main()
