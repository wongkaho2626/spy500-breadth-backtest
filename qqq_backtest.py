"""
NASDAQ 100 Breadth + Forward PE Backtest 2007 – present

Buy signal : S&P 500 breadth (% above 200-day MA) below threshold AND
             breadth (% above 50-day MA) below 25%.
             Threshold adjusts by valuation:
               FPE < 15 (cheap)    → 26%
               FPE 15-22 (normal)  → 18%
               FPE ≥ 22 or CAPE>28 → 12%

Sell signal: Bearish divergence — price rose ≥1% over 100 trading days
             while breadth fell ≥25 pts AND breadth is below sell cap.
             Cap adjusts by valuation:
               FPE < 15 (cheap)          → 65%
               FPE ≥ 22 or CAPE ≥ 32     → 40%
               otherwise (normal)         → 55%
             Minimum 15-day hold before divergence sell fires.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR     = Path(__file__).parent
NDX_FILE     = DATA_DIR / "NASDAQ100.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"
B50_FILE     = DATA_DIR / "S&P 500 Stocks Above 50-Day Average Historical Data.csv"
CAPE_FILE    = DATA_DIR / "ShillerPE.csv"
FPE_FILE     = DATA_DIR / "S&P500ForwardPE.csv"

# ── Valuation tiers ───────────────────────────────────────────────────────────
FPE_CHEAP     = 15.0
FPE_EXPENSIVE = 22.0
FPE_MAX_CLIP  = 40.0

# Buy breadth thresholds (% S&P 500 above 200-day MA)
BUY_FPE_CHEAP  = 26.0
BUY_NORMAL     = 18.0
BUY_EXPENSIVE  = 12.0
CAPE_BUY_HIGH  = 28.0
BUY_50_THRESH  = 25.0

# Divergence sell signal
DIVERGENCE_WINDOW       = 100
DIVERGENCE_PRICE_RISE   = 1.0
DIVERGENCE_BREADTH_FALL = 25.0

# Sell caps (breadth must be below this for divergence to trigger)
CAP_FPE_CHEAP  = 65.0
CAP_NORMAL     = 55.0
CAP_EXPENSIVE  = 40.0
CAPE_EXPENSIVE = 32.0

# Minimum hold to prevent whipsaws
MIN_HOLD_DAYS = 15

# ── Shared ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
BREADTH_START   = pd.Timestamp("2007-01-02")


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _buy_threshold(fpe: float, cape: float) -> float:
    if not pd.isna(fpe) and fpe < FPE_CHEAP:
        return BUY_FPE_CHEAP
    if (not pd.isna(fpe) and fpe >= FPE_EXPENSIVE) or cape > CAPE_BUY_HIGH:
        return BUY_EXPENSIVE
    return BUY_NORMAL


def _sell_cap(fpe: float, cape: float) -> float:
    if not pd.isna(fpe) and fpe < FPE_CHEAP:
        return CAP_FPE_CHEAP
    if (not pd.isna(fpe) and fpe >= FPE_EXPENSIVE) or cape >= CAPE_EXPENSIVE:
        return CAP_EXPENSIVE
    return CAP_NORMAL


def load_data() -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["date"], format="%Y-%m-%d")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"close": "price"})
    ndx["price"] = ndx["price"].astype(float)

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

    fpe = pd.read_csv(FPE_FILE)
    fpe["Date"] = pd.to_datetime(fpe["date"], format="%Y-%m-%d")
    fpe.set_index("Date", inplace=True)
    fpe = fpe.rename(columns={"forward_pe": "fpe"})
    fpe["fpe"] = fpe["fpe"].clip(upper=FPE_MAX_CLIP)

    merged = ndx[["price"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged = merged.join(b50[["Price"]].rename(columns={"Price": "b50"}), how="left")
    merged = merged.join(cape[["cape"]], how="left")
    merged = merged.join(fpe[["fpe"]], how="left")
    merged["cape"] = merged["cape"].ffill()
    merged["fpe"]  = merged["fpe"].ffill()
    merged.sort_index(inplace=True)

    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)

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
        price        = row["price"]
        cape         = row["cape"]
        fpe          = row["fpe"]
        breadth      = row["breadth"]
        b50          = row["b50"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])
        has_breadth  = date >= BREADTH_START and not pd.isna(breadth)

        if position == "OUT":
            if has_breadth:
                active_buy = _buy_threshold(fpe, cape)
                do_buy = breadth < active_buy and not pd.isna(b50) and b50 < BUY_50_THRESH
            else:
                do_buy = False

            if do_buy:
                portfolio -= COMMISSION
                eff_entry  = price * (1 + SLIPPAGE)
                raw_entry  = price
                entry_date = date
                trade_low  = price
                position   = "IN"

        elif position == "IN":
            trade_low = min(trade_low, price)
            held_days = (date - entry_date).days

            if has_breadth:
                active_cap  = _sell_cap(fpe, cape)
                bearish_div = price_rose and breadth_fell and breadth < active_cap
                do_sell     = bearish_div and held_days >= MIN_HOLD_DAYS
                reason      = "bearish-divergence"
            else:
                do_sell = False
                reason  = ""

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
           f"  {'Return':>8}  {'Drawdown':>9}  {'Portfolio':>12}  Reason")
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+8.1f}%  "
            f"${t['accumulated']:>11,.0f}  {t.get('sell_reason','—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['entry_price']:>9.2f}  {open_trade['current_price']:>9.2f}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+8.1f}%  "
            f"${open_trade['accumulated']:>11,.0f}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def plot_results(df, strategy, benchmark, trades, open_trade) -> None:
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 14), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.0, 0.7]}
    )
    ax1, ax2, ax3, ax4 = axes

    current_fpe  = df["fpe"].iloc[-1]
    current_cape = df["cape"].iloc[-1]
    buy_t  = _buy_threshold(current_fpe, current_cape)
    sell_c = _sell_cap(current_fpe, current_cape)
    fig.suptitle(
        "NASDAQ 100 Breadth + Forward PE Backtest 2007–present\n"
        f"Buy: breadth<threshold (200d MA) + b50<{BUY_50_THRESH}%  |  "
        f"Sell: bearish divergence (price↑≥{DIVERGENCE_PRICE_RISE}%, breadth↓≥{DIVERGENCE_BREADTH_FALL}pt over {DIVERGENCE_WINDOW}d)\n"
        f"FPE tiers: cheap<{FPE_CHEAP} → buy<{BUY_FPE_CHEAP}/cap<{CAP_FPE_CHEAP}  |  "
        f"expensive>{FPE_EXPENSIVE} → buy<{BUY_EXPENSIVE}/cap<{CAP_EXPENSIVE}  |  "
        f"normal → buy<{BUY_NORMAL}/cap<{CAP_NORMAL}  |  min hold {MIN_HOLD_DAYS}d\n"
        f"Current: FPE={current_fpe:.1f}, CAPE={current_cape:.1f} → buy<{buy_t}, cap<{sell_c}  |  "
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=8.5, fontweight="bold"
    )

    # ── Panel 1: portfolio ────────────────────────────────────────────────────
    ax1.plot(benchmark.index, benchmark, label="Buy & Hold NDX", color="#2196F3", linewidth=1.5)
    ax1.plot(strategy.index,  strategy,  label="Strategy", color="#FF5722", linewidth=1.5)

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

    # ── Panel 2: breadth ──────────────────────────────────────────────────────
    b_df = df[df.index >= BREADTH_START]
    ax2.plot(b_df.index, b_df["breadth"], color="#7B1FA2", linewidth=1.0,
             label="% Above 200-Day MA (S&P 500)")
    ax2.plot(b_df.index, b_df["b50"], color="#1565C0", linewidth=0.8,
             linestyle="--", alpha=0.7, label="% Above 50-Day MA")
    for lbl, lvl, clr, ls in [
        (f"Buy norm: <{BUY_NORMAL}%",       BUY_NORMAL,     "green",     "--"),
        (f"Buy cheap: <{BUY_FPE_CHEAP}%",   BUY_FPE_CHEAP,  "limegreen", ":"),
        (f"Buy exp: <{BUY_EXPENSIVE}%",     BUY_EXPENSIVE,  "olive",     "-."),
    ]:
        ax2.axhline(lvl, color=clr, linestyle=ls, linewidth=0.9, label=lbl)
    ax2.fill_between(b_df.index, b_df["breadth"], BUY_NORMAL,
                     where=b_df["breadth"] < BUY_NORMAL, color="green", alpha=0.12)

    post_entries = [d for d in all_entries if d >= BREADTH_START]
    post_exits   = [d for d in all_exits   if d >= BREADTH_START]
    if post_entries:
        ax2.scatter(post_entries, b_df["breadth"].reindex(post_entries, method="nearest"),
                    marker="^", color="green", s=60, zorder=5)
    if post_exits:
        ax2.scatter(post_exits, b_df["breadth"].reindex(post_exits, method="nearest"),
                    marker="v", color="red", s=60, zorder=5)
    ax2.set_ylabel("Breadth (2007+)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Forward PE + CAPE ────────────────────────────────────────────
    fpe_full = df[df.index >= pd.Timestamp("1990-01-01")]
    ax3.plot(fpe_full.index, fpe_full["fpe"], color="#E65100", linewidth=1.2,
             label="S&P 500 Forward PE (clipped)")
    ax3.axhline(FPE_CHEAP,     color="green", linestyle="--", linewidth=1.0,
                label=f"Cheap: <{FPE_CHEAP}")
    ax3.axhline(FPE_EXPENSIVE, color="red",   linestyle="--", linewidth=1.0,
                label=f"Expensive: >{FPE_EXPENSIVE}")
    ax3.fill_between(fpe_full.index, fpe_full["fpe"], FPE_CHEAP,
                     where=fpe_full["fpe"] <= FPE_CHEAP, color="green", alpha=0.10)
    ax3.fill_between(fpe_full.index, fpe_full["fpe"], FPE_EXPENSIVE,
                     where=fpe_full["fpe"] >= FPE_EXPENSIVE, color="red", alpha=0.10)

    ax3b = ax3.twinx()
    ax3b.plot(df.index, df["cape"], color="#546E7A", linewidth=0.9, linestyle=":",
              alpha=0.7, label=f"CAPE (gate>{CAPE_BUY_HIGH})")
    ax3b.axhline(CAPE_BUY_HIGH, color="#546E7A", linestyle=":", linewidth=0.8)
    ax3b.set_ylabel("CAPE", fontsize=8, color="#546E7A")
    ax3b.tick_params(axis="y", labelcolor="#546E7A", labelsize=7)
    ax3b.set_ylim(0, 50)

    lines3,  labs3  = ax3.get_legend_handles_labels()
    lines3b, labs3b = ax3b.get_legend_handles_labels()
    ax3.legend(lines3 + lines3b, labs3 + labs3b, loc="upper left", fontsize=7)
    ax3.set_ylabel("Forward PE")
    ax3.set_ylim(0, FPE_MAX_CLIP + 5)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: NDX price ────────────────────────────────────────────────────
    ax4.plot(df.index, df["price"], color="#546E7A", linewidth=1.0, label="NASDAQ 100")
    if all_entries:
        ax4.scatter(all_entries, df["price"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=50, zorder=5)
    if all_exits:
        ax4.scatter(all_exits, df["price"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=50, zorder=5)
    ax4.set_ylabel("NDX")
    ax4.set_xlabel("Date")
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.xaxis.set_major_locator(mdates.YearLocator(4))
    fig.autofmt_xdate()

    out = DATA_DIR / "qqq_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"Date range  : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    print(f"Strategy    : Breadth + Forward PE (2007+)")

    current_fpe  = df["fpe"].iloc[-1]
    current_cape = df["cape"].iloc[-1]
    buy_t  = _buy_threshold(current_fpe, current_cape)
    sell_c = _sell_cap(current_fpe, current_cape)
    fpe_tier = (
        "cheap" if current_fpe < FPE_CHEAP
        else "expensive" if current_fpe >= FPE_EXPENSIVE
        else "fair"
    )
    print(f"Forward PE  : {current_fpe:.1f} [{fpe_tier}]  → buy<{buy_t}%, sell cap<{sell_c}%")
    print(f"CAPE        : {current_cape:.1f}")
    print(f"Min hold    : {MIN_HOLD_DAYS} days")
    print(f"Costs       : ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")

    strategy, trades, open_trade = run_strategy(df)
    benchmark = run_benchmark(df)

    print_metrics(compute_metrics(strategy, trades), compute_metrics(benchmark))
    print_trades(trades, open_trade)
    plot_results(df, strategy, benchmark, trades, open_trade)


if __name__ == "__main__":
    main()
