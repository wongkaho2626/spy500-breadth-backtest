"""
NASDAQ 100 Breadth Strategy — breadth data start to present

BUY  (while OUT): breadth200 < 26%
                  AND at least 1 of 2 vote:
                    • VIX > 30  (fear spike / panic bottom)
                    • price > MA200  (uptrend pullback — safe to buy immediately)
SELL (while IN):  Bearish divergence — price rose ≥ 3% over 60 days
                  while breadth200 fell ≥ 20 pts AND breadth200 < 60%
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

try:
    from fetch_investing_data import fetch_all_updates
    fetch_all_updates(verbose=True)
except Exception as _fetch_err:
    print(f"[data fetch skipped: {_fetch_err}]")

DATA_DIR     = Path(__file__).parent
NDX_FILE     = DATA_DIR / "NASDAQ100.csv"
BREADTH_FILE = DATA_DIR / "S5TH.csv"
VIX_FILE     = DATA_DIR / "VIX.csv"

# ── Buy thresholds ────────────────────────────────────────────────────────────
BUY_B200_THRESH = 26.0   # breadth200 must be below this
VIX_BUY_THRESH  = 30.0   # VIX vote: fear spike (VIX > 30)
MA200_WINDOW    = 200     # MA200 vote: price above 200-day moving average

# ── Sell — bearish divergence ─────────────────────────────────────────────────
DIVERGENCE_WINDOW       = 60    # trading days lookback
DIVERGENCE_PRICE_RISE   = 3.0   # % price rise over window
DIVERGENCE_BREADTH_FALL = 20.0  # pts breadth200 drop over window
DIVERGENCE_BREADTH_CAP  = 60.0  # breadth200 must be below this

# ── Shared ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
START_YEAR      = None   # e.g. 2010 to begin backtest on Jan 1 of that year; None = full history
COOLDOWN_DAYS   = 15     # calendar days to wait after a sell before the next buy is allowed


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price"})
    ndx["price"] = _parse_price(ndx["price"])

    b200 = pd.read_csv(BREADTH_FILE)
    b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
    b200.set_index("Date", inplace=True)
    b200["Price"] = _parse_price(b200["Price"])

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

    # Only keep rows where breadth200 data is present (strategy starts here)
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

    # Pre-compute divergence components
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


def run_strategy(df: pd.DataFrame, cooldown_days: int = 0) -> tuple[pd.Series, list[dict], dict | None]:
    position       = "OUT"
    eff_entry      = raw_entry = 0.0
    entry_date     = None
    trade_low      = 0.0
    portfolio      = INITIAL_CAPITAL
    cooldown_until: pd.Timestamp | None = None
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        price        = row["price"]
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])

        if position == "OUT":
            vote_gate      = bool(row["vote_gate"])
            cooldown_ok    = cooldown_until is None or date > cooldown_until
            do_buy = not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate and cooldown_ok
            if do_buy:
                portfolio -= COMMISSION
                eff_entry  = price * (1 + SLIPPAGE)
                raw_entry  = price
                entry_date = date
                trade_low  = price
                position   = "IN"
                buy_trigger = (("VIX" if row["vix_vote"] else "") +
                               ("+" if row["vix_vote"] and row["ma200_vote"] else "") +
                               ("MA200" if row["ma200_vote"] else ""))

        elif position == "IN":
            trade_low = min(trade_low, price)
            bearish_div = price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP
            if bearish_div:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                cooldown_until = date + pd.Timedelta(days=cooldown_days)
                trades.append({
                    "entry_date":       entry_date,
                    "exit_date":        date,
                    "entry_price":      raw_entry,
                    "exit_price":       price,
                    "return_pct":       gross_ret * 100,
                    "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
                    "accumulated":      portfolio,
                    "buy_trigger":      buy_trigger,
                    "sell_reason":      "bearish-divergence",
                    "cooldown_until":   cooldown_until,
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
            "buy_trigger":      buy_trigger,
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
        print(f"  {k:<20}{strat.get(k, '—'):>{col}}{bench.get(k, '—'):>{col}}")
    print(sep)


def print_trades(trades: list[dict], open_trade: dict | None = None) -> None:
    if not trades and not open_trade:
        print("\nNo completed trades.")
        return
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Entry $':>9}  {'Exit $':>9}"
           f"  {'Return':>8}  {'Drawdown':>9}  {'Portfolio':>12}  {'Buy trigger':>11}  Sell reason")
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+8.1f}%  "
            f"${t['accumulated']:>11,.0f}  {t.get('buy_trigger','—'):>11}  {t.get('sell_reason','—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['entry_price']:>9.2f}  {open_trade['current_price']:>9.2f}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+8.1f}%  "
            f"${open_trade['accumulated']:>11,.0f}  {open_trade.get('buy_trigger','—'):>11}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def print_sell_proximity(df: pd.DataFrame, open_trade: dict | None) -> None:
    """Show how close the current position is to triggering the sell signal."""
    if open_trade is None:
        return

    last      = df.iloc[-1]
    last_date = df.index[-1]

    # Look back exactly DIVERGENCE_WINDOW rows for the comparison point
    lookback_idx = max(0, len(df) - 1 - DIVERGENCE_WINDOW)
    past         = df.iloc[lookback_idx]

    price_now      = last["price"]
    price_then     = past["price"]
    price_rise_pct = (price_now - price_then) / price_then * 100

    breadth_now    = last["breadth"]
    breadth_then   = past["breadth"]
    breadth_fall   = breadth_then - breadth_now   # positive = fell

    cap_ok = breadth_now < DIVERGENCE_BREADTH_CAP

    def bar(value: float, threshold: float, invert: bool = False) -> str:
        """20-char progress bar toward the threshold."""
        ratio = min(value / threshold, 1.0) if threshold != 0 else 1.0
        filled = round(ratio * 20)
        return f"[{'█' * filled}{'░' * (20 - filled)}] {ratio:.0%}"

    price_met   = price_rise_pct >= DIVERGENCE_PRICE_RISE
    breadth_met = breadth_fall   >= DIVERGENCE_BREADTH_FALL
    all_met     = price_met and breadth_met and cap_ok

    sep = "─" * 72
    print(f"\n── Sell signal proximity  (as of {last_date.strftime('%Y-%m-%d')}) ──\n")
    print(f"  {'Condition':<28} {'Current':>10}  {'Need':>10}  Progress")
    print(f"  {sep}")

    # 1. Price rise
    status = "✓ MET" if price_met else f"need +{DIVERGENCE_PRICE_RISE - price_rise_pct:.1f}% more"
    print(f"  {'Price rise (' + str(DIVERGENCE_WINDOW) + 'd)':<28} "
          f"{price_rise_pct:>+9.1f}%  {DIVERGENCE_PRICE_RISE:>9.1f}%  "
          f"{bar(price_rise_pct, DIVERGENCE_PRICE_RISE)}  {status}")

    # 2. Breadth200 fall
    status = "✓ MET" if breadth_met else f"need {DIVERGENCE_BREADTH_FALL - breadth_fall:.1f} more pts"
    print(f"  {'Breadth200 fall (' + str(DIVERGENCE_WINDOW) + 'd)':<28} "
          f"{breadth_fall:>+9.1f}pt  {DIVERGENCE_BREADTH_FALL:>9.1f}pt  "
          f"{bar(breadth_fall, DIVERGENCE_BREADTH_FALL)}  {status}")

    # 3. Breadth200 cap
    status = "✓ MET" if cap_ok else f"need {breadth_now - DIVERGENCE_BREADTH_CAP:.1f}pt drop"
    print(f"  {'Breadth200 < cap':<28} "
          f"{breadth_now:>+9.1f}%   {'<' + str(DIVERGENCE_BREADTH_CAP) + '%':>9}   "
          f"{'✓ below cap' if cap_ok else '✗ above cap':32}  {status}")

    print(f"  {sep}")
    verdict = "YES — sell signal ACTIVE" if all_met else "NO  — not yet triggered"
    print(f"  All 3 conditions met: {verdict}\n")


def plot_results(df, strategy, benchmark, trades, open_trade) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 0.8]}
    )
    ax1, ax2, ax3 = axes

    fig.suptitle(
        "NASDAQ 100 Breadth Strategy  (+Voting Gate)\n"
        f"BUY: breadth200 < {BUY_B200_THRESH}%  AND  (VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW})  [≥1 of 2]\n"
        f"SELL: price rose ≥{DIVERGENCE_PRICE_RISE}% over {DIVERGENCE_WINDOW}d  AND  "
        f"breadth200 fell ≥{DIVERGENCE_BREADTH_FALL}pts  AND  breadth200 < {DIVERGENCE_BREADTH_CAP}%\n"
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=9, fontweight="bold"
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

    # ── Panel 3: NDX price ────────────────────────────────────────────────────
    ax3.plot(df.index, df["price"], color="#546E7A", linewidth=1.0, label="NASDAQ 100")
    ax3.plot(df.index, df["ma200"], color="orange",  linewidth=0.8, linestyle="--", label=f"MA{MA200_WINDOW}")
    if all_entries:
        ax3.scatter(all_entries, df["price"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=50, zorder=5)
    if all_exits:
        ax3.scatter(all_exits, df["price"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=50, zorder=5)
    ax3.set_ylabel("NDX")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out = DATA_DIR / "qqq_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NASDAQ 100 Breadth Strategy backtest")
    parser.add_argument("--start-year", type=int, default=START_YEAR,
                        metavar="YEAR", help="First year to include in the backtest (default: full history)")
    parser.add_argument("--cooldown-days", type=int, default=COOLDOWN_DAYS,
                        metavar="DAYS", help="Calendar-day cooldown after a sell before next buy (default: %(default)s)")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data()
    if args.start_year is not None:
        df = df[df.index.year >= args.start_year]
    print(f"Date range  : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    print(f"Buy signal  : breadth200 < {BUY_B200_THRESH}%")
    print(f"Vote gate   : VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW}  (≥1 of 2 must agree)")
    print(f"Sell signal : price rose ≥{DIVERGENCE_PRICE_RISE}% AND breadth200 fell ≥{DIVERGENCE_BREADTH_FALL}pts")
    print(f"              over {DIVERGENCE_WINDOW} days, while breadth200 < {DIVERGENCE_BREADTH_CAP}%")
    print(f"Costs       : ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")
    print(f"Cooldown    : {args.cooldown_days} calendar days after each sell")

    benchmark                    = run_benchmark(df)
    strategy, trades, open_trade = run_strategy(df, cooldown_days=args.cooldown_days)

    print_metrics(
        compute_metrics(strategy, trades),
        compute_metrics(benchmark),
    )

    print("\n── Strategy trades ──")
    print_trades(trades, open_trade)

    print_sell_proximity(df, open_trade)

    plot_results(df, strategy, benchmark, trades, open_trade)


if __name__ == "__main__":
    main()
