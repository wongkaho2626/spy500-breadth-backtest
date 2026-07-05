"""
NASDAQ 100 — Percent of Stocks Above 200-Day Average Strategy

BUY  (while OUT): pct200 < 26%
SELL (while IN):  Bearish divergence — price rose ≥ 3% over 60 days
                  while pct200 fell ≥ 20 pts AND pct200 < 60%
"""
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
BREADTH_FILE = DATA_DIR / "MMTH.csv"

# ── Buy thresholds ────────────────────────────────────────────────────────────
BUY_B200_THRESH = 26.0   # pct200 must be below this

# ── Sell — bearish divergence ─────────────────────────────────────────────────
DIVERGENCE_WINDOW       = 60    # trading days lookback
DIVERGENCE_PRICE_RISE   = 3.0   # % price rise over window
DIVERGENCE_BREADTH_FALL = 20.0  # pts pct200 drop over window
DIVERGENCE_BREADTH_CAP  = 60.0  # pct200 must be below this

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day closes; the earliest tradeable fill is the NEXT
# session. Default: a signal on day t fills at day t+1's OPEN. Set EXECUTION_LAG=0
# and FILL_PRICE="close" for the legacy same-day-close (look-ahead) fill.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

# ── Shared ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price", "Open": "open"})
    ndx["price"] = _parse_price(ndx["price"])
    ndx["open"]  = _parse_price(ndx["open"])

    b200 = pd.read_csv(BREADTH_FILE)
    b200["Date"] = pd.to_datetime(b200["time"])
    b200.set_index("Date", inplace=True)
    b200 = b200.rename(columns={"close": "breadth"})

    merged = ndx[["price", "open"]].join(b200[["breadth"]], how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]

    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)

    # Trend re-entry: fresh close back above the 200-day MA. Divergence-only exit,
    # so the gate reduces to "price back above the prior exit price".
    ma200 = merged["price"].rolling(200).mean()
    merged["ma200_recross"] = (
        (merged["price"] > ma200) & (merged["price"].shift(1) <= ma200.shift(1))
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


def run_strategy(df: pd.DataFrame, execution_lag: int = EXECUTION_LAG,
                 fill_on: str = FILL_PRICE) -> tuple[pd.Series, list[dict], dict | None]:
    """Signals are close-based; a signal on day t fills `execution_lag` bars later
    at that bar's open (fill_on="open") or close. lag=0 requires fill_on="close"
    (legacy same-day look-ahead). Mark-to-market always uses the close."""
    if fill_on == "open" and execution_lag < 1:
        raise ValueError("fill_on='open' requires execution_lag >= 1 (open precedes close)")

    position   = "OUT"
    eff_entry  = raw_entry = 0.0
    entry_date = None
    trade_low  = 0.0
    portfolio  = INITIAL_CAPITAL
    last_exit_price: float | None = None
    trades: list[dict] = []
    values: dict = {}

    pending: dict | None = None
    rows = list(df.iterrows())
    n = len(rows)

    def execute_due(i, date, fill_price):
        nonlocal position, eff_entry, raw_entry, entry_date, trade_low
        nonlocal portfolio, last_exit_price, pending
        if pending is None or pending["fill_at"] != i:
            return False
        if pending["action"] == "BUY" and position == "OUT":
            portfolio -= COMMISSION
            eff_entry  = fill_price * (1 + SLIPPAGE)
            raw_entry  = fill_price
            entry_date = date
            trade_low  = fill_price
            position = "IN"
            pending = None
            return True
        if pending["action"] == "SELL" and position == "IN":
            eff_exit  = fill_price * (1 - SLIPPAGE)
            gross_ret = (eff_exit - eff_entry) / eff_entry
            portfolio *= (1 + gross_ret)
            portfolio -= COMMISSION
            trades.append({
                "entry_date":       entry_date,
                "exit_date":        date,
                "entry_price":      raw_entry,
                "exit_price":       fill_price,
                "return_pct":       gross_ret * 100,
                "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
                "accumulated":      portfolio,
                "sell_reason":      "bearish-divergence",
            })
            position = "OUT"
            last_exit_price = fill_price
            pending = None
            return True
        pending = None
        return False

    for i in range(n):
        date, row = rows[i]
        price        = row["price"]
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])
        if fill_on == "open" and not pd.isna(row["open"]):
            fill_price = row["open"]
        else:
            fill_price = price

        executed = execute_due(i, date, fill_price)

        if not executed and pending is None:
            if position == "OUT":
                washout_buy = not pd.isna(breadth) and breadth < BUY_B200_THRESH
                # Trend re-entry: fresh MA200 recross once price is back above the
                # price we last sold at (divergence exit proven premature).
                trend_buy   = bool(row["ma200_recross"]) and (
                    last_exit_price is not None and price > last_exit_price)
                if (washout_buy or trend_buy) and i + execution_lag < n:
                    pending = {"action": "BUY", "fill_at": i + execution_lag}

            elif position == "IN":
                trade_low = min(trade_low, price)
                bearish_div = price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP
                if bearish_div and i + execution_lag < n:
                    pending = {"action": "SELL", "fill_at": i + execution_lag}

            executed = execute_due(i, date, fill_price)

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
        print(f"  {k:<20}{strat.get(k, '—'):>{col}}{bench.get(k, '—'):>{col}}")
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


def print_sell_proximity(df: pd.DataFrame, open_trade: dict | None) -> None:
    if open_trade is None:
        return

    last      = df.iloc[-1]
    last_date = df.index[-1]

    lookback_idx = max(0, len(df) - 1 - DIVERGENCE_WINDOW)
    past         = df.iloc[lookback_idx]

    price_now      = last["price"]
    price_then     = past["price"]
    price_rise_pct = (price_now - price_then) / price_then * 100

    breadth_now  = last["breadth"]
    breadth_then = past["breadth"]
    breadth_fall = breadth_then - breadth_now

    cap_ok = breadth_now < DIVERGENCE_BREADTH_CAP

    def bar(value: float, threshold: float) -> str:
        ratio  = min(value / threshold, 1.0) if threshold != 0 else 1.0
        filled = round(ratio * 20)
        return f"[{'█' * filled}{'░' * (20 - filled)}] {ratio:.0%}"

    price_met   = price_rise_pct >= DIVERGENCE_PRICE_RISE
    breadth_met = breadth_fall   >= DIVERGENCE_BREADTH_FALL
    all_met     = price_met and breadth_met and cap_ok

    sep = "─" * 72
    print(f"\n── Sell signal proximity  (as of {last_date.strftime('%Y-%m-%d')}) ──\n")
    print(f"  {'Condition':<28} {'Current':>10}  {'Need':>10}  Progress")
    print(f"  {sep}")

    status = "✓ MET" if price_met else f"need +{DIVERGENCE_PRICE_RISE - price_rise_pct:.1f}% more"
    print(f"  {'Price rise (' + str(DIVERGENCE_WINDOW) + 'd)':<28} "
          f"{price_rise_pct:>+9.1f}%  {DIVERGENCE_PRICE_RISE:>9.1f}%  "
          f"{bar(price_rise_pct, DIVERGENCE_PRICE_RISE)}  {status}")

    status = "✓ MET" if breadth_met else f"need {DIVERGENCE_BREADTH_FALL - breadth_fall:.1f} more pts"
    print(f"  {'Pct200 fall (' + str(DIVERGENCE_WINDOW) + 'd)':<28} "
          f"{breadth_fall:>+9.1f}pt  {DIVERGENCE_BREADTH_FALL:>9.1f}pt  "
          f"{bar(breadth_fall, DIVERGENCE_BREADTH_FALL)}  {status}")

    status = "✓ MET" if cap_ok else f"need {breadth_now - DIVERGENCE_BREADTH_CAP:.1f}pt drop"
    print(f"  {'Pct200 < cap':<28} "
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
        "NASDAQ 100 — Percent of Stocks Above 200-Day Average Strategy\n"
        f"BUY: pct200 < {BUY_B200_THRESH}%\n"
        f"SELL: price rose ≥{DIVERGENCE_PRICE_RISE}% over {DIVERGENCE_WINDOW}d  AND  "
        f"pct200 fell ≥{DIVERGENCE_BREADTH_FALL}pts  AND  pct200 < {DIVERGENCE_BREADTH_CAP}%\n"
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=9, fontweight="bold"
    )

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

    ax2.plot(df.index, df["breadth"], color="#7B1FA2", linewidth=1.0,
             label="% Stocks Above 200-Day MA (all markets)")
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
    ax2.set_ylabel("Pct200 (%)")
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

    out = DATA_DIR / "qqq_pct200_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"Date range  : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    print(f"Buy signal  : pct200 < {BUY_B200_THRESH}%")
    print(f"Sell signal : price rose ≥{DIVERGENCE_PRICE_RISE}% AND pct200 fell ≥{DIVERGENCE_BREADTH_FALL}pts")
    print(f"              over {DIVERGENCE_WINDOW} days, while pct200 < {DIVERGENCE_BREADTH_CAP}%")
    print(f"Costs       : ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")

    benchmark                    = run_benchmark(df)
    strategy, trades, open_trade = run_strategy(df)

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
