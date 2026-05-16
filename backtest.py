import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path(__file__).parent
SPY_FILE = DATA_DIR / "S&P 500 Historical Data.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"

BUY_THRESHOLD = 18.0
INITIAL_CAPITAL = 10_000.0

# Bearish divergence: SPY higher than N days ago by >= X%, breadth lower by >= Y pts
DIVERGENCE_WINDOW = 100         # trading days (~5 months)
DIVERGENCE_SPY_RISE_PCT = 1.0   # SPY must be up >= 1% over the window
DIVERGENCE_BREADTH_FALL_PTS = 20.0  # breadth must have fallen >= 20 pts over the window
DIVERGENCE_BREADTH_CAP = 55.0   # only check divergence when breadth < 55


def _parse_price(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", "").astype(float)


def add_divergence_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Add a bearish_div column: True when SPY makes higher high but breadth makes lower high."""
    spy_past = df["spy_price"].shift(DIVERGENCE_WINDOW)
    breadth_past = df["breadth"].shift(DIVERGENCE_WINDOW)

    spy_rose = (df["spy_price"] - spy_past) / spy_past * 100 >= DIVERGENCE_SPY_RISE_PCT
    breadth_fell = (breadth_past - df["breadth"]) >= DIVERGENCE_BREADTH_FALL_PTS
    breadth_not_extreme = df["breadth"] < DIVERGENCE_BREADTH_CAP

    return df.assign(bearish_div=spy_rose & breadth_fell & breadth_not_extreme)


def load_data() -> pd.DataFrame:
    spy_raw = pd.read_csv(SPY_FILE)
    breadth_raw = pd.read_csv(BREADTH_FILE)

    for df in (spy_raw, breadth_raw):
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["Price"] = _parse_price(df["Price"])

    merged = spy_raw[["Price"]].join(
        breadth_raw[["Price"]], lsuffix="_spy", rsuffix="_breadth", how="inner"
    )
    merged = merged.rename(columns={"Price_spy": "spy_price", "Price_breadth": "breadth"})
    merged.sort_index(inplace=True)
    return add_divergence_signal(merged)


def _realised_value(trades: list[dict]) -> float:
    value = INITIAL_CAPITAL
    for t in trades:
        value *= 1 + t["return_pct"] / 100
    return value


def _sell_reason(bearish_div: bool) -> str | None:
    if bearish_div:
        return "bearish-divergence"
    return None


def run_strategy(df: pd.DataFrame) -> tuple[pd.Series, list[dict], dict | None]:
    position = "OUT"
    entry_price = 0.0
    entry_date = None
    trade_low = 0.0   # lowest SPY seen during open trade
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        spy = row["spy_price"]
        breadth = row["breadth"]
        bearish_div = bool(row["bearish_div"])

        if position == "OUT" and breadth < BUY_THRESHOLD:
            position = "IN"
            entry_price = spy
            entry_date = date
            trade_low = spy
        elif position == "IN":
            trade_low = min(trade_low, spy)
            reason = _sell_reason(bearish_div)
            if reason:
                trade_return = (spy - entry_price) / entry_price
                realised_before = _realised_value(trades)
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": spy,
                    "return_pct": trade_return * 100,
                    "max_drawdown_pct": (trade_low - entry_price) / entry_price * 100,
                    "accumulated": realised_before * (1 + trade_return),
                    "sell_reason": reason,
                })
                position = "OUT"
        elif position == "IN":
            trade_low = min(trade_low, spy)

        realised = _realised_value(trades)
        values[date] = realised * (spy / entry_price) if position == "IN" else realised

    open_trade = None
    if position == "IN":
        last_spy = df["spy_price"].iloc[-1]
        open_trade = {
            "entry_date": entry_date,
            "entry_price": entry_price,
            "current_date": df.index[-1],
            "current_price": last_spy,
            "return_pct": (last_spy - entry_price) / entry_price * 100,
            "max_drawdown_pct": (trade_low - entry_price) / entry_price * 100,
            "accumulated": _realised_value(trades) * (last_spy / entry_price),
        }

    return pd.Series(values, name="strategy"), trades, open_trade


def run_benchmark(df: pd.DataFrame) -> pd.Series:
    first_price = df["spy_price"].iloc[0]
    return (INITIAL_CAPITAL * df["spy_price"] / first_price).rename("benchmark")


def compute_metrics(values: pd.Series, trades: list[dict] | None = None) -> dict:
    daily_returns = values.pct_change().dropna()
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    rolling_max = values.cummax()
    max_drawdown = ((values - rolling_max) / rolling_max).min()
    std = daily_returns.std()
    sharpe = (daily_returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    metrics = {
        "Total Return": f"{total_return:.1%}",
        "CAGR": f"{cagr:.1%}",
        "Max Drawdown": f"{max_drawdown:.1%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Final Value": f"${values.iloc[-1]:,.0f}",
    }

    if trades is not None:
        n = len(trades)
        win_rate = sum(1 for t in trades if t["return_pct"] > 0) / n if n else 0.0
        in_days = sum((t["exit_date"] - t["entry_date"]).days for t in trades)
        total_days = (values.index[-1] - values.index[0]).days
        metrics.update({
            "# Trades": str(n),
            "Win Rate": f"{win_rate:.1%}",
            "Time in Market": f"{in_days / total_days:.1%}" if total_days else "—",
        })

    return metrics


def print_metrics(strat: dict, bench: dict) -> None:
    all_keys = list(dict.fromkeys(list(strat) + list(bench)))
    col = 16
    header = f"{'Metric':<22}{'Strategy':>{col}}{'Buy & Hold':>{col}}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for key in all_keys:
        print(f"  {key:<20}{strat.get(key, '—'):>{col}}{bench.get(key, '—'):>{col}}")
    print(sep)


def _days_str(days: int) -> str:
    years, remainder = divmod(days, 365)
    months = remainder // 30
    if years and months:
        return f"{years}y {months}m"
    if years:
        return f"{years}y"
    if months:
        return f"{months}m"
    return f"{days}d"


def print_trades(trades: list[dict], open_trade: dict | None = None) -> None:
    if not trades and not open_trade:
        print("\nNo completed trades in dataset.")
        return
    header = (f"{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Entry $':>9}  {'Exit $':>9}"
              f"  {'Return':>8}  {'Drawdown':>9}  {'Portfolio $':>12}  Sell Reason")
    print(f"\n{header}")
    print("-" * len(header))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  "
            f"{_days_str(days):>7}  "
            f"{t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}  "
            f"{t['return_pct']:>+7.1f}%  "
            f"{t['max_drawdown_pct']:>+8.1f}%  "
            f"${t['accumulated']:>11,.0f}  "
            f"{t.get('sell_reason', '—')}"
        )
    if open_trade:
        i = len(trades) + 1
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{i:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  "
            f"{_days_str(days):>7}  "
            f"{open_trade['entry_price']:>9.2f}  {open_trade['current_price']:>9.2f}  "
            f"{open_trade['return_pct']:>+7.1f}%  "
            f"{open_trade['max_drawdown_pct']:>+8.1f}%  "
            f"${open_trade['accumulated']:>11,.0f}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def plot_results(df: pd.DataFrame, strategy: pd.Series, benchmark: pd.Series, trades: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(
        f"S&P 500: Breadth Strategy\n"
        f"Buy: breadth<{BUY_THRESHOLD}  |  Sell: bearish-divergence\n"
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=12, fontweight="bold"
    )

    ax1.plot(benchmark.index, benchmark, label="Buy & Hold", color="#2196F3", linewidth=1.5)
    ax1.plot(strategy.index, strategy, label="Breadth Strategy", color="#FF5722", linewidth=1.5)

    entry_dates = [t["entry_date"] for t in trades]
    exit_dates = [t["exit_date"] for t in trades]
    if entry_dates:
        ax1.scatter(entry_dates, strategy.reindex(entry_dates),
                    marker="^", color="green", s=90, zorder=5, label="Buy signal")
    if exit_dates:
        ax1.scatter(exit_dates, strategy.reindex(exit_dates),
                    marker="v", color="red", s=90, zorder=5, label="Sell signal")

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df["breadth"], color="#7B1FA2", linewidth=1.2, label="% Above 200-Day MA")
    ax2.axhline(BUY_THRESHOLD, color="green", linestyle="--", linewidth=1.2,
                label=f"Buy threshold ({BUY_THRESHOLD})")
    ax2.fill_between(df.index, df["breadth"], BUY_THRESHOLD,
                     where=df["breadth"] < BUY_THRESHOLD, color="green", alpha=0.15)

    if entry_dates:
        ax2.scatter(entry_dates, df["breadth"].reindex(entry_dates),
                    marker="^", color="green", s=60, zorder=5)
    if exit_dates:
        ax2.scatter(exit_dates, df["breadth"].reindex(exit_dates),
                    marker="v", color="red", s=60, zorder=5)

    ax2.set_ylabel("% Stocks Above 200-MA")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out_path = DATA_DIR / "performance.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out_path}")


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")

    print(f"Strategy   : buy breadth<{BUY_THRESHOLD} | sell bearish-divergence")

    strategy, trades, open_trade = run_strategy(df)
    benchmark = run_benchmark(df)

    strat_metrics = compute_metrics(strategy, trades)
    bench_metrics = compute_metrics(benchmark)

    print_metrics(strat_metrics, bench_metrics)
    print_trades(trades, open_trade)
    plot_results(df, strategy, benchmark, trades)


if __name__ == "__main__":
    main()
