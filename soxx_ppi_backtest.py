import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR     = Path(__file__).parent
SOXX_FILE    = DATA_DIR / "SOXX ETF Stock Price History.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"
B50_FILE     = DATA_DIR / "S&P 500 Stocks Above 50-Day Average Historical Data.csv"
PPI_FILE     = DATA_DIR / "us_ppi_yoy.csv"

BUY_THRESHOLD           = 18.0
BUY_50_THRESHOLD        = 25.0
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 5.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 50.0

PPI_BUY_MAX  = 4.0   # % yoy — skip breadth-panic buy if PPI is above this
PPI_SELL_MIN = 0.0   # % yoy — bearish div fires regardless of PPI level (buy gate does the work)

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005


def _parse_price(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    soxx_raw    = pd.read_csv(SOXX_FILE)
    breadth_raw = pd.read_csv(BREADTH_FILE)
    b50_raw     = pd.read_csv(B50_FILE)

    for df in (soxx_raw, breadth_raw, b50_raw):
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["Price"] = _parse_price(df["Price"])

    merged = soxx_raw[["Price"]].join(
        breadth_raw[["Price"]], lsuffix="_soxx", rsuffix="_breadth", how="inner"
    )
    merged = merged.rename(columns={"Price_soxx": "soxx_price", "Price_breadth": "breadth"})
    merged = merged.join(b50_raw[["Price"]].rename(columns={"Price": "b50"}), how="inner")
    merged.sort_index(inplace=True)

    ppi_raw = pd.read_csv(PPI_FILE, parse_dates=["date"])
    ppi_raw = ppi_raw.set_index("date")[["ppi_yoy_pct"]].rename(
        columns={"ppi_yoy_pct": "ppi"}
    ).sort_index()
    merged["ppi"] = ppi_raw["ppi"].reindex(merged.index, method="ffill")

    price_past   = merged["soxx_price"].shift(DIVERGENCE_WINDOW)
    breadth_past = merged["breadth"].shift(DIVERGENCE_WINDOW)
    div_base = (
        ((merged["soxx_price"] - price_past) / price_past * 100 >= DIVERGENCE_PRICE_RISE) &
        ((breadth_past - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL) &
        (merged["breadth"] < DIVERGENCE_BREADTH_CAP)
    )
    merged["bearish_div"] = div_base & (merged["ppi"] >= PPI_SELL_MIN)

    return merged


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


def run_strategy(df: pd.DataFrame) -> tuple[pd.Series, list[dict], dict | None]:
    position   = "OUT"
    eff_entry  = raw_entry = 0.0
    entry_date = None
    trade_high = trade_low = 0.0
    entry_ppi  = 0.0
    portfolio  = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        price       = row["soxx_price"]
        breadth     = row["breadth"]
        b50         = row["b50"]
        bearish_div = bool(row["bearish_div"])
        ppi         = row["ppi"] if not pd.isna(row["ppi"]) else 0.0

        if position == "OUT" and breadth < BUY_THRESHOLD and b50 < BUY_50_THRESHOLD and ppi <= PPI_BUY_MAX:
            portfolio -= COMMISSION
            eff_entry  = price * (1 + SLIPPAGE)
            raw_entry  = price
            entry_date = date
            entry_ppi  = ppi
            trade_high = trade_low = price
            position   = "IN"
        elif position == "IN":
            trade_high = max(trade_high, price)
            trade_low  = min(trade_low, price)
            if bearish_div:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                trades.append({
                    "entry_date":       entry_date,
                    "exit_date":        date,
                    "entry_price":      raw_entry,
                    "exit_price":       price,
                    "entry_ppi":        entry_ppi,
                    "exit_ppi":         ppi,
                    "return_pct":       gross_ret * 100,
                    "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
                    "accumulated":      portfolio,
                    "sell_reason":      "bearish-div+inflation",
                })
                position = "OUT"

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    open_trade = None
    if position == "IN":
        last_price = df["soxx_price"].iloc[-1]
        last_date  = df.index[-1]
        last_ppi   = df["ppi"].iloc[-1]
        eff_last   = last_price * (1 - SLIPPAGE)
        open_trade = {
            "entry_date":    entry_date,
            "entry_price":   raw_entry,
            "entry_ppi":     entry_ppi,
            "current_date":  last_date,
            "current_price": last_price,
            "current_ppi":   last_ppi,
            "return_pct":    (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "accumulated":   portfolio * (eff_last / eff_entry),
        }

    return pd.Series(values, name="strategy"), trades, open_trade


def run_benchmark(df: pd.DataFrame) -> pd.Series:
    first_price = df["soxx_price"].iloc[0]
    return (INITIAL_CAPITAL * df["soxx_price"] / first_price).rename("benchmark")


def compute_metrics(values: pd.Series, trades: list[dict] | None = None) -> dict:
    daily_returns = values.pct_change().dropna()
    total_return  = (values.iloc[-1] / values.iloc[0]) - 1
    years         = (values.index[-1] - values.index[0]).days / 365.25
    cagr          = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    rolling_max   = values.cummax()
    max_drawdown  = ((values - rolling_max) / rolling_max).min()
    std           = daily_returns.std()
    sharpe        = (daily_returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    metrics = {
        "Total Return": f"{total_return:.1%}",
        "CAGR":         f"{cagr:.1%}",
        "Max Drawdown": f"{max_drawdown:.1%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Final Value":  f"${values.iloc[-1]:,.0f}",
    }

    if trades is not None:
        n        = len(trades)
        win_rate = sum(1 for t in trades if t["return_pct"] > 0) / n if n else 0.0
        in_days  = sum((t["exit_date"] - t["entry_date"]).days for t in trades)
        tot_days = (values.index[-1] - values.index[0]).days
        metrics.update({
            "# Trades":       str(n),
            "Win Rate":       f"{win_rate:.1%}",
            "Time in Market": f"{in_days / tot_days:.1%}" if tot_days else "—",
        })

    return metrics


def print_metrics(strat: dict, bench: dict) -> None:
    all_keys = list(dict.fromkeys(list(strat) + list(bench)))
    col      = 16
    header   = f"{'Metric':<22}{'Strategy':>{col}}{'Buy & Hold':>{col}}"
    sep      = "=" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for key in all_keys:
        print(f"  {key:<20}{strat.get(key, '—'):>{col}}{bench.get(key, '—'):>{col}}")
    print(sep)


def print_trades(trades: list[dict], open_trade: dict | None = None) -> None:
    if not trades and not open_trade:
        print("\nNo completed trades in dataset.")
        return
    header = (
        f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Entry $':>8}  {'Exit $':>8}"
        f"  {'PPI@Buy':>7}  {'PPI@Sell':>8}  {'Return':>8}  {'Drawdown':>9}  "
        f"{'Portfolio $':>12}  Sell Reason"
    )
    print(header)
    print("-" * len(header))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['entry_price']:>8.2f}  {t['exit_price']:>8.2f}  "
            f"{t['entry_ppi']:>6.1f}%  {t['exit_ppi']:>7.1f}%  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+8.1f}%  "
            f"${t['accumulated']:>11,.0f}  {t.get('sell_reason', '—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['entry_price']:>8.2f}  {open_trade['current_price']:>8.2f}  "
            f"{open_trade['entry_ppi']:>6.1f}%  {open_trade['current_ppi']:>7.1f}%  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+8.1f}%  "
            f"${open_trade['accumulated']:>11,.0f}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def plot_results(df, strategy, benchmark, trades, open_trade) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
    )
    fig.suptitle(
        f"SOXX: Breadth + PPI Strategy\n"
        f"Buy: breadth200<{BUY_THRESHOLD} AND breadth50<{BUY_50_THRESHOLD} AND PPI≤{PPI_BUY_MAX}%  |  "
        f"Sell: bearish-div (PPI≥{PPI_SELL_MIN}%)\n"
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=10, fontweight="bold"
    )

    ax1.plot(benchmark.index, benchmark, label="Buy & Hold SOXX", color="#2196F3", linewidth=1.5)
    ax1.plot(strategy.index,  strategy,  label="Breadth+PPI Strategy", color="#FF5722", linewidth=1.5)

    entry_dates = [t["entry_date"] for t in trades] + ([open_trade["entry_date"]] if open_trade else [])
    exit_dates  = [t["exit_date"] for t in trades]

    if entry_dates:
        ax1.scatter(entry_dates, strategy.reindex(entry_dates, method="nearest"),
                    marker="^", color="green", s=80, zorder=5, label="Buy")
    if exit_dates:
        ax1.scatter(exit_dates, strategy.reindex(exit_dates, method="nearest"),
                    marker="v", color="red", s=80, zorder=5, label="Sell")

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df["breadth"], color="#7B1FA2", linewidth=1.2, label="% Above 200-Day MA")
    ax2.plot(df.index, df["b50"],     color="#1565C0", linewidth=1.0, linestyle="--",
             alpha=0.7, label="% Above 50-Day MA")
    ax2.axhline(BUY_THRESHOLD,    color="green",       linestyle="--", linewidth=1.2,
                label=f"Buy 200-day: <{BUY_THRESHOLD}")
    ax2.axhline(BUY_50_THRESHOLD, color="#1565C0",     linestyle=":",  linewidth=1.2,
                label=f"Buy 50-day: <{BUY_50_THRESHOLD}")
    ax2.axhline(DIVERGENCE_BREADTH_CAP, color="darkorange", linestyle=":", linewidth=1.2,
                label=f"Div cap: {DIVERGENCE_BREADTH_CAP}")
    ax2.fill_between(df.index, df["breadth"], BUY_THRESHOLD,
                     where=df["breadth"] < BUY_THRESHOLD, color="green", alpha=0.15)
    if entry_dates:
        ax2.scatter(entry_dates, df["breadth"].reindex(entry_dates, method="nearest"),
                    marker="^", color="green", s=60, zorder=5)
    if exit_dates:
        ax2.scatter(exit_dates, df["breadth"].reindex(exit_dates, method="nearest"),
                    marker="v", color="red", s=60, zorder=5)
    ax2.set_ylabel("% Stocks Above MA")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3.plot(df.index, df["ppi"], color="#E65100", linewidth=1.2, label="US PPI YoY (%)")
    ax3.axhline(PPI_BUY_MAX,  color="red",   linestyle="--", linewidth=1.2,
                label=f"Buy blocked above: {PPI_BUY_MAX}%")
    ax3.axhline(PPI_SELL_MIN, color="green", linestyle=":",  linewidth=1.2,
                label=f"Div sell floor: {PPI_SELL_MIN}%")
    ax3.axhline(0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax3.fill_between(df.index, PPI_BUY_MAX, df["ppi"],
                     where=df["ppi"] > PPI_BUY_MAX, color="red", alpha=0.12,
                     label=f"Buy-blocked zone (PPI>{PPI_BUY_MAX}%)")
    if entry_dates:
        ax3.scatter(entry_dates, df["ppi"].reindex(entry_dates, method="ffill"),
                    marker="^", color="green", s=60, zorder=5)
    if exit_dates:
        ax3.scatter(exit_dates, df["ppi"].reindex(exit_dates, method="ffill"),
                    marker="v", color="red", s=60, zorder=5)
    ax3.set_ylabel("PPI YoY (%)")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out_path = DATA_DIR / "soxx_ppi_performance.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out_path}")


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    print(
        f"Strategy   : buy breadth200<{BUY_THRESHOLD} AND breadth50<{BUY_50_THRESHOLD} AND PPI≤{PPI_BUY_MAX}% | "
        f"sell bearish-div (window={DIVERGENCE_WINDOW}d, SOXX+{DIVERGENCE_PRICE_RISE}%, "
        f"breadth200↓≥{DIVERGENCE_BREADTH_FALL}pts, cap<{DIVERGENCE_BREADTH_CAP}, PPI≥{PPI_SELL_MIN}%)"
    )
    print(f"Costs      : ${COMMISSION:.0f} commission per side + {SLIPPAGE*100:.2f}% slippage per side")

    strategy, trades, open_trade = run_strategy(df)
    benchmark                    = run_benchmark(df)

    strat_metrics = compute_metrics(strategy, trades)
    bench_metrics = compute_metrics(benchmark)

    print_metrics(strat_metrics, bench_metrics)
    print_trades(trades, open_trade)
    plot_results(df, strategy, benchmark, trades, open_trade)


if __name__ == "__main__":
    main()
