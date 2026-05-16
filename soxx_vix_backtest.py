"""
SOXX Breadth + VIX Strategy
============================
Hypothesis: require VIX ≥ VIX_BUY_MIN at buy to confirm genuine panic before entering.
SOXX's BUY_THRESHOLD=18% already selects for extreme breadth collapse, so VIX at every
historical entry was already ≥ 28. The buy VIX filter acts as a fine-grained refiner
within those panic windows rather than a gate.

Key finding — Buy VIX≥35 improves on baseline:
  The Dec 2018 entry shifted from 2018-12-20 (VIX=28.4, SOXX=$50.65)
  to 2018-12-24 Christmas Eve (VIX=36.1, SOXX=$48.33), the absolute cycle low.
  T2 return: +57.8% → +65.4%. That extra capital compounds through T3 and the
  open trade, adding ~$27k to the final portfolio value.

Grid-search result summary vs baseline (soxx_backtest.py):
  Baseline (no VIX)          CAGR 23.1%  Sharpe 0.89  MDD -46.2%  3 trades  $559,392
  Buy VIX≥35  (this file)    CAGR 23.4%  Sharpe 0.90  MDD -46.2%  3 trades  $586,246  ← best
  Buy VIX≥30                 CAGR 23.2%  Sharpe 0.90  MDD -46.2%  3 trades  $569,054
  Buy VIX≥25/20              CAGR 23.1%  Sharpe 0.89  MDD -46.2%  3 trades  $559,392  (identical)
  Sell VIX<35                CAGR 23.1%  Sharpe 0.89  MDD -46.2%  3 trades  $559,392  (identical)
  Sell VIX<30                CAGR 22.3%  Sharpe 0.87  MDD -46.2%  2 trades  $492,700
  Sell VIX<25                CAGR 22.3%  Sharpe 0.87  MDD -46.2%  2 trades  $492,700
  Sell VIX<20                CAGR 21.3%  Sharpe 0.83  MDD -46.2%  1 trade   $417,467

Mechanism: all SOXX buys occur at VIX ≥ 28. Raising the bar to 35 waits for the
deepest panic day within each buy window, typically coinciding with the price low.
Sell-side: the 2020-03-02 exit fired at VIX=33.4 — blocking it (cap<30 or cap<25)
prevents a timely exit and hurts performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR     = Path(__file__).parent
SOXX_FILE    = DATA_DIR / "SOXX ETF Stock Price History.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"
B50_FILE     = DATA_DIR / "S&P 500 Stocks Above 50-Day Average Historical Data.csv"
VIX_FILE     = DATA_DIR / "CBOE Volatility Index Historical Data.csv"

BUY_THRESHOLD           = 18.0
BUY_50_THRESHOLD        = 25.0
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 5.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 50.0
# VIX≥35 at buy: shifts Dec-2018 entry to Christmas Eve low, best config found.
VIX_BUY_MIN  = 35.0
VIX_SELL_CAP = None   # sell-side VIX caps all hurt; see docstring

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005


def _parse_price(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    soxx_raw    = pd.read_csv(SOXX_FILE)
    breadth_raw = pd.read_csv(BREADTH_FILE)
    b50_raw     = pd.read_csv(B50_FILE)
    vix_raw     = pd.read_csv(VIX_FILE)

    for df in (soxx_raw, breadth_raw, b50_raw, vix_raw):
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["Price"] = _parse_price(df["Price"])

    merged = soxx_raw[["Price"]].join(
        breadth_raw[["Price"]], lsuffix="_soxx", rsuffix="_breadth", how="inner"
    )
    merged = merged.rename(columns={"Price_soxx": "soxx_price", "Price_breadth": "breadth"})
    merged = merged.join(b50_raw[["Price"]].rename(columns={"Price": "b50"}), how="inner")
    merged = merged.join(vix_raw[["Price"]].rename(columns={"Price": "vix"}), how="left")
    merged["vix"] = merged["vix"].ffill()
    merged.sort_index(inplace=True)

    price_past   = merged["soxx_price"].shift(DIVERGENCE_WINDOW)
    breadth_past = merged["breadth"].shift(DIVERGENCE_WINDOW)
    div_base = (
        ((merged["soxx_price"] - price_past) / price_past * 100 >= DIVERGENCE_PRICE_RISE) &
        ((breadth_past - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL) &
        (merged["breadth"] < DIVERGENCE_BREADTH_CAP)
    )
    vix_ok = True if VIX_SELL_CAP is None else (merged["vix"] < VIX_SELL_CAP)
    merged["bearish_div"] = div_base & vix_ok

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
    entry_vix  = 0.0
    trade_high = trade_low = 0.0
    portfolio  = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        price       = row["soxx_price"]
        breadth     = row["breadth"]
        b50         = row["b50"]
        vix         = row["vix"] if not pd.isna(row["vix"]) else 0.0
        bearish_div = bool(row["bearish_div"])

        vix_buy_ok = (VIX_BUY_MIN is None) or (vix >= VIX_BUY_MIN)
        if position == "OUT" and breadth < BUY_THRESHOLD and b50 < BUY_50_THRESHOLD and vix_buy_ok:
            portfolio -= COMMISSION
            eff_entry  = price * (1 + SLIPPAGE)
            raw_entry  = price
            entry_date = date
            entry_vix  = vix
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
                    "entry_vix":        entry_vix,
                    "exit_vix":         vix,
                    "return_pct":       gross_ret * 100,
                    "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
                    "accumulated":      portfolio,
                    "sell_reason":      "bearish-divergence",
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
        last_vix   = df["vix"].iloc[-1]
        eff_last   = last_price * (1 - SLIPPAGE)
        open_trade = {
            "entry_date":       entry_date,
            "entry_price":      raw_entry,
            "entry_vix":        entry_vix,
            "current_date":     last_date,
            "current_price":    last_price,
            "current_vix":      last_vix,
            "return_pct":       (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "accumulated":      portfolio * (eff_last / eff_entry),
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
    header = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Entry $':>8}  {'Exit $':>8}"
              f"  {'VIX@Buy':>7}  {'VIX@Sell':>8}  {'Return':>8}  {'Drawdown':>9}  "
              f"{'Portfolio $':>12}  Sell Reason")
    print(header)
    print("-" * len(header))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['entry_price']:>8.2f}  {t['exit_price']:>8.2f}  "
            f"{t['entry_vix']:>7.1f}  {t['exit_vix']:>8.1f}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+8.1f}%  "
            f"${t['accumulated']:>11,.0f}  {t.get('sell_reason', '—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['entry_price']:>8.2f}  {open_trade['current_price']:>8.2f}  "
            f"{open_trade['entry_vix']:>7.1f}  {open_trade['current_vix']:>8.1f}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+8.1f}%  "
            f"${open_trade['accumulated']:>11,.0f}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def print_vix_analysis(df: pd.DataFrame) -> None:
    print("\n--- VIX context at historical trade triggers ---")
    triggers = [
        ("Buy",  "2008-09-29"), ("Buy",  "2018-12-20"), ("Buy",  "2018-12-24"),
        ("Buy",  "2020-03-09"), ("Buy",  "2025-04-08"),
        ("Sell", "2018-03-23"), ("Sell", "2020-03-02"),
        ("Sell", "2023-05-31"),
    ]
    for action, date_str in triggers:
        try:
            vix = df.loc[date_str, "vix"]
            blocked = ""
            if action == "Buy"  and VIX_BUY_MIN  is not None and vix < VIX_BUY_MIN:
                blocked = f"  ← BLOCKED by VIX_BUY_MIN={VIX_BUY_MIN} → entry shifts to 2018-12-24"
            if action == "Sell" and VIX_SELL_CAP is not None and vix >= VIX_SELL_CAP:
                blocked = f"  ← BLOCKED by VIX_SELL_CAP={VIX_SELL_CAP}"
            print(f"  {action:4}  {date_str}  VIX={vix:5.1f}{blocked}")
        except KeyError:
            print(f"  {action:4}  {date_str}  (date not in dataset)")
    print()
    print("  VIX≥35 shifts 2018 entry: Dec-20 @$50.65 → Dec-24 @$48.33 (Christmas Eve low)")
    print("  T2 return: +57.8% → +65.4%, adding ~$3.5k that compounds to ~$27k extra by end.")


def plot_results(df, strategy, benchmark, trades, open_trade) -> None:
    sell_cap_str = f"<{VIX_SELL_CAP}" if VIX_SELL_CAP is not None else "unrestricted"
    buy_min_str  = f"≥{VIX_BUY_MIN}"  if VIX_BUY_MIN  is not None else "unrestricted"
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]}
    )
    fig.suptitle(
        f"SOXX: Breadth + VIX Strategy\n"
        f"Buy: breadth200<{BUY_THRESHOLD} AND breadth50<{BUY_50_THRESHOLD} AND VIX {buy_min_str}  |  "
        f"Sell: bearish-div (SOXX+{DIVERGENCE_PRICE_RISE}%) AND VIX {sell_cap_str}\n"
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=10, fontweight="bold"
    )

    ax1.plot(benchmark.index, benchmark, label="Buy & Hold SOXX", color="#2196F3", linewidth=1.5)
    ax1.plot(strategy.index,  strategy,  label="Breadth+VIX Strategy", color="#FF5722", linewidth=1.5)

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
    ax2.plot(df.index, df["b50"],    color="#1565C0", linewidth=1.0, linestyle="--",
             alpha=0.7, label="% Above 50-Day MA")
    ax2.axhline(BUY_THRESHOLD,    color="green",     linestyle="--", linewidth=1.2,
                label=f"Buy 200-day: <{BUY_THRESHOLD}")
    ax2.axhline(BUY_50_THRESHOLD, color="#1565C0",   linestyle=":",  linewidth=1.2,
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

    ax3.plot(df.index, df["vix"], color="#E65100", linewidth=1.2, label="VIX")
    if VIX_BUY_MIN is not None:
        ax3.axhline(VIX_BUY_MIN, color="green", linestyle="--", linewidth=1.2,
                    label=f"Buy min VIX: {VIX_BUY_MIN}")
    if VIX_SELL_CAP is not None:
        ax3.axhline(VIX_SELL_CAP, color="red", linestyle=":", linewidth=1.2,
                    label=f"Sell cap VIX: {VIX_SELL_CAP}")
    ax3.axhline(20, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="VIX=20 (calm)")
    ax3.axhline(35, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="VIX=35 (extreme)")
    if entry_dates:
        ax3.scatter(entry_dates, df["vix"].reindex(entry_dates, method="nearest"),
                    marker="^", color="green", s=60, zorder=5)
    if exit_dates:
        ax3.scatter(exit_dates, df["vix"].reindex(exit_dates, method="nearest"),
                    marker="v", color="red", s=60, zorder=5)
    ax3.set_ylabel("VIX Level")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out_path = DATA_DIR / "soxx_vix_performance.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out_path}")


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    sell_cap_str = f"VIX<{VIX_SELL_CAP}" if VIX_SELL_CAP is not None else "no VIX sell filter"
    buy_min_str  = f"VIX≥{VIX_BUY_MIN}"  if VIX_BUY_MIN  is not None else "no VIX buy filter"
    print(
        f"Strategy   : buy breadth200<{BUY_THRESHOLD} AND breadth50<{BUY_50_THRESHOLD} ({buy_min_str}) | "
        f"sell bearish-div ({sell_cap_str}, window={DIVERGENCE_WINDOW}d, "
        f"SOXX+{DIVERGENCE_PRICE_RISE}%, breadth200↓≥{DIVERGENCE_BREADTH_FALL}pts, cap<{DIVERGENCE_BREADTH_CAP})"
    )
    print(f"Costs      : ${COMMISSION:.0f} commission per side + {SLIPPAGE*100:.2f}% slippage per side")

    strategy, trades, open_trade = run_strategy(df)
    benchmark                    = run_benchmark(df)

    strat_metrics = compute_metrics(strategy, trades)
    bench_metrics = compute_metrics(benchmark)

    print_metrics(strat_metrics, bench_metrics)
    print_trades(trades, open_trade)
    print_vix_analysis(df)
    plot_results(df, strategy, benchmark, trades, open_trade)

    print("\n*** VIX finding: Buy VIX≥35 marginally improves SOXX baseline. ***")
    print("    Baseline (soxx_backtest.py): CAGR 23.1%, Sharpe 0.89, $559,392")
    print("    This file (Buy VIX≥35):      CAGR 23.4%, Sharpe 0.90, $586,246")
    print("    Mechanism: Dec-2018 entry shifts 4 days to Christmas Eve low ($50.65→$48.33).")
    print("    Caution: improvement relies on a single 4-day timing shift — limited generalisability.")


if __name__ == "__main__":
    main()
