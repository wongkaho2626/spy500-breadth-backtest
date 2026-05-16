import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR     = Path(__file__).parent
SPY_FILE     = DATA_DIR / "SPY ETF Stock Price History.csv"
QQQ_FILE     = DATA_DIR / "QQQ ETF Stock Price History.csv"
SOXX_FILE    = DATA_DIR / "SOXX ETF Stock Price History.csv"
IGV_FILE     = DATA_DIR / "IGV ETF Stock Price History.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005

# Optimised parameters per instrument
CONFIGS = {
    "SPY": {
        "file": SPY_FILE, "col": "spy_price",
        "buy_threshold": 18.0, "window": 100,
        "price_rise": 1.0, "breadth_fall": 20.0, "breadth_cap": 55.0,
        "trail_stop": None,
    },
    "QQQ": {
        "file": QQQ_FILE, "col": "qqq_price",
        "buy_threshold": 26.0, "window": 60,
        "price_rise": 3.0, "breadth_fall": 20.0, "breadth_cap": 60.0,
        "trail_stop": None,
    },
    "SOXX": {
        "file": SOXX_FILE, "col": "soxx_price",
        "buy_threshold": 18.0, "window": 60,
        "price_rise": 5.0, "breadth_fall": 20.0, "breadth_cap": 50.0,
        "trail_stop": None,
    },
    "IGV": {
        "file": IGV_FILE, "col": "igv_price",
        "buy_threshold": 14.0, "window": 60,
        "price_rise": 3.0, "breadth_fall": 25.0, "breadth_cap": 65.0,
        "trail_stop": None,
    },
}


def _parse_price(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", "").astype(float)


def load_instrument(cfg: dict, breadth_raw: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(cfg["file"])
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    raw.set_index("Date", inplace=True)
    raw["Price"] = _parse_price(raw["Price"])

    col = cfg["col"]
    merged = raw[["Price"]].join(breadth_raw[["Price"]], lsuffix="_etf", rsuffix="_breadth", how="inner")
    merged = merged.rename(columns={"Price_etf": col, "Price_breadth": "breadth"})
    merged.sort_index(inplace=True)

    w = cfg["window"]
    price_past   = merged[col].shift(w)
    breadth_past = merged["breadth"].shift(w)
    merged["bearish_div"] = (
        ((merged[col] - price_past) / price_past * 100 >= cfg["price_rise"]) &
        ((breadth_past - merged["breadth"]) >= cfg["breadth_fall"]) &
        (merged["breadth"] < cfg["breadth_cap"])
    )
    return merged


def run_sub_strategy(df: pd.DataFrame, cfg: dict, capital: float) -> pd.Series:
    col       = cfg["col"]
    buy_thr   = cfg["buy_threshold"]
    trail_pct = cfg["trail_stop"]

    position   = "OUT"
    eff_entry  = 0.0
    trade_high = 0.0
    portfolio  = capital
    values: dict = {}

    for date, row in df.iterrows():
        price       = row[col]
        breadth     = row["breadth"]
        bearish_div = bool(row["bearish_div"])

        if position == "OUT" and breadth < buy_thr:
            portfolio -= COMMISSION
            eff_entry  = price * (1 + SLIPPAGE)
            trade_high = price
            position   = "IN"
        elif position == "IN":
            trade_high = max(trade_high, price)
            sell_reason = None
            if bearish_div:
                sell_reason = "bearish-divergence"
            elif trail_pct and price <= trade_high * (1 - trail_pct / 100):
                sell_reason = "trail-stop"
            if sell_reason:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                position  = "OUT"

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    return pd.Series(values, name=col)


def run_portfolio(weights: dict, instrument_dfs: dict) -> pd.Series:
    common_dates = instrument_dfs["SPY"].index
    for k in instrument_dfs:
        common_dates = common_dates.intersection(instrument_dfs[k].index)

    parts = []
    for name, cfg in CONFIGS.items():
        w = weights.get(name, 0.0)
        if w == 0:
            parts.append(pd.Series(0.0, index=common_dates))
        else:
            df  = instrument_dfs[name].loc[common_dates]
            sub = run_sub_strategy(df, cfg, INITIAL_CAPITAL * w)
            parts.append(sub.reindex(common_dates).ffill())

    combined      = sum(parts)
    combined.name = "portfolio"
    return combined


def run_benchmark(instrument_dfs: dict, weights: dict) -> pd.Series:
    common_dates = instrument_dfs["SPY"].index
    for k in instrument_dfs:
        common_dates = common_dates.intersection(instrument_dfs[k].index)

    total = None
    for name, cfg in CONFIGS.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        col    = cfg["col"]
        prices = instrument_dfs[name].loc[common_dates, col]
        bh     = INITIAL_CAPITAL * w * prices / prices.iloc[0]
        total  = bh if total is None else total + bh

    if total is None:
        total = pd.Series(INITIAL_CAPITAL, index=common_dates)
    total.name = "benchmark"
    return total


def compute_metrics(values: pd.Series) -> dict:
    daily_returns = values.pct_change().dropna()
    total_return  = (values.iloc[-1] / values.iloc[0]) - 1
    years         = (values.index[-1] - values.index[0]).days / 365.25
    cagr          = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    rolling_max   = values.cummax()
    max_drawdown  = ((values - rolling_max) / rolling_max).min()
    std           = daily_returns.std()
    sharpe        = (daily_returns.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    return {
        "return": total_return * 100,
        "cagr":   cagr * 100,
        "sharpe": sharpe,
        "maxdd":  max_drawdown * 100,
        "final":  values.iloc[-1],
    }


def grid_search(instrument_dfs: dict) -> pd.DataFrame:
    step = 10
    rows = []
    for spy_w in range(0, 101, step):
        for qqq_w in range(0, 101 - spy_w, step):
            for soxx_w in range(0, 101 - spy_w - qqq_w, step):
                igv_w   = 100 - spy_w - qqq_w - soxx_w
                weights = {
                    "SPY": spy_w / 100, "QQQ": qqq_w / 100,
                    "SOXX": soxx_w / 100, "IGV": igv_w / 100,
                }
                port = run_portfolio(weights, instrument_dfs)
                m    = compute_metrics(port)
                rows.append({
                    "SPY%": spy_w, "QQQ%": qqq_w, "SOXX%": soxx_w, "IGV%": igv_w,
                    "Return": m["return"], "CAGR": m["cagr"],
                    "Sharpe": m["sharpe"], "MaxDD": m["maxdd"],
                    "Final$": m["final"],
                })
    return pd.DataFrame(rows)


def plot_top_combos(instrument_dfs: dict, results: pd.DataFrame) -> None:
    top_ret    = results.nlargest(3, "Return")
    top_sharpe = results.nlargest(3, "Sharpe")
    featured   = pd.concat([top_ret, top_sharpe]).drop_duplicates(
        subset=["SPY%", "QQQ%", "SOXX%", "IGV%"]
    )

    bh_equal = run_benchmark(instrument_dfs, {"SPY": 0.25, "QQQ": 0.25, "SOXX": 0.25, "IGV": 0.25})

    fig, ax = plt.subplots(figsize=(14, 7))
    colors  = ["#E53935", "#FB8C00", "#8E24AA", "#00897B", "#1E88E5", "#3949AB"]

    for i, (_, row) in enumerate(featured.iterrows()):
        weights = {
            "SPY": row["SPY%"] / 100, "QQQ": row["QQQ%"] / 100,
            "SOXX": row["SOXX%"] / 100, "IGV": row["IGV%"] / 100,
        }
        port  = run_portfolio(weights, instrument_dfs)
        label = (f"SPY{row['SPY%']:.0f}/QQQ{row['QQQ%']:.0f}"
                 f"/SOXX{row['SOXX%']:.0f}/IGV{row['IGV%']:.0f}"
                 f"  {row['Return']:.0f}% | Sharpe {row['Sharpe']:.2f}")
        ax.plot(port.index, port, label=label, color=colors[i % len(colors)], linewidth=1.5)

    ax.plot(bh_equal.index, bh_equal, label="Buy & Hold equal 25/25/25/25", color="black",
            linewidth=1.5, linestyle="--")

    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(
        f"SPY / QQQ / SOXX / IGV — Portfolio Combination Comparison\n(${INITIAL_CAPITAL:,.0f} initial capital)",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out = DATA_DIR / "portfolio_comparison.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def print_table(label: str, df: pd.DataFrame, sort_col: str, n: int = 10) -> None:
    print(f"\n{label}:")
    print(f"\n{'SPY%':>5} {'QQQ%':>5} {'SOXX%':>6} {'IGV%':>5}  {'Return':>9}  {'CAGR':>6}  {'Sharpe':>7}  {'MaxDD':>7}  {'Final $':>12}")
    print("-" * 85)
    for _, r in df.nlargest(n, sort_col).iterrows():
        print(f"{r['SPY%']:>5.0f} {r['QQQ%']:>5.0f} {r['SOXX%']:>6.0f} {r['IGV%']:>5.0f}  "
              f"{r['Return']:>8.1f}%  {r['CAGR']:>5.1f}%  {r['Sharpe']:>7.2f}  "
              f"{r['MaxDD']:>6.1f}%  ${r['Final$']:>11,.0f}")


def main() -> None:
    breadth_raw = pd.read_csv(BREADTH_FILE)
    breadth_raw["Date"] = pd.to_datetime(breadth_raw["Date"], format="%m/%d/%Y")
    breadth_raw.set_index("Date", inplace=True)
    breadth_raw["Price"] = _parse_price(breadth_raw["Price"])

    instrument_dfs = {name: load_instrument(cfg, breadth_raw) for name, cfg in CONFIGS.items()}

    print("Grid-searching weight combinations (10% steps)...")
    results = grid_search(instrument_dfs)

    print_table("Top 10 by Total Return", results, "Return")
    print_table("Top 10 by Sharpe Ratio", results, "Sharpe")

    eq_weights = {"SPY": 0.25, "QQQ": 0.25, "SOXX": 0.25, "IGV": 0.25}
    eq_port    = run_portfolio(eq_weights, instrument_dfs)
    eq_m       = compute_metrics(eq_port)
    print(f"\nEqual 25/25/25/25:  {eq_m['return']:.1f}%  CAGR {eq_m['cagr']:.1f}%"
          f"  Sharpe {eq_m['sharpe']:.2f}  MaxDD {eq_m['maxdd']:.1f}%  ${eq_m['final']:,.0f}")

    print("\n--- Individual strategies (reference) ---")
    for name in CONFIGS:
        w    = {k: (1.0 if k == name else 0.0) for k in CONFIGS}
        port = run_portfolio(w, instrument_dfs)
        m    = compute_metrics(port)
        print(f"{name:>5}: {m['return']:>8.1f}%  CAGR {m['cagr']:.1f}%"
              f"  Sharpe {m['sharpe']:.2f}  MaxDD {m['maxdd']:.1f}%  ${m['final']:,.0f}")

    plot_top_combos(instrument_dfs, results)


if __name__ == "__main__":
    main()
