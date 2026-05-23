"""
QQQ Walk-Forward Backtest
  Phase 1 (in-sample)  : 1990-01-01 → 2015-12-31
  Phase 2 (out-of-sample): 2016-01-01 → present

Same strategy as qqq_backtest.py:
  BUY  (OUT): breadth200 < 26%  AND  (VIX > 30 OR price > MA200)
  SELL (IN):  bearish divergence — price rose ≥3% over 60d,
              breadth200 fell ≥20pts, breadth200 < 60%
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR     = Path(__file__).parent
NDX_FILE     = DATA_DIR / "NASDAQ100.csv"
BREADTH_FILE = DATA_DIR / "S5TH.csv"
VIX_FILE     = DATA_DIR / "VIX.csv"

# ── Strategy parameters ───────────────────────────────────────────────────────
BUY_B200_THRESH         = 26.0
VIX_BUY_THRESH          = 30.0
MA200_WINDOW            = 200
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005

SPLIT_DATE = pd.Timestamp("2016-01-01")


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx.columns = [c.strip().strip('"').lstrip("﻿") for c in ndx.columns]
    ndx["Date"] = pd.to_datetime(ndx["Date"].str.strip().str.strip('"'), format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price"})
    ndx["price"] = _parse_price(ndx["price"])

    b200 = pd.read_csv(BREADTH_FILE)
    b200.columns = [c.strip().strip('"').lstrip("﻿") for c in b200.columns]
    b200["Date"] = pd.to_datetime(b200["Date"].str.strip().str.strip('"'), format="%m/%d/%Y")
    b200.set_index("Date", inplace=True)
    b200["Price"] = _parse_price(b200["Price"])

    vix = pd.read_csv(VIX_FILE)
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"].str.strip().str.strip('"'), format="%m/%d/%Y")
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


def run_strategy(df: pd.DataFrame, start_portfolio: float = INITIAL_CAPITAL
                 ) -> tuple[pd.Series, list[dict], dict | None]:
    position   = "OUT"
    eff_entry  = raw_entry = 0.0
    entry_date = None
    trade_low  = 0.0
    portfolio  = start_portfolio
    buy_trigger = ""
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        price        = row["price"]
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])

        if position == "OUT":
            vote_gate = bool(row["vote_gate"])
            if not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate:
                portfolio -= COMMISSION
                eff_entry  = price * (1 + SLIPPAGE)
                raw_entry  = price
                entry_date = date
                trade_low  = price
                position   = "IN"
                vv = bool(row["vix_vote"])
                mv = bool(row["ma200_vote"])
                buy_trigger = ("VIX" if vv else "") + ("+" if vv and mv else "") + ("MA200" if mv else "")

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
                    "buy_trigger":      buy_trigger,
                    "sell_reason":      "bearish-divergence",
                })
                position = "OUT"

        values[date] = (portfolio * (price * (1 - SLIPPAGE) / eff_entry)
                        if position == "IN" else portfolio)

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


def run_benchmark(df: pd.DataFrame, start_capital: float = INITIAL_CAPITAL) -> pd.Series:
    first = df["price"].iloc[0]
    return (start_capital * df["price"] / first).rename("benchmark")


def compute_metrics(values: pd.Series, trades: list[dict] | None = None) -> dict:
    dr    = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    tr    = (values.iloc[-1] / values.iloc[0]) - 1
    cagr  = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
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
        n    = len(trades)
        wins = sum(1 for t in trades if t["return_pct"] > 0)
        in_days = sum((t["exit_date"] - t["entry_date"]).days for t in trades)
        tot     = (values.index[-1] - values.index[0]).days
        m.update({
            "# Trades":       str(n),
            "Win Rate":       f"{wins/n:.1%}" if n else "—",
            "Time in Market": f"{in_days/tot:.1%}" if tot else "—",
        })
    return m


def print_metrics(label: str, strat: dict, bench: dict) -> None:
    keys = list(dict.fromkeys(list(strat) + list(bench)))
    col  = 16
    hdr  = f"{'Metric':<22}{'Strategy':>{col}}{'Buy & Hold':>{col}}"
    sep  = "=" * len(hdr)
    print(f"\n{sep}\n  {label}\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        print(f"  {k:<20}{strat.get(k, '—'):>{col}}{bench.get(k, '—'):>{col}}")
    print(sep)


def print_trades(trades: list[dict], open_trade: dict | None = None) -> None:
    if not trades and not open_trade:
        print("\nNo completed trades.")
        return
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Entry $':>9}  {'Exit $':>9}"
           f"  {'Return':>8}  {'Drawdown':>9}  {'Portfolio':>12}  {'Trigger':>9}  Sell reason")
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+8.1f}%  "
            f"${t['accumulated']:>11,.0f}  {t.get('buy_trigger','—'):>9}  {t.get('sell_reason','—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['entry_price']:>9.2f}  {open_trade['current_price']:>9.2f}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+8.1f}%  "
            f"${open_trade['accumulated']:>11,.0f}  {open_trade.get('buy_trigger','—'):>9}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def plot_results(df_in, df_out,
                 strat_in, strat_out,
                 bench_in, bench_out,
                 trades_in, trades_out,
                 open_in, open_out) -> None:

    # Stitch series for a continuous view
    strat_all = pd.concat([strat_in, strat_out])
    bench_all = pd.concat([bench_in, bench_out])
    df_all    = pd.concat([df_in, df_out])

    all_trades  = trades_in + trades_out
    open_trade  = open_out or open_in

    entries = [t["entry_date"] for t in all_trades] + (
        [open_trade["entry_date"]] if open_trade else [])
    exits   = [t["exit_date"]   for t in all_trades]

    fig, axes = plt.subplots(
        3, 1, figsize=(18, 13), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 0.8]}
    )
    ax1, ax2, ax3 = axes

    fig.suptitle(
        "QQQ Breadth Walk-Forward Backtest\n"
        f"IN-SAMPLE: 1990–2015   |   OUT-OF-SAMPLE: 2016–present\n"
        f"BUY: breadth200 < {BUY_B200_THRESH}%  AND  (VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW})\n"
        f"SELL: price rose ≥{DIVERGENCE_PRICE_RISE}% over {DIVERGENCE_WINDOW}d  AND  "
        f"breadth200 fell ≥{DIVERGENCE_BREADTH_FALL}pts  AND  breadth200 < {DIVERGENCE_BREADTH_CAP}%",
        fontsize=9, fontweight="bold"
    )

    # Shade in-sample vs out-of-sample
    split_x = mdates.date2num(SPLIT_DATE)
    for ax in axes:
        ax.axvspan(df_in.index[0], SPLIT_DATE, alpha=0.04, color="blue", label="In-sample")
        ax.axvspan(SPLIT_DATE, df_out.index[-1], alpha=0.04, color="orange", label="Out-of-sample")
        ax.axvline(SPLIT_DATE, color="gray", linewidth=1.2, linestyle="--")

    ax1.plot(bench_all.index, bench_all, label="Buy & Hold NDX", color="#2196F3", linewidth=1.5)
    ax1.plot(strat_all.index, strat_all, label="Strategy", color="#FF5722", linewidth=1.5)
    if entries:
        ax1.scatter(entries, strat_all.reindex(entries, method="nearest"),
                    marker="^", color="green", s=80, zorder=5, label="Buy")
    if exits:
        ax1.scatter(exits, strat_all.reindex(exits, method="nearest"),
                    marker="v", color="red", s=80, zorder=5, label="Sell")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df_all.index, df_all["breadth"], color="#7B1FA2", linewidth=1.0,
             label="% S&P 500 Above 200-Day MA")
    ax2.axhline(BUY_B200_THRESH, color="green", linestyle="--", linewidth=1.0,
                label=f"Buy gate <{BUY_B200_THRESH}%")
    ax2.axhline(DIVERGENCE_BREADTH_CAP, color="red", linestyle="--", linewidth=0.9,
                label=f"Sell cap <{DIVERGENCE_BREADTH_CAP}%")
    ax2.fill_between(df_all.index, df_all["breadth"], BUY_B200_THRESH,
                     where=df_all["breadth"] < BUY_B200_THRESH, color="green", alpha=0.12)
    if entries:
        ax2.scatter(entries, df_all["breadth"].reindex(entries, method="nearest"),
                    marker="^", color="green", s=60, zorder=5)
    if exits:
        ax2.scatter(exits, df_all["breadth"].reindex(exits, method="nearest"),
                    marker="v", color="red", s=60, zorder=5)
    ax2.set_ylabel("Breadth (%)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    ax3.plot(df_all.index, df_all["price"], color="#546E7A", linewidth=1.0, label="NASDAQ 100")
    ax3.plot(df_all.index, df_all["ma200"], color="orange",  linewidth=0.8, linestyle="--",
             label=f"MA{MA200_WINDOW}")
    if entries:
        ax3.scatter(entries, df_all["price"].reindex(entries, method="nearest"),
                    marker="^", color="green", s=50, zorder=5)
    if exits:
        ax3.scatter(exits, df_all["price"].reindex(exits, method="nearest"),
                    marker="v", color="red", s=50, zorder=5)
    ax3.set_ylabel("NDX Price")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(3))
    fig.autofmt_xdate()

    out = DATA_DIR / "qqq_walkforward.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def print_conditions() -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("  STRATEGY PARAMETERS")
    print(sep)
    print(f"\n  BUY conditions (all must be true):")
    print(f"    1. breadth200  < {BUY_B200_THRESH}%  (extreme market weakness)")
    print(f"    2. Vote gate ≥1 of:")
    print(f"         a. VIX > {VIX_BUY_THRESH}  (fear/panic spike)")
    print(f"         b. Price > {MA200_WINDOW}-day MA  (uptrend pullback)")
    print(f"\n  SELL conditions (all must be true):")
    print(f"    1. Price rose  ≥ {DIVERGENCE_PRICE_RISE}%  over {DIVERGENCE_WINDOW} trading-days")
    print(f"    2. breadth200  fell ≥ {DIVERGENCE_BREADTH_FALL} pts  over same window")
    print(f"    3. breadth200  < {DIVERGENCE_BREADTH_CAP}%  (breadth cap)")
    print(f"\n  (bearish divergence: price up, breadth deteriorating → sell)")
    print(f"\n  Costs: ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")
    print(sep)


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"Full data range: {df.index[0].date()} → {df.index[-1].date()}  ({len(df)} observations)")

    # ── Split ──────────────────────────────────────────────────────────────────
    df_in  = df[df.index < SPLIT_DATE]
    df_out = df[df.index >= SPLIT_DATE]
    print(f"In-sample    : {df_in.index[0].date()} → {df_in.index[-1].date()}  ({len(df_in)} obs)")
    print(f"Out-of-sample: {df_out.index[0].date()} → {df_out.index[-1].date()}  ({len(df_out)} obs)")

    print_conditions()

    # ── In-sample run ──────────────────────────────────────────────────────────
    bench_in              = run_benchmark(df_in)
    strat_in, trades_in, open_in = run_strategy(df_in)

    # Ending portfolio value carries into out-of-sample
    end_portfolio = strat_in.iloc[-1]

    print(f"\n{'─'*60}")
    print("  PHASE 1 — IN-SAMPLE  (1990 – 2015)")
    print_metrics("In-sample", compute_metrics(strat_in, trades_in), compute_metrics(bench_in))
    print("\n── In-sample trades ──")
    print_trades(trades_in, open_in)

    # ── Out-of-sample run (fresh capital, same params) ─────────────────────────
    bench_out = run_benchmark(df_out, start_capital=INITIAL_CAPITAL)
    strat_out, trades_out, open_out = run_strategy(df_out, start_portfolio=INITIAL_CAPITAL)

    print(f"\n{'─'*60}")
    print("  PHASE 2 — OUT-OF-SAMPLE  (2016 – present)")
    print_metrics("Out-of-sample", compute_metrics(strat_out, trades_out), compute_metrics(bench_out))
    print("\n── Out-of-sample trades ──")
    print_trades(trades_out, open_out)

    plot_results(df_in, df_out, strat_in, strat_out,
                 bench_in, bench_out,
                 trades_in, trades_out, open_in, open_out)


if __name__ == "__main__":
    main()
