"""
Sweep DIVERGENCE_BREADTH_FALL threshold to find the best sell signal.
All other parameters held at qqq_backtest.py defaults.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR     = Path(__file__).parent
NDX_FILE     = DATA_DIR / "NASDAQ100.csv"
BREADTH_FILE = DATA_DIR / "S5TH.csv"
VIX_FILE     = DATA_DIR / "VIX.csv"

BUY_B200_THRESH         = 26.0
VIX_BUY_THRESH          = 30.0
MA200_WINDOW            = 200
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_CAP  = 60.0
INITIAL_CAPITAL         = 10_000.0
COMMISSION              = 1.0
SLIPPAGE                = 0.0005


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx["price"] = _parse_price(ndx["Price"])

    b200 = pd.read_csv(BREADTH_FILE)
    b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
    b200.set_index("Date", inplace=True)
    b200["breadth"] = _parse_price(b200["Price"])

    vix = pd.read_csv(VIX_FILE)
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix.set_index("Date", inplace=True)
    vix["vix"] = _parse_price(vix["Price"])

    df = ndx[["price"]].join(b200[["breadth"]], how="left").join(vix[["vix"]], how="left")
    df.sort_index(inplace=True)
    df = df[df["breadth"].notna()]
    df["vix"]   = df["vix"].ffill()
    df["ma200"] = df["price"].rolling(MA200_WINDOW).mean()

    df["vix_vote"]   = df["vix"].apply(lambda v: True if pd.isna(v) else v > VIX_BUY_THRESH)
    df["ma200_vote"] = df.apply(
        lambda r: True if pd.isna(r["ma200"]) else r["price"] > r["ma200"], axis=1)
    df["vote_gate"]  = df["vix_vote"] | df["ma200_vote"]
    return df


def run_strategy(df: pd.DataFrame, breadth_fall_thresh: float) -> dict:
    pp = df["price"].shift(DIVERGENCE_WINDOW)
    bp = df["breadth"].shift(DIVERGENCE_WINDOW)
    price_rose   = ((df["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    breadth_fell = ((bp - df["breadth"]) >= breadth_fall_thresh).fillna(False)

    position  = "OUT"
    eff_entry = raw_entry = 0.0
    entry_date = None
    trade_low  = 0.0
    portfolio  = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}
    buy_trigger = ""

    for date, row in df.iterrows():
        price  = row["price"]
        breadth = row["breadth"]
        p_rose  = bool(price_rose[date])
        b_fell  = bool(breadth_fell[date])

        if position == "OUT":
            vote_gate = bool(row["vote_gate"])
            if not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate:
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
            if p_rose and b_fell and breadth < DIVERGENCE_BREADTH_CAP:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                trades.append({
                    "entry_date": entry_date,
                    "exit_date":  date,
                    "return_pct": gross_ret * 100,
                    "max_dd_pct": (trade_low - raw_entry) / raw_entry * 100,
                })
                position = "OUT"

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    # Mark open trade
    if position == "IN":
        last_price = df["price"].iloc[-1]
        eff_last   = last_price * (1 - SLIPPAGE)
        open_ret   = (eff_last - eff_entry) / eff_entry
        portfolio_open = portfolio * (1 + open_ret)
        trades.append({
            "entry_date": entry_date,
            "exit_date":  df.index[-1],
            "return_pct": open_ret * 100,
            "max_dd_pct": (trade_low - raw_entry) / raw_entry * 100,
            "open": True,
        })
        values[df.index[-1]] = portfolio_open

    series = pd.Series(values, name="strategy")
    if len(series) < 2:
        return None

    dr    = series.pct_change().dropna()
    years = (series.index[-1] - series.index[0]).days / 365.25
    tr    = (series.iloc[-1] / series.iloc[0]) - 1
    cagr  = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
    mdd   = ((series - series.cummax()) / series.cummax()).min()
    std   = dr.std()
    sh    = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    closed = [t for t in trades if not t.get("open")]
    n      = len(closed)
    wins   = sum(1 for t in closed if t["return_pct"] > 0)
    in_days = sum((t["exit_date"] - t["entry_date"]).days for t in closed)
    tot    = (series.index[-1] - series.index[0]).days

    # Check if the trade #11 gap sell (2023-05-01) was avoided
    t11_avoided = not any(
        t["exit_date"] == pd.Timestamp("2023-05-01") for t in trades
    )

    return {
        "breadth_fall": breadth_fall_thresh,
        "total_return": tr * 100,
        "cagr":         cagr * 100,
        "max_dd":       mdd * 100,
        "sharpe":       sh,
        "final_value":  series.iloc[-1],
        "n_trades":     len(trades),
        "n_closed":     n,
        "win_rate":     wins / n * 100 if n else 0,
        "time_in_mkt":  in_days / tot * 100 if tot else 0,
        "t11_avoided":  t11_avoided,
    }


def main():
    print("Loading data...")
    df = load_data()

    # Sweep from 10 to 50 in steps of 2.5
    thresholds = [round(x * 2.5, 1) for x in range(4, 21)]  # 10.0 to 50.0

    results = []
    for thresh in thresholds:
        r = run_strategy(df, thresh)
        if r:
            results.append(r)

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    hdr = (f"{'Fall pts':>9}  {'Tot Ret%':>9}  {'CAGR%':>6}  {'MaxDD%':>7}  "
           f"{'Sharpe':>7}  {'Final $':>12}  {'Trades':>7}  {'WinRate':>8}  "
           f"{'TimeInMkt':>10}  {'T11 gap avoided':>16}")
    sep = "─" * len(hdr)
    print(f"\nRanked by Sharpe Ratio  (sell: price≥+{DIVERGENCE_PRICE_RISE}% AND breadth fell ≥X pts AND breadth<{DIVERGENCE_BREADTH_CAP}%)\n")
    print(hdr)
    print(sep)
    for r in results:
        avoided = "YES ✓" if r["t11_avoided"] else "no"
        print(
            f"{r['breadth_fall']:>9.1f}  {r['total_return']:>+9.1f}%  {r['cagr']:>+5.1f}%  "
            f"{r['max_dd']:>+6.1f}%  {r['sharpe']:>7.2f}  ${r['final_value']:>11,.0f}  "
            f"{r['n_trades']:>7}  {r['win_rate']:>7.1f}%  {r['time_in_mkt']:>9.1f}%  "
            f"{avoided:>16}"
        )

    print(sep)
    print(f"\n{'─'*60}")
    print("Also ranked by Total Return:")
    print(f"{'─'*60}")
    results_by_tr = sorted(results, key=lambda x: x["total_return"], reverse=True)
    print(f"{'Fall pts':>9}  {'Tot Ret%':>9}  {'CAGR%':>6}  {'Sharpe':>7}  {'T11 gap avoided':>16}")
    print("─" * 55)
    for r in results_by_tr:
        avoided = "YES ✓" if r["t11_avoided"] else "no"
        print(f"{r['breadth_fall']:>9.1f}  {r['total_return']:>+9.1f}%  {r['cagr']:>+5.1f}%  "
              f"{r['sharpe']:>7.2f}  {avoided:>16}")

    print(f"\nBaseline (current, 20.0 pts): ", end="")
    base = next((r for r in results if r["breadth_fall"] == 20.0), None)
    if base:
        print(f"Total Return {base['total_return']:+.1f}%  CAGR {base['cagr']:+.1f}%  Sharpe {base['sharpe']:.2f}")


if __name__ == "__main__":
    main()
