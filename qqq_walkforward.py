"""
QQQ / NASDAQ 100 Breadth Strategy — walk-forward + cross-asset validation

Test 1 — Walk-forward:
  Optimize parameters by grid search on 1990–2008 only, evaluate frozen on
  2009–2026 (and the reverse split). Report IS/OOS Sharpe and the efficiency
  ratio (OOS SR / IS SR); > 0.5 = generalizes, < 0.3 = overfit.

Test 2 — Cross-asset:
  Run the *identical, untuned* baseline rules (breadth < 26 + VIX/MA200 vote,
  bearish-divergence exit) on SPX and Russell 3000 using the same S5TH breadth
  and VIX inputs. The signal working on other indices with frozen parameters
  is stronger evidence than any single-index statistic.

Reuses the strategy engine from qqq_validation.py.
"""
import itertools

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from qqq_validation import (
    BASE_PARAMS, INITIAL_CAPITAL, MA200_WINDOW,
    _load_price_csv, load_raw, run, sharpe,
)

DATA_DIR = Path(__file__).parent

SPLIT_DATE = pd.Timestamp("2014-01-01")  # midpoint of the clean daily era (2002–2026)

GRID = {
    "buy_thresh":   [18.0, 22.0, 26.0, 30.0, 34.0],
    "vix_thresh":   [25.0, 30.0, 35.0],
    "div_window":   [40, 50, 60, 70, 80],
    "price_rise":   [2.0, 3.0, 4.0, 5.0],
    "breadth_fall": [15.0, 20.0, 25.0],
    "breadth_cap":  [50.0, 55.0, 60.0, 65.0],
}
MIN_TRADES = 2   # require at least this many closed trades in-sample


def strategy_sharpe(df: pd.DataFrame, p: dict) -> tuple[float, int]:
    values, trades = run(df, p)
    return sharpe(values.pct_change().dropna().to_numpy()), len(trades)


def optimize(df: pd.DataFrame) -> tuple[dict, float]:
    best_p, best_sr = None, -np.inf
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(keys, combo))
        sr, n_trades = strategy_sharpe(df, p)
        if n_trades >= MIN_TRADES and sr > best_sr:
            best_p, best_sr = p, sr
    return best_p, best_sr


def excess_ttest(values: pd.Series, price: pd.Series) -> tuple[float, float, float]:
    bench = INITIAL_CAPITAL * price / price.iloc[0]
    ex = (values.pct_change() - bench.pct_change()).dropna()
    t, pval = stats.ttest_1samp(ex, 0)
    return t, pval, ex.mean() * 252


def walk_forward(df: pd.DataFrame) -> None:
    early, late = df[df.index < SPLIT_DATE], df[df.index >= SPLIT_DATE]
    splits = [("2002–2013 → 2014–2026", early, late),
              ("2014–2026 → 2002–2013", late, early)]

    print("\n=== Test 1: Walk-forward ===")
    print(f"Grid: {np.prod([len(v) for v in GRID.values()]):,} combinations, "
          f"optimized by Sharpe (min {MIN_TRADES} trades)")

    for name, is_df, oos_df in splits:
        best_p, is_sr = optimize(is_df)
        oos_sr, oos_trades = strategy_sharpe(oos_df, best_p)
        base_oos_sr, _ = strategy_sharpe(oos_df, BASE_PARAMS)
        bh_oos_sr = sharpe(oos_df["price"].pct_change().dropna().to_numpy())
        eff = oos_sr / is_sr if is_sr > 0 else float("nan")

        print(f"\n  ── {name} ──")
        print(f"  IS-optimal params : {best_p}")
        print(f"  IS Sharpe         : {is_sr:.2f}")
        print(f"  OOS Sharpe (frozen params)  : {oos_sr:.2f}   ({oos_trades} trades)")
        print(f"  Efficiency ratio (OOS/IS)   : {eff:.2f}")
        print(f"  Reference — current params OOS Sharpe: {base_oos_sr:.2f}, "
              f"buy&hold OOS Sharpe: {bh_oos_sr:.2f}")


def load_asset(name: str) -> pd.Series:
    if name == "Russell3000":
        df = pd.read_csv(DATA_DIR / "Russell3000.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["close"].sort_index()
    return _load_price_csv(f"{name}.csv")


def cross_asset(base_df: pd.DataFrame) -> None:
    print("\n=== Test 2: Cross-asset, identical untuned rules ===")
    breadth, vix = base_df["breadth"], base_df["vix"]

    for asset in ["NASDAQ100", "SPX", "Russell3000"]:
        price = load_asset(asset).rename("price")
        df = pd.concat([price, breadth, vix], axis=1, sort=True)
        df = df[df["breadth"].notna() & df["price"].notna()]
        df["vix"] = df["vix"].ffill()
        df["ma200"] = df["price"].rolling(MA200_WINDOW).mean()
        df["risk_on"] = np.nan  # regime filter unused here

        values, trades = run(df, BASE_PARAMS)
        sr = sharpe(values.pct_change().dropna().to_numpy())
        bh_sr = sharpe(df["price"].pct_change().dropna().to_numpy())
        t, pval, ex_ann = excess_ttest(values, df["price"])
        mult = values.iloc[-1] / INITIAL_CAPITAL
        bh_mult = df["price"].iloc[-1] / df["price"].iloc[0]

        print(f"\n  ── {asset} ({df.index[0].date()} → {df.index[-1].date()}) ──")
        print(f"  Strategy: Sharpe {sr:.2f}, {len(trades)} trades, {mult:,.0f}x")
        print(f"  Buy&hold: Sharpe {bh_sr:.2f}, {bh_mult:,.0f}x")
        print(f"  Excess vs buy&hold: {ex_ann:+.1%}/yr  (t = {t:+.2f}, p = {pval:.3f})")


def main() -> None:
    df = load_raw()
    print(f"Data: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    walk_forward(df)
    cross_asset(df)


if __name__ == "__main__":
    main()
