"""
Portfolio combination search — QQQ / NDX Top-1 Stock / TQQQ / SPY / SOXX

Uses the upgraded signal set from qqq_portfolio_backtest.py (washout+vote-gate
entry; divergence + climax-top + NDX 25% trailing-stop exits). Because the
buy/sell dates do not depend on allocation weights, each component's $1 bucket
path is computed ONCE from the actual trade periods, and every weight
combination is evaluated as a linear blend — so the full 5-asset grid
(step 10%, 1,001 combos) is cheap.

Selection discipline: best weights are chosen by Sharpe on one half of the
clean era (2002–2013 / 2014–2026) and reported on the other half. A
full-period ranking is printed for reference (in-sample by construction).

Fold rule (matches qqq_portfolio_backtest.py): if a component has no price at
a trade's entry (e.g. TQQQ before 2010), its bucket rides QQQ for that trade.

Output: qqq_portfolio_combo_results.csv + stdout tables.
"""
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from qqq_portfolio_backtest import (
    COMMISSION, INITIAL_CAPITAL, SLIPPAGE, load_data, run_strategy,
)

DATA_DIR   = Path(__file__).parent
OUT_FILE   = DATA_DIR / "qqq_portfolio_combo_results.csv"
SPLIT_DATE = pd.Timestamp("2014-01-01")
STEP       = 10   # weight grid step in percent
COMPONENTS = ["qqq", "stock", "tqqq", "spy", "soxx"]


def trade_periods(df, top_holdings, stocks, tqqq, spy, soxx) -> list[dict]:
    """Run the strategy once to get weight-independent trade periods."""
    _, trades, open_trade, _ = run_strategy(
        df, top_holdings, stocks, tqqq, spy, soxx, cooldown_days=15)
    periods = [{"entry": t["entry_date"], "exit": t["exit_date"],
                "ticker": t.get("top1_ticker")} for t in trades]
    if open_trade:
        periods.append({"entry": open_trade["entry_date"], "exit": df.index[-1],
                        "ticker": open_trade.get("top1_ticker")})
    return periods


def bucket_path(index: pd.DatetimeIndex, periods: list[dict],
                price_for) -> np.ndarray:
    """Daily value of a $1 bucket that trades the given periods in its own
    asset (falling back to NDX when the asset has no entry price)."""
    values = np.ones(len(index))
    bucket = 1.0
    pos = {d: i for i, d in enumerate(index)}
    for p in periods:
        i0, i1 = pos[p["entry"]], pos[p["exit"]]
        px = price_for(p)
        entry_px = px[i0]
        if np.isnan(entry_px) or entry_px <= 0:
            continue
        eff_entry = entry_px * (1 + SLIPPAGE)
        seg = px[i0:i1 + 1] * (1 - SLIPPAGE) / eff_entry
        seg = np.where(np.isnan(seg), 1.0, seg)
        values[i0:i1 + 1] = bucket * seg
        bucket = values[i1]
        values[i1 + 1:] = bucket
    return values


def evaluate(path: np.ndarray, index: pd.DatetimeIndex) -> dict:
    v = pd.Series(path, index=index)
    dr = v.pct_change().dropna()
    yrs = (index[-1] - index[0]).days / 365.25
    sd = dr.std()
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1
    mdd = ((v - v.cummax()) / v.cummax()).min()
    return {
        "sharpe": (dr.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        "cagr": cagr,
        "mdd": mdd,
        "calmar": cagr / abs(mdd) if mdd < 0 else float("inf"),
        "final": v.iloc[-1] * INITIAL_CAPITAL,
    }


def main() -> None:
    print("Loading data (this fetches TQQQ/SPY/SOXX)…")
    df, top_holdings, stocks, tqqq, spy, soxx = load_data()
    periods = trade_periods(df, top_holdings, stocks, tqqq, spy, soxx)
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}, "
          f"{len(periods)} trade periods")

    index = df.index
    ndx = df["price"].to_numpy()

    def series_or_ndx(s: pd.Series | None):
        arr = s.reindex(index).ffill().to_numpy() if s is not None else None
        def price_for(p: dict) -> np.ndarray:
            if arr is None or np.isnan(arr[np.searchsorted(index, p["entry"])]):
                return ndx
            return arr
        return price_for

    def stock_price_for(p: dict) -> np.ndarray:
        t = p.get("ticker")
        s = stocks.get(t) if t else None
        if s is None:
            return ndx
        arr = s.reindex(index).ffill().to_numpy()
        i0 = np.searchsorted(index, p["entry"])
        return ndx if np.isnan(arr[i0]) else arr

    paths = {
        "qqq":   bucket_path(index, periods, lambda p: ndx),
        "stock": bucket_path(index, periods, stock_price_for),
        "tqqq":  bucket_path(index, periods, series_or_ndx(tqqq)),
        "spy":   bucket_path(index, periods, series_or_ndx(spy)),
        "soxx":  bucket_path(index, periods, series_or_ndx(soxx)),
    }

    # ── enumerate weight grid ────────────────────────────────────────────────
    steps = range(0, 101, STEP)
    combos = [c for c in itertools.product(steps, repeat=len(COMPONENTS))
              if sum(c) == 100]
    print(f"Evaluating {len(combos)} weight combinations (step {STEP}%)")

    halves = {"2002–2013": index < SPLIT_DATE, "2014–2026": index >= SPLIT_DATE}

    def blend(c: tuple, mask=None) -> np.ndarray:
        p = sum(w / 100 * paths[k] for w, k in zip(c, COMPONENTS))
        if mask is not None:
            p = p[mask]
            p = p / p[0]
        return p

    # ── walk-forward selection, one pass per objective metric ────────────────
    # mdd is negative, so maximising it = smallest drawdown for every metric
    METRICS = ["calmar", "cagr", "sharpe", "mdd"]

    for is_name, oos_name in [("2002–2013", "2014–2026"), ("2014–2026", "2002–2013")]:
        m_is, m_oos = halves[is_name], halves[oos_name]
        idx_is, idx_oos = index[m_is], index[m_oos]
        is_evals = [(c, evaluate(blend(c, m_is), idx_is)) for c in combos]
        base = evaluate(blend((100, 0, 0, 0, 0), m_oos), idx_oos)
        print(f"\n── select on {is_name} → report on {oos_name} "
              f"(OOS 100% QQQ: Calmar {base['calmar']:.2f}, CAGR {base['cagr']:.1%}, "
              f"Sharpe {base['sharpe']:.2f}, maxDD {base['mdd']:.1%}) ──")
        print(f"  {'objective':<10}{'IS-best weights':<32}{'OOS Calmar':>11}"
              f"{'OOS CAGR':>9}{'OOS Sharpe':>11}{'OOS maxDD':>10}")
        for metric in METRICS:
            best_c, _ = max(is_evals, key=lambda ce: ce[1][metric])
            m = evaluate(blend(best_c, m_oos), idx_oos)
            label = "/".join(f"{k}{w}" for w, k in zip(best_c, COMPONENTS) if w)
            print(f"  {metric:<10}{label:<32}{m['calmar']:>11.2f}{m['cagr']:>9.1%}"
                  f"{m['sharpe']:>11.2f}{m['mdd']:>10.1%}")

    # ── full-period ranking (in-sample reference) ────────────────────────────
    rows = []
    for c in combos:
        m = evaluate(blend(c), index)
        rows.append({k: w for w, k in zip(c, COMPONENTS)} | m)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_FILE, index=False)

    for metric, asc in [("calmar", False), ("cagr", False), ("sharpe", False), ("mdd", False)]:
        title = "smallest maxDD" if metric == "mdd" else f"top by {metric}"
        print(f"\n── Full period 2002–2026: {title} (IN-SAMPLE reference) ──")
        print(f"  {'weights':<38}{'Calmar':>7}{'CAGR':>8}{'Sharpe':>8}{'maxDD':>8}{'final':>14}")
        for _, r in res.sort_values(metric, ascending=asc).head(5).iterrows():
            label = "/".join(f"{k}{int(r[k])}" for k in COMPONENTS if r[k] > 0)
            print(f"  {label:<38}{r['calmar']:>7.2f}{r['cagr']:>8.1%}{r['sharpe']:>8.2f}"
                  f"{r['mdd']:>8.1%}{r['final']:>14,.0f}")

    print(f"\nSaved all {len(res)} combos → {OUT_FILE.name}")


if __name__ == "__main__":
    main()
