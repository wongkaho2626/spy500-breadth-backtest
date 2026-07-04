"""
QQQ / NASDAQ 100 Breadth Strategy — drawdown-based position throttle overlay

Classic risk-management overlay applied monthly to the baseline strategy's
equity curve:

  - if the account's drawdown from peak exceeds MDD_THRESH (10%) at month end,
    next month runs at 50% position
  - while throttled, another red month halves the position again (floor 25%)
  - a green month adds back position:
      "step"  variant: doubles the weight (25% → 50% → 100%)
      "full"  variant: restores 100% immediately

Compared against the un-throttled baseline on the full 2002–2026 clean era and
on each regime half separately.
"""
import numpy as np
import pandas as pd
from scipy import stats

from qqq_validation import BASE_PARAMS, INITIAL_CAPITAL, load_raw, run, sharpe

MDD_THRESH  = 0.10
CUT_WEIGHT  = 0.5
FLOOR       = 0.25
SPLIT_DATE  = pd.Timestamp("2014-01-01")


def throttle(strategy_returns: pd.Series, recovery: str,
             mdd_thresh: float = MDD_THRESH, cut: float = CUT_WEIGHT,
             floor: float = FLOOR) -> pd.Series:
    """Apply the monthly drawdown throttle to a daily strategy-return series."""
    months = strategy_returns.index.to_period("M")
    equity, peak, w = 1.0, 1.0, 1.0
    month_start_equity = 1.0
    out = np.empty(len(strategy_returns))

    for i, r in enumerate(strategy_returns.to_numpy()):
        equity *= 1 + w * r
        peak = max(peak, equity)
        out[i] = equity

        is_month_end = i == len(strategy_returns) - 1 or months[i + 1] != months[i]
        if is_month_end:
            month_ret = equity / month_start_equity - 1
            dd = 1 - equity / peak
            if w == 1.0:
                if dd > mdd_thresh:
                    w = cut
            else:
                if month_ret < 0:
                    w = max(w * cut, floor)
                else:
                    w = 1.0 if recovery == "full" else min(w * 2, 1.0)
            month_start_equity = equity

    return pd.Series(out * INITIAL_CAPITAL, index=strategy_returns.index)


def evaluate(values: pd.Series, price: pd.Series) -> dict:
    dr = values.pct_change().dropna()
    ex = (dr - price.pct_change()).dropna()
    t, pval = stats.ttest_1samp(ex, 0) if len(ex) else (np.nan, np.nan)
    mdd = ((values - values.cummax()) / values.cummax()).min()
    return {"sharpe": sharpe(dr.to_numpy()), "mult": values.iloc[-1] / values.iloc[0],
            "mdd": mdd, "ex_ann": ex.mean() * 252, "p": pval}


def report(label: str, values: pd.Series, price: pd.Series) -> None:
    m = evaluate(values, price)
    print(f"  {label:<26} Sharpe {m['sharpe']:>5.2f}  {m['mult']:>7.1f}x  "
          f"maxDD {m['mdd']:>6.1%}  excess {m['ex_ann']:>+6.1%}/yr (p={m['p']:.2f})")


GRID = {
    "mdd_thresh": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
    "cut":        [0.3, 0.5, 0.7],
    "floor":      [0.10, 0.25, 0.50],
    "recovery":   ["step", "full"],
}


def _all_combos() -> list[dict]:
    import itertools
    keys = list(GRID)
    return [dict(zip(keys, c)) for c in itertools.product(*GRID.values())]


def grid_search(base_ret: pd.Series, df: pd.DataFrame) -> None:
    combos = _all_combos()
    print(f"\n{'='*72}\nGrid search: {len(combos)} combinations "
          f"(mdd_thresh × cut × floor × recovery)\n{'='*72}")

    # ── Walk-forward: select on one half by Sharpe, report on the other ──────
    halves = {"2002–2013": df.index < SPLIT_DATE, "2014–2026": df.index >= SPLIT_DATE}
    for is_name, oos_name in [("2002–2013", "2014–2026"), ("2014–2026", "2002–2013")]:
        is_ret, oos_ret = base_ret[halves[is_name]], base_ret[halves[oos_name]]
        oos_price = df["price"][halves[oos_name]]

        best_kw, best_sr = None, -np.inf
        for kw in combos:
            rec = kw.pop("recovery")
            vals = throttle(is_ret, rec, **kw)
            kw["recovery"] = rec
            sr = sharpe(vals.pct_change().dropna().to_numpy())
            if sr > best_sr:
                best_kw, best_sr = dict(kw), sr

        rec = best_kw.pop("recovery")
        oos_vals = throttle(oos_ret, rec, **best_kw)
        best_kw["recovery"] = rec
        base_oos = INITIAL_CAPITAL * (1 + oos_ret).cumprod()

        print(f"\n── select on {is_name} → report on {oos_name} ──")
        print(f"  IS-best params: {best_kw}  (IS Sharpe {best_sr:.2f})")
        report("  OOS throttled", oos_vals, oos_price)
        report("  OOS baseline (no throttle)", base_oos, oos_price)

    # ── Full-period ranking (IN-SAMPLE — reference only, overfit by design) ──
    rows = []
    for kw in combos:
        rec = kw.pop("recovery")
        vals = throttle(base_ret, rec, **kw)
        kw["recovery"] = rec
        m = evaluate(vals, df["price"])
        rows.append({**kw, **m})
    ranked = pd.DataFrame(rows).sort_values("sharpe", ascending=False)

    print(f"\n── Full-period 2002–2026 top 10 by Sharpe (IN-SAMPLE, reference only) ──")
    print(f"  {'mdd':>5} {'cut':>4} {'floor':>5} {'recovery':>8} {'Sharpe':>7} "
          f"{'mult':>7} {'maxDD':>7} {'excess/yr':>10}")
    for _, r in ranked.head(10).iterrows():
        print(f"  {r['mdd_thresh']:>5.0%} {r['cut']:>4.1f} {r['floor']:>5.0%} "
              f"{r['recovery']:>8} {r['sharpe']:>7.2f} {r['mult']:>6.1f}x "
              f"{r['mdd']:>7.1%} {r['ex_ann']:>+10.1%}")
    base_m = evaluate(INITIAL_CAPITAL * (1 + base_ret).cumprod(), df["price"])
    print(f"  {'—':>5} {'—':>4} {'—':>5} {'baseline':>8} {base_m['sharpe']:>7.2f} "
          f"{base_m['mult']:>6.1f}x {base_m['mdd']:>7.1%} {base_m['ex_ann']:>+10.1%}")


def main() -> None:
    df = load_raw()
    base_values, _ = run(df, BASE_PARAMS)
    base_ret = base_values.pct_change().fillna(0.0)

    periods = {
        "full 2002–2026": np.ones(len(df), dtype=bool),
        "2002–2013":      df.index < SPLIT_DATE,
        "2014–2026":      df.index >= SPLIT_DATE,
    }

    for name, mask in periods.items():
        sub_ret = base_ret[mask]
        price = df["price"][mask]
        bh_dd = ((price / price.cummax()) - 1).min()
        print(f"\n=== {name} (buy&hold: {price.iloc[-1]/price.iloc[0]:.1f}x, DD {bh_dd:.0%}) ===")
        base_eq = INITIAL_CAPITAL * (1 + sub_ret).cumprod()
        report("baseline (no throttle)", base_eq, price)
        report("throttle, step recovery", throttle(sub_ret, "step"), price)
        report("throttle, full recovery", throttle(sub_ret, "full"), price)

    grid_search(base_ret, df)


if __name__ == "__main__":
    main()
