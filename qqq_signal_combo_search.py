"""
QQQ / NASDAQ 100 Breadth Strategy — bearish-signal combination search

Grid-searches ALL non-empty subsets of the 8 codable bearish signals
(255 subsets) × vote threshold N (1..3) × mode (ADD next to the baseline
divergence exit, or REPLACE it) — ~1,500 configurations per half.

Selection is walk-forward: the best combination is chosen by Sharpe on one
half of 2002–2026 and reported on the other half only. A full-period
in-sample top-10 is printed for reference, but with this many configurations
the in-sample winner is overfit BY CONSTRUCTION — only the OOS rows are
evidence.
"""
import itertools

import numpy as np
import pandas as pd
from pathlib import Path

from qqq_validation import INITIAL_CAPITAL, load_raw, sharpe
from qqq_bearish_composite import (
    VOTE_WINDOW, SPLIT_DATE, bearish_signals, evaluate, load_ohlcv, precompute, run,
)

DATA_DIR = Path(__file__).parent
N_GRID   = [1, 2, 3]

SIGNALS = ["macd_cross", "rsi_cross", "rsi_div", "rs_div",
           "ext10", "climax", "hanging_man", "outside_vol"]


def all_subsets() -> list[tuple[str, ...]]:
    subs = []
    for r in range(1, len(SIGNALS) + 1):
        subs.extend(itertools.combinations(SIGNALS, r))
    return subs


def main() -> None:
    df = load_raw()
    px = load_ohlcv()
    spx_df = pd.read_csv(DATA_DIR / "SPX.csv", encoding="utf-8-sig")
    spx_df.columns = [c.strip().strip('"') for c in spx_df.columns]
    spx_df["Date"] = pd.to_datetime(spx_df["Date"], format="%m/%d/%Y")
    spx = spx_df.set_index("Date")["Price"].astype(str).str.replace(",", "").astype(float).sort_index()

    sig = bearish_signals(px, spx)
    fired = sig.rolling(VOTE_WINDOW, min_periods=1).max().reindex(df.index).ffill().fillna(0)

    halves = {"2002–2013": df.index < SPLIT_DATE, "2014–2026": df.index >= SPLIT_DATE}
    pres   = {k: precompute(df[m]) for k, m in halves.items()}
    fires  = {k: {s: fired[s].to_numpy()[m] for s in SIGNALS} for k, m in halves.items()}
    dfs    = {k: df[m] for k, m in halves.items()}

    subsets = all_subsets()
    n_configs = sum(len([n for n in N_GRID if n <= len(s)]) for s in subsets) * 2
    print(f"Data: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Searching {len(subsets)} subsets × N∈{N_GRID} × 2 modes ≈ {n_configs} configs per half")

    def combo_exit(half: str, subset: tuple[str, ...], n: int, mode: str) -> np.ndarray:
        score = sum(fires[half][s] for s in subset)
        combo = score >= n
        return (pres[half]["bearish_div"] | combo) if mode == "ADD" else combo

    def search(half: str) -> tuple[dict, float]:
        best, best_sr = None, -np.inf
        for subset in subsets:
            for n in N_GRID:
                if n > len(subset):
                    continue
                for mode in ("ADD", "REPLACE"):
                    vals, _ = run(pres[half], combo_exit(half, subset, n, mode))
                    sr = sharpe(vals.pct_change().dropna().to_numpy())
                    if sr > best_sr:
                        best, best_sr = {"subset": subset, "n": n, "mode": mode}, sr
        return best, best_sr

    # ── Walk-forward ──────────────────────────────────────────────────────────
    for is_name, oos_name in [("2002–2013", "2014–2026"), ("2014–2026", "2002–2013")]:
        oos_df = dfs[oos_name]
        bh = evaluate(INITIAL_CAPITAL * oos_df["price"] / oos_df["price"].iloc[0], oos_df["price"])

        best, is_sr = search(is_name)
        vals, n_sells = run(pres[oos_name],
                            combo_exit(oos_name, best["subset"], best["n"], best["mode"]))
        m = evaluate(vals, oos_df["price"])
        base_vals, _ = run(pres[oos_name], pres[oos_name]["bearish_div"])
        mb = evaluate(base_vals, oos_df["price"])

        print(f"\n=== select on {is_name} → report on {oos_name} "
              f"(buy&hold OOS: Sharpe {bh['sharpe']:.2f}, {bh['mult']:.1f}x) ===")
        print(f"  IS-best: {best['mode']} N≥{best['n']} of {list(best['subset'])} "
              f"(IS Sharpe {is_sr:.2f})")
        print(f"  OOS combo   : Sharpe {m['sharpe']:.2f}, {m['mult']:.1f}x, "
              f"DD {m['mdd']:.0%}, excess {m['ex_ann']:+.1%}/yr (p={m['p']:.2f}), {n_sells} sells")
        print(f"  OOS baseline: Sharpe {mb['sharpe']:.2f}, {mb['mult']:.1f}x, "
              f"DD {mb['mdd']:.0%}, excess {mb['ex_ann']:+.1%}/yr (p={mb['p']:.2f})")

    # ── Full-period in-sample ranking (reference only) ────────────────────────
    fired_full = {s: fired[s].to_numpy() for s in SIGNALS}
    pre_full = precompute(df)
    rows = []
    for subset in subsets:
        for n in N_GRID:
            if n > len(subset):
                continue
            for mode in ("ADD", "REPLACE"):
                score = sum(fired_full[s] for s in subset)
                combo = score >= n
                exit_sig = (pre_full["bearish_div"] | combo) if mode == "ADD" else combo
                vals, n_sells = run(pre_full, exit_sig)
                m = evaluate(vals, df["price"])
                rows.append({"subset": "+".join(s[:6] for s in subset), "n": n,
                             "mode": mode, "sells": n_sells, **m})
    ranked = pd.DataFrame(rows).sort_values("sharpe", ascending=False)

    base_full, _ = run(pre_full, pre_full["bearish_div"])
    mb = evaluate(base_full, df["price"])
    print(f"\n── Full-period 2002–2026 top 10 by Sharpe "
          f"(IN-SAMPLE over {len(rows)} configs — overfit by construction) ──")
    print(f"  baseline: Sharpe {mb['sharpe']:.2f}, {mb['mult']:.1f}x, DD {mb['mdd']:.0%}")
    for _, r in ranked.head(10).iterrows():
        print(f"  {r['mode']:<7} N≥{r['n']}  {r['subset']:<40} Sharpe {r['sharpe']:.2f}  "
              f"{r['mult']:6.1f}x  DD {r['mdd']:.0%}  {r['sells']} sells")


if __name__ == "__main__":
    main()
