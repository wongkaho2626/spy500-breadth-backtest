"""
QQQ / NASDAQ 100 Breadth Strategy — improvement candidates, walk-forward tested

Tests three upgrade families against the baseline on the clean daily era
(2002–2026, breadth_daily.csv), with walk-forward discipline: any parameter is
selected on one half (by Sharpe) and reported on the other half only.

  A baseline      : washout entry, bearish-divergence exit (current strategy)
  B div + trail   : adds a trailing stop (sweep 20/25/30/35%)
  C div + floor   : adds a breadth-floor exit (sell if breadth < 15/20/25)
  D tiered entry  : 50% at breadth < 26, remaining 50% at breadth < 10/14/18
  E trend re-entry: while OUT, also buy when price crosses above MA200
  F E + trail     : trend re-entry with trailing stop (sweep 20/25/30/35%)

Entry gate (all variants): breadth < 26 AND (VIX > 30 OR price > MA200),
15-day cooldown after each sell — identical to the baseline.
"""
import numpy as np
import pandas as pd
from scipy import stats

from qqq_validation import (
    BASE_PARAMS, COMMISSION, INITIAL_CAPITAL, SLIPPAGE, COOLDOWN_DAYS,
    load_raw, sharpe,
)

SPLIT_DATE = pd.Timestamp("2014-01-01")
TIER1_FRACTION = 0.5


def precompute(df: pd.DataFrame) -> dict:
    p = BASE_PARAMS
    price   = df["price"].to_numpy()
    breadth = df["breadth"].to_numpy()
    vix     = df["vix"].to_numpy()
    ma200   = df["ma200"].to_numpy()

    w = int(p["div_window"])
    pp = np.concatenate([np.full(w, np.nan), price[:-w]])
    bp = np.concatenate([np.full(w, np.nan), breadth[:-w]])
    with np.errstate(invalid="ignore"):
        bearish_div = (
            ((price - pp) / pp * 100 >= p["price_rise"])
            & ((bp - breadth) >= p["breadth_fall"])
            & (breadth < p["breadth_cap"])
        )
        vote_gate = (np.isnan(vix) | (vix > p["vix_thresh"])) | (np.isnan(ma200) | (price > ma200))
        cross_up = np.zeros(len(price), dtype=bool)
        cross_up[1:] = (price[1:] > ma200[1:]) & (price[:-1] <= ma200[:-1])

    return {"price": price, "breadth": breadth, "bearish_div": bearish_div,
            "vote_gate": vote_gate, "cross_up": cross_up, "dates": df.index}


def run_flex(pre: dict, *, trail: float | None = None, floor: float | None = None,
             tier2: float | None = None, trend_reentry: bool = False,
             div_exit: bool = True) -> tuple[pd.Series, int]:
    price, breadth = pre["price"], pre["breadth"]
    bearish_div, vote_gate, cross_up, dates = (
        pre["bearish_div"], pre["vote_gate"], pre["cross_up"], pre["dates"])

    cash, shares, weight = INITIAL_CAPITAL, 0.0, 0.0
    high = 0.0
    cooldown_until = None
    n_sells = 0
    values = np.empty(len(price))

    for i in range(len(price)):
        px = price[i]
        if shares > 0:
            high = max(high, px)
            sell = (div_exit and bearish_div[i]) \
                or (trail is not None and px <= high * (1 - trail / 100)) \
                or (floor is not None and breadth[i] < floor)
            if sell:
                cash += shares * px * (1 - SLIPPAGE) - COMMISSION
                shares, weight, high = 0.0, 0.0, 0.0
                cooldown_until = dates[i] + pd.Timedelta(days=COOLDOWN_DAYS)
                n_sells += 1

        cd_ok = cooldown_until is None or dates[i] > cooldown_until
        if cd_ok and weight < 1.0:
            target = weight
            if breadth[i] < BASE_PARAMS["buy_thresh"] and vote_gate[i]:
                target = max(target, TIER1_FRACTION if tier2 is not None else 1.0)
            if tier2 is not None and breadth[i] < tier2 and vote_gate[i]:
                target = 1.0
            if trend_reentry and shares == 0 and cross_up[i]:
                target = 1.0
            if target > weight:
                portfolio = cash + shares * px * (1 - SLIPPAGE)
                invest = (target - weight) * portfolio
                invest = min(invest, cash - COMMISSION)
                if invest > 0:
                    cash -= invest + COMMISSION
                    shares += invest / (px * (1 + SLIPPAGE))
                    if weight == 0:
                        high = px
                    weight = target

        values[i] = cash + shares * px * (1 - SLIPPAGE)

    return pd.Series(values, index=dates), n_sells


def evaluate(values: pd.Series, price: pd.Series) -> dict:
    dr = values.pct_change().dropna()
    ex = (dr - price.pct_change()).dropna()
    t, pval = stats.ttest_1samp(ex, 0) if len(ex) else (np.nan, np.nan)
    mdd = ((values - values.cummax()) / values.cummax()).min()
    return {"sharpe": sharpe(dr.to_numpy()), "mult": values.iloc[-1] / values.iloc[0],
            "mdd": mdd, "ex_ann": ex.mean() * 252, "t": t, "p": pval}


FAMILIES = {
    "A baseline":       [dict()],
    "B div+trail":      [dict(trail=t) for t in (20.0, 25.0, 30.0, 35.0)],
    "C div+floor":      [dict(floor=f) for f in (15.0, 20.0, 25.0)],
    "D tiered entry":   [dict(tier2=t2) for t2 in (10.0, 14.0, 18.0)],
    "E trend re-entry": [dict(trend_reentry=True)],
    "F E+trail":        [dict(trend_reentry=True, trail=t) for t in (20.0, 25.0, 30.0, 35.0)],
}


def main() -> None:
    df = load_raw()
    print(f"Data: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")

    halves = {"2002–2013": df[df.index < SPLIT_DATE], "2014–2026": df[df.index >= SPLIT_DATE]}
    pres = {k: precompute(v) for k, v in halves.items()}

    for is_name, oos_name in [("2002–2013", "2014–2026"), ("2014–2026", "2002–2013")]:
        is_df, oos_df = halves[is_name], halves[oos_name]
        bh = evaluate(INITIAL_CAPITAL * oos_df["price"] / oos_df["price"].iloc[0], oos_df["price"])
        print(f"\n=== select on {is_name}, report on {oos_name} "
              f"(buy&hold OOS: Sharpe {bh['sharpe']:.2f}, {bh['mult']:.1f}x, DD {bh['mdd']:.0%}) ===")
        print(f"  {'family':<18}{'chosen':<28}{'IS SR':>6}{'OOS SR':>7}{'OOS mult':>9}"
              f"{'OOS DD':>8}{'excess/yr':>10}{'p':>7}{'sells':>6}")

        for fam, grid in FAMILIES.items():
            best_kw, best_sr = None, -np.inf
            for kw in grid:
                vals, _ = run_flex(pres[is_name], **kw)
                sr = sharpe(vals.pct_change().dropna().to_numpy())
                if sr > best_sr:
                    best_kw, best_sr = kw, sr
            vals, n_sells = run_flex(pres[oos_name], **best_kw)
            m = evaluate(vals, oos_df["price"])
            label = ", ".join(f"{k}={v}" for k, v in best_kw.items()) or "—"
            print(f"  {fam:<18}{label:<28}{best_sr:>6.2f}{m['sharpe']:>7.2f}{m['mult']:>8.1f}x"
                  f"{m['mdd']:>8.0%}{m['ex_ann']:>+10.1%}{m['p']:>7.2f}{n_sells:>6}")


if __name__ == "__main__":
    main()
