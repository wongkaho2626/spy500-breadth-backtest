"""
QQQ / NASDAQ 100 Breadth Strategy — robustness upgrades test

Tests four upgrades against the current full strategy (washout+vote-gate entry;
divergence + climax-top + 25% trailing-stop exits, all signals on NDX), using
the existing validation conventions (2002–2013 / 2014–2026 halves, same
metrics):

  1. Execution lag   : signals at close, execution at NEXT day's OPEN
                       (realistic — breadth is only confirmed after the close)
  2. Total return    : strategy in/out applied to QQQ *adjusted* (dividend-
                       reinvested) prices, benchmarked against QQQ TR buy&hold
  3. Ensemble div    : divergence fires when ≥2 of the 50/60/70-day windows
                       agree (removes the div_window=60 sensitivity cliff)
  4. Partial exits   : TQQQ — climax-top sells HALF, divergence/trail sell all;
                       plus a 70/30 QQQ-core / TQQQ-satellite blend
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from qqq_validation import (
    BASE_PARAMS, COMMISSION, INITIAL_CAPITAL, SLIPPAGE, COOLDOWN_DAYS,
    load_raw, sharpe,
)

DATA_DIR    = Path(__file__).parent
SPLIT_DATE  = pd.Timestamp("2014-01-01")
EXT10_PCT   = 5.0
CLIMAX_WIN  = 10
TRAIL_PCT   = 25.0
DIV_WINDOWS = [50, 60, 70]   # ensemble variant: ≥2 of 3 must agree


def load_open() -> pd.Series:
    df = pd.read_csv(DATA_DIR / "NASDAQ100.csv", encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()
    return df["Open"].astype(str).str.replace(",", "").astype(float)


def precompute(df: pd.DataFrame, ensemble: bool) -> dict:
    p = BASE_PARAMS
    price, breadth = df["price"].to_numpy(), df["breadth"].to_numpy()
    vix, ma200 = df["vix"].to_numpy(), df["ma200"].to_numpy()

    def div_for(w: int) -> np.ndarray:
        pp = np.concatenate([np.full(w, np.nan), price[:-w]])
        bp = np.concatenate([np.full(w, np.nan), breadth[:-w]])
        with np.errstate(invalid="ignore"):
            return (((price - pp) / pp * 100 >= p["price_rise"])
                    & ((bp - breadth) >= p["breadth_fall"])
                    & (breadth < p["breadth_cap"]))

    if ensemble:
        votes = sum(div_for(w).astype(int) for w in DIV_WINDOWS)
        bearish_div = votes >= 2
    else:
        bearish_div = div_for(int(p["div_window"]))

    with np.errstate(invalid="ignore"):
        vote_gate = (np.isnan(vix) | (vix > p["vix_thresh"])) | (np.isnan(ma200) | (price > ma200))
        washout = (breadth < p["buy_thresh"]) & vote_gate

    s = pd.Series(price, index=df.index)
    macd = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    hist = (macd - macd.ewm(span=9, adjust=False).mean()).to_numpy()
    macd_cross = np.zeros(len(s), dtype=bool)
    macd_cross[1:] = (hist[1:] < 0) & (hist[:-1] >= 0)
    ext10 = (s / s.rolling(10).mean() - 1 >= EXT10_PCT / 100).fillna(False).to_numpy()

    return {"price": price, "dates": df.index, "washout": washout,
            "bearish_div": bearish_div, "macd_cross": macd_cross, "ext10": ext10}


def run_engine(pre: dict, exec_price: np.ndarray | None = None, exec_lag: int = 0,
               partial_climax: bool = False) -> tuple[pd.Series, pd.Series, int]:
    """Full strategy engine. Signals decided at close of day i; executed either
    at the same day's close (exec_lag=0, matching the current backtests) or at
    exec_price[i] where exec_price is the NEXT day's open aligned to i
    (exec_lag=1). Returns (daily equity, daily target weight, n_full_sells)."""
    ndx = pre["price"]
    px = ndx if exec_price is None else exec_price
    dates = pre["dates"]
    n = len(ndx)

    cash, shares, weight = INITIAL_CAPITAL, 0.0, 0.0
    high = 0.0
    macd_age = ext_age = 10**9
    cooldown_until = None
    n_sells = 0
    pending: tuple[str, float] | None = None   # (action, target_weight)
    values = np.empty(n)
    weights = np.empty(n)

    def do_sell(i: int, target: float) -> None:
        nonlocal cash, shares, weight, cooldown_until, n_sells
        trade_px = px[i]
        sell_frac = (weight - target) / weight
        cash += shares * sell_frac * trade_px * (1 - SLIPPAGE) - COMMISSION
        shares *= 1 - sell_frac
        if target == 0.0:
            cooldown_until = dates[i] + pd.Timedelta(days=COOLDOWN_DAYS)
            n_sells += 1
        weight = target

    def do_buy(i: int) -> None:
        nonlocal cash, shares, weight
        trade_px = px[i]
        cash -= COMMISSION
        shares = cash / (trade_px * (1 + SLIPPAGE))
        cash = 0.0
        weight = 1.0

    for i in range(n):
        # 1. execute pending T+1 order (exec_price[i] is the open after the signal)
        if pending is not None:
            action, target = pending
            if action == "sell":
                do_sell(i, target)
            else:
                do_buy(i)
            pending = None

        # 2. evaluate signals at today's close (always on NDX)
        if weight > 0:
            high = max(high, ndx[i])
            macd_age = 0 if pre["macd_cross"][i] else macd_age + 1
            ext_age = 0 if pre["ext10"][i] else ext_age + 1
            climax = (macd_age < CLIMAX_WIN) and (ext_age < CLIMAX_WIN)
            trail = ndx[i] <= high * (1 - TRAIL_PCT / 100)
            target = weight
            if pre["bearish_div"][i] or trail:
                target = 0.0
            elif climax:
                target = 0.5 if (partial_climax and weight == 1.0) else 0.0
            if target < weight:
                if exec_lag == 0:
                    do_sell(i, target)
                else:
                    pending = ("sell", target)
        elif pending is None:
            cd_ok = cooldown_until is None or dates[i] > cooldown_until
            if pre["washout"][i] and cd_ok:
                high = ndx[i]
                macd_age = ext_age = 10**9
                if exec_lag == 0:
                    do_buy(i)
                else:
                    pending = ("buy", 1.0)

        mark_px = ndx[i] if exec_price is None else ndx[i]  # mark to NDX close
        values[i] = cash + shares * (px[i] if exec_lag == 0 else ndx[i]) * (1 - SLIPPAGE)
        weights[i] = weight if pending is None else pending[1]

    return pd.Series(values, index=dates), pd.Series(weights, index=dates), n_sells


def evaluate(values: pd.Series, bench: pd.Series) -> dict:
    dr = values.pct_change().dropna()
    br = bench.pct_change().reindex(dr.index)
    ex = (dr - br).dropna()
    t, pval = stats.ttest_1samp(ex, 0) if len(ex) > 1 else (np.nan, np.nan)
    mdd = ((values - values.cummax()) / values.cummax()).min()
    return {"sharpe": sharpe(dr.to_numpy()), "mult": values.iloc[-1] / values.iloc[0],
            "mdd": mdd, "ex_ann": ex.mean() * 252, "p": pval}


def report(label: str, values: pd.Series, bench: pd.Series) -> None:
    m = evaluate(values, bench)
    print(f"  {label:<34} Sharpe {m['sharpe']:5.2f}  {m['mult']:8.1f}x  "
          f"maxDD {m['mdd']:6.1%}  excess {m['ex_ann']:+6.1%}/yr (p={m['p']:.2f})")


def weight_stream_equity(weights: pd.Series, asset_ret: pd.Series) -> pd.Series:
    """Apply a weight stream (decided at close, effective next day) to an
    asset's return stream, with slippage+commission on weight changes."""
    w = weights.reindex(asset_ret.index).ffill().fillna(0.0)
    w_eff = w.shift(1).fillna(0.0)
    port = INITIAL_CAPITAL
    out = np.empty(len(asset_ret))
    prev_w = 0.0
    for i, (r, wi) in enumerate(zip(asset_ret.to_numpy(), w_eff.to_numpy())):
        port *= 1 + wi * r
        if wi != prev_w:
            port -= abs(wi - prev_w) * port * SLIPPAGE + COMMISSION
            prev_w = wi
        out[i] = port
    return pd.Series(out, index=asset_ret.index)


def halves(idx: pd.DatetimeIndex) -> dict:
    return {"full": np.ones(len(idx), dtype=bool),
            "2002–2013": idx < SPLIT_DATE, "2014–2026": idx >= SPLIT_DATE}


def main() -> None:
    df = load_raw()
    opens = load_open().reindex(df.index)
    # exec price for a signal at close of day i = open of day i+1
    next_open = opens.shift(-1).ffill().to_numpy()

    pre = precompute(df, ensemble=False)
    pre_ens = precompute(df, ensemble=True)
    bench = INITIAL_CAPITAL * df["price"] / df["price"].iloc[0]

    # ── 1. Execution lag ─────────────────────────────────────────────────────
    print("=== 1. Execution lag: same-day close vs next-day OPEN ===")
    for name, mask in halves(df.index).items():
        sub = {k: v[mask] for k, v in pre.items()}
        b = bench[mask]
        v_close, _, _ = run_engine(sub)
        v_open, _, _ = run_engine(sub, exec_price=next_open[mask], exec_lag=1)
        print(f"\n  ── {name} ──")
        report("same-day close (current)", v_close, b)
        report("T+1 open (realistic)", v_open, b)

    # ── 2. Total-return benchmark (QQQ adjusted) ────────────────────────────
    print("\n=== 2. Total return: strategy vs QQQ dividend-adjusted buy&hold ===")
    try:
        import yfinance as yf
        qqq = yf.download("QQQ", start="1999-03-10", progress=False, auto_adjust=True)["Close"]
        if isinstance(qqq, pd.DataFrame):
            qqq = qqq.iloc[:, 0]
        qqq.index = pd.to_datetime(qqq.index)
        _, w_full, _ = run_engine(pre)
        qqq_ret = qqq.pct_change().dropna()
        qqq_ret = qqq_ret[qqq_ret.index >= df.index[0]]
        strat_tr = weight_stream_equity(w_full, qqq_ret)
        bench_tr = INITIAL_CAPITAL * (1 + qqq_ret).cumprod()
        for name, m in halves(strat_tr.index).items():
            print(f"\n  ── {name} ──")
            report("strategy on QQQ total-return", strat_tr[m], bench_tr[m])
            report("QQQ TR buy&hold", bench_tr[m], bench_tr[m])
    except Exception as e:
        print(f"  [skipped: QQQ download failed: {e}]")

    # ── 3. Ensemble divergence windows ───────────────────────────────────────
    print("\n=== 3. Divergence window: single 60d vs ensemble 2-of-{50,60,70} ===")
    for name, mask in halves(df.index).items():
        sub = {k: v[mask] for k, v in pre.items()}
        sub_e = {k: v[mask] for k, v in pre_ens.items()}
        b = bench[mask]
        v1, _, s1 = run_engine(sub)
        v2, _, s2 = run_engine(sub_e)
        print(f"\n  ── {name} ──")
        report(f"single 60d window ({s1} sells)", v1, b)
        report(f"ensemble 2-of-3 ({s2} sells)", v2, b)

    # ── 4. TQQQ partial exits + core/satellite ──────────────────────────────
    print("\n=== 4. TQQQ: partial climax exits and 70/30 core-satellite ===")
    try:
        import yfinance as yf
        tqqq = yf.download("TQQQ", start="2010-01-01", progress=False, auto_adjust=True)["Close"]
        if isinstance(tqqq, pd.DataFrame):
            tqqq = tqqq.iloc[:, 0]
        tqqq.index = pd.to_datetime(tqqq.index)
        tqqq_ret = tqqq.pct_change().dropna()

        _, w_all, _ = run_engine(pre)                        # all-or-nothing
        _, w_part, _ = run_engine(pre, partial_climax=True)  # climax sells half

        eq_all = weight_stream_equity(w_all, tqqq_ret)
        eq_part = weight_stream_equity(w_part, tqqq_ret)

        qqq = yf.download("QQQ", start="2010-01-01", progress=False, auto_adjust=True)["Close"]
        if isinstance(qqq, pd.DataFrame):
            qqq = qqq.iloc[:, 0]
        qqq.index = pd.to_datetime(qqq.index)
        qqq_ret = qqq.pct_change().reindex(tqqq_ret.index).fillna(0.0)
        blend_ret = 0.7 * qqq_ret + 0.3 * tqqq_ret
        eq_blend = weight_stream_equity(w_all, blend_ret)

        bench_t = INITIAL_CAPITAL * (1 + tqqq_ret).cumprod()
        report("TQQQ all-in/all-out (current)", eq_all, bench_t)
        report("TQQQ, climax sells half", eq_part, bench_t)
        report("70/30 QQQ/TQQQ blend", eq_blend, bench_t)
        report("TQQQ buy&hold", bench_t, bench_t)
    except Exception as e:
        print(f"  [skipped: download failed: {e}]")


if __name__ == "__main__":
    main()
