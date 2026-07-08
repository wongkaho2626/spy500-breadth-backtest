"""
Sharpe / Calmar grid over QQQ / NDX Top-1 Stock / TQQQ weights.

Companion to qqq_stock_tqqq_dca_grid_search.py: same breadth-strategy signals
(qqq_portfolio_backtest.py), same TQQQ series (tqqq_backtest.load_tqqq_data(),
actual 2010+ / simulated back to 2002), same 5%-step simplex grid (231
combos). But instead of rolling DCA window finals, this computes RATIO
metrics on the full-history strategy growth path:

  - Sharpe  = mean(daily ret) / std(daily ret) * sqrt(252)   (rf = 0)
  - CAGR, max drawdown, Calmar = CAGR / |MaxDD|

Ratios are scale-free, so the path is a pure $1-growth backtest (force-entry
on day 1, signals govern exits/re-entries, NO DCA contributions — cash
inflows would distort the daily return series). Each unit bucket's daily
mark-to-market value is computed ONCE; every weight combo is a linear blend
of the three paths, so the grid costs one pass over the 6k-day history.

Selection discipline (matches qqq_portfolio_combo_search.py): full-period
rankings are in-sample by construction. The half-split tables choose the
best weights by Sharpe/Calmar on one half of 2002-2013 / 2014-2026 and
report them on the OTHER half — trust those rows.

Output: qqq_stock_tqqq_grid_ratios.csv + stdout tables.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_portfolio_backtest as qpb
import tqqq_backtest as tb
from qqq_stock_tqqq_dca_grid_search import (
    BIG, EXECUTION_LAG, FILL_PRICE, build_arrays, weight_grid,
)

WEIGHT_STEP = 5
SPLIT_DATE  = pd.Timestamp("2014-01-01")
OUT_FILE    = Path(__file__).parent / "qqq_stock_tqqq_grid_ratios.csv"


def unit_daily_paths(A: dict, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the strategy once over rows [0, n) and return the daily
    mark-to-market value paths of three $1 unit buckets (QQQ / Top-1 Stock /
    TQQQ). Force-entry on day 1; signals govern exits and re-entries. While
    OUT a bucket sits flat in cash. Stock/TQQQ NaN closes carry the previous
    mark forward."""
    price      = A["price"]
    open_      = A["open"]
    breadth    = A["breadth"]
    tqqq_close = A["tqqq_close"]
    tqqq_open  = A["tqqq_open"]

    Vq = np.empty(n)
    Vs = np.empty(n)
    Vt = np.empty(n)

    position       = "OUT"
    cooldown_until = None
    last_sell_reason = None
    last_exit_price  = None
    stock_close_arr  = None
    stock_open_arr   = None
    ndx_high = 0.0
    macd_age = ext_age = BIG

    qqq_bucket = stock_bucket = tqqq_bucket = 1.0
    qqq_shares = stock_shares = tqqq_shares = 0.0
    stock_active = tqqq_active = False

    use_open = FILL_PRICE == "open"

    for i in range(n):
        si = i - EXECUTION_LAG
        sig_price = price[si] if si >= 0 else price[i]

        if position == "OUT":
            if i == 0:
                do_buy = True
            elif si >= 0:
                b           = breadth[si]
                cooldown_ok = cooldown_until is None or A["dates"][i] > cooldown_until
                washout_buy = (not np.isnan(b) and b < qpb.BUY_B200_THRESH
                               and A["vote_gate"][si])
                recross_ok  = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and sig_price > last_exit_price)
                trend_buy   = A["ma200_recross"][si] and recross_ok
                do_buy = cooldown_ok and (washout_buy or trend_buy)
            else:
                do_buy = False
            if do_buy:
                year   = A["years"][i]
                ticker = (A["top_holdings"].get(year)
                          or A["top_holdings"].get(year - 1))
                stock_close_arr = A["stock_close"].get(ticker) if ticker else None
                stock_open_arr  = A["stock_open"].get(ticker) if ticker else None

                qqq_px = open_[i] if use_open and not np.isnan(open_[i]) else price[i]
                stock_px = np.nan
                if stock_close_arr is not None:
                    if use_open and stock_open_arr is not None and not np.isnan(stock_open_arr[i]):
                        stock_px = stock_open_arr[i]
                    else:
                        stock_px = stock_close_arr[i]
                tqqq_px = (tqqq_open[i] if use_open and not np.isnan(tqqq_open[i])
                           else tqqq_close[i])

                stock_active = not np.isnan(stock_px)
                tqqq_active  = not np.isnan(tqqq_px)
                qqq_shares   = qqq_bucket / (qqq_px * (1 + qpb.SLIPPAGE))
                stock_shares = (stock_bucket / (stock_px * (1 + qpb.SLIPPAGE))
                                if stock_active else 0.0)
                tqqq_shares  = (tqqq_bucket / (tqqq_px * (1 + qpb.SLIPPAGE))
                                if tqqq_active else 0.0)

                ndx_high = sig_price
                macd_age = ext_age = BIG
                position = "IN"

        else:  # IN
            if sig_price > ndx_high:
                ndx_high = sig_price
            macd_age = 0 if (si >= 0 and A["macd_cross"][si]) else macd_age + 1
            ext_age  = 0 if (si >= 0 and A["ext10"][si])      else ext_age + 1
            if si >= 0:
                b           = breadth[si]
                bearish_div = (A["price_rose"][si] and A["breadth_fell"][si]
                               and not np.isnan(b) and b < qpb.DIVERGENCE_BREADTH_CAP)
            else:
                bearish_div = False
            climax    = (macd_age < qpb.CLIMAX_VOTE_WINDOW) and (ext_age < qpb.CLIMAX_VOTE_WINDOW)
            trail_hit = sig_price <= ndx_high * (1 - qpb.TRAILING_STOP_PCT / 100)

            if bearish_div or climax or trail_hit:
                qqq_px = open_[i] if use_open and not np.isnan(open_[i]) else price[i]
                stock_px = np.nan
                if stock_close_arr is not None:
                    if use_open and stock_open_arr is not None and not np.isnan(stock_open_arr[i]):
                        stock_px = stock_open_arr[i]
                    else:
                        stock_px = stock_close_arr[i]
                tqqq_px = (tqqq_open[i] if use_open and not np.isnan(tqqq_open[i])
                           else tqqq_close[i])

                qqq_bucket   = qqq_shares * qqq_px * (1 - qpb.SLIPPAGE)
                stock_bucket = (stock_shares * stock_px * (1 - qpb.SLIPPAGE)
                                if stock_active and not np.isnan(stock_px) else stock_bucket)
                tqqq_bucket  = (tqqq_shares * tqqq_px * (1 - qpb.SLIPPAGE)
                                if tqqq_active and not np.isnan(tqqq_px) else tqqq_bucket)

                position         = "OUT"
                cooldown_until   = A["dates"][i] + A["cooldown"]
                last_sell_reason = ("bearish-divergence" if bearish_div
                                    else "climax-top" if climax else "trailing-stop")
                last_exit_price  = sig_price
                qqq_shares = stock_shares = tqqq_shares = 0.0

        # daily mark-to-market at closes
        if position == "IN":
            Vq[i] = qqq_shares * price[i]
            if stock_active and stock_close_arr is not None and not np.isnan(stock_close_arr[i]):
                Vs[i] = stock_shares * stock_close_arr[i]
            else:
                Vs[i] = Vs[i - 1] if i > 0 else stock_bucket
            if tqqq_active and not np.isnan(tqqq_close[i]):
                Vt[i] = tqqq_shares * tqqq_close[i]
            else:
                Vt[i] = Vt[i - 1] if i > 0 else tqqq_bucket
        else:
            Vq[i] = qqq_bucket
            Vs[i] = stock_bucket
            Vt[i] = tqqq_bucket

    return Vq, Vs, Vt


def path_metrics(v: np.ndarray, index: pd.DatetimeIndex) -> dict:
    """CAGR, annualized Sharpe (rf=0), max drawdown, Calmar for a value path."""
    yrs  = (index[-1] - index[0]).days / 365.25
    ret  = np.diff(v) / v[:-1]
    sd   = ret.std(ddof=1)
    sharpe = ret.mean() / sd * np.sqrt(252) if sd > 0 else np.nan
    cagr = (v[-1] / v[0]) ** (1 / yrs) - 1
    peak = np.maximum.accumulate(v)
    mdd  = ((v - peak) / peak).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    return {"cagr_pct": cagr * 100, "sharpe": sharpe,
            "max_dd_pct": mdd * 100, "calmar": calmar}


def main() -> None:
    print("Loading data (breadth via breadth_daily.csv)...")
    merged, top_holdings, aligned_stocks, _tqqq, _spy, _soxx = qpb.load_data()
    stock_opens, _tqqq_o, _spy_o, _soxx_o = qpb.load_open_series(top_holdings, merged.index)

    tqqq_merged = tb.load_tqqq_data()
    tqqq_close  = tqqq_merged["price"].reindex(merged.index)
    tqqq_open   = tqqq_merged["open"].reindex(merged.index)

    L = len(merged)
    print(f"Merged rows: {L:,}  ({merged.index[0].date()} -> {merged.index[-1].date()})")

    A = build_arrays(merged, top_holdings, aligned_stocks, stock_opens,
                     tqqq_close, tqqq_open)
    Vq, Vs, Vt = unit_daily_paths(A, L)

    idx    = merged.index
    h1     = idx < SPLIT_DATE
    h2     = ~h1
    combos = weight_grid(WEIGHT_STEP)
    print(f"Weight grid: {len(combos)} combos (step {WEIGHT_STEP}%)")

    rows = []
    for (pq, ps, pt) in combos:
        v = (pq * Vq + ps * Vs + pt * Vt) / 100.0
        full = path_metrics(v, idx)
        a    = path_metrics(v[h1], idx[h1])
        b    = path_metrics(v[h2], idx[h2])
        rows.append({
            "qqq_pct": pq, "stock_pct": ps, "tqqq_pct": pt,
            "cagr_pct": round(full["cagr_pct"], 2),
            "sharpe": round(full["sharpe"], 3),
            "max_dd_pct": round(full["max_dd_pct"], 1),
            "calmar": round(full["calmar"], 3),
            "h1_sharpe": round(a["sharpe"], 3), "h1_calmar": round(a["calmar"], 3),
            "h1_cagr_pct": round(a["cagr_pct"], 2), "h1_max_dd_pct": round(a["max_dd_pct"], 1),
            "h2_sharpe": round(b["sharpe"], 3), "h2_calmar": round(b["calmar"], 3),
            "h2_cagr_pct": round(b["cagr_pct"], 2), "h2_max_dd_pct": round(b["max_dd_pct"], 1),
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)

    cols = ["qqq_pct", "stock_pct", "tqqq_pct", "cagr_pct", "sharpe",
            "max_dd_pct", "calmar"]
    print("\n== Top 10 by FULL-PERIOD Sharpe (in-sample) ==")
    print(df.nlargest(10, "sharpe")[cols].to_string(index=False))
    print("\n== Top 10 by FULL-PERIOD Calmar (in-sample) ==")
    print(df.nlargest(10, "calmar")[cols].to_string(index=False))

    ref = df[(df.qqq_pct == 60) & (df.stock_pct == 30) & (df.tqqq_pct == 10)]
    print("\n== Reference mixes ==")
    refs = df[((df.qqq_pct == 100) & (df.stock_pct == 0)) |
              ((df.qqq_pct == 60) & (df.stock_pct == 30) & (df.tqqq_pct == 10)) |
              ((df.qqq_pct == 0) & (df.stock_pct == 100)) |
              ((df.qqq_pct == 0) & (df.stock_pct == 0) & (df.tqqq_pct == 100))]
    print(refs[cols].to_string(index=False))

    print("\n== Walk-forward (choose on one half, report the OTHER half) ==")
    for metric in ("sharpe", "calmar"):
        w1 = df.loc[df[f"h1_{metric}"].idxmax()]
        w2 = df.loc[df[f"h2_{metric}"].idxmax()]
        print(f"  best {metric} on 2002-2013: "
              f"{w1.qqq_pct:.0f}/{w1.stock_pct:.0f}/{w1.tqqq_pct:.0f} "
              f"-> 2014-2026 OOS: sharpe {w1.h2_sharpe:.2f}, calmar {w1.h2_calmar:.2f}, "
              f"cagr {w1.h2_cagr_pct:.1f}%, mdd {w1.h2_max_dd_pct:.1f}%")
        print(f"  best {metric} on 2014-2026: "
              f"{w2.qqq_pct:.0f}/{w2.stock_pct:.0f}/{w2.tqqq_pct:.0f} "
              f"-> 2002-2013 OOS: sharpe {w2.h1_sharpe:.2f}, calmar {w2.h1_calmar:.2f}, "
              f"cagr {w2.h1_cagr_pct:.1f}%, mdd {w2.h1_max_dd_pct:.1f}%")
    if not ref.empty:
        r = ref.iloc[0]
        print(f"  current 60/30/10 halves: "
              f"h1 sharpe {r.h1_sharpe:.2f} / calmar {r.h1_calmar:.2f}, "
              f"h2 sharpe {r.h2_sharpe:.2f} / calmar {r.h2_calmar:.2f}")

    print(f"\nWrote {len(df)} rows -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
