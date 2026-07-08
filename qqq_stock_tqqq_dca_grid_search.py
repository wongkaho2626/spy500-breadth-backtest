"""
Weight grid search over QQQ / NDX Top-1 Stock / TQQQ for the rolling-window
DCA framework of qqq60stock30tqqq10_dca_rolling.py.

Same setup: $1,000,000 initial capital + $200,000 at every 252-trading-day
anniversary, horizons 1..20 years, 1-day roll step, force-entry on window
day 1, breadth-strategy signals from qqq_portfolio_backtest.py, TQQQ leg from
tqqq_backtest.load_tqqq_data() (actual 2010+, simulated 3x-NDX-minus-drag
back to 2002).

Because entry/exit dates do not depend on allocation weights (signals are all
NDX-based) and contributions split proportionally, each window's final value
is linear in the weights — the same cheap linear-blend trick as
qqq_portfolio_combo_search.py. So the sweep runs ONCE per window tracking
three unit buckets (100% QQQ, 100% Stock, 100% TQQQ), and every weight combo
on a 5%-step simplex grid (231 combos) is evaluated as
    final(w) = w_qqq*F_qqq + w_stock*F_stock + w_tqqq*F_tqqq.
The only nonlinearity is the flat $1 commission (≤ a few dollars per trade on
a $1M+ portfolio — negligible).

Selection is IN-SAMPLE by construction (one 2002–2026 history, overlapping
windows): the "best" rows are descriptive of this sample, not out-of-sample
forecasts.

Outputs:
  qqq_stock_tqqq_dca_grid_results.csv — full grid (231 combos x 20 horizons)
  stdout — per-horizon best combos by median return and by worst window
"""
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_portfolio_backtest as qpb
import tqqq_backtest as tb

YEAR_ROWS       = 252
HORIZONS        = range(1, 21)
INITIAL_CAPITAL = 1_000_000.0
CONTRIBUTION    = 200_000.0
WEIGHT_STEP     = 5            # grid step in percent
OUT_FILE        = Path(__file__).parent / "qqq_stock_tqqq_dca_grid_results.csv"

# ── Execution timing (matches qqq60stock30tqqq10_dca_rolling.py) ─────────────
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

BIG = 10**9


def build_arrays(merged: pd.DataFrame, top_holdings: dict[int, str],
                 aligned_stocks: dict[str, pd.Series],
                 stock_opens: dict[str, pd.Series],
                 tqqq_close: pd.Series, tqqq_open: pd.Series) -> dict:
    """Flatten the merged signal frame and aligned leg series into numpy
    arrays for the hot loop."""
    return {
        "dates":         merged.index.values,
        "years":         merged.index.year.to_numpy(),
        "price":         merged["price"].to_numpy(float),
        "open":          merged["open"].to_numpy(float),
        "breadth":       merged["breadth"].to_numpy(float),
        "vote_gate":     merged["vote_gate"].to_numpy(bool),
        "price_rose":    merged["price_rose"].to_numpy(bool),
        "breadth_fell":  merged["breadth_fell"].to_numpy(bool),
        "macd_cross":    merged["macd_cross"].to_numpy(bool),
        "ext10":         merged["ext10"].to_numpy(bool),
        "ma200_recross": merged["ma200_recross"].to_numpy(bool),
        "stock_close":   {t: s.to_numpy(float) for t, s in aligned_stocks.items()},
        "stock_open":    {t: s.to_numpy(float) for t, s in stock_opens.items()},
        "tqqq_close":    tqqq_close.to_numpy(float),
        "tqqq_open":     tqqq_open.to_numpy(float),
        "top_holdings":  top_holdings,
        "cooldown":      np.timedelta64(qpb.COOLDOWN_DAYS, "D"),
    }


def run_window_unit(A: dict, s: int, e: int, n_contributions: int,
                    execution_lag: int = EXECUTION_LAG,
                    fill_on: str = FILL_PRICE) -> tuple[float, float, float]:
    """Simulate rows [s, e) once and return the final values of three UNIT
    buckets — 100% QQQ, 100% Top-1 Stock, 100% TQQQ — each seeded with
    INITIAL_CAPITAL and receiving the full CONTRIBUTION at each anniversary.
    Signal state (entries/exits) is shared; the buckets differ only in the
    asset they hold, so any weight mix blends linearly from these finals."""
    price      = A["price"]
    open_      = A["open"]
    breadth    = A["breadth"]
    tqqq_close = A["tqqq_close"]
    tqqq_open  = A["tqqq_open"]

    position       = "OUT"
    cooldown_until = None
    last_sell_reason = None
    last_exit_price  = None
    stock_close_arr  = None
    stock_open_arr   = None
    ndx_high = 0.0
    macd_age = ext_age = BIG

    qqq_bucket = stock_bucket = tqqq_bucket = INITIAL_CAPITAL
    qqq_shares = stock_shares = tqqq_shares = 0.0
    stock_active = tqqq_active = False

    qqq_reserve = stock_reserve = tqqq_reserve = 0.0
    contributions_done = 0

    use_open = fill_on == "open"

    for i in range(s, e):
        j  = i - s
        si = i - execution_lag if i - execution_lag >= s else -1
        sig_price = price[si] if si >= 0 else price[i]

        if j > 0 and j % YEAR_ROWS == 0 and contributions_done < n_contributions:
            qqq_reserve   += CONTRIBUTION
            stock_reserve += CONTRIBUTION
            tqqq_reserve  += CONTRIBUTION
            contributions_done += 1

        if position == "OUT":
            if j == 0:
                do_buy = True
            elif si >= 0:
                b           = breadth[si]
                cooldown_ok = cooldown_until is None or A["dates"][i] > cooldown_until
                washout_buy = (not np.isnan(b) and b < qpb.BUY_B200_THRESH
                               and A["vote_gate"][si])
                # Trend re-entry on a fresh MA200 recross (NDX): rejoin when the
                # last exit was a climax-top or price is back above the prior exit.
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

                qqq_bucket   += qqq_reserve;   qqq_reserve   = 0.0
                stock_bucket += stock_reserve; stock_reserve = 0.0
                tqqq_bucket  += tqqq_reserve;  tqqq_reserve  = 0.0

                qqq_bucket   -= qpb.COMMISSION if qqq_bucket   > qpb.COMMISSION else 0.0
                stock_bucket -= qpb.COMMISSION if stock_bucket > qpb.COMMISSION else 0.0
                tqqq_bucket  -= qpb.COMMISSION if tqqq_bucket  > qpb.COMMISSION else 0.0

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

                gross_qqq   = qqq_shares * qqq_px * (1 - qpb.SLIPPAGE)
                gross_stock = (stock_shares * stock_px * (1 - qpb.SLIPPAGE)
                               if stock_active and not np.isnan(stock_px) else stock_bucket)
                gross_tqqq  = (tqqq_shares * tqqq_px * (1 - qpb.SLIPPAGE)
                               if tqqq_active and not np.isnan(tqqq_px) else tqqq_bucket)

                qqq_bucket   = gross_qqq   - (qpb.COMMISSION if gross_qqq   > qpb.COMMISSION else 0.0)
                stock_bucket = gross_stock - (qpb.COMMISSION if gross_stock > qpb.COMMISSION else 0.0)
                tqqq_bucket  = gross_tqqq  - (qpb.COMMISSION if gross_tqqq  > qpb.COMMISSION else 0.0)

                position         = "OUT"
                cooldown_until   = A["dates"][i] + A["cooldown"]
                last_sell_reason = ("bearish-divergence" if bearish_div
                                    else "climax-top" if climax else "trailing-stop")
                last_exit_price  = sig_price
                qqq_shares = stock_shares = tqqq_shares = 0.0

    last = e - 1
    if position == "IN":
        fq = qqq_shares * price[last]
        if stock_active and stock_close_arr is not None and not np.isnan(stock_close_arr[last]):
            fs = stock_shares * stock_close_arr[last]
        else:
            fs = stock_bucket
        if tqqq_active and not np.isnan(tqqq_close[last]):
            ft = tqqq_shares * tqqq_close[last]
        else:
            ft = tqqq_bucket
    else:
        fq, fs, ft = qqq_bucket, stock_bucket, tqqq_bucket
    return fq + qqq_reserve, fs + stock_reserve, ft + tqqq_reserve


def weight_grid(step: int) -> list[tuple[int, int, int]]:
    """All (qqq, stock, tqqq) percent triples on the simplex, in `step` steps."""
    return [(q, s, 100 - q - s)
            for q in range(0, 101, step)
            for s in range(0, 101 - q, step)]


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

    combos = weight_grid(WEIGHT_STEP)
    print(f"Weight grid: {len(combos)} combos (step {WEIGHT_STEP}%)")

    rows_out = []
    for Y in HORIZONS:
        win_len   = YEAR_ROWS * Y
        n_windows = L - win_len + 1
        if n_windows <= 0:
            print(f"  {Y}y: not enough data, stopping")
            break
        n_contrib = Y - 1
        deployed  = INITIAL_CAPITAL + CONTRIBUTION * n_contrib

        Fq = np.empty(n_windows)
        Fs = np.empty(n_windows)
        Ft = np.empty(n_windows)
        for w in range(n_windows):
            Fq[w], Fs[w], Ft[w] = run_window_unit(A, w, w + win_len, n_contrib)

        best_med = best_worst = None
        for (pq, ps, pt) in combos:
            finals = (pq * Fq + ps * Fs + pt * Ft) / 100.0
            ret    = (finals - deployed) / deployed * 100
            med    = float(np.median(ret))
            worst  = float(finals.min())
            row = {
                "qqq_pct": pq, "stock_pct": ps, "tqqq_pct": pt,
                "years": Y, "n_windows": n_windows, "deployed_usd": int(deployed),
                "avg_final_usd": round(finals.mean()),
                "avg_return_pct": round(float(ret.mean()), 1),
                "median_final_usd": round(float(np.median(finals))),
                "median_return_pct": round(med, 1),
                "worst_final_usd": round(worst),
                "best_final_usd": round(finals.max()),
                "pct_losing_windows": round(float((ret < 0).mean() * 100), 1),
            }
            rows_out.append(row)
            if best_med is None or med > best_med["median_return_pct"]:
                best_med = row
            if best_worst is None or worst > best_worst["worst_final_usd"]:
                best_worst = row

        print(f"  {Y:>2}y ({n_windows:,} windows): "
              f"max-median {best_med['qqq_pct']}/{best_med['stock_pct']}/{best_med['tqqq_pct']} "
              f"med {best_med['median_return_pct']:+.1f}%  |  "
              f"max-worst {best_worst['qqq_pct']}/{best_worst['stock_pct']}/{best_worst['tqqq_pct']} "
              f"worst ${best_worst['worst_final_usd']:,}")

    out = pd.DataFrame(rows_out)
    out.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
