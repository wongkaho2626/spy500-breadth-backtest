"""
Rolling-window DCA analysis: QQQ 70% / NDX Top-1 Stock 30%, with a 10% TQQQ
"washout boost" — washout entries (breadth < 26 washout signal) carve 10% of
capital out of the QQQ bucket into TQQQ for the life of that trade; trend
re-entries and the day-1 force-entry are NOT boosted (conservative: forced
entries aren't genuine washout signals).

TQQQ daily returns: actual (yfinance) from 2010; before that a synthetic
3x NDX daily return minus an overlap-calibrated constant drag. Note the drag
is calibrated on the 2010+ low-rate era and likely understates pre-2010
financing costs — see the drag stress test in the session audit.

Everything else matches qqq70stock30_dca_rolling.py: $1,000,000 initial,
$200,000 at each 252-trading-day anniversary (deployed immediately, 70/30,
when IN), 1..20-year horizons rolled 1 day at a time, NDX buy&hold benchmark
on the same DCA schedule, next-day-open fills.

Output: qqq70stock30_washoutboost_dca_rolling.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_portfolio_backtest as qpb
import qqq70stock30_dca_rolling as qsr

BOOST           = 0.10
YEAR_ROWS       = 252
HORIZONS        = range(1, 21)
INITIAL_CAPITAL = 1_000_000.0
CONTRIBUTION    = 200_000.0
QQQ_WEIGHT      = 0.70
STOCK_WEIGHT    = 0.30
OUT_FILE        = Path(__file__).parent / "qqq70stock30_washoutboost_dca_rolling.csv"

BIG = 10**9


def build_tqqq_returns(merged: pd.DataFrame, aligned_tqqq: pd.Series | None) -> np.ndarray:
    """Daily TQQQ returns aligned to merged.index: actual where available,
    synthetic 3x NDX minus calibrated drag elsewhere."""
    ndx_r = merged["price"].pct_change()
    if aligned_tqqq is None:
        return (3 * ndx_r).to_numpy(float)
    tq = aligned_tqqq.pct_change()
    ov = pd.concat([tq, ndx_r], axis=1, keys=["tq", "ndx"]).dropna()
    ov = ov[ov["tq"] != 0]
    drag = float((3 * ov["ndx"] - ov["tq"]).mean())
    synth = 3 * ndx_r - drag
    out = tq.where(tq.notna() & (tq != 0), synth)
    print(f"[TQQQ proxy drag: {drag * 252:.2%}/yr over {len(ov)} overlap days]")
    return out.to_numpy(float)


def run_window(A: dict, tqqq_r: np.ndarray, s: int, e: int,
               n_contributions: int) -> float:
    """qqq70stock30_dca_rolling.run_window plus the washout TQQQ boost."""
    price   = A["price"]
    open_   = A["open"]
    breadth = A["breadth"]

    position       = "OUT"
    cooldown_until = None
    last_sell_reason = None
    last_exit_price  = None
    stock_close_arr  = None
    stock_open_arr   = None
    ndx_high = 0.0
    macd_age = ext_age = BIG

    qqq_bucket   = INITIAL_CAPITAL * QQQ_WEIGHT
    stock_bucket = INITIAL_CAPITAL * STOCK_WEIGHT
    qqq_shares = stock_shares = 0.0
    tqqq_val   = 0.0
    stock_active = False

    cash_reserve       = 0.0
    contributions_done = 0

    for i in range(s, e):
        j  = i - s
        si = i - 1 if i - 1 >= s else -1
        sig_price = price[si] if si >= 0 else price[i]

        if j > 0 and j % YEAR_ROWS == 0 and contributions_done < n_contributions:
            contributions_done += 1
            if position == "IN":
                q_add = CONTRIBUTION * QQQ_WEIGHT - qpb.COMMISSION
                qqq_shares += q_add / (price[i] * (1 + qpb.SLIPPAGE))
                s_add = CONTRIBUTION * STOCK_WEIGHT - qpb.COMMISSION
                if (stock_active and stock_close_arr is not None
                        and not np.isnan(stock_close_arr[i])):
                    stock_shares += s_add / (stock_close_arr[i] * (1 + qpb.SLIPPAGE))
                else:
                    stock_bucket += CONTRIBUTION * STOCK_WEIGHT
            else:
                cash_reserve += CONTRIBUTION

        if position == "OUT":
            is_washout = False
            if j == 0:
                do_buy = True   # force-entry, never boosted
            elif si >= 0:
                b = breadth[si]
                cooldown_ok = cooldown_until is None or A["dates"][i] > cooldown_until
                washout_buy = (not np.isnan(b) and b < qpb.BUY_B200_THRESH
                               and A["vote_gate"][si])
                recross_ok  = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and sig_price > last_exit_price)
                trend_buy   = A["ma200_recross"][si] and recross_ok
                do_buy = cooldown_ok and (washout_buy or trend_buy)
                is_washout = do_buy and washout_buy
            else:
                do_buy = False
            if do_buy:
                year   = A["years"][i]
                ticker = (A["top_holdings"].get(year)
                          or A["top_holdings"].get(year - 1))
                stock_close_arr = A["stock_close"].get(ticker) if ticker else None
                stock_open_arr  = A["stock_open"].get(ticker) if ticker else None

                qqq_px = open_[i] if not np.isnan(open_[i]) else price[i]
                stock_px = np.nan
                if stock_close_arr is not None:
                    if stock_open_arr is not None and not np.isnan(stock_open_arr[i]):
                        stock_px = stock_open_arr[i]
                    else:
                        stock_px = stock_close_arr[i]

                if cash_reserve > 0:
                    qqq_bucket   += cash_reserve * QQQ_WEIGHT
                    stock_bucket += cash_reserve * STOCK_WEIGHT
                    cash_reserve = 0.0

                total_pre  = qqq_bucket + stock_bucket
                comm_scale = (total_pre - qpb.COMMISSION) / total_pre if total_pre > 0 else 1.0
                qqq_bucket   *= comm_scale
                stock_bucket *= comm_scale

                # Washout boost: carve the TQQQ slice out of the QQQ bucket
                tqqq_val = 0.0
                if is_washout and not np.isnan(tqqq_r[i]):
                    tqqq_val = min((qqq_bucket + stock_bucket) * BOOST, qqq_bucket)

                stock_active = not np.isnan(stock_px)
                qqq_shares   = (qqq_bucket - tqqq_val) / (qqq_px * (1 + qpb.SLIPPAGE))
                stock_shares = (stock_bucket / (stock_px * (1 + qpb.SLIPPAGE))
                                if stock_active else 0.0)

                ndx_high = sig_price
                macd_age = ext_age = BIG
                position = "IN"

        else:  # IN
            if not np.isnan(tqqq_r[i]):
                tqqq_val *= 1 + tqqq_r[i]
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
                qqq_px = open_[i] if not np.isnan(open_[i]) else price[i]
                stock_px = np.nan
                if stock_close_arr is not None:
                    if stock_open_arr is not None and not np.isnan(stock_open_arr[i]):
                        stock_px = stock_open_arr[i]
                    else:
                        stock_px = stock_close_arr[i]

                gross_qqq   = (qqq_shares * qqq_px + tqqq_val) * (1 - qpb.SLIPPAGE)
                gross_stock = (stock_shares * stock_px * (1 - qpb.SLIPPAGE)
                               if stock_active and not np.isnan(stock_px) else stock_bucket)
                gross_total = gross_qqq + gross_stock
                comm_frac   = qpb.COMMISSION / gross_total if gross_total > 0 else 0.0

                qqq_bucket   = gross_qqq   * (1 - comm_frac)
                stock_bucket = gross_stock * (1 - comm_frac)
                tqqq_val     = 0.0

                position         = "OUT"
                cooldown_until   = A["dates"][i] + A["cooldown"]
                last_sell_reason = ("bearish-divergence" if bearish_div
                                    else "climax-top" if climax else "trailing-stop")
                last_exit_price  = sig_price
                qqq_shares = stock_shares = 0.0

    last = e - 1
    if position == "IN":
        qv = qqq_shares * price[last] + tqqq_val
        if stock_active and stock_close_arr is not None and not np.isnan(stock_close_arr[last]):
            sv = stock_shares * stock_close_arr[last]
        else:
            sv = stock_bucket
        return qv + sv + cash_reserve
    return qqq_bucket + stock_bucket + cash_reserve


def main() -> None:
    print("Loading data (breadth via breadth_daily.csv)...")
    merged, top_holdings, aligned_stocks, aligned_tqqq, _spy, _soxx = qpb.load_data()
    stock_opens, _t, _s, _x = qpb.load_open_series(top_holdings, merged.index)
    L = len(merged)
    print(f"Merged rows: {L:,}  ({merged.index[0].date()} -> {merged.index[-1].date()})")

    A = qsr.build_arrays(merged, top_holdings, aligned_stocks, stock_opens)
    tqqq_r = build_tqqq_returns(merged, aligned_tqqq)
    price = A["price"]

    rows_out = []
    for Y in HORIZONS:
        win_len   = YEAR_ROWS * Y
        n_windows = L - win_len + 1
        if n_windows <= 0:
            print(f"  {Y}y: not enough data, stopping")
            break
        n_contrib = Y - 1
        deployed  = INITIAL_CAPITAL + CONTRIBUTION * n_contrib

        strat_finals = np.empty(n_windows)
        bh_finals    = np.empty(n_windows)
        for w in range(n_windows):
            strat_finals[w] = run_window(A, tqqq_r, w, w + win_len, n_contrib)
            bh_finals[w]    = qsr.run_buyhold(price, w, w + win_len, n_contrib)

        strat_ret = (strat_finals - deployed) / deployed * 100
        bh_ret    = (bh_finals - deployed) / deployed * 100

        rows_out.append({
            "years": Y,
            "n_windows": n_windows,
            "deployed_usd": int(deployed),
            "strategy_avg_final_usd": round(strat_finals.mean()),
            "strategy_avg_return_pct": round(strat_ret.mean(), 1),
            "strategy_median_final_usd": round(float(np.median(strat_finals))),
            "strategy_median_return_pct": round(float(np.median(strat_ret)), 1),
            "strategy_worst_final_usd": round(strat_finals.min()),
            "strategy_best_final_usd": round(strat_finals.max()),
            "strategy_pct_losing_windows": round(float((strat_ret < 0).mean() * 100), 1),
            "ndx_buyhold_avg_final_usd": round(bh_finals.mean()),
            "ndx_buyhold_avg_return_pct": round(bh_ret.mean(), 1),
            "ndx_buyhold_median_final_usd": round(float(np.median(bh_finals))),
            "ndx_buyhold_median_return_pct": round(float(np.median(bh_ret)), 1),
            "ndx_buyhold_worst_final_usd": round(bh_finals.min()),
            "ndx_buyhold_best_final_usd": round(bh_finals.max()),
            "ndx_buyhold_pct_losing_windows": round(float((bh_ret < 0).mean() * 100), 1),
        })
        print(f"  {Y:>2}y: {n_windows:,} windows -- "
              f"strategy avg {strat_ret.mean():+.1f}%, buy&hold avg {bh_ret.mean():+.1f}%")

    out = pd.DataFrame(rows_out)
    out.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
