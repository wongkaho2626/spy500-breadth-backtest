"""
Rolling-window analysis: the tqqq_backtest breadth strategy over 1..20-year
horizons, $100,000 initial capital, no contributions.

Reuses tqqq_backtest.load_tqqq_data() for prices/signals: actual TQQQ from
yfinance (2010-02-11 inception onward) spliced with simulated pre-inception
prices (3x NDX daily returns minus an overlap-calibrated drag) back to 2002.
All signals are computed on NDX (signal_price); execution is in TQQQ.

Each window force-enters the market on day 1 (fully invested in TQQQ
immediately) — the breadth/VIX/climax/trailing-stop signals from
tqqq_backtest.run_strategy then govern exits and any later re-entries within
the window. Signals read the NDX close from EXECUTION_LAG bars ago; fills
happen on the current bar at TQQQ's open (FILL_PRICE="open") or close.
Mark-to-market always uses TQQQ closes. Costs: $1 commission + 0.05%
slippage per side, 15-calendar-day cooldown after each sell.

For each horizon of 1..20 years (252 trading days per year), rolls a
1-trading-day step across the full history and reports strategy vs. TQQQ
buy & hold outcomes. The inner loop runs on plain numpy arrays because the
full sweep touches ~100M window-days.

Output: tqqq_strategy_rolling.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import tqqq_backtest as tb

YEAR_ROWS       = 252
HORIZONS        = range(1, 21)
INITIAL_CAPITAL = 100_000.0
OUT_FILE        = Path(__file__).parent / "tqqq_strategy_rolling.csv"

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day NDX closes; the earliest tradeable fill is the
# NEXT session. Default: a signal on day t fills at day t+1's TQQQ OPEN. Set
# EXECUTION_LAG=0 and FILL_PRICE="close" for the legacy same-day-close fill.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

BIG = 10**9


def build_arrays(merged: pd.DataFrame) -> dict:
    """Flatten the merged signal frame into numpy arrays for the hot loop."""
    return {
        "dates":         merged.index.values,
        "price":         merged["price"].to_numpy(float),        # TQQQ close
        "open":          merged["open"].to_numpy(float),         # TQQQ open
        "sig":           merged["signal_price"].to_numpy(float), # NDX close
        "breadth":       merged["breadth"].to_numpy(float),
        "vote_gate":     merged["vote_gate"].to_numpy(bool),
        "price_rose":    merged["price_rose"].to_numpy(bool),
        "breadth_fell":  merged["breadth_fell"].to_numpy(bool),
        "macd_cross":    merged["macd_cross"].to_numpy(bool),
        "ext10":         merged["ext10"].to_numpy(bool),
        "ma200_recross": merged["ma200_recross"].to_numpy(bool),
        "cooldown":      np.timedelta64(tb.COOLDOWN_DAYS, "D"),
    }


def run_window(A: dict, s: int, e: int,
               execution_lag: int = EXECUTION_LAG,
               fill_on: str = FILL_PRICE) -> float:
    """Simulate the TQQQ breadth strategy over rows [s, e). Returns the final
    portfolio value (mark-to-market at the last close)."""
    price = A["price"]
    open_ = A["open"]
    sig   = A["sig"]

    position         = "OUT"
    cooldown_until   = None
    last_sell_reason = None
    last_exit_price  = None
    eff_entry        = 0.0
    ndx_high         = 0.0
    macd_age = ext_age = BIG

    portfolio = INITIAL_CAPITAL
    use_open  = fill_on == "open"

    for i in range(s, e):
        j  = i - s
        si = i - execution_lag if i - execution_lag >= s else -1
        sig_price = sig[si] if si >= 0 else sig[i]

        if position == "OUT":
            if j == 0:
                do_buy = True
            elif si >= 0:
                b           = A["breadth"][si]
                cooldown_ok = cooldown_until is None or A["dates"][i] > cooldown_until
                washout_buy = (not np.isnan(b) and b < tb.BUY_B200_THRESH
                               and A["vote_gate"][si])
                # Trend re-entry on a fresh MA200 recross (NDX): rejoin when the
                # last exit was a climax-top or NDX is back above the prior exit.
                recross_ok  = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and sig_price > last_exit_price)
                trend_buy   = A["ma200_recross"][si] and recross_ok
                do_buy = cooldown_ok and (washout_buy or trend_buy)
            else:
                do_buy = False
            if do_buy:
                fill_px = open_[i] if use_open and not np.isnan(open_[i]) else price[i]
                portfolio -= tb.COMMISSION
                eff_entry = fill_px * (1 + tb.SLIPPAGE)
                ndx_high  = sig_price
                macd_age  = ext_age = BIG
                position  = "IN"

        else:  # IN
            if sig_price > ndx_high:
                ndx_high = sig_price
            macd_age = 0 if (si >= 0 and A["macd_cross"][si]) else macd_age + 1
            ext_age  = 0 if (si >= 0 and A["ext10"][si])      else ext_age + 1
            if si >= 0:
                b           = A["breadth"][si]
                bearish_div = (A["price_rose"][si] and A["breadth_fell"][si]
                               and not np.isnan(b) and b < tb.DIVERGENCE_BREADTH_CAP)
            else:
                bearish_div = False
            climax    = (macd_age < tb.CLIMAX_VOTE_WINDOW) and (ext_age < tb.CLIMAX_VOTE_WINDOW)
            trail_hit = sig_price <= ndx_high * (1 - tb.TRAILING_STOP_PCT / 100)

            if bearish_div or climax or trail_hit:
                fill_px  = open_[i] if use_open and not np.isnan(open_[i]) else price[i]
                eff_exit = fill_px * (1 - tb.SLIPPAGE)
                portfolio *= eff_exit / eff_entry
                portfolio -= tb.COMMISSION

                position         = "OUT"
                cooldown_until   = A["dates"][i] + A["cooldown"]
                last_sell_reason = ("bearish-divergence" if bearish_div
                                    else "climax-top" if climax else "trailing-stop")
                last_exit_price  = sig_price

    if position == "IN":
        return portfolio * (price[e - 1] * (1 - tb.SLIPPAGE) / eff_entry)
    return portfolio


def main() -> None:
    merged = tb.load_tqqq_data()
    A = build_arrays(merged)
    price = A["price"]
    L = len(price)
    print(f"TQQQ rows: {L:,}  ({merged.index[0].date()} -> {merged.index[-1].date()})")

    rows_out = []
    for Y in HORIZONS:
        win_len   = YEAR_ROWS * Y
        n_windows = L - win_len + 1
        if n_windows <= 0:
            print(f"  {Y}y: not enough data, stopping")
            break

        strat_finals = np.empty(n_windows)
        for w in range(n_windows):
            strat_finals[w] = run_window(A, w, w + win_len)
        bh_finals = INITIAL_CAPITAL * price[win_len - 1:] / price[:n_windows]

        strat_ret = (strat_finals - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        bh_ret    = (bh_finals - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        rows_out.append({
            "years": Y,
            "n_windows": n_windows,
            "initial_usd": int(INITIAL_CAPITAL),
            "strategy_avg_final_usd": round(strat_finals.mean()),
            "strategy_avg_return_pct": round(float(strat_ret.mean()), 1),
            "strategy_median_final_usd": round(float(np.median(strat_finals))),
            "strategy_median_return_pct": round(float(np.median(strat_ret)), 1),
            "strategy_worst_final_usd": round(strat_finals.min()),
            "strategy_best_final_usd": round(strat_finals.max()),
            "strategy_pct_losing_windows": round(float((strat_ret < 0).mean() * 100), 1),
            "buyhold_avg_final_usd": round(bh_finals.mean()),
            "buyhold_avg_return_pct": round(float(bh_ret.mean()), 1),
            "buyhold_median_final_usd": round(float(np.median(bh_finals))),
            "buyhold_median_return_pct": round(float(np.median(bh_ret)), 1),
            "buyhold_worst_final_usd": round(bh_finals.min()),
            "buyhold_best_final_usd": round(bh_finals.max()),
            "buyhold_pct_losing_windows": round(float((bh_ret < 0).mean() * 100), 1),
        })
        print(f"  {Y:>2}y: {n_windows:,} windows -- "
              f"strategy avg {strat_ret.mean():+,.1f}% / median {np.median(strat_ret):+,.1f}%, "
              f"buy&hold avg {bh_ret.mean():+,.1f}% / median {np.median(bh_ret):+,.1f}%")

    out = pd.DataFrame(rows_out)
    out.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
