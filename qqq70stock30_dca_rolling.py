"""
Rolling-window DCA analysis: QQQ 70% / NDX Top-1 Stock 30%.

Reuses the exact breadth-driven buy/sell signal from qqq_portfolio_backtest.py
(which loads breadth via breadth_daily.csv — the continuous daily series built
by build_breadth_daily.py from S5TH + MMTH). Only the QQQ (NDX proxy) and
NDX Top-1 stock buckets are active. If an annual NDX Top-1 holding is missing,
the latest prior-year SPY Top-1 holding is used before falling back to cash.

For each horizon of 1..20 years: $1,000,000 initial capital, plus $200,000
contributed at every subsequent 252-trading-day "anniversary" of the window's
start (so a Y-year window gets Y-1 contributions — matches deployed_usd =
1,000,000 + 200,000*(Y-1)). Rolls a 1-trading-day step across the full
history. Reports strategy vs. NDX buy&hold (same DCA schedule) outcomes.

Each window force-enters the market on day 1 (fully invested in QQQ/Stock
immediately) — the breadth/VIX/climax/trailing-stop signals then govern
exits and any later re-entries within the window. Signals read the NDX close
from EXECUTION_LAG bars ago; fills happen on the current bar at each leg's
open (FILL_PRICE="open") or close. Mark-to-market always uses closes.

The inner loop runs on plain numpy arrays (not DataFrame rows) because the
full sweep touches ~80M window-days.

Outputs:
  - qqq70stock30_dca_rolling.csv: original daily-step descriptive results.
  - qqq70stock30_dca_nonoverlap.csv: canonical non-overlapping cohorts.
  - qqq70stock30_dca_overlap_diagnostics.csv: overlap-aware inference.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_portfolio_backtest as qpb

QQQ_WEIGHT   = 0.70
STOCK_WEIGHT = 0.30

YEAR_ROWS       = 252
HORIZONS        = range(1, 21)
INITIAL_CAPITAL = 1_000_000.0
CONTRIBUTION    = 200_000.0
OUT_FILE          = Path(__file__).parent / "qqq70stock30_dca_rolling.csv"
NONOVERLAP_FILE   = Path(__file__).parent / "qqq70stock30_dca_nonoverlap.csv"
DIAGNOSTICS_FILE  = Path(__file__).parent / "qqq70stock30_dca_overlap_diagnostics.csv"
BOOTSTRAP_SAMPLES = 2_000
RANDOM_SEED       = 20_260_817

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day NDX closes; the earliest tradeable fill is the NEXT
# session. Default: a signal on day t fills at day t+1's OPEN of the traded legs.
# Set EXECUTION_LAG=0 and FILL_PRICE="close" for the legacy same-day-close fill.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

BIG = 10**9


def build_arrays(merged: pd.DataFrame, top_holdings: dict[int, str],
                 aligned_stocks: dict[str, pd.Series],
                 stock_opens: dict[str, pd.Series]) -> dict:
    """Flatten the merged signal frame and aligned stock series into numpy
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
        "top_holdings":  top_holdings,
        "cooldown":      np.timedelta64(qpb.COOLDOWN_DAYS, "D"),
    }


def run_window(A: dict, s: int, e: int, n_contributions: int,
               execution_lag: int = EXECUTION_LAG,
               fill_on: str = FILL_PRICE) -> float:
    """Simulate the QQQ/Stock breadth-signal strategy over rows [s, e),
    injecting CONTRIBUTION every YEAR_ROWS rows. Returns the final value."""
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
    stock_active = False

    cash_reserve       = 0.0
    contributions_done = 0

    use_open = fill_on == "open"

    for i in range(s, e):
        j  = i - s
        si = i - execution_lag if i - execution_lag >= s else -1
        sig_price = price[si] if si >= 0 else price[i]

        if j > 0 and j % YEAR_ROWS == 0 and contributions_done < n_contributions:
            contributions_done += 1
            if position == "IN":
                # Deploy immediately into the open positions (pro-rata, at
                # today's close) — same treatment as combined_n30q70.py's
                # mid-trade contributions, instead of idling as cash.
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

                if cash_reserve > 0:
                    qqq_bucket   += cash_reserve * QQQ_WEIGHT
                    stock_bucket += cash_reserve * STOCK_WEIGHT
                    cash_reserve = 0.0

                total_pre  = qqq_bucket + stock_bucket
                comm_scale = (total_pre - qpb.COMMISSION) / total_pre if total_pre > 0 else 1.0
                qqq_bucket   *= comm_scale
                stock_bucket *= comm_scale

                stock_active = not np.isnan(stock_px)
                qqq_shares   = qqq_bucket / (qqq_px * (1 + qpb.SLIPPAGE))
                stock_shares = (stock_bucket / (stock_px * (1 + qpb.SLIPPAGE))
                                if stock_active else 0.0)

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

                gross_qqq   = qqq_shares * qqq_px * (1 - qpb.SLIPPAGE)
                gross_stock = (stock_shares * stock_px * (1 - qpb.SLIPPAGE)
                               if stock_active and not np.isnan(stock_px) else stock_bucket)
                gross_total = gross_qqq + gross_stock
                comm_frac   = qpb.COMMISSION / gross_total if gross_total > 0 else 0.0

                qqq_bucket   = gross_qqq   * (1 - comm_frac)
                stock_bucket = gross_stock * (1 - comm_frac)

                position         = "OUT"
                cooldown_until   = A["dates"][i] + A["cooldown"]
                last_sell_reason = ("bearish-divergence" if bearish_div
                                    else "climax-top" if climax else "trailing-stop")
                last_exit_price  = sig_price
                qqq_shares = stock_shares = 0.0

    last = e - 1
    if position == "IN":
        qv = qqq_shares * price[last]
        if stock_active and stock_close_arr is not None and not np.isnan(stock_close_arr[last]):
            sv = stock_shares * stock_close_arr[last]
        else:
            sv = stock_bucket
        return qv + sv + cash_reserve
    return qqq_bucket + stock_bucket + cash_reserve


def run_buyhold(price: np.ndarray, s: int, e: int, n_contributions: int) -> float:
    """NDX buy & hold with the same $1M + $200k/anniversary DCA schedule."""
    shares = INITIAL_CAPITAL / price[s]
    contributions_done = 0
    for i in range(s + 1, e):
        if (i - s) % YEAR_ROWS == 0 and contributions_done < n_contributions:
            shares += CONTRIBUTION / price[i]
            contributions_done += 1
    return shares * price[e - 1]


def newey_west_mean_stats(values: np.ndarray, max_lag: int) -> dict[str, float]:
    """Estimate mean uncertainty with Bartlett-kernel Newey-West errors.

    FFT autocovariances keep the calculation practical when the overlap lag is
    several thousand sessions. The returned effective sample size is only a
    dependence-adjusted diagnostic, not a literal count of independent windows.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    lag = min(max_lag, n - 1)
    centered = x - x.mean()

    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=nfft)
    autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[:lag + 1] / n
    if lag:
        weights = 1 - np.arange(1, lag + 1) / (lag + 1)
        long_run_variance = autocov[0] + 2 * float(np.dot(weights, autocov[1:]))
    else:
        long_run_variance = float(autocov[0])
    long_run_variance = max(float(long_run_variance), 0.0)

    mean = float(x.mean())
    standard_error = math.sqrt(long_run_variance / n)
    if standard_error > 0:
        z_stat = mean / standard_error
        p_value = math.erfc(abs(z_stat) / math.sqrt(2))
        sample_variance = float(np.var(x, ddof=1)) if n > 1 else 0.0
        effective_n = min(float(n), max(1.0, sample_variance / standard_error**2))
    else:
        z_stat = math.nan
        p_value = math.nan
        effective_n = float(n)

    return {
        "lag": lag,
        "mean": mean,
        "standard_error": standard_error,
        "ci_low": mean - 1.96 * standard_error,
        "ci_high": mean + 1.96 * standard_error,
        "z_stat": z_stat,
        "p_value": p_value,
        "effective_n": effective_n,
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    """Bootstrap the mean of already non-overlapping cohorts."""
    x = np.asarray(values, dtype=float)
    if len(x) < 3:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_SAMPLES, len(x)))
    means = x[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def inference_quality(n_nonoverlap: int) -> str:
    if n_nonoverlap >= 20:
        return "moderate"
    if n_nonoverlap >= 3:
        return "exploratory"
    return "insufficient (<3 non-overlap windows)"


def main() -> None:
    print("Loading data (breadth via breadth_daily.csv)...")
    merged, top_holdings, aligned_stocks, _tqqq, _spy, _soxx = qpb.load_data()
    stock_opens, _tqqq_o, _spy_o, _soxx_o = qpb.load_open_series(top_holdings, merged.index)
    L = len(merged)
    print(f"Merged rows: {L:,}  ({merged.index[0].date()} -> {merged.index[-1].date()})")

    A = build_arrays(merged, top_holdings, aligned_stocks, stock_opens)
    price = A["price"]

    rows_out = []
    nonoverlap_rows = []
    diagnostic_rows = []
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
            strat_finals[w] = run_window(A, w, w + win_len, n_contrib)
            bh_finals[w]    = run_buyhold(price, w, w + win_len, n_contrib)

        strat_ret = (strat_finals - deployed) / deployed * 100
        bh_ret    = (bh_finals - deployed) / deployed * 100
        excess_ret = strat_ret - bh_ret

        # Offset zero is the canonical cohort. Each selected window begins only
        # after the prior selected horizon has ended, so these observations do
        # not reuse market sessions within a horizon.
        nonoverlap_idx = np.arange(0, n_windows, win_len)
        nonoverlap_excess = excess_ret[nonoverlap_idx]
        for window_id, w in enumerate(nonoverlap_idx, start=1):
            nonoverlap_rows.append({
                "years": Y,
                "window_id": window_id,
                "start_date": pd.Timestamp(A["dates"][w]).date(),
                "end_date": pd.Timestamp(A["dates"][w + win_len - 1]).date(),
                "deployed_usd": int(deployed),
                "strategy_final_usd": round(strat_finals[w]),
                "strategy_return_pct": round(strat_ret[w], 4),
                "ndx_buyhold_final_usd": round(bh_finals[w]),
                "ndx_buyhold_return_pct": round(bh_ret[w], 4),
                "excess_return_pp": round(excess_ret[w], 4),
            })

        nw = newey_west_mean_stats(excess_ret, win_len - 1)
        bootstrap_low, bootstrap_high = bootstrap_mean_ci(
            nonoverlap_excess, RANDOM_SEED + Y
        )
        diagnostic_rows.append({
            "years": Y,
            "raw_window_count": n_windows,
            "nonoverlap_window_count": len(nonoverlap_idx),
            "structural_effective_n": round(n_windows / win_len, 4),
            "newey_west_effective_n": round(nw["effective_n"], 4),
            "overlap_ratio_pct": round((1 - len(nonoverlap_idx) / n_windows) * 100, 4),
            "newey_west_lag": nw["lag"],
            "excess_mean_pp": round(nw["mean"], 4),
            "newey_west_se_pp": round(nw["standard_error"], 4),
            "newey_west_ci_low_pp": round(nw["ci_low"], 4),
            "newey_west_ci_high_pp": round(nw["ci_high"], 4),
            "newey_west_z_stat": round(nw["z_stat"], 4),
            "newey_west_p_value": round(nw["p_value"], 6),
            "nonoverlap_excess_mean_pp": round(float(nonoverlap_excess.mean()), 4),
            "nonoverlap_bootstrap_ci_low_pp": round(bootstrap_low, 4),
            "nonoverlap_bootstrap_ci_high_pp": round(bootstrap_high, 4),
            "inference_quality": inference_quality(len(nonoverlap_idx)),
        })

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
    nonoverlap_out = pd.DataFrame(nonoverlap_rows)
    nonoverlap_out.to_csv(NONOVERLAP_FILE, index=False)
    diagnostics_out = pd.DataFrame(diagnostic_rows)
    diagnostics_out.to_csv(DIAGNOSTICS_FILE, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_FILE.name}")
    print(f"Wrote {len(nonoverlap_out)} rows -> {NONOVERLAP_FILE.name}")
    print(f"Wrote {len(diagnostics_out)} rows -> {DIAGNOSTICS_FILE.name}")


if __name__ == "__main__":
    main()
