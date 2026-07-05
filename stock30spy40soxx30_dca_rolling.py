"""
Rolling-window DCA analysis: NDX Top-1 Stock 30% / SPY 40% / SOXX 30%.

Reuses the exact breadth-driven buy/sell signal from qqq_portfolio_backtest.py
(which loads breadth via breadth_daily.csv — the continuous daily series built
by build_breadth_daily.py from S5TH + MMTH). QQQ and TQQQ weights are zeroed
out here; only the Stock/SPY/SOXX buckets are active.

For each horizon of 1..20 years: $1,000,000 initial capital, plus $200,000
contributed at every subsequent 252-trading-day "anniversary" of the window's
start (so a Y-year window gets Y-1 contributions — matches deployed_usd =
1,000,000 + 200,000*(Y-1)). Rolls a 1-trading-day step across the full
history. Reports strategy vs. NDX buy&hold (same DCA schedule) outcomes.

Each window force-enters the market on day 1 (fully invested in Stock/SPY/
SOXX immediately, matching the sibling qqq70stock30_dca_rolling.csv
methodology) — the breadth/VIX/climax/trailing-stop signals then govern
exits and any later re-entries within the window.

Output: stock30spy40soxx30_dca_rolling.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_portfolio_backtest as qpb

qpb.QQQ_WEIGHT   = 0.0
qpb.STOCK_WEIGHT = 0.30
qpb.SPY_WEIGHT   = 0.40
qpb.SOXX_WEIGHT  = 0.30
qpb.TQQQ_WEIGHT  = 0.0

YEAR_ROWS       = 252
HORIZONS        = range(1, 21)
INITIAL_CAPITAL = 1_000_000.0
CONTRIBUTION    = 200_000.0
OUT_FILE        = Path(__file__).parent / "stock30spy40soxx30_dca_rolling.csv"

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day NDX closes; the earliest tradeable fill is the NEXT
# session. Default: a signal on day t fills at day t+1's OPEN of the traded legs.
# Set EXECUTION_LAG=0 and FILL_PRICE="close" for the legacy same-day-close fill.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar


def run_window(window: pd.DataFrame, top_holdings: dict, aligned_stocks: dict,
                aligned_spy: pd.Series, aligned_soxx: pd.Series,
                n_contributions: int,
                stock_opens: dict | None = None,
                spy_open: pd.Series | None = None,
                soxx_open: pd.Series | None = None,
                execution_lag: int = EXECUTION_LAG,
                fill_on: str = FILL_PRICE) -> float:
    """Simulate the Stock/SPY/SOXX breadth-signal strategy over one window,
    injecting CONTRIBUTION every YEAR_ROWS rows. Returns the final value.

    Signals read the NDX close from `execution_lag` bars ago; fills happen on the
    current bar at each leg's open (fill_on="open") or close. lag=0 reproduces the
    legacy same-day look-ahead fill. Mark-to-market always uses closes."""
    stock_opens = stock_opens or {}

    def fill_px(close_s, open_s, date):
        if fill_on == "open" and open_s is not None:
            v = qpb._safe(open_s, date)
            if not pd.isna(v):
                return v
        return qpb._safe(close_s, date)

    position       = "OUT"
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_price: float | None = None
    holding_ticker: str | None = None
    ndx_high = 0.0
    macd_age = ext_age = 10**9

    stock_bucket = INITIAL_CAPITAL * qpb.STOCK_WEIGHT
    spy_bucket   = INITIAL_CAPITAL * qpb.SPY_WEIGHT
    soxx_bucket  = INITIAL_CAPITAL * qpb.SOXX_WEIGHT
    stock_active = spy_active = soxx_active = False
    stock_shares = spy_shares = soxx_shares = 0.0

    cash_reserve        = 0.0
    contributions_done  = 0

    idx = window.index
    for i, date in enumerate(idx):
        row = window.iloc[i]
        # Signal bar: the row `execution_lag` bars ago (what was known at that close).
        srow = window.iloc[i - execution_lag] if i - execution_lag >= 0 else None
        sig_price = float(srow["price"]) if srow is not None else float(row["price"])
        breadth   = srow["breadth"] if srow is not None else float("nan")

        if i > 0 and i % YEAR_ROWS == 0 and contributions_done < n_contributions:
            cash_reserve += CONTRIBUTION
            contributions_done += 1

        price = float(row["price"])   # today's close (mark-to-market)

        if position == "OUT":
            if i == 0:
                do_buy = True
            elif srow is not None:
                vote_gate   = bool(srow["vote_gate"])
                cooldown_ok = cooldown_until is None or date > cooldown_until
                washout_buy = (not pd.isna(breadth) and breadth < qpb.BUY_B200_THRESH
                               and vote_gate)
                # Trend re-entry on a fresh MA200 recross (NDX): rejoin when the
                # last exit was a climax-top or price is back above the prior exit.
                recross_ok  = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and sig_price > last_exit_price)
                trend_buy   = bool(srow["ma200_recross"]) and recross_ok
                do_buy = cooldown_ok and (washout_buy or trend_buy)
            else:
                do_buy = False
            if do_buy:
                year         = date.year
                stock_ticker = top_holdings.get(year) or top_holdings.get(year - 1)
                stock_px = fill_px(aligned_stocks.get(stock_ticker) if stock_ticker else None,
                                   stock_opens.get(stock_ticker) if stock_ticker else None, date)
                spy_px   = fill_px(aligned_spy,  spy_open,  date)
                soxx_px  = fill_px(aligned_soxx, soxx_open, date)

                if cash_reserve > 0:
                    stock_bucket += cash_reserve * qpb.STOCK_WEIGHT
                    spy_bucket   += cash_reserve * qpb.SPY_WEIGHT
                    soxx_bucket  += cash_reserve * qpb.SOXX_WEIGHT
                    cash_reserve = 0.0

                total_pre  = stock_bucket + spy_bucket + soxx_bucket
                comm_scale = (total_pre - qpb.COMMISSION) / total_pre if total_pre > 0 else 1.0
                stock_bucket *= comm_scale
                spy_bucket   *= comm_scale
                soxx_bucket  *= comm_scale

                stock_active = not pd.isna(stock_px)
                spy_active   = not pd.isna(spy_px)
                soxx_active  = not pd.isna(soxx_px)

                stock_entry_px = stock_px * (1 + qpb.SLIPPAGE) if stock_active else 0.0
                spy_entry_px   = spy_px   * (1 + qpb.SLIPPAGE) if spy_active   else 0.0
                soxx_entry_px  = soxx_px  * (1 + qpb.SLIPPAGE) if soxx_active  else 0.0

                stock_shares = (stock_bucket / stock_entry_px) if stock_active else 0.0
                spy_shares   = (spy_bucket   / spy_entry_px)   if spy_active   else 0.0
                soxx_shares  = (soxx_bucket  / soxx_entry_px)  if soxx_active  else 0.0

                holding_ticker = stock_ticker
                ndx_high       = sig_price
                macd_age = ext_age = 10**9
                position       = "IN"

        elif position == "IN":
            ndx_high     = max(ndx_high, sig_price)
            macd_age     = 0 if (srow is not None and bool(srow["macd_cross"])) else macd_age + 1
            ext_age      = 0 if (srow is not None and bool(srow["ext10"]))      else ext_age + 1
            price_rose   = bool(srow["price_rose"])   if srow is not None else False
            breadth_fell = bool(srow["breadth_fell"]) if srow is not None else False
            bearish_div  = (price_rose and breadth_fell
                            and not pd.isna(breadth) and breadth < qpb.DIVERGENCE_BREADTH_CAP)
            climax       = (macd_age < qpb.CLIMAX_VOTE_WINDOW) and (ext_age < qpb.CLIMAX_VOTE_WINDOW)
            trail_hit    = sig_price <= ndx_high * (1 - qpb.TRAILING_STOP_PCT / 100)

            if bearish_div or climax or trail_hit:
                stock_px_exit = fill_px(aligned_stocks.get(holding_ticker) if holding_ticker else None,
                                        stock_opens.get(holding_ticker) if holding_ticker else None, date)
                spy_px_exit   = fill_px(aligned_spy,  spy_open,  date)
                soxx_px_exit  = fill_px(aligned_soxx, soxx_open, date)

                gross_stock = (stock_shares * stock_px_exit * (1 - qpb.SLIPPAGE)
                               if stock_active and not pd.isna(stock_px_exit) else stock_bucket)
                gross_spy   = (spy_shares * spy_px_exit * (1 - qpb.SLIPPAGE)
                               if spy_active and not pd.isna(spy_px_exit) else spy_bucket)
                gross_soxx  = (soxx_shares * soxx_px_exit * (1 - qpb.SLIPPAGE)
                               if soxx_active and not pd.isna(soxx_px_exit) else soxx_bucket)
                gross_total = gross_stock + gross_spy + gross_soxx
                comm_frac   = qpb.COMMISSION / gross_total if gross_total > 0 else 0.0

                stock_bucket = gross_stock * (1 - comm_frac)
                spy_bucket   = gross_spy   * (1 - comm_frac)
                soxx_bucket  = gross_soxx  * (1 - comm_frac)

                position       = "OUT"
                cooldown_until = date + pd.Timedelta(days=qpb.COOLDOWN_DAYS)
                last_sell_reason = ("bearish-divergence" if bearish_div
                                    else "climax-top" if climax else "trailing-stop")
                last_exit_price = sig_price
                stock_shares = spy_shares = soxx_shares = 0.0

    last_date = window.index[-1]
    if position == "IN":
        last_stock = qpb._safe(aligned_stocks.get(holding_ticker) if holding_ticker else None, last_date)
        last_spy   = qpb._safe(aligned_spy,  last_date)
        last_soxx  = qpb._safe(aligned_soxx, last_date)
        sv = stock_shares * last_stock if stock_active and not pd.isna(last_stock) else stock_bucket
        pv = spy_shares   * last_spy   if spy_active   and not pd.isna(last_spy)   else spy_bucket
        xv = soxx_shares  * last_soxx  if soxx_active  and not pd.isna(last_soxx)  else soxx_bucket
        return sv + pv + xv + cash_reserve
    return stock_bucket + spy_bucket + soxx_bucket + cash_reserve


def run_buyhold(window: pd.DataFrame, n_contributions: int) -> float:
    """NDX buy & hold with the same $1M + $200k/anniversary DCA schedule."""
    prices = window["price"].to_numpy()
    shares = INITIAL_CAPITAL / prices[0]
    contributions_done = 0
    for i in range(1, len(prices)):
        if i % YEAR_ROWS == 0 and contributions_done < n_contributions:
            shares += CONTRIBUTION / prices[i]
            contributions_done += 1
    return shares * prices[-1]


def main() -> None:
    print("Loading data (breadth via breadth_daily.csv)...")
    merged, top_holdings, aligned_stocks, _aligned_tqqq, aligned_spy, aligned_soxx = qpb.load_data()
    stock_opens, _tqqq_open, spy_open, soxx_open = qpb.load_open_series(top_holdings, merged.index)
    L = len(merged)
    print(f"Merged rows: {L:,}  ({merged.index[0].date()} -> {merged.index[-1].date()})")

    rows_out = []
    for Y in HORIZONS:
        win_len = YEAR_ROWS * Y
        n_windows = L - win_len + 1
        if n_windows <= 0:
            print(f"  {Y}y: not enough data, stopping")
            break
        n_contrib = Y - 1
        deployed  = INITIAL_CAPITAL + CONTRIBUTION * n_contrib

        strat_finals = np.empty(n_windows)
        bh_finals    = np.empty(n_windows)
        for w in range(n_windows):
            window = merged.iloc[w: w + win_len]
            strat_finals[w] = run_window(window, top_holdings, aligned_stocks,
                                          aligned_spy, aligned_soxx, n_contrib,
                                          stock_opens=stock_opens, spy_open=spy_open,
                                          soxx_open=soxx_open)
            bh_finals[w] = run_buyhold(window, n_contrib)

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
