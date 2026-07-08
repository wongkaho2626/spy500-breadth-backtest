"""
Rolling-window buy & hold analysis: TQQQ total return over 1..20-year horizons.

Reuses tqqq_backtest.load_tqqq_data() for the price series: actual TQQQ from
yfinance (2010-02-11 inception onward) spliced with simulated pre-inception
prices (3x NDX daily returns minus an overlap-calibrated drag) back to 2002.

For each horizon of 1..20 years (252 trading days per year): invest
$100,000 at the window's first close, hold to the window's last close, no
contributions and no strategy signals. Rolls a 1-trading-day step across the
full history and reports the average / median / worst / best final value and
total return, plus the share of losing windows.

Output: tqqq_buyhold_rolling.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

import tqqq_backtest as tb

YEAR_ROWS       = 252
HORIZONS        = range(1, 21)
INITIAL_CAPITAL = 100_000.0
OUT_FILE        = Path(__file__).parent / "tqqq_buyhold_rolling.csv"


def main() -> None:
    merged = tb.load_tqqq_data()
    price = merged["price"].to_numpy(float)
    L = len(price)
    print(f"TQQQ rows: {L:,}  ({merged.index[0].date()} -> {merged.index[-1].date()})")

    rows_out = []
    for Y in HORIZONS:
        win_len   = YEAR_ROWS * Y
        n_windows = L - win_len + 1
        if n_windows <= 0:
            print(f"  {Y}y: not enough data, stopping")
            break

        finals  = INITIAL_CAPITAL * price[win_len - 1:] / price[:n_windows]
        returns = (finals - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        rows_out.append({
            "years": Y,
            "n_windows": n_windows,
            "initial_usd": int(INITIAL_CAPITAL),
            "avg_final_usd": round(finals.mean()),
            "avg_return_pct": round(float(returns.mean()), 1),
            "median_final_usd": round(float(np.median(finals))),
            "median_return_pct": round(float(np.median(returns)), 1),
            "worst_final_usd": round(finals.min()),
            "worst_return_pct": round(float(returns.min()), 1),
            "best_final_usd": round(finals.max()),
            "best_return_pct": round(float(returns.max()), 1),
            "pct_losing_windows": round(float((returns < 0).mean() * 100), 1),
        })
        print(f"  {Y:>2}y: {n_windows:,} windows -- "
              f"avg {returns.mean():+,.1f}%, median {np.median(returns):+,.1f}%")

    out = pd.DataFrame(rows_out)
    out.to_csv(OUT_FILE, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_FILE.name}")


if __name__ == "__main__":
    main()
