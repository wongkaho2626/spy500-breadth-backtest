"""
Rolling-window buy & hold analysis on a FULLY MOCKED TQQQ, built from NASDAQ-100.

Unlike tqqq_buyhold_rolling.py (which splices actual TQQQ prices after its
2010-02-11 inception onto a simulated pre-inception path), this script mocks
the ENTIRE series from 2002 onward out of NASDAQ100.csv alone:

    mock daily return = LEVERAGE x NDX daily return - daily drag

The drag (expense ratio + financing cost) is calibrated the same way as
tqqq_backtest._simulate_pre_inception: the mean daily shortfall of actual
TQQQ returns vs LEVERAGE x NDX returns over the post-inception overlap
(actual TQQQ fetched from yfinance for calibration/validation only — it never
enters the mocked path).

For each horizon of 1..20 years (252 trading days per year): invest $100,000
at the window's first mocked close, hold to the window's last, rolling the
start one trading day at a time. Reports average / median / worst / best
final value and total return, plus the share of losing windows, and prints a
mock-vs-actual validation over the 2010+ overlap.

Output: tqqq_ndx_mock_rolling.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import tqqq_backtest as tb

YEAR_ROWS       = 252
HORIZONS        = range(1, 21)
INITIAL_CAPITAL = 100_000.0
START_DATE      = "2002-01-01"
OUT_FILE        = Path(__file__).parent / "tqqq_ndx_mock_rolling.csv"


def fetch_actual_tqqq() -> pd.Series:
    print("Fetching actual TQQQ from yfinance (drag calibration only)…")
    raw = yf.download("TQQQ", start="2010-01-01", progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index)
    return close.rename("tqqq")


def build_mock_series(ndx_price: pd.Series, actual: pd.Series) -> pd.Series:
    """Mock TQQQ closes over the full NDX index: cumprod of
    LEVERAGE x NDX return - drag, based at 100 on the first row."""
    ndx_ret = ndx_price.pct_change()
    overlap = pd.concat(
        [actual.pct_change(), ndx_ret.rename("ndx")], axis=1
    ).dropna()
    if overlap.empty:
        raise ValueError("no TQQQ/NDX overlap to calibrate the mock drag")
    drag = float((tb.LEVERAGE * overlap["ndx"] - overlap["tqqq"]).mean())
    corr = float((tb.LEVERAGE * overlap["ndx"]).corr(overlap["tqqq"]))
    print(f"Calibrated drag {drag * 252:.2%}/yr on {len(overlap):,} overlap days "
          f"(corr {corr:.4f})")

    mock_ret = (tb.LEVERAGE * ndx_ret - drag).fillna(0.0)
    worst = float(mock_ret.min())
    if worst <= -1.0:
        raise ValueError(f"mocked daily return {worst:.1%} wipes out the fund")
    return (100.0 * (1.0 + mock_ret).cumprod()).rename("mock_price")


def print_overlap_validation(mock: pd.Series, actual: pd.Series) -> None:
    """Compare mock vs actual total return over the post-inception overlap."""
    common = mock.index.intersection(actual.index)
    m, a = mock.loc[common], actual.loc[common]
    mock_tr   = float(m.iloc[-1] / m.iloc[0] - 1)
    actual_tr = float(a.iloc[-1] / a.iloc[0] - 1)
    daily_corr = float(m.pct_change().corr(a.pct_change()))
    print(f"Overlap check {common[0].date()} -> {common[-1].date()}: "
          f"mock {mock_tr * 100:+,.0f}% vs actual {actual_tr * 100:+,.0f}% "
          f"total return, daily-return corr {daily_corr:.4f}")


def main() -> None:
    ndx_price = tb._load_ndx()["ndx_price"]
    ndx_price = ndx_price[ndx_price.index >= START_DATE]
    actual = fetch_actual_tqqq()

    mock = build_mock_series(ndx_price, actual)
    print_overlap_validation(mock, actual)

    price = mock.to_numpy(float)
    L = len(price)
    print(f"Mocked TQQQ rows: {L:,}  ({mock.index[0].date()} -> {mock.index[-1].date()})")

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
