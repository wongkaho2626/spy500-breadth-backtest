"""
qqq_ndx_top1_sweep.py

Rolling-window statistics for all QQQ / NDX Top-1 Stock allocation mixes.

For every horizon (1y, 3y, 5y, 10y, 15y), slides a daily-rolling window
from 2010-01-01 through 2025-12-31.  At each window, simulates 101
portfolios (QQQ 0%...100% / NDX-Top-1 100%...0%) on a buy-and-hold basis.
The NDX annual Top-1 stock resets at each calendar-year boundary.

Summary statistics across all windows per (horizon, allocation):
  mean, median, std, 25th-pct, 75th-pct
  for Total Return, Final Value, CAGR, Max Drawdown, Sharpe Ratio

Output: qqq_ndx_top1_sweep_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR          = Path(__file__).parent
NDX_FILE          = DATA_DIR / "NASDAQ100.csv"
TOP_HOLDINGS_FILE = DATA_DIR / "NASDAQ100" / "nasdaq100_top10_holdings.csv"
STOCK_PRICE_DIR   = DATA_DIR / "NASDAQ100" / "stock_prices"
OUTPUT_FILE       = DATA_DIR / "qqq_ndx_top1_sweep_results.csv"

INITIAL_CAPITAL = 10_000.0
GLOBAL_START    = pd.Timestamp("2010-01-01")
GLOBAL_END      = pd.Timestamp("2025-12-31")
HORIZONS        = [1, 3, 5, 10, 15]
ALLOCATIONS     = np.arange(0, 101) / 100.0   # QQQ weight: 0.00 → 1.00

_NAME_TO_TICKER: list[tuple[str, str]] = [
    ("cisco",             "CSCO"),
    ("microsoft",         "MSFT"),
    ("intel",             "INTC"),
    ("oracle",            "ORCL"),
    ("qualcomm",          "QCOM"),
    ("apple",             "AAPL"),
    ("alphabet",          "GOOGL"),
    ("google",            "GOOGL"),
    ("amazon",            "AMZN"),
    ("tesla",             "TSLA"),
    ("nvidia",            "NVDA"),
    ("meta",              "META"),
    ("facebook",          "META"),
    ("paypal",            "PYPL"),
    ("netflix",           "NFLX"),
    ("broadcom",          "AVGO"),
    ("costco",            "COST"),
    ("pepsico",           "PEP"),
    ("t-mobile",          "TMUS"),
    ("ebay",              "EBAY"),
    ("dell",              "DELL"),
    ("comcast",           "CMCSA"),
    ("amgen",             "AMGN"),
    ("gilead",            "GILD"),
    ("charter",           "CHTR"),
    ("texas instruments", "TXN"),
]


def _name_to_ticker(name: str) -> str | None:
    nl = name.lower()
    for key, ticker in _NAME_TO_TICKER:
        if key in nl:
            return ticker
    return None


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> tuple[pd.Series, dict[int, str], dict[str, pd.Series]]:
    # NDX price series (used as QQQ proxy)
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price"})
    ndx["price"] = _parse_price(ndx["price"])
    ndx.sort_index(inplace=True)
    qqq_prices = ndx["price"]

    # Annual Top-1 holdings (Rank == 1)
    holdings_df = pd.read_csv(TOP_HOLDINGS_FILE)
    top_holdings: dict[int, str] = {}
    for _, row in holdings_df.iterrows():
        if int(row["Rank"]) == 1:
            ticker = _name_to_ticker(str(row["Holding"]))
            if ticker:
                top_holdings[int(row["Year"])] = ticker

    # Stock price CSVs
    unique_tickers = set(top_holdings.values())
    aligned_stocks: dict[str, pd.Series] = {}
    for ticker in sorted(unique_tickers):
        path = STOCK_PRICE_DIR / f"{ticker}.csv"
        if not path.exists():
            print(f"  [missing] {ticker}.csv — will fall back to QQQ when active")
            continue
        df = pd.read_csv(path)
        df["Date"] = (
            pd.to_datetime(df["Date"], format="mixed", utc=True)
            .dt.tz_localize(None)
        )
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        aligned_stocks[ticker] = df["Close"].astype(float)

    return qqq_prices, top_holdings, aligned_stocks


def build_chained_stock_series(
    trading_days: pd.DatetimeIndex,
    top_holdings: dict[int, str],
    aligned_stocks: dict[str, pd.Series],
    qqq_prices: pd.Series,
) -> pd.Series:
    """
    Build a wealth-index series for the NDX annual Top-1 rolling strategy.

    At the first trading day of each calendar year the position switches to
    that year's Top-1 stock.  If a stock's price data is unavailable the
    QQQ return for that year is used as a fallback.  The index starts at
    1.0 on the first trading day.
    """
    chain      = pd.Series(np.nan, index=trading_days, dtype=float)
    prev_value = 1.0
    years      = sorted(set(d.year for d in trading_days))

    for year in years:
        mask       = trading_days.year == year
        year_dates = trading_days[mask]
        if len(year_dates) == 0:
            continue

        ticker = top_holdings.get(year) or top_holdings.get(year - 1)

        if ticker and ticker in aligned_stocks:
            raw = aligned_stocks[ticker].reindex(year_dates).ffill()
        else:
            if ticker:
                print(f"  [fallback QQQ] {year}: {ticker} not found")
            raw = qqq_prices.reindex(year_dates).ffill()

        first_idx = raw.first_valid_index()
        if first_idx is None:
            chain[year_dates] = prev_value
            continue

        first_px = float(raw[first_idx])
        if first_px == 0 or np.isnan(first_px):
            chain[year_dates] = prev_value
            continue

        chain[year_dates] = prev_value * (raw / first_px)

        last_idx = raw.last_valid_index()
        if last_idx is not None:
            last_val = float(chain[last_idx])
            if not np.isnan(last_val):
                prev_value = last_val

    chain = chain.ffill()
    return chain


def _metrics_vectorized(
    portfolio: np.ndarray,
    years: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute five metrics for all allocations in one pass.

    portfolio : (N_alloc, T) array of daily portfolio values
    Returns   : total_return, final_value, cagr, mdd, sharpe  — each (N_alloc,)
    """
    final = portfolio[:, -1]
    tr    = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    fv    = final
    cagr  = (final / INITIAL_CAPITAL) ** (1.0 / years) - 1.0

    run_max = np.maximum.accumulate(portfolio, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd  = np.where(run_max > 0, (portfolio - run_max) / run_max, 0.0)
    mdd = dd.min(axis=1)

    d_ret  = np.diff(portfolio, axis=1) / portfolio[:, :-1]   # (N_alloc, T-1)
    d_mean = d_ret.mean(axis=1)
    d_std  = d_ret.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(d_std > 0, d_mean / d_std * np.sqrt(252), 0.0)

    return tr, fv, cagr, mdd, sharpe


def analyze_horizon(
    H_years: int,
    qqq_arr: np.ndarray,
    stock_arr: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Run all rolling windows for one time horizon and return summary stats."""
    q_col = ALLOCATIONS          # QQQ weights  (N_alloc,)
    s_col = 1.0 - ALLOCATIONS    # Stock weights (N_alloc,)

    acc_tr  : list[np.ndarray] = []
    acc_fv  : list[np.ndarray] = []
    acc_cag : list[np.ndarray] = []
    acc_mdd : list[np.ndarray] = []
    acc_sh  : list[np.ndarray] = []
    n_win = 0

    offset  = pd.DateOffset(years=H_years)
    n_dates = len(dates)

    for i0 in range(n_dates):
        end_target = dates[i0] + offset
        if end_target > GLOBAL_END:
            break

        i1 = dates.searchsorted(end_target, side="right") - 1
        if i1 <= i0:
            continue

        actual_years = (dates[i1] - dates[i0]).days / 365.25
        if actual_years < H_years * 0.9:
            continue

        q0 = qqq_arr[i0]
        s0 = stock_arr[i0]
        if q0 == 0 or s0 == 0 or np.isnan(q0) or np.isnan(s0):
            continue

        qqq_n   = qqq_arr[i0 : i1 + 1]   / q0   # (T,)
        stock_n = stock_arr[i0 : i1 + 1] / s0   # (T,)

        # (N_alloc, T) — all 101 allocations computed simultaneously
        portfolio = INITIAL_CAPITAL * (
            np.outer(q_col, qqq_n) + np.outer(s_col, stock_n)
        )

        tr, fv, cagr, mdd, sh = _metrics_vectorized(portfolio, actual_years)
        acc_tr.append(tr)
        acc_fv.append(fv)
        acc_cag.append(cagr)
        acc_mdd.append(mdd)
        acc_sh.append(sh)
        n_win += 1

    print(f"  {H_years}y: {n_win:,} windows")

    if n_win == 0:
        return pd.DataFrame()

    def _agg(mat: np.ndarray) -> dict[str, np.ndarray]:
        """mat: (n_win, N_alloc) → five (N_alloc,) stat arrays."""
        return {
            "mean":   mat.mean(axis=0),
            "median": np.median(mat, axis=0),
            "std":    mat.std(axis=0, ddof=1),
            "p25":    np.percentile(mat, 25, axis=0),
            "p75":    np.percentile(mat, 75, axis=0),
        }

    s_tr  = _agg(np.vstack(acc_tr))
    s_fv  = _agg(np.vstack(acc_fv))
    s_cag = _agg(np.vstack(acc_cag))
    s_mdd = _agg(np.vstack(acc_mdd))
    s_sh  = _agg(np.vstack(acc_sh))

    metric_stats = [
        ("total_return",  s_tr),
        ("final_value",   s_fv),
        ("cagr",          s_cag),
        ("max_drawdown",  s_mdd),
        ("sharpe_ratio",  s_sh),
    ]

    rows = []
    for i, q in enumerate(ALLOCATIONS):
        qqq_pct = int(round(q * 100))
        row: dict = {
            "horizon_years": H_years,
            "n_windows":     n_win,
            "qqq_pct":       qqq_pct,
            "stock_pct":     100 - qqq_pct,
        }
        for name, s in metric_stats:
            for stat in ("mean", "median", "std", "p25", "p75"):
                row[f"{name}_{stat}"] = round(float(s[stat][i]), 6)
        rows.append(row)

    return pd.DataFrame(rows)


def print_top_holdings(top_holdings: dict[int, str], aligned_stocks: dict[str, pd.Series]) -> None:
    print("\nNDX Top-1 holdings used (2010–2025):")
    header = f"  {'Year':>4}  {'Ticker':>6}  {'Data':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for yr in sorted(k for k in top_holdings if 2010 <= k <= 2025):
        t = top_holdings[yr]
        status = "found" if t in aligned_stocks else "MISSING"
        print(f"  {yr:>4}  {t:>6}  {status:>7}")


def main() -> None:
    print("Loading data...")
    qqq_prices, top_holdings, aligned_stocks = load_data()

    print_top_holdings(top_holdings, aligned_stocks)

    qqq_global   = qqq_prices.loc[GLOBAL_START:GLOBAL_END]
    trading_days = qqq_global.index
    print(
        f"\nTrading days: {trading_days[0].date()} → {trading_days[-1].date()}"
        f"  ({len(trading_days):,} days)"
    )

    print("\nBuilding NDX Top-1 annual chain series...")
    stock_chain = build_chained_stock_series(
        trading_days, top_holdings, aligned_stocks, qqq_global
    )
    nan_count = stock_chain.isna().sum()
    if nan_count:
        print(f"  [warning] {nan_count} NaN values in chain — check stock coverage")
    print(f"  Chain: 1.0000 → {stock_chain.iloc[-1]:.4f}")

    qqq_arr   = qqq_global.values.astype(float)
    stock_arr = stock_chain.values.astype(float)

    print("\nRunning rolling-window analysis...")
    print(f"  Allocations : {len(ALLOCATIONS)}  (QQQ 0%…100%, Stock 100%…0%)")
    print(f"  Horizons    : {HORIZONS} years")
    print(f"  Window type : daily rolling (slides 1 trading day at a time)")

    frames = []
    for H in HORIZONS:
        df_h = analyze_horizon(H, qqq_arr, stock_arr, trading_days)
        if not df_h.empty:
            frames.append(df_h)

    if not frames:
        print("\nNo results — check data coverage.")
        return

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved → {OUTPUT_FILE}")
    print(f"  Rows: {len(result):,}  ({len(frames)} horizons × {len(ALLOCATIONS)} allocations)")

    # Quick sanity print: 15y horizon at key allocations
    h15 = result[result["horizon_years"] == 15]
    if not h15.empty:
        cols = ["qqq_pct", "stock_pct",
                "total_return_mean", "cagr_mean",
                "max_drawdown_mean", "sharpe_ratio_mean"]
        sample = h15[h15["qqq_pct"].isin([0, 25, 50, 75, 100])][cols]
        print("\n── 15-year windows: mean metrics at key allocations ──")
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
