"""
qqq_ndx_topN_sweep.py

Improved rolling-window sweep over QQQ / NDX Top-N basket allocations.

Improvements over qqq_ndx_top1_sweep.py:
  1. Extended date range 2001–2025 (adds dot-com bust + 2008 financial crisis)
  2. Three basket variants: top-1, equal-weight top-2, equal-weight top-3
     — At each calendar-year boundary the basket rebalances to equal weight
       among the new year's top-N NDX holdings.
  3. Non-overlapping windows alongside rolling windows for honest significance:
     — n_eff, t-stat, p-value computed on truly independent samples
  4. Extra column: pct_windows_positive (% of rolling windows with TR > 0)

Output: qqq_ndx_topN_sweep_results.csv
"""

import numpy as np
import pandas as pd
from math import sqrt, erfc
from pathlib import Path

DATA_DIR          = Path(__file__).parent
NDX_FILE          = DATA_DIR / "NASDAQ100.csv"
TOP_HOLDINGS_FILE = DATA_DIR / "NASDAQ100" / "nasdaq100_top10_holdings.csv"
STOCK_PRICE_DIR   = DATA_DIR / "NASDAQ100" / "stock_prices"
OUTPUT_FILE       = DATA_DIR / "qqq_ndx_topN_sweep_results.csv"

INITIAL_CAPITAL = 10_000.0
GLOBAL_START    = pd.Timestamp("2001-01-01")   # extended: dot-com + 2008
GLOBAL_END      = pd.Timestamp("2025-12-31")
HORIZONS        = [1, 3, 5, 10, 15]
BASKET_SIZES    = [1, 2, 3]
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
    ("booking",           "BKNG"),
    ("honeywell",         "HON"),
    ("intuit",            "INTU"),
    ("adobe",             "ADBE"),
    ("walgreens",         "WBA"),
    ("regeneron",         "REGN"),
    ("applied materials", "AMAT"),
    ("lam research",      "LRCX"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _name_to_ticker(name: str) -> str | None:
    nl = name.lower()
    for key, ticker in _NAME_TO_TICKER:
        if key in nl:
            return ticker
    return None


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _norm_cdf(x: float) -> float:
    return 0.5 * erfc(-x / sqrt(2))


def _t_pvalue(t: float, df: int) -> float:
    """Two-sided p-value. Cornish-Fisher approx for df < 30, normal for df >= 30."""
    if df < 1:
        return float("nan")
    if df >= 30:
        z = t
    else:
        z = t * (1.0 - 1.0 / (4 * df)) / sqrt(1.0 + t * t / (2.0 * df))
    return 2.0 * (1.0 - _norm_cdf(abs(z)))


# ── data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.Series, dict[tuple[int, int], str], dict[str, pd.Series]]:
    """Return qqq_prices, {(year, rank): ticker}, {ticker: price_series}."""

    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price"})
    ndx["price"] = _parse_price(ndx["price"])
    ndx.sort_index(inplace=True)
    qqq_prices = ndx["price"]

    holdings_df = pd.read_csv(TOP_HOLDINGS_FILE)
    top_holdings: dict[tuple[int, int], str] = {}
    for _, row in holdings_df.iterrows():
        rank = int(row["Rank"])
        if rank <= max(BASKET_SIZES):
            ticker = _name_to_ticker(str(row["Holding"]))
            if ticker:
                top_holdings[(int(row["Year"]), rank)] = ticker

    unique_tickers = set(top_holdings.values())
    aligned_stocks: dict[str, pd.Series] = {}
    missing = []
    for ticker in sorted(unique_tickers):
        path = STOCK_PRICE_DIR / f"{ticker}.csv"
        if not path.exists():
            missing.append(ticker)
            continue
        df = pd.read_csv(path)
        df["Date"] = (
            pd.to_datetime(df["Date"], format="mixed", utc=True)
            .dt.tz_localize(None)
        )
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        aligned_stocks[ticker] = df["Close"].astype(float)

    if missing:
        print(f"  [missing CSVs → QQQ fallback]: {', '.join(missing)}")

    return qqq_prices, top_holdings, aligned_stocks


# ── basket chain ──────────────────────────────────────────────────────────────

def build_chained_basket_series(
    trading_days: pd.DatetimeIndex,
    top_holdings: dict[tuple[int, int], str],
    aligned_stocks: dict[str, pd.Series],
    qqq_prices: pd.Series,
    N: int,
) -> pd.Series:
    """
    Build a wealth-index for an equal-weight top-N annual basket strategy.

    At the first trading day of each calendar year the basket rebalances
    equally across that year's top-N NDX holdings.  A slot with no CSV data
    falls back to QQQ returns.  Index starts at 1.0 on the first trading day.
    """
    chain      = pd.Series(np.nan, index=trading_days, dtype=float)
    prev_value = 1.0
    years      = sorted(set(d.year for d in trading_days))

    for year in years:
        mask       = trading_days.year == year
        year_dates = trading_days[mask]
        if len(year_dates) == 0:
            continue

        slot_norms: list[pd.Series] = []
        for rank in range(1, N + 1):
            ticker = top_holdings.get((year, rank)) or top_holdings.get((year - 1, rank))

            if ticker and ticker in aligned_stocks:
                raw = aligned_stocks[ticker].reindex(year_dates).ffill()
            else:
                raw = qqq_prices.reindex(year_dates).ffill()

            first_idx = raw.first_valid_index()
            if first_idx is None:
                slot_norms.append(pd.Series(1.0, index=year_dates))
                continue

            first_px = float(raw[first_idx])
            if first_px == 0 or np.isnan(first_px):
                slot_norms.append(pd.Series(1.0, index=year_dates))
                continue

            slot_norms.append(raw / first_px)

        avg_norm = sum(slot_norms) / len(slot_norms)
        chain[year_dates] = prev_value * avg_norm

        last_val = float(chain[year_dates[-1]])
        if not np.isnan(last_val):
            prev_value = last_val

    chain = chain.ffill()
    return chain


# ── vectorised metrics ────────────────────────────────────────────────────────

def _metrics_vectorized(
    portfolio: np.ndarray,
    years: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    portfolio : (N_alloc, T) array of daily portfolio values
    Returns   : total_return, final_value, cagr, mdd, sharpe — each (N_alloc,)
    """
    final = portfolio[:, -1]
    tr    = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    fv    = final
    cagr  = (final / INITIAL_CAPITAL) ** (1.0 / years) - 1.0

    run_max = np.maximum.accumulate(portfolio, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd  = np.where(run_max > 0, (portfolio - run_max) / run_max, 0.0)
    mdd = dd.min(axis=1)

    d_ret  = np.diff(portfolio, axis=1) / portfolio[:, :-1]
    d_mean = d_ret.mean(axis=1)
    d_std  = d_ret.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(d_std > 0, d_mean / d_std * np.sqrt(252), 0.0)

    return tr, fv, cagr, mdd, sharpe


# ── non-overlapping window generator ─────────────────────────────────────────

def nonoverlap_windows(H_years: int, dates: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """
    Index pairs (i_start, i_end) for non-overlapping H-year windows
    anchored at the first available trading day, stepping H calendar years.
    """
    offset  = pd.DateOffset(years=H_years)
    windows = []
    anchor  = dates[0]

    while True:
        end_target = anchor + offset
        if end_target > GLOBAL_END:
            break
        i0 = int(dates.searchsorted(anchor,     side="left"))
        i1 = int(dates.searchsorted(end_target, side="right")) - 1
        if i1 > i0 and i0 < len(dates):
            windows.append((i0, i1))
        anchor = end_target

    return windows


# ── per-horizon analysis ──────────────────────────────────────────────────────

def analyze_horizon(
    H_years: int,
    N: int,
    qqq_arr: np.ndarray,
    stock_arr: np.ndarray,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Rolling + non-overlapping stats for one (horizon, basket_size) combo."""

    q_col  = ALLOCATIONS
    s_col  = 1.0 - ALLOCATIONS
    offset = pd.DateOffset(years=H_years)

    # ── rolling windows ───────────────────────────────────────────────────────
    acc_tr: list[np.ndarray] = []
    acc_fv: list[np.ndarray] = []
    acc_cag: list[np.ndarray] = []
    acc_mdd: list[np.ndarray] = []
    acc_sh: list[np.ndarray] = []

    for i0 in range(len(dates)):
        end_target = dates[i0] + offset
        if end_target > GLOBAL_END:
            break
        i1 = int(dates.searchsorted(end_target, side="right")) - 1
        if i1 <= i0:
            continue
        actual_years = (dates[i1] - dates[i0]).days / 365.25
        if actual_years < H_years * 0.9:
            continue
        q0, s0 = qqq_arr[i0], stock_arr[i0]
        if q0 == 0 or s0 == 0 or np.isnan(q0) or np.isnan(s0):
            continue

        qqq_n   = qqq_arr[i0 : i1 + 1]   / q0
        stock_n = stock_arr[i0 : i1 + 1] / s0
        portfolio = INITIAL_CAPITAL * (
            np.outer(q_col, qqq_n) + np.outer(s_col, stock_n)
        )
        tr, fv, cagr, mdd, sh = _metrics_vectorized(portfolio, actual_years)
        acc_tr.append(tr)
        acc_fv.append(fv)
        acc_cag.append(cagr)
        acc_mdd.append(mdd)
        acc_sh.append(sh)

    n_rolling = len(acc_tr)
    print(f"    top-{N}  {H_years}y: {n_rolling:,} rolling", end="")

    if n_rolling == 0:
        print()
        return pd.DataFrame()

    def _agg(mat: np.ndarray) -> dict[str, np.ndarray]:
        return {
            "mean":   mat.mean(axis=0),
            "median": np.median(mat, axis=0),
            "std":    mat.std(axis=0, ddof=1),
            "p25":    np.percentile(mat, 25, axis=0),
            "p75":    np.percentile(mat, 75, axis=0),
        }

    TR  = np.vstack(acc_tr)
    s_tr  = _agg(TR)
    s_fv  = _agg(np.vstack(acc_fv))
    s_cag = _agg(np.vstack(acc_cag))
    s_mdd = _agg(np.vstack(acc_mdd))
    s_sh  = _agg(np.vstack(acc_sh))
    pct_pos = (TR > 0).mean(axis=0)

    # ── non-overlapping windows ───────────────────────────────────────────────
    nw_pairs = nonoverlap_windows(H_years, dates)
    n_eff    = len(nw_pairs)
    no_cag: list[np.ndarray] = []
    no_sh:  list[np.ndarray] = []

    for i0, i1 in nw_pairs:
        q0, s0 = qqq_arr[i0], stock_arr[i0]
        if q0 == 0 or s0 == 0 or np.isnan(q0) or np.isnan(s0):
            continue
        actual_years = (dates[i1] - dates[i0]).days / 365.25
        qqq_n   = qqq_arr[i0 : i1 + 1]   / q0
        stock_n = stock_arr[i0 : i1 + 1] / s0
        portfolio = INITIAL_CAPITAL * (
            np.outer(q_col, qqq_n) + np.outer(s_col, stock_n)
        )
        _, _, cagr, _, sh = _metrics_vectorized(portfolio, actual_years)
        no_cag.append(cagr)
        no_sh.append(sh)

    print(f",  {n_eff} non-overlapping")

    if no_cag and n_eff > 1:
        NC = np.vstack(no_cag)
        NS = np.vstack(no_sh)
        no_cag_mean = NC.mean(axis=0)
        no_cag_std  = NC.std(axis=0, ddof=1)
        no_sh_mean  = NS.mean(axis=0)
        no_sh_std   = NS.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_cag = np.where(no_cag_std > 0, no_cag_mean / (no_cag_std / sqrt(n_eff)), np.nan)
            t_sh  = np.where(no_sh_std  > 0, no_sh_mean  / (no_sh_std  / sqrt(n_eff)), np.nan)
        df_t = n_eff - 1
        p_cag = np.array([_t_pvalue(float(t), df_t) if not np.isnan(t) else np.nan for t in t_cag])
        p_sh  = np.array([_t_pvalue(float(t), df_t) if not np.isnan(t) else np.nan for t in t_sh])
    elif no_cag:
        NC = np.vstack(no_cag)
        NS = np.vstack(no_sh)
        no_cag_mean = NC.mean(axis=0)
        no_cag_std  = np.full(len(ALLOCATIONS), np.nan)
        no_sh_mean  = NS.mean(axis=0)
        no_sh_std   = np.full(len(ALLOCATIONS), np.nan)
        t_cag = p_cag = t_sh = p_sh = np.full(len(ALLOCATIONS), np.nan)
    else:
        no_cag_mean = no_cag_std = t_cag = p_cag = np.full(len(ALLOCATIONS), np.nan)
        no_sh_mean  = no_sh_std  = t_sh  = p_sh  = np.full(len(ALLOCATIONS), np.nan)

    # ── assemble output rows ──────────────────────────────────────────────────
    metric_stats = [
        ("total_return", s_tr),
        ("final_value",  s_fv),
        ("cagr",         s_cag),
        ("max_drawdown", s_mdd),
        ("sharpe_ratio", s_sh),
    ]
    rows = []
    for i, q in enumerate(ALLOCATIONS):
        qqq_pct = int(round(q * 100))
        row: dict = {
            "basket_size":       N,
            "horizon_years":     H_years,
            "qqq_pct":           qqq_pct,
            "stock_pct":         100 - qqq_pct,
            "n_windows_rolling": n_rolling,
        }
        for name, s in metric_stats:
            for stat in ("mean", "median", "std", "p25", "p75"):
                row[f"{name}_{stat}"] = round(float(s[stat][i]), 6)
        row["pct_windows_positive"]   = round(float(pct_pos[i]), 4)
        row["n_windows_nonoverlap"]   = n_eff

        def _r(v: float) -> float:
            return round(v, 6) if not np.isnan(v) else float("nan")

        row["cagr_nonoverlap_mean"]   = _r(float(no_cag_mean[i]))
        row["cagr_nonoverlap_std"]    = _r(float(no_cag_std[i]))
        row["cagr_nonoverlap_tstat"]  = _r(float(t_cag[i]))
        row["cagr_nonoverlap_pvalue"] = _r(float(p_cag[i]))
        row["sharpe_nonoverlap_mean"] = _r(float(no_sh_mean[i]))
        row["sharpe_nonoverlap_std"]  = _r(float(no_sh_std[i]))
        row["sharpe_nonoverlap_tstat"]  = _r(float(t_sh[i]))
        row["sharpe_nonoverlap_pvalue"] = _r(float(p_sh[i]))
        rows.append(row)

    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    qqq_prices, top_holdings, aligned_stocks = load_data()

    qqq_global   = qqq_prices.loc[GLOBAL_START:GLOBAL_END]
    trading_days = qqq_global.index
    print(
        f"Trading days: {trading_days[0].date()} → {trading_days[-1].date()}"
        f"  ({len(trading_days):,} days)"
    )

    qqq_arr = qqq_global.values.astype(float)

    print(f"\nBuilding basket chains and running analysis...")
    print(f"  Horizons: {HORIZONS}y   Baskets: top-{BASKET_SIZES}   Allocations: {len(ALLOCATIONS)}")

    frames = []
    for N in BASKET_SIZES:
        print(f"\nBasket top-{N}:")
        basket_chain = build_chained_basket_series(
            trading_days, top_holdings, aligned_stocks, qqq_global, N
        )
        stock_arr = basket_chain.values.astype(float)

        for H in HORIZONS:
            df_h = analyze_horizon(H, N, qqq_arr, stock_arr, trading_days)
            if not df_h.empty:
                frames.append(df_h)

    if not frames:
        print("No results generated.")
        return

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")
    print(f"  Rows: {len(result):,}  ({len(BASKET_SIZES)} baskets × {len(HORIZONS)} horizons × {len(ALLOCATIONS)} allocations)")

    # ── summary tables ────────────────────────────────────────────────────────
    print("\n── Non-overlapping 1y windows at QQQ 70%: proper significance ──")
    h1 = result[(result.horizon_years == 1) & (result.qqq_pct == 70)]
    cols = ["basket_size", "n_windows_nonoverlap",
            "cagr_nonoverlap_mean", "cagr_nonoverlap_std",
            "cagr_nonoverlap_tstat", "cagr_nonoverlap_pvalue",
            "pct_windows_positive"]
    print(h1[cols].to_string(index=False))

    print("\n── Basket comparison at QQQ 70%, 5y rolling ──")
    h5 = result[(result.horizon_years == 5) & (result.qqq_pct == 70)]
    print(h5[["basket_size", "cagr_mean", "cagr_std",
               "max_drawdown_mean", "sharpe_ratio_mean",
               "pct_windows_positive"]].to_string(index=False))

    print("\n── Optimal allocation by Sharpe (10y rolling, per basket) ──")
    h10 = result[result.horizon_years == 10]
    for N in BASKET_SIZES:
        sub  = h10[h10.basket_size == N]
        best = sub.loc[sub.sharpe_ratio_mean.idxmax()]
        print(
            f"  top-{N}: best Sharpe at QQQ={int(best.qqq_pct):3d}%  "
            f"CAGR={best.cagr_mean:.2%}  SR={best.sharpe_ratio_mean:.3f}  "
            f"MDD={best.max_drawdown_mean:.2%}"
        )


if __name__ == "__main__":
    main()
