"""
Grid search: NDX Top-1 vs QQQ allocation
Sweeps NDXA_PCT from 0% to 100% in 5% steps (21 combinations).
Ranks results by CAGR, Sharpe, Calmar, and Total Return.
Exports results to n_q_optimize_results.csv
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR      = Path(__file__).parent
BREADTH_FILE  = DATA_DIR / "S5TH.csv"
NDX_FILE      = DATA_DIR / "NASDAQ100.csv"
HOLDINGS_FILE = DATA_DIR / "NASDAQ100" / "nasdaq100_top_holdings.csv"
PRICES_DIR    = DATA_DIR / "NASDAQ100" / "stock_prices"

# ── Signal parameters ─────────────────────────────────────────────────────────
BUY_B200_THRESH         = 26.0
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0
COMMISSION              = 1.0
SLIPPAGE                = 0.0005

# ── Portfolio ─────────────────────────────────────────────────────────────────
INITIAL_CAPITAL    = 153_402.0
ANNUAL_CONTRIB     = 26_880.0
CONTRIB_MONTH      = 4
CONTRIB_DAY        = 6
CONTRIB_START_YEAR = 2012
START_DATE         = pd.Timestamp("2011-01-01")

NAME_TO_TICKER = {
    "Cisco Systems Inc.":    "CSCO",
    "Microsoft Corporation": "MSFT",
    "Microsoft Corp.":       "MSFT",
    "Intel Corporation":     "INTC",
    "QUALCOMM Inc.":         "QCOM",
    "eBay Inc.":             "EBAY",
    "Apple Computer Inc.":   "AAPL",
    "Apple Inc.":            "AAPL",
    "Google Inc. Class A":   "GOOGL",
    "Google Inc. Class C":   "GOOGL",
    "Alphabet Inc.":         "GOOGL",
    "Amazon.com Inc.":       "AMZN",
    "NVIDIA Corp.":          "NVDA",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (run once, shared across all combos)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> tuple[pd.DataFrame, dict, dict]:
    b = pd.read_csv(BREADTH_FILE)
    b["Date"] = pd.to_datetime(b["Date"], format="%m/%d/%Y")
    b.set_index("Date", inplace=True)
    b["breadth"] = _parse_price(b["Price"])

    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx["price"] = _parse_price(ndx["Price"])

    merged = ndx[["price"]].join(b[["breadth"]], how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]
    merged = merged[merged.index >= START_DATE]

    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)

    df_hold = pd.read_csv(HOLDINGS_FILE)
    holdings: dict[int, list[tuple[str, float]]] = {}
    for _, row in df_hold.iterrows():
        year  = int(row["Year"])
        name  = str(row.get("#1 Holding", "")).strip()
        val   = float(row.get("#1 Value ($B)", 0) or 0)
        ticker = NAME_TO_TICKER.get(name)
        if ticker and val > 0:
            holdings[year] = [(ticker, 1.0)]

    tickers = {t for comps in holdings.values() for t, _ in comps}
    stock_prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = PRICES_DIR / f"{ticker}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        col = "Close" if "Close" in df.columns else df.columns[0]
        stock_prices[ticker] = df[col].dropna()

    return merged, holdings, stock_prices


def _contribution_dates(trading_days: pd.DatetimeIndex) -> set[pd.Timestamp]:
    end_year = trading_days[-1].year
    dates: set[pd.Timestamp] = set()
    for year in range(CONTRIB_START_YEAR, end_year + 1):
        target     = pd.Timestamp(year, CONTRIB_MONTH, CONTRIB_DAY)
        candidates = trading_days[trading_days >= target]
        if len(candidates) > 0:
            dates.add(candidates[0])
    return dates


def _get_price(prices: dict[str, pd.Series], ticker: str, date: pd.Timestamp) -> float | None:
    s = prices.get(ticker)
    if s is None or s.empty:
        return None
    idx = s.index[s.index <= date]
    return float(s.loc[idx[-1]]) if not idx.empty else None


def _build_basket(cash: float, composition: list[tuple[str, float]],
                  prices: dict[str, pd.Series], date: pd.Timestamp) -> dict[str, float]:
    available = [(t, w) for t, w in composition if _get_price(prices, t, date) is not None]
    if not available:
        return {}
    total_w = sum(w for _, w in available)
    basket: dict[str, float] = {}
    for ticker, weight in available:
        alloc          = cash * (weight / total_w)
        price          = _get_price(prices, ticker, date) * (1 + SLIPPAGE)
        basket[ticker] = alloc / price
    return basket


def _basket_value(basket: dict[str, float], prices: dict[str, pd.Series],
                  date: pd.Timestamp) -> float:
    return sum(
        shares * p
        for ticker, shares in basket.items()
        if (p := _get_price(prices, ticker, date)) is not None
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation (parameterised by allocation)
# ─────────────────────────────────────────────────────────────────────────────

def simulate(df_ndx: pd.DataFrame,
             h1: dict[int, list[tuple[str, float]]],
             stock_prices: dict[str, pd.Series],
             ndxa_pct: float) -> pd.Series:
    """Run one portfolio simulation and return daily equity series."""
    qqqb_pct = 1.0 - ndxa_pct

    common_idx    = df_ndx.index.sort_values()
    contrib_dates = _contribution_dates(common_idx)

    n_pos           = "OUT"
    n_basket: dict[str, float] = {}
    n_cash          = INITIAL_CAPITAL * ndxa_pct
    n_orig_val      = 0.0
    n_mid_contrib   = 0.0
    n_entry_date    = None
    prev_year: int | None = None

    q_pos         = "OUT"
    q_shares      = 0.0
    q_cash        = INITIAL_CAPITAL * qqqb_pct
    q_entry_price = 0.0
    q_mid_contrib = 0.0

    trade_open      = False
    pending_contrib = 0.0
    combined_values: dict = {}

    def q_mktval(price: float) -> float:
        return q_shares * price if q_pos == "IN" else q_cash

    def n_mktval(date: pd.Timestamp) -> float:
        return _basket_value(n_basket, stock_prices, date) if n_pos == "IN" else n_cash

    for date in common_idx:
        row_n   = df_ndx.loc[date]
        n_price = float(row_n["price"])
        breadth = float(row_n["breadth"])
        sig_pr  = bool(row_n["price_rose"])
        sig_bf  = bool(row_n["breadth_fell"])
        year    = date.year

        # Annual contribution
        if date in contrib_dates:
            if not trade_open:
                pending_contrib += ANNUAL_CONTRIB
            else:
                n_add = ANNUAL_CONTRIB * ndxa_pct
                q_add = ANNUAL_CONTRIB * qqqb_pct
                if n_pos == "IN" and ndxa_pct > 0:
                    cur_comp = h1.get(year, h1.get(year - 1, []))
                    if cur_comp:
                        extra = _build_basket(n_add - COMMISSION, cur_comp, stock_prices, date)
                        for ticker, shares in extra.items():
                            n_basket[ticker] = n_basket.get(ticker, 0.0) + shares
                    n_mid_contrib += n_add
                elif ndxa_pct > 0:
                    n_cash += n_add
                if q_pos == "IN" and qqqb_pct > 0:
                    new_sh = (q_add - COMMISSION) / (n_price * (1 + SLIPPAGE))
                    q_shares += new_sh
                    q_mid_contrib += q_add
                elif qqqb_pct > 0:
                    q_cash += q_add

        # NDX-A annual rebalance
        if n_pos == "IN" and prev_year is not None and year != prev_year:
            new_comp = h1.get(year, h1.get(year - 1, []))
            cur = _basket_value(n_basket, stock_prices, date)
            if cur > 0 and new_comp:
                after_sell = cur * (1 - SLIPPAGE) - COMMISSION
                n_basket   = _build_basket(after_sell, new_comp, stock_prices, date)
        prev_year = year

        # Sell
        if sig_pr and sig_bf and breadth < DIVERGENCE_BREADTH_CAP and trade_open:
            if n_pos == "IN":
                cur_n  = _basket_value(n_basket, stock_prices, date)
                n_cash = cur_n * (1 - SLIPPAGE) - COMMISSION
                n_basket = {}
                n_pos  = "OUT"
            if q_pos == "IN":
                q_cash   = q_shares * n_price * (1 - SLIPPAGE) - COMMISSION
                q_shares = 0.0
                q_pos    = "OUT"
            trade_open = False

        # Mark to market
        n_val = n_mktval(date)
        q_val = q_mktval(n_price)
        total = n_val + q_val

        # Buy
        if not trade_open and not pd.isna(breadth) and breadth < BUY_B200_THRESH:
            total_capital   = n_cash + q_cash + pending_contrib
            pending_contrib = 0.0

            if ndxa_pct > 0:
                comp = h1.get(year, [])
                if comp:
                    n_buy    = total_capital * ndxa_pct - COMMISSION
                    n_basket = _build_basket(n_buy, comp, stock_prices, date)
                    n_orig_val    = _basket_value(n_basket, stock_prices, date)
                    n_mid_contrib = 0.0
                    n_entry_date  = date
                    n_pos  = "IN"
                    n_cash = 0.0
                else:
                    n_cash = total_capital * ndxa_pct
            else:
                n_cash = 0.0

            if qqqb_pct > 0:
                q_buy         = total_capital * qqqb_pct - COMMISSION
                q_shares      = q_buy / (n_price * (1 + SLIPPAGE))
                q_entry_price = n_price
                q_mid_contrib = 0.0
                q_pos  = "IN"
                q_cash = 0.0
            else:
                q_cash = 0.0

            n_val = n_mktval(date)
            q_val = q_mktval(n_price)
            total = n_val + q_val
            trade_open = True

        combined_values[date] = total

    return pd.Series(combined_values, name="portfolio")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def metrics(equity: pd.Series, ndxa_pct: float) -> dict:
    dr    = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    init  = equity.iloc[0]
    final = equity.iloc[-1]
    tr    = (final / init) - 1
    cagr  = (final / init) ** (1 / years) - 1 if years > 0 else 0.0
    mdd   = float(((equity - equity.cummax()) / equity.cummax()).min())
    std   = float(dr.std())
    sh    = float(dr.mean()) / std * np.sqrt(252) if std > 0 else 0.0
    calmar = abs(cagr / mdd) if mdd != 0 else 0.0
    return {
        "NDX_Top1_%": round(ndxa_pct * 100),
        "QQQ_%":      round((1 - ndxa_pct) * 100),
        "CAGR_%":     round(cagr * 100, 2),
        "Total_Return_%": round(tr * 100, 1),
        "Max_DD_%":   round(mdd * 100, 2),
        "Sharpe":     round(sh, 3),
        "Calmar":     round(calmar, 3),
        "Final_$":    round(final),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data…")
    df_ndx, h1, stock_prices = load_data()
    print(f"Backtest period : {df_ndx.index[0].date()} → {df_ndx.index[-1].date()}")
    print(f"Initial capital : ${INITIAL_CAPITAL:,.0f}  |  Annual contrib: ${ANNUAL_CONTRIB:,.0f}")
    print()

    steps = [i / 100 for i in range(0, 101, 5)]   # 0%, 5%, 10%, … 100%
    results = []
    for ndxa_pct in steps:
        equity = simulate(df_ndx, h1, stock_prices, ndxa_pct)
        results.append(metrics(equity, ndxa_pct))
        print(f"  NDX {ndxa_pct:4.0%} / QQQ {1-ndxa_pct:4.0%}  →  "
              f"CAGR {results[-1]['CAGR_%']:5.2f}%  "
              f"Sharpe {results[-1]['Sharpe']:5.3f}  "
              f"Calmar {results[-1]['Calmar']:5.3f}  "
              f"MDD {results[-1]['Max_DD_%']:6.2f}%  "
              f"Final ${results[-1]['Final_$']:>15,.0f}")

    df = pd.DataFrame(results)

    # ── Rankings ──────────────────────────────────────────────────────────────
    def rank_table(df: pd.DataFrame, sort_col: str, ascending: bool = False) -> pd.DataFrame:
        return (df.sort_values(sort_col, ascending=ascending)
                  .reset_index(drop=True)
                  .assign(Rank=lambda x: x.index + 1)
                  [["Rank","NDX_Top1_%","QQQ_%","CAGR_%","Total_Return_%",
                    "Max_DD_%","Sharpe","Calmar","Final_$"]])

    for sort_col, label in [
        ("CAGR_%",          "CAGR"),
        ("Sharpe",          "Sharpe Ratio"),
        ("Calmar",          "Calmar Ratio"),
        ("Total_Return_%",  "Total Return"),
    ]:
        print(f"\n{'='*90}")
        print(f"  Ranked by {label}")
        print(f"{'='*90}")
        rt = rank_table(df, sort_col)
        hdr = (f"  {'Rank':>4}  {'NDX%':>5}  {'QQQ%':>5}  {'CAGR%':>7}  "
               f"{'TotRet%':>9}  {'MaxDD%':>7}  {'Sharpe':>7}  {'Calmar':>7}  {'Final $':>15}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for _, r in rt.iterrows():
            print(f"  {int(r['Rank']):>4}  {int(r['NDX_Top1_%']):>4}%  {int(r['QQQ_%']):>4}%  "
                  f"{r['CAGR_%']:>7.2f}  {r['Total_Return_%']:>9.1f}  "
                  f"{r['Max_DD_%']:>7.2f}  {r['Sharpe']:>7.3f}  {r['Calmar']:>7.3f}  "
                  f"${r['Final_$']:>14,.0f}")

    out = DATA_DIR / "n_q_optimize_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
