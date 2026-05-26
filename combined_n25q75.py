"""
Combined Portfolio: NDX Top-1 25% / QQQ 75%  —  Signal + April Rebalance

Initial capital : $153,402
Annual funding  : $26,880 on April 6 each year (starting 2012)
Start date      : 2011-01-01
Rebalancing     : Full 25/75 rebalance every April while a trade is open;
                  pending cash on April dates when out of market
Signals:
  BUY : S&P 500 breadth200 < 26%
  SELL: NDX price rose >= 3% over 60d AND breadth fell >= 20pts AND breadth < 60%

NDX Top-1 leg rotates to the current year's #1 NDX holding at each year-start
(and at every April rebalance).

Exports:
  combined_n25q75_metrics.csv
  combined_n25q75_combined.csv   — combined portfolio trade log
  combined_n25q75_ndxa.csv       — NDX Top-1 leg trade log
  combined_n25q75_qqqb.csv       — QQQ-B (NDX proxy) leg trade log
  combined_n25q75_rebal.csv      — April rebalance log
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

NDXA_PCT = 0.25
QQQB_PCT = 0.75

# ── Name → ticker ─────────────────────────────────────────────────────────────
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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _load_breadth() -> pd.DataFrame:
    b = pd.read_csv(BREADTH_FILE)
    b["Date"] = pd.to_datetime(b["Date"], format="%m/%d/%Y")
    b.set_index("Date", inplace=True)
    b["Price"] = _parse_price(b["Price"])
    return b[["Price"]].rename(columns={"Price": "breadth"})


def load_ndx(breadth: pd.DataFrame) -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx["price"] = _parse_price(ndx["Price"])
    merged = ndx[["price"]].join(breadth, how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]
    merged = merged[merged.index >= START_DATE]
    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)
    return merged


def load_holdings(top_n: int = 1) -> dict[int, list[tuple[str, float]]]:
    df = pd.read_csv(HOLDINGS_FILE)
    holdings: dict[int, list[tuple[str, float]]] = {}
    for _, row in df.iterrows():
        year  = int(row["Year"])
        pairs = []
        for i in [str(n) for n in range(1, top_n + 1)]:
            name   = str(row.get(f"#{i} Holding", "")).strip()
            val    = float(row.get(f"#{i} Value ($B)", 0) or 0)
            ticker = NAME_TO_TICKER.get(name)
            if ticker and val > 0:
                pairs.append((ticker, val))
        if pairs:
            total = sum(v for _, v in pairs)
            holdings[year] = [(t, v / total) for t, v in pairs]
    return holdings


def load_stock_prices(tickers: set[str]) -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    for ticker in tickers:
        path = PRICES_DIR / f"{ticker}.csv"
        if not path.exists():
            print(f"  [WARNING] No price file for {ticker}, skipping.")
            continue
        df  = pd.read_csv(path, index_col=0, parse_dates=True)
        col = "Close" if "Close" in df.columns else df.columns[0]
        prices[ticker] = df[col].dropna()
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# Basket helpers
# ─────────────────────────────────────────────────────────────────────────────

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
# Contribution dates
# ─────────────────────────────────────────────────────────────────────────────

def _contribution_dates(trading_days: pd.DatetimeIndex) -> set[pd.Timestamp]:
    end_year = trading_days[-1].year
    dates: set[pd.Timestamp] = set()
    for year in range(CONTRIB_START_YEAR, end_year + 1):
        target     = pd.Timestamp(year, CONTRIB_MONTH, CONTRIB_DAY)
        candidates = trading_days[trading_days >= target]
        if len(candidates) > 0:
            dates.add(candidates[0])
    return dates


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio(
    df_ndx:       pd.DataFrame,
    h1:           dict[int, list[tuple[str, float]]],
    stock_prices: dict[str, pd.Series],
) -> tuple[pd.Series, list[dict], list[dict], list[dict], list[dict]]:
    """
    Signal-based entry/exit with full 25/75 April rebalance while in a trade.
    Returns (equity_series, comb_trades, ndxa_trades, qqqb_trades, rebal_log).
    """
    common_idx    = df_ndx.index.sort_values()
    contrib_dates = _contribution_dates(common_idx)

    # ── NDX Top-1 leg ─────────────────────────────────────────────────────────
    n_pos           = "OUT"
    n_basket: dict[str, float] = {}
    n_cash          = INITIAL_CAPITAL * NDXA_PCT
    n_entry_date: pd.Timestamp | None = None
    n_entry_val     = 0.0
    n_orig_val      = 0.0
    n_mid_contrib   = 0.0
    n_price_low     = 0.0
    n_port_peak     = n_cash
    n_worst_dd      = 0.0
    n_trough        = n_cash
    n_entry_tickers: list[str] = []
    prev_year: int | None = None

    # ── QQQ-B leg ─────────────────────────────────────────────────────────────
    q_pos         = "OUT"
    q_shares      = 0.0
    q_cash        = INITIAL_CAPITAL * QQQB_PCT
    q_entry_date: pd.Timestamp | None = None
    q_entry_val   = 0.0
    q_entry_price = 0.0
    q_mid_contrib = 0.0
    q_price_low   = 0.0
    q_port_peak   = q_cash
    q_worst_dd    = 0.0
    q_trough      = q_cash

    # ── Combined state ────────────────────────────────────────────────────────
    trade_open        = False
    comb_entry_date: pd.Timestamp | None = None
    comb_entry_val    = 0.0
    comb_mid_contribs = 0.0
    comb_peak         = 0.0
    comb_worst_dd     = 0.0
    comb_trough       = 0.0
    pending_contrib   = 0.0

    combined_values: dict = {}
    comb_trades: list[dict] = []
    ndxa_trades: list[dict] = []
    qqqb_trades: list[dict] = []
    rebal_log:   list[dict] = []

    def n_mktval(date: pd.Timestamp) -> float:
        return _basket_value(n_basket, stock_prices, date) if n_pos == "IN" else n_cash

    def q_mktval(price: float) -> float:
        return q_shares * price if q_pos == "IN" else q_cash

    for date in common_idx:
        row_n   = df_ndx.loc[date]
        n_price = float(row_n["price"])
        breadth = float(row_n["breadth"])
        sig_pr  = bool(row_n["price_rose"])
        sig_bf  = bool(row_n["breadth_fell"])
        year    = date.year

        # ── 0. April contribution / rebalance ─────────────────────────────────
        if date in contrib_dates:
            if not trade_open:
                pending_contrib += ANNUAL_CONTRIB
                print(f"  [contrib] {date.date()}: +${ANNUAL_CONTRIB:,.0f} pending (OUT)")
            else:
                # Full rebalance: liquidate both legs, add contrib, re-buy at 25/75
                n_val_pre = _basket_value(n_basket, stock_prices, date) if n_pos == "IN" else n_cash
                q_val_pre = q_shares * n_price if q_pos == "IN" else q_cash
                liquid = (n_val_pre * (1 - SLIPPAGE) - COMMISSION +
                          q_val_pre * (1 - SLIPPAGE) - COMMISSION +
                          ANNUAL_CONTRIB)

                comp = h1.get(year, h1.get(year - 1, []))
                if comp:
                    n_basket = _build_basket(liquid * NDXA_PCT - COMMISSION, comp, stock_prices, date)
                    n_orig_val    = _basket_value(n_basket, stock_prices, date)
                    n_entry_tickers = list(n_basket.keys())
                    n_pos  = "IN"; n_cash = 0.0
                else:
                    n_cash = liquid * NDXA_PCT; n_pos = "OUT"; n_basket = {}

                q_shares      = (liquid * QQQB_PCT - COMMISSION) / (n_price * (1 + SLIPPAGE))
                q_entry_price = n_price
                q_pos = "IN"; q_cash = 0.0

                n_val_post = _basket_value(n_basket, stock_prices, date) if n_pos == "IN" else n_cash
                q_val_post = q_shares * n_price
                post_total = n_val_post + q_val_post
                ndx_pct    = n_val_post / post_total * 100 if post_total > 0 else 0

                rebal_log.append({
                    "date":          date,
                    "pre_total":     round(n_val_pre + q_val_pre),
                    "contrib":       ANNUAL_CONTRIB,
                    "liquid":        round(liquid),
                    "post_ndx_val":  round(n_val_post),
                    "post_qqq_val":  round(q_val_post),
                    "post_total":    round(post_total),
                    "ndx_pct":       round(ndx_pct, 1),
                    "qqq_pct":       round(100 - ndx_pct, 1),
                    "ndx_holding":   "+".join(n_basket.keys()),
                })
                comb_mid_contribs += ANNUAL_CONTRIB
                n_mid_contrib = 0.0; q_mid_contrib = 0.0
                print(f"  [rebal]   {date.date()}: rebalanced ${post_total:,.0f} → "
                      f"NDX {ndx_pct:.0f}% (${n_val_post:,.0f}) / "
                      f"QQQ {100-ndx_pct:.0f}% (${q_val_post:,.0f})  [{'+'.join(n_basket.keys())}]")

        # ── 1. NDX-A annual rotation at year-start (skip if April rebal today) ─
        if n_pos == "IN" and prev_year is not None and year != prev_year and date not in contrib_dates:
            new_comp = h1.get(year, h1.get(year - 1, []))
            cur = _basket_value(n_basket, stock_prices, date)
            if cur > 0 and new_comp:
                after_sell = cur * (1 - SLIPPAGE) - COMMISSION
                n_basket   = _build_basket(after_sell, new_comp, stock_prices, date)
        prev_year = year

        # ── 2. Sell check ─────────────────────────────────────────────────────
        bearish_div = sig_pr and sig_bf and breadth < DIVERGENCE_BREADTH_CAP
        if bearish_div and trade_open:
            if n_pos == "IN":
                cur_n  = _basket_value(n_basket, stock_prices, date)
                exit_n = cur_n * (1 - SLIPPAGE) - COMMISSION
                deployed_n = n_orig_val + n_mid_contrib
                ret_n = (exit_n - deployed_n) / deployed_n * 100 if deployed_n > 0 else 0.0
                h_str = "+".join(n_entry_tickers)
                if list(n_basket.keys()) != n_entry_tickers:
                    h_str += " → " + "+".join(n_basket.keys())
                ndxa_trades.append({
                    "entry_date":   n_entry_date,
                    "exit_date":    date,
                    "entry_value":  n_orig_val,
                    "mid_contribs": n_mid_contrib,
                    "exit_value":   exit_n,
                    "net_gain":     exit_n - deployed_n,
                    "return_pct":   ret_n,
                    "price_dd":     (n_price_low - n_orig_val) / n_orig_val * 100 if n_orig_val > 0 else 0.0,
                    "port_peak":    n_port_peak,
                    "port_trough":  n_trough,
                    "port_dd":      n_worst_dd,
                    "sell_reason":  "bearish-div",
                    "holdings":     h_str,
                    "status":       "closed",
                })
                n_cash = exit_n; n_basket = {}; n_pos = "OUT"

            if q_pos == "IN":
                exit_q = q_shares * n_price * (1 - SLIPPAGE) - COMMISSION
                deployed_q = q_entry_val + q_mid_contrib
                ret_q = (exit_q - deployed_q) / deployed_q * 100 if deployed_q > 0 else 0.0
                qqqb_trades.append({
                    "entry_date":   q_entry_date,
                    "exit_date":    date,
                    "entry_value":  q_entry_val,
                    "mid_contribs": q_mid_contrib,
                    "exit_value":   exit_q,
                    "net_gain":     exit_q - deployed_q,
                    "return_pct":   ret_q,
                    "price_dd":     (q_price_low - q_entry_price) / q_entry_price * 100 if q_entry_price > 0 else 0.0,
                    "port_peak":    q_port_peak,
                    "port_trough":  q_trough,
                    "port_dd":      q_worst_dd,
                    "sell_reason":  "bearish-div",
                    "holdings":     "QQQ(NDX)",
                    "status":       "closed",
                })
                q_cash = exit_q; q_shares = 0.0; q_pos = "OUT"

            total      = n_cash + q_cash
            deployed_c = comb_entry_val + comb_mid_contribs
            ret_c      = (total - deployed_c) / deployed_c * 100 if deployed_c > 0 else 0.0
            comb_trades.append({
                "entry_date":   comb_entry_date,
                "exit_date":    date,
                "held_days":    (date - comb_entry_date).days,
                "entry_value":  comb_entry_val,
                "mid_contribs": comb_mid_contribs,
                "exit_value":   total,
                "net_gain":     total - deployed_c,
                "return_pct":   ret_c,
                "port_peak":    comb_peak,
                "port_trough":  comb_trough,
                "port_dd":      comb_worst_dd,
                "portfolio":    total,
                "status":       "closed",
            })
            trade_open = False

        # ── 3. Mark to market ─────────────────────────────────────────────────
        n_val = n_mktval(date)
        q_val = q_mktval(n_price)
        total = n_val + q_val

        # ── 4. Update peaks / troughs ─────────────────────────────────────────
        if trade_open:
            comb_peak = max(comb_peak, total)
            dd = (total - comb_peak) / comb_peak * 100
            if dd < comb_worst_dd:
                comb_worst_dd = dd; comb_trough = total

        if n_pos == "IN":
            cur_n_val   = _basket_value(n_basket, stock_prices, date)
            n_price_low = min(n_price_low, cur_n_val)
            n_port_peak = max(n_port_peak, cur_n_val)
            dd = (cur_n_val - n_port_peak) / n_port_peak * 100
            if dd < n_worst_dd:
                n_worst_dd = dd; n_trough = cur_n_val

        if q_pos == "IN":
            q_price_low = min(q_price_low, n_price)
            q_port_peak = max(q_port_peak, q_val)
            dd = (q_val - q_port_peak) / q_port_peak * 100
            if dd < q_worst_dd:
                q_worst_dd = dd; q_trough = q_val

        # ── 5. Buy check ──────────────────────────────────────────────────────
        if not trade_open and not pd.isna(breadth) and breadth < BUY_B200_THRESH:
            total_capital   = n_cash + q_cash + pending_contrib
            pending_contrib = 0.0

            n_alloc = total_capital * NDXA_PCT
            q_alloc = total_capital * QQQB_PCT

            comp = h1.get(year, [])
            if comp:
                n_buy         = n_alloc - COMMISSION
                n_basket      = _build_basket(n_buy, comp, stock_prices, date)
                n_orig_val    = _basket_value(n_basket, stock_prices, date)
                n_entry_val   = n_orig_val
                n_mid_contrib = 0.0
                n_price_low   = n_orig_val
                n_port_peak   = n_orig_val; n_worst_dd = 0.0; n_trough = n_orig_val
                n_entry_tickers = list(n_basket.keys())
                n_entry_date  = date; n_pos = "IN"; n_cash = 0.0
            else:
                n_cash = n_alloc

            q_buy         = q_alloc - COMMISSION
            q_shares      = q_buy / (n_price * (1 + SLIPPAGE))
            q_entry_val   = q_buy
            q_entry_price = n_price
            q_mid_contrib = 0.0
            q_price_low   = n_price
            q_port_peak   = q_buy; q_worst_dd = 0.0; q_trough = q_buy
            q_entry_date  = date; q_pos = "IN"; q_cash = 0.0

            n_val = n_mktval(date)
            q_val = q_mktval(n_price)
            total = n_val + q_val

            comb_entry_date   = date
            comb_entry_val    = total
            comb_mid_contribs = 0.0
            comb_peak         = total; comb_worst_dd = 0.0; comb_trough = total
            trade_open        = True

        combined_values[date] = total

    # ── Open positions at end ─────────────────────────────────────────────────
    last_date = common_idx[-1]
    if trade_open:
        last_n = float(df_ndx["price"].iloc[-1])

        if n_pos == "IN":
            cur_n      = _basket_value(n_basket, stock_prices, last_date)
            exit_n_est = cur_n * (1 - SLIPPAGE)
            deployed_n = n_orig_val + n_mid_contrib
            h_str = "+".join(n_entry_tickers)
            if list(n_basket.keys()) != n_entry_tickers:
                h_str += " → " + "+".join(n_basket.keys())
            ndxa_trades.append({
                "entry_date":   n_entry_date,
                "exit_date":    None,
                "entry_value":  n_orig_val,
                "mid_contribs": n_mid_contrib,
                "exit_value":   exit_n_est,
                "net_gain":     exit_n_est - deployed_n,
                "return_pct":   (exit_n_est - deployed_n) / deployed_n * 100 if deployed_n > 0 else 0.0,
                "price_dd":     (n_price_low - n_orig_val) / n_orig_val * 100 if n_orig_val > 0 else 0.0,
                "port_peak":    n_port_peak,
                "port_trough":  n_trough,
                "port_dd":      n_worst_dd,
                "sell_reason":  "(open)",
                "holdings":     h_str,
                "status":       "(open)",
            })

        if q_pos == "IN":
            exit_q_est = q_shares * last_n * (1 - SLIPPAGE)
            deployed_q = q_entry_val + q_mid_contrib
            qqqb_trades.append({
                "entry_date":   q_entry_date,
                "exit_date":    None,
                "entry_value":  q_entry_val,
                "mid_contribs": q_mid_contrib,
                "exit_value":   exit_q_est,
                "net_gain":     exit_q_est - deployed_q,
                "return_pct":   (exit_q_est - deployed_q) / deployed_q * 100 if deployed_q > 0 else 0.0,
                "price_dd":     (q_price_low - q_entry_price) / q_entry_price * 100 if q_entry_price > 0 else 0.0,
                "port_peak":    q_port_peak,
                "port_trough":  q_trough,
                "port_dd":      q_worst_dd,
                "sell_reason":  "(open)",
                "holdings":     "QQQ(NDX)",
                "status":       "(open)",
            })

        last_total = combined_values.get(last_date, 0.0)
        deployed_c = comb_entry_val + comb_mid_contribs
        ret_c = (last_total - deployed_c) / deployed_c * 100 if deployed_c > 0 else 0.0
        comb_trades.append({
            "entry_date":   comb_entry_date,
            "exit_date":    None,
            "held_days":    (last_date - comb_entry_date).days,
            "entry_value":  comb_entry_val,
            "mid_contribs": comb_mid_contribs,
            "exit_value":   last_total,
            "net_gain":     last_total - deployed_c,
            "return_pct":   ret_c,
            "port_peak":    comb_peak,
            "port_trough":  comb_trough,
            "port_dd":      comb_worst_dd,
            "portfolio":    last_total,
            "status":       "(open)",
        })

    return (pd.Series(combined_values, name="portfolio"),
            comb_trades, ndxa_trades, qqqb_trades, rebal_log)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _days_str(days: int) -> str:
    y, rem = divmod(days, 365); m = rem // 30
    if y and m:  return f"{y}y {m}m"
    if y:        return f"{y}y"
    if m:        return f"{m}m"
    return f"{days}d"


def compute_metrics(label: str, values: pd.Series, trades: list[dict]) -> dict:
    dr    = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    init  = values.iloc[0]
    final = values.iloc[-1]
    tr    = (final / init) - 1
    cagr  = (final / init) ** (1 / years) - 1 if years > 0 else 0.0
    mdd   = float(((values - values.cummax()) / values.cummax()).min())
    std   = float(dr.std())
    sh    = float(dr.mean()) / std * np.sqrt(252) if std > 0 else 0.0
    calmar = abs(cagr / mdd) if mdd != 0 else 0.0

    closed  = [t for t in trades if t.get("exit_date") is not None]
    n       = len(trades)
    wins    = sum(1 for t in trades if t["return_pct"] > 0)
    in_days = sum(
        t.get("held_days", (t["exit_date"] - t["entry_date"]).days)
        for t in closed
    )
    tot = (values.index[-1] - values.index[0]).days

    return {
        "Label":        label,
        "Total Return": f"{tr:+.1%}",
        "CAGR":         f"{cagr*100:.2f}%",
        "Max Drawdown": f"{mdd*100:.2f}%",
        "Sharpe":       f"{sh:.3f}",
        "Calmar":       f"{calmar:.3f}",
        "Final $":      f"${final:,.0f}",
        "# Trades":     str(n),
        "Win Rate":     f"{wins/n:.0%}" if n else "—",
        "Time in Mkt":  f"{in_days/tot:.0%}" if tot else "—",
    }


def print_metrics_table(rows: list[dict]) -> None:
    keys  = list(rows[0].keys())
    cols  = [r["Label"] for r in rows]
    col_w = max(18, max(len(c) for c in cols) + 2)
    hdr   = f"  {'Metric':<16}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep   = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        if k == "Label": continue
        print(f"  {k:<16}" + "".join(f"{r.get(k, '—'):>{col_w}}" for r in rows))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_combined_log(trades: list[dict]) -> None:
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  "
           f"{'EntryVal':>13}  {'MidContrib':>11}  {'ExitVal':>13}  "
           f"{'NetGain':>12}  {'Ret%':>7}  {'PortPeak':>13}  "
           f"{'PortTrough':>13}  {'PortDD%':>8}  Status")
    bar = "─" * len(hdr.strip())
    print(f"\n{bar}\n  Combined Portfolio Trade Log\n{bar}{hdr}")
    print("─" * (len(hdr) - 1))
    for i, r in enumerate(trades, 1):
        exit_str = r["exit_date"].strftime("%Y-%m-%d") if r["exit_date"] else "(open)    "
        print(
            f"{i:>3}  {r['entry_date'].strftime('%Y-%m-%d'):10}  {exit_str:10}  "
            f"{_days_str(r['held_days']):>7}  "
            f"${r['entry_value']:>12,.0f}  ${r['mid_contribs']:>10,.0f}  "
            f"${r['exit_value']:>12,.0f}  ${r['net_gain']:>+11,.0f}  "
            f"{r['return_pct']:>+6.1f}%  "
            f"${r['port_peak']:>12,.0f}  ${r['port_trough']:>12,.0f}  "
            f"{r['port_dd']:>+7.1f}%  {r['status']}"
        )
    print("─" * (len(hdr) - 1))


def print_leg_log(label: str, trades: list[dict], last_date: pd.Timestamp,
                  show_holdings: bool = False) -> None:
    hold_col = "  Holdings" if show_holdings else ""
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  "
           f"{'EntryVal':>13}  {'MidContrib':>11}  {'ExitVal':>13}  "
           f"{'NetGain':>12}  {'Ret%':>7}  {'PortPeak':>13}  "
           f"{'PortTrough':>13}  {'PortDD%':>8}  Status{hold_col}")
    print(f"\n── {label} ──{hdr}")
    print("─" * (len(hdr) - 1))
    for i, t in enumerate(trades, 1):
        is_open  = t["exit_date"] is None
        exit_str = t["exit_date"].strftime("%Y-%m-%d") if not is_open else "(open)    "
        held     = ((t["exit_date"] - t["entry_date"]).days if not is_open
                    else (last_date - t["entry_date"]).days)
        h_str    = f"  {t.get('holdings', '')}" if show_holdings else ""
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  {exit_str:10}  "
            f"{_days_str(held):>7}  "
            f"${t['entry_value']:>12,.0f}  ${t['mid_contribs']:>10,.0f}  "
            f"${t['exit_value']:>12,.0f}  ${t['net_gain']:>+11,.0f}  "
            f"{t['return_pct']:>+6.1f}%  "
            f"${t['port_peak']:>12,.0f}  ${t['port_trough']:>12,.0f}  "
            f"{t['port_dd']:>+7.1f}%  {t['status']}{h_str}"
        )
    print("─" * (len(hdr) - 1))


def print_rebal_log(log: list[dict]) -> None:
    hdr = (f"\n{'#':>3}  {'Date':10}  {'Pre-Total':>13}  {'Contrib':>9}  "
           f"{'Liquid':>13}  {'Post NDX':>13}  {'Post QQQ':>13}  "
           f"{'Total':>13}  {'NDX%':>5}  {'QQQ%':>5}  Holding")
    bar = "─" * len(hdr.strip())
    print(f"\n{bar}\n  April Rebalance Log\n{bar}{hdr}")
    print("─" * (len(hdr) - 1))
    for i, r in enumerate(log, 1):
        print(
            f"{i:>3}  {str(r['date'])[:10]:10}  "
            f"${r['pre_total']:>12,.0f}  ${r['contrib']:>8,.0f}  "
            f"${r['liquid']:>12,.0f}  ${r['post_ndx_val']:>12,.0f}  "
            f"${r['post_qqq_val']:>12,.0f}  ${r['post_total']:>12,.0f}  "
            f"{r['ndx_pct']:>4.1f}%  {r['qqq_pct']:>4.1f}%  {r['ndx_holding']}"
        )
    print("─" * (len(hdr) - 1))


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def _to_df(trades: list[dict]) -> pd.DataFrame:
    rows = []
    for i, t in enumerate(trades, 1):
        row = {"trade_num": i}; row.update(t); rows.append(row)
    return pd.DataFrame(rows)


def export_csvs(
    metrics_rows: list[dict],
    comb_trades:  list[dict],
    ndxa_trades:  list[dict],
    qqqb_trades:  list[dict],
    rebal_log:    list[dict],
) -> None:
    prefix = DATA_DIR / "combined_n25q75"
    pd.DataFrame(metrics_rows).to_csv(f"{prefix}_metrics.csv", index=False)
    for name, trades in [("combined", comb_trades), ("ndxa", ndxa_trades), ("qqqb", qqqb_trades)]:
        _to_df(trades).to_csv(f"{prefix}_{name}.csv", index=False)
        print(f"Saved → {prefix}_{name}.csv")
    pd.DataFrame(rebal_log).to_csv(f"{prefix}_rebal.csv", index=False)
    print(f"Saved → {prefix}_metrics.csv")
    print(f"Saved → {prefix}_rebal.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    breadth = _load_breadth()
    df_ndx  = load_ndx(breadth)

    print(f"Backtest period : {df_ndx.index[0].date()} → {df_ndx.index[-1].date()}")
    print(f"Initial capital : ${INITIAL_CAPITAL:,.0f}")
    print(f"Annual funding  : ${ANNUAL_CONTRIB:,.0f} on April {CONTRIB_DAY} "
          f"each year (starting {CONTRIB_START_YEAR})")
    print(f"Allocation      : NDX-Top1 {NDXA_PCT:.0%} / QQQ {QQQB_PCT:.0%}")
    print(f"Strategy        : Signal (breadth<{BUY_B200_THRESH}% buy / bearish-div sell) "
          f"+ April rebalance to {NDXA_PCT:.0%}/{QQQB_PCT:.0%}")
    print()

    h1           = load_holdings(top_n=1)
    tickers      = {t for comps in h1.values() for t, _ in comps}
    print(f"Loading stock prices: {sorted(tickers)}")
    stock_prices = load_stock_prices(tickers)

    print("\nRunning portfolio simulation…")
    equity, comb_trades, ndxa_trades, qqqb_trades, rebal_log = run_portfolio(
        df_ndx, h1, stock_prices
    )

    last_date = df_ndx.index[-1]

    def _leg_equity(trades: list[dict], init: float) -> pd.Series:
        pts: dict[pd.Timestamp, float] = {equity.index[0]: init}
        running = init
        for t in trades:
            if t["exit_date"] is not None:
                running = t["exit_value"]
                pts[t["exit_date"]] = running
        return pd.Series(pts).sort_index()

    m_comb = compute_metrics("Combined",       equity,     comb_trades)
    m_ndxa = compute_metrics("NDX-Top1 (25%)", _leg_equity(ndxa_trades, INITIAL_CAPITAL * NDXA_PCT), ndxa_trades)
    m_qqqb = compute_metrics("QQQ-B (75%)",    _leg_equity(qqqb_trades, INITIAL_CAPITAL * QQQB_PCT), qqqb_trades)

    print_metrics_table([m_comb, m_ndxa, m_qqqb])
    print_combined_log(comb_trades)
    print_leg_log("NDX Top-1 Leg", ndxa_trades, last_date, show_holdings=True)
    print_leg_log("QQQ-B Leg",     qqqb_trades, last_date, show_holdings=False)
    print_rebal_log(rebal_log)

    print()
    export_csvs([m_comb, m_ndxa, m_qqqb], comb_trades, ndxa_trades, qqqb_trades, rebal_log)


if __name__ == "__main__":
    main()
