"""
NDX Top-1 25% / QQQ 75% — Three-way comparison (2011 → present)

  1. Signal     — buy when breadth < 26%, sell on bearish divergence
  2. Buy & Hold — always invested; contributions deployed 25/75 each April;
                  NDX Top-1 rotates to new year's holding at each year-start
  3. Apr Rebal  — same as Buy & Hold but full portfolio rebalanced to 25/75
                  every April (sell all, re-buy at target weights + contribution)

Initial capital : $153,402
Annual funding  : $26,880 on April 6 each year (starting 2012)
Start date      : 2011-01-01
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

BUY_B200_THRESH         = 26.0
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0
COMMISSION              = 1.0
SLIPPAGE                = 0.0005

INITIAL_CAPITAL    = 153_402.0
ANNUAL_CONTRIB     = 26_880.0
CONTRIB_MONTH      = 4
CONTRIB_DAY        = 6
CONTRIB_START_YEAR = 2012
START_DATE         = pd.Timestamp("2011-01-01")

NDXA_PCT = 0.25
QQQB_PCT = 0.75

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
        year = int(row["Year"])
        name = str(row.get("#1 Holding", "")).strip()
        val  = float(row.get("#1 Value ($B)", 0) or 0)
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
# Strategy 1: Signal-based (buy breadth<26%, sell bearish-div)
# ─────────────────────────────────────────────────────────────────────────────

def run_signal(df: pd.DataFrame, h1: dict, stock_prices: dict) -> tuple[pd.Series, list[dict]]:
    common_idx    = df.index.sort_values()
    contrib_dates = _contribution_dates(common_idx)

    n_pos = "OUT"; n_basket: dict[str, float] = {}; n_cash = INITIAL_CAPITAL * NDXA_PCT
    n_orig_val = 0.0; n_mid_contrib = 0.0; prev_year = None

    q_pos = "OUT"; q_shares = 0.0; q_cash = INITIAL_CAPITAL * QQQB_PCT
    q_entry_price = 0.0; q_mid_contrib = 0.0

    trade_open = False; pending_contrib = 0.0
    trades: list[dict] = []
    comb_entry_date = None; comb_entry_val = 0.0; comb_mid_contribs = 0.0
    values: dict = {}

    for date in common_idx:
        row     = df.loc[date]
        n_price = float(row["price"])
        breadth = float(row["breadth"])
        year    = date.year

        # Contribution
        if date in contrib_dates:
            if not trade_open:
                pending_contrib += ANNUAL_CONTRIB
            else:
                n_add = ANNUAL_CONTRIB * NDXA_PCT
                q_add = ANNUAL_CONTRIB * QQQB_PCT
                if n_pos == "IN":
                    cur_comp = h1.get(year, h1.get(year - 1, []))
                    if cur_comp:
                        extra = _build_basket(n_add - COMMISSION, cur_comp, stock_prices, date)
                        for t, s in extra.items():
                            n_basket[t] = n_basket.get(t, 0.0) + s
                    n_mid_contrib += n_add
                else:
                    n_cash += n_add
                if q_pos == "IN":
                    q_shares += (q_add - COMMISSION) / (n_price * (1 + SLIPPAGE))
                    q_mid_contrib += q_add
                else:
                    q_cash += q_add
                comb_mid_contribs += ANNUAL_CONTRIB

        # NDX-A annual rotation
        if n_pos == "IN" and prev_year is not None and year != prev_year:
            new_comp = h1.get(year, h1.get(year - 1, []))
            cur = _basket_value(n_basket, stock_prices, date)
            if cur > 0 and new_comp:
                n_basket = _build_basket(cur * (1 - SLIPPAGE) - COMMISSION, new_comp, stock_prices, date)
        prev_year = year

        # Sell signal
        bearish = bool(row["price_rose"]) and bool(row["breadth_fell"]) and breadth < DIVERGENCE_BREADTH_CAP
        if bearish and trade_open:
            n_exit = q_exit = 0.0
            if n_pos == "IN":
                n_exit = _basket_value(n_basket, stock_prices, date) * (1 - SLIPPAGE) - COMMISSION
                n_cash = n_exit; n_basket = {}; n_pos = "OUT"
            if q_pos == "IN":
                q_exit = q_shares * n_price * (1 - SLIPPAGE) - COMMISSION
                q_cash = q_exit; q_shares = 0.0; q_pos = "OUT"
            total = n_cash + q_cash
            deployed = comb_entry_val + comb_mid_contribs
            trades.append({
                "entry_date": comb_entry_date, "exit_date": date,
                "held_days":  (date - comb_entry_date).days,
                "entry_value": comb_entry_val, "mid_contribs": comb_mid_contribs,
                "exit_value": total, "net_gain": total - deployed,
                "return_pct": (total - deployed) / deployed * 100 if deployed else 0,
                "status": "closed",
            })
            trade_open = False

        n_val = _basket_value(n_basket, stock_prices, date) if n_pos == "IN" else n_cash
        q_val = q_shares * n_price if q_pos == "IN" else q_cash
        total = n_val + q_val

        # Buy signal
        if not trade_open and not pd.isna(breadth) and breadth < BUY_B200_THRESH:
            cap = n_cash + q_cash + pending_contrib; pending_contrib = 0.0
            comp = h1.get(year, [])
            if comp:
                n_basket = _build_basket(cap * NDXA_PCT - COMMISSION, comp, stock_prices, date)
                n_orig_val = _basket_value(n_basket, stock_prices, date)
                n_mid_contrib = 0.0; n_pos = "IN"; n_cash = 0.0
            else:
                n_cash = cap * NDXA_PCT
            q_shares = (cap * QQQB_PCT - COMMISSION) / (n_price * (1 + SLIPPAGE))
            q_entry_price = n_price; q_mid_contrib = 0.0; q_pos = "IN"; q_cash = 0.0
            n_val = _basket_value(n_basket, stock_prices, date) if n_pos == "IN" else n_cash
            q_val = q_shares * n_price
            total = n_val + q_val
            comb_entry_date = date; comb_entry_val = total
            comb_mid_contribs = 0.0; trade_open = True

        values[date] = total

    # Open trade
    if trade_open:
        last_date = common_idx[-1]
        total = values.get(last_date, 0.0)
        deployed = comb_entry_val + comb_mid_contribs
        trades.append({
            "entry_date": comb_entry_date, "exit_date": None,
            "held_days":  (last_date - comb_entry_date).days,
            "entry_value": comb_entry_val, "mid_contribs": comb_mid_contribs,
            "exit_value": total, "net_gain": total - deployed,
            "return_pct": (total - deployed) / deployed * 100 if deployed else 0,
            "status": "(open)",
        })

    return pd.Series(values, name="Signal"), trades


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Buy & Hold (always invested, April contributions, annual rotation)
# ─────────────────────────────────────────────────────────────────────────────

def run_buy_hold(df: pd.DataFrame, h1: dict, stock_prices: dict) -> pd.Series:
    common_idx    = df.index.sort_values()
    contrib_dates = _contribution_dates(common_idx)

    first_date = common_idx[0]
    first_price = float(df.loc[first_date, "price"])
    year0 = first_date.year
    comp0 = h1.get(year0, h1.get(year0 + 1, []))

    n_basket = _build_basket(INITIAL_CAPITAL * NDXA_PCT - COMMISSION, comp0, stock_prices, first_date)
    q_shares = (INITIAL_CAPITAL * QQQB_PCT - COMMISSION) / (first_price * (1 + SLIPPAGE))

    prev_year = first_date.year
    values: dict = {}

    for date in common_idx:
        n_price = float(df.loc[date, "price"])
        year    = date.year

        # Annual NDX-Top1 rotation at year start
        if year != prev_year:
            new_comp = h1.get(year, h1.get(year - 1, []))
            cur = _basket_value(n_basket, stock_prices, date)
            if cur > 0 and new_comp:
                n_basket = _build_basket(cur * (1 - SLIPPAGE) - COMMISSION, new_comp, stock_prices, date)
        prev_year = year

        # April contribution: deploy 25/75 into existing positions
        if date in contrib_dates:
            n_add = ANNUAL_CONTRIB * NDXA_PCT
            q_add = ANNUAL_CONTRIB * QQQB_PCT
            comp = h1.get(year, h1.get(year - 1, []))
            if comp:
                extra = _build_basket(n_add - COMMISSION, comp, stock_prices, date)
                for t, s in extra.items():
                    n_basket[t] = n_basket.get(t, 0.0) + s
            q_shares += (q_add - COMMISSION) / (n_price * (1 + SLIPPAGE))

        n_val = _basket_value(n_basket, stock_prices, date)
        q_val = q_shares * n_price
        values[date] = n_val + q_val

    return pd.Series(values, name="Buy & Hold")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: April Rebalance (B&H + full 25/75 rebalance every April)
# ─────────────────────────────────────────────────────────────────────────────

def run_apr_rebal(df: pd.DataFrame, h1: dict, stock_prices: dict) -> tuple[pd.Series, list[dict]]:
    common_idx    = df.index.sort_values()
    contrib_dates = _contribution_dates(common_idx)

    first_date  = common_idx[0]
    first_price = float(df.loc[first_date, "price"])
    year0  = first_date.year
    comp0  = h1.get(year0, h1.get(year0 + 1, []))

    n_basket = _build_basket(INITIAL_CAPITAL * NDXA_PCT - COMMISSION, comp0, stock_prices, first_date)
    q_shares = (INITIAL_CAPITAL * QQQB_PCT - COMMISSION) / (first_price * (1 + SLIPPAGE))

    prev_year = first_date.year
    rebal_log: list[dict] = []
    values: dict = {}

    for date in common_idx:
        n_price = float(df.loc[date, "price"])
        year    = date.year

        # Annual NDX-Top1 rotation at year start (skipped if April rebal happens same day)
        if year != prev_year and date not in contrib_dates:
            new_comp = h1.get(year, h1.get(year - 1, []))
            cur = _basket_value(n_basket, stock_prices, date)
            if cur > 0 and new_comp:
                n_basket = _build_basket(cur * (1 - SLIPPAGE) - COMMISSION, new_comp, stock_prices, date)
        prev_year = year

        # April: add contribution + full rebalance to 25/75
        if date in contrib_dates:
            n_val_pre = _basket_value(n_basket, stock_prices, date)
            q_val_pre = q_shares * n_price
            total = n_val_pre + q_val_pre + ANNUAL_CONTRIB

            # Liquidate both legs
            liquid_n = n_val_pre * (1 - SLIPPAGE) - COMMISSION
            liquid_q = q_val_pre * (1 - SLIPPAGE) - COMMISSION
            liquid   = liquid_n + liquid_q + ANNUAL_CONTRIB

            # Re-buy at 25/75 with current year's NDX Top-1
            comp = h1.get(year, h1.get(year - 1, []))
            n_basket = _build_basket(liquid * NDXA_PCT - COMMISSION, comp, stock_prices, date) if comp else {}
            q_shares = (liquid * QQQB_PCT - COMMISSION) / (n_price * (1 + SLIPPAGE))

            n_val_post = _basket_value(n_basket, stock_prices, date)
            q_val_post = q_shares * n_price
            actual_ndx_pct = n_val_post / (n_val_post + q_val_post) * 100 if (n_val_post + q_val_post) > 0 else 0
            rebal_log.append({
                "date":          date,
                "pre_total":     round(total),
                "contrib":       ANNUAL_CONTRIB,
                "post_ndx_val":  round(n_val_post),
                "post_qqq_val":  round(q_val_post),
                "post_ndx_pct":  round(actual_ndx_pct, 1),
                "ndx_holding":   "+".join(n_basket.keys()),
            })

        n_val = _basket_value(n_basket, stock_prices, date)
        q_val = q_shares * n_price
        values[date] = n_val + q_val

    return pd.Series(values, name="Apr Rebal"), rebal_log


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _days_str(days: int) -> str:
    y, rem = divmod(days, 365); m = rem // 30
    if y and m: return f"{y}y {m}m"
    if y:       return f"{y}y"
    if m:       return f"{m}m"
    return f"{days}d"


def compute_metrics(label: str, equity: pd.Series) -> dict:
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
        "Label":        label,
        "Total Return": f"{tr:+.1%}",
        "CAGR":         f"{cagr*100:.2f}%",
        "Max Drawdown": f"{mdd*100:.2f}%",
        "Sharpe":       f"{sh:.3f}",
        "Calmar":       f"{calmar:.3f}",
        "Final $":      f"${final:,.0f}",
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


def print_trade_log(label: str, trades: list[dict]) -> None:
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  "
           f"{'EntryVal':>13}  {'MidContrib':>11}  {'ExitVal':>13}  "
           f"{'NetGain':>12}  {'Ret%':>7}  Status")
    bar = "─" * len(hdr.strip())
    print(f"\n{bar}\n  {label} — Trade Log\n{bar}{hdr}")
    print("─" * (len(hdr) - 1))
    for i, r in enumerate(trades, 1):
        exit_str = r["exit_date"].strftime("%Y-%m-%d") if r["exit_date"] else "(open)    "
        print(
            f"{i:>3}  {r['entry_date'].strftime('%Y-%m-%d'):10}  {exit_str:10}  "
            f"{_days_str(r['held_days']):>7}  "
            f"${r['entry_value']:>12,.0f}  ${r['mid_contribs']:>10,.0f}  "
            f"${r['exit_value']:>12,.0f}  ${r['net_gain']:>+11,.0f}  "
            f"{r['return_pct']:>+6.1f}%  {r['status']}"
        )
    print("─" * (len(hdr) - 1))


def print_rebal_log(log: list[dict]) -> None:
    hdr = (f"\n{'#':>3}  {'Date':10}  {'Pre-Total':>13}  {'Contrib':>9}  "
           f"{'Post NDX Val':>13}  {'Post QQQ Val':>13}  {'NDX%':>6}  Holding")
    bar = "─" * len(hdr.strip())
    print(f"\n{bar}\n  April Rebalance Log\n{bar}{hdr}")
    print("─" * (len(hdr) - 1))
    for i, r in enumerate(log, 1):
        print(
            f"{i:>3}  {str(r['date'])[:10]:10}  "
            f"${r['pre_total']:>12,.0f}  ${r['contrib']:>8,.0f}  "
            f"${r['post_ndx_val']:>12,.0f}  ${r['post_qqq_val']:>12,.0f}  "
            f"{r['post_ndx_pct']:>5.1f}%  {r['ndx_holding']}"
        )
    print("─" * (len(hdr) - 1))


def export_csvs(
    metrics_rows: list[dict],
    signal_trades: list[dict],
    rebal_log: list[dict],
    equity_signal: pd.Series,
    equity_bh: pd.Series,
    equity_ar: pd.Series,
) -> None:
    prefix = DATA_DIR / "combined_n25q75"
    pd.DataFrame(metrics_rows).to_csv(f"{prefix}_metrics.csv", index=False)

    rows = []
    for i, t in enumerate(signal_trades, 1):
        row = {"trade_num": i}; row.update(t); rows.append(row)
    pd.DataFrame(rows).to_csv(f"{prefix}_signal_trades.csv", index=False)

    pd.DataFrame(rebal_log).to_csv(f"{prefix}_rebal_log.csv", index=False)

    eq = pd.DataFrame({"Signal": equity_signal, "Buy_Hold": equity_bh, "Apr_Rebal": equity_ar})
    eq.to_csv(f"{prefix}_equity.csv")

    for name in [f"{prefix}_metrics.csv", f"{prefix}_signal_trades.csv",
                 f"{prefix}_rebal_log.csv", f"{prefix}_equity.csv"]:
        print(f"Saved → {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data…")
    df, h1, stock_prices = load_data()
    print(f"Backtest period : {df.index[0].date()} → {df.index[-1].date()}")
    print(f"Allocation      : NDX Top-1 {NDXA_PCT:.0%} / QQQ {QQQB_PCT:.0%}")
    print(f"Initial capital : ${INITIAL_CAPITAL:,.0f}  |  Annual contrib: ${ANNUAL_CONTRIB:,.0f}\n")

    print("Running Signal strategy…")
    equity_signal, signal_trades = run_signal(df, h1, stock_prices)

    print("Running Buy & Hold strategy…")
    equity_bh = run_buy_hold(df, h1, stock_prices)

    print("Running April Rebalance strategy…")
    equity_ar, rebal_log = run_apr_rebal(df, h1, stock_prices)

    m_signal = compute_metrics("Signal (25/75)",   equity_signal)
    m_bh     = compute_metrics("Buy & Hold",        equity_bh)
    m_ar     = compute_metrics("April Rebalance",   equity_ar)

    print_metrics_table([m_signal, m_bh, m_ar])
    print_trade_log("Signal Strategy", signal_trades)
    print_rebal_log(rebal_log)

    print()
    export_csvs([m_signal, m_bh, m_ar], signal_trades, rebal_log,
                equity_signal, equity_bh, equity_ar)


if __name__ == "__main__":
    main()
