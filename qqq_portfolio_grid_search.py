"""
qqq_portfolio_grid_search.py

Grid search over QQQ / NDX Top-1 Stock allocation pairs using the
signal-based strategy from qqq_portfolio_backtest.py.

For each allocation (QQQ 0%→100% / Stock 100%→0%, step 5%), runs the full
signal-based backtest and records: Total Return, CAGR, Max Drawdown,
Sharpe Ratio, Final Value, # Trades, Win Rate, Time in Market.

Buy/sell signals (same as qqq_portfolio_backtest.py defaults):
  BUY : breadth200 < 26% AND (VIX > 30 OR price > MA200)
  SELL: price rose >= 3% over 60 days AND breadth200 fell >= 20 pts
        AND breadth200 < 60%

TQQQ / SPY / SOXX are excluded — 2-asset grid only.

Output: qqq_portfolio_grid_results.csv
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR          = Path(__file__).parent
NDX_FILE          = DATA_DIR / "NASDAQ100.csv"
BREADTH_FILE      = DATA_DIR / "S5TH.csv"
# Continuous daily breadth (2002+) built by build_breadth_daily.py.
# S5TH.csv alone is only daily from 2007 — before that it is bimonthly, which
# corrupts row-based lookback windows (a "60-day" window spans ~10 years).
BREADTH_DAILY_FILE = DATA_DIR / "breadth_daily.csv"
BREADTH_DAILY_MIN  = "2007-01-01"  # fallback cutoff when daily file is absent
VIX_FILE          = DATA_DIR / "VIX.csv"
TOP_HOLDINGS_FILE = DATA_DIR / "NASDAQ100" / "nasdaq100_top10_holdings.csv"
STOCK_PRICE_DIR   = DATA_DIR / "NASDAQ100" / "stock_prices"
OUTPUT_FILE       = DATA_DIR / "qqq_portfolio_grid_results.csv"

# ── Signal params (match qqq_portfolio_backtest.py defaults) ──────────────────
BUY_B200_THRESH         = 26.0
VIX_BUY_THRESH          = 30.0
MA200_WINDOW            = 200
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0
COOLDOWN_DAYS           = 0
COMMISSION              = 1.0
SLIPPAGE                = 0.0005
INITIAL_CAPITAL         = 10_000.0

_NAME_TO_TICKER: list[tuple[str, str]] = [
    ("cisco",              "CSCO"),
    ("microsoft",          "MSFT"),
    ("intel",              "INTC"),
    ("oracle",             "ORCL"),
    ("qualcomm",           "QCOM"),
    ("apple",              "AAPL"),
    ("alphabet",           "GOOGL"),
    ("google",             "GOOGL"),
    ("amazon",             "AMZN"),
    ("tesla",              "TSLA"),
    ("nvidia",             "NVDA"),
    ("meta",               "META"),
    ("facebook",           "META"),
    ("paypal",             "PYPL"),
    ("netflix",            "NFLX"),
    ("broadcom",           "AVGO"),
    ("costco",             "COST"),
    ("pepsico",            "PEP"),
    ("t-mobile",           "TMUS"),
    ("ebay",               "EBAY"),
    ("dell",               "DELL"),
    ("comcast",            "CMCSA"),
    ("amgen",              "AMGN"),
    ("gilead",             "GILD"),
    ("charter",            "CHTR"),
    ("texas instruments",  "TXN"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _load_breadth() -> pd.DataFrame:
    """Prefer the continuous daily series (breadth_daily.csv, 2002+); S5TH.csv
    alone is bimonthly before 2007, which corrupts row-based windows."""
    if BREADTH_DAILY_FILE.exists():
        b200 = pd.read_csv(BREADTH_DAILY_FILE)
        b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
        b200.set_index("Date", inplace=True)
        return b200.rename(columns={"breadth": "Price"})
    b200 = pd.read_csv(BREADTH_FILE)
    b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
    b200.set_index("Date", inplace=True)
    b200["Price"] = _parse_price(b200["Price"])
    # S5TH is bimonthly before 2007 — drop the sparse era
    return b200[b200.index >= BREADTH_DAILY_MIN]


def _name_to_ticker(name: str) -> str | None:
    nl = name.lower()
    for key, ticker in _NAME_TO_TICKER:
        if key in nl:
            return ticker
    return None


def _safe(series: pd.Series | None, date: pd.Timestamp) -> float:
    if series is None:
        return float("nan")
    try:
        v = series.loc[date]
        return float(v) if not pd.isna(v) else float("nan")
    except KeyError:
        return float("nan")


# ── data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, dict[int, str], dict[str, pd.Series]]:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price"})
    ndx["price"] = _parse_price(ndx["price"])

    b200 = _load_breadth()

    vix = pd.read_csv(VIX_FILE)
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix.set_index("Date", inplace=True)
    vix["vix"] = _parse_price(vix["Price"])

    merged = ndx[["price"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged = merged.join(vix[["vix"]], how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]
    merged["vix"]   = merged["vix"].ffill()
    merged["ma200"] = merged["price"].rolling(MA200_WINDOW).mean()

    merged["vix_vote"]   = merged["vix"].apply(
        lambda v: True if pd.isna(v) else v > VIX_BUY_THRESH
    )
    merged["ma200_vote"] = merged.apply(
        lambda r: True if pd.isna(r["ma200"]) else r["price"] > r["ma200"], axis=1
    )
    merged["vote_gate"] = merged["vix_vote"] | merged["ma200_vote"]

    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)
    # Trend re-entry: fresh close back above MA200 (gate reduces to "above prior exit").
    merged["ma200_recross"] = (
        (merged["price"] > merged["ma200"]) & (merged["price"].shift(1) <= merged["ma200"].shift(1))
    ).fillna(False)

    holdings_df = pd.read_csv(TOP_HOLDINGS_FILE)
    top_holdings: dict[int, str] = {}
    for _, row in holdings_df.iterrows():
        if int(row["Rank"]) == 1:
            ticker = _name_to_ticker(str(row["Holding"]))
            if ticker:
                top_holdings[int(row["Year"])] = ticker

    unique_tickers = set(top_holdings.values())
    aligned_stocks: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in sorted(unique_tickers):
        path = STOCK_PRICE_DIR / f"{ticker}.csv"
        if not path.exists():
            missing.append(ticker)
            continue
        df = pd.read_csv(path)
        df["Date"] = (
            pd.to_datetime(df["Date"], format="mixed", utc=True).dt.tz_localize(None)
        )
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        aligned_stocks[ticker] = df["Close"].astype(float).reindex(merged.index).ffill()

    if missing:
        print(f"  [stock CSVs not found → QQQ fallback]: {', '.join(missing)}")

    return merged, top_holdings, aligned_stocks


# ── strategy ──────────────────────────────────────────────────────────────────

def run_strategy(
    df: pd.DataFrame,
    top_holdings: dict[int, str],
    aligned_stocks: dict[str, pd.Series],
    qqq_weight: float,
    stock_weight: float,
) -> tuple[pd.Series, list[dict]]:
    """
    Run the signal-based strategy for one (qqq_weight, stock_weight) pair.
    Both weights should sum to 1.0. Returns (daily_portfolio_values, trades).

    Independent cash buckets: each compounds separately (same as the
    multi-asset backtest). If the stock CSV is missing on entry day, the
    stock bucket rolls into QQQ for that trade.
    """
    position       = "OUT"
    cooldown_until: pd.Timestamp | None = None
    last_exit_ndx: float | None = None
    trades: list[dict] = []
    values: dict[pd.Timestamp, float] = {}

    qqq_bucket   = INITIAL_CAPITAL * qqq_weight
    stock_bucket = INITIAL_CAPITAL * stock_weight

    qqq_shares     = 0.0
    stock_shares   = 0.0
    qqq_entry_px   = 0.0
    stock_entry_px = 0.0
    stock_qqq_frac = 0.0   # fraction of the QQQ position owned by stock bucket
    stock_active   = False
    holding_ticker: str | None = None
    entry_date:     pd.Timestamp | None = None
    entry_val       = 0.0

    for date, row in df.iterrows():
        ndx_price    = float(row["price"])
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])

        if position == "OUT":
            cooldown_ok = cooldown_until is None or date > cooldown_until
            washout_buy = (
                not pd.isna(breadth)
                and breadth < BUY_B200_THRESH
                and bool(row["vote_gate"])
            )
            # Trend re-entry: fresh MA200 recross once NDX is back above the prior exit.
            trend_buy = bool(row["ma200_recross"]) and (
                last_exit_ndx is not None and ndx_price > last_exit_ndx)
            do_buy = cooldown_ok and (washout_buy or trend_buy)
            if do_buy:
                year         = date.year
                stock_ticker = top_holdings.get(year) or top_holdings.get(year - 1)
                stock_px     = _safe(
                    aligned_stocks.get(stock_ticker) if stock_ticker else None, date
                )

                # Deduct $1 commission proportionally across buckets
                total_pre  = qqq_bucket + stock_bucket
                comm_scale = (total_pre - COMMISSION) / total_pre if total_pre > 0 else 1.0
                qqq_bucket   *= comm_scale
                stock_bucket *= comm_scale

                stock_active = not pd.isna(stock_px)

                # Fold unavailable stock bucket into QQQ for this trade
                eff_qqq   = qqq_bucket + (0.0 if stock_active else stock_bucket)
                eff_stock = stock_bucket if stock_active else 0.0

                stock_qqq_frac = (
                    (stock_bucket / eff_qqq) if (not stock_active and eff_qqq > 0) else 0.0
                )

                qqq_entry_px   = ndx_price * (1 + SLIPPAGE)
                stock_entry_px = stock_px  * (1 + SLIPPAGE) if stock_active else 0.0

                qqq_shares   = eff_qqq   / qqq_entry_px   if qqq_entry_px   > 0 else 0.0
                stock_shares = eff_stock / stock_entry_px if stock_entry_px > 0 else 0.0

                holding_ticker = stock_ticker
                entry_date     = date
                entry_val      = qqq_bucket + stock_bucket
                position       = "IN"

        else:  # IN
            bearish_div = (
                price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP
            )
            if bearish_div:
                stock_px_exit = _safe(
                    aligned_stocks.get(holding_ticker) if holding_ticker else None, date
                )
                spx = stock_px_exit if not pd.isna(stock_px_exit) else 0.0

                gross_qqq   = qqq_shares   * ndx_price * (1 - SLIPPAGE)
                gross_stock = stock_shares * spx        * (1 - SLIPPAGE)
                gross_total = gross_qqq + gross_stock
                comm_frac   = COMMISSION / gross_total if gross_total > 0 else 0.0

                # Distribute QQQ proceeds back to the buckets that funded them
                qqq_bucket   = gross_qqq * (1.0 - stock_qqq_frac) * (1.0 - comm_frac)
                stock_bucket = (
                    gross_qqq * stock_qqq_frac + gross_stock
                ) * (1.0 - comm_frac)
                total_proc = qqq_bucket + stock_bucket

                gross_ret = (total_proc - entry_val) / entry_val if entry_val > 0 else 0.0
                trades.append({
                    "entry_date":   entry_date,
                    "exit_date":    date,
                    "return_pct":   gross_ret * 100,
                    "entry_val":    entry_val,
                    "exit_val":     total_proc,
                    "top1_ticker":  holding_ticker,
                    "stock_active": stock_active,
                })
                cooldown_until = date + pd.Timedelta(days=COOLDOWN_DAYS)
                last_exit_ndx  = ndx_price
                position       = "OUT"
                qqq_shares = stock_shares = 0.0

        # Mark-to-market
        if position == "IN":
            sn = 0.0
            if stock_active:
                raw = _safe(
                    aligned_stocks.get(holding_ticker) if holding_ticker else None, date
                )
                sn = raw if not pd.isna(raw) else 0.0
            values[date] = qqq_shares * ndx_price + stock_shares * sn
        else:
            values[date] = qqq_bucket + stock_bucket

    return pd.Series(values, name="portfolio"), trades


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(values: pd.Series, trades: list[dict]) -> dict:
    dr    = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    tr    = (values.iloc[-1] / values.iloc[0]) - 1
    cagr  = (values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0
    mdd   = ((values - values.cummax()) / values.cummax()).min()
    std   = dr.std()
    sh    = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    n       = len(trades)
    wins    = sum(1 for t in trades if t["return_pct"] > 0)
    in_days = sum((t["exit_date"] - t["entry_date"]).days for t in trades)
    tot     = (values.index[-1] - values.index[0]).days

    return {
        "total_return": tr,
        "cagr":         cagr,
        "max_drawdown": mdd,
        "sharpe_ratio": sh,
        "final_value":  values.iloc[-1],
        "n_trades":     n,
        "win_rate":     wins / n if n else 0.0,
        "time_in_mkt":  in_days / tot if tot else 0.0,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search: QQQ vs NDX Top-1 Stock allocation (signal-based strategy)"
    )
    parser.add_argument(
        "--step", type=int, default=5, metavar="PCT",
        help="Allocation step size in %% (default: %(default)s). "
             "E.g. 5 → 21 combos, 10 → 11 combos.",
    )
    parser.add_argument(
        "--sort", choices=["sharpe", "cagr", "return", "drawdown"],
        default="sharpe",
        help="Primary sort column for the results table (default: %(default)s).",
    )
    parser.add_argument(
        "--start-date", type=str, default=None, metavar="DATE",
        help="First date to include, ISO format YYYY-MM-DD (default: full history).",
    )
    parser.add_argument(
        "--end-date", type=str, default=None, metavar="DATE",
        help="Last date to include, ISO format YYYY-MM-DD (default: full history).",
    )
    args = parser.parse_args()

    print("Loading data...")
    df, top_holdings, aligned_stocks = load_data()

    # Slice to requested date range
    if args.start_date:
        start_ts = pd.Timestamp(args.start_date)
        if start_ts < df.index[0]:
            print(f"[warning] --start-date {start_ts.date()} is before data start "
                  f"{df.index[0].date()}; using data start instead]")
        else:
            df = df[df.index >= start_ts]
            aligned_stocks = {t: s[s.index.isin(df.index)] for t, s in aligned_stocks.items()}
    if args.end_date:
        end_ts = pd.Timestamp(args.end_date)
        df = df[df.index <= end_ts]
        aligned_stocks = {t: s[s.index.isin(df.index)] for t, s in aligned_stocks.items()}

    print(
        f"Date range : {df.index[0].date()} → {df.index[-1].date()}"
        f"  ({len(df):,} trading days)"
    )
    print(f"Top-1 holdings: { {yr: tk for yr, tk in sorted(top_holdings.items())} }")

    STEP        = args.step
    allocations = list(range(0, 101, STEP))
    n_combos    = len(allocations)

    sort_key = {
        "sharpe":   "sharpe_ratio",
        "cagr":     "cagr",
        "return":   "total_return",
        "drawdown": "max_drawdown",
    }[args.sort]
    ascending = args.sort == "drawdown"

    print(
        f"\nRunning grid search: {n_combos} allocation combos "
        f"(step {STEP}%, sorted by {args.sort})"
    )
    print(
        f"Buy : breadth200 < {BUY_B200_THRESH}%  AND  "
        f"(VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW})"
    )
    print(
        f"Sell: price rose >= {DIVERGENCE_PRICE_RISE}% over {DIVERGENCE_WINDOW}d  AND  "
        f"breadth200 fell >= {DIVERGENCE_BREADTH_FALL}pts  AND  "
        f"breadth200 < {DIVERGENCE_BREADTH_CAP}%"
    )
    print()

    rows: list[dict] = []
    for qqq_pct in allocations:
        stock_pct = 100 - qqq_pct
        qqq_w     = qqq_pct   / 100.0
        stock_w   = stock_pct / 100.0

        values, trades = run_strategy(df, top_holdings, aligned_stocks, qqq_w, stock_w)
        m = compute_metrics(values, trades)

        rows.append({
            "qqq_pct":      qqq_pct,
            "stock_pct":    stock_pct,
            "cagr":         round(m["cagr"],          6),
            "total_return": round(m["total_return"],   6),
            "max_drawdown": round(m["max_drawdown"],   6),
            "sharpe_ratio": round(m["sharpe_ratio"],   6),
            "final_value":  round(m["final_value"],    2),
            "n_trades":     m["n_trades"],
            "win_rate":     round(m["win_rate"],        6),
            "time_in_mkt":  round(m["time_in_mkt"],    6),
        })
        print(
            f"  QQQ {qqq_pct:3d}% / Stock {stock_pct:3d}%  |"
            f"  CAGR {m['cagr']:>+7.2%}  |"
            f"  Sharpe {m['sharpe_ratio']:>6.3f}  |"
            f"  MDD {m['max_drawdown']:>+7.2%}  |"
            f"  Final ${m['final_value']:>9,.0f}  |"
            f"  {m['n_trades']} trades"
        )

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved → {OUTPUT_FILE}")

    # ── Summary table ─────────────────────────────────────────────────────────
    sorted_result = result.sort_values(sort_key, ascending=ascending)

    print(
        f"\n{'='*97}\n"
        f"  Grid Results — sorted by {args.sort}\n"
        f"{'='*97}"
    )
    print(
        f"  {'QQQ':>5}  {'Stock':>5}  {'CAGR':>8}  {'Tot Ret':>8}  {'MDD':>8}  "
        f"{'Sharpe':>7}  {'Final $':>11}  {'Trades':>6}  {'WinRate':>8}  {'In Mkt':>7}"
    )
    print("  " + "-" * 95)
    for _, r in sorted_result.iterrows():
        print(
            f"  {int(r.qqq_pct):>5}  {int(r.stock_pct):>5}  "
            f"{r.cagr:>+7.2%}  {r.total_return:>+7.2%}  "
            f"{r.max_drawdown:>+7.2%}  "
            f"{r.sharpe_ratio:>7.3f}  "
            f"${r.final_value:>10,.0f}  "
            f"{int(r.n_trades):>6}  "
            f"{r.win_rate:>7.1%}  "
            f"{r.time_in_mkt:>7.1%}"
        )
    print("  " + "=" * 95)

    best_sharpe = result.loc[result["sharpe_ratio"].idxmax()]
    best_cagr   = result.loc[result["cagr"].idxmax()]
    best_mdd    = result.loc[result["max_drawdown"].idxmax()]  # least negative

    print(
        f"\nBest Sharpe  :  QQQ {int(best_sharpe.qqq_pct):3d}% / Stock {int(best_sharpe.stock_pct):3d}%"
        f"  →  {float(best_sharpe.sharpe_ratio):.3f}"
        f"  (CAGR {float(best_sharpe.cagr):+.2%}, MDD {float(best_sharpe.max_drawdown):+.2%})"
    )
    print(
        f"Best CAGR    :  QQQ {int(best_cagr.qqq_pct):3d}% / Stock {int(best_cagr.stock_pct):3d}%"
        f"  →  {float(best_cagr.cagr):+.2%}"
        f"  (Sharpe {float(best_cagr.sharpe_ratio):.3f}, MDD {float(best_cagr.max_drawdown):+.2%})"
    )
    print(
        f"Shallowest MDD:  QQQ {int(best_mdd.qqq_pct):3d}% / Stock {int(best_mdd.stock_pct):3d}%"
        f"  →  {float(best_mdd.max_drawdown):+.2%}"
        f"  (CAGR {float(best_mdd.cagr):+.2%}, Sharpe {float(best_mdd.sharpe_ratio):.3f})"
    )


if __name__ == "__main__":
    main()
