"""
S&P 500 Top-17 Breadth Strategy — equal weight vs market-cap weight.

Applies the canonical qqq_backtest.py buy/sell state machine, computed on the
SPX (as spy_backtest.py does), to a basket of the top 17 S&P 500 stocks by
market cap. The top-17 list is reconstituted yearly from
SP500/sp500_top17_holdings.csv (year Y's list = ranking at the END of year
Y-1, so the composition is point-in-time knowable). Two weight modes are run
through the same engine and compared:

  • equal   — 1/N across the names with price data at the build date
  • mktcap  — proportional to the year's (approximate) market caps

BUY  (while OUT): either entry path —
                  • Washout: breadth200 < 26% AND ≥1 of 2 vote:
                      · VIX > 30  (fear spike / panic bottom)
                      · SPX > MA200  (uptrend pullback)
                  • Trend re-entry: SPX closes back above MA200 (fresh cross),
                    allowed when the previous exit was a climax-top OR SPX is
                    back above the price we last sold at.
                  A 15-day cooldown must have elapsed since the last exit.
SELL (while IN):  any of —
                  • Bearish divergence: SPX rose ≥ 3% over 60 days while
                    breadth200 fell ≥ 20 pts AND breadth200 < 60%
                  • Climax top: within 10 days, SPX extended ≥ 5% above its
                    10-day MA AND MACD(12,26,9) flipped bearish (post-entry)
                  • Trailing stop: SPX 25% below the high since entry

All signals come from the SPX close; fills happen at the NEXT day's open of
the traded stocks (EXECUTION_LAG=1 / FILL_PRICE="open"). While IN, the basket
is rebalanced to the new year's composition at each year boundary.

⚠ Data caveats: the holdings file is hand-curated from historical year-end
market-cap rankings — ranks ~10–17 and the cap figures are approximate, and
yearly reconstitution reduces but does not eliminate survivorship/curation
bias. Tickers with no price file in SP500/stock_prices/ are skipped with the
weights renormalized (run SP500/fetch_stock_history.py to download them).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR      = Path(__file__).parent
SPX_FILE      = DATA_DIR / "SPX.csv"
BREADTH_FILE  = DATA_DIR / "S5TH.csv"
# Continuous daily breadth (2002+) built by build_breadth_daily.py.
# S5TH.csv alone is only daily from 2007 — before that it is bimonthly, which
# corrupts row-based lookback windows (a "60-day" window spans ~10 years).
BREADTH_DAILY_FILE = DATA_DIR / "breadth_daily.csv"
BREADTH_DAILY_MIN  = "2007-01-01"  # fallback cutoff when daily file is absent
VIX_FILE      = DATA_DIR / "VIX.csv"
HOLDINGS_FILE = DATA_DIR / "SP500" / "sp500_top17_holdings.csv"
PRICES_DIR    = DATA_DIR / "SP500" / "stock_prices"

TOP_N = 17

# ── Buy thresholds ────────────────────────────────────────────────────────────
BUY_B200_THRESH = 26.0   # breadth200 must be below this
VIX_BUY_THRESH  = 30.0   # VIX vote: fear spike (VIX > 30)
MA200_WINDOW    = 200     # MA200 vote: price above 200-day moving average

# ── Sell — bearish divergence ─────────────────────────────────────────────────
DIVERGENCE_WINDOW       = 60    # trading days lookback
DIVERGENCE_PRICE_RISE   = 3.0   # % price rise over window
DIVERGENCE_BREADTH_FALL = 20.0  # pts breadth200 drop over window
DIVERGENCE_BREADTH_CAP  = 60.0  # breadth200 must be below this

# ── Sell — climax top (extension + momentum break within a window) ───────────
EXT10_PCT          = 5.0   # % above 10-day MA that counts as "extended"
CLIMAX_VOTE_WINDOW = 10    # days within which both climax signals must fire

# ── Sell — trailing stop ──────────────────────────────────────────────────────
TRAILING_STOP_PCT = 25.0   # % below the high since entry

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day SPX closes, so the earliest tradeable fill is
# the NEXT session. Default: a signal on day t fills at day t+1's OPEN of the
# traded stocks. The annual rebalance is a scheduled event (composition known
# in advance), so it fills same-day without a lag.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

# ── Shared ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
COOLDOWN_DAYS   = 15     # calendar days to wait after a sell before the next buy


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    """SPX close/open + breadth + VIX with all signal pre-computes
    (same block as qqq_backtest.load_data, on the SPX)."""
    spx = pd.read_csv(SPX_FILE)
    spx.columns = [c.strip().strip('"').lstrip("﻿") for c in spx.columns]
    spx["Date"] = pd.to_datetime(spx["Date"], format="%m/%d/%Y")
    spx.set_index("Date", inplace=True)
    spx = spx.rename(columns={"Price": "price", "Open": "open"})
    spx["price"] = _parse_price(spx["price"])
    spx["open"]  = _parse_price(spx["open"])

    if BREADTH_DAILY_FILE.exists():
        b200 = pd.read_csv(BREADTH_DAILY_FILE)
        b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
        b200.set_index("Date", inplace=True)
        b200 = b200.rename(columns={"breadth": "Price"})
    else:
        b200 = pd.read_csv(BREADTH_FILE)
        b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
        b200.set_index("Date", inplace=True)
        b200["Price"] = _parse_price(b200["Price"])
        # S5TH is bimonthly before 2007 — drop the sparse era
        b200 = b200[b200.index >= BREADTH_DAILY_MIN]

    vix = pd.read_csv(VIX_FILE)
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix.set_index("Date", inplace=True)
    vix["vix"] = _parse_price(vix["Price"])

    merged = spx[["price", "open"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged = merged.join(vix[["vix"]], how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]

    merged["vix"]   = merged["vix"].ffill()
    merged["ma200"] = merged["price"].rolling(MA200_WINDOW).mean()

    # Vote gate: at least 1 of [VIX > 30, price > MA200] must be True
    # NaN → True (don't restrict when data is missing)
    merged["vix_vote"]   = merged["vix"].apply(
        lambda v: True if pd.isna(v) else v > VIX_BUY_THRESH)
    merged["ma200_vote"] = merged.apply(
        lambda r: True if pd.isna(r["ma200"]) else r["price"] > r["ma200"], axis=1)
    merged["vote_gate"]  = merged["vix_vote"] | merged["ma200_vote"]

    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)

    close = merged["price"]
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    merged["macd_cross"] = ((hist < 0) & (hist.shift(1) >= 0)).fillna(False)
    merged["ext10"] = (close / close.rolling(10).mean() - 1 >= EXT10_PCT / 100).fillna(False)

    merged["ma200_recross"] = (
        (close > merged["ma200"]) & (close.shift(1) <= merged["ma200"].shift(1))
    ).fillna(False)

    return merged


def load_holdings() -> dict[int, list[tuple[str, float]]]:
    """{year: [(ticker, market cap $B), ...]} in rank order, top 17 per year."""
    df = pd.read_csv(HOLDINGS_FILE)
    holdings: dict[int, list[tuple[str, float]]] = {}
    for year, grp in df.groupby("Year"):
        grp = grp.sort_values("Rank").head(TOP_N)
        holdings[int(year)] = [
            (str(r["Ticker"]), float(r["MarketCap ($B)"])) for _, r in grp.iterrows()
        ]
    return holdings


def year_composition(holdings: dict[int, list[tuple[str, float]]],
                     year: int, weight_mode: str) -> list[tuple[str, float]]:
    """Weights for a year: equal (1/N) or proportional to market cap."""
    rows = holdings.get(year) or holdings.get(max(y for y in holdings if y <= year), [])
    if not rows:
        return []
    if weight_mode == "equal":
        return [(t, 1.0 / len(rows)) for t, _ in rows]
    total = sum(c for _, c in rows)
    return [(t, c / total) for t, c in rows]


def load_stock_prices(tickers: set[str], col: str = "Close") -> dict[str, pd.Series]:
    prices: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in sorted(tickers):
        path = PRICES_DIR / f"{ticker}.csv"
        if not path.exists():
            missing.append(ticker)
            continue
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, format="ISO8601", utc=True).tz_localize(None)
        use = col if col in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
        prices[ticker] = df[use].dropna()
    if missing and col == "Close":
        print(f"\n  [WARNING] No price file for {len(missing)} ticker(s): {', '.join(missing)}")
        print("            They are skipped (weights renormalized over the rest).")
        print("            Run `python SP500/fetch_stock_history.py` to download them.\n")
    return prices


# ── Basket helpers (same model as ndx_top1_backtest.py) ──────────────────────

def get_price(prices: dict[str, pd.Series], ticker: str, date: pd.Timestamp) -> float | None:
    s = prices.get(ticker)
    if s is None or s.empty:
        return None
    idx = s.index[s.index <= date]
    if idx.empty:
        return None
    return float(s.loc[idx[-1]])


def build_basket(cash: float, composition: list[tuple[str, float]],
                 prices: dict[str, pd.Series], date: pd.Timestamp) -> dict[str, float]:
    """Allocate `cash` across the weighted composition; return {ticker: shares}."""
    basket: dict[str, float] = {}
    available = [(t, w) for t, w in composition if get_price(prices, t, date) is not None]
    if not available:
        return basket
    total_w = sum(w for _, w in available)
    for ticker, weight in available:
        alloc = cash * (weight / total_w)
        price = get_price(prices, ticker, date) * (1 + SLIPPAGE)
        basket[ticker] = alloc / price
    return basket


def basket_value(basket: dict[str, float], prices: dict[str, pd.Series],
                 date: pd.Timestamp) -> float:
    total = 0.0
    for ticker, shares in basket.items():
        p = get_price(prices, ticker, date)
        if p is not None:
            total += shares * p
    return total


def _days_str(days: int) -> str:
    years, rem = divmod(days, 365)
    months = rem // 30
    if years and months:
        return f"{years}y {months}m"
    if years:
        return f"{years}y"
    if months:
        return f"{months}m"
    return f"{days}d"


# ── Strategy ──────────────────────────────────────────────────────────────────

def run_strategy(df: pd.DataFrame, holdings: dict[int, list[tuple[str, float]]],
                 prices: dict[str, pd.Series], opens: dict[str, pd.Series],
                 weight_mode: str, cooldown_days: int = COOLDOWN_DAYS,
                 execution_lag: int = EXECUTION_LAG,
                 fill_on: str = FILL_PRICE) -> tuple[pd.Series, list[dict], dict | None]:
    """Canonical qqq_backtest signal machine on the SPX; fills trade the top-17
    basket. Signals (incl. trailing stop / climax ages) run on the SPX close;
    mark-to-market uses the basket's closes; fills use the fill-day opens."""
    if fill_on == "open" and execution_lag < 1:
        raise ValueError("fill_on='open' requires execution_lag >= 1 (open precedes close)")
    fill_src = opens if (fill_on == "open" and opens) else prices

    position           = "OUT"
    basket: dict[str, float] = {}
    entry_date         = None
    original_entry_val = 0.0   # basket value at trade open — never overwritten
    trade_min_val      = 0.0   # lowest basket value during the trade (for MAE)
    entry_names        = 0
    trade_low = trade_high = 0.0   # SPX levels: drive the trailing stop
    macd_age = ext_age = 10**9
    buy_trigger        = None
    portfolio          = INITIAL_CAPITAL
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_spx: float | None = None
    trades: list[dict] = []
    values: dict = {}

    pending: dict | None = None
    rows = list(df.iterrows())
    n = len(rows)
    prev_year: int | None = None

    def execute_due(i, date, year, spx_fill):
        """Fill a pending basket order due this bar at fill_src prices.
        `spx_fill` is the SPX fill-bar price (open by default) — it seeds the
        trailing-stop levels on entry and the re-entry reference on exit,
        mirroring qqq_backtest/spy_backtest exactly."""
        nonlocal position, basket, original_entry_val, trade_min_val, entry_names
        nonlocal entry_date, trade_low, trade_high, macd_age, ext_age, buy_trigger
        nonlocal portfolio, cooldown_until, last_sell_reason, last_exit_spx, pending
        if pending is None or pending["fill_at"] != i:
            return False
        if pending["action"] == "BUY" and position == "OUT":
            comp = year_composition(holdings, year, weight_mode)
            if comp:
                cash_to_invest     = portfolio - COMMISSION
                basket             = build_basket(cash_to_invest, comp, fill_src, date)
                original_entry_val = basket_value(basket, fill_src, date)
                trade_min_val      = original_entry_val
                entry_names        = len(basket)
                entry_date         = date
                trade_low = trade_high = spx_fill
                macd_age = ext_age = 10**9   # climax ages reset (post-entry only)
                buy_trigger        = pending["trigger"]
                position           = "IN"
                portfolio          = 0.0
                pending = None
                return True
            pending = None
            return False
        if pending["action"] == "SELL" and position == "IN":
            exit_val     = basket_value(basket, fill_src, date)
            eff_exit_val = exit_val * (1 - SLIPPAGE) - COMMISSION
            gross_ret    = (eff_exit_val - original_entry_val) / original_entry_val if original_entry_val else 0
            mae          = (trade_min_val - original_entry_val) / original_entry_val * 100
            portfolio    = eff_exit_val
            cooldown_until   = date + pd.Timedelta(days=cooldown_days)
            last_sell_reason = pending["reason"]
            last_exit_spx    = spx_fill
            trades.append({
                "entry_date":       entry_date,
                "exit_date":        date,
                "entry_basket_val": original_entry_val,
                "exit_basket_val":  eff_exit_val,
                "return_pct":       gross_ret * 100,
                "max_drawdown_pct": mae,
                "accumulated":      eff_exit_val,
                "buy_trigger":      buy_trigger,
                "sell_reason":      pending["reason"],
                "n_holdings":       entry_names,
            })
            basket   = {}
            position = "OUT"
            pending = None
            return True
        pending = None
        return False

    for i in range(n):
        date, row = rows[i]
        spx_price    = row["price"]
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])
        year         = date.year
        if fill_on == "open" and not pd.isna(row["open"]):
            spx_fill = row["open"]
        else:
            spx_fill = spx_price

        # ── Annual rebalance during an open trade (scheduled, no signal lag) ──
        if position == "IN" and prev_year is not None and year != prev_year:
            new_comp = year_composition(holdings, year, weight_mode)
            current_val = basket_value(basket, fill_src, date)
            if current_val > 0 and new_comp:
                after_sell = current_val * (1 - SLIPPAGE) - COMMISSION
                basket     = build_basket(after_sell, new_comp, fill_src, date)
                # original_entry_val and trade_min_val intentionally NOT reset

        prev_year = year

        # ── Execute an order that comes due today (from an earlier signal) ────
        executed = execute_due(i, date, year, spx_fill)

        # ── Evaluate signals on today's SPX close → schedule a fill ──────────
        if not executed and pending is None:
            if position == "OUT":
                vote_gate   = bool(row["vote_gate"])
                cooldown_ok = cooldown_until is None or date > cooldown_until
                washout_buy = not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate
                recross_ok  = last_sell_reason == "climax-top" or (
                    last_exit_spx is not None and spx_price > last_exit_spx)
                trend_buy   = bool(row["ma200_recross"]) and recross_ok
                do_buy = cooldown_ok and (washout_buy or trend_buy)
                if do_buy and i + execution_lag < n:
                    if washout_buy:
                        trigger = (("VIX" if row["vix_vote"] else "") +
                                   ("+" if row["vix_vote"] and row["ma200_vote"] else "") +
                                   ("MA200" if row["ma200_vote"] else ""))
                    else:
                        trigger = "MA200-recross"
                    pending = {"action": "BUY", "fill_at": i + execution_lag,
                               "trigger": trigger}

            elif position == "IN":
                trade_low  = min(trade_low, spx_price)
                trade_high = max(trade_high, spx_price)
                macd_age = 0 if bool(row["macd_cross"]) else macd_age + 1
                ext_age  = 0 if bool(row["ext10"])      else ext_age + 1
                current_val   = basket_value(basket, prices, date)
                trade_min_val = min(trade_min_val, current_val)
                bearish_div = price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP
                climax      = (macd_age < CLIMAX_VOTE_WINDOW) and (ext_age < CLIMAX_VOTE_WINDOW)
                trail_hit   = spx_price <= trade_high * (1 - TRAILING_STOP_PCT / 100)
                if bearish_div:
                    reason = "bearish-divergence"
                elif climax:
                    reason = "climax-top"
                elif trail_hit:
                    reason = "trailing-stop"
                else:
                    reason = None
                if reason and i + execution_lag < n:
                    pending = {"action": "SELL", "fill_at": i + execution_lag,
                               "reason": reason}

            # Same-day execution (lag 0): the order just set is due this bar.
            executed = execute_due(i, date, year, spx_fill)

        # ── Mark-to-market (always on closes) ────────────────────────────────
        if position == "IN":
            values[date] = basket_value(basket, prices, date)
        else:
            values[date] = portfolio

    open_trade = None
    if position == "IN":
        last_date = df.index[-1]
        last_val  = basket_value(basket, prices, last_date)
        eff_last  = last_val * (1 - SLIPPAGE)
        gross_ret = (eff_last - original_entry_val) / original_entry_val if original_entry_val else 0
        trade_min_val = min(trade_min_val, last_val)
        open_trade = {
            "entry_date":         entry_date,
            "entry_basket_val":   original_entry_val,
            "current_date":       last_date,
            "current_basket_val": last_val,
            "return_pct":         gross_ret * 100,
            "max_drawdown_pct":   (trade_min_val - original_entry_val) / original_entry_val * 100,
            "accumulated":        eff_last,
            "buy_trigger":        buy_trigger,
            "n_holdings":         entry_names,
        }

    return pd.Series(values, name=f"strategy-{weight_mode}"), trades, open_trade


def run_basket_buyhold(df: pd.DataFrame, holdings: dict[int, list[tuple[str, float]]],
                       prices: dict[str, pd.Series], weight_mode: str) -> pd.Series:
    """Buy & hold the top-17 basket from day one, rebalanced to the new list at
    each year boundary (with slippage + commission on the churn)."""
    first_date = df.index[0]
    basket = build_basket(INITIAL_CAPITAL - COMMISSION,
                          year_composition(holdings, first_date.year, weight_mode),
                          prices, first_date)
    values: dict = {}
    prev_year = first_date.year
    for date in df.index:
        if date.year != prev_year:
            comp = year_composition(holdings, date.year, weight_mode)
            val  = basket_value(basket, prices, date)
            if val > 0 and comp:
                basket = build_basket(val * (1 - SLIPPAGE) - COMMISSION, comp, prices, date)
            prev_year = date.year
        values[date] = basket_value(basket, prices, date)
    return pd.Series(values, name=f"buyhold-{weight_mode}")


def run_spx_benchmark(df: pd.DataFrame) -> pd.Series:
    first = df["price"].iloc[0]
    return (INITIAL_CAPITAL * df["price"] / first).rename("SPX")


# ── Metrics / printing ────────────────────────────────────────────────────────

def compute_metrics(values: pd.Series, trades: list[dict] | None = None,
                    open_trade: dict | None = None) -> dict:
    dr    = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    tr    = (values.iloc[-1] / values.iloc[0]) - 1
    cagr  = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    mdd   = ((values - values.cummax()) / values.cummax()).min()
    std   = dr.std()
    sh    = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    m = {
        "Total Return": f"{tr:.1%}",
        "CAGR":         f"{cagr:.1%}",
        "Max Drawdown": f"{mdd:.1%}",
        "Sharpe Ratio": f"{sh:.2f}",
        "Final Value":  f"${values.iloc[-1]:,.0f}",
    }
    if trades is not None:
        n       = len(trades)
        wins    = sum(1 for t in trades if t["return_pct"] > 0)
        in_days = sum((t["exit_date"] - t["entry_date"]).days for t in trades)
        if open_trade:
            in_days += (open_trade["current_date"] - open_trade["entry_date"]).days
        tot = (values.index[-1] - values.index[0]).days
        m.update({
            "# Trades":       str(n),
            "Win Rate":       f"{wins/n:.1%}" if n else "—",
            "Time in Market": f"{in_days/tot:.1%}" if tot else "—",
        })
    return m


def print_metrics_table(columns: list[tuple[str, dict]]) -> None:
    keys = list(dict.fromkeys(k for _, m in columns for k in m))
    col  = 15
    hdr  = f"{'Metric':<16}" + "".join(f"{name:>{col}}" for name, _ in columns)
    sep  = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        print(f"  {k:<14}" + "".join(f"{m.get(k, '—'):>{col}}" for _, m in columns))
    print(sep)


def print_trades(label: str, trades: list[dict], open_trade: dict | None = None) -> None:
    print(f"\n── {label} — trade log ──")
    if not trades and not open_trade:
        print("No completed trades.")
        return
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}"
           f"  {'Return':>8}  {'MaxDD':>7}  {'Entry Port':>12}  {'Exit Port':>12}"
           f"  {'Names':>5}  {'Buy trigger':>12}  Sell reason")
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+6.1f}%"
            f"  ${t['entry_basket_val']:>11,.0f}  ${t['accumulated']:>11,.0f}"
            f"  {t['n_holdings']:>5}  {t.get('buy_trigger','—'):>12}  {t.get('sell_reason','—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+6.1f}%"
            f"  ${open_trade['entry_basket_val']:>11,.0f}  ${open_trade['accumulated']:>11,.0f}"
            f"  {open_trade['n_holdings']:>5}  {open_trade.get('buy_trigger','—'):>12}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def print_coverage(df: pd.DataFrame, holdings: dict[int, list[tuple[str, float]]],
                   prices: dict[str, pd.Series]) -> None:
    """Per-year count of top-17 names with price data (Jan-2 check date)."""
    print("\n── Data coverage (names with price data / 17, by year) ──")
    years = sorted(y for y in holdings if df.index[0].year <= y <= df.index[-1].year)
    parts = []
    for y in years:
        check = pd.Timestamp(year=y, month=1, day=15)
        have  = sum(1 for t, _ in holdings[y] if get_price(prices, t, check) is not None)
        parts.append(f"{y}:{have}")
    for i in range(0, len(parts), 9):
        print("  " + "  ".join(parts[i:i + 9]))
    if any(int(p.split(":")[1]) < TOP_N for p in parts):
        print(f"  ⚠ Years below 17 are running on a PARTIAL basket — missing tickers are")
        print(f"    skipped and weights renormalized. Results are not the full top-17 story")
        print(f"    until SP500/fetch_stock_history.py has downloaded every ticker.")


def plot_results(df, curves: dict[str, pd.Series], trades, open_trade) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 0.8]}
    )
    ax1, ax2, ax3 = axes

    fig.suptitle(
        "S&P 500 Top-17 Breadth Strategy — Equal Weight vs Market-Cap Weight\n"
        f"BUY: breadth200 < {BUY_B200_THRESH}%  AND  (VIX > {VIX_BUY_THRESH} OR SPX > MA{MA200_WINDOW})"
        f"   OR  SPX re-crosses above MA{MA200_WINDOW} (after climax-top exit or above prior exit)\n"
        f"SELL: divergence (SPX ≥{DIVERGENCE_PRICE_RISE}%/{DIVERGENCE_WINDOW}d + breadth200 "
        f"-{DIVERGENCE_BREADTH_FALL}pts < {DIVERGENCE_BREADTH_CAP}%)  OR  climax top  OR  "
        f"trailing stop ({TRAILING_STOP_PCT:.0f}%)  |  Yearly top-17 reconstitution  |  "
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}",
        fontsize=9, fontweight="bold"
    )

    colors = {"Equal-weight strategy":  "#FF5722",
              "Cap-weight strategy":    "#9C27B0",
              "Equal-weight B&H":       "#FFB74D",
              "Cap-weight B&H":         "#CE93D8",
              "Buy & Hold SPX":         "#2196F3"}
    for name, series in curves.items():
        ax1.plot(series.index, series, label=name,
                 color=colors.get(name), linewidth=1.4 if "strategy" in name else 1.0,
                 alpha=1.0 if "strategy" in name else 0.8)

    strat = curves["Equal-weight strategy"]
    all_entries = [t["entry_date"] for t in trades] + (
        [open_trade["entry_date"]] if open_trade else [])
    all_exits = [t["exit_date"] for t in trades]
    if all_entries:
        ax1.scatter(all_entries, strat.reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=80, zorder=5, label="Buy")
    if all_exits:
        ax1.scatter(all_exits, strat.reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=80, zorder=5, label="Sell")

    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio Value ($, log)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df["breadth"], color="#7B1FA2", linewidth=1.0,
             label="% Above 200-Day MA (S&P 500)")
    ax2.axhline(BUY_B200_THRESH, color="green", linestyle="--", linewidth=1.0,
                label=f"Buy gate: <{BUY_B200_THRESH}%")
    ax2.axhline(DIVERGENCE_BREADTH_CAP, color="red", linestyle="--", linewidth=0.9,
                label=f"Sell cap: <{DIVERGENCE_BREADTH_CAP}%")
    ax2.fill_between(df.index, df["breadth"], BUY_B200_THRESH,
                     where=df["breadth"] < BUY_B200_THRESH, color="green", alpha=0.12)
    if all_entries:
        ax2.scatter(all_entries, df["breadth"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=60, zorder=5)
    if all_exits:
        ax2.scatter(all_exits, df["breadth"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=60, zorder=5)
    ax2.set_ylabel("Breadth (%)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    ax3.plot(df.index, df["price"], color="#546E7A", linewidth=1.0, label="S&P 500")
    ax3.plot(df.index, df["ma200"], color="orange", linewidth=0.8, linestyle="--",
             label=f"MA{MA200_WINDOW}")
    if all_entries:
        ax3.scatter(all_entries, df["price"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=50, zorder=5)
    if all_exits:
        ax3.scatter(all_exits, df["price"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=50, zorder=5)
    ax3.set_ylabel("SPX")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out = DATA_DIR / "sp500_top17_backtest.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def main() -> None:
    print("Loading SPX + breadth data...")
    df = load_data()
    print(f"Date range  : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    print(f"Buy signal  : breadth200 < {BUY_B200_THRESH}%  (washout entry)")
    print(f"Vote gate   : VIX > {VIX_BUY_THRESH} OR SPX > MA{MA200_WINDOW}  (≥1 of 2 must agree)")
    print(f"           OR trend re-entry: SPX re-crosses above MA{MA200_WINDOW} after a climax-top")
    print(f"              exit or when back above the prior exit price")
    print(f"Sell signal : divergence / climax top / {TRAILING_STOP_PCT:.0f}% trailing stop (on SPX)")
    print(f"Costs       : ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")
    print(f"Cooldown    : {COOLDOWN_DAYS} calendar days after each sell")
    print(f"Execution   : signal on SPX close → fill at next day's stock OPEN")
    print(f"Basket      : top {TOP_N} S&P 500 stocks by market cap, reconstituted yearly")
    print(f"              (year Y uses the end-of-Y-1 ranking; annual rebalance while IN)")
    print("\n⚠ CAVEAT: the top-17 list is hand-curated from historical year-end market-cap")
    print("  rankings; ranks ~10–17 and cap figures are approximate. Yearly reconstitution")
    print("  reduces but does not eliminate survivorship/curation bias.")

    holdings    = load_holdings()
    all_tickers = {t for comps in holdings.values() for t, _ in comps}
    print(f"\nLoading stock prices for {len(all_tickers)} unique tickers...")
    prices = load_stock_prices(all_tickers)
    opens  = load_stock_prices(all_tickers, col="Open")

    print_coverage(df, holdings, prices)

    strat_eq, trades_eq, open_eq = run_strategy(df, holdings, prices, opens, "equal")
    strat_mc, trades_mc, open_mc = run_strategy(df, holdings, prices, opens, "mktcap")
    bh_eq  = run_basket_buyhold(df, holdings, prices, "equal")
    bh_mc  = run_basket_buyhold(df, holdings, prices, "mktcap")
    bh_spx = run_spx_benchmark(df)

    print_metrics_table([
        ("EqualW Strat",  compute_metrics(strat_eq, trades_eq, open_eq)),
        ("CapW Strat",    compute_metrics(strat_mc, trades_mc, open_mc)),
        ("EqualW B&H",    compute_metrics(bh_eq)),
        ("CapW B&H",      compute_metrics(bh_mc)),
        ("SPX B&H",       compute_metrics(bh_spx)),
    ])

    print_trades("Equal-weight strategy", trades_eq, open_eq)
    print_trades("Market-cap-weight strategy", trades_mc, open_mc)

    plot_results(df, {
        "Equal-weight strategy": strat_eq,
        "Cap-weight strategy":   strat_mc,
        "Equal-weight B&H":      bh_eq,
        "Cap-weight B&H":        bh_mc,
        "Buy & Hold SPX":        bh_spx,
    }, trades_eq, open_eq)


if __name__ == "__main__":
    main()
