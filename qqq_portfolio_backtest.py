"""
Portfolio Backtest: QQQ 60% + NDX Top-1 Stock 30% + TQQQ 10%

Same buy/sell signals as qqq_backtest.py applied to a fixed-weight portfolio:
  - 60% QQQ   (tracked via NASDAQ100.csv — NDX proxy)
  - 30% NDX top-1 holding for that year (from NASDAQ100/stock_prices/*.csv)
  - 10% TQQQ  (fetched from yfinance)

BUY  (while OUT): breadth200 < 26%
                  AND at least 1 of 2 vote:
                    VIX > 30  (fear spike / panic bottom)
                    price > MA200  (uptrend pullback — safe to buy)
SELL (while IN):  Bearish divergence — price rose >= 3% over 60 days
                  while breadth200 fell >= 20 pts AND breadth200 < 60%

Top-1 stock is locked at trade entry (no mid-trade rebalance on year change).
If TQQQ data is unavailable (pre-2010), that 10% rolls into QQQ.
If the top-1 stock CSV is missing, that 30% rolls into QQQ.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    from fetch_investing_data import fetch_all_updates
    fetch_all_updates(verbose=True)
except Exception as _fetch_err:
    print(f"[data fetch skipped: {_fetch_err}]")

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

# ── Portfolio weights ─────────────────────────────────────────────────────────
QQQ_WEIGHT   = 0.60
STOCK_WEIGHT = 0.30
TQQQ_WEIGHT  = 0.10
SPY_WEIGHT   = 0.0
SOXX_WEIGHT  = 0.0

# ── Buy thresholds ────────────────────────────────────────────────────────────
BUY_B200_THRESH = 26.0
VIX_BUY_THRESH  = 30.0
MA200_WINDOW    = 200

# ── Sell — bearish divergence ─────────────────────────────────────────────────
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0

# ── Sell — climax top (NDX extension + MACD break within a window) ───────────
EXT10_PCT           = 5.0   # % above 10-day MA that counts as "extended"
CLIMAX_VOTE_WINDOW  = 10    # days within which both climax signals must fire

# ── Sell — trailing stop (on NDX, the signal index) ──────────────────────────
TRAILING_STOP_PCT = 25.0    # % below the NDX high since entry

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day NDX closes, so the earliest tradeable fill is the
# NEXT session. Default: a signal on day t fills at day t+1's OPEN of every leg.
# Set EXECUTION_LAG=0 and FILL_PRICE="close" for the legacy same-day-close
# (look-ahead) fill that trades at the very close that produced the signal.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

# ── Shared ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL      = 10_000.0
COMMISSION           = 1.0
SLIPPAGE             = 0.0005
COOLDOWN_DAYS        = 15   # aligned with qqq_backtest.py (30 delayed the 2008-09 re-entries)
MONTHLY_CONTRIBUTION = 0.0
YEARLY_CONTRIBUTION  = 0.0
START_DATE: str | None = None
END_DATE:   str | None = None

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


# ── Name -> ticker mapping for NDX top-1 holdings ────────────────────────────
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


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _name_to_ticker(name: str) -> str | None:
    nl = name.lower()
    for key, ticker in _NAME_TO_TICKER:
        if key in nl:
            return ticker
    return None


def load_top_holdings() -> dict[int, str]:
    """Return {year: ticker} for the #1 NDX holding each year."""
    df = pd.read_csv(TOP_HOLDINGS_FILE)
    result: dict[int, str] = {}
    for _, row in df.iterrows():
        if int(row["Rank"]) == 1:
            ticker = _name_to_ticker(str(row["Holding"]))
            if ticker:
                result[int(row["Year"])] = ticker
    return result


def _load_stock_series(ticker: str, col: str = "Close") -> pd.Series | None:
    path = STOCK_PRICE_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # Stock CSVs from yfinance may have mixed tz-aware/naive date strings;
    # parse with utc=True then strip tz so the index stays tz-naive.
    df["Date"] = (pd.to_datetime(df["Date"], format="mixed", utc=True)
                  .dt.tz_localize(None))
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    use = col if col in df.columns else "Close"
    return df[use].rename(ticker).astype(float)


def _load_etf(ticker: str, start: str = "1993-01-01", col: str = "Close") -> pd.Series | None:
    if not _HAS_YF:
        print(f"[yfinance not installed -- {ticker} weight will be merged into QQQ]")
        return None
    try:
        print(f"Fetching {ticker} from yfinance...")
        raw = yf.download(ticker, start=start, progress=False)
        series = raw[col] if col in raw.columns else raw.iloc[:, 0]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.rename(ticker).astype(float)
        series.index = pd.to_datetime(series.index)
        series.index.name = "Date"
        return series
    except Exception as exc:
        print(f"[{ticker} fetch failed: {exc} -- weight merged into QQQ]")
        return None


def _safe(series: pd.Series | None, date: pd.Timestamp) -> float:
    """Return series value at date, or NaN if missing."""
    if series is None:
        return float("nan")
    try:
        v = series.loc[date]
        return float(v) if not pd.isna(v) else float("nan")
    except KeyError:
        return float("nan")


def load_data() -> tuple[pd.DataFrame, dict[int, str], dict[str, pd.Series], pd.Series | None, pd.Series | None, pd.Series | None]:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price", "Open": "open"})
    ndx["price"] = _parse_price(ndx["price"])
    ndx["open"]  = _parse_price(ndx["open"])

    b200 = _load_breadth()

    vix = pd.read_csv(VIX_FILE)
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix.set_index("Date", inplace=True)
    vix["vix"] = _parse_price(vix["Price"])

    merged = ndx[["price", "open"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged = merged.join(vix[["vix"]], how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]
    merged["vix"]   = merged["vix"].ffill()
    merged["ma200"] = merged["price"].rolling(MA200_WINDOW).mean()

    merged["vix_vote"]   = merged["vix"].apply(
        lambda v: True if pd.isna(v) else v > VIX_BUY_THRESH)
    merged["ma200_vote"] = merged.apply(
        lambda r: True if pd.isna(r["ma200"]) else r["price"] > r["ma200"], axis=1)
    merged["vote_gate"]  = merged["vix_vote"] | merged["ma200_vote"]

    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)

    # Climax-top components on NDX (exit fires only when both occur post-entry,
    # within CLIMAX_VOTE_WINDOW days — tracked in run_strategy)
    close = merged["price"]
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    merged["macd_cross"] = ((hist < 0) & (hist.shift(1) >= 0)).fillna(False)
    merged["ext10"] = (close / close.rolling(10).mean() - 1 >= EXT10_PCT / 100).fillna(False)

    # Trend re-entry: fresh close back above MA200 (NDX signal series).
    merged["ma200_recross"] = (
        (close > merged["ma200"]) & (close.shift(1) <= merged["ma200"].shift(1))
    ).fillna(False)

    top_holdings = load_top_holdings()
    unique_tickers = set(top_holdings.values())

    aligned_stocks: dict[str, pd.Series] = {}
    missing: list[str] = []
    for ticker in sorted(unique_tickers):
        s = _load_stock_series(ticker)
        if s is not None:
            aligned_stocks[ticker] = s.reindex(merged.index).ffill()
        else:
            missing.append(ticker)
    if missing:
        print(f"[stock CSV not found for: {', '.join(missing)} -- weight shifted to QQQ]")

    def _align(raw: pd.Series | None) -> pd.Series | None:
        return raw.reindex(merged.index).ffill() if raw is not None else None

    aligned_tqqq = _align(_load_etf("TQQQ", start="2010-01-01"))
    aligned_spy  = _align(_load_etf("SPY",  start="1993-01-01"))
    aligned_soxx = _align(_load_etf("SOXX", start="2001-07-01"))

    return merged, top_holdings, aligned_stocks, aligned_tqqq, aligned_spy, aligned_soxx


def load_open_series(
    top_holdings: dict[int, str],
    index: pd.Index,
) -> tuple[dict[str, pd.Series], pd.Series | None, pd.Series | None, pd.Series | None]:
    """Open-price counterparts of the aligned Close series returned by load_data(),
    used to fill next-day-open orders. NDX's own open lives in df["open"]."""
    def _align(raw: pd.Series | None) -> pd.Series | None:
        return raw.reindex(index).ffill() if raw is not None else None

    stocks_open: dict[str, pd.Series] = {}
    for ticker in sorted(set(top_holdings.values())):
        s = _load_stock_series(ticker, col="Open")
        if s is not None:
            stocks_open[ticker] = s.reindex(index).ffill()
    tqqq_open = _align(_load_etf("TQQQ", start="2010-01-01", col="Open"))
    spy_open  = _align(_load_etf("SPY",  start="1993-01-01", col="Open"))
    soxx_open = _align(_load_etf("SOXX", start="2001-07-01", col="Open"))
    return stocks_open, tqqq_open, spy_open, soxx_open


# ─────────────────────────────────────────────────────────────────────────────

def _position_at_date(
    df_pre: pd.DataFrame,
    top_holdings: dict[int, str],
    cooldown_days: int = COOLDOWN_DAYS,
) -> tuple[bool, str | None]:
    """
    Simulate buy/sell signals on df_pre to determine position at the end.
    Returns (is_in_market, holding_ticker_or_None).
    Used to detect whether the strategy would be IN on a given start date.
    """
    position       = "OUT"
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_price: float | None = None
    holding_ticker: str | None = None
    ndx_high = 0.0
    macd_age = ext_age = 10**9

    for date, row in df_pre.iterrows():
        if position == "OUT":
            ndx_price   = float(row["price"])
            cooldown_ok = cooldown_until is None or date > cooldown_until
            vote_gate   = bool(row["vote_gate"])
            washout_buy = (
                not pd.isna(row["breadth"])
                and row["breadth"] < BUY_B200_THRESH
                and vote_gate
            )
            recross_ok  = last_sell_reason == "climax-top" or (
                last_exit_price is not None and ndx_price > last_exit_price)
            trend_buy   = bool(row["ma200_recross"]) and recross_ok
            if cooldown_ok and (washout_buy or trend_buy):
                position = "IN"
                year           = date.year
                holding_ticker = top_holdings.get(year) or top_holdings.get(year - 1)
                ndx_high = ndx_price
                macd_age = ext_age = 10**9
        else:
            ndx_price = float(row["price"])
            ndx_high  = max(ndx_high, ndx_price)
            macd_age = 0 if bool(row["macd_cross"]) else macd_age + 1
            ext_age  = 0 if bool(row["ext10"])      else ext_age + 1
            bearish_div = (
                bool(row["price_rose"]) and bool(row["breadth_fell"])
                and row["breadth"] < DIVERGENCE_BREADTH_CAP
            )
            climax    = (macd_age < CLIMAX_VOTE_WINDOW) and (ext_age < CLIMAX_VOTE_WINDOW)
            trail_hit = ndx_price <= ndx_high * (1 - TRAILING_STOP_PCT / 100)
            if bearish_div or climax or trail_hit:
                position       = "OUT"
                holding_ticker = None
                last_sell_reason = ("bearish-divergence" if bearish_div
                                    else "climax-top" if climax else "trailing-stop")
                last_exit_price = ndx_price
                cooldown_until = date + pd.Timedelta(days=cooldown_days)

    return position == "IN", holding_ticker


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


def run_strategy(
    df: pd.DataFrame,
    top_holdings: dict[int, str],
    aligned_stocks: dict[str, pd.Series],
    aligned_tqqq: pd.Series | None,
    aligned_spy:  pd.Series | None,
    aligned_soxx: pd.Series | None,
    cooldown_days: int = 0,
    initial_capital: float = INITIAL_CAPITAL,
    monthly_contribution: float = 0.0,
    yearly_contribution: float = 0.0,
    force_entry_on_start: bool = False,
    force_ticker: str | None = None,
    execution_lag: int = EXECUTION_LAG,
    fill_on: str = FILL_PRICE,
    aligned_stocks_open: dict[str, pd.Series] | None = None,
    aligned_tqqq_open: pd.Series | None = None,
    aligned_spy_open:  pd.Series | None = None,
    aligned_soxx_open: pd.Series | None = None,
) -> tuple[pd.Series, list[dict], dict | None, float]:
    """Signals are computed on the NDX close; a signal on day t fills
    `execution_lag` bars later at that bar's open (fill_on="open", the default and
    realistic choice) or close. lag=0 requires fill_on="close" — the legacy
    same-day look-ahead fill. Mark-to-market always uses closes. When an asset's
    open series isn't supplied, that leg falls back to the fill-bar close."""
    if fill_on == "open" and execution_lag < 1:
        raise ValueError("fill_on='open' requires execution_lag >= 1 (open precedes close)")
    aligned_stocks_open = aligned_stocks_open or {}

    position       = "OUT"
    portfolio      = initial_capital
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_price: float | None = None
    trades: list[dict] = []
    values: dict[pd.Timestamp, float] = {}

    # Open-trade state
    qqq_shares    = stock_shares   = tqqq_shares   = 0.0
    spy_shares    = soxx_shares    = 0.0
    qqq_entry_px  = stock_entry_px = tqqq_entry_px = 0.0
    spy_entry_px  = soxx_entry_px  = 0.0
    holding_ticker: str | None = None
    entry_date: pd.Timestamp | None = None
    buy_trigger = ""
    trade_low_val = 0.0

    # Independent cash buckets — each compounds without forced rebalancing
    qqq_bucket   = initial_capital * QQQ_WEIGHT
    stock_bucket = initial_capital * STOCK_WEIGHT
    tqqq_bucket  = initial_capital * TQQQ_WEIGHT
    spy_bucket   = initial_capital * SPY_WEIGHT
    soxx_bucket  = initial_capital * SOXX_WEIGHT
    # Fractions of QQQ position owned by each bucket when that component is unavailable
    qqq_qqq_frac   = 1.0
    stock_qqq_frac = 0.0
    tqqq_qqq_frac  = 0.0
    spy_qqq_frac   = 0.0
    soxx_qqq_frac  = 0.0
    stock_active = tqqq_active = spy_active = soxx_active = False

    force_buy_pending    = force_entry_on_start
    pending_force_ticker = force_ticker

    cash_reserve      = 0.0   # contributions accumulated, deployed at next buy
    total_contributed = 0.0
    prev_month: int | None = None
    prev_year:  int | None = None

    # ── Execution lag via signal shift ───────────────────────────────────────
    # Decisions read data from `execution_lag` bars ago (what was actually known
    # at that close); the fill then happens on the current bar. lag=0 is the
    # identity shift, so with fill_on="close" this reproduces the legacy same-day
    # look-ahead exactly. All *_sig arrays are positional (indexed by i).
    lag = execution_lag
    _b   = df["breadth"].shift(lag).to_numpy()
    _ndx = df["price"].shift(lag).to_numpy()          # NDX close as of the signal bar
    _vg  = df["vote_gate"].shift(lag).fillna(False).to_numpy()
    _vv  = df["vix_vote"].shift(lag).fillna(False).to_numpy()
    _mv  = df["ma200_vote"].shift(lag).fillna(False).to_numpy()
    _pr  = df["price_rose"].shift(lag).fillna(False).to_numpy()
    _bf  = df["breadth_fell"].shift(lag).fillna(False).to_numpy()
    _rc  = df["ma200_recross"].shift(lag).fillna(False).to_numpy()
    _mc  = df["macd_cross"].shift(lag).fillna(False).to_numpy()
    _ex  = df["ext10"].shift(lag).fillna(False).to_numpy()
    _open = df["open"].to_numpy() if "open" in df.columns else df["price"].to_numpy()

    def _fillpx(close_s, open_s, date):
        """Fill price for one leg: the fill-bar open when available, else close."""
        if fill_on == "open" and open_s is not None:
            v = _safe(open_s, date)
            if not pd.isna(v):
                return v
        return _safe(close_s, date)

    for i, (date, row) in enumerate(df.iterrows()):
        # ── Periodic contributions (skip the very first row) ──────────────────
        if prev_month is not None:
            contrib = 0.0
            if monthly_contribution > 0 and date.month != prev_month:
                contrib += monthly_contribution
            if yearly_contribution > 0 and date.year != prev_year:
                contrib += yearly_contribution
            if contrib > 0:
                cash_reserve      += contrib
                total_contributed += contrib
        prev_month = date.month
        prev_year  = date.year

        ndx_price    = float(row["price"])       # today's NDX close (fill legacy + MTM)
        ndx_fill     = float(_open[i]) if (fill_on == "open" and not pd.isna(_open[i])) else ndx_price
        ndx_sig      = float(_ndx[i]) if not pd.isna(_ndx[i]) else ndx_price   # signal-bar NDX
        breadth      = _b[i]                      # signal-bar breadth (may be NaN early)
        price_rose   = bool(_pr[i])
        breadth_fell = bool(_bf[i])

        if position == "OUT":
            if force_buy_pending:
                do_buy               = True
                force_buy_pending    = False
                stock_ticker         = pending_force_ticker
                pending_force_ticker = None
            else:
                vote_gate   = bool(_vg[i])
                cooldown_ok = cooldown_until is None or date > cooldown_until
                washout_buy = (
                    not pd.isna(breadth)
                    and breadth < BUY_B200_THRESH
                    and vote_gate
                )
                # Trend re-entry on a fresh MA200 recross (NDX): rejoin when the
                # last exit was a climax-top or NDX is back above the prior exit.
                recross_ok  = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and ndx_sig > last_exit_price)
                trend_buy   = bool(_rc[i]) and recross_ok
                do_buy = cooldown_ok and (washout_buy or trend_buy)
                if do_buy:
                    year         = date.year
                    stock_ticker = top_holdings.get(year) or top_holdings.get(year - 1)

            if do_buy:
                stock_px = _fillpx(aligned_stocks.get(stock_ticker) if stock_ticker else None,
                                   aligned_stocks_open.get(stock_ticker) if stock_ticker else None, date)
                tqqq_px  = _fillpx(aligned_tqqq, aligned_tqqq_open, date)
                spy_px   = _fillpx(aligned_spy,  aligned_spy_open,  date)
                soxx_px  = _fillpx(aligned_soxx, aligned_soxx_open, date)

                # Sweep accumulated contributions into buckets before buying
                if cash_reserve > 0:
                    qqq_bucket   += cash_reserve * QQQ_WEIGHT
                    stock_bucket += cash_reserve * STOCK_WEIGHT
                    tqqq_bucket  += cash_reserve * TQQQ_WEIGHT
                    spy_bucket   += cash_reserve * SPY_WEIGHT
                    soxx_bucket  += cash_reserve * SOXX_WEIGHT
                    cash_reserve = 0.0

                # Deduct commission proportionally across all buckets
                total_pre  = qqq_bucket + stock_bucket + tqqq_bucket + spy_bucket + soxx_bucket
                comm_scale = (total_pre - COMMISSION) / total_pre if total_pre > 0 else 1.0
                qqq_bucket   *= comm_scale
                stock_bucket *= comm_scale
                tqqq_bucket  *= comm_scale
                spy_bucket   *= comm_scale
                soxx_bucket  *= comm_scale

                stock_active = not pd.isna(stock_px)
                tqqq_active  = not pd.isna(tqqq_px)
                spy_active   = not pd.isna(spy_px)
                soxx_active  = not pd.isna(soxx_px)

                # Fold unavailable buckets into QQQ for this trade
                eff_qqq  = (qqq_bucket
                            + (0.0 if stock_active else stock_bucket)
                            + (0.0 if tqqq_active  else tqqq_bucket)
                            + (0.0 if spy_active   else spy_bucket)
                            + (0.0 if soxx_active  else soxx_bucket))
                eff_stock = stock_bucket if stock_active else 0.0
                eff_tqqq  = tqqq_bucket  if tqqq_active  else 0.0
                eff_spy   = spy_bucket   if spy_active   else 0.0
                eff_soxx  = soxx_bucket  if soxx_active  else 0.0

                # Track what fraction of the QQQ position each bucket owns
                if eff_qqq > 0:
                    qqq_qqq_frac   = qqq_bucket   / eff_qqq
                    stock_qqq_frac = (stock_bucket / eff_qqq) if not stock_active else 0.0
                    tqqq_qqq_frac  = (tqqq_bucket  / eff_qqq) if not tqqq_active  else 0.0
                    spy_qqq_frac   = (spy_bucket   / eff_qqq) if not spy_active   else 0.0
                    soxx_qqq_frac  = (soxx_bucket  / eff_qqq) if not soxx_active  else 0.0
                else:
                    qqq_qqq_frac = 1.0
                    stock_qqq_frac = tqqq_qqq_frac = spy_qqq_frac = soxx_qqq_frac = 0.0

                qqq_entry_px   = ndx_fill  * (1 + SLIPPAGE)
                stock_entry_px = stock_px  * (1 + SLIPPAGE) if stock_active else 0.0
                tqqq_entry_px  = tqqq_px   * (1 + SLIPPAGE) if tqqq_active  else 0.0
                spy_entry_px   = spy_px    * (1 + SLIPPAGE) if spy_active   else 0.0
                soxx_entry_px  = soxx_px   * (1 + SLIPPAGE) if soxx_active  else 0.0

                qqq_shares   = eff_qqq   / qqq_entry_px
                stock_shares = eff_stock / stock_entry_px if stock_entry_px > 0 else 0.0
                tqqq_shares  = eff_tqqq  / tqqq_entry_px  if tqqq_entry_px  > 0 else 0.0
                spy_shares   = eff_spy   / spy_entry_px   if spy_entry_px   > 0 else 0.0
                soxx_shares  = eff_soxx  / soxx_entry_px  if soxx_entry_px  > 0 else 0.0

                holding_ticker = stock_ticker
                entry_date     = date
                trade_low_val  = eff_qqq + eff_stock + eff_tqqq + eff_spy + eff_soxx
                ndx_high       = ndx_sig
                macd_age = ext_age = 10**9   # climax signals must fire AFTER entry
                position       = "IN"
                buy_trigger    = (
                    ("VIX" if _vv[i] else "")
                    + ("+" if _vv[i] and _mv[i] else "")
                    + ("MA200" if _mv[i] else "")
                )
                # Bucket-level entry/peak/low values
                qqq_entry_val  = qqq_bucket;  qqq_peak_val  = qqq_bucket;  qqq_low_val  = qqq_bucket
                stock_entry_val = stock_bucket; stock_peak_val = stock_bucket; stock_low_val = stock_bucket
                tqqq_entry_val = tqqq_bucket; tqqq_peak_val = tqqq_bucket; tqqq_low_val = tqqq_bucket
                spy_entry_val  = spy_bucket;  spy_peak_val  = spy_bucket;  spy_low_val  = spy_bucket
                soxx_entry_val = soxx_bucket; soxx_peak_val = soxx_bucket; soxx_low_val = soxx_bucket

        elif position == "IN":
            ndx_high = max(ndx_high, ndx_sig)
            macd_age = 0 if bool(_mc[i]) else macd_age + 1
            ext_age  = 0 if bool(_ex[i]) else ext_age + 1
            bearish_div = price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP
            climax      = (macd_age < CLIMAX_VOTE_WINDOW) and (ext_age < CLIMAX_VOTE_WINDOW)
            trail_hit   = ndx_sig <= ndx_high * (1 - TRAILING_STOP_PCT / 100)
            if bearish_div:
                sell_reason = "bearish-divergence"
            elif climax:
                sell_reason = "climax-top"
            elif trail_hit:
                sell_reason = "trailing-stop"
            else:
                sell_reason = None

            if sell_reason:
                stock_px_exit = _fillpx(aligned_stocks.get(holding_ticker) if holding_ticker else None,
                                        aligned_stocks_open.get(holding_ticker) if holding_ticker else None, date)
                tqqq_px_exit  = _fillpx(aligned_tqqq, aligned_tqqq_open, date)
                spy_px_exit   = _fillpx(aligned_spy,  aligned_spy_open,  date)
                soxx_px_exit  = _fillpx(aligned_soxx, aligned_soxx_open, date)

                spx  = stock_px_exit if not pd.isna(stock_px_exit) else 0.0
                tpx  = tqqq_px_exit  if not pd.isna(tqqq_px_exit)  else 0.0
                spyx = spy_px_exit   if not pd.isna(spy_px_exit)   else 0.0
                sxx  = soxx_px_exit  if not pd.isna(soxx_px_exit)  else 0.0

                gross_qqq   = qqq_shares   * ndx_fill  * (1 - SLIPPAGE)
                gross_stock = stock_shares * spx        * (1 - SLIPPAGE)
                gross_tqqq  = tqqq_shares  * tpx        * (1 - SLIPPAGE)
                gross_spy   = spy_shares   * spyx       * (1 - SLIPPAGE)
                gross_soxx  = soxx_shares  * sxx        * (1 - SLIPPAGE)
                gross_total = gross_qqq + gross_stock + gross_tqqq + gross_spy + gross_soxx
                comm_frac   = COMMISSION / gross_total if gross_total > 0 else 0.0

                # Update each bucket: split QQQ proceeds back to contributing buckets
                qqq_bucket   = (gross_qqq * qqq_qqq_frac)                          * (1 - comm_frac)
                stock_bucket = (gross_qqq * stock_qqq_frac + gross_stock)          * (1 - comm_frac)
                tqqq_bucket  = (gross_qqq * tqqq_qqq_frac  + gross_tqqq)           * (1 - comm_frac)
                spy_bucket   = (gross_qqq * spy_qqq_frac   + gross_spy)            * (1 - comm_frac)
                soxx_bucket  = (gross_qqq * soxx_qqq_frac  + gross_soxx)           * (1 - comm_frac)
                total_proc   = qqq_bucket + stock_bucket + tqqq_bucket + spy_bucket + soxx_bucket

                qqq_exit_val  = qqq_bucket;  stock_exit_val = stock_bucket
                tqqq_exit_val = tqqq_bucket; spy_exit_val   = spy_bucket
                soxx_exit_val = soxx_bucket

                entry_val  = qqq_entry_val + stock_entry_val + tqqq_entry_val + spy_entry_val + soxx_entry_val
                gross_ret  = (total_proc - entry_val) / entry_val if entry_val > 0 else 0.0
                max_dd_pct = (trade_low_val - entry_val) / entry_val * 100 if entry_val > 0 else 0.0

                portfolio      = total_proc
                cooldown_until = date + pd.Timedelta(days=cooldown_days)
                last_sell_reason = sell_reason
                last_exit_price = ndx_fill

                def _mdd(low: float, peak: float) -> float:
                    return (low - peak) / peak * 100 if peak > 0 else 0.0

                trades.append({
                    "entry_date":       entry_date,
                    "exit_date":        date,
                    "return_pct":       gross_ret * 100,
                    "max_drawdown_pct": max_dd_pct,
                    "accumulated":      portfolio,
                    "buy_trigger":      buy_trigger,
                    "sell_reason":      sell_reason,
                    "top1_ticker":      holding_ticker,
                    "cooldown_until":   cooldown_until,
                    "qqq_entry_px":    qqq_entry_px / (1 + SLIPPAGE),
                    "qqq_exit_px":     ndx_fill,
                    "qqq_entry_val":   qqq_entry_val,
                    "qqq_exit_val":    qqq_exit_val,
                    "qqq_peak_val":    qqq_peak_val,
                    "qqq_mdd":         _mdd(qqq_low_val, qqq_peak_val),
                    "qqq_earning":     qqq_exit_val - qqq_entry_val,
                    "stock_active":    stock_active,
                    "stock_entry_px":  stock_entry_px / (1 + SLIPPAGE) if stock_active else None,
                    "stock_exit_px":   spx if stock_active else None,
                    "stock_entry_val": stock_entry_val,
                    "stock_exit_val":  stock_exit_val,
                    "stock_peak_val":  stock_peak_val,
                    "stock_mdd":       _mdd(stock_low_val, stock_peak_val),
                    "stock_earning":   stock_exit_val - stock_entry_val,
                    "tqqq_active":    tqqq_active,
                    "tqqq_entry_px":  tqqq_entry_px / (1 + SLIPPAGE) if tqqq_active else None,
                    "tqqq_exit_px":   tpx if tqqq_active else None,
                    "tqqq_entry_val": tqqq_entry_val,
                    "tqqq_exit_val":  tqqq_exit_val,
                    "tqqq_peak_val":  tqqq_peak_val,
                    "tqqq_mdd":       _mdd(tqqq_low_val, tqqq_peak_val),
                    "tqqq_earning":   tqqq_exit_val - tqqq_entry_val,
                    "spy_active":    spy_active,
                    "spy_entry_px":  spy_entry_px / (1 + SLIPPAGE) if spy_active else None,
                    "spy_exit_px":   spyx if spy_active else None,
                    "spy_entry_val": spy_entry_val,
                    "spy_exit_val":  spy_exit_val,
                    "spy_peak_val":  spy_peak_val,
                    "spy_mdd":       _mdd(spy_low_val, spy_peak_val),
                    "spy_earning":   spy_exit_val - spy_entry_val,
                    "soxx_active":    soxx_active,
                    "soxx_entry_px":  soxx_entry_px / (1 + SLIPPAGE) if soxx_active else None,
                    "soxx_exit_px":   sxx if soxx_active else None,
                    "soxx_entry_val": soxx_entry_val,
                    "soxx_exit_val":  soxx_exit_val,
                    "soxx_peak_val":  soxx_peak_val,
                    "soxx_mdd":       _mdd(soxx_low_val, soxx_peak_val),
                    "soxx_earning":   soxx_exit_val - soxx_entry_val,
                })
                position = "OUT"
                qqq_shares = stock_shares = tqqq_shares = spy_shares = soxx_shares = 0.0

        # ── Mark-to-market + per-component peak/trough tracking ──────────────
        if position == "IN":
            stock_now = _safe(aligned_stocks.get(holding_ticker) if holding_ticker else None, date)
            tqqq_now  = _safe(aligned_tqqq,  date)
            spy_now   = _safe(aligned_spy,   date)
            soxx_now  = _safe(aligned_soxx,  date)
            sn   = stock_now if not pd.isna(stock_now) else 0.0
            tn   = tqqq_now  if not pd.isna(tqqq_now)  else 0.0
            spyn = spy_now   if not pd.isna(spy_now)   else 0.0
            sxn  = soxx_now  if not pd.isna(soxx_now)  else 0.0

            qqq_cur   = qqq_shares   * ndx_price
            stock_cur = stock_shares * sn
            tqqq_cur  = tqqq_shares  * tn
            spy_cur   = spy_shares   * spyn
            soxx_cur  = soxx_shares  * sxn
            cur_val   = qqq_cur + stock_cur + tqqq_cur + spy_cur + soxx_cur

            qqq_b   = qqq_cur * qqq_qqq_frac
            stock_b = qqq_cur * stock_qqq_frac + stock_cur
            tqqq_b  = qqq_cur * tqqq_qqq_frac  + tqqq_cur
            spy_b   = qqq_cur * spy_qqq_frac   + spy_cur
            soxx_b  = qqq_cur * soxx_qqq_frac  + soxx_cur

            qqq_peak_val   = max(qqq_peak_val,   qqq_b)
            stock_peak_val = max(stock_peak_val, stock_b)
            tqqq_peak_val  = max(tqqq_peak_val,  tqqq_b)
            spy_peak_val   = max(spy_peak_val,   spy_b)
            soxx_peak_val  = max(soxx_peak_val,  soxx_b)
            qqq_low_val    = min(qqq_low_val,    qqq_b)
            stock_low_val  = min(stock_low_val,  stock_b)
            tqqq_low_val   = min(tqqq_low_val,   tqqq_b)
            spy_low_val    = min(spy_low_val,    spy_b)
            soxx_low_val   = min(soxx_low_val,   soxx_b)

            trade_low_val = min(trade_low_val, cur_val)
            values[date]  = cur_val + cash_reserve
        else:
            values[date] = qqq_bucket + stock_bucket + tqqq_bucket + spy_bucket + soxx_bucket + cash_reserve

    open_trade: dict | None = None
    if position == "IN":
        last_date  = df.index[-1]
        last_ndx   = float(df["price"].iloc[-1])
        last_stock = _safe(aligned_stocks.get(holding_ticker) if holding_ticker else None, last_date)
        last_tqqq  = _safe(aligned_tqqq,  last_date)
        last_spy   = _safe(aligned_spy,   last_date)
        last_soxx  = _safe(aligned_soxx,  last_date)
        ls   = last_stock if not pd.isna(last_stock) else 0.0
        lt   = last_tqqq  if not pd.isna(last_tqqq)  else 0.0
        lspy = last_spy   if not pd.isna(last_spy)   else 0.0
        lsx  = last_soxx  if not pd.isna(last_soxx)  else 0.0

        qqq_cv   = qqq_shares   * last_ndx
        stock_cv = stock_shares * ls
        tqqq_cv  = tqqq_shares  * lt
        spy_cv   = spy_shares   * lspy
        soxx_cv  = soxx_shares  * lsx
        last_val = qqq_cv + stock_cv + tqqq_cv + spy_cv + soxx_cv

        qqq_b_cur   = qqq_cv  * qqq_qqq_frac
        stock_b_cur = qqq_cv  * stock_qqq_frac + stock_cv
        tqqq_b_cur  = qqq_cv  * tqqq_qqq_frac  + tqqq_cv
        spy_b_cur   = qqq_cv  * spy_qqq_frac   + spy_cv
        soxx_b_cur  = qqq_cv  * soxx_qqq_frac  + soxx_cv

        entry_val = qqq_entry_val + stock_entry_val + tqqq_entry_val + spy_entry_val + soxx_entry_val

        def _mdd(low: float, peak: float) -> float:
            return (low - peak) / peak * 100 if peak > 0 else 0.0

        open_trade = {
            "entry_date":       entry_date,
            "current_date":     last_date,
            "return_pct":       (last_val - entry_val) / entry_val * 100 if entry_val > 0 else 0.0,
            "max_drawdown_pct": (trade_low_val - entry_val) / entry_val * 100 if entry_val > 0 else 0.0,
            "accumulated":      last_val + cash_reserve,
            "buy_trigger":      buy_trigger,
            "top1_ticker":      holding_ticker,
            "qqq_entry_px":    qqq_entry_px / (1 + SLIPPAGE),
            "qqq_exit_px":     last_ndx,
            "qqq_entry_val":   qqq_entry_val,
            "qqq_exit_val":    qqq_b_cur,
            "qqq_peak_val":    max(qqq_peak_val, qqq_b_cur),
            "qqq_mdd":         _mdd(qqq_low_val, qqq_peak_val),
            "qqq_earning":     qqq_b_cur - qqq_entry_val,
            "stock_active":    stock_active,
            "stock_entry_px":  stock_entry_px / (1 + SLIPPAGE) if stock_active else None,
            "stock_exit_px":   ls if stock_active else None,
            "stock_entry_val": stock_entry_val,
            "stock_exit_val":  stock_b_cur,
            "stock_peak_val":  max(stock_peak_val, stock_b_cur),
            "stock_mdd":       _mdd(stock_low_val, stock_peak_val),
            "stock_earning":   stock_b_cur - stock_entry_val,
            "tqqq_active":    tqqq_active,
            "tqqq_entry_px":  tqqq_entry_px / (1 + SLIPPAGE) if tqqq_active else None,
            "tqqq_exit_px":   lt if tqqq_active else None,
            "tqqq_entry_val": tqqq_entry_val,
            "tqqq_exit_val":  tqqq_b_cur,
            "tqqq_peak_val":  max(tqqq_peak_val, tqqq_b_cur),
            "tqqq_mdd":       _mdd(tqqq_low_val, tqqq_peak_val),
            "tqqq_earning":   tqqq_b_cur - tqqq_entry_val,
            "spy_active":    spy_active,
            "spy_entry_px":  spy_entry_px / (1 + SLIPPAGE) if spy_active else None,
            "spy_exit_px":   lspy if spy_active else None,
            "spy_entry_val": spy_entry_val,
            "spy_exit_val":  spy_b_cur,
            "spy_peak_val":  max(spy_peak_val, spy_b_cur),
            "spy_mdd":       _mdd(spy_low_val, spy_peak_val),
            "spy_earning":   spy_b_cur - spy_entry_val,
            "soxx_active":    soxx_active,
            "soxx_entry_px":  soxx_entry_px / (1 + SLIPPAGE) if soxx_active else None,
            "soxx_exit_px":   lsx if soxx_active else None,
            "soxx_entry_val": soxx_entry_val,
            "soxx_exit_val":  soxx_b_cur,
            "soxx_peak_val":  max(soxx_peak_val, soxx_b_cur),
            "soxx_mdd":       _mdd(soxx_low_val, soxx_peak_val),
            "soxx_earning":   soxx_b_cur - soxx_entry_val,
        }

    return pd.Series(values, name="portfolio"), trades, open_trade, total_contributed


def run_benchmark(df: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL) -> pd.Series:
    first = df["price"].iloc[0]
    return (initial_capital * df["price"] / first).rename("benchmark")


def compute_metrics(values: pd.Series, trades: list[dict] | None = None) -> dict:
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
        tot     = (values.index[-1] - values.index[0]).days
        m.update({
            "# Trades":       str(n),
            "Win Rate":       f"{wins/n:.1%}" if n else "--",
            "Time in Market": f"{in_days/tot:.1%}" if tot else "--",
        })
    return m


def print_metrics(strat: dict, bench: dict) -> None:
    keys = list(dict.fromkeys(list(strat) + list(bench)))
    col  = 16
    hdr  = f"{'Metric':<22}{'Portfolio':>{col}}{'Buy & Hold NDX':>{col}}"
    sep  = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        print(f"  {k:<20}{strat.get(k, '--'):>{col}}{bench.get(k, '--'):>{col}}")
    print(sep)


def _all_trades(trades: list[dict], open_trade: dict | None) -> list[dict]:
    return trades + ([open_trade] if open_trade else [])


def _is_open(t: dict) -> bool:
    return "exit_date" not in t


def _exit_date(t: dict) -> pd.Timestamp:
    return t["current_date"] if _is_open(t) else t["exit_date"]


def _exit_str(t: dict) -> str:
    return "(open)" if _is_open(t) else t["exit_date"].strftime("%Y-%m-%d")


def _fmt_dollars(v: float | None) -> str:
    if v is None:
        return "       n/a"
    return f"${v:>9,.0f}"


def _fmt_ret(v: float | None) -> str:
    if v is None:
        return "    n/a"
    return f"{v:>+6.1f}%"


def _fmt_mdd(v: float | None) -> str:
    if v is None:
        return "    n/a"
    return f"{v:>+6.1f}%"


def _fmt_earn(v: float | None) -> str:
    if v is None:
        return "       n/a"
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:>8,.0f}"


def _component_summary(label: str, ts: list[dict],
                       entry_key: str, exit_key: str, earn_key: str) -> None:
    valid = [t for t in ts if t.get(entry_key) is not None and t.get(exit_key) is not None]
    if not valid:
        print("  No data for this component in the date range.")
        return
    total_in   = sum(t[entry_key] for t in valid)
    total_out  = sum(t[exit_key]  for t in valid)
    total_earn = sum(t[earn_key]  for t in valid)
    wins = sum(1 for t in valid if (t.get(earn_key) or 0) > 0)
    print(
        f"  {label}:  {len(valid)} trades   Win rate: {wins}/{len(valid)}"
        f"   Total in: ${total_in:,.0f}"
        f"   Total out: ${total_out:,.0f}"
        f"   Net earning: {'+' if total_earn >= 0 else ''}${total_earn:,.0f}"
    )


def print_trades(trades: list[dict], open_trade: dict | None = None) -> None:
    if not trades and not open_trade:
        print("\nNo completed trades.")
        return

    all_t = _all_trades(trades, open_trade)

    col_hdr = (f"{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Ticker':>6}"
               f"  {'Entry $':>10}  {'Exit $':>10}  {'Peak $':>10}"
               f"  {'Max DD':>7}  {'Earning':>10}  {'Return':>7}")
    div = "-" * len(col_hdr)

    def _row(i: int, t: dict, entry_key: str, exit_key: str, peak_key: str,
             mdd_key: str, earn_key: str, ticker: str) -> None:
        days     = (_exit_date(t) - t["entry_date"]).days
        open_tag = " (open)" if _is_open(t) else ""
        ev   = t.get(entry_key)
        xv   = t.get(exit_key)
        ret  = (xv - ev) / ev * 100 if ev and xv else None
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{_exit_str(t):10}  {_days_str(days):>7}  {ticker:>6}"
            f"  {_fmt_dollars(ev)}  {_fmt_dollars(xv)}  {_fmt_dollars(t.get(peak_key))}"
            f"  {_fmt_mdd(t.get(mdd_key))}  {_fmt_earn(t.get(earn_key))}  {_fmt_ret(ret)}{open_tag}"
        )

    # ── Section 1: QQQ (60%) ─────────────────────────────────────────────────
    print(f"\n{'=' * len(col_hdr)}")
    print(f"  Section 1: QQQ  ({QQQ_WEIGHT:.0%} of portfolio)")
    print(f"{'=' * len(col_hdr)}")
    print(col_hdr)
    print(div)
    for i, t in enumerate(all_t, 1):
        _row(i, t, "qqq_entry_val", "qqq_exit_val", "qqq_peak_val",
             "qqq_mdd", "qqq_earning", "--")
    print(div)
    _component_summary("QQQ", all_t, "qqq_entry_val", "qqq_exit_val", "qqq_earning")

    # ── Section 2: Top-1 Stock (30%) ─────────────────────────────────────────
    print(f"\n{'=' * len(col_hdr)}")
    print(f"  Section 2: NDX Top-1 Stock  ({STOCK_WEIGHT:.0%} of portfolio, bucket compounds independently)")
    print(f"{'=' * len(col_hdr)}")
    print(col_hdr)
    print(div)
    for i, t in enumerate(all_t, 1):
        active = t.get("stock_active", False)
        ticker = (t.get("top1_ticker") or "--") if active else "(QQQ)"
        _row(i, t, "stock_entry_val", "stock_exit_val", "stock_peak_val",
             "stock_mdd", "stock_earning", ticker)
    print(div)
    _component_summary("Stock", all_t, "stock_entry_val", "stock_exit_val", "stock_earning")

    # ── Section 3: TQQQ (10%) ────────────────────────────────────────────────
    print(f"\n{'=' * len(col_hdr)}")
    print(f"  Section 3: TQQQ  ({TQQQ_WEIGHT:.0%} of portfolio, bucket compounds independently)")
    print(f"{'=' * len(col_hdr)}")
    print(col_hdr)
    print(div)
    for i, t in enumerate(all_t, 1):
        active = t.get("tqqq_active", False)
        ticker = "TQQQ" if active else "(QQQ)"
        _row(i, t, "tqqq_entry_val", "tqqq_exit_val", "tqqq_peak_val",
             "tqqq_mdd", "tqqq_earning", ticker)
    print(div)
    _component_summary("TQQQ", all_t, "tqqq_entry_val", "tqqq_exit_val", "tqqq_earning")

    # ── Section 4: SPY ───────────────────────────────────────────────────────
    if SPY_WEIGHT > 0:
        print(f"\n{'=' * len(col_hdr)}")
        print(f"  Section 4: SPY  ({SPY_WEIGHT:.0%} of portfolio, bucket compounds independently)")
        print(f"{'=' * len(col_hdr)}")
        print(col_hdr)
        print(div)
        for i, t in enumerate(all_t, 1):
            active = t.get("spy_active", False)
            ticker = "SPY" if active else "(QQQ)"
            _row(i, t, "spy_entry_val", "spy_exit_val", "spy_peak_val",
                 "spy_mdd", "spy_earning", ticker)
        print(div)
        _component_summary("SPY", all_t, "spy_entry_val", "spy_exit_val", "spy_earning")

    # ── Section 5: SOXX ──────────────────────────────────────────────────────
    if SOXX_WEIGHT > 0:
        print(f"\n{'=' * len(col_hdr)}")
        print(f"  Section 5: SOXX  ({SOXX_WEIGHT:.0%} of portfolio, bucket compounds independently)")
        print(f"{'=' * len(col_hdr)}")
        print(col_hdr)
        print(div)
        for i, t in enumerate(all_t, 1):
            active = t.get("soxx_active", False)
            ticker = "SOXX" if active else "(QQQ)"
            _row(i, t, "soxx_entry_val", "soxx_exit_val", "soxx_peak_val",
                 "soxx_mdd", "soxx_earning", ticker)
        print(div)
        _component_summary("SOXX", all_t, "soxx_entry_val", "soxx_exit_val", "soxx_earning")


def print_sell_proximity(df: pd.DataFrame, open_trade: dict | None) -> None:
    if open_trade is None:
        return

    last         = df.iloc[-1]
    last_date    = df.index[-1]
    lookback_idx = max(0, len(df) - 1 - DIVERGENCE_WINDOW)
    past         = df.iloc[lookback_idx]

    price_rise_pct = (last["price"] - past["price"]) / past["price"] * 100
    breadth_fall   = past["breadth"] - last["breadth"]
    cap_ok         = last["breadth"] < DIVERGENCE_BREADTH_CAP

    def bar(value: float, threshold: float) -> str:
        ratio  = min(value / threshold, 1.0) if threshold else 1.0
        filled = round(ratio * 20)
        return f"[{'#' * filled}{'-' * (20 - filled)}] {ratio:.0%}"

    price_met   = price_rise_pct >= DIVERGENCE_PRICE_RISE
    breadth_met = breadth_fall   >= DIVERGENCE_BREADTH_FALL
    all_met     = price_met and breadth_met and cap_ok

    sep = "-" * 72
    print(f"\n-- Sell signal proximity  (as of {last_date.strftime('%Y-%m-%d')}) --\n")
    print(f"  {'Condition':<28} {'Current':>10}  {'Need':>10}  Progress")
    print(f"  {sep}")

    status = "MET" if price_met else f"need +{DIVERGENCE_PRICE_RISE - price_rise_pct:.1f}% more"
    print(f"  {'Price rise (' + str(DIVERGENCE_WINDOW) + 'd)':<28} "
          f"{price_rise_pct:>+9.1f}%  {DIVERGENCE_PRICE_RISE:>9.1f}%  "
          f"{bar(price_rise_pct, DIVERGENCE_PRICE_RISE)}  {status}")

    status = "MET" if breadth_met else f"need {DIVERGENCE_BREADTH_FALL - breadth_fall:.1f} more pts"
    print(f"  {'Breadth200 fall (' + str(DIVERGENCE_WINDOW) + 'd)':<28} "
          f"{breadth_fall:>+9.1f}pt  {DIVERGENCE_BREADTH_FALL:>9.1f}pt  "
          f"{bar(breadth_fall, DIVERGENCE_BREADTH_FALL)}  {status}")

    status = "MET" if cap_ok else f"need {last['breadth'] - DIVERGENCE_BREADTH_CAP:.1f}pt drop"
    print(f"  {'Breadth200 < cap':<28} "
          f"{last['breadth']:>+9.1f}%   {'<' + str(DIVERGENCE_BREADTH_CAP) + '%':>9}   "
          f"{'below cap' if cap_ok else 'ABOVE cap':32}  {status}")

    print(f"  {sep}")
    verdict = "YES -- sell signal ACTIVE" if all_met else "NO  -- not yet triggered"
    print(f"  All 3 conditions met: {verdict}\n")


def plot_results(
    df: pd.DataFrame,
    strategy: pd.Series,
    benchmark: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    initial_capital: float = INITIAL_CAPITAL,
) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 0.8]},
    )
    ax1, ax2, ax3 = axes

    extra = ("" + (f" + SPY {SPY_WEIGHT:.0%}" if SPY_WEIGHT > 0 else "")
                + (f" + SOXX {SOXX_WEIGHT:.0%}" if SOXX_WEIGHT > 0 else ""))
    fig.suptitle(
        f"Portfolio Strategy: QQQ {QQQ_WEIGHT:.0%} + NDX Top-1 Stock {STOCK_WEIGHT:.0%} + TQQQ {TQQQ_WEIGHT:.0%}{extra}\n"
        f"BUY: breadth200 < {BUY_B200_THRESH}%  AND  (VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW})\n"
        f"SELL: price rose >={DIVERGENCE_PRICE_RISE}% over {DIVERGENCE_WINDOW}d  AND  "
        f"breadth200 fell >={DIVERGENCE_BREADTH_FALL}pts  AND  breadth200 < {DIVERGENCE_BREADTH_CAP}%\n"
        f"Starting capital: ${initial_capital:,.0f}",
        fontsize=9, fontweight="bold",
    )

    ax1.plot(benchmark.index, benchmark, label="Buy & Hold NDX", color="#2196F3", linewidth=1.5)
    ax1.plot(strategy.index,  strategy,  label=f"Portfolio ({QQQ_WEIGHT:.0%}/{STOCK_WEIGHT:.0%}/{TQQQ_WEIGHT:.0%})",
             color="#FF5722", linewidth=1.5)

    all_entries = [t["entry_date"] for t in trades] + ([open_trade["entry_date"]] if open_trade else [])
    all_exits   = [t["exit_date"]  for t in trades]

    if all_entries:
        ax1.scatter(all_entries, strategy.reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=80, zorder=5, label="Buy")
    if all_exits:
        ax1.scatter(all_exits, strategy.reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=80, zorder=5, label="Sell")

    ax1.set_ylabel("Portfolio Value ($)")
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

    ax3.plot(df.index, df["price"], color="#546E7A", linewidth=1.0, label="NASDAQ 100")
    ax3.plot(df.index, df["ma200"], color="orange",  linewidth=0.8, linestyle="--",
             label=f"MA{MA200_WINDOW}")
    if all_entries:
        ax3.scatter(all_entries, df["price"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=50, zorder=5)
    if all_exits:
        ax3.scatter(all_exits, df["price"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=50, zorder=5)
    ax3.set_ylabel("NDX")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out = DATA_DIR / "qqq_portfolio_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved -> {out}")


def main() -> None:
    global QQQ_WEIGHT, STOCK_WEIGHT, TQQQ_WEIGHT, SPY_WEIGHT, SOXX_WEIGHT

    parser = argparse.ArgumentParser(
        description="Portfolio backtest: QQQ / NDX Top-1 Stock / TQQQ"
    )
    parser.add_argument("--start-date", type=str, default=START_DATE,
                        metavar="DATE",
                        help="First date to include, ISO format YYYY-MM-DD (default: full history). "
                             "If the strategy would be IN at this date, buys immediately on open.")
    parser.add_argument("--end-date", type=str, default=END_DATE,
                        metavar="DATE",
                        help="Last date to include, ISO format YYYY-MM-DD (default: full history).")
    parser.add_argument("--cooldown-days", type=int, default=COOLDOWN_DAYS,
                        metavar="DAYS",
                        help="Calendar-day cooldown after a sell (default: %(default)s)")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL,
                        metavar="AMOUNT",
                        help="Starting portfolio value in dollars (default: %(default)s)")
    parser.add_argument("--qqq", type=float, default=None,
                        metavar="PCT",
                        help="QQQ allocation %% (default: 60 when no weights given). "
                             "If any weight flag is specified, all unspecified ones default to 0. "
                             "Weights are auto-normalized.")
    parser.add_argument("--stock", type=float, default=None,
                        metavar="PCT",
                        help="Top-1 stock allocation %% (default: 30 when no weights given).")
    parser.add_argument("--tqqq", type=float, default=None,
                        metavar="PCT",
                        help="TQQQ allocation %% (default: 10 when no weights given).")
    parser.add_argument("--spy", type=float, default=None,
                        metavar="PCT",
                        help="SPY allocation %% (default: 0 when no weights given).")
    parser.add_argument("--soxx", type=float, default=None,
                        metavar="PCT",
                        help="SOXX allocation %% (default: 0 when no weights given).")
    parser.add_argument("--monthly-contribution", type=float, default=MONTHLY_CONTRIBUTION,
                        metavar="AMOUNT",
                        help="Cash added every month in dollars (default: %(default)s). "
                             "Contributions accumulate and are deployed at the next buy signal.")
    parser.add_argument("--yearly-contribution", type=float, default=YEARLY_CONTRIBUTION,
                        metavar="AMOUNT",
                        help="Cash added every year in dollars (default: %(default)s). "
                             "Contributions accumulate and are deployed at the next buy signal.")
    parser.add_argument("--fill", choices=["next-open", "next-close", "same-close"],
                        default=None,
                        help="Execution model: next-open (default, realistic), next-close, "
                             "or same-close (legacy same-day look-ahead fill)")
    args = parser.parse_args()

    fill_lag, fill_on = EXECUTION_LAG, FILL_PRICE
    if args.fill == "next-open":
        fill_lag, fill_on = 1, "open"
    elif args.fill == "next-close":
        fill_lag, fill_on = 1, "close"
    elif args.fill == "same-close":
        fill_lag, fill_on = 0, "close"

    # If any weight flag was explicitly given, unspecified ones default to 0.
    # If none were given at all, fall back to module-level defaults.
    _explicit = [args.qqq, args.stock, args.tqqq, args.spy, args.soxx]
    if any(w is not None for w in _explicit):
        args.qqq   = args.qqq   if args.qqq   is not None else 0.0
        args.stock = args.stock if args.stock is not None else 0.0
        args.tqqq  = args.tqqq  if args.tqqq  is not None else 0.0
        args.spy   = args.spy   if args.spy   is not None else 0.0
        args.soxx  = args.soxx  if args.soxx  is not None else 0.0
    else:
        args.qqq   = QQQ_WEIGHT   * 100
        args.stock = STOCK_WEIGHT * 100
        args.tqqq  = TQQQ_WEIGHT  * 100
        args.spy   = SPY_WEIGHT   * 100
        args.soxx  = SOXX_WEIGHT  * 100

    total_w = args.qqq + args.stock + args.tqqq + args.spy + args.soxx
    if total_w <= 0:
        parser.error("Allocation weights must sum to a positive number.")
    QQQ_WEIGHT   = args.qqq   / total_w
    STOCK_WEIGHT = args.stock / total_w
    TQQQ_WEIGHT  = args.tqqq  / total_w
    SPY_WEIGHT   = args.spy   / total_w
    SOXX_WEIGHT  = args.soxx  / total_w

    print("Loading data...")
    df, top_holdings, aligned_stocks, aligned_tqqq, aligned_spy, aligned_soxx = load_data()
    stocks_open, tqqq_open, spy_open, soxx_open = load_open_series(top_holdings, df.index)

    # Parse and validate date range
    start_date = pd.Timestamp(args.start_date) if args.start_date else None
    end_date   = pd.Timestamp(args.end_date)   if args.end_date   else None

    if start_date is not None and start_date < df.index[0]:
        print(f"[warning] --start-date {start_date.date()} is before data start "
              f"{df.index[0].date()}; using data start instead]")
        start_date = None
    if end_date is not None and end_date > df.index[-1]:
        print(f"[warning] --end-date {end_date.date()} is after data end "
              f"{df.index[-1].date()}; using data end instead]")
        end_date = None

    # Determine if the strategy would be IN at start_date (pre-pass on prior data)
    force_entry_on_start = False
    force_ticker: str | None = None
    if start_date is not None:
        df_pre = df[df.index < start_date]
        if not df_pre.empty:
            force_entry_on_start, force_ticker = _position_at_date(
                df_pre, top_holdings, cooldown_days=args.cooldown_days
            )
            if force_entry_on_start:
                print(f"[info] Strategy would be IN at {start_date.date()} "
                      f"(holding {force_ticker or 'QQQ'}). Buying on first row.")

    # Slice the data to the requested range
    if start_date is not None:
        df = df[df.index >= start_date]
    if end_date is not None:
        df = df[df.index <= end_date]

    aligned_stocks = {t: s[s.index.isin(df.index)] for t, s in aligned_stocks.items()}
    if aligned_tqqq is not None:
        aligned_tqqq = aligned_tqqq[aligned_tqqq.index.isin(df.index)]
    if aligned_spy is not None:
        aligned_spy  = aligned_spy[aligned_spy.index.isin(df.index)]
    if aligned_soxx is not None:
        aligned_soxx = aligned_soxx[aligned_soxx.index.isin(df.index)]

    def _etf_status(s: pd.Series | None, w: float) -> str:
        if w == 0:
            return "not included"
        return "available" if s is not None else "unavailable (-> QQQ)"

    print(f"Date range  : {df.index[0].date()} -> {df.index[-1].date()} ({len(df)} trading days)")
    weights_str = (f"QQQ {QQQ_WEIGHT:.0%}  /  NDX top-1 stock {STOCK_WEIGHT:.0%}  /  TQQQ {TQQQ_WEIGHT:.0%}"
                   + (f"  /  SPY {SPY_WEIGHT:.0%}" if SPY_WEIGHT > 0 else "")
                   + (f"  /  SOXX {SOXX_WEIGHT:.0%}" if SOXX_WEIGHT > 0 else ""))
    print(f"Portfolio   : {weights_str}")
    print(f"TQQQ data   : {_etf_status(aligned_tqqq, TQQQ_WEIGHT)}")
    if SPY_WEIGHT > 0:
        print(f"SPY data    : {_etf_status(aligned_spy,  SPY_WEIGHT)}")
    if SOXX_WEIGHT > 0:
        print(f"SOXX data   : {_etf_status(aligned_soxx, SOXX_WEIGHT)}")
    print(f"Buy signal  : breadth200 < {BUY_B200_THRESH}%")
    print(f"Vote gate   : VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW}  (>= 1 of 2 must agree)")
    print(f"Sell signal : price rose >={DIVERGENCE_PRICE_RISE}% AND breadth200 fell >={DIVERGENCE_BREADTH_FALL}pts")
    print(f"              over {DIVERGENCE_WINDOW} days, while breadth200 < {DIVERGENCE_BREADTH_CAP}%")
    print(f"           OR climax top: >={EXT10_PCT:.0f}% above 10d MA + MACD cross (within {CLIMAX_VOTE_WINDOW}d, post-entry)")
    print(f"           OR trailing stop: {TRAILING_STOP_PCT:.0f}% below NDX high since entry")
    print(f"Costs       : ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")
    print(f"Cooldown    : {args.cooldown_days} calendar days after each sell")
    print(f"Capital     : ${args.initial_capital:,.0f}")
    if args.monthly_contribution > 0:
        print(f"Monthly DCA : ${args.monthly_contribution:,.0f}/month")
    if args.yearly_contribution > 0:
        print(f"Yearly DCA  : ${args.yearly_contribution:,.0f}/year")

    benchmark                                    = run_benchmark(df, initial_capital=args.initial_capital)
    strategy, trades, open_trade, total_contrib = run_strategy(
        df, top_holdings, aligned_stocks, aligned_tqqq, aligned_spy, aligned_soxx,
        cooldown_days=args.cooldown_days,
        initial_capital=args.initial_capital,
        monthly_contribution=args.monthly_contribution,
        yearly_contribution=args.yearly_contribution,
        force_entry_on_start=force_entry_on_start,
        force_ticker=force_ticker,
        execution_lag=fill_lag,
        fill_on=fill_on,
        aligned_stocks_open=stocks_open,
        aligned_tqqq_open=tqqq_open,
        aligned_spy_open=spy_open,
        aligned_soxx_open=soxx_open,
    )

    print_metrics(
        compute_metrics(strategy, trades),
        compute_metrics(benchmark),
    )
    if total_contrib > 0:
        total_deployed = args.initial_capital + total_contrib
        final_val      = strategy.iloc[-1]
        net_gain       = final_val - total_deployed
        print(f"\n  Total contributions : ${total_contrib:,.0f}")
        print(f"  Total deployed      : ${total_deployed:,.0f}  (initial ${args.initial_capital:,.0f} + contributions ${total_contrib:,.0f})")
        print(f"  Final value         : ${final_val:,.0f}")
        print(f"  Net gain on capital : {'+'if net_gain>=0 else ''}${net_gain:,.0f}")

    print("\n-- Portfolio trades --")
    print_trades(trades, open_trade)

    print_sell_proximity(df, open_trade)

    plot_results(df, strategy, benchmark, trades, open_trade,
                 initial_capital=args.initial_capital)


if __name__ == "__main__":
    main()
