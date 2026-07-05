"""
TQQQ Breadth Strategy — same signals as QQQ backtest, applied to 3× leveraged TQQQ.

BUY  (while OUT): breadth200 < 26%
                  AND at least 1 of 2 vote:
                    • VIX > 30  (fear spike / panic bottom)
                    • price > MA200  (uptrend pullback — safe to buy immediately)
SELL (while IN):  any of —
                  • Bearish divergence: price rose ≥ 3% over 60 days
                    while breadth200 fell ≥ 20 pts AND breadth200 < 60%
                  • Climax top: within 10 days, price extended ≥ 5% above its
                    10-day MA AND MACD(12,26,9) flipped bearish (post-entry)
                  • Trailing stop: price 25% below the high since entry

Price data fetched from yfinance (TQQQ, since its 2010-02-11 inception). Before
inception the series is SIMULATED back to 2002 from NASDAQ100.csv: LEVERAGE× the
NDX daily close-to-close return minus a constant daily drag (expense ratio +
financing cost), calibrated on the actual-TQQQ/NDX overlap — same splice idea as
build_breadth_daily.py. Synthetic opens use LEVERAGE× the NDX overnight gap.
Breadth data from S5TH.csv (S&P 500 % above 200-day MA).
VIX data from VIX.csv.

Comparison chart saved as tqqq_vs_qqq_performance.png.
"""
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from pathlib import Path

DATA_DIR     = Path(__file__).parent
BREADTH_FILE = DATA_DIR / "S5TH.csv"
# Continuous daily breadth (2002+) built by build_breadth_daily.py.
# S5TH.csv alone is only daily from 2007 — before that it is bimonthly, which
# corrupts row-based lookback windows (a "60-day" window spans ~10 years).
BREADTH_DAILY_FILE = DATA_DIR / "breadth_daily.csv"
BREADTH_DAILY_MIN  = "2007-01-01"  # fallback cutoff when daily file is absent
NDX_FILE     = DATA_DIR / "NASDAQ100.csv"   # used for QQQ comparison
VIX_FILE     = DATA_DIR / "VIX.csv"

# ── Pre-inception simulation ──────────────────────────────────────────────────
# TQQQ only exists since 2010-02-11. Earlier history is simulated from NDX:
# sim_ret = LEVERAGE × ndx_ret − drag, where the constant daily drag (expense
# ratio + financing cost of the 2× borrowed exposure) is calibrated on the
# actual-TQQQ/NDX overlap.
LEVERAGE       = 3.0
TQQQ_INCEPTION = "2010-02-11"

# ── Buy thresholds ────────────────────────────────────────────────────────────
BUY_B200_THRESH = 26.0   # breadth200 must be below this
VIX_BUY_THRESH  = 30.0   # VIX vote: fear spike (VIX > 30)
MA200_WINDOW    = 200     # MA200 vote: price above 200-day moving average

# ── Sell — bearish divergence ─────────────────────────────────────────────────
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0

# ── Sell — climax top (extension + momentum break within a window) ───────────
EXT10_PCT           = 5.0   # % above 10-day MA that counts as "extended"
CLIMAX_VOTE_WINDOW  = 10    # days within which both climax signals must fire

# ── Sell — trailing stop ──────────────────────────────────────────────────────
TRAILING_STOP_PCT = 25.0    # % below the high since entry

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day closes (on NDX), so the earliest tradeable fill is
# the NEXT session. Default: a signal on day t fills at day t+1's OPEN of the
# EXECUTION asset (TQQQ, or NDX on the QQQ leg). Set EXECUTION_LAG=0 and
# FILL_PRICE="close" for the legacy same-day-close (look-ahead) fill.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
START_YEAR      = None   # e.g. 2015 to begin backtest on Jan 1 of that year; None = full history
COOLDOWN_DAYS   = 15     # calendar days to wait after a sell before the next buy is allowed


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


def _load_vix() -> pd.Series:
    vix = pd.read_csv(VIX_FILE)
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix.set_index("Date", inplace=True)
    return _parse_price(vix["Price"]).rename("vix")


def _load_ndx() -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx.sort_index(inplace=True)
    return pd.DataFrame({
        "ndx_price": _parse_price(ndx["Price"]),
        "ndx_open":  _parse_price(ndx["Open"]),
    })


def _simulate_pre_inception(
    actual_close: pd.Series, ndx: pd.DataFrame
) -> tuple[pd.Series, pd.Series, float]:
    """Extend TQQQ before its inception with LEVERAGE× daily NDX returns minus a
    constant daily drag, scaled so the simulated path lands exactly on the first
    actual TQQQ close. Returns (sim_close, sim_open, daily_drag).

    The drag (expense ratio + financing cost) is the mean daily shortfall of
    actual TQQQ returns vs LEVERAGE× NDX returns over the post-inception
    overlap — the same fit-on-overlap splice idea as build_breadth_daily.py.
    Synthetic opens apply LEVERAGE× the NDX overnight gap to the prior
    simulated close, so next-open fills stay meaningful in the pre-2010 era.
    """
    first_date  = actual_close.index[0]
    first_close = float(actual_close.iloc[0])

    ndx_ret = ndx["ndx_price"].pct_change()
    overlap = pd.concat(
        [actual_close.pct_change().rename("tqqq"), ndx_ret.rename("ndx")], axis=1
    ).dropna()
    if overlap.empty:
        raise ValueError("no TQQQ/NDX overlap to calibrate the simulation drag")
    drag = float((LEVERAGE * overlap["ndx"] - overlap["tqqq"]).mean())
    corr = float((LEVERAGE * overlap["ndx"]).corr(overlap["tqqq"]))

    pre = ndx[ndx.index < first_date]
    sim_ret = (LEVERAGE * pre["ndx_price"].pct_change() - drag).fillna(0.0)
    worst = float(sim_ret.min())
    if worst <= -1.0:
        raise ValueError(f"simulated daily return {worst:.1%} wipes out the fund")

    # Grow an unscaled path forward, then scale it so that compounding through
    # the boundary return (last pre-inception close → first actual close) lands
    # exactly on the first real TQQQ print.
    cum = (1.0 + sim_ret).cumprod()
    boundary_ret = LEVERAGE * float(ndx_ret.get(first_date, 0.0)) - drag
    scale = first_close / (float(cum.iloc[-1]) * (1.0 + boundary_ret))
    sim_close = (cum * scale).rename("price")

    gap = ndx["ndx_open"] / ndx["ndx_price"].shift(1) - 1.0
    sim_open = (sim_close.shift(1) * (1.0 + LEVERAGE * gap.reindex(sim_close.index))).rename("open")

    print(f"Simulated TQQQ {sim_close.index[0].date()} → {sim_close.index[-1].date()} "
          f"from {LEVERAGE:.0f}× NDX daily returns "
          f"(drag {drag*252:.2%}/yr, overlap corr {corr:.4f}, worst day {worst:.1%})")
    return sim_close, sim_open, drag


def load_tqqq_data() -> pd.DataFrame:
    print("Fetching TQQQ from yfinance…")
    raw = yf.download("TQQQ", start="2010-01-01", progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    tqqq = close.rename("price")
    tqqq.index = pd.to_datetime(tqqq.index)
    tqqq.index.name = "Date"
    # TQQQ's own open is the fill price when execution lags to the next session.
    tqqq_open = raw["Open"]
    if isinstance(tqqq_open, pd.DataFrame):
        tqqq_open = tqqq_open.iloc[:, 0]
    tqqq_open = tqqq_open.rename("open")
    tqqq_open.index = pd.to_datetime(tqqq_open.index)
    tqqq_open.index.name = "Date"

    b200 = _load_breadth()

    ndx = _load_ndx()
    ndx_price = ndx["ndx_price"]

    # Splice: simulated 3× NDX before inception, actual TQQQ after.
    sim_close, sim_open, _ = _simulate_pre_inception(tqqq, ndx)
    tqqq      = pd.concat([sim_close, tqqq])
    tqqq_open = pd.concat([sim_open, tqqq_open])

    merged = tqqq.to_frame().join(tqqq_open, how="left").join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged = merged.join(_load_vix(), how="left")
    merged = merged.join(ndx_price, how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]

    merged["vix"]        = merged["vix"].ffill()
    merged["ndx_price"]  = merged["ndx_price"].ffill()
    merged["ma200"]      = merged["ndx_price"].rolling(MA200_WINDOW).mean()

    merged["vix_vote"]   = merged["vix"].apply(
        lambda v: True if pd.isna(v) else v > VIX_BUY_THRESH)
    merged["ma200_vote"] = merged.apply(
        lambda r: True if pd.isna(r["ma200"]) else r["ndx_price"] > r["ma200"], axis=1)
    merged["vote_gate"]  = merged["vix_vote"] | merged["ma200_vote"]

    # ALL signals are computed on the UNDERLYING INDEX (NDX), not on TQQQ's own
    # 3× price series — the strategy's thresholds are calibrated to index
    # volatility, and on TQQQ they would fire constantly. Execution stays in
    # TQQQ ("price"), decisions come from NDX ("signal_price").
    merged["signal_price"] = merged["ndx_price"]
    pp = merged["ndx_price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["ndx_price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)

    # Climax-top components on NDX (exit fires only when both occur post-entry,
    # within CLIMAX_VOTE_WINDOW days — tracked in run_strategy)
    close = merged["ndx_price"]
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    merged["macd_cross"] = ((hist < 0) & (hist.shift(1) >= 0)).fillna(False)
    merged["ext10"] = (close / close.rolling(10).mean() - 1 >= EXT10_PCT / 100).fillna(False)

    # Trend re-entry: fresh close back above MA200 on the signal series (NDX).
    merged["ma200_recross"] = (
        (merged["signal_price"] > merged["ma200"])
        & (merged["signal_price"].shift(1) <= merged["ma200"].shift(1))
    ).fillna(False)

    return merged


def load_qqq_data() -> pd.DataFrame:
    """Load NDX data (proxy for QQQ) + breadth + VIX, same pipeline as qqq_backtest.py."""
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price", "Open": "open"})
    ndx["price"] = _parse_price(ndx["price"])
    ndx["open"]  = _parse_price(ndx["open"])

    b200 = _load_breadth()

    merged = ndx[["price", "open"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged = merged.join(_load_vix(), how="left")
    merged.sort_index(inplace=True)
    merged = merged[merged["breadth"].notna()]

    merged["vix"]   = merged["vix"].ffill()
    merged["ma200"] = merged["price"].rolling(MA200_WINDOW).mean()

    merged["vix_vote"]   = merged["vix"].apply(
        lambda v: True if pd.isna(v) else v > VIX_BUY_THRESH)
    merged["ma200_vote"] = merged.apply(
        lambda r: True if pd.isna(r["ma200"]) else r["price"] > r["ma200"], axis=1)
    merged["vote_gate"]  = merged["vix_vote"] | merged["ma200_vote"]

    merged["signal_price"] = merged["price"]  # QQQ leg: signals on its own (index) price
    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)

    # Climax-top components (exit fires only when both occur post-entry,
    # within CLIMAX_VOTE_WINDOW days — tracked in run_strategy)
    close = merged["price"]
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    merged["macd_cross"] = ((hist < 0) & (hist.shift(1) >= 0)).fillna(False)
    merged["ext10"] = (close / close.rolling(10).mean() - 1 >= EXT10_PCT / 100).fillna(False)

    # Trend re-entry: fresh close back above MA200 on the signal series (NDX).
    merged["ma200_recross"] = (
        (merged["signal_price"] > merged["ma200"])
        & (merged["signal_price"].shift(1) <= merged["ma200"].shift(1))
    ).fillna(False)

    return merged


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


def run_strategy(df: pd.DataFrame, cooldown_days: int = 0,
                 execution_lag: int = EXECUTION_LAG,
                 fill_on: str = FILL_PRICE) -> tuple[pd.Series, list[dict], dict | None]:
    """Signals are close-based (on NDX signal_price); a signal on day t fills
    `execution_lag` bars later at the EXECUTION asset's open (fill_on="open") or
    close. lag=0 requires fill_on="close" (legacy same-day look-ahead). Mark-to-
    market always uses the execution close ("price")."""
    if fill_on == "open" and execution_lag < 1:
        raise ValueError("fill_on='open' requires execution_lag >= 1 (open precedes close)")

    position           = "OUT"
    eff_entry          = raw_entry = 0.0
    entry_date         = None
    trade_low          = trade_high = 0.0
    macd_age = ext_age = 10**9
    buy_trigger        = ""
    portfolio          = INITIAL_CAPITAL
    port_peak          = INITIAL_CAPITAL
    trade_port_peak    = INITIAL_CAPITAL
    trade_port_low     = 0.0
    trade_port_trough_val = INITIAL_CAPITAL
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_price: float | None = None
    trades: list[dict] = []
    values: dict = {}

    pending: dict | None = None
    rows = list(df.iterrows())
    n = len(rows)

    def execute_due(i, date, fill_price, sig_price):
        nonlocal position, eff_entry, raw_entry, entry_date, trade_low, trade_high
        nonlocal macd_age, ext_age, buy_trigger, portfolio, port_peak
        nonlocal trade_port_peak, trade_port_low, trade_port_trough_val
        nonlocal cooldown_until, last_sell_reason, last_exit_price, pending
        if pending is None or pending["fill_at"] != i:
            return False
        if pending["action"] == "BUY" and position == "OUT":
            portfolio -= COMMISSION
            eff_entry  = fill_price * (1 + SLIPPAGE)
            raw_entry  = fill_price
            entry_date = date
            trade_low  = fill_price
            trade_high = sig_price          # trail tracks the signal price (NDX)
            macd_age = ext_age = 10**9
            trade_port_peak       = portfolio
            trade_port_low        = 0.0
            trade_port_trough_val = portfolio
            buy_trigger = pending["trigger"]
            position = "IN"
            pending = None
            return True
        if pending["action"] == "SELL" and position == "IN":
            eff_exit  = fill_price * (1 - SLIPPAGE)
            gross_ret = (eff_exit - eff_entry) / eff_entry
            portfolio *= (1 + gross_ret)
            portfolio -= COMMISSION
            port_peak = max(port_peak, portfolio)
            cooldown_until   = date + pd.Timedelta(days=cooldown_days)
            last_sell_reason = pending["reason"]
            last_exit_price  = sig_price
            trades.append({
                "entry_date":       entry_date,
                "exit_date":        date,
                "entry_price":      raw_entry,
                "exit_price":       fill_price,
                "return_pct":       gross_ret * 100,
                "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
                "port_dd_pct":      trade_port_low,
                "port_peak":        trade_port_peak,
                "port_trough":      trade_port_trough_val,
                "accumulated":      portfolio,
                "buy_trigger":      buy_trigger,
                "sell_reason":      pending["reason"],
                "cooldown_until":   cooldown_until,
            })
            position = "OUT"
            pending = None
            return True
        pending = None
        return False

    for i in range(n):
        date, row = rows[i]
        price        = row["price"]          # execution close: fills (legacy) + MTM
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])
        sig_price    = row["signal_price"]   # NDX: drives all signals
        if fill_on == "open" and not pd.isna(row["open"]):
            fill_price = row["open"]         # execution asset's open
        else:
            fill_price = price

        executed = execute_due(i, date, fill_price, sig_price)

        if not executed and pending is None:
            if position == "OUT":
                vote_gate   = bool(row["vote_gate"])
                cooldown_ok = cooldown_until is None or date > cooldown_until
                washout_buy = not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate
                # Trend re-entry on a fresh MA200 recross (NDX): rejoin when the last
                # exit was a climax-top or the signal price is back above the price we
                # last sold at (market proved the exit premature).
                recross_ok  = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and sig_price > last_exit_price)
                trend_buy   = bool(row["ma200_recross"]) and recross_ok
                do_buy = cooldown_ok and (washout_buy or trend_buy)
                if do_buy and i + execution_lag < n:
                    trigger = (("VIX" if row["vix_vote"] else "") +
                               ("+" if row["vix_vote"] and row["ma200_vote"] else "") +
                               ("MA200" if row["ma200_vote"] else ""))
                    pending = {"action": "BUY", "fill_at": i + execution_lag,
                               "trigger": trigger}

            elif position == "IN":
                trade_low = min(trade_low, price)
                cur_port_val = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
                # Update running peak first, then measure trough relative to current peak
                trade_port_peak = max(trade_port_peak, cur_port_val)
                cur_dd = (cur_port_val - trade_port_peak) / trade_port_peak * 100
                if cur_dd < trade_port_low:   # trade_port_low reused as worst_dd_pct
                    trade_port_low = cur_dd
                    trade_port_trough_val = cur_port_val

                trade_high = max(trade_high, sig_price)
                macd_age = 0 if bool(row["macd_cross"]) else macd_age + 1
                ext_age  = 0 if bool(row["ext10"])      else ext_age + 1
                bearish_div = price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP
                climax      = (macd_age < CLIMAX_VOTE_WINDOW) and (ext_age < CLIMAX_VOTE_WINDOW)
                trail_hit   = sig_price <= trade_high * (1 - TRAILING_STOP_PCT / 100)
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

            executed = execute_due(i, date, fill_price, sig_price)

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    open_trade = None
    if position == "IN":
        last_price   = df["price"].iloc[-1]
        last_date    = df.index[-1]
        eff_last     = last_price * (1 - SLIPPAGE)
        cur_port_val = portfolio * (eff_last / eff_entry)
        trade_port_peak = max(trade_port_peak, cur_port_val)
        cur_dd = (cur_port_val - trade_port_peak) / trade_port_peak * 100
        if cur_dd < trade_port_low:
            trade_port_low = cur_dd
            trade_port_trough_val = cur_port_val
        port_dd = trade_port_low
        open_trade = {
            "entry_date":    entry_date,
            "entry_price":   raw_entry,
            "current_date":  last_date,
            "current_price": last_price,
            "return_pct":    (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "port_dd_pct":   port_dd,
            "port_peak":     trade_port_peak,
            "port_trough":   trade_port_trough_val,
            "accumulated":   cur_port_val,
            "buy_trigger":   buy_trigger,
        }

    return pd.Series(values, name="strategy"), trades, open_trade


def run_benchmark(df: pd.DataFrame) -> pd.Series:
    first = df["price"].iloc[0]
    return (INITIAL_CAPITAL * df["price"] / first).rename("benchmark")


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
            "Win Rate":       f"{wins/n:.1%}" if n else "—",
            "Time in Market": f"{in_days/tot:.1%}" if tot else "—",
        })
    return m


def print_metrics_quad(tqqq_strat: dict, qqq_strat: dict, tqqq_bench: dict, qqq_bench: dict) -> None:
    keys = list(dict.fromkeys(list(tqqq_strat) + list(qqq_strat) + list(tqqq_bench) + list(qqq_bench)))
    col  = 16
    hdr  = f"{'Metric':<22}{'TQQQ Strat':>{col}}{'QQQ Strat':>{col}}{'TQQQ B&H':>{col}}{'QQQ B&H':>{col}}"
    sep  = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        print(f"  {k:<20}{tqqq_strat.get(k,'—'):>{col}}{qqq_strat.get(k,'—'):>{col}}"
              f"{tqqq_bench.get(k,'—'):>{col}}{qqq_bench.get(k,'—'):>{col}}")
    print(sep)


def print_trades(label: str, trades: list[dict], open_trade: dict | None = None) -> None:
    print(f"\n── {label} trades ──")
    if not trades and not open_trade:
        print("  No completed trades.")
        return
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Entry $':>9}  {'Exit $':>9}"
           f"  {'Return':>8}  {'PricDD':>7}  {'PortPeak':>13}  {'PortTrough':>13}  {'PortDD':>7}  {'Portfolio':>13}  {'Buy trigger':>11}  Reason")
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}  "
            f"{t['return_pct']:>+7.1f}%  "
            f"{t['max_drawdown_pct']:>+6.1f}%  "
            f"${t['port_peak']:>12,.0f}  "
            f"${t['port_trough']:>12,.0f}  "
            f"{t['port_dd_pct']:>+6.1f}%  "
            f"${t['accumulated']:>12,.0f}  {t.get('buy_trigger','—'):>11}  {t.get('sell_reason','—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['entry_price']:>9.2f}  {open_trade['current_price']:>9.2f}  "
            f"{open_trade['return_pct']:>+7.1f}%  "
            f"{open_trade['max_drawdown_pct']:>+6.1f}%  "
            f"${open_trade['port_peak']:>12,.0f}  "
            f"${open_trade['port_trough']:>12,.0f}  "
            f"{open_trade['port_dd_pct']:>+6.1f}%  "
            f"${open_trade['accumulated']:>12,.0f}  {open_trade.get('buy_trigger','—'):>11}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def plot_comparison(
    tqqq_df, tqqq_strat, tqqq_bench, tqqq_trades, tqqq_open,
    qqq_df,  qqq_strat,  qqq_bench,  qqq_trades,  qqq_open,
) -> None:
    start = max(tqqq_strat.index[0], qqq_strat.index[0])
    t_s = tqqq_strat[tqqq_strat.index >= start]
    q_s = qqq_strat[qqq_strat.index >= start]
    t_b = tqqq_bench[tqqq_bench.index >= start]
    q_b = qqq_bench[qqq_bench.index >= start]

    def _rebase(s: pd.Series) -> pd.Series:
        return s / s.iloc[0] * INITIAL_CAPITAL

    t_s = _rebase(t_s)
    q_s = _rebase(q_s)
    t_b = _rebase(t_b)
    q_b = _rebase(q_b)

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 13), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 0.8]}
    )
    ax1, ax2, ax3 = axes

    fig.suptitle(
        "TQQQ vs QQQ — Same Breadth Strategy  (+Voting Gate)\n"
        f"BUY: breadth200 < {BUY_B200_THRESH}%  AND  (VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW})  [≥1 of 2]\n"
        f"SELL: price ≥+{DIVERGENCE_PRICE_RISE}% over {DIVERGENCE_WINDOW}d  AND  "
        f"breadth fell ≥{DIVERGENCE_BREADTH_FALL}pts  AND  breadth < {DIVERGENCE_BREADTH_CAP}%\n"
        f"Both rebased to ${INITIAL_CAPITAL:,.0f} at {start.strftime('%Y-%m-%d')}",
        fontsize=9, fontweight="bold"
    )

    ax1.plot(q_b.index, q_b, label="QQQ Buy & Hold",  color="#2196F3", linewidth=1.2, alpha=0.7)
    ax1.plot(t_b.index, t_b, label="TQQQ Buy & Hold", color="#00BCD4", linewidth=1.2, alpha=0.7)
    ax1.plot(q_s.index, q_s, label="QQQ Strategy",    color="#FF9800", linewidth=1.8)
    ax1.plot(t_s.index, t_s, label="TQQQ Strategy",   color="#E91E63", linewidth=1.8)

    def _scatter(ax, trades, open_trade, series, buy_color, sell_color, sz=55):
        entries = [t["entry_date"] for t in trades] + ([open_trade["entry_date"]] if open_trade else [])
        exits   = [t["exit_date"]  for t in trades]
        entries = [d for d in entries if d in series.index]
        exits   = [d for d in exits   if d in series.index]
        if entries:
            ax.scatter(entries, series.reindex(entries, method="nearest"),
                       marker="^", color=buy_color,  s=sz, zorder=5)
        if exits:
            ax.scatter(exits,   series.reindex(exits, method="nearest"),
                       marker="v", color=sell_color, s=sz, zorder=5)

    _scatter(ax1, tqqq_trades, tqqq_open, t_s, "#880E4F", "#880E4F")
    _scatter(ax1, qqq_trades,  qqq_open,  q_s, "#E65100", "#E65100")

    ax1.set_ylabel("Portfolio Value ($, log scale)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    breadth_common = tqqq_df["breadth"][tqqq_df.index >= start]
    ax2.plot(breadth_common.index, breadth_common, color="#7B1FA2", linewidth=1.0,
             label="S&P 500 % Above 200-Day MA")
    ax2.axhline(BUY_B200_THRESH,        color="green", linestyle="--", linewidth=1.0,
                label=f"Buy gate: <{BUY_B200_THRESH}%")
    ax2.axhline(DIVERGENCE_BREADTH_CAP, color="red",   linestyle="--", linewidth=0.9,
                label=f"Sell cap: <{DIVERGENCE_BREADTH_CAP}%")
    ax2.fill_between(breadth_common.index, breadth_common, BUY_B200_THRESH,
                     where=breadth_common < BUY_B200_THRESH, color="green", alpha=0.12)
    ax2.set_ylabel("Breadth (%)")
    ax2.legend(loc="upper left", fontsize=7)
    ax2.grid(True, alpha=0.3)

    tqqq_price = tqqq_df["price"][tqqq_df.index >= start]
    ax3.plot(tqqq_price.index, tqqq_price, color="#546E7A", linewidth=1.0, label="TQQQ")
    ax3.set_ylabel("TQQQ ($)")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", fontsize=7)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out = DATA_DIR / "tqqq_vs_qqq_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="TQQQ vs QQQ Breadth Strategy backtest")
    parser.add_argument("--start-year", type=int, default=START_YEAR,
                        metavar="YEAR", help="First year to include in the backtest (default: full history)")
    parser.add_argument("--cooldown-days", type=int, default=COOLDOWN_DAYS,
                        metavar="DAYS", help="Calendar-day cooldown after a sell before next buy (default: %(default)s)")
    args = parser.parse_args()

    tqqq_df = load_tqqq_data()
    if args.start_year is not None:
        tqqq_df = tqqq_df[tqqq_df.index.year >= args.start_year]
    print(f"TQQQ range  : {tqqq_df.index[0].date()} → {tqqq_df.index[-1].date()} ({len(tqqq_df)} rows)")

    tqqq_bench                         = run_benchmark(tqqq_df)
    tqqq_strat, tqqq_trades, tqqq_open = run_strategy(tqqq_df, cooldown_days=args.cooldown_days)

    qqq_df = load_qqq_data()
    qqq_start = tqqq_df.index[0]
    qqq_df = qqq_df[qqq_df.index >= qqq_start]
    print(f"QQQ range   : {qqq_df.index[0].date()} → {qqq_df.index[-1].date()} ({len(qqq_df)} rows)")

    qqq_bench                        = run_benchmark(qqq_df)
    qqq_strat, qqq_trades, qqq_open  = run_strategy(qqq_df, cooldown_days=args.cooldown_days)

    print(f"\nBuy signal  : breadth200 < {BUY_B200_THRESH}%")
    print(f"Vote gate   : VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW}  (≥1 of 2 must agree)")
    print(f"Sell signal : price rose ≥{DIVERGENCE_PRICE_RISE}% AND breadth200 fell ≥{DIVERGENCE_BREADTH_FALL}pts")
    print(f"              over {DIVERGENCE_WINDOW} days, while breadth200 < {DIVERGENCE_BREADTH_CAP}%")
    print(f"Costs       : ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")
    print(f"Cooldown    : {args.cooldown_days} calendar days after each sell\n")

    print_metrics_quad(
        compute_metrics(tqqq_strat, tqqq_trades),
        compute_metrics(qqq_strat,  qqq_trades),
        compute_metrics(tqqq_bench),
        compute_metrics(qqq_bench),
    )

    print_trades("TQQQ Strategy", tqqq_trades, tqqq_open)
    print_trades("QQQ Strategy",  qqq_trades,  qqq_open)

    plot_comparison(
        tqqq_df, tqqq_strat, tqqq_bench, tqqq_trades, tqqq_open,
        qqq_df,  qqq_strat,  qqq_bench,  qqq_trades,  qqq_open,
    )


if __name__ == "__main__":
    main()
