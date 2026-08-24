"""
S&P 500 tactical backtest — SPX prices with a NASDAQ-100 reference

The legacy breadth strategy uses the same signals as qqq_backtest.py, applied
to the S&P 500:

BUY  (while OUT): either entry path —
                  • Washout: breadth200 < 26% AND at least 1 of 2 vote:
                      · VIX > 30  (fear spike / panic bottom)
                      · price > MA200  (uptrend pullback)
                  • Trend re-entry: price closes back above MA200 (fresh cross),
                    allowed when either the previous exit was a climax-top or
                    price is back above the price at the previous exit.
SELL (while IN):  any of —
                  • Bearish divergence: price rose ≥ 3% over 60 days
                    while breadth200 fell ≥ 20 pts AND breadth200 < 60%
                  • Climax top: within 10 days, price was extended ≥ 5% above
                    its 10-day MA AND MACD(12,26,9) flipped bearish
                  • Trailing stop: price 25% below the high since entry
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from spy_spx_tactical import TacticalConfig, TacticalResult, run_tactical_strategy

DATA_DIR     = Path(__file__).parent
SPX_FILE     = DATA_DIR / "SPX.csv"
NDX_BENCHMARK_FILE = DATA_DIR / "NASDAQ100.csv"
VIX_FILE     = DATA_DIR / "VIX.csv"
BREADTH_FILE = DATA_DIR / "S5TH.csv"
# Continuous daily breadth (2002+) built by build_breadth_daily.py.
# S5TH.csv alone is only daily from 2007 — before that it is bimonthly, which
# corrupts row-based lookback windows (a "60-day" window spans ~10 years).
BREADTH_DAILY_FILE = DATA_DIR / "breadth_daily.csv"
BREADTH_DAILY_MIN  = "2007-01-01"  # fallback cutoff when daily file is absent

# ── Buy thresholds ────────────────────────────────────────────────────────────
BUY_B200_THRESH = 26.0   # breadth200 must be below this
VIX_BUY_THRESH  = 30.0   # VIX vote: fear spike (VIX > 30)
MA200_WINDOW    = 200    # MA200 vote: price above 200-day moving average

# ── Sell — bearish divergence ─────────────────────────────────────────────────
DIVERGENCE_WINDOW       = 60    # trading days lookback
DIVERGENCE_PRICE_RISE   = 3.0   # % price rise over window
DIVERGENCE_BREADTH_FALL = 20.0  # pts breadth200 drop over window
DIVERGENCE_BREADTH_CAP  = 60.0  # breadth200 must be below this

# ── Sell — climax top (extension + momentum break within a window) ───────────
EXT10_PCT           = 5.0   # % above 10-day MA that counts as "extended"
CLIMAX_VOTE_WINDOW  = 10    # days within which both climax signals must fire

# ── Sell — trailing stop ──────────────────────────────────────────────────────
TRAILING_STOP_PCT = 25.0    # % below the high since entry

# ── Execution timing ──────────────────────────────────────────────────────────
# Signals come from end-of-day closes, so the earliest tradeable fill is the NEXT
# session. Default: a signal on day t fills at day t+1's OPEN. Set EXECUTION_LAG=0
# and FILL_PRICE="close" for the legacy same-day-close (look-ahead) fill.
EXECUTION_LAG = 1        # bars between signal and fill (0 = same day, look-ahead)
FILL_PRICE    = "open"   # "open" or "close" of the fill bar

# ── Shared ────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
START_YEAR      = None   # e.g. 2010 to begin on Jan 1 of that year; None = full history
COOLDOWN_DAYS   = 15     # calendar days to wait after a sell before the next buy

# ── SPX tactical challenger ────────────────────────────────────────────
# A standard 12-month trend regime controls a synthetic 3x SPX exposure.  The
# defensive sleeve remains at 0.5x, and a high-volatility gate also cuts to
# 0.5x.  The 1% annual drag is deliberately charged to the entire portfolio.
TACTICAL_TREND_MONTHS = 12
TACTICAL_RISK_ON      = 3.0
TACTICAL_RISK_OFF     = 0.5
TACTICAL_VOL_WINDOW   = 40
TACTICAL_VOL_THRESHOLD = 0.40
TACTICAL_STRESS       = 0.5
TACTICAL_ANNUAL_DRAG  = 0.01


def refresh_data() -> None:
    """Refresh local inputs for CLI runs without causing import-time I/O."""
    try:
        from fetch_investing_data import fetch_all_updates
        fetch_all_updates(verbose=True)
    except Exception as fetch_error:
        print(f"[data fetch skipped: {fetch_error}]")


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


def load_data(price_file: Path = SPX_FILE) -> pd.DataFrame:
    market = pd.read_csv(price_file)
    market["Date"] = pd.to_datetime(market["Date"], format="%m/%d/%Y")
    market.set_index("Date", inplace=True)
    market = market.rename(columns={"Price": "price", "Open": "open"})
    market["price"] = _parse_price(market["price"])
    market["open"]  = _parse_price(market["open"])

    b200 = _load_breadth()

    merged = market[["price", "open"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="left"
    )
    merged.sort_index(inplace=True)

    merged = merged[merged["breadth"].notna()]

    vix = pd.read_csv(VIX_FILE)
    vix.columns = [c.strip().strip('"').lstrip("﻿") for c in vix.columns]
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix.set_index("Date", inplace=True)
    merged = merged.join(_parse_price(vix["Price"]).rename("vix"), how="left")
    merged["vix"] = merged["vix"].ffill()

    merged["ma200"] = merged["price"].rolling(MA200_WINDOW).mean()

    # Vote gate: at least 1 of [VIX > 30, price > MA200]; NaN → True
    merged["vix_vote"]   = merged["vix"].isna() | (merged["vix"] > VIX_BUY_THRESH)
    merged["ma200_vote"] = merged["ma200"].isna() | (merged["price"] > merged["ma200"])
    merged["vote_gate"]  = merged["vix_vote"] | merged["ma200_vote"]

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

    # Trend re-entry: fresh close back above MA200 (see run_strategy for the gate).
    merged["ma200_recross"] = (
        (close > merged["ma200"]) & (close.shift(1) <= merged["ma200"].shift(1))
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
    """Run the QQQ buy/sell rules against the supplied market data.

    Signals are computed on the close and fill ``execution_lag`` bars later.
    A zero-lag fill is only valid at the same close; next-session fills may use
    either the open or close. Mark-to-market values always use the close.
    """
    if fill_on == "open" and execution_lag < 1:
        raise ValueError("fill_on='open' requires execution_lag >= 1 (open precedes close)")

    position   = "OUT"
    eff_entry  = raw_entry = 0.0
    entry_date = None
    trade_low  = trade_high = 0.0
    macd_age   = ext_age = 10**9
    buy_trigger = None
    portfolio  = INITIAL_CAPITAL
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_price: float | None = None
    trades: list[dict] = []
    values: dict = {}

    pending: dict | None = None
    rows = list(df.iterrows())
    n = len(rows)

    def execute_due(i, date, fill_price):
        nonlocal position, eff_entry, raw_entry, entry_date, trade_low, trade_high
        nonlocal macd_age, ext_age, buy_trigger, portfolio, cooldown_until
        nonlocal last_sell_reason, last_exit_price, pending
        if pending is None or pending["fill_at"] != i:
            return False
        if pending["action"] == "BUY" and position == "OUT":
            portfolio -= COMMISSION
            eff_entry  = fill_price * (1 + SLIPPAGE)
            raw_entry  = fill_price
            entry_date = date
            trade_low  = trade_high = fill_price
            macd_age = ext_age = 10**9
            buy_trigger = pending["trigger"]
            position = "IN"
            pending = None
            return True
        if pending["action"] == "SELL" and position == "IN":
            eff_exit  = fill_price * (1 - SLIPPAGE)
            gross_ret = (eff_exit - eff_entry) / eff_entry
            portfolio *= (1 + gross_ret)
            portfolio -= COMMISSION
            cooldown_until   = date + pd.Timedelta(days=cooldown_days)
            last_sell_reason = pending["reason"]
            last_exit_price  = fill_price
            trades.append({
                "entry_date":       entry_date,
                "exit_date":        date,
                "entry_price":      raw_entry,
                "exit_price":       fill_price,
                "return_pct":       gross_ret * 100,
                "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
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
        price        = row["price"]
        breadth      = row["breadth"]
        price_rose   = bool(row["price_rose"])
        breadth_fell = bool(row["breadth_fell"])
        if fill_on == "open" and not pd.isna(row["open"]):
            fill_price = row["open"]
        else:
            fill_price = price

        executed = execute_due(i, date, fill_price)

        if not executed and pending is None:
            if position == "OUT":
                vote_gate   = bool(row["vote_gate"])
                cooldown_ok = cooldown_until is None or date > cooldown_until
                washout_buy = not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate
                # Trend re-entry on a fresh MA200 recross: rejoin the trend when the
                # last exit was a climax-top (a premature froth shakeout) or price is
                # back above the price we last sold at (the market proved the exit
                # premature). Recrosses still below the prior exit stay filtered as
                # failed bounces in a real downtrend.
                recross_ok  = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and price > last_exit_price)
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
                trade_low  = min(trade_low, price)
                trade_high = max(trade_high, price)
                macd_age = 0 if bool(row["macd_cross"]) else macd_age + 1
                ext_age  = 0 if bool(row["ext10"])      else ext_age + 1
                bearish_div = price_rose and breadth_fell and breadth < DIVERGENCE_BREADTH_CAP
                climax      = (macd_age < CLIMAX_VOTE_WINDOW) and (ext_age < CLIMAX_VOTE_WINDOW)
                trail_hit   = price <= trade_high * (1 - TRAILING_STOP_PCT / 100)
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

            executed = execute_due(i, date, fill_price)

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    open_trade = None
    if position == "IN":
        last_price = df["price"].iloc[-1]
        last_date  = df.index[-1]
        eff_last   = last_price * (1 - SLIPPAGE)
        open_trade = {
            "entry_date":       entry_date,
            "entry_price":      raw_entry,
            "current_date":     last_date,
            "current_price":    last_price,
            "return_pct":       (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "accumulated":      portfolio * (eff_last / eff_entry),
            "buy_trigger":      buy_trigger,
        }

    return pd.Series(values, name="strategy"), trades, open_trade


def run_calendar_strategy(
    df: pd.DataFrame,
    source_trades: list[dict],
    source_open_trade: dict | None = None,
    *,
    fill_on: str = FILL_PRICE,
) -> tuple[pd.Series, list[dict], dict | None]:
    """Trade SPX on the exact fill dates produced by the QQQ strategy.

    ``source_trades`` and ``source_open_trade`` supply dates and signal labels
    only. Every fill and mark-to-market value comes from ``df`` so NASDAQ prices
    cannot leak into SPX returns. Exposure is strictly long/cash (0x or 1x).
    """
    if fill_on not in {"open", "close"}:
        raise ValueError("fill_on must be 'open' or 'close'")

    events: dict[pd.Timestamp, tuple[str, dict]] = {}

    def add_event(date: pd.Timestamp, action: str, source: dict) -> None:
        if date not in df.index:
            raise ValueError(f"QQQ {action.lower()} date {date.date()} is absent from SPX data")
        if date in events:
            raise ValueError(f"multiple QQQ calendar events on {date.date()}")
        events[date] = (action, source)

    for source in source_trades:
        add_event(source["entry_date"], "BUY", source)
        add_event(source["exit_date"], "SELL", source)
    if source_open_trade is not None:
        add_event(source_open_trade["entry_date"], "BUY", source_open_trade)

    position = "OUT"
    portfolio = INITIAL_CAPITAL
    eff_entry = raw_entry = 0.0
    entry_date: pd.Timestamp | None = None
    trade_low = trade_high = 0.0
    buy_trigger: str | None = None
    values: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []

    for date, row in df.iterrows():
        price = float(row["price"])
        fill_price = (
            float(row["open"])
            if fill_on == "open" and not pd.isna(row["open"])
            else price
        )
        event = events.get(date)

        if event is not None:
            action, source = event
            if action == "BUY":
                if position != "OUT":
                    raise ValueError(f"QQQ calendar buys while already invested on {date.date()}")
                portfolio -= COMMISSION
                eff_entry = fill_price * (1 + SLIPPAGE)
                raw_entry = fill_price
                entry_date = date
                trade_low = trade_high = fill_price
                buy_trigger = source.get("buy_trigger")
                position = "IN"
            else:
                if position != "IN" or entry_date is None:
                    raise ValueError(f"QQQ calendar sells while out of market on {date.date()}")
                eff_exit = fill_price * (1 - SLIPPAGE)
                gross_return = (eff_exit - eff_entry) / eff_entry
                portfolio *= 1 + gross_return
                portfolio -= COMMISSION
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": raw_entry,
                        "exit_price": fill_price,
                        "return_pct": gross_return * 100,
                        "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
                        "accumulated": portfolio,
                        "buy_trigger": buy_trigger,
                        "sell_reason": source.get("sell_reason", "QQQ-calendar"),
                    }
                )
                position = "OUT"

        elif position == "IN":
            trade_low = min(trade_low, price)
            trade_high = max(trade_high, price)

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    open_trade = None
    if position == "IN" and entry_date is not None:
        last_price = float(df["price"].iloc[-1])
        last_date = df.index[-1]
        eff_last = last_price * (1 - SLIPPAGE)
        open_trade = {
            "entry_date": entry_date,
            "entry_price": raw_entry,
            "current_date": last_date,
            "current_price": last_price,
            "return_pct": (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "accumulated": portfolio * (eff_last / eff_entry),
            "buy_trigger": buy_trigger,
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


def print_metrics(strat: dict, bench: dict, *,
                  strategy_label: str = "Strategy",
                  benchmark_label: str = "Buy & Hold") -> None:
    keys = list(dict.fromkeys(list(strat) + list(bench)))
    col  = 16
    hdr  = f"{'Metric':<22}{strategy_label:>{col}}{benchmark_label:>{col}}"
    sep  = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        print(f"  {k:<20}{strat.get(k, '—'):>{col}}{bench.get(k, '—'):>{col}}")
    print(sep)


def print_trades(trades: list[dict], open_trade: dict | None = None) -> None:
    if not trades and not open_trade:
        print("\nNo completed trades.")
        return
    hdr = (f"\n{'#':>3}  {'Entry':10}  {'Exit':10}  {'Held':>7}  {'Entry $':>9}  {'Exit $':>9}"
           f"  {'Return':>8}  {'Drawdown':>9}  {'Portfolio':>12}  {'Buy trigger':>11}  Sell reason")
    print(hdr)
    print("-" * len(hdr))
    for i, t in enumerate(trades, 1):
        days = (t["exit_date"] - t["entry_date"]).days
        print(
            f"{i:>3}  {t['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{t['exit_date'].strftime('%Y-%m-%d'):10}  {_days_str(days):>7}  "
            f"{t['entry_price']:>9.2f}  {t['exit_price']:>9.2f}  "
            f"{t['return_pct']:>+7.1f}%  {t['max_drawdown_pct']:>+8.1f}%  "
            f"${t['accumulated']:>11,.0f}  {t.get('buy_trigger','—'):>11}  {t.get('sell_reason','—')}"
        )
    if open_trade:
        days = (open_trade["current_date"] - open_trade["entry_date"]).days
        print(
            f"{len(trades)+1:>3}  {open_trade['entry_date'].strftime('%Y-%m-%d'):10}  "
            f"{'(open)':10}  {_days_str(days):>7}  "
            f"{open_trade['entry_price']:>9.2f}  {open_trade['current_price']:>9.2f}  "
            f"{open_trade['return_pct']:>+7.1f}%  {open_trade['max_drawdown_pct']:>+8.1f}%  "
            f"${open_trade['accumulated']:>11,.0f}  {open_trade.get('buy_trigger','—'):>11}  "
            f"still holding (as of {open_trade['current_date'].strftime('%Y-%m-%d')})"
        )


def print_sell_proximity(df: pd.DataFrame, open_trade: dict | None) -> None:
    """Show how close the current position is to triggering the sell signal."""
    if open_trade is None:
        return

    last      = df.iloc[-1]
    last_date = df.index[-1]

    lookback_idx = max(0, len(df) - 1 - DIVERGENCE_WINDOW)
    past         = df.iloc[lookback_idx]

    price_now      = last["price"]
    price_then     = past["price"]
    price_rise_pct = (price_now - price_then) / price_then * 100

    breadth_now  = last["breadth"]
    breadth_then = past["breadth"]
    breadth_fall = breadth_then - breadth_now

    cap_ok = breadth_now < DIVERGENCE_BREADTH_CAP

    def bar(value: float, threshold: float) -> str:
        ratio  = min(value / threshold, 1.0) if threshold != 0 else 1.0
        filled = round(ratio * 20)
        return f"[{'█' * filled}{'░' * (20 - filled)}] {ratio:.0%}"

    price_met   = price_rise_pct >= DIVERGENCE_PRICE_RISE
    breadth_met = breadth_fall   >= DIVERGENCE_BREADTH_FALL
    all_met     = price_met and breadth_met and cap_ok

    sep = "─" * 72
    print(f"\n── Sell signal proximity  (as of {last_date.strftime('%Y-%m-%d')}) ──\n")
    print(f"  {'Condition':<28} {'Current':>10}  {'Need':>10}  Progress")
    print(f"  {sep}")

    status = "✓ MET" if price_met else f"need +{DIVERGENCE_PRICE_RISE - price_rise_pct:.1f}% more"
    print(f"  {'Price rise (' + str(DIVERGENCE_WINDOW) + 'd)':<28} "
          f"{price_rise_pct:>+9.1f}%  {DIVERGENCE_PRICE_RISE:>9.1f}%  "
          f"{bar(price_rise_pct, DIVERGENCE_PRICE_RISE)}  {status}")

    status = "✓ MET" if breadth_met else f"need {DIVERGENCE_BREADTH_FALL - breadth_fall:.1f} more pts"
    print(f"  {'Breadth200 fall (' + str(DIVERGENCE_WINDOW) + 'd)':<28} "
          f"{breadth_fall:>+9.1f}pt  {DIVERGENCE_BREADTH_FALL:>9.1f}pt  "
          f"{bar(breadth_fall, DIVERGENCE_BREADTH_FALL)}  {status}")

    status = "✓ MET" if cap_ok else f"need {breadth_now - DIVERGENCE_BREADTH_CAP:.1f}pt drop"
    print(f"  {'Breadth200 < cap':<28} "
          f"{breadth_now:>+9.1f}%   {'<' + str(DIVERGENCE_BREADTH_CAP) + '%':>9}   "
          f"{'✓ below cap' if cap_ok else '✗ above cap':32}  {status}")

    print(f"  {sep}")
    verdict = "YES — sell signal ACTIVE" if all_met else "NO  — not yet triggered"
    print(f"  All 3 conditions met: {verdict}\n")


def plot_results(
    df,
    strategy,
    benchmark,
    trades,
    open_trade,
    *,
    title: str | None = None,
    benchmark_label: str = "Buy & Hold S&P 500",
) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 0.8]}
    )
    ax1, ax2, ax3 = axes

    if title is None:
        title = (
            "S&P 500 Breadth Strategy  (+Voting Gate +Trend Re-entry)\n"
            f"BUY: breadth200 < {BUY_B200_THRESH}% AND "
            f"(VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW})"
            f" OR price re-crosses above MA{MA200_WINDOW} after a qualifying exit\n"
            f"SELL: divergence (price ≥{DIVERGENCE_PRICE_RISE}%/{DIVERGENCE_WINDOW}d + "
            f"breadth200 -{DIVERGENCE_BREADTH_FALL}pts < {DIVERGENCE_BREADTH_CAP}%) OR "
            f"climax top OR trailing stop ({TRAILING_STOP_PCT:.0f}%)\n"
            f"Starting capital: ${INITIAL_CAPITAL:,.0f}"
        )
    fig.suptitle(title, fontsize=9, fontweight="bold")

    ax1.plot(benchmark.index, benchmark, label=benchmark_label,
             color="#2196F3", linewidth=1.5)
    ax1.plot(strategy.index,  strategy,  label="Strategy", color="#FF5722", linewidth=1.5)

    all_entries = [t["entry_date"] for t in trades] + (
        [open_trade["entry_date"]] if open_trade else [])
    all_exits = [t["exit_date"] for t in trades]
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

    ax3.plot(df.index, df["price"], color="#546E7A", linewidth=1.0, label="S&P 500")
    ax3.plot(df.index, df["ma200"], color="orange", linewidth=0.8,
             linestyle="--", label=f"MA{MA200_WINDOW}")
    if all_entries:
        ax3.scatter(all_entries, df["price"].reindex(all_entries, method="nearest"),
                    marker="^", color="green", s=50, zorder=5)
    if all_exits:
        ax3.scatter(all_exits, df["price"].reindex(all_exits, method="nearest"),
                    marker="v", color="red", s=50, zorder=5)
    ax3.set_ylabel("S&P 500")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out = DATA_DIR / "spy_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def print_tactical_rebalances(result: TacticalResult, limit: int = 12) -> None:
    print(f"\n── Tactical exposure changes ({len(result.rebalances)} total; latest {limit}) ──")
    print(f"  {'Date':10}  {'From':>6}  {'To':>6}  {'Turnover':>8}  {'SPX open':>10}")
    for item in result.rebalances[-limit:]:
        print(
            f"  {item['date'].strftime('%Y-%m-%d'):10}  "
            f"{item['from_exposure']:>5.2f}x  {item['to_exposure']:>5.2f}x  "
            f"{item['turnover']:>7.2f}x  {item['open_price']:>10.2f}"
        )


def plot_tactical_results(df: pd.DataFrame, result: TacticalResult,
                          ndx_reference: pd.Series) -> None:
    fig, axes = plt.subplots(
        3, 1, figsize=(16, 12), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.5]},
    )
    ax1, ax2, ax3 = axes
    fig.suptitle(
        "SPX Tactical Trend Strategy vs NASDAQ-100 Breadth Strategy\n"
        f"12-month trend: {TACTICAL_RISK_ON:.1f}x risk-on / "
        f"{TACTICAL_RISK_OFF:.1f}x defensive; volatility stress at "
        f"{TACTICAL_VOL_THRESHOLD:.0%}; annual drag {TACTICAL_ANNUAL_DRAG:.1%}",
        fontsize=10, fontweight="bold",
    )
    ax1.plot(result.equity.index, result.equity, color="#FF5722", linewidth=1.4,
             label="SPX tactical challenger")
    ax1.plot(ndx_reference.index, ndx_reference, color="#2196F3", linewidth=1.3,
             label="NASDAQ-100 breadth baseline")
    ax1.set_yscale("log")
    ax1.set_ylabel("Portfolio value ($, log)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.step(result.exposure.index, result.exposure, where="post", color="#7B1FA2")
    ax2.axhline(TACTICAL_RISK_ON, color="green", linestyle="--", linewidth=0.8)
    ax2.axhline(TACTICAL_RISK_OFF, color="orange", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("SPX exposure")
    ax2.grid(True, alpha=0.3)

    ax3.plot(df.index, df["price"], color="#546E7A", linewidth=1.0, label="S&P 500")
    risk_on = result.trend_regime.reindex(df.index).fillna(False)
    ax3.fill_between(df.index, df["price"].min(), df["price"], where=risk_on,
                     color="green", alpha=0.08, label="12-month risk-on")
    ax3.set_ylabel("S&P 500")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    fig.autofmt_xdate()

    out = DATA_DIR / "spy_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="S&P 500 strategy backtest")
    parser.add_argument(
        "--strategy",
        choices=["qqq-calendar", "tactical", "breadth"],
        default="qqq-calendar",
        help=(
            "trade SPX on QQQ fill dates (default), use the tactical challenger, "
            "or use the legacy SPX breadth strategy"
        ),
    )
    parser.add_argument("--start-year", type=int, default=START_YEAR,
                        metavar="YEAR", help="First year to include (default: full history)")
    parser.add_argument("--cooldown-days", type=int, default=COOLDOWN_DAYS,
                        metavar="DAYS", help="Calendar-day cooldown after a sell (default: %(default)s)")
    parser.add_argument("--fill", choices=["next-open", "next-close", "same-close"],
                        default=None,
                        help="Execution model: next-open (default), next-close, or same-close")
    parser.add_argument("--trend-months", type=int, default=TACTICAL_TREND_MONTHS,
                        metavar="MONTHS", help="Monthly trend lookback for tactical mode")
    parser.add_argument("--risk-on-exposure", type=float, default=TACTICAL_RISK_ON,
                        metavar="MULTIPLE", help="SPX exposure in the tactical risk-on regime")
    parser.add_argument("--annual-drag", type=float, default=TACTICAL_ANNUAL_DRAG,
                        metavar="RATE", help="Annual portfolio drag in tactical mode")
    args = parser.parse_args()

    lag, fill_on = EXECUTION_LAG, FILL_PRICE
    if args.fill == "next-open":
        lag, fill_on = 1, "open"
    elif args.fill == "next-close":
        lag, fill_on = 1, "close"
    elif args.fill == "same-close":
        lag, fill_on = 0, "close"
    fill_desc = {("1", "open"): "next trading day's OPEN (realistic)",
                 ("1", "close"): "next trading day's CLOSE",
                 ("0", "close"): "same-day CLOSE (legacy look-ahead)"}.get(
                    (str(lag), fill_on), f"lag {lag} / {fill_on}")

    refresh_data()
    print("Loading data...")
    df = load_data()
    if args.start_year is not None:
        df = df[df.index.year >= args.start_year]
    print(f"Date range  : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")

    if args.strategy == "qqq-calendar":
        ndx_df = load_data(NDX_BENCHMARK_FILE)
        if args.start_year is not None:
            ndx_df = ndx_df[ndx_df.index.year >= args.start_year]
        qqq_equity, qqq_trades, qqq_open_trade = run_strategy(
            ndx_df,
            cooldown_days=args.cooldown_days,
            execution_lag=lag,
            fill_on=fill_on,
        )
        strategy, trades, open_trade = run_calendar_strategy(
            df,
            qqq_trades,
            qqq_open_trade,
            fill_on=fill_on,
        )
        print("Strategy    : trade SPX on the exact QQQ strategy fill dates")
        print(f"Source      : NASDAQ-100 signals ({len(qqq_trades)} completed trades)")
        print("Exposure    : long/cash only (maximum 1.00x)")
        print(f"Execution   : fill at {fill_desc}")
        print(f"Costs       : ${COMMISSION:.0f} commission + "
              f"{SLIPPAGE*100:.2f}% slippage per side")
        print_metrics(
            compute_metrics(strategy, trades),
            compute_metrics(qqq_equity, qqq_trades),
            strategy_label="SPX calendar",
            benchmark_label="QQQ source",
        )
        print("\n── SPX trades copied from the QQQ fill calendar ──")
        print_trades(trades, open_trade)
        plot_results(
            df,
            strategy,
            qqq_equity,
            trades,
            open_trade,
            title=(
                "SPX traded on the exact QQQ strategy fill dates\n"
                "QQQ signals determine dates; SPX open/close prices determine all returns\n"
                f"Starting capital: ${INITIAL_CAPITAL:,.0f}"
            ),
            benchmark_label="QQQ source strategy",
        )
        return

    if args.strategy == "tactical":
        config = TacticalConfig(
            trend_months=args.trend_months,
            risk_on_exposure=args.risk_on_exposure,
            risk_off_exposure=TACTICAL_RISK_OFF,
            volatility_window=TACTICAL_VOL_WINDOW,
            volatility_threshold=TACTICAL_VOL_THRESHOLD,
            stress_exposure=TACTICAL_STRESS,
            annual_drag=args.annual_drag,
            initial_capital=INITIAL_CAPITAL,
            slippage=SLIPPAGE,
            commission=COMMISSION,
        )
        print(f"Strategy    : {config.trend_months}-month SPX tactical trend")
        print(f"Exposure    : {config.risk_on_exposure:.2f}x risk-on / "
              f"{config.risk_off_exposure:.2f}x defensive")
        print(f"Volatility  : reduce to {config.stress_exposure:.2f}x above "
              f"{config.volatility_threshold:.0%} ({config.volatility_window}d)")
        print(f"Annual drag : {config.annual_drag:.2%}")
        print("Execution   : prior-close signal, rebalance next trading day's OPEN")

        result = run_tactical_strategy(df, config)
        ndx_df = load_data(NDX_BENCHMARK_FILE)
        if args.start_year is not None:
            ndx_df = ndx_df[ndx_df.index.year >= args.start_year]
        ndx_reference, ndx_trades, _ = run_strategy(
            ndx_df,
            cooldown_days=args.cooldown_days,
            execution_lag=lag,
            fill_on=fill_on,
        )
        tactical_metrics = compute_metrics(result.equity)
        tactical_metrics["# Rebalances"] = str(len(result.rebalances))
        tactical_metrics["Average Exposure"] = f"{result.exposure.mean():.2f}x"
        ndx_metrics = compute_metrics(ndx_reference, ndx_trades)
        print_metrics(
            tactical_metrics,
            ndx_metrics,
            strategy_label="SPX tactical",
            benchmark_label="NASDAQ base",
        )
        spx_buy_hold = run_benchmark(df)
        margin = result.equity.iloc[-1] / ndx_reference.iloc[-1] - 1.0
        print(f"\nSPX buy & hold final : ${spx_buy_hold.iloc[-1]:,.0f}")
        print(f"NASDAQ target final  : ${ndx_reference.iloc[-1]:,.0f}")
        print(f"SPX tactical final   : ${result.equity.iloc[-1]:,.0f} ({margin:+.1%} vs target)")
        print_tactical_rebalances(result)
        plot_tactical_results(df, result, ndx_reference)
        return

    print(f"Buy signal  : breadth200 < {BUY_B200_THRESH}%  (washout entry)")
    print(f"Vote gate   : VIX > {VIX_BUY_THRESH} OR price > MA{MA200_WINDOW}  (≥1 of 2 must agree)")
    print(f"           OR trend re-entry: price re-crosses above MA{MA200_WINDOW} after a climax-top exit")
    print("              or when it re-crosses back above the prior exit price")
    print(f"Sell signal : price rose ≥{DIVERGENCE_PRICE_RISE}% AND breadth200 fell ≥{DIVERGENCE_BREADTH_FALL}pts")
    print(f"              over {DIVERGENCE_WINDOW} days, while breadth200 < {DIVERGENCE_BREADTH_CAP}%")
    print(f"           OR climax top: ≥{EXT10_PCT:.0f}% above 10d MA + MACD bearish cross "
          f"(within {CLIMAX_VOTE_WINDOW}d)")
    print(f"           OR trailing stop: {TRAILING_STOP_PCT:.0f}% below high since entry")
    print(f"Costs       : ${COMMISSION:.0f} commission + {SLIPPAGE*100:.2f}% slippage per side")
    print(f"Cooldown    : {args.cooldown_days} calendar days after each sell")
    print(f"Execution   : fill at {fill_desc}")

    benchmark                    = run_benchmark(df)
    strategy, trades, open_trade = run_strategy(
        df, cooldown_days=args.cooldown_days, execution_lag=lag, fill_on=fill_on
    )

    print_metrics(
        compute_metrics(strategy, trades),
        compute_metrics(benchmark),
    )

    print("\n── Strategy trades ──")
    print_trades(trades, open_trade)

    print_sell_proximity(df, open_trade)

    plot_results(df, strategy, benchmark, trades, open_trade)


if __name__ == "__main__":
    main()
