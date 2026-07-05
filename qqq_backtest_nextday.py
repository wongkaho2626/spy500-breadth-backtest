"""
NASDAQ 100 Breadth Strategy — NEXT-DAY execution study.

The canonical qqq_backtest.py computes each signal from end-of-day data
(breadth, VIX, MA200, MACD, …) and then fills the trade at the SAME day's
close. That is not tradeable: you only know the signal once the session has
closed, so the earliest realistic fill is the NEXT trading day.

This script reuses the identical signal state machine but delays every fill by
one bar — a signal that fires on day t is executed at day t+1's close. It runs
both variants (same-day = baseline, next-day = realistic) and prints them side
by side so the execution-lag cost is explicit.

Signal logic is copied verbatim from qqq_backtest.py (do not re-derive — see
CLAUDE.md). Only the execution timing differs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR     = Path(__file__).parent
NDX_FILE     = DATA_DIR / "NASDAQ100.csv"
BREADTH_FILE = DATA_DIR / "S5TH.csv"
BREADTH_DAILY_FILE = DATA_DIR / "breadth_daily.csv"
BREADTH_DAILY_MIN  = "2007-01-01"
VIX_FILE     = DATA_DIR / "VIX.csv"

# ── Buy thresholds ──
BUY_B200_THRESH = 26.0
VIX_BUY_THRESH  = 30.0
MA200_WINDOW    = 200

# ── Sell — bearish divergence ──
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0

# ── Sell — climax top ──
EXT10_PCT          = 5.0
CLIMAX_VOTE_WINDOW = 10

# ── Sell — trailing stop ──
TRAILING_STOP_PCT = 25.0

# ── Shared ──
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
COOLDOWN_DAYS   = 15


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    ndx = pd.read_csv(NDX_FILE)
    ndx["Date"] = pd.to_datetime(ndx["Date"], format="%m/%d/%Y")
    ndx.set_index("Date", inplace=True)
    ndx = ndx.rename(columns={"Price": "price"})
    ndx["price"] = _parse_price(ndx["price"])

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
        b200 = b200[b200.index >= BREADTH_DAILY_MIN]

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


def _buy_signal(row, date, cooldown_until, last_sell_reason, last_exit_price):
    """Return (do_buy, buy_trigger) for an OUT day — identical rules to canonical."""
    price   = row["price"]
    breadth = row["breadth"]
    vote_gate   = bool(row["vote_gate"])
    cooldown_ok = cooldown_until is None or date > cooldown_until
    washout_buy = not pd.isna(breadth) and breadth < BUY_B200_THRESH and vote_gate
    recross_ok  = last_sell_reason == "climax-top" or (
        last_exit_price is not None and price > last_exit_price)
    trend_buy = bool(row["ma200_recross"]) and recross_ok
    do_buy = cooldown_ok and (washout_buy or trend_buy)
    if not do_buy:
        return False, None
    if washout_buy:
        trigger = (("VIX" if row["vix_vote"] else "") +
                   ("+" if row["vix_vote"] and row["ma200_vote"] else "") +
                   ("MA200" if row["ma200_vote"] else ""))
    else:
        trigger = "MA200-recross"
    return True, trigger


def run_strategy(df: pd.DataFrame, cooldown_days: int = 0,
                 execution_lag: int = 0) -> tuple[pd.Series, list[dict], dict | None]:
    """
    execution_lag = 0 → same-day fill at signal-day close (baseline).
    execution_lag = 1 → next trading day's close (realistic).

    A signal detected on day t is stored as a pending order and filled at the
    close `execution_lag` bars later. All signal computations are unchanged.
    """
    position       = "OUT"
    eff_entry      = raw_entry = 0.0
    entry_date     = None
    trade_low = trade_high = 0.0
    macd_age = ext_age = 10**9
    buy_trigger    = None
    portfolio      = INITIAL_CAPITAL
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_price: float | None = None
    trades: list[dict] = []
    values: dict = {}

    # pending order: dict with keys action ("BUY"/"SELL"), fill_at (row index i),
    # and (for BUY) the trigger captured at signal time / (for SELL) the reason.
    pending: dict | None = None

    rows = list(df.iterrows())
    n = len(rows)

    def execute_due(i, date, price):
        """Fill a pending order whose fill bar is today. Returns True if filled."""
        nonlocal position, eff_entry, raw_entry, entry_date, trade_low, trade_high
        nonlocal macd_age, ext_age, buy_trigger, portfolio, cooldown_until
        nonlocal last_sell_reason, last_exit_price, pending
        if pending is None or pending["fill_at"] != i:
            return False
        if pending["action"] == "BUY" and position == "OUT":
            portfolio -= COMMISSION
            eff_entry  = price * (1 + SLIPPAGE)
            raw_entry  = price
            entry_date = date
            trade_low  = trade_high = price
            macd_age = ext_age = 10**9
            buy_trigger = pending["trigger"]
            position = "IN"
            pending = None
            return True
        if pending["action"] == "SELL" and position == "IN":
            eff_exit  = price * (1 - SLIPPAGE)
            gross_ret = (eff_exit - eff_entry) / eff_entry
            portfolio *= (1 + gross_ret)
            portfolio -= COMMISSION
            cooldown_until   = date + pd.Timedelta(days=cooldown_days)
            last_sell_reason = pending["reason"]
            last_exit_price  = price
            trades.append({
                "entry_date":       entry_date,
                "exit_date":        date,
                "entry_price":      raw_entry,
                "exit_price":       price,
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
        price = row["price"]

        # 1) Execute any order that came due today (carried over from a prior day).
        executed = execute_due(i, date, price)

        # 2) Evaluate signals on today's data → set pending for a future fill.
        #    Skip on a day we just executed a fill, to mirror the canonical loop
        #    (entry day does no sell check; exit day does no buy check).
        if not executed and pending is None:
            if position == "OUT":
                do_buy, trigger = _buy_signal(
                    row, date, cooldown_until, last_sell_reason, last_exit_price)
                if do_buy and i + execution_lag < n:
                    pending = {"action": "BUY", "fill_at": i + execution_lag,
                               "trigger": trigger}
            elif position == "IN":
                trade_low  = min(trade_low, price)
                trade_high = max(trade_high, price)
                macd_age = 0 if bool(row["macd_cross"]) else macd_age + 1
                ext_age  = 0 if bool(row["ext10"])      else ext_age + 1
                price_rose   = bool(row["price_rose"])
                breadth_fell = bool(row["breadth_fell"])
                breadth      = row["breadth"]
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
            # Same-day execution (lag 0): the order set above is due today.
            executed = execute_due(i, date, price)

        # 3) Mark to market at end of day on the resulting position.
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
        "Total Return": tr, "CAGR": cagr, "Max Drawdown": mdd,
        "Sharpe Ratio": sh, "Final Value": values.iloc[-1],
    }
    if trades is not None:
        n       = len(trades)
        wins    = sum(1 for t in trades if t["return_pct"] > 0)
        in_days = sum((t["exit_date"] - t["entry_date"]).days for t in trades)
        tot     = (values.index[-1] - values.index[0]).days
        m["# Trades"]       = n
        m["Win Rate"]       = wins / n if n else float("nan")
        m["Time in Market"] = in_days / tot if tot else float("nan")
    return m


def _fmt(k, v):
    if k in ("Total Return", "CAGR", "Max Drawdown", "Win Rate", "Time in Market"):
        return f"{v:.1%}"
    if k == "Sharpe Ratio":
        return f"{v:.2f}"
    if k == "Final Value":
        return f"${v:,.0f}"
    if k == "# Trades":
        return str(v)
    return str(v)


def print_comparison(same: dict, nxt: dict, bench: dict) -> None:
    keys = ["Total Return", "CAGR", "Max Drawdown", "Sharpe Ratio", "Final Value",
            "# Trades", "Win Rate", "Time in Market"]
    col = 16
    hdr = (f"{'Metric':<18}{'Same-day':>{col}}{'Next-day':>{col}}"
           f"{'Delta':>{col}}{'Buy&Hold':>{col}}")
    sep = "=" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        s, x = same.get(k), nxt.get(k)
        delta = "—"
        if k in ("Total Return", "CAGR", "Max Drawdown", "Win Rate", "Time in Market"):
            delta = f"{(x - s):+.1%}"
        elif k == "Sharpe Ratio":
            delta = f"{(x - s):+.2f}"
        elif k == "Final Value":
            delta = f"{(x - s):+,.0f}"
        elif k == "# Trades":
            delta = f"{x - s:+d}"
        b = bench.get(k)
        bstr = _fmt(k, b) if b is not None else "—"
        print(f"  {k:<16}{_fmt(k, s):>{col}}{_fmt(k, x):>{col}}{delta:>{col}}{bstr:>{col}}")
    print(sep)


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"Date range : {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")

    bench = compute_metrics(run_benchmark(df))
    same_v, same_t, _ = run_strategy(df, cooldown_days=COOLDOWN_DAYS, execution_lag=0)
    next_v, next_t, _ = run_strategy(df, cooldown_days=COOLDOWN_DAYS, execution_lag=1)

    print("\nSame-day  = fill at signal-day close (baseline, look-ahead)")
    print("Next-day  = fill at the NEXT trading day's close (realistic)")
    print_comparison(
        compute_metrics(same_v, same_t),
        compute_metrics(next_v, next_t),
        bench,
    )


if __name__ == "__main__":
    main()
