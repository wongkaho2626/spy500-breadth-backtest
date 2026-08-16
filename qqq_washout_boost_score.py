"""Retrospective Backtest-Score audit of the fixed 10% washout TQQQ boost.

The frozen single-asset engine is not modified.  This isolated harness runs a
70% NDX / 30% annual NDX-top-1 portfolio and a challenger that carves 10
percentage points from the NDX bucket into TQQQ only on canonical washout
entries.  MA200-recross trades keep that sleeve in unlevered NDX.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_backtest as qbt
import qqq_portfolio_backtest as qpb
import qqq_vector_crash_exit as analytics
import qqq_vector_recross_filter as recross
import tqqq_backtest as tqbt


DATA_DIR = Path(__file__).parent
IDEA_CARD = DATA_DIR / "docs/research/washout_boost_score_idea.md"
RESULTS_FILE = DATA_DIR / "qqq_washout_boost_score_results.json"
TRADES_FILE = DATA_DIR / "qqq_washout_boost_score_trades.csv"
REPORT_FILE = DATA_DIR / "docs/research/washout_boost_score_report.md"

BOOST_PRIMARY = 0.10
BOOST_SENSITIVITY = (0.05, 0.10, 0.15)
RELATED_TRIALS = 4_594
SPLIT_DATE = pd.Timestamp("2014-01-01")
REAL_BREADTH_START = pd.Timestamp("2007-01-01")
CLEAN_FORWARD_START = pd.Timestamp("2026-07-05")
INITIAL_CAPITAL = 10_000.0


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_inputs(drag_multiplier: float = 1.0) -> tuple[
    pd.DataFrame,
    dict[int, str],
    dict[str, pd.Series],
    dict[str, pd.Series],
    pd.DataFrame,
    float,
]:
    """Load canonical signals, constituent prices, and a continuous TQQQ proxy."""
    df = qbt.load_data()
    top_holdings = qpb.load_top_holdings()
    tickers = set(top_holdings.values())
    stocks = {
        ticker: qpb._load_stock_series(ticker).reindex(df.index).ffill()
        for ticker in tickers
        if qpb._load_stock_series(ticker) is not None
    }
    stock_opens = {
        ticker: qpb._load_stock_series(ticker, col="Open").reindex(df.index).ffill()
        for ticker in tickers
        if qpb._load_stock_series(ticker, col="Open") is not None
    }

    base = tqbt.load_tqqq_data()[["price", "open"]]
    first_actual = pd.Timestamp(tqbt.TQQQ_INCEPTION)
    actual = base.loc[first_actual:].copy()
    ndx_close = df["price"]
    ndx_open = df["open"]
    actual_returns = actual["price"].pct_change()
    overlap = pd.concat(
        [actual_returns.rename("tqqq"), ndx_close.pct_change().rename("ndx")],
        axis=1,
    ).dropna()
    drag = float((tqbt.LEVERAGE * overlap["ndx"] - overlap["tqqq"]).mean())

    pre_ndx = df.loc[df.index < first_actual, ["price", "open"]]
    pre_return = (
        tqbt.LEVERAGE * pre_ndx["price"].pct_change()
        - drag * drag_multiplier
    ).fillna(0.0)
    cumulative = (1 + pre_return).cumprod()
    boundary_ndx_return = float(ndx_close.pct_change().get(first_actual, 0.0))
    boundary_return = tqbt.LEVERAGE * boundary_ndx_return - drag * drag_multiplier
    scale = float(actual["price"].iloc[0]) / (
        float(cumulative.iloc[-1]) * (1 + boundary_return)
    )
    synthetic_close = cumulative * scale
    overnight_gap = ndx_open / ndx_close.shift(1) - 1
    synthetic_open = synthetic_close.shift(1) * (
        1 + tqbt.LEVERAGE * overnight_gap.reindex(synthetic_close.index)
    )
    synthetic_open = synthetic_open.fillna(synthetic_close)
    proxy = pd.concat(
        [
            pd.DataFrame({"price": synthetic_close, "open": synthetic_open}),
            actual,
        ]
    ).sort_index()
    proxy = proxy[~proxy.index.duplicated(keep="last")].reindex(df.index).ffill()
    return df, top_holdings, stocks, stock_opens, proxy, drag


def _safe(series: pd.Series | None, date: pd.Timestamp) -> float:
    if series is None or date not in series.index:
        return float("nan")
    value = series.loc[date]
    return float(value) if not pd.isna(value) else float("nan")


def run_portfolio(
    df: pd.DataFrame,
    top_holdings: dict[int, str],
    stocks: dict[str, pd.Series],
    stock_opens: dict[str, pd.Series],
    tqqq: pd.DataFrame,
    boost: float,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict[str, Any]], dict[str, Any] | None, pd.Series]:
    """Run the fixed portfolio with the conditional sleeve and no contributions."""
    if not 0 <= boost <= 0.70:
        raise ValueError("boost must be between zero and the 70% QQQ allocation")

    # The boost is re-targeted at each washout; it is not an independently
    # compounding permanent bucket.  Proceeds merge back into QQQ after exit.
    q_bucket = INITIAL_CAPITAL * 0.70
    stock_bucket = INITIAL_CAPITAL * 0.30
    conditional_bucket = 0.0
    q_shares = stock_shares = conditional_shares = 0.0
    q_frac = stock_q_frac = conditional_q_frac = 0.0
    stock_active = conditional_tqqq = False
    holding_ticker: str | None = None
    position = "OUT"
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_price: float | None = None
    entry_date: pd.Timestamp | None = None
    entry_value = trade_low = 0.0
    entry_type = ""
    ndx_high = 0.0
    macd_age = ext_age = 10**9
    trades: list[dict[str, Any]] = []
    values: dict[pd.Timestamp, float] = {}
    positions: dict[pd.Timestamp, bool] = {}

    lag = qbt.EXECUTION_LAG
    breadth_signal = df["breadth"].shift(lag).to_numpy()
    ndx_signal = df["price"].shift(lag).to_numpy()
    vote_gate = df["vote_gate"].shift(lag).fillna(False).to_numpy(bool)
    price_rose = df["price_rose"].shift(lag).fillna(False).to_numpy(bool)
    breadth_fell = df["breadth_fell"].shift(lag).fillna(False).to_numpy(bool)
    recross = df["ma200_recross"].shift(lag).fillna(False).to_numpy(bool)
    macd_cross = df["macd_cross"].shift(lag).fillna(False).to_numpy(bool)
    extension = df["ext10"].shift(lag).fillna(False).to_numpy(bool)
    commission = qbt.COMMISSION * cost_multiplier
    slippage = qbt.SLIPPAGE * cost_multiplier

    for i, (date, row) in enumerate(df.iterrows()):
        close = float(row["price"])
        fill = float(row["open"]) if not pd.isna(row["open"]) else close
        signal_close = float(ndx_signal[i]) if not pd.isna(ndx_signal[i]) else close
        breadth = breadth_signal[i]

        if position == "OUT":
            cooldown_ok = cooldown_until is None or date > cooldown_until
            washout = (
                not pd.isna(breadth)
                and breadth < qbt.BUY_B200_THRESH
                and bool(vote_gate[i])
            )
            recross_ok = last_sell_reason == "climax-top" or (
                last_exit_price is not None and signal_close > last_exit_price
            )
            trend = bool(recross[i]) and recross_ok
            if cooldown_ok and (washout or trend):
                holding_ticker = top_holdings.get(date.year) or top_holdings.get(date.year - 1)
                stock_close = stocks.get(holding_ticker) if holding_ticker else None
                stock_open = stock_opens.get(holding_ticker) if holding_ticker else None
                stock_fill = _safe(stock_open, date)
                if pd.isna(stock_fill):
                    stock_fill = _safe(stock_close, date)
                tq_fill = float(tqqq.loc[date, "open"]) if date in tqqq.index else np.nan
                if pd.isna(tq_fill) and date in tqqq.index:
                    tq_fill = float(tqqq.loc[date, "price"])

                total_pre = q_bucket + stock_bucket
                scale = (total_pre - commission) / total_pre if total_pre > 0 else 1.0
                q_bucket *= scale
                stock_bucket *= scale

                stock_active = not pd.isna(stock_fill)
                conditional_tqqq = washout and boost > 0 and not pd.isna(tq_fill)
                conditional_bucket = 0.0
                if conditional_tqqq:
                    conditional_bucket = min(
                        (q_bucket + stock_bucket) * boost,
                        q_bucket,
                    )
                    q_bucket -= conditional_bucket
                effective_q = q_bucket
                if not stock_active:
                    effective_q += stock_bucket
                q_frac = q_bucket / effective_q if effective_q else 0.0
                stock_q_frac = stock_bucket / effective_q if not stock_active and effective_q else 0.0
                conditional_q_frac = 0.0
                q_shares = effective_q / (fill * (1 + slippage))
                stock_shares = (
                    stock_bucket / (stock_fill * (1 + slippage))
                    if stock_active
                    else 0.0
                )
                conditional_shares = (
                    conditional_bucket / (tq_fill * (1 + slippage))
                    if conditional_tqqq
                    else 0.0
                )
                entry_date = date
                entry_type = "washout" if washout else "ma200-recross"
                entry_value = q_bucket + stock_bucket + conditional_bucket
                trade_low = entry_value
                ndx_high = signal_close
                macd_age = ext_age = 10**9
                position = "IN"

        else:
            ndx_high = max(ndx_high, signal_close)
            macd_age = 0 if bool(macd_cross[i]) else macd_age + 1
            ext_age = 0 if bool(extension[i]) else ext_age + 1
            bearish = (
                bool(price_rose[i])
                and bool(breadth_fell[i])
                and not pd.isna(breadth)
                and breadth < qbt.DIVERGENCE_BREADTH_CAP
            )
            climax = macd_age < qbt.CLIMAX_VOTE_WINDOW and ext_age < qbt.CLIMAX_VOTE_WINDOW
            trailing = signal_close <= ndx_high * (1 - qbt.TRAILING_STOP_PCT / 100)
            reason = (
                "bearish-divergence" if bearish
                else "climax-top" if climax
                else "trailing-stop" if trailing
                else None
            )
            if reason:
                stock_close = stocks.get(holding_ticker) if holding_ticker else None
                stock_open = stock_opens.get(holding_ticker) if holding_ticker else None
                stock_fill = _safe(stock_open, date)
                if pd.isna(stock_fill):
                    stock_fill = _safe(stock_close, date)
                tq_fill = float(tqqq.loc[date, "open"]) if date in tqqq.index else np.nan
                if pd.isna(tq_fill) and date in tqqq.index:
                    tq_fill = float(tqqq.loc[date, "price"])

                gross_q = q_shares * fill * (1 - slippage)
                gross_stock = (
                    stock_shares * stock_fill * (1 - slippage)
                    if stock_active and not pd.isna(stock_fill)
                    else 0.0
                )
                gross_conditional = (
                    conditional_shares * tq_fill * (1 - slippage)
                    if conditional_tqqq and not pd.isna(tq_fill)
                    else 0.0
                )
                gross_total = gross_q + gross_stock + gross_conditional
                commission_fraction = commission / gross_total if gross_total > 0 else 0.0
                q_bucket = (
                    gross_q * q_frac + gross_conditional
                ) * (1 - commission_fraction)
                stock_bucket = (
                    gross_q * stock_q_frac + gross_stock
                ) * (1 - commission_fraction)
                conditional_bucket = 0.0
                proceeds = q_bucket + stock_bucket + conditional_bucket
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_type": entry_type,
                        "conditional_asset": "TQQQ" if conditional_tqqq else "NDX",
                        "return_pct": (proceeds / entry_value - 1) * 100,
                        "max_drawdown_pct": (trade_low / entry_value - 1) * 100,
                        "accumulated": proceeds,
                        "buy_trigger": entry_type,
                        "sell_reason": reason,
                        "top1_ticker": holding_ticker,
                    }
                )
                cooldown_until = date + pd.Timedelta(days=qbt.COOLDOWN_DAYS)
                last_sell_reason = reason
                last_exit_price = fill
                position = "OUT"
                q_shares = stock_shares = conditional_shares = 0.0

        if position == "IN":
            stock_close = stocks.get(holding_ticker) if holding_ticker else None
            stock_now = _safe(stock_close, date)
            tq_now = float(tqqq.loc[date, "price"]) if date in tqqq.index else np.nan
            q_now = q_shares * close
            stock_now_value = stock_shares * stock_now if stock_active and not pd.isna(stock_now) else 0.0
            conditional_now = (
                conditional_shares * tq_now
                if conditional_tqqq and not pd.isna(tq_now)
                else 0.0
            )
            current = q_now + stock_now_value + conditional_now
            trade_low = min(trade_low, current)
            values[date] = current
        else:
            values[date] = q_bucket + stock_bucket + conditional_bucket
        positions[date] = position == "IN"

    open_trade = None
    if position == "IN":
        last_date = df.index[-1]
        current = values[last_date]
        open_trade = {
            "entry_date": entry_date,
            "current_date": last_date,
            "entry_type": entry_type,
            "conditional_asset": "TQQQ" if conditional_tqqq else "NDX",
            "return_pct": (current / entry_value - 1) * 100,
            "max_drawdown_pct": (trade_low / entry_value - 1) * 100,
            "accumulated": current,
            "buy_trigger": entry_type,
            "top1_ticker": holding_ticker,
        }
    return (
        pd.Series(values, name="portfolio"),
        trades,
        open_trade,
        pd.Series(positions, name="position"),
    )


def canonical_portfolio_baseline(
    df: pd.DataFrame,
    top_holdings: dict[int, str],
    stocks: dict[str, pd.Series],
    stock_opens: dict[str, pd.Series],
) -> tuple[pd.Series, list[dict], dict | None]:
    old_weights = (
        qpb.QQQ_WEIGHT,
        qpb.STOCK_WEIGHT,
        qpb.TQQQ_WEIGHT,
        qpb.SPY_WEIGHT,
        qpb.SOXX_WEIGHT,
    )
    try:
        qpb.QQQ_WEIGHT, qpb.STOCK_WEIGHT = 0.70, 0.30
        qpb.TQQQ_WEIGHT = qpb.SPY_WEIGHT = qpb.SOXX_WEIGHT = 0.0
        equity, trades, open_trade, _ = qpb.run_strategy(
            df,
            top_holdings,
            stocks,
            None,
            None,
            None,
            cooldown_days=qbt.COOLDOWN_DAYS,
            initial_capital=INITIAL_CAPITAL,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
            aligned_stocks_open=stock_opens,
        )
    finally:
        (
            qpb.QQQ_WEIGHT,
            qpb.STOCK_WEIGHT,
            qpb.TQQQ_WEIGHT,
            qpb.SPY_WEIGHT,
            qpb.SOXX_WEIGHT,
        ) = old_weights
    return equity, trades, open_trade


def signature(trades: list[dict]) -> list[tuple]:
    return [
        (trade["entry_date"], trade["exit_date"], trade["sell_reason"])
        for trade in trades
    ]


def score(metrics: dict[str, Any], diagnostics: dict[str, Any], robustness: dict[str, Any]) -> dict[str, Any]:
    t_stat = metrics["mean_return_t_stat"]
    t_points = 8 if t_stat > 3 else 6 if t_stat > 2 else 4 if t_stat > 1.65 else 0
    psr = metrics["psr_vs_zero"]
    psr_points = 7 if psr > 0.95 else 5 if psr > 0.90 else 3 if psr > 0.80 else 0
    dsr = diagnostics["deflated_sharpe_probability"]
    dsr_points = 8 if dsr > 0.95 else 4 if dsr > 0.80 else 0
    sample_points = 4 if metrics["completed_trades"] < 30 else 7
    component_a = t_points + psr_points + dsr_points + sample_points

    sharpe = metrics["sharpe"]
    sharpe_points = 10 if sharpe > 2 else 7 if sharpe > 1 else 4 if sharpe > 0.5 else 0
    sortino = metrics["sortino"]
    calmar = metrics["calmar"]
    ratio_points = max(
        8 if sortino > 2.5 else 6 if sortino > 1.5 else 4 if sortino > 0.7 else 0,
        8 if calmar > 2 else 6 if calmar > 1 else 4 if calmar > 0.5 else 0,
    )
    mdd = abs(metrics["max_drawdown"])
    drawdown_points = 7 if mdd < 0.10 else 5 if mdd < 0.20 else 3 if mdd < 0.30 else 0
    component_b = sharpe_points + ratio_points + drawdown_points

    # Pseudo-OOS halves are useful but are not clean forward WFA.  The thin
    # 17-trade bootstrap is marginal; sensitivity earns full points only when
    # both neighbouring fixed weights retain positive deltas.
    wfa_points = 4 if robustness["split_direction_consistent"] else 0
    bootstrap_points = 4
    sensitivity_points = 7 if robustness["sensitivity_consistent"] else 4
    component_c = wfa_points + bootstrap_points + sensitivity_points

    pf = metrics["profit_factor"]
    pf_points = 7 if pf > 2 else 5 if pf > 1.5 else 3 if pf > 1.2 else 0
    coherence_points = 6 if metrics["expectancy"] > 0 and metrics["win_rate"] > 0 else 0
    positive_months = metrics["positive_months"]
    consistency_points = 7 if positive_months > 0.65 else 5 if positive_months > 0.55 else 3 if positive_months > 0.50 else 0
    component_d = pf_points + coherence_points + consistency_points
    raw = component_a + component_b + component_c + component_d
    cap = 40 if metrics["completed_trades"] < 30 else 100
    return {
        "A_statistical_validity": component_a,
        "B_risk_adjusted_performance": component_b,
        "C_robustness_oos": component_c,
        "D_trade_quality_consistency": component_d,
        "raw_score": raw,
        "hard_cap": cap,
        "cap_reason": "fewer than 30 independent completed trades" if cap == 40 else None,
        "final_score": min(raw, cap),
    }


def evaluate_run(
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    position: pd.Series,
) -> dict[str, Any]:
    metrics = analytics.strategy_metrics(equity, trades, position)
    return {
        "metrics": metrics,
        "early_period": analytics.slice_metrics(equity, str(equity.index[0].date()), "2013-12-31"),
        "late_period": analytics.slice_metrics(equity, "2014-01-01"),
        "real_breadth_period": analytics.slice_metrics(equity, "2007-01-01"),
        "clean_forward_slice": analytics.slice_metrics(equity, "2026-07-05"),
        "statistical_diagnostics": recross.statistical_diagnostics(equity, metrics, RELATED_TRIALS),
        "trade_bootstrap": recross.trade_bootstrap(trades),
        "open_trade": open_trade,
    }


def main() -> None:
    df, top, stocks, stock_opens, tqqq, drag = load_inputs()
    direct_equity, direct_trades, direct_open = canonical_portfolio_baseline(
        df, top, stocks, stock_opens
    )
    baseline_equity, baseline_trades, baseline_open, baseline_position = run_portfolio(
        df, top, stocks, stock_opens, tqqq, boost=0.0
    )
    parity = {
        "equity_max_absolute_difference": float((direct_equity - baseline_equity).abs().max()),
        "trade_signatures_identical": signature(direct_trades) == signature(baseline_trades),
        "open_entry_identical": (
            direct_open is None and baseline_open is None
        ) or (
            direct_open is not None
            and baseline_open is not None
            and direct_open["entry_date"] == baseline_open["entry_date"]
        ),
    }
    parity["passed"] = (
        parity["equity_max_absolute_difference"] < 1e-8
        and parity["trade_signatures_identical"]
        and parity["open_entry_identical"]
    )
    if not parity["passed"]:
        raise AssertionError(f"baseline parity failed: {parity}")

    base = evaluate_run(baseline_equity, baseline_trades, baseline_open, baseline_position)
    sensitivity: dict[str, Any] = {}
    run_cache: dict[float, tuple] = {0.0: (baseline_equity, baseline_trades, baseline_open, baseline_position)}
    for weight in BOOST_SENSITIVITY:
        run_cache[weight] = run_portfolio(df, top, stocks, stock_opens, tqqq, boost=weight)
        equity, trades, open_trade, position = run_cache[weight]
        evaluated = evaluate_run(equity, trades, open_trade, position)
        sensitivity[f"{weight:.0%}"] = evaluated
    challenger = sensitivity["10%"]

    cost_stress: dict[str, Any] = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = run_portfolio(df, top, stocks, stock_opens, tqqq, 0.0, multiplier)
        challenge_cost = run_portfolio(df, top, stocks, stock_opens, tqqq, BOOST_PRIMARY, multiplier)
        base_metric = analytics.strategy_metrics(base_cost[0], base_cost[1], base_cost[3])
        challenge_metric = analytics.strategy_metrics(challenge_cost[0], challenge_cost[1], challenge_cost[3])
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": base_metric["cagr"],
            "challenger_cagr": challenge_metric["cagr"],
            "cagr_delta": challenge_metric["cagr"] - base_metric["cagr"],
            "baseline_sharpe": base_metric["sharpe"],
            "challenger_sharpe": challenge_metric["sharpe"],
        }

    drag_stress: dict[str, Any] = {}
    for multiplier in (1.0, 3.0):
        if multiplier == 1.0:
            stress_tqqq = tqqq
        else:
            _, _, _, _, stress_tqqq, _ = load_inputs(drag_multiplier=multiplier)
        run = run_portfolio(df, top, stocks, stock_opens, stress_tqqq, BOOST_PRIMARY)
        metric = analytics.strategy_metrics(run[0], run[1], run[3])
        drag_stress[f"{multiplier:.0f}x"] = {
            "cagr": metric["cagr"],
            "sharpe": metric["sharpe"],
            "max_drawdown": metric["max_drawdown"],
        }

    primary_metrics = challenger["metrics"]
    base_metrics = base["metrics"]
    cagr_deltas = {
        period: challenger[period]["cagr"] - base[period]["cagr"]
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    sharpe_deltas = {
        period: challenger[period]["sharpe"] - base[period]["sharpe"]
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    sensitivity_cagr = {
        weight: value["metrics"]["cagr"] - base_metrics["cagr"]
        for weight, value in sensitivity.items()
    }
    robustness = {
        "split_direction_consistent": all(
            cagr_deltas[p] > 0 and sharpe_deltas[p] > 0
            for p in ("early_period", "late_period")
        ),
        "sensitivity_consistent": all(value > 0 for value in sensitivity_cagr.values()),
        "cagr_deltas": cagr_deltas,
        "sharpe_deltas": sharpe_deltas,
        "sensitivity_cagr_deltas": sensitivity_cagr,
    }
    base_robustness = {
        "split_direction_consistent": True,
        "sensitivity_consistent": True,
    }
    base["score"] = score(base_metrics, base["statistical_diagnostics"], base_robustness)
    challenger["score"] = score(primary_metrics, challenger["statistical_diagnostics"], robustness)

    paired = analytics.paired_hac_and_bootstrap(
        run_cache[BOOST_PRIMARY][0], baseline_equity
    )
    unchanged_signals = all(
        signature(run_cache[weight][1]) == signature(baseline_trades)
        for weight in BOOST_SENSITIVITY
    )
    guardrails = {
        "raw_score_improved": challenger["score"]["raw_score"] > base["score"]["raw_score"],
        "final_score_at_least_80": challenger["score"]["final_score"] >= 80,
        "no_hard_cap": challenger["score"]["hard_cap"] == 100,
        "historical_halves_positive": robustness["split_direction_consistent"],
        "mdd_within_five_points": primary_metrics["max_drawdown"] >= base_metrics["max_drawdown"] - 0.05,
        "five_x_cost_positive": cost_stress["5x"]["cagr_delta"] > 0,
        "three_x_drag_positive_vs_baseline": drag_stress["3x"]["cagr"] > base_metrics["cagr"],
        "signal_dates_unchanged": unchanged_signals,
        "baseline_parity": parity["passed"],
    }
    decision = "track" if all(guardrails.values()) else "reject"

    all_trades = []
    for label, trades in (("baseline", baseline_trades), ("washout_boost_10pct", run_cache[BOOST_PRIMARY][1])):
        for trade in trades:
            all_trades.append({"strategy": label, **trade})
    pd.DataFrame(all_trades).to_csv(TRADES_FILE, index=False)

    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "tqqq_daily_drag": drag,
            "related_prior_trials": RELATED_TRIALS,
        },
        "configuration": {
            "baseline_weights": {"ndx": 0.70, "ndx_top1": 0.30},
            "challenger_weights_on_washout": {"ndx": 0.60, "ndx_top1": 0.30, "tqqq": 0.10},
            "challenger_weights_on_recross": {"ndx": 0.70, "ndx_top1": 0.30, "tqqq": 0.0},
            "sensitivity": list(BOOST_SENSITIVITY),
            "execution": "close signal, next-session open fill",
        },
        "baseline_parity": parity,
        "baseline": base,
        "challenger": challenger,
        "paired_inference": paired,
        "robustness": robustness,
        "cost_stress": cost_stress,
        "drag_stress": drag_stress,
        "guardrails": guardrails,
    }
    RESULTS_FILE.write_text(json.dumps(_finite(results), indent=2), encoding="utf-8")
    write_report(results)
    print(json.dumps(_finite({
        "decision": decision,
        "baseline_score": base["score"],
        "challenger_score": challenger["score"],
        "baseline_metrics": base_metrics,
        "challenger_metrics": primary_metrics,
        "guardrails": guardrails,
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    base = results["baseline"]
    challenge = results["challenger"]
    bm, cm = base["metrics"], challenge["metrics"]
    bs, cs = base["score"], challenge["score"]
    verdict = "Reject" if results["decision"] == "reject" else "Track as research challenger"
    lines = [
        "# Backtest Verification Report — Washout-only 10% TQQQ boost",
        "",
        f"## Verdict: {verdict}",
        "",
        f"Canonical-signal 70/30 portfolio raw/final score: **{bs['raw_score']} / {bs['final_score']}**.  "
        f"Washout boost raw/final score: **{cs['raw_score']} / {cs['final_score']}**.  "
        "The final score is the number that determines the 80-point objective.",
        "",
        "## Backtest Scores",
        "",
        "| Component | Baseline | 10% boost | Max |",
        "|---|---:|---:|---:|",
        f"| A. Statistical validity | {bs['A_statistical_validity']} | {cs['A_statistical_validity']} | 30 |",
        f"| B. Risk-adjusted performance | {bs['B_risk_adjusted_performance']} | {cs['B_risk_adjusted_performance']} | 25 |",
        f"| C. Robustness / OOS | {bs['C_robustness_oos']} | {cs['C_robustness_oos']} | 25 |",
        f"| D. Trade quality / consistency | {bs['D_trade_quality_consistency']} | {cs['D_trade_quality_consistency']} | 20 |",
        f"| **Raw total** | **{bs['raw_score']}** | **{cs['raw_score']}** | **100** |",
        f"| Hard cap | {bs['hard_cap']} | {cs['hard_cap']} | |",
        f"| **Final score** | **{bs['final_score']}** | **{cs['final_score']}** | **100** |",
        "",
        "Both strategies have fewer than 30 independent completed trades.  Under the installed rubric, that is a 40-point hard cap; stronger CAGR cannot remove it.",
        "",
        "## Performance",
        "",
        "| Metric | Baseline | 10% boost | Delta |",
        "|---|---:|---:|---:|",
        f"| CAGR | {bm['cagr']:.2%} | {cm['cagr']:.2%} | {cm['cagr']-bm['cagr']:+.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {cm['sharpe']:.3f} | {cm['sharpe']-bm['sharpe']:+.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {cm['sortino']:.3f} | {cm['sortino']-bm['sortino']:+.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {cm['calmar']:.3f} | {cm['calmar']-bm['calmar']:+.3f} |",
        f"| Maximum drawdown | {bm['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} | {cm['max_drawdown']-bm['max_drawdown']:+.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} | {cm['ulcer_index']-bm['ulcer_index']:+.2%} |",
        f"| Positive months | {bm['positive_months']:.2%} | {cm['positive_months']:.2%} | {cm['positive_months']-bm['positive_months']:+.2%} |",
        f"| Completed trades | {bm['completed_trades']} | {cm['completed_trades']} | 0 |",
        "",
        "## Validity and robustness",
        "",
        f"- Baseline parity max equity difference: {results['baseline_parity']['equity_max_absolute_difference']:.3g}; signatures identical: {results['baseline_parity']['trade_signatures_identical']}.",
        f"- Paired HAC t-stat: {results['paired_inference']['hac_t_stat']:.3f}; p={results['paired_inference']['hac_two_sided_p']:.3f}; 95% block-bootstrap annual-return interval {results['paired_inference']['bootstrap_95_interval_annualized']}.",
        f"- Historical-half direction consistent: {results['robustness']['split_direction_consistent']}.",
        f"- 5%/10%/15% sensitivity all retain positive CAGR delta: {results['robustness']['sensitivity_consistent']}.",
        f"- 5x-cost CAGR delta: {results['cost_stress']['5x']['cagr_delta']:+.2%}.",
        f"- 3x pre-inception drag CAGR: {results['drag_stress']['3x']['cagr']:.2%} versus baseline {bm['cagr']:.2%}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Close signal, next-session open fill |",
        "| Survivorship | Cannot fully verify | Aggregate NDX signal plus annual top-1 history; delisted constituent completeness is not proven |",
        "| Data snooping | Present, material | The 10% result was already disclosed and thousands of related trials exist |",
        "| Costs | Included | 1x/2x/5x/10x commission and slippage stress |",
        "| Synthetic data | Present before 2010 | TQQQ proxy uses 3x NDX less overlap-calibrated drag; 3x drag stressed |",
        "| Clean forward OOS | Insufficient | Freeze occurred 2026-07-05; no completed forward round trip |",
        "",
        "## Decision",
        "",
        "This challenger is rejected as the requested 80-score strategy if any pre-registered guardrail fails.  Historical improvement may still be economically interesting, but it cannot be promoted into the frozen baseline without sufficient independent forward evidence.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
