"""Exit-only challenger: month-end NDX<MA200 and S&P breadth below a floor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_backtest as qbt
import qqq_vector_crash_exit as analytics
import qqq_vector_recross_filter as research


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/monthly_breadth_regime_exit_idea.md"
RESULTS_FILE = ROOT / "qqq_monthly_breadth_regime_exit_results.json"
EQUITY_FILE = ROOT / "qqq_monthly_breadth_regime_exit_equity.csv"
TRADES_FILE = ROOT / "qqq_monthly_breadth_regime_exit_trades.csv"
SIGNALS_FILE = ROOT / "qqq_monthly_breadth_regime_exit_signals.csv"
REPORT_FILE = ROOT / "docs/research/monthly_breadth_regime_exit_report.md"

PRIMARY_BREADTH = 50.0
SENSITIVITY = (40.0, 50.0, 60.0)
RELATED_TRIALS = 4_595
SPLIT_DATE = pd.Timestamp("2014-01-01")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def monthly_regime_signal(df: pd.DataFrame, breadth_threshold: float) -> pd.Series:
    months = pd.Series(df.index.to_period("M"), index=df.index)
    month_end = months.ne(months.shift(-1)).fillna(True)
    return (
        month_end
        & (df["price"] < df["ma200"])
        & (df["breadth"] < breadth_threshold)
    ).fillna(False).rename("monthly_regime_exit")


def run_challenger(
    df: pd.DataFrame,
    extra_exit: pd.Series,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    """Canonical state machine plus one lower-priority extra exit.

    Existing bearish-divergence, climax, and trailing-stop exits retain their
    original priority.  A close signal fills at the next session open.
    """
    extra = extra_exit.reindex(df.index).fillna(False).astype(bool)
    position = "OUT"
    eff_entry = raw_entry = 0.0
    entry_date = None
    trade_low = trade_high = 0.0
    macd_age = ext_age = 10**9
    buy_trigger = None
    portfolio = qbt.INITIAL_CAPITAL
    cooldown_until = None
    last_sell_reason = None
    last_exit_price = None
    trades: list[dict] = []
    values: dict[pd.Timestamp, float] = {}
    pending: dict | None = None
    rows = list(df.iterrows())
    n = len(rows)
    commission = qbt.COMMISSION * cost_multiplier
    slippage = qbt.SLIPPAGE * cost_multiplier

    def execute_due(i: int, date: pd.Timestamp, fill_price: float) -> bool:
        nonlocal position, eff_entry, raw_entry, entry_date, trade_low, trade_high
        nonlocal macd_age, ext_age, buy_trigger, portfolio, cooldown_until
        nonlocal last_sell_reason, last_exit_price, pending
        if pending is None or pending["fill_at"] != i:
            return False
        if pending["action"] == "BUY" and position == "OUT":
            portfolio -= commission
            eff_entry = fill_price * (1 + slippage)
            raw_entry = fill_price
            entry_date = date
            trade_low = trade_high = fill_price
            macd_age = ext_age = 10**9
            buy_trigger = pending["trigger"]
            position = "IN"
            pending = None
            return True
        if pending["action"] == "SELL" and position == "IN":
            eff_exit = fill_price * (1 - slippage)
            gross_return = (eff_exit - eff_entry) / eff_entry
            portfolio *= 1 + gross_return
            portfolio -= commission
            cooldown_until = date + pd.Timedelta(days=qbt.COOLDOWN_DAYS)
            last_sell_reason = pending["reason"]
            last_exit_price = fill_price
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
                    "sell_reason": pending["reason"],
                    "signal_date": pending["signal_date"],
                    "cooldown_until": cooldown_until,
                }
            )
            position = "OUT"
            pending = None
            return True
        pending = None
        return False

    for i, (date, row) in enumerate(rows):
        price = float(row["price"])
        breadth = float(row["breadth"])
        fill_price = (
            float(row["open"])
            if qbt.FILL_PRICE == "open" and not pd.isna(row["open"])
            else price
        )
        executed = execute_due(i, date, fill_price)

        if not executed and pending is None:
            if position == "OUT":
                cooldown_ok = cooldown_until is None or date > cooldown_until
                washout = breadth < qbt.BUY_B200_THRESH and bool(row["vote_gate"])
                recross_ok = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and price > last_exit_price
                )
                trend = bool(row["ma200_recross"]) and recross_ok
                if cooldown_ok and (washout or trend) and i + qbt.EXECUTION_LAG < n:
                    if washout:
                        trigger = (
                            ("VIX" if row["vix_vote"] else "")
                            + ("+" if row["vix_vote"] and row["ma200_vote"] else "")
                            + ("MA200" if row["ma200_vote"] else "")
                        )
                    else:
                        trigger = "MA200-recross"
                    pending = {
                        "action": "BUY",
                        "fill_at": i + qbt.EXECUTION_LAG,
                        "trigger": trigger,
                        "signal_date": date,
                    }
            else:
                trade_low = min(trade_low, price)
                trade_high = max(trade_high, price)
                macd_age = 0 if bool(row["macd_cross"]) else macd_age + 1
                ext_age = 0 if bool(row["ext10"]) else ext_age + 1
                bearish = (
                    bool(row["price_rose"])
                    and bool(row["breadth_fell"])
                    and breadth < qbt.DIVERGENCE_BREADTH_CAP
                )
                climax = (
                    macd_age < qbt.CLIMAX_VOTE_WINDOW
                    and ext_age < qbt.CLIMAX_VOTE_WINDOW
                )
                trailing = price <= trade_high * (1 - qbt.TRAILING_STOP_PCT / 100)
                reason = (
                    "bearish-divergence" if bearish
                    else "climax-top" if climax
                    else "trailing-stop" if trailing
                    else "monthly-breadth-regime" if bool(extra.loc[date])
                    else None
                )
                if reason and i + qbt.EXECUTION_LAG < n:
                    pending = {
                        "action": "SELL",
                        "fill_at": i + qbt.EXECUTION_LAG,
                        "reason": reason,
                        "signal_date": date,
                    }
            execute_due(i, date, fill_price)

        values[date] = (
            portfolio * (price * (1 - slippage) / eff_entry)
            if position == "IN"
            else portfolio
        )

    open_trade = None
    if position == "IN":
        last_date = df.index[-1]
        last_price = float(df["price"].iloc[-1])
        eff_last = last_price * (1 - slippage)
        open_trade = {
            "entry_date": entry_date,
            "entry_price": raw_entry,
            "current_date": last_date,
            "current_price": last_price,
            "return_pct": (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "accumulated": portfolio * eff_last / eff_entry,
            "buy_trigger": buy_trigger,
        }
    return pd.Series(values, name="strategy"), trades, open_trade


def trade_signature(trades: list[dict]) -> list[tuple]:
    return [
        (
            trade["entry_date"],
            trade["exit_date"],
            trade["sell_reason"],
            round(float(trade["return_pct"]), 10),
        )
        for trade in trades
    ]


def open_trade_equal(left: dict | None, right: dict | None) -> bool:
    if left is None or right is None:
        return left is right
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            if not np.isclose(float(a), float(b), rtol=1e-12, atol=1e-8):
                return False
        elif a != b:
            return False
    return True


def evaluate(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
) -> dict[str, Any]:
    position = analytics.position_series(df.index, trades, open_trade)
    metrics = analytics.strategy_metrics(equity, trades, position)
    return {
        "metrics": metrics,
        "early_period": analytics.slice_metrics(equity, "2002-01-01", "2013-12-31"),
        "late_period": analytics.slice_metrics(equity, "2014-01-01"),
        "real_breadth_period": analytics.slice_metrics(equity, "2007-01-01"),
        "clean_forward_slice": analytics.slice_metrics(equity, "2026-07-05"),
        "statistical_diagnostics": research.statistical_diagnostics(
            equity, metrics, RELATED_TRIALS
        ),
        "trade_bootstrap": research.trade_bootstrap(trades),
        "exit_reason_counts": pd.Series(
            [trade["sell_reason"] for trade in trades]
        ).value_counts().to_dict(),
        "open_trade": open_trade,
    }


def efficiency(evaluation: dict[str, Any]) -> float:
    early = float(evaluation["early_period"]["sharpe"])
    late = float(evaluation["late_period"]["sharpe"])
    return min(early, late) / max(early, late) if max(early, late) > 0 else 0.0


def score(
    evaluation: dict[str, Any],
    wfa_efficiency: float,
    bootstrap_stable: bool,
    sensitivity_stable: bool,
) -> dict[str, Any]:
    metrics = evaluation["metrics"]
    diagnostics = evaluation["statistical_diagnostics"]
    t_stat = metrics["mean_return_t_stat"]
    t_points = 8 if t_stat > 3 else 6 if t_stat > 2 else 4 if t_stat > 1.65 else 0
    psr = metrics["psr_vs_zero"]
    psr_points = 7 if psr > 0.95 else 5 if psr > 0.90 else 3 if psr > 0.80 else 0
    dsr = diagnostics["deflated_sharpe_probability"]
    dsr_points = 8 if dsr > 0.95 else 4 if dsr > 0.80 else 0
    sample_points = 7 if metrics["completed_trades"] >= 30 else 4
    component_a = t_points + psr_points + dsr_points + sample_points

    sharpe = metrics["sharpe"]
    sharpe_points = 10 if sharpe > 2 else 7 if sharpe > 1 else 4 if sharpe > 0.5 else 0
    sortino, calmar = metrics["sortino"], metrics["calmar"]
    ratio_points = max(
        8 if sortino > 2.5 else 6 if sortino > 1.5 else 4 if sortino > 0.7 else 0,
        8 if calmar > 2 else 6 if calmar > 1 else 4 if calmar > 0.5 else 0,
    )
    mdd = abs(metrics["max_drawdown"])
    drawdown_points = 7 if mdd < 0.10 else 5 if mdd < 0.20 else 3 if mdd < 0.30 else 0
    component_b = sharpe_points + ratio_points + drawdown_points

    wfa_points = 10 if wfa_efficiency > 0.7 else 7 if wfa_efficiency > 0.5 else 4 if wfa_efficiency > 0.3 else 0
    bootstrap_points = 8 if bootstrap_stable else 4
    sensitivity_points = 7 if sensitivity_stable else 4
    component_c = wfa_points + bootstrap_points + sensitivity_points

    pf = metrics["profit_factor"]
    pf_points = 7 if pf > 2 else 5 if pf > 1.5 else 3 if pf > 1.2 else 0
    coherence_points = 6 if metrics["expectancy"] > 0 and metrics["win_rate"] > 0 else 0
    positive = metrics["positive_months"]
    consistency_points = 7 if positive > 0.65 else 5 if positive > 0.55 else 3 if positive > 0.50 else 0
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
        "band": (
            "Tradeable" if min(raw, cap) >= 80
            else "Promising" if min(raw, cap) >= 65
            else "Needs work" if min(raw, cap) >= 45
            else "Weak" if min(raw, cap) >= 25
            else "Reject"
        ),
    }


def write_artifacts(
    df: pd.DataFrame,
    variants: dict[str, tuple[pd.Series, list[dict], dict | None]],
    signals: dict[str, pd.Series],
) -> None:
    pd.DataFrame({name: run[0] for name, run in variants.items()}).to_csv(EQUITY_FILE)
    trade_rows = []
    for name, (_, trades, _) in variants.items():
        for trade in trades:
            trade_rows.append({"variant": name, **trade})
    pd.DataFrame(trade_rows).to_csv(TRADES_FILE, index=False)
    signal_frame = pd.DataFrame(signals)
    signal_frame["price"] = df["price"]
    signal_frame["ma200"] = df["ma200"]
    signal_frame["breadth"] = df["breadth"]
    signal_frame.to_csv(SIGNALS_FILE)


def main() -> None:
    df = qbt.load_data()
    disabled = pd.Series(False, index=df.index)
    direct_equity, direct_trades, direct_open = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    parity_equity, parity_trades, parity_open = run_challenger(df, disabled)
    parity = {
        "equity_max_absolute_difference": float((direct_equity - parity_equity).abs().max()),
        "trade_signatures_identical": trade_signature(direct_trades) == trade_signature(parity_trades),
        "open_trade_identical": open_trade_equal(direct_open, parity_open),
    }
    parity["passed"] = bool(
        parity["equity_max_absolute_difference"] < 1e-8
        and parity["trade_signatures_identical"]
        and parity["open_trade_identical"]
    )
    if not parity["passed"]:
        raise AssertionError(f"baseline parity failed: {parity}")

    baseline = evaluate(df, direct_equity, direct_trades, direct_open)
    signals = {f"breadth_{threshold:.0f}": monthly_regime_signal(df, threshold) for threshold in SENSITIVITY}
    runs = {
        f"breadth_{threshold:.0f}": run_challenger(df, signals[f"breadth_{threshold:.0f}"])
        for threshold in SENSITIVITY
    }
    evaluations = {
        name: evaluate(df, *run)
        for name, run in runs.items()
    }
    primary = evaluations["breadth_50"]
    primary_run = runs["breadth_50"]
    paired = analytics.paired_hac_and_bootstrap(primary_run[0], direct_equity)

    sensitivity_rows = {}
    for name, value in evaluations.items():
        sensitivity_rows[name] = {
            metric: value["metrics"][metric]
            for metric in ("cagr", "sharpe", "calmar", "max_drawdown", "completed_trades", "expectancy")
        }
    calmar_deltas = [
        value["metrics"]["calmar"] - baseline["metrics"]["calmar"]
        for value in evaluations.values()
    ]
    sensitivity_stable = all(delta > 0 for delta in calmar_deltas)
    bootstrap_stable = paired["bootstrap_95_interval_annualized"][0] > 0
    baseline["score"] = score(
        baseline,
        efficiency(baseline),
        bootstrap_stable=True,
        sensitivity_stable=True,
    )
    primary["score"] = score(
        primary,
        efficiency(primary),
        bootstrap_stable=bootstrap_stable,
        sensitivity_stable=sensitivity_stable,
    )

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = run_challenger(df, disabled, multiplier)
        challenge_cost = run_challenger(df, signals["breadth_50"], multiplier)
        base_eval = evaluate(df, *base_cost)
        challenge_eval = evaluate(df, *challenge_cost)
        cost_pair = analytics.paired_hac_and_bootstrap(challenge_cost[0], base_cost[0])
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": base_eval["metrics"]["cagr"],
            "challenger_cagr": challenge_eval["metrics"]["cagr"],
            "cagr_delta": challenge_eval["metrics"]["cagr"] - base_eval["metrics"]["cagr"],
            "paired_annualized_mean": cost_pair["annualized_mean_difference"],
            "paired_hac_t": cost_pair["hac_t_stat"],
        }

    period_deltas = {
        period: {
            metric: primary[period][metric] - baseline[period][metric]
            for metric in ("cagr", "sharpe", "max_drawdown")
        }
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    # Calmar for each segment, computed from segment CAGR and segment MDD.
    for period in period_deltas:
        base_calmar = baseline[period]["cagr"] / abs(baseline[period]["max_drawdown"])
        challenge_calmar = primary[period]["cagr"] / abs(primary[period]["max_drawdown"])
        period_deltas[period]["calmar"] = challenge_calmar - base_calmar

    bm, cm = baseline["metrics"], primary["metrics"]
    guardrails = {
        "baseline_parity": parity["passed"],
        "final_score_at_least_80": primary["score"]["final_score"] >= 80,
        "calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "cagr_within_two_points": cm["cagr"] >= bm["cagr"] - 0.02,
        "positive_expectancy": cm["expectancy"] > 0,
        "profit_factor_above_1_2": cm["profit_factor"] > 1.2,
        "historical_halves_calmar_positive": all(
            period_deltas[period]["calmar"] >= 0
            for period in ("early_period", "late_period")
        ),
        "turnover_guardrail": (
            cm["completed_trades"] <= 2 * bm["completed_trades"]
            or cm["expectancy"] > bm["expectancy"]
        ),
        "five_x_paired_return_positive": cost_stress["5x"]["paired_annualized_mean"] > 0,
        "real_breadth_direction_positive": period_deltas["real_breadth_period"]["calmar"] >= 0,
        "sensitivity_not_cliff_edge": sensitivity_stable,
    }
    decision = "track" if all(guardrails.values()) else "reject"

    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "clean_forward_start": "2026-07-05",
            "related_prior_trials": RELATED_TRIALS,
        },
        "configuration": {
            "primary_breadth_threshold": PRIMARY_BREADTH,
            "sensitivity": list(SENSITIVITY),
            "ma_window": qbt.MA200_WINDOW,
            "frequency": "last trading close of calendar month",
            "fill": "next-session open",
            "change": "additional exit only",
        },
        "baseline_parity": parity,
        "baseline": baseline,
        "challenger": primary,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": efficiency(baseline),
            "challenger": efficiency(primary),
            "interpretation": "fixed-rule bidirectional historical-half efficiency; pseudo-OOS, not clean post-freeze OOS",
        },
        "sensitivity": sensitivity_rows,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "guardrails": guardrails,
        "current_signal": {
            "date": df.index[-1],
            "active": bool(signals["breadth_50"].iloc[-1]),
            "price_below_ma200": bool(df["price"].iloc[-1] < df["ma200"].iloc[-1]),
            "breadth_below_50": bool(df["breadth"].iloc[-1] < 50),
            "is_month_end": bool(pd.Series(df.index.to_period("M"), index=df.index).ne(pd.Series(df.index.to_period("M"), index=df.index).shift(-1)).iloc[-1]),
        },
    }
    variants = {"baseline": (direct_equity, direct_trades, direct_open), **runs}
    write_artifacts(df, variants, signals)
    RESULTS_FILE.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    write_report(results)
    print(json.dumps(_jsonable({
        "decision": decision,
        "baseline_score": baseline["score"],
        "challenger_score": primary["score"],
        "baseline_metrics": bm,
        "challenger_metrics": cm,
        "guardrails": guardrails,
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    base, challenge = results["baseline"], results["challenger"]
    bm, cm = base["metrics"], challenge["metrics"]
    bs, cs = base["score"], challenge["score"]
    verdict = "Reject" if results["decision"] == "reject" else "Track as research challenger"
    lines = [
        "# Backtest Verification Report — Monthly MA200 × breadth regime exit",
        "",
        f"## Verdict: {verdict}",
        "",
        f"The 50% month-end challenger scores **{cs['final_score']} / 100 ({cs['band']})** versus the frozen baseline **{bs['final_score']} / 100 ({bs['band']})**.",
        "",
        "## Backtest Scores",
        "",
        "| Component | Baseline | Challenger | Max |",
        "|---|---:|---:|---:|",
        f"| A. Statistical validity | {bs['A_statistical_validity']} | {cs['A_statistical_validity']} | 30 |",
        f"| B. Risk-adjusted performance | {bs['B_risk_adjusted_performance']} | {cs['B_risk_adjusted_performance']} | 25 |",
        f"| C. Robustness / OOS | {bs['C_robustness_oos']} | {cs['C_robustness_oos']} | 25 |",
        f"| D. Trade quality / consistency | {bs['D_trade_quality_consistency']} | {cs['D_trade_quality_consistency']} | 20 |",
        f"| **Raw total** | **{bs['raw_score']}** | **{cs['raw_score']}** | **100** |",
        f"| Hard cap | {bs['hard_cap']} | {cs['hard_cap']} | |",
        f"| **Final score** | **{bs['final_score']}** | **{cs['final_score']}** | **100** |",
        "",
        "## Performance",
        "",
        "| Metric | Baseline | Challenger | Delta |",
        "|---|---:|---:|---:|",
        f"| CAGR | {bm['cagr']:.2%} | {cm['cagr']:.2%} | {cm['cagr']-bm['cagr']:+.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {cm['sharpe']:.3f} | {cm['sharpe']-bm['sharpe']:+.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {cm['sortino']:.3f} | {cm['sortino']-bm['sortino']:+.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {cm['calmar']:.3f} | {cm['calmar']-bm['calmar']:+.3f} |",
        f"| Maximum drawdown | {bm['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} | {cm['max_drawdown']-bm['max_drawdown']:+.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} | {cm['ulcer_index']-bm['ulcer_index']:+.2%} |",
        f"| Positive months | {bm['positive_months']:.2%} | {cm['positive_months']:.2%} | {cm['positive_months']-bm['positive_months']:+.2%} |",
        f"| Completed trades | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        f"| Profit factor | {bm['profit_factor']:.2f} | {cm['profit_factor']:.2f} | |",
        f"| Expectancy | {bm['expectancy']:.2%} | {cm['expectancy']:.2%} | |",
        "",
        "## Statistical significance and robustness",
        "",
        f"- Paired annual mean difference: {results['paired_inference']['annualized_mean_difference']:+.2%}; HAC t={results['paired_inference']['hac_t_stat']:.3f}, p={results['paired_inference']['hac_two_sided_p']:.3f}.",
        f"- 21-session block-bootstrap 95% interval: {results['paired_inference']['bootstrap_95_interval_annualized']}.",
        f"- Fixed-rule historical-half efficiency: baseline {results['wfa_efficiency']['baseline']:.3f}, challenger {results['wfa_efficiency']['challenger']:.3f}.  These are pseudo-OOS robustness checks, not clean forward evidence.",
        f"- Sensitivity Calmar values: " + ", ".join(f"{key}={value['calmar']:.3f}" for key, value in results['sensitivity'].items()) + ".",
        f"- 5x-cost paired annual mean: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Month-end close signal, next-session-open fill |",
        "| Survivorship | Cannot fully verify | Aggregate index and breadth; historical constituent reconstruction is not used |",
        "| Data snooping | Present, material | At least 4,595 related trials; DSR penalty applied |",
        "| Costs | Included | 1x/2x/5x/10x commission and slippage stress |",
        "| Synthetic breadth | Present before 2007 | 2007+ real-breadth period reported separately |",
        "| Clean forward OOS | Insufficient | No completed post-2026-07-05 forward round trip |",
        "",
        "## Guardrails",
        "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in results["guardrails"].items()
    )
    lines += [
        "",
        "## Decision",
        "",
        "The decision follows the pre-registered rule without moving the target.  A historical score at or above 80 is still only research evidence until sufficient clean post-freeze observations accumulate.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
