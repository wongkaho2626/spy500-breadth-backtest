"""Fixed 70/30 breadth/MA200 ensemble with a washout-only TQQQ allocation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_backtest as qbt
import qqq_breadth_ma200_core_satellite as core
import qqq_breadth_tqqq_ma200_satellite as proxy_source
import qqq_monthly_breadth_regime_exit as scorelib
import qqq_vector_crash_exit as analytics
import qqq_vector_recross_filter as research
import tqqq_backtest as tqbt


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/breadth_ma200_washout_boost_idea.md"
RESULTS_FILE = ROOT / "qqq_breadth_ma200_washout_boost_results.json"
EQUITY_FILE = ROOT / "qqq_breadth_ma200_washout_boost_equity.csv"
TRADES_FILE = ROOT / "qqq_breadth_ma200_washout_boost_trades.csv"
REPORT_FILE = ROOT / "docs/research/breadth_ma200_washout_boost_report.md"

TREND_WEIGHT = 0.30
PRIMARY_BOOST = 0.10
SENSITIVITY = (0.05, 0.10, 0.15)
RELATED_TRIALS = 4_598


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


def _trade_schedule(
    df: pd.DataFrame,
    initial_capital: float,
    cost_multiplier: float,
) -> tuple[dict[pd.Timestamp, dict], dict[pd.Timestamp, dict]]:
    """Get the unchanged frozen entry/exit schedule and trigger labels."""
    _, trades, open_trade, _ = core.run_breadth(
        df, initial_capital, cost_multiplier
    )
    entries: dict[pd.Timestamp, dict] = {}
    exits: dict[pd.Timestamp, dict] = {}
    for trade in trades:
        entries[trade["entry_date"]] = trade
        exits[trade["exit_date"]] = trade
    if open_trade is not None:
        entries[open_trade["entry_date"]] = open_trade
    return entries, exits


def run_boosted_breadth(
    df: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    boost: float,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None, pd.Series]:
    """Follow frozen breadth dates; replace part of NDX with TQQQ on washouts."""
    if boost < 0 or boost > 1 - TREND_WEIGHT:
        raise ValueError("boost must fit inside the breadth sleeve")
    if boost == 0:
        return core.run_breadth(df, initial_capital, cost_multiplier)

    entries, exits = _trade_schedule(df, initial_capital, cost_multiplier)
    commission = qbt.COMMISSION * cost_multiplier
    slippage = qbt.SLIPPAGE * cost_multiplier
    boost_fraction = boost / (1 - TREND_WEIGHT)
    cash = initial_capital
    ndx_shares = tqqq_shares = 0.0
    position = False
    leveraged = False
    entry_date = entry_signal_date = None
    entry_capital = entry_price = trade_low = 0.0
    buy_trigger = ""
    values: dict[pd.Timestamp, float] = {}
    positions: dict[pd.Timestamp, bool] = {}
    trades: list[dict] = []
    locations = {date: i for i, date in enumerate(df.index)}

    for date, row in df.iterrows():
        ndx_close = float(row["price"])
        ndx_open = float(row["open"]) if not pd.isna(row["open"]) else ndx_close
        tq_close = float(tqqq.loc[date, "price"])
        tq_open = float(tqqq.loc[date, "open"])
        if pd.isna(tq_open):
            tq_open = tq_close

        if date in exits and position:
            record = exits[date]
            legs = 2 if leveraged else 1
            proceeds = ndx_shares * ndx_open * (1 - slippage)
            if leveraged:
                proceeds += tqqq_shares * tq_open * (1 - slippage)
            proceeds -= legs * commission
            exit_i = locations[date]
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": ndx_open,
                    "return_pct": (proceeds / entry_capital - 1) * 100,
                    "max_drawdown_pct": (trade_low / entry_capital - 1) * 100,
                    "accumulated": proceeds,
                    "buy_trigger": buy_trigger,
                    "sell_reason": record["sell_reason"],
                    "entry_signal_date": entry_signal_date,
                    "signal_date": df.index[exit_i - qbt.EXECUTION_LAG],
                    "tqqq_boosted": leveraged,
                }
            )
            cash = proceeds
            ndx_shares = tqqq_shares = 0.0
            position = leveraged = False

        if date in entries and not position:
            record = entries[date]
            buy_trigger = record["buy_trigger"]
            leveraged = buy_trigger != "MA200-recross"
            legs = 2 if leveraged else 1
            entry_capital = cash
            investable = cash - legs * commission
            tq_allocation = investable * boost_fraction if leveraged else 0.0
            ndx_allocation = investable - tq_allocation
            ndx_shares = ndx_allocation / (ndx_open * (1 + slippage))
            tqqq_shares = (
                tq_allocation / (tq_open * (1 + slippage)) if leveraged else 0.0
            )
            cash = 0.0
            entry_date = date
            entry_signal_date = df.index[locations[date] - qbt.EXECUTION_LAG]
            entry_price = ndx_open
            current = (
                ndx_shares * ndx_close * (1 - slippage)
                + tqqq_shares * tq_close * (1 - slippage)
            )
            trade_low = current
            position = True

        if position:
            current = (
                ndx_shares * ndx_close * (1 - slippage)
                + tqqq_shares * tq_close * (1 - slippage)
            )
            trade_low = min(trade_low, current)
            values[date] = current
        else:
            values[date] = cash
        positions[date] = position

    open_trade = None
    if position:
        last_date = df.index[-1]
        current = values[last_date]
        open_trade = {
            "entry_date": entry_date,
            "entry_price": entry_price,
            "current_date": last_date,
            "current_price": float(df["price"].iloc[-1]),
            "return_pct": (current / entry_capital - 1) * 100,
            "max_drawdown_pct": (trade_low / entry_capital - 1) * 100,
            "accumulated": current,
            "buy_trigger": buy_trigger,
            "entry_signal_date": entry_signal_date,
            "tqqq_boosted": leveraged,
        }
    return (
        pd.Series(values, name="boosted_breadth_equity"),
        trades,
        open_trade,
        pd.Series(positions, name="boosted_breadth_position"),
    )


def run_ensemble(
    df: pd.DataFrame,
    tqqq: pd.DataFrame,
    boost: float,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    breadth = run_boosted_breadth(
        df,
        tqqq,
        core.INITIAL_CAPITAL * (1 - TREND_WEIGHT),
        boost,
        cost_multiplier,
    )
    trend = core.run_ma200(
        df, core.INITIAL_CAPITAL * TREND_WEIGHT, cost_multiplier
    )
    combined = (breadth[0] + trend[0]).rename("combined_equity")
    events = core.component_events("breadth", breadth[1], breadth[0], combined)
    events += core.component_events("ma200", trend[1], trend[0], combined)
    return {
        "equity": combined,
        "position": breadth[3] | trend[3],
        "breadth": breadth,
        "trend": trend,
        "events": events,
        "clustered_trades": core.cluster_events(df.index, events),
    }


def evaluate(df: pd.DataFrame, run: dict[str, Any]) -> dict[str, Any]:
    metrics = analytics.strategy_metrics(
        run["equity"], run["clustered_trades"], run["position"]
    )
    return {
        "metrics": metrics,
        "early_period": analytics.slice_metrics(
            run["equity"], "2002-01-01", "2013-12-31"
        ),
        "late_period": analytics.slice_metrics(run["equity"], "2014-01-01"),
        "real_breadth_period": analytics.slice_metrics(
            run["equity"], "2007-01-01"
        ),
        "clean_forward_slice": analytics.slice_metrics(
            run["equity"], "2026-07-05"
        ),
        "statistical_diagnostics": research.statistical_diagnostics(
            run["equity"], metrics, RELATED_TRIALS
        ),
        "trade_bootstrap": research.trade_bootstrap(run["clustered_trades"]),
        "raw_component_exits": len(run["events"]),
        "clustered_exit_events": len(run["clustered_trades"]),
        "breadth_completed_trades": len(run["breadth"][1]),
        "ma200_completed_trades": len(run["trend"][1]),
    }


def period_calmar(period: dict[str, Any]) -> float:
    return period["cagr"] / abs(period["max_drawdown"])


def score_evaluation(
    evaluation: dict[str, Any],
    paired: dict[str, Any],
    sensitivity_stable: bool,
) -> dict[str, Any]:
    return scorelib.score(
        evaluation,
        scorelib.efficiency(evaluation),
        bootstrap_stable=paired["bootstrap_95_interval_annualized"][0] > 0,
        sensitivity_stable=sensitivity_stable,
    )


def evaluate_family(
    df: pd.DataFrame,
    proxy: pd.DataFrame,
    frozen_equity: pd.Series,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    runs = {f"{boost:.0%}": run_ensemble(df, proxy, boost) for boost in SENSITIVITY}
    evaluations = {name: evaluate(df, run) for name, run in runs.items()}
    paired = {
        name: analytics.paired_hac_and_bootstrap(run["equity"], frozen_equity)
        for name, run in runs.items()
    }
    return runs, evaluations, paired


def sensitivity_is_stable(
    evaluations: dict[str, dict[str, Any]],
    paired: dict[str, dict[str, Any]],
    baseline_metrics: dict[str, Any],
) -> bool:
    for name, evaluation in evaluations.items():
        metrics = evaluation["metrics"]
        if not (
            metrics["cagr"] > baseline_metrics["cagr"]
            and metrics["sharpe"] > baseline_metrics["sharpe"]
            and metrics["calmar"] > baseline_metrics["calmar"]
            and paired[name]["annualized_mean_difference"] > 0
        ):
            return False
    return True


def main() -> None:
    df = qbt.load_data()
    loaded = tqbt.load_tqqq_data()[["price", "open"]]
    proxy_1x, daily_drag = proxy_source.load_tqqq_proxy(df, 1.0, loaded=loaded)
    proxy_3x, _ = proxy_source.load_tqqq_proxy(df, 3.0, loaded=loaded)

    frozen_equity, frozen_trades, frozen_open, frozen_position = core.run_breadth(
        df, core.INITIAL_CAPITAL
    )
    frozen_control = core.run_ensemble(df, 0.0)
    prior_70_30 = core.run_ensemble(df, TREND_WEIGHT)
    zero_boost = run_ensemble(df, proxy_1x, 0.0)
    parity = {
        "frozen_equity_max_absolute_difference": float(
            (frozen_equity - frozen_control["equity"]).abs().max()
        ),
        "frozen_trade_signatures_identical": [
            (t["entry_date"], t["exit_date"], t["sell_reason"])
            for t in frozen_trades
        ] == [
            (t["entry_date"], t["exit_date"], t["sell_reason"])
            for t in frozen_control["breadth"][1]
        ],
        "zero_boost_70_30_equity_max_absolute_difference": float(
            (prior_70_30["equity"] - zero_boost["equity"]).abs().max()
        ),
        "zero_boost_trade_signatures_identical": [
            (t["entry_date"], t["exit_date"], t["sell_reason"])
            for t in prior_70_30["breadth"][1]
        ] == [
            (t["entry_date"], t["exit_date"], t["sell_reason"])
            for t in zero_boost["breadth"][1]
        ],
    }
    parity["passed"] = bool(
        parity["frozen_equity_max_absolute_difference"] < 1e-8
        and parity["frozen_trade_signatures_identical"]
        and parity["zero_boost_70_30_equity_max_absolute_difference"] < 1e-8
        and parity["zero_boost_trade_signatures_identical"]
    )
    if not parity["passed"]:
        raise AssertionError(f"parity failed: {parity}")

    baseline_run = {
        "equity": frozen_equity,
        "position": frozen_position,
        "breadth": (frozen_equity, frozen_trades, frozen_open, frozen_position),
        "trend": core.run_ma200(df, 0.0),
        "events": core.component_events(
            "breadth", frozen_trades, frozen_equity, frozen_equity
        ),
        "clustered_trades": [dict(trade) for trade in frozen_trades],
    }
    baseline = evaluate(df, baseline_run)
    baseline["score"] = scorelib.score(
        baseline,
        scorelib.efficiency(baseline),
        bootstrap_stable=True,
        sensitivity_stable=True,
    )

    runs_1x, evals_1x, pairs_1x = evaluate_family(
        df, proxy_1x, frozen_equity
    )
    family_stable = sensitivity_is_stable(
        evals_1x, pairs_1x, baseline["metrics"]
    )
    for name, evaluation in evals_1x.items():
        evaluation["score"] = score_evaluation(
            evaluation, pairs_1x[name], family_stable
        )
    score_spread = max(
        value["score"]["final_score"] for value in evals_1x.values()
    ) - min(value["score"]["final_score"] for value in evals_1x.values())
    family_stable = family_stable and score_spread <= 5
    for name, evaluation in evals_1x.items():
        evaluation["score"] = score_evaluation(
            evaluation, pairs_1x[name], family_stable
        )

    primary_run = runs_1x["10%"]
    primary = evals_1x["10%"]
    paired = pairs_1x["10%"]
    component_correlation = float(
        primary_run["breadth"][0].pct_change().fillna(0).corr(
            primary_run["trend"][0].pct_change().fillna(0)
        )
    )
    sensitivity = {
        name: {
            **{
                metric: evaluation["metrics"][metric]
                for metric in (
                    "cagr", "sharpe", "calmar", "max_drawdown",
                    "completed_trades", "profit_factor", "expectancy",
                )
            },
            "paired_annualized_mean": pairs_1x[name]["annualized_mean_difference"],
            "score": evaluation["score"]["final_score"],
        }
        for name, evaluation in evals_1x.items()
    }

    cost_stress: dict[str, Any] = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = core.run_ensemble(df, 0.0, multiplier)
        challenge_cost = run_ensemble(
            df, proxy_1x, PRIMARY_BOOST, multiplier
        )
        pair = analytics.paired_hac_and_bootstrap(
            challenge_cost["equity"], base_cost["equity"]
        )
        base_metrics = analytics.strategy_metrics(
            base_cost["equity"], base_cost["breadth"][1], base_cost["position"]
        )
        challenge_metrics = evaluate(df, challenge_cost)["metrics"]
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": base_metrics["cagr"],
            "challenger_cagr": challenge_metrics["cagr"],
            "cagr_delta": challenge_metrics["cagr"] - base_metrics["cagr"],
            "paired_annualized_mean": pair["annualized_mean_difference"],
            "paired_hac_t": pair["hac_t_stat"],
        }

    period_deltas = {}
    for period in ("early_period", "late_period", "real_breadth_period"):
        period_deltas[period] = {
            "cagr": primary[period]["cagr"] - baseline[period]["cagr"],
            "sharpe": primary[period]["sharpe"] - baseline[period]["sharpe"],
            "max_drawdown": primary[period]["max_drawdown"] - baseline[period]["max_drawdown"],
            "calmar": period_calmar(primary[period]) - period_calmar(baseline[period]),
        }

    runs_3x, evals_3x, pairs_3x = evaluate_family(
        df, proxy_3x, frozen_equity
    )
    stable_3x = sensitivity_is_stable(
        evals_3x, pairs_3x, baseline["metrics"]
    )
    for name, evaluation in evals_3x.items():
        evaluation["score"] = score_evaluation(
            evaluation, pairs_3x[name], stable_3x
        )
    scores_3x = [v["score"]["final_score"] for v in evals_3x.values()]
    stable_3x = stable_3x and max(scores_3x) - min(scores_3x) <= 5
    for name, evaluation in evals_3x.items():
        evaluation["score"] = score_evaluation(
            evaluation, pairs_3x[name], stable_3x
        )
    primary_3x = evals_3x["10%"]
    paired_3x = pairs_3x["10%"]

    bm, cm = baseline["metrics"], primary["metrics"]
    guardrails = {
        "baseline_and_zero_boost_parity": parity["passed"],
        "at_least_30_clustered_events": cm["completed_trades"] >= 30,
        "component_correlation_below_0_95": component_correlation < 0.95,
        "final_score_at_least_80": primary["score"]["final_score"] >= 80,
        "cagr_improved": cm["cagr"] > bm["cagr"],
        "sharpe_improved": cm["sharpe"] > bm["sharpe"],
        "calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_within_two_points": cm["max_drawdown"] >= bm["max_drawdown"] - 0.02,
        "paired_mean_positive": paired["annualized_mean_difference"] > 0,
        "paired_bootstrap_excludes_zero": paired["bootstrap_95_interval_annualized"][0] > 0,
        "all_periods_positive": all(
            period_deltas[period][metric] > 0
            for period in period_deltas
            for metric in ("cagr", "sharpe", "calmar")
        ),
        "five_x_paired_return_positive": cost_stress["5x"]["paired_annualized_mean"] > 0,
        "sensitivity_stable": family_stable,
        "profit_factor_above_1_2": cm["profit_factor"] > 1.2,
        "positive_expectancy": cm["expectancy"] > 0,
        "three_x_drag_paired_positive": paired_3x["annualized_mean_difference"] > 0,
        "three_x_drag_drawdown_within_two_points": primary_3x["metrics"]["max_drawdown"] >= bm["max_drawdown"] - 0.02,
        "three_x_drag_score_at_least_80": primary_3x["score"]["final_score"] >= 80,
    }
    decision = "track" if all(guardrails.values()) else "reject"

    equity = pd.DataFrame(
        {
            "frozen_baseline": frozen_equity,
            "prior_70_30": prior_70_30["equity"],
            **{f"boost_{name}": run["equity"] for name, run in runs_1x.items()},
            "primary_breadth_bucket": primary_run["breadth"][0],
            "primary_ma200_bucket": primary_run["trend"][0],
            "primary_3x_drag": runs_3x["10%"]["equity"],
        }
    )
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)
    records = []
    for name, run in runs_1x.items():
        for trade in run["breadth"][1]:
            records.append({
                "variant": name,
                "record_type": "component_trade",
                "component": "breadth",
                **trade,
            })
        for trade in run["trend"][1]:
            records.append({
                "variant": name,
                "record_type": "component_trade",
                "component": "ma200",
                **trade,
            })
        for event in run["events"]:
            records.append({"variant": name, "record_type": "component_exit", **event})
        for cluster in run["clustered_trades"]:
            records.append({"variant": name, "record_type": "21_session_cluster", **cluster})
    pd.DataFrame(records).to_csv(TRADES_FILE, index=False)

    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "actual_tqqq_start": tqbt.TQQQ_INCEPTION,
            "clean_forward_start": "2026-07-05",
            "related_prior_trials": RELATED_TRIALS,
        },
        "configuration": {
            "breadth_weight": 1 - TREND_WEIGHT,
            "ma200_weight": TREND_WEIGHT,
            "primary_washout_tqqq_boost": PRIMARY_BOOST,
            "sensitivity": list(SENSITIVITY),
            "rebalancing": "none within trade; breadth sleeve re-split only on frozen entry",
            "fill": "close signal, next-session open",
            "event_cluster_sessions": core.EVENT_CLUSTER_SESSIONS,
        },
        "baseline_parity": parity,
        "baseline": baseline,
        "prior_70_30": evaluate(df, prior_70_30),
        "challenger": primary,
        "component_correlation": component_correlation,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": scorelib.efficiency(baseline),
            "challenger": scorelib.efficiency(primary),
            "interpretation": "historical-half pseudo-OOS only",
        },
        "sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "drag_stress": {
            "daily_drag_1x": daily_drag,
            "annualized_drag_1x": daily_drag * 252,
            "1x": {"metrics": cm, "score": primary["score"], "paired": paired},
            "3x": {
                "metrics": primary_3x["metrics"],
                "score": primary_3x["score"],
                "paired": paired_3x,
                "sensitivity_stable": stable_3x,
            },
        },
        "guardrails": guardrails,
        "current_state": {
            "date": df.index[-1],
            "breadth_sleeve_in": bool(primary_run["breadth"][3].iloc[-1]),
            "ma200_sleeve_in": bool(primary_run["trend"][3].iloc[-1]),
            "current_breadth_entry_boosted": bool(
                primary_run["breadth"][2]
                and primary_run["breadth"][2].get("tqqq_boosted", False)
            ),
        },
    }
    RESULTS_FILE.write_text(
        json.dumps(_jsonable(results), indent=2), encoding="utf-8"
    )
    write_report(results)
    print(json.dumps(_jsonable({
        "decision": decision,
        "baseline_score": baseline["score"],
        "challenger_score": primary["score"],
        "three_x_drag_score": primary_3x["score"],
        "baseline_metrics": bm,
        "challenger_metrics": cm,
        "paired_inference": paired,
        "period_deltas": period_deltas,
        "sensitivity": sensitivity,
        "guardrails": guardrails,
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    baseline = results["baseline"]
    challenger = results["challenger"]
    bm, cm = baseline["metrics"], challenger["metrics"]
    bs, cs = baseline["score"], challenger["score"]
    paired = results["paired_inference"]
    sensitivity_scores = ", ".join(
        f"{key}={value['score']}"
        for key, value in results["sensitivity"].items()
    )
    verdict = "Track as research challenger" if results["decision"] == "track" else "Reject"
    lines = [
        "# Backtest Verification Report — MA200 ensemble + washout TQQQ boost",
        "",
        f"## Verdict: {verdict}",
        "",
        f"The fixed 70/30 ensemble with a 10% washout-only TQQQ allocation scores **{cs['final_score']} / 100 ({cs['band']})** versus the frozen baseline **{bs['final_score']} / 100 ({bs['band']})**.",
        "",
        "## Backtest Scores",
        "",
        "| Component | Frozen baseline | Challenger | Max |",
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
        "| Metric | Frozen baseline | Challenger | Delta |",
        "|---|---:|---:|---:|",
        f"| CAGR | {bm['cagr']:.2%} | {cm['cagr']:.2%} | {cm['cagr']-bm['cagr']:+.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {cm['sharpe']:.3f} | {cm['sharpe']-bm['sharpe']:+.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {cm['sortino']:.3f} | {cm['sortino']-bm['sortino']:+.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {cm['calmar']:.3f} | {cm['calmar']-bm['calmar']:+.3f} |",
        f"| Maximum drawdown | {bm['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} | {cm['max_drawdown']-bm['max_drawdown']:+.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} | {cm['ulcer_index']-bm['ulcer_index']:+.2%} |",
        f"| Positive months | {bm['positive_months']:.2%} | {cm['positive_months']:.2%} | {cm['positive_months']-bm['positive_months']:+.2%} |",
        f"| Clustered events | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        "",
        "## Robustness",
        "",
        f"- Paired annual mean: {paired['annualized_mean_difference']:+.2%}; HAC t={paired['hac_t_stat']:.3f}, p={paired['hac_two_sided_p']:.3f}.",
        f"- Block-bootstrap 95% interval: {paired['bootstrap_95_interval_annualized']}.",
        f"- Sensitivity scores: {sensitivity_scores}.",
        f"- 5x-cost paired annual mean: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%}.",
        f"- 3x proxy-drag score: {results['drag_stress']['3x']['score']['final_score']}; paired annual mean {results['drag_stress']['3x']['paired']['annualized_mean_difference']:+.2%}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Close signals fill next-session open |",
        "| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |",
        f"| Data snooping | Present, material | At least {results['data']['related_prior_trials']:,} related trials; DSR penalty applied |",
        "| Trade independence | Adjusted | Component exits within 21 sessions form one event |",
        "| Costs | Included | Every traded leg; 1x/2x/5x/10x stress |",
        "| Synthetic TQQQ | Present before 2010 | Actual post-inception; 1x and punitive 3x drag |",
        "| Synthetic breadth | Present before 2007 | 2007+ period reported separately |",
        "| Clean forward OOS | Insufficient | Post-freeze sample is not meaningful |",
        "",
        "## Guardrails",
        "",
        *[
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in results["guardrails"].items()
        ],
        "",
        "## Decision",
        "",
        "The decision follows every pre-registered performance, inference, period, sensitivity, cost and synthetic-data guardrail. A historical pass can only be forward-tracked and does not change the frozen baseline.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
