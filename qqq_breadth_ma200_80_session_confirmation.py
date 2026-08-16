"""Confirm the selected 80-session washout-boost rule under stricter checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import qqq_backtest as qbt
import qqq_breadth_ma200_core_satellite as core
import qqq_breadth_ma200_timed_washout_boost as timed
import qqq_breadth_tqqq_ma200_satellite as proxy_source
import qqq_monthly_breadth_regime_exit as scorelib
import qqq_vector_crash_exit as analytics
import qqq_vector_recross_filter as research
import tqqq_backtest as tqbt


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/breadth_ma200_80_session_confirmation_idea.md"
RESULTS_FILE = ROOT / "qqq_breadth_ma200_80_session_confirmation_results.json"
EQUITY_FILE = ROOT / "qqq_breadth_ma200_80_session_confirmation_equity.csv"
TRADES_FILE = ROOT / "qqq_breadth_ma200_80_session_confirmation_trades.csv"
REPORT_FILE = ROOT / "docs/research/breadth_ma200_80_session_confirmation_report.md"

PRIMARY_SESSIONS = 80
SENSITIVITY = (60, 80, 100)
RELATED_TRIALS = 4_600
ACTUAL_TQQQ_START = pd.Timestamp(tqbt.TQQQ_INCEPTION)


def _jsonable(value: Any) -> Any:
    return timed._jsonable(value)


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
        "actual_tqqq_period": analytics.slice_metrics(
            run["equity"], str(ACTUAL_TQQQ_START.date())
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
        "completed_rotations": sum(
            trade.get("rotation_date") is not None for trade in run["breadth"][1]
        ),
    }


def period_calmar(period: dict[str, Any]) -> float:
    return period["cagr"] / abs(period["max_drawdown"])


def period_delta(
    challenger: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    return {
        "cagr": challenger["cagr"] - baseline["cagr"],
        "sharpe": challenger["sharpe"] - baseline["sharpe"],
        "max_drawdown": challenger["max_drawdown"] - baseline["max_drawdown"],
        "calmar": period_calmar(challenger) - period_calmar(baseline),
    }


def calendar_parity_split(
    challenger: pd.Series, baseline: pd.Series
) -> dict[str, dict[str, float | int]]:
    aligned = pd.concat(
        [
            challenger.pct_change().rename("challenger"),
            baseline.pct_change().rename("baseline"),
        ],
        axis=1,
    ).dropna()
    aligned["difference"] = aligned["challenger"] - aligned["baseline"]
    output = {}
    for name, parity in (("odd_years", 1), ("even_years", 0)):
        sample = aligned.loc[aligned.index.year % 2 == parity, "difference"]
        output[name] = {
            "observations": len(sample),
            "annualized_mean_difference": float(sample.mean() * 252),
            "positive_days": float((sample > 0).mean()),
        }
    return output


def evaluate_family(
    df: pd.DataFrame,
    proxy: pd.DataFrame,
    frozen_equity: pd.Series,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    runs = {
        f"{sessions}_sessions": timed.run_ensemble(df, proxy, sessions)
        for sessions in SENSITIVITY
    }
    evaluations = {name: evaluate(df, run) for name, run in runs.items()}
    paired = {
        name: analytics.paired_hac_and_bootstrap(run["equity"], frozen_equity)
        for name, run in runs.items()
    }
    return runs, evaluations, paired


def family_is_stable(
    evaluations: dict[str, dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
) -> bool:
    ordered = [
        evaluations[f"{sessions}_sessions"]["metrics"]
        for sessions in SENSITIVITY
    ]
    if not all(
        metrics["cagr"] > 0
        and metrics["sharpe"] > 1.0
        and metrics["calmar"] > 0.5
        and pairs[f"{sessions}_sessions"]["annualized_mean_difference"] > 0
        for sessions, metrics in zip(SENSITIVITY, ordered)
    ):
        return False
    for left, right in zip(ordered, ordered[1:]):
        for metric in ("cagr", "sharpe", "calmar"):
            if abs(right[metric] / left[metric] - 1) >= 0.20:
                return False
    return True


def score_family(
    evaluations: dict[str, dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
    stable: bool,
) -> bool:
    for name, evaluation in evaluations.items():
        evaluation["score"] = scorelib.score(
            evaluation,
            scorelib.efficiency(evaluation),
            bootstrap_stable=(
                pairs[name]["bootstrap_95_interval_annualized"][0] > 0
            ),
            sensitivity_stable=stable,
        )
    scores = [value["score"]["final_score"] for value in evaluations.values()]
    stable = stable and max(scores) - min(scores) <= 5
    for name, evaluation in evaluations.items():
        evaluation["score"] = scorelib.score(
            evaluation,
            scorelib.efficiency(evaluation),
            bootstrap_stable=(
                pairs[name]["bootstrap_95_interval_annualized"][0] > 0
            ),
            sensitivity_stable=stable,
        )
    return stable


def main() -> None:
    df = qbt.load_data()
    loaded = tqbt.load_tqqq_data()[["price", "open"]]
    proxy_1x, daily_drag = proxy_source.load_tqqq_proxy(df, 1.0, loaded=loaded)
    proxy_3x, _ = proxy_source.load_tqqq_proxy(df, 3.0, loaded=loaded)

    frozen_equity, frozen_trades, frozen_open, frozen_position = core.run_breadth(
        df, core.INITIAL_CAPITAL
    )
    frozen_control = core.run_ensemble(df, 0.0)
    prior_engine = timed.run_ensemble(df, proxy_1x, PRIMARY_SESSIONS)
    confirmation = timed.run_ensemble(df, proxy_1x, PRIMARY_SESSIONS)
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
        "prior_engine_equity_max_absolute_difference": float(
            (prior_engine["equity"] - confirmation["equity"]).abs().max()
        ),
        "prior_engine_trade_signatures_identical": [
            (
                t["entry_date"], t["exit_date"], t["sell_reason"],
                t.get("rotation_date"),
            )
            for t in prior_engine["breadth"][1]
        ] == [
            (
                t["entry_date"], t["exit_date"], t["sell_reason"],
                t.get("rotation_date"),
            )
            for t in confirmation["breadth"][1]
        ],
    }
    parity["passed"] = bool(
        parity["frozen_equity_max_absolute_difference"] < 1e-8
        and parity["frozen_trade_signatures_identical"]
        and parity["prior_engine_equity_max_absolute_difference"] < 1e-8
        and parity["prior_engine_trade_signatures_identical"]
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

    runs_1x, evals_1x, pairs_1x = evaluate_family(df, proxy_1x, frozen_equity)
    stable_1x = score_family(
        evals_1x, pairs_1x, family_is_stable(evals_1x, pairs_1x)
    )
    primary_run = runs_1x["80_sessions"]
    primary = evals_1x["80_sessions"]
    paired = pairs_1x["80_sessions"]
    correlation = float(
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
            "completed_rotations": evaluation["completed_rotations"],
            "paired_annualized_mean": pairs_1x[name]["annualized_mean_difference"],
            "paired_bootstrap_95_interval": pairs_1x[name]["bootstrap_95_interval_annualized"],
            "score": evaluation["score"]["final_score"],
        }
        for name, evaluation in evals_1x.items()
    }

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = core.run_ensemble(df, 0.0, multiplier)
        challenge_cost = timed.run_ensemble(
            df, proxy_1x, PRIMARY_SESSIONS, multiplier
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

    periods = (
        "early_period", "late_period", "real_breadth_period", "actual_tqqq_period"
    )
    period_deltas = {
        period: period_delta(primary[period], baseline[period])
        for period in periods
    }
    calendar_parity = calendar_parity_split(primary_run["equity"], frozen_equity)

    runs_3x, evals_3x, pairs_3x = evaluate_family(df, proxy_3x, frozen_equity)
    stable_3x = score_family(
        evals_3x, pairs_3x, family_is_stable(evals_3x, pairs_3x)
    )
    primary_3x = evals_3x["80_sessions"]
    paired_3x = pairs_3x["80_sessions"]

    bm, cm = baseline["metrics"], primary["metrics"]
    guardrails = {
        "frozen_and_prior_engine_parity": parity["passed"],
        "at_least_30_clustered_events": cm["completed_trades"] >= 30,
        "component_correlation_below_0_95": correlation < 0.95,
        "final_score_at_least_80": primary["score"]["final_score"] >= 80,
        "no_hard_cap": primary["score"]["hard_cap"] == 100,
        "cagr_improved": cm["cagr"] > bm["cagr"],
        "sharpe_improved": cm["sharpe"] > bm["sharpe"],
        "calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_below_30pct": abs(cm["max_drawdown"]) < 0.30,
        "max_drawdown_within_two_points": cm["max_drawdown"] >= bm["max_drawdown"] - 0.02,
        "paired_mean_positive": paired["annualized_mean_difference"] > 0,
        "paired_hac_p_below_0_05": paired["hac_two_sided_p"] < 0.05,
        "paired_bootstrap_excludes_zero": paired["bootstrap_95_interval_annualized"][0] > 0,
        "all_periods_positive": all(
            period_deltas[period][metric] > 0
            for period in period_deltas
            for metric in ("cagr", "sharpe", "calmar")
        ),
        "odd_even_years_positive": all(
            value["annualized_mean_difference"] > 0
            for value in calendar_parity.values()
        ),
        "five_x_paired_return_positive": cost_stress["5x"]["paired_annualized_mean"] > 0,
        "sensitivity_stable": stable_1x,
        "profit_factor_above_1_2": cm["profit_factor"] > 1.2,
        "positive_expectancy": cm["expectancy"] > 0,
        "three_x_drag_paired_positive": paired_3x["annualized_mean_difference"] > 0,
        "three_x_drag_drawdown_below_30pct": abs(primary_3x["metrics"]["max_drawdown"]) < 0.30,
        "three_x_drag_score_at_least_80": primary_3x["score"]["final_score"] >= 80,
    }
    decision = "track" if all(guardrails.values()) else "reject"

    equity = pd.DataFrame(
        {
            "frozen_baseline": frozen_equity,
            **{name: run["equity"] for name, run in runs_1x.items()},
            "primary_breadth_bucket": primary_run["breadth"][0],
            "primary_ma200_bucket": primary_run["trend"][0],
            "primary_3x_drag": runs_3x["80_sessions"]["equity"],
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
            if trade.get("rotation_date") is not None:
                records.append({
                    "variant": name,
                    "record_type": "tqqq_to_ndx_rotation",
                    "component": "breadth",
                    "entry_date": trade["entry_date"],
                    "signal_date": trade["rotation_signal_date"],
                    "exit_date": trade["rotation_date"],
                    "buy_trigger": trade["buy_trigger"],
                    "max_boost_sessions": trade["max_boost_sessions"],
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
        "selection_status": "selected from prior sensitivity; confirmatory, not blind",
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "actual_tqqq_start": ACTUAL_TQQQ_START,
            "clean_forward_start": "2026-07-05",
            "related_prior_trials": RELATED_TRIALS,
        },
        "configuration": {
            "breadth_weight": 0.70,
            "ma200_weight": 0.30,
            "washout_tqqq_boost": timed.BOOST,
            "primary_max_boost_sessions": PRIMARY_SESSIONS,
            "sensitivity_sessions": list(SENSITIVITY),
            "rotation": "age close signal, next-session open TQQQ sale and NDX buy",
            "event_cluster_sessions": core.EVENT_CLUSTER_SESSIONS,
        },
        "baseline_parity": parity,
        "baseline": baseline,
        "challenger": primary,
        "component_correlation": correlation,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": scorelib.efficiency(baseline),
            "challenger": scorelib.efficiency(primary),
            "interpretation": "historical-half pseudo-OOS only",
        },
        "sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "calendar_parity_split": calendar_parity,
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
            "current_open_trade": primary_run["breadth"][2],
        },
    }
    RESULTS_FILE.write_text(
        json.dumps(_jsonable(results), indent=2), encoding="utf-8"
    )
    write_report(results)
    print(json.dumps(_jsonable({
        "decision": decision,
        "selection_status": results["selection_status"],
        "baseline_score": baseline["score"],
        "challenger_score": primary["score"],
        "three_x_drag_score": primary_3x["score"],
        "baseline_metrics": bm,
        "challenger_metrics": cm,
        "paired_inference": paired,
        "period_deltas": period_deltas,
        "calendar_parity_split": calendar_parity,
        "sensitivity": sensitivity,
        "guardrails": guardrails,
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    baseline, challenger = results["baseline"], results["challenger"]
    bm, cm = baseline["metrics"], challenger["metrics"]
    bs, cs = baseline["score"], challenger["score"]
    paired = results["paired_inference"]
    sensitivity_scores = ", ".join(
        f"{key}={value['score']}"
        for key, value in results["sensitivity"].items()
    )
    verdict = "Track as research challenger" if results["decision"] == "track" else "Reject"
    lines = [
        "# Backtest Verification Report — Fixed 80-session confirmation",
        "",
        f"## Verdict: {verdict}",
        "",
        f"The selected fixed 80-session rule scores **{cs['final_score']} / 100 ({cs['band']})** versus the frozen baseline **{bs['final_score']} / 100 ({bs['band']})**.",
        "",
        f"Selection status: **{results['selection_status']}**. Historical confirmation is not clean forward OOS.",
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
        "## Confirmation evidence",
        "",
        f"- Paired annual mean: {paired['annualized_mean_difference']:+.2%}; HAC t={paired['hac_t_stat']:.3f}, p={paired['hac_two_sided_p']:.4f}.",
        f"- Block-bootstrap 95% interval: {paired['bootstrap_95_interval_annualized']}.",
        f"- Sensitivity scores: {sensitivity_scores}.",
        f"- Odd-year paired mean: {results['calendar_parity_split']['odd_years']['annualized_mean_difference']:+.2%}; even-year: {results['calendar_parity_split']['even_years']['annualized_mean_difference']:+.2%}.",
        f"- 5x-cost paired annual mean: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%}.",
        f"- 3x proxy-drag score: {results['drag_stress']['3x']['score']['final_score']}; paired annual mean {results['drag_stress']['3x']['paired']['annualized_mean_difference']:+.2%}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Every entry, exit and age rotation fills next-session open |",
        "| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |",
        f"| Data snooping | Present, material | 80 sessions was selected from prior sensitivity; {results['data']['related_prior_trials']:,} trials in DSR |",
        "| Trade independence | Adjusted | Component exits within 21 sessions form one event |",
        "| Costs | Included | Entry, exit and two-leg rotation; 1x/2x/5x/10x stress |",
        "| Synthetic TQQQ | Present before 2010 | Actual-only period plus 1x/3x proxy drag reported |",
        "| Synthetic breadth | Present before 2007 | 2007+ period reported separately |",
        "| Clean forward OOS | Insufficient | Only a very short post-freeze sample exists |",
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
        "A complete historical pass reaches the research-score objective and supports forward tracking only. It does not justify frozen-baseline adoption without meaningful clean forward OOS evidence and an explicit user request.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
