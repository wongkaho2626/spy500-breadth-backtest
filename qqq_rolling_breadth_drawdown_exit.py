"""QQQ challenger using drawdown from the rolling 60-session breadth high."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import qqq_breadth_only_divergence_exit as base


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/rolling_breadth_drawdown_exit_idea.md"
REPORT_FILE = ROOT / "docs/research/rolling_breadth_drawdown_exit_report.md"
RESULTS_FILE = ROOT / "qqq_rolling_breadth_drawdown_exit_results.json"
EQUITY_FILE = ROOT / "qqq_rolling_breadth_drawdown_exit_equity.csv"
TRADES_FILE = ROOT / "qqq_rolling_breadth_drawdown_exit_trades.csv"
SIGNALS_FILE = ROOT / "qqq_rolling_breadth_drawdown_exit_signals.csv"
RAW_EVENTS_FILE = ROOT / "qqq_rolling_breadth_drawdown_exit_raw_events.csv"
ONSETS_FILE = ROOT / "qqq_rolling_breadth_drawdown_exit_episode_onsets.csv"

START_DATE = "2002-01-01"
LOOKBACK = 60
BREADTH_CAP = 60.0
PRIMARY_DRAWDOWN = 20.0
SENSITIVITY = (15.0, 20.0, 25.0)
RELATED_TRIALS = 4_605

qbt = base.qbt
framework = base.framework
analytics = base.analytics


def rolling_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    rolling_max = df["breadth"].rolling(LOOKBACK, min_periods=LOOKBACK).max()
    return pd.DataFrame(
        {
            "rolling_breadth_max_60": rolling_max,
            "breadth": df["breadth"],
            "breadth_drawdown_60_points": rolling_max - df["breadth"],
            "breadth_below_60": df["breadth"] < BREADTH_CAP,
            "original_price_rise_met": df["price_rose"].astype(bool),
            "canonical_divergence": (
                df["price_rose"].astype(bool)
                & df["breadth_fell"].astype(bool)
                & (df["breadth"] < BREADTH_CAP)
            ),
        },
        index=df.index,
    )


def rolling_breadth_signal(
    features: pd.DataFrame, drawdown_threshold: float
) -> pd.Series:
    return (
        (features["breadth_drawdown_60_points"] >= drawdown_threshold)
        & features["breadth_below_60"].astype(bool)
    ).fillna(False).rename(f"drawdown_{drawdown_threshold:.0f}")


def run_variant(
    df: pd.DataFrame,
    rolling_signal: pd.Series | None,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    equity, trades, open_trade = base.run_variant(
        df, rolling_signal, cost_multiplier
    )
    features = rolling_breadth_features(df)
    labelled = []
    for source in trades:
        trade = dict(source)
        if trade["sell_reason"] == "breadth-only-60d-fall":
            signal_date = trade["signal_date"]
            trade["sell_reason"] = "rolling-breadth-drawdown"
            trade["rolling_breadth_max_60"] = float(
                features.loc[signal_date, "rolling_breadth_max_60"]
            )
            trade["breadth_on_signal"] = float(
                features.loc[signal_date, "breadth"]
            )
            trade["breadth_drawdown_60_points"] = float(
                features.loc[signal_date, "breadth_drawdown_60_points"]
            )
            for key in (
                "breadth_60_sessions_ago",
                "breadth_fall_60_points",
                "ndx_return_60_pct",
            ):
                trade.pop(key, None)
        labelled.append(trade)
    return equity, labelled, open_trade


def sensitivity_is_stable(
    baseline_calmar: float,
    evaluations: dict[str, dict[str, Any]],
) -> bool:
    calmars = [
        float(evaluations[f"drawdown_{value:.0f}"]["metrics"]["calmar"])
        for value in SENSITIVITY
    ]
    if not all(value > baseline_calmar for value in calmars):
        return False
    return all(
        abs(right / left - 1) < 0.25
        for left, right in zip(calmars, calmars[1:])
        if left != 0
    )


def write_artifacts(
    df: pd.DataFrame,
    features: pd.DataFrame,
    benchmark: pd.Series,
    baseline_run: tuple[pd.Series, list[dict], dict | None],
    runs: dict[str, tuple[pd.Series, list[dict], dict | None]],
    signals: dict[str, pd.Series],
) -> None:
    equity = pd.DataFrame(
        {
            "ndx_buy_hold": benchmark,
            "baseline": baseline_run[0],
            **{name: run[0] for name, run in runs.items()},
        }
    )
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)

    rows = []
    for name, run in {"baseline": baseline_run, **runs}.items():
        rows.extend({"variant": name, **trade} for trade in run[1])
    pd.DataFrame(rows).to_csv(TRADES_FILE, index=False)

    signal_frame = features.copy()
    for name, signal in signals.items():
        signal_frame[name] = signal
    signal_frame["price"] = df["price"]
    signal_frame["next_session_open"] = df["open"].shift(-1)
    signal_frame.index.name = "Date"
    signal_frame.to_csv(SIGNALS_FILE)

    primary = signals[f"drawdown_{PRIMARY_DRAWDOWN:.0f}"]
    raw_events = signal_frame.loc[primary].copy()
    raw_events["episode_onset"] = (
        primary & ~primary.shift(1, fill_value=False)
    ).loc[raw_events.index]
    raw_events.to_csv(RAW_EVENTS_FILE)
    raw_events.loc[raw_events["episode_onset"].astype(bool)].to_csv(ONSETS_FILE)


def main() -> None:
    if not IDEA_CARD.exists():
        raise FileNotFoundError("pre-registered idea card is required")

    framework.RELATED_TRIALS = RELATED_TRIALS
    df = qbt.load_data().loc[START_DATE:].copy()
    features = rolling_breadth_features(df)
    signals = {
        f"drawdown_{value:.0f}": rolling_breadth_signal(features, value)
        for value in SENSITIVITY
    }

    direct = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    parity_run = run_variant(df, None)
    parity = {
        "equity_max_absolute_difference": float(
            (direct[0] - parity_run[0]).abs().max()
        ),
        "trade_signatures_identical": (
            framework.trade_signature(direct[1])
            == framework.trade_signature(parity_run[1])
        ),
        "open_trade_identical": framework.open_trade_equal(
            direct[2], parity_run[2]
        ),
    }
    parity["passed"] = bool(
        parity["equity_max_absolute_difference"] < 1e-8
        and parity["trade_signatures_identical"]
        and parity["open_trade_identical"]
    )
    if not parity["passed"]:
        raise AssertionError(f"baseline parity failed: {parity}")

    baseline_run = parity_run
    baseline = framework.evaluate(df, *baseline_run)
    runs = {
        name: run_variant(df, signal) for name, signal in signals.items()
    }
    evaluations = {
        name: framework.evaluate(df, *run) for name, run in runs.items()
    }
    primary_name = f"drawdown_{PRIMARY_DRAWDOWN:.0f}"
    primary_run = runs[primary_name]
    primary = evaluations[primary_name]
    paired = analytics.paired_hac_and_bootstrap(
        primary_run[0], baseline_run[0]
    )

    stable = sensitivity_is_stable(
        float(baseline["metrics"]["calmar"]), evaluations
    )
    bootstrap_stable = paired["bootstrap_95_interval_annualized"][0] > 0
    baseline["score"] = framework.score(
        baseline, framework.efficiency(baseline), True, True
    )
    primary["score"] = framework.score(
        primary,
        framework.efficiency(primary),
        bootstrap_stable,
        stable,
    )

    sensitivity = {
        name: {
            metric: evaluation["metrics"][metric]
            for metric in (
                "cagr", "sharpe", "calmar", "max_drawdown",
                "completed_trades", "expectancy", "round_trips_per_year",
            )
        }
        for name, evaluation in evaluations.items()
    }

    cost_stress: dict[str, dict[str, float]] = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = run_variant(df, None, multiplier)
        challenge_cost = run_variant(
            df, signals[primary_name], multiplier
        )
        base_eval = framework.evaluate(df, *base_cost)
        challenge_eval = framework.evaluate(df, *challenge_cost)
        cost_pair = analytics.paired_hac_and_bootstrap(
            challenge_cost[0], base_cost[0]
        )
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": float(base_eval["metrics"]["cagr"]),
            "challenger_cagr": float(challenge_eval["metrics"]["cagr"]),
            "cagr_delta": float(
                challenge_eval["metrics"]["cagr"]
                - base_eval["metrics"]["cagr"]
            ),
            "paired_annualized_mean": float(
                cost_pair["annualized_mean_difference"]
            ),
            "paired_hac_t": float(cost_pair["hac_t_stat"]),
        }

    period_deltas = {
        period: base.period_delta(primary[period], baseline[period])
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    calendar_split = base.calendar_parity_split(
        primary_run[0], baseline_run[0]
    )
    bm, cm = baseline["metrics"], primary["metrics"]
    guardrails = {
        "baseline_parity": parity["passed"],
        "calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "cagr_within_two_points": cm["cagr"] >= bm["cagr"] - 0.02,
        "expectancy_not_worse": cm["expectancy"] >= bm["expectancy"],
        "turnover_guardrail": (
            cm["completed_trades"] <= bm["completed_trades"] * 1.5
            or cm["expectancy"] > bm["expectancy"]
        ),
        "historical_halves_calmar_nonnegative": all(
            period_deltas[period]["calmar"] >= 0
            for period in ("early_period", "late_period")
        ),
        "real_breadth_calmar_nonnegative": (
            period_deltas["real_breadth_period"]["calmar"] >= 0
        ),
        "five_x_paired_return_positive": (
            cost_stress["5x"]["paired_annualized_mean"] > 0
        ),
        "sensitivity_not_cliff_edge": stable,
    }
    decision = "track" if all(guardrails.values()) else "reject"

    primary_signal = signals[primary_name]
    onsets = primary_signal & ~primary_signal.shift(1, fill_value=False)
    rolling_sells = [
        trade for trade in primary_run[1]
        if trade["sell_reason"] == "rolling-breadth-drawdown"
    ]
    benchmark = qbt.run_benchmark(df)
    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "clean_forward_start": "2026-07-05",
            "related_prior_trials": RELATED_TRIALS,
            "pre_2007_breadth": "synthetic daily splice",
        },
        "configuration": {
            "rolling_window_sessions": LOOKBACK,
            "window_includes_signal_session": True,
            "primary_breadth_drawdown_points": PRIMARY_DRAWDOWN,
            "breadth_cap": BREADTH_CAP,
            "price_rise_condition": "removed",
            "sensitivity_drawdown_points": list(SENSITIVITY),
            "fill": "next-session open",
            "change": "replace bearish-divergence exit only",
        },
        "baseline_parity": parity,
        "benchmark": {
            "name": "NDX price-index buy and hold",
            "metrics": analytics.slice_metrics(benchmark, START_DATE),
        },
        "baseline": baseline,
        "challenger": primary,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": framework.efficiency(baseline),
            "challenger": framework.efficiency(primary),
            "interpretation": "fixed-rule historical-half pseudo-OOS only",
        },
        "sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "calendar_parity_split": calendar_split,
        "guardrails": guardrails,
        "signal_counts": {
            "raw_active_days": int(primary_signal.sum()),
            "raw_episode_onsets": int(onsets.sum()),
            "executed_rolling_breadth_sells": len(rolling_sells),
            "all_executed_exit_reasons": pd.Series(
                [trade["sell_reason"] for trade in primary_run[1]]
            ).value_counts().to_dict(),
        },
        "current_signal": {
            "date": df.index[-1],
            "active": bool(primary_signal.iloc[-1]),
            "breadth": float(features["breadth"].iloc[-1]),
            "rolling_breadth_max_60": float(
                features["rolling_breadth_max_60"].iloc[-1]
            ),
            "breadth_drawdown_60_points": float(
                features["breadth_drawdown_60_points"].iloc[-1]
            ),
            "baseline_position_open": baseline_run[2] is not None,
            "challenger_position_open": primary_run[2] is not None,
            "earliest_fill": "next available session open",
        },
    }

    write_artifacts(
        df, features, benchmark, baseline_run, runs, signals
    )
    RESULTS_FILE.write_text(
        json.dumps(framework._jsonable(results), indent=2), encoding="utf-8"
    )
    write_report(results)
    print(
        json.dumps(
            framework._jsonable(
                {
                    "decision": decision,
                    "baseline_score": baseline["score"],
                    "challenger_score": primary["score"],
                    "baseline_metrics": bm,
                    "challenger_metrics": cm,
                    "paired_inference": paired,
                    "guardrails": guardrails,
                    "signal_counts": results["signal_counts"],
                    "current_signal": results["current_signal"],
                    "parity": parity,
                }
            ),
            indent=2,
        )
    )


def write_report(results: dict[str, Any]) -> None:
    base_eval, challenge = results["baseline"], results["challenger"]
    bm, cm = base_eval["metrics"], challenge["metrics"]
    bs, cs = base_eval["score"], challenge["score"]
    verdict = (
        "Reject" if results["decision"] == "reject"
        else "Track as research challenger"
    )
    lines = [
        "# Backtest Verification Report — rolling breadth-drawdown exit",
        "",
        f"## Verdict: {verdict}",
        "",
        "## Executive Summary",
        "",
        (
            f"The rolling-max challenger scores **{cs['final_score']} / 100 "
            f"({cs['band']})** versus baseline **{bs['final_score']} / 100 "
            f"({bs['band']})**.  The verdict follows the pre-registered Calmar "
            "objective and guardrails."
        ),
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
        "## Performance and risk",
        "",
        "| Metric | Baseline | Challenger | Delta |",
        "|---|---:|---:|---:|",
        f"| CAGR | {bm['cagr']:.2%} | {cm['cagr']:.2%} | {cm['cagr']-bm['cagr']:+.2%} |",
        f"| Volatility | {bm['annual_volatility']:.2%} | {cm['annual_volatility']:.2%} | {cm['annual_volatility']-bm['annual_volatility']:+.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {cm['sharpe']:.3f} | {cm['sharpe']-bm['sharpe']:+.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {cm['sortino']:.3f} | {cm['sortino']-bm['sortino']:+.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {cm['calmar']:.3f} | {cm['calmar']-bm['calmar']:+.3f} |",
        f"| Maximum drawdown | {bm['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} | {cm['max_drawdown']-bm['max_drawdown']:+.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} | {cm['ulcer_index']-bm['ulcer_index']:+.2%} |",
        f"| Time underwater | {bm['time_underwater']:.2%} | {cm['time_underwater']:.2%} | {cm['time_underwater']-bm['time_underwater']:+.2%} |",
        f"| Completed trades | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        f"| Exposure | {bm['exposure']:.2%} | {cm['exposure']:.2%} | {cm['exposure']-bm['exposure']:+.2%} |",
        f"| Win rate | {bm['win_rate']:.2%} | {cm['win_rate']:.2%} | {cm['win_rate']-bm['win_rate']:+.2%} |",
        f"| Payoff ratio | {bm['payoff_ratio']:.2f} | {cm['payoff_ratio']:.2f} | |",
        f"| Profit factor | {bm['profit_factor']:.2f} | {cm['profit_factor']:.2f} | |",
        f"| Expectancy | {bm['expectancy']:.2%} | {cm['expectancy']:.2%} | {cm['expectancy']-bm['expectancy']:+.2%} |",
        "",
        "## Statistical significance",
        "",
        f"- Effective observations: {bm['effective_daily_observations']:.0f} / {cm['effective_daily_observations']:.0f}.",
        f"- t-stat: {bm['mean_return_t_stat']:.3f} / {cm['mean_return_t_stat']:.3f}; PSR: {bm['psr_vs_zero']:.4f} / {cm['psr_vs_zero']:.4f}.",
        f"- DSR after {results['data']['related_prior_trials']:,} trials: {base_eval['statistical_diagnostics']['deflated_sharpe_probability']:.4f} / {challenge['statistical_diagnostics']['deflated_sharpe_probability']:.4f}.",
        f"- Jarque-Bera p: {bm['jarque_bera_p']:.3g} / {cm['jarque_bera_p']:.3g}; Ljung-Box p: {base_eval['statistical_diagnostics']['ljung_box_p']:.3g} / {challenge['statistical_diagnostics']['ljung_box_p']:.3g}.",
        f"- Paired annual mean: {results['paired_inference']['annualized_mean_difference']:+.2%}; HAC t={results['paired_inference']['hac_t_stat']:.3f}, p={results['paired_inference']['hac_two_sided_p']:.3f}.",
        f"- Block-bootstrap 95% interval: {results['paired_inference']['bootstrap_95_interval_annualized']}.",
        "",
        "## Robustness",
        "",
        f"- Historical-half efficiency: {results['wfa_efficiency']['baseline']:.3f} / {results['wfa_efficiency']['challenger']:.3f} (pseudo-OOS).",
        "- Sensitivity Calmar: " + ", ".join(
            f"{name}={value['calmar']:.3f}"
            for name, value in results["sensitivity"].items()
        ) + ".",
        f"- Odd/even paired means: {results['calendar_parity_split']['odd_years']['annualized_mean_difference']:+.2%} / {results['calendar_parity_split']['even_years']['annualized_mean_difference']:+.2%}.",
        f"- Trade bootstrap simulations: {challenge['trade_bootstrap'].get('simulations', 0):,}.",
        f"- 5x/10x-cost paired means: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%} / {results['cost_stress']['10x']['paired_annualized_mean']:+.2%}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Rolling window ends at signal close; next-open fill |",
        "| Survivorship | Cannot fully verify | Aggregate index and breadth series |",
        f"| Data snooping | Present, material | {results['data']['related_prior_trials']:,} trials penalised |",
        "| Costs | Included | 1x/2x/5x/10x commission and slippage |",
        "| Liquidity | Low concern | Liquid index proxy; NDX price-index approximation |",
        "| Synthetic breadth | Present before 2007 | 2007+ reported separately |",
        "| Clean forward OOS | Insufficient | Too few post-freeze observations/trades |",
        "| Regime overfit | Tested | Halves, odd/even, real-breadth, sensitivity |",
        "",
        "## Guardrails",
        "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in results["guardrails"].items()
    )
    current = results["current_signal"]
    lines += [
        "",
        "## Current signal",
        "",
        (
            f"As of {pd.Timestamp(current['date']).date()}, rolling 60-session "
            f"breadth max is {current['rolling_breadth_max_60']:.2f}% and current "
            f"breadth is {current['breadth']:.2f}%, a drawdown of "
            f"{current['breadth_drawdown_60_points']:.2f} points.  The signal is "
            f"{'ACTIVE' if current['active'] else 'inactive'}."
        ),
        "",
        "## Red Flags",
        "",
        "1. A rolling peak can keep the condition active long after the first breakdown, increasing whipsaw.",
        "2. Pre-2007 breadth is synthetic; pre-freeze results are not clean forward OOS.",
        "3. A score increase caused only by clearing the 30-trade cap is not an economic improvement.",
        "",
        "## Improvement Recommendations",
        "",
        "1. Follow the pre-registered verdict and leave the frozen baseline unchanged.",
        "2. Inspect raw events and actual fills separately; repeated active days are not independent signals.",
        "",
        "## Decision",
        "",
        "The verdict follows the primary Calmar objective and every guardrail without retuning.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
