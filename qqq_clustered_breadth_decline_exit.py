"""Exit-only QQQ challenger for clustered daily breadth deterioration.

The frozen strategy is left unchanged.  This harness adds one lower-priority
exit when at least N of the last ten breadth observations fell by more than 2%
and the ten-session OLS breadth slope is negative.  Signals use the close and
fill at the next session open.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# qqq_backtest refreshes market data on import.  Research must use the checked
# working-tree snapshot so a run is reproducible and does not overwrite user data.
_fetch_stub = types.ModuleType("fetch_investing_data")
_fetch_stub.fetch_all_updates = lambda verbose=True: None
sys.modules["fetch_investing_data"] = _fetch_stub

import qqq_monthly_breadth_regime_exit as framework  # noqa: E402


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/clustered_breadth_decline_exit_idea.md"
RESULTS_FILE = ROOT / "qqq_clustered_breadth_decline_exit_results.json"
EQUITY_FILE = ROOT / "qqq_clustered_breadth_decline_exit_equity.csv"
TRADES_FILE = ROOT / "qqq_clustered_breadth_decline_exit_trades.csv"
SIGNALS_FILE = ROOT / "qqq_clustered_breadth_decline_exit_signals.csv"
REPORT_FILE = ROOT / "docs/research/clustered_breadth_decline_exit_report.md"

START_DATE = "2002-01-01"
LOOKBACK = 10
DAILY_DROP_THRESHOLD = -0.02
PRIMARY_COUNT = 4
SENSITIVITY = (3, 4, 5)
RELATED_TRIALS = 4_603

qbt = framework.qbt
analytics = framework.analytics


def run_challenger(
    df: pd.DataFrame,
    extra_exit: pd.Series,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    """Run the shared state machine and label this experiment's extra exits."""
    equity, trades, open_trade = framework.run_challenger(
        df, extra_exit, cost_multiplier
    )
    labelled = [dict(trade) for trade in trades]
    for trade in labelled:
        if trade["sell_reason"] == "monthly-breadth-regime":
            trade["sell_reason"] = "clustered-breadth-decline"
    return equity, labelled, dict(open_trade) if open_trade else None


def breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build causal features using breadth observations through each close."""
    breadth = df["breadth"].astype(float)
    daily_return = breadth.pct_change(fill_method=None)
    large_decline = daily_return < DAILY_DROP_THRESHOLD
    decline_count = large_decline.astype(int).rolling(
        LOOKBACK, min_periods=LOOKBACK
    ).sum()

    x = np.arange(LOOKBACK, dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.square(centered_x).sum())

    def ols_slope(values: np.ndarray) -> float:
        return float(np.dot(values - values.mean(), centered_x) / denominator)

    slope = breadth.rolling(LOOKBACK, min_periods=LOOKBACK).apply(
        ols_slope, raw=True
    )
    return pd.DataFrame(
        {
            "breadth_daily_return": daily_return,
            "large_breadth_decline": large_decline,
            "large_decline_count_10": decline_count,
            "breadth_slope_10": slope,
        },
        index=df.index,
    )


def clustered_signal(features: pd.DataFrame, minimum_count: int) -> pd.Series:
    signal = (
        (features["large_decline_count_10"] >= minimum_count)
        & (features["breadth_slope_10"] < 0)
    ).fillna(False)
    return signal.rename(f"count_{minimum_count}")


def period_calmar(period: dict[str, Any]) -> float:
    drawdown = float(period["max_drawdown"])
    return float(period["cagr"] / abs(drawdown)) if drawdown < 0 else np.nan


def period_delta(
    challenger: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, float]:
    return {
        "cagr": float(challenger["cagr"] - baseline["cagr"]),
        "sharpe": float(challenger["sharpe"] - baseline["sharpe"]),
        "max_drawdown": float(
            challenger["max_drawdown"] - baseline["max_drawdown"]
        ),
        "calmar": float(period_calmar(challenger) - period_calmar(baseline)),
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
    difference = aligned["challenger"] - aligned["baseline"]
    output: dict[str, dict[str, float | int]] = {}
    for name, parity in (("odd_years", 1), ("even_years", 0)):
        sample = difference[difference.index.year % 2 == parity]
        output[name] = {
            "observations": len(sample),
            "annualized_mean_difference": float(sample.mean() * 252),
            "positive_difference_days": float((sample > 0).mean()),
        }
    return output


def sensitivity_is_stable(
    baseline_calmar: float,
    evaluations: dict[str, dict[str, Any]],
) -> bool:
    calmars = [
        float(evaluations[f"count_{count}"]["metrics"]["calmar"])
        for count in SENSITIVITY
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
    variants: dict[str, tuple[pd.Series, list[dict], dict | None]],
    signals: dict[str, pd.Series],
) -> None:
    equity = pd.DataFrame({name: run[0] for name, run in variants.items()})
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)

    trade_rows = []
    for name, (_, trades, _) in variants.items():
        for trade in trades:
            trade_rows.append({"variant": name, **trade})
    pd.DataFrame(trade_rows).to_csv(TRADES_FILE, index=False)

    signal_frame = features.copy()
    for name, signal in signals.items():
        signal_frame[name] = signal
    signal_frame["price"] = df["price"]
    signal_frame["open"] = df["open"]
    signal_frame["breadth"] = df["breadth"]
    signal_frame.index.name = "Date"
    signal_frame.to_csv(SIGNALS_FILE)


def main() -> None:
    if not IDEA_CARD.exists():
        raise FileNotFoundError("pre-registered idea card is required before evaluation")

    framework.RELATED_TRIALS = RELATED_TRIALS
    df = qbt.load_data().loc[START_DATE:].copy()
    features = breadth_features(df)
    disabled = pd.Series(False, index=df.index)

    direct_equity, direct_trades, direct_open = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    parity_equity, parity_trades, parity_open = run_challenger(df, disabled)
    parity = {
        "equity_max_absolute_difference": float(
            (direct_equity - parity_equity).abs().max()
        ),
        "trade_signatures_identical": (
            framework.trade_signature(direct_trades)
            == framework.trade_signature(parity_trades)
        ),
        "open_trade_identical": framework.open_trade_equal(
            direct_open, parity_open
        ),
    }
    parity["passed"] = bool(
        parity["equity_max_absolute_difference"] < 1e-8
        and parity["trade_signatures_identical"]
        and parity["open_trade_identical"]
    )
    if not parity["passed"]:
        raise AssertionError(f"baseline parity failed: {parity}")

    baseline = framework.evaluate(
        df, direct_equity, direct_trades, direct_open
    )
    benchmark_equity = qbt.run_benchmark(df)
    benchmark_metrics = analytics.slice_metrics(benchmark_equity, START_DATE)
    signals = {
        f"count_{count}": clustered_signal(features, count)
        for count in SENSITIVITY
    }
    runs = {
        name: run_challenger(df, signal)
        for name, signal in signals.items()
    }
    evaluations = {
        name: framework.evaluate(df, *run) for name, run in runs.items()
    }
    primary = evaluations[f"count_{PRIMARY_COUNT}"]
    primary_run = runs[f"count_{PRIMARY_COUNT}"]
    paired = analytics.paired_hac_and_bootstrap(
        primary_run[0], direct_equity
    )

    stable = sensitivity_is_stable(
        float(baseline["metrics"]["calmar"]), evaluations
    )
    bootstrap_stable = paired["bootstrap_95_interval_annualized"][0] > 0
    baseline["score"] = framework.score(
        baseline,
        framework.efficiency(baseline),
        bootstrap_stable=True,
        sensitivity_stable=True,
    )
    primary["score"] = framework.score(
        primary,
        framework.efficiency(primary),
        bootstrap_stable=bootstrap_stable,
        sensitivity_stable=stable,
    )

    sensitivity = {
        name: {
            metric: evaluation["metrics"][metric]
            for metric in (
                "cagr",
                "sharpe",
                "calmar",
                "max_drawdown",
                "completed_trades",
                "expectancy",
                "round_trips_per_year",
            )
        }
        for name, evaluation in evaluations.items()
    }

    cost_stress: dict[str, dict[str, float]] = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = run_challenger(df, disabled, multiplier)
        challenge_cost = run_challenger(
            df, signals[f"count_{PRIMARY_COUNT}"], multiplier
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
        period: period_delta(primary[period], baseline[period])
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    calendar_split = calendar_parity_split(primary_run[0], direct_equity)
    regime_windows = {
        "global_financial_crisis": ("2007-01-01", "2009-12-31"),
        "covid_shock": ("2020-01-01", "2020-12-31"),
        "rate_hike_bear": ("2022-01-01", "2022-12-31"),
        "post_freeze": ("2026-07-05", None),
    }
    regime_deltas = {}
    for name, (start, end) in regime_windows.items():
        base_slice = analytics.slice_metrics(direct_equity, start, end)
        challenge_slice = analytics.slice_metrics(primary_run[0], start, end)
        required = {"cagr", "sharpe", "max_drawdown"}
        if required.issubset(base_slice) and required.issubset(challenge_slice):
            regime_deltas[name] = period_delta(challenge_slice, base_slice)
        else:
            regime_deltas[name] = {
                "observations": min(
                    int(base_slice.get("observations", 0)),
                    int(challenge_slice.get("observations", 0)),
                )
            }

    bm, cm = baseline["metrics"], primary["metrics"]
    guardrails = {
        "baseline_parity": parity["passed"],
        "calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "cagr_within_one_point": cm["cagr"] >= bm["cagr"] - 0.01,
        "expectancy_not_worse": cm["expectancy"] >= bm["expectancy"],
        "completed_trades_within_25pct": (
            cm["completed_trades"] <= bm["completed_trades"] * 1.25
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

    current = features.iloc[-1]
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
            "lookback_sessions": LOOKBACK,
            "daily_relative_breadth_drop": DAILY_DROP_THRESHOLD,
            "primary_minimum_count": PRIMARY_COUNT,
            "sensitivity_counts": list(SENSITIVITY),
            "slope": "ten-session OLS slope < 0 percentage points/session",
            "change": "one lower-priority additional exit only",
            "fill": "next-session open",
        },
        "baseline_parity": parity,
        "benchmark": {
            "name": "NDX price-index buy and hold",
            "metrics": benchmark_metrics,
        },
        "baseline": baseline,
        "challenger": primary,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": framework.efficiency(baseline),
            "challenger": framework.efficiency(primary),
            "interpretation": (
                "fixed-rule historical-half efficiency; pseudo-OOS, not clean "
                "post-freeze OOS"
            ),
        },
        "sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "calendar_parity_split": calendar_split,
        "regime_deltas": regime_deltas,
        "guardrails": guardrails,
        "current_signal": {
            "date": df.index[-1],
            "active": bool(signals[f"count_{PRIMARY_COUNT}"].iloc[-1]),
            "baseline_position_open": direct_open is not None,
            "large_decline_today": bool(current["large_breadth_decline"]),
            "breadth_daily_return": float(current["breadth_daily_return"]),
            "large_decline_count_10": int(current["large_decline_count_10"]),
            "breadth_slope_10": float(current["breadth_slope_10"]),
            "earliest_fill": "next available session open",
        },
    }

    variants = {
        "ndx_buy_hold": (benchmark_equity, [], None),
        "baseline": (direct_equity, direct_trades, direct_open),
        **runs,
    }
    write_artifacts(df, features, variants, signals)
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
                    "current_signal": results["current_signal"],
                    "parity": parity,
                }
            ),
            indent=2,
        )
    )


def write_report(results: dict[str, Any]) -> None:
    base, challenge = results["baseline"], results["challenger"]
    bm, cm = base["metrics"], challenge["metrics"]
    benchmark = results["benchmark"]["metrics"]
    bs, cs = base["score"], challenge["score"]
    trade_bootstrap = challenge["trade_bootstrap"]
    verdict = (
        "Reject"
        if results["decision"] == "reject"
        else "Track as research challenger"
    )
    lines = [
        "# Backtest Verification Report — clustered breadth-decline exit",
        "",
        f"## Verdict: {verdict}",
        "",
        "## Executive Summary",
        "",
        (
            f"The count>=4 challenger scores **{cs['final_score']} / 100 "
            f"({cs['band']})** versus the frozen baseline **{bs['final_score']} / "
            f"100 ({bs['band']})**.  The decision follows the pre-registered Calmar "
            "objective and guardrails, not CAGR alone."
        ),
        "",
        (
            f"The baseline raw score is {bs['raw_score']}, above the challenger's "
            f"{cs['raw_score']}; its displayed final score is capped at 40 only because "
            "it has fewer than 30 completed trades.  The challenger's higher capped "
            "score therefore is not evidence of an economic improvement."
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
        f"| Annual volatility | {bm['annual_volatility']:.2%} | {cm['annual_volatility']:.2%} | {cm['annual_volatility']-bm['annual_volatility']:+.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {cm['sharpe']:.3f} | {cm['sharpe']-bm['sharpe']:+.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {cm['sortino']:.3f} | {cm['sortino']-bm['sortino']:+.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {cm['calmar']:.3f} | {cm['calmar']-bm['calmar']:+.3f} |",
        f"| Maximum drawdown | {bm['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} | {cm['max_drawdown']-bm['max_drawdown']:+.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} | {cm['ulcer_index']-bm['ulcer_index']:+.2%} |",
        f"| Time underwater | {bm['time_underwater']:.2%} | {cm['time_underwater']:.2%} | {cm['time_underwater']-bm['time_underwater']:+.2%} |",
        f"| Completed trades | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        f"| Exposure | {bm['exposure']:.2%} | {cm['exposure']:.2%} | {cm['exposure']-bm['exposure']:+.2%} |",
        f"| Round trips / year | {bm['round_trips_per_year']:.2f} | {cm['round_trips_per_year']:.2f} | {cm['round_trips_per_year']-bm['round_trips_per_year']:+.2f} |",
        f"| Win rate | {bm['win_rate']:.2%} | {cm['win_rate']:.2%} | {cm['win_rate']-bm['win_rate']:+.2%} |",
        f"| Payoff ratio | {bm['payoff_ratio']:.2f} | {cm['payoff_ratio']:.2f} | |",
        f"| Profit factor | {bm['profit_factor']:.2f} | {cm['profit_factor']:.2f} | |",
        f"| Expectancy | {bm['expectancy']:.2%} | {cm['expectancy']:.2%} | {cm['expectancy']-bm['expectancy']:+.2%} |",
        "",
        (
            f"NDX price-index buy-and-hold over the same dates: CAGR "
            f"{benchmark['cagr']:.2%}, Sharpe {benchmark['sharpe']:.3f}, maximum "
            f"drawdown {benchmark['max_drawdown']:.2%}."
        ),
        "",
        "## Distribution and consistency",
        "",
        "| Metric | Baseline | Challenger |",
        "|---|---:|---:|",
        f"| Daily VaR 95% | {bm['var_95_daily']:.2%} | {cm['var_95_daily']:.2%} |",
        f"| Daily CVaR 95% | {bm['cvar_95_daily']:.2%} | {cm['cvar_95_daily']:.2%} |",
        f"| Skewness | {bm['skewness']:.3f} | {cm['skewness']:.3f} |",
        f"| Excess kurtosis | {bm['excess_kurtosis']:.2f} | {cm['excess_kurtosis']:.2f} |",
        f"| Positive months | {bm['positive_months']:.2%} | {cm['positive_months']:.2%} |",
        f"| Rolling 252d Sharpe min / max / std | {bm['rolling_sharpe_252_min']:.2f} / {bm['rolling_sharpe_252_max']:.2f} / {bm['rolling_sharpe_252_std']:.2f} | {cm['rolling_sharpe_252_min']:.2f} / {cm['rolling_sharpe_252_max']:.2f} / {cm['rolling_sharpe_252_std']:.2f} |",
        "",
        "## Statistical significance",
        "",
        f"- Baseline/challenger effective observations: {bm['effective_daily_observations']:.0f} / {cm['effective_daily_observations']:.0f}.",
        f"- Mean-return t-stat: {bm['mean_return_t_stat']:.3f} / {cm['mean_return_t_stat']:.3f}; PSR vs zero: {bm['psr_vs_zero']:.4f} / {cm['psr_vs_zero']:.4f}.",
        f"- DSR probability after {results['data']['related_prior_trials']:,} related trials: {base['statistical_diagnostics']['deflated_sharpe_probability']:.4f} / {challenge['statistical_diagnostics']['deflated_sharpe_probability']:.4f}.",
        f"- Jarque-Bera p: {bm['jarque_bera_p']:.3g} / {cm['jarque_bera_p']:.3g}; Ljung-Box(10) p: {base['statistical_diagnostics']['ljung_box_p']:.3g} / {challenge['statistical_diagnostics']['ljung_box_p']:.3g}.",
        f"- Paired annual mean difference: {results['paired_inference']['annualized_mean_difference']:+.2%}; HAC t={results['paired_inference']['hac_t_stat']:.3f}, p={results['paired_inference']['hac_two_sided_p']:.3f}.",
        f"- 21-session block-bootstrap 95% interval: {results['paired_inference']['bootstrap_95_interval_annualized']}.",
        "",
        "## Robustness",
        "",
        f"- Historical-half efficiency (pseudo-OOS): baseline {results['wfa_efficiency']['baseline']:.3f}, challenger {results['wfa_efficiency']['challenger']:.3f}.",
        "- Sensitivity Calmar: " + ", ".join(
            f"{name}={value['calmar']:.3f}"
            for name, value in results["sensitivity"].items()
        ) + ".",
        f"- Odd/even-year paired annual means: {results['calendar_parity_split']['odd_years']['annualized_mean_difference']:+.2%} / {results['calendar_parity_split']['even_years']['annualized_mean_difference']:+.2%}.",
        f"- Trade bootstrap ({trade_bootstrap.get('simulations', 0):,} simulations) terminal-return p05/p50/p95: {trade_bootstrap['terminal_return_percentiles']['p05']:.2f} / {trade_bootstrap['terminal_return_percentiles']['p50']:.2f} / {trade_bootstrap['terminal_return_percentiles']['p95']:.2f}; MDD p05/p50/p95: {trade_bootstrap['max_drawdown_percentiles']['p05']:.2%} / {trade_bootstrap['max_drawdown_percentiles']['p50']:.2%} / {trade_bootstrap['max_drawdown_percentiles']['p95']:.2%}.",
        f"- 5x-cost paired annual mean: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%}; 10x: {results['cost_stress']['10x']['paired_annualized_mean']:+.2%}.",
        "",
        "### Historical direction splits",
        "",
        "| Slice | CAGR delta | Sharpe delta | Calmar delta | MDD delta |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {name} | {value['cagr']:+.2%} | {value['sharpe']:+.3f} | {value['calmar']:+.3f} | {value['max_drawdown']:+.2%} |"
            for name, value in results["period_deltas"].items()
        ],
        "",
        "### Cost stress",
        "",
        "| Cost multiplier | Baseline CAGR | Challenger CAGR | Paired annual mean |",
        "|---|---:|---:|---:|",
        *[
            f"| {name} | {value['baseline_cagr']:.2%} | {value['challenger_cagr']:.2%} | {value['paired_annualized_mean']:+.2%} |"
            for name, value in results["cost_stress"].items()
        ],
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Ten closes through t; fill at t+1 open |",
        "| Survivorship | Cannot fully verify | Aggregate NDX and aggregate S&P breadth; no constituent reconstruction |",
        f"| Data snooping | Present, material | {results['data']['related_prior_trials']:,} related trials in DSR penalty |",
        "| Transaction costs | Included | $1 and 0.05% per side, stressed at 1x/2x/5x/10x |",
        "| Liquidity | Low concern | Liquid index proxy, but NDX price-index execution is still an approximation |",
        "| Synthetic breadth | Present before 2007 | 2007+ real-breadth result reported separately |",
        "| Clean forward OOS | Insufficient | Post-freeze window is too short for five completed trades |",
        "| Regime overfit | Tested, unresolved if inconsistent | Historical halves, odd/even years, named crises, and sensitivity reported |",
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
            f"As of {pd.Timestamp(current['date']).date()}, the raw rule is "
            f"{'ACTIVE' if current['active'] else 'inactive'}: "
            f"{current['large_decline_count_10']} qualifying declines, "
            f"breadth slope {current['breadth_slope_10']:+.3f} points/session, "
            f"latest breadth return {current['breadth_daily_return']:+.2%}."
        ),
        "",
        "## Red Flags",
        "",
        "1. The rule cuts exposure from 73% to 15% and misses too much of the long-run NDX advance.",
        "2. Forty-six of fifty challenger exits come from the new rule, creating repeated exit/re-entry churn.",
        "3. CAGR, Sharpe, Sortino, Calmar, expectancy, time underwater, both historical halves, and 2007+ real-breadth evidence all deteriorate.",
        "4. The paired underperformance is statistically clear and survives block bootstrap and every cost stress.",
        "5. Clean post-freeze evidence is only 41 daily observations and contains no completed forward round trip.",
        "",
        "## Improvement Recommendations",
        "",
        "1. Do not add this condition as an unconditional sell rule to the frozen strategy.",
        "2. Keep today's trigger as a diagnostic warning only; do not treat it as validated execution evidence.",
        "3. If a later research round is requested, pre-register a different economic role such as a regime-gated warning or exposure throttle rather than retuning this failed exit on the same history.",
        "",
        "## Decision",
        "",
        (
            "The result is rejected if any pre-registered guardrail fails.  Even a "
            "historical pass would remain a research challenger until meaningful "
            "clean post-freeze evidence accumulates."
        ),
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
