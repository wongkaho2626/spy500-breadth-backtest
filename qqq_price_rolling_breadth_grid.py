"""Fixed grid search for price-confirmed rolling breadth exits."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import qqq_price_confirmed_rolling_breadth_exit as source


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/price_rolling_breadth_grid_idea.md"
REPORT_FILE = ROOT / "docs/research/price_rolling_breadth_grid_report.md"
RESULTS_FILE = ROOT / "qqq_price_rolling_breadth_grid_results.json"
GRID_FILE = ROOT / "qqq_price_rolling_breadth_grid.csv"
EQUITY_FILE = ROOT / "qqq_price_rolling_breadth_grid_equity.csv"
TRADES_FILE = ROOT / "qqq_price_rolling_breadth_grid_trades.csv"
SIGNALS_FILE = ROOT / "qqq_price_rolling_breadth_grid_signals.csv"

START_DATE = "2002-01-01"
LOOKBACK = 60
PRICE_GRID = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
DRAWDOWN_GRID = (10.0, 15.0, 20.0, 25.0, 30.0, 35.0)
CAP_GRID = (30.0, 40.0, 50.0, 60.0, 70.0)
RELATED_TRIALS = 4_816

qbt = source.qbt
framework = source.framework
analytics = source.analytics


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = source.build_features(df)
    return features[[
        "ndx_return_60_pct",
        "rolling_breadth_max_60",
        "breadth",
        "breadth_drawdown_60_points",
    ]].copy()


def grid_signal(
    features: pd.DataFrame,
    price_rise: float,
    breadth_drawdown: float,
    breadth_cap: float,
) -> pd.Series:
    return (
        (features["ndx_return_60_pct"] >= price_rise)
        & (features["breadth_drawdown_60_points"] >= breadth_drawdown)
        & (features["breadth"] < breadth_cap)
    ).fillna(False).rename("grid_signal")


def annotate_trades(
    df: pd.DataFrame,
    trades: list[dict],
    features: pd.DataFrame,
    params: tuple[float, float, float],
) -> list[dict]:
    price_rise, breadth_drawdown, breadth_cap = params
    output = []
    for original in trades:
        trade = dict(original)
        exit_location = df.index.get_loc(trade["exit_date"])
        signal_date = df.index[exit_location - qbt.EXECUTION_LAG]
        trade["signal_date"] = signal_date
        trade["grid_price_rise_pct"] = price_rise
        trade["grid_breadth_drawdown_points"] = breadth_drawdown
        trade["grid_breadth_cap"] = breadth_cap
        if trade["sell_reason"] == "grid-price-rolling-breadth":
            trade["ndx_return_60_pct"] = float(
                features.loc[signal_date, "ndx_return_60_pct"]
            )
            trade["rolling_breadth_max_60"] = float(
                features.loc[signal_date, "rolling_breadth_max_60"]
            )
            trade["breadth_on_signal"] = float(
                features.loc[signal_date, "breadth"]
            )
            trade["breadth_drawdown_60_points"] = float(
                features.loc[signal_date, "breadth_drawdown_60_points"]
            )
        output.append(trade)
    return output


def run_combo(
    df: pd.DataFrame,
    features: pd.DataFrame,
    params: tuple[float, float, float],
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    signal = grid_signal(features, *params)
    equity, trades, open_trade = analytics.run_replacement_exit(
        df,
        signal,
        reason="grid-price-rolling-breadth",
        commission_multiplier=cost_multiplier,
    )
    return equity, annotate_trades(df, trades, features, params), open_trade


def run_baseline_harness(
    df: pd.DataFrame,
    features: pd.DataFrame,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    equity, trades, open_trade = analytics.run_replacement_exit(
        df,
        analytics.baseline_divergence_signal(df),
        reason="bearish-divergence",
        commission_multiplier=cost_multiplier,
    )
    return equity, annotate_trades(
        df, trades, features, (qbt.DIVERGENCE_PRICE_RISE,
                               qbt.DIVERGENCE_BREADTH_FALL,
                               qbt.DIVERGENCE_BREADTH_CAP)
    ), open_trade


def segment_metrics(
    equity: pd.Series, start: str, end: str | None = None
) -> dict[str, float]:
    metrics = analytics.slice_metrics(equity, start, end)
    drawdown = float(metrics["max_drawdown"])
    calmar = (
        float(metrics["cagr"] / abs(drawdown))
        if drawdown < 0 else math.inf
    )
    return {
        "cagr": float(metrics["cagr"]),
        "sharpe": float(metrics["sharpe"]),
        "max_drawdown": drawdown,
        "calmar": calmar,
    }


def grid_row(
    df: pd.DataFrame,
    features: pd.DataFrame,
    params: tuple[float, float, float],
) -> dict[str, float | int]:
    equity, trades, open_trade = run_combo(df, features, params)
    full = segment_metrics(equity, "2002-01-01")
    early = segment_metrics(equity, "2002-01-01", "2013-12-31")
    late = segment_metrics(equity, "2014-01-01")
    real = segment_metrics(equity, "2007-01-01")
    position = analytics.position_series(df.index, trades, open_trade)
    price_rise, drawdown, cap = params
    return {
        "price_rise_pct": price_rise,
        "breadth_drawdown_points": drawdown,
        "breadth_cap": cap,
        "full_cagr": full["cagr"],
        "full_sharpe": full["sharpe"],
        "full_max_drawdown": full["max_drawdown"],
        "full_calmar": full["calmar"],
        "early_cagr": early["cagr"],
        "early_sharpe": early["sharpe"],
        "early_max_drawdown": early["max_drawdown"],
        "early_calmar": early["calmar"],
        "late_cagr": late["cagr"],
        "late_sharpe": late["sharpe"],
        "late_max_drawdown": late["max_drawdown"],
        "late_calmar": late["calmar"],
        "real_breadth_cagr": real["cagr"],
        "real_breadth_sharpe": real["sharpe"],
        "real_breadth_max_drawdown": real["max_drawdown"],
        "real_breadth_calmar": real["calmar"],
        "min_half_calmar": min(early["calmar"], late["calmar"]),
        "completed_trades": len(trades),
        "exposure": float(position.mean()),
        "raw_signal_days": int(grid_signal(features, *params).sum()),
    }


def row_params(row: pd.Series | dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["price_rise_pct"]),
        float(row["breadth_drawdown_points"]),
        float(row["breadth_cap"]),
    )


def select_winners(grid: pd.DataFrame) -> dict[str, dict[str, Any]]:
    robust = grid.sort_values(
        ["min_half_calmar", "full_calmar", "completed_trades"],
        ascending=[False, False, True],
    ).iloc[0]
    early = grid.sort_values(
        ["early_calmar", "late_calmar", "completed_trades"],
        ascending=[False, False, True],
    ).iloc[0]
    late = grid.sort_values(
        ["late_calmar", "early_calmar", "completed_trades"],
        ascending=[False, False, True],
    ).iloc[0]
    full = grid.sort_values(
        ["full_calmar", "min_half_calmar", "completed_trades"],
        ascending=[False, False, True],
    ).iloc[0]
    return {
        "robust_consensus": robust.to_dict(),
        "early_selected": early.to_dict(),
        "late_selected": late.to_dict(),
        "full_sample": full.to_dict(),
    }


def neighbour_stability(
    grid: pd.DataFrame, winner: dict[str, Any]
) -> dict[str, Any]:
    params = row_params(winner)
    grids = (PRICE_GRID, DRAWDOWN_GRID, CAP_GRID)
    boundary = any(
        value in (values[0], values[-1])
        for value, values in zip(params, grids)
    )
    neighbours = []
    for dimension, values in enumerate(grids):
        location = values.index(params[dimension])
        for offset in (-1, 1):
            neighbour_location = location + offset
            if not 0 <= neighbour_location < len(values):
                continue
            target = list(params)
            target[dimension] = values[neighbour_location]
            match = grid[
                (grid["price_rise_pct"] == target[0])
                & (grid["breadth_drawdown_points"] == target[1])
                & (grid["breadth_cap"] == target[2])
            ].iloc[0]
            stable = bool(
                match["full_calmar"] >= 0.9 * winner["full_calmar"]
                and match["early_calmar"] > 0
                and match["late_calmar"] > 0
            )
            neighbours.append({**match.to_dict(), "stable": stable})
    stable_count = sum(item["stable"] for item in neighbours)
    return {
        "candidate_on_grid_boundary": boundary,
        "neighbour_count": len(neighbours),
        "stable_neighbour_count": stable_count,
        "stable_neighbour_fraction": (
            stable_count / len(neighbours) if neighbours else 0.0
        ),
        "passed": bool(
            not boundary
            and neighbours
            and stable_count / len(neighbours) >= 0.75
        ),
        "neighbours": neighbours,
    }


def main() -> None:
    if not IDEA_CARD.exists():
        raise FileNotFoundError("pre-registered grid card is required")
    framework.RELATED_TRIALS = RELATED_TRIALS
    df = qbt.load_data().loc[START_DATE:].copy()
    features = build_features(df)

    direct = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    baseline_run = run_baseline_harness(df, features)
    parity = {
        "equity_max_absolute_difference": float(
            (direct[0] - baseline_run[0]).abs().max()
        ),
        "trade_signatures_identical": (
            framework.trade_signature(direct[1])
            == framework.trade_signature(baseline_run[1])
        ),
        "open_trade_identical": framework.open_trade_equal(
            direct[2], baseline_run[2]
        ),
    }
    parity["passed"] = bool(
        parity["equity_max_absolute_difference"] < 1e-8
        and parity["trade_signatures_identical"]
        and parity["open_trade_identical"]
    )
    if not parity["passed"]:
        raise AssertionError(f"baseline parity failed: {parity}")

    rows = [
        grid_row(df, features, params)
        for params in itertools.product(PRICE_GRID, DRAWDOWN_GRID, CAP_GRID)
    ]
    grid = pd.DataFrame(rows)
    grid.sort_values(
        ["min_half_calmar", "full_calmar"], ascending=False
    ).to_csv(GRID_FILE, index=False)
    winners = select_winners(grid)
    robust_params = row_params(winners["robust_consensus"])
    robust_run = run_combo(df, features, robust_params)
    stability = neighbour_stability(grid, winners["robust_consensus"])

    baseline = framework.evaluate(df, *baseline_run)
    challenger = framework.evaluate(df, *robust_run)
    paired = analytics.paired_hac_and_bootstrap(
        robust_run[0], baseline_run[0]
    )
    bootstrap_stable = paired["bootstrap_95_interval_annualized"][0] > 0
    baseline["score"] = framework.score(
        baseline, framework.efficiency(baseline), True, True
    )
    challenger["score"] = framework.score(
        challenger,
        framework.efficiency(challenger),
        bootstrap_stable,
        stability["passed"],
    )

    cost_stress: dict[str, dict[str, float]] = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = run_baseline_harness(df, features, multiplier)
        challenge_cost = run_combo(
            df, features, robust_params, multiplier
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
        period: source.rolling.base.period_delta(
            challenger[period], baseline[period]
        )
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    calendar_split = source.rolling.base.calendar_parity_split(
        robust_run[0], baseline_run[0]
    )
    bm, cm = baseline["metrics"], challenger["metrics"]
    guardrails = {
        "baseline_parity": parity["passed"],
        "calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "cagr_within_two_points": cm["cagr"] >= bm["cagr"] - 0.02,
        "expectancy_not_worse": cm["expectancy"] >= bm["expectancy"],
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
        "candidate_not_on_boundary": not stability[
            "candidate_on_grid_boundary"
        ],
        "local_neighbour_stability": stability["passed"],
    }
    decision = "track" if all(guardrails.values()) else "reject"

    candidate_runs: dict[str, tuple[pd.Series, list[dict], dict | None]] = {}
    for name, winner in winners.items():
        candidate_runs[name] = run_combo(df, features, row_params(winner))

    benchmark = qbt.run_benchmark(df)
    equity = pd.DataFrame(
        {
            "ndx_buy_hold": benchmark,
            "baseline": baseline_run[0],
            **{name: run[0] for name, run in candidate_runs.items()},
        }
    )
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)
    trade_rows = []
    for name, run in candidate_runs.items():
        trade_rows.extend({"selection": name, **trade} for trade in run[1])
    pd.DataFrame(trade_rows).to_csv(TRADES_FILE, index=False)

    robust_signal = grid_signal(features, *robust_params)
    signal_frame = features.copy()
    signal_frame["robust_signal"] = robust_signal
    signal_frame["next_session_open"] = df["open"].shift(-1)
    signal_frame.index.name = "Date"
    signal_frame.to_csv(SIGNALS_FILE)

    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "clean_forward_start": "2026-07-05",
            "grid_variants": len(grid),
            "related_trials_including_grid": RELATED_TRIALS,
        },
        "grid": {
            "price_rise_pct": list(PRICE_GRID),
            "breadth_drawdown_points": list(DRAWDOWN_GRID),
            "breadth_cap": list(CAP_GRID),
            "lookback_sessions": LOOKBACK,
            "selection_objective": (
                "maximise min(early_calmar, late_calmar), then full_calmar, "
                "then lower completed trades"
            ),
        },
        "baseline_parity": parity,
        "winners": winners,
        "neighbour_stability": stability,
        "benchmark": {
            "name": "NDX price-index buy and hold",
            "metrics": analytics.slice_metrics(benchmark, START_DATE),
        },
        "baseline": baseline,
        "challenger": challenger,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": framework.efficiency(baseline),
            "challenger": framework.efficiency(challenger),
            "interpretation": "historical-half pseudo-OOS; grid family is now seen",
        },
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "calendar_parity_split": calendar_split,
        "guardrails": guardrails,
        "current_signal": {
            "date": df.index[-1],
            "active": bool(robust_signal.iloc[-1]),
            "selected_params": robust_params,
            "ndx_return_60_pct": float(
                features["ndx_return_60_pct"].iloc[-1]
            ),
            "breadth": float(features["breadth"].iloc[-1]),
            "rolling_breadth_max_60": float(
                features["rolling_breadth_max_60"].iloc[-1]
            ),
            "breadth_drawdown_60_points": float(
                features["breadth_drawdown_60_points"].iloc[-1]
            ),
        },
        "top_20_robust": grid.sort_values(
            ["min_half_calmar", "full_calmar"], ascending=False
        ).head(20).to_dict(orient="records"),
    }
    RESULTS_FILE.write_text(
        json.dumps(framework._jsonable(results), indent=2), encoding="utf-8"
    )
    write_report(results)
    print(json.dumps(framework._jsonable({
        "decision": decision,
        "winners": winners,
        "neighbour_stability": stability,
        "baseline_score": baseline["score"],
        "challenger_score": challenger["score"],
        "baseline_metrics": bm,
        "challenger_metrics": cm,
        "paired_inference": paired,
        "guardrails": guardrails,
        "current_signal": results["current_signal"],
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    base_eval, challenge = results["baseline"], results["challenger"]
    bm, cm = base_eval["metrics"], challenge["metrics"]
    bs, cs = base_eval["score"], challenge["score"]
    winner = results["winners"]["robust_consensus"]
    verdict = (
        "Reject" if results["decision"] == "reject"
        else "Track as research challenger"
    )
    lines = [
        "# Backtest Verification Report — price × rolling breadth grid",
        "",
        f"## Verdict: {verdict}",
        "",
        "## Executive Summary",
        "",
        (
            f"The pre-registered 210-cell robust winner is P={winner['price_rise_pct']:.0f}%, "
            f"D={winner['breadth_drawdown_points']:.0f} points, C={winner['breadth_cap']:.0f}%. "
            f"It scores **{cs['final_score']} / 100 ({cs['band']})** versus "
            f"baseline **{bs['final_score']} / 100 ({bs['band']})**."
        ),
        "",
        "## Selected combinations",
        "",
        "| Selection | P | D | C | Full CAGR | Full Calmar | Early Calmar | Late Calmar | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in results["winners"].items():
        lines.append(
            f"| {name} | {value['price_rise_pct']:.0f}% | {value['breadth_drawdown_points']:.0f} | "
            f"{value['breadth_cap']:.0f}% | {value['full_cagr']:.2%} | "
            f"{value['full_calmar']:.3f} | {value['early_calmar']:.3f} | "
            f"{value['late_calmar']:.3f} | {int(value['completed_trades'])} |"
        )
    lines += [
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
        "## Performance and risk — robust candidate",
        "",
        "| Metric | Baseline | Challenger | Delta |",
        "|---|---:|---:|---:|",
        f"| CAGR | {bm['cagr']:.2%} | {cm['cagr']:.2%} | {cm['cagr']-bm['cagr']:+.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {cm['sharpe']:.3f} | {cm['sharpe']-bm['sharpe']:+.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {cm['sortino']:.3f} | {cm['sortino']-bm['sortino']:+.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {cm['calmar']:.3f} | {cm['calmar']-bm['calmar']:+.3f} |",
        f"| Maximum drawdown | {bm['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} | {cm['max_drawdown']-bm['max_drawdown']:+.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} | {cm['ulcer_index']-bm['ulcer_index']:+.2%} |",
        f"| Time underwater | {bm['time_underwater']:.2%} | {cm['time_underwater']:.2%} | {cm['time_underwater']-bm['time_underwater']:+.2%} |",
        f"| Completed trades | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        f"| Exposure | {bm['exposure']:.2%} | {cm['exposure']:.2%} | {cm['exposure']-bm['exposure']:+.2%} |",
        f"| Profit factor | {bm['profit_factor']:.2f} | {cm['profit_factor']:.2f} | |",
        f"| Expectancy | {bm['expectancy']:.2%} | {cm['expectancy']:.2%} | {cm['expectancy']-bm['expectancy']:+.2%} |",
        "",
        "## Statistical significance",
        "",
        f"- DSR after {results['data']['related_trials_including_grid']:,} trials: {base_eval['statistical_diagnostics']['deflated_sharpe_probability']:.4f} / {challenge['statistical_diagnostics']['deflated_sharpe_probability']:.4f}.",
        f"- Paired annual mean: {results['paired_inference']['annualized_mean_difference']:+.2%}; HAC t={results['paired_inference']['hac_t_stat']:.3f}, p={results['paired_inference']['hac_two_sided_p']:.3f}.",
        f"- Block-bootstrap 95% interval: {results['paired_inference']['bootstrap_95_interval_annualized']}.",
        "",
        "## Robustness",
        "",
        f"- Candidate boundary: {results['neighbour_stability']['candidate_on_grid_boundary']}.",
        f"- Stable immediate neighbours: {results['neighbour_stability']['stable_neighbour_count']} / {results['neighbour_stability']['neighbour_count']} ({results['neighbour_stability']['stable_neighbour_fraction']:.0%}).",
        f"- Historical-half efficiency: {results['wfa_efficiency']['baseline']:.3f} / {results['wfa_efficiency']['challenger']:.3f} (pseudo-OOS).",
        f"- Odd/even paired means: {results['calendar_parity_split']['odd_years']['annualized_mean_difference']:+.2%} / {results['calendar_parity_split']['even_years']['annualized_mean_difference']:+.2%}.",
        f"- 5x/10x-cost paired means: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%} / {results['cost_stress']['10x']['paired_annualized_mean']:+.2%}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Close features; next-open fill |",
        "| Survivorship | Cannot fully verify | Aggregate index and breadth series |",
        f"| Data snooping | Present, material | 210 new cells; {results['data']['related_trials_including_grid']:,} total trials in DSR |",
        "| Costs | Included | 1x/2x/5x/10x stress |",
        "| Liquidity | Low concern | Liquid index proxy; NDX price-index approximation |",
        "| Synthetic breadth | Present before 2007 | 2007+ reported separately |",
        "| Clean forward OOS | Insufficient | Grid and latest history are now seen |",
        "| Regime overfit | Tested, not eliminated | Halves, reverse direction, neighbours, odd/even |",
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
            f"As of {pd.Timestamp(current['date']).date()}, the selected signal is "
            f"{'ACTIVE' if current['active'] else 'inactive'}: NDX 60-session "
            f"return {current['ndx_return_60_pct']:+.2f}%, breadth drawdown "
            f"{current['breadth_drawdown_60_points']:.2f} points, current breadth "
            f"{current['breadth']:.2f}%."
        ),
        "",
        "## Red Flags",
        "",
        "1. This is a 210-cell search on already-seen historical data.",
        "2. A full-sample winner is descriptive, not clean OOS evidence.",
        "3. Boundary or locally unstable winners indicate the grid has found a ridge rather than a robust plateau.",
        "",
        "## Improvement Recommendations",
        "",
        "1. Do not expand the grid after seeing these results in the same round.",
        "2. Keep any passing robust candidate isolated and forward-track it from the frozen boundary.",
        "",
        "## Decision",
        "",
        "The verdict follows the pre-registered maximin selection and guardrails.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
