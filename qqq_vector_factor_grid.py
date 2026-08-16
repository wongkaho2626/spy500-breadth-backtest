"""Exhaustive, split-validated factor-subset search for the vector crash exit."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_vector_crash_exit as crash


DATA_DIR = Path(__file__).parent
DEFAULT_GRID = DATA_DIR / "qqq_vector_factor_grid.csv"
DEFAULT_RESULT = DATA_DIR / "qqq_vector_factor_grid_results.json"
DEFAULT_PROBABILITIES = DATA_DIR / "qqq_vector_factor_probabilities.csv"
SPLIT_DATE = pd.Timestamp("2014-01-01")
THRESHOLDS = tuple(np.arange(0.10, 0.501, 0.05).round(2))
FEATURES = crash.FEATURE_COLUMNS

_WORKER_DF: pd.DataFrame | None = None
_WORKER_VECTOR: pd.DataFrame | None = None
_WORKER_LABELS: pd.Series | None = None
_WORKER_FUTURE_RETURN: pd.Series | None = None
_WORKER_EPISODES: list[dict[str, Any]] | None = None


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def feature_subsets() -> list[tuple[str, ...]]:
    return [
        subset
        for size in range(1, len(FEATURES) + 1)
        for subset in itertools.combinations(FEATURES, size)
    ]


def _load_research_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    list[dict[str, Any]],
]:
    df = crash.qbt.load_data()
    spx_full = crash.load_spx()["close"]
    spx = spx_full.reindex(df.index)
    vector = crash.build_market_vector(df, spx)
    labels, future_return = crash.forward_crash_labels(spx)
    episodes = crash.spx_crash_episodes(spx_full)
    return df, vector, labels, future_return, episodes


def _worker_init() -> None:
    global _WORKER_DF
    global _WORKER_VECTOR
    global _WORKER_LABELS
    global _WORKER_FUTURE_RETURN
    global _WORKER_EPISODES
    (
        _WORKER_DF,
        _WORKER_VECTOR,
        _WORKER_LABELS,
        _WORKER_FUTURE_RETURN,
        _WORKER_EPISODES,
    ) = _load_research_data()


def _period_bounds(period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if period == "early":
        return pd.Timestamp("2002-01-01"), pd.Timestamp("2013-12-31")
    if period == "late":
        return SPLIT_DATE, pd.Timestamp("2100-01-01")
    raise ValueError(f"unknown period: {period}")


def _period_strategy_metrics(
    equity: pd.Series,
    trades: list[dict],
    period: str,
) -> dict[str, float]:
    start, end = _period_bounds(period)
    segment = equity.loc[start:end]
    returns = segment.pct_change().dropna()
    years = (segment.index[-1] - segment.index[0]).days / 365.25
    cagr = (segment.iloc[-1] / segment.iloc[0]) ** (1 / years) - 1
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    drawdown = segment / segment.cummax() - 1
    completed = sum(
        start <= trade["exit_date"] <= end for trade in trades
    )
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "round_trips_per_year": float(completed / years),
    }


def _vector_exit_outcomes(
    trades: list[dict],
    future_return: pd.Series,
) -> list[dict[str, Any]]:
    outcomes = []
    for trade in trades:
        if trade["sell_reason"] != "vector-crash":
            continue
        exit_location = future_return.index.get_loc(trade["exit_date"])
        signal_location = exit_location - crash.qbt.EXECUTION_LAG
        signal_date = future_return.index[signal_location]
        realized = float(future_return.loc[signal_date])
        resolved = np.isfinite(realized)
        outcomes.append(
            {
                "signal_date": signal_date,
                "exit_date": trade["exit_date"],
                "resolved": bool(resolved),
                "true_warning": bool(resolved and realized <= crash.CRASH_DROP),
                "future_min_return": realized,
            }
        )
    return outcomes


def _period_event_metrics(
    position: pd.Series,
    outcomes: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    period: str,
) -> dict[str, Any]:
    start, end = _period_bounds(period)
    period_outcomes = [
        outcome
        for outcome in outcomes
        if start <= outcome["signal_date"] <= end and outcome["resolved"]
    ]
    true_count = sum(outcome["true_warning"] for outcome in period_outcomes)
    false_count = len(period_outcomes) - true_count
    precision = true_count / len(period_outcomes) if period_outcomes else np.nan

    event_rows = []
    protected = 0
    for episode in episodes:
        breach = episode["breach_date"]
        if not (start <= breach <= end) or breach not in position.index:
            continue
        breach_location = position.index.get_loc(breach)
        warning_start = position.index[
            max(0, breach_location - crash.CRASH_HORIZON)
        ]
        qualifying = [
            outcome
            for outcome in period_outcomes
            if outcome["true_warning"]
            and warning_start <= outcome["signal_date"] < breach
        ]
        out_at_breach = not bool(position.loc[breach])
        vector_protected = bool(qualifying and out_at_breach)
        protected += int(vector_protected)
        event_rows.append(
            {
                "breach_date": breach,
                "out_at_breach": out_at_breach,
                "vector_protected": vector_protected,
                "qualifying_vector_exit": (
                    qualifying[0]["exit_date"] if qualifying else None
                ),
            }
        )
    return {
        "evaluable_crashes": len(event_rows),
        "protected_crashes": protected,
        "resolved_vector_exits": len(period_outcomes),
        "true_vector_exits": true_count,
        "false_vector_exits": false_count,
        "precision": float(precision),
        "events": event_rows,
    }


def _compact_configuration(
    subset: tuple[str, ...],
    threshold: float,
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    future_return: pd.Series,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    position = crash.position_series(equity.index, trades, open_trade)
    outcomes = _vector_exit_outcomes(trades, future_return)
    row: dict[str, Any] = {
        "features": "|".join(subset),
        "feature_count": len(subset),
        "threshold": threshold,
        "completed_trades_full": len(trades),
    }
    for period in ("early", "late"):
        strategy = _period_strategy_metrics(equity, trades, period)
        events = _period_event_metrics(
            position, outcomes, episodes, period
        )
        for key, value in strategy.items():
            row[f"{period}_{key}"] = value
        for key, value in events.items():
            if key != "events":
                row[f"{period}_{key}"] = value
    return row


def _evaluate_subset(subset: tuple[str, ...]) -> list[dict[str, Any]]:
    if (
        _WORKER_DF is None
        or _WORKER_VECTOR is None
        or _WORKER_LABELS is None
        or _WORKER_FUTURE_RETURN is None
        or _WORKER_EPISODES is None
    ):
        raise RuntimeError("grid worker was not initialized")
    probability = crash.online_crash_probability(
        _WORKER_VECTOR,
        _WORKER_LABELS,
        feature_columns=subset,
    )["crash_probability"]
    rows = []
    for threshold in THRESHOLDS:
        equity, trades, open_trade = crash.run_replacement_exit(
            _WORKER_DF,
            probability >= threshold,
            reason="vector-crash",
        )
        rows.append(
            _compact_configuration(
                subset,
                threshold,
                equity,
                trades,
                open_trade,
                _WORKER_FUTURE_RETURN,
                _WORKER_EPISODES,
            )
        )
    print(f"completed {'|'.join(subset)}", flush=True)
    return rows


def _baseline_periods(
    df: pd.DataFrame,
) -> tuple[pd.Series, list[dict], dict | None, dict[str, dict[str, float]]]:
    equity, trades, open_trade = crash.qbt.run_strategy(
        df,
        cooldown_days=crash.qbt.COOLDOWN_DAYS,
        execution_lag=crash.qbt.EXECUTION_LAG,
        fill_on=crash.qbt.FILL_PRICE,
    )
    metrics = {
        period: _period_strategy_metrics(equity, trades, period)
        for period in ("early", "late")
    }
    return equity, trades, open_trade, metrics


def _eligible(
    grid: pd.DataFrame,
    baseline: dict[str, dict[str, float]],
    period: str,
) -> pd.Series:
    return (
        (grid[f"{period}_max_drawdown"] >= baseline[period]["max_drawdown"])
        & (grid[f"{period}_cagr"] >= baseline[period]["cagr"] - 0.02)
        & (
            grid[f"{period}_round_trips_per_year"]
            <= baseline[period]["round_trips_per_year"] + 1.0
        )
        & (grid[f"{period}_protected_crashes"] >= 1)
        & (grid[f"{period}_true_vector_exits"] >= 1)
    )


def select_configuration(
    grid: pd.DataFrame,
    baseline: dict[str, dict[str, float]],
    period: str,
) -> pd.Series | None:
    candidates = grid[_eligible(grid, baseline, period)].copy()
    if candidates.empty:
        return None
    candidates[f"{period}_precision_sort"] = candidates[
        f"{period}_precision"
    ].fillna(-1)
    candidates = candidates.sort_values(
        [
            f"{period}_protected_crashes",
            f"{period}_precision_sort",
            f"{period}_sharpe",
            "feature_count",
            "threshold",
        ],
        ascending=[False, False, False, True, False],
        kind="stable",
    )
    return candidates.iloc[0]


def descriptive_leader(
    grid: pd.DataFrame,
    period: str,
) -> pd.Series | None:
    """Return the least-bad crash protector without calling it validated."""
    candidates = grid[
        (grid[f"{period}_protected_crashes"] >= 1)
        & (grid[f"{period}_true_vector_exits"] >= 1)
    ].copy()
    if candidates.empty:
        return None
    candidates = candidates.sort_values(
        [
            f"{period}_false_vector_exits",
            f"{period}_cagr",
            f"{period}_sharpe",
            "feature_count",
            "threshold",
        ],
        ascending=[True, False, False, True, False],
        kind="stable",
    )
    return candidates.iloc[0]


def descriptive_consensus_leader(
    grid: pd.DataFrame,
    baseline: dict[str, dict[str, float]],
) -> pd.Series | None:
    """Rank configurations that protected at least one crash in both halves."""
    candidates = grid[
        (grid["early_protected_crashes"] >= 1)
        & (grid["late_protected_crashes"] >= 1)
    ].copy()
    if candidates.empty:
        return None
    candidates["total_false_exits"] = (
        candidates["early_false_vector_exits"]
        + candidates["late_false_vector_exits"]
    )
    candidates["worst_cagr_gap"] = np.minimum(
        candidates["early_cagr"] - baseline["early"]["cagr"],
        candidates["late_cagr"] - baseline["late"]["cagr"],
    )
    candidates["worst_drawdown_gap"] = np.minimum(
        candidates["early_max_drawdown"]
        - baseline["early"]["max_drawdown"],
        candidates["late_max_drawdown"]
        - baseline["late"]["max_drawdown"],
    )
    candidates = candidates.sort_values(
        [
            "total_false_exits",
            "worst_cagr_gap",
            "worst_drawdown_gap",
            "feature_count",
            "threshold",
        ],
        ascending=[True, False, False, True, False],
        kind="stable",
    )
    return candidates.iloc[0]


def _subset_from_row(row: pd.Series) -> tuple[str, ...]:
    return tuple(str(row["features"]).split("|"))


def _detailed_selected(
    row: pd.Series,
    df: pd.DataFrame,
    vector: pd.DataFrame,
    labels: pd.Series,
    future_return: pd.Series,
    episodes: list[dict[str, Any]],
    baseline_equity: pd.Series,
) -> tuple[dict[str, Any], pd.Series]:
    subset = _subset_from_row(row)
    threshold = float(row["threshold"])
    risk = crash.online_crash_probability(
        vector,
        labels,
        feature_columns=subset,
    )
    probability = risk["crash_probability"]
    equity, trades, open_trade, details = crash.evaluate_threshold(
        df,
        probability,
        threshold,
        crash.load_spx()["close"].reindex(df.index),
        future_return,
        episodes,
    )
    details["features"] = list(subset)
    details["paired_inference"] = crash.paired_hac_and_bootstrap(
        equity, baseline_equity
    )
    details["current_probability"] = probability.iloc[-1]
    details["current_signal"] = bool(probability.iloc[-1] >= threshold)
    details["grid_row"] = row.to_dict()
    details["cost_stress"] = {}
    for multiplier in (1, 5):
        baseline_cost, _, _ = crash.run_replacement_exit(
            df,
            crash.baseline_divergence_signal(df),
            reason="bearish-divergence",
            commission_multiplier=multiplier,
        )
        candidate_cost, _, _, _ = crash.evaluate_threshold(
            df,
            probability,
            threshold,
            crash.load_spx()["close"].reindex(df.index),
            future_return,
            episodes,
            commission_multiplier=multiplier,
        )
        details["cost_stress"][str(multiplier)] = {
            "baseline_cagr": crash.slice_metrics(
                baseline_cost, str(df.index[0].date())
            )["cagr"],
            "candidate_cagr": crash.slice_metrics(
                candidate_cost, str(df.index[0].date())
            )["cagr"],
        }
    return details, probability


def _opposite_validation(
    row: pd.Series | None,
    period: str,
    baseline: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if row is None:
        return {"selected": False, "passed": False}
    opposite = "late" if period == "early" else "early"
    protected = int(row[f"{opposite}_protected_crashes"])
    cagr_ok = (
        row[f"{opposite}_cagr"] >= baseline[opposite]["cagr"] - 0.02
    )
    drawdown_ok = (
        row[f"{opposite}_max_drawdown"]
        >= baseline[opposite]["max_drawdown"]
    )
    precision = row[f"{opposite}_precision"]
    passed = bool(
        protected >= 1
        and cagr_ok
        and drawdown_ok
        and np.isfinite(precision)
        and precision > 0
    )
    return {
        "selected": True,
        "selection_period": period,
        "validation_period": opposite,
        "protected_crashes": protected,
        "cagr_guardrail_passed": bool(cagr_ok),
        "drawdown_guardrail_passed": bool(drawdown_ok),
        "vector_exit_precision": precision,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-output", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--probabilities-output",
        type=Path,
        default=DEFAULT_PROBABILITIES,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(6, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--reuse-grid",
        action="store_true",
        help="reuse an existing complete grid CSV and only rebuild summaries",
    )
    args = parser.parse_args()

    parity = crash.parity_check(crash.qbt.load_data())
    if not parity["passed"]:
        raise RuntimeError(f"baseline parity failed: {parity}")

    df, vector, labels, future_return, episodes = _load_research_data()
    baseline_equity, baseline_trades, baseline_open, baseline_periods = (
        _baseline_periods(df)
    )
    subsets = feature_subsets()
    if args.reuse_grid:
        grid = pd.read_csv(args.grid_output)
        expected = len(subsets) * len(THRESHOLDS)
        if len(grid) != expected:
            raise RuntimeError(
                f"existing grid has {len(grid)} rows; expected {expected}"
            )
    else:
        rows: list[dict[str, Any]] = []
        if args.workers == 1:
            _worker_init()
            for subset in subsets:
                rows.extend(_evaluate_subset(subset))
        else:
            context = mp.get_context("spawn")
            with context.Pool(
                processes=args.workers,
                initializer=_worker_init,
            ) as pool:
                for subset_rows in pool.imap_unordered(
                    _evaluate_subset,
                    subsets,
                ):
                    rows.extend(subset_rows)
        grid = pd.DataFrame(rows).sort_values(
            ["feature_count", "features", "threshold"]
        )
    grid["early_guardrails_pass"] = _eligible(
        grid, baseline_periods, "early"
    )
    grid["late_guardrails_pass"] = _eligible(
        grid, baseline_periods, "late"
    )
    grid.to_csv(args.grid_output, index=False)

    early_selected = select_configuration(
        grid, baseline_periods, "early"
    )
    late_selected = select_configuration(grid, baseline_periods, "late")
    early_validation = _opposite_validation(
        early_selected, "early", baseline_periods
    )
    late_validation = _opposite_validation(
        late_selected, "late", baseline_periods
    )
    diagnostic_early = descriptive_leader(grid, "early")
    diagnostic_late = descriptive_leader(grid, "late")
    diagnostic_consensus = descriptive_consensus_leader(
        grid, baseline_periods
    )

    detailed = {}
    probability_output = pd.DataFrame(index=df.index)
    for name, selected in (
        ("early_selected", early_selected),
        ("late_selected", late_selected),
        ("diagnostic_early", diagnostic_early),
        ("diagnostic_late", diagnostic_late),
        ("diagnostic_consensus", diagnostic_consensus),
    ):
        if selected is None:
            detailed[name] = None
            continue
        details, probability = _detailed_selected(
            selected,
            df,
            vector,
            labels,
            future_return,
            episodes,
            baseline_equity,
        )
        detailed[name] = details
        probability_output[name] = probability
    probability_output.index.name = "Date"
    probability_output.reset_index().to_csv(
        args.probabilities_output, index=False
    )

    baseline_position = crash.position_series(
        df.index, baseline_trades, baseline_open
    )
    result = {
        "idea_card": (
            DATA_DIR / "docs/research/vector_factor_grid_idea.md"
        ).resolve(),
        "configuration": {
            "features": list(FEATURES),
            "feature_subsets": len(subsets),
            "thresholds": list(THRESHOLDS),
            "total_configurations": len(grid),
            "neighbors": crash.NEIGHBORS,
            "horizon_sessions": crash.CRASH_HORIZON,
            "crash_drop": crash.CRASH_DROP,
            "split_date": SPLIT_DATE,
            "workers": args.workers,
        },
        "baseline_parity": parity,
        "baseline": {
            "periods": baseline_periods,
            "full_metrics": crash.strategy_metrics(
                baseline_equity, baseline_trades, baseline_position
            ),
            "crash_avoidance": crash.crash_avoidance(
                episodes, baseline_position, baseline_trades
            ),
        },
        "early_selected": (
            early_selected.to_dict()
            if early_selected is not None
            else None
        ),
        "early_to_late_validation": early_validation,
        "late_selected": (
            late_selected.to_dict()
            if late_selected is not None
            else None
        ),
        "late_to_early_validation": late_validation,
        "diagnostic_leaders": {
            "early": (
                diagnostic_early.to_dict()
                if diagnostic_early is not None
                else None
            ),
            "late": (
                diagnostic_late.to_dict()
                if diagnostic_late is not None
                else None
            ),
            "consensus": (
                diagnostic_consensus.to_dict()
                if diagnostic_consensus is not None
                else None
            ),
        },
        "selected_details": detailed,
        "decision": (
            "track"
            if early_validation["passed"] and late_validation["passed"]
            else "reject"
        ),
        "limitations": [
            "Only three independent SPX 20% episodes occur in the strategy "
            "sample.",
            "All 567 configurations are historically selected and require a "
            "large multiple-testing penalty.",
            "The two half-splits are pseudo-out-of-sample; data through "
            "2026-07-02 were previously seen.",
            "Pre-2007 breadth is a synthetic MMTH-mapped splice.",
            "No resolved 126-session clean forward label exists after the "
            "2026-07-05 freeze.",
        ],
        "artifacts": {
            "grid_csv": args.grid_output.resolve(),
            "selected_probabilities_csv": args.probabilities_output.resolve(),
            "results_json": args.result_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
