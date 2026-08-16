"""Evaluate sell-only 3D vectors against post-exit economic targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import qqq_current_sell_state_vector as shared
import qqq_sell_signal_3d_vector as sell3d


DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_sell_3d_economic_targets_results.json"
DEFAULT_EVENTS = DATA_DIR / "qqq_sell_3d_economic_target_events.csv"
DEFAULT_PREDICTIONS = DATA_DIR / "qqq_sell_3d_economic_predictions.csv"
HORIZONS = (20, 60)
TARGET_HORIZON = 60
NEIGHBORS = 3
MIN_TRAINING_EVENTS = 4
ROUND_TRIP_SLIPPAGE = 0.001
CONTINUOUS_TARGETS = (
    "future_min_ndx_return_60",
    "risk_adjusted_return_20",
    "risk_adjusted_return_60",
    "cash_advantage_20",
    "cash_advantage_60",
    "oracle_reentry_delay_60",
)
BINARY_TARGETS = ("cash_wins_20", "cash_wins_60")


def _finite(value: Any) -> Any:
    return shared._finite(value)


def build_event_targets(
    strategy: pd.DataFrame,
    trades: list[dict],
    vector: pd.DataFrame,
) -> pd.DataFrame:
    """Create causal signal rows and future labels kept in separate columns."""
    rows = []
    index = strategy.index
    lag = shared.base_analysis.transition.qbt.EXECUTION_LAG
    for trade in trades:
        exit_location = index.get_loc(trade["exit_date"])
        signal_location = int(exit_location) - lag
        end_location = signal_location + TARGET_HORIZON
        if end_location >= len(index):
            continue
        signal_date = index[signal_location]
        execution_date = index[int(exit_location)]
        if signal_date not in vector.index:
            continue
        feature_values = vector.loc[
            signal_date, list(sell3d.SELL_3D_COLUMNS)
        ]
        if feature_values.isna().any():
            continue

        execution_row = strategy.iloc[int(exit_location)]
        start_price = float(execution_row["open"])
        if not math.isfinite(start_price):
            start_price = float(execution_row["price"])
        closes_60 = strategy["price"].iloc[
            int(exit_location) : end_location + 1
        ].astype(float)
        if len(closes_60) != TARGET_HORIZON:
            raise RuntimeError("unexpected forward horizon length")

        row: dict[str, Any] = {
            "signal_date": signal_date,
            "execution_date": execution_date,
            "target_end_date": index[end_location],
            "sell_reason": trade["sell_reason"],
            **feature_values.to_dict(),
            "future_min_ndx_return_60": float(
                closes_60.min() / start_price - 1
            ),
        }
        for horizon in HORIZONS:
            closes = closes_60.iloc[:horizon]
            values = np.concatenate([[start_price], closes.to_numpy()])
            daily_returns = values[1:] / values[:-1] - 1
            terminal_return = float(closes.iloc[-1] / start_price - 1)
            path_volatility = float(
                np.std(daily_returns, ddof=1) * np.sqrt(horizon)
            )
            risk_adjusted = (
                terminal_return / path_volatility
                if path_volatility > 0
                else np.nan
            )
            cash_advantage = -terminal_return - ROUND_TRIP_SLIPPAGE
            row[f"hold_return_{horizon}"] = terminal_return
            row[f"risk_adjusted_return_{horizon}"] = risk_adjusted
            row[f"cash_advantage_{horizon}"] = cash_advantage
            row[f"cash_wins_{horizon}"] = bool(cash_advantage > 0)
        row["oracle_reentry_delay_60"] = int(
            np.argmin(closes_60.to_numpy())
        )
        row["oracle_reentry_date_60"] = closes_60.index[
            int(row["oracle_reentry_delay_60"])
        ]
        rows.append(row)

    return pd.DataFrame(rows).set_index("signal_date").sort_index()


def distance_weights(distances: pd.Series) -> np.ndarray:
    raw = 1.0 / (distances.to_numpy(dtype=float) + 0.25)
    return raw / raw.sum()


def expanding_predictions(
    events: pd.DataFrame,
    feature_columns: tuple[str, ...] = sell3d.SELL_3D_COLUMNS,
) -> pd.DataFrame:
    rows = []
    for signal_date, query in events.iterrows():
        training = events.loc[
            (events.index < signal_date)
            & (events["target_end_date"] < signal_date)
        ]
        if len(training) < MIN_TRAINING_EVENTS:
            continue
        center, scale = shared.base_analysis.robust_scale(
            training.loc[:, feature_columns]
        )
        distances = shared.base_analysis.standardized_distance(
            query,
            training,
            center,
            scale,
            feature_columns,
        ).sort_values()
        selected = training.loc[distances.index[:NEIGHBORS]]
        weights = distance_weights(distances.iloc[:NEIGHBORS])
        row: dict[str, Any] = {
            "signal_date": signal_date,
            "training_events": len(training),
            "neighbors": len(selected),
            "neighbor_dates": "|".join(
                date.strftime("%Y-%m-%d") for date in selected.index
            ),
        }
        for target in CONTINUOUS_TARGETS:
            actual = float(query[target])
            prediction = float(
                np.dot(weights, selected[target].to_numpy(dtype=float))
            )
            naive = float(training[target].median())
            row[f"actual_{target}"] = actual
            row[f"model_{target}"] = prediction
            row[f"naive_{target}"] = naive
        for target in BINARY_TARGETS:
            actual = float(bool(query[target]))
            prediction = float(
                np.dot(weights, selected[target].to_numpy(dtype=float))
            )
            naive = float(training[target].mean())
            row[f"actual_{target}"] = actual
            row[f"model_{target}"] = prediction
            row[f"naive_{target}"] = naive
        rows.append(row)
    return pd.DataFrame(rows).set_index("signal_date").sort_index()


def continuous_diagnostics(
    predictions: pd.DataFrame, target: str
) -> dict[str, Any]:
    actual = predictions[f"actual_{target}"].to_numpy(dtype=float)
    model = predictions[f"model_{target}"].to_numpy(dtype=float)
    naive = predictions[f"naive_{target}"].to_numpy(dtype=float)
    model_mae = float(np.mean(np.abs(model - actual)))
    naive_mae = float(np.mean(np.abs(naive - actual)))
    correlation = stats.spearmanr(actual, model).statistic
    return {
        "observations": len(actual),
        "model_mae": model_mae,
        "naive_mae": naive_mae,
        "mae_improvement_pct": (
            (1 - model_mae / naive_mae) * 100
            if naive_mae > 0
            else np.nan
        ),
        "spearman_correlation": correlation,
        "model_beats_naive": bool(model_mae < naive_mae),
    }


def binary_diagnostics(
    predictions: pd.DataFrame, target: str
) -> dict[str, Any]:
    actual = predictions[f"actual_{target}"].to_numpy(dtype=float)
    model = predictions[f"model_{target}"].to_numpy(dtype=float)
    naive = predictions[f"naive_{target}"].to_numpy(dtype=float)
    model_brier = float(np.mean(np.square(model - actual)))
    naive_brier = float(np.mean(np.square(naive - actual)))
    return {
        "observations": len(actual),
        "observed_rate": float(actual.mean()),
        "model_brier": model_brier,
        "naive_brier": naive_brier,
        "brier_improvement_pct": (
            (1 - model_brier / naive_brier) * 100
            if naive_brier > 0
            else np.nan
        ),
        "model_accuracy_at_50pct": float(
            np.mean((model >= 0.5) == actual.astype(bool))
        ),
        "naive_accuracy_at_50pct": float(
            np.mean((naive >= 0.5) == actual.astype(bool))
        ),
        "model_beats_naive": bool(model_brier < naive_brier),
    }


def primary_period_diagnostics(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    target = "future_min_ndx_return_60"
    periods = {
        "early_through_2013": predictions.loc[
            predictions.index < pd.Timestamp("2014-01-01")
        ],
        "late_from_2014": predictions.loc[
            predictions.index >= pd.Timestamp("2014-01-01")
        ],
    }
    return {
        name: continuous_diagnostics(frame, target)
        for name, frame in periods.items()
    }


def current_counterfactual(
    current: pd.Series,
    events: pd.DataFrame,
    feature_columns: tuple[str, ...] = sell3d.SELL_3D_COLUMNS,
) -> dict[str, Any]:
    center, scale = shared.base_analysis.robust_scale(
        events.loc[:, feature_columns]
    )
    distances = shared.base_analysis.standardized_distance(
        current,
        events,
        center,
        scale,
        feature_columns,
    ).sort_values()
    selected = events.loc[distances.index[:NEIGHBORS]]
    weights = distance_weights(distances.iloc[:NEIGHBORS])
    forecasts = {
        target: float(
            np.dot(weights, selected[target].to_numpy(dtype=float))
        )
        for target in CONTINUOUS_TARGETS
    }
    forecasts.update(
        {
            target: float(
                np.dot(weights, selected[target].to_numpy(dtype=float))
            )
            for target in BINARY_TARGETS
        }
    )
    return {
        "neighbors": [date for date in selected.index],
        "distances": distances.iloc[:NEIGHBORS].tolist(),
        "forecasts": forecasts,
        "warning": (
            "Counterfactual only: no canonical sell signal is active, and "
            "the model must pass causal validation before use."
        ),
    }


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    output = frame.reset_index()
    for column in output.columns:
        if column.endswith("date"):
            output[column] = pd.to_datetime(output[column]).dt.strftime(
                "%Y-%m-%d"
            )
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--events-output", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument(
        "--predictions-output", type=Path, default=DEFAULT_PREDICTIONS
    )
    args = parser.parse_args()

    vector, strategy = sell3d.build_sell_3d_vector()
    qbt = shared.base_analysis.transition.qbt
    _, trades, open_trade = qbt.run_strategy(
        strategy,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    events = build_event_targets(strategy, trades, vector)
    predictions = expanding_predictions(events)

    continuous_results = {
        target: continuous_diagnostics(predictions, target)
        for target in CONTINUOUS_TARGETS
    }
    binary_results = {
        target: binary_diagnostics(predictions, target)
        for target in BINARY_TARGETS
    }
    primary = continuous_results["future_min_ndx_return_60"]
    primary_periods = primary_period_diagnostics(predictions)
    secondary_wins = sum(
        result["model_beats_naive"]
        for target, result in continuous_results.items()
        if target != "future_min_ndx_return_60"
    ) + sum(result["model_beats_naive"] for result in binary_results.values())
    secondary_total = len(continuous_results) - 1 + len(binary_results)

    current_date = vector.index[-1]
    current = vector.loc[current_date]
    current_state = shared.current_strategy_sell_state(strategy, open_trade)
    counterfactual = current_counterfactual(current, events)

    write_frame(args.events_output, events)
    write_frame(args.predictions_output, predictions)

    stable_primary = all(
        period["model_beats_naive"] for period in primary_periods.values()
    )
    result = {
        "decision": (
            "track"
            if primary["model_beats_naive"] and stable_primary
            else "reject"
        ),
        "hypothesis": (
            "The sell-only 3D vector predicts post-exit economic outcomes "
            "better than expanding historical naive forecasts."
        ),
        "method": {
            "features": list(sell3d.SELL_3D_COLUMNS),
            "horizons_sessions": list(HORIZONS),
            "neighbors": NEIGHBORS,
            "minimum_training_events": MIN_TRAINING_EVENTS,
            "weighting": "1 / (robust-scaled Euclidean distance + 0.25)",
            "training_rule": (
                "expanding causal; prior event admitted only after its full "
                "60-session label window ended"
            ),
            "naive_continuous": "expanding historical median",
            "naive_binary": "expanding historical base rate",
            "round_trip_slippage": ROUND_TRIP_SLIPPAGE,
            "cash_yield": 0.0,
            "new_tuned_parameters": 0,
        },
        "data": {
            "start": strategy.index[0],
            "end": strategy.index[-1],
            "canonical_exits": len(trades),
            "events_with_complete_3d_and_targets": len(events),
            "causal_predictions": len(predictions),
            "exit_reason_counts": events["sell_reason"].value_counts().to_dict(),
            "clean_forward_oos_start": "2026-07-05",
            "clean_forward_completed_exits": 0,
        },
        "primary_target": {
            "name": "future_min_ndx_return_60",
            **primary,
        },
        "primary_early_late_stability": primary_periods,
        "continuous_targets": continuous_results,
        "cash_vs_hold_targets": binary_results,
        "secondary_targets_beating_naive": secondary_wins,
        "secondary_targets_total": secondary_total,
        "decision_reason": (
            "Reject: the 0.16% full-sample primary MAE improvement reverses "
            "from +7.01% in the early period to -7.43% from 2014 onward, "
            "and none of seven secondary targets beats its naive benchmark."
        ),
        "current": {
            "date": current_date,
            "vector": {
                column: current[column]
                for column in sell3d.SELL_3D_COLUMNS
            },
            "canonical_sell_state": current_state,
            "counterfactual_if_sold_now": counterfactual,
        },
        "limitations": [
            "Only 17 canonical exits are sampled and just 13 produce causal predictions after the minimum training period.",
            "The three-dimensional vector omits climax, trailing-stop, entry, trade-high and cooldown state.",
            "Cash-vs-hold subtracts fixed round-trip slippage but not dollar commission or cash yield.",
            "Oracle re-entry delay is the hindsight trough within 60 sessions and is only a supervised target.",
            "There are no completed clean forward-OOS exits after the 2026-07-05 freeze.",
            "Counterfactual current forecasts are not actionable when no sell signal is active.",
        ],
        "artifacts": {
            "result_json": args.result_output.resolve(),
            "event_targets_csv": args.events_output.resolve(),
            "causal_predictions_csv": args.predictions_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
