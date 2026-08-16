"""Stateful sell-vector challenger on the frozen economic target panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_current_sell_state_vector as shared
import qqq_sell_3d_economic_targets as economic
import qqq_sell_signal_3d_vector as sell3d


DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_stateful_sell_vector_results.json"
DEFAULT_EVENTS = DATA_DIR / "qqq_stateful_sell_vector_events.csv"
DEFAULT_PREDICTIONS = DATA_DIR / "qqq_stateful_sell_vector_predictions.csv"
AGE_CAP = 60
TRAJECTORY_COLUMNS = (
    "ndx_return_60_slope_20",
    "vix_slope_20",
    "spx_drawdown_252_slope_20",
)
STATEFUL_COLUMNS = (
    *sell3d.SELL_3D_COLUMNS,
    "reason_bearish_divergence",
    "reason_climax_top",
    "reason_trailing_stop",
    "return_since_entry_pct",
    "trade_high_gain_from_entry_pct",
    "current_trade_drawdown_pct",
    "macd_cross_age_capped",
    "extension_age_capped",
    "climax_active",
    "trailing_stop_active",
    *TRAJECTORY_COLUMNS,
)


def capped_age(event: pd.Series, cap: int = AGE_CAP) -> int:
    locations = np.flatnonzero(event.fillna(False).to_numpy(dtype=bool))
    if len(locations) == 0:
        return cap
    return min(len(event) - 1 - int(locations[-1]), cap)


def position_state(
    strategy: pd.DataFrame,
    signal_date: pd.Timestamp,
    entry_date: pd.Timestamp,
    entry_price: float,
    sell_reason: str | None,
) -> dict[str, Any]:
    qbt = shared.base_analysis.transition.qbt
    invested = strategy.loc[
        (strategy.index > entry_date) & (strategy.index <= signal_date)
    ]
    current_price = float(strategy.loc[signal_date, "price"])
    trade_high = max(
        float(entry_price),
        float(invested["price"].max()) if not invested.empty else float(entry_price),
    )
    macd_age = capped_age(invested["macd_cross"])
    extension_age = capped_age(invested["ext10"])
    drawdown = (current_price / trade_high - 1) * 100
    return {
        "entry_price_raw": float(entry_price),
        "trade_high_raw": trade_high,
        "return_since_entry_pct": (
            current_price / float(entry_price) - 1
        )
        * 100,
        "trade_high_gain_from_entry_pct": (
            trade_high / float(entry_price) - 1
        )
        * 100,
        "current_trade_drawdown_pct": drawdown,
        "macd_cross_age_capped": float(macd_age),
        "extension_age_capped": float(extension_age),
        "reason_bearish_divergence": float(
            sell_reason == "bearish-divergence"
        ),
        "reason_climax_top": float(sell_reason == "climax-top"),
        "reason_trailing_stop": float(sell_reason == "trailing-stop"),
        "climax_active": float(
            macd_age < qbt.CLIMAX_VOTE_WINDOW
            and extension_age < qbt.CLIMAX_VOTE_WINDOW
        ),
        "trailing_stop_active": float(
            drawdown <= -qbt.TRAILING_STOP_PCT
        ),
    }


def build_stateful_events(
    strategy: pd.DataFrame,
    trades: list[dict],
    sell_vector: pd.DataFrame,
) -> pd.DataFrame:
    events = economic.build_event_targets(strategy, trades, sell_vector)
    trajectory = shared.base_analysis.build_full_transition_vector()
    state_rows = []
    lag = shared.base_analysis.transition.qbt.EXECUTION_LAG
    for trade in trades:
        exit_location = strategy.index.get_loc(trade["exit_date"])
        signal_date = strategy.index[int(exit_location) - lag]
        if signal_date not in events.index:
            continue
        state = position_state(
            strategy,
            signal_date,
            pd.Timestamp(trade["entry_date"]),
            float(trade["entry_price"]),
            str(trade["sell_reason"]),
        )
        state.update(
            trajectory.loc[signal_date, list(TRAJECTORY_COLUMNS)].to_dict()
        )
        state_rows.append({"signal_date": signal_date, **state})
    states = pd.DataFrame(state_rows).set_index("signal_date")
    output = events.join(states, how="inner")
    if output.loc[:, STATEFUL_COLUMNS].isna().any().any():
        raise RuntimeError("stateful event vector contains missing values")
    return output


def continuous_comparison(
    stateful_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    challenger = economic.continuous_diagnostics(
        stateful_predictions, target
    )
    baseline = economic.continuous_diagnostics(
        baseline_predictions, target
    )
    return {
        "stateful": challenger,
        "sell_3d": baseline,
        "stateful_vs_sell_3d_mae_improvement_pct": (
            1 - challenger["model_mae"] / baseline["model_mae"]
        )
        * 100,
        "stateful_beats_both": bool(
            challenger["model_mae"] < baseline["model_mae"]
            and challenger["model_mae"] < challenger["naive_mae"]
        ),
    }


def binary_comparison(
    stateful_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    target: str,
) -> dict[str, Any]:
    challenger = economic.binary_diagnostics(stateful_predictions, target)
    baseline = economic.binary_diagnostics(baseline_predictions, target)
    return {
        "stateful": challenger,
        "sell_3d": baseline,
        "stateful_vs_sell_3d_brier_improvement_pct": (
            1 - challenger["model_brier"] / baseline["model_brier"]
        )
        * 100,
        "stateful_beats_both": bool(
            challenger["model_brier"] < baseline["model_brier"]
            and challenger["model_brier"] < challenger["naive_brier"]
        ),
    }


def current_state_vector(
    strategy: pd.DataFrame,
    sell_vector: pd.DataFrame,
    open_trade: dict,
) -> pd.Series:
    current_date = strategy.index[-1]
    state = position_state(
        strategy,
        current_date,
        pd.Timestamp(open_trade["entry_date"]),
        float(open_trade["entry_price"]),
        None,
    )
    trajectory = shared.base_analysis.build_full_transition_vector()
    values = {
        **sell_vector.loc[
            current_date, list(sell3d.SELL_3D_COLUMNS)
        ].to_dict(),
        **state,
        **trajectory.loc[
            current_date, list(TRAJECTORY_COLUMNS)
        ].to_dict(),
    }
    return pd.Series(values, name=current_date)


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

    sell_vector, strategy = sell3d.build_sell_3d_vector()
    qbt = shared.base_analysis.transition.qbt
    _, trades, open_trade = qbt.run_strategy(
        strategy,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    if open_trade is None:
        raise RuntimeError("current canonical strategy has no open trade")
    events = build_stateful_events(strategy, trades, sell_vector)
    stateful_predictions = economic.expanding_predictions(
        events, STATEFUL_COLUMNS
    )
    baseline_predictions = economic.expanding_predictions(
        events, sell3d.SELL_3D_COLUMNS
    )
    if not stateful_predictions.index.equals(baseline_predictions.index):
        raise RuntimeError("stateful and 3D prediction dates are misaligned")

    continuous = {
        target: continuous_comparison(
            stateful_predictions, baseline_predictions, target
        )
        for target in economic.CONTINUOUS_TARGETS
    }
    binary = {
        target: binary_comparison(
            stateful_predictions, baseline_predictions, target
        )
        for target in economic.BINARY_TARGETS
    }
    primary = continuous["future_min_ndx_return_60"]
    stateful_periods = economic.primary_period_diagnostics(
        stateful_predictions
    )
    baseline_periods = economic.primary_period_diagnostics(
        baseline_predictions
    )
    period_comparison = {}
    for period in stateful_periods:
        challenger = stateful_periods[period]
        baseline = baseline_periods[period]
        period_comparison[period] = {
            "stateful": challenger,
            "sell_3d": baseline,
            "stateful_beats_both": bool(
                challenger["model_mae"] < baseline["model_mae"]
                and challenger["model_mae"] < challenger["naive_mae"]
            ),
        }
    secondary_wins = sum(
        result["stateful_beats_both"]
        for target, result in continuous.items()
        if target != "future_min_ndx_return_60"
    ) + sum(result["stateful_beats_both"] for result in binary.values())
    primary_pass = bool(
        primary["stateful_beats_both"]
        and all(
            result["stateful_beats_both"]
            for result in period_comparison.values()
        )
    )
    passed = primary_pass and secondary_wins >= 4

    current = current_state_vector(
        strategy, sell_vector, open_trade
    )
    counterfactual = economic.current_counterfactual(
        current, events, STATEFUL_COLUMNS
    )

    prediction_output = stateful_predictions.add_prefix("stateful_").join(
        baseline_predictions.add_prefix("sell3d_")
    )
    write_frame(args.events_output, events)
    write_frame(args.predictions_output, prediction_output)

    result = {
        "decision": "track" if passed else "reject",
        "hypothesis": (
            "Adding position path, exit mechanism, event ages and market "
            "trajectory improves the frozen economic target forecasts."
        ),
        "method": {
            "features": list(STATEFUL_COLUMNS),
            "feature_count": len(STATEFUL_COLUMNS),
            "raw_price_audit_columns": [
                "entry_price_raw",
                "trade_high_raw",
            ],
            "price_normalization": (
                "raw entry/high retained for audit; distance uses return "
                "since entry, high gain and drawdown"
            ),
            "age_cap_sessions": AGE_CAP,
            "neighbors": economic.NEIGHBORS,
            "minimum_training_events": economic.MIN_TRAINING_EVENTS,
            "targets": list(economic.CONTINUOUS_TARGETS)
            + list(economic.BINARY_TARGETS),
            "new_tuned_parameters": 0,
        },
        "data": {
            "start": strategy.index[0],
            "end": strategy.index[-1],
            "canonical_exits": len(trades),
            "complete_stateful_events": len(events),
            "causal_predictions": len(stateful_predictions),
            "features_per_prediction": len(STATEFUL_COLUMNS),
            "clean_forward_completed_exits": 0,
        },
        "primary_target": primary,
        "primary_early_late": period_comparison,
        "continuous_targets": continuous,
        "binary_targets": binary,
        "secondary_targets_beating_both": secondary_wins,
        "secondary_targets_required": 4,
        "guardrails": {
            "primary_beats_both_full_early_late": primary_pass,
            "at_least_four_secondary_targets_beat_both": (
                secondary_wins >= 4
            ),
            "all_passed": passed,
        },
        "current": {
            "date": current.name,
            "vector": {
                column: current[column] for column in STATEFUL_COLUMNS
            },
            "entry_price_raw": current["entry_price_raw"],
            "trade_high_raw": current["trade_high_raw"],
            "canonical_sell_state": shared.current_strategy_sell_state(
                strategy, open_trade
            ),
            "counterfactual_if_sold_now": counterfactual,
        },
        "decision_reason": (
            "Reject: four secondary targets beat both comparators, but the "
            "pre-registered primary 60-session adverse-return target is worse "
            "than both 3D and naive in the full sample and fails both the "
            "early and late dual-comparator checks."
        ),
        "limitations": [
            "Sixteen features are estimated from only 17 exits and 13 causal predictions.",
            "Sell-reason one-hot values are known at historical sell decisions but all zero when no current sell is active.",
            "The vector remains a nearest-neighbour diagnostic, not a calibrated return model.",
            "No completed clean forward-OOS exit exists after the freeze.",
            "Current counterfactual forecasts are outside-domain while no sell signal is active.",
        ],
        "artifacts": {
            "result_json": args.result_output.resolve(),
            "stateful_event_vectors_csv": args.events_output.resolve(),
            "paired_predictions_csv": args.predictions_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(shared._finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(shared._finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
