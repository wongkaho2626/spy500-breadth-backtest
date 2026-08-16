"""Evaluate only the three continuous inputs of the frozen QQQ sell rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import qqq_current_sell_state_vector as shared


DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_sell_signal_3d_vector_results.json"
DEFAULT_SIGNALS = DATA_DIR / "qqq_sell_signal_3d_vectors.csv"
DEFAULT_EXITS = DATA_DIR / "qqq_sell_signal_3d_exits.csv"
SELL_3D_COLUMNS = (
    "sell_ndx_return_60_pct",
    "sell_breadth_fall_60_points",
    "sell_breadth_level_pct",
)


def build_sell_3d_vector() -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy = shared.base_analysis.transition.qbt.load_data()
    sell = shared.canonical_sell_vote_frame(strategy)
    return sell, strategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--signals-output", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--exits-output", type=Path, default=DEFAULT_EXITS)
    args = parser.parse_args()

    vector, strategy = build_sell_3d_vector()
    complete = vector.dropna(subset=list(SELL_3D_COLUMNS))
    peak_rows = shared.base_analysis.load_peak_rows()
    comparable_peaks = peak_rows.loc[
        peak_rows.index.intersection(complete.index)
    ]
    signal_dates, future_return = shared.raw_signal_dates_and_labels(strategy)
    signal_dates = signal_dates.intersection(complete.index)

    full = shared.full_sample_similarity(
        complete, comparable_peaks, signal_dates, SELL_3D_COLUMNS
    ).add_prefix("full_")
    causal = shared.causal_similarity_rows(
        complete, comparable_peaks, signal_dates, SELL_3D_COLUMNS
    )

    output = pd.DataFrame(index=signal_dates)
    output.index.name = "signal_date"
    output["future_min_spx_return_126"] = future_return.reindex(signal_dates)
    output["followed_by_20pct_drop"] = (
        output["future_min_spx_return_126"] <= -0.20
    )
    output = output.join(complete.loc[signal_dates, list(SELL_3D_COLUMNS)])
    output = output.join(full).join(causal)

    causal_output = output.dropna(
        subset=["causal_peak_similarity_percentile"]
    )
    labels = causal_output["followed_by_20pct_drop"]
    causal_auc = shared.base_analysis.rank_auc(
        labels,
        causal_output["causal_peak_similarity_percentile"],
    )
    full_auc = shared.base_analysis.rank_auc(
        output["followed_by_20pct_drop"],
        output["full_peak_similarity_percentile"],
    )

    qbt = shared.base_analysis.transition.qbt
    _, trades, open_trade = qbt.run_strategy(
        strategy,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    exits = shared.canonical_exit_outcomes(
        strategy, trades, future_return, vector
    )
    divergence_exits = exits.loc[
        exits["sell_reason"] == "bearish-divergence"
    ]
    divergence_drops = int(
        divergence_exits["followed_by_20pct_drop"].sum()
    )

    current_date = complete.index[-1]
    current = complete.loc[current_date]
    current_causal = shared.base_analysis.causal_prior_peak_similarity(
        current_date,
        current,
        complete,
        comparable_peaks,
        SELL_3D_COLUMNS,
    )
    center, scale = shared.base_analysis.robust_scale(
        complete.loc[:, SELL_3D_COLUMNS]
    )
    current_signal_distances = shared.base_analysis.standardized_distance(
        current,
        complete.reindex(signal_dates),
        center,
        scale,
        SELL_3D_COLUMNS,
    ).sort_values()
    nearest_signal_date = current_signal_distances.index[0]

    serialized = output.reset_index()
    for column in serialized.columns:
        if column.endswith("date"):
            serialized[column] = pd.to_datetime(serialized[column]).dt.strftime(
                "%Y-%m-%d"
            )
    serialized.to_csv(args.signals_output, index=False)
    serialized_exits = exits.reset_index()
    for column in ("signal_date", "exit_date"):
        serialized_exits[column] = pd.to_datetime(
            serialized_exits[column]
        ).dt.strftime("%Y-%m-%d")
    serialized_exits.to_csv(args.exits_output, index=False)

    result = {
        "decision": "reject",
        "hypothesis": (
            "The three continuous inputs of the frozen bearish-divergence "
            "sell rule can predict a 126-session SPX drop of at least 20%."
        ),
        "method": {
            "features": list(SELL_3D_COLUMNS),
            "excluded_features": list(shared.BASE_COLUMNS),
            "target": "SPX future minimum return over 126 sessions <= -20%",
            "scaling": "historical median/IQR",
            "distance": "Euclidean after robust scaling",
            "causal_reference": (
                "only breadth-comparable peak episodes whose -20% breach "
                "predates the query"
            ),
            "new_thresholds": 0,
        },
        "data": {
            "start": strategy.index[0],
            "end": strategy.index[-1],
            "bars": len(strategy),
            "signal_clusters": len(output),
            "causal_comparable_clusters": len(causal_output),
            "causal_positive_clusters": int(labels.sum()),
            "independent_positive_crash_episodes": 1,
            "comparable_peak_episodes": int(
                comparable_peaks["episode"].nunique()
            ),
        },
        "classification": {
            "full_sample_auc_hindsight_only": full_auc,
            "strict_causal_auc": causal_auc,
            "true_median_causal_similarity": causal_output.loc[
                labels, "causal_peak_similarity_percentile"
            ].median(),
            "false_median_causal_similarity": causal_output.loc[
                ~labels, "causal_peak_similarity_percentile"
            ].median(),
        },
        "canonical_bearish_divergence_outcomes": {
            "signals": len(divergence_exits),
            "drops": divergence_drops,
            **shared.jeffreys_rate(
                divergence_drops, len(divergence_exits)
            ),
        },
        "current": {
            "date": current_date,
            "vector": {
                column: current[column] for column in SELL_3D_COLUMNS
            },
            "conditions": {
                "price_condition_met": current[
                    "sell_price_condition_met"
                ],
                "breadth_fall_condition_met": current[
                    "sell_breadth_fall_condition_met"
                ],
                "breadth_cap_condition_met": current[
                    "sell_breadth_cap_condition_met"
                ],
                "bearish_divergence_active": current[
                    "canonical_bearish_divergence"
                ],
            },
            "strategy_sell_state": shared.current_strategy_sell_state(
                strategy, open_trade
            ),
            "causal_peak_similarity_percentile": current_causal[
                "causal_peak_similarity_percentile"
            ],
            "nearest_peak_zone_date": current_causal[
                "causal_nearest_peak_zone_date"
            ],
            "nearest_peak_episode": current_causal[
                "causal_nearest_peak_episode"
            ],
            "nearest_historical_signal_date": nearest_signal_date,
            "nearest_historical_signal_distance": (
                current_signal_distances.iloc[0]
            ),
            "nearest_signal_followed_by_20pct_drop": bool(
                output.loc[nearest_signal_date, "followed_by_20pct_drop"]
            ),
        },
        "decision_reason": (
            "The strict causal result has only one independent positive crash "
            "episode, so the three-dimensional sell vector cannot be "
            "validated or used for a calibrated drop prediction."
        ),
        "limitations": [
            "All positive signal clusters are the same 2020 crash episode.",
            "Pre-2007 breadth is synthetic and the 2000 peak is unavailable.",
            "Full-sample AUC uses future peak labels and is descriptive only.",
            "Similarity percentile is not a probability of loss.",
            "The frozen strategy is unchanged.",
        ],
        "artifacts": {
            "result_json": args.result_output.resolve(),
            "signal_vectors_csv": args.signals_output.resolve(),
            "canonical_exits_csv": args.exits_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(shared._finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(shared._finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
