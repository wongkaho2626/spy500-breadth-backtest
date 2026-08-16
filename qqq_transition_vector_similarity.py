"""Compare triple-transition exit vectors with historical SPX peak vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_triple_trajectory_reversal_sell as transition
import spx_20pct_peak_vector_analysis as peak_analysis


DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_transition_vector_similarity_results.json"
DEFAULT_EVENTS = DATA_DIR / "qqq_transition_exit_vectors.csv"
DEFAULT_NEIGHBORS = DATA_DIR / "qqq_transition_peak_neighbors.csv"
DEFAULT_RAW_SIGNALS = DATA_DIR / "qqq_transition_raw_signal_vectors.csv"
TRANSITION_RESULT = DATA_DIR / "qqq_triple_trajectory_sell_results.json"
PEAK_ZONE_FILE = DATA_DIR / "spx_20pct_peak_zone_vectors.csv"
FEATURE_COLUMNS = (
    "ndx_return_60_pct",
    "ndx_return_60_slope_20",
    "vix",
    "vix_slope_20",
    "spx_drawdown_252_pct",
    "spx_drawdown_252_slope_20",
)
TRAJECTORY_COLUMNS = (
    "ndx_return_60_slope_20",
    "vix_slope_20",
    "spx_drawdown_252_slope_20",
)
NEAREST_PEAK_ROWS = 5


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


def build_full_transition_vector() -> pd.DataFrame:
    vector, _ = peak_analysis.build_full_market_vector()
    vector["ndx_return_60_slope_20"] = transition.trajectory.rolling_linear_slope(
        vector["ndx_return_60_pct"], transition.SLOPE_WINDOW
    )
    return vector.loc[:, FEATURE_COLUMNS]


def robust_scale(values: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    center = values.median()
    scale = values.quantile(0.75) - values.quantile(0.25)
    fallback = values.std(ddof=0)
    scale = scale.where(scale > 0, fallback)
    scale = scale.where(scale > 0, 1.0)
    return center, scale


def standardized_distance(
    query: pd.Series,
    candidates: pd.DataFrame,
    center: pd.Series,
    scale: pd.Series,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.Series:
    query_values = (
        query.loc[list(columns)].astype(float) - center.astype(float)
    ) / scale.astype(float)
    candidate_values = (
        candidates.loc[:, columns]
        .astype(float)
        .sub(center.astype(float))
        .div(scale.astype(float))
    )
    distance = np.sqrt(
        np.square(candidate_values - query_values).sum(axis=1)
    )
    return pd.Series(distance, index=candidates.index)


def load_executed_events() -> pd.DataFrame:
    result = json.loads(TRANSITION_RESULT.read_text())
    outcomes = result["challenger_primary"]["transition_exit_outcomes"][
        "outcomes"
    ]
    frame = pd.DataFrame(outcomes)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"])
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    return frame.set_index("signal_date").sort_index()


def load_peak_rows() -> pd.DataFrame:
    frame = pd.read_csv(PEAK_ZONE_FILE, parse_dates=["Date"])
    for column in ("peak_date", "breach_date", "trough_date", "recovery_date"):
        frame[column] = pd.to_datetime(frame[column])
    return frame.set_index("Date").sort_index()


def causal_prior_peak_similarity(
    signal_date: pd.Timestamp,
    query: pd.Series,
    vector: pd.DataFrame,
    peak_rows: pd.DataFrame,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Compare only with peak episodes confirmed before the signal date."""
    eligible_metadata = peak_rows[
        (peak_rows["breach_date"] < signal_date)
        & (peak_rows.index < signal_date)
    ]
    eligible_peak = vector.reindex(eligible_metadata.index)[list(columns)].dropna()
    history = vector.loc[vector.index < signal_date, list(columns)].dropna()
    ordinary = history.loc[~history.index.isin(eligible_peak.index)]
    center, scale = robust_scale(ordinary)
    distance = standardized_distance(
        query, eligible_peak, center, scale, columns
    ).sort_values()
    peak_array = ((eligible_peak - center) / scale).to_numpy()
    ordinary_array = ((ordinary - center) / scale).to_numpy()
    ordinary_min_distance = np.sqrt(
        np.square(
            ordinary_array[:, None, :] - peak_array[None, :, :]
        ).sum(axis=2)
    ).min(axis=1)
    nearest_date = distance.index[0]
    return {
        "causal_nearest_peak_zone_date": nearest_date,
        "causal_nearest_peak_episode": int(
            peak_rows.loc[nearest_date, "episode"]
        ),
        "causal_nearest_peak_distance": float(distance.iloc[0]),
        "causal_peak_similarity_percentile": float(
            (ordinary_min_distance >= distance.iloc[0]).mean()
        ),
        "resolved_peak_episodes_available": int(
            eligible_metadata["episode"].nunique()
        ),
    }


def similarity_bundle(
    query: pd.Series,
    candidates: pd.DataFrame,
    control: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[str, float]:
    """Return nearest distance and control-calibrated similarity."""
    center, scale = robust_scale(control.loc[:, columns])
    distance = standardized_distance(
        query, candidates.loc[:, columns], center, scale, columns
    ).sort_values()
    candidate_array = (
        candidates.loc[:, columns].sub(center).div(scale).to_numpy()
    )
    control_array = control.loc[:, columns].sub(center).div(scale).to_numpy()
    control_min_distance = np.sqrt(
        np.square(
            control_array[:, None, :] - candidate_array[None, :, :]
        ).sum(axis=2)
    ).min(axis=1)
    return {
        "distance": float(distance.iloc[0]),
        "similarity_percentile": float(
            (control_min_distance >= distance.iloc[0]).mean()
        ),
    }


def rank_auc(labels: pd.Series, scores: pd.Series) -> float:
    """Mann-Whitney rank AUC, with average ranks for ties."""
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = scores.rank(method="average")
    positive_rank_sum = float(ranks.loc[labels].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def pairwise_event_neighbors(
    event_vectors: pd.DataFrame,
    center: pd.Series,
    scale: pd.Series,
    columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> dict[pd.Timestamp, dict[str, Any]]:
    standardized = event_vectors.loc[:, columns].sub(center).div(scale)
    output = {}
    for date, row in standardized.iterrows():
        distances = np.sqrt(np.square(standardized - row).sum(axis=1))
        distances = distances.drop(date)
        nearest_date = distances.idxmin()
        output[date] = {
            "nearest_transition_date": nearest_date,
            "nearest_transition_distance": float(distances.loc[nearest_date]),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--events-output", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument(
        "--neighbors-output", type=Path, default=DEFAULT_NEIGHBORS
    )
    parser.add_argument(
        "--raw-signals-output", type=Path, default=DEFAULT_RAW_SIGNALS
    )
    args = parser.parse_args()

    vector = build_full_transition_vector().dropna()
    events = load_executed_events()
    peak_rows = load_peak_rows()
    peak_vectors = vector.reindex(peak_rows.index).dropna()
    exact_peak_dates = pd.DatetimeIndex(
        pd.to_datetime(peak_rows["peak_date"].unique())
    )
    exact_peak_vectors = vector.reindex(exact_peak_dates).dropna()
    event_vectors = vector.reindex(events.index).dropna()

    excluded_dates = peak_vectors.index.union(event_vectors.index)
    ordinary = vector.loc[~vector.index.isin(excluded_dates)]
    center, scale = robust_scale(ordinary)

    trajectory_center, trajectory_scale = robust_scale(
        ordinary.loc[:, TRAJECTORY_COLUMNS]
    )

    peak_array = ((peak_vectors - center) / scale).to_numpy()
    ordinary_array = ((ordinary - center) / scale).to_numpy()
    ordinary_min_peak_distance = np.sqrt(
        np.square(
            ordinary_array[:, None, :] - peak_array[None, :, :]
        ).sum(axis=2)
    ).min(axis=1)
    trajectory_peak_array = (
        peak_vectors.loc[:, TRAJECTORY_COLUMNS]
        .sub(trajectory_center)
        .div(trajectory_scale)
        .to_numpy()
    )
    trajectory_ordinary_array = (
        ordinary.loc[:, TRAJECTORY_COLUMNS]
        .sub(trajectory_center)
        .div(trajectory_scale)
        .to_numpy()
    )
    trajectory_ordinary_min_peak_distance = np.sqrt(
        np.square(
            trajectory_ordinary_array[:, None, :]
            - trajectory_peak_array[None, :, :]
        ).sum(axis=2)
    ).min(axis=1)

    strategy_df = transition.qbt.load_data()
    strategy_spx = transition.crash.load_spx()["close"].reindex(
        strategy_df.index
    )
    strategy_features = transition.build_transition_features(
        strategy_df, strategy_spx
    )
    raw_signal = transition.transition_signal(
        strategy_features, transition.PRIMARY_WINDOW
    )
    raw_onset = raw_signal & ~raw_signal.shift(1, fill_value=False)
    _, raw_future_return = transition.crash.forward_crash_labels(
        strategy_spx
    )
    raw_vectors = vector.reindex(strategy_df.index[raw_onset]).dropna()
    raw_rows = []
    for signal_date, query in raw_vectors.iterrows():
        peak_distance = standardized_distance(
            query, peak_vectors, center, scale
        ).sort_values()
        nearest_peak_date = peak_distance.index[0]
        metadata = peak_rows.loc[nearest_peak_date]
        future_minimum = float(raw_future_return.loc[signal_date])
        causal = causal_prior_peak_similarity(
            signal_date, query, vector, peak_rows
        )
        trajectory_distance = standardized_distance(
            query,
            peak_vectors,
            trajectory_center,
            trajectory_scale,
            TRAJECTORY_COLUMNS,
        ).sort_values()
        causal_trajectory = causal_prior_peak_similarity(
            signal_date,
            query,
            vector,
            peak_rows,
            TRAJECTORY_COLUMNS,
        )
        raw_rows.append(
            {
                "signal_date": signal_date,
                "future_min_spx_return_126": future_minimum,
                "followed_by_20pct_drop": bool(future_minimum <= -0.20),
                **query.to_dict(),
                "nearest_peak_zone_date": nearest_peak_date,
                "nearest_peak_episode": int(metadata["episode"]),
                "nearest_peak_phase": metadata["phase"],
                "nearest_peak_zone_distance": float(
                    peak_distance.iloc[0]
                ),
                "peak_similarity_percentile": float(
                    (
                        ordinary_min_peak_distance
                        >= peak_distance.iloc[0]
                    ).mean()
                ),
                "trajectory_nearest_peak_zone_distance": float(
                    trajectory_distance.iloc[0]
                ),
                "trajectory_peak_similarity_percentile": float(
                    (
                        trajectory_ordinary_min_peak_distance
                        >= trajectory_distance.iloc[0]
                    ).mean()
                ),
                "causal_trajectory_peak_similarity_percentile": (
                    causal_trajectory[
                        "causal_peak_similarity_percentile"
                    ]
                ),
                **causal,
            }
        )
    raw_output = pd.DataFrame(raw_rows).set_index("signal_date")
    raw_trajectory_neighbors = pairwise_event_neighbors(
        raw_vectors,
        trajectory_center,
        trajectory_scale,
        TRAJECTORY_COLUMNS,
    )
    for signal_date, neighbor in raw_trajectory_neighbors.items():
        nearest_date = neighbor["nearest_transition_date"]
        raw_output.loc[
            signal_date, "trajectory_nearest_signal_date"
        ] = nearest_date
        raw_output.loc[
            signal_date, "trajectory_nearest_signal_distance"
        ] = neighbor["nearest_transition_distance"]
        raw_output.loc[
            signal_date,
            "trajectory_nearest_signal_followed_by_20pct_drop",
        ] = bool(raw_output.loc[nearest_date, "followed_by_20pct_drop"])

    episode_centroids = peak_vectors.join(
        peak_rows[["episode"]], how="left"
    ).groupby("episode")[list(FEATURE_COLUMNS)].median()

    pair_neighbors = pairwise_event_neighbors(
        event_vectors, center, scale
    )
    event_rows = []
    neighbor_rows = []
    for signal_date, event in events.iterrows():
        query = event_vectors.loc[signal_date]
        peak_distance = standardized_distance(
            query, peak_vectors, center, scale
        ).sort_values()
        exact_distance = standardized_distance(
            query, exact_peak_vectors, center, scale
        ).sort_values()
        episode_distance = standardized_distance(
            query, episode_centroids, center, scale
        ).sort_values()
        similarity_percentile = float(
            (ordinary_min_peak_distance >= peak_distance.iloc[0]).mean()
        )
        nearest_peak_date = peak_distance.index[0]
        nearest_metadata = peak_rows.loc[nearest_peak_date]
        pair = pair_neighbors[signal_date]
        nearest_other_outcome = events.loc[
            pair["nearest_transition_date"], "followed_by_20pct_drop"
        ]
        causal = causal_prior_peak_similarity(
            signal_date, query, vector, peak_rows
        )
        trajectory_distance = standardized_distance(
            query,
            peak_vectors,
            trajectory_center,
            trajectory_scale,
            TRAJECTORY_COLUMNS,
        ).sort_values()
        causal_trajectory = causal_prior_peak_similarity(
            signal_date,
            query,
            vector,
            peak_rows,
            TRAJECTORY_COLUMNS,
        )
        row = {
            "signal_date": signal_date,
            "exit_date": event["exit_date"],
            "followed_by_20pct_drop": bool(
                event["followed_by_20pct_drop"]
            ),
            "future_min_spx_return_126": event[
                "future_min_spx_return_126"
            ],
            **query.to_dict(),
            "nearest_peak_zone_date": nearest_peak_date,
            "nearest_peak_episode": int(nearest_metadata["episode"]),
            "nearest_peak_phase": nearest_metadata["phase"],
            "nearest_peak_zone_distance": float(peak_distance.iloc[0]),
            "peak_similarity_percentile": similarity_percentile,
            "trajectory_nearest_peak_zone_distance": float(
                trajectory_distance.iloc[0]
            ),
            "trajectory_peak_similarity_percentile": float(
                (
                    trajectory_ordinary_min_peak_distance
                    >= trajectory_distance.iloc[0]
                ).mean()
            ),
            "causal_trajectory_peak_similarity_percentile": (
                causal_trajectory["causal_peak_similarity_percentile"]
            ),
            "nearest_exact_peak_date": exact_distance.index[0],
            "nearest_exact_peak_distance": float(exact_distance.iloc[0]),
            "nearest_episode_centroid": int(episode_distance.index[0]),
            "nearest_episode_centroid_distance": float(
                episode_distance.iloc[0]
            ),
            **pair,
            "nearest_transition_followed_by_20pct_drop": bool(
                nearest_other_outcome
            ),
            **causal,
        }
        event_rows.append(row)

        for rank, (peak_date, distance) in enumerate(
            peak_distance.iloc[:NEAREST_PEAK_ROWS].items(), start=1
        ):
            metadata = peak_rows.loc[peak_date]
            neighbor_rows.append(
                {
                    "signal_date": signal_date,
                    "rank": rank,
                    "peak_zone_date": peak_date,
                    "episode": int(metadata["episode"]),
                    "phase": metadata["phase"],
                    "sessions_from_peak": int(
                        metadata["sessions_from_peak"]
                    ),
                    "distance": float(distance),
                }
            )

    event_output = pd.DataFrame(event_rows).set_index("signal_date")
    neighbors_output = pd.DataFrame(neighbor_rows)

    current_date = vector.index[-1]
    current = vector.loc[current_date]
    current_peak_distance = standardized_distance(
        current, peak_vectors, center, scale
    ).sort_values()
    current_event_distance = standardized_distance(
        current, event_vectors, center, scale
    ).sort_values()
    current_similarity = float(
        (ordinary_min_peak_distance >= current_peak_distance.iloc[0]).mean()
    )
    current_trajectory_distance = standardized_distance(
        current,
        peak_vectors,
        trajectory_center,
        trajectory_scale,
        TRAJECTORY_COLUMNS,
    ).sort_values()
    current_trajectory_similarity = float(
        (
            trajectory_ordinary_min_peak_distance
            >= current_trajectory_distance.iloc[0]
        ).mean()
    )
    current_raw_trajectory_distance = standardized_distance(
        current,
        raw_vectors,
        trajectory_center,
        trajectory_scale,
        TRAJECTORY_COLUMNS,
    ).sort_values()

    true_events = event_output[event_output["followed_by_20pct_drop"]]
    false_events = event_output[~event_output["followed_by_20pct_drop"]]
    comparison = {
        "true_event_count": len(true_events),
        "false_event_count": len(false_events),
        "true_event_peak_similarity_percentile": (
            true_events["peak_similarity_percentile"].median()
        ),
        "false_event_median_peak_similarity_percentile": (
            false_events["peak_similarity_percentile"].median()
        ),
        "false_event_similarity_range": [
            false_events["peak_similarity_percentile"].min(),
            false_events["peak_similarity_percentile"].max(),
        ],
        "events_above_90pct_peak_similarity": int(
            (event_output["peak_similarity_percentile"] >= 0.90).sum()
        ),
        "true_events_above_90pct_peak_similarity": int(
            (
                true_events["peak_similarity_percentile"] >= 0.90
            ).sum()
        ),
        "true_event_causal_peak_similarity_percentile": true_events[
            "causal_peak_similarity_percentile"
        ].median(),
        "true_event_trajectory_similarity_percentile": true_events[
            "trajectory_peak_similarity_percentile"
        ].median(),
        "false_event_median_trajectory_similarity_percentile": (
            false_events["trajectory_peak_similarity_percentile"].median()
        ),
        "true_event_causal_trajectory_similarity_percentile": true_events[
            "causal_trajectory_peak_similarity_percentile"
        ].median(),
        "false_event_median_causal_trajectory_similarity_percentile": (
            false_events[
                "causal_trajectory_peak_similarity_percentile"
            ].median()
        ),
        "false_event_median_causal_peak_similarity_percentile": false_events[
            "causal_peak_similarity_percentile"
        ].median(),
    }
    raw_true = raw_output[raw_output["followed_by_20pct_drop"]]
    raw_false = raw_output[~raw_output["followed_by_20pct_drop"]]
    raw_comparison = {
        "signal_clusters": len(raw_output),
        "true_clusters": len(raw_true),
        "false_clusters": len(raw_false),
        "precision": len(raw_true) / len(raw_output),
        "true_median_peak_similarity_percentile": raw_true[
            "peak_similarity_percentile"
        ].median(),
        "false_median_peak_similarity_percentile": raw_false[
            "peak_similarity_percentile"
        ].median(),
        "true_similarity_range": [
            raw_true["peak_similarity_percentile"].min(),
            raw_true["peak_similarity_percentile"].max(),
        ],
        "false_similarity_range": [
            raw_false["peak_similarity_percentile"].min(),
            raw_false["peak_similarity_percentile"].max(),
        ],
        "true_median_causal_peak_similarity_percentile": raw_true[
            "causal_peak_similarity_percentile"
        ].median(),
        "false_median_causal_peak_similarity_percentile": raw_false[
            "causal_peak_similarity_percentile"
        ].median(),
        "true_causal_similarity_range": [
            raw_true["causal_peak_similarity_percentile"].min(),
            raw_true["causal_peak_similarity_percentile"].max(),
        ],
        "false_causal_similarity_range": [
            raw_false["causal_peak_similarity_percentile"].min(),
            raw_false["causal_peak_similarity_percentile"].max(),
        ],
        "six_dim_similarity_auc": rank_auc(
            raw_output["followed_by_20pct_drop"],
            raw_output["peak_similarity_percentile"],
        ),
        "causal_six_dim_similarity_auc": rank_auc(
            raw_output["followed_by_20pct_drop"],
            raw_output["causal_peak_similarity_percentile"],
        ),
        "trajectory_similarity_auc": rank_auc(
            raw_output["followed_by_20pct_drop"],
            raw_output["trajectory_peak_similarity_percentile"],
        ),
        "causal_trajectory_similarity_auc": rank_auc(
            raw_output["followed_by_20pct_drop"],
            raw_output[
                "causal_trajectory_peak_similarity_percentile"
            ],
        ),
        "true_median_trajectory_similarity_percentile": raw_true[
            "trajectory_peak_similarity_percentile"
        ].median(),
        "false_median_trajectory_similarity_percentile": raw_false[
            "trajectory_peak_similarity_percentile"
        ].median(),
        "true_median_causal_trajectory_similarity_percentile": raw_true[
            "causal_trajectory_peak_similarity_percentile"
        ].median(),
        "false_median_causal_trajectory_similarity_percentile": raw_false[
            "causal_trajectory_peak_similarity_percentile"
        ].median(),
        "true_vector_median": raw_true[list(FEATURE_COLUMNS)]
        .median()
        .to_dict(),
        "false_vector_median": raw_false[list(FEATURE_COLUMNS)]
        .median()
        .to_dict(),
    }

    serializable_events = event_output.reset_index().copy()
    date_columns = [
        "signal_date",
        "exit_date",
        "nearest_peak_zone_date",
        "nearest_exact_peak_date",
        "nearest_transition_date",
    ]
    for column in date_columns:
        serializable_events[column] = pd.to_datetime(
            serializable_events[column]
        ).dt.strftime("%Y-%m-%d")
    serializable_events.to_csv(args.events_output, index=False)
    for column in ("signal_date", "peak_zone_date"):
        neighbors_output[column] = pd.to_datetime(
            neighbors_output[column]
        ).dt.strftime("%Y-%m-%d")
    neighbors_output.to_csv(args.neighbors_output, index=False)
    serializable_raw = raw_output.reset_index()
    for column in (
        "signal_date",
        "nearest_peak_zone_date",
        "trajectory_nearest_signal_date",
    ):
        serializable_raw[column] = pd.to_datetime(
            serializable_raw[column]
        ).dt.strftime("%Y-%m-%d")
    serializable_raw.to_csv(args.raw_signals_output, index=False)

    result = {
        "method": {
            "features": list(FEATURE_COLUMNS),
            "trajectory_features": list(TRAJECTORY_COLUMNS),
            "scaling": "ordinary-market median and IQR",
            "distance": "Euclidean distance after robust scaling",
            "peak_reference_rows": len(peak_vectors),
            "exact_peak_rows": len(exact_peak_vectors),
            "ordinary_control_rows": len(ordinary),
            "similarity_percentile": (
                "fraction of ordinary days whose nearest peak-zone distance "
                "is greater than or equal to the event distance"
            ),
        },
        "event_comparison": comparison,
        "raw_signal_comparison": raw_comparison,
        "events": event_output.reset_index().to_dict(orient="records"),
        "current": {
            "date": current_date,
            "vector": current.to_dict(),
            "nearest_peak_zone_date": current_peak_distance.index[0],
            "nearest_peak_episode": int(
                peak_rows.loc[current_peak_distance.index[0], "episode"]
            ),
            "nearest_peak_zone_distance": current_peak_distance.iloc[0],
            "peak_similarity_percentile": current_similarity,
            "trajectory_nearest_peak_zone_date": (
                current_trajectory_distance.index[0]
            ),
            "trajectory_nearest_peak_zone_distance": (
                current_trajectory_distance.iloc[0]
            ),
            "trajectory_peak_similarity_percentile": (
                current_trajectory_similarity
            ),
            "trajectory_nearest_signal_date": (
                current_raw_trajectory_distance.index[0]
            ),
            "trajectory_nearest_signal_distance": (
                current_raw_trajectory_distance.iloc[0]
            ),
            "trajectory_nearest_signal_followed_by_20pct_drop": bool(
                raw_output.loc[
                    current_raw_trajectory_distance.index[0],
                    "followed_by_20pct_drop",
                ]
            ),
            "nearest_transition_date": current_event_distance.index[0],
            "nearest_transition_distance": current_event_distance.iloc[0],
        "nearest_transition_followed_by_20pct_drop": bool(
                events.loc[
                    current_event_distance.index[0],
                    "followed_by_20pct_drop",
                ]
            ),
        },
        "limitations": [
            "Only four independent peak episodes and one true transition exit exist.",
            "Peak dates are identified with hindsight; similarity is descriptive, not a forecast.",
            "Full-sample robust scaling is used for diagnosis and must not be treated as a live model.",
            "Nearby peak-zone rows are correlated; they are not 86 independent crashes.",
        ],
        "artifacts": {
            "results_json": args.result_output.resolve(),
            "event_vectors_csv": args.events_output.resolve(),
            "nearest_peak_rows_csv": args.neighbors_output.resolve(),
            "raw_signal_vectors_csv": args.raw_signals_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
