"""Test whether the frozen bearish-divergence sell state improves crash vectors.

The challenger adds one causal feature to the existing six-dimensional
trajectory vector: the fraction of the three frozen bearish-divergence
conditions currently met. It is a diagnostic classifier, not a strategy rule.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import qqq_transition_vector_similarity as base_analysis


DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_current_sell_state_vector_results.json"
DEFAULT_SIGNALS = DATA_DIR / "qqq_current_sell_state_vector_signals.csv"
DEFAULT_EXITS = DATA_DIR / "qqq_current_sell_state_vector_exits.csv"
BASE_COLUMNS = base_analysis.FEATURE_COLUMNS
SELL_FEATURE = "canonical_sell_vote_fraction"
AUGMENTED_COLUMNS = (*BASE_COLUMNS, SELL_FEATURE)


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


def canonical_sell_vote_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the three frozen divergence votes using close-time data only."""
    qbt = base_analysis.transition.qbt
    price_anchor = df["price"].shift(qbt.DIVERGENCE_WINDOW)
    breadth_anchor = df["breadth"].shift(qbt.DIVERGENCE_WINDOW)
    ndx_return = (df["price"] / price_anchor - 1).mul(100)
    breadth_fall = breadth_anchor - df["breadth"]
    price_met = ndx_return >= qbt.DIVERGENCE_PRICE_RISE
    breadth_fall_met = breadth_fall >= qbt.DIVERGENCE_BREADTH_FALL
    breadth_cap_met = df["breadth"] < qbt.DIVERGENCE_BREADTH_CAP
    vote_fraction = pd.concat(
        [price_met, breadth_fall_met, breadth_cap_met], axis=1
    ).astype(float).mean(axis=1)
    return pd.DataFrame(
        {
            "sell_ndx_return_60_pct": ndx_return,
            "sell_breadth_fall_60_points": breadth_fall,
            "sell_breadth_level_pct": df["breadth"],
            "sell_price_condition_met": price_met,
            "sell_breadth_fall_condition_met": breadth_fall_met,
            "sell_breadth_cap_condition_met": breadth_cap_met,
            SELL_FEATURE: vote_fraction,
            "canonical_bearish_divergence": (
                price_met & breadth_fall_met & breadth_cap_met
            ),
        },
        index=df.index,
    )


def build_augmented_vector() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = base_analysis.build_full_transition_vector()
    strategy = base_analysis.transition.qbt.load_data()
    sell = canonical_sell_vote_frame(strategy)
    augmented = base.join(sell, how="left")
    return augmented, strategy, sell


def full_sample_similarity(
    vector: pd.DataFrame,
    peak_rows: pd.DataFrame,
    query_dates: pd.DatetimeIndex,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    peak_vectors = vector.reindex(peak_rows.index).dropna(subset=list(columns))
    ordinary = vector.loc[
        ~vector.index.isin(peak_vectors.index), list(columns)
    ].dropna()
    center, scale = base_analysis.robust_scale(ordinary)
    peak_array = peak_vectors.loc[:, columns].sub(center).div(scale).to_numpy()
    ordinary_array = ordinary.sub(center).div(scale).to_numpy()
    ordinary_min_distance = np.sqrt(
        np.square(
            ordinary_array[:, None, :] - peak_array[None, :, :]
        ).sum(axis=2)
    ).min(axis=1)
    rows = []
    for date in query_dates:
        query = vector.loc[date]
        distance = base_analysis.standardized_distance(
            query, peak_vectors, center, scale, columns
        ).sort_values()
        rows.append(
            {
                "signal_date": date,
                "nearest_peak_zone_date": distance.index[0],
                "nearest_peak_distance": float(distance.iloc[0]),
                "peak_similarity_percentile": float(
                    (ordinary_min_distance >= distance.iloc[0]).mean()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("signal_date")


def causal_similarity_rows(
    vector: pd.DataFrame,
    peak_rows: pd.DataFrame,
    query_dates: pd.DatetimeIndex,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for date in query_dates:
        try:
            result = base_analysis.causal_prior_peak_similarity(
                date, vector.loc[date], vector, peak_rows, columns
            )
        except (IndexError, ValueError):
            continue
        rows.append(
            {
                "signal_date": date,
                "causal_nearest_peak_zone_date": result[
                    "causal_nearest_peak_zone_date"
                ],
                "causal_nearest_peak_distance": result[
                    "causal_nearest_peak_distance"
                ],
                "causal_peak_similarity_percentile": result[
                    "causal_peak_similarity_percentile"
                ],
                "resolved_peak_episodes_available": result[
                    "resolved_peak_episodes_available"
                ],
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("signal_date")


def raw_signal_dates_and_labels(
    strategy: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, pd.Series]:
    spx = base_analysis.transition.crash.load_spx()["close"].reindex(
        strategy.index
    )
    features = base_analysis.transition.build_transition_features(strategy, spx)
    raw = base_analysis.transition.transition_signal(
        features, base_analysis.transition.PRIMARY_WINDOW
    )
    onset = raw & ~raw.shift(1, fill_value=False)
    _, future_return = base_analysis.transition.crash.forward_crash_labels(spx)
    return strategy.index[onset], future_return


def current_strategy_sell_state(
    strategy: pd.DataFrame,
    open_trade: dict | None,
) -> dict[str, Any]:
    qbt = base_analysis.transition.qbt
    row = strategy.iloc[-1]
    divergence = bool(
        row["price_rose"]
        and row["breadth_fell"]
        and row["breadth"] < qbt.DIVERGENCE_BREADTH_CAP
    )
    if open_trade is None:
        return {
            "position": "OUT",
            "bearish_divergence_active": divergence,
            "climax_top_active": False,
            "trailing_stop_active": False,
            "canonical_sell_reason": None,
        }

    after_entry = strategy.loc[strategy.index > open_trade["entry_date"]]

    def age_since(event: pd.Series) -> int:
        locations = np.flatnonzero(event.to_numpy(dtype=bool))
        if len(locations) == 0:
            return 10**9
        return len(event) - 1 - int(locations[-1])

    macd_age = age_since(after_entry["macd_cross"])
    ext_age = age_since(after_entry["ext10"])
    climax = bool(
        macd_age < qbt.CLIMAX_VOTE_WINDOW
        and ext_age < qbt.CLIMAX_VOTE_WINDOW
    )
    trade_high = max(
        float(open_trade["entry_price"]),
        float(after_entry["price"].max()),
    )
    drawdown = (float(row["price"]) / trade_high - 1) * 100
    trailing = bool(drawdown <= -qbt.TRAILING_STOP_PCT)
    if divergence:
        reason = "bearish-divergence"
    elif climax:
        reason = "climax-top"
    elif trailing:
        reason = "trailing-stop"
    else:
        reason = None
    return {
        "position": "IN",
        "bearish_divergence_active": divergence,
        "climax_top_active": climax,
        "trailing_stop_active": trailing,
        "canonical_sell_reason": reason,
        "macd_cross_age_sessions": macd_age,
        "extension_age_sessions": ext_age,
        "drawdown_from_trade_high_pct": drawdown,
    }


def canonical_exit_outcomes(
    strategy: pd.DataFrame,
    trades: list[dict],
    future_return: pd.Series,
    vector: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for trade in trades:
        exit_location = strategy.index.get_loc(trade["exit_date"])
        signal_date = strategy.index[exit_location - base_analysis.transition.qbt.EXECUTION_LAG]
        future_minimum = float(future_return.loc[signal_date])
        rows.append(
            {
                "signal_date": signal_date,
                "exit_date": trade["exit_date"],
                "sell_reason": trade["sell_reason"],
                SELL_FEATURE: vector.loc[signal_date, SELL_FEATURE],
                "future_min_spx_return_126": future_minimum,
                "followed_by_20pct_drop": bool(future_minimum <= -0.20),
            }
        )
    return pd.DataFrame(rows).set_index("signal_date")


def jeffreys_rate(successes: int, observations: int) -> dict[str, Any]:
    alpha = 0.5 + successes
    beta = 0.5 + observations - successes
    return {
        "raw_rate": successes / observations if observations else np.nan,
        "posterior_probability": alpha / (alpha + beta),
        "jeffreys_90_interval": stats.beta.ppf(
            [0.05, 0.95], alpha, beta
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--signals-output", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--exits-output", type=Path, default=DEFAULT_EXITS)
    args = parser.parse_args()

    vector, strategy, sell = build_augmented_vector()
    complete = vector.dropna(subset=list(AUGMENTED_COLUMNS))
    peak_rows = base_analysis.load_peak_rows()
    comparable_peak_rows = peak_rows.loc[
        peak_rows.index.intersection(complete.index)
    ]
    signal_dates, future_return = raw_signal_dates_and_labels(strategy)
    signal_dates = signal_dates.intersection(complete.index)

    baseline_full = full_sample_similarity(
        complete, comparable_peak_rows, signal_dates, BASE_COLUMNS
    ).add_prefix("baseline_")
    challenger_full = full_sample_similarity(
        complete, comparable_peak_rows, signal_dates, AUGMENTED_COLUMNS
    ).add_prefix("challenger_")
    baseline_causal = causal_similarity_rows(
        complete, comparable_peak_rows, signal_dates, BASE_COLUMNS
    ).add_prefix("baseline_")
    challenger_causal = causal_similarity_rows(
        complete, comparable_peak_rows, signal_dates, AUGMENTED_COLUMNS
    ).add_prefix("challenger_")

    signal_output = pd.DataFrame(index=signal_dates)
    signal_output.index.name = "signal_date"
    signal_output["future_min_spx_return_126"] = future_return.reindex(
        signal_dates
    )
    signal_output["followed_by_20pct_drop"] = (
        signal_output["future_min_spx_return_126"] <= -0.20
    )
    signal_output = signal_output.join(
        complete.loc[signal_dates, list(AUGMENTED_COLUMNS)]
    )
    for frame in (
        baseline_full,
        challenger_full,
        baseline_causal,
        challenger_causal,
    ):
        signal_output = signal_output.join(frame)

    causal = signal_output.dropna(
        subset=[
            "baseline_causal_peak_similarity_percentile",
            "challenger_causal_peak_similarity_percentile",
        ]
    )
    labels = causal["followed_by_20pct_drop"]
    baseline_auc = base_analysis.rank_auc(
        labels, causal["baseline_causal_peak_similarity_percentile"]
    )
    challenger_auc = base_analysis.rank_auc(
        labels, causal["challenger_causal_peak_similarity_percentile"]
    )

    qbt = base_analysis.transition.qbt
    baseline_equity, trades, open_trade = qbt.run_strategy(
        strategy,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    exits = canonical_exit_outcomes(
        strategy, trades, future_return, complete
    )

    current_date = complete.index[-1]
    current_base = base_analysis.causal_prior_peak_similarity(
        current_date,
        complete.loc[current_date],
        complete,
        comparable_peak_rows,
        BASE_COLUMNS,
    )
    current_challenger = base_analysis.causal_prior_peak_similarity(
        current_date,
        complete.loc[current_date],
        complete,
        comparable_peak_rows,
        AUGMENTED_COLUMNS,
    )
    current_sell = sell.loc[current_date]

    reason_summary = {}
    for reason, group in exits.groupby("sell_reason"):
        drops = int(group["followed_by_20pct_drop"].sum())
        reason_summary[reason] = {
            "signals": len(group),
            "drops": drops,
            **jeffreys_rate(drops, len(group)),
        }

    serializable_signals = signal_output.reset_index()
    for column in serializable_signals.columns:
        if column.endswith("date"):
            serializable_signals[column] = pd.to_datetime(
                serializable_signals[column]
            ).dt.strftime("%Y-%m-%d")
    serializable_signals.to_csv(args.signals_output, index=False)
    serializable_exits = exits.reset_index()
    for column in ("signal_date", "exit_date"):
        serializable_exits[column] = pd.to_datetime(
            serializable_exits[column]
        ).dt.strftime("%Y-%m-%d")
    serializable_exits.to_csv(args.exits_output, index=False)

    result = {
        "hypothesis": (
            "Adding the frozen bearish-divergence vote fraction to the six-"
            "dimensional trajectory vector improves causal ranking of a "
            "126-session SPX drop of at least 20%."
        ),
        "method": {
            "baseline_features": list(BASE_COLUMNS),
            "challenger_features": list(AUGMENTED_COLUMNS),
            "sell_vote_definition": (
                "fraction of frozen divergence conditions met: NDX 60d "
                "return >=3%, breadth 60d fall >=20 points, breadth <60%"
            ),
            "target": "SPX future minimum return over 126 sessions <= -20%",
            "scaling": "expanding historical median/IQR",
            "distance": "Euclidean after robust scaling",
            "causal_reference": (
                "only peak episodes whose -20% breach predates the query"
            ),
            "new_thresholds": 0,
        },
        "data": {
            "start": strategy.index[0],
            "end": strategy.index[-1],
            "bars": len(strategy),
            "raw_signal_clusters": len(signal_output),
            "causal_comparable_clusters": len(causal),
            "causal_positive_clusters": int(labels.sum()),
            "independent_positive_crash_episodes": 1,
            "comparable_peak_episodes": int(
                comparable_peak_rows["episode"].nunique()
            ),
            "real_breadth_start": "2007-01-01",
            "clean_forward_oos_start": "2026-07-05",
        },
        "causal_comparison": {
            "baseline_auc": baseline_auc,
            "challenger_auc": challenger_auc,
            "auc_delta": challenger_auc - baseline_auc,
            "baseline_true_median_similarity": causal.loc[
                labels, "baseline_causal_peak_similarity_percentile"
            ].median(),
            "baseline_false_median_similarity": causal.loc[
                ~labels, "baseline_causal_peak_similarity_percentile"
            ].median(),
            "challenger_true_median_similarity": causal.loc[
                labels, "challenger_causal_peak_similarity_percentile"
            ].median(),
            "challenger_false_median_similarity": causal.loc[
                ~labels, "challenger_causal_peak_similarity_percentile"
            ].median(),
        },
        "canonical_exit_outcomes": {
            "signals": len(exits),
            "drops": int(exits["followed_by_20pct_drop"].sum()),
            **jeffreys_rate(
                int(exits["followed_by_20pct_drop"].sum()), len(exits)
            ),
            "by_reason": reason_summary,
        },
        "current": {
            "date": current_date,
            "sell_vote_fraction": current_sell[SELL_FEATURE],
            "conditions": {
                "price_condition_met": current_sell[
                    "sell_price_condition_met"
                ],
                "breadth_fall_condition_met": current_sell[
                    "sell_breadth_fall_condition_met"
                ],
                "breadth_cap_condition_met": current_sell[
                    "sell_breadth_cap_condition_met"
                ],
                "bearish_divergence_active": current_sell[
                    "canonical_bearish_divergence"
                ],
            },
            "strategy_sell_state": current_strategy_sell_state(
                strategy, open_trade
            ),
            "baseline_causal_peak_similarity_percentile": current_base[
                "causal_peak_similarity_percentile"
            ],
            "challenger_causal_peak_similarity_percentile": (
                current_challenger["causal_peak_similarity_percentile"]
            ),
            "challenger_nearest_peak_zone_date": current_challenger[
                "causal_nearest_peak_zone_date"
            ],
            "challenger_nearest_peak_episode": current_challenger[
                "causal_nearest_peak_episode"
            ],
        },
        "decision": "reject",
        "decision_reason": (
            "Causal AUC improved, but every positive cluster is the same "
            "2020 crash episode, triggering the pre-registered single-episode "
            "falsification rule; canonical divergence precision is also low."
        ),
        "limitations": [
            "All positive raw clusters belong to the same 2020 crash episode.",
            "Pre-2007 breadth is synthetic and the 2000 peak lacks a comparable sell vote.",
            "Similarity percentile is a rank, not a calibrated probability of loss.",
            "Only observations after 2026-07-05 are clean forward OOS and there is no completed forward trade.",
            "The frozen baseline strategy and execution are unchanged by this diagnostic.",
        ],
        "artifacts": {
            "result_json": args.result_output.resolve(),
            "signal_vectors_csv": args.signals_output.resolve(),
            "canonical_exits_csv": args.exits_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
