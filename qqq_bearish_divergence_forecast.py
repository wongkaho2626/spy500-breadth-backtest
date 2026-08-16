"""Nearest-neighbour forecast for the canonical QQQ bearish-divergence exit.

The model represents each invested close with six causal features and compares
the latest state with one nearest historical state per completed trade. Other
exit reasons are treated as competing risks rather than pretending that every
historical position eventually reached bearish divergence.

This is a descriptive analogue model, not a price target or trading rule.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# Keep the forecast reproducible and read-only: importing qqq_backtest normally
# invokes the repository's network updater.
_fetch_stub = types.ModuleType("fetch_investing_data")
_fetch_stub.fetch_all_updates = lambda verbose=True: None
sys.modules["fetch_investing_data"] = _fetch_stub

import qqq_backtest as qbt  # noqa: E402


DATA_DIR = Path(__file__).parent
DEFAULT_VECTOR_OUTPUT = DATA_DIR / "qqq_bearish_divergence_vector.csv"
DEFAULT_FORECAST_OUTPUT = DATA_DIR / "qqq_bearish_divergence_forecast.json"
SPY_PROXY_FILE = DATA_DIR / "SPX.csv"
DEFAULT_NEIGHBORS = 7
SENSITIVITY_NEIGHBORS = (5, 7, 9)
FORECAST_HORIZONS = (20, 60, 126, 252)
FEATURE_COLUMNS = (
    "spy_daily_change_pct",
    "ndx_return_60_pct",
    "breadth",
    "breadth_fall_60_points",
    "drawdown_from_trade_high_pct",
    "vix",
)


def load_spy_daily_change(path: Path = SPY_PROXY_FILE) -> pd.Series:
    """Load close-to-close S&P 500 change as the local SPY return proxy."""
    spy = pd.read_csv(path)
    spy.columns = [column.strip().strip('"').lstrip("﻿") for column in spy]
    spy["Date"] = pd.to_datetime(spy["Date"], format="%m/%d/%Y")
    spy["Price"] = (
        spy["Price"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    spy = spy.sort_values("Date").set_index("Date")
    change = spy["Price"].pct_change(fill_method=None).mul(100)
    change.name = "spy_daily_change_pct"
    return change


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


def _signal_date(
    index: pd.DatetimeIndex,
    fill_date: pd.Timestamp,
    execution_lag: int = qbt.EXECUTION_LAG,
) -> pd.Timestamp:
    location = index.get_loc(fill_date)
    if not isinstance(location, (int, np.integer)):
        raise ValueError(f"duplicate or missing fill date: {fill_date}")
    signal_location = int(location) - execution_lag
    if signal_location < 0:
        raise ValueError(f"fill date has no preceding signal bar: {fill_date}")
    return index[signal_location]


def _trade_feature_rows(
    features: pd.DataFrame,
    index: pd.DatetimeIndex,
    trade_id: int,
    entry_date: pd.Timestamp,
    entry_price: float,
    end_date: pd.Timestamp,
    outcome_reason: str | None,
) -> pd.DataFrame:
    """Return causal daily vectors for one position through its signal bar."""
    start_location = index.get_loc(entry_date)
    end_location = index.get_loc(end_date)
    if not isinstance(start_location, (int, np.integer)) or not isinstance(
        end_location, (int, np.integer)
    ):
        raise ValueError("trade dates must be unique members of the price index")
    if end_location < start_location:
        raise ValueError("trade end precedes entry")

    dates = index[int(start_location) : int(end_location) + 1]
    rows = features.loc[dates].copy()

    # Canonical state initializes the high at the fill-bar open and starts
    # updating it with closes on later bars.
    price = rows["price"]
    high_input = price.copy()
    high_input.iloc[0] = float(entry_price)
    trade_high = high_input.cummax().clip(lower=float(entry_price))
    rows["drawdown_from_trade_high_pct"] = (
        price / trade_high - 1
    ).mul(100)
    rows["trade_id"] = trade_id
    rows["entry_date"] = entry_date
    rows["outcome_reason"] = outcome_reason
    rows["outcome_signal_date"] = end_date if outcome_reason else pd.NaT

    if outcome_reason:
        remaining = int(end_location) - np.arange(
            int(start_location), int(end_location) + 1
        )
        rows["sessions_to_outcome"] = remaining
        rows["bearish_divergence_before_other_exit"] = (
            outcome_reason == "bearish-divergence"
        )
    else:
        rows["sessions_to_outcome"] = np.nan
        rows["bearish_divergence_before_other_exit"] = pd.NA
    return rows


def build_vector_frame(
    df: pd.DataFrame,
    trades: list[dict],
    open_trade: dict | None,
) -> pd.DataFrame:
    """Build all invested daily state vectors and separated future labels."""
    required = {"price", "breadth", "vix", "spy_daily_change_pct"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"canonical data is missing columns: {sorted(missing)}")

    price_anchor = df["price"].shift(qbt.DIVERGENCE_WINDOW)
    breadth_anchor = df["breadth"].shift(qbt.DIVERGENCE_WINDOW)
    base = pd.DataFrame(
        {
            "price": df["price"],
            "spy_daily_change_pct": df["spy_daily_change_pct"],
            "ndx_return_60_pct": (
                df["price"] / price_anchor - 1
            ).mul(100),
            "breadth": df["breadth"],
            "breadth_fall_60_points": breadth_anchor - df["breadth"],
            "vix": df["vix"],
        },
        index=df.index,
    )

    frames = []
    for trade_id, trade in enumerate(trades, start=1):
        signal_date = _signal_date(df.index, trade["exit_date"])
        frames.append(
            _trade_feature_rows(
                base,
                df.index,
                trade_id,
                trade["entry_date"],
                trade["entry_price"],
                signal_date,
                trade["sell_reason"],
            )
        )

    if open_trade:
        frames.append(
            _trade_feature_rows(
                base,
                df.index,
                len(trades) + 1,
                open_trade["entry_date"],
                open_trade["entry_price"],
                df.index[-1],
                None,
            )
        )
    if not frames:
        raise ValueError("the canonical strategy has no invested observations")

    vector = pd.concat(frames).dropna(subset=list(FEATURE_COLUMNS))
    vector.index.name = "Date"
    vector["is_current"] = False
    if open_trade and df.index[-1] in vector.index:
        vector.loc[df.index[-1], "is_current"] = True
    return vector


def robust_scale(historical: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Fit median/IQR scaling using historical feature rows only."""
    center = historical.loc[:, FEATURE_COLUMNS].median()
    scale = (
        historical.loc[:, FEATURE_COLUMNS].quantile(0.75)
        - historical.loc[:, FEATURE_COLUMNS].quantile(0.25)
    )
    fallback = historical.loc[:, FEATURE_COLUMNS].std(ddof=0)
    scale = scale.where(scale > 0, fallback).fillna(1.0)
    scale = scale.where(scale > 0, 1.0)
    return center, scale


def trade_level_analogues(
    vector: pd.DataFrame,
    current: pd.Series,
    center: pd.Series,
    scale: pd.Series,
) -> pd.DataFrame:
    """Choose exactly one nearest historical row from each completed trade."""
    historical = vector[vector["outcome_reason"].notna()].copy()
    standardized_current = (
        current.loc[list(FEATURE_COLUMNS)].astype(float) - center
    ) / scale
    standardized = (
        historical.loc[:, FEATURE_COLUMNS].astype(float) - center
    ) / scale
    historical["distance"] = np.sqrt(
        np.square(
            standardized.sub(standardized_current, axis="columns")
        ).sum(axis=1)
    )

    rows = []
    for _, group in historical.groupby("trade_id", sort=True):
        rows.append(group.sort_values(["distance"], kind="stable").iloc[0])
    analogues = pd.DataFrame(rows)
    analogues.index.name = "Date"
    return analogues.sort_values(["distance"], kind="stable")


def _weights(distances: pd.Series) -> np.ndarray:
    """Bound inverse-distance weights so one match cannot dominate."""
    raw = 1.0 / (distances.to_numpy(dtype=float) + 0.25)
    return raw / raw.sum()


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    return float(sorted_values[np.searchsorted(cumulative, quantile)])


def _approximate_date(
    start: pd.Timestamp,
    sessions: float,
) -> pd.Timestamp:
    calendar_days = round(float(sessions) * 365.25 / 252)
    return start + pd.Timedelta(days=calendar_days)


def _jeffreys_binary(
    raw_probability: float,
    effective_observations: float,
) -> tuple[float, list[float]]:
    alpha = 0.5 + raw_probability * effective_observations
    beta = 0.5 + (1 - raw_probability) * effective_observations
    probability = alpha / (alpha + beta)
    interval = stats.beta.ppf([0.05, 0.95], alpha, beta)
    return float(probability), interval.tolist()


def forecast_from_analogues(
    analogues: pd.DataFrame,
    current_date: pd.Timestamp,
    neighbors: int = DEFAULT_NEIGHBORS,
) -> dict:
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    selected = analogues.head(min(neighbors, len(analogues))).copy()
    if selected.empty:
        raise ValueError("no completed-trade analogues are available")

    weights = _weights(selected["distance"])
    divergence = (
        selected["outcome_reason"] == "bearish-divergence"
    ).to_numpy(dtype=float)
    raw_probability = float(np.dot(weights, divergence))

    # Inverse-distance weights reduce the independent information below k.
    # A Jeffreys prior prevents seven similar outcomes from becoming a false
    # 100% certainty and gives an honest small-sample interval.
    effective_neighbors = float(1.0 / np.square(weights).sum())
    probability, probability_interval = _jeffreys_binary(
        raw_probability, effective_neighbors
    )

    horizon_probabilities = {}
    sessions = selected["sessions_to_outcome"].to_numpy(dtype=float)
    for horizon in FORECAST_HORIZONS:
        occurred = divergence * (sessions <= horizon)
        raw_horizon = float(np.dot(weights, occurred))
        horizon_probability, horizon_interval = _jeffreys_binary(
            raw_horizon, effective_neighbors
        )
        horizon_probabilities[str(horizon)] = {
            "raw_weighted_rate": raw_horizon,
            "posterior_probability": horizon_probability,
            "jeffreys_90_interval": horizon_interval,
        }

    divergence_mask = divergence.astype(bool)
    timing = None
    if divergence_mask.any():
        divergence_sessions = sessions[divergence_mask]
        divergence_weights = weights[divergence_mask]
        divergence_weights = divergence_weights / divergence_weights.sum()
        q25 = _weighted_quantile(
            divergence_sessions, divergence_weights, 0.25
        )
        median = _weighted_quantile(
            divergence_sessions, divergence_weights, 0.50
        )
        q75 = _weighted_quantile(
            divergence_sessions, divergence_weights, 0.75
        )
        timing = {
            "sessions_q25_median_q75": [q25, median, q75],
            "approximate_dates_q25_median_q75": [
                _approximate_date(current_date, value)
                for value in (q25, median, q75)
            ],
        }

    reason_probabilities = {}
    for reason in ("bearish-divergence", "climax-top", "trailing-stop"):
        mask = (selected["outcome_reason"] == reason).to_numpy(dtype=float)
        reason_probabilities[reason] = float(np.dot(weights, mask))

    neighbour_rows = []
    for (date, row), weight in zip(selected.iterrows(), weights, strict=True):
        neighbour_rows.append(
            {
                "date": date,
                "trade_id": row["trade_id"],
                "distance": row["distance"],
                "weight": weight,
                "outcome_reason": row["outcome_reason"],
                "sessions_to_outcome": row["sessions_to_outcome"],
                "outcome_signal_date": row["outcome_signal_date"],
                "features": {
                    column: row[column] for column in FEATURE_COLUMNS
                },
            }
        )

    return {
        "neighbors_requested": neighbors,
        "neighbors_used": len(selected),
        "effective_neighbor_count": effective_neighbors,
        "raw_weighted_divergence_rate": raw_probability,
        "probability_divergence_before_other_exit": probability,
        "jeffreys_90_interval": probability_interval,
        "probability_by_horizon_sessions": horizon_probabilities,
        "raw_neighbor_reason_weights": reason_probabilities,
        "conditional_timing_if_divergence": timing,
        "neighbors": neighbour_rows,
    }


def landmark_validation(
    vector: pd.DataFrame,
    neighbors: int = DEFAULT_NEIGHBORS,
    landmark_session: int = 60,
) -> dict:
    """Leave one trade out at a fixed causal session after entry."""
    completed = vector[vector["outcome_reason"].notna()]
    predictions = []
    for trade_id, group in completed.groupby("trade_id", sort=True):
        ordered = group.sort_index()
        if len(ordered) <= landmark_session:
            continue
        query = ordered.iloc[landmark_session]
        training = completed[completed["trade_id"] != trade_id]
        center, scale = robust_scale(training)
        analogues = trade_level_analogues(
            training, query, center, scale
        )
        prediction = forecast_from_analogues(
            analogues,
            ordered.index[landmark_session],
            neighbors=neighbors,
        )["probability_divergence_before_other_exit"]
        observed = float(
            query["outcome_reason"] == "bearish-divergence"
        )
        predictions.append(
            {
                "trade_id": trade_id,
                "landmark_date": ordered.index[landmark_session],
                "predicted_probability": prediction,
                "observed_divergence": observed,
                "outcome_reason": query["outcome_reason"],
            }
        )
    if not predictions:
        return {
            "landmark_session": landmark_session,
            "trades_tested": 0,
            "status": "unavailable",
        }

    predicted = np.array(
        [row["predicted_probability"] for row in predictions]
    )
    observed = np.array(
        [row["observed_divergence"] for row in predictions]
    )
    return {
        "landmark_session": landmark_session,
        "trades_tested": len(predictions),
        "brier_score": float(np.mean(np.square(predicted - observed))),
        "classification_accuracy_at_50pct": float(
            np.mean((predicted >= 0.5) == observed)
        ),
        "observed_divergence_rate": float(observed.mean()),
        "mean_predicted_probability": float(predicted.mean()),
        "predictions": predictions,
    }


def current_threshold_distances(
    df: pd.DataFrame,
    current: pd.Series,
) -> dict:
    price_anchor = float(
        df["price"].shift(qbt.DIVERGENCE_WINDOW).iloc[-1]
    )
    breadth_anchor = float(
        df["breadth"].shift(qbt.DIVERGENCE_WINDOW).iloc[-1]
    )
    required_price = price_anchor * (
        1 + qbt.DIVERGENCE_PRICE_RISE / 100
    )
    required_breadth = min(
        qbt.DIVERGENCE_BREADTH_CAP,
        breadth_anchor - qbt.DIVERGENCE_BREADTH_FALL,
    )
    return {
        "price_condition_met": (
            current["ndx_return_60_pct"] >= qbt.DIVERGENCE_PRICE_RISE
        ),
        "breadth_fall_condition_met": (
            current["breadth_fall_60_points"]
            >= qbt.DIVERGENCE_BREADTH_FALL
        ),
        "breadth_cap_condition_met": (
            current["breadth"] < qbt.DIVERGENCE_BREADTH_CAP
        ),
        "ndx_level_needed_if_anchor_static": required_price,
        "ndx_rise_from_current_needed_pct": (
            required_price / float(df["price"].iloc[-1]) - 1
        )
        * 100,
        "breadth_needed_if_anchor_static": required_breadth,
        "breadth_drop_from_current_needed_points": (
            float(current["breadth"]) - required_breadth
        ),
        "warning": (
            "Both 60-session anchors roll daily; these threshold distances "
            "are a current snapshot, not fixed future targets."
        ),
    }


def write_vector(path: Path, vector: pd.DataFrame) -> None:
    output = vector.reset_index()
    output["Date"] = output["Date"].dt.strftime("%Y-%m-%d")
    output["entry_date"] = pd.to_datetime(
        output["entry_date"]
    ).dt.strftime("%Y-%m-%d")
    output["outcome_signal_date"] = pd.to_datetime(
        output["outcome_signal_date"]
    ).dt.strftime("%Y-%m-%d")
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vector-output",
        type=Path,
        default=DEFAULT_VECTOR_OUTPUT,
    )
    parser.add_argument(
        "--forecast-output",
        type=Path,
        default=DEFAULT_FORECAST_OUTPUT,
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=DEFAULT_NEIGHBORS,
    )
    args = parser.parse_args()

    df = qbt.load_data()
    df["spy_daily_change_pct"] = load_spy_daily_change().reindex(df.index)
    _, trades, open_trade = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    if open_trade is None:
        raise RuntimeError(
            "the canonical strategy is not invested, so there is no active "
            "bearish-divergence exit to forecast"
        )

    vector = build_vector_frame(df, trades, open_trade)
    current_rows = vector[vector["is_current"]]
    if len(current_rows) != 1:
        raise RuntimeError(
            f"expected one current vector row, found {len(current_rows)}"
        )
    current = current_rows.iloc[0]
    historical = vector[vector["outcome_reason"].notna()]
    center, scale = robust_scale(historical)
    analogues = trade_level_analogues(vector, current, center, scale)

    forecast = forecast_from_analogues(
        analogues,
        df.index[-1],
        neighbors=args.neighbors,
    )
    sensitivity = {
        str(neighbors): forecast_from_analogues(
            analogues,
            df.index[-1],
            neighbors=neighbors,
        )["probability_divergence_before_other_exit"]
        for neighbors in SENSITIVITY_NEIGHBORS
    }

    write_vector(args.vector_output, vector)
    result = {
        "method": {
            "model": "robust-scaled nearest historical states",
            "features": list(FEATURE_COLUMNS),
            "one_analogue_per_completed_trade": True,
            "competing_risks": ["climax-top", "trailing-stop"],
            "default_neighbors": args.neighbors,
            "sensitivity_neighbors": list(SENSITIVITY_NEIGHBORS),
            "forecast_horizons_sessions": list(FORECAST_HORIZONS),
            "uncertainty": (
                "Jeffreys beta posterior using inverse-distance effective "
                "neighbor count"
            ),
        },
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "completed_trades": len(trades),
            "historical_vector_rows": len(historical),
            "trade_level_analogues": len(analogues),
            "spy_daily_change_source": (
                "SPX.csv close-to-close return (local SPY proxy)"
            ),
            "pre_2007_breadth": "synthetic MMTH-mapped splice",
        },
        "current_vector": {
            column: current[column] for column in FEATURE_COLUMNS
        },
        "threshold_distances": current_threshold_distances(df, current),
        "forecast": forecast,
        "neighbor_sensitivity": sensitivity,
        "leave_one_trade_out_validation": landmark_validation(
            vector,
            neighbors=args.neighbors,
            landmark_session=60,
        ),
        "limitations": [
            "Only one nearest state per 17 completed trades is counted.",
            "Nine completed trades ended in bearish divergence.",
            "The timing interval is conditional on divergence occurring before "
            "a competing exit; it is not an unconditional calendar promise.",
            "All observations through the latest local close have already been "
            "seen and are not clean forward out-of-sample evidence.",
            "The current feature vector changes every session as both 60-day "
            "anchors roll.",
        ],
        "artifacts": {
            "vector_csv": args.vector_output.resolve(),
            "forecast_json": args.forecast_output.resolve(),
        },
    }
    args.forecast_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
