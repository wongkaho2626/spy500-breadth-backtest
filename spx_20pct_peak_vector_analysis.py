"""Describe common vectors near peaks preceding S&P 500 drawdowns of 20%."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_vector_crash_exit as crash
from qqq_vector_trajectory_recross_filter import rolling_linear_slope


DATA_DIR = Path(__file__).parent
SPX_FILE = DATA_DIR / "SPX.csv"
NDX_FILE = DATA_DIR / "NASDAQ100.csv"
VIX_FILE = DATA_DIR / "VIX.csv"
BREADTH_FILE = DATA_DIR / "breadth_daily.csv"
DEFAULT_RESULT = DATA_DIR / "spx_20pct_peak_vector_results.json"
DEFAULT_EPISODES = DATA_DIR / "spx_20pct_drawdown_episodes.csv"
DEFAULT_VECTORS = DATA_DIR / "spx_20pct_peak_zone_vectors.csv"
DEFAULT_COMMON = DATA_DIR / "spx_20pct_peak_vector_commonalities.csv"
DRAW_THRESHOLD = -0.20
PEAK_BAND = 0.02
PRE_PEAK_LOOKBACK = 126
SLOPE_WINDOW = 20
FEATURE_COLUMNS = (
    "spx_daily_change_pct",
    "ndx_return_60_pct",
    "breadth",
    "breadth_fall_60_points",
    "spx_drawdown_252_pct",
    "vix",
    "vix_slope_20",
    "breadth_slope_20",
    "spx_drawdown_252_slope_20",
)
REDUCED_FEATURE_COLUMNS = tuple(
    column
    for column in FEATURE_COLUMNS
    if "breadth" not in column
)


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


def load_price_series(path: Path, name: str) -> pd.Series:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [
        column.strip().strip('"').lstrip("﻿") for column in frame.columns
    ]
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    frame = frame.sort_values("Date").set_index("Date")
    values = (
        frame["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    return values.rename(name)


def load_breadth(path: Path = BREADTH_FILE) -> pd.Series:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    frame = frame.sort_values("Date").set_index("Date")
    return frame["breadth"].astype(float).rename("breadth")


def build_full_market_vector() -> tuple[pd.DataFrame, pd.Series]:
    spx = load_price_series(SPX_FILE, "spx")
    ndx = load_price_series(NDX_FILE, "ndx").reindex(spx.index)
    vix = load_price_series(VIX_FILE, "vix").reindex(spx.index).ffill()
    breadth = load_breadth().reindex(spx.index)
    spx_high = spx.rolling(252, min_periods=60).max()
    vector = pd.DataFrame(
        {
            "spx_daily_change_pct": spx.pct_change(
                fill_method=None
            ).mul(100),
            "ndx_return_60_pct": (ndx / ndx.shift(60) - 1).mul(100),
            "breadth": breadth,
            "breadth_fall_60_points": breadth.shift(60) - breadth,
            "spx_drawdown_252_pct": (spx / spx_high - 1).mul(100),
            "vix": vix,
        },
        index=spx.index,
    )
    vector["vix_slope_20"] = rolling_linear_slope(
        vector["vix"], SLOPE_WINDOW
    )
    vector["breadth_slope_20"] = rolling_linear_slope(
        vector["breadth"], SLOPE_WINDOW
    )
    vector["spx_drawdown_252_slope_20"] = rolling_linear_slope(
        vector["spx_drawdown_252_pct"], SLOPE_WINDOW
    )
    vector.index.name = "Date"
    return vector, spx


def detect_drawdown_episodes(
    close: pd.Series,
) -> list[dict[str, Any]]:
    return crash.spx_crash_episodes(close)


def peak_zone_rows(
    close: pd.Series,
    vector: pd.DataFrame,
    episodes: list[dict[str, Any]],
    peak_band: float = PEAK_BAND,
    pre_peak_lookback: int = PRE_PEAK_LOOKBACK,
) -> pd.DataFrame:
    """Collect all closes within peak_band of each peak around the peak date."""
    rows = []
    locations = {date: i for i, date in enumerate(close.index)}
    for episode_number, episode in enumerate(episodes, start=1):
        peak_date = episode["peak_date"]
        breach_date = episode["breach_date"]
        peak_location = locations[peak_date]
        start_location = max(0, peak_location - pre_peak_lookback)
        start_date = close.index[start_location]
        peak_price = float(close.loc[peak_date])
        window = close.loc[start_date:breach_date]
        zone_dates = window.index[
            window >= peak_price * (1 - peak_band)
        ]
        for date in zone_dates:
            date_location = locations[date]
            if date < peak_date:
                phase = "before_peak"
            elif date > peak_date:
                phase = "after_peak"
            else:
                phase = "peak"
            row = {
                "episode": episode_number,
                "peak_date": peak_date,
                "breach_date": breach_date,
                "trough_date": episode["trough_date"],
                "recovery_date": episode["recovery_date"],
                "peak_to_trough_pct": (
                    float(episode["peak_to_trough"]) * 100
                ),
                "Date": date,
                "phase": phase,
                "sessions_from_peak": date_location - peak_location,
                "spx_close": float(close.loc[date]),
                "distance_below_peak_pct": (
                    close.loc[date] / peak_price - 1
                )
                * 100,
            }
            row.update(vector.loc[date, list(FEATURE_COLUMNS)].to_dict())
            rows.append(row)
    return pd.DataFrame(rows).set_index("Date").sort_index()


def episode_feature_medians(
    peak_rows: pd.DataFrame,
    vector: pd.DataFrame,
) -> pd.DataFrame:
    medians = peak_rows.groupby("episode")[list(FEATURE_COLUMNS)].median()
    metadata = (
        peak_rows.groupby("episode")
        .agg(
            peak_date=("peak_date", "first"),
            breach_date=("breach_date", "first"),
            peak_to_trough_pct=("peak_to_trough_pct", "first"),
            peak_zone_days=("phase", "size"),
        )
    )
    exact_peak_rows = []
    for episode_number, row in metadata.iterrows():
        exact = vector.loc[
            row["peak_date"], list(FEATURE_COLUMNS)
        ].to_dict()
        exact_peak_rows.append(
            {
                "episode": episode_number,
                **{
                    f"exact_peak_{feature}": value
                    for feature, value in exact.items()
                },
            }
        )
    exact_peaks = pd.DataFrame(exact_peak_rows).set_index("episode")
    return metadata.join(medians).join(exact_peaks)


def commonality_table(
    vector: pd.DataFrame,
    peak_rows: pd.DataFrame,
    episode_medians: pd.DataFrame,
    value_prefix: str = "",
    sample_scope: str = "peak_zone_median",
) -> pd.DataFrame:
    """Compare equal-weighted episode medians with non-peak market days."""
    control = vector.loc[~vector.index.isin(peak_rows.index)]
    rows = []
    for feature in FEATURE_COLUMNS:
        episode_values = episode_medians[
            f"{value_prefix}{feature}"
        ].dropna()
        ordinary = control[feature].dropna()
        ordinary_median = float(ordinary.median())
        q25, q75 = ordinary.quantile([0.25, 0.75])
        iqr = float(q75 - q25)
        top_median = float(episode_values.median())
        difference = top_median - ordinary_median
        direction = "above" if difference >= 0 else "below"
        if direction == "above":
            consistent = int((episode_values > ordinary_median).sum())
        else:
            consistent = int((episode_values < ordinary_median).sum())
        percentiles = [
            float((ordinary <= value).mean())
            for value in episode_values
        ]
        robust_effect = difference / iqr if iqr > 0 else np.nan
        consistency = consistent / len(episode_values)
        rows.append(
            {
                "sample_scope": sample_scope,
                "feature": feature,
                "episodes_available": len(episode_values),
                "top_episode_median": top_median,
                "ordinary_day_median": ordinary_median,
                "difference": difference,
                "direction_vs_ordinary": direction,
                "consistent_episodes": consistent,
                "consistency_fraction": consistency,
                "median_percentile_vs_ordinary": float(
                    np.median(percentiles)
                ),
                "robust_effect_iqr": robust_effect,
                "commonality_score": (
                    abs(robust_effect) * consistency
                    if np.isfinite(robust_effect)
                    else np.nan
                ),
                "is_common": bool(
                    len(episode_values) >= 3
                    and consistency >= 0.75
                    and abs(robust_effect) >= 0.25
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["is_common", "commonality_score"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )


def episode_records(
    episodes: list[dict[str, Any]],
    peak_rows: pd.DataFrame,
    episode_medians: pd.DataFrame,
) -> list[dict[str, Any]]:
    records = []
    for episode_number, episode in enumerate(episodes, start=1):
        rows = peak_rows[peak_rows["episode"] == episode_number]
        median = episode_medians.loc[episode_number]
        record = {
            "episode": episode_number,
            **episode,
            "peak_zone_start": rows.index.min(),
            "peak_zone_end": rows.index.max(),
            "peak_zone_days": len(rows),
            "before_peak_days": int((rows["phase"] == "before_peak").sum()),
            "after_peak_days": int((rows["phase"] == "after_peak").sum()),
            "complete_nine_feature_vector": bool(
                median[list(FEATURE_COLUMNS)].notna().all()
            ),
            "episode_median_vector": {
                feature: median[feature] for feature in FEATURE_COLUMNS
            },
            "exact_peak_vector": {
                feature: median[f"exact_peak_{feature}"]
                for feature in FEATURE_COLUMNS
            },
        }
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--episodes-output", type=Path, default=DEFAULT_EPISODES
    )
    parser.add_argument(
        "--vectors-output", type=Path, default=DEFAULT_VECTORS
    )
    parser.add_argument(
        "--common-output", type=Path, default=DEFAULT_COMMON
    )
    args = parser.parse_args()

    vector, spx = build_full_market_vector()
    episodes = detect_drawdown_episodes(spx)
    peak_rows = peak_zone_rows(spx, vector, episodes)
    medians = episode_feature_medians(peak_rows, vector)
    zone_common = commonality_table(vector, peak_rows, medians)
    exact_common = commonality_table(
        vector,
        peak_rows,
        medians,
        value_prefix="exact_peak_",
        sample_scope="exact_peak",
    )
    common = pd.concat(
        [exact_common, zone_common],
        ignore_index=True,
    ).sort_values(
        ["sample_scope", "is_common", "commonality_score"],
        ascending=[True, False, False],
    )
    records = episode_records(episodes, peak_rows, medians)

    episode_output = medians.reset_index()
    for column in ("peak_date", "breach_date"):
        episode_output[column] = pd.to_datetime(
            episode_output[column]
        ).dt.strftime("%Y-%m-%d")
    episode_output.to_csv(args.episodes_output, index=False)

    vector_output = peak_rows.reset_index()
    for column in (
        "Date",
        "peak_date",
        "breach_date",
        "trough_date",
        "recovery_date",
    ):
        vector_output[column] = pd.to_datetime(
            vector_output[column]
        ).dt.strftime("%Y-%m-%d")
    vector_output.to_csv(args.vectors_output, index=False)
    common.to_csv(args.common_output, index=False)

    common_features = common[common["is_common"]]
    result = {
        "definition": {
            "episode": (
                "non-overlapping running-peak drawdown first breaching -20%"
            ),
            "peak_zone": (
                "all closes at least 98% of the episode peak, from 126 "
                "sessions before the peak through the first -20% breach"
            ),
            "feature_aggregation": (
                "median across peak-zone days, then equal weight per episode"
            ),
            "drawdown_threshold": DRAW_THRESHOLD,
            "peak_band": PEAK_BAND,
            "pre_peak_lookback_sessions": PRE_PEAK_LOOKBACK,
        },
        "data": {
            "start": spx.index[0],
            "end": spx.index[-1],
            "bars": len(spx),
            "breadth_start": vector["breadth"].first_valid_index(),
            "episodes": len(episodes),
            "episodes_with_complete_nine_feature_vector": sum(
                record["complete_nine_feature_vector"]
                for record in records
            ),
        },
        "features": {
            "full_nine": list(FEATURE_COLUMNS),
            "all_episode_reduced_six": list(REDUCED_FEATURE_COLUMNS),
        },
        "episodes": records,
        "common_features": common_features.to_dict(orient="records"),
        "all_feature_comparisons": common.to_dict(orient="records"),
        "limitations": [
            (
                "Only four independent 20% drawdown episodes exist in the "
                "1990-2026 close series."
            ),
            (
                "Breadth starts in 2002, so the 2000 episode has only the "
                "reduced six-feature vector and the full nine-feature "
                "commonalities rely on three episodes."
            ),
            (
                "This is descriptive event analysis, not a causal forecast "
                "or a backtested trade rule."
            ),
            (
                "Peak-zone dates are correlated observations; episode "
                "medians are used so long plateaus do not receive extra "
                "weight."
            ),
        ],
        "artifacts": {
            "results_json": args.result_output.resolve(),
            "episodes_csv": args.episodes_output.resolve(),
            "peak_zone_vectors_csv": args.vectors_output.resolve(),
            "commonalities_csv": args.common_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
