import numpy as np
import pandas as pd

import qqq_vector_crash_exit as challenger


def test_forward_crash_labels_use_only_next_horizon() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    prices = pd.Series(
        [100.0, 99.0, 98.0, 79.0, 90.0, 91.0, 92.0, 93.0],
        index=dates,
    )
    labels, future_return = challenger.forward_crash_labels(
        prices,
        horizon=3,
        crash_drop=-0.20,
    )

    assert labels.iloc[0] == 1.0
    assert labels.iloc[1] == 1.0
    assert labels.iloc[3] == 0.0
    assert future_return.iloc[-3:].isna().all()


def test_online_probability_is_causal_under_future_mutation() -> None:
    dates = pd.date_range("2010-01-01", periods=90, freq="B")
    base = np.linspace(0.0, 1.0, len(dates))
    vector = pd.DataFrame(
        {
            column: base + offset
            for offset, column in enumerate(challenger.FEATURE_COLUMNS)
        },
        index=dates,
    )
    labels = pd.Series(
        (np.arange(len(dates)) % 9 == 0).astype(float),
        index=dates,
    )
    original = challenger.online_crash_probability(
        vector,
        labels,
        horizon=5,
        neighbors=3,
    )

    mutated_vector = vector.copy()
    mutated_labels = labels.copy()
    mutated_vector.iloc[71:] = 10_000
    mutated_labels.iloc[71:] = 1 - mutated_labels.iloc[71:]
    mutated = challenger.online_crash_probability(
        mutated_vector,
        mutated_labels,
        horizon=5,
        neighbors=3,
    )

    pd.testing.assert_series_equal(
        original["crash_probability"].iloc[:71],
        mutated["crash_probability"].iloc[:71],
    )


def test_baseline_replacement_harness_has_exact_parity() -> None:
    frame = challenger.qbt.load_data()
    result = challenger.parity_check(frame)

    assert result["passed"]
    assert result["equity_max_absolute_difference"] == 0.0


def test_crash_episode_detection() -> None:
    dates = pd.date_range("2020-01-01", periods=7, freq="B")
    prices = pd.Series(
        [100.0, 105.0, 100.0, 83.0, 70.0, 90.0, 106.0],
        index=dates,
    )
    episodes = challenger.spx_crash_episodes(prices)

    assert len(episodes) == 1
    assert episodes[0]["peak_date"] == dates[1]
    assert episodes[0]["breach_date"] == dates[3]
    assert episodes[0]["trough_date"] == dates[4]
    assert episodes[0]["recovery_date"] == dates[6]
