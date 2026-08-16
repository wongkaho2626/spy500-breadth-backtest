import numpy as np
import pandas as pd

import qqq_vector_buy_signal as challenger


def test_forward_buy_labels_enforce_return_and_adverse_path() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="B")
    prices = pd.Series(
        [100.0, 95.0, 90.0, 111.0, 100.0, 101.0, 102.0, 103.0],
        index=dates,
    )
    labels, diagnostics = challenger.forward_buy_labels(
        prices,
        horizon=3,
        target_return=0.10,
        max_adverse_return=-0.15,
    )

    assert labels.iloc[0] == 1.0
    assert labels.iloc[1] == 0.0
    assert diagnostics.iloc[-3:].isna().all().all()


def test_vector_buy_wrapper_restores_canonical_constants() -> None:
    df = challenger.qbt.load_data()
    old_threshold = challenger.qbt.BUY_B200_THRESH
    old_commission = challenger.qbt.COMMISSION
    old_slippage = challenger.qbt.SLIPPAGE
    signal = pd.Series(False, index=df.index)

    challenger.run_vector_buy(df, signal)

    assert challenger.qbt.BUY_B200_THRESH == old_threshold
    assert challenger.qbt.COMMISSION == old_commission
    assert challenger.qbt.SLIPPAGE == old_slippage


def test_baseline_branch_has_exact_parity() -> None:
    df = challenger.qbt.load_data()
    parity = challenger.parity_check(df)

    assert parity["passed"]
    assert parity["equity_max_absolute_difference"] == 0.0


def test_online_buy_probability_is_causal() -> None:
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
        (np.arange(len(dates)) % 7 == 0).astype(float),
        index=dates,
    )
    original = challenger.analytics.online_crash_probability(
        vector,
        labels,
        horizon=5,
        neighbors=3,
        feature_columns=challenger.FEATURE_COLUMNS,
    )
    mutated = vector.copy()
    mutated.iloc[71:] = 1000
    changed = challenger.analytics.online_crash_probability(
        mutated,
        labels,
        horizon=5,
        neighbors=3,
        feature_columns=challenger.FEATURE_COLUMNS,
    )

    pd.testing.assert_series_equal(
        original["crash_probability"].iloc[:71],
        changed["crash_probability"].iloc[:71],
    )
