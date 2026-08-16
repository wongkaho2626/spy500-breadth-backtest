import numpy as np
import pandas as pd

import qqq_bearish_divergence_forecast as forecast


def test_trade_feature_rows_keep_future_labels_out_of_features() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    features = pd.DataFrame(
        {
            "price": [100.0, 102.0, 101.0, 104.0],
            "spy_daily_change_pct": [0.1, 2.0, -1.0, 3.0],
            "ndx_return_60_pct": [1.0, 2.0, 1.5, 3.0],
            "breadth": [70.0, 68.0, 66.0, 49.0],
            "breadth_fall_60_points": [0.0, 2.0, 4.0, 21.0],
            "vix": [15.0, 16.0, 17.0, 18.0],
        },
        index=dates,
    )

    rows = forecast._trade_feature_rows(
        features,
        dates,
        trade_id=1,
        entry_date=dates[0],
        entry_price=99.0,
        end_date=dates[-1],
        outcome_reason="bearish-divergence",
    )

    assert rows["sessions_to_outcome"].tolist() == [3, 2, 1, 0]
    assert rows["bearish_divergence_before_other_exit"].all()
    assert set(forecast.FEATURE_COLUMNS).isdisjoint(
        {
            "sessions_to_outcome",
            "outcome_reason",
            "outcome_signal_date",
        }
    )


def test_trade_level_analogues_select_one_row_per_trade() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    vector = pd.DataFrame(
        {
            "trade_id": [1, 1, 2, 2],
            "outcome_reason": [
                "bearish-divergence",
                "bearish-divergence",
                "climax-top",
                "climax-top",
            ],
            "sessions_to_outcome": [5, 4, 8, 7],
            **{
                column: [0.0, 1.0, 10.0, 9.0]
                for column in forecast.FEATURE_COLUMNS
            },
        },
        index=dates,
    )
    current = vector.iloc[1].copy()
    center = pd.Series(0.0, index=forecast.FEATURE_COLUMNS)
    scale = pd.Series(1.0, index=forecast.FEATURE_COLUMNS)

    analogues = forecast.trade_level_analogues(
        vector, current, center, scale
    )

    assert len(analogues) == 2
    assert analogues["trade_id"].nunique() == 2
    assert analogues.index[0] == dates[1]


def test_forecast_treats_other_exit_reasons_as_competing_risks() -> None:
    dates = pd.date_range("2024-01-02", periods=3, freq="B")
    analogues = pd.DataFrame(
        {
            "trade_id": [1, 2, 3],
            "distance": [0.1, 0.2, 0.3],
            "outcome_reason": [
                "bearish-divergence",
                "climax-top",
                "trailing-stop",
            ],
            "sessions_to_outcome": [40.0, 10.0, 20.0],
            "outcome_signal_date": dates,
            **{
                column: [1.0, 2.0, 3.0]
                for column in forecast.FEATURE_COLUMNS
            },
        },
        index=dates,
    )

    result = forecast.forecast_from_analogues(
        analogues,
        pd.Timestamp("2026-07-29"),
        neighbors=3,
    )

    assert 0 < result["probability_divergence_before_other_exit"] < 1
    assert result["jeffreys_90_interval"][0] < result[
        "probability_divergence_before_other_exit"
    ]
    assert result["jeffreys_90_interval"][1] > result[
        "probability_divergence_before_other_exit"
    ]
    assert result["probability_by_horizon_sessions"]["20"][
        "raw_weighted_rate"
    ] == 0
    assert result["probability_by_horizon_sessions"]["20"][
        "posterior_probability"
    ] > 0
    assert result["probability_by_horizon_sessions"]["60"][
        "posterior_probability"
    ] > result["probability_by_horizon_sessions"]["20"][
        "posterior_probability"
    ]
    assert np.isclose(
        sum(result["raw_neighbor_reason_weights"].values()), 1.0
    )


def test_current_vector_is_causal_under_future_truncation() -> None:
    dates = pd.date_range("2024-01-02", periods=70, freq="B")
    price = pd.Series(np.linspace(100, 130, len(dates)), index=dates)
    breadth = pd.Series(np.linspace(70, 40, len(dates)), index=dates)
    frame = pd.DataFrame(
        {
            "price": price,
            "spy_daily_change_pct": price.pct_change().fillna(0).mul(100),
            "breadth": breadth,
            "vix": 20.0,
        },
        index=dates,
    )
    trade = {
        "entry_date": dates[60],
        "exit_date": dates[-1],
        "entry_price": float(price.iloc[60]),
        "sell_reason": "bearish-divergence",
    }

    full = forecast.build_vector_frame(frame, [trade], None)
    truncated_frame = frame.iloc[:-1]
    truncated_trade = {
        **trade,
        "exit_date": dates[-2],
    }
    truncated = forecast.build_vector_frame(
        truncated_frame, [truncated_trade], None
    )
    comparison_date = dates[-3]

    np.testing.assert_allclose(
        full.loc[comparison_date, list(forecast.FEATURE_COLUMNS)].to_numpy(
            dtype=float
        ),
        truncated.loc[
            comparison_date, list(forecast.FEATURE_COLUMNS)
        ].to_numpy(dtype=float),
    )
