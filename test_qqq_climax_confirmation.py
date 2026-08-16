import pandas as pd

import qqq_climax_confirmation as challenger


def test_confirmation_thresholds_are_nested() -> None:
    df = challenger.qbt.load_data()
    _, diagnostics_2 = challenger.climax_confirmation_frame(df, 2.0)
    _, diagnostics_3 = challenger.climax_confirmation_frame(df, 3.0)
    _, diagnostics_4 = challenger.climax_confirmation_frame(df, 4.0)

    signal_2 = diagnostics_2["confirmed_macd_cross"]
    signal_3 = diagnostics_3["confirmed_macd_cross"]
    signal_4 = diagnostics_4["confirmed_macd_cross"]
    assert (signal_3 <= signal_2).all()
    assert (signal_4 <= signal_3).all()


def test_zero_confirmation_preserves_baseline_exactly() -> None:
    df = challenger.qbt.load_data()
    parity = challenger.parity_check(df)

    assert parity["passed"]
    assert parity["equity_max_absolute_difference"] == 0.0


def test_confirmation_uses_trailing_close_high_only() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="B")
    frame = pd.DataFrame(
        {
            "price": [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                110,
                106,
                105,
            ],
            "macd_cross": [False] * 10 + [True, True],
        },
        index=index,
    )
    experiment, diagnostics = challenger.climax_confirmation_frame(
        frame, 3.0
    )

    assert diagnostics["pullback_from_10_close_high_pct"].iloc[10] < -3
    assert bool(experiment["macd_cross"].iloc[10])
    assert bool(experiment["macd_cross"].iloc[11])
