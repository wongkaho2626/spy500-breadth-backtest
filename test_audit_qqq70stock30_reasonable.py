import math

import audit_qqq70stock30_reasonable as audit


def test_future_value_and_implied_return_are_inverses() -> None:
    expected_return = 0.12
    terminal = audit.future_value(expected_return, 20)

    assert math.isclose(
        audit.implied_annual_return(terminal, 20), expected_return, abs_tol=1e-12
    )


def test_point_in_time_holdings_delay_and_correct_snapshots() -> None:
    snapshots = {
        2015: "AAPL",
        2016: "MSFT",  # corrected to AAPL by the verified override
        2020: "AAPL",
        2021: "MSFT",  # corrected to AAPL by the verified override
        2023: "AMZN",  # corrected to AAPL by the verified override
        2025: "NVDA",
    }
    spy = {2001: "GE"}

    result = audit.build_point_in_time_holdings(
        [2001, 2016, 2017, 2021, 2022, 2024, 2026], snapshots, spy
    )

    assert result == {
        2001: "GE",
        2016: "AAPL",
        2017: "AAPL",
        2021: "AAPL",
        2022: "AAPL",
        2024: "AAPL",
        2026: "NVDA",
    }
