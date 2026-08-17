from pathlib import Path

import numpy as np
import pandas as pd

import qqq70stock30_dca_rolling as rolling
import qqq_portfolio_backtest as qpb


def test_spy_top1_fills_missing_nasdaq_years_without_overriding_nasdaq() -> None:
    holdings = qpb.load_top_holdings()

    assert holdings[1996] == "GE"
    assert holdings[1998] == "GE"
    assert holdings[1999] == "MSFT"
    assert holdings[2000] == "MSFT"
    assert holdings[2001] == "CSCO"


def test_spy_fallback_prices_are_adjusted_like_nasdaq_stock_prices() -> None:
    raw = pd.read_csv(
        Path(__file__).parent / "SPY" / "stock_prices" / "prices" / "GE.csv"
    )
    close = qpb._load_stock_series("GE", col="Close")
    open_ = qpb._load_stock_series("GE", col="Open")

    assert close is not None
    assert open_ is not None
    assert np.isclose(close.iloc[0], raw["Adj Close"].iloc[0])
    expected_open = raw["Open"].iloc[0] * raw["Adj Close"].iloc[0] / raw["Close"].iloc[0]
    assert np.isclose(open_.iloc[0], expected_open)


def test_early_rolling_window_invests_the_spy_fallback_bucket() -> None:
    dates = np.array(["1996-01-02", "1996-01-03"], dtype="datetime64[ns]")
    false = np.array([False, False])
    arrays = {
        "dates": dates,
        "years": np.array([1996, 1996]),
        "price": np.array([100.0, 100.0]),
        "open": np.array([100.0, 100.0]),
        "breadth": np.array([50.0, 50.0]),
        "vote_gate": false,
        "price_rose": false,
        "breadth_fell": false,
        "macd_cross": false,
        "ext10": false,
        "ma200_recross": false,
        "stock_close": {"GE": np.array([10.0, 11.0])},
        "stock_open": {"GE": np.array([10.0, 10.5])},
        "top_holdings": {1996: "GE"},
        "cooldown": np.timedelta64(qpb.COOLDOWN_DAYS, "D"),
    }

    final_value = rolling.run_window(arrays, 0, 2, n_contributions=0)

    assert final_value > 1_020_000
