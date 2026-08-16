from pathlib import Path

import pandas as pd

from download_spy_historical_prices import (
    build_holdings,
    discover_reports,
    extract_report_date,
    yfinance_symbol,
)


def test_extract_report_dates_from_text_and_html() -> None:
    oldest = next(Path("SPY").glob("1996-03-04_*.txt"))
    newest = next(Path("SPY").glob("2026-05-29_*.htm"))

    assert extract_report_date(oldest) == pd.Timestamp("1995-12-31")
    assert extract_report_date(newest) == pd.Timestamp("2026-03-31")


def test_discover_all_downloaded_reports() -> None:
    reports = discover_reports(Path("SPY"))

    assert len(reports) == 48
    assert reports["report_date"].min() == pd.Timestamp("1995-12-31")
    assert reports["report_date"].max() == pd.Timestamp("2026-03-31")


def test_point_in_time_snapshot_selection_and_symbol_conversion() -> None:
    reports = pd.DataFrame(
        {
            "filing_date": pd.to_datetime(["1996-03-04", "2026-05-29"]),
            "report_date": pd.to_datetime(["1995-12-31", "2026-03-31"]),
            "source_filing": ["old.txt", "new.htm"],
        }
    )
    membership = pd.DataFrame(
        {
            "date": ["1996-01-02", "2026-03-23"],
            "tickers": ["AAPL,BF.B", "AAPL,BRK.B"],
        }
    )

    holdings, tickers = build_holdings(reports, membership)

    assert set(holdings["ticker"]) == {"AAPL", "BF.B", "BRK.B"}
    assert holdings.iloc[0]["snapshot_resolution"] == "earliest_available"
    assert set(tickers["yfinance_ticker"]) == {"AAPL", "BF-B", "BRK-B"}
    assert tickers["first_membership_date"].min() == pd.Timestamp("1996-01-02")
    assert yfinance_symbol("BRK.B") == "BRK-B"
