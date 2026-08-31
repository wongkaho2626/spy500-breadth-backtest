import unittest
from unittest.mock import patch

import pandas as pd

import fetch_investing_data


class FetchUpdatesTests(unittest.TestCase):
    def test_fetch_all_updates_rebuilds_daily_breadth_after_sources(self):
        calls = []

        with (
            patch.object(
                fetch_investing_data,
                "_fetch_instruments",
                side_effect=lambda instruments, verbose: calls.append(
                    ("fetch", instruments, verbose)
                ),
            ),
            patch.object(
                fetch_investing_data,
                "build_breadth_daily",
                side_effect=lambda verbose: calls.append(("rebuild", verbose)),
            ),
        ):
            fetch_investing_data.fetch_all_updates(verbose=False)

        self.assertEqual(
            calls,
            [
                ("fetch", fetch_investing_data.INSTRUMENTS, False),
                ("rebuild", False),
            ],
        )

    def test_fetch_spy_updates_rebuilds_daily_breadth_after_sources(self):
        calls = []

        with (
            patch.object(
                fetch_investing_data,
                "_fetch_instruments",
                side_effect=lambda instruments, verbose: calls.append(
                    ("fetch", instruments, verbose)
                ),
            ),
            patch.object(
                fetch_investing_data,
                "build_breadth_daily",
                side_effect=lambda verbose: calls.append(("rebuild", verbose)),
            ),
        ):
            fetch_investing_data.fetch_spy_updates(verbose=True)

        self.assertEqual(
            calls,
            [
                ("fetch", fetch_investing_data.SPY_INSTRUMENTS, True),
                ("rebuild", True),
            ],
        )


class FetchBreadthInstrumentTests(unittest.TestCase):
    def _existing_breadth_seed(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp("2024-01-02"),
                    "Price": "50.00",
                    "Open": "50.00",
                    "High": "50.00",
                    "Low": "50.00",
                    "Vol.": "",
                    "Change %": "",
                }
            ]
        )

    def _close_download_frame(self, closes_by_ticker: dict[str, list[float]]) -> pd.DataFrame:
        index = pd.to_datetime(["2024-01-02", "2024-01-03"])
        columns = pd.MultiIndex.from_tuples(
            [("Close", ticker) for ticker in closes_by_ticker]
        )
        rows = list(zip(*closes_by_ticker.values()))
        return pd.DataFrame(rows, index=index, columns=columns)

    def test_fetch_breadth_instrument_retries_tickers_missing_from_bulk_download(self):
        instrument = fetch_investing_data.INSTRUMENTS[2]
        saved_frames = []
        download_calls = []

        bulk = self._close_download_frame(
            {
                "AAA": [10.0, 11.0],
                "BBB": [20.0, 21.0],
            }
        )
        retry = self._close_download_frame({"CCC": [30.0, 31.0]})

        def fake_download(tickers, **kwargs):
            download_calls.append((tickers, kwargs["threads"]))
            if tickers == ["AAA", "BBB", "CCC"]:
                return bulk
            if tickers == "CCC":
                return retry
            raise AssertionError(f"unexpected download request: {tickers!r}")

        with (
            patch.object(fetch_investing_data, "_read_existing", return_value=self._existing_breadth_seed()),
            patch.object(fetch_investing_data, "_sp500_tickers", return_value=["AAA", "BBB", "CCC"]),
            patch.object(fetch_investing_data, "MA_WINDOW", 2),
            patch.object(fetch_investing_data, "MIN_VALID_CONSTITUENTS", 3),
            patch.object(fetch_investing_data.yf, "download", side_effect=fake_download),
            patch.object(
                fetch_investing_data,
                "_merge_and_save",
                side_effect=lambda new_df, existing, csv_file: saved_frames.append(new_df.copy()),
            ),
        ):
            rows_added = fetch_investing_data._fetch_breadth_instrument(instrument, verbose=False)

        self.assertEqual(rows_added, 1)
        self.assertEqual(
            download_calls,
            [
                (["AAA", "BBB", "CCC"], fetch_investing_data.BREADTH_DOWNLOAD_THREADS),
                ("CCC", fetch_investing_data.BREADTH_RETRY_THREADS),
            ],
        )
        self.assertEqual(len(saved_frames), 1)
        self.assertEqual(saved_frames[0].iloc[0]["Price"], "100.00")

    def test_fetch_breadth_instrument_retries_all_nan_histories_before_counting_valid_constituents(self):
        instrument = fetch_investing_data.INSTRUMENTS[2]
        saved_frames = []
        download_calls = []

        bulk = self._close_download_frame(
            {
                "AAA": [10.0, 11.0],
                "BBB": [20.0, 21.0],
                "CCC": [float("nan"), float("nan")],
            }
        )
        retry = self._close_download_frame({"CCC": [30.0, 31.0]})

        def fake_download(tickers, **kwargs):
            download_calls.append(tickers)
            if tickers == ["AAA", "BBB", "CCC"]:
                return bulk
            if tickers == "CCC":
                return retry
            raise AssertionError(f"unexpected download request: {tickers!r}")

        with (
            patch.object(fetch_investing_data, "_read_existing", return_value=self._existing_breadth_seed()),
            patch.object(fetch_investing_data, "_sp500_tickers", return_value=["AAA", "BBB", "CCC"]),
            patch.object(fetch_investing_data, "MA_WINDOW", 2),
            patch.object(fetch_investing_data, "MIN_VALID_CONSTITUENTS", 3),
            patch.object(fetch_investing_data.yf, "download", side_effect=fake_download),
            patch.object(
                fetch_investing_data,
                "_merge_and_save",
                side_effect=lambda new_df, existing, csv_file: saved_frames.append(new_df.copy()),
            ),
        ):
            rows_added = fetch_investing_data._fetch_breadth_instrument(instrument, verbose=False)

        self.assertEqual(rows_added, 1)
        self.assertEqual(download_calls, [["AAA", "BBB", "CCC"], "CCC"])
        self.assertEqual(len(saved_frames), 1)
        self.assertEqual(saved_frames[0].iloc[0]["Price"], "100.00")
