import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from curl_cffi.requests.exceptions import HTTPError, RequestException

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
            patch.object(
                fetch_investing_data,
                "_publish_breadth_files",
                side_effect=lambda: calls.append(("publish",)),
            ),
        ):
            fetch_investing_data.fetch_all_updates(verbose=False)

        self.assertEqual(
            calls,
            [
                ("fetch", fetch_investing_data.INSTRUMENTS, False),
                ("rebuild", False),
                ("publish",),
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
            patch.object(
                fetch_investing_data,
                "_publish_breadth_files",
                side_effect=lambda: calls.append(("publish",)),
            ),
        ):
            fetch_investing_data.fetch_spy_updates(verbose=True)

        self.assertEqual(
            calls,
            [
                ("fetch", fetch_investing_data.SPY_INSTRUMENTS, True),
                ("rebuild", True),
                ("publish",),
            ],
        )


def published_s5th_html(rows=None, *, identity="S5TH", timeframe="Daily"):
    if rows is None:
        rows = [
            ["Sep 11, 2026", "61.25", "60.00", "62.00", "59.75", "", "+2.08%"],
            ["Sep 10, 2026", "60.00", "59.50", "61.00", "59.00", "", "+0.84%"],
        ]
    headers = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]
    return (
        f"<html><h1>S&amp;P 500 Above 200-Day MA ({identity}) Historical Data</h1>"
        f"<div>Time Frame: <span>{timeframe}</span></div><table><thead><tr>"
        + "".join(f"<th>{header}</th>" for header in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "</tbody></table></html>"
    )


class ParseS5THTests(unittest.TestCase):
    def test_parses_published_daily_ohlc_without_synthesizing_values(self):
        actual = fetch_investing_data._parse_s5th_html(published_s5th_html())

        self.assertEqual(actual.columns.tolist(), fetch_investing_data.CSV_COLUMNS)
        self.assertEqual(
            actual.iloc[0].to_dict(),
            {"Date": "09/11/2026", "Price": "61.25", "Open": "60.00", "High": "62.00",
             "Low": "59.75", "Vol.": "", "Change %": "+2.08%"},
        )
        self.assertEqual(actual["Date"].tolist(), ["09/11/2026", "09/10/2026"])

    def test_rejects_another_instrument(self):
        with self.assertRaises(ValueError):
            fetch_investing_data._parse_s5th_html(published_s5th_html(identity="SPX"))

    def test_rejects_weekly_data(self):
        with self.assertRaises(ValueError):
            fetch_investing_data._parse_s5th_html(published_s5th_html(timeframe="Weekly"))

    def test_rejects_page_without_daily_timeframe(self):
        with self.assertRaises(ValueError):
            fetch_investing_data._parse_s5th_html(published_s5th_html(timeframe=""))

    def test_rejects_challenge_page_without_historical_table(self):
        with self.assertRaises(ValueError):
            fetch_investing_data._parse_s5th_html("<html><h1>Just a moment...</h1></html>")

    def test_rejects_missing_required_column(self):
        for column in ["Date", "Price", "Open", "High", "Low"]:
            with self.subTest(column=column), self.assertRaises(ValueError):
                fetch_investing_data._parse_s5th_html(
                    published_s5th_html().replace(f"<th>{column}</th>", "<th>Unknown</th>")
                )

    def test_rejects_invalid_dates(self):
        with self.assertRaises(ValueError):
            fetch_investing_data._parse_s5th_html(published_s5th_html().replace("Sep 11, 2026", "Feb 30, 2026"))

    def test_rejects_duplicate_dates(self):
        with self.assertRaises(ValueError):
            fetch_investing_data._parse_s5th_html(published_s5th_html().replace("Sep 10, 2026", "Sep 11, 2026"))

    def test_rejects_missing_nonfinite_and_out_of_range_ohlc(self):
        for column in range(1, 5):
            for invalid in ["", "nan", "inf", "-inf", "-0.01", "100.01"]:
                row = ["Sep 11, 2026", "61.25", "60.00", "62.00", "59.75", "", "+2.08%"]
                row[column] = invalid
                with self.subTest(column=column, invalid=invalid), self.assertRaises(ValueError):
                    fetch_investing_data._parse_s5th_html(published_s5th_html([row]))

    def test_rejects_impossible_high_low_bounds(self):
        for price, open_, high, low in [
            ("63", "60", "62", "59"), ("61", "63", "62", "59"),
            ("58", "60", "62", "59"), ("61", "58", "62", "59"),
            ("61", "60", "59", "62"),
        ]:
            with self.subTest(price=price, open=open_, high=high, low=low), self.assertRaises(ValueError):
                fetch_investing_data._parse_s5th_html(
                    published_s5th_html([["Sep 11, 2026", price, open_, high, low, "", ""]])
                )


class FetchS5THInstrumentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.csv_file = Path(self.temp_dir.name) / "S5TH.csv"
        self.instrument = dict(fetch_investing_data.INSTRUMENTS[2], csv_file=self.csv_file,
                               url="https://example.test/s5th-historical-data")
        self.yahoo = patch.object(fetch_investing_data.yf, "download", side_effect=AssertionError("S5TH must never use Yahoo estimates"))
        self.yahoo_mock = self.yahoo.start()
        self.addCleanup(self.yahoo.stop)
        self.addCleanup(self.yahoo_mock.assert_not_called)

    def write_seed(self):
        pd.DataFrame([
            ["09/10/2026", "55.00", "55.00", "55.00", "55.00", "", ""],
            ["09/09/2026", "54.00", "53.00", "55.00", "52.00", "", "+1.89%"],
        ], columns=fetch_investing_data.CSV_COLUMNS).to_csv(self.csv_file, index=False)

    def read_csv(self):
        return pd.read_csv(self.csv_file, dtype=str, keep_default_na=False)

    def fetch(self, html=None):
        response = Mock(text=html if html is not None else published_s5th_html())
        with patch.object(fetch_investing_data.requests, "get", return_value=response) as get:
            added = fetch_investing_data._fetch_s5th_instrument(self.instrument, verbose=False)
        self.assertEqual(get.call_args.args[0], self.instrument["url"])
        self.assertEqual(get.call_args.kwargs["impersonate"], "chrome")
        self.assertEqual(get.call_args.kwargs["timeout"], 30)
        self.assertNotIn("headers", get.call_args.kwargs)
        response.raise_for_status.assert_called_once_with()
        return added

    def test_config_routes_breadth_to_actual_s5th_source(self):
        self.assertEqual(fetch_investing_data.INSTRUMENTS[2]["source"], "s5th")
        self.assertIn("url", fetch_investing_data.INSTRUMENTS[2])

    def test_replaces_estimated_overlap_and_preserves_earlier_history(self):
        self.write_seed()
        older_row = self.read_csv().iloc[1].to_dict()

        self.fetch()

        actual = self.read_csv().set_index("Date")
        self.assertEqual(len(actual), 3)
        self.assertEqual(actual.loc["09/10/2026", ["Price", "Open", "High", "Low"]].tolist(),
                         ["60.00", "59.50", "61.00", "59.00"])
        for column in ["Price", "Open", "High", "Low"]:
            self.assertEqual(float(actual.loc["09/09/2026", column]), float(older_row[column]))
        for column in ["Vol.", "Change %"]:
            self.assertEqual(actual.loc["09/09/2026", column], older_row[column])
        self.assertEqual(actual.loc["09/11/2026", "Price"], "61.25")

    def test_returns_only_count_of_new_dates(self):
        self.write_seed()
        self.assertEqual(self.fetch(), 1)

    def test_can_create_daily_s5th_file_without_computed_seed(self):
        self.assertEqual(self.fetch(), 2)
        self.assertEqual(self.read_csv()["Date"].tolist(), ["09/11/2026", "09/10/2026"])

    def test_repeated_published_data_is_idempotent(self):
        self.fetch()
        original = self.csv_file.read_bytes()
        self.assertEqual(self.fetch(), 0)
        self.assertEqual(self.csv_file.read_bytes(), original)

    def test_corrects_existing_date_even_without_a_new_date(self):
        self.write_seed()
        html = published_s5th_html([["Sep 10, 2026", "60.00", "59.50", "61.00", "59.00", "", "+0.84%"]])
        self.assertEqual(self.fetch(html), 0)
        self.assertEqual(self.read_csv().set_index("Date").loc["09/10/2026", "Price"], "60.00")

    def test_http_failure_warns_and_preserves_file(self):
        self.write_seed()
        original = self.csv_file.read_bytes()
        response = Mock()
        response.raise_for_status.side_effect = HTTPError("403 Forbidden")
        output = io.StringIO()
        with patch.object(fetch_investing_data.requests, "get", return_value=response), patch("sys.stdout", output):
            self.assertEqual(fetch_investing_data._fetch_s5th_instrument(self.instrument, verbose=False), 0)
        self.assertEqual(self.csv_file.read_bytes(), original)
        self.assertIn("403", output.getvalue())
        self.assertTrue(output.getvalue().strip())

    def test_transport_failure_warns_and_preserves_file(self):
        self.write_seed()
        original = self.csv_file.read_bytes()
        output = io.StringIO()
        with (
            patch.object(fetch_investing_data.requests, "get",
                         side_effect=RequestException("timeout")),
            patch("sys.stdout", output),
        ):
            self.assertEqual(fetch_investing_data._fetch_s5th_instrument(self.instrument, verbose=False), 0)
        self.assertEqual(self.csv_file.read_bytes(), original)
        self.assertIn("timeout", output.getvalue())

    def test_invalid_source_warns_and_preserves_file(self):
        self.write_seed()
        original = self.csv_file.read_bytes()
        output = io.StringIO()
        with patch("sys.stdout", output):
            self.assertEqual(self.fetch("<html>Cloudflare challenge</html>"), 0)
        self.assertEqual(self.csv_file.read_bytes(), original)
        self.assertTrue(output.getvalue().strip())

    def test_import_replaces_overlap_with_actual_published_ohlc(self):
        self.write_seed()
        source = Path(self.temp_dir.name) / "publisher-export.csv"
        pd.DataFrame([
            ["09/10/2026", "60.00", "59.50", "61.00", "59.00", "", "+0.84%"],
            ["09/11/2026", "61.25", "60.00", "62.00", "59.75", "", "+2.08%"],
        ], columns=fetch_investing_data.CSV_COLUMNS).to_csv(source, index=False)
        with patch.object(fetch_investing_data, "INSTRUMENTS", [{}, {}, self.instrument]):
            self.assertEqual(fetch_investing_data.import_s5th_csv(source, verbose=False), 1)
        actual = self.read_csv().set_index("Date")
        self.assertEqual(actual.loc["09/10/2026", "Price"], "60.00")
        self.assertEqual(actual.loc["09/11/2026", "High"], "62.00")
        self.assertEqual(actual.loc["09/11/2026", "Vol."], "")

    def test_import_rejects_invalid_ohlc_before_modifying_saved_history(self):
        self.write_seed()
        original = self.csv_file.read_bytes()
        source = Path(self.temp_dir.name) / "invalid-export.csv"
        for invalid in ["", "inf", "-inf", "nan", "100.01"]:
            with self.subTest(invalid=invalid):
                pd.DataFrame([
                    ["09/11/2026", invalid, "60.00", "62.00", "59.75", "", ""],
                ], columns=fetch_investing_data.CSV_COLUMNS).to_csv(source, index=False)
                with patch.object(fetch_investing_data, "INSTRUMENTS", [{}, {}, self.instrument]), self.assertRaises(ValueError):
                    fetch_investing_data.import_s5th_csv(source, verbose=False)
                self.assertEqual(self.csv_file.read_bytes(), original)
