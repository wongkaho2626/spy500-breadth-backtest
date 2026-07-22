import unittest
from unittest.mock import patch

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
