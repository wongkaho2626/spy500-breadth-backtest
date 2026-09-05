import json
import unittest
from pathlib import Path

import pandas as pd

import qqq_price_confirmed_rolling_breadth_exit as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_price_confirmed_rolling_breadth_exit_results.json"
TRADES = ROOT / "qqq_price_confirmed_rolling_breadth_exit_trades.csv"
SIGNALS = ROOT / "qqq_price_confirmed_rolling_breadth_exit_signals.csv"
RAW_EVENTS = ROOT / "qqq_price_confirmed_rolling_breadth_exit_raw_events.csv"


class PriceConfirmedRollingBreadthUnitTests(unittest.TestCase):
    def test_signal_requires_all_three_registered_conditions(self):
        features = pd.DataFrame(
            {
                "ndx_return_60_pct": [3.0, 5.0, 2.9, 5.0, 5.0],
                "breadth_drawdown_60_points": [20.0, 19.9, 30.0, 25.0, 25.0],
                "breadth_below_60": [True, True, True, False, True],
            }
        )
        signal = research.price_confirmed_signal(features, 20.0)
        self.assertEqual(signal.tolist(), [True, False, False, False, True])


class PriceConfirmedRollingBreadthArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.trades = pd.read_csv(
            TRADES, parse_dates=["entry_date", "exit_date", "signal_date"]
        )
        cls.signals = pd.read_csv(SIGNALS, parse_dates=["Date"]).set_index("Date")
        cls.raw_events = pd.read_csv(
            RAW_EVENTS, parse_dates=["Date"]
        ).set_index("Date")

    def test_disabled_harness_has_exact_baseline_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["trade_signatures_identical"])
        self.assertTrue(parity["open_trade_identical"])

    def test_every_raw_event_meets_price_and_breadth_conditions(self):
        self.assertEqual(
            len(self.raw_events),
            self.results["signal_counts"]["raw_active_days"],
        )
        self.assertTrue((self.raw_events["ndx_return_60_pct"] >= 3.0).all())
        self.assertTrue(
            (self.raw_events["breadth_drawdown_60_points"] >= 20.0).all()
        )
        self.assertTrue((self.raw_events["breadth"] < 60.0).all())

    def test_executed_sells_fill_next_session_open(self):
        primary = self.trades[self.trades["variant"] == "drawdown_20"]
        exits = primary[
            primary["sell_reason"] == "price-confirmed-rolling-breadth"
        ]
        self.assertEqual(
            len(exits),
            self.results["signal_counts"]["executed_replacement_sells"],
        )
        for row in exits.itertuples():
            self.assertTrue(
                bool(self.signals.loc[row.signal_date, "drawdown_20"])
            )
            signal_location = self.signals.index.get_loc(row.signal_date)
            self.assertEqual(self.signals.index[signal_location + 1], row.exit_date)
            self.assertAlmostEqual(
                row.exit_price,
                self.signals.loc[row.signal_date, "next_session_open"],
                places=8,
            )

    def test_current_snapshot_fails_all_three_votes(self):
        current = self.results["current_signal"]
        self.assertFalse(current["active"])
        self.assertFalse(current["price_rise_3pct"])
        self.assertLess(current["breadth_drawdown_60_points"], 20.0)
        self.assertGreaterEqual(current["breadth"], 60.0)

    def test_pre_registered_failures_force_rejection(self):
        self.assertEqual(self.results["decision"], "reject")
        guardrails = self.results["guardrails"]
        self.assertFalse(guardrails["calmar_improved"])
        self.assertFalse(guardrails["max_drawdown_not_worse"])
        self.assertFalse(guardrails["cagr_within_two_points"])
        interval = self.results["paired_inference"][
            "bootstrap_95_interval_annualized"
        ]
        self.assertLess(interval[1], 0)


if __name__ == "__main__":
    unittest.main()
