import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import qqq_rolling_breadth_drawdown_exit as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_rolling_breadth_drawdown_exit_results.json"
TRADES = ROOT / "qqq_rolling_breadth_drawdown_exit_trades.csv"
SIGNALS = ROOT / "qqq_rolling_breadth_drawdown_exit_signals.csv"
RAW_EVENTS = ROOT / "qqq_rolling_breadth_drawdown_exit_raw_events.csv"


class RollingBreadthDrawdownUnitTests(unittest.TestCase):
    def test_features_are_causal(self):
        index = pd.bdate_range("2025-01-02", periods=80)
        original = pd.DataFrame(
            {
                "breadth": np.linspace(80.0, 40.0, len(index)),
                "price_rose": False,
                "breadth_fell": False,
            },
            index=index,
        )
        changed = original.copy()
        changed.loc[index[75]:, "breadth"] = [90, 20, 85, 15, 80]
        before = research.rolling_breadth_features(original).loc[: index[74]]
        after = research.rolling_breadth_features(changed).loc[: index[74]]
        assert_frame_equal(before, after)

    def test_signal_uses_rolling_drawdown_and_current_cap(self):
        features = pd.DataFrame(
            {
                "breadth_drawdown_60_points": [20.0, 25.0, 19.9, 30.0],
                "breadth_below_60": [True, False, True, True],
            }
        )
        signal = research.rolling_breadth_signal(features, 20.0)
        self.assertEqual(signal.tolist(), [True, False, False, True])


class RollingBreadthDrawdownArtifactTests(unittest.TestCase):
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

    def test_every_raw_event_meets_registered_conditions(self):
        self.assertEqual(
            len(self.raw_events),
            self.results["signal_counts"]["raw_active_days"],
        )
        self.assertTrue(
            (self.raw_events["breadth_drawdown_60_points"] >= 20.0).all()
        )
        self.assertTrue((self.raw_events["breadth"] < 60.0).all())

    def test_executed_sells_fill_next_session_open(self):
        primary = self.trades[self.trades["variant"] == "drawdown_20"]
        exits = primary[
            primary["sell_reason"] == "rolling-breadth-drawdown"
        ]
        self.assertEqual(
            len(exits),
            self.results["signal_counts"]["executed_rolling_breadth_sells"],
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

    def test_current_snapshot_is_not_active(self):
        current = self.results["current_signal"]
        self.assertFalse(current["active"])
        self.assertLess(current["breadth_drawdown_60_points"], 20.0)

    def test_pre_registered_failures_force_rejection(self):
        self.assertEqual(self.results["decision"], "reject")
        guardrails = self.results["guardrails"]
        self.assertFalse(guardrails["calmar_improved"])
        self.assertFalse(guardrails["cagr_within_two_points"])
        self.assertFalse(guardrails["expectancy_not_worse"])
        interval = self.results["paired_inference"][
            "bootstrap_95_interval_annualized"
        ]
        self.assertLess(interval[1], 0)


if __name__ == "__main__":
    unittest.main()
