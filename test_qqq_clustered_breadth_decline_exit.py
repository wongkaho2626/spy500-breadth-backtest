import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import qqq_clustered_breadth_decline_exit as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_clustered_breadth_decline_exit_results.json"
TRADES = ROOT / "qqq_clustered_breadth_decline_exit_trades.csv"
SIGNALS = ROOT / "qqq_clustered_breadth_decline_exit_signals.csv"


class ClusteredBreadthDeclineUnitTests(unittest.TestCase):
    def test_features_do_not_change_when_only_future_breadth_changes(self):
        index = pd.bdate_range("2025-01-02", periods=20)
        original = pd.DataFrame(
            {"breadth": np.linspace(80.0, 60.0, len(index))}, index=index
        )
        changed = original.copy()
        changed.loc[index[15]:, "breadth"] = [95, 40, 90, 35, 85]

        before = research.breadth_features(original).loc[: index[14]]
        after = research.breadth_features(changed).loc[: index[14]]
        assert_frame_equal(before, after)

    def test_registered_signal_requires_count_and_negative_slope(self):
        index = pd.bdate_range("2025-01-02", periods=12)
        breadth = pd.Series(
            [100, 97, 99, 96, 98, 95, 97, 94, 96, 93, 92, 91],
            index=index,
        )
        features = research.breadth_features(pd.DataFrame({"breadth": breadth}))
        signal = research.clustered_signal(features, 4)
        active = signal[signal]
        self.assertGreater(len(active), 0)
        self.assertTrue((features.loc[active.index, "large_decline_count_10"] >= 4).all())
        self.assertTrue((features.loc[active.index, "breadth_slope_10"] < 0).all())


class ClusteredBreadthDeclineArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.trades = pd.read_csv(
            TRADES, parse_dates=["entry_date", "exit_date", "signal_date"]
        )
        cls.signals = pd.read_csv(SIGNALS, parse_dates=["Date"]).set_index("Date")

    def test_disabled_harness_has_exact_baseline_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["trade_signatures_identical"])
        self.assertTrue(parity["open_trade_identical"])

    def test_primary_extra_exits_follow_registered_close_signal(self):
        primary = self.trades[self.trades["variant"] == "count_4"]
        extra = primary[
            primary["sell_reason"] == "clustered-breadth-decline"
        ]
        self.assertGreater(len(extra), 0)
        for row in extra.itertuples():
            self.assertTrue(bool(self.signals.loc[row.signal_date, "count_4"]))
            self.assertGreaterEqual(
                self.signals.loc[row.signal_date, "large_decline_count_10"], 4
            )
            self.assertLess(
                self.signals.loc[row.signal_date, "breadth_slope_10"], 0
            )
            signal_location = self.signals.index.get_loc(row.signal_date)
            self.assertEqual(self.signals.index[signal_location + 1], row.exit_date)
            self.assertAlmostEqual(
                row.exit_price,
                self.signals.loc[row.exit_date, "open"],
                places=8,
            )

    def test_current_observation_matches_user_rule(self):
        current = self.results["current_signal"]
        self.assertTrue(current["active"])
        self.assertEqual(current["large_decline_count_10"], 4)
        self.assertLess(current["breadth_slope_10"], 0)
        self.assertLess(current["breadth_daily_return"], -0.02)

    def test_pre_registered_failures_force_rejection(self):
        self.assertEqual(self.results["decision"], "reject")
        guardrails = self.results["guardrails"]
        self.assertFalse(guardrails["calmar_improved"])
        self.assertFalse(guardrails["cagr_within_one_point"])
        self.assertFalse(guardrails["expectancy_not_worse"])
        self.assertFalse(guardrails["completed_trades_within_25pct"])
        interval = self.results["paired_inference"][
            "bootstrap_95_interval_annualized"
        ]
        self.assertLess(interval[1], 0)


if __name__ == "__main__":
    unittest.main()
