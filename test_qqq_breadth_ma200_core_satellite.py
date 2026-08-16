import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_breadth_ma200_core_satellite_results.json"
EQUITY = ROOT / "qqq_breadth_ma200_core_satellite_equity.csv"
TRADES = ROOT / "qqq_breadth_ma200_core_satellite_trades.csv"


class BreadthMa200CoreSatelliteArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.equity = pd.read_csv(EQUITY, parse_dates=["Date"]).set_index("Date")
        cls.records = pd.read_csv(
            TRADES,
            parse_dates=["entry_date", "exit_date", "signal_date"],
        )

    def test_zero_weight_has_exact_frozen_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["breadth_trade_signatures_identical"])
        self.assertTrue(parity["trend_bucket_zero"])

    def test_component_exits_fill_on_next_session(self):
        exits = self.records[
            (self.records["variant"] == "30%")
            & (self.records["record_type"] == "component_exit")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertGreater(len(exits), 0)
        for row in exits.itertuples():
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)

    def test_independence_adjustment_still_leaves_enough_events(self):
        challenger = self.results["challenger"]
        self.assertEqual(challenger["raw_component_exits"], 85)
        self.assertEqual(challenger["clustered_exit_events"], 48)
        self.assertLess(self.results["component_correlation"], 0.95)
        self.assertGreaterEqual(challenger["metrics"]["completed_trades"], 30)

    def test_promising_score_does_not_reach_target(self):
        score = self.results["challenger"]["score"]
        self.assertEqual(score["hard_cap"], 100)
        self.assertEqual(score["final_score"], 76)
        self.assertEqual(score["band"], "Promising")
        self.assertLess(score["final_score"], 80)
        self.assertEqual(self.results["decision"], "reject")

    def test_rejection_is_tied_to_pre_registered_robustness(self):
        guardrails = self.results["guardrails"]
        self.assertTrue(guardrails["sharpe_improved"])
        self.assertTrue(guardrails["calmar_improved"])
        self.assertFalse(guardrails["historical_halves_positive"])
        self.assertFalse(guardrails["five_x_paired_return_positive"])
        self.assertFalse(guardrails["sensitivity_stable"])


if __name__ == "__main__":
    unittest.main()
