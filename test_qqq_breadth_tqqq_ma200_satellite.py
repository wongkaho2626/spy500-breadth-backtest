import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_breadth_tqqq_ma200_satellite_results.json"
EQUITY = ROOT / "qqq_breadth_tqqq_ma200_satellite_equity.csv"
TRADES = ROOT / "qqq_breadth_tqqq_ma200_satellite_trades.csv"


class BreadthTqqqMa200SatelliteArtifactTests(unittest.TestCase):
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
            (self.records["variant"] == "15%")
            & (self.records["record_type"] == "component_exit")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(exits), 85)
        for row in exits.itertuples():
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)

    def test_independence_adjustment_still_leaves_enough_events(self):
        challenger = self.results["challenger"]
        self.assertEqual(challenger["raw_component_exits"], 85)
        self.assertEqual(challenger["clustered_exit_events"], 48)
        self.assertEqual(challenger["breadth_completed_trades"], 17)
        self.assertEqual(challenger["tqqq_ma200_completed_trades"], 68)
        self.assertLess(self.results["component_correlation"], 0.95)
        self.assertGreaterEqual(challenger["metrics"]["completed_trades"], 30)

    def test_score_and_decision_match_pre_registered_result(self):
        score = self.results["challenger"]["score"]
        self.assertEqual(score["hard_cap"], 100)
        self.assertEqual(score["final_score"], 73)
        self.assertEqual(score["band"], "Promising")
        self.assertLess(score["final_score"], 80)
        self.assertEqual(self.results["decision"], "reject")

    def test_rejection_is_tied_to_pre_registered_guardrails(self):
        guardrails = self.results["guardrails"]
        self.assertTrue(guardrails["cagr_improved"])
        self.assertTrue(guardrails["calmar_improved"])
        self.assertTrue(guardrails["five_x_paired_return_positive"])
        self.assertFalse(guardrails["sharpe_improved"])
        self.assertFalse(guardrails["paired_bootstrap_excludes_zero"])
        self.assertFalse(guardrails["historical_halves_positive"])
        self.assertFalse(guardrails["real_breadth_positive"])
        self.assertFalse(guardrails["sensitivity_stable"])

    def test_punitive_proxy_drag_does_not_reach_target(self):
        drag = self.results["drag_stress"]["3x"]
        self.assertGreater(drag["paired"]["annualized_mean_difference"], 0)
        self.assertEqual(drag["score"]["final_score"], 73)
        self.assertFalse(drag["sensitivity_stable"])
        self.assertFalse(self.results["guardrails"]["three_x_drag_score_at_least_80"])


if __name__ == "__main__":
    unittest.main()
