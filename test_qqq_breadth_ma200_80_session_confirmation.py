import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_breadth_ma200_80_session_confirmation_results.json"
EQUITY = ROOT / "qqq_breadth_ma200_80_session_confirmation_equity.csv"
TRADES = ROOT / "qqq_breadth_ma200_80_session_confirmation_trades.csv"


class EightySessionConfirmationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.equity = pd.read_csv(EQUITY, parse_dates=["Date"]).set_index("Date")
        cls.records = pd.read_csv(
            TRADES,
            parse_dates=[
                "entry_date", "exit_date", "entry_signal_date", "signal_date",
                "rotation_signal_date", "rotation_date",
            ],
        )

    def test_frozen_and_prior_engine_controls_have_exact_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["frozen_equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["frozen_trade_signatures_identical"])
        self.assertEqual(parity["prior_engine_equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["prior_engine_trade_signatures_identical"])

    def test_primary_rotations_are_80_sessions_and_next_open(self):
        rotations = self.records[
            (self.records["variant"] == "80_sessions")
            & (self.records["record_type"] == "tqqq_to_ndx_rotation")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(rotations), 7)
        for row in rotations.itertuples():
            self.assertEqual(locations[row.exit_date] - locations[row.entry_date], 80)
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)
            self.assertEqual(int(row.max_boost_sessions), 80)

    def test_original_entries_and_exits_remain_next_session(self):
        trades = self.records[
            (self.records["variant"] == "80_sessions")
            & (self.records["record_type"] == "component_trade")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(trades), 85)
        for row in trades.itertuples():
            self.assertEqual(
                locations[row.entry_date] - locations[row.entry_signal_date], 1
            )
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)

    def test_selected_primary_score_is_83_without_cap(self):
        score = self.results["challenger"]["score"]
        self.assertEqual(score["final_score"], 83)
        self.assertEqual(score["hard_cap"], 100)
        self.assertEqual(score["band"], "Tradeable")
        self.assertIn("selected from prior sensitivity", self.results["selection_status"])

    def test_paired_confirmation_is_positive_but_borderline(self):
        paired = self.results["paired_inference"]
        self.assertGreater(paired["annualized_mean_difference"], 0)
        self.assertLess(paired["hac_two_sided_p"], 0.05)
        self.assertGreater(paired["bootstrap_95_interval_annualized"][0], 0)
        self.assertLess(paired["bootstrap_95_interval_annualized"][0], 0.001)

    def test_sensitivity_does_not_replace_fixed_primary(self):
        sensitivity = self.results["sensitivity"]
        self.assertEqual(sensitivity["60_sessions"]["score"], 79)
        self.assertEqual(sensitivity["80_sessions"]["score"], 83)
        self.assertEqual(sensitivity["100_sessions"]["score"], 80)
        self.assertTrue(self.results["guardrails"]["sensitivity_stable"])
        self.assertEqual(
            self.results["configuration"]["primary_max_boost_sessions"], 80
        )

    def test_actual_period_and_calendar_parity_are_explicit(self):
        periods = self.results["period_deltas"]
        calendar = self.results["calendar_parity_split"]
        self.assertIn("actual_tqqq_period", periods)
        self.assertGreater(calendar["odd_years"]["annualized_mean_difference"], 0)
        self.assertGreater(calendar["even_years"]["annualized_mean_difference"], 0)
        self.assertTrue(self.results["guardrails"]["odd_even_years_positive"])

    def test_regime_calmar_reversal_prevents_confirmation(self):
        periods = self.results["period_deltas"]
        guardrails = self.results["guardrails"]
        self.assertLess(periods["late_period"]["calmar"], 0)
        self.assertLess(periods["actual_tqqq_period"]["calmar"], 0)
        self.assertFalse(guardrails["all_periods_positive"])
        self.assertEqual(self.results["decision"], "reject")

    def test_three_x_drag_preserves_minimum_score(self):
        drag = self.results["drag_stress"]["3x"]
        self.assertEqual(drag["score"]["final_score"], 80)
        self.assertLess(abs(drag["metrics"]["max_drawdown"]), 0.30)
        self.assertGreater(drag["paired"]["annualized_mean_difference"], 0)


if __name__ == "__main__":
    unittest.main()
