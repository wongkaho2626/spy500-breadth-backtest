import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_breadth_ma200_timed_washout_boost_results.json"
EQUITY = ROOT / "qqq_breadth_ma200_timed_washout_boost_equity.csv"
TRADES = ROOT / "qqq_breadth_ma200_timed_washout_boost_trades.csv"


class TimedWashoutBoostArtifactTests(unittest.TestCase):
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

    def test_frozen_and_unlimited_controls_have_exact_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["frozen_equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["frozen_trade_signatures_identical"])
        self.assertEqual(parity["unlimited_equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["unlimited_trade_signatures_identical"])

    def test_primary_rotations_use_exact_age_and_next_open(self):
        rotations = self.records[
            (self.records["variant"] == "60_sessions")
            & (self.records["record_type"] == "tqqq_to_ndx_rotation")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(rotations), 7)
        for row in rotations.itertuples():
            self.assertEqual(locations[row.exit_date] - locations[row.entry_date], 60)
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)
            self.assertEqual(int(row.max_boost_sessions), 60)

    def test_original_component_signals_still_fill_next_session(self):
        trades = self.records[
            (self.records["variant"] == "60_sessions")
            & (self.records["record_type"] == "component_trade")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(trades), 85)
        for row in trades.itertuples():
            self.assertEqual(
                locations[row.entry_date] - locations[row.entry_signal_date], 1
            )
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)

    def test_primary_reaches_79_but_not_80(self):
        score = self.results["challenger"]["score"]
        self.assertEqual(score["hard_cap"], 100)
        self.assertEqual(score["final_score"], 79)
        self.assertEqual(score["band"], "Promising")
        self.assertEqual(self.results["decision"], "reject")

    def test_sensitivity_winner_is_not_promoted_after_viewing(self):
        sensitivity = self.results["sensitivity"]
        self.assertEqual(sensitivity["40_sessions"]["score"], 79)
        self.assertEqual(sensitivity["60_sessions"]["score"], 79)
        self.assertEqual(sensitivity["80_sessions"]["score"], 83)
        self.assertEqual(
            self.results["configuration"]["primary_max_boost_sessions"], 60
        )

    def test_primary_improves_all_full_period_risk_return_guardrails(self):
        guardrails = self.results["guardrails"]
        self.assertTrue(guardrails["cagr_improved"])
        self.assertTrue(guardrails["sharpe_improved"])
        self.assertTrue(guardrails["calmar_improved"])
        self.assertTrue(guardrails["max_drawdown_within_two_points"])
        self.assertTrue(guardrails["sensitivity_stable"])
        self.assertLess(
            abs(self.results["challenger"]["metrics"]["max_drawdown"]), 0.30
        )

    def test_statistical_and_period_guardrails_explain_rejection(self):
        paired = self.results["paired_inference"]
        guardrails = self.results["guardrails"]
        self.assertGreater(paired["annualized_mean_difference"], 0)
        self.assertLess(paired["bootstrap_95_interval_annualized"][0], 0)
        self.assertFalse(guardrails["paired_bootstrap_excludes_zero"])
        self.assertLess(self.results["period_deltas"]["early_period"]["cagr"], 0)
        self.assertFalse(guardrails["all_periods_positive"])
        self.assertEqual(self.results["drag_stress"]["3x"]["score"]["final_score"], 79)


if __name__ == "__main__":
    unittest.main()
