import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_breadth_ma200_trajectory_extension_results.json"
EQUITY = ROOT / "qqq_breadth_ma200_trajectory_extension_equity.csv"
TRADES = ROOT / "qqq_breadth_ma200_trajectory_extension_trades.csv"
SIGNALS = ROOT / "qqq_breadth_ma200_trajectory_extension_signals.csv"


class TrajectoryExtensionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.equity = pd.read_csv(EQUITY, parse_dates=["Date"]).set_index("Date")
        cls.trades = pd.read_csv(
            TRADES,
            parse_dates=[
                "entry_date", "exit_date", "entry_signal_date", "signal_date",
                "rotation_signal_date", "rotation_date", "extension_decision_date",
            ],
        )
        cls.signals = pd.read_csv(
            SIGNALS,
            parse_dates=[
                "entry_date", "decision_date", "decision_fill_date",
                "scheduled_rotation_date",
            ],
        )

    def test_all_three_parity_controls_are_exact(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["frozen_equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["frozen_trade_signatures_identical"])
        self.assertEqual(parity["force_short_equity_max_absolute_difference"], 0.0)
        self.assertEqual(parity["force_long_equity_max_absolute_difference"], 0.0)

    def test_extension_decisions_use_session_60_close_only(self):
        decisions = self.signals[self.signals["variant"] == "20_sessions"]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(decisions), 8)
        for row in decisions.itertuples():
            self.assertEqual(
                locations[row.decision_date] - locations[row.entry_date], 59
            )
            self.assertEqual(
                locations[row.decision_fill_date] - locations[row.decision_date], 1
            )
            expected_gate = bool(
                row.decision_ndx_above_ma200
                and row.decision_breadth > row.decision_past_breadth
            )
            self.assertEqual(bool(row.extended_to_80), expected_gate)

    def test_adaptive_rotations_fill_after_60_or_80_sessions(self):
        rotations = self.trades[
            (self.trades["variant"] == "20_sessions")
            & (self.trades["record_type"] == "tqqq_to_ndx_rotation")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(rotations), 7)
        for row in rotations.itertuples():
            age = locations[row.exit_date] - locations[row.entry_date]
            self.assertEqual(age, 80 if bool(row.extended_to_80) else 60)
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)

    def test_original_component_fills_remain_next_session(self):
        trades = self.trades[
            (self.trades["variant"] == "20_sessions")
            & (self.trades["record_type"] == "component_trade")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(trades), 85)
        for row in trades.itertuples():
            self.assertEqual(
                locations[row.entry_date] - locations[row.entry_signal_date], 1
            )
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)

    def test_primary_score_is_79_and_rejected(self):
        score = self.results["challenger"]["score"]
        self.assertEqual(score["final_score"], 79)
        self.assertEqual(score["hard_cap"], 100)
        self.assertEqual(score["band"], "Promising")
        self.assertEqual(self.results["decision"], "reject")

    def test_recent_periods_improve_but_early_cagr_does_not(self):
        periods = self.results["period_deltas"]
        for period in ("late_period", "real_breadth_period", "actual_tqqq_period"):
            for metric in ("cagr", "sharpe", "calmar"):
                self.assertGreater(periods[period][metric], 0)
        self.assertLess(periods["early_period"]["cagr"], 0)
        self.assertFalse(self.results["guardrails"]["all_periods_positive"])

    def test_inference_and_drag_explain_missing_score(self):
        paired = self.results["paired_inference"]
        self.assertGreater(paired["annualized_mean_difference"], 0)
        self.assertGreater(paired["hac_two_sided_p"], 0.05)
        self.assertLess(paired["bootstrap_95_interval_annualized"][0], 0)
        self.assertEqual(self.results["drag_stress"]["3x"]["score"]["final_score"], 79)

    def test_lookback_family_is_stable_but_not_tradeable(self):
        sensitivity = self.results["sensitivity"]
        self.assertEqual(set(sensitivity), {"10_sessions", "20_sessions", "40_sessions"})
        self.assertTrue(self.results["guardrails"]["sensitivity_stable"])
        self.assertTrue(all(row["score"] == 79 for row in sensitivity.values()))


if __name__ == "__main__":
    unittest.main()
