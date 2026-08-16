import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_breadth_only_trajectory_extension_results.json"
EQUITY = ROOT / "qqq_breadth_only_trajectory_extension_equity.csv"
TRADES = ROOT / "qqq_breadth_only_trajectory_extension_trades.csv"
SIGNALS = ROOT / "qqq_breadth_only_trajectory_extension_signals.csv"


class BreadthOnlyTrajectoryExtensionArtifactTests(unittest.TestCase):
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

    def test_all_control_paths_have_exact_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["frozen_equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["frozen_trade_signatures_identical"])
        self.assertEqual(parity["combined_default_equity_max_absolute_difference"], 0.0)
        self.assertEqual(parity["force_short_equity_max_absolute_difference"], 0.0)
        self.assertEqual(parity["force_long_equity_max_absolute_difference"], 0.0)

    def test_session_60_gate_uses_breadth_only(self):
        decisions = self.signals[self.signals["variant"] == "20_sessions"]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(decisions), 8)
        self.assertEqual(int(decisions["extended_to_80"].sum()), 5)
        for row in decisions.itertuples():
            self.assertEqual(
                locations[row.decision_date] - locations[row.entry_date], 59
            )
            self.assertEqual(
                locations[row.decision_fill_date] - locations[row.decision_date], 1
            )
            self.assertFalse(bool(row.decision_requires_ma200))
            self.assertEqual(
                bool(row.extended_to_80),
                bool(row.decision_breadth > row.decision_past_breadth),
            )
        lagging_ma_extensions = decisions[
            decisions["extended_to_80"].astype(bool)
            & ~decisions["decision_ndx_above_ma200"].astype(bool)
        ]
        self.assertGreaterEqual(len(lagging_ma_extensions), 2)

    def test_rotations_fill_after_60_or_80_sessions(self):
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

    def test_original_component_orders_remain_next_session(self):
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

    def test_primary_score_reaches_tradeable_band_without_cap(self):
        score = self.results["challenger"]["score"]
        self.assertEqual(score["A_statistical_validity"], 26)
        self.assertEqual(score["B_risk_adjusted_performance"], 14)
        self.assertEqual(score["C_robustness_oos"], 25)
        self.assertEqual(score["D_trade_quality_consistency"], 18)
        self.assertEqual(score["final_score"], 83)
        self.assertEqual(score["hard_cap"], 100)
        self.assertEqual(score["band"], "Tradeable")
        self.assertEqual(self.results["decision"], "track")

    def test_incremental_inference_passes_both_tests(self):
        paired = self.results["paired_inference"]
        self.assertGreater(paired["annualized_mean_difference"], 0)
        self.assertLess(paired["hac_two_sided_p"], 0.05)
        self.assertGreater(paired["bootstrap_95_interval_annualized"][0], 0)

    def test_every_registered_period_and_calendar_subset_is_positive(self):
        periods = self.results["period_deltas"]
        for period in (
            "early_period", "late_period", "real_breadth_period",
            "actual_tqqq_period",
        ):
            for metric in ("cagr", "sharpe", "calmar"):
                self.assertGreater(periods[period][metric], 0)
        calendar = self.results["calendar_parity_split"]
        self.assertGreater(calendar["odd_years"]["annualized_mean_difference"], 0)
        self.assertGreater(calendar["even_years"]["annualized_mean_difference"], 0)

    def test_sensitivity_family_is_stable_and_above_target(self):
        sensitivity = self.results["sensitivity"]
        self.assertEqual(set(sensitivity), {"10_sessions", "20_sessions", "40_sessions"})
        self.assertTrue(all(row["score"] == 83 for row in sensitivity.values()))
        self.assertTrue(self.results["guardrails"]["sensitivity_stable"])

    def test_cost_and_proxy_drag_stress_pass(self):
        self.assertGreater(
            self.results["cost_stress"]["5x"]["paired_annualized_mean"], 0
        )
        drag = self.results["drag_stress"]["3x"]
        self.assertEqual(drag["score"]["final_score"], 83)
        self.assertLess(abs(drag["metrics"]["max_drawdown"]), 0.30)
        self.assertGreater(drag["paired"]["annualized_mean_difference"], 0)

    def test_all_pre_registered_guardrails_pass(self):
        failures = [
            name for name, passed in self.results["guardrails"].items()
            if not passed
        ]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
