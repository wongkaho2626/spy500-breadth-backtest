import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_breadth_ma200_washout_boost_results.json"
EQUITY = ROOT / "qqq_breadth_ma200_washout_boost_equity.csv"
TRADES = ROOT / "qqq_breadth_ma200_washout_boost_trades.csv"


class BreadthMa200WashoutBoostArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.equity = pd.read_csv(EQUITY, parse_dates=["Date"]).set_index("Date")
        cls.records = pd.read_csv(
            TRADES,
            parse_dates=[
                "entry_date", "exit_date", "entry_signal_date", "signal_date"
            ],
        )

    def test_controls_have_exact_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["frozen_equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["frozen_trade_signatures_identical"])
        self.assertEqual(
            parity["zero_boost_70_30_equity_max_absolute_difference"], 0.0
        )
        self.assertTrue(parity["zero_boost_trade_signatures_identical"])

    def test_every_component_trade_fills_one_session_after_signal(self):
        trades = self.records[
            (self.records["variant"] == "10%")
            & (self.records["record_type"] == "component_trade")
        ]
        locations = {date: i for i, date in enumerate(self.equity.index)}
        self.assertEqual(len(trades), 85)
        for row in trades.itertuples():
            self.assertEqual(
                locations[row.entry_date] - locations[row.entry_signal_date], 1
            )
            self.assertEqual(locations[row.exit_date] - locations[row.signal_date], 1)

    def test_boost_is_limited_to_frozen_washout_entries(self):
        trades = self.records[
            (self.records["variant"] == "10%")
            & (self.records["record_type"] == "component_trade")
            & (self.records["component"] == "breadth")
        ]
        self.assertEqual(len(trades), 17)
        recross = trades[trades["buy_trigger"] == "MA200-recross"]
        washout = trades[trades["buy_trigger"] != "MA200-recross"]
        self.assertGreater(len(recross), 0)
        self.assertGreater(len(washout), 0)
        self.assertFalse(recross["tqqq_boosted"].astype(bool).any())
        self.assertTrue(washout["tqqq_boosted"].astype(bool).all())

    def test_independence_clustering_removes_overlapping_events(self):
        challenger = self.results["challenger"]
        self.assertEqual(challenger["raw_component_exits"], 85)
        self.assertEqual(challenger["clustered_exit_events"], 48)
        self.assertGreaterEqual(challenger["metrics"]["completed_trades"], 30)
        self.assertLess(self.results["component_correlation"], 0.95)

    def test_score_is_new_high_but_below_target(self):
        score = self.results["challenger"]["score"]
        self.assertEqual(score["hard_cap"], 100)
        self.assertEqual(score["final_score"], 77)
        self.assertEqual(score["band"], "Promising")
        self.assertLess(score["final_score"], 80)
        self.assertEqual(self.results["decision"], "reject")

    def test_incremental_return_is_statistically_positive(self):
        paired = self.results["paired_inference"]
        self.assertGreater(paired["annualized_mean_difference"], 0)
        self.assertGreater(paired["bootstrap_95_interval_annualized"][0], 0)
        self.assertLess(paired["hac_two_sided_p"], 0.01)
        self.assertGreater(
            self.results["cost_stress"]["5x"]["paired_annualized_mean"], 0
        )

    def test_pre_registered_risk_guardrails_fail(self):
        guardrails = self.results["guardrails"]
        self.assertTrue(guardrails["cagr_improved"])
        self.assertTrue(guardrails["calmar_improved"])
        self.assertFalse(guardrails["sharpe_improved"])
        self.assertFalse(guardrails["max_drawdown_within_two_points"])
        self.assertFalse(guardrails["all_periods_positive"])
        self.assertFalse(guardrails["sensitivity_stable"])
        self.assertFalse(guardrails["three_x_drag_score_at_least_80"])


if __name__ == "__main__":
    unittest.main()
