import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_monthly_breadth_regime_exit_results.json"
TRADES = ROOT / "qqq_monthly_breadth_regime_exit_trades.csv"
SIGNALS = ROOT / "qqq_monthly_breadth_regime_exit_signals.csv"


class MonthlyBreadthRegimeExitArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.trades = pd.read_csv(TRADES, parse_dates=["entry_date", "exit_date", "signal_date"])
        cls.signals = pd.read_csv(SIGNALS, parse_dates=["Date"]).set_index("Date")

    def test_disabled_harness_has_exact_baseline_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["trade_signatures_identical"])
        self.assertTrue(parity["open_trade_identical"])

    def test_primary_extra_exits_only_fire_on_registered_signal_dates(self):
        primary = self.trades[self.trades["variant"] == "breadth_50"]
        extra = primary[primary["sell_reason"] == "monthly-breadth-regime"]
        self.assertGreater(len(extra), 0)
        for date in extra["signal_date"]:
            self.assertTrue(bool(self.signals.loc[date, "breadth_50"]))
            self.assertLess(self.signals.loc[date, "price"], self.signals.loc[date, "ma200"])
            self.assertLess(self.signals.loc[date, "breadth"], 50.0)

    def test_challenger_clears_trade_count_cap_but_not_score_target(self):
        score = self.results["challenger"]["score"]
        self.assertGreaterEqual(
            self.results["challenger"]["metrics"]["completed_trades"], 30
        )
        self.assertEqual(score["hard_cap"], 100)
        self.assertEqual(score["final_score"], 61)
        self.assertLess(score["final_score"], 80)

    def test_pre_registered_failure_forces_rejection(self):
        guardrails = self.results["guardrails"]
        self.assertEqual(self.results["decision"], "reject")
        self.assertFalse(guardrails["calmar_improved"])
        self.assertFalse(guardrails["cagr_within_two_points"])
        self.assertFalse(guardrails["sensitivity_not_cliff_edge"])
        self.assertTrue(guardrails["max_drawdown_not_worse"])


if __name__ == "__main__":
    unittest.main()
