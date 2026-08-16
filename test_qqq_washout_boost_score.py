import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_washout_boost_score_results.json"
TRADES = ROOT / "qqq_washout_boost_score_trades.csv"


class WashoutBoostScoreArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.trades = pd.read_csv(TRADES)

    def test_baseline_parity_is_exact(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["trade_signatures_identical"])

    def test_allocation_change_does_not_change_signals(self):
        baseline = self.trades[self.trades["strategy"] == "baseline"].reset_index(drop=True)
        boost = self.trades[
            self.trades["strategy"] == "washout_boost_10pct"
        ].reset_index(drop=True)
        columns = ["entry_date", "exit_date", "sell_reason"]
        pd.testing.assert_frame_equal(baseline[columns], boost[columns])

    def test_conditional_sleeve_only_uses_tqqq_for_washouts(self):
        boost = self.trades[self.trades["strategy"] == "washout_boost_10pct"]
        washouts = boost[boost["entry_type"] == "washout"]
        recrosses = boost[boost["entry_type"] == "ma200-recross"]
        self.assertGreater(len(washouts), 0)
        self.assertGreater(len(recrosses), 0)
        self.assertTrue((washouts["conditional_asset"] == "TQQQ").all())
        self.assertTrue((recrosses["conditional_asset"] == "NDX").all())

    def test_score_cannot_pass_80_with_thin_trade_sample(self):
        score = self.results["challenger"]["score"]
        self.assertLess(score["raw_score"], 80)
        self.assertEqual(score["hard_cap"], 40)
        self.assertEqual(score["final_score"], 40)
        self.assertEqual(self.results["decision"], "reject")


if __name__ == "__main__":
    unittest.main()
