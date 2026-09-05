import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_rolling_only_exit_results.json"
TRADES = ROOT / "qqq_rolling_only_exit_trades.csv"
SIGNALS = ROOT / "qqq_rolling_only_exit_signals.csv"


class RollingOnlyExitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.trades = pd.read_csv(
            TRADES, parse_dates=["entry_date", "exit_date", "signal_date"]
        )
        cls.signals = pd.read_csv(SIGNALS, parse_dates=["Date"]).set_index("Date")

    def test_disabled_switches_preserve_exact_baseline(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["trade_signatures_identical"])
        self.assertTrue(parity["open_trade_identical"])

    def test_primary_has_no_auxiliary_exit_reasons(self):
        primary = self.trades[self.trades["variant"] == "drawdown_30"]
        self.assertEqual(set(primary["sell_reason"]), {"rolling-only"})
        self.assertEqual(len(primary), 5)

    def test_every_rolling_exit_meets_signal_and_fills_next_open(self):
        primary = self.trades[self.trades["variant"] == "drawdown_30"]
        for row in primary.itertuples():
            self.assertTrue(bool(self.signals.loc[row.signal_date, "drawdown_30"]))
            signal_location = self.signals.index.get_loc(row.signal_date)
            self.assertEqual(self.signals.index[signal_location + 1], row.exit_date)
            self.assertAlmostEqual(
                row.exit_price,
                self.signals.loc[row.signal_date, "next_session_open"],
                places=8,
            )

    def test_current_signal_is_inactive_but_position_is_open(self):
        current = self.results["current_signal"]
        self.assertFalse(current["active"])
        self.assertTrue(current["position_open"])

    def test_drawdown_guardrail_forces_rejection(self):
        self.assertEqual(self.results["decision"], "reject")
        self.assertLess(
            self.results["challenger"]["metrics"]["max_drawdown"], -0.49
        )
        guardrails = self.results["guardrails"]
        self.assertFalse(guardrails["max_drawdown_not_worse"])
        self.assertFalse(guardrails["calmar_beats_baseline"])
        self.assertFalse(guardrails["calmar_beats_auxiliary_on"])


if __name__ == "__main__":
    unittest.main()
