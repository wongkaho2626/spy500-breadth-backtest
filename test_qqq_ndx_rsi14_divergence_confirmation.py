import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_ndx_rsi14_divergence_confirmation as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_ndx_rsi14_divergence_confirmation_results.json"
TRADES = ROOT / "qqq_ndx_rsi14_divergence_confirmation_trades.csv"
SIGNALS = ROOT / "qqq_ndx_rsi14_divergence_confirmation_signals.csv"


class RSI14UnitTests(unittest.TestCase):
    def test_rsi_reaches_expected_extremes_for_one_way_prices(self):
        index = pd.date_range("2020-01-01", periods=30, freq="D")
        rising = pd.Series(np.arange(1.0, 31.0), index=index)
        falling = pd.Series(np.arange(30.0, 0.0, -1.0), index=index)
        self.assertEqual(research.calculate_rsi14(rising).iloc[-1], 100.0)
        self.assertEqual(research.calculate_rsi14(falling).iloc[-1], 0.0)

    def test_confirmation_requires_divergence_and_rsi_at_or_below_threshold(self):
        features = pd.DataFrame(
            {
                "canonical_divergence": [True, True, True, False],
                "ndx_rsi14": [49.9, 50.0, 50.1, 40.0],
            }
        )
        actual = research.confirmation_signal(features, 50.0)
        self.assertEqual(actual.tolist(), [True, True, False, False])


class RSI14ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.trades = pd.read_csv(
            TRADES, parse_dates=["entry_date", "exit_date", "signal_date"]
        )
        cls.signals = pd.read_csv(SIGNALS, parse_dates=["Date"]).set_index("Date")

    def test_disabled_harness_has_exact_baseline_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["trade_signatures_identical"])
        self.assertTrue(parity["open_trade_identical"])

    def test_saved_rsi_uses_complete_nasdaq100_csv_history(self):
        expected = research.calculate_rsi14(
            research.load_ndx_close_csv()
        ).reindex(self.signals.index)
        np.testing.assert_allclose(
            self.signals["ndx_rsi14"].to_numpy(),
            expected.to_numpy(),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )

    def test_confirmed_exits_use_signal_close_and_fill_next_open(self):
        primary = self.trades[self.trades["variant"] == "rsi_le_50"]
        exits = primary[
            primary["sell_reason"] == "rsi14-confirmed-divergence"
        ]
        self.assertEqual(
            len(exits),
            self.results["signal_counts"]["executed_confirmed_divergence_exits"],
        )
        for row in exits.itertuples():
            self.assertTrue(bool(self.signals.loc[row.signal_date, "canonical_divergence"]))
            self.assertLessEqual(self.signals.loc[row.signal_date, "ndx_rsi14"], 50.0)
            signal_location = self.signals.index.get_loc(row.signal_date)
            self.assertEqual(self.signals.index[signal_location + 1], row.exit_date)
            self.assertAlmostEqual(
                row.exit_price,
                self.signals.loc[row.signal_date, "next_session_open"],
                places=8,
            )

    def test_pre_registered_failures_force_rejection(self):
        self.assertEqual(self.results["decision"], "reject")
        guardrails = self.results["guardrails"]
        self.assertFalse(guardrails["primary_calmar_improved"])
        self.assertFalse(guardrails["historical_halves_calmar_nonnegative"])
        self.assertFalse(guardrails["real_breadth_calmar_nonnegative"])
        self.assertFalse(guardrails["five_x_paired_return_positive"])
        self.assertFalse(guardrails["sensitivity_not_cliff_edge"])

    def test_challenger_does_not_improve_primary_or_risk_adjusted_metrics(self):
        baseline = self.results["baseline"]["metrics"]
        challenger = self.results["challenger"]["metrics"]
        self.assertLess(challenger["calmar"], baseline["calmar"])
        self.assertLess(challenger["cagr"], baseline["cagr"])
        self.assertLess(challenger["sharpe"], baseline["sharpe"])
        self.assertGreater(challenger["ulcer_index"], baseline["ulcer_index"])


if __name__ == "__main__":
    unittest.main()
