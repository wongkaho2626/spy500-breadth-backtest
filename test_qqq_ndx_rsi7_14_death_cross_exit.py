import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_ndx_rsi7_14_death_cross_exit as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_ndx_rsi7_14_death_cross_exit_results.json"
TRADES = ROOT / "qqq_ndx_rsi7_14_death_cross_exit_trades.csv"
SIGNALS = ROOT / "qqq_ndx_rsi7_14_death_cross_exit_signals.csv"


class RSICrossoverUnitTests(unittest.TestCase):
    def test_rsi_reaches_expected_extremes_for_one_way_prices(self):
        index = pd.date_range("2020-01-01", periods=30, freq="D")
        rising = pd.Series(np.arange(1.0, 31.0), index=index)
        falling = pd.Series(np.arange(30.0, 0.0, -1.0), index=index)
        self.assertEqual(research.calculate_wilder_rsi(rising, 7).iloc[-1], 100.0)
        self.assertEqual(research.calculate_wilder_rsi(falling, 14).iloc[-1], 0.0)

    def test_invalid_rsi_window_is_rejected(self):
        with self.assertRaises(ValueError):
            research.calculate_wilder_rsi(pd.Series([1.0, 2.0]), 1)

    def test_golden_and_death_crosses_are_mutually_exclusive(self):
        close = research.common.load_ndx_close_csv()
        features = research.build_crossover_features(close.index, close, 7)
        both = (
            features["death_cross_7_14"]
            & features["golden_cross_7_14"]
        )
        self.assertFalse(bool(both.any()))


class RSICrossoverArtifactTests(unittest.TestCase):
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
        close = research.common.load_ndx_close_csv()
        expected_short = research.calculate_wilder_rsi(close, 7).reindex(
            self.signals.index
        )
        expected_long = research.calculate_wilder_rsi(close, 14).reindex(
            self.signals.index
        )
        np.testing.assert_allclose(
            self.signals["rsi_7"], expected_short,
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )
        np.testing.assert_allclose(
            self.signals["rsi_14"], expected_long,
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )

    def test_death_cross_exits_fill_at_next_session_open(self):
        primary = self.trades[self.trades["variant"] == "rsi_7_14"]
        exits = primary[primary["sell_reason"] == "rsi-death-cross"]
        self.assertEqual(
            len(exits), self.results["signal_counts"]["executed_death_cross_exits"]
        )
        for row in exits.itertuples():
            self.assertTrue(bool(self.signals.loc[row.signal_date, "death_cross_7_14"]))
            signal_location = self.signals.index.get_loc(row.signal_date)
            self.assertEqual(self.signals.index[signal_location + 1], row.exit_date)
            self.assertAlmostEqual(
                row.exit_price,
                self.signals.loc[row.signal_date, "next_session_open"],
                places=8,
            )

    def test_golden_cross_is_not_used_as_an_entry_in_this_round(self):
        primary = self.trades[self.trades["variant"] == "rsi_7_14"]
        self.assertNotIn("rsi-golden-cross", set(primary["buy_trigger"]))
        self.assertGreater(self.results["signal_counts"]["raw_golden_cross_days"], 0)

    def test_pre_registered_failures_force_rejection(self):
        self.assertEqual(self.results["decision"], "reject")
        failed = [
            name for name, passed in self.results["guardrails"].items()
            if not passed
        ]
        self.assertEqual(
            failed,
            [
                "primary_calmar_improved",
                "max_drawdown_not_worse",
                "cagr_within_two_points",
                "expectancy_not_worse",
                "turnover_guardrail",
                "historical_halves_calmar_nonnegative",
                "real_breadth_calmar_nonnegative",
                "five_x_paired_return_positive",
                "sensitivity_not_cliff_edge",
            ],
        )

    def test_challenger_materially_underperforms_baseline(self):
        baseline = self.results["baseline"]["metrics"]
        challenger = self.results["challenger"]["metrics"]
        self.assertLess(challenger["cagr"], baseline["cagr"] - 0.10)
        self.assertLess(challenger["sharpe"], baseline["sharpe"])
        self.assertLess(challenger["calmar"], baseline["calmar"])
        self.assertLess(challenger["max_drawdown"], baseline["max_drawdown"])
        self.assertGreater(
            challenger["turnover_position_changes_per_year"],
            baseline["turnover_position_changes_per_year"] * 2,
        )


if __name__ == "__main__":
    unittest.main()
