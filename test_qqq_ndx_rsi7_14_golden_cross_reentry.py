import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_ndx_rsi7_14_golden_cross_reentry as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_ndx_rsi7_14_golden_cross_reentry_results.json"
TRADES = ROOT / "qqq_ndx_rsi7_14_golden_cross_reentry_trades.csv"
SIGNALS = ROOT / "qqq_ndx_rsi7_14_golden_cross_reentry_signals.csv"


class GoldenCrossUnitTests(unittest.TestCase):
    def test_cross_formula_matches_rsi_relationship_changes(self):
        close = research.common.load_ndx_close_csv()
        features = research.build_features(close.index, close, 7)
        expected = (
            (features["rsi_7"] > features["rsi_14"])
            & (features["rsi_7"].shift(1) <= features["rsi_14"].shift(1))
        ).fillna(False)
        pd.testing.assert_series_equal(
            features["golden_cross_7_14"],
            expected,
            check_names=False,
        )

    def test_golden_and_death_crosses_are_mutually_exclusive(self):
        close = research.common.load_ndx_close_csv()
        features = research.build_features(close.index, close, 7)
        self.assertFalse(bool(
            (features["golden_cross_7_14"] & features["death_cross_7_14"]).any()
        ))


class GoldenCrossArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.trades = pd.read_csv(
            TRADES, parse_dates=["entry_date", "exit_date", "cooldown_until"]
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
        expected_short = research.crossover.calculate_wilder_rsi(
            close, 7
        ).reindex(self.signals.index)
        expected_long = research.crossover.calculate_wilder_rsi(
            close, 14
        ).reindex(self.signals.index)
        np.testing.assert_allclose(
            self.signals["rsi_7"], expected_short,
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )
        np.testing.assert_allclose(
            self.signals["rsi_14"], expected_long,
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )

    def test_golden_cross_entries_fill_next_open_after_prior_exit_and_cooldown(self):
        primary = (
            self.trades[self.trades["variant"] == "rsi_7_14"]
            .sort_values("entry_date")
            .reset_index(drop=True)
        )
        golden = primary[primary["buy_trigger"] == "RSI-golden-cross"]
        open_trade = self.results["challenger"]["open_trade"]
        open_is_golden = bool(
            open_trade
            and open_trade["buy_trigger"] == "RSI-golden-cross"
        )
        self.assertEqual(
            len(golden) + int(open_is_golden),
            self.results["signal_counts"]["executed_golden_cross_entries"],
        )
        entries = [
            (row.entry_date, row.entry_price)
            for row in golden.itertuples()
        ]
        if open_is_golden:
            entries.append(
                (
                    pd.Timestamp(open_trade["entry_date"]),
                    float(open_trade["entry_price"]),
                )
            )
        for entry_date, entry_price in entries:
            location = self.signals.index.get_loc(entry_date)
            signal_date = self.signals.index[location - 1]
            self.assertTrue(bool(self.signals.loc[signal_date, "golden_cross_7_14"]))
            self.assertAlmostEqual(
                entry_price,
                self.signals.loc[signal_date, "next_session_open"],
                places=8,
            )
            earlier = primary[primary["exit_date"] < entry_date]
            self.assertFalse(earlier.empty)
            previous_exit = earlier.iloc[-1]
            self.assertGreater(signal_date, previous_exit["cooldown_until"])

    def test_only_entry_logic_changed(self):
        primary = self.trades[self.trades["variant"] == "rsi_7_14"]
        allowed_reasons = {"bearish-divergence", "climax-top", "trailing-stop"}
        self.assertTrue(set(primary["sell_reason"]).issubset(allowed_reasons))
        self.assertNotIn("rsi-death-cross", set(primary["sell_reason"]))

    def test_pre_registered_failures_force_rejection(self):
        self.assertEqual(self.results["decision"], "reject")
        guardrails = self.results["guardrails"]
        self.assertTrue(guardrails["baseline_parity"])
        self.assertTrue(guardrails["turnover_guardrail"])
        self.assertTrue(guardrails["sensitivity_not_cliff_edge"])
        for name in (
            "primary_calmar_improved",
            "max_drawdown_not_worse",
            "cagr_within_two_points",
            "expectancy_not_worse",
            "historical_halves_calmar_nonnegative",
            "real_breadth_calmar_nonnegative",
            "five_x_paired_return_positive",
        ):
            self.assertFalse(guardrails[name])

    def test_challenger_underperforms_baseline(self):
        baseline = self.results["baseline"]["metrics"]
        challenger = self.results["challenger"]["metrics"]
        self.assertLess(challenger["cagr"], baseline["cagr"])
        self.assertLess(challenger["sharpe"], baseline["sharpe"])
        self.assertLess(challenger["calmar"], baseline["calmar"])
        self.assertLess(challenger["max_drawdown"], baseline["max_drawdown"])
        self.assertGreater(challenger["exposure"], baseline["exposure"])


if __name__ == "__main__":
    unittest.main()
