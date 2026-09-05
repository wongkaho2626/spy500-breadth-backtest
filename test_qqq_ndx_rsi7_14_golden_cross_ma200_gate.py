import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import qqq_ndx_rsi7_14_golden_cross_ma200_gate as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_ndx_rsi7_14_golden_cross_ma200_gate_results.json"
TRADES = ROOT / "qqq_ndx_rsi7_14_golden_cross_ma200_gate_trades.csv"
SIGNALS = ROOT / "qqq_ndx_rsi7_14_golden_cross_ma200_gate_signals.csv"


class MA200GateUnitTests(unittest.TestCase):
    def test_qualified_signal_requires_both_golden_cross_and_trend(self):
        frame = pd.DataFrame(
            {
                "qualified_golden_ma_200": [True, False, False],
            }
        )
        self.assertEqual(
            research.qualified_signal(frame, 200).tolist(),
            [True, False, False],
        )

    def test_feature_formula_is_exact_conjunction(self):
        close = research.common.load_ndx_close_csv()
        features = research.build_features(close.index, close)
        expected = (
            features["golden_cross_7_14"]
            & (features["price"] > features["ma_200"])
        ).fillna(False)
        pd.testing.assert_series_equal(
            features["qualified_golden_ma_200"],
            expected,
            check_names=False,
        )


class MA200GateArtifactTests(unittest.TestCase):
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

    def test_saved_indicators_use_complete_nasdaq100_history(self):
        close = research.common.load_ndx_close_csv()
        expected_rsi = research.golden.crossover.calculate_wilder_rsi(
            close, 7
        ).reindex(self.signals.index)
        expected_ma = close.rolling(200, min_periods=200).mean().reindex(
            self.signals.index
        )
        np.testing.assert_allclose(
            self.signals["rsi_7"], expected_rsi,
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )
        np.testing.assert_allclose(
            self.signals["ma_200"], expected_ma,
            rtol=1e-12, atol=1e-12, equal_nan=True,
        )

    def test_gated_entries_meet_all_rules_and_fill_next_open(self):
        primary = (
            self.trades[self.trades["variant"] == "ma_200"]
            .sort_values("entry_date")
            .reset_index(drop=True)
        )
        completed = primary[primary["buy_trigger"] == "RSI-golden-cross"]
        open_trade = self.results["challenger"]["open_trade"]
        open_is_gated = bool(
            open_trade
            and open_trade["buy_trigger"] == "RSI-golden-cross"
        )
        self.assertEqual(
            len(completed) + int(open_is_gated),
            self.results["signal_counts"]["executed_gated_entries"],
        )
        entries = [
            (row.entry_date, row.entry_price)
            for row in completed.itertuples()
        ]
        if open_is_gated:
            entries.append(
                (pd.Timestamp(open_trade["entry_date"]), open_trade["entry_price"])
            )
        for entry_date, entry_price in entries:
            location = self.signals.index.get_loc(entry_date)
            signal_date = self.signals.index[location - 1]
            self.assertTrue(bool(
                self.signals.loc[signal_date, "entry_signal_ma_200"]
            ))
            self.assertTrue(bool(
                self.signals.loc[signal_date, "golden_cross_7_14"]
            ))
            self.assertGreater(
                self.signals.loc[signal_date, "price"],
                self.signals.loc[signal_date, "ma_200"],
            )
            self.assertAlmostEqual(
                entry_price,
                self.signals.loc[signal_date, "next_session_open"],
                places=8,
            )
            earlier = primary[primary["exit_date"] < entry_date]
            self.assertFalse(earlier.empty)
            self.assertGreater(signal_date, earlier.iloc[-1]["cooldown_until"])

    def test_only_entry_logic_changed(self):
        primary = self.trades[self.trades["variant"] == "ma_200"]
        allowed = {"bearish-divergence", "climax-top", "trailing-stop"}
        self.assertTrue(set(primary["sell_reason"]).issubset(allowed))
        self.assertNotIn("rsi-death-cross", set(primary["sell_reason"]))

    def test_gate_improves_some_ungated_metrics_but_not_drawdown(self):
        ungated = self.results["ungated_golden_cross_reference"]["metrics"]
        gated = self.results["challenger"]["metrics"]
        self.assertGreater(gated["cagr"], ungated["cagr"])
        self.assertGreater(gated["sharpe"], ungated["sharpe"])
        self.assertLess(gated["max_drawdown"], ungated["max_drawdown"])
        self.assertLess(gated["calmar"], ungated["calmar"])

    def test_pre_registered_result_is_reject(self):
        self.assertEqual(self.results["decision"], "reject")
        guardrails = self.results["guardrails"]
        self.assertTrue(guardrails["baseline_parity"])
        self.assertTrue(guardrails["turnover_guardrail"])
        self.assertTrue(guardrails["sensitivity_not_cliff_edge"])
        for name in (
            "primary_calmar_improved",
            "max_drawdown_not_worse",
            "expectancy_not_worse",
            "historical_halves_calmar_nonnegative",
            "real_breadth_calmar_nonnegative",
            "five_x_paired_return_positive",
        ):
            self.assertFalse(guardrails[name])

    def test_gated_challenger_still_underperforms_baseline(self):
        baseline = self.results["baseline"]["metrics"]
        challenger = self.results["challenger"]["metrics"]
        self.assertLess(challenger["cagr"], baseline["cagr"])
        self.assertLess(challenger["sharpe"], baseline["sharpe"])
        self.assertLess(challenger["calmar"], baseline["calmar"])
        self.assertLess(challenger["max_drawdown"], baseline["max_drawdown"])


if __name__ == "__main__":
    unittest.main()
