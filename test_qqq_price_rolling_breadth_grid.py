import itertools
import json
import unittest
from pathlib import Path

import pandas as pd

import qqq_price_rolling_breadth_grid as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_price_rolling_breadth_grid_results.json"
GRID = ROOT / "qqq_price_rolling_breadth_grid.csv"
TRADES = ROOT / "qqq_price_rolling_breadth_grid_trades.csv"
SIGNALS = ROOT / "qqq_price_rolling_breadth_grid_signals.csv"


class PriceRollingBreadthGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.grid = pd.read_csv(GRID)
        cls.trades = pd.read_csv(
            TRADES, parse_dates=["entry_date", "exit_date", "signal_date"]
        )
        cls.signals = pd.read_csv(SIGNALS, parse_dates=["Date"]).set_index("Date")

    def test_grid_is_exact_pre_registered_cartesian_product(self):
        expected = set(
            itertools.product(
                research.PRICE_GRID,
                research.DRAWDOWN_GRID,
                research.CAP_GRID,
            )
        )
        actual = set(
            self.grid[
                ["price_rise_pct", "breadth_drawdown_points", "breadth_cap"]
            ].itertuples(index=False, name=None)
        )
        self.assertEqual(actual, expected)
        self.assertEqual(len(self.grid), 210)
        self.assertEqual(len(self.grid.drop_duplicates(
            ["price_rise_pct", "breadth_drawdown_points", "breadth_cap"]
        )), 210)

    def test_disabled_harness_has_exact_baseline_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["trade_signatures_identical"])
        self.assertTrue(parity["open_trade_identical"])

    def test_robust_winner_matches_registered_sort_order(self):
        expected = self.grid.sort_values(
            ["min_half_calmar", "full_calmar", "completed_trades"],
            ascending=[False, False, True],
        ).iloc[0]
        winner = self.results["winners"]["robust_consensus"]
        self.assertEqual(
            (winner["price_rise_pct"], winner["breadth_drawdown_points"], winner["breadth_cap"]),
            (expected["price_rise_pct"], expected["breadth_drawdown_points"], expected["breadth_cap"]),
        )

    def test_full_sample_winner_is_descriptive_maximum(self):
        expected = self.grid.sort_values(
            ["full_calmar", "min_half_calmar", "completed_trades"],
            ascending=[False, False, True],
        ).iloc[0]
        winner = self.results["winners"]["full_sample"]
        self.assertEqual(winner["full_calmar"], expected["full_calmar"])
        self.assertEqual(winner["price_rise_pct"], expected["price_rise_pct"])

    def test_every_robust_raw_signal_meets_selected_conditions(self):
        winner = self.results["winners"]["robust_consensus"]
        active = self.signals[self.signals["robust_signal"].astype(bool)]
        self.assertTrue(
            (active["ndx_return_60_pct"] >= winner["price_rise_pct"]).all()
        )
        self.assertTrue(
            (active["breadth_drawdown_60_points"]
             >= winner["breadth_drawdown_points"]).all()
        )
        self.assertTrue((active["breadth"] < winner["breadth_cap"]).all())

    def test_robust_replacement_exits_fill_next_session_open(self):
        trades = self.trades[
            (self.trades["selection"] == "robust_consensus")
            & (self.trades["sell_reason"] == "grid-price-rolling-breadth")
        ]
        self.assertGreater(len(trades), 0)
        for row in trades.itertuples():
            self.assertTrue(bool(self.signals.loc[row.signal_date, "robust_signal"]))
            signal_location = self.signals.index.get_loc(row.signal_date)
            self.assertEqual(self.signals.index[signal_location + 1], row.exit_date)
            self.assertAlmostEqual(
                row.exit_price,
                self.signals.loc[row.signal_date, "next_session_open"],
                places=8,
            )

    def test_pre_registered_guardrails_force_rejection(self):
        self.assertEqual(self.results["decision"], "reject")
        guardrails = self.results["guardrails"]
        self.assertFalse(guardrails["calmar_improved"])
        self.assertFalse(guardrails["candidate_not_on_boundary"])
        self.assertFalse(guardrails["five_x_paired_return_positive"])


if __name__ == "__main__":
    unittest.main()
