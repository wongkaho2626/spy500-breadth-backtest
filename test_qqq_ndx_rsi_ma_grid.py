import itertools
import json
import unittest
from pathlib import Path
from typing import Any

import pandas as pd

import qqq_ndx_rsi_ma_grid as research


ROOT = Path(__file__).parent
RESULTS = ROOT / "qqq_ndx_rsi_ma_grid_results.json"
GRID = ROOT / "qqq_ndx_rsi_ma_grid_grid.csv"
TRADES = ROOT / "qqq_ndx_rsi_ma_grid_trades.csv"
SIGNALS = ROOT / "qqq_ndx_rsi_ma_grid_signals.csv"


class NdxRsiMaGridUnitTests(unittest.TestCase):
    def test_grid_is_exact_pre_registered_product(self):
        expected = {
            (short_window, long_window, ma_window)
            for short_window, long_window in itertools.product(
                research.SHORT_WINDOWS, research.LONG_WINDOWS
            )
            if short_window < long_window
            for ma_window in research.MA_WINDOWS
        }
        self.assertEqual(set(research.GRID), expected)
        self.assertEqual(len(research.GRID), 95)

    def test_build_features_requires_golden_cross_and_above_ma(self):
        index = pd.date_range("2020-01-01", periods=4, freq="D")
        close = pd.Series([90.0, 101.0, 105.0, 110.0], index=index)
        short_rsi = pd.Series([40.0, 45.0, 55.0, 60.0], index=index)
        long_rsi = pd.Series([50.0, 50.0, 50.0, 58.0], index=index)
        ma_value = pd.Series([95.0, 100.0, 100.0, 120.0], index=index)
        features = research.build_features(index, close, short_rsi, long_rsi, ma_value)
        expected_cross = pd.Series([False, False, True, False], index=index)
        expected_above_ma = pd.Series([False, True, True, False], index=index)
        expected_signal = expected_cross & expected_above_ma
        pd.testing.assert_series_equal(
            features["golden_cross"], expected_cross, check_names=False
        )
        pd.testing.assert_series_equal(
            features["above_ma"], expected_above_ma, check_names=False
        )
        pd.testing.assert_series_equal(
            features["selected_signal"], expected_signal, check_names=False
        )


class NdxRsiMaGridArtifactTests(unittest.TestCase):
    results: dict[str, Any]
    grid: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(RESULTS.read_text())
        cls.grid = pd.read_csv(GRID)
        cls.trades = pd.read_csv(
            TRADES,
            parse_dates=[
                "entry_date",
                "exit_date",
                "cooldown_until",
                "signal_date",
                "current_date",
            ],
        )
        cls.signals = pd.read_csv(SIGNALS, parse_dates=["Date"]).set_index("Date")

    def primary_trade_rows(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        primary = self.trades[self.trades["selection"] == "primary_train_2002_2013"].copy()
        open_mask = primary["open_trade"].eq(True)
        completed = primary.loc[~open_mask].copy()
        open_rows = primary.loc[
            open_mask & (primary["buy_trigger"] == "RSI-golden-cross")
        ].copy()
        return primary, completed, open_rows

    def test_disabled_harness_has_exact_baseline_parity(self):
        parity = self.results["baseline_parity"]
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)
        self.assertTrue(parity["trade_signatures_identical"])
        self.assertTrue(parity["open_trade_identical"])

    def test_grid_file_contains_all_95_unique_combinations(self):
        expected = set(research.GRID)
        actual = set(
            self.grid[["short_window", "long_window", "ma_window"]]
            .itertuples(index=False, name=None)
        )
        self.assertEqual(actual, expected)
        self.assertEqual(len(self.grid), 95)
        self.assertEqual(
            len(self.grid.drop_duplicates(["short_window", "long_window", "ma_window"])),
            95,
        )

    def test_primary_winner_matches_registered_train_sort(self):
        expected = self.grid.sort_values(
            [
                "train_calmar",
                "train_sharpe",
                "train_cagr",
                "short_window",
                "long_window",
                "ma_window",
            ],
            ascending=[False, False, False, True, True, True],
            kind="stable",
        ).iloc[0]
        winner = self.results["winners"]["primary_train_2002_2013"]
        self.assertEqual(
            (winner["short_window"], winner["long_window"], winner["ma_window"]),
            (
                int(expected["short_window"]),
                int(expected["long_window"]),
                int(expected["ma_window"]),
            ),
        )

    def test_reverse_and_full_winners_match_registered_sort(self):
        reverse_expected = self.grid.sort_values(
            [
                "test_calmar",
                "test_sharpe",
                "test_cagr",
                "short_window",
                "long_window",
                "ma_window",
            ],
            ascending=[False, False, False, True, True, True],
            kind="stable",
        ).iloc[0]
        reverse = self.results["winners"]["reverse_train_2014_plus"]
        self.assertEqual(
            (reverse["short_window"], reverse["long_window"], reverse["ma_window"]),
            (
                int(reverse_expected["short_window"]),
                int(reverse_expected["long_window"]),
                int(reverse_expected["ma_window"]),
            ),
        )
        full_expected = self.grid.sort_values(
            [
                "full_calmar",
                "full_sharpe",
                "full_cagr",
                "short_window",
                "long_window",
                "ma_window",
            ],
            ascending=[False, False, False, True, True, True],
            kind="stable",
        ).iloc[0]
        full = self.results["winners"]["full_history_descriptive"]
        self.assertEqual(
            (full["short_window"], full["long_window"], full["ma_window"]),
            (
                int(full_expected["short_window"]),
                int(full_expected["long_window"]),
                int(full_expected["ma_window"]),
            ),
        )

    def test_selected_signal_rows_meet_rsi_cross_and_ma_gate(self):
        active = self.signals[self.signals["selected_signal"].astype(bool)]
        self.assertGreater(len(active), 0)
        self.assertTrue((active["short_rsi"] > active["long_rsi"]).all())
        previous_short = self.signals["short_rsi"].shift(1).reindex(active.index)
        previous_long = self.signals["long_rsi"].shift(1).reindex(active.index)
        self.assertTrue((previous_short <= previous_long).all())
        self.assertTrue((active["ndx_close"] > active["ma_value"]).all())

    def test_golden_cross_entries_fill_next_open_after_prior_exit_and_cooldown(self):
        _, completed, open_rows = self.primary_trade_rows()
        golden = completed[completed["buy_trigger"] == "RSI-golden-cross"].sort_values(
            "entry_date"
        )
        entries = [(row.entry_date, float(row.entry_price)) for row in golden.itertuples()]
        for row in open_rows.itertuples():
            entries.append((row.entry_date, float(row.entry_price)))
        self.assertEqual(
            len(entries), self.results["signal_counts"]["executed_golden_cross_entries"]
        )
        for entry_date, entry_price in entries:
            location = self.signals.index.get_loc(entry_date)
            signal_date = self.signals.index[location - 1]
            self.assertTrue(bool(self.signals.loc[signal_date, "selected_signal"]))
            self.assertAlmostEqual(
                entry_price,
                self.signals.loc[signal_date, "next_session_open"],
                places=8,
            )
            earlier = completed[completed["exit_date"] < entry_date]
            self.assertFalse(earlier.empty)
            previous_exit = earlier.sort_values("exit_date").iloc[-1]
            self.assertGreater(signal_date, previous_exit["cooldown_until"])

    def test_trials_count_and_decision_are_consistent(self):
        self.assertEqual(self.results["data"]["related_trials_including_grid"], 4918)
        expected = "track" if all(self.results["guardrails"].values()) else "reject"
        self.assertEqual(self.results["decision"], expected)

    def test_no_grid_combination_beats_baseline_calmar(self):
        baseline = self.results["baseline_segments"]
        diagnostics = self.results["grid_diagnostics"]
        comparisons = (
            ("train", "train_calmar", "train_combinations_beating_baseline_calmar"),
            ("test", "test_calmar", "test_combinations_beating_baseline_calmar"),
        )
        for segment, column, key in comparisons:
            count = int((self.grid[column] > baseline[segment]["calmar"]).sum())
            self.assertEqual(count, diagnostics[key])
            self.assertEqual(count, 0)
        full_count = int(
            (
                self.grid["full_calmar"]
                > self.results["baseline"]["metrics"]["calmar"]
            ).sum()
        )
        self.assertEqual(
            full_count,
            diagnostics["full_combinations_beating_baseline_calmar"],
        )
        self.assertEqual(full_count, 0)

    def test_local_neighbourhood_rows_recompute_the_reported_fraction(self):
        stability = self.results["local_neighbourhood_stability"]
        rows = stability["rows"]
        fraction = (
            sum(bool(row["supports_plateau"]) for row in rows) / len(rows)
            if rows
            else 0.0
        )
        self.assertAlmostEqual(
            stability["supportive_neighbour_fraction"], fraction, places=12
        )


if __name__ == "__main__":
    unittest.main()
