import unittest

import numpy as np
import pandas as pd

import spx_20pct_peak_vector_analysis as analysis


class SpxPeakVectorAnalysisTests(unittest.TestCase):
    def test_detects_non_overlapping_twenty_percent_episode(self) -> None:
        dates = pd.date_range("2020-01-01", periods=8, freq="B")
        close = pd.Series(
            [100.0, 105.0, 103.0, 83.0, 75.0, 90.0, 105.0, 106.0],
            index=dates,
        )
        episodes = analysis.detect_drawdown_episodes(close)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["peak_date"], dates[1])
        self.assertEqual(episodes[0]["breach_date"], dates[3])
        self.assertEqual(episodes[0]["trough_date"], dates[4])

    def test_peak_zone_contains_only_closes_within_two_percent(self) -> None:
        dates = pd.date_range("2020-01-01", periods=8, freq="B")
        close = pd.Series(
            [97.0, 99.0, 100.0, 99.0, 97.0, 79.9, 90.0, 101.0],
            index=dates,
        )
        vector = pd.DataFrame(
            {
                feature: np.arange(len(dates), dtype=float)
                for feature in analysis.FEATURE_COLUMNS
            },
            index=dates,
        )
        episodes = analysis.detect_drawdown_episodes(close)
        rows = analysis.peak_zone_rows(
            close, vector, episodes, pre_peak_lookback=4
        )

        self.assertEqual(
            list(rows.index), [dates[1], dates[2], dates[3]]
        )
        self.assertTrue(
            (rows["distance_below_peak_pct"] >= -2.0).all()
        )
        self.assertEqual(
            set(rows["phase"]), {"before_peak", "peak", "after_peak"}
        )

    def test_future_mutation_does_not_change_past_vector(self) -> None:
        vector, spx = analysis.build_full_market_vector()
        cutoff = 1000
        original = vector.iloc[: cutoff + 1].copy()
        mutated_spx = spx.copy()
        mutated_spx.iloc[cutoff + 1 :] *= 10

        spx_high = mutated_spx.rolling(252, min_periods=60).max()
        changed_drawdown = (
            mutated_spx / spx_high - 1
        ).mul(100)
        changed_slope = analysis.rolling_linear_slope(
            changed_drawdown, analysis.SLOPE_WINDOW
        )

        pd.testing.assert_series_equal(
            original["spx_drawdown_252_pct"],
            changed_drawdown.iloc[: cutoff + 1],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            original["spx_drawdown_252_slope_20"],
            changed_slope.iloc[: cutoff + 1],
            check_names=False,
        )


if __name__ == "__main__":
    unittest.main()
