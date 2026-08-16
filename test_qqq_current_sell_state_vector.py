import unittest

import numpy as np
import pandas as pd

import qqq_current_sell_state_vector as analysis


class CurrentSellStateVectorTests(unittest.TestCase):
    def test_vote_fraction_reaches_one_when_all_conditions_are_met(self) -> None:
        index = pd.date_range("2020-01-01", periods=61, freq="B")
        price = np.full(61, 100.0)
        breadth = np.full(61, 80.0)
        price[-1] = 104.0
        breadth[-1] = 55.0
        frame = pd.DataFrame(
            {"price": price, "breadth": breadth}, index=index
        )

        result = analysis.canonical_sell_vote_frame(frame).iloc[-1]

        self.assertEqual(result[analysis.SELL_FEATURE], 1.0)
        self.assertTrue(result["canonical_bearish_divergence"])

    def test_vote_fraction_counts_partial_sell_state(self) -> None:
        index = pd.date_range("2020-01-01", periods=61, freq="B")
        price = np.full(61, 100.0)
        breadth = np.full(61, 80.0)
        price[-1] = 102.0
        breadth[-1] = 55.0
        frame = pd.DataFrame(
            {"price": price, "breadth": breadth}, index=index
        )

        result = analysis.canonical_sell_vote_frame(frame).iloc[-1]

        self.assertAlmostEqual(result[analysis.SELL_FEATURE], 2 / 3)
        self.assertFalse(result["canonical_bearish_divergence"])

    def test_augmented_vector_preserves_six_baseline_features(self) -> None:
        augmented, _, _ = analysis.build_augmented_vector()
        baseline = analysis.base_analysis.build_full_transition_vector()
        overlap = augmented.index.intersection(baseline.index)[-500:]

        pd.testing.assert_frame_equal(
            augmented.loc[overlap, list(analysis.BASE_COLUMNS)],
            baseline.loc[overlap, list(analysis.BASE_COLUMNS)],
            check_freq=False,
        )


if __name__ == "__main__":
    unittest.main()
