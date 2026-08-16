import unittest

import numpy as np
import pandas as pd

import qqq_transition_vector_similarity as analysis


class TransitionVectorSimilarityTests(unittest.TestCase):
    def test_robust_scaling_uses_median_and_iqr(self) -> None:
        frame = pd.DataFrame(
            {
                column: [0.0, 1.0, 2.0, 3.0, 100.0]
                for column in analysis.FEATURE_COLUMNS
            }
        )
        center, scale = analysis.robust_scale(frame)

        np.testing.assert_allclose(center, 2.0)
        np.testing.assert_allclose(scale, 2.0)

    def test_identical_vector_has_zero_distance(self) -> None:
        rows = pd.DataFrame(
            {
                column: [1.0, 2.0, 3.0]
                for column in analysis.FEATURE_COLUMNS
            },
            index=pd.date_range("2020-01-01", periods=3),
        )
        center, scale = analysis.robust_scale(rows)
        distance = analysis.standardized_distance(
            rows.iloc[1], rows, center, scale
        )

        self.assertEqual(distance.iloc[1], 0.0)

    def test_trajectory_distance_uses_only_three_slopes(self) -> None:
        rows = pd.DataFrame(
            {
                column: [1.0, 2.0, 3.0]
                for column in analysis.FEATURE_COLUMNS
            },
            index=pd.date_range("2020-01-01", periods=3),
        )
        center, scale = analysis.robust_scale(
            rows.loc[:, analysis.TRAJECTORY_COLUMNS]
        )
        query = rows.iloc[1].copy()
        query["vix"] = 999.0
        distance = analysis.standardized_distance(
            query,
            rows,
            center,
            scale,
            analysis.TRAJECTORY_COLUMNS,
        )

        self.assertEqual(distance.iloc[1], 0.0)

    def test_rank_auc(self) -> None:
        labels = pd.Series([False, False, True, True])
        perfect = pd.Series([0.1, 0.2, 0.8, 0.9])
        reversed_scores = perfect.iloc[::-1].reset_index(drop=True)

        self.assertEqual(analysis.rank_auc(labels, perfect), 1.0)
        self.assertEqual(analysis.rank_auc(labels, reversed_scores), 0.0)

    def test_full_vector_matches_strategy_features_on_overlap(self) -> None:
        full = analysis.build_full_transition_vector()
        df = analysis.transition.qbt.load_data()
        spx = analysis.transition.crash.load_spx()["close"].reindex(df.index)
        strategy = analysis.transition.build_transition_features(df, spx)
        overlap = full.index.intersection(strategy.index)[-500:]

        pd.testing.assert_frame_equal(
            full.loc[overlap, list(analysis.FEATURE_COLUMNS)],
            strategy.loc[overlap, list(analysis.FEATURE_COLUMNS)],
            check_freq=False,
        )


if __name__ == "__main__":
    unittest.main()
