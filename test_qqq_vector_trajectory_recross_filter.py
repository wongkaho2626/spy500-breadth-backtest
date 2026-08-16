import unittest

import numpy as np
import pandas as pd

import qqq_vector_trajectory_recross_filter as challenger


class VectorTrajectoryRecrossFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = challenger.qbt.load_data()
        cls.spx = challenger.analytics.load_spx()["close"].reindex(
            cls.df.index
        )

    def test_rolling_linear_slope_matches_known_line(self) -> None:
        dates = pd.date_range("2020-01-01", periods=30, freq="B")
        series = pd.Series(3 + 2 * np.arange(30), index=dates)
        slope = challenger.rolling_linear_slope(series, window=20)

        self.assertTrue(slope.iloc[:19].isna().all())
        np.testing.assert_allclose(slope.iloc[19:], 2.0)

    def test_trajectory_vector_preserves_static_features(self) -> None:
        static = challenger.analytics.build_market_vector(
            self.df, self.spx
        )
        trajectory = challenger.build_trajectory_vector(
            self.df, self.spx
        )

        pd.testing.assert_frame_equal(
            trajectory[list(challenger.vector_buy.FEATURE_COLUMNS)],
            static[list(challenger.vector_buy.FEATURE_COLUMNS)],
        )
        self.assertEqual(
            set(challenger.SLOPE_FEATURE_COLUMNS),
            set(trajectory.columns) - set(static.columns),
        )

    def test_trajectory_features_are_causal(self) -> None:
        cutoff = 1000
        original = challenger.build_trajectory_vector(
            self.df, self.spx
        )
        mutated_df = self.df.copy()
        mutated_spx = self.spx.copy()
        mutated_df.loc[mutated_df.index[cutoff + 1 :], "vix"] = 999.0
        mutated_df.loc[
            mutated_df.index[cutoff + 1 :], "breadth"
        ] = 999.0
        mutated_spx.iloc[cutoff + 1 :] *= 10.0
        changed = challenger.build_trajectory_vector(
            mutated_df, mutated_spx
        )

        pd.testing.assert_frame_equal(
            original.iloc[: cutoff + 1],
            changed.iloc[: cutoff + 1],
        )

    def test_executed_recross_entries_meet_trajectory_threshold(self) -> None:
        vector = challenger.build_trajectory_vector(self.df, self.spx)
        labels, _ = challenger.vector_buy.forward_buy_labels(
            self.df["price"]
        )
        probability = challenger.online_trajectory_probability(
            vector, labels
        )["trajectory_buy_probability"]
        run = challenger.static_filter.run_vector_recross_filter(
            self.df, probability >= challenger.PRIMARY_THRESHOLD
        )
        audit = challenger.static_filter.challenger_entry_audit(
            self.df.index,
            run[1],
            run[2],
            probability,
            challenger.PRIMARY_THRESHOLD,
        )

        self.assertEqual(audit["recross_filter_violations"], 0)


if __name__ == "__main__":
    unittest.main()
