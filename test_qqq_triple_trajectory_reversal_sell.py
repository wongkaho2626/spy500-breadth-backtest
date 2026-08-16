import unittest

import numpy as np
import pandas as pd

import qqq_triple_trajectory_reversal_sell as challenger


class TripleTrajectoryReversalTests(unittest.TestCase):
    def test_three_crosses_must_cluster_inside_window(self) -> None:
        dates = pd.date_range("2020-01-01", periods=8, freq="B")
        features = pd.DataFrame(
            {
                "ndx_deceleration_cross": [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
                "vix_slope_up_cross": [
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                ],
                "drawdown_slope_down_cross": [
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                ],
                "positive_ndx_momentum_gate": True,
                "near_high_gate": True,
            },
            index=dates,
        )

        five_day = challenger.transition_signal(features, 5)
        three_day = challenger.transition_signal(features, 3)

        self.assertTrue(five_day.iloc[6])
        self.assertFalse(three_day.iloc[6])

    def test_gates_block_clustered_crosses(self) -> None:
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        features = pd.DataFrame(
            {
                "ndx_deceleration_cross": [True, False, False],
                "vix_slope_up_cross": [False, True, False],
                "drawdown_slope_down_cross": [False, False, True],
                "positive_ndx_momentum_gate": [True, True, True],
                "near_high_gate": [True, True, False],
            },
            index=dates,
        )

        signal = challenger.transition_signal(features, 3)

        self.assertFalse(signal.iloc[-1])

    def test_future_mutation_does_not_change_past_features(self) -> None:
        df = challenger.qbt.load_data()
        spx = challenger.crash.load_spx()["close"].reindex(df.index)
        original = challenger.build_transition_features(df, spx)
        cutoff = 1000
        mutated_df = df.copy()
        mutated_spx = spx.copy()
        mutated_df.loc[
            mutated_df.index[cutoff + 1 :], "price"
        ] *= 10
        mutated_df.loc[
            mutated_df.index[cutoff + 1 :], "vix"
        ] = 999
        mutated_spx.iloc[cutoff + 1 :] *= 10
        changed = challenger.build_transition_features(
            mutated_df, mutated_spx
        )

        pd.testing.assert_frame_equal(
            original.iloc[: cutoff + 1],
            changed.iloc[: cutoff + 1],
        )

    def test_baseline_and_buy_input_integrity(self) -> None:
        df = challenger.qbt.load_data()
        parity = challenger.crash.parity_check(df)
        integrity = challenger.sell_harness.entry_logic_integrity(df)

        self.assertTrue(parity["passed"])
        self.assertTrue(integrity["passed"])
        self.assertEqual(parity["equity_max_absolute_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
