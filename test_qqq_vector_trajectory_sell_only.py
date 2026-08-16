import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import qqq_vector_trajectory_sell_only as challenger


class VectorTrajectorySellOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = challenger.qbt.load_data()

    def test_sell_wrapper_preserves_all_buy_inputs(self) -> None:
        captured = {}

        def fake_run_strategy(experiment, **kwargs):
            captured["frame"] = experiment
            return pd.Series(1.0, index=experiment.index), [], None

        with patch.object(
            challenger.qbt,
            "run_strategy",
            side_effect=fake_run_strategy,
        ):
            challenger.run_sell_only(
                self.df, pd.Series(False, index=self.df.index)
            )

        experiment = captured["frame"]
        for column in (
            "breadth",
            "vix_vote",
            "ma200_vote",
            "vote_gate",
            "ma200_recross",
        ):
            pd.testing.assert_series_equal(
                experiment[column], self.df[column]
            )

    def test_baseline_replacement_has_exact_parity(self) -> None:
        baseline = challenger.qbt.run_strategy(
            self.df,
            cooldown_days=challenger.qbt.COOLDOWN_DAYS,
            execution_lag=challenger.qbt.EXECUTION_LAG,
            fill_on=challenger.qbt.FILL_PRICE,
        )
        replacement = challenger.run_sell_only(
            self.df,
            challenger.crash.baseline_divergence_signal(self.df),
            reason="bearish-divergence",
        )

        np.testing.assert_allclose(baseline[0], replacement[0])
        self.assertEqual(
            challenger.research.vector_buy.trade_signature(baseline[1]),
            challenger.research.vector_buy.trade_signature(replacement[1]),
        )
        self.assertEqual(baseline[2], replacement[2])

    def test_trajectory_crash_probability_is_causal(self) -> None:
        spx = challenger.crash.load_spx()["close"].reindex(self.df.index)
        vector = challenger.trajectory.build_trajectory_vector(
            self.df, spx
        )
        labels, _ = challenger.crash.forward_crash_labels(spx)
        original = challenger.online_trajectory_crash_probability(
            vector, labels
        )
        cutoff = 1000
        mutated_vector = vector.copy()
        mutated_labels = labels.copy()
        mutated_vector.iloc[cutoff + 1 :] = 10_000
        mutated_labels.iloc[cutoff + 1 :] = (
            1 - mutated_labels.iloc[cutoff + 1 :]
        )
        changed = challenger.online_trajectory_crash_probability(
            mutated_vector, mutated_labels
        )

        pd.testing.assert_series_equal(
            original["trajectory_crash_probability"].iloc[: cutoff + 1],
            changed["trajectory_crash_probability"].iloc[: cutoff + 1],
        )


if __name__ == "__main__":
    unittest.main()
