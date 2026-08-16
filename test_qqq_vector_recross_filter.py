import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import qqq_vector_recross_filter as challenger


class VectorRecrossFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = challenger.qbt.load_data()

    def test_filter_only_changes_ma200_recross_column(self) -> None:
        allowed = pd.Series(False, index=self.df.index)
        captured = {}

        def fake_run_strategy(experiment, **kwargs):
            captured["experiment"] = experiment
            return pd.Series(1.0, index=experiment.index), [], None

        with patch.object(
            challenger.qbt, "run_strategy", side_effect=fake_run_strategy
        ):
            challenger.run_vector_recross_filter(self.df, allowed)

        experiment = captured["experiment"]
        unchanged = [
            column
            for column in self.df.columns
            if column != "ma200_recross"
        ]
        pd.testing.assert_frame_equal(
            experiment[unchanged], self.df[unchanged]
        )
        self.assertFalse(experiment["ma200_recross"].any())

    def test_filter_restores_cost_constants(self) -> None:
        old_commission = challenger.qbt.COMMISSION
        old_slippage = challenger.qbt.SLIPPAGE
        allowed = pd.Series(True, index=self.df.index)

        challenger.run_vector_recross_filter(
            self.df, allowed, cost_multiplier=5
        )

        self.assertEqual(challenger.qbt.COMMISSION, old_commission)
        self.assertEqual(challenger.qbt.SLIPPAGE, old_slippage)

    def test_all_true_filter_has_exact_baseline_parity(self) -> None:
        baseline = challenger.vector_buy.baseline_run(self.df)
        filtered = challenger.run_vector_recross_filter(
            self.df, pd.Series(True, index=self.df.index)
        )

        np.testing.assert_allclose(baseline[0], filtered[0])
        self.assertEqual(
            challenger.vector_buy.trade_signature(baseline[1]),
            challenger.vector_buy.trade_signature(filtered[1]),
        )
        self.assertEqual(baseline[2], filtered[2])

    def test_executed_recross_entries_meet_threshold(self) -> None:
        spx = challenger.analytics.load_spx()["close"].reindex(self.df.index)
        vector = challenger.analytics.build_market_vector(self.df, spx)
        labels, _ = challenger.vector_buy.forward_buy_labels(
            self.df["price"]
        )
        probability = challenger.vector_buy.online_buy_probability(
            vector, labels
        )["buy_probability"]
        _, trades, open_trade = challenger.run_vector_recross_filter(
            self.df, probability >= challenger.PRIMARY_THRESHOLD
        )
        audit = challenger.challenger_entry_audit(
            self.df.index,
            trades,
            open_trade,
            probability,
            challenger.PRIMARY_THRESHOLD,
        )

        self.assertEqual(audit["recross_filter_violations"], 0)


if __name__ == "__main__":
    unittest.main()
