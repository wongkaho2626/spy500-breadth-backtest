import unittest

import numpy as np
import pandas as pd

import qqq_sell_3d_economic_targets as analysis


class Sell3DEconomicTargetsTests(unittest.TestCase):
    def test_distance_weights_sum_to_one(self) -> None:
        weights = analysis.distance_weights(pd.Series([0.0, 1.0, 2.0]))

        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_expanding_predictions_do_not_use_unresolved_targets(self) -> None:
        dates = pd.date_range("2020-01-01", periods=6, freq="100D")
        frame = pd.DataFrame(index=dates)
        for column in analysis.sell3d.SELL_3D_COLUMNS:
            frame[column] = np.arange(6, dtype=float)
        frame["target_end_date"] = dates + pd.Timedelta(days=90)
        for target in analysis.CONTINUOUS_TARGETS:
            frame[target] = np.arange(6, dtype=float)
        for target in analysis.BINARY_TARGETS:
            frame[target] = [False, True, False, True, False, True]

        predictions = analysis.expanding_predictions(frame)

        self.assertEqual(len(predictions), 2)
        self.assertEqual(predictions.iloc[0]["training_events"], 4)
        self.assertNotIn(dates[4].strftime("%Y-%m-%d"), predictions.iloc[0]["neighbor_dates"])


if __name__ == "__main__":
    unittest.main()
