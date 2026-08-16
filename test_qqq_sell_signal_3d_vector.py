import unittest

import pandas as pd

import qqq_sell_signal_3d_vector as analysis


class SellSignal3DVectorTests(unittest.TestCase):
    def test_vector_has_exactly_three_continuous_sell_inputs(self) -> None:
        vector, _ = analysis.build_sell_3d_vector()

        self.assertEqual(len(analysis.SELL_3D_COLUMNS), 3)
        self.assertTrue(
            all(
                pd.api.types.is_float_dtype(vector[column])
                for column in analysis.SELL_3D_COLUMNS
            )
        )

    def test_old_six_dimensions_are_excluded(self) -> None:
        self.assertTrue(
            set(analysis.SELL_3D_COLUMNS).isdisjoint(
                analysis.shared.BASE_COLUMNS
            )
        )


if __name__ == "__main__":
    unittest.main()
