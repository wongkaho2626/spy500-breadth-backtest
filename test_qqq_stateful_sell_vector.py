import unittest

import pandas as pd

import qqq_stateful_sell_vector as analysis


class StatefulSellVectorTests(unittest.TestCase):
    def test_age_is_capped_and_causal(self) -> None:
        event = pd.Series([False] * 70)
        self.assertEqual(analysis.capped_age(event), analysis.AGE_CAP)
        event.iloc[-4] = True
        self.assertEqual(analysis.capped_age(event), 3)

    def test_stateful_vector_contains_requested_state(self) -> None:
        required = {
            "reason_bearish_divergence",
            "reason_climax_top",
            "reason_trailing_stop",
            "return_since_entry_pct",
            "trade_high_gain_from_entry_pct",
            "current_trade_drawdown_pct",
            "macd_cross_age_capped",
            "extension_age_capped",
            "climax_active",
            "trailing_stop_active",
            "ndx_return_60_slope_20",
            "vix_slope_20",
            "spx_drawdown_252_slope_20",
        }
        self.assertTrue(required.issubset(analysis.STATEFUL_COLUMNS))

    def test_raw_prices_are_audit_only(self) -> None:
        self.assertNotIn("entry_price_raw", analysis.STATEFUL_COLUMNS)
        self.assertNotIn("trade_high_raw", analysis.STATEFUL_COLUMNS)

    def test_historical_sell_reason_is_one_hot(self) -> None:
        sell_vector, strategy = analysis.sell3d.build_sell_3d_vector()
        qbt = analysis.shared.base_analysis.transition.qbt
        _, trades, _ = qbt.run_strategy(
            strategy,
            cooldown_days=qbt.COOLDOWN_DAYS,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
        )
        events = analysis.build_stateful_events(
            strategy, trades, sell_vector
        )
        reason_columns = [
            "reason_bearish_divergence",
            "reason_climax_top",
            "reason_trailing_stop",
        ]

        self.assertEqual(len(events), 17)
        self.assertTrue((events[reason_columns].sum(axis=1) == 1).all())


if __name__ == "__main__":
    unittest.main()
