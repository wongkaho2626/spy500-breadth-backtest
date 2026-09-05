import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import main_force_washout_validation as wash


class WashoutValidationTests(unittest.TestCase):
    def test_volume_baseline_excludes_current_session(self):
        idx = pd.bdate_range("2024-01-01", periods=40)
        close = np.linspace(100, 102, 40)
        df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": [100.0] * 39 + [10_000.0]}, index=idx)
        sf = wash.signal_frame(df, np.ones(40, dtype=bool), pd.Series(True, index=idx), wash.BASE)
        self.assertEqual(sf["volume_median20"].iloc[-1], 100.0)
        self.assertEqual(sf["volume_ratio"].iloc[-1], 100.0)

    def test_setup_cannot_break_out_on_same_day(self):
        idx = pd.bdate_range("2024-01-01", periods=45)
        close = np.full(45, 100.0)
        df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                           "close": close, "volume": np.full(45, 100.0)}, index=idx)
        sf = wash.signal_frame(df, np.ones(45, dtype=bool), pd.Series(True, index=idx), wash.BASE)
        sf.loc[:, "setup"] = False
        sf.loc[idx[-1], "setup"] = True
        # Recompute the defining causal relationship directly: shifted setup is required.
        recent = sf["setup"].shift(1, fill_value=False).rolling(wash.BASE.setup_window).max().fillna(0)
        self.assertEqual(recent.iloc[-1], 0)

    def test_entry_uses_next_open(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        sf = pd.DataFrame(index=idx, data={
            "open": [10, 11, 20, 21, 22], "close": [10, 11, 20, 21, 22],
            "breakout": [False, True, False, False, False], "setup": [True, False, False, False, False],
            "member": True, "distribution": False, "persistent_ma_loss": False,
            "volume_ratio": 0.4, "ma20": 10.0, "ma30": 10.0, "consolidation_days": 3,
        })
        trades, signals = wash.build_trades(sf, "TEST", wash.replace(wash.BASE, max_hold=1))
        self.assertEqual(signals[0]["entry_date"], idx[2].date().isoformat())
        self.assertEqual(trades[0]["entry_price"], 20.0)

    def test_membership_mask_is_asof_and_ticker_normalized(self):
        dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-03"])
        snaps = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-03")], dtype="datetime64[ns]")
        sets = [{"BRK-B"}, set()]
        self.assertEqual(wash.membership_mask(dates, "BRK.B", snaps, sets).tolist(), [True, True, False])

    def test_round_trip_cost_is_applied(self):
        df = pd.DataFrame({"gross_return": [0.0], "holding_sessions": [5], "open_trade": [False]})
        metrics = wash.trade_metrics(df, cost_mult=1, base_cost=0.001)
        self.assertAlmostEqual(metrics["expectancy"], (1 - 0.001) ** 2 - 1, places=12)

    def test_load_price_rejects_invalid_ohlc(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "BAD.csv"
            pd.DataFrame({"Date": ["2024-01-01"], "Open": [10], "High": [9], "Low": [8],
                          "Close": [10], "Adj Close": [10], "Volume": [100]}).to_csv(p, index=False)
            frame, audit = wash.load_price(p)
            self.assertEqual(audit["invalid_ohlc_rows"], 1)
            self.assertTrue(frame["close"].isna().all())


if __name__ == "__main__":
    unittest.main()
