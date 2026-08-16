import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import qqq_trajectory_backtest as runner


class QQQTrajectoryBacktestTests(unittest.TestCase):
    def test_import_chain_parses_as_python_3_11(self):
        dependency_files = (
            "qqq_breadth_only_trajectory_extension.py",
            "qqq_breadth_ma200_trajectory_extension.py",
            "qqq_breadth_ma200_80_session_confirmation.py",
            "qqq_breadth_ma200_timed_washout_boost.py",
            "qqq_breadth_ma200_washout_boost.py",
        )
        for filename in dependency_files:
            path = Path(__file__).with_name(filename)
            with self.subTest(filename=filename):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 11),
                )

    def test_runner_uses_fixed_breadth_only_gate(self):
        index = pd.date_range("2025-01-01", periods=2, freq="B")
        df = pd.DataFrame(index=index)
        proxy = pd.DataFrame(index=index)
        sentinel = {"equity": "result"}
        with patch.object(runner.strategy, "run_ensemble", return_value=sentinel) as mocked:
            actual = runner.run_backtest(df, proxy)
        self.assertIs(actual, sentinel)
        mocked.assert_called_once_with(
            df,
            proxy,
            runner.TRAJECTORY_LOOKBACK,
        )

    def test_series_metrics_match_known_path(self):
        index = pd.date_range("2024-01-01", periods=253, freq="B")
        equity = pd.Series(10_000 * (1.001 ** pd.RangeIndex(253)), index=index)
        metrics = runner._series_metrics(equity)
        self.assertAlmostEqual(metrics["total_return"], 1.001**252 - 1)
        self.assertGreater(metrics["cagr"], 0)
        self.assertEqual(metrics["max_drawdown"], 0.0)
        self.assertEqual(metrics["ulcer_index"], 0.0)

    def test_save_artifacts_keeps_components_separate(self):
        index = pd.date_range("2025-01-01", periods=3, freq="B")
        equity = pd.Series([10_000.0, 10_050.0, 10_100.0], index=index)
        position = pd.Series([False, True, True], index=index)
        trade = {
            "entry_date": index[1],
            "exit_date": index[2],
            "entry_price": 100.0,
            "exit_price": 101.0,
            "return_pct": 1.0,
            "max_drawdown_pct": 0.0,
            "accumulated": 7_070.0,
            "buy_trigger": "washout",
            "sell_reason": "test",
        }
        run = {
            "equity": equity,
            "position": position,
            "breadth": (equity * 0.7, [trade], None, position),
            "trend": (equity * 0.3, [], None, position),
            "extension_decisions": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = runner.save_artifacts(Path(tmp), pd.DataFrame(index=index), equity, run)
            self.assertTrue(all(path.exists() for path in paths))
            trades = pd.read_csv(paths[1])
            self.assertEqual(trades.loc[0, "component"], "breadth")


if __name__ == "__main__":
    unittest.main()
