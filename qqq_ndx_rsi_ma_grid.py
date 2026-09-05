"""Fixed-grid research challenger: NDX RSI crossover plus MA regime gate.

This experiment keeps the frozen QQQ strategy unchanged and adds one lower-
priority re-entry path while flat after a prior exit and the canonical
cooldown. The extra entry requires a short RSI crossing above a longer RSI and
the NDX close above a fixed moving-average window, all observed at the close
and filled on the next session open.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_ndx_rsi14_divergence_confirmation as common
import qqq_ndx_rsi7_14_death_cross_exit as crossover
import qqq_ndx_rsi7_14_golden_cross_reentry as reentry


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/ndx_rsi_ma_grid_idea.md"
REPORT_FILE = ROOT / "docs/research/ndx_rsi_ma_grid_report.md"
RESULTS_FILE = ROOT / "qqq_ndx_rsi_ma_grid_results.json"
GRID_FILE = ROOT / "qqq_ndx_rsi_ma_grid_grid.csv"
EQUITY_FILE = ROOT / "qqq_ndx_rsi_ma_grid_equity.csv"
TRADES_FILE = ROOT / "qqq_ndx_rsi_ma_grid_trades.csv"
SIGNALS_FILE = ROOT / "qqq_ndx_rsi_ma_grid_signals.csv"

START_DATE = "2002-01-01"
TRAIN_END = "2013-12-31"
TEST_START = "2014-01-01"
REAL_BREADTH_START = "2007-01-01"
SHORT_WINDOWS = (3, 5, 7, 9, 11)
LONG_WINDOWS = (10, 14, 21, 28)
MA_WINDOWS = (100, 150, 200, 250, 300)
RSI_PAIRS = tuple(
    (short_window, long_window)
    for short_window in SHORT_WINDOWS
    for long_window in LONG_WINDOWS
    if short_window < long_window
)
GRID = tuple(
    (short_window, long_window, ma_window)
    for short_window, long_window in RSI_PAIRS
    for ma_window in MA_WINDOWS
)
RELATED_TRIALS = 4_918

qbt = common.qbt
framework = common.framework
analytics = common.analytics


def build_features(
    index: pd.DatetimeIndex,
    ndx_close: pd.Series,
    short_rsi: pd.Series,
    long_rsi: pd.Series,
    ma_series: pd.Series,
) -> pd.DataFrame:
    aligned_close = ndx_close.reindex(index)
    aligned_short = short_rsi.reindex(index)
    aligned_long = long_rsi.reindex(index)
    aligned_ma = ma_series.reindex(index)
    valid = aligned_short.notna() & aligned_long.notna() & aligned_ma.notna()
    golden_cross = (
        valid
        & (aligned_short > aligned_long)
        & (aligned_short.shift(1) <= aligned_long.shift(1))
    ).fillna(False)
    above_ma = (valid & (aligned_close > aligned_ma)).fillna(False)
    return pd.DataFrame(
        {
            "ndx_close": aligned_close,
            "short_rsi": aligned_short,
            "long_rsi": aligned_long,
            "ma_value": aligned_ma,
            "golden_cross": golden_cross,
            "above_ma": above_ma,
            "selected_signal": (golden_cross & above_ma).fillna(False),
        },
        index=index,
    )


def row_params(row: pd.Series | dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["short_window"]),
        int(row["long_window"]),
        int(row["ma_window"]),
    )


def combo_name(params: tuple[int, int, int]) -> str:
    short_window, long_window, ma_window = params
    return f"rsi_{short_window}_{long_window}_ma_{ma_window}"


def segment_metrics(
    equity: pd.Series, start: str, end: str | None = None
) -> dict[str, float | int]:
    metrics = analytics.slice_metrics(equity, start, end)
    if metrics.get("observations", 0) < 2:
        return metrics
    drawdown = float(metrics["max_drawdown"])
    calmar = (
        float(metrics["cagr"] / abs(drawdown))
        if drawdown < 0
        else np.nan
    )
    return {**metrics, "calmar": calmar}


def annotate_run(
    df: pd.DataFrame,
    features: pd.DataFrame,
    params: tuple[int, int, int],
    run: tuple[pd.Series, list[dict], dict | None],
) -> tuple[pd.Series, list[dict], dict | None]:
    short_window, long_window, ma_window = params
    annotated_trades: list[dict] = []
    for original in run[1]:
        trade = dict(original)
        trade["short_window"] = short_window
        trade["long_window"] = long_window
        trade["ma_window"] = ma_window
        if trade.get("buy_trigger") == "RSI-golden-cross":
            entry_location = df.index.get_loc(trade["entry_date"])
            signal_date = df.index[entry_location - qbt.EXECUTION_LAG]
            trade["signal_date"] = signal_date
            trade["short_rsi_on_signal"] = float(
                features.loc[signal_date, "short_rsi"]
            )
            trade["long_rsi_on_signal"] = float(
                features.loc[signal_date, "long_rsi"]
            )
            trade["ma_value_on_signal"] = float(
                features.loc[signal_date, "ma_value"]
            )
            trade["ndx_close_on_signal"] = float(
                features.loc[signal_date, "ndx_close"]
            )
            trade["selected_signal_on_signal_date"] = bool(
                features.loc[signal_date, "selected_signal"]
            )
        annotated_trades.append(trade)
    open_trade = dict(run[2]) if run[2] else None
    if open_trade is not None:
        open_trade["short_window"] = short_window
        open_trade["long_window"] = long_window
        open_trade["ma_window"] = ma_window
        if open_trade.get("buy_trigger") == "RSI-golden-cross":
            entry_location = df.index.get_loc(open_trade["entry_date"])
            signal_date = df.index[entry_location - qbt.EXECUTION_LAG]
            open_trade["signal_date"] = signal_date
            open_trade["short_rsi_on_signal"] = float(
                features.loc[signal_date, "short_rsi"]
            )
            open_trade["long_rsi_on_signal"] = float(
                features.loc[signal_date, "long_rsi"]
            )
            open_trade["ma_value_on_signal"] = float(
                features.loc[signal_date, "ma_value"]
            )
            open_trade["ndx_close_on_signal"] = float(
                features.loc[signal_date, "ndx_close"]
            )
            open_trade["selected_signal_on_signal_date"] = bool(
                features.loc[signal_date, "selected_signal"]
            )
    return run[0], annotated_trades, open_trade


def run_combo(
    df: pd.DataFrame,
    features: pd.DataFrame,
    params: tuple[int, int, int],
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    raw = reentry.run_variant(
        df, features["selected_signal"], cost_multiplier=cost_multiplier
    )
    return annotate_run(df, features, params, raw)


def grid_row(
    df: pd.DataFrame,
    features: pd.DataFrame,
    params: tuple[int, int, int],
) -> dict[str, float | int]:
    run = run_combo(df, features, params)
    equity, trades, open_trade = run
    position = analytics.position_series(df.index, trades, open_trade)
    full = analytics.strategy_metrics(equity, trades, position)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    turnover = (
        position.astype(int).diff().abs().fillna(int(position.iloc[0])).sum() / years
    )
    train = segment_metrics(equity, START_DATE, TRAIN_END)
    test = segment_metrics(equity, TEST_START)
    real = segment_metrics(equity, REAL_BREADTH_START)
    short_window, long_window, ma_window = params
    executed_entries = [
        trade for trade in trades if trade.get("buy_trigger") == "RSI-golden-cross"
    ]
    if open_trade and open_trade.get("buy_trigger") == "RSI-golden-cross":
        executed_entries.append(open_trade)
    return {
        "short_window": short_window,
        "long_window": long_window,
        "ma_window": ma_window,
        "full_cagr": float(full["cagr"]),
        "full_sharpe": float(full["sharpe"]),
        "full_max_drawdown": float(full["max_drawdown"]),
        "full_calmar": float(full["calmar"]),
        "train_cagr": float(train["cagr"]),
        "train_sharpe": float(train["sharpe"]),
        "train_max_drawdown": float(train["max_drawdown"]),
        "train_calmar": float(train["calmar"]),
        "test_cagr": float(test["cagr"]),
        "test_sharpe": float(test["sharpe"]),
        "test_max_drawdown": float(test["max_drawdown"]),
        "test_calmar": float(test["calmar"]),
        "real_breadth_cagr": float(real["cagr"]),
        "real_breadth_sharpe": float(real["sharpe"]),
        "real_breadth_max_drawdown": float(real["max_drawdown"]),
        "real_breadth_calmar": float(real["calmar"]),
        "completed_trades": int(full["completed_trades"]),
        "exposure": float(full["exposure"]),
        "expectancy": float(full["expectancy"]),
        "turnover_position_changes_per_year": float(turnover),
        "raw_signal_days": int(features["selected_signal"].sum()),
        "executed_golden_cross_entries": int(len(executed_entries)),
    }


def sort_grid(grid: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return grid.sort_values(
        [
            f"{prefix}_calmar",
            f"{prefix}_sharpe",
            f"{prefix}_cagr",
            "short_window",
            "long_window",
            "ma_window",
        ],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def select_winners(grid: pd.DataFrame) -> dict[str, dict[str, Any]]:
    primary = sort_grid(grid, "train").iloc[0]
    reverse = sort_grid(grid, "test").iloc[0]
    full = sort_grid(grid, "full").iloc[0]
    return {
        "primary_train_2002_2013": primary.to_dict(),
        "reverse_train_2014_plus": reverse.to_dict(),
        "full_history_descriptive": full.to_dict(),
    }


def rank_columns(grid: pd.DataFrame) -> pd.DataFrame:
    ranked = grid.copy()
    for rank_name, prefix in (
        ("primary_rank", "train"),
        ("reverse_rank", "test"),
        ("full_rank", "full"),
    ):
        ordering = sort_grid(ranked, prefix).loc[
            :, ["short_window", "long_window", "ma_window"]
        ]
        ordering[rank_name] = np.arange(1, len(ordering) + 1)
        ranked = ranked.merge(
            ordering, on=["short_window", "long_window", "ma_window"], how="left"
        )
    return ranked.sort_values("primary_rank", kind="stable").reset_index(drop=True)


def local_neighbourhood_stability(
    grid: pd.DataFrame, winner: dict[str, Any]
) -> dict[str, Any]:
    short_window, long_window, ma_window = row_params(winner)
    key = {
        (int(row.short_window), int(row.long_window), int(row.ma_window)): row
        for row in grid.itertuples(index=False)
    }
    neighbours: list[dict[str, Any]] = []
    for candidate_short in (
        SHORT_WINDOWS[max(0, SHORT_WINDOWS.index(short_window) - 1)]
        if short_window != SHORT_WINDOWS[0]
        else None,
        SHORT_WINDOWS[
            min(len(SHORT_WINDOWS) - 1, SHORT_WINDOWS.index(short_window) + 1)
        ]
        if short_window != SHORT_WINDOWS[-1]
        else None,
    ):
        if candidate_short is None or candidate_short >= long_window:
            continue
        row = key.get((candidate_short, long_window, ma_window))
        if row is None:
            continue
        neighbours.append({"dimension": "short_window", "row": row})
    for candidate_long in (
        LONG_WINDOWS[max(0, LONG_WINDOWS.index(long_window) - 1)]
        if long_window != LONG_WINDOWS[0]
        else None,
        LONG_WINDOWS[
            min(len(LONG_WINDOWS) - 1, LONG_WINDOWS.index(long_window) + 1)
        ]
        if long_window != LONG_WINDOWS[-1]
        else None,
    ):
        if candidate_long is None or short_window >= candidate_long:
            continue
        row = key.get((short_window, candidate_long, ma_window))
        if row is None:
            continue
        neighbours.append({"dimension": "long_window", "row": row})
    for candidate_ma in (
        MA_WINDOWS[max(0, MA_WINDOWS.index(ma_window) - 1)]
        if ma_window != MA_WINDOWS[0]
        else None,
        MA_WINDOWS[min(len(MA_WINDOWS) - 1, MA_WINDOWS.index(ma_window) + 1)]
        if ma_window != MA_WINDOWS[-1]
        else None,
    ):
        if candidate_ma is None:
            continue
        row = key.get((short_window, long_window, candidate_ma))
        if row is None:
            continue
        neighbours.append({"dimension": "ma_window", "row": row})

    winner_train = float(winner["train_calmar"])
    winner_full = float(winner["full_calmar"])
    rows: list[dict[str, Any]] = []
    supportive = 0
    for neighbour in neighbours:
        row = neighbour["row"]
        train_ratio = (
            float(row.train_calmar) / winner_train
            if np.isfinite(winner_train) and winner_train != 0
            else np.nan
        )
        full_ratio = (
            float(row.full_calmar) / winner_full
            if np.isfinite(winner_full) and winner_full != 0
            else np.nan
        )
        supports = bool(
            np.isfinite(train_ratio)
            and np.isfinite(full_ratio)
            and train_ratio >= 0.85
            and full_ratio >= 0.80
        )
        supportive += int(supports)
        rows.append(
            {
                "dimension": neighbour["dimension"],
                "short_window": int(row.short_window),
                "long_window": int(row.long_window),
                "ma_window": int(row.ma_window),
                "train_calmar": float(row.train_calmar),
                "test_calmar": float(row.test_calmar),
                "full_calmar": float(row.full_calmar),
                "train_calmar_ratio_to_winner": float(train_ratio),
                "full_calmar_ratio_to_winner": float(full_ratio),
                "supports_plateau": supports,
            }
        )
    fraction = supportive / len(rows) if rows else 0.0
    return {
        "definition": (
            "immediate neighbour supports plateau when train Calmar >= 85% "
            "and full-history Calmar >= 80% of the selected point"
        ),
        "winner_params": [short_window, long_window, ma_window],
        "neighbour_count": len(rows),
        "supportive_neighbour_count": supportive,
        "supportive_neighbour_fraction": float(fraction),
        "passed": bool(rows and fraction >= 0.5),
        "rows": rows,
    }


def write_artifacts(
    df: pd.DataFrame,
    benchmark: pd.Series,
    baseline_run: tuple[pd.Series, list[dict], dict | None],
    candidate_runs: dict[str, tuple[pd.Series, list[dict], dict | None]],
    primary_features: pd.DataFrame,
    primary_params: tuple[int, int, int],
    grid: pd.DataFrame,
) -> None:
    equity = pd.DataFrame(
        {
            "ndx_buy_hold": benchmark,
            "baseline": baseline_run[0],
            **{name: run[0] for name, run in candidate_runs.items()},
        }
    )
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)

    trade_rows: list[dict] = []
    for name, run in candidate_runs.items():
        trade_rows.extend({"selection": name, **trade} for trade in run[1])
        if run[2]:
            trade_rows.append({"selection": name, "open_trade": True, **run[2]})
    pd.DataFrame(trade_rows).to_csv(TRADES_FILE, index=False)

    short_window, long_window, ma_window = primary_params
    signal_frame = primary_features.copy()
    signal_frame["short_window"] = short_window
    signal_frame["long_window"] = long_window
    signal_frame["ma_window"] = ma_window
    signal_frame["next_session_open"] = df["open"].shift(-1)
    signal_frame.index.name = "Date"
    signal_frame.to_csv(SIGNALS_FILE)

    grid.to_csv(GRID_FILE, index=False)


def write_report(results: dict[str, Any]) -> None:
    baseline = results["baseline"]
    challenger = results["challenger"]
    bm, cm = baseline["metrics"], challenger["metrics"]
    bs, cs = baseline["score"], challenger["score"]
    primary = results["winners"]["primary_train_2002_2013"]
    reverse = results["winners"]["reverse_train_2014_plus"]
    full = results["winners"]["full_history_descriptive"]
    verdict = "拒絕採用" if results["decision"] == "reject" else "保留作研究追蹤"
    lines = [
        "# 回測驗證報告 — NDX RSI 黃金交叉 + MA 固定網格",
        "",
        f"## 研究決定：{verdict}",
        "",
        "## 摘要",
        "",
        (
            f"依照 2026 年 9 月 2 日預先登記的 95 組固定網格，"
            f"主選組合是 RSI({int(primary['short_window'])}/{int(primary['long_window'])}) "
            f"+ MA{int(primary['ma_window'])}。"
            f"挑戰策略評分 **{cs['final_score']} / 100（{cs['band']}）**，"
            f"原有策略 **{bs['final_score']} / 100（{bs['band']}）**。"
        ),
        (
            "主方向排序用 2002-01-01 至 2013-12-31 訓練、"
            "2014-01-01 起測試；反向排序則相反。"
            "所有歷史資料至 2026-07-02 都屬已見樣本，所以只可視為 pseudo-OOS。"
        ),
        (
            "95 組之中，訓練半段、測試半段及全期 Calmar 高過原有策略的組合數"
            f"分別是 {results['grid_diagnostics']['train_combinations_beating_baseline_calmar']}、"
            f"{results['grid_diagnostics']['test_combinations_beating_baseline_calmar']}、"
            f"{results['grid_diagnostics']['full_combinations_beating_baseline_calmar']}。"
        ),
        "",
        "## 入選組合",
        "",
        "| 排名用途 | 參數 | 訓練 Calmar | 測試 Calmar | 全期 Calmar | 全期 CAGR | 交易數 |",
        "|---|---|---:|---:|---:|---:|---:|",
        (
            f"| 主方向（2002-2013 訓練） | RSI({int(primary['short_window'])}/"
            f"{int(primary['long_window'])}) + MA{int(primary['ma_window'])} | "
            f"{primary['train_calmar']:.3f} | {primary['test_calmar']:.3f} | "
            f"{primary['full_calmar']:.3f} | {primary['full_cagr']:.2%} | "
            f"{int(primary['completed_trades'])} |"
        ),
        (
            f"| 反方向（2014+ 訓練） | RSI({int(reverse['short_window'])}/"
            f"{int(reverse['long_window'])}) + MA{int(reverse['ma_window'])} | "
            f"{reverse['test_calmar']:.3f} | {reverse['train_calmar']:.3f} | "
            f"{reverse['full_calmar']:.3f} | {reverse['full_cagr']:.2%} | "
            f"{int(reverse['completed_trades'])} |"
        ),
        (
            f"| 全樣本描述 | RSI({int(full['short_window'])}/{int(full['long_window'])}) "
            f"+ MA{int(full['ma_window'])} | {full['train_calmar']:.3f} | "
            f"{full['test_calmar']:.3f} | {full['full_calmar']:.3f} | "
            f"{full['full_cagr']:.2%} | {int(full['completed_trades'])} |"
        ),
        "",
        "## 回測評分",
        "",
        "| 組成 | 原有策略 | 主選挑戰策略 | 滿分 |",
        "|---|---:|---:|---:|",
        f"| A. 統計有效性 | {bs['A_statistical_validity']} | {cs['A_statistical_validity']} | 30 |",
        f"| B. 風險調整表現 | {bs['B_risk_adjusted_performance']} | {cs['B_risk_adjusted_performance']} | 25 |",
        f"| C. 穩健性／OOS | {bs['C_robustness_oos']} | {cs['C_robustness_oos']} | 25 |",
        f"| D. 交易質素／一致性 | {bs['D_trade_quality_consistency']} | {cs['D_trade_quality_consistency']} | 20 |",
        f"| 原始總分 | {bs['raw_score']} | {cs['raw_score']} | 100 |",
        f"| 硬性上限 | {bs['hard_cap']} | {cs['hard_cap']} | |",
        f"| **最終分數** | **{bs['final_score']}** | **{cs['final_score']}** | **100** |",
        "",
        "## 主選組合 vs 原有策略",
        "",
        "| 指標 | 原有策略 | 主選挑戰策略 | 差異 |",
        "|---|---:|---:|---:|",
        f"| CAGR | {bm['cagr']:.2%} | {cm['cagr']:.2%} | {cm['cagr']-bm['cagr']:+.2%} |",
        f"| 波動率 | {bm['annual_volatility']:.2%} | {cm['annual_volatility']:.2%} | {cm['annual_volatility']-bm['annual_volatility']:+.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {cm['sharpe']:.3f} | {cm['sharpe']-bm['sharpe']:+.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {cm['sortino']:.3f} | {cm['sortino']-bm['sortino']:+.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {cm['calmar']:.3f} | {cm['calmar']-bm['calmar']:+.3f} |",
        f"| 最大回撤 | {bm['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} | {cm['max_drawdown']-bm['max_drawdown']:+.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} | {cm['ulcer_index']-bm['ulcer_index']:+.2%} |",
        f"| 水底時間 | {bm['time_underwater']:.2%} | {cm['time_underwater']:.2%} | {cm['time_underwater']-bm['time_underwater']:+.2%} |",
        f"| 完成交易 | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        f"| 勝率 | {bm['win_rate']:.2%} | {cm['win_rate']:.2%} | {cm['win_rate']-bm['win_rate']:+.2%} |",
        f"| 盈虧比 | {bm['payoff_ratio']:.2f} | {cm['payoff_ratio']:.2f} | {cm['payoff_ratio']-bm['payoff_ratio']:+.2f} |",
        f"| Profit factor | {bm['profit_factor']:.2f} | {cm['profit_factor']:.2f} | {cm['profit_factor']-bm['profit_factor']:+.2f} |",
        f"| 每宗期望 | {bm['expectancy']:.2%} | {cm['expectancy']:.2%} | {cm['expectancy']-bm['expectancy']:+.2%} |",
        f"| 持倉比例 | {bm['exposure']:.2%} | {cm['exposure']:.2%} | {cm['exposure']-bm['exposure']:+.2%} |",
        f"| 每年倉位轉換 | {bm['turnover_position_changes_per_year']:.2f} | {cm['turnover_position_changes_per_year']:.2f} | {cm['turnover_position_changes_per_year']-bm['turnover_position_changes_per_year']:+.2f} |",
        "",
        "### 尾部風險、回撤及基準比較",
        "",
        "| 指標 | 原有策略 | 主選挑戰策略 |",
        "|---|---:|---:|",
        f"| Omega（0% 門檻） | {bm['omega_zero']:.3f} | {cm['omega_zero']:.3f} |",
        f"| 每日 VaR 95% | {bm['var_95_daily']:.2%} | {cm['var_95_daily']:.2%} |",
        f"| 每日 CVaR 95% | {bm['cvar_95_daily']:.2%} | {cm['cvar_95_daily']:.2%} |",
        f"| 每日 VaR 99% | {bm['var_99_daily']:.2%} | {cm['var_99_daily']:.2%} |",
        f"| 每日 CVaR 99% | {bm['cvar_99_daily']:.2%} | {cm['cvar_99_daily']:.2%} |",
        f"| 平均回撤事件 | {bm['average_episode_drawdown']:.2%} | {cm['average_episode_drawdown']:.2%} |",
        f"| 平均修復時間 | {bm['average_recovery_sessions']:.1f} 日 | {cm['average_recovery_sessions']:.1f} 日 |",
        f"| 最長修復時間 | {bm['maximum_recovery_sessions']} 日 | {cm['maximum_recovery_sessions']} 日 |",
        f"| Pain ratio | {bm['pain_ratio']:.3f} | {cm['pain_ratio']:.3f} |",
        f"| 年化 alpha（相對 NDX） | {bm['benchmark_alpha_annualized']:.2%} | {cm['benchmark_alpha_annualized']:.2%} |",
        f"| Beta（相對 NDX） | {bm['benchmark_beta']:.3f} | {cm['benchmark_beta']:.3f} |",
        f"| 與 NDX 相關係數 | {bm['benchmark_correlation']:.3f} | {cm['benchmark_correlation']:.3f} |",
        "",
        "## 統計顯著性",
        "",
        f"- 有效日數：{bm['effective_daily_observations']:.0f} / {cm['effective_daily_observations']:.0f}。",
        f"- t-stat：{bm['mean_return_t_stat']:.3f} / {cm['mean_return_t_stat']:.3f}；PSR：{bm['psr_vs_zero']:.4f} / {cm['psr_vs_zero']:.4f}。",
        (
            f"- DSR 以固定 {results['data']['related_trials_including_grid']:,} 次相關測試計算："
            f"{baseline['statistical_diagnostics']['deflated_sharpe_probability']:.4f} / "
            f"{challenger['statistical_diagnostics']['deflated_sharpe_probability']:.4f}。"
        ),
        f"- 偏度：{bm['skewness']:.3f} / {cm['skewness']:.3f}；超額峰度：{bm['excess_kurtosis']:.3f} / {cm['excess_kurtosis']:.3f}。",
        f"- Jarque-Bera p：{bm['jarque_bera_p']:.3g} / {cm['jarque_bera_p']:.3g}；Ljung-Box p：{baseline['statistical_diagnostics']['ljung_box_p']:.3g} / {challenger['statistical_diagnostics']['ljung_box_p']:.3g}。",
        f"- 回報 ADF p：{bm['adf_return_p']:.3g} / {cm['adf_return_p']:.3g}。",
        (
            f"- 配對年化平均差 {results['paired_inference']['annualized_mean_difference']:+.2%}；"
            f"HAC t={results['paired_inference']['hac_t_stat']:.3f}，"
            f"p={results['paired_inference']['hac_two_sided_p']:.3f}。"
        ),
        (
            f"- 區塊 bootstrap 95% 區間："
            f"{results['paired_inference']['bootstrap_95_interval_annualized']}。"
        ),
        "",
        "## 穩健性",
        "",
        (
            f"- 主方向入選在 2014-01-01 起測試 Calmar "
            f"{primary['test_calmar']:.3f}，對照 baseline "
            f"{results['baseline_segments']['test']['calmar']:.3f}。"
        ),
        (
            f"- 反方向入選在 2002-01-01 至 2013-12-31 測試 Calmar "
            f"{reverse['train_calmar']:.3f}，對照 baseline "
            f"{results['baseline_segments']['train']['calmar']:.3f}。"
        ),
        (
            f"- 2007+ 真實 breadth Calmar 差："
            f"{results['period_deltas']['real_breadth_period']['calmar']:+.3f}。"
        ),
        (
            f"- 5x / 10x 成本下配對年化差："
            f"{results['cost_stress']['5x']['paired_annualized_mean']:+.2%} / "
            f"{results['cost_stress']['10x']['paired_annualized_mean']:+.2%}。"
        ),
        f"- 交易 bootstrap：{challenger['trade_bootstrap'].get('simulations', 0):,} 次。",
        "- 挑戰策略交易 bootstrap 終值回報分位：" + ", ".join(
            f"{name}={value:.1%}"
            for name, value in challenger["trade_bootstrap"]["terminal_return_percentiles"].items()
        ) + "。",
        "- 挑戰策略交易 bootstrap 最大回撤分位：" + ", ".join(
            f"{name}={value:.1%}"
            for name, value in challenger["trade_bootstrap"]["max_drawdown_percentiles"].items()
        ) + "。",
        (
            f"- 鄰近穩定性："
            f"{results['local_neighbourhood_stability']['supportive_neighbour_count']} / "
            f"{results['local_neighbourhood_stability']['neighbour_count']} 個即時鄰點支撐平台，"
            f"比例 {results['local_neighbourhood_stability']['supportive_neighbour_fraction']:.0%}。"
        ),
        "",
        "## 市況分段",
        "",
        "| 市況 | CAGR 差 | Sharpe 差 | 最大回撤差 | Calmar 差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in results["regime_attribution"].items():
        delta = row["delta"]
        lines.append(
            f"| {name} | {delta['cagr']:+.2%} | {delta['sharpe']:+.3f} | "
            f"{delta['max_drawdown']:+.2%} | {delta['calmar']:+.3f} |"
        )
    lines += [
        "",
        "## 偏誤檢查",
        "",
        "| 偏誤 | 狀態 | 證據 |",
        "|---|---|---|",
        "| 前視偏誤 | 無 | 只用當日及以前 NDX 收市價，下一交易日開市成交 |",
        "| 特徵偷看未來 | 無 | RSI 與 MA 都由 `NASDAQ100.csv` 因果計算 |",
        "| 生存者偏誤 | 未能完全核實 | 使用 NDX 指數與廣度彙總資料 |",
        (
            f"| 數據挖掘 | 明顯存在 | 固定 95 組 grid，DSR 累計"
            f"{results['data']['related_trials_including_grid']:,} 次測試 |"
        ),
        "| 交易成本低估 | 已壓力測試 | 1x / 2x / 5x / 10x 成本都已重跑 |",
        "| 流動性偏誤 | 風險較低但未完全核實 | 以高流動性 ETF/指數代理，未加入 ADV 限制 |",
        "| 頻率錯配 | 無 | 日線收市訊號配下一個日線開市成交 |",
        "| 2007 年前合成 breadth | 存在 | 2007+ 真實 breadth 另行匯報 |",
        "| 市況過度擬合 | 已測但不能排除 | 主/反方向、全樣本、成本、鄰近穩定性及分段歸因 |",
        "| 乾淨 forward OOS | 不足 | 2026-07-05 之後觀察期仍太短，不能據此採用 |",
        "",
        "## 預先登記限制條件",
        "",
    ]
    lines.extend(
        f"- {name}：{'通過' if passed else '失敗'}"
        for name, passed in results["guardrails"].items()
    )
    current = results["current_signal"]
    lines += [
        "",
        "## 最新訊號",
        "",
        (
            f"截至 {pd.Timestamp(current['date']).date()}，主選組合 "
            f"RSI({current['short_window']}/{current['long_window']}) + "
            f"MA{current['ma_window']}：短 RSI={current['short_rsi']:.2f}，"
            f"長 RSI={current['long_rsi']:.2f}，NDX={current['ndx_close']:.2f}，"
            f"MA={current['ma_value']:.2f}；"
            f"{'已觸發' if current['selected_signal'] else '未觸發'} 黃金交叉再入場訊號。"
        ),
        "",
        "## 結論",
        "",
        "今輪只屬歷史研究證據，不構成投資建議。",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not IDEA_CARD.exists():
        raise FileNotFoundError("pre-registered idea card is required")

    framework.RELATED_TRIALS = RELATED_TRIALS
    all_data = qbt.load_data()
    df = all_data.loc[all_data.index >= pd.Timestamp(START_DATE)].copy()
    complete_close = common.load_ndx_close_csv()
    ndx_close = complete_close.reindex(df.index)
    rsi_cache = {
        window: crossover.calculate_wilder_rsi(complete_close, window).reindex(df.index)
        for window in sorted(set(SHORT_WINDOWS).union(LONG_WINDOWS))
    }
    ma_cache = {
        window: complete_close.rolling(window, min_periods=window).mean().reindex(df.index)
        for window in MA_WINDOWS
    }
    benchmark = qbt.run_benchmark(df)

    direct = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    parity_run = reentry.run_variant(df, None)
    parity = {
        "equity_max_absolute_difference": float((direct[0] - parity_run[0]).abs().max()),
        "trade_signatures_identical": (
            framework.trade_signature(direct[1]) == framework.trade_signature(parity_run[1])
        ),
        "open_trade_identical": framework.open_trade_equal(direct[2], parity_run[2]),
    }
    parity["passed"] = bool(
        parity["equity_max_absolute_difference"] < 1e-8
        and parity["trade_signatures_identical"]
        and parity["open_trade_identical"]
    )
    if not parity["passed"]:
        raise AssertionError(f"baseline parity failed: {parity}")

    baseline = common.evaluate(df, parity_run, benchmark)
    baseline["score"] = framework.score(
        baseline, framework.efficiency(baseline), True, True
    )
    baseline_segments = {
        "train": segment_metrics(parity_run[0], START_DATE, TRAIN_END),
        "test": segment_metrics(parity_run[0], TEST_START),
        "real_breadth": segment_metrics(parity_run[0], REAL_BREADTH_START),
    }

    grid_rows: list[dict[str, float | int]] = []
    feature_cache: dict[tuple[int, int, int], pd.DataFrame] = {}
    for params in GRID:
        short_window, long_window, ma_window = params
        features = build_features(
            df.index,
            ndx_close,
            rsi_cache[short_window],
            rsi_cache[long_window],
            ma_cache[ma_window],
        )
        feature_cache[params] = features
        grid_rows.append(grid_row(df, features, params))

    grid = rank_columns(pd.DataFrame(grid_rows))
    winners = select_winners(grid)
    grid_diagnostics = {
        "train_combinations_beating_baseline_calmar": int(
            (grid["train_calmar"] > baseline_segments["train"]["calmar"]).sum()
        ),
        "test_combinations_beating_baseline_calmar": int(
            (grid["test_calmar"] > baseline_segments["test"]["calmar"]).sum()
        ),
        "full_combinations_beating_baseline_calmar": int(
            (grid["full_calmar"] > baseline["metrics"]["calmar"]).sum()
        ),
    }

    primary_params = row_params(winners["primary_train_2002_2013"])
    reverse_params = row_params(winners["reverse_train_2014_plus"])
    full_params = row_params(winners["full_history_descriptive"])
    primary_features = feature_cache[primary_params]
    primary_run = run_combo(df, primary_features, primary_params)
    reverse_run = run_combo(df, feature_cache[reverse_params], reverse_params)
    full_run = run_combo(df, feature_cache[full_params], full_params)

    challenger = common.evaluate(df, primary_run, benchmark)
    paired = analytics.paired_hac_and_bootstrap(primary_run[0], parity_run[0])
    stability = local_neighbourhood_stability(grid, winners["primary_train_2002_2013"])
    bootstrap_stable = paired["bootstrap_95_interval_annualized"][0] > 0
    challenger["score"] = framework.score(
        challenger, framework.efficiency(challenger), bootstrap_stable, stability["passed"]
    )

    cost_stress: dict[str, dict[str, Any]] = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost_run = reentry.run_variant(df, None, multiplier)
        challenger_cost_run = run_combo(
            df, primary_features, primary_params, cost_multiplier=multiplier
        )
        baseline_cost = common.evaluate(df, baseline_cost_run, benchmark)
        challenger_cost = common.evaluate(df, challenger_cost_run, benchmark)
        pair = analytics.paired_hac_and_bootstrap(
            challenger_cost_run[0], baseline_cost_run[0]
        )
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": baseline_cost["metrics"]["cagr"],
            "challenger_cagr": challenger_cost["metrics"]["cagr"],
            "cagr_delta": (
                challenger_cost["metrics"]["cagr"] - baseline_cost["metrics"]["cagr"]
            ),
            "baseline_expectancy": baseline_cost["metrics"]["expectancy"],
            "challenger_expectancy": challenger_cost["metrics"]["expectancy"],
            "expectancy_delta": (
                challenger_cost["metrics"]["expectancy"]
                - baseline_cost["metrics"]["expectancy"]
            ),
            "paired_annualized_mean": pair["annualized_mean_difference"],
            "paired_hac_t": pair["hac_t_stat"],
            "paired_hac_p": pair["hac_two_sided_p"],
            "bootstrap_95_interval_annualized": pair[
                "bootstrap_95_interval_annualized"
            ],
        }

    period_deltas = {
        period: common.period_delta(challenger[period], baseline[period])
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    regime_attribution = {}
    for name, (start, end) in common.REGIMES.items():
        base_period = analytics.slice_metrics(parity_run[0], start, end)
        challenge_period = analytics.slice_metrics(primary_run[0], start, end)
        regime_attribution[name] = {
            "baseline": base_period,
            "challenger": challenge_period,
            "delta": common.period_delta(challenge_period, base_period),
        }

    bm, cm = baseline["metrics"], challenger["metrics"]
    guardrails = {
        "baseline_parity": parity["passed"],
        "primary_test_calmar_improved": (
            float(winners["primary_train_2002_2013"]["test_calmar"])
            > float(baseline_segments["test"]["calmar"])
        ),
        "reverse_test_calmar_improved": (
            float(winners["reverse_train_2014_plus"]["train_calmar"])
            > float(baseline_segments["train"]["calmar"])
        ),
        "full_history_calmar_not_worse": cm["calmar"] >= bm["calmar"],
        "full_history_max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "expectancy_not_worse": cm["expectancy"] >= bm["expectancy"],
        "turnover_guardrail": (
            cm["turnover_position_changes_per_year"]
            <= bm["turnover_position_changes_per_year"] * 1.5
            or cm["expectancy"] > bm["expectancy"]
        ),
        "real_breadth_calmar_nonnegative": (
            period_deltas["real_breadth_period"]["calmar"] >= 0
        ),
        "five_x_paired_return_positive": (
            cost_stress["5x"]["paired_annualized_mean"] > 0
        ),
        "local_neighbourhood_stable": stability["passed"],
    }
    decision = "track" if all(guardrails.values()) else "reject"

    all_primary_entries = [*primary_run[1]]
    if primary_run[2]:
        all_primary_entries.append(primary_run[2])
    signal_counts = {
        "raw_selected_signal_days": int(primary_features["selected_signal"].sum()),
        "executed_golden_cross_entries": int(
            sum(
                trade.get("buy_trigger") == "RSI-golden-cross"
                for trade in all_primary_entries
            )
        ),
        "entry_triggers": pd.Series(
            [trade.get("buy_trigger") for trade in all_primary_entries],
            dtype="object",
        ).value_counts().to_dict(),
        "exit_reasons": pd.Series(
            [trade["sell_reason"] for trade in primary_run[1]],
            dtype="object",
        ).value_counts().to_dict(),
    }
    current_signal = {
        "date": df.index[-1],
        "short_window": primary_params[0],
        "long_window": primary_params[1],
        "ma_window": primary_params[2],
        "short_rsi": float(primary_features["short_rsi"].iloc[-1]),
        "long_rsi": float(primary_features["long_rsi"].iloc[-1]),
        "ndx_close": float(primary_features["ndx_close"].iloc[-1]),
        "ma_value": float(primary_features["ma_value"].iloc[-1]),
        "golden_cross": bool(primary_features["golden_cross"].iloc[-1]),
        "above_ma": bool(primary_features["above_ma"].iloc[-1]),
        "selected_signal": bool(primary_features["selected_signal"].iloc[-1]),
        "baseline_position_open": parity_run[2] is not None,
        "challenger_position_open": primary_run[2] is not None,
        "earliest_fill": "next available session open",
    }

    candidate_runs = {
        "primary_train_2002_2013": primary_run,
        "reverse_train_2014_plus": reverse_run,
        "full_history_descriptive": full_run,
    }
    write_artifacts(
        df,
        benchmark,
        parity_run,
        candidate_runs,
        primary_features,
        primary_params,
        grid,
    )

    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "source": "NASDAQ100.csv",
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "clean_forward_start": "2026-07-05",
            "clean_forward_observations": challenger["clean_forward_slice"].get(
                "observations", 0
            ),
            "related_trials_including_grid": RELATED_TRIALS,
            "pre_2007_breadth": "synthetic daily splice",
        },
        "grid": {
            "short_windows": list(SHORT_WINDOWS),
            "long_windows": list(LONG_WINDOWS),
            "ma_windows": list(MA_WINDOWS),
            "valid_rsi_pairs": [list(pair) for pair in RSI_PAIRS],
            "combinations": len(GRID),
            "rsi_smoothing": "Wilder ewm(alpha=1/window, adjust=False)",
            "selection_objective": (
                "maximize Calmar, then Sharpe, then CAGR, then lexicographically "
                "smaller (short, long, ma)"
            ),
        },
        "baseline_parity": parity,
        "baseline_segments": baseline_segments,
        "winners": winners,
        "grid_diagnostics": grid_diagnostics,
        "top_rankings": {
            "primary_train_top10": sort_grid(grid, "train")
            .head(10)
            .to_dict(orient="records"),
            "reverse_train_top10": sort_grid(grid, "test")
            .head(10)
            .to_dict(orient="records"),
            "full_history_top10": sort_grid(grid, "full")
            .head(10)
            .to_dict(orient="records"),
        },
        "local_neighbourhood_stability": stability,
        "benchmark": {
            "name": "NDX price-index buy and hold",
            "metrics": analytics.slice_metrics(benchmark, START_DATE),
        },
        "baseline": baseline,
        "challenger": challenger,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": framework.efficiency(baseline),
            "challenger": framework.efficiency(challenger),
            "interpretation": "historical split pseudo-OOS only; full history is seen",
        },
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "calendar_parity_split": common.harness.calendar_parity_split(
            primary_run[0], parity_run[0]
        ),
        "regime_attribution": regime_attribution,
        "guardrails": guardrails,
        "signal_counts": signal_counts,
        "current_signal": current_signal,
    }

    RESULTS_FILE.write_text(
        json.dumps(framework._jsonable(results), indent=2), encoding="utf-8"
    )
    write_report(results)
    print(
        json.dumps(
            framework._jsonable(
                {
                    "decision": decision,
                    "winners": winners,
                    "local_neighbourhood_stability": stability,
                    "baseline_score": baseline["score"],
                    "challenger_score": challenger["score"],
                    "baseline_metrics": bm,
                    "challenger_metrics": cm,
                    "paired_inference": paired,
                    "guardrails": guardrails,
                    "current_signal": current_signal,
                    "parity": parity,
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
