"""Research challenger: add an NDX RSI(7/14) golden-cross re-entry."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_ndx_rsi14_divergence_confirmation as common
import qqq_ndx_rsi7_14_death_cross_exit as crossover


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/ndx_rsi7_14_golden_cross_reentry_idea.md"
REPORT_FILE = ROOT / "docs/research/ndx_rsi7_14_golden_cross_reentry_report.md"
RESULTS_FILE = ROOT / "qqq_ndx_rsi7_14_golden_cross_reentry_results.json"
EQUITY_FILE = ROOT / "qqq_ndx_rsi7_14_golden_cross_reentry_equity.csv"
TRADES_FILE = ROOT / "qqq_ndx_rsi7_14_golden_cross_reentry_trades.csv"
SIGNALS_FILE = ROOT / "qqq_ndx_rsi7_14_golden_cross_reentry_signals.csv"

START_DATE = "2002-01-01"
LONG_WINDOW = 14
PRIMARY_SHORT_WINDOW = 7
SHORT_WINDOW_SENSITIVITY = (5, 7, 9)
RELATED_TRIALS = 4_822

qbt = common.qbt
framework = common.framework
analytics = common.analytics


def build_features(
    index: pd.DatetimeIndex,
    complete_ndx_close: pd.Series,
    short_window: int,
) -> pd.DataFrame:
    return crossover.build_crossover_features(
        index, complete_ndx_close, short_window
    )


def run_variant(
    df: pd.DataFrame,
    golden_cross: pd.Series | None,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    """Run the canonical strategy with one optional lower-priority entry path."""
    extra_entry = (
        pd.Series(False, index=df.index)
        if golden_cross is None
        else golden_cross.reindex(df.index).fillna(False).astype(bool)
    )
    commission = qbt.COMMISSION * cost_multiplier
    slippage = qbt.SLIPPAGE * cost_multiplier
    position = "OUT"
    eff_entry = raw_entry = 0.0
    entry_date = None
    trade_low = trade_high = 0.0
    macd_age = ext_age = 10**9
    buy_trigger = None
    portfolio = qbt.INITIAL_CAPITAL
    cooldown_until = None
    last_sell_reason = None
    last_exit_price = None
    trades: list[dict] = []
    values: dict[pd.Timestamp, float] = {}
    pending: dict | None = None
    rows = list(df.iterrows())
    n = len(rows)

    def execute_due(i: int, date: pd.Timestamp, fill_price: float) -> bool:
        nonlocal position, eff_entry, raw_entry, entry_date, trade_low, trade_high
        nonlocal macd_age, ext_age, buy_trigger, portfolio, cooldown_until
        nonlocal last_sell_reason, last_exit_price, pending
        if pending is None or pending["fill_at"] != i:
            return False
        if pending["action"] == "BUY" and position == "OUT":
            portfolio -= commission
            eff_entry = fill_price * (1 + slippage)
            raw_entry = fill_price
            entry_date = date
            trade_low = trade_high = fill_price
            macd_age = ext_age = 10**9
            buy_trigger = pending["trigger"]
            position = "IN"
            pending = None
            return True
        if pending["action"] == "SELL" and position == "IN":
            eff_exit = fill_price * (1 - slippage)
            gross_return = (eff_exit - eff_entry) / eff_entry
            portfolio *= 1 + gross_return
            portfolio -= commission
            cooldown_until = date + pd.Timedelta(days=qbt.COOLDOWN_DAYS)
            last_sell_reason = pending["reason"]
            last_exit_price = fill_price
            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": raw_entry,
                    "exit_price": fill_price,
                    "return_pct": gross_return * 100,
                    "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
                    "accumulated": portfolio,
                    "buy_trigger": buy_trigger,
                    "sell_reason": pending["reason"],
                    "cooldown_until": cooldown_until,
                }
            )
            position = "OUT"
            pending = None
            return True
        pending = None
        return False

    for i, (date, row) in enumerate(rows):
        price = float(row["price"])
        breadth = float(row["breadth"])
        fill_price = (
            float(row["open"])
            if qbt.FILL_PRICE == "open" and not pd.isna(row["open"])
            else price
        )
        executed = execute_due(i, date, fill_price)

        if not executed and pending is None:
            if position == "OUT":
                cooldown_ok = cooldown_until is None or date > cooldown_until
                washout = (
                    breadth < qbt.BUY_B200_THRESH
                    and bool(row["vote_gate"])
                )
                recross_ok = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and price > last_exit_price
                )
                trend = bool(row["ma200_recross"]) and recross_ok
                golden = (
                    last_exit_price is not None
                    and bool(extra_entry.loc[date])
                )
                if (
                    cooldown_ok
                    and (washout or trend or golden)
                    and i + qbt.EXECUTION_LAG < n
                ):
                    if washout:
                        trigger = (
                            ("VIX" if row["vix_vote"] else "")
                            + ("+" if row["vix_vote"] and row["ma200_vote"] else "")
                            + ("MA200" if row["ma200_vote"] else "")
                        )
                    elif trend:
                        trigger = "MA200-recross"
                    else:
                        trigger = "RSI-golden-cross"
                    pending = {
                        "action": "BUY",
                        "fill_at": i + qbt.EXECUTION_LAG,
                        "trigger": trigger,
                    }
            else:
                trade_low = min(trade_low, price)
                trade_high = max(trade_high, price)
                macd_age = 0 if bool(row["macd_cross"]) else macd_age + 1
                ext_age = 0 if bool(row["ext10"]) else ext_age + 1
                bearish = (
                    bool(row["price_rose"])
                    and bool(row["breadth_fell"])
                    and breadth < qbt.DIVERGENCE_BREADTH_CAP
                )
                climax = (
                    macd_age < qbt.CLIMAX_VOTE_WINDOW
                    and ext_age < qbt.CLIMAX_VOTE_WINDOW
                )
                trailing = price <= trade_high * (1 - qbt.TRAILING_STOP_PCT / 100)
                reason = (
                    "bearish-divergence" if bearish
                    else "climax-top" if climax
                    else "trailing-stop" if trailing
                    else None
                )
                if reason and i + qbt.EXECUTION_LAG < n:
                    pending = {
                        "action": "SELL",
                        "fill_at": i + qbt.EXECUTION_LAG,
                        "reason": reason,
                    }
            execute_due(i, date, fill_price)

        values[date] = (
            portfolio * (price * (1 - slippage) / eff_entry)
            if position == "IN"
            else portfolio
        )

    open_trade = None
    if position == "IN":
        last_date = df.index[-1]
        last_price = float(df["price"].iloc[-1])
        eff_last = last_price * (1 - slippage)
        open_trade = {
            "entry_date": entry_date,
            "entry_price": raw_entry,
            "current_date": last_date,
            "current_price": last_price,
            "return_pct": (eff_last - eff_entry) / eff_entry * 100,
            "max_drawdown_pct": (trade_low - raw_entry) / raw_entry * 100,
            "accumulated": portfolio * eff_last / eff_entry,
            "buy_trigger": buy_trigger,
        }
    return pd.Series(values, name="strategy"), trades, open_trade


def sensitivity_is_stable(
    _baseline_calmar: float,
    evaluations: dict[str, dict[str, Any]],
) -> bool:
    calmars = np.asarray(
        [evaluations[f"rsi_{window}_14"]["metrics"]["calmar"]
         for window in SHORT_WINDOW_SENSITIVITY],
        dtype=float,
    )
    if not np.isfinite(calmars).all():
        return False
    primary = calmars[SHORT_WINDOW_SENSITIVITY.index(PRIMARY_SHORT_WINDOW)]
    return bool((np.abs(calmars / primary - 1) <= 0.25).all())


def write_artifacts(
    df: pd.DataFrame,
    benchmark: pd.Series,
    features: dict[str, pd.DataFrame],
    signals: dict[str, pd.Series],
    baseline_run: tuple[pd.Series, list[dict], dict | None],
    runs: dict[str, tuple[pd.Series, list[dict], dict | None]],
) -> None:
    equity = pd.DataFrame(
        {
            "ndx_buy_hold": benchmark,
            "baseline": baseline_run[0],
            **{name: run[0] for name, run in runs.items()},
        }
    )
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)

    rows: list[dict] = []
    for name, run in {"baseline": baseline_run, **runs}.items():
        rows.extend({"variant": name, **trade} for trade in run[1])
    pd.DataFrame(rows).to_csv(TRADES_FILE, index=False)

    primary = features[f"rsi_{PRIMARY_SHORT_WINDOW}_14"].copy()
    for name, signal in signals.items():
        primary[f"golden_cross_{name.removeprefix('rsi_')}"] = signal
    primary["price"] = df["price"]
    primary["next_session_open"] = df["open"].shift(-1)
    primary.index.name = "Date"
    primary.to_csv(SIGNALS_FILE)


def main() -> None:
    if not IDEA_CARD.exists():
        raise FileNotFoundError("pre-registered idea card is required")

    framework.RELATED_TRIALS = RELATED_TRIALS
    all_data = qbt.load_data()
    df = all_data.loc[all_data.index >= pd.Timestamp(START_DATE)].copy()
    complete_close = common.load_ndx_close_csv()
    benchmark = qbt.run_benchmark(df)
    features = {
        f"rsi_{window}_14": build_features(df.index, complete_close, window)
        for window in SHORT_WINDOW_SENSITIVITY
    }
    signals = {
        name: frame[f"golden_cross_{window}_14"].astype(bool)
        for (name, frame), window in zip(
            features.items(), SHORT_WINDOW_SENSITIVITY, strict=True
        )
    }

    direct = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    parity_run = run_variant(df, None)
    parity = {
        "equity_max_absolute_difference": float(
            (direct[0] - parity_run[0]).abs().max()
        ),
        "trade_signatures_identical": (
            framework.trade_signature(direct[1])
            == framework.trade_signature(parity_run[1])
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
    runs = {name: run_variant(df, signal) for name, signal in signals.items()}
    evaluations = {
        name: common.evaluate(df, run, benchmark) for name, run in runs.items()
    }
    primary_name = f"rsi_{PRIMARY_SHORT_WINDOW}_14"
    primary_run = runs[primary_name]
    primary = evaluations[primary_name]
    paired = analytics.paired_hac_and_bootstrap(primary_run[0], parity_run[0])

    stable = sensitivity_is_stable(baseline["metrics"]["calmar"], evaluations)
    bootstrap_stable = paired["bootstrap_95_interval_annualized"][0] > 0
    baseline["score"] = framework.score(
        baseline, framework.efficiency(baseline), True, True
    )
    primary["score"] = framework.score(
        primary, framework.efficiency(primary), bootstrap_stable, stable
    )

    sensitivity = {
        name: {
            key: evaluation["metrics"][key]
            for key in (
                "cagr", "sharpe", "calmar", "max_drawdown",
                "completed_trades", "expectancy", "exposure",
                "turnover_position_changes_per_year",
            )
        }
        for name, evaluation in evaluations.items()
    }

    cost_stress: dict[str, dict[str, float]] = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost_run = run_variant(df, None, multiplier)
        challenger_cost_run = run_variant(df, signals[primary_name], multiplier)
        baseline_cost = common.evaluate(df, baseline_cost_run, benchmark)
        challenger_cost = common.evaluate(df, challenger_cost_run, benchmark)
        pair = analytics.paired_hac_and_bootstrap(
            challenger_cost_run[0], baseline_cost_run[0]
        )
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": baseline_cost["metrics"]["cagr"],
            "challenger_cagr": challenger_cost["metrics"]["cagr"],
            "cagr_delta": (
                challenger_cost["metrics"]["cagr"]
                - baseline_cost["metrics"]["cagr"]
            ),
            "paired_annualized_mean": pair["annualized_mean_difference"],
            "paired_hac_t": pair["hac_t_stat"],
        }

    period_deltas = {
        period: common.period_delta(primary[period], baseline[period])
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

    bm, cm = baseline["metrics"], primary["metrics"]
    guardrails = {
        "baseline_parity": parity["passed"],
        "primary_calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "cagr_within_two_points": cm["cagr"] >= bm["cagr"] - 0.02,
        "expectancy_not_worse": cm["expectancy"] >= bm["expectancy"],
        "turnover_guardrail": (
            cm["turnover_position_changes_per_year"]
            <= bm["turnover_position_changes_per_year"] * 1.5
            or cm["expectancy"] > bm["expectancy"]
        ),
        "historical_halves_calmar_nonnegative": all(
            period_deltas[period]["calmar"] >= 0
            for period in ("early_period", "late_period")
        ),
        "real_breadth_calmar_nonnegative": (
            period_deltas["real_breadth_period"]["calmar"] >= 0
        ),
        "five_x_paired_return_positive": (
            cost_stress["5x"]["paired_annualized_mean"] > 0
        ),
        "sensitivity_not_cliff_edge": stable,
    }
    decision = "track" if all(guardrails.values()) else "reject"

    primary_features = features[primary_name]
    primary_signal = signals[primary_name]
    all_entries = [*primary_run[1]]
    if primary_run[2]:
        all_entries.append(primary_run[2])
    golden_entries = [
        trade for trade in all_entries
        if trade.get("buy_trigger") == "RSI-golden-cross"
    ]
    current = {
        "date": df.index[-1],
        "short_rsi_7": float(primary_features["rsi_7"].iloc[-1]),
        "long_rsi_14": float(primary_features["rsi_14"].iloc[-1]),
        "golden_cross": bool(primary_signal.iloc[-1]),
        "death_cross": bool(primary_features["death_cross_7_14"].iloc[-1]),
        "baseline_position_open": parity_run[2] is not None,
        "challenger_position_open": primary_run[2] is not None,
        "earliest_fill": "next available session open",
    }

    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "source": "NASDAQ100.csv",
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "clean_forward_start": "2026-07-05",
            "clean_forward_observations": primary["clean_forward_slice"].get("observations", 0),
            "related_prior_trials": RELATED_TRIALS,
            "pre_2007_breadth": "synthetic daily splice",
        },
        "configuration": {
            "long_rsi_window": LONG_WINDOW,
            "primary_short_rsi_window": PRIMARY_SHORT_WINDOW,
            "short_window_sensitivity": list(SHORT_WINDOW_SENSITIVITY),
            "rsi_smoothing": "Wilder ewm(alpha=1/window, adjust=False)",
            "signal": "RSI7 crosses from <= RSI14 to > RSI14",
            "change": "add lower-priority entry only while flat after cooldown",
            "fill": "next-session open",
            "commission": qbt.COMMISSION,
            "slippage_per_side": qbt.SLIPPAGE,
        },
        "baseline_parity": parity,
        "benchmark": {
            "name": "NDX price-index buy and hold",
            "metrics": analytics.slice_metrics(benchmark, START_DATE),
        },
        "baseline": baseline,
        "challenger": primary,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": framework.efficiency(baseline),
            "challenger": framework.efficiency(primary),
            "interpretation": "fixed-rule historical-half pseudo-OOS only",
        },
        "sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "calendar_parity_split": common.harness.calendar_parity_split(
            primary_run[0], parity_run[0]
        ),
        "regime_attribution": regime_attribution,
        "guardrails": guardrails,
        "signal_counts": {
            "raw_golden_cross_days": int(primary_signal.sum()),
            "executed_golden_cross_entries": len(golden_entries),
            "all_executed_entry_triggers": pd.Series(
                [trade.get("buy_trigger") for trade in all_entries]
            ).value_counts().to_dict(),
            "all_executed_exit_reasons": pd.Series(
                [trade["sell_reason"] for trade in primary_run[1]]
            ).value_counts().to_dict(),
        },
        "current_signal": current,
    }

    write_artifacts(df, benchmark, features, signals, parity_run, runs)
    RESULTS_FILE.write_text(
        json.dumps(framework._jsonable(results), indent=2), encoding="utf-8"
    )
    write_report(results)
    print(json.dumps(framework._jsonable({
        "decision": decision,
        "baseline_score": baseline["score"],
        "challenger_score": primary["score"],
        "baseline_metrics": bm,
        "challenger_metrics": cm,
        "paired_inference": paired,
        "guardrails": guardrails,
        "signal_counts": results["signal_counts"],
        "current_signal": current,
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    baseline, challenger = results["baseline"], results["challenger"]
    bm, cm = baseline["metrics"], challenger["metrics"]
    bs, cs = baseline["score"], challenger["score"]
    paired = results["paired_inference"]
    verdict = "拒絕採用" if results["decision"] == "reject" else "保留作研究追蹤"
    lines = [
        "# 回測驗證報告 — NDX RSI(7/14) 黃金交叉再入場",
        "",
        f"## 研究決定：{verdict}",
        "",
        "## 摘要",
        "",
        f"黃金交叉挑戰策略評分 **{cs['final_score']} / 100（{cs['band']}）**，"
        f"原有策略 **{bs['final_score']} / 100（{bs['band']}）**。研究決定依照預先登記的 "
        "Calmar 目標及全部限制條件，而非單一回報指標。",
        "今輪只增加空倉後的黃金交叉入口；上輪失敗的死亡交叉賣出沒有加入。",
        "",
        "## 回測評分",
        "",
        "| 組成 | 原有策略 | 挑戰策略 | 滿分 |",
        "|---|---:|---:|---:|",
        f"| A. 統計有效性 | {bs['A_statistical_validity']} | {cs['A_statistical_validity']} | 30 |",
        f"| B. 風險調整表現 | {bs['B_risk_adjusted_performance']} | {cs['B_risk_adjusted_performance']} | 25 |",
        f"| C. 穩健性／OOS | {bs['C_robustness_oos']} | {cs['C_robustness_oos']} | 25 |",
        f"| D. 交易質素／一致性 | {bs['D_trade_quality_consistency']} | {cs['D_trade_quality_consistency']} | 20 |",
        f"| 原始總分 | {bs['raw_score']} | {cs['raw_score']} | 100 |",
        f"| 硬性上限 | {bs['hard_cap']} | {cs['hard_cap']} | |",
        f"| **最終分數** | **{bs['final_score']}** | **{cs['final_score']}** | **100** |",
        "",
        "## 表現、風險與交易",
        "",
        "| 指標 | 原有策略 | 挑戰策略 | 差異 |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = [
        ("CAGR", "cagr", ".2%"), ("波動率", "annual_volatility", ".2%"),
        ("Sharpe", "sharpe", ".3f"), ("Sortino", "sortino", ".3f"),
        ("Calmar", "calmar", ".3f"), ("最大回撤", "max_drawdown", ".2%"),
        ("Ulcer Index", "ulcer_index", ".2%"), ("水底時間", "time_underwater", ".2%"),
        ("勝率", "win_rate", ".2%"), ("盈虧比", "payoff_ratio", ".2f"),
        ("Profit factor", "profit_factor", ".2f"), ("每宗期望", "expectancy", ".2%"),
        ("持倉比例", "exposure", ".2%"),
        ("每年倉位轉換", "turnover_position_changes_per_year", ".2f"),
    ]
    for label, key, fmt in metric_rows:
        lines.append(
            f"| {label} | {format(bm[key], fmt)} | {format(cm[key], fmt)} | "
            f"{format(cm[key] - bm[key], '+' + fmt)} |"
        )
    lines += [
        f"| 完成交易 | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        "",
        "### 尾部風險、回撤及基準比較",
        "",
        "| 指標 | 原有策略 | 挑戰策略 |",
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
        f"- 以 {results['data']['related_prior_trials']:,} 次相關測試計算 DSR：{baseline['statistical_diagnostics']['deflated_sharpe_probability']:.4f} / {challenger['statistical_diagnostics']['deflated_sharpe_probability']:.4f}。",
        f"- 偏度：{bm['skewness']:.3f} / {cm['skewness']:.3f}；超額峰度：{bm['excess_kurtosis']:.3f} / {cm['excess_kurtosis']:.3f}。",
        f"- Jarque-Bera p：{bm['jarque_bera_p']:.3g} / {cm['jarque_bera_p']:.3g}；Ljung-Box p：{baseline['statistical_diagnostics']['ljung_box_p']:.3g} / {challenger['statistical_diagnostics']['ljung_box_p']:.3g}。",
        f"- 回報 ADF p：{bm['adf_return_p']:.3g} / {cm['adf_return_p']:.3g}。",
        f"- 配對年化平均差：{paired['annualized_mean_difference']:+.2%}；HAC t={paired['hac_t_stat']:.3f}，p={paired['hac_two_sided_p']:.3f}。",
        f"- 區塊 bootstrap 95% 區間：{paired['bootstrap_95_interval_annualized']}。",
        "",
        "## 穩健性",
        "",
        f"- 歷史兩半效率：{results['wfa_efficiency']['baseline']:.3f} / {results['wfa_efficiency']['challenger']:.3f}（pseudo-OOS）。",
        "- 短線 RSI 週期敏感度 Calmar：" + ", ".join(
            f"{name}={value['calmar']:.3f}" for name, value in results["sensitivity"].items()
        ) + "。",
        f"- 奇數／偶數年份配對年化差：{results['calendar_parity_split']['odd_years']['annualized_mean_difference']:+.2%} / {results['calendar_parity_split']['even_years']['annualized_mean_difference']:+.2%}。",
        f"- 交易 bootstrap：{challenger['trade_bootstrap'].get('simulations', 0):,} 次。",
        "- 挑戰策略交易 bootstrap 終值回報分位：" + ", ".join(
            f"{name}={value:.1%}"
            for name, value in challenger["trade_bootstrap"]["terminal_return_percentiles"].items()
        ) + "。",
        "- 挑戰策略交易 bootstrap 最大回撤分位：" + ", ".join(
            f"{name}={value:.1%}"
            for name, value in challenger["trade_bootstrap"]["max_drawdown_percentiles"].items()
        ) + "。",
        f"- 5x／10x 成本配對差：{results['cost_stress']['5x']['paired_annualized_mean']:+.2%} / {results['cost_stress']['10x']['paired_annualized_mean']:+.2%}。",
        f"- 乾淨 forward OOS 只有 {results['data']['clean_forward_observations']} 個交易日。",
        "",
        "### 市況分段歸因",
        "",
        "| 市況 | CAGR 差 | Sharpe 差 | 最大回撤差 | Calmar 差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in results["regime_attribution"].items():
        delta = values["delta"]
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
        "| 前視偏誤 | 無 | 當日收市確認交叉，下一交易日開市買入 |",
        "| 特徵偷看未來 | 無 | RSI 只使用當日及過往 NDX 收市價 |",
        "| 生存者偏誤 | 未能完全核實 | 使用 NDX 指數及整體廣度 |",
        f"| 數據挖掘 | 明顯存在 | DSR 計入 {results['data']['related_prior_trials']:,} 次相關測試 |",
        "| 交易成本 | 已計算 | 每邊 $1 加 0.05%，另測 2x／5x／10x |",
        "| 流動性偏誤 | 風險較低但未完全核實 | 指數代理沒有倉位相對 ADV 模型 |",
        "| 頻率錯配 | 無 | 每日收市訊號配下一個每日開市價 |",
        "| 合成數據 | 2007 年前存在 | 2007 年後真實廣度另行匯報 |",
        "| 市況過度擬合 | 已測但不能排除 | 兩半、奇偶年份、市況分段及敏感度 |",
        "| 乾淨 OOS | 不足 | 凍結後時間及完成交易不足 |",
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
        f"截至 {pd.Timestamp(current['date']).date()}，RSI(7)={current['short_rsi_7']:.2f}，"
        f"RSI(14)={current['long_rsi_14']:.2f}；黃金交叉"
        f"{'已出現' if current['golden_cross'] else '未出現'}。",
        "",
        "## 主要風險",
        "",
        "1. 黃金交叉可能在熊市反彈中過早重新入場，削弱原策略耐心等待洗倉的優勢。",
        "2. 過往相關測試數量龐大，歷史改善容易受選擇偏誤影響。",
        "3. 2007 年前廣度屬合成數據，凍結後 forward OOS 仍不足。",
        "",
        "## 決定",
        "",
        "依照預先登記的 Calmar 目標及全部限制條件決定，凍結基準不會自動更改。",
        "",
        "以上只屬研究證據，並非投資建議。",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
