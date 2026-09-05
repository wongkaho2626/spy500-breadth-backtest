"""Research challenger: confirm QQQ bearish-divergence exits with NDX RSI(14).

The canonical strategy is not modified.  The challenger changes only the
bearish-divergence exit, requiring Wilder RSI(14) calculated from
``NASDAQ100.csv`` closes to be at or below a fixed threshold.  Signals are
observed at the close and filled at the next session open.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

import qqq_breadth_only_divergence_exit as harness


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/ndx_rsi14_divergence_confirmation_idea.md"
REPORT_FILE = ROOT / "docs/research/ndx_rsi14_divergence_confirmation_report.md"
RESULTS_FILE = ROOT / "qqq_ndx_rsi14_divergence_confirmation_results.json"
EQUITY_FILE = ROOT / "qqq_ndx_rsi14_divergence_confirmation_equity.csv"
TRADES_FILE = ROOT / "qqq_ndx_rsi14_divergence_confirmation_trades.csv"
SIGNALS_FILE = ROOT / "qqq_ndx_rsi14_divergence_confirmation_signals.csv"

START_DATE = "2002-01-01"
RSI_WINDOW = 14
PRIMARY_THRESHOLD = 50.0
SENSITIVITY = (45.0, 50.0, 55.0)
RELATED_TRIALS = 4_820
REGIMES = {
    "synthetic_breadth_2002_2006": ("2002-01-01", "2006-12-31"),
    "gfc_and_recovery_2007_2013": ("2007-01-01", "2013-12-31"),
    "pre_pandemic_2014_2019": ("2014-01-01", "2019-12-31"),
    "pandemic_and_inflation_2020_2022": ("2020-01-01", "2022-12-31"),
    "recent_2023_present": ("2023-01-01", None),
}

qbt = harness.qbt
framework = harness.framework
analytics = harness.analytics


def load_ndx_close_csv() -> pd.Series:
    """Load the complete NASDAQ100.csv close history for RSI warm-up."""
    raw = pd.read_csv(ROOT / "NASDAQ100.csv", encoding="utf-8-sig")
    raw.columns = [column.strip().strip('"').lstrip("﻿") for column in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    close = (
        raw.set_index("Date")["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
        .sort_index()
    )
    return close.rename("ndx_close")


def calculate_rsi14(close: pd.Series) -> pd.Series:
    """Return causal Wilder RSI(14), using closes through each row only."""
    delta = close.astype(float).diff()
    average_gain = delta.clip(lower=0).ewm(
        alpha=1 / RSI_WINDOW, adjust=False, min_periods=RSI_WINDOW
    ).mean()
    average_loss = (-delta.clip(upper=0)).ewm(
        alpha=1 / RSI_WINDOW, adjust=False, min_periods=RSI_WINDOW
    ).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_gain == 0) & (average_loss == 0), 50.0)
    return rsi.rename("ndx_rsi14")


def build_features(
    df: pd.DataFrame,
    complete_ndx_close: pd.Series | None = None,
) -> pd.DataFrame:
    canonical = (
        df["price_rose"].astype(bool)
        & df["breadth_fell"].astype(bool)
        & (df["breadth"] < qbt.DIVERGENCE_BREADTH_CAP)
    )
    rsi_source = df["price"] if complete_ndx_close is None else complete_ndx_close
    rsi = calculate_rsi14(rsi_source).reindex(df.index)
    return pd.DataFrame(
        {
            "price": df["price"],
            "breadth": df["breadth"],
            "canonical_divergence": canonical,
            "ndx_rsi14": rsi,
            "next_session_open": df["open"].shift(-1),
        },
        index=df.index,
    )


def confirmation_signal(features: pd.DataFrame, threshold: float) -> pd.Series:
    return (
        features["canonical_divergence"].astype(bool)
        & (features["ndx_rsi14"] <= threshold)
    ).fillna(False).rename(f"rsi_le_{threshold:.0f}")


def run_variant(
    df: pd.DataFrame,
    signal: pd.Series | None,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    equity, trades, open_trade = harness.run_variant(
        df, signal, cost_multiplier
    )
    rsi = calculate_rsi14(load_ndx_close_csv()).reindex(df.index)
    labelled: list[dict] = []
    for source in trades:
        trade = dict(source)
        if signal is not None and trade["sell_reason"] == "breadth-only-60d-fall":
            trade["sell_reason"] = "rsi14-confirmed-divergence"
            trade["ndx_rsi14_on_signal"] = float(rsi.loc[trade["signal_date"]])
        labelled.append(trade)
    return equity, labelled, dict(open_trade) if open_trade else None


def drawdown_diagnostics(equity: pd.Series) -> dict[str, Any]:
    drawdown = equity / equity.cummax() - 1
    episodes: list[tuple[float, int]] = []
    active: list[float] = []
    for value in drawdown:
        if value < 0:
            active.append(float(value))
        elif active:
            episodes.append((min(active), len(active)))
            active = []
    if active:
        episodes.append((min(active), len(active)))
    ulcer = float(np.sqrt(np.mean(np.square(drawdown))))
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)
    return {
        "average_episode_drawdown": float(np.mean([x[0] for x in episodes])),
        "average_recovery_sessions": float(np.mean([x[1] for x in episodes])),
        "maximum_recovery_sessions": int(max(x[1] for x in episodes)),
        "drawdown_episodes": len(episodes),
        "pain_ratio": cagr / ulcer if ulcer > 0 else None,
    }


def supplemental_metrics(
    equity: pd.Series,
    position: pd.Series,
    benchmark: pd.Series,
) -> dict[str, Any]:
    returns = equity.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().reindex(returns.index).dropna()
    returns = returns.reindex(benchmark_returns.index)
    losses = returns[returns < 0]
    gains = returns[returns > 0]
    difference = returns - benchmark_returns
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    beta = float(returns.cov(benchmark_returns) / benchmark_returns.var())
    alpha = float((returns.mean() - beta * benchmark_returns.mean()) * 252)
    tracking_error = float(difference.std(ddof=1) * np.sqrt(252))
    information_ratio = (
        float(difference.mean() * 252 / tracking_error)
        if tracking_error > 0 else None
    )
    adf = adfuller(returns, autolag="AIC")
    metrics = {
        "omega_zero": float(gains.sum() / abs(losses.sum())),
        "var_99_daily": float(returns.quantile(0.01)),
        "cvar_99_daily": float(returns[returns <= returns.quantile(0.01)].mean()),
        "positive_quarters": float(
            (equity.resample("QE").last().pct_change().dropna() > 0).mean()
        ),
        "turnover_position_changes_per_year": float(
            position.astype(int).diff().abs().fillna(position.iloc[0]).sum() / years
        ),
        "benchmark_alpha_annualized": alpha,
        "benchmark_beta": beta,
        "benchmark_correlation": float(returns.corr(benchmark_returns)),
        "tracking_error": tracking_error,
        "information_ratio": information_ratio,
        "adf_return_statistic": float(adf[0]),
        "adf_return_p": float(adf[1]),
    }
    metrics.update(drawdown_diagnostics(equity))
    return metrics


def evaluate(
    df: pd.DataFrame,
    run: tuple[pd.Series, list[dict], dict | None],
    benchmark: pd.Series,
) -> dict[str, Any]:
    evaluation = framework.evaluate(df, *run)
    position = analytics.position_series(df.index, run[1], run[2])
    evaluation["metrics"].update(
        supplemental_metrics(run[0], position, benchmark)
    )
    return evaluation


def period_delta(challenger: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return harness.period_delta(challenger, baseline)


def sensitivity_is_stable(
    baseline_calmar: float,
    evaluations: dict[str, dict[str, Any]],
) -> bool:
    calmars = np.asarray(
        [evaluations[f"rsi_le_{x:.0f}"]["metrics"]["calmar"] for x in SENSITIVITY],
        dtype=float,
    )
    if not np.isfinite(calmars).all() or not (calmars > baseline_calmar).all():
        return False
    primary = calmars[SENSITIVITY.index(PRIMARY_THRESHOLD)]
    return bool((np.abs(calmars / primary - 1) <= 0.25).all())


def write_artifacts(
    df: pd.DataFrame,
    features: pd.DataFrame,
    benchmark: pd.Series,
    baseline_run: tuple[pd.Series, list[dict], dict | None],
    runs: dict[str, tuple[pd.Series, list[dict], dict | None]],
    signals: dict[str, pd.Series],
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

    signal_frame = features.copy()
    for name, signal in signals.items():
        signal_frame[name] = signal
    signal_frame.index.name = "Date"
    signal_frame.to_csv(SIGNALS_FILE)


def main() -> None:
    if not IDEA_CARD.exists():
        raise FileNotFoundError("pre-registered idea card is required")

    framework.RELATED_TRIALS = RELATED_TRIALS
    all_data = qbt.load_data()
    df = all_data.loc[all_data.index >= pd.Timestamp(START_DATE)].copy()
    benchmark = qbt.run_benchmark(df)
    features = build_features(df, load_ndx_close_csv())
    signals = {
        f"rsi_le_{threshold:.0f}": confirmation_signal(features, threshold)
        for threshold in SENSITIVITY
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

    baseline = evaluate(df, parity_run, benchmark)
    runs = {name: run_variant(df, signal) for name, signal in signals.items()}
    evaluations = {
        name: evaluate(df, run, benchmark) for name, run in runs.items()
    }
    primary_name = f"rsi_le_{PRIMARY_THRESHOLD:.0f}"
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
            )
        }
        for name, evaluation in evaluations.items()
    }

    cost_stress: dict[str, dict[str, float]] = {}
    for multiplier in (1, 2, 5, 10):
        base_cost_run = run_variant(df, None, multiplier)
        challenger_cost_run = run_variant(df, signals[primary_name], multiplier)
        base_cost = evaluate(df, base_cost_run, benchmark)
        challenger_cost = evaluate(df, challenger_cost_run, benchmark)
        pair = analytics.paired_hac_and_bootstrap(
            challenger_cost_run[0], base_cost_run[0]
        )
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": base_cost["metrics"]["cagr"],
            "challenger_cagr": challenger_cost["metrics"]["cagr"],
            "cagr_delta": challenger_cost["metrics"]["cagr"] - base_cost["metrics"]["cagr"],
            "paired_annualized_mean": pair["annualized_mean_difference"],
            "paired_hac_t": pair["hac_t_stat"],
        }

    period_deltas = {
        period: period_delta(primary[period], baseline[period])
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    regime_attribution = {}
    for name, (start, end) in REGIMES.items():
        base_period = analytics.slice_metrics(parity_run[0], start, end)
        challenger_period = analytics.slice_metrics(primary_run[0], start, end)
        regime_attribution[name] = {
            "baseline": base_period,
            "challenger": challenger_period,
            "delta": period_delta(challenger_period, base_period),
        }

    bm, cm = baseline["metrics"], primary["metrics"]
    guardrails = {
        "baseline_parity": parity["passed"],
        "primary_calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "cagr_within_two_points": cm["cagr"] >= bm["cagr"] - 0.02,
        "expectancy_not_worse": cm["expectancy"] >= bm["expectancy"],
        "turnover_not_increased": (
            cm["turnover_position_changes_per_year"]
            <= bm["turnover_position_changes_per_year"]
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

    confirmed_exits = [
        trade for trade in primary_run[1]
        if trade["sell_reason"] == "rsi14-confirmed-divergence"
    ]
    primary_signal = signals[primary_name]
    current = {
        "date": df.index[-1],
        "ndx_close": float(features["price"].iloc[-1]),
        "ndx_rsi14": float(features["ndx_rsi14"].iloc[-1]),
        "canonical_divergence": bool(features["canonical_divergence"].iloc[-1]),
        "rsi_confirmation": bool(features["ndx_rsi14"].iloc[-1] <= PRIMARY_THRESHOLD),
        "active": bool(primary_signal.iloc[-1]),
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
            "rsi_window": RSI_WINDOW,
            "rsi_smoothing": "Wilder ewm(alpha=1/14, adjust=False)",
            "primary_threshold": PRIMARY_THRESHOLD,
            "sensitivity_thresholds": list(SENSITIVITY),
            "fill": "next-session open",
            "commission": qbt.COMMISSION,
            "slippage_per_side": qbt.SLIPPAGE,
            "change": "confirm canonical bearish-divergence exit only",
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
        "calendar_parity_split": harness.calendar_parity_split(
            primary_run[0], parity_run[0]
        ),
        "regime_attribution": regime_attribution,
        "guardrails": guardrails,
        "signal_counts": {
            "canonical_divergence_active_days": int(features["canonical_divergence"].sum()),
            "confirmed_active_days": int(primary_signal.sum()),
            "executed_confirmed_divergence_exits": len(confirmed_exits),
            "all_executed_exit_reasons": pd.Series(
                [trade["sell_reason"] for trade in primary_run[1]]
            ).value_counts().to_dict(),
        },
        "current_signal": current,
    }

    write_artifacts(df, features, benchmark, parity_run, runs, signals)
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
    verdict = "Reject" if results["decision"] == "reject" else "Track as research challenger"
    paired = results["paired_inference"]
    lines = [
        "# Backtest Verification Report — NDX RSI(14) divergence confirmation",
        "",
        f"## Verdict: {verdict}",
        "",
        "## Executive Summary",
        "",
        f"The RSI-confirmed challenger scores **{cs['final_score']} / 100 ({cs['band']})** "
        f"versus baseline **{bs['final_score']} / 100 ({bs['band']})**. The decision follows "
        "the pre-registered Calmar objective and guardrails; historical splits are pseudo-OOS, "
        "not untouched forward evidence.",
        "",
        "## Backtest Scores",
        "",
        "| Component | Baseline | Challenger | Max |",
        "|---|---:|---:|---:|",
        f"| A. Statistical validity | {bs['A_statistical_validity']} | {cs['A_statistical_validity']} | 30 |",
        f"| B. Risk-adjusted performance | {bs['B_risk_adjusted_performance']} | {cs['B_risk_adjusted_performance']} | 25 |",
        f"| C. Robustness / OOS | {bs['C_robustness_oos']} | {cs['C_robustness_oos']} | 25 |",
        f"| D. Trade quality / consistency | {bs['D_trade_quality_consistency']} | {cs['D_trade_quality_consistency']} | 20 |",
        f"| **Raw total** | **{bs['raw_score']}** | **{cs['raw_score']}** | **100** |",
        f"| Hard cap | {bs['hard_cap']} | {cs['hard_cap']} | |",
        f"| **Final score** | **{bs['final_score']}** | **{cs['final_score']}** | **100** |",
        "",
        "## Performance and risk",
        "",
        "| Metric | Baseline | Challenger | Delta |",
        "|---|---:|---:|---:|",
    ]
    metric_rows = [
        ("CAGR", "cagr", ".2%"), ("Volatility", "annual_volatility", ".2%"),
        ("Sharpe", "sharpe", ".3f"), ("Sortino", "sortino", ".3f"),
        ("Calmar", "calmar", ".3f"), ("Maximum drawdown", "max_drawdown", ".2%"),
        ("Ulcer Index", "ulcer_index", ".2%"), ("Time underwater", "time_underwater", ".2%"),
        ("Win rate", "win_rate", ".2%"), ("Payoff ratio", "payoff_ratio", ".2f"),
        ("Profit factor", "profit_factor", ".2f"), ("Expectancy", "expectancy", ".2%"),
        ("Exposure", "exposure", ".2%"),
        ("Turnover changes/year", "turnover_position_changes_per_year", ".2f"),
    ]
    for label, key, fmt in metric_rows:
        lines.append(
            f"| {label} | {format(bm[key], fmt)} | {format(cm[key], fmt)} | "
            f"{format(cm[key] - bm[key], '+' + fmt)} |"
        )
    lines += [
        f"| Completed trades | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        "",
        "### Tail, drawdown, and benchmark diagnostics",
        "",
        "| Metric | Baseline | Challenger |",
        "|---|---:|---:|",
        f"| Omega (0% threshold) | {bm['omega_zero']:.3f} | {cm['omega_zero']:.3f} |",
        f"| Daily VaR 95% | {bm['var_95_daily']:.2%} | {cm['var_95_daily']:.2%} |",
        f"| Daily CVaR 95% | {bm['cvar_95_daily']:.2%} | {cm['cvar_95_daily']:.2%} |",
        f"| Daily VaR 99% | {bm['var_99_daily']:.2%} | {cm['var_99_daily']:.2%} |",
        f"| Daily CVaR 99% | {bm['cvar_99_daily']:.2%} | {cm['cvar_99_daily']:.2%} |",
        f"| Average drawdown episode | {bm['average_episode_drawdown']:.2%} | {cm['average_episode_drawdown']:.2%} |",
        f"| Average recovery | {bm['average_recovery_sessions']:.1f} sessions | {cm['average_recovery_sessions']:.1f} sessions |",
        f"| Maximum recovery | {bm['maximum_recovery_sessions']} sessions | {cm['maximum_recovery_sessions']} sessions |",
        f"| Pain ratio | {bm['pain_ratio']:.3f} | {cm['pain_ratio']:.3f} |",
        f"| Annual alpha vs NDX | {bm['benchmark_alpha_annualized']:.2%} | {cm['benchmark_alpha_annualized']:.2%} |",
        f"| Beta vs NDX | {bm['benchmark_beta']:.3f} | {cm['benchmark_beta']:.3f} |",
        f"| Correlation vs NDX | {bm['benchmark_correlation']:.3f} | {cm['benchmark_correlation']:.3f} |",
        f"| Information ratio | {bm['information_ratio']:.3f} | {cm['information_ratio']:.3f} |",
        "",
        "## Statistical significance",
        "",
        f"- Effective observations: {bm['effective_daily_observations']:.0f} / {cm['effective_daily_observations']:.0f}.",
        f"- t-stat: {bm['mean_return_t_stat']:.3f} / {cm['mean_return_t_stat']:.3f}; PSR: {bm['psr_vs_zero']:.4f} / {cm['psr_vs_zero']:.4f}.",
        f"- DSR after {results['data']['related_prior_trials']:,} trials: {baseline['statistical_diagnostics']['deflated_sharpe_probability']:.4f} / {challenger['statistical_diagnostics']['deflated_sharpe_probability']:.4f}.",
        f"- Skewness: {bm['skewness']:.3f} / {cm['skewness']:.3f}; excess kurtosis: {bm['excess_kurtosis']:.3f} / {cm['excess_kurtosis']:.3f}.",
        f"- Jarque-Bera p: {bm['jarque_bera_p']:.3g} / {cm['jarque_bera_p']:.3g}; Ljung-Box p: {baseline['statistical_diagnostics']['ljung_box_p']:.3g} / {challenger['statistical_diagnostics']['ljung_box_p']:.3g}.",
        f"- Return ADF p: {bm['adf_return_p']:.3g} / {cm['adf_return_p']:.3g} (stationary when below 0.05).",
        f"- Paired annual mean: {paired['annualized_mean_difference']:+.2%}; HAC t={paired['hac_t_stat']:.3f}, p={paired['hac_two_sided_p']:.3f}.",
        f"- Block-bootstrap 95% interval: {paired['bootstrap_95_interval_annualized']}.",
        "",
        "## Robustness",
        "",
        f"- Historical-half efficiency: {results['wfa_efficiency']['baseline']:.3f} / {results['wfa_efficiency']['challenger']:.3f} (pseudo-OOS).",
        "- Sensitivity Calmar: " + ", ".join(
            f"{name}={value['calmar']:.3f}" for name, value in results["sensitivity"].items()
        ) + ".",
        f"- Odd/even paired means: {results['calendar_parity_split']['odd_years']['annualized_mean_difference']:+.2%} / {results['calendar_parity_split']['even_years']['annualized_mean_difference']:+.2%}.",
        f"- Trade bootstrap simulations: {challenger['trade_bootstrap'].get('simulations', 0):,}.",
        "- Challenger trade-bootstrap terminal-return percentiles: "
        + ", ".join(
            f"{name}={value:.1%}"
            for name, value in challenger["trade_bootstrap"]["terminal_return_percentiles"].items()
        ) + ".",
        "- Challenger trade-bootstrap max-drawdown percentiles: "
        + ", ".join(
            f"{name}={value:.1%}"
            for name, value in challenger["trade_bootstrap"]["max_drawdown_percentiles"].items()
        ) + ".",
        f"- 5x/10x-cost paired means: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%} / {results['cost_stress']['10x']['paired_annualized_mean']:+.2%}.",
        f"- Clean forward slice: {results['data']['clean_forward_observations']} daily observations; not meaningful under the frozen plan's 3-year / 5-trade checkpoint.",
        "",
        "### Regime attribution",
        "",
        "| Regime | Delta CAGR | Delta Sharpe | Delta max DD | Delta Calmar |",
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
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead bias | Absent | RSI ends at signal close; every changed exit fills next session open |",
        "| Look-forward in features | Absent | RSI uses current and prior NDX closes only; no future label or shift(-1) enters the signal |",
        "| Survivorship | Cannot fully verify | Aggregate NDX price index and breadth series; constituent history is not modeled |",
        f"| Data snooping / overfitting | Present, material | {results['data']['related_prior_trials']:,} prior/current trials penalized in DSR |",
        "| Transaction costs | Included | $1 plus 0.05% per side, stressed at 2x/5x/10x |",
        "| Liquidity | Low concern, not fully verified | Liquid index proxy; no position-size-versus-ADV model |",
        "| Data frequency mismatch | Absent | Daily close signal and following daily open fill |",
        "| Synthetic data | Present before 2007 | Real-breadth period is reported separately |",
        "| Regime overfit | Tested, not eliminated | Historical halves, named regimes, odd/even years, and sensitivity |",
        "| Clean forward OOS | Insufficient | Post-freeze sample is short and contains too few completed trades |",
        "",
        "## Guardrails",
        "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in results["guardrails"].items()
    )
    current = results["current_signal"]
    lines += [
        "",
        "## Current signal",
        "",
        f"As of {pd.Timestamp(current['date']).date()}, NDX RSI(14) is {current['ndx_rsi14']:.2f}. "
        f"Canonical divergence is {'active' if current['canonical_divergence'] else 'inactive'}; "
        f"the combined challenger signal is {'ACTIVE' if current['active'] else 'inactive'}.",
        "",
        "## Red Flags",
        "",
        "1. Thousands of prior repository trials make historical improvements vulnerable to selection bias.",
        "2. The RSI filter can delay a valid exit until after a sharp gap or until the trailing stop fires.",
        "3. Pre-2007 breadth is synthetic, and post-freeze forward history is too short for adoption.",
        "",
        "## Improvement Recommendations",
        "",
        "1. Follow the pre-registered verdict and leave the frozen baseline unchanged.",
        "2. If tracked, record the fixed RSI<=50 signal without retuning until the forward checkpoint.",
        "",
        "## Decision",
        "",
        "The verdict follows the primary Calmar objective and every pre-registered guardrail.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
