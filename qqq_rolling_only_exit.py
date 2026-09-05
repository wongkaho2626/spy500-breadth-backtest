"""Use the selected rolling breadth rule as the sole QQQ sell signal."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

import qqq_price_rolling_breadth_grid as grid


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/rolling_only_exit_idea.md"
REPORT_FILE = ROOT / "docs/research/rolling_only_exit_report.md"
RESULTS_FILE = ROOT / "qqq_rolling_only_exit_results.json"
EQUITY_FILE = ROOT / "qqq_rolling_only_exit_equity.csv"
TRADES_FILE = ROOT / "qqq_rolling_only_exit_trades.csv"
SIGNALS_FILE = ROOT / "qqq_rolling_only_exit_signals.csv"

START_DATE = "2002-01-01"
PRICE_RISE = 2.0
BREADTH_CAP = 70.0
PRIMARY_DRAWDOWN = 30.0
SENSITIVITY = (25.0, 30.0, 35.0)
RELATED_TRIALS = 4_819

qbt = grid.qbt
framework = grid.framework
analytics = grid.analytics


@contextmanager
def engine_override(
    cost_multiplier: float,
    disable_auxiliary_exits: bool,
) -> Iterator[None]:
    old_commission = qbt.COMMISSION
    old_slippage = qbt.SLIPPAGE
    old_cap = qbt.DIVERGENCE_BREADTH_CAP
    old_trailing = qbt.TRAILING_STOP_PCT
    qbt.COMMISSION = old_commission * cost_multiplier
    qbt.SLIPPAGE = old_slippage * cost_multiplier
    qbt.DIVERGENCE_BREADTH_CAP = float("inf")
    if disable_auxiliary_exits:
        qbt.TRAILING_STOP_PCT = 1_000.0
    try:
        yield
    finally:
        qbt.COMMISSION = old_commission
        qbt.SLIPPAGE = old_slippage
        qbt.DIVERGENCE_BREADTH_CAP = old_cap
        qbt.TRAILING_STOP_PCT = old_trailing


def run_engine(
    df: pd.DataFrame,
    features: pd.DataFrame,
    signal: pd.Series,
    reason: str,
    disable_auxiliary_exits: bool,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    experiment = df.copy()
    experiment["price_rose"] = signal.reindex(df.index).fillna(False).astype(bool)
    experiment["breadth_fell"] = True
    if disable_auxiliary_exits:
        experiment["macd_cross"] = False
        experiment["ext10"] = False
    with engine_override(cost_multiplier, disable_auxiliary_exits):
        equity, trades, open_trade = qbt.run_strategy(
            experiment,
            cooldown_days=qbt.COOLDOWN_DAYS,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
        )

    output = []
    for original in trades:
        trade = dict(original)
        exit_location = df.index.get_loc(trade["exit_date"])
        signal_date = df.index[exit_location - qbt.EXECUTION_LAG]
        trade["signal_date"] = signal_date
        if trade["sell_reason"] == "bearish-divergence":
            trade["sell_reason"] = reason
            trade["ndx_return_60_pct"] = float(
                features.loc[signal_date, "ndx_return_60_pct"]
            )
            trade["rolling_breadth_max_60"] = float(
                features.loc[signal_date, "rolling_breadth_max_60"]
            )
            trade["breadth_on_signal"] = float(
                features.loc[signal_date, "breadth"]
            )
            trade["breadth_drawdown_60_points"] = float(
                features.loc[signal_date, "breadth_drawdown_60_points"]
            )
        output.append(trade)
    return equity, output, open_trade


def rolling_signal(
    features: pd.DataFrame, drawdown: float
) -> pd.Series:
    return grid.grid_signal(features, PRICE_RISE, drawdown, BREADTH_CAP)


def sensitivity_is_stable(
    baseline_calmar: float,
    evaluations: dict[str, dict[str, Any]],
) -> bool:
    calmars = [
        float(evaluations[f"drawdown_{value:.0f}"]["metrics"]["calmar"])
        for value in SENSITIVITY
    ]
    if not all(value > baseline_calmar for value in calmars):
        return False
    return all(
        abs(right / left - 1) < 0.25
        for left, right in zip(calmars, calmars[1:])
        if left != 0
    )


def main() -> None:
    if not IDEA_CARD.exists():
        raise FileNotFoundError("pre-registered idea card is required")
    framework.RELATED_TRIALS = RELATED_TRIALS
    df = qbt.load_data().loc[START_DATE:].copy()
    features = grid.build_features(df)
    canonical_signal = analytics.baseline_divergence_signal(df)

    direct = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    baseline_run = run_engine(
        df, features, canonical_signal, "bearish-divergence", False
    )
    parity = {
        "equity_max_absolute_difference": float(
            (direct[0] - baseline_run[0]).abs().max()
        ),
        "trade_signatures_identical": (
            framework.trade_signature(direct[1])
            == framework.trade_signature(baseline_run[1])
        ),
        "open_trade_identical": framework.open_trade_equal(
            direct[2], baseline_run[2]
        ),
    }
    parity["passed"] = bool(
        parity["equity_max_absolute_difference"] < 1e-8
        and parity["trade_signatures_identical"]
        and parity["open_trade_identical"]
    )
    if not parity["passed"]:
        raise AssertionError(f"baseline parity failed: {parity}")

    signals = {
        f"drawdown_{value:.0f}": rolling_signal(features, value)
        for value in SENSITIVITY
    }
    solo_runs = {
        name: run_engine(
            df, features, signal, "rolling-only", True
        )
        for name, signal in signals.items()
    }
    primary_name = f"drawdown_{PRIMARY_DRAWDOWN:.0f}"
    primary_run = solo_runs[primary_name]
    auxiliary_on_run = grid.run_combo(
        df, features, (PRICE_RISE, PRIMARY_DRAWDOWN, BREADTH_CAP)
    )

    baseline = framework.evaluate(df, *baseline_run)
    challenger = framework.evaluate(df, *primary_run)
    auxiliary_on = framework.evaluate(df, *auxiliary_on_run)
    evaluations = {
        name: framework.evaluate(df, *run)
        for name, run in solo_runs.items()
    }
    paired = analytics.paired_hac_and_bootstrap(
        primary_run[0], baseline_run[0]
    )
    paired_vs_auxiliary = analytics.paired_hac_and_bootstrap(
        primary_run[0], auxiliary_on_run[0]
    )
    stable = sensitivity_is_stable(
        float(baseline["metrics"]["calmar"]), evaluations
    )
    baseline["score"] = framework.score(
        baseline, framework.efficiency(baseline), True, True
    )
    challenger["score"] = framework.score(
        challenger,
        framework.efficiency(challenger),
        paired["bootstrap_95_interval_annualized"][0] > 0,
        stable,
    )

    sensitivity = {
        name: {
            metric: evaluation["metrics"][metric]
            for metric in (
                "cagr", "sharpe", "calmar", "max_drawdown",
                "completed_trades", "expectancy", "exposure",
            )
        }
        for name, evaluation in evaluations.items()
    }

    cost_stress: dict[str, dict[str, float]] = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = run_engine(
            df, features, canonical_signal, "bearish-divergence", False,
            multiplier,
        )
        challenge_cost = run_engine(
            df, features, signals[primary_name], "rolling-only", True,
            multiplier,
        )
        base_eval = framework.evaluate(df, *base_cost)
        challenge_eval = framework.evaluate(df, *challenge_cost)
        pair = analytics.paired_hac_and_bootstrap(
            challenge_cost[0], base_cost[0]
        )
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": float(base_eval["metrics"]["cagr"]),
            "challenger_cagr": float(challenge_eval["metrics"]["cagr"]),
            "cagr_delta": float(
                challenge_eval["metrics"]["cagr"]
                - base_eval["metrics"]["cagr"]
            ),
            "paired_annualized_mean": float(pair["annualized_mean_difference"]),
            "paired_hac_t": float(pair["hac_t_stat"]),
        }

    period_deltas = {
        period: grid.source.rolling.base.period_delta(
            challenger[period], baseline[period]
        )
        for period in ("early_period", "late_period", "real_breadth_period")
    }
    calendar_split = grid.source.rolling.base.calendar_parity_split(
        primary_run[0], baseline_run[0]
    )
    bm, cm, am = (
        baseline["metrics"], challenger["metrics"], auxiliary_on["metrics"]
    )
    guardrails = {
        "baseline_parity": parity["passed"],
        "calmar_beats_baseline": cm["calmar"] > bm["calmar"],
        "calmar_beats_auxiliary_on": cm["calmar"] > am["calmar"],
        "max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "cagr_within_two_points": cm["cagr"] >= bm["cagr"] - 0.02,
        "expectancy_not_worse": cm["expectancy"] >= bm["expectancy"],
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

    benchmark = qbt.run_benchmark(df)
    equity = pd.DataFrame(
        {
            "ndx_buy_hold": benchmark,
            "baseline": baseline_run[0],
            "rolling_auxiliary_on": auxiliary_on_run[0],
            **{name: run[0] for name, run in solo_runs.items()},
        }
    )
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)
    trade_rows = []
    for name, run in {
        "baseline": baseline_run,
        "rolling_auxiliary_on": auxiliary_on_run,
        **solo_runs,
    }.items():
        trade_rows.extend({"variant": name, **trade} for trade in run[1])
    pd.DataFrame(trade_rows).to_csv(TRADES_FILE, index=False)
    signal_frame = features.copy()
    for name, signal in signals.items():
        signal_frame[name] = signal
    signal_frame["next_session_open"] = df["open"].shift(-1)
    signal_frame.index.name = "Date"
    signal_frame.to_csv(SIGNALS_FILE)

    current_signal = signals[primary_name]
    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "clean_forward_start": "2026-07-05",
            "related_trials": RELATED_TRIALS,
        },
        "configuration": {
            "price_rise_pct": PRICE_RISE,
            "breadth_drawdown_points": PRIMARY_DRAWDOWN,
            "breadth_cap": BREADTH_CAP,
            "rolling_window_sessions": 60,
            "climax_top": "disabled",
            "trailing_stop": "disabled",
            "fill": "next-session open",
            "sensitivity_drawdown_points": list(SENSITIVITY),
        },
        "baseline_parity": parity,
        "benchmark": {
            "name": "NDX price-index buy and hold",
            "metrics": analytics.slice_metrics(benchmark, START_DATE),
        },
        "baseline": baseline,
        "rolling_auxiliary_on": auxiliary_on,
        "challenger": challenger,
        "paired_inference": paired,
        "paired_vs_auxiliary_on": paired_vs_auxiliary,
        "wfa_efficiency": {
            "baseline": framework.efficiency(baseline),
            "challenger": framework.efficiency(challenger),
            "interpretation": "historical-half pseudo-OOS only",
        },
        "sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "calendar_parity_split": calendar_split,
        "guardrails": guardrails,
        "current_signal": {
            "date": df.index[-1],
            "active": bool(current_signal.iloc[-1]),
            "ndx_return_60_pct": float(
                features["ndx_return_60_pct"].iloc[-1]
            ),
            "breadth": float(features["breadth"].iloc[-1]),
            "rolling_breadth_max_60": float(
                features["rolling_breadth_max_60"].iloc[-1]
            ),
            "breadth_drawdown_60_points": float(
                features["breadth_drawdown_60_points"].iloc[-1]
            ),
            "position_open": primary_run[2] is not None,
        },
    }
    RESULTS_FILE.write_text(
        json.dumps(framework._jsonable(results), indent=2), encoding="utf-8"
    )
    write_report(results)
    print(json.dumps(framework._jsonable({
        "decision": decision,
        "baseline_score": baseline["score"],
        "challenger_score": challenger["score"],
        "baseline_metrics": bm,
        "rolling_auxiliary_on_metrics": am,
        "rolling_only_metrics": cm,
        "paired_inference": paired,
        "paired_vs_auxiliary_on": paired_vs_auxiliary,
        "sensitivity": sensitivity,
        "guardrails": guardrails,
        "current_signal": results["current_signal"],
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    bm = results["baseline"]["metrics"]
    am = results["rolling_auxiliary_on"]["metrics"]
    cm = results["challenger"]["metrics"]
    bs = results["baseline"]["score"]
    cs = results["challenger"]["score"]
    verdict = "Reject" if results["decision"] == "reject" else "Track"
    lines = [
        "# Backtest Verification Report — rolling-only exit",
        "",
        f"## Verdict: {verdict}",
        "",
        "## Executive Summary",
        "",
        "The 2% / 30-point / 70% rolling rule is the sole sell signal; climax-top and trailing-stop exits are disabled.",
        "",
        "## Performance comparison",
        "",
        "| Metric | Frozen baseline | Rolling + auxiliary exits | Rolling only |",
        "|---|---:|---:|---:|",
        f"| CAGR | {bm['cagr']:.2%} | {am['cagr']:.2%} | {cm['cagr']:.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {am['sharpe']:.3f} | {cm['sharpe']:.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {am['sortino']:.3f} | {cm['sortino']:.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {am['calmar']:.3f} | {cm['calmar']:.3f} |",
        f"| Maximum drawdown | {bm['max_drawdown']:.2%} | {am['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {am['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} |",
        f"| Exposure | {bm['exposure']:.2%} | {am['exposure']:.2%} | {cm['exposure']:.2%} |",
        f"| Completed trades | {bm['completed_trades']} | {am['completed_trades']} | {cm['completed_trades']} |",
        f"| Expectancy | {bm['expectancy']:.2%} | {am['expectancy']:.2%} | {cm['expectancy']:.2%} |",
        "",
        "## Backtest Score",
        "",
        f"Baseline raw/final: {bs['raw_score']} / {bs['final_score']}; rolling-only raw/final: {cs['raw_score']} / {cs['final_score']}.",
        "",
        "## Statistical significance",
        "",
        f"- Versus baseline paired annual mean: {results['paired_inference']['annualized_mean_difference']:+.2%}; HAC p={results['paired_inference']['hac_two_sided_p']:.3f}; bootstrap 95% interval {results['paired_inference']['bootstrap_95_interval_annualized']}.",
        f"- Versus rolling auxiliary-on paired annual mean: {results['paired_vs_auxiliary_on']['annualized_mean_difference']:+.2%}; HAC p={results['paired_vs_auxiliary_on']['hac_two_sided_p']:.3f}.",
        "",
        "## Robustness",
        "",
        "- Sensitivity Calmar: " + ", ".join(
            f"{name}={value['calmar']:.3f}"
            for name, value in results["sensitivity"].items()
        ) + ".",
        f"- 5x/10x cost paired means: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%} / {results['cost_stress']['10x']['paired_annualized_mean']:+.2%}.",
        f"- Historical-half efficiency: {results['wfa_efficiency']['baseline']:.3f} / {results['wfa_efficiency']['challenger']:.3f}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead | Absent | Close features; next-open fill |",
        "| Data snooping | Present | Rolling parameters were selected from the prior grid |",
        "| Costs | Included | 1x/2x/5x/10x stress |",
        "| Synthetic breadth | Present before 2007 | 2007+ reported separately |",
        "| Clean forward OOS | Insufficient | Parameters and auxiliary removal are historically evaluated |",
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
        f"As of {pd.Timestamp(current['date']).date()}, signal is {'ACTIVE' if current['active'] else 'inactive'}; NDX return {current['ndx_return_60_pct']:+.2f}%, breadth drawdown {current['breadth_drawdown_60_points']:.2f} points, breadth {current['breadth']:.2f}%.",
        "",
        "## Red Flags",
        "",
        "1. Without a trailing stop, a persistent bear market can remain unprotected once the price-rise vote turns negative.",
        "2. Removing climax exits also removes the special re-entry treatment following a climax exit.",
        "",
        "## Verdict",
        "",
        "The decision follows the pre-registered Calmar and drawdown guardrails.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
