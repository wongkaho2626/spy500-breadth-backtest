"""Fixed 70/30 ensemble of frozen breadth timing and an MA200 trend sleeve."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import qqq_backtest as qbt
import qqq_monthly_breadth_regime_exit as scorelib
import qqq_vector_crash_exit as analytics
import qqq_vector_recross_filter as research


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/breadth_ma200_core_satellite_idea.md"
RESULTS_FILE = ROOT / "qqq_breadth_ma200_core_satellite_results.json"
EQUITY_FILE = ROOT / "qqq_breadth_ma200_core_satellite_equity.csv"
TRADES_FILE = ROOT / "qqq_breadth_ma200_core_satellite_trades.csv"
REPORT_FILE = ROOT / "docs/research/breadth_ma200_core_satellite_report.md"

PRIMARY_WEIGHT = 0.30
SENSITIVITY = (0.20, 0.30, 0.40)
EVENT_CLUSTER_SESSIONS = 21
RELATED_TRIALS = 4_596
INITIAL_CAPITAL = 10_000.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@contextmanager
def baseline_settings(initial_capital: float, cost_multiplier: float) -> Iterator[None]:
    old = (qbt.INITIAL_CAPITAL, qbt.COMMISSION, qbt.SLIPPAGE)
    qbt.INITIAL_CAPITAL = initial_capital
    qbt.COMMISSION = old[1] * cost_multiplier
    qbt.SLIPPAGE = old[2] * cost_multiplier
    try:
        yield
    finally:
        qbt.INITIAL_CAPITAL, qbt.COMMISSION, qbt.SLIPPAGE = old


def run_breadth(
    df: pd.DataFrame,
    initial_capital: float,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None, pd.Series]:
    with baseline_settings(initial_capital, cost_multiplier):
        equity, trades, open_trade = qbt.run_strategy(
            df,
            cooldown_days=qbt.COOLDOWN_DAYS,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
        )
    position = analytics.position_series(df.index, trades, open_trade)
    return equity, trades, open_trade, position


def run_ma200(
    df: pd.DataFrame,
    initial_capital: float,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None, pd.Series]:
    if initial_capital <= 0:
        zero = pd.Series(0.0, index=df.index, name="ma200_equity")
        return zero, [], None, pd.Series(False, index=df.index)
    commission = qbt.COMMISSION * cost_multiplier
    slippage = qbt.SLIPPAGE * cost_multiplier
    above = (df["price"] > df["ma200"]).fillna(False)
    cash = initial_capital
    shares = 0.0
    position = False
    pending: dict | None = None
    entry_date = entry_signal_date = None
    entry_price = entry_capital = trade_low = 0.0
    values: dict[pd.Timestamp, float] = {}
    positions: dict[pd.Timestamp, bool] = {}
    trades: list[dict] = []
    rows = list(df.iterrows())

    for i, (date, row) in enumerate(rows):
        close = float(row["price"])
        fill = float(row["open"]) if not pd.isna(row["open"]) else close
        executed = False
        if pending is not None and pending["fill_at"] == i:
            if pending["action"] == "BUY" and not position:
                entry_capital = cash
                cash -= commission
                entry_price = fill
                shares = cash / (fill * (1 + slippage))
                cash = 0.0
                entry_date = date
                entry_signal_date = pending["signal_date"]
                trade_low = shares * close * (1 - slippage)
                position = True
                executed = True
            elif pending["action"] == "SELL" and position:
                proceeds = shares * fill * (1 - slippage) - commission
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": fill,
                        "return_pct": (proceeds / entry_capital - 1) * 100,
                        "max_drawdown_pct": (trade_low / entry_capital - 1) * 100,
                        "accumulated": proceeds,
                        "buy_trigger": "NDX-above-MA200",
                        "sell_reason": "NDX-below-MA200",
                        "entry_signal_date": entry_signal_date,
                        "signal_date": pending["signal_date"],
                    }
                )
                cash = proceeds
                shares = 0.0
                position = False
                executed = True
            pending = None

        if not executed and pending is None and i + qbt.EXECUTION_LAG < len(rows):
            desired = bool(above.loc[date])
            if desired and not position:
                pending = {
                    "action": "BUY",
                    "fill_at": i + qbt.EXECUTION_LAG,
                    "signal_date": date,
                }
            elif not desired and position:
                pending = {
                    "action": "SELL",
                    "fill_at": i + qbt.EXECUTION_LAG,
                    "signal_date": date,
                }

        if position:
            current = shares * close * (1 - slippage)
            trade_low = min(trade_low, current)
            values[date] = current
        else:
            values[date] = cash
        positions[date] = position

    open_trade = None
    if position:
        last_date = df.index[-1]
        current = values[last_date]
        open_trade = {
            "entry_date": entry_date,
            "entry_price": entry_price,
            "current_date": last_date,
            "current_price": float(df["price"].iloc[-1]),
            "return_pct": (current / entry_capital - 1) * 100,
            "max_drawdown_pct": (trade_low / entry_capital - 1) * 100,
            "accumulated": current,
            "buy_trigger": "NDX-above-MA200",
            "entry_signal_date": entry_signal_date,
        }
    return (
        pd.Series(values, name="ma200_equity"),
        trades,
        open_trade,
        pd.Series(positions, name="ma200_position"),
    )


def component_events(
    component: str,
    trades: list[dict],
    component_equity: pd.Series,
    combined_equity: pd.Series,
) -> list[dict[str, Any]]:
    events = []
    locations = {date: i for i, date in enumerate(combined_equity.index)}
    for trade in trades:
        entry_i = locations[trade["entry_date"]]
        exit_i = locations[trade["exit_date"]]
        before_i = max(0, entry_i - 1)
        allocation = (
            float(component_equity.iloc[before_i] / combined_equity.iloc[before_i])
            if combined_equity.iloc[before_i] > 0
            else 0.0
        )
        events.append(
            {
                "component": component,
                "entry_date": trade["entry_date"],
                "exit_date": trade["exit_date"],
                "signal_date": combined_equity.index[max(0, exit_i - qbt.EXECUTION_LAG)],
                "component_return_pct": float(trade["return_pct"]),
                "allocation_at_entry": allocation,
                "portfolio_contribution_pct": float(trade["return_pct"]) * allocation,
                "sell_reason": trade["sell_reason"],
            }
        )
    return events


def cluster_events(
    index: pd.DatetimeIndex,
    events: list[dict[str, Any]],
    sessions: int = EVENT_CLUSTER_SESSIONS,
) -> list[dict[str, Any]]:
    locations = {date: i for i, date in enumerate(index)}
    ordered = sorted(events, key=lambda event: event["exit_date"])
    clusters: list[list[dict[str, Any]]] = []
    for event in ordered:
        if (
            not clusters
            or locations[event["exit_date"]]
            - locations[clusters[-1][-1]["exit_date"]]
            > sessions
        ):
            clusters.append([event])
        else:
            clusters[-1].append(event)
    output = []
    for number, cluster in enumerate(clusters, 1):
        contribution = sum(event["portfolio_contribution_pct"] for event in cluster)
        output.append(
            {
                "cluster": number,
                "entry_date": min(event["entry_date"] for event in cluster),
                "exit_date": max(event["exit_date"] for event in cluster),
                "return_pct": contribution,
                "max_drawdown_pct": np.nan,
                "accumulated": np.nan,
                "buy_trigger": "+".join(sorted({event["component"] for event in cluster})),
                "sell_reason": "+".join(sorted({event["sell_reason"] for event in cluster})),
                "component_events": len(cluster),
            }
        )
    return output


def run_ensemble(
    df: pd.DataFrame,
    trend_weight: float,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    breadth = run_breadth(
        df, INITIAL_CAPITAL * (1 - trend_weight), cost_multiplier
    )
    trend = run_ma200(df, INITIAL_CAPITAL * trend_weight, cost_multiplier)
    combined = (breadth[0] + trend[0]).rename("combined_equity")
    position = breadth[3] | trend[3]
    events = component_events("breadth", breadth[1], breadth[0], combined)
    events += component_events("ma200", trend[1], trend[0], combined)
    clusters = cluster_events(df.index, events)
    return {
        "equity": combined,
        "position": position,
        "breadth": breadth,
        "trend": trend,
        "events": events,
        "clustered_trades": clusters,
    }


def evaluate(df: pd.DataFrame, run: dict[str, Any]) -> dict[str, Any]:
    equity = run["equity"]
    trades = run["clustered_trades"]
    metrics = analytics.strategy_metrics(equity, trades, run["position"])
    return {
        "metrics": metrics,
        "early_period": analytics.slice_metrics(equity, "2002-01-01", "2013-12-31"),
        "late_period": analytics.slice_metrics(equity, "2014-01-01"),
        "real_breadth_period": analytics.slice_metrics(equity, "2007-01-01"),
        "clean_forward_slice": analytics.slice_metrics(equity, "2026-07-05"),
        "statistical_diagnostics": research.statistical_diagnostics(
            equity, metrics, RELATED_TRIALS
        ),
        "trade_bootstrap": research.trade_bootstrap(trades),
        "raw_component_exits": len(run["events"]),
        "clustered_exit_events": len(trades),
        "breadth_completed_trades": len(run["breadth"][1]),
        "ma200_completed_trades": len(run["trend"][1]),
    }


def period_calmar(period: dict[str, Any]) -> float:
    return period["cagr"] / abs(period["max_drawdown"])


def main() -> None:
    df = qbt.load_data()
    direct_equity, direct_trades, direct_open, direct_position = run_breadth(
        df, INITIAL_CAPITAL
    )
    zero = run_ensemble(df, 0.0)
    parity = {
        "equity_max_absolute_difference": float(
            (direct_equity - zero["equity"]).abs().max()
        ),
        "breadth_trade_signatures_identical": [
            (trade["entry_date"], trade["exit_date"], trade["sell_reason"])
            for trade in direct_trades
        ] == [
            (trade["entry_date"], trade["exit_date"], trade["sell_reason"])
            for trade in zero["breadth"][1]
        ],
        "trend_bucket_zero": bool((zero["trend"][0] == 0).all()),
    }
    parity["passed"] = bool(
        parity["equity_max_absolute_difference"] < 1e-8
        and parity["breadth_trade_signatures_identical"]
        and parity["trend_bucket_zero"]
    )
    if not parity["passed"]:
        raise AssertionError(f"baseline parity failed: {parity}")

    baseline_run = {
        "equity": direct_equity,
        "position": direct_position,
        "breadth": (direct_equity, direct_trades, direct_open, direct_position),
        "trend": run_ma200(df, 0.0),
        "events": component_events(
            "breadth", direct_trades, direct_equity, direct_equity
        ),
        "clustered_trades": [dict(trade) for trade in direct_trades],
    }
    baseline = evaluate(df, baseline_run)
    runs = {f"{weight:.0%}": run_ensemble(df, weight) for weight in SENSITIVITY}
    evaluations = {name: evaluate(df, run) for name, run in runs.items()}
    primary_run = runs["30%"]
    primary = evaluations["30%"]

    base_return = primary_run["breadth"][0].pct_change().fillna(0.0)
    trend_return = primary_run["trend"][0].pct_change().fillna(0.0)
    component_correlation = float(base_return.corr(trend_return))
    paired = analytics.paired_hac_and_bootstrap(
        primary_run["equity"], direct_equity
    )

    sensitivity = {}
    for name, value in evaluations.items():
        sensitivity[name] = {
            metric: value["metrics"][metric]
            for metric in (
                "cagr", "sharpe", "calmar", "max_drawdown",
                "completed_trades", "profit_factor", "expectancy",
            )
        }
    bm, cm = baseline["metrics"], primary["metrics"]
    sensitivity_stable = all(
        value["sharpe"] >= bm["sharpe"]
        and value["calmar"] >= bm["calmar"]
        for value in sensitivity.values()
    )
    bootstrap_stable = paired["bootstrap_95_interval_annualized"][0] > 0
    baseline["score"] = scorelib.score(
        baseline,
        scorelib.efficiency(baseline),
        bootstrap_stable=True,
        sensitivity_stable=True,
    )
    primary["score"] = scorelib.score(
        primary,
        scorelib.efficiency(primary),
        bootstrap_stable=bootstrap_stable,
        sensitivity_stable=sensitivity_stable,
    )

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = run_ensemble(df, 0.0, multiplier)
        challenge_cost = run_ensemble(df, PRIMARY_WEIGHT, multiplier)
        paired_cost = analytics.paired_hac_and_bootstrap(
            challenge_cost["equity"], base_cost["equity"]
        )
        base_metric = analytics.strategy_metrics(
            base_cost["equity"], base_cost["breadth"][1], base_cost["position"]
        )
        challenge_eval = evaluate(df, challenge_cost)
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": base_metric["cagr"],
            "challenger_cagr": challenge_eval["metrics"]["cagr"],
            "cagr_delta": challenge_eval["metrics"]["cagr"] - base_metric["cagr"],
            "paired_annualized_mean": paired_cost["annualized_mean_difference"],
            "paired_hac_t": paired_cost["hac_t_stat"],
        }

    period_deltas = {}
    for period in ("early_period", "late_period", "real_breadth_period"):
        period_deltas[period] = {
            "cagr": primary[period]["cagr"] - baseline[period]["cagr"],
            "sharpe": primary[period]["sharpe"] - baseline[period]["sharpe"],
            "max_drawdown": primary[period]["max_drawdown"] - baseline[period]["max_drawdown"],
            "calmar": period_calmar(primary[period]) - period_calmar(baseline[period]),
        }

    guardrails = {
        "baseline_parity": parity["passed"],
        "at_least_30_clustered_events": cm["completed_trades"] >= 30,
        "component_correlation_below_0_95": component_correlation < 0.95,
        "final_score_at_least_80": primary["score"]["final_score"] >= 80,
        "sharpe_improved": cm["sharpe"] > bm["sharpe"],
        "calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_not_worse": cm["max_drawdown"] >= bm["max_drawdown"],
        "cagr_within_two_points": cm["cagr"] >= bm["cagr"] - 0.02,
        "historical_halves_positive": all(
            period_deltas[period]["sharpe"] > 0
            and period_deltas[period]["calmar"] > 0
            for period in ("early_period", "late_period")
        ),
        "real_breadth_positive": (
            period_deltas["real_breadth_period"]["sharpe"] > 0
            and period_deltas["real_breadth_period"]["calmar"] > 0
        ),
        "five_x_paired_return_positive": cost_stress["5x"]["paired_annualized_mean"] > 0,
        "sensitivity_stable": sensitivity_stable,
        "profit_factor_above_1_2": cm["profit_factor"] > 1.2,
        "positive_expectancy": cm["expectancy"] > 0,
    }
    decision = "track" if all(guardrails.values()) else "reject"

    equity_frame = pd.DataFrame(
        {
            "baseline": direct_equity,
            **{f"ensemble_{name}": run["equity"] for name, run in runs.items()},
            "primary_breadth_bucket": primary_run["breadth"][0],
            "primary_ma200_bucket": primary_run["trend"][0],
        }
    )
    equity_frame.index.name = "Date"
    equity_frame.to_csv(EQUITY_FILE)
    event_rows = []
    for name, run in runs.items():
        for event in run["events"]:
            event_rows.append({"variant": name, "record_type": "component_exit", **event})
        for cluster in run["clustered_trades"]:
            event_rows.append({"variant": name, "record_type": "21_session_cluster", **cluster})
    pd.DataFrame(event_rows).to_csv(TRADES_FILE, index=False)

    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "clean_forward_start": "2026-07-05",
            "related_prior_trials": RELATED_TRIALS,
        },
        "configuration": {
            "breadth_weight": 0.70,
            "primary_ma200_weight": PRIMARY_WEIGHT,
            "sensitivity_ma200_weights": list(SENSITIVITY),
            "ma_window": qbt.MA200_WINDOW,
            "independent_buckets": True,
            "rebalancing": "none",
            "fill": "close signal, next-session open",
            "event_cluster_sessions": EVENT_CLUSTER_SESSIONS,
        },
        "baseline_parity": parity,
        "baseline": baseline,
        "challenger": primary,
        "component_correlation": component_correlation,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": scorelib.efficiency(baseline),
            "challenger": scorelib.efficiency(primary),
            "interpretation": "fixed-rule historical-half efficiency; pseudo-OOS, not clean forward OOS",
        },
        "sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "guardrails": guardrails,
        "current_state": {
            "date": df.index[-1],
            "breadth_sleeve_in": bool(primary_run["breadth"][3].iloc[-1]),
            "ma200_sleeve_in": bool(primary_run["trend"][3].iloc[-1]),
            "ndx_above_ma200": bool(df["price"].iloc[-1] > df["ma200"].iloc[-1]),
        },
    }
    RESULTS_FILE.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    write_report(results)
    print(json.dumps(_jsonable({
        "decision": decision,
        "baseline_score": baseline["score"],
        "challenger_score": primary["score"],
        "baseline_metrics": bm,
        "challenger_metrics": cm,
        "component_correlation": component_correlation,
        "guardrails": guardrails,
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    base, challenge = results["baseline"], results["challenger"]
    bm, cm = base["metrics"], challenge["metrics"]
    bs, cs = base["score"], challenge["score"]
    verdict = "Reject" if results["decision"] == "reject" else "Track as research challenger"
    lines = [
        "# Backtest Verification Report — Breadth + MA200 core-satellite",
        "",
        f"## Verdict: {verdict}",
        "",
        f"The fixed 70/30 ensemble scores **{cs['final_score']} / 100 ({cs['band']})** versus the frozen baseline **{bs['final_score']} / 100 ({bs['band']})**.",
        "",
        "## Backtest Scores",
        "",
        "| Component | Baseline | 70/30 ensemble | Max |",
        "|---|---:|---:|---:|",
        f"| A. Statistical validity | {bs['A_statistical_validity']} | {cs['A_statistical_validity']} | 30 |",
        f"| B. Risk-adjusted performance | {bs['B_risk_adjusted_performance']} | {cs['B_risk_adjusted_performance']} | 25 |",
        f"| C. Robustness / OOS | {bs['C_robustness_oos']} | {cs['C_robustness_oos']} | 25 |",
        f"| D. Trade quality / consistency | {bs['D_trade_quality_consistency']} | {cs['D_trade_quality_consistency']} | 20 |",
        f"| **Raw total** | **{bs['raw_score']}** | **{cs['raw_score']}** | **100** |",
        f"| Hard cap | {bs['hard_cap']} | {cs['hard_cap']} | |",
        f"| **Final score** | **{bs['final_score']}** | **{cs['final_score']}** | **100** |",
        "",
        "## Performance",
        "",
        "| Metric | Baseline | Ensemble | Delta |",
        "|---|---:|---:|---:|",
        f"| CAGR | {bm['cagr']:.2%} | {cm['cagr']:.2%} | {cm['cagr']-bm['cagr']:+.2%} |",
        f"| Sharpe | {bm['sharpe']:.3f} | {cm['sharpe']:.3f} | {cm['sharpe']-bm['sharpe']:+.3f} |",
        f"| Sortino | {bm['sortino']:.3f} | {cm['sortino']:.3f} | {cm['sortino']-bm['sortino']:+.3f} |",
        f"| Calmar | {bm['calmar']:.3f} | {cm['calmar']:.3f} | {cm['calmar']-bm['calmar']:+.3f} |",
        f"| Maximum drawdown | {bm['max_drawdown']:.2%} | {cm['max_drawdown']:.2%} | {cm['max_drawdown']-bm['max_drawdown']:+.2%} |",
        f"| Ulcer Index | {bm['ulcer_index']:.2%} | {cm['ulcer_index']:.2%} | {cm['ulcer_index']-bm['ulcer_index']:+.2%} |",
        f"| Positive months | {bm['positive_months']:.2%} | {cm['positive_months']:.2%} | {cm['positive_months']-bm['positive_months']:+.2%} |",
        f"| Completed / clustered events | {bm['completed_trades']} | {cm['completed_trades']} | {cm['completed_trades']-bm['completed_trades']:+d} |",
        f"| Profit factor | {bm['profit_factor']:.2f} | {cm['profit_factor']:.2f} | |",
        f"| Expectancy | {bm['expectancy']:.2%} | {cm['expectancy']:.2%} | |",
        "",
        "## Independence and robustness",
        "",
        f"- Component daily-return correlation: {results['component_correlation']:.3f}.",
        f"- Raw component exits / 21-session clusters: {challenge['raw_component_exits']} / {challenge['clustered_exit_events']}.",
        f"- Paired annual mean: {results['paired_inference']['annualized_mean_difference']:+.2%}; HAC t={results['paired_inference']['hac_t_stat']:.3f}, p={results['paired_inference']['hac_two_sided_p']:.3f}.",
        f"- Block-bootstrap 95% interval: {results['paired_inference']['bootstrap_95_interval_annualized']}.",
        f"- Historical-half efficiency: baseline {results['wfa_efficiency']['baseline']:.3f}; ensemble {results['wfa_efficiency']['challenger']:.3f} (pseudo-OOS only).",
        f"- Sensitivity Sharpes: " + ", ".join(f"{key}={value['sharpe']:.3f}" for key, value in results['sensitivity'].items()) + ".",
        f"- 5x-cost paired annual mean: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Both sleeves signal on close and fill next-session open |",
        "| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |",
        "| Data snooping | Present, material | At least 4,596 related trials; DSR penalty applied |",
        "| Trade independence | Adjusted | Component exits within 21 sessions counted as one event |",
        "| Costs | Included | Every component transition; 1x/2x/5x/10x stress |",
        "| Synthetic breadth | Present before 2007 | 2007+ result reported separately |",
        "| Clean forward OOS | Insufficient | No completed post-freeze ensemble evaluation period |",
        "",
        "## Guardrails",
        "",
    ]
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in results["guardrails"].items()
    )
    lines += [
        "",
        "## Decision",
        "",
        "The decision follows the pre-registered score and economic guardrails.  A historical score of 80 or more would still require clean forward tracking before adoption.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
