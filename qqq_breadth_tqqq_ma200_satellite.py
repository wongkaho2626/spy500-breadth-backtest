"""Fixed 85/15 frozen-breadth plus TQQQ MA200 trend-satellite audit."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_backtest as qbt
import qqq_breadth_ma200_core_satellite as core
import qqq_monthly_breadth_regime_exit as scorelib
import qqq_vector_crash_exit as analytics
import qqq_vector_recross_filter as research
import tqqq_backtest as tqbt


ROOT = Path(__file__).parent
IDEA_CARD = ROOT / "docs/research/breadth_tqqq_ma200_satellite_idea.md"
RESULTS_FILE = ROOT / "qqq_breadth_tqqq_ma200_satellite_results.json"
EQUITY_FILE = ROOT / "qqq_breadth_tqqq_ma200_satellite_equity.csv"
TRADES_FILE = ROOT / "qqq_breadth_tqqq_ma200_satellite_trades.csv"
REPORT_FILE = ROOT / "docs/research/breadth_tqqq_ma200_satellite_report.md"

PRIMARY_WEIGHT = 0.15
SENSITIVITY = (0.10, 0.15, 0.20)
DRAG_STRESS = (1.0, 3.0)
RELATED_TRIALS = 4_597


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


def load_tqqq_proxy(
    df: pd.DataFrame,
    drag_multiplier: float,
    loaded: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float]:
    """Return actual TQQQ plus a reconstructed pre-inception proxy."""
    base = tqbt.load_tqqq_data()[["price", "open"]] if loaded is None else loaded
    first_actual = pd.Timestamp(tqbt.TQQQ_INCEPTION)
    actual = base.loc[first_actual:].copy()
    overlap = pd.concat(
        [
            actual["price"].pct_change().rename("tqqq"),
            df["price"].pct_change().rename("ndx"),
        ],
        axis=1,
    ).dropna()
    daily_drag = float(
        (tqbt.LEVERAGE * overlap["ndx"] - overlap["tqqq"]).mean()
    )
    pre = df.loc[df.index < first_actual, ["price", "open"]]
    simulated_return = (
        tqbt.LEVERAGE * pre["price"].pct_change()
        - daily_drag * drag_multiplier
    ).fillna(0.0)
    cumulative = (1 + simulated_return).cumprod()
    boundary_ndx = float(df["price"].pct_change().get(first_actual, 0.0))
    boundary = tqbt.LEVERAGE * boundary_ndx - daily_drag * drag_multiplier
    scale = float(actual["price"].iloc[0]) / (
        float(cumulative.iloc[-1]) * (1 + boundary)
    )
    simulated_close = cumulative * scale
    overnight = df["open"] / df["price"].shift(1) - 1
    simulated_open = simulated_close.shift(1) * (
        1 + tqbt.LEVERAGE * overnight.reindex(simulated_close.index)
    )
    simulated_open = simulated_open.fillna(simulated_close)
    proxy = pd.concat(
        [
            pd.DataFrame({"price": simulated_close, "open": simulated_open}),
            actual,
        ]
    ).sort_index()
    proxy = proxy[~proxy.index.duplicated(keep="last")].reindex(df.index).ffill()
    return proxy, daily_drag


def run_tqqq_ma200(
    df: pd.DataFrame,
    tqqq: pd.DataFrame,
    initial_capital: float,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None, pd.Series]:
    if initial_capital <= 0:
        zero = pd.Series(0.0, index=df.index, name="tqqq_ma200_equity")
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

    for i, (date, _) in enumerate(rows):
        close = float(tqqq.loc[date, "price"])
        open_price = float(tqqq.loc[date, "open"])
        fill = open_price if not pd.isna(open_price) else close
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
                        "buy_trigger": "NDX-above-MA200/TQQQ",
                        "sell_reason": "NDX-below-MA200/TQQQ",
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
            "current_price": float(tqqq["price"].iloc[-1]),
            "return_pct": (current / entry_capital - 1) * 100,
            "max_drawdown_pct": (trade_low / entry_capital - 1) * 100,
            "accumulated": current,
            "buy_trigger": "NDX-above-MA200/TQQQ",
            "entry_signal_date": entry_signal_date,
        }
    return (
        pd.Series(values, name="tqqq_ma200_equity"),
        trades,
        open_trade,
        pd.Series(positions, name="tqqq_ma200_position"),
    )


def run_ensemble(
    df: pd.DataFrame,
    tqqq: pd.DataFrame,
    tqqq_weight: float,
    cost_multiplier: float = 1.0,
) -> dict[str, Any]:
    breadth = core.run_breadth(
        df, core.INITIAL_CAPITAL * (1 - tqqq_weight), cost_multiplier
    )
    trend = run_tqqq_ma200(
        df, tqqq, core.INITIAL_CAPITAL * tqqq_weight, cost_multiplier
    )
    combined = (breadth[0] + trend[0]).rename("combined_equity")
    position = breadth[3] | trend[3]
    events = core.component_events(
        "breadth", breadth[1], breadth[0], combined
    )
    events += core.component_events(
        "tqqq_ma200", trend[1], trend[0], combined
    )
    clusters = core.cluster_events(df.index, events)
    return {
        "equity": combined,
        "position": position,
        "breadth": breadth,
        "trend": trend,
        "events": events,
        "clustered_trades": clusters,
    }


def evaluate(df: pd.DataFrame, run: dict[str, Any]) -> dict[str, Any]:
    metrics = analytics.strategy_metrics(
        run["equity"], run["clustered_trades"], run["position"]
    )
    return {
        "metrics": metrics,
        "early_period": analytics.slice_metrics(
            run["equity"], "2002-01-01", "2013-12-31"
        ),
        "late_period": analytics.slice_metrics(
            run["equity"], "2014-01-01"
        ),
        "real_breadth_period": analytics.slice_metrics(
            run["equity"], "2007-01-01"
        ),
        "clean_forward_slice": analytics.slice_metrics(
            run["equity"], "2026-07-05"
        ),
        "statistical_diagnostics": research.statistical_diagnostics(
            run["equity"], metrics, RELATED_TRIALS
        ),
        "trade_bootstrap": research.trade_bootstrap(run["clustered_trades"]),
        "raw_component_exits": len(run["events"]),
        "clustered_exit_events": len(run["clustered_trades"]),
        "breadth_completed_trades": len(run["breadth"][1]),
        "tqqq_ma200_completed_trades": len(run["trend"][1]),
    }


def period_calmar(period: dict[str, Any]) -> float:
    return period["cagr"] / abs(period["max_drawdown"])


def evaluate_family(
    df: pd.DataFrame,
    proxy: pd.DataFrame,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    runs = {f"{weight:.0%}": run_ensemble(df, proxy, weight) for weight in SENSITIVITY}
    evaluations = {name: evaluate(df, run) for name, run in runs.items()}
    return runs, evaluations


def family_sensitivity_stable(
    evaluations: dict[str, dict[str, Any]],
    baseline: dict[str, Any],
) -> bool:
    bm = baseline["metrics"]
    scores = []
    for value in evaluations.values():
        metrics = value["metrics"]
        if not (
            metrics["cagr"] > bm["cagr"]
            and metrics["sharpe"] > bm["sharpe"]
            and metrics["calmar"] > bm["calmar"]
        ):
            return False
        scores.append(value["score"]["final_score"] if "score" in value else np.nan)
    return True


def main() -> None:
    df = qbt.load_data()
    loaded = tqbt.load_tqqq_data()[["price", "open"]]
    proxy_1x, daily_drag = load_tqqq_proxy(df, 1.0, loaded=loaded)
    proxy_3x, _ = load_tqqq_proxy(df, 3.0, loaded=loaded)

    direct_equity, direct_trades, direct_open, direct_position = core.run_breadth(
        df, core.INITIAL_CAPITAL
    )
    zero = run_ensemble(df, proxy_1x, 0.0)
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
        "breadth": (
            direct_equity, direct_trades, direct_open, direct_position
        ),
        "trend": run_tqqq_ma200(df, proxy_1x, 0.0),
        "events": core.component_events(
            "breadth", direct_trades, direct_equity, direct_equity
        ),
        "clustered_trades": [dict(trade) for trade in direct_trades],
    }
    baseline = evaluate(df, baseline_run)

    runs_1x, evals_1x = evaluate_family(df, proxy_1x)
    primary_run = runs_1x["15%"]
    primary = evals_1x["15%"]
    paired = analytics.paired_hac_and_bootstrap(
        primary_run["equity"], direct_equity
    )
    base_component_return = primary_run["breadth"][0].pct_change().fillna(0)
    trend_component_return = primary_run["trend"][0].pct_change().fillna(0)
    correlation = float(base_component_return.corr(trend_component_return))

    # Score baseline and every sensitivity row before applying the fixed
    # sensitivity guardrail.  Each row receives its own paired-bootstrap test.
    baseline["score"] = scorelib.score(
        baseline,
        scorelib.efficiency(baseline),
        bootstrap_stable=True,
        sensitivity_stable=True,
    )
    for name, evaluation in evals_1x.items():
        pair = analytics.paired_hac_and_bootstrap(
            runs_1x[name]["equity"], direct_equity
        )
        evaluation["score"] = scorelib.score(
            evaluation,
            scorelib.efficiency(evaluation),
            bootstrap_stable=(pair["bootstrap_95_interval_annualized"][0] > 0),
            sensitivity_stable=True,
        )
    sensitivity_stable = family_sensitivity_stable(evals_1x, baseline)
    # Re-score primary with the family-level sensitivity verdict.
    primary["score"] = scorelib.score(
        primary,
        scorelib.efficiency(primary),
        bootstrap_stable=(paired["bootstrap_95_interval_annualized"][0] > 0),
        sensitivity_stable=sensitivity_stable,
    )

    sensitivity = {}
    for name, value in evals_1x.items():
        sensitivity[name] = {
            **{
                metric: value["metrics"][metric]
                for metric in (
                    "cagr", "sharpe", "calmar", "max_drawdown",
                    "completed_trades", "profit_factor", "expectancy",
                )
            },
            "score": value["score"]["final_score"],
        }
    neighbouring_scores_stable = (
        max(value["score"] for value in sensitivity.values())
        - min(value["score"] for value in sensitivity.values())
        <= 5
    )
    sensitivity_stable = sensitivity_stable and neighbouring_scores_stable
    primary["score"] = scorelib.score(
        primary,
        scorelib.efficiency(primary),
        bootstrap_stable=(paired["bootstrap_95_interval_annualized"][0] > 0),
        sensitivity_stable=sensitivity_stable,
    )

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        base_cost = run_ensemble(df, proxy_1x, 0.0, multiplier)
        challenge_cost = run_ensemble(
            df, proxy_1x, PRIMARY_WEIGHT, multiplier
        )
        pair = analytics.paired_hac_and_bootstrap(
            challenge_cost["equity"], base_cost["equity"]
        )
        base_metrics = analytics.strategy_metrics(
            base_cost["equity"], base_cost["breadth"][1], base_cost["position"]
        )
        challenge = evaluate(df, challenge_cost)
        cost_stress[f"{multiplier}x"] = {
            "baseline_cagr": base_metrics["cagr"],
            "challenger_cagr": challenge["metrics"]["cagr"],
            "cagr_delta": challenge["metrics"]["cagr"] - base_metrics["cagr"],
            "paired_annualized_mean": pair["annualized_mean_difference"],
            "paired_hac_t": pair["hac_t_stat"],
        }

    period_deltas = {}
    for period in ("early_period", "late_period", "real_breadth_period"):
        period_deltas[period] = {
            "cagr": primary[period]["cagr"] - baseline[period]["cagr"],
            "sharpe": primary[period]["sharpe"] - baseline[period]["sharpe"],
            "max_drawdown": primary[period]["max_drawdown"] - baseline[period]["max_drawdown"],
            "calmar": period_calmar(primary[period]) - period_calmar(baseline[period]),
        }

    # Punitive 3x-drag family, including sensitivity and score.
    runs_3x, evals_3x = evaluate_family(df, proxy_3x)
    primary_3x = evals_3x["15%"]
    paired_3x = analytics.paired_hac_and_bootstrap(
        runs_3x["15%"]["equity"], direct_equity
    )
    for name, evaluation in evals_3x.items():
        pair = analytics.paired_hac_and_bootstrap(
            runs_3x[name]["equity"], direct_equity
        )
        evaluation["score"] = scorelib.score(
            evaluation,
            scorelib.efficiency(evaluation),
            bootstrap_stable=(pair["bootstrap_95_interval_annualized"][0] > 0),
            sensitivity_stable=True,
        )
    stable_3x = family_sensitivity_stable(evals_3x, baseline)
    scores_3x = [value["score"]["final_score"] for value in evals_3x.values()]
    stable_3x = stable_3x and max(scores_3x) - min(scores_3x) <= 5
    primary_3x["score"] = scorelib.score(
        primary_3x,
        scorelib.efficiency(primary_3x),
        bootstrap_stable=(paired_3x["bootstrap_95_interval_annualized"][0] > 0),
        sensitivity_stable=stable_3x,
    )
    drag_stress = {
        "daily_drag_1x": daily_drag,
        "annualized_drag_1x": daily_drag * 252,
        "1x": {
            "metrics": primary["metrics"],
            "score": primary["score"],
            "paired": paired,
        },
        "3x": {
            "metrics": primary_3x["metrics"],
            "score": primary_3x["score"],
            "paired": paired_3x,
            "sensitivity_stable": stable_3x,
        },
    }

    bm, cm = baseline["metrics"], primary["metrics"]
    guardrails = {
        "baseline_parity": parity["passed"],
        "at_least_30_clustered_events": cm["completed_trades"] >= 30,
        "component_correlation_below_0_95": correlation < 0.95,
        "final_score_at_least_80": primary["score"]["final_score"] >= 80,
        "cagr_improved": cm["cagr"] > bm["cagr"],
        "sharpe_improved": cm["sharpe"] > bm["sharpe"],
        "calmar_improved": cm["calmar"] > bm["calmar"],
        "max_drawdown_within_two_points": cm["max_drawdown"] >= bm["max_drawdown"] - 0.02,
        "paired_mean_positive": paired["annualized_mean_difference"] > 0,
        "paired_bootstrap_excludes_zero": paired["bootstrap_95_interval_annualized"][0] > 0,
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
        "three_x_drag_paired_positive": paired_3x["annualized_mean_difference"] > 0,
        "three_x_drag_drawdown_within_two_points": (
            primary_3x["metrics"]["max_drawdown"] >= bm["max_drawdown"] - 0.02
        ),
        "three_x_drag_score_at_least_80": primary_3x["score"]["final_score"] >= 80,
    }
    decision = "track" if all(guardrails.values()) else "reject"

    equity = pd.DataFrame(
        {
            "baseline": direct_equity,
            **{f"ensemble_{name}": run["equity"] for name, run in runs_1x.items()},
            "primary_breadth_bucket": primary_run["breadth"][0],
            "primary_tqqq_ma200_bucket": primary_run["trend"][0],
            "primary_3x_drag": runs_3x["15%"]["equity"],
        }
    )
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)
    rows = []
    for name, run in runs_1x.items():
        for event in run["events"]:
            rows.append({"variant": name, "record_type": "component_exit", **event})
        for cluster in run["clustered_trades"]:
            rows.append({"variant": name, "record_type": "21_session_cluster", **cluster})
    pd.DataFrame(rows).to_csv(TRADES_FILE, index=False)

    results = {
        "decision": decision,
        "idea_card": IDEA_CARD,
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "actual_tqqq_start": tqbt.TQQQ_INCEPTION,
            "clean_forward_start": "2026-07-05",
            "related_prior_trials": RELATED_TRIALS,
        },
        "configuration": {
            "breadth_weight": 0.85,
            "primary_tqqq_ma200_weight": PRIMARY_WEIGHT,
            "sensitivity_tqqq_weights": list(SENSITIVITY),
            "ma_window": qbt.MA200_WINDOW,
            "independent_buckets": True,
            "rebalancing": "none",
            "fill": "close signal, next-session open",
            "event_cluster_sessions": core.EVENT_CLUSTER_SESSIONS,
        },
        "baseline_parity": parity,
        "baseline": baseline,
        "challenger": primary,
        "component_correlation": correlation,
        "paired_inference": paired,
        "wfa_efficiency": {
            "baseline": scorelib.efficiency(baseline),
            "challenger": scorelib.efficiency(primary),
            "interpretation": "fixed-rule historical-half efficiency; pseudo-OOS, not clean forward OOS",
        },
        "sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "period_deltas": period_deltas,
        "drag_stress": drag_stress,
        "guardrails": guardrails,
        "current_state": {
            "date": df.index[-1],
            "breadth_sleeve_in": bool(primary_run["breadth"][3].iloc[-1]),
            "tqqq_ma200_sleeve_in": bool(primary_run["trend"][3].iloc[-1]),
            "ndx_above_ma200": bool(df["price"].iloc[-1] > df["ma200"].iloc[-1]),
        },
    }
    RESULTS_FILE.write_text(json.dumps(_jsonable(results), indent=2), encoding="utf-8")
    write_report(results)
    print(json.dumps(_jsonable({
        "decision": decision,
        "baseline_score": baseline["score"],
        "challenger_score": primary["score"],
        "three_x_drag_score": primary_3x["score"],
        "baseline_metrics": bm,
        "challenger_metrics": cm,
        "component_correlation": correlation,
        "guardrails": guardrails,
        "parity": parity,
    }), indent=2))


def write_report(results: dict[str, Any]) -> None:
    base, challenge = results["baseline"], results["challenger"]
    bm, cm = base["metrics"], challenge["metrics"]
    bs, cs = base["score"], challenge["score"]
    drag3 = results["drag_stress"]["3x"]
    verdict = "Reject" if results["decision"] == "reject" else "Track as research challenger"
    lines = [
        "# Backtest Verification Report — Breadth + TQQQ MA200 satellite",
        "",
        f"## Verdict: {verdict}",
        "",
        f"The fixed 85/15 ensemble scores **{cs['final_score']} / 100 ({cs['band']})** versus the frozen baseline **{bs['final_score']} / 100 ({bs['band']})**.  Under 3× pre-inception drag it scores **{drag3['score']['final_score']} / 100**.",
        "",
        "## Backtest Scores",
        "",
        "| Component | Baseline | 85/15 ensemble | Max |",
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
        f"- Sensitivity scores: " + ", ".join(f"{key}={value['score']}" for key, value in results['sensitivity'].items()) + ".",
        f"- 5x-cost paired annual mean: {results['cost_stress']['5x']['paired_annualized_mean']:+.2%}.",
        f"- 3× proxy-drag paired annual mean: {drag3['paired']['annualized_mean_difference']:+.2%}; MDD {drag3['metrics']['max_drawdown']:.2%}; score {drag3['score']['final_score']}.",
        "",
        "## Bias assessment",
        "",
        "| Bias | Status | Evidence |",
        "|---|---|---|",
        "| Lookahead / frequency mismatch | Absent | Both sleeves signal on close and fill next-session open |",
        "| Survivorship | Cannot fully verify | Aggregate NDX and breadth series |",
        "| Data snooping | Present, material | At least 4,597 related trials; DSR penalty applied |",
        "| Trade independence | Adjusted | Component exits within 21 sessions counted as one event |",
        "| Costs | Included | Every component transition; 1x/2x/5x/10x stress |",
        "| Synthetic TQQQ | Present before 2010 | Actual post-inception; calibrated 1× and punitive 3× drag tested |",
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
        "The decision follows every pre-registered score, risk, cost, sensitivity, and synthetic-data guardrail.  Historical success can only justify forward tracking, not an immediate frozen-baseline change.",
        "",
        "Research evidence only; not investment advice.",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
