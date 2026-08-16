"""Causal vector filter for canonical QQQ MA200 trend re-entries.

The canonical washout entry and every exit rule remain unchanged.  The only
challenger change is that an otherwise-valid MA200-recross entry is allowed
when the expanding-history vector buy probability meets a fixed threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_vector_buy_signal as vector_buy


analytics = vector_buy.analytics
qbt = vector_buy.qbt
DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_vector_recross_filter_results.json"
DEFAULT_SIGNALS = DATA_DIR / "qqq_vector_recross_filter_signals.csv"
DEFAULT_EQUITY = DATA_DIR / "qqq_vector_recross_filter_equity.csv"
DEFAULT_TRADES = DATA_DIR / "qqq_vector_recross_filter_trades.csv"
IDEA_CARD = DATA_DIR / "docs/research/vector_ma200_recross_filter_idea.md"
PRIMARY_THRESHOLD = 0.60
SENSITIVITY_THRESHOLDS = (0.50, 0.60, 0.70)
RELATED_PRIOR_TRIALS = 570


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def run_vector_recross_filter(
    df: pd.DataFrame,
    recross_allowed: pd.Series,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    """Run one isolated entry challenger through the canonical state machine."""
    allowed = recross_allowed.reindex(df.index).fillna(False).astype(bool)
    experiment = df.copy()
    experiment["ma200_recross"] = (
        df["ma200_recross"].astype(bool) & allowed
    )
    with vector_buy._cost_override(cost_multiplier):
        return qbt.run_strategy(
            experiment,
            cooldown_days=qbt.COOLDOWN_DAYS,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
        )


def trigger_counts(
    trades: list[dict],
    open_trade: dict | None,
) -> dict[str, int]:
    triggers = [str(trade["buy_trigger"]) for trade in trades]
    if open_trade:
        triggers.append(str(open_trade["buy_trigger"]))
    counts = pd.Series(triggers, dtype="object").value_counts()
    return {
        "washout": int(
            sum(count for trigger, count in counts.items()
                if trigger != "MA200-recross")
        ),
        "ma200_recross": int(counts.get("MA200-recross", 0)),
        "all_entries": int(counts.sum()),
    }


def baseline_recross_audit(
    index: pd.DatetimeIndex,
    trades: list[dict],
    open_trade: dict | None,
    probability: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    entries = [dict(trade) for trade in trades]
    if open_trade:
        entries.append(dict(open_trade))
    rows = []
    for entry in entries:
        if entry.get("buy_trigger") != "MA200-recross":
            continue
        entry_location = index.get_loc(entry["entry_date"])
        signal_location = entry_location - qbt.EXECUTION_LAG
        signal_date = index[signal_location]
        value = float(probability.loc[signal_date])
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry["entry_date"],
                "buy_probability": value,
                "filter_passed": bool(
                    np.isfinite(value) and value >= threshold
                ),
            }
        )
    return {
        "baseline_recross_entries": len(rows),
        "passed_filter": sum(row["filter_passed"] for row in rows),
        "vetoed_by_filter": sum(not row["filter_passed"] for row in rows),
        "decisions": rows,
    }


def challenger_entry_audit(
    index: pd.DatetimeIndex,
    trades: list[dict],
    open_trade: dict | None,
    probability: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    entries = [dict(trade) for trade in trades]
    if open_trade:
        entries.append(dict(open_trade))
    rows = []
    violations = 0
    for entry in entries:
        entry_location = index.get_loc(entry["entry_date"])
        signal_location = entry_location - qbt.EXECUTION_LAG
        signal_date = index[signal_location]
        value = float(probability.loc[signal_date])
        is_recross = entry.get("buy_trigger") == "MA200-recross"
        passes = bool(np.isfinite(value) and value >= threshold)
        violations += int(is_recross and not passes)
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry["entry_date"],
                "buy_trigger": entry.get("buy_trigger"),
                "buy_probability": value,
                "recross_filter_passed": passes if is_recross else None,
            }
        )
    return {
        "entries": len(rows),
        "recross_filter_violations": violations,
        "outcomes": rows,
    }


def trade_bootstrap(
    trades: list[dict],
    simulations: int = 5000,
) -> dict[str, Any]:
    """Bootstrap completed trade returns and their trade-level drawdowns."""
    returns = np.asarray(
        [float(trade["return_pct"]) / 100 for trade in trades],
        dtype=float,
    )
    if len(returns) == 0:
        return {"completed_trades": 0, "simulations": 0}
    rng = np.random.default_rng(42)
    terminal = np.empty(simulations)
    max_drawdown = np.empty(simulations)
    for simulation in range(simulations):
        sample = rng.choice(returns, size=len(returns), replace=True)
        path = np.cumprod(1 + sample)
        path = np.concatenate(([1.0], path))
        drawdown = path / np.maximum.accumulate(path) - 1
        terminal[simulation] = path[-1] - 1
        max_drawdown[simulation] = drawdown.min()
    return {
        "completed_trades": len(returns),
        "simulations": simulations,
        "terminal_return_percentiles": dict(
            zip(
                ("p05", "p50", "p95"),
                np.percentile(terminal, [5, 50, 95]),
            )
        ),
        "max_drawdown_percentiles": dict(
            zip(
                ("p05", "p50", "p95"),
                np.percentile(max_drawdown, [5, 50, 95]),
            )
        ),
    }


def statistical_diagnostics(
    equity: pd.Series,
    metrics: dict[str, Any],
    trials: int = RELATED_PRIOR_TRIALS,
) -> dict[str, Any]:
    """Return serial-correlation and multiple-testing diagnostics."""
    returns = equity.pct_change().dropna()
    observations = len(returns)
    ljung_box_q = 0.0
    for lag in range(1, 11):
        rho = float(returns.autocorr(lag=lag))
        ljung_box_q += rho**2 / (observations - lag)
    ljung_box_q *= observations * (observations + 2)
    ljung_box_p = float(analytics.stats.chi2.sf(ljung_box_q, 10))

    effective_n = float(metrics["effective_daily_observations"])
    daily_sharpe = float(metrics["sharpe"]) / np.sqrt(252)
    skew = float(metrics["skewness"])
    kurtosis = float(metrics["excess_kurtosis"]) + 3
    sharpe_variance = (
        1
        - skew * daily_sharpe
        + ((kurtosis - 1) / 4) * daily_sharpe**2
    ) / (effective_n - 1)
    gamma = 0.5772156649
    expected_max_sharpe = np.sqrt(sharpe_variance) * (
        (1 - gamma) * analytics.stats.norm.ppf(1 - 1 / trials)
        + gamma
        * analytics.stats.norm.ppf(1 - 1 / (trials * np.e))
    )
    psr_denominator = np.sqrt(
        1
        - skew * daily_sharpe
        + ((kurtosis - 1) / 4) * daily_sharpe**2
    )
    dsr = analytics.stats.norm.cdf(
        (daily_sharpe - expected_max_sharpe)
        * np.sqrt(effective_n - 1)
        / psr_denominator
    )
    return {
        "ljung_box_lag": 10,
        "ljung_box_q": float(ljung_box_q),
        "ljung_box_p": ljung_box_p,
        "related_trials": trials,
        "expected_max_annual_sharpe": float(
            expected_max_sharpe * np.sqrt(252)
        ),
        "deflated_sharpe_probability": float(dsr),
        "adf_return_stationarity": (
            "not computed: statsmodels is unavailable; daily strategy "
            "returns, not price levels, are evaluated"
        ),
    }


def evaluate(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    labels: pd.Series,
    label_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    result = vector_buy.evaluate(
        df,
        equity,
        trades,
        open_trade,
        labels,
        label_diagnostics,
    )
    result["trigger_counts"] = trigger_counts(trades, open_trade)
    result["trade_bootstrap"] = trade_bootstrap(trades)
    result["statistical_diagnostics"] = statistical_diagnostics(
        equity, result["metrics"]
    )
    return result


def period_deltas(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for period in ("early_period", "late_period", "real_breadth_period"):
        output[period] = {
            metric: (
                challenger[period].get(metric, np.nan)
                - baseline[period].get(metric, np.nan)
            )
            for metric in ("cagr", "sharpe", "max_drawdown")
        }
    return output


def guardrail_results(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
    period_delta: dict[str, Any],
) -> dict[str, bool]:
    base_metrics = baseline["metrics"]
    challenger_metrics = challenger["metrics"]
    early_cagr = period_delta["early_period"]["cagr"]
    late_cagr = period_delta["late_period"]["cagr"]
    return {
        "primary_calmar_improved": (
            challenger_metrics["calmar"] > base_metrics["calmar"]
        ),
        "cagr_within_two_percentage_points": (
            challenger_metrics["cagr"] >= base_metrics["cagr"] - 0.02
        ),
        "max_drawdown_not_worse": (
            challenger_metrics["max_drawdown"]
            >= base_metrics["max_drawdown"]
        ),
        "completed_trades_not_increased": (
            challenger_metrics["completed_trades"]
            <= base_metrics["completed_trades"]
        ),
        "historical_cagr_effect_not_reversed": bool(
            np.isfinite(early_cagr)
            and np.isfinite(late_cagr)
            and early_cagr * late_cagr >= 0
        ),
    }


def write_trades(
    path: Path,
    variants: dict[str, tuple[list[dict], dict | None]],
) -> None:
    rows = []
    for variant, (trades, open_trade) in variants.items():
        for trade in trades:
            row = {"variant": variant, "trade_status": "closed"}
            row.update(trade)
            rows.append(row)
        if open_trade:
            row = {"variant": variant, "trade_status": "open"}
            row.update(open_trade)
            rows.append(row)
    output = pd.DataFrame(rows)
    for column in (
        "entry_date",
        "exit_date",
        "current_date",
        "cooldown_until",
    ):
        if column in output:
            output[column] = pd.to_datetime(output[column]).dt.strftime(
                "%Y-%m-%d"
            )
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--signals-output", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--equity-output", type=Path, default=DEFAULT_EQUITY)
    parser.add_argument("--trades-output", type=Path, default=DEFAULT_TRADES)
    args = parser.parse_args()

    df = qbt.load_data()
    parity = vector_buy.parity_check(df)
    if not parity["passed"]:
        raise RuntimeError(f"baseline parity failed: {parity}")

    spx = analytics.load_spx()["close"].reindex(df.index)
    market_vector = analytics.build_market_vector(df, spx)
    labels, label_diagnostics = vector_buy.forward_buy_labels(df["price"])
    risk = vector_buy.online_buy_probability(market_vector, labels)
    probability = risk["buy_probability"]

    baseline_equity, baseline_trades, baseline_open = vector_buy.baseline_run(df)
    baseline = evaluate(
        df,
        baseline_equity,
        baseline_trades,
        baseline_open,
        labels,
        label_diagnostics,
    )

    sensitivity: dict[str, Any] = {}
    equities = {"baseline": baseline_equity}
    trade_variants = {"baseline": (baseline_trades, baseline_open)}
    primary_equity: pd.Series | None = None
    primary_trades: list[dict] | None = None
    primary_open: dict | None = None

    for threshold in SENSITIVITY_THRESHOLDS:
        equity, trades, open_trade = run_vector_recross_filter(
            df,
            probability >= threshold,
        )
        name = f"{threshold:.2f}"
        details = evaluate(
            df,
            equity,
            trades,
            open_trade,
            labels,
            label_diagnostics,
        )
        details["baseline_recross_audit"] = baseline_recross_audit(
            df.index,
            baseline_trades,
            baseline_open,
            probability,
            threshold,
        )
        details["challenger_entry_audit"] = challenger_entry_audit(
            df.index,
            trades,
            open_trade,
            probability,
            threshold,
        )
        details["period_deltas_vs_baseline"] = period_deltas(
            baseline, details
        )
        sensitivity[name] = details
        equities[name] = equity
        trade_variants[name] = (trades, open_trade)
        if np.isclose(threshold, PRIMARY_THRESHOLD):
            primary_equity = equity
            primary_trades = trades
            primary_open = open_trade

    if primary_equity is None or primary_trades is None:
        raise RuntimeError("primary threshold was not evaluated")

    primary = sensitivity[f"{PRIMARY_THRESHOLD:.2f}"]
    primary["paired_inference"] = analytics.paired_hac_and_bootstrap(
        primary_equity, baseline_equity
    )
    primary["guardrails"] = guardrail_results(
        baseline,
        primary,
        primary["period_deltas_vs_baseline"],
    )

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost, _, _ = vector_buy.baseline_run(df, multiplier)
        challenger_cost, _, _ = run_vector_recross_filter(
            df,
            probability >= PRIMARY_THRESHOLD,
            multiplier,
        )
        baseline_cagr = analytics.slice_metrics(
            baseline_cost, str(df.index[0].date())
        )["cagr"]
        challenger_cagr = analytics.slice_metrics(
            challenger_cost, str(df.index[0].date())
        )["cagr"]
        cost_stress[str(multiplier)] = {
            "baseline_cagr": baseline_cagr,
            "challenger_cagr": challenger_cagr,
            "challenger_minus_baseline_cagr": (
                challenger_cagr - baseline_cagr
            ),
        }
    primary["guardrails"]["five_x_cost_improvement_retained"] = (
        cost_stress["1"]["challenger_minus_baseline_cagr"] > 0
        and cost_stress["5"]["challenger_minus_baseline_cagr"] > 0
    )
    primary["guardrails"]["all_passed"] = all(
        primary["guardrails"].values()
    )

    signal_output = market_vector.join(risk).join(label_diagnostics)
    signal_output["successful_buy_outcome"] = labels
    signal_output["raw_ma200_recross"] = df["ma200_recross"]
    for threshold in SENSITIVITY_THRESHOLDS:
        signal_output[f"recross_filter_pass_{threshold:.2f}"] = (
            probability >= threshold
        )
        signal_output[f"filtered_ma200_recross_{threshold:.2f}"] = (
            df["ma200_recross"].astype(bool)
            & (probability >= threshold)
        )
    signal_output.index.name = "Date"
    signal_output.reset_index().to_csv(args.signals_output, index=False)

    equity_output = pd.DataFrame(equities)
    equity_output["baseline_return"] = baseline_equity.pct_change()
    equity_output["challenger_0.60_return"] = primary_equity.pct_change()
    equity_output["baseline_position"] = analytics.position_series(
        df.index, baseline_trades, baseline_open
    )
    equity_output["challenger_0.60_position"] = analytics.position_series(
        df.index, primary_trades, primary_open
    )
    equity_output.index.name = "Date"
    equity_output.reset_index().to_csv(args.equity_output, index=False)
    write_trades(args.trades_output, trade_variants)

    result = {
        "idea_card": IDEA_CARD.resolve(),
        "configuration": {
            "features": list(vector_buy.FEATURE_COLUMNS),
            "buy_horizon_sessions": vector_buy.BUY_HORIZON,
            "target_terminal_return": vector_buy.TARGET_RETURN,
            "maximum_adverse_return": vector_buy.MAX_ADVERSE_RETURN,
            "neighbors": vector_buy.NEIGHBORS,
            "primary_probability_threshold": PRIMARY_THRESHOLD,
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "entry_change": (
                "retain washout; require vector threshold for MA200 recross"
            ),
            "retained_exits": [
                "bearish-divergence",
                "climax-top",
                "25% trailing-stop",
            ],
            "signal_timing": "close",
            "fill_timing": "next-session open",
            "attempted_variants_this_round": len(SENSITIVITY_THRESHOLDS),
            "related_prior_trials_for_multiplicity": RELATED_PRIOR_TRIALS,
        },
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "resolved_positive_label_rate": labels.mean(),
            "real_breadth_start": "2007-01-01",
            "clean_forward_oos_start": "2026-07-05",
            "clean_forward_oos_bars": int(
                (df.index >= pd.Timestamp("2026-07-05")).sum()
            ),
        },
        "baseline_parity": parity,
        "baseline": baseline,
        "challenger_primary": primary,
        "parameter_sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "current": {
            "date": df.index[-1],
            "buy_probability": probability.iloc[-1],
            "raw_ma200_recross": bool(df["ma200_recross"].iloc[-1]),
            "primary_filter_pass": bool(
                np.isfinite(probability.iloc[-1])
                and probability.iloc[-1] >= PRIMARY_THRESHOLD
            ),
            "filtered_ma200_recross": bool(
                df["ma200_recross"].iloc[-1]
                and np.isfinite(probability.iloc[-1])
                and probability.iloc[-1] >= PRIMARY_THRESHOLD
            ),
        },
        "bias_audit": {
            "lookahead": (
                "Absent by construction: a label becomes eligible only after "
                "its complete 126-session path is historical; close signals "
                "fill at the next-session open."
            ),
            "survivorship": (
                "Cannot fully verify constituent history because aggregate "
                "index and breadth series are used."
            ),
            "data_snooping": (
                "Present as a material research risk: 570 related prior "
                "vector configurations are counted for multiplicity."
            ),
            "transaction_costs": (
                "Included and stressed at 1x/2x/5x/10x."
            ),
            "liquidity": (
                "Low concern for QQQ at modeled size; ADV participation is "
                "not explicitly modeled."
            ),
            "frequency_alignment": (
                "Daily close probability and next-session-open fills align."
            ),
            "synthetic_breadth": (
                "Present before 2007; 2007+ results reported separately."
            ),
            "clean_forward_oos": (
                "Insufficient: observations after 2026-07-05 were already "
                "viewed during this research round and there are too few "
                "completed post-freeze trades."
            ),
        },
        "artifacts": {
            "results_json": args.result_output.resolve(),
            "signals_csv": args.signals_output.resolve(),
            "equity_csv": args.equity_output.resolve(),
            "trades_csv": args.trades_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
