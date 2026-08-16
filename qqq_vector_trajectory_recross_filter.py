"""Twenty-session trajectory vector for canonical MA200 trend re-entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_vector_recross_filter as static_filter


vector_buy = static_filter.vector_buy
analytics = static_filter.analytics
qbt = static_filter.qbt
DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_vector_trajectory_recross_results.json"
DEFAULT_SIGNALS = DATA_DIR / "qqq_vector_trajectory_recross_signals.csv"
DEFAULT_EQUITY = DATA_DIR / "qqq_vector_trajectory_recross_equity.csv"
DEFAULT_TRADES = DATA_DIR / "qqq_vector_trajectory_recross_trades.csv"
IDEA_CARD = DATA_DIR / "docs/research/vector_trajectory_recross_filter_idea.md"
SLOPE_WINDOW = 20
PRIMARY_THRESHOLD = 0.60
SENSITIVITY_THRESHOLDS = (0.50, 0.60, 0.70)
RELATED_PRIOR_TRIALS = 573
SLOPE_FEATURE_COLUMNS = (
    "vix_slope_20",
    "breadth_slope_20",
    "spx_drawdown_252_slope_20",
)
FEATURE_COLUMNS = vector_buy.FEATURE_COLUMNS + SLOPE_FEATURE_COLUMNS


def rolling_linear_slope(
    series: pd.Series,
    window: int = SLOPE_WINDOW,
) -> pd.Series:
    """Causal rolling OLS slope using the current and prior window-1 rows."""
    if window < 2:
        raise ValueError("slope window must be at least two")
    x = np.arange(window, dtype=float)
    x -= x.mean()
    denominator = float(np.dot(x, x))

    def slope(values: np.ndarray) -> float:
        return float(np.dot(x, values) / denominator)

    return series.astype(float).rolling(
        window,
        min_periods=window,
    ).apply(slope, raw=True)


def build_trajectory_vector(
    df: pd.DataFrame,
    spx_close: pd.Series,
) -> pd.DataFrame:
    """Append three causal twenty-session slopes to the frozen six features."""
    vector = analytics.build_market_vector(df, spx_close)
    vector["vix_slope_20"] = rolling_linear_slope(
        vector["vix"], SLOPE_WINDOW
    )
    vector["breadth_slope_20"] = rolling_linear_slope(
        vector["breadth"], SLOPE_WINDOW
    )
    vector["spx_drawdown_252_slope_20"] = rolling_linear_slope(
        vector["spx_drawdown_252_pct"], SLOPE_WINDOW
    )
    return vector


def online_trajectory_probability(
    vector: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    result = analytics.online_crash_probability(
        vector,
        labels,
        horizon=vector_buy.BUY_HORIZON,
        neighbors=vector_buy.NEIGHBORS,
        feature_columns=FEATURE_COLUMNS,
    )
    return result.rename(
        columns={"crash_probability": "trajectory_buy_probability"}
    )


def exact_equity_parity(
    baseline: tuple[pd.Series, list[dict], dict | None],
    challenger: tuple[pd.Series, list[dict], dict | None],
) -> dict[str, Any]:
    equity_difference = float(
        np.max(np.abs(baseline[0] - challenger[0]))
    )
    trades_equal = (
        vector_buy.trade_signature(baseline[1])
        == vector_buy.trade_signature(challenger[1])
    )
    open_equal = baseline[2] == challenger[2]
    return {
        "equity_max_absolute_difference": equity_difference,
        "trade_signatures_identical": trades_equal,
        "open_trade_identical": open_equal,
        "passed": bool(
            np.allclose(baseline[0], challenger[0])
            and trades_equal
            and open_equal
        ),
    }


def evaluate_variant(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    labels: pd.Series,
    label_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    result = static_filter.evaluate(
        df,
        equity,
        trades,
        open_trade,
        labels,
        label_diagnostics,
    )
    result["statistical_diagnostics"] = (
        static_filter.statistical_diagnostics(
            equity,
            result["metrics"],
            trials=RELATED_PRIOR_TRIALS,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--signals-output", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--equity-output", type=Path, default=DEFAULT_EQUITY)
    parser.add_argument("--trades-output", type=Path, default=DEFAULT_TRADES)
    args = parser.parse_args()

    df = qbt.load_data()
    baseline_parity = vector_buy.parity_check(df)
    if not baseline_parity["passed"]:
        raise RuntimeError(f"baseline parity failed: {baseline_parity}")

    spx = analytics.load_spx()["close"].reindex(df.index)
    static_vector = analytics.build_market_vector(df, spx)
    trajectory_vector = build_trajectory_vector(df, spx)
    labels, label_diagnostics = vector_buy.forward_buy_labels(df["price"])
    static_risk = vector_buy.online_buy_probability(static_vector, labels)
    static_probability = static_risk["buy_probability"]
    trajectory_risk = online_trajectory_probability(
        trajectory_vector, labels
    )
    trajectory_probability = trajectory_risk[
        "trajectory_buy_probability"
    ]

    baseline_run = vector_buy.baseline_run(df)
    all_true_run = static_filter.run_vector_recross_filter(
        df, pd.Series(True, index=df.index)
    )
    all_true_parity = exact_equity_parity(baseline_run, all_true_run)
    if not all_true_parity["passed"]:
        raise RuntimeError(f"all-true filter parity failed: {all_true_parity}")

    baseline_equity, baseline_trades, baseline_open = baseline_run
    baseline = evaluate_variant(
        df,
        baseline_equity,
        baseline_trades,
        baseline_open,
        labels,
        label_diagnostics,
    )

    static_run = static_filter.run_vector_recross_filter(
        df, static_probability >= PRIMARY_THRESHOLD
    )
    static_control = evaluate_variant(
        df,
        static_run[0],
        static_run[1],
        static_run[2],
        labels,
        label_diagnostics,
    )

    sensitivity: dict[str, Any] = {}
    equities = {
        "baseline": baseline_equity,
        "static_0.60": static_run[0],
    }
    trade_variants = {
        "baseline": (baseline_trades, baseline_open),
        "static_0.60": (static_run[1], static_run[2]),
    }
    primary_run: tuple[pd.Series, list[dict], dict | None] | None = None

    for threshold in SENSITIVITY_THRESHOLDS:
        run = static_filter.run_vector_recross_filter(
            df, trajectory_probability >= threshold
        )
        name = f"{threshold:.2f}"
        details = evaluate_variant(
            df,
            run[0],
            run[1],
            run[2],
            labels,
            label_diagnostics,
        )
        details["baseline_recross_audit"] = (
            static_filter.baseline_recross_audit(
                df.index,
                baseline_trades,
                baseline_open,
                trajectory_probability,
                threshold,
            )
        )
        details["challenger_entry_audit"] = (
            static_filter.challenger_entry_audit(
                df.index,
                run[1],
                run[2],
                trajectory_probability,
                threshold,
            )
        )
        details["period_deltas_vs_baseline"] = (
            static_filter.period_deltas(baseline, details)
        )
        sensitivity[name] = details
        equities[f"trajectory_{name}"] = run[0]
        trade_variants[f"trajectory_{name}"] = (run[1], run[2])
        if np.isclose(threshold, PRIMARY_THRESHOLD):
            primary_run = run

    if primary_run is None:
        raise RuntimeError("primary threshold was not evaluated")
    primary = sensitivity[f"{PRIMARY_THRESHOLD:.2f}"]
    primary["paired_inference_vs_baseline"] = (
        analytics.paired_hac_and_bootstrap(
            primary_run[0], baseline_equity
        )
    )
    primary["paired_inference_vs_static_vector"] = (
        analytics.paired_hac_and_bootstrap(
            primary_run[0], static_run[0]
        )
    )
    primary["guardrails"] = static_filter.guardrail_results(
        baseline,
        primary,
        primary["period_deltas_vs_baseline"],
    )

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost = vector_buy.baseline_run(df, multiplier)[0]
        trajectory_cost = static_filter.run_vector_recross_filter(
            df,
            trajectory_probability >= PRIMARY_THRESHOLD,
            multiplier,
        )[0]
        baseline_cagr = analytics.slice_metrics(
            baseline_cost, str(df.index[0].date())
        )["cagr"]
        challenger_cagr = analytics.slice_metrics(
            trajectory_cost, str(df.index[0].date())
        )["cagr"]
        cost_stress[str(multiplier)] = {
            "baseline_cagr": baseline_cagr,
            "trajectory_cagr": challenger_cagr,
            "trajectory_minus_baseline_cagr": (
                challenger_cagr - baseline_cagr
            ),
        }
    primary["guardrails"]["five_x_cost_improvement_retained"] = (
        cost_stress["1"]["trajectory_minus_baseline_cagr"] > 0
        and cost_stress["5"]["trajectory_minus_baseline_cagr"] > 0
    )
    primary["guardrails"]["all_passed"] = all(
        primary["guardrails"].values()
    )

    signal_output = trajectory_vector.join(trajectory_risk)
    signal_output["static_buy_probability"] = static_probability
    signal_output = signal_output.join(label_diagnostics)
    signal_output["successful_buy_outcome"] = labels
    signal_output["raw_ma200_recross"] = df["ma200_recross"]
    for threshold in SENSITIVITY_THRESHOLDS:
        signal_output[f"trajectory_filter_pass_{threshold:.2f}"] = (
            trajectory_probability >= threshold
        )
        signal_output[f"filtered_ma200_recross_{threshold:.2f}"] = (
            df["ma200_recross"].astype(bool)
            & (trajectory_probability >= threshold)
        )
    signal_output.index.name = "Date"
    signal_output.reset_index().to_csv(args.signals_output, index=False)

    equity_output = pd.DataFrame(equities)
    equity_output["baseline_return"] = baseline_equity.pct_change()
    equity_output["static_0.60_return"] = static_run[0].pct_change()
    equity_output["trajectory_0.60_return"] = primary_run[0].pct_change()
    equity_output["baseline_position"] = analytics.position_series(
        df.index, baseline_trades, baseline_open
    )
    equity_output["trajectory_0.60_position"] = analytics.position_series(
        df.index, primary_run[1], primary_run[2]
    )
    equity_output.index.name = "Date"
    equity_output.reset_index().to_csv(args.equity_output, index=False)
    static_filter.write_trades(args.trades_output, trade_variants)

    result = {
        "idea_card": IDEA_CARD.resolve(),
        "configuration": {
            "base_features": list(vector_buy.FEATURE_COLUMNS),
            "trajectory_features": list(SLOPE_FEATURE_COLUMNS),
            "all_features": list(FEATURE_COLUMNS),
            "slope_window_sessions": SLOPE_WINDOW,
            "slope_method": (
                "OLS slope over signal close and prior 19 sessions"
            ),
            "neighbors": vector_buy.NEIGHBORS,
            "primary_probability_threshold": PRIMARY_THRESHOLD,
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "entry_change": (
                "retain washout; require nine-feature vector threshold "
                "for MA200 recross"
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
            "clean_forward_oos_start": "2026-07-05",
            "clean_forward_oos_bars": int(
                (df.index >= pd.Timestamp("2026-07-05")).sum()
            ),
        },
        "baseline_parity": baseline_parity,
        "all_true_filter_parity": all_true_parity,
        "baseline": baseline,
        "static_six_feature_control": static_control,
        "challenger_primary": primary,
        "parameter_sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "current": {
            "date": df.index[-1],
            "static_buy_probability": static_probability.iloc[-1],
            "trajectory_buy_probability": trajectory_probability.iloc[-1],
            "raw_ma200_recross": bool(df["ma200_recross"].iloc[-1]),
            "trajectory_filter_pass": bool(
                np.isfinite(trajectory_probability.iloc[-1])
                and trajectory_probability.iloc[-1] >= PRIMARY_THRESHOLD
            ),
            "trajectory": {
                column: trajectory_vector[column].iloc[-1]
                for column in SLOPE_FEATURE_COLUMNS
            },
        },
        "bias_audit": {
            "lookahead": (
                "Absent by construction: each slope uses only the current "
                "and prior 19 sessions; resolved labels are embargoed for "
                "the complete 126-session horizon."
            ),
            "survivorship": (
                "Cannot fully verify constituent history because aggregate "
                "index and breadth series are used."
            ),
            "data_snooping": (
                "Present as a material risk: at least 573 related vector "
                "variants are counted."
            ),
            "transaction_costs": (
                "Included and stressed at 1x/2x/5x/10x."
            ),
            "frequency_alignment": (
                "Daily close trajectory and probability feed a "
                "next-session-open fill."
            ),
            "synthetic_breadth": (
                "Present before 2007; 2007+ results reported separately."
            ),
            "clean_forward_oos": (
                "Insufficient: only a small number of post-freeze bars and "
                "no meaningful set of completed forward trades."
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
        json.dumps(
            static_filter._finite(result),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            static_filter._finite(result),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
