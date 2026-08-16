"""Nine-feature trajectory vector used only as a QQQ sell signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import qqq_vector_crash_exit as crash
import qqq_vector_recross_filter as research
import qqq_vector_trajectory_recross_filter as trajectory


qbt = crash.qbt
DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_vector_trajectory_sell_results.json"
DEFAULT_SIGNALS = DATA_DIR / "qqq_vector_trajectory_sell_signals.csv"
DEFAULT_EQUITY = DATA_DIR / "qqq_vector_trajectory_sell_equity.csv"
DEFAULT_TRADES = DATA_DIR / "qqq_vector_trajectory_sell_trades.csv"
IDEA_CARD = DATA_DIR / "docs/research/vector_trajectory_sell_only_idea.md"
PRIMARY_THRESHOLD = 0.50
SENSITIVITY_THRESHOLDS = (0.40, 0.50, 0.60)
RELATED_PRIOR_TRIALS = 4587
FEATURE_COLUMNS = trajectory.FEATURE_COLUMNS


def build_sell_only_frame(
    df: pd.DataFrame,
    sell_signal: pd.Series,
) -> pd.DataFrame:
    """Change only the two canonical bearish-divergence input columns."""
    signal = sell_signal.reindex(df.index).fillna(False).astype(bool)
    experiment = df.copy()
    experiment["price_rose"] = signal
    experiment["breadth_fell"] = True
    return experiment


def run_sell_only(
    df: pd.DataFrame,
    sell_signal: pd.Series,
    cost_multiplier: float = 1.0,
    reason: str = "vector-crash",
) -> tuple[pd.Series, list[dict], dict | None]:
    """Replace bearish divergence while preserving both canonical buy paths."""
    experiment = build_sell_only_frame(df, sell_signal)
    with crash._cost_override(cost_multiplier):
        equity, trades, open_trade = qbt.run_strategy(
            experiment,
            cooldown_days=qbt.COOLDOWN_DAYS,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
        )
    copied_trades = [dict(trade) for trade in trades]
    for trade in copied_trades:
        if trade["sell_reason"] == "bearish-divergence":
            trade["sell_reason"] = reason
    copied_open = dict(open_trade) if open_trade else None
    return equity, copied_trades, copied_open


def online_trajectory_crash_probability(
    vector: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    return crash.online_crash_probability(
        vector,
        labels,
        horizon=crash.CRASH_HORIZON,
        neighbors=crash.NEIGHBORS,
        feature_columns=FEATURE_COLUMNS,
    ).rename(
        columns={"crash_probability": "trajectory_crash_probability"}
    )


def evaluate(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    spx_episodes: list[dict[str, Any]],
    future_return: pd.Series,
) -> dict[str, Any]:
    position = crash.position_series(df.index, trades, open_trade)
    metrics = crash.strategy_metrics(equity, trades, position)
    return {
        "metrics": metrics,
        "crash_avoidance": crash.crash_avoidance(
            spx_episodes, position, trades
        ),
        "false_exits": crash.false_vector_exits(trades, future_return),
        "early_period": crash.slice_metrics(
            equity, "2002-01-01", "2013-12-31"
        ),
        "late_period": crash.slice_metrics(equity, "2014-01-01"),
        "real_breadth_period": crash.slice_metrics(
            equity, "2007-01-01"
        ),
        "clean_forward_slice": crash.slice_metrics(
            equity, "2026-07-05"
        ),
        "exit_reason_counts": pd.Series(
            [trade["sell_reason"] for trade in trades],
            dtype="object",
        ).value_counts().to_dict(),
        "trade_bootstrap": research.trade_bootstrap(trades),
        "statistical_diagnostics": research.statistical_diagnostics(
            equity,
            metrics,
            trials=RELATED_PRIOR_TRIALS,
        ),
    }


def required_episode_audit(
    avoidance: list[dict[str, Any]],
    years: tuple[int, ...] = (2008, 2020),
) -> dict[str, Any]:
    rows = []
    for year in years:
        matches = [
            episode
            for episode in avoidance
            if pd.Timestamp(episode["breach_date"]).year == year
        ]
        if len(matches) != 1:
            rows.append(
                {
                    "breach_year": year,
                    "found": False,
                    "protected_by_vector_exit": False,
                }
            )
            continue
        episode = dict(matches[0])
        episode["breach_year"] = year
        episode["found"] = True
        episode["protected_by_vector_exit"] = bool(
            episode["out_at_first_20pct_breach"]
            and episode["first_exit_reason"] == "vector-crash"
        )
        rows.append(episode)
    return {
        "required_years": list(years),
        "all_protected_by_vector_exit": all(
            row["protected_by_vector_exit"] for row in rows
        ),
        "episodes": rows,
    }


def guardrail_results(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
) -> dict[str, bool]:
    base_metrics = baseline["metrics"]
    challenger_metrics = challenger["metrics"]
    deltas = research.period_deltas(baseline, challenger)
    early = deltas["early_period"]["cagr"]
    late = deltas["late_period"]["cagr"]
    return {
        "required_2008_2020_episodes_protected": (
            challenger["required_episode_audit"][
                "all_protected_by_vector_exit"
            ]
        ),
        "cagr_within_two_percentage_points": (
            challenger_metrics["cagr"] >= base_metrics["cagr"] - 0.02
        ),
        "positive_cost_adjusted_expectancy": (
            challenger_metrics["expectancy"] > 0
        ),
        "max_drawdown_not_worse": (
            challenger_metrics["max_drawdown"]
            >= base_metrics["max_drawdown"]
        ),
        "turnover_increase_within_one_round_trip_per_year": (
            challenger_metrics["round_trips_per_year"]
            <= base_metrics["round_trips_per_year"] + 1
        ),
        "historical_cagr_effect_not_reversed": bool(
            np.isfinite(early)
            and np.isfinite(late)
            and early * late >= 0
        ),
    }


def entry_logic_integrity(df: pd.DataFrame) -> dict[str, Any]:
    """Verify the experiment leaves every canonical buy input unchanged."""
    experiment = build_sell_only_frame(
        df, pd.Series(False, index=df.index)
    )
    buy_columns = (
        "breadth",
        "vix_vote",
        "ma200_vote",
        "vote_gate",
        "ma200_recross",
    )
    identical = {
        column: bool(experiment[column].equals(df[column]))
        for column in buy_columns
    }
    return {
        "checked_columns": list(buy_columns),
        "identical": identical,
        "passed": all(identical.values()),
    }


def write_trades(
    path: Path,
    variants: dict[str, tuple[list[dict], dict | None]],
) -> None:
    research.write_trades(path, variants)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--signals-output", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--equity-output", type=Path, default=DEFAULT_EQUITY)
    parser.add_argument("--trades-output", type=Path, default=DEFAULT_TRADES)
    args = parser.parse_args()

    df = qbt.load_data()
    parity = crash.parity_check(df)
    if not parity["passed"]:
        raise RuntimeError(f"baseline parity failed: {parity}")
    buy_integrity = entry_logic_integrity(df)
    if not buy_integrity["passed"]:
        raise RuntimeError(f"buy inputs changed: {buy_integrity}")

    spx_full = crash.load_spx()["close"]
    spx = spx_full.reindex(df.index)
    vector = trajectory.build_trajectory_vector(df, spx)
    labels, future_return = crash.forward_crash_labels(spx)
    trajectory_risk = online_trajectory_crash_probability(vector, labels)
    trajectory_probability = trajectory_risk[
        "trajectory_crash_probability"
    ]
    static_vector = crash.build_market_vector(df, spx)
    static_risk = crash.online_crash_probability(
        static_vector, labels
    )
    static_probability = static_risk["crash_probability"]
    episodes = crash.spx_crash_episodes(spx_full)

    baseline_run = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    baseline = evaluate(
        df,
        baseline_run[0],
        baseline_run[1],
        baseline_run[2],
        episodes,
        future_return,
    )

    static_run = run_sell_only(
        df, static_probability >= PRIMARY_THRESHOLD
    )
    static_control = evaluate(
        df,
        static_run[0],
        static_run[1],
        static_run[2],
        episodes,
        future_return,
    )

    sensitivity: dict[str, Any] = {}
    equities = {
        "baseline": baseline_run[0],
        "static_0.50": static_run[0],
    }
    trade_variants = {
        "baseline": (baseline_run[1], baseline_run[2]),
        "static_0.50": (static_run[1], static_run[2]),
    }
    primary_run: tuple[pd.Series, list[dict], dict | None] | None = None
    for threshold in SENSITIVITY_THRESHOLDS:
        run = run_sell_only(
            df, trajectory_probability >= threshold
        )
        details = evaluate(
            df,
            run[0],
            run[1],
            run[2],
            episodes,
            future_return,
        )
        details["threshold"] = threshold
        details["required_episode_audit"] = required_episode_audit(
            details["crash_avoidance"]
        )
        details["period_deltas_vs_baseline"] = research.period_deltas(
            baseline, details
        )
        name = f"{threshold:.2f}"
        sensitivity[name] = details
        equities[f"trajectory_{name}"] = run[0]
        trade_variants[f"trajectory_{name}"] = (run[1], run[2])
        if np.isclose(threshold, PRIMARY_THRESHOLD):
            primary_run = run

    if primary_run is None:
        raise RuntimeError("primary threshold was not evaluated")
    primary = sensitivity[f"{PRIMARY_THRESHOLD:.2f}"]
    primary["paired_inference_vs_baseline"] = (
        crash.paired_hac_and_bootstrap(
            primary_run[0], baseline_run[0]
        )
    )
    primary["paired_inference_vs_static_vector"] = (
        crash.paired_hac_and_bootstrap(
            primary_run[0], static_run[0]
        )
    )
    primary["guardrails"] = guardrail_results(baseline, primary)

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost = run_sell_only(
            df,
            crash.baseline_divergence_signal(df),
            multiplier,
        )[0]
        trajectory_cost = run_sell_only(
            df,
            trajectory_probability >= PRIMARY_THRESHOLD,
            multiplier,
        )[0]
        baseline_cagr = crash.slice_metrics(
            baseline_cost, str(df.index[0].date())
        )["cagr"]
        trajectory_cagr = crash.slice_metrics(
            trajectory_cost, str(df.index[0].date())
        )["cagr"]
        cost_stress[str(multiplier)] = {
            "baseline_cagr": baseline_cagr,
            "trajectory_cagr": trajectory_cagr,
            "trajectory_minus_baseline_cagr": (
                trajectory_cagr - baseline_cagr
            ),
        }
    primary["guardrails"]["five_x_cost_benefit_retained"] = (
        cost_stress["1"]["trajectory_minus_baseline_cagr"] > 0
        and cost_stress["5"]["trajectory_minus_baseline_cagr"] > 0
    )
    primary["guardrails"]["all_passed"] = all(
        primary["guardrails"].values()
    )

    signal_output = vector.join(trajectory_risk)
    signal_output["static_crash_probability"] = static_probability
    signal_output["future_min_return_126"] = future_return
    signal_output["future_spx_drop_at_least_20pct"] = labels
    signal_output["baseline_bearish_divergence"] = (
        crash.baseline_divergence_signal(df)
    )
    for threshold in SENSITIVITY_THRESHOLDS:
        signal_output[f"trajectory_sell_signal_{threshold:.2f}"] = (
            trajectory_probability >= threshold
        )
    signal_output.index.name = "Date"
    signal_output.reset_index().to_csv(args.signals_output, index=False)

    equity_output = pd.DataFrame(equities)
    equity_output["baseline_return"] = baseline_run[0].pct_change()
    equity_output["static_0.50_return"] = static_run[0].pct_change()
    equity_output["trajectory_0.50_return"] = primary_run[0].pct_change()
    equity_output["baseline_position"] = crash.position_series(
        df.index, baseline_run[1], baseline_run[2]
    )
    equity_output["trajectory_0.50_position"] = crash.position_series(
        df.index, primary_run[1], primary_run[2]
    )
    equity_output.index.name = "Date"
    equity_output.reset_index().to_csv(args.equity_output, index=False)
    write_trades(args.trades_output, trade_variants)

    result = {
        "idea_card": IDEA_CARD.resolve(),
        "configuration": {
            "features": list(FEATURE_COLUMNS),
            "trajectory_features": list(
                trajectory.SLOPE_FEATURE_COLUMNS
            ),
            "slope_window_sessions": trajectory.SLOPE_WINDOW,
            "crash_horizon_sessions": crash.CRASH_HORIZON,
            "crash_threshold_from_current_close": crash.CRASH_DROP,
            "neighbors": crash.NEIGHBORS,
            "primary_probability_threshold": PRIMARY_THRESHOLD,
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "buy_rules": (
                "canonical washout and MA200-recross unchanged"
            ),
            "replaced_exit": "bearish-divergence",
            "retained_exits": ["climax-top", "25% trailing-stop"],
            "signal_timing": "close",
            "fill_timing": "next-session open",
            "related_prior_trials_for_multiplicity": (
                RELATED_PRIOR_TRIALS
            ),
        },
        "data": {
            "start": df.index[0],
            "end": df.index[-1],
            "bars": len(df),
            "real_breadth_start": "2007-01-01",
            "clean_forward_oos_start": "2026-07-05",
            "clean_forward_oos_bars": int(
                (df.index >= pd.Timestamp("2026-07-05")).sum()
            ),
        },
        "baseline_parity": parity,
        "buy_input_integrity": buy_integrity,
        "baseline": baseline,
        "static_six_feature_control": static_control,
        "challenger_primary": primary,
        "threshold_sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "current": {
            "date": df.index[-1],
            "static_crash_probability": static_probability.iloc[-1],
            "trajectory_crash_probability": (
                trajectory_probability.iloc[-1]
            ),
            "primary_sell_signal": bool(
                np.isfinite(trajectory_probability.iloc[-1])
                and trajectory_probability.iloc[-1]
                >= PRIMARY_THRESHOLD
            ),
            "trajectory": {
                column: vector[column].iloc[-1]
                for column in trajectory.SLOPE_FEATURE_COLUMNS
            },
        },
        "bias_audit": {
            "lookahead": (
                "Absent by construction: slopes are backward-looking and "
                "crash labels enter only after 126 sessions resolve."
            ),
            "survivorship": (
                "Cannot fully verify aggregate breadth constituent history."
            ),
            "data_snooping": (
                "Present as a material risk after thousands of related "
                "baseline, composite, and vector trials."
            ),
            "transaction_costs": (
                "Included and stressed at 1x/2x/5x/10x."
            ),
            "frequency_alignment": (
                "Daily close sell probability and next-session-open fill."
            ),
            "synthetic_breadth": (
                "Present before 2007; 2007+ results reported separately."
            ),
            "clean_forward_oos": (
                "Insufficient: no meaningful number of completed "
                "post-freeze trades or resolved crash labels."
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
            research._finite(result),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            research._finite(result),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
