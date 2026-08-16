"""Sell-only challenger using three coordinated trajectory reversals."""

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
import qqq_vector_trajectory_sell_only as sell_harness


qbt = crash.qbt
DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_triple_trajectory_sell_results.json"
DEFAULT_SIGNALS = DATA_DIR / "qqq_triple_trajectory_sell_signals.csv"
DEFAULT_EQUITY = DATA_DIR / "qqq_triple_trajectory_sell_equity.csv"
DEFAULT_TRADES = DATA_DIR / "qqq_triple_trajectory_sell_trades.csv"
IDEA_CARD = DATA_DIR / "docs/research/triple_trajectory_reversal_sell_idea.md"
PRIMARY_WINDOW = 10
SENSITIVITY_WINDOWS = (5, 10, 20)
SLOPE_WINDOW = 20
NEAR_HIGH_DRAWDOWN_PCT = -2.0
EXIT_REASON = "triple-trajectory"
RELATED_PRIOR_TRIALS = 4588


def build_transition_features(
    df: pd.DataFrame,
    spx_close: pd.Series,
) -> pd.DataFrame:
    frame = trajectory.build_trajectory_vector(df, spx_close)
    frame["ndx_return_60_slope_20"] = (
        trajectory.rolling_linear_slope(
            frame["ndx_return_60_pct"], SLOPE_WINDOW
        )
    )
    frame["ndx_deceleration_cross"] = (
        (frame["ndx_return_60_slope_20"] < 0)
        & (frame["ndx_return_60_slope_20"].shift(1) >= 0)
    ).fillna(False)
    frame["vix_slope_up_cross"] = (
        (frame["vix_slope_20"] > 0)
        & (frame["vix_slope_20"].shift(1) <= 0)
    ).fillna(False)
    frame["drawdown_slope_down_cross"] = (
        (frame["spx_drawdown_252_slope_20"] < 0)
        & (frame["spx_drawdown_252_slope_20"].shift(1) >= 0)
    ).fillna(False)
    frame["positive_ndx_momentum_gate"] = (
        frame["ndx_return_60_pct"] > 0
    )
    frame["near_high_gate"] = (
        frame["spx_drawdown_252_pct"] >= NEAR_HIGH_DRAWDOWN_PCT
    )
    return frame


def transition_signal(
    features: pd.DataFrame,
    confirmation_window: int = PRIMARY_WINDOW,
) -> pd.Series:
    if confirmation_window < 1:
        raise ValueError("confirmation window must be positive")
    event_columns = (
        "ndx_deceleration_cross",
        "vix_slope_up_cross",
        "drawdown_slope_down_cross",
    )
    recent_events = [
        features[column]
        .astype(int)
        .rolling(confirmation_window, min_periods=1)
        .max()
        .astype(bool)
        for column in event_columns
    ]
    signal = (
        recent_events[0]
        & recent_events[1]
        & recent_events[2]
        & features["positive_ndx_momentum_gate"].astype(bool)
        & features["near_high_gate"].astype(bool)
    )
    signal.name = f"triple_trajectory_signal_{confirmation_window}"
    return signal.fillna(False)


def transition_exit_outcomes(
    index: pd.DatetimeIndex,
    trades: list[dict],
    future_return: pd.Series,
) -> dict[str, Any]:
    outcomes = []
    for trade in trades:
        if trade["sell_reason"] != EXIT_REASON:
            continue
        exit_location = index.get_loc(trade["exit_date"])
        signal_location = exit_location - qbt.EXECUTION_LAG
        signal_date = index[signal_location]
        realized = float(future_return.loc[signal_date])
        outcomes.append(
            {
                "signal_date": signal_date,
                "exit_date": trade["exit_date"],
                "future_min_spx_return_126": realized,
                "followed_by_20pct_drop": bool(realized <= -0.20),
            }
        )
    true_exits = sum(row["followed_by_20pct_drop"] for row in outcomes)
    return {
        "transition_exits": len(outcomes),
        "true_exits": true_exits,
        "false_exits": len(outcomes) - true_exits,
        "precision": true_exits / len(outcomes) if outcomes else np.nan,
        "outcomes": outcomes,
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
                    "protected_by_transition_exit": False,
                }
            )
            continue
        row = dict(matches[0])
        row["breach_year"] = year
        row["found"] = True
        row["protected_by_transition_exit"] = bool(
            row["out_at_first_20pct_breach"]
            and row["first_exit_reason"] == EXIT_REASON
        )
        rows.append(row)
    return {
        "required_years": list(years),
        "all_protected_by_transition_exit": all(
            row["protected_by_transition_exit"] for row in rows
        ),
        "episodes": rows,
    }


def evaluate(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    episodes: list[dict[str, Any]],
    future_return: pd.Series,
) -> dict[str, Any]:
    result = sell_harness.evaluate(
        df,
        equity,
        trades,
        open_trade,
        episodes,
        future_return,
    )
    result["transition_exit_outcomes"] = transition_exit_outcomes(
        df.index, trades, future_return
    )
    result["required_episode_audit"] = required_episode_audit(
        result["crash_avoidance"]
    )
    result["statistical_diagnostics"] = (
        research.statistical_diagnostics(
            equity,
            result["metrics"],
            trials=RELATED_PRIOR_TRIALS,
        )
    )
    return result


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
                "all_protected_by_transition_exit"
            ]
        ),
        "cagr_within_two_percentage_points": (
            challenger_metrics["cagr"] >= base_metrics["cagr"] - 0.02
        ),
        "max_drawdown_not_worse": (
            challenger_metrics["max_drawdown"]
            >= base_metrics["max_drawdown"]
        ),
        "turnover_increase_within_one_round_trip_per_year": (
            challenger_metrics["round_trips_per_year"]
            <= base_metrics["round_trips_per_year"] + 1
        ),
        "positive_cost_adjusted_expectancy": (
            challenger_metrics["expectancy"] > 0
        ),
        "historical_cagr_effect_not_reversed": bool(
            np.isfinite(early)
            and np.isfinite(late)
            and early * late >= 0
        ),
    }


def event_age(event: pd.Series) -> pd.Series:
    ages = np.full(len(event), np.nan)
    last_event: int | None = None
    for i, occurred in enumerate(event.fillna(False).astype(bool)):
        if occurred:
            last_event = i
        if last_event is not None:
            ages[i] = i - last_event
    return pd.Series(ages, index=event.index)


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
    buy_integrity = sell_harness.entry_logic_integrity(df)
    if not buy_integrity["passed"]:
        raise RuntimeError(f"buy inputs changed: {buy_integrity}")

    spx_full = crash.load_spx()["close"]
    spx = spx_full.reindex(df.index)
    features = build_transition_features(df, spx)
    _, future_return = crash.forward_crash_labels(spx)
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

    sensitivity: dict[str, Any] = {}
    equities = {"baseline": baseline_run[0]}
    trade_variants = {
        "baseline": (baseline_run[1], baseline_run[2])
    }
    primary_run: tuple[pd.Series, list[dict], dict | None] | None = None
    for window in SENSITIVITY_WINDOWS:
        signal = transition_signal(features, window)
        run = sell_harness.run_sell_only(
            df,
            signal,
            reason=EXIT_REASON,
        )
        details = evaluate(
            df,
            run[0],
            run[1],
            run[2],
            episodes,
            future_return,
        )
        details["confirmation_window"] = window
        details["period_deltas_vs_baseline"] = research.period_deltas(
            baseline, details
        )
        name = str(window)
        sensitivity[name] = details
        equities[f"window_{window}"] = run[0]
        trade_variants[f"window_{window}"] = (run[1], run[2])
        if window == PRIMARY_WINDOW:
            primary_run = run

    if primary_run is None:
        raise RuntimeError("primary confirmation window was not evaluated")
    primary = sensitivity[str(PRIMARY_WINDOW)]
    primary["paired_inference_vs_baseline"] = (
        crash.paired_hac_and_bootstrap(
            primary_run[0], baseline_run[0]
        )
    )
    primary["guardrails"] = guardrail_results(baseline, primary)

    primary_signal = transition_signal(features, PRIMARY_WINDOW)
    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost = sell_harness.run_sell_only(
            df,
            crash.baseline_divergence_signal(df),
            multiplier,
            reason="bearish-divergence",
        )[0]
        challenger_cost = sell_harness.run_sell_only(
            df,
            primary_signal,
            multiplier,
            reason=EXIT_REASON,
        )[0]
        baseline_cagr = crash.slice_metrics(
            baseline_cost, str(df.index[0].date())
        )["cagr"]
        challenger_cagr = crash.slice_metrics(
            challenger_cost, str(df.index[0].date())
        )["cagr"]
        cost_stress[str(multiplier)] = {
            "baseline_cagr": baseline_cagr,
            "challenger_cagr": challenger_cagr,
            "challenger_minus_baseline_cagr": (
                challenger_cagr - baseline_cagr
            ),
        }
    primary["guardrails"]["five_x_cost_benefit_retained"] = (
        cost_stress["1"]["challenger_minus_baseline_cagr"] > 0
        and cost_stress["5"]["challenger_minus_baseline_cagr"] > 0
    )
    primary["guardrails"]["all_passed"] = all(
        primary["guardrails"].values()
    )

    signal_output = features.copy()
    for column in (
        "ndx_deceleration_cross",
        "vix_slope_up_cross",
        "drawdown_slope_down_cross",
    ):
        signal_output[f"{column}_age"] = event_age(features[column])
    for window in SENSITIVITY_WINDOWS:
        signal_output[f"sell_signal_window_{window}"] = (
            transition_signal(features, window)
        )
    signal_output["baseline_bearish_divergence"] = (
        crash.baseline_divergence_signal(df)
    )
    signal_output["future_min_spx_return_126"] = future_return
    signal_output.index.name = "Date"
    signal_output.reset_index().to_csv(args.signals_output, index=False)

    equity_output = pd.DataFrame(equities)
    equity_output["baseline_return"] = baseline_run[0].pct_change()
    equity_output["challenger_return"] = primary_run[0].pct_change()
    equity_output["baseline_position"] = crash.position_series(
        df.index, baseline_run[1], baseline_run[2]
    )
    equity_output["challenger_position"] = crash.position_series(
        df.index, primary_run[1], primary_run[2]
    )
    equity_output.index.name = "Date"
    equity_output.reset_index().to_csv(args.equity_output, index=False)
    research.write_trades(args.trades_output, trade_variants)

    last = features.iloc[-1]
    result = {
        "idea_card": IDEA_CARD.resolve(),
        "configuration": {
            "ndx_momentum": "NDX trailing 60-session return",
            "slope_window_sessions": SLOPE_WINDOW,
            "primary_confirmation_window_sessions": PRIMARY_WINDOW,
            "sensitivity_windows": list(SENSITIVITY_WINDOWS),
            "near_high_drawdown_gate_pct": NEAR_HIGH_DRAWDOWN_PCT,
            "positive_ndx_momentum_gate": True,
            "crosses": {
                "ndx": "20-session momentum slope >=0 to <0",
                "vix": "20-session VIX slope <=0 to >0",
                "spx_drawdown": (
                    "20-session drawdown slope >=0 to <0"
                ),
            },
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
        "challenger_primary": primary,
        "window_sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "current": {
            "date": df.index[-1],
            "primary_sell_signal": bool(primary_signal.iloc[-1]),
            "ndx_return_60_pct": last["ndx_return_60_pct"],
            "ndx_return_60_slope_20": last[
                "ndx_return_60_slope_20"
            ],
            "vix_slope_20": last["vix_slope_20"],
            "spx_drawdown_252_pct": last[
                "spx_drawdown_252_pct"
            ],
            "spx_drawdown_252_slope_20": last[
                "spx_drawdown_252_slope_20"
            ],
            "positive_ndx_momentum_gate": bool(
                last["positive_ndx_momentum_gate"]
            ),
            "near_high_gate": bool(last["near_high_gate"]),
            "event_ages": {
                column: event_age(features[column]).iloc[-1]
                for column in (
                    "ndx_deceleration_cross",
                    "vix_slope_up_cross",
                    "drawdown_slope_down_cross",
                )
            },
        },
        "bias_audit": {
            "lookahead": (
                "Absent: every slope and cross uses the signal close or "
                "earlier; fill is next-session open."
            ),
            "hindsight_hypothesis_selection": (
                "Present: the three transitions were chosen after inspecting "
                "four historical drawdown peaks."
            ),
            "survivorship": (
                "Low for index prices; aggregate constituent history cannot "
                "be fully verified."
            ),
            "data_snooping": (
                "Severe after thousands of related prior trials."
            ),
            "transaction_costs": (
                "Included and stressed at 1x/2x/5x/10x."
            ),
            "frequency_alignment": (
                "Daily close signal and next-session-open fill."
            ),
            "clean_forward_oos": (
                "Insufficient: no meaningful post-freeze completed-trade "
                "sample."
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
