"""Isolated challenger adding price confirmation to QQQ climax exits."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import qqq_vector_crash_exit as analytics


qbt = analytics.qbt
DATA_DIR = Path(__file__).parent
DEFAULT_RESULT = DATA_DIR / "qqq_climax_confirmation_results.json"
DEFAULT_EQUITY = DATA_DIR / "qqq_climax_confirmation_equity.csv"
DEFAULT_TRADES = DATA_DIR / "qqq_climax_confirmation_trades.csv"
PRIMARY_PULLBACK_PCT = 3.0
SENSITIVITY_PULLBACK_PCT = (2.0, 3.0, 4.0)
TRAILING_HIGH_WINDOW = 10


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


def climax_confirmation_frame(
    df: pd.DataFrame,
    pullback_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require the MACD-cross close to be below its trailing ten-day high."""
    if pullback_pct < 0:
        raise ValueError("pullback confirmation cannot be negative")
    trailing_high = df["price"].rolling(
        TRAILING_HIGH_WINDOW,
        min_periods=TRAILING_HIGH_WINDOW,
    ).max()
    pullback = (df["price"] / trailing_high - 1).mul(100)
    confirmed = (
        df["macd_cross"].astype(bool) & (pullback <= -pullback_pct)
    )
    experiment = df.copy()
    experiment["macd_cross"] = confirmed
    diagnostics = pd.DataFrame(
        {
            "ndx_close": df["price"],
            "trailing_10_close_high": trailing_high,
            "pullback_from_10_close_high_pct": pullback,
            "raw_macd_cross": df["macd_cross"].astype(bool),
            "confirmed_macd_cross": confirmed,
        },
        index=df.index,
    )
    return experiment, diagnostics


@contextmanager
def _cost_override(multiplier: float) -> Iterator[None]:
    old_commission = qbt.COMMISSION
    old_slippage = qbt.SLIPPAGE
    qbt.COMMISSION = old_commission * multiplier
    qbt.SLIPPAGE = old_slippage * multiplier
    try:
        yield
    finally:
        qbt.COMMISSION = old_commission
        qbt.SLIPPAGE = old_slippage


def run_confirmation(
    df: pd.DataFrame,
    pullback_pct: float,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None, pd.DataFrame]:
    experiment, diagnostics = climax_confirmation_frame(df, pullback_pct)
    with _cost_override(cost_multiplier):
        equity, trades, open_trade = qbt.run_strategy(
            experiment,
            cooldown_days=qbt.COOLDOWN_DAYS,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
        )
    return equity, trades, open_trade, diagnostics


def baseline_run(
    df: pd.DataFrame,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    with _cost_override(cost_multiplier):
        return qbt.run_strategy(
            df,
            cooldown_days=qbt.COOLDOWN_DAYS,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
        )


def trade_signature(trades: list[dict]) -> list[tuple[Any, ...]]:
    return [
        (
            trade["entry_date"],
            trade["exit_date"],
            trade["sell_reason"],
            round(float(trade["return_pct"]), 10),
        )
        for trade in trades
    ]


def parity_check(df: pd.DataFrame) -> dict[str, Any]:
    baseline_equity, baseline_trades, baseline_open = baseline_run(df)
    parity_equity, parity_trades, parity_open, _ = run_confirmation(df, 0.0)
    maximum_difference = float(
        np.max(np.abs(baseline_equity - parity_equity))
    )
    signatures_identical = (
        trade_signature(baseline_trades) == trade_signature(parity_trades)
    )
    open_identical = baseline_open == parity_open
    passed = bool(
        np.allclose(baseline_equity, parity_equity)
        and signatures_identical
        and open_identical
    )
    return {
        "equity_max_absolute_difference": maximum_difference,
        "trade_signatures_identical": signatures_identical,
        "open_trade_identical": open_identical,
        "passed": passed,
    }


def exit_reason_counts(trades: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        reason = str(trade["sell_reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def future_path_after_climax(
    df: pd.DataFrame,
    trades: list[dict],
    horizon: int = 126,
) -> list[dict[str, Any]]:
    paths = []
    for trade in trades:
        if trade["sell_reason"] != "climax-top":
            continue
        exit_location = df.index.get_loc(trade["exit_date"])
        future = df["price"].iloc[
            exit_location : min(len(df), exit_location + horizon + 1)
        ]
        exit_price = float(trade["exit_price"])
        paths.append(
            {
                "exit_date": trade["exit_date"],
                "trade_return_pct": trade["return_pct"],
                "future_max_return_pct": (
                    future.max() / exit_price - 1
                )
                * 100,
                "future_min_return_pct": (
                    future.min() / exit_price - 1
                )
                * 100,
                "future_60_return_pct": (
                    (
                        future.iloc[60] / exit_price - 1
                        if len(future) > 60
                        else np.nan
                    )
                    * 100
                ),
                "future_126_return_pct": (
                    (
                        future.iloc[126] / exit_price - 1
                        if len(future) > 126
                        else np.nan
                    )
                    * 100
                ),
            }
        )
    return paths


def evaluate(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
) -> dict[str, Any]:
    position = analytics.position_series(df.index, trades, open_trade)
    return {
        "metrics": analytics.strategy_metrics(equity, trades, position),
        "early_period": analytics.slice_metrics(
            equity, "2002-01-01", "2013-12-31"
        ),
        "late_period": analytics.slice_metrics(equity, "2014-01-01"),
        "real_breadth_period": analytics.slice_metrics(
            equity, "2007-01-01"
        ),
        "clean_forward_slice": analytics.slice_metrics(
            equity, "2026-07-05"
        ),
        "exit_reason_counts": exit_reason_counts(trades),
        "climax_exit_future_paths": future_path_after_climax(df, trades),
    }


def write_trades(
    path: Path,
    variants: dict[str, list[dict]],
) -> None:
    rows = []
    for variant, trades in variants.items():
        for trade in trades:
            row = {"variant": variant}
            row.update(trade)
            rows.append(row)
    output = pd.DataFrame(rows)
    for column in ("entry_date", "exit_date", "cooldown_until"):
        output[column] = pd.to_datetime(output[column]).dt.strftime(
            "%Y-%m-%d"
        )
    output.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--equity-output", type=Path, default=DEFAULT_EQUITY)
    parser.add_argument("--trades-output", type=Path, default=DEFAULT_TRADES)
    args = parser.parse_args()

    df = qbt.load_data()
    parity = parity_check(df)
    if not parity["passed"]:
        raise RuntimeError(f"baseline parity failed: {parity}")

    baseline_equity, baseline_trades, baseline_open = baseline_run(df)
    baseline = evaluate(
        df, baseline_equity, baseline_trades, baseline_open
    )

    sensitivity: dict[str, Any] = {}
    equities = {"baseline": baseline_equity}
    trade_variants = {"baseline": baseline_trades}
    primary_equity = None
    primary_trades = None
    primary_open = None
    primary_diagnostics = None
    for pullback_pct in SENSITIVITY_PULLBACK_PCT:
        equity, trades, open_trade, diagnostics = run_confirmation(
            df, pullback_pct
        )
        name = f"{pullback_pct:.0f}pct"
        equities[name] = equity
        trade_variants[name] = trades
        sensitivity[name] = evaluate(df, equity, trades, open_trade)
        if np.isclose(pullback_pct, PRIMARY_PULLBACK_PCT):
            primary_equity = equity
            primary_trades = trades
            primary_open = open_trade
            primary_diagnostics = diagnostics
    if (
        primary_equity is None
        or primary_trades is None
        or primary_diagnostics is None
    ):
        raise RuntimeError("primary threshold was not evaluated")

    primary = sensitivity["3pct"]
    primary["paired_inference"] = analytics.paired_hac_and_bootstrap(
        primary_equity, baseline_equity
    )
    primary_position = analytics.position_series(
        df.index, primary_trades, primary_open
    )

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost, _, _ = baseline_run(df, multiplier)
        challenger_cost, _, _, _ = run_confirmation(
            df, PRIMARY_PULLBACK_PCT, multiplier
        )
        cost_stress[str(multiplier)] = {
            "baseline_cagr": analytics.slice_metrics(
                baseline_cost, str(df.index[0].date())
            )["cagr"],
            "challenger_cagr": analytics.slice_metrics(
                challenger_cost, str(df.index[0].date())
            )["cagr"],
        }

    equity_output = pd.DataFrame(equities)
    equity_output["baseline_return"] = baseline_equity.pct_change()
    equity_output["challenger_3pct_return"] = primary_equity.pct_change()
    equity_output["baseline_position"] = analytics.position_series(
        df.index, baseline_trades, baseline_open
    )
    equity_output["challenger_3pct_position"] = primary_position
    equity_output.index.name = "Date"
    equity_output.reset_index().to_csv(args.equity_output, index=False)
    write_trades(args.trades_output, trade_variants)

    for pullback_pct in SENSITIVITY_PULLBACK_PCT:
        _, diagnostics = climax_confirmation_frame(df, pullback_pct)
        primary_diagnostics[
            f"confirmed_macd_cross_{pullback_pct:.0f}pct"
        ] = diagnostics["confirmed_macd_cross"]
    signal_dates = primary_diagnostics[
        [
            "ndx_close",
            "trailing_10_close_high",
            "pullback_from_10_close_high_pct",
            "raw_macd_cross",
            "confirmed_macd_cross_2pct",
            "confirmed_macd_cross_3pct",
            "confirmed_macd_cross_4pct",
        ]
    ]
    signal_dates.index.name = "Date"
    signal_path = args.equity_output.with_name(
        "qqq_climax_confirmation_signals.csv"
    )
    signal_dates.reset_index().to_csv(signal_path, index=False)

    result = {
        "idea_card": (
            DATA_DIR
            / "docs/research/climax_price_confirmation_idea.md"
        ).resolve(),
        "configuration": {
            "primary_pullback_confirmation_pct": PRIMARY_PULLBACK_PCT,
            "sensitivity_pullback_pct": list(
                SENSITIVITY_PULLBACK_PCT
            ),
            "trailing_high_window_sessions": TRAILING_HIGH_WINDOW,
            "unchanged_climax_extension_pct": qbt.EXT10_PCT,
            "unchanged_climax_vote_window": qbt.CLIMAX_VOTE_WINDOW,
            "signal_timing": "close",
            "fill_timing": "next-session open",
            "retained_exits": [
                "bearish-divergence",
                "25% trailing-stop",
            ],
            "attempted_variants": len(SENSITIVITY_PULLBACK_PCT),
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
        "baseline": baseline,
        "challenger_primary": primary,
        "parameter_sensitivity": sensitivity,
        "cost_stress": cost_stress,
        "current": {
            "date": df.index[-1],
            "pullback_from_10_close_high_pct": primary_diagnostics[
                "pullback_from_10_close_high_pct"
            ].iloc[-1],
            "raw_macd_cross": primary_diagnostics[
                "raw_macd_cross"
            ].iloc[-1],
            "confirmed_macd_cross_3pct": primary_diagnostics[
                "confirmed_macd_cross_3pct"
            ].iloc[-1],
        },
        "bias_audit": {
            "lookahead": (
                "Absent: all confirmation inputs end at the signal close and "
                "fills occur at the next session open."
            ),
            "survivorship": (
                "Cannot verify aggregate breadth constituent history; index "
                "and breadth series are used."
            ),
            "data_snooping": (
                "Material risk due to extensive prior repository trials; this "
                "three-value sensitivity set is additionally counted."
            ),
            "transaction_costs": (
                "Included and stressed at 1x/2x/5x/10x."
            ),
            "liquidity": (
                "Low concern for QQQ at modeled size, but explicit ADV "
                "participation is not modeled."
            ),
            "frequency_alignment": (
                "Daily close confirmation and next-session-open fills align."
            ),
            "synthetic_breadth": (
                "Present before 2007; 2007+ results reported separately."
            ),
            "clean_forward_oos": (
                "Insufficient: only 18 bars after the freeze and no completed "
                "trade in that slice."
            ),
        },
        "artifacts": {
            "results_json": args.result_output.resolve(),
            "equity_csv": args.equity_output.resolve(),
            "trades_csv": args.trades_output.resolve(),
            "signals_csv": signal_path.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
