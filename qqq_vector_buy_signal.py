"""Causal nearest-neighbour vector challenger replacing QQQ buy rules."""

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
DEFAULT_RESULT = DATA_DIR / "qqq_vector_buy_results.json"
DEFAULT_SIGNALS = DATA_DIR / "qqq_vector_buy_signals.csv"
DEFAULT_EQUITY = DATA_DIR / "qqq_vector_buy_equity.csv"
DEFAULT_TRADES = DATA_DIR / "qqq_vector_buy_trades.csv"
BUY_HORIZON = 126
TARGET_RETURN = 0.10
MAX_ADVERSE_RETURN = -0.15
NEIGHBORS = 15
PRIMARY_THRESHOLD = 0.60
SENSITIVITY_THRESHOLDS = (0.50, 0.60, 0.70)
FEATURE_COLUMNS = analytics.FEATURE_COLUMNS


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


def forward_buy_labels(
    ndx_close: pd.Series,
    horizon: int = BUY_HORIZON,
    target_return: float = TARGET_RETURN,
    max_adverse_return: float = MAX_ADVERSE_RETURN,
) -> tuple[pd.Series, pd.DataFrame]:
    """Label favorable six-month outcomes using strictly future closes."""
    values = ndx_close.to_numpy(dtype=float)
    terminal = np.full(len(values), np.nan)
    future_min = np.full(len(values), np.nan)
    if len(values) > horizon:
        terminal[: len(values) - horizon] = values[horizon:]
        windows = np.lib.stride_tricks.sliding_window_view(
            values[1:], horizon
        )
        future_min[: len(windows)] = windows.min(axis=1)
    terminal_return = terminal / values - 1
    adverse_return = future_min / values - 1
    resolved = np.isfinite(terminal_return) & np.isfinite(adverse_return)
    labels = np.full(len(values), np.nan)
    labels[resolved] = (
        (terminal_return[resolved] >= target_return)
        & (adverse_return[resolved] >= max_adverse_return)
    ).astype(float)
    label_series = pd.Series(
        labels,
        index=ndx_close.index,
        name="successful_buy_outcome",
    )
    diagnostics = pd.DataFrame(
        {
            "future_ndx_return_126": terminal_return,
            "future_ndx_min_return_126": adverse_return,
        },
        index=ndx_close.index,
    )
    return label_series, diagnostics


def online_buy_probability(
    vector: pd.DataFrame,
    labels: pd.Series,
) -> pd.DataFrame:
    result = analytics.online_crash_probability(
        vector,
        labels,
        horizon=BUY_HORIZON,
        neighbors=NEIGHBORS,
        feature_columns=FEATURE_COLUMNS,
    )
    return result.rename(
        columns={"crash_probability": "buy_probability"}
    )


@contextmanager
def _buy_override(
    cost_multiplier: float,
) -> Iterator[None]:
    old_buy_threshold = qbt.BUY_B200_THRESH
    old_commission = qbt.COMMISSION
    old_slippage = qbt.SLIPPAGE
    qbt.BUY_B200_THRESH = float("inf")
    qbt.COMMISSION = old_commission * cost_multiplier
    qbt.SLIPPAGE = old_slippage * cost_multiplier
    try:
        yield
    finally:
        qbt.BUY_B200_THRESH = old_buy_threshold
        qbt.COMMISSION = old_commission
        qbt.SLIPPAGE = old_slippage


@contextmanager
def _cost_override(cost_multiplier: float) -> Iterator[None]:
    old_commission = qbt.COMMISSION
    old_slippage = qbt.SLIPPAGE
    qbt.COMMISSION = old_commission * cost_multiplier
    qbt.SLIPPAGE = old_slippage * cost_multiplier
    try:
        yield
    finally:
        qbt.COMMISSION = old_commission
        qbt.SLIPPAGE = old_slippage


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


def run_vector_buy(
    df: pd.DataFrame,
    buy_signal: pd.Series,
    cost_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    """Use the canonical state machine with vector probability as the sole buy."""
    signal = buy_signal.reindex(df.index).fillna(False).astype(bool)
    experiment = df.copy()
    experiment["vote_gate"] = signal
    experiment["vix_vote"] = signal
    experiment["ma200_vote"] = False
    experiment["ma200_recross"] = False
    with _buy_override(cost_multiplier):
        equity, trades, open_trade = qbt.run_strategy(
            experiment,
            cooldown_days=qbt.COOLDOWN_DAYS,
            execution_lag=qbt.EXECUTION_LAG,
            fill_on=qbt.FILL_PRICE,
        )
    copied_trades = [dict(trade) for trade in trades]
    for trade in copied_trades:
        trade["buy_trigger"] = "VECTOR"
    copied_open = dict(open_trade) if open_trade else None
    if copied_open:
        copied_open["buy_trigger"] = "VECTOR"
    return equity, copied_trades, copied_open


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
    """Verify the harness baseline branch delegates exactly to the canonical run."""
    direct = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    harness = baseline_run(df)
    equity_equal = np.allclose(direct[0], harness[0])
    trades_equal = trade_signature(direct[1]) == trade_signature(harness[1])
    open_equal = direct[2] == harness[2]
    return {
        "engine": "canonical qqq_backtest.run_strategy in both branches",
        "equity_max_absolute_difference": float(
            np.max(np.abs(direct[0] - harness[0]))
        ),
        "trade_signatures_identical": trades_equal,
        "open_trade_identical": open_equal,
        "passed": bool(equity_equal and trades_equal and open_equal),
    }


def entry_label_outcomes(
    index: pd.DatetimeIndex,
    trades: list[dict],
    open_trade: dict | None,
    labels: pd.Series,
    diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    entries = [
        {"entry_date": trade["entry_date"], "closed": True}
        for trade in trades
    ]
    if open_trade:
        entries.append(
            {"entry_date": open_trade["entry_date"], "closed": False}
        )
    rows = []
    for entry in entries:
        entry_location = index.get_loc(entry["entry_date"])
        signal_location = entry_location - qbt.EXECUTION_LAG
        signal_date = index[signal_location]
        label = float(labels.loc[signal_date])
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": entry["entry_date"],
                "closed_trade": entry["closed"],
                "label_resolved": np.isfinite(label),
                "successful_buy_outcome": (
                    bool(label) if np.isfinite(label) else None
                ),
                "future_ndx_return_126": diagnostics.loc[
                    signal_date, "future_ndx_return_126"
                ],
                "future_ndx_min_return_126": diagnostics.loc[
                    signal_date, "future_ndx_min_return_126"
                ],
            }
        )
    resolved = [row for row in rows if row["label_resolved"]]
    successes = sum(row["successful_buy_outcome"] for row in resolved)
    return {
        "entries": len(rows),
        "resolved_entries": len(resolved),
        "successful_entries": successes,
        "resolved_success_rate": (
            successes / len(resolved) if resolved else np.nan
        ),
        "outcomes": rows,
    }


def evaluate(
    df: pd.DataFrame,
    equity: pd.Series,
    trades: list[dict],
    open_trade: dict | None,
    labels: pd.Series,
    label_diagnostics: pd.DataFrame,
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
        "entry_label_outcomes": entry_label_outcomes(
            df.index, trades, open_trade, labels, label_diagnostics
        ),
        "exit_reason_counts": pd.Series(
            [trade["sell_reason"] for trade in trades]
        ).value_counts().to_dict(),
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
    parser.add_argument("--signals-output", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--equity-output", type=Path, default=DEFAULT_EQUITY)
    parser.add_argument("--trades-output", type=Path, default=DEFAULT_TRADES)
    args = parser.parse_args()

    df = qbt.load_data()
    parity = parity_check(df)
    if not parity["passed"]:
        raise RuntimeError(f"baseline parity failed: {parity}")
    spx = analytics.load_spx()["close"].reindex(df.index)
    vector = analytics.build_market_vector(df, spx)
    labels, label_diagnostics = forward_buy_labels(df["price"])
    risk = online_buy_probability(vector, labels)
    probability = risk["buy_probability"]

    baseline_equity, baseline_trades, baseline_open = baseline_run(df)
    baseline = evaluate(
        df,
        baseline_equity,
        baseline_trades,
        baseline_open,
        labels,
        label_diagnostics,
    )

    sensitivity = {}
    equities = {"baseline": baseline_equity}
    trade_variants = {"baseline": baseline_trades}
    primary_equity = None
    primary_trades = None
    primary_open = None
    for threshold in SENSITIVITY_THRESHOLDS:
        equity, trades, open_trade = run_vector_buy(
            df, probability >= threshold
        )
        name = f"{threshold:.2f}"
        equities[name] = equity
        trade_variants[name] = trades
        sensitivity[name] = evaluate(
            df,
            equity,
            trades,
            open_trade,
            labels,
            label_diagnostics,
        )
        if np.isclose(threshold, PRIMARY_THRESHOLD):
            primary_equity = equity
            primary_trades = trades
            primary_open = open_trade
    if primary_equity is None or primary_trades is None:
        raise RuntimeError("primary threshold was not evaluated")
    primary = sensitivity["0.60"]
    primary["paired_inference"] = analytics.paired_hac_and_bootstrap(
        primary_equity, baseline_equity
    )

    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost, _, _ = baseline_run(df, multiplier)
        challenger_cost, _, _ = run_vector_buy(
            df,
            probability >= PRIMARY_THRESHOLD,
            multiplier,
        )
        cost_stress[str(multiplier)] = {
            "baseline_cagr": analytics.slice_metrics(
                baseline_cost, str(df.index[0].date())
            )["cagr"],
            "challenger_cagr": analytics.slice_metrics(
                challenger_cost, str(df.index[0].date())
            )["cagr"],
        }

    signal_output = vector.join(risk).join(label_diagnostics)
    signal_output["successful_buy_outcome"] = labels
    for threshold in SENSITIVITY_THRESHOLDS:
        signal_output[f"buy_signal_{threshold:.2f}"] = (
            probability >= threshold
        )
    signal_output.index.name = "Date"
    signal_output.reset_index().to_csv(args.signals_output, index=False)

    equity_output = pd.DataFrame(equities)
    equity_output["baseline_return"] = baseline_equity.pct_change()
    equity_output["challenger_0.60_return"] = primary_equity.pct_change()
    equity_output["baseline_position"] = analytics.position_series(
        df.index, baseline_trades, baseline_open
    )
    equity_output["challenger_0.60_position"] = (
        analytics.position_series(
            df.index, primary_trades, primary_open
        )
    )
    equity_output.index.name = "Date"
    equity_output.reset_index().to_csv(args.equity_output, index=False)
    write_trades(args.trades_output, trade_variants)

    result = {
        "idea_card": (
            DATA_DIR / "docs/research/vector_buy_signal_idea.md"
        ).resolve(),
        "configuration": {
            "features": list(FEATURE_COLUMNS),
            "buy_horizon_sessions": BUY_HORIZON,
            "target_terminal_return": TARGET_RETURN,
            "maximum_adverse_return": MAX_ADVERSE_RETURN,
            "neighbors": NEIGHBORS,
            "primary_probability_threshold": PRIMARY_THRESHOLD,
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "disabled_buy_rules": [
                "breadth washout",
                "MA200 trend recross",
            ],
            "retained_exits": [
                "bearish-divergence",
                "climax-top",
                "25% trailing-stop",
            ],
            "signal_timing": "close",
            "fill_timing": "next-session open",
            "attempted_variants": len(SENSITIVITY_THRESHOLDS),
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
            "primary_buy_signal": bool(
                probability.iloc[-1] >= PRIMARY_THRESHOLD
            ),
            "vector": {
                column: vector[column].iloc[-1]
                for column in FEATURE_COLUMNS
            },
        },
        "bias_audit": {
            "lookahead": (
                "Absent by construction: a training label enters only after "
                "its complete 126-session path is historical; fills are next "
                "session open."
            ),
            "survivorship": (
                "Cannot verify aggregate breadth constituent history; index "
                "and aggregate breadth series are used."
            ),
            "data_snooping": (
                "Material risk: the vector features and analogue mechanism "
                "were viewed in prior exit research, and extensive repository "
                "trials require a severe multiplicity penalty."
            ),
            "transaction_costs": (
                "Included and stressed at 1x/2x/5x/10x."
            ),
            "liquidity": (
                "Low concern for QQQ at modeled size, but explicit ADV "
                "participation is not modeled."
            ),
            "frequency_alignment": (
                "Daily close probability and next-session-open fills align."
            ),
            "synthetic_breadth": (
                "Present before 2007; 2007+ results reported separately."
            ),
            "clean_forward_oos": (
                "Insufficient: only 18 bars after the freeze, and the latest "
                "126 labels are unresolved."
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
