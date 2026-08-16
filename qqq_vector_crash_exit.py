"""Causal vector challenger for S&P 500 bear-market avoidance.

This research harness leaves ``qqq_backtest.py`` unchanged. It replaces only
the canonical bearish-divergence exit with an expanding-history nearest-
neighbour estimate of a 20% S&P 500 decline within 126 sessions. Signals are
computed at the close and filled at the next session open.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from scipy import stats


_fetch_stub = types.ModuleType("fetch_investing_data")
_fetch_stub.fetch_all_updates = lambda verbose=True: None
sys.modules["fetch_investing_data"] = _fetch_stub

import qqq_backtest as qbt  # noqa: E402


DATA_DIR = Path(__file__).parent
SPX_FILE = DATA_DIR / "SPX.csv"
DEFAULT_RESULT = DATA_DIR / "qqq_vector_crash_exit_results.json"
DEFAULT_SIGNALS = DATA_DIR / "qqq_vector_crash_exit_signals.csv"
CRASH_HORIZON = 126
CRASH_DROP = -0.20
NEIGHBORS = 15
PRIMARY_THRESHOLD = 0.50
SENSITIVITY_THRESHOLDS = (0.40, 0.50, 0.60)
FEATURE_COLUMNS = (
    "spx_daily_change_pct",
    "ndx_return_60_pct",
    "breadth",
    "breadth_fall_60_points",
    "spx_drawdown_252_pct",
    "vix",
)


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


def load_spx(path: Path = SPX_FILE) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [
        column.strip().strip('"').lstrip("﻿") for column in frame.columns
    ]
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y")
    frame = frame.sort_values("Date").set_index("Date")
    for source, target in (("Price", "close"), ("Open", "open")):
        frame[target] = (
            frame[source]
            .astype(str)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
    return frame[["close", "open"]]


def build_market_vector(
    df: pd.DataFrame,
    spx_close: pd.Series,
) -> pd.DataFrame:
    """Return a causal six-feature vector on canonical strategy dates."""
    spx = spx_close.reindex(df.index)
    ndx_anchor = df["price"].shift(qbt.DIVERGENCE_WINDOW)
    breadth_anchor = df["breadth"].shift(qbt.DIVERGENCE_WINDOW)
    spx_high = spx.rolling(252, min_periods=60).max()
    vector = pd.DataFrame(
        {
            "spx_daily_change_pct": spx.pct_change(fill_method=None).mul(100),
            "ndx_return_60_pct": (df["price"] / ndx_anchor - 1).mul(100),
            "breadth": df["breadth"],
            "breadth_fall_60_points": breadth_anchor - df["breadth"],
            "spx_drawdown_252_pct": (spx / spx_high - 1).mul(100),
            "vix": df["vix"],
        },
        index=df.index,
    )
    vector.index.name = "Date"
    return vector


def forward_crash_labels(
    spx_close: pd.Series,
    horizon: int = CRASH_HORIZON,
    crash_drop: float = CRASH_DROP,
) -> tuple[pd.Series, pd.Series]:
    """Label a future 20% fall from today's close without leaking into features."""
    values = spx_close.to_numpy(dtype=float)
    future_min = np.full(len(values), np.nan)
    if len(values) > horizon:
        windows = np.lib.stride_tricks.sliding_window_view(
            values[1:], horizon
        )
        future_min[: len(windows)] = windows.min(axis=1)
    future_return = pd.Series(
        future_min / values - 1,
        index=spx_close.index,
        name="future_min_return_126",
    )
    labels = (future_return <= crash_drop).astype(float)
    labels[future_return.isna()] = np.nan
    labels.name = "future_spx_drop_at_least_20pct"
    return labels, future_return


def _robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(values, axis=0)
    q25 = np.nanpercentile(values, 25, axis=0)
    q75 = np.nanpercentile(values, 75, axis=0)
    scale = q75 - q25
    fallback = np.nanstd(values, axis=0)
    scale = np.where(scale > 0, scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return center, scale


def online_crash_probability(
    vector: pd.DataFrame,
    labels: pd.Series,
    horizon: int = CRASH_HORIZON,
    neighbors: int = NEIGHBORS,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Expanding-history monthly-independent nearest-neighbour probabilities."""
    if not feature_columns:
        raise ValueError("at least one feature is required")
    unknown = set(feature_columns).difference(vector.columns)
    if unknown:
        raise ValueError(f"unknown vector features: {sorted(unknown)}")
    features = vector.loc[:, feature_columns].to_numpy(dtype=float)
    y = labels.reindex(vector.index).to_numpy(dtype=float)
    months = vector.index.to_period("M")
    probabilities = np.full(len(vector), np.nan)
    nearest_positive = np.full(len(vector), np.nan)
    nearest_distance = np.full(len(vector), np.nan)
    effective_neighbors = np.full(len(vector), np.nan)

    valid_feature = np.isfinite(features).all(axis=1)
    for i in range(len(vector)):
        if not valid_feature[i]:
            continue
        resolved_end = i - horizon
        if resolved_end < 0:
            continue
        eligible = np.flatnonzero(
            valid_feature[: resolved_end + 1]
            & np.isfinite(y[: resolved_end + 1])
        )
        if len(eligible) < neighbors:
            continue

        train = features[eligible]
        center, scale = _robust_scale(train)
        query = (features[i] - center) / scale
        standardized = (train - center) / scale
        distances = np.sqrt(np.square(standardized - query).sum(axis=1))
        order = np.argsort(distances, kind="stable")

        chosen: list[int] = []
        seen_months: set[pd.Period] = set()
        for order_position in order:
            original_position = int(eligible[order_position])
            month = months[original_position]
            if month in seen_months:
                continue
            seen_months.add(month)
            chosen.append(int(order_position))
            if len(chosen) == neighbors:
                break
        if len(chosen) < neighbors:
            continue

        selected_distance = distances[chosen]
        selected_label = y[eligible[np.asarray(chosen)]]
        weights = 1.0 / (selected_distance + 0.25)
        weights = weights / weights.sum()
        probabilities[i] = float(np.dot(weights, selected_label))
        nearest_positive[i] = float(selected_label[0])
        nearest_distance[i] = float(selected_distance[0])
        effective_neighbors[i] = float(1.0 / np.square(weights).sum())

    return pd.DataFrame(
        {
            "crash_probability": probabilities,
            "nearest_label": nearest_positive,
            "nearest_distance": nearest_distance,
            "effective_neighbors": effective_neighbors,
        },
        index=vector.index,
    )


def baseline_divergence_signal(df: pd.DataFrame) -> pd.Series:
    signal = (
        df["price_rose"].astype(bool)
        & df["breadth_fell"].astype(bool)
        & (df["breadth"] < qbt.DIVERGENCE_BREADTH_CAP)
    )
    signal.name = "baseline_divergence_signal"
    return signal


@contextmanager
def _cost_override(
    commission_multiplier: float,
) -> Iterator[None]:
    old_commission = qbt.COMMISSION
    old_slippage = qbt.SLIPPAGE
    old_cap = qbt.DIVERGENCE_BREADTH_CAP
    qbt.COMMISSION = old_commission * commission_multiplier
    qbt.SLIPPAGE = old_slippage * commission_multiplier
    qbt.DIVERGENCE_BREADTH_CAP = float("inf")
    try:
        yield
    finally:
        qbt.COMMISSION = old_commission
        qbt.SLIPPAGE = old_slippage
        qbt.DIVERGENCE_BREADTH_CAP = old_cap


def run_replacement_exit(
    df: pd.DataFrame,
    replacement_signal: pd.Series,
    reason: str,
    commission_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    """Run the canonical state machine with one isolated divergence replacement."""
    experiment = df.copy()
    signal = replacement_signal.reindex(df.index).fillna(False).astype(bool)
    experiment["price_rose"] = signal
    experiment["breadth_fell"] = True
    with _cost_override(commission_multiplier):
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


def position_series(
    index: pd.DatetimeIndex,
    trades: list[dict],
    open_trade: dict | None,
) -> pd.Series:
    position = pd.Series(False, index=index)
    locations = {date: i for i, date in enumerate(index)}
    for trade in trades:
        start = locations[trade["entry_date"]]
        end = locations[trade["exit_date"]]
        if end > start:
            position.iloc[start:end] = True
    if open_trade:
        start = locations[open_trade["entry_date"]]
        position.iloc[start:] = True
    return position


def strategy_metrics(
    equity: pd.Series,
    trades: list[dict],
    position: pd.Series,
) -> dict[str, Any]:
    returns = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    volatility = returns.std(ddof=1) * np.sqrt(252)
    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(252)
        if returns.std(ddof=1) > 0
        else np.nan
    )
    downside = returns[returns < 0]
    downside_deviation = np.sqrt(np.mean(np.square(downside))) * np.sqrt(252)
    sortino = (
        returns.mean() * 252 / downside_deviation
        if downside_deviation > 0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    ulcer = float(np.sqrt(np.mean(np.square(drawdown))))
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    lake_ratio = float((drawdown < 0).mean())
    var95 = float(returns.quantile(0.05))
    cvar95 = float(returns[returns <= var95].mean())
    skew = float(stats.skew(returns, bias=False))
    excess_kurtosis = float(stats.kurtosis(returns, fisher=True, bias=False))
    jb = stats.jarque_bera(returns)

    autocorrelations = [
        float(returns.autocorr(lag=lag)) for lag in range(1, 11)
    ]
    positive_rho = sum(max(value, 0.0) for value in autocorrelations)
    effective_n = len(returns) / (1 + 2 * positive_rho)
    daily_std = float(returns.std(ddof=1))
    t_stat = (
        float(returns.mean() / (daily_std / np.sqrt(effective_n)))
        if daily_std > 0
        else np.nan
    )
    daily_sharpe = float(returns.mean() / daily_std) if daily_std > 0 else 0.0
    denominator = np.sqrt(
        max(
            1e-12,
            1
            - skew * daily_sharpe
            + ((excess_kurtosis + 3) - 1)
            / 4
            * daily_sharpe**2,
        )
    )
    psr = float(
        stats.norm.cdf(
            daily_sharpe * np.sqrt(max(effective_n - 1, 1)) / denominator
        )
    )

    monthly = equity.resample("ME").last().pct_change().dropna()
    rolling_mean = returns.rolling(252).mean()
    rolling_std = returns.rolling(252).std(ddof=1)
    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252)).dropna()

    trade_returns = np.asarray(
        [trade["return_pct"] / 100 for trade in trades],
        dtype=float,
    )
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    average_win = float(wins.mean()) if len(wins) else np.nan
    average_loss = float(losses.mean()) if len(losses) else np.nan
    payoff = (
        average_win / abs(average_loss)
        if np.isfinite(average_win)
        and np.isfinite(average_loss)
        and average_loss != 0
        else np.nan
    )

    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": float(cagr),
        "annual_volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_drawdown": max_drawdown,
        "ulcer_index": ulcer,
        "time_underwater": lake_ratio,
        "var_95_daily": var95,
        "cvar_95_daily": cvar95,
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera_stat": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
        "autocorrelation_lags_1_10": autocorrelations,
        "effective_daily_observations": float(effective_n),
        "mean_return_t_stat": t_stat,
        "psr_vs_zero": psr,
        "positive_months": float((monthly > 0).mean()),
        "rolling_sharpe_252_min": float(rolling_sharpe.min()),
        "rolling_sharpe_252_max": float(rolling_sharpe.max()),
        "rolling_sharpe_252_std": float(rolling_sharpe.std(ddof=1)),
        "completed_trades": len(trades),
        "win_rate": float((trade_returns > 0).mean()),
        "profit_factor": float(profit_factor),
        "expectancy": float(trade_returns.mean()),
        "average_win": average_win,
        "average_loss": average_loss,
        "payoff_ratio": float(payoff),
        "exposure": float(position.mean()),
        "round_trips_per_year": float(len(trades) / years),
    }


def slice_metrics(equity: pd.Series, start: str, end: str | None = None) -> dict:
    segment = equity.loc[start:end]
    returns = segment.pct_change().dropna()
    if len(returns) < 2:
        return {"observations": len(returns)}
    years = (segment.index[-1] - segment.index[0]).days / 365.25
    cagr = (segment.iloc[-1] / segment.iloc[0]) ** (1 / years) - 1
    daily_std = returns.std(ddof=1)
    volatility = daily_std * np.sqrt(252)
    sharpe = (
        returns.mean() / daily_std * np.sqrt(252)
        if daily_std > 0
        else np.nan
    )
    drawdown = segment / segment.cummax() - 1
    return {
        "start": segment.index[0],
        "end": segment.index[-1],
        "observations": len(returns),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "annual_volatility": float(volatility),
    }


def spx_crash_episodes(spx_close: pd.Series) -> list[dict[str, Any]]:
    """Find all-time-peak drawdown episodes that first breach minus 20%."""
    running_peak = spx_close.cummax()
    drawdown = spx_close / running_peak - 1
    episodes: list[dict[str, Any]] = []
    active = False
    for date, value in drawdown.items():
        if not active and value <= CRASH_DROP:
            peak_date = running_peak.loc[:date].idxmax()
            breach_date = date
            active = True
        if active and value >= 0:
            period = spx_close.loc[peak_date:date]
            trough_date = period.idxmin()
            episodes.append(
                {
                    "peak_date": peak_date,
                    "breach_date": breach_date,
                    "trough_date": trough_date,
                    "recovery_date": date,
                    "peak_to_trough": float(drawdown.loc[trough_date]),
                }
            )
            active = False
    if active:
        period = spx_close.loc[peak_date:]
        trough_date = period.idxmin()
        episodes.append(
            {
                "peak_date": peak_date,
                "breach_date": breach_date,
                "trough_date": trough_date,
                "recovery_date": None,
                "peak_to_trough": float(drawdown.loc[trough_date]),
            }
        )
    return episodes


def crash_avoidance(
    episodes: list[dict[str, Any]],
    position: pd.Series,
    trades: list[dict],
) -> list[dict[str, Any]]:
    results = []
    for episode in episodes:
        peak = episode["peak_date"]
        breach = episode["breach_date"]
        if peak < position.index[0] or breach > position.index[-1]:
            continue
        window = position.loc[peak:breach]
        exposed = bool(window.any())
        out_at_breach = not bool(position.loc[breach])
        exits = [
            trade
            for trade in trades
            if peak <= trade["exit_date"] <= breach
        ]
        row = dict(episode)
        row.update(
            {
                "exposed_peak_to_breach": exposed,
                "out_at_first_20pct_breach": out_at_breach,
                "avoided_while_exposed": exposed and out_at_breach,
                "first_exit_date": exits[0]["exit_date"] if exits else None,
                "first_exit_reason": exits[0]["sell_reason"] if exits else None,
            }
        )
        results.append(row)
    return results


def paired_hac_and_bootstrap(
    challenger: pd.Series,
    baseline: pd.Series,
    block_size: int = 21,
    simulations: int = 2000,
) -> dict[str, Any]:
    difference = (
        challenger.pct_change() - baseline.pct_change()
    ).dropna()
    values = difference.to_numpy(dtype=float)
    n = len(values)
    mean = float(values.mean())
    gamma_zero = float(np.mean(np.square(values - mean)))
    long_run_variance = gamma_zero
    max_lag = min(block_size, n - 1)
    for lag in range(1, max_lag + 1):
        covariance = float(
            np.mean((values[lag:] - mean) * (values[:-lag] - mean))
        )
        weight = 1 - lag / (max_lag + 1)
        long_run_variance += 2 * weight * covariance
    standard_error = np.sqrt(max(long_run_variance, 0) / n)
    t_stat = mean / standard_error if standard_error > 0 else np.nan

    rng = np.random.default_rng(42)
    starts = np.arange(n)
    bootstrap_mean = np.empty(simulations)
    blocks_needed = int(np.ceil(n / block_size))
    for simulation in range(simulations):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [
                values[(start + np.arange(block_size)) % n]
                for start in sampled_starts
            ]
        )[:n]
        bootstrap_mean[simulation] = sample.mean() * 252
    interval = np.percentile(bootstrap_mean, [2.5, 97.5])
    return {
        "annualized_mean_difference": mean * 252,
        "newey_west_lags": max_lag,
        "hac_t_stat": float(t_stat),
        "hac_two_sided_p": float(2 * stats.norm.sf(abs(t_stat))),
        "block_size": block_size,
        "bootstrap_simulations": simulations,
        "bootstrap_95_interval_annualized": interval.tolist(),
    }


def false_vector_exits(
    trades: list[dict],
    future_return: pd.Series,
) -> dict[str, Any]:
    vector_exits = [
        trade for trade in trades if trade["sell_reason"] == "vector-crash"
    ]
    outcomes = []
    for trade in vector_exits:
        exit_location = future_return.index.get_loc(trade["exit_date"])
        signal_location = exit_location - qbt.EXECUTION_LAG
        signal_date = future_return.index[signal_location]
        realized = float(future_return.loc[signal_date])
        outcomes.append(
            {
                "signal_date": signal_date,
                "exit_date": trade["exit_date"],
                "future_min_return_126": realized,
                "followed_by_20pct_drop": realized <= CRASH_DROP,
            }
        )
    false_count = sum(not row["followed_by_20pct_drop"] for row in outcomes)
    return {
        "vector_exits": len(vector_exits),
        "false_vector_exits": false_count,
        "precision": (
            1 - false_count / len(vector_exits) if vector_exits else np.nan
        ),
        "outcomes": outcomes,
    }


def parity_check(df: pd.DataFrame) -> dict[str, Any]:
    direct_equity, direct_trades, direct_open = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    harness_equity, harness_trades, harness_open = run_replacement_exit(
        df,
        baseline_divergence_signal(df),
        reason="bearish-divergence",
    )
    direct_signature = [
        (
            trade["entry_date"],
            trade["exit_date"],
            trade["sell_reason"],
            round(trade["return_pct"], 10),
        )
        for trade in direct_trades
    ]
    harness_signature = [
        (
            trade["entry_date"],
            trade["exit_date"],
            trade["sell_reason"],
            round(trade["return_pct"], 10),
        )
        for trade in harness_trades
    ]
    return {
        "equity_max_absolute_difference": float(
            np.max(np.abs(direct_equity - harness_equity))
        ),
        "trade_signatures_identical": direct_signature == harness_signature,
        "open_trade_identical": direct_open == harness_open,
        "passed": bool(
            np.allclose(direct_equity, harness_equity)
            and direct_signature == harness_signature
            and direct_open == harness_open
        ),
    }


def evaluate_threshold(
    df: pd.DataFrame,
    probability: pd.Series,
    threshold: float,
    spx_close: pd.Series,
    future_return: pd.Series,
    episodes: list[dict[str, Any]],
    commission_multiplier: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None, dict[str, Any]]:
    signal = probability >= threshold
    equity, trades, open_trade = run_replacement_exit(
        df,
        signal,
        reason="vector-crash",
        commission_multiplier=commission_multiplier,
    )
    position = position_series(df.index, trades, open_trade)
    details = {
        "threshold": threshold,
        "metrics": strategy_metrics(equity, trades, position),
        "crash_avoidance": crash_avoidance(episodes, position, trades),
        "false_exits": false_vector_exits(trades, future_return),
        "early_period": slice_metrics(equity, "2002-01-01", "2013-12-31"),
        "late_period": slice_metrics(equity, "2014-01-01"),
        "real_breadth_period": slice_metrics(equity, "2007-01-01"),
    }
    return equity, trades, open_trade, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--signals-output", type=Path, default=DEFAULT_SIGNALS)
    args = parser.parse_args()

    df = qbt.load_data()
    spx_full = load_spx()["close"]
    spx = spx_full.reindex(df.index)
    vector = build_market_vector(df, spx)
    labels, future_return = forward_crash_labels(spx)
    risk = online_crash_probability(vector, labels)
    probability = risk["crash_probability"]
    episodes = spx_crash_episodes(spx_full)

    parity = parity_check(df)
    if not parity["passed"]:
        raise RuntimeError(f"baseline parity failed: {parity}")

    baseline_equity, baseline_trades, baseline_open = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=qbt.EXECUTION_LAG,
        fill_on=qbt.FILL_PRICE,
    )
    baseline_position = position_series(
        df.index, baseline_trades, baseline_open
    )
    baseline = {
        "metrics": strategy_metrics(
            baseline_equity, baseline_trades, baseline_position
        ),
        "crash_avoidance": crash_avoidance(
            episodes, baseline_position, baseline_trades
        ),
        "early_period": slice_metrics(
            baseline_equity, "2002-01-01", "2013-12-31"
        ),
        "late_period": slice_metrics(baseline_equity, "2014-01-01"),
        "real_breadth_period": slice_metrics(
            baseline_equity, "2007-01-01"
        ),
    }

    sensitivity = {}
    primary_equity = None
    primary_details = None
    for threshold in SENSITIVITY_THRESHOLDS:
        equity, trades, open_trade, details = evaluate_threshold(
            df,
            probability,
            threshold,
            spx,
            future_return,
            episodes,
        )
        sensitivity[f"{threshold:.2f}"] = details
        if np.isclose(threshold, PRIMARY_THRESHOLD):
            primary_equity = equity
            primary_details = details
    if primary_equity is None or primary_details is None:
        raise RuntimeError("primary threshold was not evaluated")

    paired = paired_hac_and_bootstrap(primary_equity, baseline_equity)
    cost_stress = {}
    for multiplier in (1, 2, 5, 10):
        baseline_cost_equity, _, _ = run_replacement_exit(
            df,
            baseline_divergence_signal(df),
            reason="bearish-divergence",
            commission_multiplier=multiplier,
        )
        challenger_cost_equity, _, _, _ = evaluate_threshold(
            df,
            probability,
            PRIMARY_THRESHOLD,
            spx,
            future_return,
            episodes,
            commission_multiplier=multiplier,
        )
        cost_stress[str(multiplier)] = {
            "baseline_cagr": slice_metrics(
                baseline_cost_equity,
                str(df.index[0].date()),
            )["cagr"],
            "challenger_cagr": slice_metrics(
                challenger_cost_equity,
                str(df.index[0].date()),
            )["cagr"],
        }

    signal_output = vector.join(risk)
    signal_output["future_min_return_126"] = future_return
    signal_output["future_spx_drop_at_least_20pct"] = labels
    signal_output["primary_signal"] = probability >= PRIMARY_THRESHOLD
    signal_output.reset_index().to_csv(args.signals_output, index=False)

    result = {
        "idea_card": (
            DATA_DIR / "docs/research/vector_spx_crash_exit_idea.md"
        ).resolve(),
        "configuration": {
            "features": list(FEATURE_COLUMNS),
            "crash_horizon_sessions": CRASH_HORIZON,
            "crash_threshold_from_current_close": CRASH_DROP,
            "neighbors": NEIGHBORS,
            "one_nearest_state_per_calendar_month": True,
            "primary_probability_threshold": PRIMARY_THRESHOLD,
            "sensitivity_thresholds": list(SENSITIVITY_THRESHOLDS),
            "signal_timing": "close",
            "fill_timing": "next-session open",
            "replaced_exit": "bearish-divergence",
            "retained_exits": ["climax-top", "25% trailing-stop"],
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
            "independent_spx_20pct_episodes_in_strategy_sample": len(
                [
                    episode
                    for episode in episodes
                    if episode["peak_date"] >= df.index[0]
                    and episode["breach_date"] <= df.index[-1]
                ]
            ),
        },
        "baseline_parity": parity,
        "baseline": baseline,
        "challenger_primary": primary_details,
        "threshold_sensitivity": sensitivity,
        "paired_inference": paired,
        "cost_stress": cost_stress,
        "current": {
            "date": df.index[-1],
            "crash_probability": probability.iloc[-1],
            "primary_signal": bool(
                probability.iloc[-1] >= PRIMARY_THRESHOLD
            ),
            "vector": {
                column: vector[column].iloc[-1]
                for column in FEATURE_COLUMNS
            },
        },
        "bias_audit": {
            "lookahead": (
                "Absent by construction: training labels must fully resolve "
                "before each prediction date; fills occur next open."
            ),
            "survivorship": (
                "Cannot verify for breadth constituents; SPX/NDX index and "
                "aggregate breadth series are used."
            ),
            "data_snooping": (
                "Present as a material risk: extensive prior repository "
                "searches and only three independent in-sample SPX bear "
                "episodes."
            ),
            "transaction_costs": (
                "Included and stressed at 1x/2x/5x/10x."
            ),
            "liquidity": (
                "Low concern for QQQ at modeled size, but volume participation "
                "is not explicitly modeled."
            ),
            "frequency_alignment": (
                "Aligned daily close features and next-session-open fills."
            ),
            "synthetic_breadth": (
                "Present before 2007; 2007+ results reported separately."
            ),
            "clean_forward_oos": (
                "Insufficient: only the short post-2026-07-05 slice is clean "
                "and it contains no resolved 126-session label."
            ),
        },
        "artifacts": {
            "signals_csv": args.signals_output.resolve(),
            "results_json": args.result_output.resolve(),
        },
    }
    args.result_output.write_text(
        json.dumps(_finite(result), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_finite(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
