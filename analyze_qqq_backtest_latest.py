#!/usr/bin/env python3
"""Reproducible statistical audit of qqq_backtest.py on the latest breadth data."""

from __future__ import annotations

import itertools
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

import qqq_backtest as qbt


ROOT = Path(__file__).resolve().parent
EQUITY_FILE = ROOT / "qqq_backtest_latest_equity.csv"
TRADES_FILE = ROOT / "qqq_backtest_latest_trades.csv"
RESULTS_FILE = ROOT / "qqq_backtest_latest_analysis.json"
RNG_SEED = 20260817
TRIALS_ASSUMED = 1_000

BASE_PARAMS = {
    "buy_thresh": qbt.BUY_B200_THRESH,
    "vix_thresh": qbt.VIX_BUY_THRESH,
    "div_window": qbt.DIVERGENCE_WINDOW,
    "price_rise": qbt.DIVERGENCE_PRICE_RISE,
    "breadth_fall": qbt.DIVERGENCE_BREADTH_FALL,
    "breadth_cap": qbt.DIVERGENCE_BREADTH_CAP,
    "ext10": qbt.EXT10_PCT,
    "climax_window": qbt.CLIMAX_VOTE_WINDOW,
    "trailing_stop": qbt.TRAILING_STOP_PCT,
    "cooldown_days": qbt.COOLDOWN_DAYS,
}


@contextmanager
def qbt_parameters(params: dict[str, float]):
    mapping = {
        "buy_thresh": "BUY_B200_THRESH",
        "vix_thresh": "VIX_BUY_THRESH",
        "div_window": "DIVERGENCE_WINDOW",
        "price_rise": "DIVERGENCE_PRICE_RISE",
        "breadth_fall": "DIVERGENCE_BREADTH_FALL",
        "breadth_cap": "DIVERGENCE_BREADTH_CAP",
        "ext10": "EXT10_PCT",
        "climax_window": "CLIMAX_VOTE_WINDOW",
        "trailing_stop": "TRAILING_STOP_PCT",
    }
    original = {attr: getattr(qbt, attr) for attr in mapping.values()}
    try:
        for key, attr in mapping.items():
            setattr(qbt, attr, params[key])
        yield
    finally:
        for attr, value in original.items():
            setattr(qbt, attr, value)


def prepare_features(df: pd.DataFrame, params: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    price = out["price"]
    breadth = out["breadth"]
    window = int(params["div_window"])
    out["vix_vote"] = out["vix"].isna() | (out["vix"] > params["vix_thresh"])
    out["ma200_vote"] = out["ma200"].isna() | (price > out["ma200"])
    out["vote_gate"] = out["vix_vote"] | out["ma200_vote"]
    out["price_rose"] = (
        (price / price.shift(window) - 1) * 100 >= params["price_rise"]
    ).fillna(False)
    out["breadth_fell"] = (
        breadth.shift(window) - breadth >= params["breadth_fall"]
    ).fillna(False)
    out["ext10"] = (
        price / price.rolling(10).mean() - 1 >= params["ext10"] / 100
    ).fillna(False)
    return out


def run_fast(
    base_df: pd.DataFrame,
    params: dict[str, float],
    cost_multiple: float = 1.0,
) -> tuple[pd.Series, list[float], np.ndarray]:
    """Equivalent next-open engine used for repeated robustness runs."""
    df = prepare_features(base_df, params)
    dates = df.index
    price = df["price"].to_numpy(float)
    opens = df["open"].fillna(df["price"]).to_numpy(float)
    breadth = df["breadth"].to_numpy(float)
    vote_gate = df["vote_gate"].to_numpy(bool)
    price_rose = df["price_rose"].to_numpy(bool)
    breadth_fell = df["breadth_fell"].to_numpy(bool)
    macd_cross = df["macd_cross"].to_numpy(bool)
    ext10 = df["ext10"].to_numpy(bool)
    ma200_recross = df["ma200_recross"].to_numpy(bool)

    commission = qbt.COMMISSION * cost_multiple
    slippage = qbt.SLIPPAGE * cost_multiple
    portfolio = qbt.INITIAL_CAPITAL
    values = np.empty(len(df), dtype=float)
    positions = np.zeros(len(df), dtype=bool)
    trade_returns: list[float] = []
    in_market = False
    pending: tuple[str, int, str] | None = None
    eff_entry = 0.0
    trade_high = 0.0
    cooldown_until: pd.Timestamp | None = None
    last_sell_reason: str | None = None
    last_exit_price: float | None = None
    macd_age = ext_age = 10**9

    for i, date in enumerate(dates):
        executed = False
        if pending is not None and pending[1] == i:
            action, _, reason = pending
            if action == "BUY" and not in_market:
                portfolio -= commission
                eff_entry = opens[i] * (1 + slippage)
                trade_high = opens[i]
                macd_age = ext_age = 10**9
                in_market = True
                executed = True
            elif action == "SELL" and in_market:
                eff_exit = opens[i] * (1 - slippage)
                trade_return = eff_exit / eff_entry - 1
                portfolio *= 1 + trade_return
                portfolio -= commission
                trade_returns.append(trade_return)
                cooldown_until = date + pd.Timedelta(days=int(params["cooldown_days"]))
                last_sell_reason = reason
                last_exit_price = opens[i]
                in_market = False
                executed = True
            pending = None

        if not executed:
            if not in_market:
                cooldown_ok = cooldown_until is None or date > cooldown_until
                washout = breadth[i] < params["buy_thresh"] and vote_gate[i]
                recross_ok = last_sell_reason == "climax-top" or (
                    last_exit_price is not None and price[i] > last_exit_price
                )
                trend_buy = ma200_recross[i] and recross_ok
                if cooldown_ok and (washout or trend_buy) and i + 1 < len(df):
                    pending = ("BUY", i + 1, "washout" if washout else "MA200-recross")
            else:
                trade_high = max(trade_high, price[i])
                macd_age = 0 if macd_cross[i] else macd_age + 1
                ext_age = 0 if ext10[i] else ext_age + 1
                bearish_div = (
                    price_rose[i]
                    and breadth_fell[i]
                    and breadth[i] < params["breadth_cap"]
                )
                climax = (
                    macd_age < params["climax_window"]
                    and ext_age < params["climax_window"]
                )
                trail = price[i] <= trade_high * (1 - params["trailing_stop"] / 100)
                reason = (
                    "bearish-divergence" if bearish_div
                    else "climax-top" if climax
                    else "trailing-stop" if trail
                    else ""
                )
                if reason and i + 1 < len(df):
                    pending = ("SELL", i + 1, reason)

        if in_market:
            values[i] = portfolio * price[i] * (1 - slippage) / eff_entry
            positions[i] = True
        else:
            values[i] = portfolio

    return pd.Series(values, index=dates, name="strategy"), trade_returns, positions


def basic_metrics(values: pd.Series) -> dict[str, float]:
    returns = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    drawdown = values / values.cummax() - 1
    downside = np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0) ** 2)) * np.sqrt(252)
    annual_return = returns.mean() * 252
    ulcer = float(np.sqrt(np.mean(drawdown.to_numpy() ** 2)))
    return {
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1),
        "cagr": float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1),
        "annual_volatility": float(returns.std() * np.sqrt(252)),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)),
        "sortino": float(annual_return / downside),
        "max_drawdown": float(drawdown.min()),
        "calmar": float(
            ((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) / abs(drawdown.min())
        ),
        "ulcer_index": ulcer,
        "pain_ratio": float(
            ((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1) / ulcer
        ),
        "lake_ratio": float((drawdown < 0).mean()),
        "omega": float(returns.clip(lower=0).sum() / -returns.clip(upper=0).sum()),
        "var_95": float(-returns.quantile(0.05)),
        "var_99": float(-returns.quantile(0.01)),
        "cvar_95": float(-returns[returns <= returns.quantile(0.05)].mean()),
        "cvar_99": float(-returns[returns <= returns.quantile(0.01)].mean()),
    }


def drawdown_episodes(values: pd.Series) -> dict[str, float | int]:
    dd = values / values.cummax() - 1
    depths: list[float] = []
    durations: list[int] = []
    start: pd.Timestamp | None = None
    trough = 0.0
    for date, value in dd.items():
        if value < 0 and start is None:
            start, trough = date, value
        elif value < 0:
            trough = min(trough, value)
        elif start is not None:
            depths.append(trough)
            durations.append((date - start).days)
            start = None
    if start is not None:
        depths.append(trough)
        durations.append((dd.index[-1] - start).days)
    return {
        "episode_count": len(depths),
        "average_depth": float(np.mean(depths)),
        "average_duration_calendar_days": float(np.mean(durations)),
        "maximum_duration_calendar_days": int(max(durations)),
    }


def statistical_tests(returns: pd.Series) -> dict[str, object]:
    arr = returns.to_numpy(float)
    n = len(arr)
    skew = float(stats.skew(arr, bias=False))
    excess_kurtosis = float(stats.kurtosis(arr, fisher=True, bias=False))
    jb = stats.jarque_bera(arr)
    lb = acorr_ljungbox(arr, lags=[5, 10, 20], return_df=True)
    autocorr = [float(returns.autocorr(lag)) for lag in range(1, 21)]
    variance_inflation = max(0.05, 1 + 2 * sum(autocorr))
    n_eff = min(float(n), float(n / variance_inflation))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    t_naive = mean / (std / np.sqrt(n))
    t_eff = mean / (std / np.sqrt(n_eff))
    p_naive = float(2 * stats.t.sf(abs(t_naive), n - 1))
    p_eff = float(2 * stats.t.sf(abs(t_eff), max(1, int(n_eff) - 1)))
    daily_sr = mean / std
    pearson_kurtosis = excess_kurtosis + 3
    psr_denom = np.sqrt(
        (1 - skew * daily_sr + (pearson_kurtosis - 1) / 4 * daily_sr**2)
        / (n_eff - 1)
    )
    psr = float(stats.norm.cdf(daily_sr / psr_denom))
    gamma = np.euler_gamma
    sr_std_null = 1 / np.sqrt(n_eff - 1)
    expected_max_daily_sr = sr_std_null * (
        (1 - gamma) * stats.norm.ppf(1 - 1 / TRIALS_ASSUMED)
        + gamma * stats.norm.ppf(1 - 1 / (TRIALS_ASSUMED * np.e))
    )
    dsr = float(stats.norm.cdf((daily_sr - expected_max_daily_sr) / psr_denom))
    adf = adfuller(arr, autolag="AIC")
    return {
        "observations": n,
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera_stat": float(jb.statistic),
        "jarque_bera_p": float(jb.pvalue),
        "ljung_box_p": {str(i): float(lb.loc[i, "lb_pvalue"]) for i in lb.index},
        "effective_sample_size_20_lags": n_eff,
        "t_stat_naive": float(t_naive),
        "p_value_naive": p_naive,
        "t_stat_effective": float(t_eff),
        "p_value_effective": p_eff,
        "psr_vs_zero": psr,
        "dsr_1000_trials": dsr,
        "dsr_benchmark_annual_sharpe": float(expected_max_daily_sr * np.sqrt(252)),
        "adf_stat": float(adf[0]),
        "adf_p": float(adf[1]),
    }


def benchmark_regression(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> dict[str, float]:
    paired = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")], axis=1
    ).dropna()
    model = sm.OLS(paired["strategy"], sm.add_constant(paired["benchmark"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    active = paired["strategy"] - paired["benchmark"]
    tracking_error = active.std() * np.sqrt(252)
    return {
        "annual_alpha_hac": float(model.params["const"] * 252),
        "alpha_hac_p": float(model.pvalues["const"]),
        "beta": float(model.params["benchmark"]),
        "correlation": float(paired.corr().iloc[0, 1]),
        "tracking_error": float(tracking_error),
        "information_ratio": float(active.mean() * 252 / tracking_error),
        "annual_mean_excess": float(active.mean() * 252),
    }


def consistency(returns: pd.Series) -> dict[str, float]:
    monthly = (1 + returns).resample("ME").prod() - 1
    quarterly = (1 + returns).resample("QE").prod() - 1
    rolling = returns.rolling(252).apply(
        lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252) if x.std(ddof=1) > 0 else np.nan,
        raw=False,
    ).dropna()
    return {
        "positive_months": float((monthly > 0).mean()),
        "nonnegative_months": float((monthly >= 0).mean()),
        "positive_quarters": float((quarterly > 0).mean()),
        "rolling_252_sharpe_min": float(rolling.min()),
        "rolling_252_sharpe_median": float(rolling.median()),
        "rolling_252_sharpe_max": float(rolling.max()),
        "rolling_252_sharpe_std": float(rolling.std()),
    }


def block_bootstrap(returns: pd.Series, simulations: int = 1_000, block: int = 20) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED)
    arr = returns.to_numpy(float)
    n = len(arr)
    blocks = int(np.ceil(n / block))
    sharpes, cagrs, mdds = [], [], []
    offsets = np.arange(block)
    for _ in range(simulations):
        starts = rng.integers(0, n, size=blocks)
        sample = arr[((starts[:, None] + offsets) % n).ravel()[:n]]
        std = sample.std(ddof=1)
        sharpes.append(sample.mean() / std * np.sqrt(252) if std > 0 else 0)
        equity = np.cumprod(1 + sample)
        cagrs.append(equity[-1] ** (252 / n) - 1)
        mdds.append(np.min(equity / np.maximum.accumulate(equity) - 1))
    return {
        "simulations": simulations,
        "block_sessions": block,
        "sharpe_95_ci": [float(x) for x in np.percentile(sharpes, [2.5, 97.5])],
        "cagr_95_ci": [float(x) for x in np.percentile(cagrs, [2.5, 97.5])],
        "mdd_5_50_95_percentiles": [float(x) for x in np.percentile(mdds, [5, 50, 95])],
    }


def trade_bootstrap(trade_returns: list[float], simulations: int = 1_000) -> dict[str, object]:
    rng = np.random.default_rng(RNG_SEED + 1)
    arr = np.asarray(trade_returns)
    terminals, mdds = [], []
    for _ in range(simulations):
        sample = rng.choice(arr, size=len(arr), replace=True)
        equity = np.cumprod(1 + sample)
        terminals.append(equity[-1])
        mdds.append(np.min(equity / np.maximum.accumulate(equity) - 1))
    return {
        "simulations": simulations,
        "completed_trades": len(arr),
        "terminal_multiple_5_50_95": [float(x) for x in np.percentile(terminals, [5, 50, 95])],
        "mdd_5_50_95_percentiles": [float(x) for x in np.percentile(mdds, [5, 50, 95])],
    }


def trade_statistics(trades: list[dict]) -> dict[str, float | int]:
    returns = np.array([trade["return_pct"] / 100 for trade in trades])
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / len(returns)
    avg_win = float(wins.mean())
    avg_loss = float(losses.mean())
    payoff = avg_win / abs(avg_loss)
    return {
        "completed_trades": len(returns),
        "win_rate": float(win_rate),
        "profit_factor": float(wins.sum() / abs(losses.sum())),
        "expectancy": float(returns.mean()),
        "average_win": avg_win,
        "average_loss": avg_loss,
        "payoff_ratio": float(payoff),
        "kelly_fraction": float(win_rate - (1 - win_rate) / payoff),
    }


def subset_metrics(values: pd.Series, start: str, end: str | None = None) -> dict[str, float]:
    subset = values.loc[start:end]
    normalized = subset / subset.iloc[0] * qbt.INITIAL_CAPITAL
    return basic_metrics(normalized)


def source_era_runs(df: pd.DataFrame) -> dict[str, object]:
    eras = {
        "spy_reconstructed_1996_2001": ("1996-01-02", "2001-12-31"),
        "mmth_mapped_2002_2006": ("2002-01-02", "2006-12-31"),
        "actual_s5th_2007_plus": ("2007-01-03", None),
        "existing_series_2002_plus": ("2002-01-02", None),
    }
    out = {}
    for name, (start, end) in eras.items():
        section = df.loc[start:end]
        values, trade_returns, positions = run_fast(section, BASE_PARAMS)
        benchmark = qbt.INITIAL_CAPITAL * section["price"] / section["price"].iloc[0]
        out[name] = {
            "start": section.index[0].date().isoformat(),
            "end": section.index[-1].date().isoformat(),
            "sessions": len(section),
            "completed_trades": len(trade_returns),
            "time_in_market": float(positions.mean()),
            "strategy": basic_metrics(values),
            "benchmark": basic_metrics(benchmark),
        }
    return out


def regime_metrics(strategy: pd.Series, benchmark: pd.Series) -> dict[str, object]:
    regimes = {
        "dotcom_bust": ("2000-03-10", "2002-10-09"),
        "global_financial_crisis": ("2007-10-09", "2009-03-09"),
        "covid_crash": ("2020-02-19", "2020-03-23"),
        "2022_bear": ("2022-01-03", "2022-10-14"),
    }
    return {
        name: {
            "strategy_total_return": subset_metrics(strategy, start, end)["total_return"],
            "strategy_max_drawdown": subset_metrics(strategy, start, end)["max_drawdown"],
            "benchmark_total_return": subset_metrics(benchmark, start, end)["total_return"],
            "benchmark_max_drawdown": subset_metrics(benchmark, start, end)["max_drawdown"],
        }
        for name, (start, end) in regimes.items()
    }


def cost_stress(df: pd.DataFrame) -> list[dict[str, float]]:
    rows = []
    for multiple in [1, 2, 5, 10]:
        values, trades, _ = run_fast(df, BASE_PARAMS, multiple)
        m = basic_metrics(values)
        rows.append({"cost_multiple": multiple, "completed_trades": len(trades), **m})
    return rows


def sensitivity(df: pd.DataFrame) -> list[dict[str, object]]:
    grids = {
        "buy_thresh": [13, 20.8, 26, 31.2, 39],
        "vix_thresh": [15, 24, 30, 36, 45],
        "div_window": [30, 48, 60, 72, 90],
        "price_rise": [1.5, 2.4, 3, 3.6, 4.5],
        "breadth_fall": [10, 16, 20, 24, 30],
        "breadth_cap": [30, 48, 60, 72, 90],
        "ext10": [2.5, 4, 5, 6, 7.5],
        "climax_window": [5, 8, 10, 12, 15],
        "trailing_stop": [12.5, 20, 25, 30, 37.5],
        "cooldown_days": [0, 7, 15, 23, 30],
    }
    rows = []
    for parameter, values_to_test in grids.items():
        for value in values_to_test:
            params = BASE_PARAMS | {parameter: value}
            result, trades, _ = run_fast(df, params)
            m = basic_metrics(result)
            rows.append(
                {"parameter": parameter, "value": value, "completed_trades": len(trades),
                 "cagr": m["cagr"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]}
            )
    return rows


def walk_forward(df: pd.DataFrame) -> list[dict[str, object]]:
    grid = {
        "buy_thresh": [22.0, 26.0, 30.0],
        "vix_thresh": [25.0, 30.0, 35.0],
        "div_window": [50, 60, 70],
        "price_rise": [2.0, 3.0, 4.0],
        "breadth_fall": [15.0, 20.0, 25.0],
        "breadth_cap": [55.0, 60.0, 65.0],
    }
    candidates = [
        BASE_PARAMS | dict(zip(grid, combo))
        for combo in itertools.product(*grid.values())
    ]
    folds = [
        ("1996-2010_to_2011-2017", "1996-01-02", "2010-12-31", "2011-01-01", "2017-12-31"),
        ("1996-2017_to_2018-2026", "1996-01-02", "2017-12-31", "2018-01-01", None),
    ]
    rows = []
    for name, is_start, is_end, oos_start, oos_end in folds:
        is_df = df.loc[is_start:is_end]
        oos_df = df.loc[oos_start:oos_end]
        best_params = None
        best_sharpe = -np.inf
        best_trades = 0
        for params in candidates:
            values, trades, _ = run_fast(is_df, params)
            if len(trades) < 4:
                continue
            candidate_sharpe = basic_metrics(values)["sharpe"]
            if candidate_sharpe > best_sharpe:
                best_params, best_sharpe, best_trades = params, candidate_sharpe, len(trades)
        oos_values, oos_trades, _ = run_fast(oos_df, best_params)
        base_values, base_trades, _ = run_fast(oos_df, BASE_PARAMS)
        oos_sharpe = basic_metrics(oos_values)["sharpe"]
        rows.append(
            {
                "fold": name,
                "grid_candidates": len(candidates),
                "is_best_params": best_params,
                "is_sharpe": best_sharpe,
                "is_completed_trades": best_trades,
                "oos_sharpe": oos_sharpe,
                "oos_completed_trades": len(oos_trades),
                "efficiency_ratio": oos_sharpe / best_sharpe,
                "current_params_oos_sharpe": basic_metrics(base_values)["sharpe"],
                "current_params_oos_trades": len(base_trades),
            }
        )
    return rows


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return value


def main() -> None:
    df = qbt.load_data()
    strategy, trades, open_trade = qbt.run_strategy(
        df,
        cooldown_days=qbt.COOLDOWN_DAYS,
        execution_lag=1,
        fill_on="open",
    )
    benchmark = qbt.run_benchmark(df)
    fast_strategy, fast_trade_returns, positions = run_fast(df, BASE_PARAMS)
    if not np.allclose(strategy, fast_strategy, rtol=0, atol=1e-8):
        raise AssertionError("Fast robustness engine does not match qqq_backtest.py")
    if len(trades) != len(fast_trade_returns):
        raise AssertionError("Fast robustness engine trade count does not match")

    returns = strategy.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()
    active = returns - benchmark_returns
    paired_t = stats.ttest_1samp(active.dropna(), 0)

    trade_rows = [dict(trade, status="closed") for trade in trades]
    if open_trade:
        trade_rows.append(dict(open_trade, status="open", exit_date=pd.NaT, sell_reason=""))
    pd.DataFrame(trade_rows).to_csv(TRADES_FILE, index=False)

    breadth_source = pd.read_csv(ROOT / "breadth_daily.csv")
    breadth_source["Date"] = pd.to_datetime(breadth_source["Date"], format="%m/%d/%Y")
    breadth_source = breadth_source.set_index("Date")["source"]
    equity = df[["price", "open", "breadth", "vix"]].copy()
    equity["strategy"] = strategy
    equity["benchmark"] = benchmark
    equity["position"] = positions.astype(int)
    equity["breadth_source"] = breadth_source.reindex(equity.index)
    equity.index.name = "Date"
    equity.to_csv(EQUITY_FILE)

    completed_trade_stats = trade_statistics(trades)
    results = {
        "backtest_score": {
            "statistical_validity": {"score": 27, "max": 30},
            "risk_adjusted_performance": {"score": 13, "max": 25},
            "robustness_and_oos": {"score": 16, "max": 25},
            "trade_quality_and_consistency": {"score": 16, "max": 20},
            "raw_total": 72,
            "caps": {
                "unresolved_survivorship_or_missing_history_bias": 20,
                "fewer_than_30_completed_trades": 40,
                "no_genuinely_untouched_oos_period": 55,
            },
            "final_score": 20,
            "band": "Reject",
        },
        "run": {
            "start": df.index[0].date().isoformat(),
            "end": df.index[-1].date().isoformat(),
            "sessions": len(df),
            "execution": "signal at close, fill next session open",
            "commission_per_side": qbt.COMMISSION,
            "slippage_per_side": qbt.SLIPPAGE,
            "principal_parameter_count": 11,
            "multiple_testing_trials_assumed": TRIALS_ASSUMED,
        },
        "strategy": basic_metrics(strategy),
        "benchmark": basic_metrics(benchmark),
        "drawdowns": drawdown_episodes(strategy),
        "distribution_and_significance": statistical_tests(returns),
        "benchmark_regression_hac": benchmark_regression(returns, benchmark_returns),
        "paired_excess_test": {
            "annual_mean_excess": float(active.mean() * 252),
            "t_stat": float(paired_t.statistic),
            "p_value": float(paired_t.pvalue),
        },
        "consistency": consistency(returns),
        "trade_statistics": completed_trade_stats,
        "open_trade": serializable(open_trade),
        "source_era_standalone_runs": source_era_runs(df),
        "crisis_regimes_from_full_run": regime_metrics(strategy, benchmark),
        "cost_stress": cost_stress(df),
        "parameter_sensitivity": sensitivity(df),
        "walk_forward_pseudo_oos": walk_forward(df),
        "block_bootstrap": block_bootstrap(returns),
        "trade_bootstrap": trade_bootstrap(fast_trade_returns),
    }
    RESULTS_FILE.write_text(json.dumps(serializable(results), indent=2) + "\n")
    print(f"Wrote {EQUITY_FILE.name}, {TRADES_FILE.name}, and {RESULTS_FILE.name}")
    print(json.dumps(serializable(results), indent=2))


if __name__ == "__main__":
    main()
