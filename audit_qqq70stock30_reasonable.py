#!/usr/bin/env python3
"""Recalculate defensible statistics for the QQQ 70 / Top-1 stock 30 strategy.

The original rolling output is useful descriptively, but its annual holding map
uses a September portfolio snapshot during the same calendar year and its daily
rolling windows are almost perfectly dependent.  This audit therefore:

* delays every Nasdaq-100 annual snapshot until the following calendar year;
* corrects three Top-1 extraction errors verified against the source filings;
* treats overlapping windows as descriptive and reports their structural n;
* uses 2007+ S5TH breadth for continuous risk and trade statistics; and
* reports block-bootstrap uncertainty and transparent return scenarios.

Only local cached market data are read.  No downloader is run by this module.
"""

from __future__ import annotations

import json
import math
import sys
import types
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import brentq
from statsmodels.stats.diagnostic import acorr_ljungbox


# qqq_portfolio_backtest refreshes remote inputs at import time.  The audit must
# be deterministic and read-only, so provide the one symbol that import expects.
_fetch_stub = types.ModuleType("fetch_investing_data")


def _noop_fetch(verbose: bool = True) -> None:
    del verbose


setattr(_fetch_stub, "fetch_all_updates", _noop_fetch)
sys.modules.setdefault("fetch_investing_data", _fetch_stub)

import qqq70stock30_dca_rolling as rolling  # noqa: E402
import qqq_portfolio_backtest as qpb  # noqa: E402


ROOT = Path(__file__).resolve().parent
ROLLING_FILE = ROOT / "qqq70stock30_reasonable_rolling.csv"
NONOVERLAP_FILE = ROOT / "qqq70stock30_reasonable_nonoverlap.csv"
RESULTS_FILE = ROOT / "qqq70stock30_reasonable_results.json"
REPORT_FILE = ROOT / "docs" / "research" / "qqq70stock30_reasonable_report.md"

HORIZONS = (1, 3, 5, 10, 15, 20)
CLEAN_START = "2007-01-01"
INITIAL_CAPITAL = 1_000_000.0
ANNUAL_CONTRIBUTION = 200_000.0
TRIALS_ASSUMED = 4_819
BOOTSTRAP_SAMPLES = 2_000
BLOCK_SESSIONS = 20
RANDOM_SEED = 20_260_903

# Source Schedule of Investments values show Apple was larger than the ticker
# recorded in the derived CSV for these snapshots.
TOP1_CORRECTIONS = {2016: "AAPL", 2021: "AAPL", 2023: "AAPL"}


def future_value(
    annual_return: float,
    years: int,
    initial: float = INITIAL_CAPITAL,
    annual_contribution: float = ANNUAL_CONTRIBUTION,
) -> float:
    """Terminal wealth with contributions after years 1..years-1."""
    value = initial
    for year in range(years):
        value *= 1 + annual_return
        if year < years - 1:
            value += annual_contribution
    return float(value)


def implied_annual_return(
    terminal_value: float,
    years: int,
    initial: float = INITIAL_CAPITAL,
    annual_contribution: float = ANNUAL_CONTRIBUTION,
) -> float:
    """Constant annual return that reproduces a terminal DCA value."""
    def objective(rate: float) -> float:
        return (
            future_value(rate, years, initial, annual_contribution) - terminal_value
        )

    return float(brentq(objective, -0.99, 3.0))


def build_point_in_time_holdings(
    years: list[int] | range,
    snapshots: dict[int, str],
    spy_fallback: dict[int, str],
) -> dict[int, str]:
    """Map a calendar year to information available before that year.

    Nasdaq snapshots are September fiscal-year schedules and are conservatively
    delayed to the following calendar year.  The SPY fallback file already maps
    each year to a filing available before that calendar year.
    """
    corrected = snapshots | TOP1_CORRECTIONS
    result: dict[int, str] = {}
    for year in years:
        ticker = corrected.get(year - 1)
        if ticker is None:
            ticker = spy_fallback.get(year) or spy_fallback.get(year - 1)
        if ticker:
            result[int(year)] = ticker
    return result


def load_point_in_time_holdings(years: list[int] | range) -> dict[int, str]:
    raw = pd.read_csv(qpb.TOP_HOLDINGS_FILE)
    snapshots: dict[int, str] = {}
    for _, row in raw.loc[raw["Rank"].eq(1)].iterrows():
        ticker = qpb._name_to_ticker(str(row["Holding"]))
        if ticker:
            snapshots[int(row["Year"])] = ticker

    spy = pd.read_csv(qpb.SPY_TOP_HOLDINGS_FILE)
    spy_fallback = {
        int(row["Year"]): str(row["Ticker"]).strip()
        for _, row in spy.iterrows()
        if str(row["Ticker"]).strip()
    }
    return build_point_in_time_holdings(years, snapshots, spy_fallback)


@contextmanager
def portfolio_settings(
    qqq_weight: float,
    stock_weight: float,
    cost_multiple: float = 1.0,
):
    attrs = {
        "QQQ_WEIGHT": qpb.QQQ_WEIGHT,
        "STOCK_WEIGHT": qpb.STOCK_WEIGHT,
        "TQQQ_WEIGHT": qpb.TQQQ_WEIGHT,
        "SPY_WEIGHT": qpb.SPY_WEIGHT,
        "SOXX_WEIGHT": qpb.SOXX_WEIGHT,
        "COMMISSION": qpb.COMMISSION,
        "SLIPPAGE": qpb.SLIPPAGE,
    }
    try:
        qpb.QQQ_WEIGHT = qqq_weight
        qpb.STOCK_WEIGHT = stock_weight
        qpb.TQQQ_WEIGHT = qpb.SPY_WEIGHT = qpb.SOXX_WEIGHT = 0.0
        qpb.COMMISSION = attrs["COMMISSION"] * cost_multiple
        qpb.SLIPPAGE = attrs["SLIPPAGE"] * cost_multiple
        yield
    finally:
        for name, value in attrs.items():
            setattr(qpb, name, value)


def run_continuous(
    df: pd.DataFrame,
    point_in_time: dict[int, str],
    stocks: dict[str, pd.Series],
    stock_opens: dict[str, pd.Series],
    qqq_weight: float = 0.70,
    stock_weight: float = 0.30,
    cost_multiple: float = 1.0,
) -> tuple[pd.Series, list[dict], dict | None]:
    first_ticker = point_in_time.get(int(df.index[0].year))
    with portfolio_settings(qqq_weight, stock_weight, cost_multiple):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            equity, trades, open_trade, _ = qpb.run_strategy(
                df,
                point_in_time,
                stocks,
                None,
                None,
                None,
                cooldown_days=qpb.COOLDOWN_DAYS,
                initial_capital=INITIAL_CAPITAL,
                force_entry_on_start=True,
                force_ticker=first_ticker,
                execution_lag=1,
                fill_on="open",
                aligned_stocks_open=stock_opens,
            )
    return equity, trades, open_trade


def performance_metrics(values: pd.Series) -> dict[str, float]:
    returns = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    drawdown = values / values.cummax() - 1
    downside = np.sqrt(np.mean(np.minimum(returns.to_numpy(), 0.0) ** 2)) * np.sqrt(252)
    annual_mean = float(returns.mean() * 252)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(252))
    cagr = float((values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1)
    max_drawdown = float(drawdown.min())
    ulcer = float(np.sqrt(np.mean(drawdown.to_numpy() ** 2)))
    positive = returns.clip(lower=0).sum()
    negative = -returns.clip(upper=0).sum()
    return {
        "start": values.index[0].date().isoformat(),
        "end": values.index[-1].date().isoformat(),
        "sessions": int(len(values)),
        "years": float(years),
        "final_value_no_dca": float(values.iloc[-1]),
        "total_return": float(values.iloc[-1] / values.iloc[0] - 1),
        "cagr": cagr,
        "annual_mean_return": annual_mean,
        "annual_volatility": annual_vol,
        "sharpe": float(annual_mean / annual_vol),
        "sortino": float(annual_mean / downside),
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)),
        "ulcer_index": ulcer,
        "pain_ratio": float(cagr / ulcer),
        "lake_ratio": float((drawdown < 0).mean()),
        "omega_zero": float(positive / negative),
        "var_95": float(-returns.quantile(0.05)),
        "var_99": float(-returns.quantile(0.01)),
        "cvar_95": float(-returns.loc[returns <= returns.quantile(0.05)].mean()),
        "cvar_99": float(-returns.loc[returns <= returns.quantile(0.01)].mean()),
    }


def statistical_tests(returns: pd.Series) -> dict[str, object]:
    arr = returns.to_numpy(float)
    n = len(arr)
    mean_model = sm.OLS(arr, np.ones((n, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 20}
    )
    sample_var = float(np.var(arr, ddof=1))
    hac_se = float(mean_model.bse[0])
    effective_n = min(float(n), max(1.0, sample_var / hac_se**2))
    skew = float(stats.skew(arr, bias=False))
    excess_kurtosis = float(stats.kurtosis(arr, fisher=True, bias=False))
    daily_sharpe = float(np.mean(arr) / np.std(arr, ddof=1))
    pearson_kurtosis = excess_kurtosis + 3
    psr_denom = math.sqrt(
        (
            1
            - skew * daily_sharpe
            + (pearson_kurtosis - 1) / 4 * daily_sharpe**2
        )
        / (effective_n - 1)
    )
    psr = float(stats.norm.cdf(daily_sharpe / psr_denom))
    gamma = np.euler_gamma
    sr_null_std = 1 / math.sqrt(effective_n - 1)
    expected_max = sr_null_std * (
        (1 - gamma) * stats.norm.ppf(1 - 1 / TRIALS_ASSUMED)
        + gamma * stats.norm.ppf(1 - 1 / (TRIALS_ASSUMED * np.e))
    )
    dsr = float(stats.norm.cdf((daily_sharpe - expected_max) / psr_denom))
    jb = stats.jarque_bera(arr)
    lb = acorr_ljungbox(arr, lags=[5, 10, 20], return_df=True)
    return {
        "observations": n,
        "effective_observations_hac20": effective_n,
        "hac_t_stat_mean": float(mean_model.tvalues[0]),
        "hac_p_value_mean": float(mean_model.pvalues[0]),
        "skewness": skew,
        "excess_kurtosis": excess_kurtosis,
        "jarque_bera_p": float(jb.pvalue),
        "ljung_box_p": {str(lag): float(lb.loc[lag, "lb_pvalue"]) for lag in lb.index},
        "psr_vs_zero": psr,
        "dsr_4819_trials": dsr,
        "dsr_benchmark_annual_sharpe": float(expected_max * np.sqrt(252)),
    }


def benchmark_regression(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> dict[str, float]:
    paired = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    model = sm.OLS(paired["strategy"], sm.add_constant(paired["benchmark"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 20}
    )
    active = paired["strategy"] - paired["benchmark"]
    tracking_error = float(active.std(ddof=1) * np.sqrt(252))
    return {
        "annual_alpha_hac": float(model.params["const"] * 252),
        "alpha_hac_p_value": float(model.pvalues["const"]),
        "beta": float(model.params["benchmark"]),
        "correlation": float(paired.corr().iloc[0, 1]),
        "tracking_error": tracking_error,
        "information_ratio": float(active.mean() * 252 / tracking_error),
    }


def consistency(returns: pd.Series) -> dict[str, float]:
    monthly = (1 + returns).resample("ME").prod() - 1
    quarterly = (1 + returns).resample("QE").prod() - 1
    rolling_sharpe = returns.rolling(252).apply(
        lambda x: x.mean() / x.std(ddof=1) * np.sqrt(252)
        if x.std(ddof=1) > 0
        else np.nan,
        raw=False,
    ).dropna()
    return {
        "positive_months": float((monthly > 0).mean()),
        "positive_quarters": float((quarterly > 0).mean()),
        "rolling_252_sharpe_min": float(rolling_sharpe.min()),
        "rolling_252_sharpe_median": float(rolling_sharpe.median()),
        "rolling_252_sharpe_max": float(rolling_sharpe.max()),
        "rolling_252_sharpe_std": float(rolling_sharpe.std()),
    }


def trade_statistics(trades: list[dict]) -> dict[str, float | int | None]:
    values = np.asarray([trade["return_pct"] / 100 for trade in trades], dtype=float)
    wins = values[values > 0]
    losses = values[values <= 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.size else math.inf
    average_loss = float(losses.mean()) if losses.size else None
    payoff = float(wins.mean() / abs(average_loss)) if average_loss else math.inf
    win_rate = float((values > 0).mean())
    return {
        "completed_trades": int(len(values)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": float(values.mean()),
        "average_win": float(wins.mean()) if wins.size else None,
        "average_loss": average_loss,
        "payoff_ratio": payoff,
        "kelly_fraction": float(win_rate - (1 - win_rate) / payoff),
    }


def block_bootstrap(returns: pd.Series) -> dict[str, object]:
    rng = np.random.default_rng(RANDOM_SEED)
    arr = returns.to_numpy(float)
    horizon_sessions = 20 * 252
    blocks_needed = int(np.ceil(horizon_sessions / BLOCK_SESSIONS))
    offsets = np.arange(BLOCK_SESSIONS)
    terminals: list[float] = []
    cagrs: list[float] = []
    max_drawdowns: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        starts = rng.integers(0, len(arr), size=blocks_needed)
        sample = arr[((starts[:, None] + offsets) % len(arr)).ravel()[:horizon_sessions]]
        value = INITIAL_CAPITAL
        path = np.empty(horizon_sessions)
        for day, daily_return in enumerate(sample):
            value *= 1 + daily_return
            if day > 0 and day % 252 == 0:
                value += ANNUAL_CONTRIBUTION
            path[day] = value
        terminals.append(float(value))
        no_flow = np.cumprod(1 + sample)
        cagrs.append(float(no_flow[-1] ** (1 / 20) - 1))
        max_drawdowns.append(float(np.min(path / np.maximum.accumulate(path) - 1)))
    return {
        "samples": BOOTSTRAP_SAMPLES,
        "block_sessions": BLOCK_SESSIONS,
        "warning": "In-sample stationary block bootstrap; not a calibrated forecast.",
        "terminal_20y_percentiles": {
            str(q): float(np.percentile(terminals, q)) for q in (5, 25, 50, 75, 95)
        },
        "cagr_20y_95_ci": [float(x) for x in np.percentile(cagrs, [2.5, 97.5])],
        "max_drawdown_20y_5_50_95": [
            float(x) for x in np.percentile(max_drawdowns, [5, 50, 95])
        ],
        "probability_terminal_at_least_80m_in_sample_model": float(
            np.mean(np.asarray(terminals) >= 80_000_000)
        ),
    }


def rolling_statistics(
    arrays: dict,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    length = len(arrays["dates"])
    rows: list[dict] = []
    cohorts: list[dict] = []
    for years in horizons:
        window = rolling.YEAR_ROWS * years
        count = length - window + 1
        if count <= 0:
            continue
        contributions = years - 1
        deployed = INITIAL_CAPITAL + ANNUAL_CONTRIBUTION * contributions
        strategy = np.empty(count)
        benchmark = np.empty(count)
        for start in range(count):
            strategy[start] = rolling.run_window(
                arrays, start, start + window, contributions
            )
            benchmark[start] = rolling.run_buyhold(
                arrays["price"], start, start + window, contributions
            )

        excess_pp = (strategy - benchmark) / deployed * 100
        nonoverlap_idx = np.arange(0, count, window)
        nonoverlap = excess_pp[nonoverlap_idx]
        if len(nonoverlap) >= 3:
            t_result = stats.ttest_1samp(nonoverlap, popmean=0)
            rng = np.random.default_rng(RANDOM_SEED + years)
            sample_idx = rng.integers(
                0, len(nonoverlap), size=(BOOTSTRAP_SAMPLES, len(nonoverlap))
            )
            boot_means = nonoverlap[sample_idx].mean(axis=1)
            boot_low, boot_high = np.percentile(boot_means, [2.5, 97.5])
            t_stat, p_value = float(t_result.statistic), float(t_result.pvalue)
        else:
            t_stat = p_value = boot_low = boot_high = math.nan

        hac = rolling.newey_west_mean_stats(excess_pp, window - 1)
        row = {
            "years": years,
            "raw_overlapping_windows": count,
            "structural_effective_n": count / window,
            "overlap_pct": (1 - len(nonoverlap_idx) / count) * 100,
            "deployed_usd": deployed,
            "strategy_mean_usd": float(strategy.mean()),
            "strategy_p05_usd": float(np.quantile(strategy, 0.05)),
            "strategy_p25_usd": float(np.quantile(strategy, 0.25)),
            "strategy_median_usd": float(np.median(strategy)),
            "strategy_p75_usd": float(np.quantile(strategy, 0.75)),
            "strategy_p95_usd": float(np.quantile(strategy, 0.95)),
            "strategy_min_usd": float(strategy.min()),
            "strategy_max_usd": float(strategy.max()),
            "strategy_median_implied_annual_return": implied_annual_return(
                float(np.median(strategy)), years
            ),
            "benchmark_median_usd": float(np.median(benchmark)),
            "benchmark_median_implied_annual_return": implied_annual_return(
                float(np.median(benchmark)), years
            ),
            "nonoverlap_windows": len(nonoverlap_idx),
            "nonoverlap_excess_mean_pp": float(nonoverlap.mean()),
            "nonoverlap_t_stat": t_stat,
            "nonoverlap_p_value": p_value,
            "nonoverlap_bootstrap_ci_low_pp": float(boot_low),
            "nonoverlap_bootstrap_ci_high_pp": float(boot_high),
            "hac_excess_mean_pp": float(hac["mean"]),
            "hac_excess_ci_low_pp": float(hac["ci_low"]),
            "hac_excess_ci_high_pp": float(hac["ci_high"]),
            "hac_excess_p_value": float(hac["p_value"]),
        }
        if years == 20:
            row["overlap_fraction_at_least_80m"] = float(np.mean(strategy >= 80e6))
            row["overlap_lag1_autocorrelation"] = float(
                np.corrcoef(strategy[:-1], strategy[1:])[0, 1]
            )
        rows.append(row)

        for cohort_id, start in enumerate(nonoverlap_idx, start=1):
            cohorts.append(
                {
                    "years": years,
                    "cohort_id": cohort_id,
                    "start_date": pd.Timestamp(arrays["dates"][start]).date(),
                    "end_date": pd.Timestamp(arrays["dates"][start + window - 1]).date(),
                    "strategy_final_usd": float(strategy[start]),
                    "benchmark_final_usd": float(benchmark[start]),
                    "excess_return_pp": float(excess_pp[start]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(cohorts)


def score_report(metrics: dict, tests: dict, trades: dict, robustness: dict) -> dict:
    # Statistical validity: strong in-sample mean/PSR, but DSR and thin trade
    # count prevent full credit.  Risk points follow the skill thresholds.
    a = 8 if tests["hac_t_stat_mean"] > 3 else 6 if tests["hac_t_stat_mean"] > 2 else 0
    a += 7 if tests["psr_vs_zero"] > 0.95 else 5 if tests["psr_vs_zero"] > 0.90 else 0
    a += 8 if tests["dsr_4819_trials"] > 0.95 else 4 if tests["dsr_4819_trials"] > 0.80 else 0
    a += 4  # 4,947 returns, but only 13 completed trades.

    b = 10 if metrics["sharpe"] > 2 else 7 if metrics["sharpe"] > 1 else 4
    b += 8 if metrics["sortino"] > 1.5 else 4 if metrics["sortino"] > 0.7 else 0
    b += 7 if abs(metrics["max_drawdown"]) < 0.10 else 5 if abs(metrics["max_drawdown"]) < 0.20 else 3 if abs(metrics["max_drawdown"]) < 0.30 else 0

    c = 0  # no genuine walk-forward/OOS
    c += 4 if robustness["block_bootstrap"]["cagr_20y_95_ci"][0] > 0 else 0
    c += 0  # no pre-registered parameter sensitivity for this exact portfolio

    d = 7 if trades["profit_factor"] > 2 else 5 if trades["profit_factor"] > 1.5 else 3
    d += 6 if trades["expectancy"] > 0 else 0
    positive_months = robustness["consistency"]["positive_months"]
    d += 7 if positive_months > 0.65 else 5 if positive_months > 0.55 else 3 if positive_months > 0.50 else 0

    raw = int(a + b + c + d)
    caps = {"fewer_than_30_completed_trades": 40, "no_true_oos_walk_forward": 55}
    final = min(raw, *caps.values())
    bands = [(80, "Tradeable"), (65, "Promising"), (45, "Needs work"), (25, "Weak"), (0, "Reject")]
    band = next(label for threshold, label in bands if final >= threshold)
    return {
        "A_statistical_validity": {"score": int(a), "max": 30},
        "B_risk_adjusted_performance": {"score": int(b), "max": 25},
        "C_robustness_oos": {"score": int(c), "max": 25},
        "D_trade_quality_consistency": {"score": int(d), "max": 20},
        "raw_score": raw,
        "caps": caps,
        "final_score": int(final),
        "band": band,
    }


def markdown_report(results: dict, rolling_df: pd.DataFrame) -> str:
    clean = results["clean_2007_plus"]
    metrics = clean["strategy_metrics"]
    benchmark = clean["benchmark_metrics"]
    tests = clean["statistical_tests"]
    trades = clean["trade_statistics"]
    score = results["backtest_score"]
    row20 = rolling_df.loc[rolling_df["years"].eq(20)].iloc[0]
    boot = results["robustness"]["block_bootstrap"]
    scenario = results["planning_scenarios"]
    lines = [
        "# QQQ 70 / Point-in-Time Top-1 30：合理統計重算",
        "",
        f"資料截至：{results['data_end']}；審核只使用本機快取資料。",
        "",
        f"## Backtest Score: {score['final_score']} / 100 — {score['band']}",
        "",
        "已移除年度持倉的直接前視偏差，並只用 2007+ S5TH 時段計算風險；但只有 "
        f"{trades['completed_trades']} 宗完成交易、沒有真正 OOS，因此分數受 40 分上限約束。",
        "",
        "| Component | Score | Max |",
        "|---|---:|---:|",
        f"| A. Statistical validity | {score['A_statistical_validity']['score']} | 30 |",
        f"| B. Risk-adjusted performance | {score['B_risk_adjusted_performance']['score']} | 25 |",
        f"| C. Robustness / OOS | {score['C_robustness_oos']['score']} | 25 |",
        f"| D. Trade quality / consistency | {score['D_trade_quality_consistency']['score']} | 20 |",
        f"| Raw | **{score['raw_score']}** | **100** |",
        f"| Final after caps | **{score['final_score']}** | **100** |",
        "",
        "## 20 年 DCA 歷史描述（非預測）",
        "",
        "| Metric | Corrected result |",
        "|---|---:|",
        f"| Total contributions | ${row20['deployed_usd']:,.0f} |",
        f"| Median terminal | ${row20['strategy_median_usd']:,.0f} |",
        f"| 5th–95th percentile | ${row20['strategy_p05_usd']:,.0f} – ${row20['strategy_p95_usd']:,.0f} |",
        f"| Minimum–maximum | ${row20['strategy_min_usd']:,.0f} – ${row20['strategy_max_usd']:,.0f} |",
        f"| Median implied annual return | {row20['strategy_median_implied_annual_return']:.2%} |",
        f"| NDX buy-and-hold median | ${row20['benchmark_median_usd']:,.0f} |",
        f"| Structural effective n | {row20['structural_effective_n']:.2f} |",
        f"| Lag-1 correlation | {row20['overlap_lag1_autocorrelation']:.4f} |",
        "",
        "The overlapping-window percent above $80m is deliberately not reported as a probability.",
        "",
        "## 2007+ Clean-source risk profile（無 DCA）",
        "",
        "| Metric | Strategy | NDX buy-and-hold |",
        "|---|---:|---:|",
        f"| CAGR | {metrics['cagr']:.2%} | {benchmark['cagr']:.2%} |",
        f"| Annual volatility | {metrics['annual_volatility']:.2%} | {benchmark['annual_volatility']:.2%} |",
        f"| Sharpe | {metrics['sharpe']:.3f} | {benchmark['sharpe']:.3f} |",
        f"| Sortino | {metrics['sortino']:.3f} | {benchmark['sortino']:.3f} |",
        f"| Maximum drawdown | {metrics['max_drawdown']:.2%} | {benchmark['max_drawdown']:.2%} |",
        f"| Completed trades | {trades['completed_trades']} | — |",
        f"| Win rate | {trades['win_rate']:.2%} | — |",
        f"| HAC alpha p-value | {clean['benchmark_regression']['alpha_hac_p_value']:.4g} | — |",
        f"| DSR after {TRIALS_ASSUMED:,} trials | {tests['dsr_4819_trials']:.3f} | — |",
        "",
        "## Block bootstrap（仍屬 in-sample model）",
        "",
        f"20-year terminal 5/50/95 percentiles: ${boot['terminal_20y_percentiles']['5']:,.0f} / "
        f"${boot['terminal_20y_percentiles']['50']:,.0f} / ${boot['terminal_20y_percentiles']['95']:,.0f}.",
        "這是假設 2007+ 回報生成機制不變的區塊重抽樣，不可當實際成功概率。",
        "",
        "## Planning scenarios",
        "",
        "| Annual return | 20y nominal terminal | 3% inflation real value |",
        "|---:|---:|---:|",
    ]
    for item in scenario:
        lines.append(
            f"| {item['annual_return']:.2%} | ${item['nominal_terminal_usd']:,.0f} | "
            f"${item['real_terminal_usd_at_3pct_inflation']:,.0f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "Point-in-time correction lowers the historical 20-year median, but the estimate remains "
            "dominated by one market history and an in-sample signal search. Use the scenario table for "
            "planning; do not use the overlapping-window frequency or bootstrap frequency as a promised probability.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    qpb._HAS_YF = False
    merged, original_holdings, stocks, *_ = qpb.load_data()
    stock_opens, *_ = qpb.load_open_series(original_holdings, merged.index)
    years = list(range(int(merged.index.year.min()), int(merged.index.year.max()) + 1))
    point_in_time = load_point_in_time_holdings(years)

    arrays = rolling.build_arrays(merged, point_in_time, stocks, stock_opens)
    rolling_df, nonoverlap_df = rolling_statistics(arrays)
    rolling_df.to_csv(ROLLING_FILE, index=False)
    nonoverlap_df.to_csv(NONOVERLAP_FILE, index=False)

    clean_df = merged.loc[pd.Timestamp(CLEAN_START) :].copy()
    strategy, trades, open_trade = run_continuous(
        clean_df, point_in_time, stocks, stock_opens
    )
    qqq_timing, _, _ = run_continuous(
        clean_df,
        point_in_time,
        stocks,
        stock_opens,
        qqq_weight=1.0,
        stock_weight=0.0,
    )
    benchmark = (
        INITIAL_CAPITAL * clean_df["price"] / clean_df["price"].iloc[0]
    ).rename("ndx_buyhold")
    strategy_returns = strategy.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()

    cost_stress_rows: list[dict[str, object]] = []
    robustness: dict[str, object] = {
        "block_bootstrap": block_bootstrap(strategy_returns),
        "consistency": consistency(strategy_returns),
        "cost_stress": cost_stress_rows,
        "qqq_only_timing_metrics": performance_metrics(qqq_timing),
        "holding_corrections": TOP1_CORRECTIONS,
    }
    for multiple in (1, 2, 5, 10):
        stressed, stressed_trades, _ = run_continuous(
            clean_df,
            point_in_time,
            stocks,
            stock_opens,
            cost_multiple=float(multiple),
        )
        cost_stress_rows.append(
            {
                "cost_multiple": multiple,
                "completed_trades": len(stressed_trades),
                **performance_metrics(stressed),
            }
        )

    metrics = performance_metrics(strategy)
    tests = statistical_tests(strategy_returns)
    trade_stats = trade_statistics(trades)
    results = {
        "methodology": {
            "holding_rule": "Use September snapshot from year t only in calendar year t+1.",
            "holding_corrections": TOP1_CORRECTIONS,
            "execution": "Signal at close t, fill at open t+1; 5 bps baseline slippage.",
            "rolling_use": "Descriptive only; non-overlap/HAC used for inference.",
            "risk_period": "2007+ actual S5TH breadth source; no contributions.",
        },
        "data_start": merged.index[0].date().isoformat(),
        "data_end": merged.index[-1].date().isoformat(),
        "clean_2007_plus": {
            "strategy_metrics": metrics,
            "benchmark_metrics": performance_metrics(benchmark),
            "statistical_tests": tests,
            "benchmark_regression": benchmark_regression(
                strategy_returns, benchmark_returns
            ),
            "trade_statistics": trade_stats,
            "open_trade_present": open_trade is not None,
        },
        "robustness": robustness,
        "planning_scenarios": [
            {
                "annual_return": rate,
                "nominal_terminal_usd": future_value(rate, 20),
                "real_terminal_usd_at_3pct_inflation": future_value(rate, 20)
                / 1.03**20,
            }
            for rate in (0.08, 0.10, 0.12, 0.15, 0.18, 0.20)
        ],
    }
    results["backtest_score"] = score_report(metrics, tests, trade_stats, robustness)
    RESULTS_FILE.write_text(json.dumps(results, indent=2, allow_nan=True) + "\n")
    REPORT_FILE.write_text(markdown_report(results, rolling_df))

    row20 = rolling_df.loc[rolling_df["years"].eq(20)].iloc[0]
    print(f"Data: {results['data_start']} to {results['data_end']}")
    print(f"Corrected 20y median: ${row20['strategy_median_usd']:,.0f}")
    print(f"Structural effective n: {row20['structural_effective_n']:.3f}")
    print(
        f"Clean 2007+ CAGR/Sharpe/MDD: {metrics['cagr']:.2%} / "
        f"{metrics['sharpe']:.3f} / {metrics['max_drawdown']:.2%}"
    )
    print(
        f"Backtest score: {results['backtest_score']['final_score']} / 100 "
        f"({results['backtest_score']['band']})"
    )
    print(f"Wrote {ROLLING_FILE.name}, {NONOVERLAP_FILE.name}, {RESULTS_FILE.name}")
    print(f"Wrote {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
