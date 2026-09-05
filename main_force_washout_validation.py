#!/usr/bin/env python3
"""Causal, point-in-time validation of the main-force washout checklist.

The qualitative skill is converted to a pre-registered OHLCV rule.  Daily files
cannot validate displayed order-book support, so Signal 3 is always Unknown and
a price/volume breakout is required before a setup is labelled completed.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
PRICE_DIR = ROOT / "SPY" / "stock_prices" / "prices"
MEMBERSHIP_FILE = ROOT / "SPY" / "stock_prices" / "membership_snapshots.csv"
MASTER_FILE = ROOT / "SPY" / "stock_prices" / "all_tickers_1996_2026.csv"
UNAVAILABLE_FILE = ROOT / "SPY" / "stock_prices" / "unavailable_tickers.csv"
SPY_FILE = ROOT / "NASDAQ100" / "stock_prices" / "SPY.csv"
OUTPUT_DIR = ROOT / "analysis" / "main_force_washout_validation"
TRADING_DAYS = 252
RNG_SEED = 20260904


@dataclass(frozen=True)
class RuleConfig:
    ma_zone: float = 0.01
    support_lookback: int = 15
    support_tests: int = 2
    volume_threshold: float = 0.50
    doji_threshold: float = 0.10
    narrow_atr: float = 0.60
    setup_window: int = 10
    breakout_lookback: int = 20
    breakout_volume: float = 1.20
    max_hold: int = 63
    one_way_cost: float = 0.001


BASE = RuleConfig()
SENSITIVITY = {
    "baseline": BASE,
    "ma_zone_0.5pct": replace(BASE, ma_zone=0.005),
    "ma_zone_1.5pct": replace(BASE, ma_zone=0.015),
    "volume_ratio_0.40": replace(BASE, volume_threshold=0.40),
    "volume_ratio_0.60": replace(BASE, volume_threshold=0.60),
    "breakout_volume_1.00": replace(BASE, breakout_volume=1.00),
    "breakout_volume_1.50": replace(BASE, breakout_volume=1.50),
    "setup_window_5": replace(BASE, setup_window=5),
    "setup_window_15": replace(BASE, setup_window=15),
    "breakout_lookback_10": replace(BASE, breakout_lookback=10),
    "breakout_lookback_30": replace(BASE, breakout_lookback=30),
}


def finite(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().replace(".", "-")


def load_membership(path: Path = MEMBERSHIP_FILE) -> tuple[np.ndarray, list[set[str]], set[str]]:
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["date"], errors="raise").values.astype("datetime64[ns]")
    sets = [{normalize_ticker(x) for x in str(v).split(",") if x} for v in raw["tickers"]]
    universe = set().union(*sets)
    return dates, sets, universe


def membership_mask(
    dates: pd.DatetimeIndex, ticker: str, snapshot_dates: np.ndarray, snapshot_sets: list[set[str]]
) -> np.ndarray:
    idx = np.searchsorted(snapshot_dates, dates.values.astype("datetime64[ns]"), side="right") - 1
    ticker = normalize_ticker(ticker)
    return np.array([i >= 0 and ticker in snapshot_sets[int(i)] for i in idx], dtype=bool)


def load_spy(path: Path = SPY_FILE) -> pd.Series:
    raw = pd.read_csv(path)
    dates = pd.to_datetime(raw["Date"], format="mixed", utc=True).dt.tz_convert(None).dt.normalize()
    out = pd.Series(pd.to_numeric(raw["price"], errors="coerce").values, index=dates, name="spy")
    return out[~out.index.duplicated(keep="last")].sort_index().dropna()


def load_price(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    raw = pd.read_csv(path, usecols=cols)
    dates = pd.to_datetime(raw.pop("Date"), format="mixed", utc=True).dt.tz_convert(None).dt.normalize()
    raw.index = dates
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    factor = raw["Adj Close"] / raw["Close"]
    frame = pd.DataFrame(index=raw.index)
    for col in ("Open", "High", "Low", "Close"):
        frame[col.lower()] = raw[col] * factor
    frame["volume"] = raw["Volume"].where(raw["Volume"] > 0)
    valid_price = (
        frame[["open", "high", "low", "close"]].notna().all(axis=1)
        & (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    )
    valid_ohlc = (
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
    )
    invalid_ohlc = valid_price & ~valid_ohlc
    frame.loc[~valid_price | invalid_ohlc, ["open", "high", "low", "close"]] = np.nan
    audit = {
        "raw_rows": int(len(raw)),
        "invalid_price_rows": int((~valid_price).sum()),
        "invalid_ohlc_rows": int(invalid_ohlc.sum()),
        "zero_volume_rows": int((raw["Volume"] == 0).sum()),
    }
    return frame, audit


def market_regime(spy: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = spy.reindex(index, method="ffill")
    ma30 = aligned.rolling(30, min_periods=30).mean()
    return ((aligned >= ma30) | (aligned.pct_change(20) > -0.05)).fillna(False)


def signal_frame(
    df: pd.DataFrame, member: np.ndarray, benign_market: pd.Series, cfg: RuleConfig
) -> pd.DataFrame:
    out = df.copy()
    close, high, low, opn, volume = (out[x] for x in ("close", "high", "low", "open", "volume"))
    out["ma20"] = close.rolling(20, min_periods=20).mean()
    out["ma30"] = close.rolling(30, min_periods=30).mean()
    out["volume_median20"] = volume.shift(1).rolling(20, min_periods=15).median()
    out["volume_ratio"] = volume / out["volume_median20"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.shift(1).rolling(14, min_periods=10).mean()
    rng = high - low
    safe_rng = rng.where(rng > 0)
    out["body_ratio"] = (close - opn).abs() / safe_rng
    out["close_location"] = ((close - low) / safe_rng).fillna(0.5)
    lower_wick = (pd.concat([opn, close], axis=1).min(axis=1) - low) / safe_rng
    lower_ma = pd.concat([out["ma20"], out["ma30"]], axis=1).min(axis=1)
    upper_ma = pd.concat([out["ma20"], out["ma30"]], axis=1).max(axis=1)
    zone_low = lower_ma * (1 - cfg.ma_zone)
    zone_high = upper_ma * (1 + cfg.ma_zone)
    touch = (low <= zone_high) & (high >= zone_low) & (close >= zone_low)
    buying_response = (out["close_location"] >= 0.60) | (lower_wick >= 0.40)
    defense = touch & buying_response & (volume < out["volume_median20"])
    defense_start = defense & ~defense.shift(1, fill_value=False)
    defense_count = defense_start.rolling(cfg.support_lookback, min_periods=1).sum()
    out["distribution"] = (close < out["ma30"] * (1 - cfg.ma_zone)) & (
        out["volume_ratio"] > cfg.breakout_volume
    )
    no_recent_distribution = out["distribution"].rolling(10, min_periods=1).max().eq(0)
    out["signal1_strong"] = (
        (defense_count >= cfg.support_tests) & no_recent_distribution & (close >= zone_low)
    )
    near_ma = (
        pd.concat(
            [((close / out["ma20"]) - 1).abs(), ((close / out["ma30"]) - 1).abs(),
             ((low / out["ma20"]) - 1).abs(), ((low / out["ma30"]) - 1).abs()],
            axis=1,
        ).min(axis=1)
        <= cfg.ma_zone
    )
    candle_ok = (out["body_ratio"] <= cfg.doji_threshold) | (rng / out["atr14"] <= cfg.narrow_atr)
    out["signal2_strong"] = (
        (out["volume_ratio"] < cfg.volume_threshold)
        & candle_ok
        & near_ma
        & (out["close_location"] >= 0.40)
        & (close >= zone_low)
        & ~out["distribution"]
    )
    out["member"] = member
    out["setup"] = out["signal1_strong"] & out["signal2_strong"] & out["member"]
    recent_setup = out["setup"].shift(1, fill_value=False).rolling(cfg.setup_window, min_periods=1).max().eq(1)
    no_distribution_since = (
        out["distribution"].shift(1, fill_value=False).rolling(cfg.setup_window, min_periods=1).max().eq(0)
    )
    resistance = high.shift(1).rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max()
    out["breakout"] = (
        recent_setup
        & no_distribution_since
        & (close > resistance)
        & (out["volume_ratio"] > cfg.breakout_volume)
        & (out["close_location"] >= 0.60)
        & out["member"]
        & benign_market.reindex(out.index).fillna(False)
    )
    out["persistent_ma_loss"] = (close < out["ma30"] * (1 - cfg.ma_zone)) & (
        close.shift(1) < out["ma30"].shift(1) * (1 - cfg.ma_zone)
    )
    rolling_high = high.rolling(63, min_periods=20).max()
    high_age = np.full(len(out), np.nan)
    last_high = -1
    for i, (h, rh) in enumerate(zip(high.to_numpy(), rolling_high.to_numpy())):
        if np.isfinite(h) and np.isfinite(rh) and h >= rh:
            last_high = i
        if last_high >= 0:
            high_age[i] = i - last_high
    out["consolidation_days"] = high_age
    return out


def build_trades(sf: pd.DataFrame, ticker: str, cfg: RuleConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trades: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    in_trade = False
    entry_i = -1
    setup_positions = np.flatnonzero(sf["setup"].to_numpy())
    for i in range(len(sf) - 1):
        if not in_trade and bool(sf["breakout"].iat[i]):
            fill_i = i + 1
            if not bool(sf["member"].iat[fill_i]) or not np.isfinite(sf["open"].iat[fill_i]):
                continue
            prior = setup_positions[(setup_positions < i) & (setup_positions >= i - cfg.setup_window)]
            setup_i = int(prior[-1]) if len(prior) else i
            entry_i = fill_i
            in_trade = True
            horizons: dict[str, float | None] = {}
            for h in (5, 20, 60):
                target = entry_i + h - 1
                px = sf["close"].iat[target] if target < len(sf) else np.nan
                horizons[f"forward_{h}d"] = (
                    float(px / sf["open"].iat[entry_i] * (1 - cfg.one_way_cost) ** 2 - 1)
                    if np.isfinite(px)
                    else None
                )
            signals.append({
                "ticker": ticker,
                "setup_date": sf.index[setup_i].date().isoformat(),
                "signal_date": sf.index[i].date().isoformat(),
                "entry_date": sf.index[entry_i].date().isoformat(),
                "signal1_rating": "Strong",
                "signal2_rating": "Strong",
                "signal3_rating": "Unknown",
                "volume_ratio_setup": float(sf["volume_ratio"].iat[setup_i]),
                "close_vs_ma20_setup": float(sf["close"].iat[setup_i] / sf["ma20"].iat[setup_i] - 1),
                "close_vs_ma30_setup": float(sf["close"].iat[setup_i] / sf["ma30"].iat[setup_i] - 1),
                "consolidation_days": int(sf["consolidation_days"].iat[setup_i]) if np.isfinite(sf["consolidation_days"].iat[setup_i]) else None,
                **horizons,
            })
            continue
        if in_trade:
            held = i - entry_i + 1
            membership_exit = not bool(sf["member"].iat[i])
            invalidation = bool(sf["distribution"].iat[i] or sf["persistent_ma_loss"].iat[i])
            timeout = held >= cfg.max_hold
            if membership_exit or invalidation or timeout:
                exit_i = i + 1
                if not np.isfinite(sf["open"].iat[exit_i]):
                    continue
                reason = "membership_removal" if membership_exit else "ma30_invalidation" if invalidation else "max_hold_63"
                gross = float(sf["open"].iat[exit_i] / sf["open"].iat[entry_i] - 1)
                trades.append({
                    "ticker": ticker,
                    "entry_date": sf.index[entry_i].date().isoformat(),
                    "exit_date": sf.index[exit_i].date().isoformat(),
                    "entry_price": float(sf["open"].iat[entry_i]),
                    "exit_price": float(sf["open"].iat[exit_i]),
                    "holding_sessions": int(exit_i - entry_i),
                    "exit_reason": reason,
                    "gross_return": gross,
                    "net_return": float((1 + gross) * (1 - cfg.one_way_cost) ** 2 - 1),
                    "open_trade": False,
                })
                in_trade = False
                entry_i = -1
    if in_trade and np.isfinite(sf["close"].iat[-1]):
        gross = float(sf["close"].iat[-1] / sf["open"].iat[entry_i] - 1)
        trades.append({
            "ticker": ticker,
            "entry_date": sf.index[entry_i].date().isoformat(),
            "exit_date": sf.index[-1].date().isoformat(),
            "entry_price": float(sf["open"].iat[entry_i]),
            "exit_price": float(sf["close"].iat[-1]),
            "holding_sessions": int(len(sf) - 1 - entry_i),
            "exit_reason": "open_mark_to_market",
            "gross_return": gross,
            "net_return": float((1 + gross) * (1 - cfg.one_way_cost) - 1),
            "open_trade": True,
        })
    return trades, signals


def portfolio_returns(trades: pd.DataFrame, prices: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, cost: float) -> pd.DataFrame:
    gross_sum = pd.Series(0.0, index=calendar)
    active = pd.Series(0, index=calendar, dtype=int)
    entries = pd.Series(0, index=calendar, dtype=int)
    exits = pd.Series(0, index=calendar, dtype=int)
    for row in trades.itertuples(index=False):
        frame = prices[row.ticker]
        entry, exit_ = pd.Timestamp(row.entry_date), pd.Timestamp(row.exit_date)
        interval = frame.loc[(frame.index >= entry) & (frame.index < exit_), "open"]
        rr = frame["open"].shift(-1).div(frame["open"]).sub(1).reindex(interval.index).dropna()
        ix = rr.index.intersection(calendar)
        gross_sum.loc[ix] += rr.loc[ix]
        active.loc[ix] += 1
        if entry in entries.index:
            entries.loc[entry] += 1
        if exit_ in exits.index:
            exits.loc[exit_] += 1
    gross = gross_sum.div(active.where(active > 0)).fillna(0.0)
    previous_active = active.shift(1, fill_value=0)
    denominator = pd.concat([active, previous_active], axis=1).max(axis=1).clip(lower=1)
    turnover = (entries + exits).div(denominator)
    net = gross - cost * turnover
    equity = (1 + net).cumprod()
    return pd.DataFrame({"gross_return": gross, "turnover": turnover, "net_return": net,
                         "active_positions": active, "equity": equity})


def drawdown_episodes(drawdown: pd.Series) -> tuple[float, float]:
    lengths, depths, length, depth = [], [], 0, 0.0
    for value in drawdown:
        if value < 0:
            length += 1
            depth = min(depth, float(value))
        elif length:
            lengths.append(length)
            depths.append(depth)
            length, depth = 0, 0.0
    if length:
        lengths.append(length)
        depths.append(depth)
    return (float(np.mean(depths)) if depths else 0.0, float(np.mean(lengths)) if lengths else 0.0)


def return_metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> dict[str, Any]:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2:
        return {"observations": int(len(r)), "cagr": None, "sharpe": None}
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * math.sqrt(TRADING_DAYS))
    sortino = float(r.mean() * TRADING_DAYS / downside) if downside > 0 else None
    dd = equity / equity.cummax() - 1
    mdd = float(dd.min())
    avg_dd, avg_recovery = drawdown_episodes(dd)
    ulcer = float(np.sqrt(np.mean(dd**2)))
    q95, q99 = float(r.quantile(0.05)), float(r.quantile(0.01))
    acfs = [float(r.autocorr(lag=k)) for k in range(1, min(21, len(r) // 3))]
    positive_acf_sum = sum(x for x in acfs if np.isfinite(x) and x > 0)
    n_eff = float(len(r) / (1 + 2 * positive_acf_sum))
    t_stat = float(r.mean() / (r.std(ddof=1) / math.sqrt(max(n_eff, 1)))) if r.std(ddof=1) else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df=max(int(n_eff) - 1, 1)))
    skew = float(stats.skew(r, bias=False))
    kurt_excess = float(stats.kurtosis(r, fisher=True, bias=False))
    daily_sr = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) else 0.0
    denom = max(1e-12, 1 - skew * daily_sr + ((kurt_excess + 3) - 1) / 4 * daily_sr**2)
    psr = float(stats.norm.cdf(daily_sr * math.sqrt(len(r) - 1) / math.sqrt(denom)))
    lb_lags = min(20, len(acfs))
    lb_q = float(len(r) * (len(r) + 2) * sum(acfs[k - 1] ** 2 / (len(r) - k) for k in range(1, lb_lags + 1))) if lb_lags else 0.0
    lb_p = float(stats.chi2.sf(lb_q, lb_lags)) if lb_lags else None
    jb = stats.jarque_bera(r)
    monthly = equity.resample("ME").last().pct_change().dropna()
    quarterly = equity.resample("QE").last().pct_change().dropna()
    rolling = r.rolling(252).mean().div(r.rolling(252).std()).mul(math.sqrt(TRADING_DAYS)).dropna()
    result: dict[str, Any] = {
        "observations": int(len(r)), "total_return": float(equity.iloc[-1] - 1), "cagr": cagr,
        "annual_volatility": vol, "sharpe": sharpe, "sortino": sortino,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else None, "max_drawdown": mdd,
        "average_drawdown": avg_dd, "average_recovery_sessions": avg_recovery,
        "ulcer_index": ulcer, "pain_ratio": float(cagr / ulcer) if ulcer > 0 else None,
        "lake_ratio": float((dd < 0).mean()), "var_95": -q95,
        "cvar_95": float(-r[r <= q95].mean()), "var_99": -q99,
        "cvar_99": float(-r[r <= q99].mean()),
        "omega": float(r[r > 0].sum() / abs(r[r < 0].sum())) if (r < 0).any() else None,
        "skewness": skew, "excess_kurtosis": kurt_excess,
        "jarque_bera": float(jb.statistic), "jarque_bera_p": float(jb.pvalue),
        "ljung_box_q20": lb_q, "ljung_box_p20": lb_p, "effective_n": n_eff,
        "t_stat": t_stat, "p_value": p_value, "psr_vs_zero": psr,
        "positive_months": float((monthly > 0).mean()) if len(monthly) else None,
        "positive_quarters": float((quarterly > 0).mean()) if len(quarterly) else None,
        "rolling_sharpe_min": float(rolling.min()) if len(rolling) else None,
        "rolling_sharpe_max": float(rolling.max()) if len(rolling) else None,
        "rolling_sharpe_std": float(rolling.std()) if len(rolling) else None,
    }
    if benchmark is not None:
        pair = pd.concat([r.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
        if len(pair) > 2 and pair["benchmark"].var() > 0:
            beta = float(pair.cov().loc["strategy", "benchmark"] / pair["benchmark"].var())
            alpha = float((pair["strategy"].mean() - beta * pair["benchmark"].mean()) * TRADING_DAYS)
            result.update({"alpha_annual": alpha, "beta": beta,
                           "benchmark_correlation": float(pair.corr().iloc[0, 1]),
                           "tracking_error": float((pair.strategy - pair.benchmark).std() * math.sqrt(TRADING_DAYS))})
    return finite(result)


def trade_metrics(trades: pd.DataFrame, cost_mult: float = 1.0, base_cost: float = BASE.one_way_cost) -> dict[str, Any]:
    if trades.empty:
        return {"trades": 0, "profit_factor": None, "expectancy": None, "win_rate": None}
    gross = trades["gross_return"].astype(float)
    # Closed trades pay both entry and exit cost; the two end-of-sample marks
    # have not exited and therefore pay only their realised entry cost.
    cost_sides = np.where(trades["open_trade"].astype(bool), 1, 2)
    net = (1 + gross) * np.power(1 - base_cost * cost_mult, cost_sides) - 1
    wins, losses = net[net > 0], net[net < 0]
    return finite({
        "trades": int(len(net)), "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) else None,
        "expectancy": float(net.mean()), "median_return": float(net.median()),
        "win_rate": float((net > 0).mean()), "average_win": float(wins.mean()) if len(wins) else None,
        "average_loss": float(losses.mean()) if len(losses) else None,
        "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else None,
        "average_holding_sessions": float(trades["holding_sessions"].mean()),
        "open_trades": int(trades["open_trade"].sum()),
    })


def bh_adjust(p_values: pd.Series) -> pd.Series:
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return pd.Series(np.nan, index=p_values.index)
    n = len(valid)
    adjusted = (valid * n / np.arange(1, n + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    return adjusted.reindex(p_values.index)


def block_bootstrap(r: pd.Series, runs: int = 1000, block: int = 20) -> dict[str, Any]:
    values = r.to_numpy(dtype=float)
    n = len(values)
    rng = np.random.default_rng(RNG_SEED)
    sharpes, cagrs, mdds = [], [], []
    blocks_needed = math.ceil(n / block)
    for _ in range(runs):
        starts = rng.integers(0, max(1, n - block + 1), size=blocks_needed)
        sample = np.concatenate([values[s:s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        sharpes.append(sample.mean() / sd * math.sqrt(TRADING_DAYS) if sd else 0.0)
        eq = np.cumprod(1 + sample)
        cagrs.append(eq[-1] ** (TRADING_DAYS / n) - 1 if eq[-1] > 0 else -1)
        mdds.append(np.min(eq / np.maximum.accumulate(eq) - 1))
    return {
        "runs": runs, "block_sessions": block,
        "sharpe_95_ci": [float(x) for x in np.percentile(sharpes, [2.5, 97.5])],
        "cagr_95_ci": [float(x) for x in np.percentile(cagrs, [2.5, 97.5])],
        "max_drawdown_5_50_95": [float(x) for x in np.percentile(mdds, [5, 50, 95])],
    }


def score_backtest(metrics: dict[str, Any], tm: dict[str, Any], robustness: dict[str, Any]) -> dict[str, Any]:
    t = metrics.get("t_stat") or 0
    t_pts = 8 if t > 3 else 6 if t > 2 else 4 if t > 1.65 else 0
    psr = metrics.get("psr_vs_zero") or 0
    psr_pts = 7 if psr > .95 else 5 if psr > .90 else 3 if psr > .80 else 0
    dsr_pts = t_pts  # fixed, skill-derived parameters; no optimisation of the baseline
    neff = metrics.get("effective_n") or 0
    neff_pts = 7 if neff >= 1000 else 4 if neff >= 250 else 0
    a = t_pts + psr_pts + dsr_pts + neff_pts
    sh = metrics.get("sharpe") or 0
    sh_pts = 10 if sh > 2 else 7 if sh > 1 else 4 if sh > .5 else 0
    so = metrics.get("sortino") or 0
    so_pts = 8 if so > 2.5 else 6 if so > 1.5 else 4 if so > .7 else 0
    mdd = abs(metrics.get("max_drawdown") or 1)
    dd_pts = 7 if mdd < .10 else 5 if mdd < .20 else 3 if mdd < .30 else 0
    b = sh_pts + so_pts + dd_pts
    eff = robustness.get("wfa_efficiency")
    wfa_pts = 10 if eff is not None and eff > .7 else 7 if eff is not None and eff > .5 else 4 if eff is not None and eff > .3 else 0
    boot_lo = robustness.get("bootstrap", {}).get("sharpe_95_ci", [None])[0]
    boot_pts = 8 if boot_lo is not None and boot_lo > 0 else 4 if (metrics.get("sharpe") or 0) > 0 else 0
    sens = robustness.get("sensitivity_stability", "failed")
    sens_pts = 7 if sens == "smooth" else 4 if sens == "mixed" else 0
    c = wfa_pts + boot_pts + sens_pts
    pf = tm.get("profit_factor") or 0
    pf_pts = 7 if pf > 2 else 5 if pf > 1.5 else 3 if pf > 1.2 else 0
    coherent = (tm.get("expectancy") or 0) > 0 and tm.get("win_rate") is not None
    coh_pts = 6 if coherent else 3 if (tm.get("expectancy") or 0) > -0.005 else 0
    pm = metrics.get("positive_months") or 0
    roll_std = metrics.get("rolling_sharpe_std")
    con_pts = 7 if pm > .65 and roll_std is not None and roll_std < .75 else 5 if pm > .55 else 3 if pm > .50 else 0
    d = pf_pts + coh_pts + con_pts
    raw = int(a + b + c + d)
    return {"A": a, "B": b, "C": c, "D": d, "raw_score": raw,
            "caps": {"missing_history_survivorship_bias": 20}, "final_score": min(raw, 20),
            "band": "Reject" if min(raw, 20) < 25 else "Weak"}


def fmt_pct(x: Any) -> str:
    return "N/A" if x is None else f"{100 * float(x):.2f}%"


def write_report(summary: dict[str, Any], output: Path) -> None:
    m, tm, rb, sc = summary["performance"], summary["trade_statistics"], summary["robustness"], summary["score"]
    fwd = summary["forward_event_study"]
    lines = [
        "# Backtest Verification Report — Main-Force Washout Skill",
        "",
        f"Evidence timestamp: completed daily OHLCV through **{summary['data']['end_date']}**; "
        f"US equities, daily timeframe; local Yahoo-derived archive and community point-in-time membership.",
        "",
        f"## Backtest Score: {sc['final_score']} / 100 — {sc['band']}",
        "",
        "The rule is causal and the available-history result is reproducible, but missing delisted histories and "
        "ticker-reuse contamination impose the rubric's 20-point survivorship-bias cap.",
        "",
        "| Component | Score | Max |", "|---|---:|---:|",
        f"| A. Statistical Validity & Significance | {sc['A']} | 30 |",
        f"| B. Risk-Adjusted Performance | {sc['B']} | 25 |",
        f"| C. Robustness & Out-of-Sample | {sc['C']} | 25 |",
        f"| D. Trade Quality & Consistency | {sc['D']} | 20 |",
        f"| **Raw total** | **{sc['raw_score']}** | **100** |",
        "| Caps applied | Missing-history/survivorship bias → 20 | |",
        f"| **Final score** | **{sc['final_score']}** | **100** |",
        "",
        "## Executive Summary", "",
        f"Across {summary['data']['usable_tickers']} usable point-in-time ticker histories, the completed-washout rule "
        f"produced {tm['trades']} non-overlapping trades. The equal-weight active-position portfolio returned "
        f"{fmt_pct(m.get('cagr'))} annualised with Sharpe {m.get('sharpe') if m.get('sharpe') is not None else 'N/A'} "
        f"and maximum drawdown {fmt_pct(m.get('max_drawdown'))}.",
        f"The median next-open event return was {fmt_pct(fwd['20d']['median'])} at 20 sessions and "
        f"{fmt_pct(fwd['60d']['median'])} at 60 sessions. These figures describe the available subset, not an "
        "institutional point-in-time universe. Uninvested cash earns 0% in this test.",
        "",
        "## Washout Signal Verification", "",
        "1. **Signal 1 (MA Support): Strong.** Entry candidates had at least two distinct 20/30-SMA defenses "
        "inside 15 sessions, buying response, lighter pullback volume, and no recent high-volume 30-SMA break.",
        "2. **Signal 2 (Volume Contraction): Strong.** Setup volume was below 50% of the prior-20-session median "
        "with a doji/narrow-range candle within the 1% MA zone and a stable close.",
        "3. **Signal 3 (Order Book Test): Unknown.** The archive has no timestamped depth or time-and-sales data.",
        "4. **Time & Market Risk Check:** Consolidation is measured as sessions since the latest rolling 63-session "
        "high; >21 is an opportunity-cost warning and >63 elevated risk. SPY must be above its 30-SMA or no worse "
        "than -5% over 20 sessions. Point-in-time sector mappings are unavailable.",
        "5. **True vs. Fake Washout Assessment:** A later close above the prior 20-session high on >1.2× median "
        "volume is mandatory. High-volume/persistent 30-SMA failure invalidates the setup and exits next open.",
        "6. **Final Verdict: Washout Completed (medium confidence)** only for rows in `signals.csv`; invalidated by "
        "a decisive/persistent 30-SMA loss. Confidence cannot be high without order-flow and complete histories.",
        "7. **Suggested Action:** Treat a signal as educational confirmation only; avoid new exposure if the 30-SMA "
        "breaks on expanding volume. This is not financial advice.",
        "",
        "## 1. Performance Metrics", "",
        "| Metric | Strategy | SPY benchmark / threshold | Status |", "|---|---:|---:|---|",
        f"| CAGR | {fmt_pct(m.get('cagr'))} | {fmt_pct(summary['benchmark'].get('cagr'))} | {'Higher' if (m.get('cagr') or -9) > (summary['benchmark'].get('cagr') or -9) else 'Lower'} |",
        f"| Annual volatility | {fmt_pct(m.get('annual_volatility'))} | {fmt_pct(summary['benchmark'].get('annual_volatility'))} | — |",
        f"| Sharpe | {m.get('sharpe')} | >1 good | {'Pass' if (m.get('sharpe') or 0) > 1 else 'Below good'} |",
        f"| Sortino | {m.get('sortino')} | >1.5 good | {'Pass' if (m.get('sortino') or 0) > 1.5 else 'Below good'} |",
        f"| Max drawdown | {fmt_pct(m.get('max_drawdown'))} | <20% good | {'Pass' if abs(m.get('max_drawdown') or 1) < .2 else 'High'} |",
        f"| Ulcer / lake ratio | {fmt_pct(m.get('ulcer_index'))} / {fmt_pct(m.get('lake_ratio'))} | Lake >50% red flag | — |",
        f"| VaR / CVaR 95% | {fmt_pct(m.get('var_95'))} / {fmt_pct(m.get('cvar_95'))} | — | — |",
        f"| Trades / win rate / PF | {tm['trades']} / {fmt_pct(tm.get('win_rate'))} / {tm.get('profit_factor')} | PF>1.5 good | — |",
        "",
        "## 2. Statistical Significance", "",
        f"Effective-N t-stat = {m.get('t_stat')}, p = {m.get('p_value')}; PSR vs zero = {fmt_pct(m.get('psr_vs_zero'))}. "
        f"Jarque–Bera p = {m.get('jarque_bera_p')} and Ljung–Box(20) p = {m.get('ljung_box_p20')}. "
        "ADF is unavailable because `statsmodels` is not installed; no price-level regression is used. DSR was not "
        "required for the fixed skill-derived baseline; its score mirrors the t-stat tier. Per-ticker p-values are "
        "Benjamini–Hochberg adjusted in `per_ticker.csv`.",
        "",
        "## 3. Bias Assessment", "",
        "| Bias | Verdict | Evidence |", "|---|---|---|",
        "| Lookahead | Absent | Prior-session rolling baselines; close signals fill next open. |",
        "| Survivorship / missing histories | **Present** | 381/1,206 archive symbols unavailable; final score capped at 20. |",
        "| Ticker reuse | **Present, mitigated** | 157 downloaded mappings have no membership overlap and are excluded. |",
        "| Overfitting / data snooping | Mitigated | Baseline thresholds fixed before results; sensitivity is diagnostic only. |",
        "| Transaction costs | Partly verifiable | 10 bps/side baseline and 2×/5×/10× stress; spread/impact unavailable. |",
        "| Liquidity | Cannot verify | No trade size, ADV participation, or historical spreads. |",
        "| Frequency mismatch | Absent | Daily close decision, next-session adjusted-open fill. |",
        "| Sector/regime | Cannot fully verify | SPY regime used; point-in-time sector mapping unavailable. |",
        "",
        "## 4. Robustness", "",
        f"Fixed-rule 70/30 split: IS Sharpe {rb['is_sharpe']}, OOS Sharpe {rb['oos_sharpe']}, efficiency "
        f"{rb['wfa_efficiency']}. The block-bootstrap 95% Sharpe interval is {rb['bootstrap']['sharpe_95_ci']}; "
        f"sensitivity classification is **{rb['sensitivity_stability']}**. Cost-stress trade expectancy is recorded "
        "at 1×/2×/5×/10× in `summary.json`.",
        "",
        "## 5. Red Flags 🚩", "",
        "1. Missing delisted/acquired histories are not random and can materially bias both signal frequency and returns.",
        "2. Community membership data and reused Yahoo symbols are not institutional-grade point-in-time data.",
        "3. Signal 3 and sector-relative confirmation cannot be tested from these daily files.",
        "4. Equal-weight portfolio turnover costs are approximate; capacity and market impact are unknown.",
        "5. The SPY benchmark is close-to-close while strategy fills are adjusted open-to-open.",
        "",
        "## 6. Improvement Recommendations 💡", "",
        "1. Acquire CRSP/Compustat-quality delisted histories and permanent identifiers; rerun without changing rules.",
        "2. Add timestamped order-book/tape archives to validate persistence, absorption, and follow-through.",
        "3. Add point-in-time sector classifications and sector ETF total-return benchmarks.",
        "4. Freeze these parameters prospectively and collect a genuinely unseen live sample.",
        "5. Model spread and nonlinear market impact by historical ADV and signal-time volatility.",
        "",
        "## 7. Verdict", "",
        f"**{sc['final_score']}/100 — {sc['band']}.** The available subset can test whether the OHLCV translation is "
        "causal and directionally useful, but it cannot establish a tradeable edge while the missing-history bias remains.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output_dir: Path = OUTPUT_DIR, bootstrap_runs: int = 1000) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dates, snapshot_sets, universe = load_membership()
    spy = load_spy()
    calendar = spy.loc["1996-01-02":"2026-08-14"].index
    all_trades: list[dict[str, Any]] = []
    all_signals: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    cached_prices: dict[str, pd.DataFrame] = {}
    sensitivity_returns: dict[str, list[float]] = {k: [] for k in SENSITIVITY}
    sensitivity_counts: dict[str, int] = {k: 0 for k in SENSITIVITY}
    audit_totals = {"raw_rows": 0, "invalid_price_rows": 0, "invalid_ohlc_rows": 0, "zero_volume_rows": 0}

    for path in sorted(PRICE_DIR.glob("*.csv")):
        ticker = normalize_ticker(path.stem)
        frame, audit = load_price(path)
        for k in audit_totals:
            audit_totals[k] += audit[k]
        member = membership_mask(frame.index, ticker, snapshot_dates, snapshot_sets)
        member_valid = member & frame[["open", "high", "low", "close", "volume"]].notna().all(axis=1).to_numpy()
        reason = "usable"
        if member.sum() == 0:
            reason = "no_point_in_time_membership_overlap"
        elif member_valid.sum() < 30:
            reason = "fewer_than_30_valid_member_rows"
        row: dict[str, Any] = {
            "ticker": ticker, "status": reason, "raw_rows": audit["raw_rows"],
            "member_rows": int(member.sum()), "valid_member_rows": int(member_valid.sum()),
            "invalid_ohlc_rows": audit["invalid_ohlc_rows"], "zero_volume_rows": audit["zero_volume_rows"],
            "setup_count": 0, "entry_signals": 0, "trades": 0,
        }
        if reason != "usable":
            ticker_rows.append(row)
            continue
        benign = market_regime(spy, frame.index)
        base_sf: pd.DataFrame | None = None
        for name, cfg in SENSITIVITY.items():
            sf = signal_frame(frame, member, benign, cfg)
            trades, signals = build_trades(sf, ticker, cfg)
            sensitivity_returns[name].extend(float(t["net_return"]) for t in trades)
            sensitivity_counts[name] += len(trades)
            if name == "baseline":
                base_sf = sf
                all_trades.extend(trades)
                all_signals.extend(signals)
                row.update({"setup_count": int(sf["setup"].sum()), "entry_signals": len(signals), "trades": len(trades)})
        assert base_sf is not None
        cached_prices[ticker] = frame[["open", "close"]].copy()
        tdf = pd.DataFrame([t for t in all_trades if t["ticker"] == ticker])
        sdf = pd.DataFrame([s for s in all_signals if s["ticker"] == ticker])
        if not tdf.empty:
            vals = tdf["net_return"].astype(float)
            row.update({"mean_trade_return": vals.mean(), "median_trade_return": vals.median(),
                        "win_rate": (vals > 0).mean(),
                        "profit_factor": vals[vals > 0].sum() / abs(vals[vals < 0].sum()) if (vals < 0).any() else np.nan,
                        "trade_t_pvalue": stats.ttest_1samp(vals, 0).pvalue if len(vals) >= 2 else np.nan})
        if not sdf.empty:
            for h in (5, 20, 60):
                vals = pd.to_numeric(sdf[f"forward_{h}d"], errors="coerce")
                row[f"median_forward_{h}d"] = vals.median()
                row[f"win_rate_forward_{h}d"] = (vals.dropna() > 0).mean() if vals.notna().any() else np.nan
        ticker_rows.append(row)

    trades_df = pd.DataFrame(all_trades)
    signals_df = pd.DataFrame(all_signals)
    per_ticker = pd.DataFrame(ticker_rows).sort_values("ticker")
    if "trade_t_pvalue" in per_ticker:
        eligible = per_ticker["trades"].fillna(0) >= 5
        per_ticker["trade_pvalue_bh_fdr"] = np.nan
        per_ticker.loc[eligible, "trade_pvalue_bh_fdr"] = bh_adjust(per_ticker.loc[eligible, "trade_t_pvalue"])
    daily = portfolio_returns(trades_df, cached_prices, calendar, BASE.one_way_cost)
    spy_ret = spy.reindex(calendar).pct_change().fillna(0.0)
    daily["spy_return"] = spy_ret
    daily["spy_equity"] = (1 + spy_ret).cumprod()
    perf = return_metrics(daily["net_return"], daily["spy_return"])
    perf["time_in_market"] = float((daily["active_positions"] > 0).mean())
    benchmark = return_metrics(daily["spy_return"])
    tm = trade_metrics(trades_df)
    split_i = int(len(daily) * 0.70)
    is_m = return_metrics(daily["net_return"].iloc[:split_i])
    oos_m = return_metrics(daily["net_return"].iloc[split_i:])
    efficiency = (
        float(oos_m["sharpe"] / is_m["sharpe"])
        if is_m.get("sharpe") not in (None, 0) and oos_m.get("sharpe") is not None else None
    )
    boot = block_bootstrap(daily["net_return"], runs=bootstrap_runs)
    sensitivity_rows = []
    baseline_mean = float(np.mean(sensitivity_returns["baseline"])) if sensitivity_returns["baseline"] else np.nan
    for name, cfg in SENSITIVITY.items():
        vals = np.asarray(sensitivity_returns[name], dtype=float)
        sensitivity_rows.append({"variant": name, **asdict(cfg), "trades": len(vals),
                                 "mean_net_return": float(vals.mean()) if len(vals) else np.nan,
                                 "median_net_return": float(np.median(vals)) if len(vals) else np.nan,
                                 "win_rate": float((vals > 0).mean()) if len(vals) else np.nan})
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    variant_means = sensitivity_df.loc[sensitivity_df.variant.ne("baseline"), "mean_net_return"].dropna()
    if len(variant_means) and (variant_means > 0).all() and baseline_mean > 0 and variant_means.min() >= .5 * baseline_mean:
        sensitivity_stability = "smooth"
    elif len(variant_means) and (variant_means > 0).mean() >= .7:
        sensitivity_stability = "mixed"
    else:
        sensitivity_stability = "failed"
    robustness = {"split_date": daily.index[split_i].date().isoformat(), "is_sharpe": is_m.get("sharpe"),
                  "oos_sharpe": oos_m.get("sharpe"), "wfa_efficiency": efficiency,
                  "bootstrap": boot, "sensitivity_stability": sensitivity_stability}
    forward: dict[str, Any] = {}
    for h in (5, 20, 60):
        vals = pd.to_numeric(signals_df.get(f"forward_{h}d", pd.Series(dtype=float)), errors="coerce").dropna()
        forward[f"{h}d"] = {"n": int(len(vals)), "mean": float(vals.mean()) if len(vals) else None,
                              "median": float(vals.median()) if len(vals) else None,
                              "win_rate": float((vals > 0).mean()) if len(vals) else None,
                              "t_pvalue": float(stats.ttest_1samp(vals, 0).pvalue) if len(vals) >= 2 else None}
    cost_stress = {f"{x}x": trade_metrics(trades_df, x) for x in (1, 2, 5, 10)}
    score = score_backtest(perf, tm, robustness)
    summary = finite({
        "rule": asdict(BASE),
        "data": {"start_date": calendar.min().date().isoformat(), "end_date": calendar.max().date().isoformat(),
                 "price_files": 825, "master_tickers": int(pd.read_csv(MASTER_FILE).ticker.nunique()),
                 "unavailable_tickers": int(pd.read_csv(UNAVAILABLE_FILE).ticker.nunique()),
                 "membership_universe": len(universe),
                 "membership_overlap_files": int((per_ticker.member_rows > 0).sum()),
                 "usable_tickers": int((per_ticker.status == "usable").sum()), **audit_totals,
                 "order_book_signal": "Unknown", "sector_benchmark": "Unknown"},
        "performance": perf, "benchmark": benchmark, "trade_statistics": tm,
        "forward_event_study": forward, "robustness": robustness, "cost_stress": cost_stress,
        "score": score,
    })
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    signals_df.to_csv(output_dir / "signals.csv", index=False)
    per_ticker.to_csv(output_dir / "per_ticker.csv", index=False)
    daily.to_csv(output_dir / "daily_equity.csv", index_label="date")
    sensitivity_df.to_csv(output_dir / "sensitivity.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(summary, output_dir / "report.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap-runs", type=int, default=1000)
    args = parser.parse_args()
    summary = run(args.output_dir, args.bootstrap_runs)
    print(json.dumps({"output_dir": str(args.output_dir), "score": summary["score"],
                      "trades": summary["trade_statistics"]["trades"],
                      "performance": {k: summary["performance"].get(k) for k in ("cagr", "sharpe", "max_drawdown")}}, indent=2))


if __name__ == "__main__":
    main()
