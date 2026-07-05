"""
QQQ / NASDAQ 100 Breadth Strategy — statistical validation

Runs three robustness checks on the baseline strategy (and the XLY/XLP
regime-filter variant) from qqq_sector_experiment.py:

  1. Daily-return significance — t-stat and Probabilistic Sharpe Ratio (PSR)
     on ~9k daily observations instead of 11 trades; plus a paired test of
     strategy-minus-benchmark excess returns.
  2. Parameter sensitivity map — each key parameter swept ±50% around its
     tuned value; a robust strategy degrades smoothly, not cliff-edge.
  3. Monte Carlo — (a) entry/exit timing jitter of ±5 trading days, 1,000
     runs; (b) bootstrap resampling of trade returns for a max-drawdown
     distribution.

Data: NASDAQ100.csv, S5TH.csv, VIX.csv, XLY.csv, XLP.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path(__file__).parent

# ── Baseline parameters (same as qqq_backtest.py) ─────────────────────────────
BASE_PARAMS = {
    "buy_thresh":   26.0,   # breadth200 buy threshold
    "vix_thresh":   30.0,   # VIX vote
    "div_window":   60,     # divergence lookback (trading days)
    "price_rise":   3.0,    # % price rise over window
    "breadth_fall": 20.0,   # pts breadth drop over window
    "breadth_cap":  60.0,   # breadth must be below this to sell
}
MA200_WINDOW    = 200
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
COOLDOWN_DAYS   = 15

N_MC          = 1000
JITTER_DAYS   = 5
RNG           = np.random.default_rng(42)


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _load_price_csv(name: str) -> pd.Series:
    df = pd.read_csv(DATA_DIR / name, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()
    return _parse_price(df["Price"])


def _load_breadth() -> pd.Series:
    """Prefer the continuous daily series (breadth_daily.csv, 2002+) built by
    build_breadth_daily.py; S5TH.csv alone is only daily from 2007 and sparse
    (bimonthly) before that, which corrupts row-based windows."""
    daily = DATA_DIR / "breadth_daily.csv"
    if daily.exists():
        df = pd.read_csv(daily)
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        return df.set_index("Date")["breadth"].sort_index()
    s = _load_price_csv("S5TH.csv")
    return s[s.index >= "2007-01-01"]


def load_raw() -> pd.DataFrame:
    ndx     = _load_price_csv("NASDAQ100.csv").rename("price")
    breadth = _load_breadth().rename("breadth")
    vix     = _load_price_csv("VIX.csv").rename("vix")
    xly     = _load_price_csv("XLY.csv").rename("xly")
    xlp     = _load_price_csv("XLP.csv").rename("xlp")

    df = pd.concat([ndx, breadth, vix, xly, xlp], axis=1, sort=True)
    df = df[df["breadth"].notna() & df["price"].notna()]
    df["vix"] = df["vix"].ffill()
    df["ma200"] = df["price"].rolling(MA200_WINDOW).mean()

    # compute the ratio on ETF trading days only — the merged index has NaN gaps
    # that would poison the rolling mean
    ratio = (xly / xlp).dropna()
    risk_on = ratio > ratio.rolling(MA200_WINDOW).mean()
    df["risk_on"] = risk_on.reindex(df.index).ffill()
    return df


def run(df: pd.DataFrame, p: dict, regime: bool = False):
    """Vectorised precompute + state-machine loop. Returns (daily values, trades)."""
    price   = df["price"].to_numpy()
    breadth = df["breadth"].to_numpy()
    vix     = df["vix"].to_numpy()
    ma200   = df["ma200"].to_numpy()
    riskon  = df["risk_on"].to_numpy(dtype=object)
    dates   = df.index

    w = int(p["div_window"])
    pp = np.concatenate([np.full(w, np.nan), price[:-w]])
    bp = np.concatenate([np.full(w, np.nan), breadth[:-w]])
    with np.errstate(invalid="ignore"):
        price_rose   = (price - pp) / pp * 100 >= p["price_rise"]
        breadth_fell = (bp - breadth) >= p["breadth_fall"]
        vix_vote     = np.isnan(vix) | (vix > p["vix_thresh"])
        ma_vote      = np.isnan(ma200) | (price > ma200)
    vote_gate = vix_vote | ma_vote
    # Trend re-entry: fresh close back above MA200. Divergence-only exit here, so
    # the gate reduces to "price back above the prior exit price".
    cross_up = np.zeros(len(price), dtype=bool)
    cross_up[1:] = (price[1:] > ma200[1:]) & (price[:-1] <= ma200[:-1])

    position, portfolio = False, INITIAL_CAPITAL
    eff_entry = 0.0
    entry_i = -1
    cooldown_until = None
    last_exit_price = None
    trades, values = [], np.empty(len(price))

    for i in range(len(price)):
        if not position:
            cd_ok = cooldown_until is None or dates[i] > cooldown_until
            washout = breadth[i] < p["buy_thresh"] and vote_gate[i]
            trend = bool(cross_up[i]) and (last_exit_price is not None and price[i] > last_exit_price)
            buy = cd_ok and (washout or trend)
            if buy and regime:
                r = riskon[i]
                buy = (r is None) or (isinstance(r, float) and np.isnan(r)) or bool(r)
            if buy:
                portfolio -= COMMISSION
                eff_entry = price[i] * (1 + SLIPPAGE)
                entry_i = i
                position = True
        else:
            if price_rose[i] and breadth_fell[i] and breadth[i] < p["breadth_cap"]:
                eff_exit = price[i] * (1 - SLIPPAGE)
                ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= 1 + ret
                portfolio -= COMMISSION
                cooldown_until = dates[i] + pd.Timedelta(days=COOLDOWN_DAYS)
                last_exit_price = price[i]
                trades.append({"entry_i": entry_i, "exit_i": i, "return": ret})
                position = False
        values[i] = portfolio * (price[i] * (1 - SLIPPAGE) / eff_entry) if position else portfolio

    return pd.Series(values, index=dates), trades


def sharpe(dr: np.ndarray) -> float:
    sd = dr.std(ddof=1)
    return dr.mean() / sd * np.sqrt(252) if sd > 0 else 0.0


def psr(dr: np.ndarray, sr_benchmark_daily: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012), daily units."""
    n = len(dr)
    sr = dr.mean() / dr.std(ddof=1)
    g3 = stats.skew(dr)
    g4 = stats.kurtosis(dr, fisher=False)
    denom = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2)
    z = (sr - sr_benchmark_daily) * np.sqrt(n - 1) / denom
    return stats.norm.cdf(z)


def significance_report(name: str, values: pd.Series, bench: pd.Series) -> None:
    dr  = values.pct_change().dropna()
    br  = bench.pct_change().dropna()
    ex  = (dr - br).dropna()

    active = dr[dr != 0]  # days actually in the market
    t_mean, p_mean = stats.ttest_1samp(active, 0)
    t_ex, p_ex = stats.ttest_1samp(ex, 0)

    print(f"\n── {name}: daily-return significance ──")
    print(f"  Daily obs (total / in-market)  : {len(dr):,} / {len(active):,}")
    print(f"  Annualised Sharpe              : {sharpe(dr.to_numpy()):.2f}")
    print(f"  t-stat, in-market daily returns: {t_mean:+.2f}  (p = {p_mean:.2e})")
    print(f"  PSR vs SR*=0                   : {psr(dr.to_numpy()):.1%}")
    print(f"  Excess vs buy&hold: t = {t_ex:+.2f}  (p = {p_ex:.3f})  "
          f"mean = {ex.mean()*252:+.1%}/yr")


def sensitivity_map(df: pd.DataFrame) -> None:
    multipliers = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5]
    print("\n── Parameter sensitivity (baseline, Sharpe ratio per multiplier) ──")
    header = "  " + f"{'parameter':<14}" + "".join(f"{m:>8.2f}x" for m in multipliers)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in BASE_PARAMS:
        row = []
        for m in multipliers:
            p = dict(BASE_PARAMS)
            p[key] = p[key] * m
            if key == "div_window":
                p[key] = max(5, int(round(p[key])))
            values, _ = run(df, p)
            row.append(sharpe(values.pct_change().dropna().to_numpy()))
        cells = "".join(f"{v:>9.2f}" for v in row)
        print(f"  {key:<14}{cells}")
    print("  (baseline column = 1.00x; look for cliffs vs smooth decay)")


def mc_timing_jitter(df: pd.DataFrame, trades: list[dict]) -> None:
    price = df["price"].to_numpy()
    n = len(price)
    finals = np.empty(N_MC)
    for k in range(N_MC):
        portfolio = INITIAL_CAPITAL
        for t in trades:
            e = int(np.clip(t["entry_i"] + RNG.integers(-JITTER_DAYS, JITTER_DAYS + 1), 0, n - 1))
            x = int(np.clip(t["exit_i"] + RNG.integers(-JITTER_DAYS, JITTER_DAYS + 1), 0, n - 1))
            if x <= e:
                x = min(e + 1, n - 1)
            eff_in  = price[e] * (1 + SLIPPAGE)
            eff_out = price[x] * (1 - SLIPPAGE)
            portfolio = (portfolio - COMMISSION) * (eff_out / eff_in) - COMMISSION
        finals[k] = portfolio

    actual = INITIAL_CAPITAL
    for t in trades:
        actual = (actual - COMMISSION) * (1 + t["return"]) - COMMISSION

    pct = np.percentile(finals, [5, 25, 50, 75, 95])
    print(f"\n── Monte Carlo: entry/exit timing jitter ±{JITTER_DAYS}d ({N_MC} runs) ──")
    print(f"  Actual final value : ${actual:,.0f}")
    print(f"  Jittered percentiles: 5% ${pct[0]:,.0f} | 25% ${pct[1]:,.0f} | "
          f"median ${pct[2]:,.0f} | 75% ${pct[3]:,.0f} | 95% ${pct[4]:,.0f}")
    print(f"  Share of runs beating buy&hold multiple: "
          f"{(finals / INITIAL_CAPITAL > price[-1] / price[0]).mean():.0%}")


def bootstrap_drawdown(trades: list[dict]) -> None:
    rets = np.array([t["return"] for t in trades])
    mdds = np.empty(N_MC)
    for k in range(N_MC):
        sample = RNG.choice(rets, size=len(rets), replace=True)
        equity = np.cumprod(1 + sample)
        peak = np.maximum.accumulate(equity)
        mdds[k] = ((equity - peak) / peak).min()
    pct = np.percentile(mdds, [5, 50, 95])
    print(f"\n── Bootstrap of {len(rets)} trade returns ({N_MC} resamples) ──")
    print(f"  Trade-level max-drawdown distribution: "
          f"5% {pct[0]:.1%} | median {pct[1]:.1%} | 95% {pct[2]:.1%}")
    print("  (trade-close granularity — intra-trade drawdown is deeper)")


def main() -> None:
    df = load_raw()
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")

    base_values, base_trades = run(df, BASE_PARAMS)
    regime_values, regime_trades = run(df, BASE_PARAMS, regime=True)
    bench = INITIAL_CAPITAL * df["price"] / df["price"].iloc[0]

    print(f"Baseline: {len(base_trades)} closed trades, final ${base_values.iloc[-1]:,.0f}")
    print(f"Regime  : {len(regime_trades)} closed trades, final ${regime_values.iloc[-1]:,.0f}")

    significance_report("Baseline", base_values, bench)
    significance_report("Regime (XLY/XLP)", regime_values, bench)

    sensitivity_map(df)

    mc_timing_jitter(df, base_trades)
    bootstrap_drawdown(base_trades)


if __name__ == "__main__":
    main()
