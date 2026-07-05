"""
QQQ / NASDAQ 100 Breadth Strategy — folder-wide indicator scan

Tests every unused daily indicator in this folder as an add-on to the baseline
strategy, with walk-forward discipline (parameters selected by Sharpe on one
half of 2002–2026, reported on the other half only):

  entry gates (block a washout buy unless the condition holds):
    HY-calm  : high-yield credit spread (BAMLH0A0HYM2) below its 200d MA + X
  extra exits (sell on top of the baseline bearish-divergence exit):
    HY-stress: HY spread rose ≥ X pts over 60 days
    NHNL     : 20d avg of (new highs − new lows) < X while price rose ≥ 3%/60d
    ADline   : cumulative advance-decline line fell over 60d while price rose
    S5OH-div : % above 100-day MA fell ≥ X pts over 60d while price rose
    10Y-spike: 10-year yield rose ≥ X pts over 60 days
    RSI exit : SPX 14-day RSI ≥ X (overbought) while holding
    MACD exit: SPX MACD(12,26,9) bearish cross while MACD > X% of price
  more entry gates:
    RSI entry : SPX 14-day RSI < X at the washout buy (oversold confirm)
    MACD entry: SPX MACD histogram > X% of price at the buy (momentum turning)

Skipped (too slow-moving for an ~8-trade strategy): ShillerPE, ForwardPE,
us_ppi_yoy. Skipped as uninformative: UCHG (unchanged issues count).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from qqq_validation import (
    COMMISSION, INITIAL_CAPITAL, SLIPPAGE, COOLDOWN_DAYS, BASE_PARAMS,
    _load_price_csv, load_raw, sharpe,
)

DATA_DIR   = Path(__file__).parent
SPLIT_DATE = pd.Timestamp("2014-01-01")
WINDOW     = 60   # lookback for all change-based conditions (matches divergence window)


def _load_tv_close(name: str, date_col: str) -> pd.Series:
    """TradingView-style export: <date_col>,open,high,low,close[,...] with ISO dates."""
    df = pd.read_csv(DATA_DIR / name)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col)["close"].sort_index()


def load_indicators(index: pd.DatetimeIndex) -> pd.DataFrame:
    hy   = _load_tv_close("BAMLH0A0HYM2.csv", "date")
    mahn = _load_tv_close("MAHN.csv", "time")
    maln = _load_tv_close("MALN.csv", "time")
    adv  = _load_tv_close("ADV.csv", "time")
    decl = _load_tv_close("DECL.csv", "time")
    s5oh = _load_price_csv("S5OH.csv")
    us10 = _load_price_csv("US10Y.csv")
    spx  = _load_price_csv("SPX.csv")

    # SPX 14-day RSI (Wilder smoothing)
    delta = spx.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)

    # SPX MACD(12,26,9), normalised to % of price so thresholds are scale-free
    ema12 = spx.ewm(span=12, adjust=False).mean()
    ema26 = spx.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / spx * 100
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    ind = pd.DataFrame(index=index)
    ind["hy"]      = hy.reindex(index).ffill()
    ind["hy_ma"]   = hy.rolling(200).mean().reindex(index).ffill()
    ind["nhnl"]    = (mahn - maln).rolling(20).mean().reindex(index).ffill()
    ind["ad_line"] = (adv - decl).cumsum().reindex(index).ffill()
    ind["s5oh"]    = s5oh.reindex(index).ffill()
    ind["us10"]    = us10.reindex(index).ffill()
    ind["rsi"]     = rsi.reindex(index).ffill()
    ind["macd"]    = macd.reindex(index).ffill()
    ind["hist"]    = hist.reindex(index).ffill()
    return ind


def precompute(df: pd.DataFrame, ind: pd.DataFrame) -> dict:
    p = BASE_PARAMS
    price, breadth = df["price"].to_numpy(), df["breadth"].to_numpy()
    vix, ma200 = df["vix"].to_numpy(), df["ma200"].to_numpy()

    w = int(p["div_window"])
    pp = np.concatenate([np.full(w, np.nan), price[:-w]])
    bp = np.concatenate([np.full(w, np.nan), breadth[:-w]])
    with np.errstate(invalid="ignore"):
        price_rose  = (price - pp) / pp * 100 >= p["price_rise"]
        bearish_div = (price_rose & ((bp - breadth) >= p["breadth_fall"])
                       & (breadth < p["breadth_cap"]))
        vote_gate = (np.isnan(vix) | (vix > p["vix_thresh"])) | (np.isnan(ma200) | (price > ma200))
        washout   = (breadth < p["buy_thresh"]) & vote_gate
    # Trend re-entry: fresh close back above MA200 (gate reduces to "above prior exit").
    cross_up = np.zeros(len(price), dtype=bool)
    cross_up[1:] = (price[1:] > ma200[1:]) & (price[:-1] <= ma200[:-1])

    def chg(col: str) -> np.ndarray:
        a = ind[col].to_numpy()
        prev = np.concatenate([np.full(WINDOW, np.nan), a[:-WINDOW]])
        return a - prev

    macd = ind["macd"].to_numpy()
    hist = ind["hist"].to_numpy()
    # bearish cross: histogram (macd - signal) flips from >=0 to <0
    bear_cross = np.zeros(len(macd), dtype=bool)
    bear_cross[1:] = (hist[1:] < 0) & (hist[:-1] >= 0)

    return {
        "price": price, "dates": df.index, "cross_up": cross_up,
        "washout": washout, "bearish_div": bearish_div, "price_rose": price_rose,
        "hy": ind["hy"].to_numpy(), "hy_ma": ind["hy_ma"].to_numpy(),
        "hy_chg": chg("hy"), "nhnl": ind["nhnl"].to_numpy(),
        "ad_chg": chg("ad_line"), "s5oh_chg": chg("s5oh"), "us10_chg": chg("us10"),
        "rsi": ind["rsi"].to_numpy(), "macd": macd, "hist": hist, "bear_cross": bear_cross,
    }


def run(pre: dict, entry_gate: np.ndarray | None = None,
        extra_exit: np.ndarray | None = None) -> tuple[pd.Series, int]:
    price, dates = pre["price"], pre["dates"]
    washout, bearish_div, cross_up = pre["washout"], pre["bearish_div"], pre["cross_up"]

    cash, shares = INITIAL_CAPITAL, 0.0
    cooldown_until = None
    last_exit_price = None
    n_sells = 0
    values = np.empty(len(price))

    for i in range(len(price)):
        px = price[i]
        if shares > 0:
            if bearish_div[i] or (extra_exit is not None and extra_exit[i]):
                cash += shares * px * (1 - SLIPPAGE) - COMMISSION
                shares = 0.0
                cooldown_until = dates[i] + pd.Timedelta(days=COOLDOWN_DAYS)
                last_exit_price = px
                n_sells += 1
        else:
            cd_ok = cooldown_until is None or dates[i] > cooldown_until
            gate_ok = entry_gate is None or entry_gate[i]
            # Trend re-entry: fresh MA200 recross once price is back above the prior exit.
            trend = bool(cross_up[i]) and (last_exit_price is not None and px > last_exit_price)
            if (washout[i] or trend) and cd_ok and gate_ok:
                cash -= COMMISSION
                shares = cash / (px * (1 + SLIPPAGE))
                cash = 0.0
        values[i] = cash + shares * px * (1 - SLIPPAGE)

    return pd.Series(values, index=dates), n_sells


def build_condition(pre: dict, family: str, x: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Returns (entry_gate, extra_exit) boolean arrays for a family/parameter."""
    nan_true = lambda cond, ref: np.where(np.isnan(ref), True, cond)   # missing data → allow
    nan_false = lambda cond, ref: np.where(np.isnan(ref), False, cond)  # missing data → no signal

    if family == "HY-calm entry":
        cond = pre["hy"] < pre["hy_ma"] + x
        return nan_true(cond, pre["hy_ma"]), None
    if family == "HY-stress exit":
        return None, nan_false(pre["hy_chg"] >= x, pre["hy_chg"])
    if family == "NHNL exit":
        cond = pre["price_rose"] & (pre["nhnl"] < x)
        return None, nan_false(cond, pre["nhnl"])
    if family == "ADline exit":
        cond = pre["price_rose"] & (pre["ad_chg"] < x)
        return None, nan_false(cond, pre["ad_chg"])
    if family == "S5OH-div exit":
        cond = pre["price_rose"] & (pre["s5oh_chg"] <= -x)
        return None, nan_false(cond, pre["s5oh_chg"])
    if family == "10Y-spike exit":
        return None, nan_false(pre["us10_chg"] >= x, pre["us10_chg"])
    if family == "RSI entry":
        return nan_true(pre["rsi"] < x, pre["rsi"]), None
    if family == "RSI exit":
        return None, nan_false(pre["rsi"] >= x, pre["rsi"])
    if family == "MACD entry":
        return nan_true(pre["hist"] > x, pre["hist"]), None
    if family == "MACD exit":
        cond = pre["bear_cross"] & (pre["macd"] > x)
        return None, nan_false(cond, pre["macd"])
    raise ValueError(family)


FAMILIES = {
    "A baseline":     [0.0],
    "HY-calm entry":  [0.0, 0.5, 1.0],
    "HY-stress exit": [0.75, 1.0, 1.5],
    "NHNL exit":      [0.0, -20.0, -50.0],
    "ADline exit":    [0.0, -2000.0, -5000.0],
    "S5OH-div exit":  [15.0, 20.0, 25.0],
    "10Y-spike exit": [0.5, 0.75, 1.0],
    "RSI entry":      [35.0, 40.0, 50.0],
    "RSI exit":       [70.0, 75.0, 80.0],
    "MACD entry":     [-0.25, 0.0, 0.25],
    "MACD exit":      [0.0, 0.5, 1.0],
}


def evaluate(values: pd.Series, price: pd.Series) -> dict:
    dr = values.pct_change().dropna()
    ex = (dr - price.pct_change()).dropna()
    t, pval = stats.ttest_1samp(ex, 0) if len(ex) else (np.nan, np.nan)
    mdd = ((values - values.cummax()) / values.cummax()).min()
    return {"sharpe": sharpe(dr.to_numpy()), "mult": values.iloc[-1] / values.iloc[0],
            "mdd": mdd, "ex_ann": ex.mean() * 252, "p": pval}


def main() -> None:
    df = load_raw()
    ind = load_indicators(df.index)
    print(f"Data: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")

    halves = {"2002–2013": df.index < SPLIT_DATE, "2014–2026": df.index >= SPLIT_DATE}
    pres = {k: precompute(df[m], ind[m]) for k, m in halves.items()}
    dfs  = {k: df[m] for k, m in halves.items()}

    for is_name, oos_name in [("2002–2013", "2014–2026"), ("2014–2026", "2002–2013")]:
        oos_df = dfs[oos_name]
        bh = evaluate(INITIAL_CAPITAL * oos_df["price"] / oos_df["price"].iloc[0], oos_df["price"])
        print(f"\n=== select on {is_name}, report on {oos_name} "
              f"(buy&hold OOS: Sharpe {bh['sharpe']:.2f}, {bh['mult']:.1f}x, DD {bh['mdd']:.0%}) ===")
        print(f"  {'family':<16}{'chosen':>8}{'IS SR':>7}{'OOS SR':>7}{'OOS mult':>9}"
              f"{'OOS DD':>8}{'excess/yr':>10}{'p':>7}{'sells':>6}")

        for fam, grid in FAMILIES.items():
            best_x, best_sr = None, -np.inf
            for x in grid:
                gate, exit_ = (None, None) if fam == "A baseline" else build_condition(pres[is_name], fam, x)
                vals, _ = run(pres[is_name], gate, exit_)
                sr = sharpe(vals.pct_change().dropna().to_numpy())
                if sr > best_sr:
                    best_x, best_sr = x, sr
            gate, exit_ = (None, None) if fam == "A baseline" else build_condition(pres[oos_name], fam, best_x)
            vals, n_sells = run(pres[oos_name], gate, exit_)
            m = evaluate(vals, oos_df["price"])
            print(f"  {fam:<16}{best_x:>8.2f}{best_sr:>7.2f}{m['sharpe']:>7.2f}{m['mult']:>8.1f}x"
                  f"{m['mdd']:>8.0%}{m['ex_ann']:>+10.1%}{m['p']:>7.2f}{n_sells:>6}")


if __name__ == "__main__":
    main()
