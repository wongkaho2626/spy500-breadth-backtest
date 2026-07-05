"""
QQQ / NASDAQ 100 Breadth Strategy — composite bearish-signal exit

Codes 8 classic bearish technical signals on the NASDAQ 100 (OHLC + volume)
and exits when ≥ N distinct signals fire within a 10-day window:

  1. macd_cross : MACD(12,26,9) histogram flips negative
  2. rsi_cross  : RSI(14) crosses back below 70 from overbought
  3. rsi_div    : price rose ≥3%/60d while RSI fell (momentum divergence)
  4. rs_div     : price up over 20d while RS line (NDX/SPX) down — leadership loss
  5. ext10      : price extended ≥5% above its 10-day MA
  6. climax     : price ≥50% above MA200 AND 10-day gain ≥6% (climax-top proxy)
  7. hanging_man: small body, long lower shadow after a 10-day advance
  8. outside_vol: bearish outside bar on volume ≥150% of its 50-day average

Not codable objectively (skipped): Fib retracement, trendline resistance,
support/resistance levels — they depend on subjective anchor choices.

Tested walk-forward (N selected by Sharpe on one half, reported on the other),
both as an ADDITIONAL exit next to the baseline bearish-divergence exit and as
a full REPLACEMENT for it.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from qqq_validation import (
    BASE_PARAMS, COMMISSION, INITIAL_CAPITAL, SLIPPAGE, COOLDOWN_DAYS,
    load_raw, sharpe,
)

DATA_DIR   = Path(__file__).parent
SPLIT_DATE = pd.Timestamp("2014-01-01")
VOTE_WINDOW = 10          # days within which distinct signals are counted
N_GRID      = [2, 3, 4, 5]


def _parse_num(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(",", "").str.strip()
    mult = s.str[-1].map({"K": 1e3, "M": 1e6, "B": 1e9})
    num = pd.to_numeric(s.str.rstrip("KMB"), errors="coerce")
    return num * mult.fillna(1.0)


def load_ohlcv() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "NASDAQ100.csv", encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()
    out = pd.DataFrame(index=df.index)
    for col, src in [("close", "Price"), ("open", "Open"), ("high", "High"), ("low", "Low")]:
        out[col] = _parse_num(df[src])
    out["volume"] = _parse_num(df["Vol."])
    return out


def bearish_signals(px: pd.DataFrame, spx: pd.Series) -> pd.DataFrame:
    c, o, h, l, v = px["close"], px["open"], px["high"], px["low"], px["volume"]

    # 1. MACD bearish cross
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    macd_cross = (hist < 0) & (hist.shift(1) >= 0)

    # 2. RSI crosses back below 70
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    rsi_cross = (rsi < 70) & (rsi.shift(1) >= 70)

    # 3. RSI divergence: price up ≥3% over 60d, RSI down over 60d
    rsi_div = (c.pct_change(60) * 100 >= 3.0) & (rsi.diff(60) < 0) & (rsi > 50)

    # 4. RS-line divergence: price up over 20d while NDX/SPX ratio down
    rs = c / spx.reindex(c.index).ffill()
    rs_div = (c.pct_change(20) > 0) & (rs.pct_change(20) < 0)

    # 5. Extended ≥5% above 10-day MA
    ext10 = c / c.rolling(10).mean() - 1 >= 0.05

    # 6. Climax-top proxy: far above MA200 and accelerating
    climax = (c / c.rolling(200).mean() - 1 >= 0.50) & (c.pct_change(10) >= 0.06)

    # 7. Hanging man after a 10-day advance
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    lower_shadow = np.minimum(c, o) - l
    hanging_man = ((body <= 0.3 * rng) & (lower_shadow >= 0.6 * rng)
                   & (c.pct_change(10) >= 0.04))

    # 8. Bearish outside bar on ≥150% average volume
    outside_vol = ((h > h.shift(1)) & (l < l.shift(1)) & (c < o)
                   & (v >= 1.5 * v.rolling(50).mean()))

    sig = pd.DataFrame({
        "macd_cross": macd_cross, "rsi_cross": rsi_cross, "rsi_div": rsi_div,
        "rs_div": rs_div, "ext10": ext10, "climax": climax,
        "hanging_man": hanging_man, "outside_vol": outside_vol,
    }).fillna(False)
    return sig


def composite_score(sig: pd.DataFrame) -> pd.Series:
    """Number of DISTINCT signal types that fired within the last VOTE_WINDOW days."""
    fired = sig.rolling(VOTE_WINDOW, min_periods=1).max()
    return fired.sum(axis=1)


def precompute(df: pd.DataFrame) -> dict:
    p = BASE_PARAMS
    price, breadth = df["price"].to_numpy(), df["breadth"].to_numpy()
    vix, ma200 = df["vix"].to_numpy(), df["ma200"].to_numpy()
    w = int(p["div_window"])
    pp = np.concatenate([np.full(w, np.nan), price[:-w]])
    bp = np.concatenate([np.full(w, np.nan), breadth[:-w]])
    with np.errstate(invalid="ignore"):
        bearish_div = (((price - pp) / pp * 100 >= p["price_rise"])
                       & ((bp - breadth) >= p["breadth_fall"]) & (breadth < p["breadth_cap"]))
        vote_gate = (np.isnan(vix) | (vix > p["vix_thresh"])) | (np.isnan(ma200) | (price > ma200))
        washout = (breadth < p["buy_thresh"]) & vote_gate
    # Trend re-entry: fresh close back above MA200 (gate reduces to "above prior exit").
    cross_up = np.zeros(len(price), dtype=bool)
    cross_up[1:] = (price[1:] > ma200[1:]) & (price[:-1] <= ma200[:-1])
    return {"price": price, "dates": df.index, "washout": washout,
            "bearish_div": bearish_div, "cross_up": cross_up}


def run(pre: dict, exit_signal: np.ndarray) -> tuple[pd.Series, int]:
    price, dates, cross_up = pre["price"], pre["dates"], pre["cross_up"]
    cash, shares = INITIAL_CAPITAL, 0.0
    cooldown_until, n_sells = None, 0
    last_exit_price = None
    values = np.empty(len(price))
    for i in range(len(price)):
        px = price[i]
        if shares > 0:
            if exit_signal[i]:
                cash += shares * px * (1 - SLIPPAGE) - COMMISSION
                shares = 0.0
                cooldown_until = dates[i] + pd.Timedelta(days=COOLDOWN_DAYS)
                last_exit_price = px
                n_sells += 1
        else:
            cd_ok = cooldown_until is None or dates[i] > cooldown_until
            trend = bool(cross_up[i]) and (last_exit_price is not None and px > last_exit_price)
            if (pre["washout"][i] or trend) and cd_ok:
                cash -= COMMISSION
                shares = cash / (px * (1 + SLIPPAGE))
                cash = 0.0
        values[i] = cash + shares * px * (1 - SLIPPAGE)
    return pd.Series(values, index=dates), n_sells


def evaluate(values: pd.Series, price: pd.Series) -> dict:
    dr = values.pct_change().dropna()
    ex = (dr - price.pct_change()).dropna()
    t, pval = stats.ttest_1samp(ex, 0) if len(ex) else (np.nan, np.nan)
    mdd = ((values - values.cummax()) / values.cummax()).min()
    return {"sharpe": sharpe(dr.to_numpy()), "mult": values.iloc[-1] / values.iloc[0],
            "mdd": mdd, "ex_ann": ex.mean() * 252, "p": pval}


def main() -> None:
    df = load_raw()
    px = load_ohlcv()
    spx_df = pd.read_csv(DATA_DIR / "SPX.csv", encoding="utf-8-sig")
    spx_df.columns = [c.strip().strip('"') for c in spx_df.columns]
    spx_df["Date"] = pd.to_datetime(spx_df["Date"], format="%m/%d/%Y")
    spx = spx_df.set_index("Date")["Price"].astype(str).str.replace(",", "").astype(float).sort_index()

    sig = bearish_signals(px, spx)
    score = composite_score(sig).reindex(df.index).ffill().fillna(0).to_numpy()

    print(f"Data: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")
    freq = pd.Series(score, index=df.index)
    print("Composite score distribution (share of days at each level):")
    print("  " + "  ".join(f"≥{n}: {(freq >= n).mean():.0%}" for n in N_GRID))

    halves = {"2002–2013": df.index < SPLIT_DATE, "2014–2026": df.index >= SPLIT_DATE}
    pres = {k: precompute(df[m]) for k, m in halves.items()}
    scores = {k: score[m] for k, m in halves.items()}
    dfs = {k: df[m] for k, m in halves.items()}

    for is_name, oos_name in [("2002–2013", "2014–2026"), ("2014–2026", "2002–2013")]:
        oos_df = dfs[oos_name]
        bh = evaluate(INITIAL_CAPITAL * oos_df["price"] / oos_df["price"].iloc[0], oos_df["price"])
        print(f"\n=== select on {is_name}, report on {oos_name} "
              f"(buy&hold OOS: Sharpe {bh['sharpe']:.2f}, {bh['mult']:.1f}x, DD {bh['mdd']:.0%}) ===")
        print(f"  {'variant':<22}{'N':>3}{'IS SR':>7}{'OOS SR':>7}{'OOS mult':>9}"
              f"{'OOS DD':>8}{'excess/yr':>10}{'p':>7}{'sells':>6}")

        for mode in ["baseline", "composite ADD", "composite REPLACE"]:
            best_n, best_sr = None, -np.inf
            for n in (N_GRID if mode != "baseline" else [0]):
                pre_is = pres[is_name]
                if mode == "baseline":
                    exit_sig = pre_is["bearish_div"]
                elif mode == "composite ADD":
                    exit_sig = pre_is["bearish_div"] | (scores[is_name] >= n)
                else:
                    exit_sig = scores[is_name] >= n
                vals, _ = run(pre_is, exit_sig)
                sr = sharpe(vals.pct_change().dropna().to_numpy())
                if sr > best_sr:
                    best_n, best_sr = n, sr
            pre_oos = pres[oos_name]
            if mode == "baseline":
                exit_sig = pre_oos["bearish_div"]
            elif mode == "composite ADD":
                exit_sig = pre_oos["bearish_div"] | (scores[oos_name] >= best_n)
            else:
                exit_sig = scores[oos_name] >= best_n
            vals, n_sells = run(pre_oos, exit_sig)
            m = evaluate(vals, oos_df["price"])
            print(f"  {mode:<22}{best_n:>3}{best_sr:>7.2f}{m['sharpe']:>7.2f}{m['mult']:>8.1f}x"
                  f"{m['mdd']:>8.0%}{m['ex_ann']:>+10.1%}{m['p']:>7.2f}{n_sells:>6}")


if __name__ == "__main__":
    main()
