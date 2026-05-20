"""
Grid-search parameter optimization for soxx_backtest.py.
Uses vectorized NumPy for speed (~seconds vs minutes).
Ranks results by Sharpe Ratio (primary) and CAGR (secondary).
"""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR     = Path(__file__).parent
SOXX_FILE    = DATA_DIR / "SOXX.csv"
BREADTH_FILE = DATA_DIR / "S5TH.csv"

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005

PARAM_GRID = {
    "buy_thresh":   [14, 16, 18, 20, 22, 24, 26, 28, 30],
    "div_window":   [40, 50, 60, 70, 80, 100],
    "price_rise":   [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
    "breadth_fall": [8, 10, 12, 15, 18, 20, 25],
    "breadth_cap":  [45, 50, 55, 60, 65, 70],
}


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _load_raw() -> pd.DataFrame:
    soxx = pd.read_csv(SOXX_FILE)
    soxx["Date"] = pd.to_datetime(soxx["Date"], format="%m/%d/%Y")
    soxx.set_index("Date", inplace=True)
    soxx["price"] = _parse_price(soxx["Price"])

    b200 = pd.read_csv(BREADTH_FILE)
    b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
    b200.set_index("Date", inplace=True)
    b200["breadth"] = _parse_price(b200["Price"])

    merged = soxx[["price"]].join(b200[["breadth"]], how="left")
    merged.sort_index(inplace=True)
    return merged[merged["breadth"].notna()].copy()


def _run_vectorized(price: np.ndarray, breadth: np.ndarray,
                    buy_thresh, div_window, price_rise, breadth_fall, breadth_cap) -> dict | None:
    n = len(price)
    # Pre-compute divergence signals
    pp = np.empty(n); pp[:div_window] = np.nan; pp[div_window:] = price[:-div_window]
    bp = np.empty(n); bp[:div_window] = np.nan; bp[div_window:] = breadth[:-div_window]

    with np.errstate(invalid="ignore", divide="ignore"):
        price_rose   = (price - pp) / pp * 100 >= price_rise
        breadth_fell = (bp - breadth) >= breadth_fall
    bearish_div = price_rose & breadth_fell & (breadth < breadth_cap)
    buy_signal  = breadth < buy_thresh

    # State-machine simulation (minimal Python loop — only trade events)
    portfolio  = INITIAL_CAPITAL
    position   = False
    eff_entry  = raw_entry = 0.0
    entry_idx  = 0
    trade_low  = 0.0
    trades     = []
    values     = np.empty(n)

    for i in range(n):
        p = price[i]
        b = breadth[i]

        if not position:
            if buy_signal[i]:
                portfolio -= COMMISSION
                eff_entry = p * (1 + SLIPPAGE)
                raw_entry = p
                entry_idx = i
                trade_low = p
                position  = True
        else:
            if p < trade_low:
                trade_low = p
            if bearish_div[i]:
                eff_exit  = p * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                trades.append((entry_idx, i, raw_entry, p, gross_ret))
                position = False

        values[i] = portfolio * (p * (1 - SLIPPAGE) / eff_entry) if position else portfolio

    if len(trades) < 2:
        return None

    dr    = np.diff(values) / values[:-1]
    years = (n - 1) / 252
    tr    = values[-1] / values[0] - 1
    cagr  = (values[-1] / values[0]) ** (1 / years) - 1
    cum   = np.maximum.accumulate(values)
    mdd   = np.min((values - cum) / cum)
    std   = dr.std()
    sh    = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    nt   = len(trades)
    wins = sum(1 for *_, gr in trades if gr > 0)
    in_d = sum(e - s for s, e, *_ in trades)
    tot  = n - 1

    return {
        "buy_thresh":   buy_thresh,
        "div_window":   div_window,
        "price_rise":   price_rise,
        "breadth_fall": breadth_fall,
        "breadth_cap":  breadth_cap,
        "cagr":         round(cagr * 100, 2),
        "total_ret":    round(tr * 100, 1),
        "mdd":          round(mdd * 100, 1),
        "sharpe":       round(sh, 3),
        "trades":       nt,
        "win_rate":     round(wins / nt * 100, 1),
        "time_in_mkt":  round(in_d / tot * 100, 1),
        "final_value":  round(values[-1], 0),
    }


def main() -> None:
    print("Loading data...")
    df     = _load_raw()
    price  = df["price"].to_numpy(dtype=float)
    breadth = df["breadth"].to_numpy(dtype=float)

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)
    print(f"Running {total:,} combinations...")

    results = []
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        r = _run_vectorized(price, breadth, **params)
        if r:
            results.append(r)
        if i % 5000 == 0:
            print(f"  {i:,}/{total:,}...")

    if not results:
        print("No valid results.")
        return

    res_df = pd.DataFrame(results)
    res_df.sort_values(["sharpe", "cagr"], ascending=False, inplace=True)
    res_df.reset_index(drop=True, inplace=True)

    print(f"\nTop 20 by Sharpe Ratio (from {len(res_df):,} valid combos):\n")
    top = res_df.head(20)
    hdr = (f"{'#':>3}  {'Buy':>5}  {'Win':>4}  {'PRise':>6}  {'BFall':>6}  {'BCap':>5}"
           f"  {'CAGR':>7}  {'TotRet':>8}  {'MDD':>7}  {'Sharpe':>7}"
           f"  {'Trades':>6}  {'WinR':>6}  {'InMkt':>6}  {'FinalVal':>10}")
    print(hdr)
    print("-" * len(hdr))
    for i, row in top.iterrows():
        print(
            f"{i+1:>3}  {row.buy_thresh:>5}  {row.div_window:>4}  {row.price_rise:>6}  "
            f"{row.breadth_fall:>6}  {row.breadth_cap:>5}  "
            f"{row.cagr:>6.1f}%  {row.total_ret:>7.1f}%  {row.mdd:>6.1f}%  "
            f"{row.sharpe:>7.3f}  {row.trades:>6}  {row.win_rate:>5.1f}%  "
            f"{row.time_in_mkt:>5.1f}%  ${row.final_value:>9,.0f}"
        )

    best = res_df.iloc[0]
    print(f"\nBest params (Sharpe {best.sharpe:.3f}, CAGR {best.cagr:.1f}%):")
    print(f"  BUY_B200_THRESH         = {best.buy_thresh}")
    print(f"  DIVERGENCE_WINDOW       = {int(best.div_window)}")
    print(f"  DIVERGENCE_PRICE_RISE   = {best.price_rise}")
    print(f"  DIVERGENCE_BREADTH_FALL = {best.breadth_fall}")
    print(f"  DIVERGENCE_BREADTH_CAP  = {best.breadth_cap}")


if __name__ == "__main__":
    main()
