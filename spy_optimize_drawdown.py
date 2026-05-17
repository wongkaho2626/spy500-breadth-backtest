"""
Optimize CAPE-based drawdown minimization for the SPY breadth strategy.

Two CAPE levers tested:
  1. CAPE drop stop: exit when CAPE falls X% from entry-level (crash confirmed)
  2. CAPE-tiered buy threshold: require lower breadth when CAPE is elevated
"""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR     = Path(__file__).parent
SPY_FILE     = DATA_DIR / "SPY.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"
B50_FILE     = DATA_DIR / "S&P 500 Stocks Above 50-Day Average Historical Data.csv"
CAPE_FILE    = DATA_DIR / "ShillerPE.csv"

INITIAL_CAPITAL  = 10_000.0
COMMISSION       = 1.0
SLIPPAGE         = 0.0005
DIV_WINDOW       = 100
DIV_PRICE_RISE   = 1.0
DIV_BREADTH_FALL = 20.0
CAP_BASE         = 55.0
CAPE_HIGH        = 32.0
CAP_HIGH_PE      = 52.0
CAPE_VERY_HIGH   = 38.0
CAP_VERY_HIGH_PE = 35.0
BUY_50_THRESHOLD = 25.0


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    spy = pd.read_csv(SPY_FILE)
    spy["Date"] = pd.to_datetime(spy["date"], format="%Y-%m-%d")
    spy.set_index("Date", inplace=True)
    spy = spy.rename(columns={"close": "spy_price"})

    b200 = pd.read_csv(BREADTH_FILE)
    b50  = pd.read_csv(B50_FILE)
    for df in (b200, b50):
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["Price"] = _parse_price(df["Price"])

    cape = pd.read_csv(CAPE_FILE)
    cape["Date"] = pd.to_datetime(cape["date"], format="%Y-%m-%d")
    cape.set_index("Date", inplace=True)
    cape = cape.rename(columns={"close": "cape"})

    merged = spy[["spy_price"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="inner"
    )
    merged = merged.join(b50[["Price"]].rename(columns={"Price": "b50"}), how="inner")
    merged = merged.join(cape[["cape"]], how="left")
    merged["cape"] = merged["cape"].ffill()
    merged.sort_index(inplace=True)

    div_cap = merged["cape"].apply(
        lambda c: CAP_VERY_HIGH_PE if c >= CAPE_VERY_HIGH
        else (CAP_HIGH_PE if c >= CAPE_HIGH else CAP_BASE)
    )
    pp = merged["spy_price"].shift(DIV_WINDOW)
    bp = merged["breadth"].shift(DIV_WINDOW)
    merged["bearish_div"] = (
        ((merged["spy_price"] - pp) / pp * 100 >= DIV_PRICE_RISE) &
        ((bp - merged["breadth"]) >= DIV_BREADTH_FALL) &
        (merged["breadth"] < div_cap)
    )
    return merged


def run_strategy(df: pd.DataFrame, params: dict) -> tuple[pd.Series, list[dict]]:
    buy_thresh     = params["buy_thresh"]
    buy_thresh_hi  = params["buy_thresh_hi"]
    cape_buy_hi    = params["cape_buy_hi"]
    cape_drop_stop = params["cape_drop_stop"]
    stop_cooldown  = params["stop_cooldown"]

    position       = "OUT"
    eff_entry      = raw_entry = 0.0
    entry_date     = None
    cape_entry     = 0.0
    portfolio      = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}
    stop_exit_date = None

    for date, row in df.iterrows():
        price       = row["spy_price"]
        breadth     = row["breadth"]
        b50         = row["b50"]
        cape        = row["cape"]
        bearish_div = bool(row["bearish_div"])

        active_buy  = buy_thresh_hi if cape > cape_buy_hi else buy_thresh
        in_cooldown = (stop_exit_date is not None and
                       (date - stop_exit_date).days < stop_cooldown)

        if position == "OUT" and not in_cooldown and breadth < active_buy and b50 < BUY_50_THRESHOLD:
            portfolio -= COMMISSION
            eff_entry  = price * (1 + SLIPPAGE)
            raw_entry  = price
            entry_date = date
            cape_entry = cape
            position   = "IN"

        elif position == "IN":
            cape_crashed = (cape_drop_stop > 0 and
                            cape < cape_entry * (1 - cape_drop_stop))

            if bearish_div or cape_crashed:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                reason = "cape-drop-stop" if (cape_crashed and not bearish_div) else "bearish-divergence"
                trades.append({
                    "entry_date":  entry_date,
                    "exit_date":   date,
                    "return_pct":  gross_ret * 100,
                    "accumulated": portfolio,
                    "sell_reason": reason,
                })
                position = "OUT"
                if cape_crashed and not bearish_div:
                    stop_exit_date = date

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    return pd.Series(values, name="strategy"), trades


def score(values: pd.Series, trades: list[dict]) -> dict:
    if len(values) < 2 or not trades:
        return {"sharpe": -99, "cagr": -99, "max_dd": -99, "n_trades": 0}
    daily_ret = values.pct_change().dropna()
    years     = (values.index[-1] - values.index[0]).days / 365.25
    cagr      = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    roll_max  = values.cummax()
    max_dd    = ((values - roll_max) / roll_max).min()
    std       = daily_ret.std()
    sharpe    = (daily_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    wins      = sum(1 for t in trades if t["return_pct"] > 0)
    return {
        "sharpe":   sharpe,
        "cagr":     cagr,
        "max_dd":   max_dd,
        "n_trades": len(trades),
        "win_rate": wins / len(trades) if trades else 0.0,
    }


def main() -> None:
    print("Loading data...")
    df = load_data()

    grid = {
        "buy_thresh":     [18.0],
        "buy_thresh_hi":  [8.0, 10.0, 12.0, 14.0],
        "cape_buy_hi":    [25.0, 28.0, 30.0, 32.0],
        "cape_drop_stop": [0.0, 0.10, 0.15, 0.18, 0.20, 0.22],
        "stop_cooldown":  [0, 30, 60, 90],
    }

    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"Testing {len(combos)} combinations...")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        values, trades = run_strategy(df, params)
        s = score(values, trades)
        results.append({**params, **s})

    results.sort(key=lambda r: (-r["max_dd"], -r["sharpe"]))

    print(f"\n{'='*125}")
    print("Top 25 by Max Drawdown (least negative), then Sharpe")
    print(f"{'='*125}")
    hdr = (f"{'MaxDD':>7}  {'Sharpe':>7}  {'CAGR':>7}  {'#T':>3}  {'WR':>6}"
           f"  {'buyThr':>6}  {'buyHi':>6}  {'capeHi':>7}  {'dropSt':>7}  {'cool':>5}")
    print(hdr)
    print("-" * 125)
    for r in results[:25]:
        print(
            f"{r['max_dd']:>7.2%}  {r['sharpe']:>7.3f}  {r['cagr']:>7.2%}"
            f"  {r['n_trades']:>3}  {r['win_rate']:>6.0%}"
            f"  {r['buy_thresh']:>6.1f}  {r['buy_thresh_hi']:>6.1f}"
            f"  {r['cape_buy_hi']:>7.1f}  {r['cape_drop_stop']:>7.0%}"
            f"  {r['stop_cooldown']:>5}"
        )
    print(f"{'='*125}")

    best = results[0]
    print(f"\nBest params (min drawdown):")
    for k in keys:
        print(f"  {k} = {best[k]}")
    print(f"  → MaxDD={best['max_dd']:.2%}  Sharpe={best['sharpe']:.3f}  CAGR={best['cagr']:.2%}")


if __name__ == "__main__":
    main()
