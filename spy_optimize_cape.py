"""Grid-search optimizer adding ShillerPE (CAPE) tiers to the SPY breadth strategy."""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR     = Path(__file__).parent
SPY_FILE     = DATA_DIR / "SPY.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"
B50_FILE     = DATA_DIR / "S&P 500 Stocks Above 50-Day Average Historical Data.csv"
CAPE_FILE    = DATA_DIR / "ShillerPE.csv"

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005


def _parse_price(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    spy_raw = pd.read_csv(SPY_FILE)
    spy_raw["Date"] = pd.to_datetime(spy_raw["date"], format="%Y-%m-%d")
    spy_raw.set_index("Date", inplace=True)
    spy_raw = spy_raw.rename(columns={"close": "spy_price"})

    breadth_raw = pd.read_csv(BREADTH_FILE)
    b50_raw     = pd.read_csv(B50_FILE)
    for df in (breadth_raw, b50_raw):
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["Price"] = _parse_price(df["Price"])

    cape_raw = pd.read_csv(CAPE_FILE)
    cape_raw["Date"] = pd.to_datetime(cape_raw["date"], format="%Y-%m-%d")
    cape_raw.set_index("Date", inplace=True)
    cape_raw = cape_raw.rename(columns={"close": "cape"})

    merged = spy_raw[["spy_price"]].join(
        breadth_raw[["Price"]].rename(columns={"Price": "breadth"}), how="inner"
    )
    merged = merged.join(b50_raw[["Price"]].rename(columns={"Price": "b50"}), how="inner")
    merged = merged.join(cape_raw[["cape"]], how="left")
    merged["cape"] = merged["cape"].ffill()
    merged.sort_index(inplace=True)
    return merged


def run_strategy(df: pd.DataFrame, params: dict) -> tuple[pd.Series, list[dict]]:
    buy_thresh      = params["buy_thresh"]
    buy_50_thresh   = params["buy_50_thresh"]
    div_window      = params["div_window"]
    div_price_rise  = params["div_price_rise"]
    div_bfall       = params["div_bfall"]
    cap_base        = params["cap_base"]
    cape_hi         = params["cape_hi"]
    cap_hi_pe       = params["cap_hi_pe"]
    cape_very_hi    = params["cape_very_hi"]
    cap_very_hi_pe  = params["cap_very_hi_pe"]

    price_past   = df["spy_price"].shift(div_window)
    breadth_past = df["breadth"].shift(div_window)
    price_rose   = (df["spy_price"] - price_past) / price_past * 100 >= div_price_rise
    breadth_fell = (breadth_past - df["breadth"]) >= div_bfall

    position  = "OUT"
    eff_entry = raw_entry = 0.0
    entry_date = None
    portfolio  = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        price   = row["spy_price"]
        breadth = row["breadth"]
        b50     = row["b50"]
        cape    = row["cape"]

        if cape >= cape_very_hi:
            div_cap = cap_very_hi_pe
        elif cape >= cape_hi:
            div_cap = cap_hi_pe
        else:
            div_cap = cap_base

        bearish_div = bool(price_rose[date] and breadth_fell[date] and breadth < div_cap)

        if position == "OUT" and breadth < buy_thresh and b50 < buy_50_thresh:
            portfolio -= COMMISSION
            eff_entry  = price * (1 + SLIPPAGE)
            raw_entry  = price
            entry_date = date
            position   = "IN"
        elif position == "IN" and bearish_div:
            eff_exit  = price * (1 - SLIPPAGE)
            gross_ret = (eff_exit - eff_entry) / eff_entry
            portfolio *= (1 + gross_ret)
            portfolio -= COMMISSION
            trades.append({
                "entry_date":  entry_date,
                "exit_date":   date,
                "return_pct":  gross_ret * 100,
                "accumulated": portfolio,
            })
            position = "OUT"

        if position == "IN":
            values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry)
        else:
            values[date] = portfolio

    return pd.Series(values, name="strategy"), trades


def score(values: pd.Series, trades: list[dict]) -> dict:
    if len(values) < 2:
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

    fixed = {
        "buy_thresh":     18.0,
        "buy_50_thresh":  25.0,
        "div_window":     100,
        "div_price_rise": 1.0,
        "div_bfall":      20.0,
    }

    grid = {
        "cap_base":       [55.0, 60.0],
        "cape_hi":        [28.0, 30.0, 32.0],
        "cap_hi_pe":      [42.0, 45.0, 48.0, 52.0],
        "cape_very_hi":   [34.0, 36.0, 38.0],
        "cap_very_hi_pe": [35.0, 38.0, 42.0],
    }

    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"Testing {len(combos)} combinations...")

    results = []
    for combo in combos:
        params = {**fixed, **dict(zip(keys, combo))}
        if params["cap_very_hi_pe"] >= params["cap_hi_pe"]:
            continue
        if params["cape_very_hi"] <= params["cape_hi"]:
            continue

        values, trades = run_strategy(df, params)
        s = score(values, trades)
        results.append({**params, **s})

    results.sort(key=lambda r: r["sharpe"], reverse=True)

    print(f"\n{'='*115}")
    print("Top 20 by Sharpe Ratio")
    print(f"{'='*115}")
    hdr = (f"{'Sharpe':>7}  {'CAGR':>7}  {'MaxDD':>7}  {'#T':>3}  {'WR':>6}"
           f"  {'capBase':>7}  {'capeHi':>6}  {'capHi':>6}  {'capeVH':>7}  {'capVH':>6}")
    print(hdr)
    print("-" * 115)
    for r in results[:20]:
        print(
            f"{r['sharpe']:>7.3f}  {r['cagr']:>7.2%}  {r['max_dd']:>7.2%}  {r['n_trades']:>3}  {r['win_rate']:>6.0%}"
            f"  {r['cap_base']:>7.1f}  {r['cape_hi']:>6.1f}  {r['cap_hi_pe']:>6.1f}"
            f"  {r['cape_very_hi']:>7.1f}  {r['cap_very_hi_pe']:>6.1f}"
        )
    print(f"{'='*115}")

    best = results[0]
    print(f"\nBest params:")
    for k in ["cap_base", "cape_hi", "cap_hi_pe", "cape_very_hi", "cap_very_hi_pe"]:
        print(f"  {k} = {best[k]}")


if __name__ == "__main__":
    main()
