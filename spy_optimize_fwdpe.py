"""
Optimize forward PE overlay on top of the existing CAPE-tiered sell caps.
Tests: when CAPE is in a high tier AND fwd PE > threshold, use an even
tighter breadth cap to exit sooner in dual-expensive markets.
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
FWDPE_FILE   = DATA_DIR / "S&P500ForwardPE.csv"

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
BUY_THRESHOLD    = 18.0
BUY_50_THRESHOLD = 25.0
BUY_THRESH_HI    = 12.0
CAPE_BUY_HIGH    = 28.0
CAPE_DROP_STOP   = 0.15
STOP_COOLDOWN    = 90


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_data() -> pd.DataFrame:
    spy = pd.read_csv(SPY_FILE)
    spy["Date"] = pd.to_datetime(spy["date"]); spy.set_index("Date", inplace=True)
    spy = spy.rename(columns={"close": "spy_price"})

    b200 = pd.read_csv(BREADTH_FILE)
    b200["Date"] = pd.to_datetime(b200["Date"], format="%m/%d/%Y")
    b200.set_index("Date", inplace=True); b200["Price"] = _parse_price(b200["Price"])

    b50 = pd.read_csv(B50_FILE)
    b50["Date"] = pd.to_datetime(b50["Date"], format="%m/%d/%Y")
    b50.set_index("Date", inplace=True); b50["Price"] = _parse_price(b50["Price"])

    cape = pd.read_csv(CAPE_FILE)
    cape["Date"] = pd.to_datetime(cape["date"]); cape.set_index("Date", inplace=True)
    cape = cape.rename(columns={"close": "cape"})

    fwdpe = pd.read_csv(FWDPE_FILE)
    fwdpe["Date"] = pd.to_datetime(fwdpe["date"]); fwdpe.set_index("Date", inplace=True)

    merged = spy[["spy_price"]].join(
        b200[["Price"]].rename(columns={"Price": "breadth"}), how="inner"
    )
    merged = merged.join(b50[["Price"]].rename(columns={"Price": "b50"}), how="inner")
    merged = merged.join(cape[["cape"]], how="left")
    merged = merged.join(fwdpe[["forward_pe"]], how="left")
    merged["cape"]       = merged["cape"].ffill()
    merged["forward_pe"] = merged["forward_pe"].ffill()
    merged.sort_index(inplace=True)

    # precompute shifted series for speed
    merged["_pp"] = merged["spy_price"].shift(DIV_WINDOW)
    merged["_bp"] = merged["breadth"].shift(DIV_WINDOW)
    return merged


def run_strategy(df: pd.DataFrame, params: dict) -> tuple[pd.Series, list[dict]]:
    fpe_hi_thresh  = params["fpe_hi_thresh"]
    cap_dual       = params["cap_dual"]
    fpe_mid_thresh = params["fpe_mid_thresh"]
    cap_mid_dual   = params["cap_mid_dual"]

    position       = "OUT"
    eff_entry      = raw_entry = 0.0
    entry_date     = None
    cape_entry     = 0.0
    portfolio      = INITIAL_CAPITAL
    trades: list[dict] = []
    values: dict = {}
    stop_exit_date = None

    for date, row in df.iterrows():
        price   = row["spy_price"]
        breadth = row["breadth"]
        b50     = row["b50"]
        cape    = row["cape"]
        fpe     = row["forward_pe"]
        pp, bp  = row["_pp"], row["_bp"]

        if cape >= CAPE_VERY_HIGH and not pd.isna(fpe) and fpe > fpe_hi_thresh:
            div_cap = cap_dual
        elif cape >= CAPE_VERY_HIGH:
            div_cap = CAP_VERY_HIGH_PE
        elif cape >= CAPE_HIGH and not pd.isna(fpe) and fpe > fpe_mid_thresh:
            div_cap = cap_mid_dual
        elif cape >= CAPE_HIGH:
            div_cap = CAP_HIGH_PE
        else:
            div_cap = CAP_BASE

        bearish_div = (
            not pd.isna(pp) and not pd.isna(bp) and
            (price - pp) / pp * 100 >= DIV_PRICE_RISE and
            (bp - breadth) >= DIV_BREADTH_FALL and
            breadth < div_cap
        )

        active_buy  = BUY_THRESH_HI if cape > CAPE_BUY_HIGH else BUY_THRESHOLD
        in_cooldown = (stop_exit_date is not None and
                       (date - stop_exit_date).days < STOP_COOLDOWN)

        if position == "OUT" and not in_cooldown and breadth < active_buy and b50 < BUY_50_THRESHOLD:
            portfolio -= COMMISSION
            eff_entry  = price * (1 + SLIPPAGE)
            raw_entry  = price
            entry_date = date
            cape_entry = cape
            position   = "IN"

        elif position == "IN":
            cape_crashed = cape < cape_entry * (1 - CAPE_DROP_STOP)
            if bearish_div or cape_crashed:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                reason = "cape-drop-stop" if (cape_crashed and not bearish_div) else "bearish-div"
                trades.append({"entry_date": entry_date, "exit_date": date,
                                "return_pct": gross_ret * 100, "accumulated": portfolio,
                                "sell_reason": reason})
                position = "OUT"
                if cape_crashed and not bearish_div:
                    stop_exit_date = date

        values[date] = (portfolio * (price * (1 - SLIPPAGE) / eff_entry)
                        if position == "IN" else portfolio)

    return pd.Series(values, name="strategy"), trades


def score(values: pd.Series, trades: list[dict]) -> dict:
    if len(values) < 2 or not trades:
        return {"sharpe": -99, "cagr": -99, "max_dd": -99, "n_trades": 0}
    dr    = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr  = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    mdd   = ((values - values.cummax()) / values.cummax()).min()
    std   = dr.std()
    sh    = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    wins  = sum(1 for t in trades if t["return_pct"] > 0)
    return {"sharpe": sh, "cagr": cagr, "max_dd": mdd,
            "n_trades": len(trades),
            "win_rate": wins / len(trades) if trades else 0.0}


def main() -> None:
    print("Loading data...")
    df = load_data()

    grid = {
        "fpe_hi_thresh":  [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
        "cap_dual":       [22.0, 25.0, 28.0, 30.0],
        "fpe_mid_thresh": [20.0, 22.0, 24.0, 26.0],
        "cap_mid_dual":   [38.0, 42.0, 46.0],
    }
    keys   = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    print(f"Testing {len(combos)} combinations...")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        if params["cap_dual"] >= CAP_VERY_HIGH_PE:
            continue
        if params["cap_mid_dual"] >= CAP_HIGH_PE:
            continue
        values, trades = run_strategy(df, params)
        s = score(values, trades)
        results.append({**params, **s})

    results.sort(key=lambda r: (round(r["sharpe"], 3), -r["max_dd"]), reverse=True)

    print(f"\n{'='*120}")
    print("Top 20 by Sharpe → Max Drawdown")
    print(f"{'='*120}")
    hdr = (f"{'Sharpe':>7}  {'CAGR':>7}  {'MaxDD':>7}  {'#T':>3}  {'WR':>5}"
           f"  {'fpeHiThr':>9}  {'capDual':>8}  {'fpeMidThr':>10}  {'capMidDual':>11}")
    print(hdr); print("-" * 120)
    for r in results[:20]:
        print(f"{r['sharpe']:>7.3f}  {r['cagr']:>7.2%}  {r['max_dd']:>7.2%}"
              f"  {r['n_trades']:>3}  {r['win_rate']:>5.0%}"
              f"  {r['fpe_hi_thresh']:>9.1f}  {r['cap_dual']:>8.1f}"
              f"  {r['fpe_mid_thresh']:>10.1f}  {r['cap_mid_dual']:>11.1f}")
    print(f"{'='*120}")

    best = results[0]
    print(f"\nBest params:")
    for k in keys:
        print(f"  {k} = {best[k]}")
    print(f"  Sharpe={best['sharpe']:.3f}  CAGR={best['cagr']:.2%}  MaxDD={best['max_dd']:.2%}")


if __name__ == "__main__":
    main()
