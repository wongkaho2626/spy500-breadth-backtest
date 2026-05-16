import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

DATA_DIR     = Path(__file__).parent
IGV_FILE     = DATA_DIR / "IGV ETF Stock Price History.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"
B50_FILE     = DATA_DIR / "S&P 500 Stocks Above 50-Day Average Historical Data.csv"

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005

# Grid
BUY_THRESHOLDS   = [14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0]
BUY_50_THRES     = [20.0, 25.0, 30.0, 35.0, 999.0]   # 999 = disabled
DIV_WINDOWS      = [40, 60, 80, 100]
DIV_PRICE_RISES  = [1.0, 2.0, 3.0, 5.0]
DIV_BREADTH_FALL = [10.0, 15.0, 20.0, 25.0]
DIV_BREADTH_CAPS = [50.0, 55.0, 60.0, 65.0, 70.0]


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def load_base() -> pd.DataFrame:
    igv_raw     = pd.read_csv(IGV_FILE)
    breadth_raw = pd.read_csv(BREADTH_FILE)
    b50_raw     = pd.read_csv(B50_FILE)
    for df in (igv_raw, breadth_raw, b50_raw):
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["Price"] = _parse_price(df["Price"])
    merged = igv_raw[["Price"]].join(
        breadth_raw[["Price"]], lsuffix="_igv", rsuffix="_breadth", how="inner"
    )
    merged = merged.rename(columns={"Price_igv": "igv_price", "Price_breadth": "breadth"})
    merged = merged.join(b50_raw[["Price"]].rename(columns={"Price": "b50"}), how="inner")
    merged.sort_index(inplace=True)
    return merged


def run(df: pd.DataFrame, buy_t: float, buy_50: float,
        win: int, rise: float, fall: float, cap: float) -> tuple:
    price_past   = df["igv_price"].shift(win)
    breadth_past = df["breadth"].shift(win)
    bearish_div  = (
        ((df["igv_price"] - price_past) / price_past * 100 >= rise) &
        ((breadth_past - df["breadth"]) >= fall) &
        (df["breadth"] < cap)
    )

    position  = "OUT"
    eff_entry = 0.0
    portfolio = INITIAL_CAPITAL
    values: dict = {}
    trades    = 0

    for date, row in df.iterrows():
        price   = row["igv_price"]
        breadth = row["breadth"]
        b50     = row["b50"]
        bd      = bool(bearish_div.loc[date])

        buy_ok = breadth < buy_t and (buy_50 >= 999 or b50 < buy_50)

        if position == "OUT" and buy_ok:
            portfolio -= COMMISSION
            eff_entry  = price * (1 + SLIPPAGE)
            position   = "IN"
        elif position == "IN" and bd:
            eff_exit  = price * (1 - SLIPPAGE)
            portfolio *= (1 + (eff_exit - eff_entry) / eff_entry)
            portfolio -= COMMISSION
            position   = "OUT"
            trades    += 1

        values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry) if position == "IN" else portfolio

    s      = pd.Series(values)
    dr     = s.pct_change().dropna()
    total  = (s.iloc[-1] / s.iloc[0]) - 1
    years  = (s.index[-1] - s.index[0]).days / 365.25
    cagr   = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    max_dd = ((s - s.cummax()) / s.cummax()).min()
    std    = dr.std()
    sharpe = dr.mean() / std * np.sqrt(252) if std > 0 else 0.0
    return total * 100, cagr * 100, sharpe, max_dd * 100, trades, s.iloc[-1]


def main() -> None:
    print("Loading data...")
    df = load_base()

    bh_ret = (df["igv_price"].iloc[-1] / df["igv_price"].iloc[0] - 1) * 100
    combos = (len(BUY_THRESHOLDS) * len(BUY_50_THRES) * len(DIV_WINDOWS)
              * len(DIV_PRICE_RISES) * len(DIV_BREADTH_FALL) * len(DIV_BREADTH_CAPS))
    print(f"Buy & Hold return : {bh_ret:.1f}%")
    print(f"Searching {combos:,} combinations...\n")

    rows = []
    for buy_t, buy_50, win, rise, fall, cap in product(
        BUY_THRESHOLDS, BUY_50_THRES, DIV_WINDOWS,
        DIV_PRICE_RISES, DIV_BREADTH_FALL, DIV_BREADTH_CAPS
    ):
        total, cagr, sharpe, max_dd, trades, final = run(df, buy_t, buy_50, win, rise, fall, cap)
        rows.append({
            "buy_t": buy_t, "buy_50": buy_50, "win": win,
            "rise": rise, "fall": fall, "cap": cap,
            "Return": total, "CAGR": cagr, "Sharpe": sharpe,
            "MaxDD": max_dd, "Trades": trades, "Final$": final,
        })

    results = pd.DataFrame(rows)

    def fmt(r: pd.Series) -> str:
        b50s = f"{r['buy_50']:.0f}" if r['buy_50'] < 999 else "off"
        return (
            f"buy200<{r['buy_t']:4.0f}  b50<{b50s:>3}  win={r['win']:3.0f}  "
            f"rise={r['rise']:4.1f}  fall={r['fall']:4.0f}  cap={r['cap']:4.0f}  "
            f"Ret={r['Return']:7.1f}%  CAGR={r['CAGR']:5.1f}%  "
            f"Sharpe={r['Sharpe']:5.2f}  MaxDD={r['MaxDD']:6.1f}%  "
            f"T={r['Trades']:2.0f}  ${r['Final$']:>10,.0f}"
        )

    header = "buy200   b50    win   rise  fall   cap    Return   CAGR   Sharpe   MaxDD   T   Final $"
    sep    = "-" * len(header)

    print(f"Top 15 by Total Return (beating Buy & Hold {bh_ret:.1f}%):")
    print(header); print(sep)
    for _, r in results[results["Return"] > bh_ret].nlargest(15, "Return").iterrows():
        print(fmt(r))

    print(f"\nTop 15 by Sharpe (beating Buy & Hold {bh_ret:.1f}%):")
    print(header); print(sep)
    for _, r in results[results["Return"] > bh_ret].nlargest(15, "Sharpe").iterrows():
        print(fmt(r))

    print(f"\nTop 15 by Sharpe (all results):")
    print(header); print(sep)
    for _, r in results.nlargest(15, "Sharpe").iterrows():
        print(fmt(r))

    baseline_row = results[
        (results["buy_t"] == 18) & (results["buy_50"] == 25) &
        (results["win"] == 60)   & (results["rise"] == 3) &
        (results["fall"] == 20)  & (results["cap"] == 60)
    ]
    if not baseline_row.empty:
        print(f"\nBaseline (current igv_backtest.py settings):")
        print(fmt(baseline_row.iloc[0]))


if __name__ == "__main__":
    main()
