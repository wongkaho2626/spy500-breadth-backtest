import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

DATA_DIR     = Path(__file__).parent
QQQ_FILE     = DATA_DIR / "QQQ ETF Stock Price History.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005


def _parse_price(s):
    return s.astype(str).str.replace(",", "").astype(float)


def load_base():
    qqq_raw     = pd.read_csv(QQQ_FILE)
    breadth_raw = pd.read_csv(BREADTH_FILE)
    for df in (qqq_raw, breadth_raw):
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["Price"] = _parse_price(df["Price"])
    merged = qqq_raw[["Price"]].join(
        breadth_raw[["Price"]], lsuffix="_qqq", rsuffix="_breadth", how="inner"
    )
    merged = merged.rename(columns={"Price_qqq": "qqq_price", "Price_breadth": "breadth"})
    merged.sort_index(inplace=True)
    return merged


def run(df, buy_t, win, rise, fall, cap):
    price_past   = df["qqq_price"].shift(win)
    breadth_past = df["breadth"].shift(win)
    bearish_div  = (
        ((df["qqq_price"] - price_past) / price_past * 100 >= rise) &
        ((breadth_past - df["breadth"]) >= fall) &
        (df["breadth"] < cap)
    )

    position  = "OUT"
    eff_entry = 0.0
    portfolio = INITIAL_CAPITAL
    values    = {}
    trades    = 0

    for date, row in df.iterrows():
        price = row["qqq_price"]
        bd    = bool(bearish_div.loc[date])

        if position == "OUT" and row["breadth"] < buy_t:
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

    s = pd.Series(values)
    dr       = s.pct_change().dropna()
    total    = (s.iloc[-1] / s.iloc[0]) - 1
    years    = (s.index[-1] - s.index[0]).days / 365.25
    cagr     = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    max_dd   = ((s - s.cummax()) / s.cummax()).min()
    std      = dr.std()
    sharpe   = dr.mean() / std * np.sqrt(252) if std > 0 else 0.0
    return total * 100, cagr * 100, sharpe, max_dd * 100, trades


df = load_base()

bh_ret = (df["qqq_price"].iloc[-1] / df["qqq_price"].iloc[0] - 1) * 100
print(f"Buy & Hold return: {bh_ret:.1f}%\n")

buy_thresholds  = [19.5, 20.5, 22.0, 24.0, 26.0, 28.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0]
windows         = [40, 60, 80, 100, 120]
rises           = [0.5, 1.0, 2.0, 3.0, 5.0]
falls           = [8.0, 10.0, 12.0, 15.0, 18.0, 20.0]
caps            = [45.0, 50.0, 55.0, 60.0]

combos = list(product(buy_thresholds, windows, rises, falls, caps))
print(f"Searching {len(combos):,} combinations...")

results = []
for buy_t, win, rise, fall, cap in combos:
    r = run(df, buy_t, win, rise, fall, cap)
    results.append((r[0], r[1], r[2], r[3], r[4], buy_t, win, rise, fall, cap))

results.sort(key=lambda x: -x[0])

print(f"\nTop 20 by Total Return:\n")
print(f"  {'Return':>8}  {'CAGR':>6}  {'Sharpe':>6}  {'MaxDD':>7}  {'Trades':>6}  {'BuyT':>5}  {'Win':>4}  {'Rise':>5}  {'Fall':>5}  {'Cap':>5}")
print("-" * 90)
for r in results[:20]:
    print(f"  {r[0]:>7.1f}%  {r[1]:>5.1f}%  {r[2]:>6.2f}  {r[3]:>6.1f}%  {r[4]:>6}  {r[5]:>5}  {r[6]:>4}  {r[7]:>5}  {r[8]:>5}  {r[9]:>5}")

by_sharpe = sorted(results, key=lambda x: -x[2])
print(f"\nTop 10 by Sharpe (beating BH {bh_ret:.1f}%):\n")
print(f"  {'Return':>8}  {'CAGR':>6}  {'Sharpe':>6}  {'MaxDD':>7}  {'Trades':>6}  {'BuyT':>5}  {'Win':>4}  {'Rise':>5}  {'Fall':>5}  {'Cap':>5}")
print("-" * 90)
shown = 0
for r in by_sharpe:
    if r[0] > bh_ret:
        print(f"  {r[0]:>7.1f}%  {r[1]:>5.1f}%  {r[2]:>6.2f}  {r[3]:>6.1f}%  {r[4]:>6}  {r[5]:>5}  {r[6]:>4}  {r[7]:>5}  {r[8]:>5}  {r[9]:>5}")
        shown += 1
        if shown == 10:
            break
if shown == 0:
    print("  (none beat buy & hold)")
    for r in by_sharpe[:10]:
        print(f"  {r[0]:>7.1f}%  {r[1]:>5.1f}%  {r[2]:>6.2f}  {r[3]:>6.1f}%  {r[4]:>6}  {r[5]:>5}  {r[6]:>4}  {r[7]:>5}  {r[8]:>5}  {r[9]:>5}")
