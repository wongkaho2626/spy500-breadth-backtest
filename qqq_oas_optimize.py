import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR     = Path(__file__).parent
QQQ_FILE     = DATA_DIR / "QQQ ETF Stock Price History.csv"
BREADTH_FILE = DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv"
OAS_FILE     = DATA_DIR / "BAMLH0A0HYM2.csv"

# Fixed base parameters (known-best from qqq_optimize.py)
BUY_THRESHOLD           = 26.0
DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0
INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005

# OAS grid (0.0 / 999.0 = filter disabled)
OAS_BUY_MINS  = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
OAS_SELL_CAPS = [999.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
# 8 × 8 = 64 combinations


def _parse_price(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", "").astype(float)


def load_base() -> pd.DataFrame:
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

    oas_raw = pd.read_csv(OAS_FILE, parse_dates=["date"])
    oas_raw = oas_raw.set_index("date")[["close"]].rename(columns={"close": "oas"}).sort_index()
    merged["oas"] = oas_raw["oas"].reindex(merged.index, method="ffill")

    price_past   = merged["qqq_price"].shift(DIVERGENCE_WINDOW)
    breadth_past = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["div_base"] = (
        ((merged["qqq_price"] - price_past) / price_past * 100 >= DIVERGENCE_PRICE_RISE) &
        ((breadth_past - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL) &
        (merged["breadth"] < DIVERGENCE_BREADTH_CAP)
    )
    return merged


def run_combo(df: pd.DataFrame, oas_buy_min: float, oas_sell_cap: float) -> dict:
    bearish_div_series = df["div_base"] & (df["oas"] < oas_sell_cap)

    position    = "OUT"
    eff_entry   = 0.0
    entry_price = 0.0
    entry_date  = None
    trade_high  = trade_low = 0.0
    portfolio   = INITIAL_CAPITAL
    trades      = 0
    wins        = 0
    in_days     = 0
    values: dict = {}

    for date, row in df.iterrows():
        price       = row["qqq_price"]
        breadth     = row["breadth"]
        bearish_div = bool(bearish_div_series[date])
        oas         = row["oas"] if not pd.isna(row["oas"]) else 0.0

        buy_ok = breadth < BUY_THRESHOLD and (oas_buy_min == 0.0 or oas >= oas_buy_min)

        if position == "OUT" and buy_ok:
            portfolio  -= COMMISSION
            eff_entry   = price * (1 + SLIPPAGE)
            entry_price = price
            entry_date  = date
            trade_high  = trade_low = price
            position    = "IN"
        elif position == "IN":
            trade_high = max(trade_high, price)
            trade_low  = min(trade_low, price)
            if bearish_div:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                trades    += 1
                if gross_ret > 0:
                    wins += 1
                in_days += (date - entry_date).days
                position = "OUT"

        values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry) if position == "IN" else portfolio

    series       = pd.Series(values)
    daily_ret    = series.pct_change().dropna()
    total_return = (series.iloc[-1] / series.iloc[0]) - 1
    years        = (series.index[-1] - series.index[0]).days / 365.25
    cagr         = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
    rolling_max  = series.cummax()
    max_dd       = ((series - rolling_max) / rolling_max).min()
    std          = daily_ret.std()
    sharpe       = (daily_ret.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    return {
        "oas_buy_min":  oas_buy_min,
        "oas_sell_cap": oas_sell_cap,
        "Return":       total_return * 100,
        "CAGR":         cagr * 100,
        "Sharpe":       sharpe,
        "MaxDD":        max_dd * 100,
        "Trades":       trades,
        "Final$":       series.iloc[-1],
    }


def fmt_row(r: pd.Series) -> str:
    bm = f"{r['oas_buy_min']:.1f}%" if r['oas_buy_min'] > 0 else "  off"
    sc = f"{r['oas_sell_cap']:.1f}%" if r['oas_sell_cap'] < 900 else "  off"
    return (f"{bm:>7} {sc:>8}  {r['Return']:>8.1f}%  {r['CAGR']:>5.1f}%  {r['Sharpe']:>7.2f}"
            f"  {r['MaxDD']:>6.1f}%  {r['Trades']:>7.0f}  ${r['Final$']:>11,.0f}")


def main() -> None:
    print("Loading data...")
    df = load_base()

    bh_return = (df["qqq_price"].iloc[-1] / df["qqq_price"].iloc[0] - 1) * 100
    n_combos  = len(OAS_BUY_MINS) * len(OAS_SELL_CAPS)
    print(f"Buy & Hold return: {bh_return:.1f}%")
    print(f"Searching {n_combos} OAS combinations...")

    rows = [run_combo(df, bm, sc) for bm in OAS_BUY_MINS for sc in OAS_SELL_CAPS]
    results = pd.DataFrame(rows)

    header = (f"\n{'BuyMin':>7} {'SellCap':>8}  {'Return':>9}  {'CAGR':>6}  "
              f"{'Sharpe':>7}  {'MaxDD':>7}  {'Trades':>7}  {'Final $':>12}")
    sep = "-" * len(header)

    print(f"\nAll results beating Buy & Hold ({bh_return:.1f}%), by Total Return:")
    print(header)
    print(sep)
    beating = results[results["Return"] > bh_return].nlargest(20, "Return")
    if beating.empty:
        print("  (none beat buy & hold)")
    else:
        for _, r in beating.iterrows():
            print(fmt_row(r))

    print(f"\nAll results beating Buy & Hold ({bh_return:.1f}%), by Sharpe:")
    print(header)
    print(sep)
    beating_s = results[results["Return"] > bh_return].nlargest(20, "Sharpe")
    if beating_s.empty:
        print("  (none beat buy & hold)")
    else:
        for _, r in beating_s.iterrows():
            print(fmt_row(r))

    base = run_combo(df, 0.0, 999.0)
    print(f"\nBaseline (no OAS filter):  {base['Return']:.1f}%  CAGR {base['CAGR']:.1f}%  "
          f"Sharpe {base['Sharpe']:.2f}  MaxDD {base['MaxDD']:.1f}%  {base['Trades']:.0f} trades  "
          f"${base['Final$']:,.0f}")

    print(f"\nAll {n_combos} combinations (sorted by Sharpe):")
    print(header)
    print(sep)
    for _, r in results.nlargest(n_combos, "Sharpe").iterrows():
        print(fmt_row(r))


if __name__ == "__main__":
    main()
