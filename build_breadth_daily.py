"""
Build a continuous DAILY S&P 500 breadth series (breadth_daily.csv).

Problem: S5TH.csv (% of S&P 500 stocks above their 200-day MA) is only daily
from 2007 — before that it has one point every ~2 months, which silently
corrupts any backtest window computed in "rows".

Fix: MMTH.csv (broader-universe % above 200-day MA) is daily from 2002 and
correlates ~0.94 with S5TH on their 2007+ overlap, but sits ~8 pts lower.
This script fits a linear map S5TH ≈ a + b·MMTH on the overlap, applies it to
2002–2006 MMTH, and splices:

    2002-01-02 … 2006-12-31 : regression-mapped MMTH  (source = MMTH-mapped)
    2007-01-03 … present    : actual daily S5TH       (source = S5TH)

Output: breadth_daily.csv with columns Date (MM/DD/YYYY), breadth, source.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR    = Path(__file__).parent
OUT_FILE    = DATA_DIR / "breadth_daily.csv"
DAILY_START = pd.Timestamp("2007-01-01")   # S5TH is genuinely daily from here


def load_s5th() -> pd.Series:
    df = pd.read_csv(DATA_DIR / "S5TH.csv", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()
    return df["Price"].astype(str).str.replace(",", "").astype(float)


def load_mmth() -> pd.Series:
    df = pd.read_csv(DATA_DIR / "MMTH.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")["close"].sort_index()


def main() -> None:
    s5th = load_s5th()
    mmth = load_mmth()

    overlap = pd.concat([s5th.rename("s5th"), mmth.rename("mmth")], axis=1).dropna()
    overlap = overlap[overlap.index >= DAILY_START]
    b, a = np.polyfit(overlap["mmth"], overlap["s5th"], 1)
    fitted = a + b * overlap["mmth"]
    r = overlap["s5th"].corr(overlap["mmth"])
    rmse = np.sqrt(((overlap["s5th"] - fitted) ** 2).mean())
    print(f"Overlap fit on {len(overlap):,} days: S5TH ≈ {a:.2f} + {b:.3f}·MMTH")
    print(f"  correlation r = {r:.3f}, RMSE = {rmse:.1f} pts")

    pre = mmth[mmth.index < DAILY_START]
    mapped = (a + b * pre).clip(0, 100).rename("breadth")
    real = s5th[s5th.index >= DAILY_START].rename("breadth")

    combined = pd.concat([
        mapped.to_frame().assign(source="MMTH-mapped"),
        real.to_frame().assign(source="S5TH"),
    ]).sort_index()

    out = combined.reset_index()
    out.columns = ["Date", "breadth", "source"]
    out["Date"] = out["Date"].dt.strftime("%m/%d/%Y")
    out["breadth"] = out["breadth"].round(2)
    out.to_csv(OUT_FILE, index=False)

    print(f"\nWrote {len(out):,} rows -> {OUT_FILE.name}")
    print(f"  {out['Date'].iloc[0]} … {out['Date'].iloc[-1]}")
    print(f"  {len(mapped):,} mapped MMTH rows (2002–2006), {len(real):,} real S5TH rows")


if __name__ == "__main__":
    main()
