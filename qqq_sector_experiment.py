"""
QQQ / NASDAQ 100 Breadth Strategy — sector-data experiment

Replicates the baseline strategy from qqq_backtest.py and layers on optional
sector-ETF filters, then compares all variants side by side:

  baseline   : breadth200 < 26 AND (VIX>30 OR price>MA200); sell on bearish divergence
  sector-buy : baseline + buy also requires sector breadth (% of 11 SPDR sectors
               above their own 200-day MA) below a washout threshold
  regime     : baseline + buy only when XLY/XLP ratio is above its 200-day MA
               (risk-on regime)
  def-exit   : baseline + extra sell when defensive rotation spread
               (63d return of (XLP+XLU)/2 minus XLK) exceeds a threshold
  sector-div : baseline sell additionally requires sector breadth to have
               fallen ≥ N pts over the divergence window (confirmation)

Data: NASDAQ100.csv, S5TH.csv, VIX.csv + XL*.csv (run fetch_sector_data.py first).
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR     = Path(__file__).parent
NDX_FILE     = DATA_DIR / "NASDAQ100.csv"
BREADTH_FILE = DATA_DIR / "S5TH.csv"
# Continuous daily breadth (2002+) built by build_breadth_daily.py.
# S5TH.csv alone is only daily from 2007 — before that it is bimonthly, which
# corrupts row-based lookback windows (a "60-day" window spans ~10 years).
BREADTH_DAILY_FILE = DATA_DIR / "breadth_daily.csv"
BREADTH_DAILY_MIN  = "2007-01-01"  # fallback cutoff when daily file is absent
VIX_FILE     = DATA_DIR / "VIX.csv"

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLY", "XLP", "XLE", "XLI", "XLB", "XLRE", "XLU", "XLC"]

# ── Baseline parameters (same as qqq_backtest.py) ─────────────────────────────
BUY_B200_THRESH = 26.0
VIX_BUY_THRESH  = 30.0
MA200_WINDOW    = 200

DIVERGENCE_WINDOW       = 60
DIVERGENCE_PRICE_RISE   = 3.0
DIVERGENCE_BREADTH_FALL = 20.0
DIVERGENCE_BREADTH_CAP  = 60.0

INITIAL_CAPITAL = 10_000.0
COMMISSION      = 1.0
SLIPPAGE        = 0.0005
COOLDOWN_DAYS   = 15

# ── Sector-filter parameters ──────────────────────────────────────────────────
SECTOR_BREADTH_BUY_MAX   = 50.0   # sector-buy: sector breadth must be below this at entry
DEFENSIVE_SPREAD_WINDOW  = 63     # def-exit: trailing days for rotation spread
DEFENSIVE_SPREAD_THRESH  = 8.0    # def-exit: (XLP+XLU)/2 minus XLK trailing return, pts
SECTOR_DIV_FALL          = 18.0   # sector-div: sector breadth fall (pts) over divergence window
REGIME_MA_WINDOW         = 200    # regime: XLY/XLP ratio vs its own MA


def _parse_price(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(",", "").astype(float)


def _load_price_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()
    return _parse_price(df["Price"])


def _load_breadth() -> pd.Series:
    """Prefer the continuous daily series (breadth_daily.csv, 2002+); S5TH.csv
    alone is bimonthly before 2007, which corrupts row-based windows."""
    if BREADTH_DAILY_FILE.exists():
        df = pd.read_csv(BREADTH_DAILY_FILE)
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        return df.set_index("Date")["breadth"].sort_index()
    s = _load_price_csv(BREADTH_FILE)
    # S5TH is bimonthly before 2007 — drop the sparse era
    return s[s.index >= BREADTH_DAILY_MIN]


def load_data() -> pd.DataFrame:
    ndx     = _load_price_csv(NDX_FILE).rename("price")
    breadth = _load_breadth().rename("breadth")
    vix     = _load_price_csv(VIX_FILE).rename("vix")

    merged = pd.concat([ndx, breadth, vix], axis=1).sort_index()
    merged = merged[merged["breadth"].notna() & merged["price"].notna()]

    merged["vix"]   = merged["vix"].ffill()
    merged["ma200"] = merged["price"].rolling(MA200_WINDOW).mean()

    merged["vix_vote"]   = merged["vix"].isna() | (merged["vix"] > VIX_BUY_THRESH)
    merged["ma200_vote"] = merged["ma200"].isna() | (merged["price"] > merged["ma200"])
    merged["vote_gate"]  = merged["vix_vote"] | merged["ma200_vote"]

    pp = merged["price"].shift(DIVERGENCE_WINDOW)
    bp = merged["breadth"].shift(DIVERGENCE_WINDOW)
    merged["price_rose"]   = ((merged["price"] - pp) / pp * 100 >= DIVERGENCE_PRICE_RISE).fillna(False)
    merged["breadth_fell"] = ((bp - merged["breadth"]) >= DIVERGENCE_BREADTH_FALL).fillna(False)
    # Trend re-entry: fresh close back above MA200 (gate reduces to "above prior exit").
    merged["ma200_recross"] = (
        (merged["price"] > merged["ma200"]) & (merged["price"].shift(1) <= merged["ma200"].shift(1))
    ).fillna(False)

    # ── Sector data ───────────────────────────────────────────────────────────
    sectors = pd.DataFrame({s: _load_price_csv(DATA_DIR / f"{s}.csv") for s in SECTOR_ETFS})
    sectors = sectors.sort_index()

    above = pd.DataFrame({
        s: sectors[s] > sectors[s].rolling(MA200_WINDOW).mean() for s in SECTOR_ETFS
    })
    has_data = sectors.notna()
    counts = has_data.sum(axis=1)
    sector_breadth = (above.astype(float).where(has_data).sum(axis=1) / counts.replace(0, np.nan)) * 100

    # defensive rotation spread: trailing return of (XLP+XLU)/2 minus XLK, in pts
    def _trail_ret(s: pd.Series) -> pd.Series:
        return (s / s.shift(DEFENSIVE_SPREAD_WINDOW) - 1) * 100
    def_spread = (_trail_ret(sectors["XLP"]) + _trail_ret(sectors["XLU"])) / 2 - _trail_ret(sectors["XLK"])

    # risk-appetite regime: XLY/XLP ratio above its 200-day MA
    ratio = sectors["XLY"] / sectors["XLP"]
    risk_on = ratio > ratio.rolling(REGIME_MA_WINDOW).mean()

    merged = merged.join(sector_breadth.rename("sec_breadth"), how="left")
    merged = merged.join(def_spread.rename("def_spread"), how="left")
    merged = merged.join(risk_on.rename("risk_on"), how="left")

    merged["sec_breadth"] = merged["sec_breadth"].ffill()
    merged["def_spread"]  = merged["def_spread"].ffill()
    merged["risk_on"]     = merged["risk_on"].ffill()

    sbp = merged["sec_breadth"].shift(DIVERGENCE_WINDOW)
    merged["sec_breadth_fell"] = ((sbp - merged["sec_breadth"]) >= SECTOR_DIV_FALL).fillna(False)

    return merged


def run_strategy(df: pd.DataFrame, *,
                 sector_buy: bool = False,
                 regime: bool = False,
                 def_exit: bool = False,
                 sector_div: bool = False) -> tuple[pd.Series, list[dict]]:
    position   = "OUT"
    eff_entry  = raw_entry = 0.0
    entry_date = None
    portfolio  = INITIAL_CAPITAL
    cooldown_until: pd.Timestamp | None = None
    last_exit_price: float | None = None
    trades: list[dict] = []
    values: dict = {}

    for date, row in df.iterrows():
        price = row["price"]

        if position == "OUT":
            cooldown_ok = cooldown_until is None or date > cooldown_until
            washout_buy = (not pd.isna(row["breadth"])
                           and row["breadth"] < BUY_B200_THRESH
                           and bool(row["vote_gate"]))
            # Trend re-entry: fresh MA200 recross once price is back above the prior exit.
            trend_buy   = bool(row["ma200_recross"]) and (
                last_exit_price is not None and price > last_exit_price)
            do_buy = cooldown_ok and (washout_buy or trend_buy)
            if do_buy and sector_buy:
                do_buy = pd.isna(row["sec_breadth"]) or row["sec_breadth"] < SECTOR_BREADTH_BUY_MAX
            if do_buy and regime:
                do_buy = pd.isna(row["risk_on"]) or bool(row["risk_on"])
            if do_buy:
                portfolio -= COMMISSION
                eff_entry  = price * (1 + SLIPPAGE)
                raw_entry  = price
                entry_date = date
                position   = "IN"

        elif position == "IN":
            bearish_div = (bool(row["price_rose"]) and bool(row["breadth_fell"])
                           and row["breadth"] < DIVERGENCE_BREADTH_CAP)
            if bearish_div and sector_div:
                bearish_div = bool(row["sec_breadth_fell"])
            defensive = (def_exit and not pd.isna(row["def_spread"])
                         and row["def_spread"] >= DEFENSIVE_SPREAD_THRESH)
            reason = "bearish-divergence" if bearish_div else ("defensive-rotation" if defensive else None)
            if reason:
                eff_exit  = price * (1 - SLIPPAGE)
                gross_ret = (eff_exit - eff_entry) / eff_entry
                portfolio *= (1 + gross_ret)
                portfolio -= COMMISSION
                cooldown_until = date + pd.Timedelta(days=COOLDOWN_DAYS)
                last_exit_price = price
                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "entry_price": raw_entry, "exit_price": price,
                    "return_pct": gross_ret * 100, "sell_reason": reason,
                })
                position = "OUT"

        values[date] = portfolio * (price * (1 - SLIPPAGE) / eff_entry) if position == "IN" else portfolio

    return pd.Series(values), trades


def compute_metrics(values: pd.Series, trades: list[dict]) -> dict:
    dr    = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    tr    = values.iloc[-1] / values.iloc[0] - 1
    cagr  = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    mdd   = ((values - values.cummax()) / values.cummax()).min()
    std   = dr.std()
    sh    = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    n     = len(trades)
    wins  = sum(1 for t in trades if t["return_pct"] > 0)
    in_days = sum((t["exit_date"] - t["entry_date"]).days for t in trades)
    tot     = (values.index[-1] - values.index[0]).days
    return {
        "Total Return": f"{tr:+.0%}", "CAGR": f"{cagr:.1%}",
        "Max DD": f"{mdd:.1%}", "Sharpe": f"{sh:.2f}",
        "Final": f"${values.iloc[-1]:,.0f}",
        "Trades": str(n), "Win%": f"{wins/n:.0%}" if n else "—",
        "InMkt": f"{in_days/tot:.0%}" if tot else "—",
    }


def main() -> None:
    df = load_data()
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} trading days)")

    variants = {
        "baseline":            {},
        "sector-buy":          {"sector_buy": True},
        "regime (XLY/XLP)":    {"regime": True},
        "def-exit":            {"def_exit": True},
        "sector-div confirm":  {"sector_div": True},
        "sector-buy + regime": {"sector_buy": True, "regime": True},
        "all filters":         {"sector_buy": True, "regime": True, "def_exit": True, "sector_div": True},
    }

    results = {}
    for name, kwargs in variants.items():
        values, trades = run_strategy(df, **kwargs)
        results[name] = compute_metrics(values, trades)

    bench = INITIAL_CAPITAL * df["price"] / df["price"].iloc[0]
    results["buy & hold NDX"] = compute_metrics(bench, [])

    table = pd.DataFrame(results).T
    print("\n=== Variant comparison ===")
    print(table.to_string())


if __name__ == "__main__":
    main()
