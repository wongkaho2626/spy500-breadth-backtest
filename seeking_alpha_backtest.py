#!/usr/bin/env python3
"""
Seeking Alpha annual stock picks backtest.

Compares three strategies using S&P 500 market indicators:

  A) Baseline  : Buy Jan 1 every year, sell Dec 31 / current
  B) PE Filter : Buy first day S&P 500 fwd PE < 20, sell Dec 31 / current
                 (replicates the CSV right-side timing)
  C) Enhanced  : Entry = PE < 20  OR  (VIX ≥ 22 AND breadth ≤ 50)
                 Exit  = SPX bearish-divergence  OR  trailing stop  OR  year-end

Stock universe: 10 picks / year from seeking_alpha.csv.
Portfolio return: equal-weight across all 10 stocks per year.
Entry prices: CSV prices when entry date matches CSV signal dates; SPX proxy
              (beta = 1) for all other entry dates.
Year-end exits: calibrated to actual CSV stock prices (accurate).
Intra-year early exits: SPX proxy return (approximation).
"""
import csv
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA_DIR = Path(__file__).parent

SA_CSV      = DATA_DIR / "seeking_alpha.csv"
SPX_CSV     = DATA_DIR / "SPX.csv"
FWD_PE_CSV  = DATA_DIR / "S&P500ForwardPE.csv"
BREADTH_CSV = DATA_DIR / "S5TH.csv"
VIX_CSV     = DATA_DIR / "VIX.csv"

# ── Strategy parameters ──────────────────────────────────────────────────────
INITIAL_CAPITAL     = 100_000.0   # starting capital ($)
FWD_PE_BUY          = 20.0        # primary entry: S&P 500 fwd PE < this
VIX_ALT_THRESH      = 22.0        # alt-entry (Strategy C): VIX ≥ this …
BREADTH_ALT_THRESH  = 50.0        # … AND breadth ≤ this  (fear + oversold)
DIV_WINDOW          = 60          # bearish-divergence lookback (trading days)
DIV_PRICE_RISE      = 5.0         # SPX % rise  required for divergence
DIV_BREADTH_FALL    = 20.0        # breadth pts fall required (stricter than B)
DIV_BREADTH_CAP     = 60.0        # breadth must be below this for divergence
TRAILING_STOP_PCT   = 25.0        # % drop from SPX peak triggers exit (C only)
MA200_WINDOW        = 200         # MA200 for SPX


# ── Data loading ─────────────────────────────────────────────────────────────

def _parse_price(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", "").astype(float)


def load_market_data() -> pd.DataFrame:
    """Merge SPX, forward PE, breadth (S5TH), and VIX onto daily SPX dates."""

    def read_csv_investing(path: Path, col: str = "Price") -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.strip().strip('"').lstrip("﻿") for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["val"] = _parse_price(df[col])
        return df[["val"]]

    spx = read_csv_investing(SPX_CSV).rename(columns={"val": "spx"})

    fpe = pd.read_csv(FWD_PE_CSV)
    fpe["Date"] = pd.to_datetime(fpe["date"], format="%Y-%m-%d")
    fpe.set_index("Date", inplace=True)
    fpe = fpe.rename(columns={"forward_pe": "fwd_pe"})[["fwd_pe"]]

    brd = read_csv_investing(BREADTH_CSV).rename(columns={"val": "breadth"})
    vix = read_csv_investing(VIX_CSV).rename(columns={"val": "vix"})

    df = spx.join(fpe, how="left").join(brd, how="left").join(vix, how="left")
    df.sort_index(inplace=True)

    # Forward-fill indicator data (handles weekends/missing days)
    df["fwd_pe"]  = df["fwd_pe"].ffill()
    df["breadth"] = df["breadth"].ffill()
    df["vix"]     = df["vix"].ffill()
    df["ma200"]   = df["spx"].rolling(MA200_WINDOW).mean()

    # Pre-compute SPX bearish-divergence signal (for Strategy C exit)
    spx_past = df["spx"].shift(DIV_WINDOW)
    brd_past = df["breadth"].shift(DIV_WINDOW)
    df["spx_rose"] = ((df["spx"] - spx_past) / spx_past * 100 >= DIV_PRICE_RISE).fillna(False)
    df["brd_fell"] = ((brd_past - df["breadth"]) >= DIV_BREADTH_FALL).fillna(False)
    df["div_ok"]   = df["spx_rose"] & df["brd_fell"] & (df["breadth"] < DIV_BREADTH_CAP)

    return df


# ── CSV parsing ──────────────────────────────────────────────────────────────

def _parse_date_ymd(s: str):
    m = re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s.strip())
    return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def parse_sa_csv() -> dict:
    """
    Parse seeking_alpha.csv.

    Returns dict: year (int) → {
      'left_start':  Timestamp,   Jan 1 entry date
      'right_start': Timestamp,   PE-signal entry date from CSV
      'year_end':    Timestamp|'current',
      'stocks': [{ticker, left_entry, left_exit, right_entry, right_exit}]
    }
    """
    year_data: dict = {}
    cur_year: int | None = None
    cur_stocks: list = []

    def pf(s: str) -> float | None:
        try:
            return float(str(s).replace(",", "").replace('"', "").strip())
        except Exception:
            return None

    with open(SA_CSV, newline="") as fh:
        for row in csv.reader(fh):
            while len(row) < 10:
                row.append("")
            ticker = row[0].strip()
            d_left = _parse_date_ymd(row[1]) if row[1].strip() else None

            if not ticker and d_left is not None:
                # Year header row
                if cur_year is not None:
                    year_data[cur_year]["stocks"] = cur_stocks

                cur_year = d_left.year
                cur_stocks = []

                d_right = _parse_date_ymd(row[6]) if row[6].strip() else d_left
                raw_end = row[2].strip().lower()
                year_end = (
                    "current" if raw_end == "current"
                    else _parse_date_ymd(row[2]) if row[2].strip() else None
                )
                year_data[cur_year] = {
                    "left_start":  d_left,
                    "right_start": d_right if d_right else d_left,
                    "year_end":    year_end,
                    "stocks":      [],
                }

            elif ticker:
                le = pf(row[1])
                lx = pf(row[2])
                re_ = pf(row[6])
                rx = pf(row[7])
                if le is not None:
                    cur_stocks.append({
                        "ticker":      ticker,
                        "left_entry":  le,
                        "left_exit":   lx,
                        "right_entry": re_ if re_ is not None else le,
                        "right_exit":  rx if rx is not None else lx,
                    })

    if cur_year is not None and cur_stocks:
        year_data[cur_year]["stocks"] = cur_stocks

    return year_data


# ── Strategy simulation ───────────────────────────────────────────────────────

def _nearest_date(market: pd.DataFrame, target: pd.Timestamp) -> pd.Timestamp:
    """First trading day on or after target."""
    idx = market.index[market.index >= target]
    return idx[0] if len(idx) else market.index[-1]


def _spx(market: pd.DataFrame, date: pd.Timestamp) -> float:
    if date in market.index:
        return float(market.loc[date, "spx"])
    idx = market.index[market.index <= date]
    return float(market.loc[idx[-1], "spx"]) if len(idx) else float(market["spx"].iloc[0])


def _portfolio_return(
    stocks: list[dict],
    entry_date: pd.Timestamp,
    year_start_td: pd.Timestamp,
    right_start: pd.Timestamp,
    market: pd.DataFrame,
    use_left: bool = False,
) -> float:
    """
    Compute equal-weight portfolio return from entry_date to year-end.

    If entry_date matches right_start (± 2 days): uses CSV right_entry/right_exit.
    If use_left: uses CSV left_entry/left_exit directly.
    Otherwise: estimates entry price via SPX proxy (beta = 1).
    Returns decimal return (e.g. 0.50 for +50%).
    """
    if use_left:
        rets = [s["left_exit"] / s["left_entry"] - 1 for s in stocks
                if s["left_entry"] and s["left_exit"]]
        return sum(rets) / len(rets) if rets else 0.0

    if abs((entry_date - right_start).days) <= 2:
        rets = [s["right_exit"] / s["right_entry"] - 1 for s in stocks
                if s["right_entry"] and s["right_exit"]]
        return sum(rets) / len(rets) if rets else 0.0

    # SPX proxy: estimate entry price proportional to SPX move from year start
    spx_year_start = _spx(market, year_start_td)
    spx_entry      = _spx(market, entry_date)
    rets = []
    for s in stocks:
        if s["left_entry"] and s["left_exit"]:
            entry_price = s["left_entry"] * (spx_entry / spx_year_start)
            rets.append(s["left_exit"] / entry_price - 1)
    return sum(rets) / len(rets) if rets else 0.0


def run_strategies(sa_data: dict, market: pd.DataFrame):
    """
    Simulate strategies A, B, C.

    Returns three dicts each with keys:
      'values': pd.Series (daily portfolio $)
      'trades': list[dict]
      'open_trade': dict | None
    """
    market = market[market.index >= pd.Timestamp("2022-01-01")].copy()
    results = {s: {"values": {}, "trades": [], "open_trade": None}
               for s in ("A", "B", "C")}
    caps = {"A": INITIAL_CAPITAL, "B": INITIAL_CAPITAL, "C": INITIAL_CAPITAL}

    for year in sorted(sa_data.keys()):
        info    = sa_data[year]
        stocks  = info["stocks"]
        if not stocks:
            continue

        left_start  = info["left_start"]
        right_start = info["right_start"]
        raw_end     = info["year_end"]

        year_start_td = _nearest_date(market, left_start)
        if raw_end == "current":
            year_end_td = market.index[-1]
        elif raw_end is not None:
            year_end_td = _nearest_date(market, raw_end)
        else:
            cands = market.index[market.index.year == year]
            year_end_td = cands[-1] if len(cands) else market.index[-1]

        year_slice = market.loc[year_start_td:year_end_td]
        if year_slice.empty:
            continue

        year_dates = year_slice.index

        for strat in ("A", "B", "C"):
            cap = caps[strat]

            # ── Determine entry date ──────────────────────────────────────
            entry_td   = None
            entry_note = ""

            if strat == "A":
                entry_td   = year_start_td
                entry_note = "Jan 1"

            elif strat in ("B", "C"):
                # Primary condition: PE < FWD_PE_BUY
                pe_mask = year_slice["fwd_pe"] < FWD_PE_BUY
                if pe_mask.any():
                    entry_td   = year_slice.index[pe_mask][0]
                    entry_note = f"PE<{FWD_PE_BUY:.0f}"

                if strat == "C" and entry_td is None:
                    # Alternative condition: VIX ≥ threshold AND breadth ≤ threshold
                    alt_mask = (
                        (year_slice["vix"] >= VIX_ALT_THRESH) &
                        (year_slice["breadth"] <= BREADTH_ALT_THRESH)
                    )
                    if alt_mask.any():
                        entry_td   = year_slice.index[alt_mask][0]
                        entry_note = f"VIX≥{VIX_ALT_THRESH:.0f}+B≤{BREADTH_ALT_THRESH:.0f}"

                # Fallback: if no signal triggered, enter on Jan 1 (never skip)
                if strat == "C" and entry_td is None:
                    entry_td   = year_start_td
                    entry_note = "Jan 1 (fallback)"

            if entry_td is None:
                # No signal: hold cash
                for d in year_dates:
                    results[strat]["values"][d] = cap
                results[strat]["trades"].append({
                    "year": year, "strat": strat,
                    "entry_date": None, "exit_date": None,
                    "entry_note": "skipped (no signal)",
                    "exit_note": "—",
                    "return_pct": 0.0, "entry_cap": cap, "exit_cap": cap,
                })
                continue

            spx_year_start = _spx(market, year_start_td)
            spx_entry      = _spx(market, entry_td)

            # ── Determine exit date (Strategy C) ─────────────────────────
            exit_td    = year_end_td
            exit_note  = "year-end"
            is_early   = False

            if strat == "C":
                spx_peak = spx_entry
                for d in year_slice.index[year_slice.index >= entry_td]:
                    row = market.loc[d]
                    spx_d = float(row["spx"])
                    spx_peak = max(spx_peak, spx_d)

                    if d == year_end_td:
                        break

                    # Trailing stop
                    if spx_d < spx_peak * (1.0 - TRAILING_STOP_PCT / 100.0):
                        exit_td   = d
                        exit_note = f"trailing-stop(-{TRAILING_STOP_PCT:.0f}%)"
                        is_early  = True
                        break

                    # Bearish divergence on SPX + breadth
                    if bool(row["div_ok"]):
                        exit_td   = d
                        exit_note = "bearish-div"
                        is_early  = True
                        break

            # ── Compute portfolio return ─────────────────────────────────
            if is_early:
                # SPX proxy for intra-year exit
                spx_exit = _spx(market, exit_td)
                ret = spx_exit / spx_entry - 1
            else:
                use_left = (strat == "A")
                ret = _portfolio_return(
                    stocks, entry_td, year_start_td, right_start, market, use_left=use_left
                )

            exit_cap = cap * (1.0 + ret)
            is_open  = (exit_td == market.index[-1] and raw_end == "current")

            trade = {
                "year":       year,
                "strat":      strat,
                "entry_date": entry_td,
                "exit_date":  exit_td,
                "entry_note": entry_note,
                "exit_note":  exit_note,
                "return_pct": ret * 100.0,
                "entry_cap":  cap,
                "exit_cap":   exit_cap,
                "is_open":    is_open,
            }
            if is_open:
                results[strat]["open_trade"] = trade
            else:
                results[strat]["trades"].append(trade)

            # ── Daily portfolio values (SPX proxy within holding) ────────
            for d in year_dates:
                spx_d = _spx(market, d)
                if d < entry_td:
                    results[strat]["values"][d] = cap
                elif is_early and d > exit_td:
                    results[strat]["values"][d] = exit_cap
                elif d == exit_td and not is_early:
                    results[strat]["values"][d] = exit_cap
                else:
                    # SPX proxy while holding
                    results[strat]["values"][d] = cap * (spx_d / spx_entry)

            caps[strat] = exit_cap

    # Build SPX benchmark (normalised to INITIAL_CAPITAL from 2022-01-01)
    spx_start = float(market["spx"].iloc[0])
    bench = (INITIAL_CAPITAL * market["spx"] / spx_start).rename("benchmark")

    for strat in ("A", "B", "C"):
        s = pd.Series(results[strat]["values"]).sort_index()
        results[strat]["values"] = s

    return results, bench


# ── Metrics & reporting ───────────────────────────────────────────────────────

def _days_str(days: int) -> str:
    y, r = divmod(days, 365)
    m = r // 30
    if y and m:
        return f"{y}y {m}m"
    if y:
        return f"{y}y"
    if m:
        return f"{m}m"
    return f"{days}d"


def compute_metrics(values: pd.Series, trades: list, open_trade) -> dict:
    v = values.dropna()
    if v.empty:
        return {}
    years  = (v.index[-1] - v.index[0]).days / 365.25
    tr     = v.iloc[-1] / v.iloc[0] - 1
    cagr   = (v.iloc[-1] / v.iloc[0]) ** (1 / max(years, 0.01)) - 1
    mdd    = ((v - v.cummax()) / v.cummax()).min()
    dr     = v.pct_change().dropna()
    sh     = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    all_t  = trades + ([open_trade] if open_trade else [])
    n      = len(all_t)
    wins   = sum(1 for t in all_t if t and t["return_pct"] > 0)
    closed = [t for t in all_t if t and not t.get("is_open")]
    in_days = sum(
        (t["exit_date"] - t["entry_date"]).days
        for t in closed
        if t.get("entry_date") and t.get("exit_date")
    )
    tot_days = (v.index[-1] - v.index[0]).days

    return {
        "Total Return":  f"{tr:+.1%}",
        "CAGR":          f"{cagr:+.1%}",
        "Max Drawdown":  f"{mdd:.1%}",
        "Sharpe Ratio":  f"{sh:.2f}",
        "Final Value":   f"${v.iloc[-1]:>12,.0f}",
        "# Trades":      str(n),
        "Win Rate":      f"{wins/n:.1%}" if n else "—",
        "Time in Mkt":   f"{in_days/tot_days:.1%}" if tot_days else "—",
    }


def print_metrics(results: dict, bench_metrics: dict) -> None:
    labels   = ["Strategy A", "Strategy B", "Strategy C", "Buy & Hold SPX"]
    col_data = [
        compute_metrics(results["A"]["values"], results["A"]["trades"], results["A"]["open_trade"]),
        compute_metrics(results["B"]["values"], results["B"]["trades"], results["B"]["open_trade"]),
        compute_metrics(results["C"]["values"], results["C"]["trades"], results["C"]["open_trade"]),
        bench_metrics,
    ]
    keys = list(dict.fromkeys(k for d in col_data for k in d))
    W    = 16
    hdr  = f"\n  {'Metric':<22}" + "".join(f"{l:>{W}}" for l in labels)
    sep  = "=" * len(hdr.lstrip("\n"))
    print(f"\n{sep}\n{hdr}\n{sep}")
    for k in keys:
        row = f"  {k:<22}" + "".join(f"{d.get(k,'—'):>{W}}" for d in col_data)
        print(row)
    print(sep)


def print_trades(results: dict) -> None:
    hdr = (f"\n  {'Strat':>6}  {'Year':>4}  {'Entry':10}  {'Exit':10}  "
           f"{'Held':>7}  {'Return':>8}  {'In $':>13}  {'Out $':>13}  "
           f"{'Entry cond':>18}  Exit cond")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_trades: list = []
    for strat in ("A", "B", "C"):
        for t in results[strat]["trades"]:
            all_trades.append(t)
        if results[strat]["open_trade"]:
            all_trades.append(results[strat]["open_trade"])

    all_trades.sort(key=lambda t: (t.get("year", 0), t["strat"],
                                   t.get("entry_date") or pd.Timestamp("1900-01-01")))

    for t in all_trades:
        if not t.get("entry_date"):
            ed, xd, held = "—", "—", "—"
        else:
            ed   = t["entry_date"].strftime("%Y-%m-%d")
            xd   = (t["exit_date"].strftime("%Y-%m-%d") if t.get("exit_date")
                    else "(open)")
            held = (_days_str((t["exit_date"] - t["entry_date"]).days)
                    if t.get("exit_date") else "—")
        yr = t.get("year", "?")
        print(
            f"  {t['strat']:>6}  {yr:>4}  {ed:10}  {xd:10}  {held:>7}  "
            f"{t['return_pct']:>+7.1f}%  ${t['entry_cap']:>12,.0f}  "
            f"${t['exit_cap']:>12,.0f}  {t.get('entry_note',''):>18}  "
            f"{t.get('exit_note','')}"
        )


def print_per_year_summary(results: dict, sa_data: dict) -> None:
    print("\n── Per-year returns ──\n")
    hdr = f"  {'Year':>4}  {'SA picks':45}  {'  A':>8}  {'  B':>8}  {'  C':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    all_trades = {s: {t.get("year"): t
                      for t in results[s]["trades"] + (
                          [results[s]["open_trade"]] if results[s]["open_trade"] else [])}
                  for s in ("A", "B", "C")}

    for year in sorted(sa_data.keys()):
        stocks = sa_data[year].get("stocks", [])
        tickers = ", ".join(s["ticker"] for s in stocks)
        tickers_str = tickers[:45].ljust(45) if tickers else "—" * 45

        row_vals = []
        for s in ("A", "B", "C"):
            t = all_trades[s].get(year)
            if t and t.get("entry_date"):
                row_vals.append(f"{t['return_pct']:>+7.1f}%")
            elif t:
                row_vals.append("  (skip)")
            else:
                row_vals.append("      —")

        print(f"  {year:>4}  {tickers_str}  {'  '.join(row_vals)}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, bench: pd.Series, sa_data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11),
                                   gridspec_kw={"height_ratios": [3, 1]})

    colors = {"A": "#2196F3", "B": "#4CAF50", "C": "#FF5722"}
    labels = {
        "A": "A) Buy Jan 1 / sell Dec 31",
        "B": f"B) Buy PE<{FWD_PE_BUY:.0f} / sell Dec 31",
        "C": (f"C) Buy PE<{FWD_PE_BUY:.0f} or VIX≥{VIX_ALT_THRESH:.0f}+B≤{BREADTH_ALT_THRESH:.0f} / "
              f"bearish-div or stop exit"),
    }

    ax1.plot(bench.index, bench, color="#90A4AE", linewidth=1.2,
             linestyle="--", label="SPX buy & hold", alpha=0.7)
    for s in ("A", "B", "C"):
        v = results[s]["values"]
        ax1.plot(v.index, v, color=colors[s], linewidth=1.8, label=labels[s])

    # Mark buy/sell for Strategy C (most interesting)
    def plot_markers(trades, open_trade):
        entries = [t["entry_date"] for t in trades if t.get("entry_date")]
        exits   = [t["exit_date"]  for t in trades if t.get("exit_date")]
        early   = [t["exit_date"]  for t in trades
                   if t.get("exit_date") and t.get("exit_note", "") not in ("year-end", "—")]
        if open_trade and open_trade.get("entry_date"):
            entries.append(open_trade["entry_date"])
        v = results["C"]["values"]
        if entries:
            ax1.scatter(entries, v.reindex(entries, method="nearest"),
                        marker="^", color="lime", s=90, zorder=6, label="C: buy")
        if exits:
            ax1.scatter(exits, v.reindex(exits, method="nearest"),
                        marker="v", color="red", s=90, zorder=6, label="C: sell")

    plot_markers(results["C"]["trades"], results["C"]["open_trade"])

    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.set_title(
        "Seeking Alpha Annual Picks — Strategy Comparison\n"
        f"Entry: A=Jan 1  |  B=PE<{FWD_PE_BUY:.0f}  |  "
        f"C=PE<{FWD_PE_BUY:.0f} OR (VIX≥{VIX_ALT_THRESH:.0f} AND breadth≤{BREADTH_ALT_THRESH:.0f})\n"
        f"Exit C: SPX bearish-div (SPX+{DIV_PRICE_RISE:.0f}%/{DIV_WINDOW}d + breadth-{DIV_BREADTH_FALL:.0f}pt) "
        f"OR trailing-stop(-{TRAILING_STOP_PCT:.0f}%) OR year-end\n"
        f"Starting capital: ${INITIAL_CAPITAL:,.0f}  |  Equal-weight 10 stocks/year  |  SPX proxy for intra-year exits",
        fontsize=8.5, fontweight="bold",
    )
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: annual return bars ──────────────────────────────────────────
    years = sorted(sa_data.keys())
    x = np.arange(len(years))
    width = 0.27

    all_t = {s: {t.get("year"): t
                 for t in results[s]["trades"] + (
                     [results[s]["open_trade"]] if results[s]["open_trade"] else [])}
             for s in ("A", "B", "C")}

    for i, s in enumerate(("A", "B", "C")):
        rets = [all_t[s].get(y, {}).get("return_pct", 0.0) for y in years]
        bars = ax2.bar(x + (i - 1) * width, rets, width,
                       color=colors[s], alpha=0.8, label=f"Strat {s}")
        for bar, ret in zip(bars, rets):
            if abs(ret) > 0.5:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                         f"{ret:+.0f}%", ha="center", va="bottom", fontsize=6.5)

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annual Return (%)")
    ax2.set_title("Annual Returns by Strategy (note: C early exits use SPX proxy)", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    out = DATA_DIR / "seeking_alpha_performance.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading market data...")
    market = load_market_data()

    print("Parsing Seeking Alpha picks...")
    sa_data = parse_sa_csv()

    years = sorted(sa_data.keys())
    print(f"Years in CSV : {years[0]}–{years[-1]}")
    for y in years:
        n = len(sa_data[y].get("stocks", []))
        right = sa_data[y]["right_start"].strftime("%Y-%m-%d")
        print(f"  {y}: {n} picks  |  signal entry date: {right}")

    print("\nRunning strategies A / B / C...")
    results, bench = run_strategies(sa_data, market)

    bench_m = compute_metrics(bench[bench.index >= pd.Timestamp("2022-01-01")], [], None)

    print("\n" + "─" * 80)
    print("STRATEGY PARAMETERS")
    print(f"  A  Buy Jan 1, sell Dec 31 every year")
    print(f"  B  Buy when S&P 500 fwd PE < {FWD_PE_BUY:.1f},  sell Dec 31")
    print(f"  C  Buy when PE < {FWD_PE_BUY:.1f}  OR  (VIX ≥ {VIX_ALT_THRESH:.1f} AND breadth ≤ {BREADTH_ALT_THRESH:.1f})")
    print(f"     Exit: bearish-div (SPX +{DIV_PRICE_RISE:.1f}%/{DIV_WINDOW}d + breadth -{DIV_BREADTH_FALL:.1f}pt + breadth<{DIV_BREADTH_CAP:.1f}%)")
    print(f"           OR trailing-stop -{TRAILING_STOP_PCT:.1f}% from SPX peak   OR   year-end")
    print(f"  Initial capital: ${INITIAL_CAPITAL:,.0f}")
    print("─" * 80)

    print_metrics(results, bench_m)
    print_per_year_summary(results, sa_data)
    print("\n── Trade log ──")
    print_trades(results)
    plot_results(results, bench, sa_data)


if __name__ == "__main__":
    main()
