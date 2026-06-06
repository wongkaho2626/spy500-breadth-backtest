#!/usr/bin/env python3
"""
S&P 500 broad backtest — SA Strategy C timing applied to every constituent.

Entry  (while OUT): fwd PE < 20  OR  (VIX >= 22 AND breadth <= 50)
Exit   (while IN) : SPX bearish-divergence  OR  25% trailing stop on stock price
Universe          : current S&P 500 list (Wikipedia), last 10 years of price history

REQUIREMENTS (run locally):
    pip install yfinance pandas numpy matplotlib
    python sp500_strategy_c_backtest.py

NOTE: yfinance and Wikipedia require outbound internet access.
      If run in a restricted environment without internet, the script will
      automatically fall back to SPX-proxy mode (one representative stock
      showing the pure timing-signal effect on the broad market).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent

# ── Parameters (SA Strategy C) ───────────────────────────────────────────────
START_DATE         = "2016-01-01"
END_DATE           = "2026-06-06"
FWD_PE_BUY         = 20.0
VIX_ALT_THRESH     = 22.0
BREADTH_ALT_THRESH = 50.0
DIV_WINDOW         = 60
DIV_PRICE_RISE     = 5.0
DIV_BREADTH_FALL   = 20.0
DIV_BREADTH_CAP    = 60.0
TRAILING_STOP_PCT  = 25.0
MIN_HISTORY_DAYS   = 504   # ~2 years minimum


# ── Load market indicators ────────────────────────────────────────────────────

def load_market_data() -> pd.DataFrame:
    def read_inv(path, col="Price"):
        df = pd.read_csv(path)
        df.columns = [c.strip().strip('"').lstrip("﻿") for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df.set_index("Date", inplace=True)
        df["val"] = df[col].astype(str).str.replace(",", "").astype(float)
        return df[["val"]]

    # Use local SPX.csv first; fall back to yfinance if not available
    spx_local = DATA_DIR / "SPX.csv"
    if spx_local.exists():
        spx_df = read_inv(spx_local).rename(columns={"val": "spx"})
        spx = spx_df["spx"]
    else:
        raw = yf.download("^GSPC", start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)["Close"]
        spx = raw.rename("spx")
        if hasattr(spx.index, "tz_localize"):
            spx.index = spx.index.tz_localize(None)

    fpe = pd.read_csv(DATA_DIR / "S&P500ForwardPE.csv")
    fpe["Date"] = pd.to_datetime(fpe["date"])
    fpe.set_index("Date", inplace=True)
    fpe = fpe.rename(columns={"forward_pe": "fwd_pe"})[["fwd_pe"]]

    brd = read_inv(DATA_DIR / "S5TH.csv").rename(columns={"val": "breadth"})
    vix_csv = read_inv(DATA_DIR / "VIX.csv").rename(columns={"val": "vix"})

    df = pd.DataFrame(spx).join(fpe, how="left").join(brd, how="left").join(vix_csv, how="left")
    df.sort_index(inplace=True)
    df = df[df.index >= pd.Timestamp(START_DATE)]

    df["fwd_pe"]  = df["fwd_pe"].ffill()
    df["breadth"] = df["breadth"].ffill()
    df["vix"]     = df["vix"].ffill()

    # Pre-compute entry signal
    df["entry_signal"] = (
        (df["fwd_pe"] < FWD_PE_BUY) |
        ((df["vix"] >= VIX_ALT_THRESH) & (df["breadth"] <= BREADTH_ALT_THRESH))
    )

    # Pre-compute SPX bearish-divergence
    spx_past = df["spx"].shift(DIV_WINDOW)
    brd_past = df["breadth"].shift(DIV_WINDOW)
    df["div_ok"] = (
        ((df["spx"] - spx_past) / spx_past * 100 >= DIV_PRICE_RISE) &
        ((brd_past - df["breadth"]) >= DIV_BREADTH_FALL) &
        (df["breadth"] < DIV_BREADTH_CAP)
    ).fillna(False)

    return df.dropna(subset=["spx"])


# ── Get S&P 500 tickers ───────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia, with GitHub raw fallback."""
    import urllib.request, io

    # Try Wikipedia first
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        print(f"  Found {len(tickers)} tickers (Wikipedia)")
        return tickers
    except Exception:
        pass

    # GitHub raw fallback (no auth required)
    try:
        url = ("https://raw.githubusercontent.com/datasets/"
               "s-and-p-500-companies/main/data/constituents.csv")
        data = urllib.request.urlopen(url, timeout=10).read().decode()
        df = pd.read_csv(io.StringIO(data))
        tickers = [t.replace(".", "-") for t in df["Symbol"].tolist()]
        print(f"  Found {len(tickers)} tickers (GitHub fallback)")
        return tickers
    except Exception as e:
        raise RuntimeError(f"Cannot fetch S&P 500 ticker list: {e}")


# ── Download price data in batches ────────────────────────────────────────────

def download_prices(tickers: list[str], batch_size: int = 100) -> pd.DataFrame:
    print(f"Downloading {len(tickers)} stocks ({START_DATE} → {END_DATE}) …")
    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        raw = yf.download(
            batch, start=START_DATE, end=END_DATE,
            auto_adjust=True, progress=False, threads=True
        )
        closes = raw["Close"] if "Close" in raw.columns else raw
        closes.index = closes.index.tz_localize(None)
        frames.append(closes)
        print(f"  {min(i + batch_size, len(tickers))}/{len(tickers)}", end="\r")
    print()
    return pd.concat(frames, axis=1)


# ── Run strategy on a single stock ───────────────────────────────────────────

def run_stock_strategy(prices: pd.Series, market: pd.DataFrame) -> dict:
    """Apply SA Strategy C to one stock price series. Returns metrics dict."""
    # Align stock prices with market dates
    s = prices.reindex(market.index).ffill()
    valid = s.notna()
    if valid.sum() < MIN_HISTORY_DAYS:
        return None

    position   = "OUT"
    entry_px   = 0.0
    peak_px    = 0.0
    entry_date = None
    port       = 1.0    # normalised to 1.0
    values     = {}
    trades     = []

    for date in market.index:
        if not valid[date]:
            if position == "IN":
                values[date] = port * (s[date] / entry_px) if pd.notna(s[date]) else values.get(date, port)
            else:
                values[date] = port
            continue

        px  = float(s[date])
        sig = bool(market.loc[date, "entry_signal"])
        div = bool(market.loc[date, "div_ok"])

        if position == "OUT":
            if sig:
                entry_px   = px
                peak_px    = px
                entry_date = date
                position   = "IN"
            values[date] = port

        else:  # IN
            peak_px = max(peak_px, px)

            trailing_stop = px < peak_px * (1.0 - TRAILING_STOP_PCT / 100.0)
            sell = trailing_stop or div

            if sell:
                ret = px / entry_px - 1
                port *= (1 + ret)
                trades.append({
                    "entry_date": entry_date,
                    "exit_date":  date,
                    "return_pct": ret * 100,
                    "held_days":  (date - entry_date).days,
                    "sell_reason": "trailing-stop" if trailing_stop else "bearish-div",
                })
                values[date] = port
                position = "OUT"
            else:
                values[date] = port * (px / entry_px)

    # Open trade counts as current value
    if position == "IN":
        last_px = s[s.notna()].iloc[-1]
        ret = last_px / entry_px - 1
        trades.append({
            "entry_date": entry_date,
            "exit_date":  market.index[-1],
            "return_pct": ret * 100,
            "held_days":  (market.index[-1] - entry_date).days,
            "sell_reason": "open",
        })

    v = pd.Series(values).dropna()
    if len(v) < 2:
        return None

    bh = s.reindex(v.index).ffill()
    bh = bh / bh.iloc[0]

    years = (v.index[-1] - v.index[0]).days / 365.25
    tr    = v.iloc[-1] / v.iloc[0] - 1
    cagr  = (v.iloc[-1] / v.iloc[0]) ** (1 / max(years, 0.1)) - 1
    mdd   = ((v - v.cummax()) / v.cummax()).min()
    dr    = v.pct_change().dropna()
    sh    = (dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

    bh_tr   = bh.iloc[-1] - 1
    bh_cagr = (bh.iloc[-1]) ** (1 / max(years, 0.1)) - 1
    bh_mdd  = ((bh - bh.cummax()) / bh.cummax()).min()
    bh_dr   = bh.pct_change().dropna()
    bh_sh   = (bh_dr.mean() / bh_dr.std() * np.sqrt(252)) if bh_dr.std() > 0 else 0.0

    n      = len(trades)
    wins   = sum(1 for t in trades if t["return_pct"] > 0)
    in_d   = sum(t["held_days"] for t in trades)
    tot_d  = (v.index[-1] - v.index[0]).days

    return {
        "total_return":    tr,
        "cagr":            cagr,
        "max_drawdown":    mdd,
        "sharpe":          sh,
        "n_trades":        n,
        "win_rate":        wins / n if n else 0.0,
        "time_in_mkt":     in_d / tot_d if tot_d else 0.0,
        "bh_total_return": bh_tr,
        "bh_cagr":         bh_cagr,
        "bh_max_drawdown": bh_mdd,
        "bh_sharpe":       bh_sh,
        "history_days":    valid.sum(),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def run_spx_proxy_mode(market: pd.DataFrame) -> None:
    """
    Fallback when yfinance is unavailable.
    Runs SA Strategy C on SPX itself as a single broad-market stock,
    showing the pure timing-signal effect without individual stock selection.
    """
    print("\n[PROXY MODE] yfinance unavailable — running on SPX index as single instrument.\n")
    spx_prices = market["spx"].rename("SPX")
    r = run_stock_strategy(spx_prices, market)
    if not r:
        print("Insufficient data for SPX proxy run.")
        return

    print(f"  {'Metric':<28} {'SA-C → SPX':>12} {'SPX Buy&Hold':>13}")
    print("  " + "=" * 55)
    rows = [
        ("Total Return",         f"{r['total_return']:>+11.1%}",  f"{r['bh_total_return']:>+12.1%}"),
        ("CAGR",                 f"{r['cagr']:>+11.1%}",          f"{r['bh_cagr']:>+12.1%}"),
        ("Max Drawdown",         f"{r['max_drawdown']:>11.1%}",   f"{r['bh_max_drawdown']:>12.1%}"),
        ("Sharpe Ratio",         f"{r['sharpe']:>11.2f}",         f"{r['bh_sharpe']:>12.2f}"),
        ("# Trades",             f"{r['n_trades']:>11}",          "—"),
        ("Win Rate",             f"{r['win_rate']:>11.1%}",       "—"),
        ("Time in Market",       f"{r['time_in_mkt']:>11.1%}",   "—"),
    ]
    for label, sv, bv in rows:
        print(f"  {label:<28} {sv}  {bv}")
    print("  " + "=" * 55)
    print("\n  NOTE: Full S&P 500 backtest requires internet access (yfinance).")
    print("        Run locally with: python sp500_strategy_c_backtest.py")


def main():
    print("Loading market indicators...")
    market = load_market_data()
    market = market[market.index >= pd.Timestamp(START_DATE)]
    print(f"  Market data: {market.index[0].date()} → {market.index[-1].date()}")

    entry_days = market["entry_signal"].sum()
    signal_pct = entry_days / len(market) * 100
    print(f"  Entry signal active {entry_days} of {len(market)} days ({signal_pct:.1f}%)")
    div_days = market["div_ok"].sum()
    print(f"  Bearish-div signal active {div_days} days")

    # Try to get tickers and prices; fall back to SPX proxy if network unavailable
    try:
        tickers = get_sp500_tickers()
    except Exception as e:
        print(f"\n  Cannot fetch tickers ({e})")
        run_spx_proxy_mode(market)
        return

    try:
        prices = download_prices(tickers)
    except Exception as e:
        print(f"\n  Cannot download prices ({e})")
        run_spx_proxy_mode(market)
        return

    # Also detect silent failures (all-NaN downloads)
    valid_cols = prices.columns[prices.notna().sum() > MIN_HISTORY_DAYS]
    if len(valid_cols) < 10:
        print(f"\n  Only {len(valid_cols)} stocks downloaded — likely network blocked.")
        run_spx_proxy_mode(market)
        return
    prices = prices[valid_cols]

    print(f"\nRunning SA Strategy C on {prices.shape[1]} stocks...")
    results = {}
    for i, ticker in enumerate(prices.columns):
        r = run_stock_strategy(prices[ticker], market)
        if r:
            results[ticker] = r
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(prices.columns)} processed ({len(results)} valid)")

    print(f"\n  {len(results)} stocks with sufficient history")

    # ── Save per-stock results ────────────────────────────────────────────────
    df_res = pd.DataFrame(results).T
    df_res.index.name = "ticker"
    out_csv = DATA_DIR / "sp500_strategy_c_results.csv"
    df_res.to_csv(out_csv)
    print(f"  Per-stock results saved → {out_csv}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  SA Strategy C applied to S&P 500 — Aggregate Results")
    print(f"  Universe: {len(df_res)} stocks  |  Period: {START_DATE} → {END_DATE}")
    print(f"  Entry: PE<{FWD_PE_BUY:.0f} OR (VIX≥{VIX_ALT_THRESH:.0f} AND breadth≤{BREADTH_ALT_THRESH:.0f})")
    print(f"  Exit:  SPX bearish-div OR {TRAILING_STOP_PCT:.0f}% trailing stop")
    print("=" * 68)

    metrics_cols = ["total_return","cagr","max_drawdown","sharpe",
                    "n_trades","win_rate","time_in_mkt",
                    "bh_total_return","bh_cagr","bh_max_drawdown","bh_sharpe"]
    labels = {
        "total_return":    "Total Return",
        "cagr":            "CAGR",
        "max_drawdown":    "Max Drawdown",
        "sharpe":          "Sharpe Ratio",
        "n_trades":        "Avg # Trades",
        "win_rate":        "Win Rate",
        "time_in_mkt":     "Time in Market",
        "bh_total_return": "Buy&Hold Total Return",
        "bh_cagr":         "Buy&Hold CAGR",
        "bh_max_drawdown": "Buy&Hold Max DD",
        "bh_sharpe":       "Buy&Hold Sharpe",
    }
    fmt = {
        "total_return": "{:+.1%}", "cagr": "{:+.1%}", "max_drawdown": "{:.1%}",
        "sharpe": "{:.2f}", "n_trades": "{:.1f}", "win_rate": "{:.1%}",
        "time_in_mkt": "{:.1%}",
        "bh_total_return": "{:+.1%}", "bh_cagr": "{:+.1%}",
        "bh_max_drawdown": "{:.1%}", "bh_sharpe": "{:.2f}",
    }

    print(f"\n  {'Metric':<28} {'Mean':>10} {'Median':>10} {'10th %ile':>10} {'90th %ile':>10}")
    print("  " + "-" * 62)
    sep_printed = False
    for col in metrics_cols:
        if col == "bh_total_return" and not sep_printed:
            print("  " + "-" * 62)
            sep_printed = True
        s = df_res[col]
        f = fmt[col]
        print(f"  {labels[col]:<28} "
              f"{f.format(s.mean()):>10} "
              f"{f.format(s.median()):>10} "
              f"{f.format(s.quantile(0.10)):>10} "
              f"{f.format(s.quantile(0.90)):>10}")

    # ── Winners / losers ──────────────────────────────────────────────────────
    beat_bh = (df_res["total_return"] > df_res["bh_total_return"]).sum()
    pct_beat = beat_bh / len(df_res)
    print(f"\n  Stocks where strategy beat buy&hold: {beat_bh}/{len(df_res)} ({pct_beat:.1%})")

    better_dd = (df_res["max_drawdown"].abs() < df_res["bh_max_drawdown"].abs()).sum()
    print(f"  Stocks with lower drawdown vs B&H:   {better_dd}/{len(df_res)} ({better_dd/len(df_res):.1%})")

    better_sh = (df_res["sharpe"] > df_res["bh_sharpe"]).sum()
    print(f"  Stocks with higher Sharpe vs B&H:    {better_sh}/{len(df_res)} ({better_sh/len(df_res):.1%})")

    # ── Top / bottom performers ───────────────────────────────────────────────
    top10 = df_res.nlargest(10, "total_return")
    bot10 = df_res.nsmallest(10, "total_return")

    print(f"\n── Top 10 stocks (strategy total return) ──")
    print(f"  {'Ticker':8} {'Strat Ret':>10} {'B&H Ret':>10} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print("  " + "-" * 60)
    for tk, row in top10.iterrows():
        print(f"  {tk:8} {row.total_return:>+9.1%} {row.bh_total_return:>+9.1%} "
              f"{row.cagr:>+7.1%} {row.sharpe:>8.2f} {row.max_drawdown:>7.1%} {int(row.n_trades):>7}")

    print(f"\n── Bottom 10 stocks (strategy total return) ──")
    print(f"  {'Ticker':8} {'Strat Ret':>10} {'B&H Ret':>10} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
    print("  " + "-" * 60)
    for tk, row in bot10.iterrows():
        print(f"  {tk:8} {row.total_return:>+9.1%} {row.bh_total_return:>+9.1%} "
              f"{row.cagr:>+7.1%} {row.sharpe:>8.2f} {row.max_drawdown:>7.1%} {int(row.n_trades):>7}")

    # ── Plot distribution ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"SA Strategy C → S&P 500 Stocks ({START_DATE[:4]}–{END_DATE[:4]})\n"
        f"Entry: PE<{FWD_PE_BUY:.0f} OR (VIX≥{VIX_ALT_THRESH:.0f} AND breadth≤{BREADTH_ALT_THRESH:.0f})  |  "
        f"Exit: bearish-div OR {TRAILING_STOP_PCT:.0f}% trailing stop  |  N={len(df_res)} stocks",
        fontsize=10, fontweight="bold"
    )

    excess = df_res["total_return"] - df_res["bh_total_return"]

    ax = axes[0, 0]
    ax.hist(df_res["total_return"] * 100, bins=40, color="#2196F3", alpha=0.7, label="Strategy")
    ax.hist(df_res["bh_total_return"] * 100, bins=40, color="#FF9800", alpha=0.5, label="Buy & Hold")
    ax.axvline(df_res["total_return"].mean() * 100, color="blue", linestyle="--", linewidth=1.5)
    ax.axvline(df_res["bh_total_return"].mean() * 100, color="orange", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Total Return (%)")
    ax.set_title("Total Return Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.hist(excess * 100, bins=40, color="#4CAF50", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.axvline(excess.mean() * 100, color="green", linestyle="--", linewidth=1.5,
               label=f"Mean: {excess.mean():+.1%}")
    ax.set_xlabel("Strategy − Buy&Hold Return (%)")
    ax.set_title("Excess Return vs Buy & Hold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.hist(df_res["sharpe"], bins=40, color="#9C27B0", alpha=0.7, label="Strategy")
    ax.hist(df_res["bh_sharpe"], bins=40, color="#FF9800", alpha=0.5, label="Buy & Hold")
    ax.axvline(df_res["sharpe"].mean(), color="purple", linestyle="--", linewidth=1.5)
    ax.axvline(df_res["bh_sharpe"].mean(), color="orange", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.scatter(df_res["bh_total_return"] * 100, df_res["total_return"] * 100,
               alpha=0.4, s=12, color="#607D8B")
    lim = max(abs(df_res[["total_return","bh_total_return"]].values.max()),
              abs(df_res[["total_return","bh_total_return"]].values.min())) * 100 + 50
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.0, label="y=x (equal)")
    ax.set_xlabel("Buy & Hold Total Return (%)")
    ax.set_ylabel("Strategy Total Return (%)")
    ax.set_title("Strategy vs Buy & Hold (per stock)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = DATA_DIR / "sp500_strategy_c_results.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out_png}")


if __name__ == "__main__":
    main()
