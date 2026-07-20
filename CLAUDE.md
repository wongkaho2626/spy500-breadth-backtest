# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A market-timing **backtesting playground** for NASDAQ-heavy portfolios (QQQ, TQQQ, the #1
NASDAQ-100 stock, SPY, SOXX). Every strategy is driven by the same core signal: the **S&P 500
breadth indicator** — the percentage of S&P 500 stocks trading above their 200-day moving
average. The repo ships both a Python CLI (many standalone scripts) and an interactive static
web app (`webapp/nextjs/`) that runs the same logic in the browser and deploys to GitHub Pages.

> **Repo name is historical.** It began as an S&P 500 breadth study (hence
> `spy500-breadth-backtest`) but has since centred on QQQ / NASDAQ-100. The README titles it
> "QQQ Portfolio Backtest."

## Environment & dependencies

No `requirements.txt`, no package structure, no test suite. Scripts are standalone and read CSVs
from the repo root. Install the third-party libraries directly:

```bash
pip install pandas numpy matplotlib scipy yfinance playwright
```

- **pandas / numpy** — used by nearly every script
- **matplotlib** — chart PNGs (saved next to the script; `.png` is gitignored except `docs/*.png`)
- **scipy** — statistics in the validation/research scripts (`qqq_validation.py`, `qqq_walkforward.py`, etc.)
- **yfinance** — live price fetches for TQQQ and NDX top-stock CSVs (`tqqq_backtest.py`, `qqq_portfolio_backtest.py`, `ndx_top1_backtest.py`, `combined_t5n25q70.py`, `sa_qqq_backtest.py`)
- **playwright** — Cloudflare-bypassing scraper in `fetch_investing_data.py` (`playwright install` needed once)

## The core strategy (shared signal logic)

All the "backtest" scripts implement the same signal state machine — a `position` that is `"IN"`
or `"OUT"`, driven by these rules (see `qqq_backtest.py` for the canonical, best-documented copy):

**BUY (while OUT)** — two entry paths:
- **Washout**: breadth < 26% AND a vote gate passes — either VIX > 30 (fear spike) OR price > its
  own 200-day MA (an uptrend pullback, safe to buy immediately). The vote gate avoids buying into
  a structurally broken downtrend.
- **Trend re-entry** (newest addition): price closes back above its MA200 (a fresh cross),
  allowed only when EITHER the previous exit was a "climax top" (a premature froth shakeout) OR
  price is back above the level we last sold at (the market proved the exit premature). Recrosses
  still below the prior exit stay filtered out — that rejects failed bounces in a real downtrend
  (e.g. the 2022 whipsaws) while catching real recoveries (e.g. after the false 2004 divergence).
- A **cooldown** (default 15–30 days, configurable) must have elapsed since the last exit.

> **Tested & rejected — mid-trend "dip" entry (2026-07).** A third entry path was tried: while
> FLAT in a healthy uptrend (price > MA200, breadth ≥ 26), enter on an oversold dip — RSI(14) < 35
> **or** price ≥ 10% below its 126-day high. The motive was to give a flat position a way in
> *without* waiting for a rare breadth-<26 washout. In an **isolated** forward-return scan those
> dips look good (RSI<35: +10.3% fwd-6m, better ret/drawdown than "buy any day"). But **added to
> the full strategy it hurt across the board**: Sharpe 1.12 → 0.99, CAGR 20.5% → 18.7%, MaxDD
> −32.2% → −34.8%, and it added only 2 net trades (17 → 19, still < 30). Mechanism: the dip entry
> **front-runs the washout** — it buys oversold dips *earlier and higher* that then deepen into the
> very crash the washout was built to bottom-tick (e.g. a 2007-11 RSI dip entry rode the GFC to
> −23%; the 2025 dip entry front-ran the April washout and ate a −17% drawdown vs the washout
> entry's 0%). Lesson: **a signal with standalone forward-return edge is not additive** — this
> strategy's edge *is* its patience (only buying washouts), and any "get in sooner" path destroys
> the downside-avoidance that creates the edge. Don't re-try mid-trend entries without a
> fundamentally different gate. (Scan + patched backtest were run in a scratchpad; canonical
> `qqq_backtest.py` was left unchanged.)

> **Tested & rejected — bullish-divergence entry (2026-07).** The "fundamentally different gate"
> the note above asks for was then tried: the **mirror of the bearish-divergence exit** — price
> *fell* ≥ X% over 60 days while breadth *rose* ≥ Y pts (internals healing before price confirms).
> Breadth-based, not price-oversold, so it is a genuinely different mechanism. Two findings:
>
> 1. **The strict mirror barely exists, and for SPX it is structurally impossible.** At the exact
>    mirror thresholds (−3% price / +20 pts breadth) SPX fires **0 times in 24 years** (even
>    −1%/+20pts fires once). S&P breadth *is* the S&P's own internals: if 20% more S&P stocks
>    reclaim their 200-day MA, the index cannot be falling. The asymmetry is real — a *narrow
>    rally* (mega-caps carrying a declining majority) is common, so bearish divergence works; the
>    reverse is not. NDX fires more often (14 days at −3%/+20pts) **only because NDX is a different
>    universe from S&P breadth** — "NDX down + S&P breadth up" is tech-out-of-favour *rotation*,
>    not NDX bullishness.
> 2. **As a mid-trend entry it does not exist; as a crash-recovery entry it is net-negative.**
>    Mid-trend (price > MA200): 3 episodes in 24 years, 2 of which the strategy was already IN →
>    **one usable signal ever**, and mid-trend instances *underperformed* (+1.2% fwd-6m vs +8.2%
>    baseline). Gated to below-MA200 (crash regime, price −3% / breadth +5 pts) it fires 3 times
>    and gives Sharpe 1.12 → 1.09, CAGR 20.5% → 20.1%, MaxDD unchanged at −32.2%, 17 → 18 trades.
>    Right **2 of 3** (2004-09 entry +43.4% vs baseline's later +33.4%; 2022-12 +12.2%) but the
>    miss (2022-04, −15.4%) front-ran the June washout and cost more than both wins gained.
>    Mechanism: "internals healing" looks identical at a **real bottom** (2004-09, 2022-12) and at
>    a **bear-market rally** (2022-04) — the signal cannot separate them.
>
> **⚠️ Overfitting trap — do not "fix" it this way.** A breadth floor of ~50 would exclude the one
> loser (breadth 44.5) while keeping both winners (59.6, 52.7), turning this into 2 wins / 0 losses.
> That is tuning a threshold on **n=3** to draw a circle around the single bad trade — exactly the
> curve-fitting the Deflated Sharpe penalty exists to punish. It would look better and *be* worse.
>
> Verdict: rejected, but it is the strongest of the alternative-entry attempts (sound mechanism,
> drawdown-neutral). The pattern across every attempt is now unambiguous: **any path that enters
> earlier than washout/MA200 confirmation gives back more in the bad cases than it gains in the
> good ones.** The waiting is not a bug — it is the product. (Scans + patched backtest ran in a
> scratchpad; canonical `qqq_backtest.py` unchanged.)

**SELL (while IN)** — any of:
- **Bearish divergence**: price rose ≥ 3% over a 60-day window while breadth fell ≥ 20 pts AND
  breadth is currently < 60%. Flags a narrowing rally carried by a shrinking set of leaders.
- **Climax top**: within 10 days, price was extended ≥ 5% above its 10-day MA AND MACD(12,26,9)
  flipped bearish.
- **Trailing stop**: price 25% below the high since entry.

**Costs**: `$1` flat commission + `0.05%` slippage per side, applied to the effective entry/exit
price (in all the current scripts; the older no-cost variants have been removed).

**Execution timing**: signals come from end-of-day closes, so they are only known after the
session closes. All trading backtests therefore fill orders at the **next trading day's open**
by default (`EXECUTION_LAG = 1`, `FILL_PRICE = "open"` — constants near the top of each script).
The legacy same-day-close fill (a look-ahead: it trades at the very close that produced the
signal) stays available via `EXECUTION_LAG = 0` / `FILL_PRICE = "close"`, or `--fill same-close`
on `qqq_backtest.py` / `qqq_portfolio_backtest.py` (choices: `next-open` | `next-close` |
`same-close`). Signals and mark-to-market always stay on closes; only the transaction
price/timing changes. Measured lag cost on `qqq_backtest.py` (2002–2026): next-open ≈ −0.3 pts
CAGR vs same-close (Sharpe unchanged 1.12); next-close ≈ −1.8 pts. Legs without an Open column
in their data (e.g. the web app's TQQQ/SPY/SOXX CSVs) fall back to the fill-day close.

The signal is computed on a "signal index" (NDX for the QQQ family, SPX for SPY, SOXX for the
semiconductor variant) and then applied to whatever asset(s) the script actually trades.

## Data pipeline — the breadth series

The breadth input has a historical trap worth knowing:

- **`S5TH.csv`** (% of S&P 500 above 200-day MA) is only *daily from 2007*. Before that it is
  bimonthly, which silently corrupts any row-based lookback window (a "60-day" window would span
  ~10 years of sparse points).
- **`build_breadth_daily.py`** fixes this: it fits a linear map `S5TH ≈ a + b·MMTH` on the 2007+
  overlap (MMTH is a broader-universe daily series back to 2002, ~0.94 correlated), applies it to
  2002–2006, and splices the two into **`breadth_daily.csv`** (columns: `Date`, `breadth`,
  `source`). Newer scripts read `breadth_daily.csv` for a clean continuous 2002+ series; older
  ones read `S5TH.csv` directly. **Rebuild `breadth_daily.csv` whenever `S5TH.csv` or `MMTH.csv`
  changes.**

Fetch/refresh helpers:
- **`fetch_investing_data.py`** — Playwright scraper for `NASDAQ100.csv`, `S5TH.csv`, `SPX.csv`
  from investing.com (bypasses Cloudflare).
- **`fetch_sector_data.py`** — SPDR sector ETF history (`XL*.csv`) from Yahoo Finance.
- **`sector_indicator.py`** — sector performance/rotation/breadth report (run `fetch_sector_data.py` first).

## Script catalogue

Run any script with `python <name>.py`. Each prints a metrics table + trade log to stdout and
saves a chart PNG (and some export `*_results.csv` / `*_metrics.csv` files). Grouped by purpose:

### Single-asset breadth backtests (the canonical strategy applied to one instrument)
| Script | Trades | Signal index | Notes |
|---|---|---|---|
| `qqq_backtest.py` | QQQ / NDX | NDX | **Canonical, best-documented implementation** |
| `spy_backtest.py` | SPX | SPX | Same signals on the S&P 500 |
| `soxx_backtest.py` | SOXX | SOXX | Semiconductor ETF |
| `tqqq_backtest.py` | TQQQ (3× NDX) | NDX | Actual price via yfinance (2010+); pre-inception simulated back to 2002 as 3× NDX daily returns minus an overlap-calibrated drag |
| `qqq_pct200_backtest.py` | QQQ | NDX | Pure "% above 200-day" entry, no vote gate |

### Concentrated single-stock
| Script | Notes |
|---|---|
| `ndx_top1_backtest.py` | Buys the single #1 NDX holding each year (from `nasdaq100_top_holdings.csv`), annual rebalance |
| `nasdaq100_top2_backtest.py` | Same idea, top-2 basket variant |

### Multi-asset / portfolio backtests
| Script | Allocation |
|---|---|
| `qqq_portfolio_backtest.py` | 60% QQQ / 30% NDX top-1 / 10% TQQQ (canonical portfolio engine; the web app mirrors this) |
| `combined_n30q70.py` | 30% NDX top-1 / 70% QQQ, proportional DCA, no rebalance |
| `combined_rebalance_n30q70.py` | Same weights, rebalance variant |
| `combined_t5n25q70.py` | 5% TQQQ / 25% NDX top-1 / 70% QQQ, rebalanced at each entry |
| `stock30spy40soxx30_dca_rolling.py` | Rolling-window DCA over 30% NDX top-1 / 40% SPY / 30% SOXX |

### Grid searches & sweeps (write `*_results.csv`)
| Script | Explores |
|---|---|
| `qqq_portfolio_grid_search.py` | QQQ/stock 2-asset allocation grid |
| `qqq_portfolio_combo_search.py` | Full 5-asset weight grid (cheap linear-blend trick) |
| `qqq_ndx_top1_sweep.py` | Rolling-window stats for QQQ↔NDX-top-1 mixes |
| `qqq_ndx_topN_sweep.py` | Extended sweep, top-1/2/3 baskets, non-overlapping-window significance |
| `qqq_signal_combo_search.py` | All 255 subsets of the 8 bearish exit signals × vote threshold |

### Research / validation / robustness (many use scipy)
| Script | Purpose |
|---|---|
| `qqq_validation.py` | Daily-return t-stat / PSR, parameter-sensitivity map, Monte Carlo |
| `qqq_walkforward.py` | Walk-forward optimize/freeze + cross-asset (SPX, Russell 3000) |
| `qqq_improve.py` | Walk-forward tests of trailing-stop / breadth-floor / tiered-entry / trend-reentry upgrades |
| `qqq_indicator_scan.py` | Adds every unused CSV indicator (HY spread, A/D line, 10Y, RSI…) as a candidate gate/exit |
| `qqq_upgrades_test.py` | Execution-lag, total-return (adjusted prices), ensemble divergence, partial exits |
| `qqq_bearish_composite.py` | Codes 8 classic bearish technicals, exits when ≥N fire in 10 days |
| `qqq_dd_throttle.py` | Drawdown-based position-throttle overlay |
| `qqq_sector_experiment.py` | Layers sector-ETF filters (sector breadth, XLY/XLP regime, defensive rotation) |

### Seeking Alpha
| Script | Purpose |
|---|---|
| `sa_qqq_backtest.py` | Applies QQQ breadth-timing to annual Seeking Alpha 10-pick cohorts (2022–2026); four strategies A–D |

**Walk-forward discipline** is used throughout the search/research scripts: parameters are chosen
on one half of 2002–2013 / 2014–2026 (by Sharpe) and reported on the *other* half only. In-sample
"winners" from these grids are overfit by construction — trust the OOS rows.

## Key constants

The tunable parameters live as module-level constants near the top of each script. The current
canonical values (`qqq_backtest.py` / `qqq_portfolio_backtest.py` and the web app):

| Constant | Value | Meaning |
|---|---|---|
| `BUY_B200_THRESH` | 26.0 | Breadth washout entry threshold |
| `VIX_BUY_THRESH` | 30.0 | VIX vote for entry |
| `MA200_WINDOW` | 200 | Moving-average window for the trend vote / re-entry |
| `DIVERGENCE_WINDOW` | 60 | Lookback for the bearish-divergence exit |
| `DIVERGENCE_PRICE_RISE` | 3.0% | Price-rise leg of divergence |
| `DIVERGENCE_BREADTH_FALL` | 20 pts | Breadth-fall leg of divergence |
| `DIVERGENCE_BREADTH_CAP` | 60.0 | Divergence ignored above this breadth |
| `EXT10_PCT` / `CLIMAX_VOTE_WINDOW` | 5.0% / 10 | Climax-top exit |
| `TRAILING_STOP_PCT` | 25.0% | Trailing stop from post-entry high |
| `COMMISSION` / `SLIPPAGE` | $1 / 0.05% | Per-side transaction costs |
| `EXECUTION_LAG` / `FILL_PRICE` | 1 / `"open"` | Fill signals at the next day's open (0/`"close"` = legacy same-day look-ahead) |

SPY and SOXX variants share these; `qqq_backtest.py`'s own copies live at the top of that file.

## Static Web App (`webapp/nextjs/`)

A client-side Next.js app that runs the **`qqq_portfolio_backtest.py`** logic entirely in the
browser — no server. It is a **static export** deployed to GitHub Pages.

**Live URL:** `https://wongkaho2626.github.io/spy500-breadth-backtest/`

### Local development
```bash
cd webapp/nextjs
npm install
npm run dev       # http://localhost:3000/spy500-breadth-backtest
npm run build     # static export → webapp/nextjs/out/
```

### Architecture
- **`app/page.tsx`** — main page; sidebar + tabbed charts / metrics / trade log / sell-signal panel
- **`lib/backtest.ts`** — TypeScript port of the Python portfolio engine; the constants block at
  the top mirrors `qqq_portfolio_backtest.py` and **must be kept in sync** with the Python side.
  Includes the execution model (`FillMode`: `next-open` default | `next-close` | `same-close`);
  the Sidebar exposes it as an "Order Execution" selector
- **`lib/loadData.ts`** — fetches CSVs from `public/data/` at runtime
- **`lib/types.ts` / `lib/utils.ts`** — shared types and formatting helpers
- **`components/`** — `Sidebar` (parameters), `MetricCards`, `SellSignalPanel`, and Chart.js charts
  under `components/charts/` (`GrowthChart`, `AnnualChart`, `DrawdownChart`, `SignalsChart`)
- **Charting:** Chart.js via `react-chartjs-2` (not Recharts)
- **`next.config.mjs`** — `output: 'export'`, `basePath: '/spy500-breadth-backtest'` for GitHub Pages

### Data for the web app
The app fetches its own copies from **`webapp/nextjs/public/data/`**:
`breadth_daily.csv`, `NASDAQ100.csv`, `S5TH.csv`, `VIX.csv`, `nasdaq100_top10_holdings.csv`, and
`stock_prices/*.csv`. When the root CSVs change, **copy the relevant files into `public/data/`**
or the static build will serve stale data.

### Deployment
`.github/workflows/deploy.yml` builds the app and deploys `out/` to GitHub Pages on every push to
`main` (and via `workflow_dispatch`). Uses **Node 24** with npm caching. There is no test/lint CI.

## CSV data reference

All CSVs use `MM/DD/YYYY` dates and comma-formatted prices (e.g. `"1,234.56"`); the `_parse_price()`
helper in each script strips commas before casting. Price CSVs follow the investing.com column
layout: `Date, Price, Open, High, Low, Vol., Change %`.

**Price / index series:** `NASDAQ100.csv` (NDX, the QQQ proxy), `SPX.csv`, `SOXX.csv`,
`Russell3000.csv`, `VIX.csv`, `US10Y.csv`, and the 11 SPDR sector ETFs `XLB/XLC/XLE/XLF/XLI/XLK/XLP/XLRE/XLU/XLV/XLY.csv`.

**Breadth / market internals:** `S5TH.csv` (% S&P 500 above 200-day MA — daily 2007+),
`MMTH.csv` (broader universe, daily 2002+), `S5OH.csv` (% above 100-day MA), `breadth_daily.csv`
(spliced continuous series), `ADV.csv` / `DECL.csv` (advancers/decliners), `MAHN.csv` / `MALN.csv`
(new highs/lows), `UCHG.csv` (unchanged).

**Valuation / macro:** `S&P500ForwardPE.csv`, `ShillerPE.csv`, `BAMLH0A0HYM2.csv` (HY credit
spread), `us_ppi_yoy.csv`.

**NDX holdings & constituents:** `NASDAQ100/` holds annual PDFs, `nasdaq100_top_holdings.csv` /
`nasdaq100_top10_holdings.csv`, and `NASDAQ100/stock_prices/*.csv` (per-stock price history used
by the top-1/portfolio backtests).

**Generated outputs** (checked in): `*_results.csv` (sweeps), `*_metrics.csv` and per-leg trade
logs from the `combined_*` scripts, `*_dca_rolling.csv`, `tqqq_rolling_forecast.csv`.

## Conventions & gotchas

- **`.png` charts are gitignored** (except `docs/*.png`, e.g. `docs/screenshot.png` used in the README).
- **Keep `lib/backtest.ts` constants in sync** with the Python portfolio engine when tuning.
- **Copy updated CSVs into `webapp/nextjs/public/data/`** — the app does not read the repo root.
- **Rebuild `breadth_daily.csv`** after refreshing `S5TH.csv`/`MMTH.csv`.
- Prefer `breadth_daily.csv` over raw `S5TH.csv` for any new lookback-window logic (pre-2007 trap).
- New backtests should copy the signal block from `qqq_backtest.py` rather than re-deriving it.
