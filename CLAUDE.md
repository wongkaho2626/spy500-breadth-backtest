# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Scripts

```bash
# Run the Seeking Alpha annual picks backtest (uses seeking_alpha.csv + market indicator CSVs)
python seeking_alpha_backtest.py

# Run the original S&P 500 breadth strategy (uses S&P 500 Historical Data.csv)
python backtest.py

# Run the SPY ETF backtest (uses SPY ETF Stock Price History.csv)
python spy_backtest.py

# Run the QQQ ETF backtest (uses QQQ ETF Stock Price History.csv)
python qqq_backtest.py

# Grid-search parameter optimization for SPY
python spy_optimize.py

# Grid-search parameter optimization for QQQ
python qqq_optimize.py
```

Each backtest script prints a metrics table and trade log to stdout, and saves a chart PNG in the same directory.

## Architecture

Standalone Python backtesting project — no package structure, no tests, no dependencies file. All scripts are self-contained and read CSV data from the same directory.

**Data pipeline** (each script does this independently):
1. `load_data()` — reads two CSVs (price history + breadth), parses comma-formatted prices, joins on Date index, computes `bearish_div` boolean column inline
2. `run_strategy()` — single-pass loop over rows, simulates trades with a `portfolio` float and `position` state machine (`"IN"` / `"OUT"`)
3. `run_benchmark()` — simple buy-and-hold normalised to `INITIAL_CAPITAL`
4. `compute_metrics()` / `print_metrics()` / `print_trades()` / `plot_results()` — reporting layer

**Signal logic** (shared across all scripts):
- **Buy trigger**: `breadth < BUY_THRESHOLD` while out of market
- **Sell trigger**: bearish divergence — price rose ≥ `DIVERGENCE_PRICE_RISE`% over `DIVERGENCE_WINDOW` days while breadth fell ≥ `DIVERGENCE_BREADTH_FALL` pts AND breadth is below `DIVERGENCE_BREADTH_CAP`
- `qqq_backtest.py` adds a **trailing stop** (`TRAILING_STOP_PCT = 30%`) as a second exit condition

**Costs** (`spy_backtest.py` and `qqq_backtest.py` only): `$1` flat commission + `0.05%` slippage per side applied to effective entry/exit price. `backtest.py` has no cost model.

**Optimization scripts** (`spy_optimize.py`, `qqq_optimize.py`): brute-force grid search over ~14,000 parameter combinations using `itertools.product`. Results ranked by Total Return and Sharpe Ratio.

## Key Constants (tunable per script)

| Constant | backtest.py | spy_backtest.py | qqq_backtest.py |
|---|---|---|---|
| `BUY_THRESHOLD` | 18.0 | 18.0 | 26.0 |
| `DIVERGENCE_WINDOW` | 100 days | 100 days | 60 days |
| `DIVERGENCE_PRICE_RISE` | 1.0% | 1.0% | 3.0% |
| `DIVERGENCE_BREADTH_FALL` | 20 pts | 20 pts | 20 pts |
| `DIVERGENCE_BREADTH_CAP` | 55.0 | 55.0 | 60.0 |
| `TRAILING_STOP_PCT` | — | — | 30.0% |

## Seeking Alpha Backtest (`seeking_alpha_backtest.py`)

Compares three annual stock-picking strategies using 10 Seeking Alpha picks/year:

| Strategy | Entry | Exit |
|---|---|---|
| A (baseline) | Jan 1 every year | Dec 31 every year |
| B (PE filter) | First day S&P 500 fwd PE < 20 | Dec 31 |
| C (enhanced) | PE<20 OR (VIX≥22 AND breadth≤50); fallback Jan 1 | SPX bearish-div OR trailing-stop(-25%) OR year-end |

**Key parameters:**
- `FWD_PE_BUY = 20.0` — primary S&P 500 forward PE entry threshold
- `VIX_ALT_THRESH = 22.0`, `BREADTH_ALT_THRESH = 50.0` — alt-entry (fear + oversold)
- `DIV_WINDOW = 60`, `DIV_PRICE_RISE = 5.0%`, `DIV_BREADTH_FALL = 20 pts`, `DIV_BREADTH_CAP = 60.0` — bearish divergence exit parameters
- `TRAILING_STOP_PCT = 25.0` — trailing stop for Strategy C

**Data files used:** `seeking_alpha.csv`, `SPX.csv`, `S&P500ForwardPE.csv`, `S5TH.csv`, `VIX.csv`

Entry prices for non-CSV dates are estimated via SPX beta=1 proxy. Year-end exits use actual CSV stock prices.

## Static Web App (Next.js)

A client-side backtest UI lives in `webapp/nextjs/`. It runs the same strategy logic in the browser — no server required — and is deployed to GitHub Pages on every push to `main`.

**Live URL:** `https://<github-user>.github.io/spy500-breadth-backtest/`

### Local development

```bash
cd webapp/nextjs
npm install
npm run dev       # dev server at http://localhost:3000/spy500-breadth-backtest
npm run build     # static export → webapp/nextjs/out/
```

### Architecture

- **`app/page.tsx`** — main page; orchestrates sidebar, charts, metrics, and trade log tabs
- **`lib/backtest.ts`** — TypeScript port of the Python backtest engine (same signal logic)
- **`lib/loadData.ts`** — fetches CSV data files from `public/data/` at runtime
- **`components/`** — Sidebar (parameters), MetricCards, SellSignalPanel, and Recharts-based chart components
- **`next.config.mjs`** — `output: 'export'`, `basePath: '/spy500-breadth-backtest'` for GitHub Pages

### Deployment

`.github/workflows/deploy.yml` builds the Next.js app and deploys the `out/` directory to GitHub Pages on every push to `main`. The workflow uses Node 20 and caches `npm` dependencies.

Data CSVs must be present in `webapp/nextjs/public/data/` for the static build to serve them correctly.

## CSV Data Files

All CSVs use `MM/DD/YYYY` date format and comma-formatted prices (e.g. `"1,234.56"`). The `_parse_price()` helper strips commas before casting to float.

- `S&P 500 Historical Data.csv` — used by `backtest.py` as SPY proxy
- `SPY ETF Stock Price History.csv` — used by `spy_backtest.py` and `spy_optimize.py`
- `QQQ ETF Stock Price History.csv` — used by `qqq_backtest.py` and `qqq_optimize.py`
- `S&P 500 Stocks Above 200-Day Average Historical Data.csv` — breadth indicator (% of S&P 500 stocks above 200-day MA), used by all scripts
- `seeking_alpha.csv` — annual Seeking Alpha 10-pick cohorts (2022–2026), used by `seeking_alpha_backtest.py`
- `SPX.csv`, `S&P500ForwardPE.csv`, `S5TH.csv`, `VIX.csv` — market data for `seeking_alpha_backtest.py`
