# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Scripts

```bash
# ── Core backtests (fully runnable with current CSV files) ────────────────────
python backtest.py           # S&P 500 breadth-only strategy (original)
python spy_backtest.py       # SPY breadth + CAPE tiered strategy
python qqq_backtest.py       # NASDAQ 100 hybrid (CAPE-only pre-2007 / breadth+CAPE post-2007)
python sp500_backtest.py     # S&P 500 hybrid (same two-phase structure as qqq_backtest.py)

# ── Optimization scripts (fully runnable) ─────────────────────────────────────
python qqq_optimize.py           # Grid search: NASDAQ 100 breadth+CAPE params
python spy_optimize_cape.py      # Grid search: CAPE-tiered sell caps for SPY
python spy_optimize_drawdown.py  # Grid search: CAPE drop stop + tiered buy thresholds
python spy_optimize_fwdpe.py     # Grid search: forward PE overlay on CAPE tiers

# ── Scripts requiring missing CSV files (will fail until files are restored) ──
# These reference "SPY ETF Stock Price History.csv", "QQQ ETF Stock Price History.csv",
# "SOXX ETF Stock Price History.csv", or "IGV ETF Stock Price History.csv" which
# have been superseded by the new-format CSVs (SPY.csv, QQQ.csv, SOXX.csv).
python spy_optimize.py        # SPY breadth param grid search
python spy_vix_backtest.py    # SPY + VIX filter experiment
python spy_ppi_backtest.py    # SPY + PPI inflation filter
python qqq_vix_backtest.py    # QQQ + VIX filter experiment
python qqq_ppi_backtest.py    # QQQ + PPI inflation filter
python qqq_oas_backtest.py    # QQQ + credit spread (OAS) filter
python qqq_oas_optimize.py    # QQQ OAS filter grid search
python soxx_backtest.py       # SOXX semiconductor ETF breadth strategy
python soxx_vix_backtest.py   # SOXX + VIX filter (VIX≥35 marginally helps)
python soxx_ppi_backtest.py   # SOXX + PPI inflation filter
python portfolio_backtest.py  # Multi-instrument portfolio (SPY/QQQ/SOXX/IGV)
```

Each backtest script prints a metrics table and trade log to stdout and saves a chart PNG in the same directory.

## Architecture

Standalone Python backtesting project — no package structure, no tests, no dependencies file. All scripts are self-contained and read CSV data from the same directory.

**Standard pipeline** (each script does this independently):
1. `load_data()` — reads CSVs, normalises date/price columns, joins on Date index, computes `bearish_div` boolean column
2. `run_strategy()` — single-pass loop over rows, simulates trades with a `portfolio` float and `position` state machine (`"IN"` / `"OUT"`)
3. `run_benchmark()` — buy-and-hold normalised to `INITIAL_CAPITAL`
4. `compute_metrics()` / `print_metrics()` / `print_trades()` / `plot_results()` — reporting layer

**Costs** (all scripts except `backtest.py`): `$1` flat commission + `0.05%` slippage per side, applied to effective entry/exit price. `eff_entry = price * (1 + SLIPPAGE)`, `eff_exit = price * (1 - SLIPPAGE)`.

## Script Inventory

### Core Backtests

| Script | Instrument | Strategy | Price CSV |
|--------|-----------|----------|-----------|
| `backtest.py` | S&P 500 | Breadth-only; no costs | `S&P 500 Historical Data.csv` |
| `spy_backtest.py` | SPY ETF | Breadth + b50 + CAPE tiers | `SPY.csv` |
| `qqq_backtest.py` | NASDAQ 100 | CAPE-only 1987–2006; Breadth+CAPE 2007+ | `NASDAQ100.csv` |
| `sp500_backtest.py` | S&P 500 | CAPE-only 1987–2006; Breadth+CAPE 2007+ | `S&P500.csv` |
| `soxx_backtest.py` | SOXX ETF | Breadth + b50, no CAPE | `SOXX ETF Stock Price History.csv` ✗ |
| `portfolio_backtest.py` | SPY/QQQ/SOXX/IGV | Equal-weight multi-instrument | Multiple ✗ |

### Optimization Scripts

| Script | Purpose | Combinations | Runnable |
|--------|---------|-------------|---------|
| `spy_optimize.py` | SPY breadth param grid | ~14,400 | No (missing CSV) |
| `qqq_optimize.py` | QQQ breadth+CAPE params | ~17,496 | Yes |
| `spy_optimize_cape.py` | CAPE-tiered sell caps for SPY | Large | Yes |
| `spy_optimize_drawdown.py` | CAPE drop stop + tiered buy thresholds | Medium | Yes |
| `spy_optimize_fwdpe.py` | Forward PE overlay on CAPE tiers | Medium | Yes |
| `qqq_oas_optimize.py` | OAS filter for QQQ | 64 | No (missing CSV) |

### Experiment Scripts (retained as research logs)

All experiment scripts include a module-level docstring with the full grid-search results and conclusion.

| Script | Hypothesis | Conclusion |
|--------|-----------|------------|
| `spy_vix_backtest.py` | VIX buy gate / sell cap on SPY | No improvement — breadth<18 already implies VIX≥28 |
| `qqq_vix_backtest.py` | VIX buy gate / sell cap on QQQ | No improvement — buy VIX filters block good trades |
| `soxx_vix_backtest.py` | VIX buy gate for SOXX | VIX≥35 shifts Dec-2018 entry to Christmas Eve low (+$27k final value) |
| `spy_ppi_backtest.py` | PPI blocks buys >4% YoY; gates div exits above 2.5% | Experimental |
| `qqq_ppi_backtest.py` | Same PPI logic applied to QQQ | Experimental |
| `soxx_ppi_backtest.py` | PPI buy gate for SOXX; sell unrestricted | Experimental |
| `qqq_oas_backtest.py` | OAS credit spread gate for QQQ | Experimental |

## Signal Logic

### Buy trigger (all scripts)
Enter long when `breadth < BUY_THRESHOLD` while out of market.

Extended scripts (`spy_backtest.py` onward) also require `b50 < BUY_50_THRESHOLD` (both the 200-day and 50-day breadth must confirm panic).

`spy_backtest.py` and `qqq_backtest.py` (phase 2) add a **CAPE tier**: if `CAPE > CAPE_BUY_HIGH`, use the tighter `BUY_THRESH_HI_CAPE` instead of `BUY_THRESHOLD` (requiring a deeper crash before buying when market is expensive).

### Sell trigger — bearish divergence
Exit when all three conditions hold simultaneously:
- Price rose ≥ `DIVERGENCE_PRICE_RISE`% over `DIVERGENCE_WINDOW` days
- Breadth fell ≥ `DIVERGENCE_BREADTH_FALL` pts over the same window
- Breadth is below `DIVERGENCE_BREADTH_CAP` (prevents exits when breadth is still strong)

`spy_backtest.py` adds a **CAPE sell tier**: if `CAPE >= CAPE_EXPENSIVE`, use `CAP_EXPENSIVE` (tighter cap = exits allowed more easily in expensive markets).

### Two-phase hybrid (qqq_backtest.py and sp500_backtest.py)
- **Phase 1 (pre-2007)**: CAPE-only — buy when `CAPE < CAPE_BUY_ABS`; sell when `CAPE > CAPE_SELL_ABS`
- **Phase 2 (2007+)**: Breadth + CAPE with tiered thresholds
- `BREADTH_START = pd.Timestamp("2007-01-02")` marks the transition

## Key Constants Per Script

| Constant | `backtest.py` | `spy_backtest.py` | `qqq_backtest.py` | `sp500_backtest.py` | `soxx_backtest.py` |
|----------|:---:|:---:|:---:|:---:|:---:|
| `BUY_THRESHOLD` | 18.0 | 18.0 | 18.0 | 18.0 | 18.0 |
| `BUY_50_THRESHOLD` | — | 25.0 | 25.0 | 25.0 | 25.0 |
| `BUY_THRESH_HI_CAPE` | — | 12.0 | 12.0 | 12.0 | — |
| `CAPE_BUY_HIGH` | — | 30.0 | 28.0 | 30.0 | — |
| `DIVERGENCE_WINDOW` | 100 | 100 | 100 | 100 | 60 |
| `DIVERGENCE_PRICE_RISE` | 1.0% | 1.0% | 1.0% | 1.0% | 5.0% |
| `DIVERGENCE_BREADTH_FALL` | 20.0 | 20.0 | 25.0 | 20.0 | 20.0 |
| `DIVERGENCE_BREADTH_CAP` | 55.0 | 55.0 | 55.0 | 55.0 | 50.0 |
| `CAPE_EXPENSIVE` | — | 30.0 | 32.0 | 30.0 | — |
| `CAP_EXPENSIVE` | — | 45.0 | 40.0 | 45.0 | — |
| `CAPE_BUY_ABS` (phase 1) | — | — | 22.0 | 24.0 | — |
| `CAPE_SELL_ABS` (phase 1) | — | — | 30.0 | 30.0 | — |
| `COMMISSION` | — | $1 | $1 | $1 | $1 |
| `SLIPPAGE` | — | 0.05% | 0.05% | 0.05% | 0.05% |

## CSV Data Files

### File Format — two schemas

**Old format** (Investing.com export, UTF-8 BOM, comma-formatted prices):
- Columns: `Date` (MM/DD/YYYY), `Price` (e.g. `"1,234.56"`), plus optional Open/High/Low/Vol/Change
- `_parse_price()` strips commas before casting to float
- Files: `S&P 500 Historical Data.csv`, `S&P 500 Stocks Above 200-Day Average Historical Data.csv`, `S&P 500 Stocks Above 50-Day Average Historical Data.csv`, `CBOE Volatility Index Historical Data.csv`

**New format** (clean daily OHLC):
- Columns: `date` (YYYY-MM-DD), `close` (raw float, no commas)
- No `_parse_price()` needed; loaded with `pd.read_csv(..., format="%Y-%m-%d")`
- Files: `SPY.csv`, `QQQ.csv`, `SOXX.csv`, `NASDAQ100.csv`, `S&P500.csv`, `ShillerPE.csv`

**Special formats**:
- `BAMLH0A0HYM2.csv` — ICE BofA HY OAS; columns: `date`, `open`, `high`, `low`, `close`; monthly, forward-filled to daily
- `us_ppi_yoy.csv` — columns: `date`, `ppi_yoy_pct`; monthly, forward-filled to daily
- `S&P500ForwardPE.csv` — columns: `date`, `forward_pe`; daily

### File Registry

| CSV file | Used by | Content |
|----------|---------|---------|
| `S&P 500 Historical Data.csv` | `backtest.py` | S&P 500 index price (old format) |
| `SPY.csv` | `spy_backtest.py`, `spy_optimize_cape.py`, `spy_optimize_drawdown.py`, `spy_optimize_fwdpe.py` | SPY ETF daily close |
| `NASDAQ100.csv` | `qqq_backtest.py`, `qqq_optimize.py` | NASDAQ 100 daily close |
| `S&P500.csv` | `sp500_backtest.py` | S&P 500 daily close (back to 1987) |
| `SOXX.csv` | *(available but unused — scripts still reference old filename)* | SOXX semiconductor ETF |
| `QQQ.csv` | *(available but unused — scripts still reference old filename)* | QQQ ETF |
| `S&P 500 Stocks Above 200-Day Average Historical Data.csv` | all scripts | Breadth: % of S&P 500 above 200-day MA |
| `S&P 500 Stocks Above 50-Day Average Historical Data.csv` | most scripts | Breadth: % of S&P 500 above 50-day MA |
| `ShillerPE.csv` | `spy_backtest.py`, `qqq_backtest.py`, `sp500_backtest.py`, optimize scripts | Shiller CAPE ratio (monthly) |
| `S&P500ForwardPE.csv` | `spy_optimize_fwdpe.py` | S&P 500 12-month forward P/E |
| `CBOE Volatility Index Historical Data.csv` | `spy_vix_backtest.py`, `qqq_vix_backtest.py`, `soxx_vix_backtest.py` | CBOE VIX |
| `BAMLH0A0HYM2.csv` | `qqq_oas_backtest.py`, `qqq_oas_optimize.py` | ICE BofA HY OAS spread |
| `us_ppi_yoy.csv` | `spy_ppi_backtest.py`, `qqq_ppi_backtest.py`, `soxx_ppi_backtest.py` | US PPI YoY % |
| `Russell3000.csv` | *(available, not currently referenced)* | Russell 3000 index |

### Missing files (cause import-time failures)

The following files were replaced during a CSV format migration but the referencing scripts were not updated:

| Missing file | Current equivalent | Scripts affected |
|---|---|---|
| `SPY ETF Stock Price History.csv` | `SPY.csv` (new format) | `spy_optimize.py`, `spy_vix_backtest.py`, `spy_ppi_backtest.py`, `portfolio_backtest.py` |
| `QQQ ETF Stock Price History.csv` | `QQQ.csv` (new format) | `qqq_vix_backtest.py`, `qqq_ppi_backtest.py`, `qqq_oas_backtest.py`, `qqq_oas_optimize.py` |
| `SOXX ETF Stock Price History.csv` | `SOXX.csv` (new format) | `soxx_backtest.py`, `soxx_vix_backtest.py`, `soxx_ppi_backtest.py` |
| `IGV ETF Stock Price History.csv` | *(not available)* | `portfolio_backtest.py` |

To fix a broken script: update the `FILE` constant at the top and change the date/price column loading to use `date`/`close` and `%Y-%m-%d` format (no `_parse_price()` needed). See `spy_backtest.py` for the reference pattern.

## Metrics Reported

Every backtest computes and prints:
- **Total Return**, **CAGR** (annualised), **Max Drawdown**, **Sharpe Ratio**, **Final Value**
- **# Trades**, **Win Rate**, **Time in Market** (strategy only)

Chart PNGs are saved to the repository root (excluded by `.gitignore`):
- `backtest.py` → `performance.png`
- `spy_backtest.py` → `spy_performance.png`
- `qqq_backtest.py` → `qqq_performance.png`
- `sp500_backtest.py` → `sp500_performance.png`
- etc. (pattern: `<prefix>_performance.png`)

## Conventions and Development Notes

- **No shared library**: every script is self-contained. Copy-paste duplication is intentional. When adding a new strategy variant, start from the closest existing script and modify in place.
- **State machine**: position is always `"IN"` or `"OUT"`. There is no partial position or re-entry logic.
- **Open trade reporting**: if position is `"IN"` at end of dataset, `run_strategy()` returns an `open_trade` dict alongside closed `trades`. `print_trades()` renders it with `"(open)"` in the Exit column.
- **Divergence pre-computation**: `bearish_div` is computed as a vectorised boolean Series inside `load_data()` (except `backtest.py` which uses `add_divergence_signal()`), not inside the loop, for performance.
- **CAPE forward-fill**: CAPE is monthly; `merged["cape"].ffill()` propagates it to every trading day.
- **Optimizers output only**: optimization scripts do not produce charts. Results go to stdout ranked by Sharpe then by Max Drawdown.
- **Python version**: uses `list[dict]` and `tuple[pd.Series, ...]` type hints — requires Python 3.10+.
