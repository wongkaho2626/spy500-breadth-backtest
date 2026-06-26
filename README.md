# QQQ Portfolio Backtest

A market-timing backtest engine for NASDAQ-heavy portfolios (QQQ, TQQQ, NDX top stock, SPY, SOXX), with both a Python CLI and an interactive static web app deployable to GitHub Pages.

![App Screenshot](docs/screenshot.png)

## Live Demo

**[Try it on GitHub Pages →](https://wongkaho2626.github.io/spy500-breadth-backtest/)**

## How It Works

The strategy uses a breadth indicator (% of S&P 500 stocks above their 200-day moving average) to time entries and exits:

- **Buy signal** — breadth drops below a threshold (oversold market)
- **Sell signal** — bearish divergence: price rises while breadth falls (trend exhaustion)
- **Trailing stop** — optional downside protection after entry

The web app runs this logic entirely in the browser. No server needed.

## Web App Features

- Adjustable portfolio allocation across QQQ / NDX Top-1 Stock / TQQQ / SPY / SOXX
- Initial capital + monthly/yearly DCA contributions
- Custom date range and post-sell cooldown period
- Charts: Portfolio Growth, Annual Returns, Drawdown, Market Signals
- Metrics: Total Return, CAGR, Max Drawdown, Sharpe Ratio, Win Rate, Time in Market
- All compared against a Buy & Hold NDX benchmark

## Running Locally

### Web App

```bash
cd webapp/nextjs
npm install
npm run dev     # http://localhost:3000/spy500-breadth-backtest
```

### Python Backtests

```bash
# SPY breadth strategy
python spy_backtest.py

# QQQ breadth strategy (with trailing stop)
python qqq_backtest.py

# S&P 500 breadth strategy
python backtest.py

# Seeking Alpha annual picks comparison
python seeking_alpha_backtest.py

# Parameter grid search
python spy_optimize.py
python qqq_optimize.py
```

Each script prints a metrics table and trade log to stdout, and saves a chart PNG.

## Data Files

All CSVs use `MM/DD/YYYY` dates and comma-formatted prices. Place them in the project root for the Python scripts, and in `webapp/nextjs/public/data/` for the web app.

| File | Used by |
|------|---------|
| `SPY ETF Stock Price History.csv` | `spy_backtest.py`, `spy_optimize.py` |
| `QQQ ETF Stock Price History.csv` | `qqq_backtest.py`, `qqq_optimize.py` |
| `S&P 500 Historical Data.csv` | `backtest.py` |
| `S&P 500 Stocks Above 200-Day Average Historical Data.csv` | all scripts |
| `seeking_alpha.csv`, `SPX.csv`, `S&P500ForwardPE.csv`, `S5TH.csv`, `VIX.csv` | `seeking_alpha_backtest.py` |

## Deployment

Pushing to `main` automatically builds the Next.js app and deploys to GitHub Pages via `.github/workflows/deploy.yml`.
