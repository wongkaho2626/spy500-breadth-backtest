# SP500 data — top-17 holdings & per-stock prices

Data layer for `sp500_top17_backtest.py` (S&P 500 top-17 breadth strategy,
equal-weight vs market-cap-weight).

## `sp500_top17_holdings.csv`

Columns: `Year, Rank, Company, Ticker, MarketCap ($B)` — ranks 1–17 for each
year 2002–2026.

**Point-in-time convention:** year Y's rows are the top 17 S&P 500 members
ranked by market cap at the **end of year Y−1** — the composition an investor
could actually have known when entering year Y. The `MarketCap ($B)` column is
the cap-weighting source (analogous to `Value ($B)` in
`NASDAQ100/nasdaq100_top_holdings.csv`).

**Provenance & accuracy:** hand-curated from historical year-end largest-US-
company rankings (cross-checked against public sources such as year-end
market-cap league tables; era-level ordering verified — GE/MSFT early 2000s,
XOM 2006–2011, AAPL 2012+, NVDA/AVGO 2024–25). The top ~10 ranks each year are
solid; **ranks ~10–17 and the cap figures are approximate** (±1–2 rank
positions at the tail are possible). Non-S&P-500 members are excluded (e.g.
Google before its March 2006 index add, Berkshire before February 2010,
Tesla before December 2020).

**Ticker conventions:** surviving-entity tickers are used throughout so
yfinance can serve continuous adjusted history — `T` covers SBC Communications
pre-2005 (SBC is the surviving entity behind today's AT&T Inc.), `GOOGL` for
both Google share classes, `META` for Facebook, `BRK-B` for Berkshire class B.

## `stock_prices/`

One CSV per ticker (`Date,Close,High,Low,Open,Volume`, auto-adjusted), the
same format as `NASDAQ100/stock_prices/`. Populate/refresh with:

```bash
python SP500/fetch_stock_history.py
```

which downloads `period="max"` history via yfinance for every unique ticker in
the holdings file and writes `_download_summary.csv`. Tickers without a price
file are skipped by the backtest (weights renormalized over the rest) — the
backtest prints a coverage table so partial-data runs are obvious.
