# SPY historical ticker and price data

This directory contains the historical stock-symbol universe associated with
the SPDR S&P 500 ETF Trust from 1996 through 2026 and all price histories that
Yahoo Finance returned through `yfinance`.

## Files

- `all_tickers_1996_2026.csv`: 1,206 unique historical symbols, their first
  and last membership dates, report-date coverage, and Yahoo-compatible symbol.
- `all_tickers_1996_2026.txt`: the same 1,206 symbols, one per line.
- `holdings_by_report.csv`: point-in-time symbols for each of the 48 downloaded
  SEC N-30D holdings reports.
- `report_manifest.csv`: SEC filing and Schedule of Investments dates.
- `download_status.csv`: one row per symbol with Yahoo download status, row
  count, date coverage, and output path.
- `unavailable_tickers.csv`: symbols for which Yahoo returned no history.
- `prices/`: one daily OHLCV/action CSV per available Yahoo symbol.

## Method and sources

The locally stored [SEC N-30D filings](../SPY/) are authoritative for the
Schedule of Investments dates and issuer holdings. Those reports generally
identify issuers rather than exchange tickers. Point-in-time symbols are
therefore resolved with the MIT-licensed
[`fja05680/sp500`](https://github.com/fja05680/sp500) historical component
dataset, which covers 1996 through 2026. Report-date snapshots are retained in
`holdings_by_report.csv`; the master ticker file also includes constituents
that entered or left between the semiannual filing dates.

Yahoo class-share symbols use hyphens, so examples such as `BRK.B` and `BF.B`
are requested from `yfinance` as `BRK-B` and `BF-B`. Both forms are preserved
in the manifests.

## Coverage caveat

Yahoo Finance does not retain downloadable histories for every delisted,
bankrupt, acquired, or renamed symbol. Those symbols remain in the master list
and are explicitly recorded in `unavailable_tickers.csv`; they are not silently
dropped or replaced with a successor company. Historical constituent data is
community-maintained and may contain omissions or symbol-history errors, so it
should not be treated as a licensed institutional point-in-time dataset.

Regenerate or resume the data with:

```bash
python3 download_spy_historical_prices.py
```
