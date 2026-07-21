"""
Download historical stock price data (via yfinance) for every unique ticker in
sp500_top17_holdings.csv. Saves one CSV per stock in ./stock_prices/.
"""

import os
import time
import yfinance as yf
import pandas as pd

BASE_DIR      = os.path.dirname(__file__)
HOLDINGS_FILE = os.path.join(BASE_DIR, "sp500_top17_holdings.csv")
OUTPUT_DIR    = os.path.join(BASE_DIR, "stock_prices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

holdings = pd.read_csv(HOLDINGS_FILE)
tickers  = sorted(holdings["Ticker"].unique())
names    = holdings.drop_duplicates("Ticker").set_index("Ticker")["Company"]

print(f"Downloading data for {len(tickers)} unique tickers...\n")

results = []
for ticker in tickers:
    company = names.get(ticker, ticker)
    try:
        df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
        if df.empty:
            status = "NO_DATA"
            rows = 0
        else:
            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            out_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
            df.to_csv(out_path)
            status = "OK"
            rows = len(df)
        print(f"  [{status:8s}] {ticker:6s}  {company}  ({rows} rows)")
    except Exception as e:
        status = "ERROR"
        rows = 0
        print(f"  [{status:8s}] {ticker:6s}  {company}  — {e}")

    results.append({"ticker": ticker, "company": company, "status": status, "rows": rows})
    time.sleep(0.3)   # gentle rate-limit

summary = pd.DataFrame(results)
ok  = summary[summary.status == "OK"]
bad = summary[summary.status != "OK"]

print(f"\n{'='*60}")
print(f"Done. {len(ok)}/{len(tickers)} succeeded.")
if not bad.empty:
    print("\nFailed / no data:")
    for _, r in bad.iterrows():
        print(f"  {r.ticker:6s}  {r.status}  {r.company}")

summary_path = os.path.join(OUTPUT_DIR, "_download_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\nSummary saved to {summary_path}")
print(f"Stock CSVs saved to   {OUTPUT_DIR}/")
