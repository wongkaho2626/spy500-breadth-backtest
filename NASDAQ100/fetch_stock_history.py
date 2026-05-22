"""
Download historical stock price data (via yfinance) for all unique companies
in nasdaq100_unique_stocks.csv. Saves one CSV per stock in ./stock_prices/.
"""

import os
import time
import yfinance as yf
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "stock_prices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Deduplicated company → ticker mapping.
# Defunct/acquired companies retain their original ticker so yfinance can
# still return whatever historical data it has for that symbol.
COMPANY_TICKER_MAP = {
    # ── Active ──────────────────────────────────────────────────────────────
    "Microsoft Corp.":                      "MSFT",
    "Alphabet Inc. (Google)":               "GOOGL",
    "Intel Corp.":                          "INTC",
    "Cisco Systems Inc.":                   "CSCO",
    "Apple Inc.":                           "AAPL",
    "QUALCOMM Inc.":                        "QCOM",
    "Amazon.com Inc.":                      "AMZN",
    "Oracle Corp.":                         "ORCL",
    "Meta Platforms Inc.":                  "META",
    "Comcast Corp.":                        "CMCSA",
    "Amgen Inc.":                           "AMGN",
    "NVIDIA Corp.":                         "NVDA",
    "Gilead Sciences Inc.":                 "GILD",
    "Broadcom Inc.":                        "AVGO",
    "Costco Wholesale Corp.":               "COST",
    "Dell Inc.":                            "DELL",
    "Tesla Inc.":                           "TSLA",
    "PepsiCo Inc.":                         "PEP",
    "Charter Communications Inc.":          "CHTR",
    "eBay Inc.":                            "EBAY",
    "Netflix Inc.":                         "NFLX",
    "Adobe Inc.":                           "ADBE",
    "PayPal Holdings Inc.":                 "PYPL",
    "Texas Instruments Inc.":               "TXN",
    "Intuit Inc.":                          "INTU",
    "Kraft Heinz Co.":                      "KHC",
    "Walgreens Boots Alliance Inc.":        "WBA",
    "Starbucks Corp.":                      "SBUX",
    "Teva Pharmaceutical Industries Ltd.":  "TEVA",
    "T-Mobile US Inc.":                     "TMUS",
    "Honeywell International Inc.":         "HON",
    "Booking Holdings Inc.":                "BKNG",
    "Regeneron Pharmaceuticals Inc.":       "REGN",
    "Juniper Networks Inc.":                "JNPR",
    "Ross Stores Inc.":                     "ROST",
    "PACCAR Inc.":                          "PCAR",
    "Western Digital Corp.":               "WDC",
    "Analog Devices Inc.":                  "ADI",
    # ── Acquired / renamed ──────────────────────────────────────────────────
    "Research In Motion Ltd.":              "BB",      # now BlackBerry
    "Celgene Corp.":                        "CELG",    # acquired by BMS 2019
    "Maxim Integrated Products Inc.":       "MXIM",    # acquired by ADI 2021
    "Kraft Foods Inc.":                     "KRFT",    # merged into KHC 2015
    "Priceline Group Inc.":                 "PCLN",    # now BKNG
    "JDS Uniphase Corp.":                   "JDSU",    # now VIAV
    "Symantec Corp.":                       "SYMC",    # now GEN
    "Veritas Software Corp.":               "VRTS",    # spun off from Symantec
    "Siebel Systems Inc.":                  "SEBL",    # acquired by Oracle 2006
    "Linear Technology Corp.":              "LLTC",    # acquired by ADI 2017
    # ── Bankrupt / defunct ──────────────────────────────────────────────────
    "Nextel Communications Inc.":           "NXTL",    # acquired by Sprint 2005
    "Sun Microsystems Inc.":                "SUNW",    # acquired by Oracle 2010
    "WorldCom Inc.":                        "WCOM",    # bankrupt 2002
    "Bed Bath & Beyond Inc.":               "BBBY",    # bankrupt 2023
    "JDS Uniphase Corp. (2001)":            "JDSU",
    "Starbucks Corp. (2004/2006)":          "SBUX",
    "Bed Bath & Beyond Inc. (2003)":        "BBBY",
    "Symantec Corp. (2005)":               "SYMC",
}

# Remove any accidental duplicates introduced above
seen_tickers: dict[str, str] = {}
for name, ticker in COMPANY_TICKER_MAP.items():
    seen_tickers.setdefault(ticker, name)

unique_tickers = list(seen_tickers.keys())

print(f"Downloading data for {len(unique_tickers)} unique tickers...\n")

results = []
for ticker in unique_tickers:
    company = seen_tickers[ticker]
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

# Summary
summary = pd.DataFrame(results)
ok  = summary[summary.status == "OK"]
bad = summary[summary.status != "OK"]

print(f"\n{'='*60}")
print(f"Done. {len(ok)}/{len(unique_tickers)} succeeded.")
if not bad.empty:
    print("\nFailed / no data:")
    for _, r in bad.iterrows():
        print(f"  {r.ticker:6s}  {r.status}  {r.company}")

summary_path = os.path.join(OUTPUT_DIR, "_download_summary.csv")
summary.to_csv(summary_path, index=False)
print(f"\nSummary saved to {summary_path}")
print(f"Stock CSVs saved to   {OUTPUT_DIR}/")
