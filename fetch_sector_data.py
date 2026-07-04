"""Download SPDR sector ETF price history from Yahoo Finance and save as CSVs
in the same format used elsewhere in this repo (Date,Price,Open,High,Low,Vol.,Change %).
"""
import csv
import json
import time
import urllib.request
from datetime import datetime, timezone

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLC": "Communication Services",
}

HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_ticker(symbol: str) -> list[dict]:
    # range=max silently drops to monthly granularity on this endpoint; an explicit
    # period1/period2 window keeps daily bars for the full history instead.
    period1 = 0
    period2 = int(time.time())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={period1}&period2={period2}&interval=1d"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)

    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    closes = quote["close"]
    opens = quote["open"]
    highs = quote["high"]
    lows = quote["low"]
    volumes = quote["volume"]

    rows = []
    prev_close = None
    for ts, o, h, l, c, v in zip(timestamps, opens, highs, lows, closes, volumes):
        if c is None or o is None:
            continue
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%m/%d/%Y")
        change_pct = "" if prev_close is None else f"{(c / prev_close - 1) * 100:+.2f}%"
        rows.append(
            {
                "Date": date,
                "Price": f"{c:,.2f}",
                "Open": f"{o:,.2f}",
                "High": f"{h:,.2f}",
                "Low": f"{l:,.2f}",
                "Vol.": "" if v is None else f"{v:,}",
                "Change %": change_pct,
            }
        )
        prev_close = c

    rows.reverse()  # newest first, matching existing CSVs
    return rows


def save_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    for symbol in SECTOR_ETFS:
        print(f"Fetching {symbol} ({SECTOR_ETFS[symbol]})...")
        rows = fetch_ticker(symbol)
        save_csv(f"{symbol}.csv", rows)
        print(f"  saved {len(rows)} rows -> {symbol}.csv")
        time.sleep(0.5)


if __name__ == "__main__":
    main()
