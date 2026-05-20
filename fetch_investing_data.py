"""
Fetch latest historical data from investing.com and update local CSV files.
Uses Playwright headless browser to bypass Cloudflare protection.

Instruments updated:
  - NASDAQ100.csv
  - S&P 500 Stocks Above 200-Day Average Historical Data.csv
  - S&P 500 Stocks Above 50-Day Average Historical Data.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

DATA_DIR = Path(__file__).parent

INSTRUMENTS = [
    {
        "name": "NASDAQ 100",
        "url": "https://www.investing.com/indices/nq-100-historical-data",
        "csv_file": DATA_DIR / "NASDAQ100.csv",
    },
    {
        "name": "S&P 500 Above 200-Day MA",
        "url": "https://www.investing.com/indices/sp-500-stocks-above-200-day-average-historical-data",
        "csv_file": DATA_DIR / "S&P 500 Stocks Above 200-Day Average Historical Data.csv",
    },
    {
        "name": "S&P 500 Above 50-Day MA",
        "url": "https://www.investing.com/indices/s-p-500-stocks-above-50-day-average-historical-data",
        "csv_file": DATA_DIR / "S&P 500 Stocks Above 50-Day Average Historical Data.csv",
    },
]

CSV_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]

# investing.com now uses Tailwind-style classes; match on the freeze-column prefix
HISTORICAL_TABLE_SELECTOR = "table[class*='freeze-column-w-1']"


def _read_existing(csv_file: Path) -> pd.DataFrame:
    if not csv_file.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(csv_file, encoding="utf-8-sig")  # utf-8-sig strips BOM
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df


def _dismiss_consent(page) -> None:
    for selector in [
        "button#onetrust-accept-btn-handler",
        "button[id*='accept']",
        ".js-accept-cookies",
    ]:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=2000):
                btn.click()
                time.sleep(0.5)
                return
        except Exception:
            pass


def _wait_for_table(page) -> bool:
    try:
        page.wait_for_selector(HISTORICAL_TABLE_SELECTOR, timeout=10000)
        return True
    except PlaywrightTimeout:
        return False


def _extract_rows(page, cutoff_date: pd.Timestamp | None) -> list[dict]:
    rows = []
    # Pick the first freeze-column table whose first row looks like a date row
    tables = page.query_selector_all(HISTORICAL_TABLE_SELECTOR)
    data_table = None
    for t in tables:
        first_tds = t.query_selector_all("tbody tr:first-child td")
        if first_tds:
            try:
                pd.to_datetime(first_tds[0].inner_text().strip())
                data_table = t
                break
            except Exception:
                continue

    if data_table is None:
        return rows

    for tr in data_table.query_selector_all("tbody tr"):
        cells = tr.query_selector_all("td")
        if len(cells) < 6:
            continue
        try:
            texts = [c.inner_text().strip() for c in cells]
            # Page uses "May 19, 2026" format; pd.to_datetime handles it
            date = pd.to_datetime(texts[0])
            if cutoff_date is not None and date <= cutoff_date:
                continue
            rows.append({
                "Date": date.strftime("%m/%d/%Y"),
                "Price": texts[1],
                "Open": texts[2],
                "High": texts[3],
                "Low": texts[4],
                "Vol.": texts[5] if len(texts) > 5 else "",
                "Change %": texts[6] if len(texts) > 6 else "",
            })
        except Exception:
            continue
    return rows


def _set_date_range(page, start_date: str, end_date: str) -> bool:
    """Try to extend the date range via the investing.com date picker."""
    try:
        picker = page.locator(
            "[data-test='historical-data-date-picker'], #widgetFieldDateRange"
        ).first
        if not picker.is_visible(timeout=3000):
            return False
        picker.click()
        time.sleep(0.5)

        start_input = page.locator("input[name='startDate'], #startDate").first
        end_input = page.locator("input[name='endDate'], #endDate").first
        start_input.fill(start_date)
        end_input.fill(end_date)

        apply_btn = page.locator("button[data-test='apply-button'], #applyBtn").first
        apply_btn.click()
        time.sleep(2)
        return True
    except Exception:
        return False


def _fetch_instrument(page, instrument: dict, verbose: bool) -> int:
    name = instrument["name"]
    url = instrument["url"]
    csv_file = instrument["csv_file"]

    existing = _read_existing(csv_file)
    has_dates = not existing.empty and "Date" in existing.columns and existing["Date"].notna().any()
    cutoff = existing["Date"].max() if has_dates else None

    if verbose:
        cutoff_str = cutoff.strftime("%m/%d/%Y") if cutoff is not None else "none"
        print(f"  {name}: latest in CSV = {cutoff_str}")

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
    except PlaywrightTimeout:
        print(f"  {name}: page load timed out, skipping")
        return 0

    _dismiss_consent(page)
    time.sleep(1)

    # If CSV is more than 60 days stale, try to extend the date picker range
    if cutoff is not None and (pd.Timestamp.now() - cutoff).days > 60:
        start_str = (cutoff + pd.Timedelta(days=1)).strftime("%m/%d/%Y")
        end_str = pd.Timestamp.now().strftime("%m/%d/%Y")
        _set_date_range(page, start_str, end_str)

    if not _wait_for_table(page):
        print(f"  {name}: data table not found on page, skipping")
        return 0

    new_rows = _extract_rows(page, cutoff)
    if not new_rows:
        if verbose:
            print(f"  {name}: no new rows found")
        return 0

    new_df = pd.DataFrame(new_rows, columns=CSV_COLUMNS)
    if not existing.empty:
        existing["Date"] = existing["Date"].dt.strftime("%m/%d/%Y")
        combined = pd.concat([new_df, existing], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(csv_file, index=False, quoting=1, encoding="utf-8-sig")
    if verbose:
        print(f"  {name}: added {len(new_rows)} new row(s)")
    return len(new_rows)


def fetch_all_updates(verbose: bool = True) -> None:
    if verbose:
        print("Fetching latest data from investing.com...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )
        page = context.new_page()

        total = 0
        for instrument in INSTRUMENTS:
            total += _fetch_instrument(page, instrument, verbose)

        browser.close()

    if verbose:
        print(f"Done. Total new rows added: {total}\n")


if __name__ == "__main__":
    fetch_all_updates(verbose=True)
