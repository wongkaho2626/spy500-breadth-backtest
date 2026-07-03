import { DayData, prepareData } from './backtest'

// Parse MM/DD/YYYY -> YYYY-MM-DD
function parseMMDDYYYY(s: string): string {
  const clean = s.replace(/"/g, '').trim()
  const [m, d, y] = clean.split('/')
  return `${y}-${m.padStart(2,'0')}-${d.padStart(2,'0')}`
}

// Strip commas and quotes, parse float
function parsePrice(s: string): number {
  return parseFloat(s.replace(/[",\s]/g, ''))
}

// Quote-aware CSV field splitter used for both headers and body rows
function parseCsvLine(line: string): string[] {
  const fields: string[] = []
  let cur = '', inQ = false
  for (const c of line) {
    if (c === '"') { inQ = !inQ }
    else if (c === ',' && !inQ) { fields.push(cur); cur = '' }
    else cur += c
  }
  fields.push(cur)
  return fields
}

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split('\n')
  // Use quote-aware split so header names containing commas are not split incorrectly
  const headers = parseCsvLine(lines[0]).map(h => h.replace(/[\r﻿]/g, '').trim())
  return lines.slice(1).map(line => {
    const fields = parseCsvLine(line)
    const rec: Record<string, string> = {}
    headers.forEach((h, i) => { rec[h] = (fields[i] ?? '').replace(/\r/g, '').trim() })
    return rec
  })
}

export interface AppData {
  data: DayData[]
  topHoldings: Map<number, string>
  alignedStocks: Map<string, Map<string, number>>
  alignedTqqq: Map<string, number> | null
  alignedSpy: Map<string, number> | null
  alignedSoxx: Map<string, number> | null
}

// Name->ticker mapping
const NAME_TO_TICKER_MAP: [string, string][] = [
  ["cisco", "CSCO"], ["microsoft", "MSFT"], ["intel", "INTC"],
  ["oracle", "ORCL"], ["qualcomm", "QCOM"], ["apple", "AAPL"],
  ["alphabet", "GOOGL"], ["google", "GOOGL"], ["amazon", "AMZN"],
  ["tesla", "TSLA"], ["nvidia", "NVDA"], ["meta", "META"],
  ["facebook", "META"], ["paypal", "PYPL"], ["netflix", "NFLX"],
  ["broadcom", "AVGO"], ["costco", "COST"], ["pepsico", "PEP"],
  ["t-mobile", "TMUS"], ["ebay", "EBAY"], ["dell", "DELL"],
  ["comcast", "CMCSA"], ["amgen", "AMGN"], ["gilead", "GILD"],
  ["charter", "CHTR"], ["texas instruments", "TXN"],
]

function nameToTicker(name: string): string | null {
  const lower = name.toLowerCase()
  for (const [key, ticker] of NAME_TO_TICKER_MAP) {
    if (lower.includes(key)) return ticker
  }
  return null
}

async function fetchText(path: string): Promise<string> {
  const base = process.env.NEXT_PUBLIC_BASE_PATH ?? ''
  const url = `${base}${path}`
  const res = await fetch(url)
  if (!res.ok) throw new Error(`Failed to load ${url} (HTTP ${res.status})`)
  return res.text()
}

// Continuous daily breadth (2002+) built by build_breadth_daily.py; S5TH.csv
// alone is only daily from 2007 (bimonthly before), which corrupts row-based
// lookback windows — the fallback truncates it to the daily era.
async function fetchBreadth(): Promise<{ text: string; daily: boolean }> {
  try {
    return { text: await fetchText('/data/breadth_daily.csv'), daily: true }
  } catch {
    return { text: await fetchText('/data/S5TH.csv'), daily: false }
  }
}

export async function loadAppData(): Promise<AppData> {
  // Load all files in parallel
  const [ndxText, breadthFile, vixText, holdingsText] = await Promise.all([
    fetchText('/data/NASDAQ100.csv'),
    fetchBreadth(),
    fetchText('/data/VIX.csv'),
    fetchText('/data/nasdaq100_top10_holdings.csv'),
  ])

  // Parse NDX prices (MM/DD/YYYY, "Price" column, comma-formatted)
  const ndxRows = parseCSV(ndxText)
  const ndxPrices: [string, number][] = ndxRows
    .map(r => [parseMMDDYYYY(r['Date']), parsePrice(r['Price'])] as [string, number])
    .filter(([, v]) => !isNaN(v))
    .sort((a, b) => a[0].localeCompare(b[0]))

  // Parse breadth (% S&P 500 stocks above 200-day MA):
  // breadth_daily.csv has a "breadth" column; S5TH.csv has "Price"
  const breadthRows = parseCSV(breadthFile.text)
  const breadthCol = breadthFile.daily ? 'breadth' : 'Price'
  const breadthPrices: [string, number][] = breadthRows
    .map(r => [parseMMDDYYYY(r['Date']), parsePrice(r[breadthCol])] as [string, number])
    .filter(([, v]) => !isNaN(v))
    .filter(([d]) => breadthFile.daily || d >= '2007-01-01')
    .sort((a, b) => a[0].localeCompare(b[0]))

  // Parse VIX
  const vixRows = parseCSV(vixText)
  const vixPrices: [string, number][] = vixRows
    .map(r => [parseMMDDYYYY(r['Date']), parsePrice(r['Price'])] as [string, number])
    .filter(([, v]) => !isNaN(v))
    .sort((a, b) => a[0].localeCompare(b[0]))

  // Prepare main data
  const data = prepareData(ndxPrices, breadthPrices, vixPrices)

  // Parse holdings -> top-1 per year
  const holdingsRows = parseCSV(holdingsText)
  const topHoldings = new Map<number, string>()
  for (const r of holdingsRows) {
    if (r['Rank'] === '1') {
      const year   = parseInt(r['Year'], 10)
      const ticker = nameToTicker(r['Holding'])
      if (ticker) topHoldings.set(year, ticker)
    }
  }

  // Get all unique tickers needed
  const uniqueTickers = new Set(topHoldings.values())
  const etfTickers = ['TQQQ', 'SPY', 'SOXX']

  // Load stock price CSVs in parallel
  const allTickers = [...uniqueTickers, ...etfTickers]
  const stockResults = await Promise.allSettled(
    allTickers.map(ticker =>
      fetchText(`/data/stock_prices/${ticker}.csv`)
        .then(text => ({ ticker, text }))
    )
  )

  const alignedStocks = new Map<string, Map<string, number>>()
  let alignedTqqq: Map<string, number> | null = null
  let alignedSpy: Map<string, number> | null = null
  let alignedSoxx: Map<string, number> | null = null

  // All data dates for forward-fill alignment
  const allDates = data.map(r => r.date)

  for (const result of stockResults) {
    if (result.status !== 'fulfilled') {
      console.warn('[loadData] Stock CSV load failed:', result.reason)
      continue
    }
    const { ticker, text } = result.value
    const rows = parseCSV(text)

    if (rows.length === 0) {
      console.warn(`[loadData] ${ticker}: CSV has no data rows, skipping`)
      continue
    }

    // Detect format: stock CSVs have 'Close' column; ETF CSVs (TQQQ/SPY/SOXX) have 'price' column
    const priceCol = ('price' in rows[0]) ? 'price' : 'Close'

    const rawMap = new Map<string, number>()
    for (const r of rows) {
      const date = r['Date']?.trim()
      if (!date) continue
      const val = parseFloat(r[priceCol])
      if (!isNaN(val) && val > 0) rawMap.set(date, val)
    }

    // Forward-fill to all data dates
    const filled = new Map<string, number>()
    let last = NaN
    for (const d of allDates) {
      if (rawMap.has(d)) last = rawMap.get(d)!
      if (!isNaN(last)) filled.set(d, last)
    }

    if (ticker === 'TQQQ') alignedTqqq = filled
    else if (ticker === 'SPY') alignedSpy = filled
    else if (ticker === 'SOXX') alignedSoxx = filled
    else alignedStocks.set(ticker, filled)
  }

  return { data, topHoldings, alignedStocks, alignedTqqq, alignedSpy, alignedSoxx }
}
