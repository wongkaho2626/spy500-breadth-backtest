// Constants matching Python qqq_portfolio_backtest.py
export const BUY_B200_THRESH        = 26.0
export const VIX_BUY_THRESH         = 30.0
export const MA200_WINDOW           = 200
export const DIVERGENCE_WINDOW      = 60
export const DIVERGENCE_PRICE_RISE  = 3.0
export const DIVERGENCE_BREADTH_FALL = 20.0
export const DIVERGENCE_BREADTH_CAP  = 60.0
// Climax-top exit: NDX extended above its 10-day MA AND MACD(12,26,9) flipped
// bearish, both within CLIMAX_VOTE_WINDOW days AFTER entry
export const EXT10_PCT          = 5.0
export const CLIMAX_VOTE_WINDOW = 10
// Trailing stop on NDX (the signal index)
export const TRAILING_STOP_PCT  = 25.0
export const COMMISSION = 1.0
export const SLIPPAGE   = 0.0005

// ── Execution timing ─────────────────────────────────────────────────────────
// Signals come from end-of-day NDX closes, so the earliest tradeable fill is the
// NEXT session. Default: a signal on day t fills at day t+1's OPEN of each leg.
// 'same-close' (lag 0, close) is the legacy same-day look-ahead fill and
// reproduces the prior numbers exactly; it stays available as a toggle.
export type FillMode = 'next-open' | 'next-close' | 'same-close'
export const DEFAULT_FILL_MODE: FillMode = 'next-open'
export function fillModeToParams(mode: FillMode): { lag: number; fillOn: 'open' | 'close' } {
  if (mode === 'next-open')  return { lag: 1, fillOn: 'open' }
  if (mode === 'next-close') return { lag: 1, fillOn: 'close' }
  return { lag: 0, fillOn: 'close' }  // same-close (legacy look-ahead)
}

export interface DayData {
  date: string        // ISO YYYY-MM-DD
  price: number       // NDX close
  open: number        // NDX open (fill price when execution lags to the next bar)
  breadth: number     // % above 200-day MA
  vix: number
  ma200: number | null
  vix_vote: boolean   // vix > VIX_BUY_THRESH
  ma200_vote: boolean // price > ma200
  vote_gate: boolean  // vix_vote || ma200_vote
  price_rose: boolean // (price - price[i-DIVERGENCE_WINDOW]) / price[i-DIVERGENCE_WINDOW] * 100 >= DIVERGENCE_PRICE_RISE
  breadth_fell: boolean // breadth[i-DIVERGENCE_WINDOW] - breadth[i] >= DIVERGENCE_BREADTH_FALL
  macd_cross: boolean // MACD(12,26,9) histogram flipped negative today
  ext10: boolean      // price >= EXT10_PCT% above its 10-day MA
  ma200_recross: boolean // price closed back above MA200 today (was <= yesterday)
}

export interface BacktestParams {
  qqq: number; stock: number; tqqq: number; spy: number; soxx: number
  initial_capital: number
  monthly_contribution: number
  yearly_contribution: number
  cooldown_days: number
  start_date?: string | null
  end_date?: string | null
  fill_mode?: FillMode   // execution model (default: next-open, the realistic fill)
}

export interface Trade {
  entry_date: string; exit_date?: string; current_date?: string
  top1_ticker?: string; buy_trigger?: string; sell_reason?: string
  return_pct?: number; max_drawdown_pct?: number; accumulated?: number
  qqq_entry_val?: number; qqq_exit_val?: number
  stock_entry_val?: number; stock_exit_val?: number
  tqqq_entry_val?: number; tqqq_exit_val?: number
  spy_entry_val?: number; spy_exit_val?: number
  soxx_entry_val?: number; soxx_exit_val?: number
  stock_active?: boolean; tqqq_active?: boolean
  spy_active?: boolean; soxx_active?: boolean
}

export interface MetricPoint { date: string; value: number }
export interface AnnualReturn { year: number; strategy: number; benchmark: number }

export interface BacktestResult {
  metrics: { strategy: Record<string, string>; benchmark: Record<string, string> }
  chart_data: {
    portfolio: MetricPoint[]; benchmark: MetricPoint[]
    breadth: MetricPoint[]; ndx: MetricPoint[]
  }
  trades: Trade[]
  open_trade: Trade | null
  sell_proximity: SellProximity | null
  annual_returns: AnnualReturn[]
  total_contrib: number
  weights: { qqq: number; stock: number; tqqq: number; spy: number; soxx: number }
  params: Record<string, number>
}

export interface SellProximity {
  price_rise_pct: number; breadth_fall_pts: number; breadth_current: number
  price_rise_needed: number; breadth_fall_needed: number; breadth_cap: number
  price_rise_met: boolean; breadth_fall_met: boolean; breadth_cap_met: boolean
  // Climax top: both must fire within climax_window days (post-entry)
  macd_days_ago: number | null; ext_days_ago: number | null
  climax_window: number; climax_met: boolean
  // Trailing stop on NDX since entry
  ndx_high: number; ndx_current: number
  drop_from_high_pct: number; trail_stop_pct: number; trail_met: boolean
}

// safe: get value from a price map, returns NaN if missing
function safe(map: Map<string, number> | null, date: string): number {
  if (!map) return NaN
  return map.get(date) ?? NaN
}

// Compute rolling MA (returns array of same length, null for first window-1 entries)
function rollingMean(values: number[], window: number): (number | null)[] {
  const result: (number | null)[] = []
  for (let i = 0; i < values.length; i++) {
    if (i < window - 1) { result.push(null); continue }
    let sum = 0
    for (let j = i - window + 1; j <= i; j++) sum += values[j]
    result.push(sum / window)
  }
  return result
}

export function prepareData(
  ndxPrices: [string, number][],  // [date, price] sorted ascending
  breadthPrices: [string, number][],
  vixPrices: [string, number][],
  ndxOpens: [string, number][] = [],  // [date, open] — NDX open for next-day-open fills
): DayData[] {
  // Build maps for fast lookup
  const breadthMap = new Map(breadthPrices)
  const vixMap = new Map(vixPrices)
  const openMap = new Map(ndxOpens)

  // First pass: collect rows with breadth
  const raw: { date: string; price: number; open: number; breadth: number; vix: number }[] = []
  const prices: number[] = []

  for (const [date, price] of ndxPrices) {
    const breadth = breadthMap.get(date)
    if (breadth === undefined) continue
    const vix = vixMap.get(date) ?? NaN
    const open = openMap.get(date) ?? price   // fall back to close if open missing
    raw.push({ date, price, open, breadth, vix })
    prices.push(price)
  }

  // Compute MA200
  const ma200 = rollingMean(prices, MA200_WINDOW)

  // Climax-top components: MACD(12,26,9) bearish cross + extension above 10d MA
  const ma10 = rollingMean(prices, 10)
  const macdCross: boolean[] = new Array(prices.length).fill(false)
  const ext10Arr: boolean[] = new Array(prices.length).fill(false)
  let ema12 = prices[0] ?? 0
  let ema26 = prices[0] ?? 0
  let signal = 0
  let prevHist = 0
  for (let i = 0; i < prices.length; i++) {
    if (i > 0) {
      ema12 = prices[i] * (2 / 13) + ema12 * (1 - 2 / 13)
      ema26 = prices[i] * (2 / 27) + ema26 * (1 - 2 / 27)
    }
    const macd = ema12 - ema26
    signal = i === 0 ? macd : macd * (2 / 10) + signal * (1 - 2 / 10)
    const hist = macd - signal
    macdCross[i] = i > 0 && hist < 0 && prevHist >= 0
    prevHist = hist
    const m10 = ma10[i]
    ext10Arr[i] = m10 !== null && prices[i] / m10 - 1 >= EXT10_PCT / 100
  }

  // Trend re-entry: fresh close back above MA200 (price > ma200 today, <= yesterday)
  const ma200Recross: boolean[] = new Array(prices.length).fill(false)
  for (let i = 1; i < prices.length; i++) {
    const ma = ma200[i], maPrev = ma200[i - 1]
    ma200Recross[i] = ma !== null && maPrev !== null
      && prices[i] > ma && prices[i - 1] <= maPrev
  }

  // Forward-fill VIX (some dates may be missing)
  let lastVix = NaN
  for (let i = 0; i < raw.length; i++) {
    if (!isNaN(raw[i].vix)) lastVix = raw[i].vix
    else raw[i].vix = lastVix
  }

  // Second pass: compute signals
  const rows: DayData[] = []
  for (let i = 0; i < raw.length; i++) {
    const r = raw[i]
    const ma = ma200[i]

    const vix_vote   = isNaN(r.vix) ? true : r.vix > VIX_BUY_THRESH
    const ma200_vote = ma === null ? true : r.price > ma
    const vote_gate  = vix_vote || ma200_vote

    // Divergence signals (look back DIVERGENCE_WINDOW rows)
    let price_rose = false, breadth_fell = false
    if (i >= DIVERGENCE_WINDOW) {
      const past = raw[i - DIVERGENCE_WINDOW]
      const pr = (r.price - past.price) / past.price * 100
      const bf = past.breadth - r.breadth
      price_rose   = pr >= DIVERGENCE_PRICE_RISE
      breadth_fell = bf >= DIVERGENCE_BREADTH_FALL
    }

    rows.push({ date: r.date, price: r.price, open: r.open, breadth: r.breadth, vix: r.vix,
      ma200: ma, vix_vote, ma200_vote, vote_gate, price_rose, breadth_fell,
      macd_cross: macdCross[i], ext10: ext10Arr[i], ma200_recross: ma200Recross[i] })
  }

  return rows
}

// Replay the signal state machine over the rows BEFORE the backtest window
// to decide whether the strategy would already hold a position at the start
// date. Mirrors the entry/exit conditions in runStrategy (position only).
export function isInPositionAt(allData: DayData[], startDate: string, cooldownDays: number): boolean {
  let position: 'IN' | 'OUT' = 'OUT'
  let cooldownUntil: string | null = null
  let ndxHigh = 0
  let macdAge = Number.MAX_SAFE_INTEGER
  let extAge  = Number.MAX_SAFE_INTEGER
  let lastSellReason: string | null = null
  let lastExitPrice: number | null = null

  for (const row of allData) {
    if (row.date >= startDate) break

    if (position === 'OUT') {
      const cooldownOk = cooldownUntil === null || row.date > cooldownUntil
      const washoutBuy = !isNaN(row.breadth) && row.breadth < BUY_B200_THRESH && row.vote_gate
      const recrossOk  = lastSellReason === 'climax-top'
        || (lastExitPrice !== null && row.price > lastExitPrice)
      const trendBuy   = row.ma200_recross && recrossOk
      if (cooldownOk && (washoutBuy || trendBuy)) {
        position = 'IN'
        ndxHigh  = row.price
        macdAge  = Number.MAX_SAFE_INTEGER
        extAge   = Number.MAX_SAFE_INTEGER
      }
    } else {
      ndxHigh = Math.max(ndxHigh, row.price)
      macdAge = row.macd_cross ? 0 : macdAge + 1
      extAge  = row.ext10      ? 0 : extAge + 1
      const bearishDiv = row.price_rose && row.breadth_fell && row.breadth < DIVERGENCE_BREADTH_CAP
      const climax     = macdAge < CLIMAX_VOTE_WINDOW && extAge < CLIMAX_VOTE_WINDOW
      const trailHit   = row.price <= ndxHigh * (1 - TRAILING_STOP_PCT / 100)

      if (bearishDiv || climax || trailHit) {
        position = 'OUT'
        lastSellReason = bearishDiv ? 'bearish-divergence' : climax ? 'climax-top' : 'trailing-stop'
        lastExitPrice = row.price
        const cooldownDate = new Date(row.date)
        cooldownDate.setDate(cooldownDate.getDate() + cooldownDays)
        cooldownUntil = cooldownDate.toISOString().slice(0, 10)
      }
    }
  }

  return position === 'IN'
}

export function runStrategy(
  data: DayData[],
  topHoldings: Map<number, string>,    // year -> ticker
  alignedStocks: Map<string, Map<string, number>>,  // ticker -> date->price
  alignedTqqq: Map<string, number> | null,
  alignedSpy: Map<string, number> | null,
  alignedSoxx: Map<string, number> | null,
  params: BacktestParams,
  wQQQ: number, wStock: number, wTQQQ: number, wSPY: number, wSOXX: number,
  forceEntryFirstDay = false,  // strategy was already IN before the window start
  executionLag = 1,            // bars between signal and fill (0 = same-day look-ahead)
  fillOn: 'open' | 'close' = 'open',
  alignedStocksOpen: Map<string, Map<string, number>> = new Map(),
  alignedTqqqOpen: Map<string, number> | null = null,
  alignedSpyOpen: Map<string, number> | null = null,
  alignedSoxxOpen: Map<string, number> | null = null,
): { portfolio: [string, number][]; trades: Trade[]; openTrade: Trade | null; totalContrib: number } {
  // Fill price for one leg: the fill-bar open when available, else close.
  const fillPx = (closeMap: Map<string, number> | null, openMap: Map<string, number> | null, date: string): number => {
    if (fillOn === 'open' && openMap) {
      const v = safe(openMap, date)
      if (!isNaN(v)) return v
    }
    return safe(closeMap, date)
  }

  let position: 'IN' | 'OUT' = 'OUT'
  let cooldownUntil: string | null = null
  let lastSellReason: string | null = null
  let lastExitPrice: number | null = null
  const trades: Trade[] = []
  const values: [string, number][] = []

  // Shares held
  let qqqShares = 0, stockShares = 0, tqqqShares = 0, spyShares = 0, soxxShares = 0
  let holdingTicker: string | null = null
  let entryDate: string | null = null
  let buyTrigger = ''
  let tradeLowVal = 0
  // Climax/trail state — signals only count AFTER entry
  let ndxHigh = 0
  let macdAge = Number.MAX_SAFE_INTEGER
  let extAge  = Number.MAX_SAFE_INTEGER

  // Buckets
  const ic = params.initial_capital
  let qqqBucket   = ic * wQQQ
  let stockBucket = ic * wStock
  let tqqqBucket  = ic * wTQQQ
  let spyBucket   = ic * wSPY
  let soxxBucket  = ic * wSOXX

  let qqqQqqFrac = 1.0, stockQqqFrac = 0, tqqqQqqFrac = 0, spyQqqFrac = 0, soxxQqqFrac = 0
  let stockActive = false, tqqqActive = false, spyActive = false, soxxActive = false

  let cashReserve = 0, totalContrib = 0
  let prevMonth: number | null = null, prevYear: number | null = null

  // Bucket-level entry tracking
  let qqqEntryVal = 0, stockEntryVal = 0, tqqqEntryVal = 0, spyEntryVal = 0, soxxEntryVal = 0

  for (let idx = 0; idx < data.length; idx++) {
    const row = data[idx]
    const date = row.date
    const dateMonth = parseInt(date.slice(5, 7))
    const dateYear  = parseInt(date.slice(0, 4))

    // Contributions
    if (prevMonth !== null) {
      let contrib = 0
      if (params.monthly_contribution > 0 && dateMonth !== prevMonth) contrib += params.monthly_contribution
      if (params.yearly_contribution > 0 && dateYear !== prevYear)    contrib += params.yearly_contribution
      if (contrib > 0) { cashReserve += contrib; totalContrib += contrib }
    }
    prevMonth = dateMonth; prevYear = dateYear

    const ndxPrice = row.price
    // Signal bar: the close `executionLag` bars ago (what was actually known when
    // the order was placed). lag=0 → today's row (legacy same-day look-ahead).
    const sig = executionLag > 0
      ? (idx - executionLag >= 0 ? data[idx - executionLag] : null)
      : row
    // Fill price for the NDX/QQQ leg on this bar (open, else close).
    const ndxFill = (fillOn === 'open' && !isNaN(row.open)) ? row.open : ndxPrice

    if (position === 'OUT') {
      // Determine if cooldown has passed
      const cooldownOk = cooldownUntil === null || date > cooldownUntil
      const carryIn = forceEntryFirstDay && idx === 0
      const washoutBuy = !!sig && !isNaN(sig.breadth) && sig.breadth < BUY_B200_THRESH && sig.vote_gate
      // Trend re-entry on a fresh MA200 recross (NDX): rejoin when the last exit
      // was a climax-top or the NDX price is back above the price we last sold at
      // (market proved the exit premature). Recrosses still below the prior exit
      // stay filtered as failed bounces in a real downtrend.
      const recrossOk = lastSellReason === 'climax-top'
        || (lastExitPrice !== null && !!sig && sig.price > lastExitPrice)
      const trendBuy  = !!sig && sig.ma200_recross && recrossOk
      const doBuy = carryIn || (!!sig && cooldownOk && (washoutBuy || trendBuy))

      if (doBuy) {
        const year = dateYear
        const stockTicker = topHoldings.get(year) ?? topHoldings.get(year - 1) ?? null

        const stockPx = fillPx(stockTicker ? (alignedStocks.get(stockTicker) ?? null) : null,
                               stockTicker ? (alignedStocksOpen.get(stockTicker) ?? null) : null, date)
        const tqqqPx  = fillPx(alignedTqqq, alignedTqqqOpen, date)
        const spyPx   = fillPx(alignedSpy,  alignedSpyOpen,  date)
        const soxxPx  = fillPx(alignedSoxx, alignedSoxxOpen, date)

        // Sweep cash into buckets
        if (cashReserve > 0) {
          qqqBucket   += cashReserve * wQQQ
          stockBucket += cashReserve * wStock
          tqqqBucket  += cashReserve * wTQQQ
          spyBucket   += cashReserve * wSPY
          soxxBucket  += cashReserve * wSOXX
          cashReserve = 0
        }

        // Commission
        const totalPre = qqqBucket + stockBucket + tqqqBucket + spyBucket + soxxBucket
        const commScale = totalPre > 0 ? (totalPre - COMMISSION) / totalPre : 1.0
        qqqBucket   *= commScale; stockBucket *= commScale; tqqqBucket  *= commScale
        spyBucket   *= commScale; soxxBucket  *= commScale

        stockActive = !isNaN(stockPx)
        tqqqActive  = !isNaN(tqqqPx)
        spyActive   = !isNaN(spyPx)
        soxxActive  = !isNaN(soxxPx)

        const effQQQ   = qqqBucket + (!stockActive ? stockBucket : 0) + (!tqqqActive ? tqqqBucket : 0) + (!spyActive ? spyBucket : 0) + (!soxxActive ? soxxBucket : 0)
        const effStock = stockActive ? stockBucket : 0
        const effTQQQ  = tqqqActive  ? tqqqBucket  : 0
        const effSPY   = spyActive   ? spyBucket   : 0
        const effSOXX  = soxxActive  ? soxxBucket  : 0

        if (effQQQ > 0) {
          qqqQqqFrac   = qqqBucket   / effQQQ
          stockQqqFrac = !stockActive ? stockBucket / effQQQ : 0
          tqqqQqqFrac  = !tqqqActive  ? tqqqBucket  / effQQQ : 0
          spyQqqFrac   = !spyActive   ? spyBucket   / effQQQ : 0
          soxxQqqFrac  = !soxxActive  ? soxxBucket  / effQQQ : 0
        } else {
          qqqQqqFrac = 1; stockQqqFrac = tqqqQqqFrac = spyQqqFrac = soxxQqqFrac = 0
        }

        const qqqEntryPx   = ndxFill * (1 + SLIPPAGE)
        const stockEntryPx = stockActive ? stockPx * (1 + SLIPPAGE) : 0
        const tqqqEntryPx  = tqqqActive  ? tqqqPx  * (1 + SLIPPAGE) : 0
        const spyEntryPx   = spyActive   ? spyPx   * (1 + SLIPPAGE) : 0
        const soxxEntryPx  = soxxActive  ? soxxPx  * (1 + SLIPPAGE) : 0

        qqqShares   = effQQQ   / qqqEntryPx
        stockShares = stockEntryPx > 0 ? effStock / stockEntryPx : 0
        tqqqShares  = tqqqEntryPx  > 0 ? effTQQQ  / tqqqEntryPx  : 0
        spyShares   = spyEntryPx   > 0 ? effSPY   / spyEntryPx   : 0
        soxxShares  = soxxEntryPx  > 0 ? effSOXX  / soxxEntryPx  : 0

        holdingTicker = stockTicker
        entryDate     = date
        tradeLowVal   = effQQQ + effStock + effTQQQ + effSPY + effSOXX
        ndxHigh       = sig ? sig.price : ndxPrice
        macdAge       = Number.MAX_SAFE_INTEGER
        extAge        = Number.MAX_SAFE_INTEGER
        position      = 'IN'
        buyTrigger    = (carryIn || !sig)
          ? 'carry-in'
          : (sig.vix_vote ? 'VIX' : '') + (sig.vix_vote && sig.ma200_vote ? '+' : '') + (sig.ma200_vote ? 'MA200' : '')

        qqqEntryVal  = qqqBucket;   stockEntryVal = stockBucket
        tqqqEntryVal = tqqqBucket;  spyEntryVal   = spyBucket;   soxxEntryVal = soxxBucket
      }

    } else { // IN
      // Exit signals track the signal series (close as of `executionLag` bars ago).
      ndxHigh = Math.max(ndxHigh, sig ? sig.price : ndxPrice)
      macdAge = sig ? (sig.macd_cross ? 0 : macdAge + 1) : macdAge + 1
      extAge  = sig ? (sig.ext10      ? 0 : extAge + 1)  : extAge + 1
      const sigNdx = sig ? sig.price : ndxPrice
      const bearishDiv = !!sig && sig.price_rose && sig.breadth_fell && sig.breadth < DIVERGENCE_BREADTH_CAP
      const climax     = macdAge < CLIMAX_VOTE_WINDOW && extAge < CLIMAX_VOTE_WINDOW
      const trailHit   = sigNdx <= ndxHigh * (1 - TRAILING_STOP_PCT / 100)
      const sellReason = bearishDiv ? 'bearish-divergence'
        : climax ? 'climax-top'
        : trailHit ? 'trailing-stop'
        : null

      if (sellReason) {
        const stockPxExit = fillPx(holdingTicker ? (alignedStocks.get(holdingTicker) ?? null) : null,
                                   holdingTicker ? (alignedStocksOpen.get(holdingTicker) ?? null) : null, date)
        const tqqqPxExit  = fillPx(alignedTqqq, alignedTqqqOpen, date)
        const spyPxExit   = fillPx(alignedSpy,  alignedSpyOpen,  date)
        const soxxPxExit  = fillPx(alignedSoxx, alignedSoxxOpen, date)

        const spx  = !isNaN(stockPxExit) ? stockPxExit : 0
        const tpx  = !isNaN(tqqqPxExit)  ? tqqqPxExit  : 0
        const spyx = !isNaN(spyPxExit)   ? spyPxExit   : 0
        const sxx  = !isNaN(soxxPxExit)  ? soxxPxExit  : 0

        const grossQQQ   = qqqShares   * ndxFill  * (1 - SLIPPAGE)
        const grossStock = stockShares * spx        * (1 - SLIPPAGE)
        const grossTQQQ  = tqqqShares  * tpx        * (1 - SLIPPAGE)
        const grossSPY   = spyShares   * spyx       * (1 - SLIPPAGE)
        const grossSOXX  = soxxShares  * sxx        * (1 - SLIPPAGE)
        const grossTotal = grossQQQ + grossStock + grossTQQQ + grossSPY + grossSOXX
        const commFrac   = grossTotal > 0 ? COMMISSION / grossTotal : 0

        qqqBucket   = (grossQQQ * qqqQqqFrac)                        * (1 - commFrac)
        stockBucket = (grossQQQ * stockQqqFrac + grossStock)          * (1 - commFrac)
        tqqqBucket  = (grossQQQ * tqqqQqqFrac  + grossTQQQ)           * (1 - commFrac)
        spyBucket   = (grossQQQ * spyQqqFrac   + grossSPY)            * (1 - commFrac)
        soxxBucket  = (grossQQQ * soxxQqqFrac  + grossSOXX)           * (1 - commFrac)
        const totalProc = qqqBucket + stockBucket + tqqqBucket + spyBucket + soxxBucket

        const qqqExitVal  = qqqBucket,  stockExitVal = stockBucket
        const tqqqExitVal = tqqqBucket, spyExitVal   = spyBucket, soxxExitVal = soxxBucket
        const entryVal    = qqqEntryVal + stockEntryVal + tqqqEntryVal + spyEntryVal + soxxEntryVal
        const grossRet    = entryVal > 0 ? (totalProc - entryVal) / entryVal : 0
        const maxDdPct    = entryVal > 0 ? (tradeLowVal - entryVal) / entryVal * 100 : 0

        // Add cooldown_days to date
        const cooldownDate = new Date(date)
        cooldownDate.setDate(cooldownDate.getDate() + params.cooldown_days)
        cooldownUntil = cooldownDate.toISOString().slice(0, 10)
        lastSellReason = sellReason
        lastExitPrice = ndxFill

        trades.push({
          entry_date: entryDate!, exit_date: date,
          return_pct: grossRet * 100, max_drawdown_pct: maxDdPct,
          accumulated: totalProc, buy_trigger: buyTrigger, sell_reason: sellReason,
          top1_ticker: holdingTicker ?? undefined, stock_active: stockActive,
          tqqq_active: tqqqActive, spy_active: spyActive, soxx_active: soxxActive,
          qqq_entry_val: qqqEntryVal, qqq_exit_val: qqqExitVal,
          stock_entry_val: stockEntryVal, stock_exit_val: stockExitVal,
          tqqq_entry_val: tqqqEntryVal, tqqq_exit_val: tqqqExitVal,
          spy_entry_val: spyEntryVal, spy_exit_val: spyExitVal,
          soxx_entry_val: soxxEntryVal, soxx_exit_val: soxxExitVal,
        })

        position = 'OUT'
        qqqShares = stockShares = tqqqShares = spyShares = soxxShares = 0
      }
    }

    // Mark-to-market
    if (position === 'IN') {
      const sn   = !isNaN(safe(holdingTicker ? (alignedStocks.get(holdingTicker) ?? null) : null, date)) ? safe(holdingTicker ? (alignedStocks.get(holdingTicker) ?? null) : null, date) : 0
      const tn   = !isNaN(safe(alignedTqqq, date))  ? safe(alignedTqqq, date)  : 0
      const spyn = !isNaN(safe(alignedSpy,  date))  ? safe(alignedSpy,  date)  : 0
      const sxn  = !isNaN(safe(alignedSoxx, date))  ? safe(alignedSoxx, date)  : 0

      const qqqCur   = qqqShares   * ndxPrice
      const stockCur = stockShares * sn
      const tqqqCur  = tqqqShares  * tn
      const spyCur   = spyShares   * spyn
      const soxxCur  = soxxShares  * sxn
      const curVal   = qqqCur + stockCur + tqqqCur + spyCur + soxxCur

      tradeLowVal = Math.min(tradeLowVal, curVal)
      values.push([date, curVal + cashReserve])
    } else {
      values.push([date, qqqBucket + stockBucket + tqqqBucket + spyBucket + soxxBucket + cashReserve])
    }
  }

  // Open trade
  let openTrade: Trade | null = null
  if (position === 'IN' && data.length > 0) {
    const last     = data[data.length - 1]
    const lastDate = last.date
    const lastNdx  = last.price
    const ls   = !isNaN(safe(holdingTicker ? (alignedStocks.get(holdingTicker) ?? null) : null, lastDate)) ? safe(holdingTicker ? (alignedStocks.get(holdingTicker) ?? null) : null, lastDate) : 0
    const lt   = !isNaN(safe(alignedTqqq, lastDate))  ? safe(alignedTqqq, lastDate)  : 0
    const lspy = !isNaN(safe(alignedSpy,  lastDate))  ? safe(alignedSpy,  lastDate)  : 0
    const lsx  = !isNaN(safe(alignedSoxx, lastDate))  ? safe(alignedSoxx, lastDate)  : 0

    const qqqCv   = qqqShares   * lastNdx
    const stockCv = stockShares * ls
    const tqqqCv  = tqqqShares  * lt
    const spyCv   = spyShares   * lspy
    const soxxCv  = soxxShares  * lsx
    const lastVal = qqqCv + stockCv + tqqqCv + spyCv + soxxCv

    const qqqBCur   = qqqCv  * qqqQqqFrac
    const stockBCur = qqqCv  * stockQqqFrac + stockCv
    const tqqqBCur  = qqqCv  * tqqqQqqFrac  + tqqqCv
    const spyBCur   = qqqCv  * spyQqqFrac   + spyCv
    const soxxBCur  = qqqCv  * soxxQqqFrac  + soxxCv
    const entryVal  = qqqEntryVal + stockEntryVal + tqqqEntryVal + spyEntryVal + soxxEntryVal

    openTrade = {
      entry_date: entryDate!, current_date: lastDate,
      return_pct: entryVal > 0 ? (lastVal - entryVal) / entryVal * 100 : 0,
      max_drawdown_pct: entryVal > 0 ? (tradeLowVal - entryVal) / entryVal * 100 : 0,
      accumulated: lastVal + cashReserve, buy_trigger: buyTrigger,
      top1_ticker: holdingTicker ?? undefined, stock_active: stockActive,
      tqqq_active: tqqqActive, spy_active: spyActive, soxx_active: soxxActive,
      qqq_entry_val: qqqEntryVal, qqq_exit_val: qqqBCur,
      stock_entry_val: stockEntryVal, stock_exit_val: stockBCur,
      tqqq_entry_val: tqqqEntryVal, tqqq_exit_val: tqqqBCur,
      spy_entry_val: spyEntryVal, spy_exit_val: spyBCur,
      soxx_entry_val: soxxEntryVal, soxx_exit_val: soxxBCur,
    }
  }

  return { portfolio: values, trades, openTrade, totalContrib }
}

export function runBenchmark(data: DayData[], initialCapital: number): [string, number][] {
  if (data.length === 0) return []
  const first = data[0].price
  return data.map(row => [row.date, initialCapital * row.price / first])
}

export function computeMetrics(
  values: [string, number][],
  trades?: Trade[],
): Record<string, string> {
  if (values.length < 2) return {}
  const vals = values.map(v => v[1])
  const dates = values.map(v => v[0])

  const first = vals[0], last = vals[vals.length - 1]
  const startMs = new Date(dates[0]).getTime()
  const endMs   = new Date(dates[dates.length - 1]).getTime()
  const years   = (endMs - startMs) / (365.25 * 24 * 3600 * 1000)
  const tr      = last / first - 1
  const cagr    = Math.pow(last / first, 1 / years) - 1

  // Daily returns
  const dr: number[] = []
  for (let i = 1; i < vals.length; i++) dr.push(vals[i] / vals[i-1] - 1)
  const mean = dr.reduce((a, b) => a + b, 0) / dr.length
  const std  = Math.sqrt(dr.reduce((a, b) => a + (b - mean) ** 2, 0) / dr.length)
  const sh   = std > 0 ? mean / std * Math.sqrt(252) : 0

  // Max drawdown
  let peak = vals[0], mdd = 0
  for (const v of vals) {
    if (v > peak) peak = v
    const dd = (v - peak) / peak
    if (dd < mdd) mdd = dd
  }

  const fmt = (n: number, pct: boolean) =>
    pct ? `${(n >= 0 ? '+' : '')}${(n * 100).toFixed(1)}%` : n.toFixed(2)

  const m: Record<string, string> = {
    'Total Return': fmt(tr, true),
    'CAGR':         fmt(cagr, true),
    'Max Drawdown': fmt(mdd, true),
    'Sharpe Ratio': sh.toFixed(2),
    'Final Value':  `$${Math.round(last).toLocaleString()}`,
  }

  if (trades !== undefined) {
    const n    = trades.length
    const wins = trades.filter(t => (t.return_pct ?? 0) > 0).length
    const inDays = trades.reduce((sum, t) => {
      if (!t.exit_date || !t.entry_date) return sum
      return sum + (new Date(t.exit_date).getTime() - new Date(t.entry_date).getTime()) / 86400000
    }, 0)
    const totDays = (endMs - startMs) / 86400000
    m['# Trades']       = String(n)
    m['Win Rate']        = n > 0 ? `${(wins/n*100).toFixed(1)}%` : '--'
    m['Time in Market']  = totDays > 0 ? `${(inDays/totDays*100).toFixed(1)}%` : '--'
  }
  return m
}

export function runBacktest(
  allData: DayData[],
  topHoldings: Map<number, string>,
  alignedStocks: Map<string, Map<string, number>>,
  alignedTqqq: Map<string, number> | null,
  alignedSpy: Map<string, number> | null,
  alignedSoxx: Map<string, number> | null,
  params: BacktestParams,
  alignedStocksOpen: Map<string, Map<string, number>> = new Map(),
  alignedTqqqOpen: Map<string, number> | null = null,
  alignedSpyOpen: Map<string, number> | null = null,
  alignedSoxxOpen: Map<string, number> | null = null,
): BacktestResult {
  const total = params.qqq + params.stock + params.tqqq + params.spy + params.soxx
  const wQQQ   = total > 0 ? params.qqq   / total : 0
  const wStock = total > 0 ? params.stock / total : 0
  const wTQQQ  = total > 0 ? params.tqqq  / total : 0
  const wSPY   = total > 0 ? params.spy   / total : 0
  const wSOXX  = total > 0 ? params.soxx  / total : 0

  // Date filtering
  let data = allData
  if (params.start_date) data = data.filter(r => r.date >= params.start_date!)
  if (params.end_date)   data = data.filter(r => r.date <= params.end_date!)

  // If the start date falls between a buy and its sell signal (strategy would
  // already be holding), enter the market on the first day of the window
  const carryIn = params.start_date
    ? isInPositionAt(allData, params.start_date, params.cooldown_days)
    : false

  // Slice aligned maps to matching dates
  const dateSet = new Set(data.map(r => r.date))
  function sliceMap(m: Map<string, number> | null): Map<string, number> | null {
    if (!m) return null
    const out = new Map<string, number>()
    for (const [d, v] of m) if (dateSet.has(d)) out.set(d, v)
    return out
  }
  const slicedStocks = new Map<string, Map<string, number>>()
  for (const [t, m] of alignedStocks) slicedStocks.set(t, sliceMap(m)!)
  const slicedStocksOpen = new Map<string, Map<string, number>>()
  for (const [t, m] of alignedStocksOpen) slicedStocksOpen.set(t, sliceMap(m)!)

  const { lag, fillOn } = fillModeToParams(params.fill_mode ?? DEFAULT_FILL_MODE)

  const { portfolio, trades, openTrade, totalContrib } = runStrategy(
    data, topHoldings, slicedStocks, sliceMap(alignedTqqq), sliceMap(alignedSpy), sliceMap(alignedSoxx),
    params, wQQQ, wStock, wTQQQ, wSPY, wSOXX, carryIn,
    lag, fillOn, slicedStocksOpen, sliceMap(alignedTqqqOpen), sliceMap(alignedSpyOpen), sliceMap(alignedSoxxOpen),
  )
  const benchmark = runBenchmark(data, params.initial_capital)

  const stratMetrics = computeMetrics(portfolio, trades)
  const benchMetrics = computeMetrics(benchmark)

  // Annual returns
  const annual_returns: AnnualReturn[] = []
  const portByYear = new Map<number, number[]>()
  const benchByYear = new Map<number, number[]>()
  for (const [d, v] of portfolio) {
    const y = parseInt(d.slice(0, 4))
    if (!portByYear.has(y)) portByYear.set(y, [])
    portByYear.get(y)!.push(v)
  }
  for (const [d, v] of benchmark) {
    const y = parseInt(d.slice(0, 4))
    if (!benchByYear.has(y)) benchByYear.set(y, [])
    benchByYear.get(y)!.push(v)
  }
  for (const y of [...portByYear.keys()].sort()) {
    const sp = portByYear.get(y)!
    const bp = benchByYear.get(y) ?? []
    if (sp.length < 2 || bp.length < 2) continue
    annual_returns.push({
      year: y,
      strategy:  parseFloat(((sp[sp.length-1] / sp[0] - 1) * 100).toFixed(2)),
      benchmark: parseFloat(((bp[bp.length-1] / bp[0] - 1) * 100).toFixed(2)),
    })
  }

  // Sell proximity
  let sell_proximity: SellProximity | null = null
  if (openTrade && data.length > 0) {
    const last = data[data.length - 1]
    const pastIdx = Math.max(0, data.length - 1 - DIVERGENCE_WINDOW)
    const past = data[pastIdx]
    const pr   = (last.price - past.price) / past.price * 100
    const bf   = past.breadth - last.breadth
    const capOk = last.breadth < DIVERGENCE_BREADTH_CAP

    // Replay post-entry state for climax-top and trailing-stop proximity
    // (mirrors runStrategy: signals only count AFTER the entry day)
    const entryIdx = data.findIndex(r => r.date === openTrade!.entry_date)
    let ndxHigh = entryIdx >= 0 ? data[entryIdx].price : last.price
    let macdAge = Number.MAX_SAFE_INTEGER
    let extAge  = Number.MAX_SAFE_INTEGER
    for (let i = Math.max(entryIdx, 0) + 1; i < data.length; i++) {
      ndxHigh = Math.max(ndxHigh, data[i].price)
      macdAge = data[i].macd_cross ? 0 : macdAge + 1
      extAge  = data[i].ext10      ? 0 : extAge + 1
    }
    const neverFired = data.length  // ages beyond series length mean "never"
    const macdDaysAgo = macdAge > neverFired ? null : macdAge
    const extDaysAgo  = extAge  > neverFired ? null : extAge
    const dropPct = ndxHigh > 0 ? (1 - last.price / ndxHigh) * 100 : 0

    sell_proximity = {
      price_rise_pct: parseFloat(pr.toFixed(2)),
      breadth_fall_pts: parseFloat(bf.toFixed(2)),
      breadth_current: parseFloat(last.breadth.toFixed(2)),
      price_rise_needed: DIVERGENCE_PRICE_RISE,
      breadth_fall_needed: DIVERGENCE_BREADTH_FALL,
      breadth_cap: DIVERGENCE_BREADTH_CAP,
      price_rise_met: pr >= DIVERGENCE_PRICE_RISE,
      breadth_fall_met: bf >= DIVERGENCE_BREADTH_FALL,
      breadth_cap_met: capOk,
      macd_days_ago: macdDaysAgo,
      ext_days_ago: extDaysAgo,
      climax_window: CLIMAX_VOTE_WINDOW,
      climax_met: macdAge < CLIMAX_VOTE_WINDOW && extAge < CLIMAX_VOTE_WINDOW,
      ndx_high: parseFloat(ndxHigh.toFixed(2)),
      ndx_current: parseFloat(last.price.toFixed(2)),
      drop_from_high_pct: parseFloat(dropPct.toFixed(2)),
      trail_stop_pct: TRAILING_STOP_PCT,
      trail_met: dropPct >= TRAILING_STOP_PCT,
    }
  }

  const weights = { qqq: wQQQ*100, stock: wStock*100, tqqq: wTQQQ*100, spy: wSPY*100, soxx: wSOXX*100 }

  return {
    metrics: { strategy: stratMetrics, benchmark: benchMetrics },
    chart_data: {
      portfolio: portfolio.map(([d,v]) => ({ date: d, value: parseFloat(v.toFixed(4)) })),
      benchmark: benchmark.map(([d,v]) => ({ date: d, value: parseFloat(v.toFixed(4)) })),
      breadth:   data.map(r => ({ date: r.date, value: r.breadth })),
      ndx:       data.map(r => ({ date: r.date, value: r.price })),
    },
    trades, open_trade: openTrade,
    sell_proximity, annual_returns,
    total_contrib: parseFloat(totalContrib.toFixed(2)),
    weights,
    params: {
      buy_b200_thresh: BUY_B200_THRESH, vix_buy_thresh: VIX_BUY_THRESH,
      divergence_window: DIVERGENCE_WINDOW, divergence_price_rise: DIVERGENCE_PRICE_RISE,
      divergence_breadth_fall: DIVERGENCE_BREADTH_FALL, divergence_breadth_cap: DIVERGENCE_BREADTH_CAP,
      ext10_pct: EXT10_PCT, climax_vote_window: CLIMAX_VOTE_WINDOW,
      trailing_stop_pct: TRAILING_STOP_PCT,
    },
  }
}
