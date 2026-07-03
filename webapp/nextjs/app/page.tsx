'use client'
import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import Sidebar from '@/components/Sidebar'
import MetricCards from '@/components/MetricCards'
import SellSignalPanel from '@/components/SellSignalPanel'
import { BacktestResponse, FormState } from '@/lib/types'
import { fmtDollar, fmtPct, pctCls, fmtHeld } from '@/lib/utils'
import { loadAppData, AppData } from '@/lib/loadData'
import { runBacktest as runBacktestEngine } from '@/lib/backtest'

const GrowthChart = dynamic(() => import('@/components/charts/GrowthChart'), { ssr: false })
const AnnualChart = dynamic(() => import('@/components/charts/AnnualChart'), { ssr: false })
const DrawdownChart = dynamic(() => import('@/components/charts/DrawdownChart'), { ssr: false })
const SignalsChart = dynamic(() => import('@/components/charts/SignalsChart'), { ssr: false })

type Tab = 'growth' | 'annual' | 'drawdown' | 'signals' | 'metrics' | 'trades' | 'sell'
type UIState = 'empty' | 'loading' | 'error' | 'results'

const TABS: { id: Tab; label: string }[] = [
  { id: 'growth', label: 'Portfolio Growth' },
  { id: 'annual', label: 'Annual Returns' },
  { id: 'drawdown', label: 'Drawdown' },
  { id: 'signals', label: 'Market Signals' },
  { id: 'metrics', label: 'Metrics' },
  { id: 'trades', label: 'Trade Log' },
  { id: 'sell', label: 'Sell Signal' },
]

const DEFAULT_FORM: FormState = {
  qqq: 60, stock: 30, tqqq: 10, spy: 0, soxx: 0,
  initial_capital: 10000, monthly_contribution: 0, yearly_contribution: 0,
  cooldown_days: 15, start_date: '', end_date: '',
}

export default function Page() {
  const [form, setForm] = useState<FormState>(DEFAULT_FORM)
  const [result, setResult] = useState<BacktestResponse | null>(null)
  const [uiState, setUiState] = useState<UIState>('empty')
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState<Tab>('growth')
  const [scale, setScale] = useState<'linear' | 'log'>('linear')
  const [appData, setAppData] = useState<AppData | null>(null)
  const [dataLoading, setDataLoading] = useState(true)
  const [dataError, setDataError] = useState('')

  useEffect(() => {
    loadAppData()
      .then(d => { setAppData(d); setDataLoading(false) })
      .catch(e => { setDataError((e as Error).message); setDataLoading(false) })
  }, [])

  function handleChange(k: keyof FormState, v: number | string) {
    setForm(prev => ({ ...prev, [k]: v }))
  }

  function runBacktest() {
    if (!appData) return
    setUiState('loading')
    try {
      const engineResult = runBacktestEngine(
        appData.data,
        appData.topHoldings,
        appData.alignedStocks,
        appData.alignedTqqq,
        appData.alignedSpy,
        appData.alignedSoxx,
        {
          qqq: form.qqq, stock: form.stock, tqqq: form.tqqq, spy: form.spy, soxx: form.soxx,
          initial_capital: form.initial_capital,
          monthly_contribution: form.monthly_contribution,
          yearly_contribution: form.yearly_contribution,
          cooldown_days: form.cooldown_days,
          start_date: form.start_date || null,
          end_date: form.end_date || null,
        },
      )
      // Map BacktestResult to BacktestResponse shape
      const data: BacktestResponse = {
        success: true,
        metrics: engineResult.metrics,
        chart_data: engineResult.chart_data,
        trades: engineResult.trades,
        open_trade: engineResult.open_trade ?? undefined,
        sell_proximity: engineResult.sell_proximity ?? undefined,
        annual_returns: engineResult.annual_returns,
        total_contrib: engineResult.total_contrib,
        weights: engineResult.weights,
        params: engineResult.params,
      }
      setResult(data)
      setActiveTab('growth')
      setUiState('results')
    } catch (e) {
      setError((e as Error).message)
      setUiState('error')
    }
  }

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      <Sidebar form={form} onChange={handleChange} onRun={runBacktest} loading={uiState === 'loading' || dataLoading} />

      <main style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
        {uiState === 'empty' && dataLoading && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center', color: 'var(--muted)', gap: 10 }}>
            <div style={{ fontSize: 44 }}>⏳</div>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>Loading market data…</h2>
            <p style={{ fontSize: 13, maxWidth: 380, lineHeight: 1.6 }}>Fetching CSV files. This only happens once.</p>
          </div>
        )}

        {uiState === 'empty' && !dataLoading && dataError && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center', gap: 10 }}>
            <div style={{ fontSize: 44 }}>⚠️</div>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>Failed to load data</h2>
            <p style={{ fontSize: 13, color: 'var(--red2)', maxWidth: 400 }}>{dataError}</p>
          </div>
        )}

        {uiState === 'empty' && !dataLoading && !dataError && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center', color: 'var(--muted)', gap: 10 }}>
            <div style={{ fontSize: 44 }}>📊</div>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>Configure Your Backtest</h2>
            <p style={{ fontSize: 13, maxWidth: 380, lineHeight: 1.6 }}>Set portfolio weights, capital, and date range in the sidebar, then click <strong>Run Backtest</strong>.</p>
          </div>
        )}

        {uiState === 'loading' && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center', color: 'var(--muted)', gap: 10 }}>
            <div style={{ fontSize: 44 }}>⏳</div>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>Running Backtest…</h2>
            <p style={{ fontSize: 13, maxWidth: 380, lineHeight: 1.6 }}>Fetching ETF data and simulating trades. This may take 15–30 seconds on first run.</p>
          </div>
        )}

        {uiState === 'error' && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', textAlign: 'center', gap: 10 }}>
            <div style={{ fontSize: 44 }}>⚠️</div>
            <h2 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>Error</h2>
            <p style={{ fontSize: 13, color: 'var(--red2)', maxWidth: 400 }}>{error}</p>
          </div>
        )}

        {uiState === 'results' && result && (
          <>
            {result.open_trade && (
              <div style={{ marginBottom: 14 }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7, background: '#f0fdf4', color: '#15803d', border: '1px solid #bbf7d0', borderRadius: 20, padding: '4px 12px', fontSize: 12, fontWeight: 600 }}>
                  <span className="dot-green" /> Currently in market
                </div>
              </div>
            )}

            <MetricCards data={result} />

            {result.total_contrib > 0 && (
              <div style={{ marginBottom: 16, padding: '10px 14px', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 8, fontSize: 12, color: '#92400e' }}>
                <strong>DCA summary:</strong> Initial ${form.initial_capital.toLocaleString()} + contributions ${result.total_contrib.toLocaleString()} = ${(form.initial_capital + result.total_contrib).toLocaleString()} total deployed
              </div>
            )}

            {/* Tabs panel */}
            <div style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, boxShadow: '0 1px 2px rgba(0,0,0,.05)', overflow: 'hidden' }}>
              {/* Tab bar */}
              <div style={{ display: 'flex', borderBottom: '1px solid var(--border)', padding: '0 4px', gap: 2, overflowX: 'auto' }}>
                {TABS.map(t => (
                  <button key={t.id} onClick={() => setActiveTab(t.id)}
                    style={{
                      fontSize: 12.5, fontWeight: 500, padding: '10px 14px',
                      borderBottom: activeTab === t.id ? '2px solid var(--primary)' : '2px solid transparent',
                      color: activeTab === t.id ? 'var(--primary)' : 'var(--muted)',
                      background: 'none', border: 'none',
                      cursor: 'pointer', whiteSpace: 'nowrap',
                    }}>
                    {t.label}
                  </button>
                ))}
              </div>

              {/* Tab content */}
              <div style={{ padding: 16 }}>
                {activeTab === 'growth' && (
                  <>
                    <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 6, marginBottom: 10 }}>
                      <span style={{ fontSize: 11, color: 'var(--muted)' }}>Scale:</span>
                      {(['linear', 'log'] as const).map(s => (
                        <button key={s} onClick={() => setScale(s)}
                          style={{
                            fontSize: 11, padding: '2px 8px', border: '1px solid var(--border)', borderRadius: 4, cursor: 'pointer',
                            background: scale === s ? 'var(--primary)' : '#fff',
                            color: scale === s ? '#fff' : 'var(--muted)',
                            borderColor: scale === s ? 'var(--primary)' : 'var(--border)',
                          }}>
                          {s.charAt(0).toUpperCase() + s.slice(1)}
                        </button>
                      ))}
                    </div>
                    <div style={{ position: 'relative', height: 380 }}>
                      <GrowthChart data={result} scale={scale} />
                    </div>
                  </>
                )}

                {activeTab === 'annual' && (
                  <>
                    <div style={{ position: 'relative', height: 250 }}>
                      <AnnualChart data={result.annual_returns} />
                    </div>
                    <div style={{ overflowX: 'auto', marginTop: 16 }}>
                      <table>
                        <thead><tr><th>Year</th><th>Portfolio</th><th>Buy &amp; Hold NDX</th><th>Difference</th></tr></thead>
                        <tbody>
                          {result.annual_returns.map(r => {
                            const diff = r.strategy - r.benchmark
                            return (
                              <tr key={r.year}>
                                <td>{r.year}</td>
                                <td className={pctCls(r.strategy)}>{fmtPct(r.strategy)}</td>
                                <td className={pctCls(r.benchmark)}>{fmtPct(r.benchmark)}</td>
                                <td className={pctCls(diff)}>{fmtPct(diff)}</td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                {activeTab === 'drawdown' && (
                  <div style={{ position: 'relative', height: 380 }}>
                    <DrawdownChart portfolio={result.chart_data.portfolio} benchmark={result.chart_data.benchmark} />
                  </div>
                )}

                {activeTab === 'signals' && (
                  <div style={{ position: 'relative', height: 380 }}>
                    <SignalsChart data={result} />
                  </div>
                )}

                {activeTab === 'metrics' && (
                  <table style={{ width: '100%' }}>
                    <thead><tr><th>Metric</th><th style={{ textAlign: 'right' }}>Portfolio</th><th style={{ textAlign: 'right' }}>Buy &amp; Hold NDX</th></tr></thead>
                    <tbody>
                      {Object.keys({ ...result.metrics.strategy, ...result.metrics.benchmark }).filter((k, i, a) => a.indexOf(k) === i).map(k => (
                        <tr key={k}>
                          <td style={{ color: 'var(--muted)' }}>{k}</td>
                          <td style={{ textAlign: 'right', fontWeight: 700 }}>{result.metrics.strategy[k] || '—'}</td>
                          <td style={{ textAlign: 'right' }}>{result.metrics.benchmark[k] || '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}

                {activeTab === 'trades' && (
                  <div style={{ overflowX: 'auto' }}>
                    <table>
                      <thead>
                        <tr>
                          <th>#</th><th>Entry</th><th>Exit</th><th>Held</th>
                          <th>Ticker</th><th>Buy Signal</th><th>Sell Signal</th><th>Return</th><th>Max DD</th>
                          <th>Entry $</th><th>Exit $</th><th>Net P&amp;L</th><th>Accum.</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...(result.trades || []), ...(result.open_trade ? [result.open_trade] : [])].map((t, i) => {
                          const open = !t.exit_date
                          const entryV = (t.qqq_entry_val||0)+(t.stock_entry_val||0)+(t.tqqq_entry_val||0)+(t.spy_entry_val||0)+(t.soxx_entry_val||0)
                          const exitV = (t.qqq_exit_val||0)+(t.stock_exit_val||0)+(t.tqqq_exit_val||0)+(t.spy_exit_val||0)+(t.soxx_exit_val||0)
                          const pnl = exitV - entryV
                          return (
                            <tr key={i}>
                              <td>{i+1}</td>
                              <td>{t.entry_date}</td>
                              <td>{open ? <span style={{ background: '#eff6ff', color: '#1d4ed8', fontSize: 10, padding: '1px 5px', borderRadius: 3, fontWeight: 700 }}>OPEN</span> : t.exit_date}</td>
                              <td>{open ? fmtHeld(t.entry_date, t.current_date!) : fmtHeld(t.entry_date, t.exit_date!)}</td>
                              <td><strong>{t.top1_ticker || '—'}</strong></td>
                              <td>{t.buy_trigger || '—'}</td>
                              <td>{open
                                ? <span style={{ color: 'var(--muted)', fontSize: 11 }}>holding</span>
                                : <span style={{
                                    background: t.sell_reason === 'trailing-stop' ? '#fef2f2' : t.sell_reason === 'climax-top' ? '#fff7ed' : '#f5f3ff',
                                    color: t.sell_reason === 'trailing-stop' ? '#dc2626' : t.sell_reason === 'climax-top' ? '#c2410c' : '#6d28d9',
                                    fontSize: 10, padding: '1px 6px', borderRadius: 3, fontWeight: 600, whiteSpace: 'nowrap',
                                  }}>{t.sell_reason || '—'}</span>}
                              </td>
                              <td className={pctCls(t.return_pct)}>{fmtPct(t.return_pct)}</td>
                              <td className={pctCls(t.max_drawdown_pct)}>{fmtPct(t.max_drawdown_pct)}</td>
                              <td>{fmtDollar(entryV)}</td>
                              <td>{fmtDollar(exitV)}</td>
                              <td className={pnl >= 0 ? 'pos' : 'neg'}>{pnl >= 0 ? '+' : ''}{fmtDollar(pnl)}</td>
                              <td>{fmtDollar(t.accumulated)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                {activeTab === 'sell' && (
                  <SellSignalPanel proximity={result.sell_proximity} hasOpenTrade={!!result.open_trade} />
                )}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  )
}
