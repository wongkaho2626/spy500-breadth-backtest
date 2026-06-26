'use client'
import { BacktestResponse } from '@/lib/types'

export default function MetricCards({ data }: { data: BacktestResponse }) {
  const sm = data.metrics.strategy, bm = data.metrics.benchmark
  const defs = [
    'Total Return', 'CAGR', 'Max Drawdown', 'Sharpe Ratio',
    'Final Value', '# Trades', 'Win Rate', 'Time in Market',
  ]

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 16 }}>
      {defs.map(key => {
        const sv = sm[key]; if (!sv) return null
        const bv = bm[key]
        const sn = parseFloat(sv.replace(/[%$,]/g, ''))
        let cls = ''
        if (key === 'Total Return' || key === 'CAGR') cls = sn > 0 ? 'pos' : sn < 0 ? 'neg' : ''
        if (key === 'Max Drawdown') cls = sn < 0 ? 'neg' : ''

        let bench = null
        if (bv && bv !== '--') {
          const bn = parseFloat(bv.replace(/[%$,]/g, ''))
          if (!isNaN(sn) && !isNaN(bn)) {
            const diff = sn - bn
            const unit = sv.includes('%') ? '%' : ''
            bench = <div style={{ fontSize: 11, marginTop: 5, color: 'var(--muted)' }}>
              vs NDX: <span className={diff >= 0 ? 'pos' : 'neg'}>{diff >= 0 ? '+' : ''}{diff.toFixed(1)}{unit}</span>
            </div>
          }
        }

        return (
          <div key={key} style={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 10, padding: '14px 16px', boxShadow: '0 1px 2px rgba(0,0,0,.05)' }}>
            <div style={{ fontSize: 10.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.5px', color: 'var(--muted)', marginBottom: 6 }}>{key}</div>
            <div className={cls} style={{ fontSize: 22, fontWeight: 700, lineHeight: 1 }}>{sv}</div>
            {bench}
          </div>
        )
      })}
    </div>
  )
}
