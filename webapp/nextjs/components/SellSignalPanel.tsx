'use client'
import { SellProximity } from '@/lib/types'

interface Props { proximity?: SellProximity; hasOpenTrade: boolean }

function ProgressBar({ current, needed, met }: { current: number; needed: number; met: boolean }) {
  const pct = Math.min(current / needed * 100, 100)
  const color = met ? '#22c55e' : pct > 70 ? '#f59e0b' : '#3b82f6'
  return (
    <div style={{ height: 7, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
      <div style={{ height: '100%', borderRadius: 4, width: `${pct.toFixed(0)}%`, background: color, transition: 'width .5s ease' }} />
    </div>
  )
}

export default function SellSignalPanel({ proximity: p, hasOpenTrade }: Props) {
  if (!hasOpenTrade || !p) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: 'var(--muted)' }}>
        No open position — sell signal proximity is only visible when the strategy is in the market.
      </div>
    )
  }

  const allMet = p.price_rise_met && p.breadth_fall_met && p.breadth_cap_met

  const rows = [
    { label: 'Price Rise (over window)', current: p.price_rise_pct, needed: p.price_rise_needed, unit: '%', met: p.price_rise_met },
    { label: 'Breadth Fall (pts)', current: p.breadth_fall_pts, needed: p.breadth_fall_needed, unit: 'pts', met: p.breadth_fall_met },
  ]

  return (
    <div>
      {allMet && (
        <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: '10px 14px', marginBottom: 16, fontSize: 13, fontWeight: 600, color: '#dc2626' }}>
          ⚠ All sell conditions met — divergence signal triggered
        </div>
      )}

      <div style={{ marginBottom: 14, padding: '10px 14px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: 'var(--muted)' }}>
        <strong style={{ color: 'var(--text)' }}>Current breadth:</strong> {p.breadth_current.toFixed(1)}%
        &nbsp;·&nbsp;
        <strong style={{ color: 'var(--text)' }}>Cap:</strong> &lt;{p.breadth_cap}% — <span className={p.breadth_cap_met ? 'pos' : 'neg'}>{p.breadth_cap_met ? 'Met ✓' : 'Not met ✗'}</span>
      </div>

      {rows.map(row => (
        <div key={row.label} style={{ display: 'grid', gridTemplateColumns: '200px 70px 80px 1fr 70px', alignItems: 'center', gap: 10, padding: '9px 0', borderBottom: '1px solid var(--border)', fontSize: 12 }}>
          <div>{row.label}</div>
          <div style={{ fontWeight: 700, textAlign: 'right' }}>{row.current.toFixed(1)}{row.unit}</div>
          <div style={{ color: 'var(--muted)', textAlign: 'right' }}>need ≥{row.needed}{row.unit}</div>
          <ProgressBar current={row.current} needed={row.needed} met={row.met} />
          <div style={{ fontSize: 11, fontWeight: 700, textAlign: 'center' }} className={row.met ? 'pos' : 'neg'}>
            {row.met ? 'Met ✓' : 'Not met ✗'}
          </div>
        </div>
      ))}
    </div>
  )
}
