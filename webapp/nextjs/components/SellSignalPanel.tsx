'use client'
import { SellProximity } from '@/lib/types'

interface Props { proximity?: SellProximity; hasOpenTrade: boolean }

function ProgressBar({ current, needed, met }: { current: number; needed: number; met: boolean }) {
  const pct = Math.min(Math.max(current / needed * 100, 0), 100)
  const color = met ? '#22c55e' : pct > 70 ? '#f59e0b' : '#3b82f6'
  return (
    <div style={{ height: 7, background: '#e2e8f0', borderRadius: 4, overflow: 'hidden' }}>
      <div style={{ height: '100%', borderRadius: 4, width: `${pct.toFixed(0)}%`, background: color, transition: 'width .5s ease' }} />
    </div>
  )
}

function MetBadge({ met }: { met: boolean }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 700, textAlign: 'center' }} className={met ? 'pos' : 'neg'}>
      {met ? 'Met ✓' : 'Not met ✗'}
    </div>
  )
}

function SectionHeader({ title, triggered }: { title: string; triggered: boolean }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, margin: '18px 0 6px' }}>
      <h3 style={{ fontSize: 13, fontWeight: 700, color: 'var(--text)' }}>{title}</h3>
      <span style={{
        fontSize: 10, fontWeight: 700, padding: '1px 8px', borderRadius: 10,
        background: triggered ? '#fef2f2' : '#f0fdf4',
        color: triggered ? '#dc2626' : '#15803d',
        border: `1px solid ${triggered ? '#fecaca' : '#bbf7d0'}`,
      }}>
        {triggered ? 'TRIGGERED' : 'not triggered'}
      </span>
    </div>
  )
}

const ROW_STYLE: React.CSSProperties = {
  display: 'grid', gridTemplateColumns: '200px 80px 90px 1fr 70px',
  alignItems: 'center', gap: 10, padding: '9px 0',
  borderBottom: '1px solid var(--border)', fontSize: 12,
}

export default function SellSignalPanel({ proximity: p, hasOpenTrade }: Props) {
  if (!hasOpenTrade || !p) {
    return (
      <div style={{ textAlign: 'center', padding: 40, color: 'var(--muted)' }}>
        No open position — sell signal proximity is only visible when the strategy is in the market.
      </div>
    )
  }

  const divergenceMet = p.price_rise_met && p.breadth_fall_met && p.breadth_cap_met
  const anyMet = divergenceMet || p.climax_met || p.trail_met

  const divRows = [
    { label: 'Price rise (over window)', current: p.price_rise_pct, needed: p.price_rise_needed, unit: '%', met: p.price_rise_met },
    { label: 'Breadth fall (pts)', current: p.breadth_fall_pts, needed: p.breadth_fall_needed, unit: 'pts', met: p.breadth_fall_met },
  ]

  const fmtDays = (d: number | null) => d === null ? 'never' : d === 0 ? 'today' : `${d}d ago`
  const climaxRows = [
    { label: 'MACD bearish cross', daysAgo: p.macd_days_ago, met: p.macd_days_ago !== null && p.macd_days_ago < p.climax_window },
    { label: 'Extended ≥5% above 10d MA', daysAgo: p.ext_days_ago, met: p.ext_days_ago !== null && p.ext_days_ago < p.climax_window },
  ]

  return (
    <div>
      {anyMet ? (
        <div style={{ background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, padding: '10px 14px', marginBottom: 4, fontSize: 13, fontWeight: 600, color: '#dc2626' }}>
          ⚠ Sell signal ACTIVE — {[divergenceMet && 'bearish divergence', p.climax_met && 'climax top', p.trail_met && 'trailing stop'].filter(Boolean).join(' + ')}
        </div>
      ) : (
        <div style={{ background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 8, padding: '10px 14px', marginBottom: 4, fontSize: 13, fontWeight: 600, color: '#15803d' }}>
          No sell condition triggered — position stays open
        </div>
      )}

      {/* ── 1. Bearish divergence ── */}
      <SectionHeader title="1 · Bearish Divergence" triggered={divergenceMet} />
      <div style={{ marginBottom: 8, padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: 'var(--muted)' }}>
        <strong style={{ color: 'var(--text)' }}>Current breadth:</strong> {p.breadth_current.toFixed(1)}%
        &nbsp;·&nbsp;
        <strong style={{ color: 'var(--text)' }}>Cap:</strong> &lt;{p.breadth_cap}% — <span className={p.breadth_cap_met ? 'pos' : 'neg'}>{p.breadth_cap_met ? 'Met ✓' : 'Not met ✗'}</span>
      </div>
      {divRows.map(row => (
        <div key={row.label} style={ROW_STYLE}>
          <div>{row.label}</div>
          <div style={{ fontWeight: 700, textAlign: 'right' }}>{row.current.toFixed(1)}{row.unit}</div>
          <div style={{ color: 'var(--muted)', textAlign: 'right' }}>need ≥{row.needed}{row.unit}</div>
          <ProgressBar current={row.current} needed={row.needed} met={row.met} />
          <MetBadge met={row.met} />
        </div>
      ))}

      {/* ── 2. Climax top ── */}
      <SectionHeader title="2 · Climax Top" triggered={p.climax_met} />
      <div style={{ marginBottom: 8, padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: 'var(--muted)' }}>
        Both signals must fire within <strong style={{ color: 'var(--text)' }}>{p.climax_window} days</strong> of each other (after entry).
      </div>
      {climaxRows.map(row => (
        <div key={row.label} style={ROW_STYLE}>
          <div>{row.label}</div>
          <div style={{ fontWeight: 700, textAlign: 'right' }}>{fmtDays(row.daysAgo)}</div>
          <div style={{ color: 'var(--muted)', textAlign: 'right' }}>&lt;{p.climax_window}d ago</div>
          <div />
          <MetBadge met={row.met} />
        </div>
      ))}

      {/* ── 3. Trailing stop ── */}
      <SectionHeader title="3 · Trailing Stop" triggered={p.trail_met} />
      <div style={{ marginBottom: 8, padding: '8px 12px', background: '#f8fafc', borderRadius: 8, fontSize: 12, color: 'var(--muted)' }}>
        <strong style={{ color: 'var(--text)' }}>NDX high since entry:</strong> {p.ndx_high.toLocaleString()}
        &nbsp;·&nbsp;
        <strong style={{ color: 'var(--text)' }}>Current:</strong> {p.ndx_current.toLocaleString()}
      </div>
      <div style={ROW_STYLE}>
        <div>Drop from high</div>
        <div style={{ fontWeight: 700, textAlign: 'right' }}>{p.drop_from_high_pct.toFixed(1)}%</div>
        <div style={{ color: 'var(--muted)', textAlign: 'right' }}>need ≥{p.trail_stop_pct}%</div>
        <ProgressBar current={p.drop_from_high_pct} needed={p.trail_stop_pct} met={p.trail_met} />
        <MetBadge met={p.trail_met} />
      </div>
    </div>
  )
}
