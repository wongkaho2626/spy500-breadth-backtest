'use client'
import { FormState } from '@/lib/types'

interface Props {
  form: FormState
  onChange: (k: keyof FormState, v: number | string) => void
  onRun: () => void
  loading: boolean
}

const sliders = [
  { key: 'qqq' as const, label: 'QQQ' },
  { key: 'stock' as const, label: 'NDX Top-1 Stock' },
  { key: 'tqqq' as const, label: 'TQQQ' },
  { key: 'spy' as const, label: 'SPY' },
  { key: 'soxx' as const, label: 'SOXX' },
]

export default function Sidebar({ form, onChange, onRun, loading }: Props) {
  const total = sliders.reduce((s, sl) => s + (form[sl.key] as number), 0)
  const badgeOk = total === 100

  return (
    <aside style={{ width: 270, minWidth: 270, background: 'var(--sb)', color: 'var(--sb-text)', display: 'flex', flexDirection: 'column', overflowY: 'auto', height: '100vh' }}>
      {/* Logo */}
      <div style={{ padding: '18px 16px 14px', borderBottom: '1px solid var(--sb2)' }}>
        <h1 style={{ fontSize: 14, fontWeight: 700, color: '#f8fafc', letterSpacing: '-.3px' }}>📈 QQQ Portfolio Backtest</h1>
        <p style={{ fontSize: 11, color: 'var(--sb-muted)', marginTop: 2 }}>Market-timing strategy analyser</p>
      </div>

      {/* Weights */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--sb2)' }}>
        <h2 style={{ fontSize: 9.5, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.9px', color: '#475569', marginBottom: 10 }}>Portfolio Allocation</h2>
        {sliders.map(sl => (
          <div key={sl.key} style={{ marginBottom: 9 }}>
            <label style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: '#cbd5e1', marginBottom: 4 }}>
              {sl.label}
              <span style={{ color: '#93c5fd', fontWeight: 700 }}>{form[sl.key]}%</span>
            </label>
            <input type="range" min={0} max={100} value={form[sl.key] as number}
              onChange={e => onChange(sl.key, Number(e.target.value))} />
          </div>
        ))}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 6, fontSize: 11, color: 'var(--sb-muted)' }}>
          <span>Total</span>
          <span style={{
            fontSize: 10.5, fontWeight: 700, padding: '2px 7px', borderRadius: 10,
            background: badgeOk ? '#14532d' : '#7f1d1d',
            color: badgeOk ? '#86efac' : '#fca5a5',
          }}>{total}%</span>
        </div>
      </div>

      {/* Capital */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--sb2)' }}>
        <h2 style={{ fontSize: 9.5, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.9px', color: '#475569', marginBottom: 10 }}>Capital &amp; Contributions</h2>
        {[
          { key: 'initial_capital' as const, label: 'Initial Capital ($)', step: 1000 },
          { key: 'monthly_contribution' as const, label: 'Monthly DCA ($)', step: 100 },
          { key: 'yearly_contribution' as const, label: 'Yearly DCA ($)', step: 1000 },
        ].map(f => (
          <div key={f.key} style={{ marginBottom: 9 }}>
            <label style={{ display: 'block', fontSize: 11.5, color: '#cbd5e1', marginBottom: 4 }}>{f.label}</label>
            <input type="number" value={form[f.key] as number} min={0} step={f.step}
              onChange={e => onChange(f.key, Number(e.target.value))}
              style={{ width: '100%', background: 'var(--sb2)', border: '1px solid var(--sb3)', borderRadius: 6, color: '#f1f5f9', fontSize: 12, padding: '5px 8px', outline: 'none' }} />
          </div>
        ))}
      </div>

      {/* Date range */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--sb2)' }}>
        <h2 style={{ fontSize: 9.5, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.9px', color: '#475569', marginBottom: 10 }}>Date Range</h2>
        {[
          { key: 'start_date' as const, label: 'Start Date' },
          { key: 'end_date' as const, label: 'End Date' },
        ].map(f => (
          <div key={f.key} style={{ marginBottom: 9 }}>
            <label style={{ display: 'block', fontSize: 11.5, color: '#cbd5e1', marginBottom: 4 }}>{f.label}</label>
            <input type="date" value={form[f.key] as string}
              onChange={e => onChange(f.key, e.target.value)}
              style={{ width: '100%', background: 'var(--sb2)', border: '1px solid var(--sb3)', borderRadius: 6, color: '#f1f5f9', fontSize: 12, padding: '5px 8px', outline: 'none' }} />
          </div>
        ))}
      </div>

      {/* Trade settings */}
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--sb2)' }}>
        <h2 style={{ fontSize: 9.5, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.9px', color: '#475569', marginBottom: 10 }}>Trade Settings</h2>
        <div style={{ marginBottom: 9 }}>
          <label style={{ display: 'block', fontSize: 11.5, color: '#cbd5e1', marginBottom: 4 }}>Cooldown Days After Sell</label>
          <input type="number" value={form.cooldown_days} min={0} step={5}
            onChange={e => onChange('cooldown_days', Number(e.target.value))}
            style={{ width: '100%', background: 'var(--sb2)', border: '1px solid var(--sb3)', borderRadius: 6, color: '#f1f5f9', fontSize: 12, padding: '5px 8px', outline: 'none' }} />
        </div>
        <div style={{ marginBottom: 9 }}>
          <label style={{ display: 'block', fontSize: 11.5, color: '#cbd5e1', marginBottom: 4 }}>Order Execution</label>
          <select value={form.fill_mode}
            onChange={e => onChange('fill_mode', e.target.value)}
            style={{ width: '100%', background: 'var(--sb2)', border: '1px solid var(--sb3)', borderRadius: 6, color: '#f1f5f9', fontSize: 12, padding: '5px 8px', outline: 'none' }}>
            <option value="next-open">Next day open (realistic)</option>
            <option value="next-close">Next day close</option>
            <option value="same-close">Same day close (look-ahead)</option>
          </select>
          <div style={{ fontSize: 10, color: '#64748b', marginTop: 3 }}>
            Signals are known only at the close, so orders fill the next session.
          </div>
        </div>
      </div>

      <button onClick={onRun} disabled={loading}
        style={{
          margin: '12px 16px 16px', padding: 10, width: 'calc(100% - 32px)',
          background: loading ? '#334155' : 'var(--primary)', color: '#fff',
          border: 'none', borderRadius: 8, fontSize: 13, fontWeight: 600,
          cursor: loading ? 'not-allowed' : 'pointer', display: 'flex',
          alignItems: 'center', justifyContent: 'center', gap: 7,
        }}>
        {loading ? <><span className="spin" /> Running…</> : 'Run Backtest'}
      </button>
    </aside>
  )
}
