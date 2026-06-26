export const fmtDollar = (v: number | undefined | null): string =>
  v == null ? '—' : '$' + Math.round(v).toLocaleString()

export const fmtPct = (v: number | undefined | null): string =>
  v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(1) + '%'

export const pctCls = (v: number | undefined | null): string =>
  v == null ? '' : v >= 0 ? 'pos' : 'neg'

export function fmtHeld(entry: string, exit: string): string {
  const days = Math.round((new Date(exit).getTime() - new Date(entry).getTime()) / 864e5)
  if (days >= 365) {
    const y = Math.floor(days / 365), m = Math.floor((days % 365) / 30)
    return m ? `${y}y ${m}m` : `${y}y`
  }
  return days >= 30 ? Math.floor(days / 30) + 'm' : days + 'd'
}
