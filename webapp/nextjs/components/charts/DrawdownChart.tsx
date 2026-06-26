'use client'
import { useEffect, useRef } from 'react'
import { Chart, LineController, LineElement, PointElement, LinearScale, TimeScale, Filler, Legend, Tooltip } from 'chart.js'
import 'chartjs-adapter-date-fns'
import { MetricPoint } from '@/lib/types'

Chart.register(LineController, LineElement, PointElement, LinearScale, TimeScale, Filler, Legend, Tooltip)

function computeDD(pts: MetricPoint[]): number[] {
  let peak = -Infinity
  return pts.map(p => { if (p.value > peak) peak = p.value; return peak > 0 ? (p.value - peak) / peak * 100 : 0 })
}

interface Props { portfolio: MetricPoint[]; benchmark: MetricPoint[] }

export default function DrawdownChart({ portfolio, benchmark }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    const ddPort = computeDD(portfolio)
    const ddBench = computeDD(benchmark)

    const chart = new Chart(canvasRef.current, {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Portfolio Drawdown',
            data: portfolio.map((d, i) => ({ x: d.date as unknown as number, y: ddPort[i] })),
            borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,.15)',
            borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0,
          },
          {
            label: 'NDX Drawdown',
            data: benchmark.map((d, i) => ({ x: d.date as unknown as number, y: ddBench[i] })),
            borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,.07)',
            borderWidth: 1, pointRadius: 0, fill: true, tension: 0,
          },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { font: { size: 11 }, boxWidth: 12 } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label ?? ''}: ${(c.parsed.y ?? 0).toFixed(1)}%` } }
        },
        scales: {
          x: { type: 'time', time: { unit: 'year', displayFormats: { year: 'yyyy' } }, ticks: { font: { size: 10 }, maxTicksLimit: 14 }, grid: { display: false } },
          y: { ticks: { font: { size: 10 }, callback: (v: number | string) => v + '%' }, grid: { color: 'rgba(0,0,0,.05)' } }
        }
      }
    })
    return () => chart.destroy()
  }, [portfolio, benchmark])

  return <canvas ref={canvasRef} />
}
