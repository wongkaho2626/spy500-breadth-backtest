'use client'
import { useEffect, useRef } from 'react'
import {
  Chart, LineController, LineElement, PointElement, LinearScale, LogarithmicScale,
  TimeScale, Filler, Legend, Tooltip, ScatterController,
} from 'chart.js'
import 'chartjs-adapter-date-fns'
import { BacktestResponse } from '@/lib/types'

Chart.register(LineController, LineElement, PointElement, LinearScale, LogarithmicScale,
  TimeScale, Filler, Legend, Tooltip, ScatterController)

interface Props { data: BacktestResponse; scale: 'linear' | 'log' }

export default function GrowthChart({ data, scale }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const chartRef = useRef<Chart | null>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    chartRef.current?.destroy()

    const port = data.chart_data.portfolio
    const bench = data.chart_data.benchmark
    const portMap = Object.fromEntries(port.map(d => [d.date, d.value]))

    const allTrades = [...(data.trades || []), ...(data.open_trade ? [data.open_trade] : [])]
    const buys = allTrades.map(t => ({ x: t.entry_date, y: portMap[t.entry_date] ?? null })).filter(p => p.y != null)
    const sells = (data.trades || []).map(t => ({ x: t.exit_date!, y: portMap[t.exit_date!] ?? null })).filter(p => p.y != null)

    const w = data.weights
    const wLabel = [
      w.qqq > 0 ? `QQQ ${w.qqq}%` : '',
      w.stock > 0 ? `Stock ${w.stock}%` : '',
      w.tqqq > 0 ? `TQQQ ${w.tqqq}%` : '',
      w.spy > 0 ? `SPY ${w.spy}%` : '',
      w.soxx > 0 ? `SOXX ${w.soxx}%` : '',
    ].filter(Boolean).join(' / ')

    // chart.js mixed type (line+scatter) requires a type cast for the config
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const chartConfig: any = {
      data: {
        datasets: [
          {
            type: 'line', label: `Portfolio (${wLabel})`,
            data: port.map(d => ({ x: d.date as unknown as number, y: d.value })),
            borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,.06)',
            borderWidth: 2, pointRadius: 0, fill: true, tension: 0, order: 2,
          },
          {
            type: 'line', label: 'Buy & Hold NDX',
            data: bench.map(d => ({ x: d.date as unknown as number, y: d.value })),
            borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,.04)',
            borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0, order: 3,
          },
          {
            type: 'scatter', label: 'Buy',
            data: buys as { x: unknown; y: number }[],
            backgroundColor: '#22c55e', borderColor: '#15803d', borderWidth: 1.5,
            pointStyle: 'triangle', pointRadius: 9, rotation: 0, order: 1,
          },
          {
            type: 'scatter', label: 'Sell',
            data: sells as { x: unknown; y: number }[],
            backgroundColor: '#ef4444', borderColor: '#991b1b', borderWidth: 1.5,
            pointStyle: 'triangle', pointRadius: 9, rotation: 180, order: 1,
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top', labels: { font: { size: 11 }, boxWidth: 12, usePointStyle: true } },
          tooltip: {
            filter: (item: { datasetIndex: number }) => item.datasetIndex < 2,
            callbacks: { label: (c: { dataset: { label: string }; parsed: { y: number } }) => `${c.dataset.label}: $${Math.round(c.parsed.y).toLocaleString()}` }
          }
        },
        scales: {
          x: {
            type: 'time',
            time: { unit: 'year', displayFormats: { year: 'yyyy', month: "MMM 'yy" } },
            ticks: { maxTicksLimit: 14, font: { size: 10 } },
            grid: { display: false },
          },
          y: {
            type: scale === 'log' ? 'logarithmic' : 'linear',
            ticks: {
              font: { size: 10 },
              callback: (v: number | string) => {
                const n = Number(v)
                return '$' + (n >= 1e6 ? (n / 1e6).toFixed(1) + 'M' : n >= 1000 ? (n / 1000).toFixed(0) + 'k' : n)
              }
            },
            grid: { color: 'rgba(0,0,0,.05)' },
          }
        }
      }
    }
    chartRef.current = new Chart(canvasRef.current, chartConfig)

    return () => { chartRef.current?.destroy(); chartRef.current = null }
  }, [data, scale])

  return <canvas ref={canvasRef} />
}
