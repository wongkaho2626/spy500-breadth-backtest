'use client'
import { useEffect, useRef } from 'react'
import {
  Chart, LineController, LineElement, PointElement, LinearScale, TimeScale,
  Filler, Legend, Tooltip, ScatterController,
} from 'chart.js'
import 'chartjs-adapter-date-fns'
import { BacktestResponse } from '@/lib/types'

Chart.register(LineController, LineElement, PointElement, LinearScale, TimeScale, Filler, Legend, Tooltip, ScatterController)

export default function SignalsChart({ data }: { data: BacktestResponse }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    const breadth = data.chart_data.breadth
    const ndx = data.chart_data.ndx
    const bMap = Object.fromEntries(breadth.map(d => [d.date, d.value]))
    const allT = [...(data.trades || []), ...(data.open_trade ? [data.open_trade] : [])]
    const buys = allT.map(t => ({ x: t.entry_date, y: bMap[t.entry_date] ?? null })).filter(p => p.y != null)
    const sells = (data.trades || []).map(t => ({ x: t.exit_date!, y: bMap[t.exit_date!] ?? null })).filter(p => p.y != null)

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const chartConfig: any = {
      data: {
        datasets: [
          {
            type: 'line', label: '% S&P 500 Above 200-Day MA',
            data: breadth.map(d => ({ x: d.date as unknown as number, y: d.value })),
            borderColor: '#7c3aed', backgroundColor: 'rgba(124,58,237,.07)',
            borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0, yAxisID: 'yB',
          },
          {
            type: 'line', label: 'NASDAQ 100',
            data: ndx.map(d => ({ x: d.date as unknown as number, y: d.value })),
            borderColor: '#0ea5e9', borderWidth: 1,
            pointRadius: 0, fill: false, tension: 0, yAxisID: 'yN',
          },
          {
            type: 'scatter', label: 'Buy',
            data: buys as { x: unknown; y: number }[],
            backgroundColor: '#22c55e', borderColor: '#15803d', borderWidth: 1.5,
            pointStyle: 'triangle', pointRadius: 9, rotation: 0, yAxisID: 'yB',
          },
          {
            type: 'scatter', label: 'Sell',
            data: sells as { x: unknown; y: number }[],
            backgroundColor: '#ef4444', borderColor: '#991b1b', borderWidth: 1.5,
            pointStyle: 'triangle', pointRadius: 9, rotation: 180, yAxisID: 'yB',
          },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { font: { size: 11 }, boxWidth: 12, usePointStyle: true } },
          tooltip: {
            filter: (c: { datasetIndex: number }) => c.datasetIndex < 2,
            callbacks: {
              label: (c: { datasetIndex: number; parsed: { y: number }; dataset: { label: string } }) =>
                c.datasetIndex === 0 ? `Breadth: ${c.parsed.y.toFixed(1)}%` : `NDX: ${Math.round(c.parsed.y).toLocaleString()}`
            }
          }
        },
        scales: {
          x: { type: 'time', time: { unit: 'year', displayFormats: { year: 'yyyy' } }, ticks: { font: { size: 10 }, maxTicksLimit: 14 }, grid: { display: false } },
          yB: {
            position: 'left',
            ticks: { font: { size: 10 }, callback: (v: number | string) => v + '%' },
            grid: { color: 'rgba(0,0,0,.04)' },
            title: { display: true, text: 'Breadth (%)', font: { size: 10 }, color: '#7c3aed' },
          },
          yN: {
            position: 'right',
            ticks: { font: { size: 10 }, callback: (v: number | string) => { const n = Number(v); return n >= 1000 ? Math.round(n / 1000) + 'k' : String(v) } },
            grid: { display: false },
            title: { display: true, text: 'NDX', font: { size: 10 }, color: '#0ea5e9' },
          },
        }
      }
    }
    const chart = new Chart(canvasRef.current, chartConfig)
    return () => chart.destroy()
  }, [data])

  return <canvas ref={canvasRef} />
}
