'use client'
import { useEffect, useRef } from 'react'
import { Chart, BarController, BarElement, CategoryScale, LinearScale, Legend, Tooltip } from 'chart.js'
import { AnnualReturn } from '@/lib/types'

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Legend, Tooltip)

export default function AnnualChart({ data }: { data: AnnualReturn[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!canvasRef.current) return
    const chart = new Chart(canvasRef.current, {
      type: 'bar',
      data: {
        labels: data.map(r => r.year),
        datasets: [
          {
            label: 'Portfolio',
            data: data.map(r => r.strategy),
            backgroundColor: data.map(r => r.strategy >= 0 ? 'rgba(239,68,68,.75)' : 'rgba(239,68,68,.4)'),
            borderColor: '#ef4444', borderWidth: 1, borderRadius: 3,
          },
          {
            label: 'Buy & Hold NDX',
            data: data.map(r => r.benchmark),
            backgroundColor: data.map(r => r.benchmark >= 0 ? 'rgba(59,130,246,.75)' : 'rgba(59,130,246,.4)'),
            borderColor: '#3b82f6', borderWidth: 1, borderRadius: 3,
          },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { position: 'top', labels: { font: { size: 11 }, boxWidth: 12 } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label ?? ''}: ${(c.raw as number) >= 0 ? '+' : ''}${(c.raw as number).toFixed(1)}%` } }
        },
        scales: {
          x: { ticks: { font: { size: 10 } }, grid: { display: false } },
          y: { ticks: { font: { size: 10 }, callback: (v: number | string) => v + '%' }, grid: { color: 'rgba(0,0,0,.05)' } }
        }
      }
    })
    return () => chart.destroy()
  }, [data])

  return <canvas ref={canvasRef} />
}
