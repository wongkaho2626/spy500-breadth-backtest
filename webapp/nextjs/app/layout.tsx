import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'QQQ Portfolio Backtest',
  description: 'Market-timing strategy analyser',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
