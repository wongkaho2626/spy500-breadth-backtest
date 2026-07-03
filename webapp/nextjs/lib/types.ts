export interface MetricPoint { date: string; value: number }

export interface AnnualReturn { year: number; strategy: number; benchmark: number }

export interface Trade {
  entry_date: string
  exit_date?: string
  top1_ticker?: string
  buy_trigger?: string
  sell_reason?: string
  return_pct?: number
  max_drawdown_pct?: number
  accumulated?: number
  current_date?: string
  qqq_entry_val?: number; stock_entry_val?: number; tqqq_entry_val?: number
  spy_entry_val?: number; soxx_entry_val?: number
  qqq_exit_val?: number; stock_exit_val?: number; tqqq_exit_val?: number
  spy_exit_val?: number; soxx_exit_val?: number
}

export interface SellProximity {
  price_rise_pct: number; breadth_fall_pts: number; breadth_current: number
  price_rise_needed: number; breadth_fall_needed: number; breadth_cap: number
  price_rise_met: boolean; breadth_fall_met: boolean; breadth_cap_met: boolean
  // Climax top: both must fire within climax_window days (post-entry)
  macd_days_ago: number | null; ext_days_ago: number | null
  climax_window: number; climax_met: boolean
  // Trailing stop on NDX since entry
  ndx_high: number; ndx_current: number
  drop_from_high_pct: number; trail_stop_pct: number; trail_met: boolean
}

export interface BacktestResponse {
  success: boolean; error?: string
  metrics: { strategy: Record<string, string>; benchmark: Record<string, string> }
  chart_data: { portfolio: MetricPoint[]; benchmark: MetricPoint[]; breadth: MetricPoint[]; ndx: MetricPoint[] }
  trades: Trade[]; open_trade?: Trade; sell_proximity?: SellProximity
  annual_returns: AnnualReturn[]; total_contrib: number
  weights: { qqq: number; stock: number; tqqq: number; spy: number; soxx: number }
  params: Record<string, number>
}

export interface FormState {
  qqq: number; stock: number; tqqq: number; spy: number; soxx: number
  initial_capital: number; monthly_contribution: number; yearly_contribution: number
  cooldown_days: number; start_date: string; end_date: string
}
