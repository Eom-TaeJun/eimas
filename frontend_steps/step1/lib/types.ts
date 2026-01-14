// TypeScript types for EIMAS API responses
export interface Portfolio {
  cash: number
  positions_value: number
  total_value: number
  total_pnl: number
  total_pnl_pct: number
}

export interface MarketRegime {
  regime: string
  trend: string
  volatility: string
  confidence: number
}

export interface Signal {
  source: string
  action: "BUY" | "SELL" | "HOLD"
  ticker: string
  conviction: number
  timestamp: string
}

export interface Risk {
  risk_score: number
  risk_level: "LOW" | "MEDIUM" | "HIGH"
}
