// TypeScript types for EIMAS API responses

// Main EIMAS integrated analysis result
export interface EIMASAnalysis {
  timestamp: string

  // FRED data (static, excluded from real-time updates)
  fred_summary: {
    rrp: number
    rrp_delta: number
    tga: number
    tga_delta: number
    fed_assets: number
    net_liquidity: number
    liquidity_regime: string
    fed_funds: number
    treasury_10y: number
    spread_10y2y: number
    curve_status: string
  }

  // Market data
  market_data_count: number
  crypto_data_count: number

  // Regime analysis
  regime: {
    regime: string
    trend: string
    volatility: string
    confidence: number
    description: string
    strategy: string
    gmm_regime?: string
    gmm_probabilities?: {
      Bull: number
      Neutral: number
      Bear: number
    }
    entropy?: number
    entropy_level?: string
    entropy_interpretation?: string
    gmm_report_line?: string
  }

  // Events and signals
  events_detected: any[]
  liquidity_signal: string

  // Risk metrics
  risk_score: number
  risk_level: "LOW" | "MEDIUM" | "HIGH"
  base_risk_score: number
  microstructure_adjustment: number
  bubble_risk_adjustment: number

  // Market quality (v2.1.1)
  market_quality: {
    avg_liquidity_score: number
    liquidity_scores: Record<string, number>
    high_toxicity_tickers: string[]
    illiquid_tickers: string[]
    data_quality: string
  } | null

  // Bubble risk (v2.1.1)
  bubble_risk: {
    overall_status: string
    risk_tickers: any[]
    highest_risk_ticker: string
    highest_risk_score: number
    methodology_notes: string
  } | null

  // AI debate results
  full_mode_position: string
  reference_mode_position: string
  modes_agree: boolean
  final_recommendation: string
  confidence: number

  // Phase 2-3 Enhanced Results (NEW)
  reasoning_chain: Array<{
    agent: string
    output_summary: string
    confidence: number
    key_factors: string[]
  }>
  debate_consensus: {
    enhanced?: {
      interpretation?: {
        recommended_action: string
        consensus_points: string[]
        divergence_points?: string[]
      }
      methodology?: {
        selected_methodology: string
        rationale: string
      }
    }
    verification?: {
      overall_score: number
      passed: boolean
      hallucination_risk: number
      warnings: string[]
    }
    metadata?: {
      num_agents: number
      total_debates: number
    }
  }

  // Portfolio and strategy
  portfolio_weights: Record<string, number>
  shock_propagation: any
  integrated_signals: any[]
  genius_act_regime: string
  genius_act_signals: any[]

  // Additional data
  theme_etf_analysis: any
  warnings: string[]
  realtime_signals: any[]

  // Event tracking
  tracked_events: Array<{
    ticker: string
    timestamp: string
    anomaly_type: string
    volume_zscore: number
    price_change_pct: number
    news_found: boolean
    news_summary?: string
    news_sources?: string[]
    event_type: string
    sentiment: string
    impact_score: number
  }>

  // Crypto stress test
  crypto_stress_test: {
    scenario: string
    total_value: number
    depeg_probability: number
    depeg_probability_pct: string
    estimated_loss_under_stress: number
    estimated_loss_pct: string
    breakdown_by_coin: Array<{
      ticker: string
      amount: number
      weight: number
      depeg_probability: number
      expected_loss: number
      loss_rate: number
    }>
    risk_rating: string
    methodology_note: string
  } | null

  // Volume anomalies
  volume_anomalies: Array<{
    ticker: string
    timestamp: string
    current_volume: number
    avg_volume_20d: number
    volume_ratio: number
    z_score: number
    price_change_1d: number
    price_change_5d: number
    anomaly_type: string
    information_type: string
    severity: string
    alert_message: string
  }>

  // Correlation analysis (v2.1.4)
  correlation_matrix: number[][]
  correlation_tickers: string[]

  // Metadata
  _meta?: {
    source_file: string
    file_modified: string
  }
}

// Legacy types (kept for backward compatibility)
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
