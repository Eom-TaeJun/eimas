export type AnalysisLevel = "Geopolitics" | "Monetary" | "Sector" | "Individual"
export type ResearchGoal = "Variable Selection" | "Forecasting" | "Causal Inference"
export type AnalysisStatus = "PENDING" | "RUNNING" | "COMPLETED" | "FAILED"
export type Stance = "BULLISH" | "BEARISH" | "NEUTRAL"

export interface RegimeContext {
  regime_type?: string
  confidence?: number
  indicators?: string[]
}

export interface AnalysisResult {
  analysis_id: string
  status: AnalysisStatus
  final_stance?: Stance
  confidence?: number
  executive_summary?: string
  top_down_summary?: string
  regime_context?: RegimeContext
  stages_completed?: string[]
  duration?: number
  timestamp?: string
  question?: string
  analysis_level?: AnalysisLevel
  research_goal?: ResearchGoal
}
