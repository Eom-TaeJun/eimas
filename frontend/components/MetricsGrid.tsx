"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"
import { TrendingUp, TrendingDown, Activity, PieChart as PieChartIcon, Brain, Shield, GitBranch } from "lucide-react"
import { RiskGauge } from "@/components/charts/RiskGauge"
import { PortfolioPie } from "@/components/charts/PortfolioPie"

export function MetricsGrid() {
  // Fetch latest EIMAS analysis every 5 seconds
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  })

  const getRiskColor = (level: string) => {
    switch (level) {
      case "LOW":
        return "bg-green-500/10 text-green-500 border-green-500/20"
      case "MEDIUM":
        return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
      case "HIGH":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case "BULLISH":
      case "BUY":
        return "bg-green-500/10 text-green-500 border-green-500/20"
      case "BEARISH":
      case "SELL":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      case "NEUTRAL":
      case "HOLD":
        return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  const getRegimeIcon = (regime: string) => {
    if (regime.includes("Bull")) return <TrendingUp className="w-5 h-5 text-green-400" />
    if (regime.includes("Bear")) return <TrendingDown className="w-5 h-5 text-red-400" />
    return <Activity className="w-5 h-5 text-gray-400" />
  }

  if (error) {
    return (
      <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-4">
        <p className="text-red-400">Failed to load EIMAS analysis data</p>
        <p className="text-sm text-gray-400 mt-2">Make sure FastAPI server is running on localhost:8000</p>
      </div>
    )
  }

  if (!analysis) {
    return (
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {[...Array(4)].map((_, i) => (
          <Card key={i} className="bg-[#161b22] border-[#30363d]">
            <CardHeader>
              <CardTitle className="text-sm font-medium text-gray-400">Loading...</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="animate-pulse space-y-2">
                <div className="h-8 bg-gray-700 rounded"></div>
                <div className="h-4 bg-gray-700 rounded w-2/3"></div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Main Status Banner */}
      <Card className={`bg-[#161b22] border-2 ${getRecommendationColor(analysis.final_recommendation)}`}>
        <CardContent className="pt-6">
          <div className="grid gap-6 md:grid-cols-3">
            <div>
              <div className="text-sm text-gray-400 mb-2">Final Recommendation</div>
              <Badge variant="outline" className={`text-2xl font-bold px-4 py-2 ${getRecommendationColor(analysis.final_recommendation)}`}>
                {analysis.final_recommendation}
              </Badge>
            </div>
            <div>
              <div className="text-sm text-gray-400 mb-2">Confidence</div>
              <div className="text-3xl font-bold text-white">{(analysis.confidence * 100).toFixed(1)}%</div>
              <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                  style={{ width: `${analysis.confidence * 100}%` }}
                ></div>
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400 mb-2">Risk Level</div>
              <div className="flex items-center gap-3">
                <Badge variant="outline" className={`text-xl font-bold px-3 py-1 ${getRiskColor(analysis.risk_level)}`}>
                  {analysis.risk_level}
                </Badge>
                {/* Risk Gauge Small */}
                <div className="w-24 h-16">
                  <RiskGauge score={analysis.risk_score} />
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Metrics Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* Market Regime Card */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader>
            <div className="flex items-center gap-2">
              {getRegimeIcon(analysis.regime.regime)}
              <CardTitle className="text-sm font-medium text-gray-400">Market Regime</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="text-xl font-bold text-white">{analysis.regime.regime}</div>
              <div className="space-y-1 text-sm">
                <div>
                  <span className="text-gray-400">Trend:</span>{" "}
                  <span className="text-white">{analysis.regime.trend}</span>
                </div>
                <div>
                  <span className="text-gray-400">Vol:</span>{" "}
                  <span className="text-white">{analysis.regime.volatility}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-400 text-xs">Confidence:</span>
                <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20 text-xs">
                  {(analysis.regime.confidence * 100).toFixed(0)}%
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* AI Consensus Card */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-400">AI Consensus</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="space-y-2 text-sm">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Full Mode:</span>
                  <Badge variant="outline" className={`text-xs ${getRecommendationColor(analysis.full_mode_position)}`}>
                    {analysis.full_mode_position}
                  </Badge>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Reference:</span>
                  <Badge variant="outline" className={`text-xs ${getRecommendationColor(analysis.reference_mode_position)}`}>
                    {analysis.reference_mode_position}
                  </Badge>
                </div>
              </div>
              <div className="pt-2 border-t border-gray-700">
                {analysis.modes_agree ? (
                  <div className="flex items-center gap-2 text-green-400 text-sm">
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span>Modes Agree</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-yellow-400 text-sm">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                    <span>Dissent</span>
                  </div>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Portfolio Allocation Card (NEW) */}
        <Card className="bg-[#161b22] border-[#30363d] col-span-1 md:col-span-2">
          <CardHeader>
            <div className="flex items-center gap-2">
              <PieChartIcon className="w-5 h-5 text-purple-400" />
              <CardTitle className="text-sm font-medium text-gray-400">Recommended Allocation</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[200px]">
              {analysis.portfolio_weights ? (
                <PortfolioPie weights={analysis.portfolio_weights} />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500">
                  Calculating Portfolio...
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Enhanced AI Section (Phase 2-3) */}
      {(analysis.debate_consensus?.enhanced || analysis.reasoning_chain?.length > 0) && (
        <div className="grid gap-4 md:grid-cols-3">
          {/* Enhanced Debate Card */}
          {analysis.debate_consensus?.enhanced && (
            <Card className="bg-[#161b22] border-[#30363d]">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-purple-400" />
                  <CardTitle className="text-sm font-medium text-gray-400">Enhanced Debate</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {analysis.debate_consensus.enhanced.interpretation && (
                    <div>
                      <div className="text-xs text-gray-500 mb-1">Interpretation</div>
                      <Badge variant="outline" className={`text-xs ${getRecommendationColor(analysis.debate_consensus.enhanced.interpretation.recommended_action)}`}>
                        {analysis.debate_consensus.enhanced.interpretation.recommended_action}
                      </Badge>
                      <div className="text-xs text-gray-500 mt-1">Schools: Monetarist / Keynesian / Austrian</div>
                    </div>
                  )}
                  {analysis.debate_consensus.enhanced.methodology && (
                    <div className="pt-2 border-t border-gray-700">
                      <div className="text-xs text-gray-500 mb-1">Methodology</div>
                      <div className="text-sm text-white">{analysis.debate_consensus.enhanced.methodology.selected_methodology}</div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Verification Card */}
          {analysis.debate_consensus?.verification && (
            <Card className="bg-[#161b22] border-[#30363d]">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-green-400" />
                  <CardTitle className="text-sm font-medium text-gray-400">Verification</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">Score</span>
                    <span className="text-xl font-bold text-white">{analysis.debate_consensus.verification.overall_score?.toFixed(1)}/100</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">Status</span>
                    <Badge variant="outline" className={analysis.debate_consensus.verification.passed ? "bg-green-500/10 text-green-400 border-green-500/20" : "bg-red-500/10 text-red-400 border-red-500/20"}>
                      {analysis.debate_consensus.verification.passed ? "✅ Passed" : "❌ Failed"}
                    </Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">Hallucination Risk</span>
                    <span className="text-sm text-white">{(analysis.debate_consensus.verification.hallucination_risk * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Reasoning Chain Card */}
          {analysis.reasoning_chain && analysis.reasoning_chain.length > 0 && (
            <Card className="bg-[#161b22] border-[#30363d]">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <GitBranch className="w-5 h-5 text-blue-400" />
                  <CardTitle className="text-sm font-medium text-gray-400">Reasoning Chain</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-xs text-gray-500 mb-2">{analysis.reasoning_chain.length} steps tracked</div>
                  {analysis.reasoning_chain.slice(0, 3).map((step, idx) => (
                    <div key={idx} className="flex items-center gap-2 text-sm">
                      <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                      <span className="text-gray-400">{step.agent}</span>
                      <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                        {step.confidence}%
                      </Badge>
                    </div>
                  ))}
                  {analysis.reasoning_chain.length > 3 && (
                    <div className="text-xs text-gray-500">+{analysis.reasoning_chain.length - 3} more steps...</div>
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Warnings (if any) */}
      {analysis.warnings && analysis.warnings.length > 0 && (
        <Card className="bg-yellow-500/10 border border-yellow-500/20">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-yellow-400">⚠️ Warnings</CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-1">
              {analysis.warnings.map((warning, idx) => (
                <li key={idx} className="text-sm text-gray-300">
                  • {warning}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  )
}