"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"
import { TrendingUp, TrendingDown, Activity } from "lucide-react"

export function MetricsGrid() {
  // Fetch latest EIMAS analysis every 5 seconds
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000, // 5 seconds
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
                <span className="text-2xl font-bold text-white">{analysis.risk_score.toFixed(1)}</span>
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

        {/* Data Collection Card */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-400">Data Collection</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div>
                <div className="text-2xl font-bold text-white">{analysis.market_data_count}</div>
                <div className="text-sm text-gray-400">Market Tickers</div>
              </div>
              <div>
                <div className="text-xl font-bold text-white">{analysis.crypto_data_count}</div>
                <div className="text-sm text-gray-400">Crypto Assets</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Market Quality Card (v2.1.1) */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader>
            <CardTitle className="text-sm font-medium text-gray-400">Market Quality</CardTitle>
          </CardHeader>
          <CardContent>
            {analysis.market_quality ? (
              <div className="space-y-3">
                <div>
                  <div className="text-2xl font-bold text-white">
                    {analysis.market_quality.avg_liquidity_score.toFixed(1)}
                  </div>
                  <div className="text-sm text-gray-400">Avg Liquidity</div>
                </div>
                <div className="text-xs">
                  <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                    {analysis.market_quality.data_quality}
                  </Badge>
                </div>
              </div>
            ) : (
              <div className="text-sm text-gray-400">No data</div>
            )}
          </CardContent>
        </Card>
      </div>

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
