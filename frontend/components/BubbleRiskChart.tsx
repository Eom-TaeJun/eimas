"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { AlertTriangle } from "lucide-react"

interface BubbleRiskChartProps {
  bubbleRisk: {
    overall_status: string
    risk_tickers: Array<{
      ticker: string
      risk_score: number
      run_up_2y: number
      status: string
    }>
    highest_risk_ticker: string
    highest_risk_score: number
    methodology_notes: string
  } | null
}

export function BubbleRiskChart({ bubbleRisk }: BubbleRiskChartProps) {
  if (!bubbleRisk) {
    return (
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-gray-400">Bubble Risk Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-gray-500 text-sm">No bubble risk data available</div>
        </CardContent>
      </Card>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status.toUpperCase()) {
      case "DANGER":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      case "WARNING":
        return "bg-orange-500/10 text-orange-500 border-orange-500/20"
      case "WATCH":
        return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
      case "NONE":
        return "bg-green-500/10 text-green-500 border-green-500/20"
      default:
        return "bg-gray-500/10 text-gray-500 border-gray-500/20"
    }
  }

  const getBarColor = (riskScore: number) => {
    if (riskScore >= 75) return "#f85149" // Red - DANGER
    if (riskScore >= 50) return "#fb8500" // Orange - WARNING
    if (riskScore >= 25) return "#d29922" // Yellow - WATCH
    return "#3fb950" // Green - NONE
  }

  // Prepare data for chart (top 5 risk tickers)
  const chartData = bubbleRisk.risk_tickers
    .slice(0, 5)
    .map(ticker => ({
      ticker: ticker.ticker,
      riskScore: ticker.risk_score,
      runUp: ticker.run_up_2y,
      status: ticker.status,
    }))

  return (
    <Card className="bg-[#161b22] border-[#30363d]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-orange-400" />
            <CardTitle className="text-sm font-medium text-gray-400">
              Bubble Risk Analysis
            </CardTitle>
          </div>
          <Badge
            variant="outline"
            className={`${getStatusColor(bubbleRisk.overall_status)} text-xs font-bold`}
          >
            {bubbleRisk.overall_status}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Overall Summary */}
          <div className="grid grid-cols-2 gap-4 pb-4 border-b border-gray-700">
            <div>
              <div className="text-xs text-gray-400 mb-1">Highest Risk Ticker</div>
              <div className="text-lg font-bold text-white">{bubbleRisk.highest_risk_ticker}</div>
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">Highest Risk Score</div>
              <div className="text-lg font-bold text-white">{bubbleRisk.highest_risk_score.toFixed(1)}</div>
            </div>
          </div>

          {/* Bar Chart */}
          {chartData.length > 0 ? (
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                  <XAxis
                    dataKey="ticker"
                    tick={{ fill: "#8b949e", fontSize: 12 }}
                    angle={-45}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    tick={{ fill: "#8b949e", fontSize: 12 }}
                    label={{ value: "Risk Score", angle: -90, position: "insideLeft", fill: "#8b949e" }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#161b22",
                      border: "1px solid #30363d",
                      borderRadius: "6px",
                      color: "#c9d1d9",
                    }}
                    formatter={(value: number, name: string) => {
                      if (name === "riskScore") return [value.toFixed(1), "Risk Score"]
                      if (name === "runUp") return [`${value.toFixed(1)}%`, "2Y Run-up"]
                      return [value, name]
                    }}
                  />
                  <Bar dataKey="riskScore" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getBarColor(entry.riskScore)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="text-gray-500 text-sm text-center py-8">
              No high-risk tickers detected
            </div>
          )}

          {/* Risk Tickers Table */}
          {bubbleRisk.risk_tickers.length > 0 && (
            <div className="pt-4 border-t border-gray-700">
              <div className="text-xs text-gray-400 mb-2 font-semibold">Top Risk Tickers</div>
              <div className="space-y-2">
                {bubbleRisk.risk_tickers.slice(0, 5).map((ticker, idx) => (
                  <div key={idx} className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <span className="text-white font-medium">{ticker.ticker}</span>
                      <Badge
                        variant="outline"
                        className={`${getStatusColor(ticker.status)} text-xs`}
                      >
                        {ticker.status}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="text-gray-400 text-xs">
                        2Y: <span className="text-white">{ticker.run_up_2y.toFixed(1)}%</span>
                      </div>
                      <div className="text-gray-400 text-xs">
                        Score: <span className="text-white font-bold">{ticker.risk_score.toFixed(1)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Methodology Note */}
          <div className="pt-4 border-t border-gray-700">
            <div className="text-xs text-gray-500 italic">
              {bubbleRisk.methodology_notes}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
