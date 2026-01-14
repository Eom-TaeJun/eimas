"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"
import { AlertTriangle, Shield } from "lucide-react"

export function CryptoStressTest() {
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  })

  if (error || !analysis || !analysis.crypto_stress_test) {
    return null
  }

  const test = analysis.crypto_stress_test

  const formatMoney = (value: number) => {
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`
    if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`
    return `$${value.toFixed(0)}`
  }

  const getRiskColor = (rating: string) => {
    if (rating.includes("LOW") || rating.includes("낮음")) {
      return "bg-green-500/10 text-green-400 border-green-500/20"
    }
    if (rating.includes("MEDIUM") || rating.includes("중간")) {
      return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
    }
    return "bg-red-500/10 text-red-400 border-red-500/20"
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Shield className="w-6 h-6 text-orange-400" />
        <h2 className="text-xl font-bold text-white">Crypto Stress Test</h2>
        <Badge variant="outline" className="bg-orange-500/10 text-orange-400 border-orange-500/20">
          {test.scenario}
        </Badge>
      </div>

      {/* Summary Card */}
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-gray-400">
            Depeg Scenario Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            <div>
              <div className="text-xs text-gray-400 mb-1">Total Value at Risk</div>
              <div className="text-xl font-bold text-white">{formatMoney(test.total_value)}</div>
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">Depeg Probability</div>
              <div className="text-xl font-bold text-orange-400">
                {test.depeg_probability_pct}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">Estimated Loss</div>
              <div className="text-xl font-bold text-red-400">
                {formatMoney(test.estimated_loss_under_stress)}
              </div>
              <div className="text-xs text-gray-400">{test.estimated_loss_pct} of total</div>
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">Risk Rating</div>
              <Badge variant="outline" className={getRiskColor(test.risk_rating)}>
                {test.risk_rating}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Breakdown by Coin */}
      <div className="grid gap-3 md:grid-cols-3">
        {test.breakdown_by_coin.map((coin, idx) => (
          <Card key={idx} className="bg-[#161b22] border-[#30363d]">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium text-white">{coin.ticker}</CardTitle>
                <Badge
                  variant="outline"
                  className="bg-blue-500/10 text-blue-400 border-blue-500/20 text-xs"
                >
                  {(coin.weight * 100).toFixed(1)}%
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div>
                  <div className="text-xs text-gray-400">Amount</div>
                  <div className="text-lg font-bold text-white">{formatMoney(coin.amount)}</div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <div className="text-gray-400">Depeg Prob.</div>
                    <div className="text-orange-400 font-mono">
                      {(coin.depeg_probability * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Expected Loss</div>
                    <div className="text-red-400 font-mono">{formatMoney(coin.expected_loss)}</div>
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">Loss Rate</div>
                  <div className="text-xs text-gray-300 font-mono">
                    {(coin.loss_rate * 100).toFixed(3)}%
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Methodology Note */}
      <Card className="bg-blue-500/10 border border-blue-500/20">
        <CardContent className="pt-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
            <div className="space-y-1">
              <div className="text-sm font-medium text-blue-400">Methodology</div>
              <div className="text-xs text-gray-300">{test.methodology_note}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
