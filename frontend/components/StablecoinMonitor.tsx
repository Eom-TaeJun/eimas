"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import { formatBillions, safeToFixed, safeNumber } from "@/lib/format"
import type { EIMASAnalysis } from "@/lib/types"
import { Coins, TrendingUp, TrendingDown, AlertTriangle } from "lucide-react"

export function StablecoinMonitor() {
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  })

  if (
    error ||
    !analysis ||
    !analysis.genius_act_signals ||
    analysis.genius_act_signals.length === 0
  ) {
    return null
  }

  const stablecoinSignal = analysis.genius_act_signals.find(
    (s: any) => s.type === "stablecoin_analysis"
  )

  if (!stablecoinSignal || !stablecoinSignal.metadata) {
    return null
  }

  const meta = stablecoinSignal.metadata
  const components = meta.components || {}

  // Using imported formatBillions from @/lib/format

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "growing":
        return "bg-green-500/10 text-green-400 border-green-500/20"
      case "stable":
        return "bg-blue-500/10 text-blue-400 border-blue-500/20"
      case "declining":
        return "bg-red-500/10 text-red-400 border-red-500/20"
      default:
        return "bg-gray-500/10 text-gray-400 border-gray-500/20"
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case "growing":
        return <TrendingUp className="w-4 h-4 text-green-400" />
      case "declining":
        return <TrendingDown className="w-4 h-4 text-red-400" />
      default:
        return <AlertTriangle className="w-4 h-4 text-blue-400" />
    }
  }

  const getGeniusActColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "expanding":
        return "bg-green-500/10 text-green-400 border-green-500/20"
      case "draining":
        return "bg-red-500/10 text-red-400 border-red-500/20"
      default:
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Coins className="w-6 h-6 text-yellow-400" />
        <h2 className="text-xl font-bold text-white">Genius Act Stablecoin Monitor</h2>
      </div>

      {/* Overall Summary */}
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-gray-400">
            Total Stablecoin Market
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-3">
            <div>
              <div className="text-xs text-gray-400 mb-1">Market Cap</div>
              <div className="text-2xl font-bold text-white">
                {formatBillions(meta.total_market_cap)}
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">7-Day Change</div>
              <div
                className={`text-2xl font-bold ${meta.total_delta_7d > 0 ? "text-green-400" : "text-red-400"
                  }`}
              >
                {meta.total_delta_7d > 0 ? "+" : ""}
                {formatBillions(meta.total_delta_7d)}
              </div>
              <div
                className={`text-xs ${meta.total_delta_pct > 0 ? "text-green-400" : "text-red-400"
                  }`}
              >
                {safeNumber(meta.total_delta_pct) > 0 ? "+" : ""}
                {safeToFixed(meta.total_delta_pct, 2)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-400 mb-1">Genius Act Status</div>
              <Badge variant="outline" className={getGeniusActColor(meta.genius_act_status)}>
                {meta.genius_act_status.toUpperCase()}
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Individual Stablecoins */}
      <div className="grid gap-4 md:grid-cols-3">
        {Object.entries(components).map(([coin, data]: [string, any]) => (
          <Card key={coin} className="bg-[#161b22] border-[#30363d]">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm font-medium text-gray-400">{coin}</CardTitle>
                <div className="flex items-center gap-1">
                  {getStatusIcon(data.status)}
                  <Badge variant="outline" className={getStatusColor(data.status)}>
                    {data.status}
                  </Badge>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div>
                  <div className="text-xs text-gray-400">Current</div>
                  <div className="text-xl font-bold text-white">
                    {formatBillions(data.current)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-400">7-Day Change</div>
                  <div
                    className={`text-sm font-mono ${data.delta_7d > 0 ? "text-green-400" : "text-red-400"
                      }`}
                  >
                    {data.delta_7d > 0 ? "+" : ""}
                    {formatBillions(data.delta_7d)}
                  </div>
                  <div
                    className={`text-xs ${data.delta_pct > 0 ? "text-green-400" : "text-red-400"
                      }`}
                  >
                    {safeNumber(data.delta_pct) > 0 ? "+" : ""}
                    {safeToFixed(data.delta_pct, 2)}%
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Signal Explanation */}
      <Card className="bg-yellow-500/10 border border-yellow-500/20">
        <CardContent className="pt-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5 flex-shrink-0" />
            <div className="space-y-1">
              <div className="text-sm font-medium text-yellow-400">
                {stablecoinSignal.description}
              </div>
              <div className="text-xs text-gray-300">{stablecoinSignal.why}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
