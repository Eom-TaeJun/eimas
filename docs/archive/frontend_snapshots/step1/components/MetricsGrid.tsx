"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchPortfolio, fetchMarketRegime, fetchRisk } from "@/lib/api"
import type { Portfolio, MarketRegime, Risk } from "@/lib/types"

export function MetricsGrid() {
  const { data: portfolio, error: portfolioError } = useSWR<Portfolio>("portfolio", fetchPortfolio, {
    refreshInterval: 60000,
  })

  const { data: regime, error: regimeError } = useSWR<MarketRegime>("regime", () => fetchMarketRegime("SPY"), {
    refreshInterval: 60000,
  })

  const { data: risk, error: riskError } = useSWR<Risk>("risk", fetchRisk, { refreshInterval: 60000 })

  // Consensus signal (mock for now - would come from an API endpoint)
  const consensus = {
    action: "BUY" as const,
    conviction: 72,
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value)
  }

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

  const getActionColor = (action: string) => {
    switch (action) {
      case "BUY":
        return "bg-green-500/10 text-green-500 border-green-500/20"
      case "SELL":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      case "HOLD":
        return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-2">
      {/* Portfolio Value Card */}
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-gray-400">Portfolio Value</CardTitle>
        </CardHeader>
        <CardContent>
          {portfolioError ? (
            <p className="text-sm text-red-400">Failed to load</p>
          ) : !portfolio ? (
            <p className="text-sm text-gray-400">Loading...</p>
          ) : (
            <div className="space-y-2">
              <div className="text-3xl font-bold text-white">{formatCurrency(portfolio.total_value)}</div>
              <div className="flex items-center gap-2 text-sm">
                <span className={portfolio.total_pnl >= 0 ? "text-green-400" : "text-red-400"}>
                  {portfolio.total_pnl >= 0 ? "+" : ""}
                  {formatCurrency(portfolio.total_pnl)}
                </span>
                <span className={portfolio.total_pnl_pct >= 0 ? "text-green-400" : "text-red-400"}>
                  ({portfolio.total_pnl_pct >= 0 ? "+" : ""}
                  {portfolio.total_pnl_pct.toFixed(2)}%)
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Market Regime Card */}
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-gray-400">Market Regime</CardTitle>
        </CardHeader>
        <CardContent>
          {regimeError ? (
            <p className="text-sm text-red-400">Failed to load</p>
          ) : !regime ? (
            <p className="text-sm text-gray-400">Loading...</p>
          ) : (
            <div className="space-y-3">
              <div className="text-2xl font-bold text-white">{regime.regime}</div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <span className="text-gray-400">Trend:</span> <span className="text-white">{regime.trend}</span>
                </div>
                <div>
                  <span className="text-gray-400">Volatility:</span>{" "}
                  <span className="text-white">{regime.volatility}</span>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-400 text-sm">Confidence:</span>
                <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                  {regime.confidence}%
                </Badge>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Consensus Signal Card */}
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-gray-400">Consensus Signal</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div>
              <Badge variant="outline" className={`text-lg font-bold px-3 py-1 ${getActionColor(consensus.action)}`}>
                {consensus.action}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-gray-400 text-sm">Conviction:</span>
              <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                {consensus.conviction}%
              </Badge>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Risk Level Card */}
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-sm font-medium text-gray-400">Risk Level</CardTitle>
        </CardHeader>
        <CardContent>
          {riskError ? (
            <p className="text-sm text-red-400">Failed to load</p>
          ) : !risk ? (
            <p className="text-sm text-gray-400">Loading...</p>
          ) : (
            <div className="space-y-3">
              <div>
                <Badge variant="outline" className={`text-2xl font-bold px-4 py-2 ${getRiskColor(risk.risk_level)}`}>
                  {risk.risk_level}
                </Badge>
              </div>
              <div className="text-sm text-gray-400">
                Risk Score: <span className="text-white">{risk.risk_score.toFixed(1)}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
