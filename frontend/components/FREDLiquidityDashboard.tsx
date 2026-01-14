"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"
import { TrendingUp, TrendingDown, DollarSign, AlertCircle } from "lucide-react"

export function FREDLiquidityDashboard() {
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  })

  if (error || !analysis || !analysis.fred_summary) {
    return null
  }

  const fred = analysis.fred_summary

  const formatBillions = (value: number) => {
    if (value >= 1000) {
      return `$${(value / 1000).toFixed(2)}T`
    }
    return `$${value.toFixed(1)}B`
  }

  const getDeltaBadge = (delta: number) => {
    const isPositive = delta > 0
    return (
      <Badge
        variant="outline"
        className={`${
          isPositive
            ? "bg-green-500/10 text-green-400 border-green-500/20"
            : "bg-red-500/10 text-red-400 border-red-500/20"
        } text-xs`}
      >
        {isPositive ? "+" : ""}
        {formatBillions(delta)}
      </Badge>
    )
  }

  const getLiquidityColor = (regime: string) => {
    switch (regime.toLowerCase()) {
      case "abundant":
        return "bg-green-500/10 text-green-400 border-green-500/20"
      case "tight":
        return "bg-red-500/10 text-red-400 border-red-500/20"
      default:
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
    }
  }

  const getCurveColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "normal":
        return "bg-green-500/10 text-green-400 border-green-500/20"
      case "inverted":
        return "bg-red-500/10 text-red-400 border-red-500/20"
      default:
        return "bg-gray-500/10 text-gray-400 border-gray-500/20"
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <DollarSign className="w-6 h-6 text-blue-400" />
        <h2 className="text-xl font-bold text-white">FRED Liquidity Metrics</h2>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* RRP Card */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-gray-400">
              Reverse Repo (RRP)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white">{formatBillions(fred.rrp)}</div>
              <div className="flex items-center gap-2">
                {fred.rrp_delta > 0 ? (
                  <TrendingUp className="w-4 h-4 text-green-400" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-red-400" />
                )}
                {getDeltaBadge(fred.rrp_delta)}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* TGA Card */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-gray-400">
              Treasury General Account (TGA)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white">{formatBillions(fred.tga)}</div>
              <div className="flex items-center gap-2">
                {fred.tga_delta > 0 ? (
                  <TrendingUp className="w-4 h-4 text-green-400" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-red-400" />
                )}
                {getDeltaBadge(fred.tga_delta)}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Net Liquidity Card */}
        <Card className="bg-[#161b22] border-[#30363d] border-2 border-blue-500/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-blue-400">Net Liquidity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-3xl font-bold text-blue-400">
                {formatBillions(fred.net_liquidity)}
              </div>
              <Badge variant="outline" className={getLiquidityColor(fred.liquidity_regime)}>
                {fred.liquidity_regime}
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Fed Assets Card */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-gray-400">Fed Assets</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-2xl font-bold text-white">
                {formatBillions(fred.fed_assets * 1000)}
              </div>
              <div className="text-xs text-gray-400">Total Balance Sheet</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Rates Row */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-gray-400">Fed Funds Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{fred.fed_funds.toFixed(2)}%</div>
          </CardContent>
        </Card>

        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-gray-400">10Y Treasury</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{fred.treasury_10y.toFixed(2)}%</div>
          </CardContent>
        </Card>

        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-gray-400">Yield Curve</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="text-xl font-bold text-white">
                {fred.spread_10y2y > 0 ? "+" : ""}
                {(fred.spread_10y2y * 100).toFixed(0)}bp
              </div>
              <Badge variant="outline" className={getCurveColor(fred.curve_status)}>
                {fred.curve_status}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
