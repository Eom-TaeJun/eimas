"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import { formatBillions, safeToFixed, safeNumber } from "@/lib/format"
import type { EIMASAnalysis } from "@/lib/types"
import { TrendingUp, TrendingDown, DollarSign } from "lucide-react"

export function FREDLiquidityDashboard() {
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  })

  if (error || !analysis || !analysis.fred_summary) {
    return null
  }

  const fred = analysis.fred_summary

  const getDeltaBadge = (delta: number) => {
    const isPositive = delta > 0
    return (
      <Badge
        variant="outline"
        className={`${isPositive
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
      case "abundant": return "bg-green-500/10 text-green-400 border-green-500/20"
      case "tight": return "bg-red-500/10 text-red-400 border-red-500/20"
      default: return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
    }
  }

  const getCurveColor = (status: string) => {
    switch (status.toLowerCase()) {
      case "normal": return "bg-green-500/10 text-green-400 border-green-500/20"
      case "inverted": return "bg-red-500/10 text-red-400 border-red-500/20"
      default: return "bg-gray-500/10 text-gray-400 border-gray-500/20"
    }
  }

  // Calculate proportions for composition bar
  const totalShown = fred.net_liquidity + fred.tga + fred.rrp;
  // Use fed_assets as denominator or sum of components? Usually fed_assets is much larger.
  // Let's use fed_assets as the base for the "Fed Balance Sheet" visual.
  const assetsBase = fred.fed_assets || totalShown;

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
            <CardTitle className="text-xs font-medium text-gray-400">Reverse Repo (RRP)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{formatBillions(fred.rrp)}</div>
            <div className="flex items-center gap-2 mt-1">
              {fred.rrp_delta > 0 ? <TrendingUp className="w-4 h-4 text-green-400" /> : <TrendingDown className="w-4 h-4 text-red-400" />}
              {getDeltaBadge(fred.rrp_delta)}
            </div>
            <div className="mt-3 h-1.5 w-full bg-gray-700/30 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500" style={{ width: `${Math.min((fred.rrp / assetsBase) * 100, 100)}%` }} />
            </div>
            <div className="text-[10px] text-gray-500 mt-1">{(fred.rrp / assetsBase * 100).toFixed(1)}% of Fed Assets</div>
          </CardContent>
        </Card>

        {/* TGA Card */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-gray-400">Treasury Gen. Acct (TGA)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{formatBillions(fred.tga)}</div>
            <div className="flex items-center gap-2 mt-1">
              {fred.tga_delta > 0 ? <TrendingUp className="w-4 h-4 text-green-400" /> : <TrendingDown className="w-4 h-4 text-red-400" />}
              {getDeltaBadge(fred.tga_delta)}
            </div>
            <div className="mt-3 h-1.5 w-full bg-gray-700/30 rounded-full overflow-hidden">
              <div className="h-full bg-yellow-500" style={{ width: `${Math.min((fred.tga / assetsBase) * 100, 100)}%` }} />
            </div>
            <div className="text-[10px] text-gray-500 mt-1">{(fred.tga / assetsBase * 100).toFixed(1)}% of Fed Assets</div>
          </CardContent>
        </Card>

        {/* Net Liquidity Card */}
        <Card className="bg-[#161b22] border-[#30363d] border-l-4 border-l-blue-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-blue-400">Net Liquidity</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">{formatBillions(fred.net_liquidity)}</div>
            <Badge variant="outline" className={`mt-2 ${getLiquidityColor(fred.liquidity_regime)}`}>
              {fred.liquidity_regime}
            </Badge>
          </CardContent>
        </Card>

        {/* Yield Curve Card */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-gray-400">Yield Curve (10Y-2Y)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${fred.spread_10y2y < 0 ? "text-red-400" : "text-green-400"}`}>
              {fred.spread_10y2y.toFixed(2)}%
            </div>
            <Badge variant="outline" className={`mt-2 ${getCurveColor(fred.curve_status)}`}>
              {fred.curve_status}
            </Badge>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Visualizations */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Liquidity Composition Bar */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">Fed Liquidity Composition</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 pt-2">
              <div className="h-8 w-full flex rounded-md overflow-hidden bg-gray-800">
                <div className="h-full bg-green-500 flex items-center justify-center text-xs text-black font-bold transition-all duration-500"
                  style={{ width: `${Math.min((fred.net_liquidity / assetsBase) * 100, 100)}%` }} title="Net Liquidity">
                  {((fred.net_liquidity / assetsBase) * 100).toFixed(1)}%
                </div>
                <div className="h-full bg-yellow-500 flex items-center justify-center text-xs text-black font-bold transition-all duration-500"
                  style={{ width: `${Math.min((fred.tga / assetsBase) * 100, 100)}%` }} title="TGA">
                  TGA
                </div>
                <div className="h-full bg-blue-500 flex items-center justify-center text-xs text-white font-bold transition-all duration-500"
                  style={{ width: `${Math.min((fred.rrp / assetsBase) * 100, 100)}%` }} title="RRP">
                  RRP
                </div>
              </div>
              <div className="flex justify-between text-xs text-gray-400 px-1">
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-green-500" /> Net Liquidity</div>
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-yellow-500" /> TGA</div>
                <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-blue-500" /> RRP</div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Key Interest Rates */}
        <Card className="bg-[#161b22] border-[#30363d]">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-300">Key Interest Rates</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 pt-2">
              {/* Fed Funds */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-400">Fed Funds Rate</span>
                  <span className="text-white font-mono">{fred.fed_funds.toFixed(2)}%</span>
                </div>
                <div className="w-full bg-gray-700/30 h-2 rounded-full overflow-hidden">
                  <div className="bg-purple-500 h-full transition-all duration-500" style={{ width: `${Math.min((fred.fed_funds / 6) * 100, 100)}%` }} />
                </div>
              </div>

              {/* 10Y Treasury */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-400">10Y Treasury</span>
                  <span className="text-white font-mono">{fred.treasury_10y.toFixed(2)}%</span>
                </div>
                <div className="w-full bg-gray-700/30 h-2 rounded-full overflow-hidden">
                  <div className="bg-orange-500 h-full transition-all duration-500" style={{ width: `${Math.min((fred.treasury_10y / 6) * 100, 100)}%` }} />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
