"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from "recharts"
import { Activity, Gauge, TrendingUp, TrendingDown } from "lucide-react"
import type { EIMASAnalysis } from "@/lib/types"

interface MarketSentimentGaugeProps {
    sentiment: EIMASAnalysis["sentiment_analysis"]
    hft: EIMASAnalysis["hft_microstructure"]
}

export function MarketSentimentGauge({ sentiment, hft }: MarketSentimentGaugeProps) {
    if (!sentiment || !hft) return null

    // Fear & Greed Data
    const fgValue = sentiment.fear_greed.value
    const fgData = [
        { name: "Extreme Fear", value: 25, color: "#ef4444" },
        { name: "Fear", value: 25, color: "#f97316" },
        { name: "Greed", value: 25, color: "#3b82f6" },
        { name: "Extreme Greed", value: 25, color: "#22c55e" },
    ]

    // Calculate needle rotation
    const needleRotation = (fgValue / 100) * 180 - 90

    // HFT Data
    const buyRatio = hft.tick_rule.buy_ratio * 100
    const sellRatio = hft.tick_rule.sell_ratio * 100

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Fear & Greed Gauge */}
            <Card className="bg-[#161b22] border-[#30363d]">
                <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-bold text-white flex items-center gap-2">
                        <Gauge className="w-5 h-5 text-yellow-400" />
                        Fear & Greed Level
                    </CardTitle>
                    <CardDescription className="text-gray-400">
                        Current Market Emotion
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="relative h-[180px] flex flex-col items-center justify-center">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={fgData}
                                    cx="50%"
                                    cy="70%"
                                    startAngle={180}
                                    endAngle={0}
                                    innerRadius="60%"
                                    outerRadius="80%"
                                    paddingAngle={2}
                                    dataKey="value"
                                    stroke="none"
                                >
                                    {fgData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                            </PieChart>
                        </ResponsiveContainer>

                        {/* Needle */}
                        <div
                            className="absolute top-[70%] left-[50%] w-[2px] h-[60px] bg-white origin-bottom transition-all duration-1000 ease-out"
                            style={{ transform: `translateX(-50%) translateY(-100%) rotate(${needleRotation}deg)` }}
                        >
                            <div className="absolute bottom-0 left-[-4px] w-[10px] h-[10px] rounded-full bg-white" />
                        </div>

                        <div className="absolute bottom-4 text-center">
                            <div className="text-3xl font-bold text-white">{fgValue}</div>
                            <div className="text-sm font-medium text-gray-400 capitalize">{sentiment.fear_greed.level}</div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* HFT Pressure */}
            <Card className="bg-[#161b22] border-[#30363d]">
                <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-bold text-white flex items-center gap-2">
                        <Activity className="w-5 h-5 text-cyan-400" />
                        HFT Order Pressure
                    </CardTitle>
                    <CardDescription className="text-gray-400">
                        Microstructure Analysis (Tick Rule)
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6 pt-4">
                    <div className="space-y-4">
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-green-400 font-medium flex items-center gap-1">
                                    <TrendingUp className="w-3 h-3" /> Buy Pressure
                                </span>
                                <span className="text-white font-mono">{buyRatio.toFixed(1)}%</span>
                            </div>
                            <div className="h-2 w-full bg-[#0d1117] rounded-full overflow-hidden border border-[#30363d]">
                                <div
                                    className="h-full bg-green-500 transition-all duration-500"
                                    style={{ width: `${buyRatio}%` }}
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-red-400 font-medium flex items-center gap-1">
                                    <TrendingDown className="w-3 h-3" /> Sell Pressure
                                </span>
                                <span className="text-white font-mono">{sellRatio.toFixed(1)}%</span>
                            </div>
                            <div className="h-2 w-full bg-[#0d1117] rounded-full overflow-hidden border border-[#30363d]">
                                <div
                                    className="h-full bg-red-500 transition-all duration-500"
                                    style={{ width: `${sellRatio}%` }}
                                />
                            </div>
                        </div>
                    </div>

                    <div className="pt-4 border-t border-[#30363d]">
                        <div className="flex justify-between items-center text-xs text-gray-400">
                            <span>Dominance</span>
                            <span className={`px-2 py-1 rounded border ${buyRatio > sellRatio
                                    ? "bg-green-500/10 border-green-500/20 text-green-400"
                                    : "bg-red-500/10 border-red-500/20 text-red-400"
                                }`}>
                                {hft.tick_rule.interpretation}
                            </span>
                        </div>
                        {sentiment.news_sentiment && (
                            <div className="flex justify-between items-center text-xs text-gray-400 mt-2">
                                <span>News Sentiment</span>
                                <span className="text-gray-200">{sentiment.news_sentiment.overall}</span>
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}
