"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from "recharts"
import { ArrowUpRight, ArrowDownRight, TrendingUp, TrendingDown, Layers } from "lucide-react"
import { format } from "date-fns"
import type { EIMASAnalysis } from "@/lib/types"

interface ArkAnalysisDashboardProps {
    data: EIMASAnalysis["ark_analysis"]
}

export function ArkAnalysisDashboard({ data }: ArkAnalysisDashboardProps) {
    if (!data) return null

    // Transform data for charts
    const increasesData = data.top_increases.slice(0, 5).map(item => ({
        name: item.ticker,
        value: item.weight_change_1d,
        sector: item.sector,
        fullData: item
    }))

    const decreasesData = data.top_decreases.slice(0, 5).map(item => ({
        name: item.ticker,
        value: Math.abs(item.weight_change_1d), // Use absolute value for bar length
        originalValue: item.weight_change_1d,
        sector: item.sector,
        fullData: item
    }))

    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload
            const isIncrease = d.originalValue === undefined ? true : d.originalValue > 0

            return (
                <div className="bg-[#1c2128] border border-[#30363d] p-3 rounded-lg shadow-xl text-xs">
                    <p className="font-bold text-gray-200 mb-1">{d.fullData.company} ({label})</p>
                    <div className="space-y-1">
                        <p className="text-gray-400">Sector: <span className="text-gray-300">{d.sector}</span></p>
                        <p className={isIncrease ? "text-green-400" : "text-red-400"}>
                            Change: {isIncrease ? "+" : ""}{d.originalValue ?? d.value}%
                        </p>
                        <p className="text-gray-400">Signal: <span className="text-gray-300">{d.fullData.signal_type}</span></p>
                    </div>
                </div>
            )
        }
        return null
    }

    return (
        <Card className="bg-[#161b22] border-[#30363d]">
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div className="space-y-1">
                        <CardTitle className="text-xl font-bold text-white flex items-center gap-2">
                            <Layers className="w-5 h-5 text-purple-400" />
                            ARK Invest Analysis
                        </CardTitle>
                        <CardDescription className="text-gray-400">
                            Daily ETF weight changes and consensus • {format(new Date(data.timestamp), "MMM dd, HH:mm")}
                        </CardDescription>
                    </div>
                </div>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Consensus Section */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2 bg-[#0d1117] p-4 rounded-lg border border-[#30363d]">
                        <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2">
                            <TrendingUp className="w-4 h-4 text-green-400" /> Consensus Buys
                        </h3>
                        <div className="flex flex-wrap gap-2">
                            {data.consensus_buys.length > 0 ? (
                                data.consensus_buys.map(ticker => (
                                    <Badge key={ticker} variant="outline" className="bg-green-500/10 text-green-400 border-green-500/20">
                                        {ticker}
                                    </Badge>
                                ))
                            ) : (
                                <span className="text-xs text-gray-500">None detected</span>
                            )}
                        </div>

                        {data.new_positions.length > 0 && (
                            <div className="mt-3 pt-3 border-t border-[#30363d]">
                                <h4 className="text-xs font-medium text-gray-500 mb-2">New Positions</h4>
                                <div className="flex flex-wrap gap-2">
                                    {data.new_positions.map(ticker => (
                                        <Badge key={ticker} variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/20">
                                            NEW {ticker}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="space-y-2 bg-[#0d1117] p-4 rounded-lg border border-[#30363d]">
                        <h3 className="text-sm font-medium text-gray-400 flex items-center gap-2">
                            <TrendingDown className="w-4 h-4 text-red-400" /> Consensus Sells
                        </h3>
                        <div className="flex flex-wrap gap-2">
                            {data.consensus_sells.length > 0 ? (
                                data.consensus_sells.map(ticker => (
                                    <Badge key={ticker} variant="outline" className="bg-red-500/10 text-red-400 border-red-500/20">
                                        {ticker}
                                    </Badge>
                                ))
                            ) : (
                                <span className="text-xs text-gray-500">None detected</span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Charts Section */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-2">
                    {/* Increases Chart */}
                    <div className="space-y-2">
                        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                            <ArrowUpRight className="w-4 h-4 text-green-500" /> Top Weight Increases
                        </h3>
                        <div className="h-[200px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={increasesData} layout="vertical" margin={{ left: 0, right: 30, top: 5, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#30363d" />
                                    <XAxis type="number" hide />
                                    <YAxis
                                        type="category"
                                        dataKey="name"
                                        width={50}
                                        tick={{ fill: '#8b949e', fontSize: 12 }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip cursor={{ fill: '#1c2128' }} content={<CustomTooltip />} />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                                        {increasesData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill="#238636" />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Decreases Chart */}
                    <div className="space-y-2">
                        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                            <ArrowDownRight className="w-4 h-4 text-red-500" /> Top Weight Decreases
                        </h3>
                        <div className="h-[200px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={decreasesData} layout="vertical" margin={{ left: 0, right: 30, top: 5, bottom: 5 }}>
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#30363d" />
                                    <XAxis type="number" hide />
                                    <YAxis
                                        type="category"
                                        dataKey="name"
                                        width={50}
                                        tick={{ fill: '#8b949e', fontSize: 12 }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip cursor={{ fill: '#1c2128' }} content={<CustomTooltip />} />
                                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={20}>
                                        {decreasesData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill="#da3633" />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
