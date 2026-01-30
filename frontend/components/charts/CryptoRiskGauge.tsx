"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"

export function CryptoRiskGauge() {
    const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
        refreshInterval: 5000,
    })

    // Safe fallback if data is missing
    if (error || !analysis || !analysis.crypto_stress_test) {
        return (
            <Card className="bg-[#161b22] border-[#30363d] h-full">
                <CardHeader>
                    <div className="h-6 w-32 bg-gray-700/50 rounded animate-pulse" />
                </CardHeader>
                <CardContent className="h-[200px] flex items-center justify-center">
                    <div className="text-gray-500 text-sm">Waiting for stress test data...</div>
                </CardContent>
            </Card>
        );
    }

    const test = analysis.crypto_stress_test;
    const value = test.depeg_probability || 0; // 0 to 1 scale typically
    // If value is 0-1, convert to 0-100. If typically small (e.g. 0.05), maybe log scale?
    // Assuming depeg_probability is probability (0-1). Display as % (0-100).
    const percentValue = Math.min(Math.max(value * 100, 0), 100);

    // Gauge data: [Value, Remainder]
    // We want a semi-circle gauge (180 degrees)
    const data = [
        { name: "Risk", value: percentValue },
        { name: "Safe", value: 100 - percentValue },
    ];

    // Determine color based on risk
    let riskColor = "#3fb950"; // Green (Low)
    if (percentValue > 30) riskColor = "#d29922"; // Yellow (Medium)
    if (percentValue > 70) riskColor = "#f85149"; // Red (High)

    const cx = "50%";
    const cy = "70%";
    const iR = 60;
    const oR = 80;

    return (
        <Card className="bg-[#161b22] border-[#30363d] h-full flex flex-col">
            <CardHeader className="pb-2">
                <CardTitle className="text-white text-lg">Depeg Risk Gauge</CardTitle>
                <CardDescription className="text-gray-400">
                    Probability of Stablecoin Deviation
                </CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col items-center justify-center -mt-4">
                <div className="w-full h-[160px] relative">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                dataKey="value"
                                startAngle={180}
                                endAngle={0}
                                data={data}
                                cx={cx}
                                cy={cy}
                                innerRadius={iR}
                                outerRadius={oR}
                                stroke="none"
                            >
                                <Cell fill={riskColor} />
                                <Cell fill="#30363d" />
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                    {/* Central Label */}
                    <div className="absolute top-[60%] left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-center">
                        <div className="text-3xl font-bold text-white">
                            {percentValue.toFixed(1)}%
                        </div>
                        <div className={`text-xs font-medium uppercase mt-1`} style={{ color: riskColor }}>
                            {test.risk_rating || "Unknown"}
                        </div>
                    </div>
                </div>
                <div className="text-xs text-gray-400 text-center px-4">
                    Scenario: <span className="text-gray-300">{test.scenario}</span>
                </div>
            </CardContent>
        </Card>
    )
}
