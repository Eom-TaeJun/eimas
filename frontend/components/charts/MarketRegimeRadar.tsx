"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Tooltip } from "recharts"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"

export function MarketRegimeRadar() {
    const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
        refreshInterval: 5000,
    })

    // Default data structure if not available
    const defaultProbabilities = {
        Bull: 0.33,
        Neutral: 0.34,
        Bear: 0.33
    };

    const probabilities = analysis?.regime?.gmm_probabilities || defaultProbabilities;

    const data = [
        { subject: 'Bullish', A: probabilities.Bull * 100, fullMark: 100 },
        { subject: 'Neutral', A: probabilities.Neutral * 100, fullMark: 100 },
        { subject: 'Bearish', A: probabilities.Bear * 100, fullMark: 100 },
    ];

    // Determine dominant regime color
    let color = "#8884d8";
    if (probabilities.Bull > 0.5) color = "#3fb950";
    else if (probabilities.Bear > 0.5) color = "#f85149";
    else color = "#d29922"; // Neutral/Mixed

    return (
        <Card className="bg-[#161b22] border-[#30363d] h-full">
            <CardHeader>
                <CardTitle className="text-white text-lg">Market Regime Probability</CardTitle>
                <CardDescription className="text-gray-400">
                    Gaussian Mixture Model (GMM) Analysis
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="h-[250px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
                            <PolarGrid stroke="#30363d" />
                            <PolarAngleAxis dataKey="subject" tick={{ fill: '#8b949e', fontSize: 12 }} />
                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                            <Radar
                                name="Probability"
                                dataKey="A"
                                stroke={color}
                                fill={color}
                                fillOpacity={0.5}
                            />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1c2128', borderColor: '#30363d', color: '#fff' }}
                                itemStyle={{ color: color }}
                                formatter={(value: number) => [`${value.toFixed(1)}%`, 'Probability']}
                            />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>
                <div className="text-center mt-2">
                    <div className="inline-block px-3 py-1 rounded-full text-xs font-bold border"
                        style={{ borderColor: `${color}40`, backgroundColor: `${color}10`, color: color }}>
                        Current: {analysis?.regime?.regime || "Unknown"}
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}
