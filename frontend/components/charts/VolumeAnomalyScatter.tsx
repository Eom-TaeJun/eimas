"use client"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"

export function VolumeAnomalyScatter() {
    const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
        refreshInterval: 5000,
    })

    if (error || !analysis || !analysis.volume_anomalies || analysis.volume_anomalies.length === 0) {
        return null
    }

    const data = analysis.volume_anomalies.map(item => ({
        x: item.price_change_1d, // Price Change %
        y: item.z_score,         // Volume Z-Score
        volumeRatio: item.volume_ratio,
        ticker: item.ticker,
        type: item.anomaly_type
    }));

    const CustomTooltip = ({ active, payload }: any) => {
        if (active && payload && payload.length) {
            const d = payload[0].payload;
            return (
                <div className="bg-[#1c2128] border border-[#30363d] p-3 rounded-lg shadow-xl text-xs">
                    <p className="font-bold text-gray-200 mb-1">{d.ticker}</p>
                    <p className="text-gray-400">Z-Score: <span className="text-blue-400">{d.y.toFixed(1)}σ</span></p>
                    <p className="text-gray-400">Price 1D: <span className={d.x > 0 ? "text-green-400" : "text-red-400"}>{d.x > 0 ? "+" : ""}{d.x.toFixed(2)}%</span></p>
                    <p className="text-gray-400">Vol Ratio: <span className="text-yellow-400">{d.volumeRatio.toFixed(1)}x</span></p>
                </div>
            );
        }
        return null;
    };

    return (
        <Card className="bg-[#161b22] border-[#30363d]">
            <CardHeader>
                <CardTitle className="text-white text-lg">Volume Anomaly Scatter</CardTitle>
                <CardDescription className="text-gray-400">
                    Distribution of abnormal volume events vs price change
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="h-[300px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
                            <XAxis
                                type="number"
                                dataKey="x"
                                name="Price Change"
                                unit="%"
                                stroke="#8b949e"
                                tick={{ fontSize: 12 }}
                                label={{ value: "Price Change (%)", position: "insideBottom", offset: -10, fill: "#8b949e" }}
                            />
                            <YAxis
                                type="number"
                                dataKey="y"
                                name="Z-Score"
                                stroke="#8b949e"
                                tick={{ fontSize: 12 }}
                                label={{ value: "Volume Z-Score", angle: -90, position: "insideLeft", fill: "#8b949e" }}
                            />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
                            <Scatter name="Anomalies" data={data} fill="#8884d8">
                                {data.map((entry, index) => {
                                    let fill = "#a371f7"; // purple (neutral/unknown)
                                    if (entry.type.includes("surge")) fill = "#3fb950"; // green
                                    if (entry.type.includes("drop")) fill = "#f85149"; // red
                                    return <Cell key={`cell-${index}`} fill={fill} />
                                })}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}
