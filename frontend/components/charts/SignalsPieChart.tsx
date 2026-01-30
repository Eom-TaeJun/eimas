"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts"
import useSWR from "swr"
import { fetchSignals } from "@/lib/api"
import type { Signal } from "@/lib/types"

const COLORS = {
    BUY: "#3fb950",   // Green
    SELL: "#f85149",  // Red
    HOLD: "#d29922",  // Yellow
}

export function SignalsPieChart() {
    const { data: signals, error } = useSWR<Signal[]>("signals", () => fetchSignals(20), { refreshInterval: 60000 })

    if (error || !signals) return null;

    // Process data for pie chart
    const counts = signals.reduce((acc, signal) => {
        const action = signal.action || "HOLD";
        acc[action] = (acc[action] || 0) + 1;
        return acc;
    }, {} as Record<string, number>);

    const chartData = Object.entries(counts).map(([name, value]) => ({
        name,
        value,
    }));

    if (chartData.length === 0) return null;

    return (
        <Card className="bg-[#161b22] border-[#30363d] h-full">
            <CardHeader>
                <CardTitle className="text-gray-200 text-sm">Signal Distribution</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="h-[200px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={chartData}
                                cx="50%"
                                cy="50%"
                                innerRadius={50}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                            >
                                {chartData.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={COLORS[entry.name as keyof typeof COLORS] || "#8884d8"}
                                        stroke="rgba(0,0,0,0)"
                                    />
                                ))}
                            </Pie>
                            <Tooltip
                                contentStyle={{ backgroundColor: '#161b22', borderColor: '#30363d', color: '#fff' }}
                                itemStyle={{ color: '#fff' }}
                            />
                            <Legend verticalAlign="bottom" height={36} iconType="circle" />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            </CardContent>
        </Card>
    )
}
