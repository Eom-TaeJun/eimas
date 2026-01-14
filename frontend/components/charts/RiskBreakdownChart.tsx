"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
  ReferenceLine,
} from "recharts";

interface RiskBreakdownChartProps {
  base_risk: number;
  microstructure_adj: number;
  bubble_adj: number;
  final_risk: number;
}

export function RiskBreakdownChart({
  base_risk,
  microstructure_adj,
  bubble_adj,
  final_risk,
}: RiskBreakdownChartProps) {
  const data = [
    {
      name: "Base Risk",
      value: base_risk,
      fill: "#58a6ff", // blue
    },
    {
      name: "Micro Adj",
      value: microstructure_adj,
      fill: microstructure_adj >= 0 ? "#f85149" : "#3fb950", // red if positive, green if negative
    },
    {
      name: "Bubble Adj",
      value: bubble_adj,
      fill: "#f85149", // red
    },
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const value = payload[0].value;
      const sign = value >= 0 ? "+" : "";
      return (
        <div className="bg-[#161b22] border border-[#30363d] p-3 rounded-md shadow-lg">
          <p className="text-white font-semibold">{payload[0].payload.name}</p>
          <p className="text-[#58a6ff]">
            {sign}
            {value.toFixed(1)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <CardTitle className="text-white">Risk Score Breakdown</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <p className="text-sm text-gray-400">
            Final = Base + Micro Adj + Bubble Adj
          </p>
          <p className="text-2xl font-bold text-white mt-2">
            {final_risk.toFixed(1)} / 100
          </p>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart
            data={data}
            margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
          >
            <XAxis
              dataKey="name"
              tick={{ fill: "#ffffff" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fill: "#ffffff" }}
              axisLine={false}
              tickLine={false}
              domain={["auto", "auto"]}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0} stroke="#30363d" strokeDasharray="3 3" />
            <Bar dataKey="value" radius={[8, 8, 0, 0]}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-[#58a6ff] rounded"></div>
            <span className="text-gray-400">Base: {base_risk.toFixed(1)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded ${
                microstructure_adj >= 0 ? "bg-[#f85149]" : "bg-[#3fb950]"
              }`}
            ></div>
            <span className="text-gray-400">
              Micro: {microstructure_adj >= 0 ? "+" : ""}
              {microstructure_adj.toFixed(1)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-[#f85149] rounded"></div>
            <span className="text-gray-400">
              Bubble: +{bubble_adj.toFixed(1)}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
