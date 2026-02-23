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
  extended_data_adj?: number;
  final_risk: number;
}

export function RiskBreakdownChart({
  base_risk,
  microstructure_adj,
  bubble_adj,
  extended_data_adj = 0,
  final_risk,
}: RiskBreakdownChartProps) {
  const data = [
    {
      name: "Base Risk",
      value: base_risk,
      fill: "#60A5FA", // blue
    },
    {
      name: "Micro Adj",
      value: microstructure_adj,
      fill: microstructure_adj >= 0 ? "#ef4444" : "#10B981", // red if positive, green if negative
    },
    {
      name: "Bubble Adj",
      value: bubble_adj,
      fill: bubble_adj >= 0 ? "#ef4444" : "#10B981", // red if positive, green if negative
    },
    {
      name: "Extended Adj",
      value: extended_data_adj,
      fill: extended_data_adj >= 0 ? "#ef4444" : "#10B981", // red if positive, green if negative
    },
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const value = payload[0].value;
      const sign = value >= 0 ? "+" : "";
      return (
        <div className="bg-surface-card border border-border p-3 rounded-md shadow-lg">
          <p className="text-white font-semibold">{payload[0].payload.name}</p>
          <p className="text-primary-400">
            {sign}
            {value.toFixed(1)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-surface border-border">
      <CardHeader>
        <CardTitle className="text-white">Risk Score Breakdown</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <p className="text-sm text-gray-400">
            Final = Base + Micro Adj + Bubble Adj + Extended Adj
          </p>
          <p className="text-2xl font-bold text-white mt-2">
            {final_risk.toFixed(1)} / 100
          </p>
          <p className="text-xs text-gray-500 mt-1">
            {base_risk.toFixed(1)} + ({microstructure_adj >= 0 ? "+" : ""}{microstructure_adj.toFixed(1)}) + ({bubble_adj >= 0 ? "+" : ""}{bubble_adj.toFixed(1)}) + ({extended_data_adj >= 0 ? "+" : ""}{extended_data_adj.toFixed(1)}) = {final_risk.toFixed(1)}
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
        <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-primary-400 rounded"></div>
            <span className="text-gray-400">Base: {base_risk.toFixed(1)}</span>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-3 h-3 rounded ${
                microstructure_adj >= 0 ? "bg-red-500" : "bg-secondary"
              }`}
            ></div>
            <span className="text-gray-400">
              Micro: {microstructure_adj >= 0 ? "+" : ""}
              {microstructure_adj.toFixed(1)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded ${
                bubble_adj >= 0 ? "bg-red-500" : "bg-secondary"
              }`}></div>
            <span className="text-gray-400">
              Bubble: {bubble_adj >= 0 ? "+" : ""}{bubble_adj.toFixed(1)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-3 h-3 rounded ${
                extended_data_adj >= 0 ? "bg-red-500" : "bg-secondary"
              }`}></div>
            <span className="text-gray-400">
              Extended: {extended_data_adj >= 0 ? "+" : ""}{extended_data_adj.toFixed(1)}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
