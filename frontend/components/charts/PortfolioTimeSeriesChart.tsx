"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";

interface PortfolioTimeSeriesProps {
  data?: Array<{
    timestamp: string;
    total_value: number;
    pnl: number;
    pnl_pct: number;
  }>;
}

// Mock historical data for demonstration
const generateMockHistory = (days: number = 30) => {
  const data = [];
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;
  let value = 100000;

  for (let i = days; i >= 0; i--) {
    const change = (Math.random() - 0.48) * 2000; // Slight upward bias
    value = Math.max(90000, Math.min(120000, value + change));
    const pnl = value - 100000;
    const pnl_pct = (pnl / 100000) * 100;

    data.push({
      timestamp: new Date(now - i * dayMs).toISOString().split("T")[0],
      total_value: Math.round(value),
      pnl: Math.round(pnl),
      pnl_pct: parseFloat(pnl_pct.toFixed(2)),
    });
  }

  return data;
};

export function PortfolioTimeSeriesChart({ data }: PortfolioTimeSeriesProps) {
  const [timeRange, setTimeRange] = useState<"7D" | "30D" | "90D">("30D");
  const [viewMode, setViewMode] = useState<"value" | "pnl">("value");

  // Use provided data or generate mock data
  const chartData = data || generateMockHistory(timeRange === "7D" ? 7 : timeRange === "30D" ? 30 : 90);

  // Calculate statistics
  const latestValue = chartData[chartData.length - 1];
  const firstValue = chartData[0];
  const totalReturn = latestValue.total_value - firstValue.total_value;
  const totalReturnPct = ((totalReturn / firstValue.total_value) * 100).toFixed(2);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-[#161b22] border border-[#30363d] p-4 rounded-md shadow-lg">
          <p className="text-white font-semibold mb-2">{data.timestamp}</p>
          <div className="space-y-1">
            <p className="text-[#58a6ff] text-sm">
              Value: ${data.total_value.toLocaleString()}
            </p>
            <p className={`text-sm ${data.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
              P&L: {data.pnl >= 0 ? "+" : ""}${data.pnl.toLocaleString()} ({data.pnl_pct >= 0 ? "+" : ""}
              {data.pnl_pct}%)
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-white">Portfolio Performance</CardTitle>
            <p className="text-xs text-gray-400 mt-1">Historical value and P&L tracking</p>
          </div>
          <div className="flex items-center gap-2">
            {/* Time Range Selector */}
            <div className="flex gap-1 bg-[#161b22] rounded-lg p-1">
              {(["7D", "30D", "90D"] as const).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-3 py-1 text-xs rounded transition-colors ${
                    timeRange === range
                      ? "bg-[#238636] text-white"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  {range}
                </button>
              ))}
            </div>
            {/* View Mode Selector */}
            <div className="flex gap-1 bg-[#161b22] rounded-lg p-1">
              {(["value", "pnl"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1 text-xs rounded transition-colors capitalize ${
                    viewMode === mode
                      ? "bg-[#238636] text-white"
                      : "text-gray-400 hover:text-white"
                  }`}
                >
                  {mode === "value" ? "Value" : "P&L"}
                </button>
              ))}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Summary Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-[#161b22] rounded-lg p-3 border border-[#30363d]">
            <div className="text-xs text-gray-400 mb-1">Current Value</div>
            <div className="text-xl font-bold text-white">
              ${latestValue.total_value.toLocaleString()}
            </div>
          </div>
          <div className="bg-[#161b22] rounded-lg p-3 border border-[#30363d]">
            <div className="text-xs text-gray-400 mb-1">Total Return</div>
            <div className={`text-xl font-bold ${totalReturn >= 0 ? "text-green-400" : "text-red-400"}`}>
              {totalReturn >= 0 ? "+" : ""}${totalReturn.toLocaleString()}
            </div>
          </div>
          <div className="bg-[#161b22] rounded-lg p-3 border border-[#30363d]">
            <div className="text-xs text-gray-400 mb-1">Return %</div>
            <div className={`text-xl font-bold ${parseFloat(totalReturnPct) >= 0 ? "text-green-400" : "text-red-400"}`}>
              {parseFloat(totalReturnPct) >= 0 ? "+" : ""}{totalReturnPct}%
            </div>
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={350}>
          {viewMode === "value" ? (
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#58a6ff" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#58a6ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                dataKey="timestamp"
                stroke="#8b949e"
                style={{ fontSize: "12px" }}
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getMonth() + 1}/${date.getDate()}`;
                }}
              />
              <YAxis
                stroke="#8b949e"
                style={{ fontSize: "12px" }}
                tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="total_value"
                stroke="#58a6ff"
                strokeWidth={2}
                fill="url(#colorValue)"
              />
            </AreaChart>
          ) : (
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#30363d" />
              <XAxis
                dataKey="timestamp"
                stroke="#8b949e"
                style={{ fontSize: "12px" }}
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getMonth() + 1}/${date.getDate()}`;
                }}
              />
              <YAxis
                stroke="#8b949e"
                style={{ fontSize: "12px" }}
                tickFormatter={(value) => `${value >= 0 ? "+" : ""}${value}`}
              />
              <Tooltip content={<CustomTooltip />} />
              <Line
                type="monotone"
                dataKey="pnl"
                stroke="#3fb950"
                strokeWidth={2}
                dot={false}
              />
              {/* Zero line */}
              <Line
                type="monotone"
                dataKey={() => 0}
                stroke="#30363d"
                strokeWidth={1}
                strokeDasharray="5 5"
                dot={false}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
