"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, PieChart } from "lucide-react";

interface PortfolioAllocationEvolutionProps {
  data?: Array<{
    timestamp: string;
    weights: Record<string, number>;
  }>;
}

// Color palette for assets
const ASSET_COLORS = [
  "#3fb950", "#58a6ff", "#d29922", "#f85149", "#a371f7",
  "#da3633", "#e3b341", "#f778ba", "#56d4dd", "#7ee787",
];

// Generate mock allocation history
const generateMockAllocationHistory = (days: number = 30) => {
  const tickers = ["SPY", "QQQ", "TLT", "GLD", "HYG", "XLF", "XLE"];
  const data = [];
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;

  for (let i = days; i >= 0; i--) {
    const weights: Record<string, number> = {};
    let total = 0;

    // Generate random weights
    tickers.forEach((ticker) => {
      const weight = 0.05 + Math.random() * 0.2;
      weights[ticker] = weight;
      total += weight;
    });

    // Normalize to sum to 1
    Object.keys(weights).forEach((ticker) => {
      weights[ticker] = (weights[ticker] / total) * 100; // Convert to percentage
    });

    data.push({
      timestamp: new Date(now - i * dayMs).toISOString().split("T")[0],
      ...weights,
      weights,
    });
  }

  return data;
};

export function PortfolioAllocationEvolution({ data }: PortfolioAllocationEvolutionProps) {
  const [selectedAssets, setSelectedAssets] = useState<Set<string>>(new Set());

  const chartData = data || generateMockAllocationHistory(30);

  // Extract all unique tickers from the data
  const allTickers = Array.from(
    new Set(
      chartData.flatMap((d) =>
        Object.keys(d).filter((k) => k !== "timestamp" && k !== "weights")
      )
    )
  );

  // Get top assets by average weight
  const avgWeights = allTickers.map((ticker) => {
    const avg =
      chartData.reduce((sum, d) => sum + (d[ticker as keyof typeof d] as number || 0), 0) /
      chartData.length;
    return { ticker, avg };
  });
  avgWeights.sort((a, b) => b.avg - a.avg);
  const topTickers = avgWeights.slice(0, 7).map((t) => t.ticker);

  const displayTickers = selectedAssets.size > 0 ? Array.from(selectedAssets) : topTickers;

  // Calculate statistics for selected assets
  const latestAllocation = chartData[chartData.length - 1];
  const firstAllocation = chartData[0];

  const toggleAsset = (ticker: string) => {
    const newSelection = new Set(selectedAssets);
    if (newSelection.has(ticker)) {
      newSelection.delete(ticker);
    } else {
      if (newSelection.size < 7) {
        // Limit to 7 for readability
        newSelection.add(ticker);
      }
    }
    setSelectedAssets(newSelection);
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const date = payload[0].payload.timestamp;
      return (
        <div className="bg-[#161b22] border border-[#30363d] p-4 rounded-md shadow-lg max-w-xs">
          <p className="text-white font-semibold mb-2">{date}</p>
          <div className="space-y-1 text-xs">
            {payload
              .sort((a: any, b: any) => b.value - a.value)
              .map((entry: any, index: number) => (
                <div key={index} className="flex justify-between gap-4">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: entry.color }}
                    />
                    <span className="text-gray-300">{entry.name}:</span>
                  </div>
                  <span className="text-white font-mono">{entry.value.toFixed(1)}%</span>
                </div>
              ))}
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
            <CardTitle className="text-white flex items-center gap-2">
              <PieChart className="w-5 h-5 text-purple-400" />
              Portfolio Allocation Evolution
              <Badge variant="outline" className="text-xs bg-purple-500/10 text-purple-400 border-purple-500/20">
                Time Series
              </Badge>
            </CardTitle>
            <p className="text-xs text-gray-400 mt-1">
              Historical asset weight changes • Click tickers to toggle display
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Asset Selector */}
        <div className="mb-6">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-gray-400">Select Assets:</span>
            <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
              {selectedAssets.size > 0 ? `${selectedAssets.size} selected` : "Top 7 shown"}
            </Badge>
          </div>
          <div className="flex flex-wrap gap-2">
            {allTickers.map((ticker, index) => {
              const isSelected = selectedAssets.size === 0 ? topTickers.includes(ticker) : selectedAssets.has(ticker);
              return (
                <button
                  key={ticker}
                  onClick={() => toggleAsset(ticker)}
                  className={`px-3 py-1 text-xs rounded-lg border transition-all ${
                    isSelected
                      ? "bg-[#238636] text-white border-[#238636]"
                      : "bg-[#161b22] text-gray-400 border-[#30363d] hover:border-gray-500"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: isSelected ? ASSET_COLORS[index % ASSET_COLORS.length] : "#666",
                      }}
                    />
                    {ticker}
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {displayTickers.slice(0, 4).map((ticker, index) => {
            const latest = latestAllocation[ticker as keyof typeof latestAllocation] as number || 0;
            const first = firstAllocation[ticker as keyof typeof firstAllocation] as number || 0;
            const change = latest - first;

            return (
              <div key={ticker} className="bg-[#161b22] rounded-lg p-3 border border-[#30363d]">
                <div className="flex items-center gap-2 mb-1">
                  <div
                    className="w-2 h-2 rounded-full"
                    style={{ backgroundColor: ASSET_COLORS[allTickers.indexOf(ticker) % ASSET_COLORS.length] }}
                  />
                  <div className="text-xs text-gray-400">{ticker}</div>
                </div>
                <div className="text-lg font-bold text-white">{latest.toFixed(1)}%</div>
                <div className={`text-xs flex items-center gap-1 ${change >= 0 ? "text-green-400" : "text-red-400"}`}>
                  <TrendingUp className={`w-3 h-3 ${change < 0 ? "rotate-180" : ""}`} />
                  {change >= 0 ? "+" : ""}{change.toFixed(1)}%
                </div>
              </div>
            );
          })}
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={chartData}>
            <defs>
              {displayTickers.map((ticker, index) => (
                <linearGradient key={ticker} id={`gradient-${ticker}`} x1="0" y1="0" x2="0" y2="1">
                  <stop
                    offset="5%"
                    stopColor={ASSET_COLORS[allTickers.indexOf(ticker) % ASSET_COLORS.length]}
                    stopOpacity={0.8}
                  />
                  <stop
                    offset="95%"
                    stopColor={ASSET_COLORS[allTickers.indexOf(ticker) % ASSET_COLORS.length]}
                    stopOpacity={0.1}
                  />
                </linearGradient>
              ))}
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
              tickFormatter={(value) => `${value.toFixed(0)}%`}
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ color: "#ffffff", fontSize: "12px" }}
              iconType="square"
            />
            {displayTickers.map((ticker, index) => (
              <Area
                key={ticker}
                type="monotone"
                dataKey={ticker}
                stackId="1"
                stroke={ASSET_COLORS[allTickers.indexOf(ticker) % ASSET_COLORS.length]}
                fill={`url(#gradient-${ticker})`}
                strokeWidth={1.5}
                name={ticker}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>

        {/* Legend Note */}
        <div className="mt-4 text-xs text-gray-400 text-center">
          Stacked area chart showing percentage allocation over time. Total always sums to 100%.
        </div>
      </CardContent>
    </Card>
  );
}
