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
  ReferenceLine,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, Activity } from "lucide-react";
import { useState } from "react";

interface RegimeTransitionProps {
  data?: Array<{
    timestamp: string;
    regime: "Bull" | "Neutral" | "Bear";
    confidence: number;
    bull_prob: number;
    neutral_prob: number;
    bear_prob: number;
    volatility: number;
  }>;
}

// Generate mock regime transition data
const generateMockRegimeData = (days: number = 30) => {
  const data = [];
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;

  let currentRegime: "Bull" | "Neutral" | "Bear" = "Bull";

  for (let i = days; i >= 0; i--) {
    // Simulate regime transitions with some persistence
    if (Math.random() < 0.1) {
      const regimes: ("Bull" | "Neutral" | "Bear")[] = ["Bull", "Neutral", "Bear"];
      currentRegime = regimes[Math.floor(Math.random() * 3)];
    }

    // Generate probabilities that favor current regime
    let bullProb, neutralProb, bearProb;

    if (currentRegime === "Bull") {
      bullProb = 0.5 + Math.random() * 0.3;
      bearProb = 0.1 + Math.random() * 0.2;
      neutralProb = 1 - bullProb - bearProb;
    } else if (currentRegime === "Bear") {
      bearProb = 0.5 + Math.random() * 0.3;
      bullProb = 0.1 + Math.random() * 0.2;
      neutralProb = 1 - bullProb - bearProb;
    } else {
      neutralProb = 0.4 + Math.random() * 0.3;
      bullProb = 0.2 + Math.random() * 0.3;
      bearProb = 1 - bullProb - neutralProb;
    }

    const maxProb = Math.max(bullProb, neutralProb, bearProb);
    const volatility = 15 + Math.random() * 20;

    data.push({
      timestamp: new Date(now - i * dayMs).toISOString().split("T")[0],
      regime: currentRegime,
      confidence: parseFloat(maxProb.toFixed(3)),
      bull_prob: parseFloat(bullProb.toFixed(3)),
      neutral_prob: parseFloat(neutralProb.toFixed(3)),
      bear_prob: parseFloat(bearProb.toFixed(3)),
      volatility: parseFloat(volatility.toFixed(1)),
    });
  }

  return data;
};

export function RegimeTransitionChart({ data }: RegimeTransitionProps) {
  const [viewMode, setViewMode] = useState<"probabilities" | "volatility">("probabilities");

  const chartData = data || generateMockRegimeData(30);

  // Calculate regime statistics
  const regimeCounts = chartData.reduce((acc, d) => {
    acc[d.regime] = (acc[d.regime] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const currentRegime = chartData[chartData.length - 1];
  const avgConfidence = chartData.reduce((sum, d) => sum + d.confidence, 0) / chartData.length;

  // Count regime transitions
  let transitions = 0;
  for (let i = 1; i < chartData.length; i++) {
    if (chartData[i].regime !== chartData[i - 1].regime) {
      transitions++;
    }
  }

  const getRegimeColor = (regime: string) => {
    switch (regime) {
      case "Bull":
        return "text-green-400 bg-green-500/10 border-green-500/20";
      case "Bear":
        return "text-red-400 bg-red-500/10 border-red-500/20";
      case "Neutral":
        return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
      default:
        return "text-gray-400";
    }
  };

  const getRegimeIcon = (regime: string) => {
    switch (regime) {
      case "Bull":
        return <TrendingUp className="w-4 h-4" />;
      case "Bear":
        return <TrendingDown className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-surface-card border border-border p-4 rounded-md shadow-lg">
          <p className="text-white font-semibold mb-2">{data.timestamp}</p>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className={`text-xs ${getRegimeColor(data.regime)}`}>
                {getRegimeIcon(data.regime)}
                <span className="ml-1">{data.regime}</span>
              </Badge>
              <span className="text-white text-sm">
                {(data.confidence * 100).toFixed(0)}% confidence
              </span>
            </div>
            <div className="pt-2 border-t border-gray-700 space-y-1 text-xs">
              <div className="flex justify-between gap-4">
                <span className="text-green-400">Bull:</span>
                <span className="text-white font-mono">{(data.bull_prob * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-yellow-400">Neutral:</span>
                <span className="text-white font-mono">{(data.neutral_prob * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between gap-4">
                <span className="text-red-400">Bear:</span>
                <span className="text-white font-mono">{(data.bear_prob * 100).toFixed(1)}%</span>
              </div>
              {viewMode === "volatility" && (
                <div className="flex justify-between gap-4 pt-1 border-t border-gray-700">
                  <span className="text-gray-400">Volatility:</span>
                  <span className="text-white font-mono">{data.volatility.toFixed(1)}%</span>
                </div>
              )}
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-surface border-border">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-white flex items-center gap-2">
              Market Regime Evolution
              <Badge variant="outline" className="text-xs bg-purple-500/10 text-purple-400 border-purple-500/20">
                GMM Analysis
              </Badge>
            </CardTitle>
            <p className="text-xs text-gray-400 mt-1">
              Historical regime transitions with probability distributions
            </p>
          </div>
          <div className="flex gap-1 bg-surface-card rounded-lg p-1">
            {(["probabilities", "volatility"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1 text-xs rounded transition-colors capitalize ${
                  viewMode === mode
                    ? "bg-secondary text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400 mb-1">Current Regime</div>
            <Badge variant="outline" className={`${getRegimeColor(currentRegime.regime)} mb-1`}>
              {getRegimeIcon(currentRegime.regime)}
              <span className="ml-1">{currentRegime.regime}</span>
            </Badge>
            <div className="text-sm text-gray-300">
              {(currentRegime.confidence * 100).toFixed(0)}% conf
            </div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400 mb-1">Avg Confidence</div>
            <div className="text-lg font-bold text-white">
              {(avgConfidence * 100).toFixed(0)}%
            </div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400 mb-1">Transitions</div>
            <div className="text-lg font-bold text-white">{transitions}</div>
            <div className="text-xs text-gray-400">in {chartData.length} days</div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400 mb-1">Distribution</div>
            <div className="flex gap-2 text-xs mt-1">
              <span className="text-green-400">{regimeCounts.Bull || 0}🟢</span>
              <span className="text-yellow-400">{regimeCounts.Neutral || 0}🟡</span>
              <span className="text-red-400">{regimeCounts.Bear || 0}🔴</span>
            </div>
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={350}>
          {viewMode === "probabilities" ? (
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="bullGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.6} />
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0.1} />
                </linearGradient>
                <linearGradient id="neutralGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#d97706" stopOpacity={0.6} />
                  <stop offset="95%" stopColor="#d97706" stopOpacity={0.1} />
                </linearGradient>
                <linearGradient id="bearGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.6} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.1} />
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
                tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                domain={[0, 1]}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ color: "#ffffff", fontSize: "12px" }}
              />
              <ReferenceLine
                y={0.5}
                stroke="#30363d"
                strokeDasharray="3 3"
                label={{ value: "50%", fill: "#8b949e", fontSize: 10 }}
              />
              <Area
                type="monotone"
                dataKey="bull_prob"
                stackId="1"
                stroke="#10B981"
                fill="url(#bullGradient)"
                name="Bull Probability"
              />
              <Area
                type="monotone"
                dataKey="neutral_prob"
                stackId="1"
                stroke="#d97706"
                fill="url(#neutralGradient)"
                name="Neutral Probability"
              />
              <Area
                type="monotone"
                dataKey="bear_prob"
                stackId="1"
                stroke="#ef4444"
                fill="url(#bearGradient)"
                name="Bear Probability"
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
                label={{ value: "Volatility (%)", angle: -90, position: "insideLeft", fill: "#8b949e" }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend wrapperStyle={{ color: "#ffffff", fontSize: "12px" }} />
              <Line
                type="monotone"
                dataKey="volatility"
                stroke="#a371f7"
                strokeWidth={2}
                dot={false}
                name="Volatility"
              />
              <ReferenceLine
                y={20}
                stroke="#d97706"
                strokeDasharray="3 3"
                label={{ value: "Normal", fill: "#d97706", fontSize: 10 }}
              />
              <ReferenceLine
                y={30}
                stroke="#ef4444"
                strokeDasharray="3 3"
                label={{ value: "High", fill: "#ef4444", fontSize: 10 }}
              />
            </LineChart>
          )}
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
