"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle, Shield, TrendingUp } from "lucide-react";

interface RiskTimelineProps {
  data?: Array<{
    timestamp: string;
    risk_score: number;
    risk_level: "LOW" | "MEDIUM" | "HIGH";
    base_risk: number;
    microstructure_adj: number;
    bubble_adj: number;
  }>;
}

// Generate mock historical risk data
const generateMockRiskHistory = (days: number = 30) => {
  const data = [];
  const now = Date.now();
  const dayMs = 24 * 60 * 60 * 1000;

  for (let i = days; i >= 0; i--) {
    const baseRisk = 40 + Math.random() * 20; // 40-60
    const microAdj = -5 + Math.random() * 10; // -5 to +5
    const bubbleAdj = -3 + Math.random() * 8; // -3 to +5
    const riskScore = baseRisk + microAdj + bubbleAdj;

    let riskLevel: "LOW" | "MEDIUM" | "HIGH";
    if (riskScore < 45) riskLevel = "LOW";
    else if (riskScore < 65) riskLevel = "MEDIUM";
    else riskLevel = "HIGH";

    data.push({
      timestamp: new Date(now - i * dayMs).toISOString().split("T")[0],
      risk_score: parseFloat(riskScore.toFixed(2)),
      risk_level: riskLevel,
      base_risk: parseFloat(baseRisk.toFixed(2)),
      microstructure_adj: parseFloat(microAdj.toFixed(2)),
      bubble_adj: parseFloat(bubbleAdj.toFixed(2)),
    });
  }

  return data;
};

export function RiskTimelineChart({ data }: RiskTimelineProps) {
  const chartData = data || generateMockRiskHistory(30);

  // Calculate statistics
  const latestRisk = chartData[chartData.length - 1];
  const avgRisk = chartData.reduce((sum, d) => sum + d.risk_score, 0) / chartData.length;
  const maxRisk = Math.max(...chartData.map((d) => d.risk_score));
  const minRisk = Math.min(...chartData.map((d) => d.risk_score));

  const getRiskColor = (level: string) => {
    switch (level) {
      case "LOW":
        return "text-green-400 bg-green-500/10 border-green-500/20";
      case "MEDIUM":
        return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
      case "HIGH":
        return "text-red-400 bg-red-500/10 border-red-500/20";
      default:
        return "text-gray-400";
    }
  };

  const getRiskIcon = (level: string) => {
    switch (level) {
      case "LOW":
        return <Shield className="w-4 h-4" />;
      case "MEDIUM":
        return <TrendingUp className="w-4 h-4" />;
      case "HIGH":
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return null;
    }
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-[#161b22] border border-[#30363d] p-4 rounded-md shadow-lg">
          <p className="text-white font-semibold mb-2">{data.timestamp}</p>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className={`text-xs ${getRiskColor(data.risk_level)}`}
              >
                {data.risk_level}
              </Badge>
              <span className="text-white font-bold">{data.risk_score.toFixed(1)}</span>
            </div>
            <div className="pt-2 border-t border-gray-700 space-y-1 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-400">Base Risk:</span>
                <span className="text-white">{data.base_risk.toFixed(1)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Microstructure:</span>
                <span className={data.microstructure_adj >= 0 ? "text-red-400" : "text-green-400"}>
                  {data.microstructure_adj >= 0 ? "+" : ""}
                  {data.microstructure_adj.toFixed(1)}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Bubble Risk:</span>
                <span className={data.bubble_adj >= 0 ? "text-red-400" : "text-green-400"}>
                  {data.bubble_adj >= 0 ? "+" : ""}
                  {data.bubble_adj.toFixed(1)}
                </span>
              </div>
            </div>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <Shield className="w-5 h-5 text-blue-400" />
          Risk Score Timeline
        </CardTitle>
        <p className="text-xs text-gray-400 mt-1">
          Historical risk evolution with component breakdown
        </p>
      </CardHeader>
      <CardContent>
        {/* Summary Stats */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-[#161b22] rounded-lg p-3 border border-[#30363d]">
            <div className="text-xs text-gray-400 mb-1">Current</div>
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className={`text-xs ${getRiskColor(latestRisk.risk_level)}`}
              >
                {getRiskIcon(latestRisk.risk_level)}
                <span className="ml-1">{latestRisk.risk_level}</span>
              </Badge>
            </div>
            <div className="text-lg font-bold text-white mt-1">
              {latestRisk.risk_score.toFixed(1)}
            </div>
          </div>
          <div className="bg-[#161b22] rounded-lg p-3 border border-[#30363d]">
            <div className="text-xs text-gray-400 mb-1">Average</div>
            <div className="text-lg font-bold text-white">
              {avgRisk.toFixed(1)}
            </div>
          </div>
          <div className="bg-[#161b22] rounded-lg p-3 border border-[#30363d]">
            <div className="text-xs text-gray-400 mb-1">Peak</div>
            <div className="text-lg font-bold text-red-400">
              {maxRisk.toFixed(1)}
            </div>
          </div>
          <div className="bg-[#161b22] rounded-lg p-3 border border-[#30363d]">
            <div className="text-xs text-gray-400 mb-1">Low</div>
            <div className="text-lg font-bold text-green-400">
              {minRisk.toFixed(1)}
            </div>
          </div>
        </div>

        {/* Chart */}
        <ResponsiveContainer width="100%" height={350}>
          <ComposedChart data={chartData}>
            <defs>
              <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f85149" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#f85149" stopOpacity={0} />
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
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend
              wrapperStyle={{ color: "#ffffff", fontSize: "12px" }}
              iconType="line"
            />

            {/* Risk level threshold lines */}
            <ReferenceLine
              y={45}
              stroke="#d29922"
              strokeDasharray="3 3"
              label={{
                value: "Medium Risk",
                position: "right",
                fill: "#d29922",
                fontSize: 10,
              }}
            />
            <ReferenceLine
              y={65}
              stroke="#f85149"
              strokeDasharray="3 3"
              label={{
                value: "High Risk",
                position: "right",
                fill: "#f85149",
                fontSize: 10,
              }}
            />

            {/* Risk components as stacked areas */}
            <Area
              type="monotone"
              dataKey="base_risk"
              stackId="1"
              stroke="#58a6ff"
              fill="#58a6ff"
              fillOpacity={0.6}
              name="Base Risk"
            />

            {/* Main risk score line */}
            <Line
              type="monotone"
              dataKey="risk_score"
              stroke="#f85149"
              strokeWidth={3}
              dot={false}
              name="Total Risk Score"
            />
          </ComposedChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
