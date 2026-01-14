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
  LabelList,
} from "recharts";

interface GMMProbabilityChartProps {
  probabilities: {
    Bull: number;
    Neutral: number;
    Bear: number;
  };
}

const REGIME_COLORS = {
  Bull: "#3fb950",
  Neutral: "#d29922",
  Bear: "#f85149",
};

export function GMMProbabilityChart({
  probabilities,
}: GMMProbabilityChartProps) {
  const chartData = [
    {
      regime: "Bull",
      probability: probabilities.Bull * 100,
      color: REGIME_COLORS.Bull,
    },
    {
      regime: "Neutral",
      probability: probabilities.Neutral * 100,
      color: REGIME_COLORS.Neutral,
    },
    {
      regime: "Bear",
      probability: probabilities.Bear * 100,
      color: REGIME_COLORS.Bear,
    },
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#161b22] border border-[#30363d] p-3 rounded-md shadow-lg">
          <p className="text-white font-semibold">{payload[0].payload.regime}</p>
          <p className="text-[#58a6ff]">{payload[0].value.toFixed(1)}%</p>
        </div>
      );
    }
    return null;
  };

  const renderCustomLabel = (props: any) => {
    const { x, y, width, value } = props;
    return (
      <text
        x={x + width / 2}
        y={y + 20}
        fill="white"
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize="14"
        fontWeight="bold"
      >
        {value.toFixed(1)}%
      </text>
    );
  };

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <CardTitle className="text-white">GMM Regime Probabilities</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <XAxis type="number" domain={[0, 100]} hide />
            <YAxis
              type="category"
              dataKey="regime"
              tick={{ fill: "#ffffff" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="probability" radius={[0, 8, 8, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
              <LabelList dataKey="probability" content={renderCustomLabel} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
