"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from "recharts";

interface PortfolioChartProps {
  weights: Record<string, number>;
}

const COLORS = [
  "#3fb950", // green
  "#58a6ff", // blue
  "#d29922", // yellow
  "#f85149", // red
  "#a371f7", // purple
  "#da3633", // dark red
  "#e3b341", // gold
  "#f778ba", // pink
];

export function PortfolioChart({ weights }: PortfolioChartProps) {
  // Sort by weight descending and take top 8
  const sortedEntries = Object.entries(weights)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8);

  // Calculate "Others" if there are more than 8 holdings
  const totalShown = sortedEntries.reduce((sum, [, weight]) => sum + weight, 0);
  const othersWeight = 1 - totalShown;

  const chartData = [
    ...sortedEntries.map(([name, value]) => ({
      name,
      value: value * 100, // Convert to percentage
    })),
    ...(othersWeight > 0.001
      ? [{ name: "Others", value: othersWeight * 100 }]
      : []),
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-[#161b22] border border-[#30363d] p-3 rounded-md shadow-lg">
          <p className="text-white font-semibold">{payload[0].name}</p>
          <p className="text-[#58a6ff]">{payload[0].value.toFixed(1)}%</p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <CardTitle className="text-white">Portfolio Allocation</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value.toFixed(1)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              verticalAlign="bottom"
              height={36}
              wrapperStyle={{ color: "#ffffff" }}
            />
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
