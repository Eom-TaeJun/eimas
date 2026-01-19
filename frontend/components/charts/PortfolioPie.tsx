"use client"

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip, Legend } from "recharts"

interface PortfolioPieProps {
  weights: Record<string, number>
}

const COLORS = [
  "#3b82f6", // blue-500
  "#22c55e", // green-500
  "#eab308", // yellow-500
  "#f97316", // orange-500
  "#ef4444", // red-500
  "#a855f7", // purple-500
  "#ec4899", // pink-500
  "#6366f1", // violet-500
]

export function PortfolioPie({ weights }: PortfolioPieProps) {
  // 데이터 변환: {SPY: 0.3} -> [{name: 'SPY', value: 30}]
  const data = Object.entries(weights)
    .map(([name, weight]) => ({
      name,
      value: parseFloat((weight * 100).toFixed(1)),
    }))
    .sort((a, b) => b.value - a.value) // 높은 비중 순 정렬

  // 데이터가 없으면 플레이스홀더 표시
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[300px] text-gray-500">
        No portfolio data
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={80}
          paddingAngle={5}
          dataKey="value"
        >
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip formatter={(value: number) => [`${value}%`, 'Weight']} />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  )
}
