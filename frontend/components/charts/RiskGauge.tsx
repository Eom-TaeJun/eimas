"use client"

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts"

interface RiskGaugeProps {
  score: number // 0-100
}

export function RiskGauge({ score }: RiskGaugeProps) {
  // 게이지 데이터: [현재 점수, 나머지]
  const data = [
    { name: "Risk Score", value: score },
    { name: "Remaining", value: 100 - score },
  ]

  // 색상 결정: 낮음(초록) -> 높음(빨강)
  const getColor = (s: number) => {
    if (s < 30) return "#22c55e" // green-500
    if (s < 70) return "#eab308" // yellow-500
    return "#ef4444" // red-500
  }

  const color = getColor(score)

  // 바늘 각도 계산 (180도 기준)
  // 0점 -> 180도 (왼쪽), 100점 -> 0도 (오른쪽)
  const cx = "50%"
  const cy = "70%"
  const iR = 60
  const oR = 80

  return (
    <div className="flex flex-col items-center justify-center h-full min-h-[200px]">
      <ResponsiveContainer width="100%" height={160}>
        <PieChart>
          <Pie
            dataKey="value"
            startAngle={180}
            endAngle={0}
            data={data}
            cx={cx}
            cy={cy}
            innerRadius={iR}
            outerRadius={oR}
            stroke="none"
          >
            <Cell fill={color} />
            <Cell fill="#334155" /> {/* gray-700 for background */}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
      <div className="mt-[-40px] text-center">
        <div className="text-4xl font-bold" style={{ color }}>
          {score.toFixed(1)}
        </div>
        <div className="text-sm text-gray-400">Risk Score</div>
      </div>
    </div>
  )
}
