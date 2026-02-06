"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

interface CorrelationData {
  matrix: number[][]
  assets: string[]
}

// Blue (negative) → Gray (zero) → Red (positive) — standard financial convention
const getColor = (value: number) => {
  if (value > 0.7) return "bg-red-700 text-white"
  if (value > 0.4) return "bg-red-500 text-white"
  if (value > 0.1) return "bg-red-300 text-gray-800"
  if (value > -0.1) return "bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200"
  if (value > -0.4) return "bg-blue-300 text-gray-800"
  if (value > -0.7) return "bg-blue-500 text-white"
  return "bg-blue-700 text-white"
}

export function CorrelationMatrix() {
  const [data, setData] = useState<CorrelationData | null>(null)
  const [assets, setAssets] = useState("SPY,TLT,GLD,QQQ")
  const [loading, setLoading] = useState(false)
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number } | null>(null)

  const fetchCorrelation = async (assetList: string) => {
    setLoading(true)
    try {
      const res = await fetch(`/api/correlation?assets=${assetList}`)
      const json = await res.json()
      setData(json)
    } catch (error) {
      console.error("[v0] Failed to fetch correlation data:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCorrelation(assets)
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle>Correlation Matrix</CardTitle>
        <CardDescription>Asset correlation heatmap (Blue = negative, Red = positive)</CardDescription>
        <div className="flex gap-2 mt-4">
          <Input
            placeholder="Enter assets (comma-separated)"
            value={assets}
            onChange={(e) => setAssets(e.target.value)}
            className="flex-1"
          />
          <Button onClick={() => fetchCorrelation(assets)} disabled={loading}>
            {loading ? "Loading..." : "Update"}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {data && (
          <div className="overflow-x-auto">
            <div className="inline-block min-w-full">
              <div className="grid gap-1" style={{ gridTemplateColumns: `80px repeat(${data.assets.length}, 80px)` }}>
                {/* Header row */}
                <div className="font-semibold"></div>
                {data.assets.map((asset) => (
                  <div key={asset} className="font-semibold text-center p-2 text-sm">
                    {asset}
                  </div>
                ))}

                {/* Data rows */}
                {data.matrix.map((row, i) => (
                  <div key={i} className="contents">
                    <div className="font-semibold p-2 flex items-center text-sm">{data.assets[i]}</div>
                    {row.map((value, j) => (
                      <div
                        key={j}
                        className={`p-2 text-center font-mono text-sm rounded cursor-pointer transition-all duration-200 hover:scale-110 hover:z-10 hover:shadow-lg ${getColor(value)}`}
                        onMouseEnter={() => setHoveredCell({ i, j })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        {value.toFixed(2)}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>

            {/* Hover tooltip */}
            {hoveredCell && (
              <div className="mt-3 p-2 bg-gray-100 dark:bg-gray-800 rounded border text-sm">
                <span className="font-bold">{data.assets[hoveredCell.i]}</span>
                {" ↔ "}
                <span className="font-bold">{data.assets[hoveredCell.j]}</span>
                {": "}
                <span className={data.matrix[hoveredCell.i][hoveredCell.j] > 0 ? "text-red-600 font-bold" : "text-blue-600 font-bold"}>
                  {data.matrix[hoveredCell.i][hoveredCell.j].toFixed(3)}
                </span>
                <span className="text-gray-500 ml-2">
                  ({Math.abs(data.matrix[hoveredCell.i][hoveredCell.j]) > 0.7
                    ? "Strong"
                    : Math.abs(data.matrix[hoveredCell.i][hoveredCell.j]) > 0.4
                    ? "Moderate"
                    : "Weak"} correlation)
                </span>
              </div>
            )}

            {/* Color legend */}
            <div className="mt-4 flex items-center justify-center gap-2 text-xs text-gray-500">
              <span>Negative</span>
              <div className="flex gap-0.5">
                <div className="w-8 h-3 bg-blue-700 rounded-l"></div>
                <div className="w-8 h-3 bg-blue-500"></div>
                <div className="w-8 h-3 bg-blue-300"></div>
                <div className="w-8 h-3 bg-gray-200"></div>
                <div className="w-8 h-3 bg-red-300"></div>
                <div className="w-8 h-3 bg-red-500"></div>
                <div className="w-8 h-3 bg-red-700 rounded-r"></div>
              </div>
              <span>Positive</span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
