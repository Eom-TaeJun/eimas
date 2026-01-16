"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"

interface CorrelationData {
  matrix: number[][]
  assets: string[]
}

export function CorrelationMatrix() {
  const [data, setData] = useState<CorrelationData | null>(null)
  const [assets, setAssets] = useState("SPY,TLT,GLD,QQQ")
  const [loading, setLoading] = useState(false)

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

  const getColor = (value: number) => {
    // Red for negative, green for positive
    if (value < -0.5) return "bg-red-500 text-white"
    if (value < -0.2) return "bg-red-300 text-foreground"
    if (value < 0.2) return "bg-gray-200 text-foreground dark:bg-gray-700"
    if (value < 0.5) return "bg-green-300 text-foreground"
    if (value < 1) return "bg-green-400 text-foreground"
    return "bg-green-500 text-white"
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Correlation Matrix</CardTitle>
        <CardDescription>Asset correlation heatmap</CardDescription>
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
                  <div key={asset} className="font-semibold text-center p-2">
                    {asset}
                  </div>
                ))}

                {/* Data rows */}
                {data.matrix.map((row, i) => (
                  <div key={i} className="contents">
                    <div className="font-semibold p-2 flex items-center">{data.assets[i]}</div>
                    {row.map((value, j) => (
                      <div key={j} className={`p-2 text-center font-mono text-sm rounded ${getColor(value)}`}>
                        {value.toFixed(2)}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
