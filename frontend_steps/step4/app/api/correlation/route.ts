import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const assetsParam = searchParams.get("assets") || "SPY,TLT,GLD,QQQ"
  const assets = assetsParam.split(",").map((a) => a.trim())

  // Mock correlation matrix - replace with actual correlation calculation
  const generateCorrelationMatrix = (assetList: string[]) => {
    const size = assetList.length
    const matrix: number[][] = []

    for (let i = 0; i < size; i++) {
      matrix[i] = []
      for (let j = 0; j < size; j++) {
        if (i === j) {
          matrix[i][j] = 1.0
        } else if (i < j) {
          // Generate random correlation between -1 and 1
          const correlation = Math.random() * 2 - 1
          matrix[i][j] = Math.round(correlation * 100) / 100
        } else {
          // Mirror the upper triangle
          matrix[i][j] = matrix[j][i]
        }
      }
    }

    return matrix
  }

  const matrix = generateCorrelationMatrix(assets)

  // Generate alerts based on correlation values
  const alerts = []
  for (let i = 0; i < assets.length; i++) {
    for (let j = i + 1; j < assets.length; j++) {
      const corr = matrix[i][j]

      if (Math.abs(corr) > 0.7) {
        alerts.push({
          type: corr > 0 ? "High Correlation" : "High Negative Correlation",
          message: `${assets[i]} and ${assets[j]} correlation: ${corr.toFixed(2)}`,
          severity: Math.abs(corr) > 0.85 ? "High" : "Medium",
        })
      }
    }
  }

  return NextResponse.json({
    matrix,
    assets,
    alerts,
  })
}
