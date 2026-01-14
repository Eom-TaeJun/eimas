"use client"

import useSWR from "swr"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { TrendingUp, TrendingDown } from "lucide-react"

interface PortfolioData {
  cash: number
  positions_value: number
  total_value: number
  total_pnl: number
  total_pnl_pct: number
}

const fetcher = (url: string) => fetch(url).then((res) => res.json())

export function PortfolioSummary() {
  const { data, isLoading, error } = useSWR<PortfolioData>("/api/portfolio", fetcher, { refreshInterval: 30000 })

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-destructive">Failed to load portfolio data</p>
        </CardContent>
      </Card>
    )
  }

  if (isLoading || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="space-y-2">
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-8 w-32" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  const isPositive = data.total_pnl >= 0
  const PnLIcon = isPositive ? TrendingUp : TrendingDown

  return (
    <Card>
      <CardHeader>
        <CardTitle>Portfolio Summary</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 md:grid-cols-4">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Cash Balance</p>
            <p className="text-2xl font-bold">
              ${data.cash.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Positions Value</p>
            <p className="text-2xl font-bold">
              ${data.positions_value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Total Value</p>
            <p className="text-2xl font-bold">
              ${data.total_value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </p>
          </div>

          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">Total P&L</p>
            <div className="flex items-center gap-2">
              <p
                className={`text-2xl font-bold ${isPositive ? "text-green-600 dark:text-green-500" : "text-red-600 dark:text-red-500"}`}
              >
                {isPositive ? "+" : ""}$
                {data.total_pnl.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </p>
              <PnLIcon
                className={`h-5 w-5 ${isPositive ? "text-green-600 dark:text-green-500" : "text-red-600 dark:text-red-500"}`}
              />
            </div>
            <p
              className={`text-sm ${isPositive ? "text-green-600 dark:text-green-500" : "text-red-600 dark:text-red-500"}`}
            >
              {isPositive ? "+" : ""}
              {data.total_pnl_pct.toFixed(2)}%
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
