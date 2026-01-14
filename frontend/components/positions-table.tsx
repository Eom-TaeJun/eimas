"use client"

import type React from "react"

import { useState } from "react"
import useSWR from "swr"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { ArrowUpDown, TrendingUp, TrendingDown } from "lucide-react"
import { Button } from "@/components/ui/button"

interface Position {
  ticker: string
  quantity: number
  avg_cost: number
  current_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_pct: number
}

interface PortfolioData {
  positions: Position[]
}

type SortField =
  | "ticker"
  | "quantity"
  | "avg_cost"
  | "current_price"
  | "market_value"
  | "unrealized_pnl"
  | "unrealized_pnl_pct"
type SortDirection = "asc" | "desc"

const fetcher = (url: string) => fetch(url).then((res) => res.json())

export function PositionsTable() {
  const [sortField, setSortField] = useState<SortField>("ticker")
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc")

  const { data, isLoading, error } = useSWR<PortfolioData>("/api/portfolio", fetcher, { refreshInterval: 30000 })

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-destructive">Failed to load positions</p>
        </CardContent>
      </Card>
    )
  }

  if (isLoading || !data) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Positions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        </CardContent>
      </Card>
    )
  }

  const sortedPositions = [...data.positions].sort((a, b) => {
    const aValue = a[sortField]
    const bValue = b[sortField]
    const modifier = sortDirection === "asc" ? 1 : -1

    if (typeof aValue === "string" && typeof bValue === "string") {
      return aValue.localeCompare(bValue) * modifier
    }
    return ((aValue as number) - (bValue as number)) * modifier
  })

  const SortButton = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <Button
      variant="ghost"
      size="sm"
      className="-ml-3 h-8 data-[state=open]:bg-accent"
      onClick={() => handleSort(field)}
    >
      {children}
      <ArrowUpDown className="ml-2 h-4 w-4" />
    </Button>
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Positions</CardTitle>
      </CardHeader>
      <CardContent>
        {sortedPositions.length === 0 ? (
          <p className="text-center text-muted-foreground py-8">No positions yet</p>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>
                    <SortButton field="ticker">Ticker</SortButton>
                  </TableHead>
                  <TableHead className="text-right">
                    <SortButton field="quantity">Quantity</SortButton>
                  </TableHead>
                  <TableHead className="text-right">
                    <SortButton field="avg_cost">Avg Cost</SortButton>
                  </TableHead>
                  <TableHead className="text-right">
                    <SortButton field="current_price">Current Price</SortButton>
                  </TableHead>
                  <TableHead className="text-right">
                    <SortButton field="market_value">Market Value</SortButton>
                  </TableHead>
                  <TableHead className="text-right">
                    <SortButton field="unrealized_pnl">Unrealized P&L</SortButton>
                  </TableHead>
                  <TableHead className="text-right">
                    <SortButton field="unrealized_pnl_pct">P&L %</SortButton>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sortedPositions.map((position) => {
                  const isPositive = position.unrealized_pnl >= 0
                  const PnLIcon = isPositive ? TrendingUp : TrendingDown

                  return (
                    <TableRow key={position.ticker} className="cursor-pointer hover:bg-muted/50">
                      <TableCell className="font-medium">{position.ticker}</TableCell>
                      <TableCell className="text-right">{position.quantity}</TableCell>
                      <TableCell className="text-right">${position.avg_cost.toFixed(2)}</TableCell>
                      <TableCell className="text-right">${position.current_price.toFixed(2)}</TableCell>
                      <TableCell className="text-right">
                        $
                        {position.market_value.toLocaleString("en-US", {
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2,
                        })}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1">
                          <span
                            className={
                              isPositive ? "text-green-600 dark:text-green-500" : "text-red-600 dark:text-red-500"
                            }
                          >
                            {isPositive ? "+" : ""}${position.unrealized_pnl.toFixed(2)}
                          </span>
                          <PnLIcon
                            className={`h-4 w-4 ${isPositive ? "text-green-600 dark:text-green-500" : "text-red-600 dark:text-red-500"}`}
                          />
                        </div>
                      </TableCell>
                      <TableCell
                        className={`text-right ${isPositive ? "text-green-600 dark:text-green-500" : "text-red-600 dark:text-red-500"}`}
                      >
                        {isPositive ? "+" : ""}
                        {position.unrealized_pnl_pct.toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
