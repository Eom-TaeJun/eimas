"use client"

import { useState } from "react"
import useSWR from "swr"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"

interface Trade {
  timestamp: string
  ticker: string
  side: "BUY" | "SELL"
  quantity: number
  price: number
  realized_pnl: number
}

const fetcher = (url: string) => fetch(url).then((res) => res.json())

export function TradeHistory() {
  const [days, setDays] = useState("30")

  const { data, isLoading, error } = useSWR<Trade[]>(`/api/portfolio/trades?days=${days}`, fetcher)

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-destructive">Failed to load trade history</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle>Trade History</CardTitle>
        <Select value={days} onValueChange={setDays}>
          <SelectTrigger className="w-[140px]">
            <SelectValue placeholder="Select period" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="7">Last 7 days</SelectItem>
            <SelectItem value="30">Last 30 days</SelectItem>
            <SelectItem value="90">Last 90 days</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        {isLoading || !data ? (
          <div className="space-y-2">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-12 w-full" />
            ))}
          </div>
        ) : data.length === 0 ? (
          <p className="text-center text-muted-foreground py-8">No trades in this period</p>
        ) : (
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Ticker</TableHead>
                  <TableHead>Side</TableHead>
                  <TableHead className="text-right">Quantity</TableHead>
                  <TableHead className="text-right">Price</TableHead>
                  <TableHead className="text-right">Realized P&L</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.map((trade, index) => {
                  const isBuy = trade.side === "BUY"
                  const hasPnL = trade.realized_pnl !== 0

                  return (
                    <TableRow key={index}>
                      <TableCell>
                        {new Date(trade.timestamp).toLocaleString("en-US", {
                          month: "short",
                          day: "numeric",
                          year: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </TableCell>
                      <TableCell className="font-medium">{trade.ticker}</TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={
                            isBuy
                              ? "border-green-600 text-green-600 dark:border-green-500 dark:text-green-500"
                              : "border-red-600 text-red-600 dark:border-red-500 dark:text-red-500"
                          }
                        >
                          {trade.side}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">{trade.quantity}</TableCell>
                      <TableCell className="text-right">${trade.price.toFixed(2)}</TableCell>
                      <TableCell
                        className={`text-right ${
                          !hasPnL
                            ? ""
                            : trade.realized_pnl > 0
                              ? "text-green-600 dark:text-green-500"
                              : "text-red-600 dark:text-red-500"
                        }`}
                      >
                        {hasPnL && (trade.realized_pnl > 0 ? "+" : "")}${trade.realized_pnl.toFixed(2)}
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
