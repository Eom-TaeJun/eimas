"use client"

import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import useSWR from "swr"
import { fetchSignals } from "@/lib/api"
import type { Signal } from "@/lib/types"
import { format } from "date-fns"

export function SignalsTable() {
  const { data: signals, error } = useSWR<Signal[]>("signals", () => fetchSignals(10), { refreshInterval: 60000 })

  const getActionColor = (action: string) => {
    switch (action) {
      case "BUY":
        return "bg-green-500/10 text-green-400 border-green-500/20"
      case "SELL":
        return "bg-red-500/10 text-red-400 border-red-500/20"
      case "HOLD":
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  const formatTimestamp = (timestamp: string) => {
    try {
      return format(new Date(timestamp), "MMM dd, HH:mm:ss")
    } catch {
      return timestamp
    }
  }

  return (
    <Card className="bg-[#161b22] border-[#30363d]">
      <CardHeader>
        <CardTitle className="text-gray-200">Live Signals</CardTitle>
        <p className="text-sm text-gray-400">Auto-refreshing every 60 seconds</p>
      </CardHeader>
      <CardContent>
        {error ? (
          <p className="text-sm text-red-400">Failed to load signals</p>
        ) : !signals || !Array.isArray(signals) ? (
          <p className="text-sm text-gray-400">Loading signals...</p>
        ) : (
          <div className="rounded-md border border-[#30363d]">
            <Table>
              <TableHeader>
                <TableRow className="border-[#30363d] hover:bg-[#1c2128]">
                  <TableHead className="text-gray-400">Source</TableHead>
                  <TableHead className="text-gray-400">Action</TableHead>
                  <TableHead className="text-gray-400">Ticker</TableHead>
                  <TableHead className="text-gray-400">Conviction</TableHead>
                  <TableHead className="text-gray-400">Timestamp</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {signals.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center text-gray-400">
                      No signals available
                    </TableCell>
                  </TableRow>
                ) : (
                  signals.map((signal, index) => (
                    <TableRow key={index} className="border-[#30363d] hover:bg-[#1c2128]">
                      <TableCell className="font-medium text-gray-300">{signal.source}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className={getActionColor(signal.action)}>
                          {signal.action}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-gray-300 font-mono">{signal.ticker}</TableCell>
                      <TableCell className="text-gray-300">{signal.conviction}%</TableCell>
                      <TableCell className="text-gray-400 text-sm">{formatTimestamp(signal.timestamp)}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
