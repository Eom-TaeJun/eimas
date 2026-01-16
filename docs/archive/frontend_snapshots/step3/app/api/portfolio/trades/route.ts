import { NextResponse } from "next/server"

// Mock data - Replace with actual database queries
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const days = Number.parseInt(searchParams.get("days") || "30")

  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 300))

  // Generate mock trades based on the days parameter
  const trades = [
    {
      timestamp: "2026-01-10T12:00:00",
      ticker: "SPY",
      side: "BUY" as const,
      quantity: 100,
      price: 450.0,
      realized_pnl: 0,
    },
    {
      timestamp: "2026-01-09T14:30:00",
      ticker: "AAPL",
      side: "BUY" as const,
      quantity: 50,
      price: 180.0,
      realized_pnl: 0,
    },
    {
      timestamp: "2026-01-08T10:15:00",
      ticker: "TSLA",
      side: "BUY" as const,
      quantity: 20,
      price: 250.0,
      realized_pnl: 0,
    },
    {
      timestamp: "2026-01-05T11:45:00",
      ticker: "MSFT",
      side: "BUY" as const,
      quantity: 30,
      price: 350.0,
      realized_pnl: 0,
    },
    {
      timestamp: "2026-01-04T15:20:00",
      ticker: "MSFT",
      side: "SELL" as const,
      quantity: 30,
      price: 355.0,
      realized_pnl: 150.0,
    },
  ]

  // Filter trades based on days parameter
  const cutoffDate = new Date()
  cutoffDate.setDate(cutoffDate.getDate() - days)

  const filteredTrades = trades.filter((trade) => new Date(trade.timestamp) >= cutoffDate)

  return NextResponse.json(filteredTrades)
}
