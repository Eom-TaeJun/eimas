import { NextResponse } from "next/server"

// Mock data - Replace with actual database queries
export async function GET() {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 500))

  const portfolioData = {
    cash: 100000,
    positions_value: 50000,
    total_value: 150000,
    total_pnl: 5000,
    total_pnl_pct: 3.45,
    positions: [
      {
        ticker: "SPY",
        quantity: 100,
        avg_cost: 450.0,
        current_price: 455.0,
        market_value: 45500,
        unrealized_pnl: 500,
        unrealized_pnl_pct: 1.11,
      },
      {
        ticker: "AAPL",
        quantity: 50,
        avg_cost: 180.0,
        current_price: 185.0,
        market_value: 9250,
        unrealized_pnl: 250,
        unrealized_pnl_pct: 2.78,
      },
      {
        ticker: "TSLA",
        quantity: 20,
        avg_cost: 250.0,
        current_price: 245.0,
        market_value: 4900,
        unrealized_pnl: -100,
        unrealized_pnl_pct: -2.0,
      },
    ],
  }

  return NextResponse.json(portfolioData)
}
