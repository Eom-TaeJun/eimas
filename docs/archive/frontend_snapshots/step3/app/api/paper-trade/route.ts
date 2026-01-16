import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const { ticker, side, quantity } = body

    // Validate input
    if (!ticker || !side || !quantity) {
      return NextResponse.json({ error: "Missing required fields: ticker, side, quantity" }, { status: 400 })
    }

    if (!["BUY", "SELL"].includes(side)) {
      return NextResponse.json({ error: "Invalid side. Must be BUY or SELL" }, { status: 400 })
    }

    if (quantity <= 0) {
      return NextResponse.json({ error: "Quantity must be greater than 0" }, { status: 400 })
    }

    // Simulate API delay
    await new Promise((resolve) => setTimeout(resolve, 500))

    // Mock trade execution - Replace with actual trade logic
    // In a real implementation, you would:
    // 1. Validate the ticker exists
    // 2. Check if user has sufficient funds (for BUY) or shares (for SELL)
    // 3. Get current market price
    // 4. Execute the trade
    // 5. Update portfolio in database

    const executedTrade = {
      ticker,
      side,
      quantity,
      price: 450.0, // Mock price
      timestamp: new Date().toISOString(),
      success: true,
    }

    return NextResponse.json(executedTrade)
  } catch (error) {
    console.error("[v0] Paper trade error:", error)
    return NextResponse.json({ error: "Failed to execute trade" }, { status: 500 })
  }
}
