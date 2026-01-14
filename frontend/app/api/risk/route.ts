import { NextResponse } from "next/server"

export async function GET() {
  // Mock data - replace with actual risk calculation logic
  const riskData = {
    var_95: -5000,
    var_99: -8000,
    cvar: -9500,
    max_drawdown: -12.5,
    volatility: 15.2,
    sharpe_ratio: 1.45,
    composition: {
      SPY: 45,
      TLT: 25,
      GLD: 20,
      QQQ: 10,
    },
  }

  return NextResponse.json(riskData)
}
