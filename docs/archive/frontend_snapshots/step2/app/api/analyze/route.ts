import { type NextRequest, NextResponse } from "next/server"
import type { AnalysisResult } from "@/lib/types/analysis"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // TODO: Replace with actual API endpoint
    const API_ENDPOINT = process.env.EIMAS_API_URL || "http://localhost:8000"

    const response = await fetch(`${API_ENDPOINT}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`)
    }

    const data: AnalysisResult = await response.json()

    // Add timestamp if not present
    if (!data.timestamp) {
      data.timestamp = new Date().toISOString()
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("[v0] Analysis API error:", error)
    return NextResponse.json({ error: "Failed to run analysis" }, { status: 500 })
  }
}
