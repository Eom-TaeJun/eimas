import { type NextRequest, NextResponse } from "next/server"
import type { AnalysisResult } from "@/lib/types/analysis"

export async function GET(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params

    // TODO: Replace with actual API endpoint
    const API_ENDPOINT = process.env.EIMAS_API_URL || "http://localhost:8000"

    const response = await fetch(`${API_ENDPOINT}/analyze/${id}`)

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`)
    }

    const data: AnalysisResult = await response.json()

    return NextResponse.json(data)
  } catch (error) {
    console.error("[v0] Analysis fetch error:", error)
    return NextResponse.json({ error: "Failed to fetch analysis" }, { status: 500 })
  }
}
