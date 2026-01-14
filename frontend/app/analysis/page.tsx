"use client"

import { useState } from "react"
import { AnalysisForm } from "@/components/analysis/analysis-form"
import { ResultsDisplay } from "@/components/analysis/results-display"
import { HistoricalAnalyses } from "@/components/analysis/historical-analyses"
import { Card } from "@/components/ui/card"
import type { AnalysisResult } from "@/lib/types/analysis"

export default function AnalysisPage() {
  const [currentResult, setCurrentResult] = useState<AnalysisResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [historicalResults, setHistoricalResults] = useState<AnalysisResult[]>([])

  const handleAnalysisSubmit = async (data: {
    question: string
    analysis_level: string
    research_goal: string
    use_mock: boolean
  }) => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      })

      if (!response.ok) {
        throw new Error("Analysis failed")
      }

      const result: AnalysisResult = await response.json()
      setCurrentResult(result)
      setHistoricalResults((prev) => [result, ...prev])
    } catch (error) {
      console.error("[v0] Analysis error:", error)
      // Error handling will be shown via toast or error state in form
    } finally {
      setIsLoading(false)
    }
  }

  const handleLoadHistorical = async (analysisId: string) => {
    try {
      const response = await fetch(`/api/analyze/${analysisId}`)
      if (!response.ok) {
        throw new Error("Failed to load analysis")
      }
      const result: AnalysisResult = await response.json()
      setCurrentResult(result)
    } catch (error) {
      console.error("[v0] Failed to load historical analysis:", error)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        <div className="mb-8">
          <h1 className="text-4xl font-bold tracking-tight mb-2">EIMAS Analysis</h1>
          <p className="text-muted-foreground text-lg">Run economic intelligence and market analysis reports</p>
        </div>

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-6">
            <Card className="p-6">
              <h2 className="text-2xl font-semibold mb-4">New Analysis Request</h2>
              <AnalysisForm onSubmit={handleAnalysisSubmit} isLoading={isLoading} />
            </Card>

            {currentResult && <ResultsDisplay result={currentResult} />}
          </div>

          <div className="lg:col-span-1">
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Historical Analyses</h2>
              <HistoricalAnalyses
                analyses={historicalResults}
                onLoadAnalysis={handleLoadHistorical}
                currentAnalysisId={currentResult?.analysis_id}
              />
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
