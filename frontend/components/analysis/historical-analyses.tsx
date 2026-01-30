"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Clock } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AnalysisResult, Stance } from "@/lib/types/analysis"

interface HistoricalAnalysesProps {
  analyses: AnalysisResult[]
  onLoadAnalysis: (analysisId: string) => void
  currentAnalysisId?: string
}

export function HistoricalAnalyses({ analyses, onLoadAnalysis, currentAnalysisId }: HistoricalAnalysesProps) {
  const getStanceColor = (stance?: Stance) => {
    switch (stance) {
      case "BULLISH":
        return "bg-green-500/10 text-green-500 border-green-500/20"
      case "BEARISH":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      case "NEUTRAL":
        return "bg-gray-500/10 text-gray-500 border-gray-500/20"
      default:
        return ""
    }
  }

  const formatTimestamp = (timestamp?: string) => {
    if (!timestamp) return "Unknown"
    const date = new Date(timestamp)
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }).format(date)
  }

  if (analyses.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        <Clock className="h-12 w-12 mx-auto mb-3 opacity-50" />
        <p className="text-sm">No analyses yet</p>
        <p className="text-xs mt-1">Run your first analysis to get started</p>
      </div>
    )
  }

  return (
    <div className="h-[600px] -mx-6 px-6 overflow-y-auto">
      <div className="space-y-3">
        {analyses.map((analysis) => (
          <Button
            key={analysis.analysis_id}
            onClick={() => onLoadAnalysis(analysis.analysis_id)}
            variant="outline"
            className={cn(
              "w-full h-auto p-3 justify-start text-left",
              currentAnalysisId === analysis.analysis_id && "border-primary",
            )}
          >
            <div className="w-full space-y-2">
              <div className="flex items-start justify-between gap-2">
                <code className="text-xs font-mono truncate flex-1">{analysis.analysis_id}</code>
                {analysis.final_stance && (
                  <Badge className={cn("text-xs border", getStanceColor(analysis.final_stance))}>
                    {analysis.final_stance}
                  </Badge>
                )}
              </div>

              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span>{formatTimestamp(analysis.timestamp)}</span>
                {analysis.confidence !== undefined && <span>{analysis.confidence}%</span>}
              </div>

              {analysis.question && <p className="text-xs text-muted-foreground line-clamp-2">{analysis.question}</p>}
            </div>
          </Button>
        ))}
      </div>
    </div>
  )
}
