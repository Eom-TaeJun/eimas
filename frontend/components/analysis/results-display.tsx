"use client"

import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AnalysisResult, Stance, AnalysisStatus } from "@/lib/types/analysis"
import ReactMarkdown from "react-markdown"

interface ResultsDisplayProps {
  result: AnalysisResult
}

export function ResultsDisplay({ result }: ResultsDisplayProps) {
  const getStanceColor = (stance?: Stance) => {
    switch (stance) {
      case "BULLISH":
        return "bg-green-500 hover:bg-green-600 text-white"
      case "BEARISH":
        return "bg-red-500 hover:bg-red-600 text-white"
      case "NEUTRAL":
        return "bg-gray-500 hover:bg-gray-600 text-white"
      default:
        return "bg-muted text-muted-foreground"
    }
  }

  const getStatusColor = (status: AnalysisStatus) => {
    switch (status) {
      case "COMPLETED":
        return "bg-green-500/10 text-green-500 border-green-500/20"
      case "RUNNING":
        return "bg-blue-500/10 text-blue-500 border-blue-500/20"
      case "PENDING":
        return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
      case "FAILED":
        return "bg-red-500/10 text-red-500 border-red-500/20"
      default:
        return ""
    }
  }

  return (
    <Card className="p-6">
      <div className="space-y-6">
        <div>
          <h2 className="text-2xl font-semibold mb-4">Analysis Results</h2>

          <div className="flex flex-wrap items-center gap-3 mb-4">
            <div>
              <span className="text-sm text-muted-foreground">Analysis ID:</span>
              <code className="ml-2 px-2 py-1 bg-muted rounded text-sm font-mono">{result.analysis_id}</code>
            </div>
            <Badge className={cn("border", getStatusColor(result.status))}>{result.status}</Badge>
          </div>
        </div>

        {result.final_stance && (
          <div>
            <h3 className="text-sm font-medium mb-2">Final Stance</h3>
            <Badge className={cn("text-lg py-2 px-4", getStanceColor(result.final_stance))}>
              {result.final_stance}
            </Badge>
          </div>
        )}

        {result.confidence !== undefined && (
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-medium">Confidence</h3>
              <span className="text-sm text-muted-foreground">{result.confidence}%</span>
            </div>
            <Progress value={result.confidence} className="h-2" />
          </div>
        )}

        {result.executive_summary && (
          <div>
            <h3 className="text-sm font-medium mb-2">Executive Summary</h3>
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown>{result.executive_summary}</ReactMarkdown>
            </div>
          </div>
        )}

        {result.top_down_summary && (
          <Collapsible>
            <CollapsibleTrigger className="flex items-center justify-between w-full py-2 hover:bg-muted/50 rounded px-2 -mx-2">
              <h3 className="text-sm font-medium">Top-Down Summary</h3>
              <ChevronDown className="h-4 w-4 transition-transform duration-200" />
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2">
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown>{result.top_down_summary}</ReactMarkdown>
              </div>
            </CollapsibleContent>
          </Collapsible>
        )}

        {result.regime_context && (
          <div>
            <h3 className="text-sm font-medium mb-2">Regime Context</h3>
            <div className="bg-muted rounded-lg p-4 space-y-2">
              {result.regime_context.regime_type && (
                <div>
                  <span className="text-sm text-muted-foreground">Type:</span>
                  <span className="ml-2 text-sm font-medium">{result.regime_context.regime_type}</span>
                </div>
              )}
              {result.regime_context.confidence !== undefined && (
                <div>
                  <span className="text-sm text-muted-foreground">Confidence:</span>
                  <span className="ml-2 text-sm font-medium">{result.regime_context.confidence}%</span>
                </div>
              )}
              {result.regime_context.indicators && result.regime_context.indicators.length > 0 && (
                <div>
                  <span className="text-sm text-muted-foreground block mb-1">Indicators:</span>
                  <div className="flex flex-wrap gap-1">
                    {result.regime_context.indicators.map((indicator, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {indicator}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {result.stages_completed && result.stages_completed.length > 0 && (
          <div>
            <h3 className="text-sm font-medium mb-2">Stages Completed</h3>
            <div className="flex flex-wrap gap-2">
              {result.stages_completed.map((stage, index) => (
                <Badge key={index} variant="outline">
                  {stage}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {result.duration !== undefined && (
          <div className="flex items-center justify-between pt-4 border-t">
            <span className="text-sm text-muted-foreground">Duration</span>
            <span className="text-sm font-medium">{result.duration.toFixed(1)}s</span>
          </div>
        )}
      </div>
    </Card>
  )
}
