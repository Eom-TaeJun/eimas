"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"
import { TrendingUp, TrendingDown, Volume2, Newspaper, Clock } from "lucide-react"
import { useState } from "react"

export function EventFeed() {
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  })

  const [expandedEvents, setExpandedEvents] = useState<Set<number>>(new Set())

  if (error || !analysis || !analysis.tracked_events || analysis.tracked_events.length === 0) {
    return null
  }

  const events = analysis.tracked_events.slice(0, 10) // Limit to 10 most recent

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case "positive":
        return <TrendingUp className="w-4 h-4 text-green-400" />
      case "negative":
        return <TrendingDown className="w-4 h-4 text-red-400" />
      default:
        return <Volume2 className="w-4 h-4 text-gray-400" />
    }
  }

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case "positive":
        return "bg-green-500/10 text-green-400 border-green-500/20"
      case "negative":
        return "bg-red-500/10 text-red-400 border-red-500/20"
      default:
        return "bg-gray-500/10 text-gray-400 border-gray-500/20"
    }
  }

  const getAnomalyBadgeColor = (type: string) => {
    if (type.includes("SURGE")) return "bg-green-500/10 text-green-400 border-green-500/20"
    if (type.includes("DROP")) return "bg-red-500/10 text-red-400 border-red-500/20"
    return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
  }

  const toggleExpand = (index: number) => {
    const newExpanded = new Set(expandedEvents)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedEvents(newExpanded)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Newspaper className="w-6 h-6 text-purple-400" />
        <h2 className="text-xl font-bold text-white">Event Tracking Feed</h2>
        <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/20">
          {events.length} Events
        </Badge>
      </div>

      <div className="space-y-3">
        {events.map((event, idx) => {
          const isExpanded = expandedEvents.has(idx)
          return (
            <Card
              key={idx}
              className="bg-[#161b22] border-[#30363d] hover:border-[#484f58] transition-colors cursor-pointer"
              onClick={() => toggleExpand(idx)}
            >
              <CardContent className="pt-4">
                <div className="space-y-3">
                  {/* Header Row */}
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20 font-mono text-xs">
                        {event.ticker}
                      </Badge>
                      <div className="flex items-center gap-2 text-xs text-gray-400">
                        <Clock className="w-3 h-3" />
                        {new Date(event.timestamp).toLocaleDateString()}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {getSentimentIcon(event.sentiment)}
                      <Badge variant="outline" className={getSentimentColor(event.sentiment)}>
                        {event.impact_score}/100
                      </Badge>
                    </div>
                  </div>

                  {/* Anomaly Type */}
                  <div className="flex flex-wrap gap-2">
                    {event.anomaly_type.split(",").map((type, i) => (
                      <Badge
                        key={i}
                        variant="outline"
                        className={getAnomalyBadgeColor(type.trim())}
                      >
                        {type.trim()}
                      </Badge>
                    ))}
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-gray-400">Volume Z-Score: </span>
                      <span className="text-white font-mono">{event.volume_zscore.toFixed(2)}</span>
                    </div>
                    <div>
                      <span className="text-gray-400">Price Change: </span>
                      <span
                        className={`font-mono ${
                          event.price_change_pct > 0 ? "text-green-400" : "text-red-400"
                        }`}
                      >
                        {event.price_change_pct > 0 ? "+" : ""}
                        {event.price_change_pct.toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  {/* News Summary (Expandable) */}
                  {event.news_found && event.news_summary && (
                    <div className="pt-2 border-t border-gray-700">
                      <div className="text-sm">
                        <div className="text-gray-300 font-medium mb-1">
                          {event.news_summary.split(":")[0] || "News"}
                        </div>
                        {isExpanded && (
                          <div className="text-gray-400 text-xs leading-relaxed mt-2">
                            {event.news_summary}
                          </div>
                        )}
                        {!isExpanded && (
                          <div className="text-gray-500 text-xs">
                            Click to expand...
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
