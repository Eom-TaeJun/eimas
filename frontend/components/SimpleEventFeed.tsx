"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"
import { AlertCircle, TrendingUp, Newspaper, Clock, Activity } from "lucide-react"

export function SimpleEventFeed() {
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  })

  if (error || !analysis) {
    return (
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Market Events
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-gray-400 text-sm">Loading events...</div>
        </CardContent>
      </Card>
    )
  }

  const events = analysis.events_detected || []

  if (events.length === 0) {
    return (
      <Card className="bg-[#161b22] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Market Events
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-gray-400 text-sm">No events detected</div>
        </CardContent>
      </Card>
    )
  }

  const getEventIcon = (type: string) => {
    const t = type.toLowerCase()
    if (t.includes('news') || t.includes('surge')) {
      return <Newspaper className="w-4 h-4" />
    }
    if (t.includes('price') || t.includes('shock')) {
      return <TrendingUp className="w-4 h-4" />
    }
    return <AlertCircle className="w-4 h-4" />
  }

  const getSeverityColor = (importance: string) => {
    const imp = importance?.toUpperCase() || 'LOW'
    switch (imp) {
      case 'HIGH':
      case 'CRITICAL':
        return "bg-red-500/10 text-red-400 border-red-500/20"
      case 'MEDIUM':
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
      default:
        return "bg-blue-500/10 text-blue-400 border-blue-500/20"
    }
  }

  const getEventTypeColor = (type: string) => {
    const t = type.toLowerCase()
    if (t.includes('news') || t.includes('crypto')) {
      return "bg-purple-500/10 text-purple-400 border-purple-500/20"
    }
    if (t.includes('price') || t.includes('shock')) {
      return "bg-green-500/10 text-green-400 border-green-500/20"
    }
    if (t.includes('liquidity')) {
      return "bg-blue-500/10 text-blue-400 border-blue-500/20"
    }
    return "bg-gray-500/10 text-gray-400 border-gray-500/20"
  }

  const extractCategory = (description: string): string => {
    const match = description.match(/^\[([^\]]+)\]/)
    return match ? match[1] : ''
  }

  const cleanDescription = (description: string): string => {
    return description.replace(/^\[([^\]]+)\]\s*/, '')
  }

  return (
    <Card className="bg-[#161b22] border-[#30363d]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Market Events
          </CardTitle>
          <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
            {events.length} Events
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {events.map((event: any, idx: number) => {
            const category = extractCategory(event.description)
            const cleanDesc = cleanDescription(event.description)

            return (
              <div
                key={idx}
                className="bg-[#0d1117] rounded-lg p-4 border border-[#30363d] hover:border-[#484f58] transition-colors"
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getEventIcon(event.type)}
                    <Badge variant="outline" className={getEventTypeColor(event.type)}>
                      {event.type.replace(/_/g, ' ')}
                    </Badge>
                  </div>
                  <Badge variant="outline" className={getSeverityColor(event.importance)}>
                    {event.importance || 'LOW'}
                  </Badge>
                </div>

                {/* Category */}
                {category && (
                  <div className="mb-2">
                    <Badge variant="outline" className="text-xs bg-gray-500/10 text-gray-400 border-gray-500/20">
                      {category}
                    </Badge>
                  </div>
                )}

                {/* Description */}
                <div className="text-sm text-gray-300 mb-2 leading-relaxed">
                  {cleanDesc}
                </div>

                {/* Timestamp */}
                <div className="flex items-center gap-1 text-xs text-gray-500">
                  <Clock className="w-3 h-3" />
                  {new Date(event.timestamp).toLocaleString()}
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
