"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import useSWR from "swr"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"
import { Volume2, AlertTriangle } from "lucide-react"

export function VolumeAnomalies() {
  const { data: analysis, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  })

  if (
    error ||
    !analysis ||
    !analysis.volume_anomalies ||
    analysis.volume_anomalies.length === 0
  ) {
    return null
  }

  const anomalies = analysis.volume_anomalies

  const getSeverityColor = (severity: string) => {
    switch (severity.toUpperCase()) {
      case "HIGH":
        return "bg-red-500/10 text-red-400 border-red-500/20"
      case "MEDIUM":
        return "bg-yellow-500/10 text-yellow-400 border-yellow-500/20"
      default:
        return "bg-blue-500/10 text-blue-400 border-blue-500/20"
    }
  }

  const getAnomalyColor = (type: string) => {
    if (type.includes("surge")) return "bg-green-500/10 text-green-400 border-green-500/20"
    if (type.includes("drop")) return "bg-red-500/10 text-red-400 border-red-500/20"
    return "bg-purple-500/10 text-purple-400 border-purple-500/20"
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Volume2 className="w-6 h-6 text-purple-400" />
        <h2 className="text-xl font-bold text-white">Volume Anomalies</h2>
        <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/20">
          {anomalies.length} Detected
        </Badge>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        {anomalies.map((anomaly, idx) => (
          <Card key={idx} className="bg-[#161b22] border-[#30363d]">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Badge
                    variant="outline"
                    className="bg-blue-500/10 text-blue-400 border-blue-500/20 font-mono"
                  >
                    {anomaly.ticker}
                  </Badge>
                  <Badge variant="outline" className={getSeverityColor(anomaly.severity)}>
                    {anomaly.severity}
                  </Badge>
                </div>
                <Badge variant="outline" className={getAnomalyColor(anomaly.anomaly_type)}>
                  {anomaly.anomaly_type.replace("abnormal_", "")}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {/* Volume Metrics */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-xs text-gray-400">Volume Ratio</div>
                    <div className="text-lg font-bold text-white">
                      {anomaly.volume_ratio.toFixed(1)}x
                    </div>
                    <div className="text-xs text-gray-400">avg volume</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Z-Score</div>
                    <div
                      className={`text-lg font-bold ${
                        anomaly.z_score > 3
                          ? "text-red-400"
                          : anomaly.z_score > 2
                          ? "text-yellow-400"
                          : "text-blue-400"
                      }`}
                    >
                      {anomaly.z_score.toFixed(1)}σ
                    </div>
                  </div>
                </div>

                {/* Price Changes */}
                <div className="grid grid-cols-2 gap-3 pt-2 border-t border-gray-700">
                  <div>
                    <div className="text-xs text-gray-400">1-Day</div>
                    <div
                      className={`text-sm font-mono ${
                        anomaly.price_change_1d > 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {anomaly.price_change_1d > 0 ? "+" : ""}
                      {anomaly.price_change_1d.toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">5-Day</div>
                    <div
                      className={`text-sm font-mono ${
                        anomaly.price_change_5d > 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      {anomaly.price_change_5d > 0 ? "+" : ""}
                      {anomaly.price_change_5d.toFixed(2)}%
                    </div>
                  </div>
                </div>

                {/* Alert Message */}
                <div className="pt-2 border-t border-gray-700">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                    <div className="text-xs text-gray-300 leading-relaxed">
                      {anomaly.alert_message}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
