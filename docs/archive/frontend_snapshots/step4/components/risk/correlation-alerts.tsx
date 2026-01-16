"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { AlertCircle, AlertTriangle, Info } from "lucide-react"

interface Alert {
  type: string
  message: string
  severity: "Low" | "Medium" | "High"
}

export function CorrelationAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchData() {
      try {
        const res = await fetch("/api/correlation?assets=SPY,TLT,GLD,QQQ")
        const json = await res.json()
        setAlerts(json.alerts || [])
      } catch (error) {
        console.error("[v0] Failed to fetch correlation alerts:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "High":
        return "destructive"
      case "Medium":
        return "default"
      case "Low":
        return "secondary"
      default:
        return "secondary"
    }
  }

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "High":
        return AlertCircle
      case "Medium":
        return AlertTriangle
      case "Low":
        return Info
      default:
        return Info
    }
  }

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Correlation Alerts</CardTitle>
          <CardDescription>Loading...</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Correlation Alerts</CardTitle>
        <CardDescription>Unusual correlation patterns detected</CardDescription>
      </CardHeader>
      <CardContent>
        {alerts.length === 0 ? (
          <p className="text-sm text-muted-foreground">No alerts at this time</p>
        ) : (
          <div className="space-y-3">
            {alerts.map((alert, index) => {
              const Icon = getSeverityIcon(alert.severity)
              return (
                <div key={index} className="flex items-start gap-3 p-3 rounded-lg border bg-card">
                  <Icon className="h-5 w-5 mt-0.5 text-muted-foreground" />
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm">{alert.type}</span>
                      <Badge variant={getSeverityColor(alert.severity)}>{alert.severity}</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{alert.message}</p>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
