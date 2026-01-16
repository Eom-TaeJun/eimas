import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { TrendingDown, Activity, AlertTriangle, BarChart3, Target } from "lucide-react"

async function getRiskData() {
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000"}/api/risk`, {
    cache: "no-store",
  })
  if (!res.ok) throw new Error("Failed to fetch risk data")
  return res.json()
}

export async function RiskMetrics() {
  const data = await getRiskData()

  const metrics = [
    {
      title: "VaR 95%",
      value: `$${data.var_95.toLocaleString()}`,
      icon: TrendingDown,
      color: "text-red-500",
    },
    {
      title: "VaR 99%",
      value: `$${data.var_99.toLocaleString()}`,
      icon: AlertTriangle,
      color: "text-orange-500",
    },
    {
      title: "CVaR",
      value: `$${data.cvar.toLocaleString()}`,
      icon: TrendingDown,
      color: "text-red-600",
    },
    {
      title: "Max Drawdown",
      value: `${data.max_drawdown}%`,
      icon: BarChart3,
      color: "text-amber-500",
    },
    {
      title: "Volatility",
      value: `${data.volatility}%`,
      icon: Activity,
      color: "text-blue-500",
    },
    {
      title: "Sharpe Ratio",
      value: data.sharpe_ratio,
      icon: Target,
      color: "text-green-500",
    },
  ]

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      {metrics.map((metric) => {
        const Icon = metric.icon
        return (
          <Card key={metric.title}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{metric.title}</CardTitle>
              <Icon className={`h-4 w-4 ${metric.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{metric.value}</div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
