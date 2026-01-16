import { Suspense } from "react"
import { RiskMetrics } from "@/components/risk/risk-metrics"
import { PortfolioComposition } from "@/components/risk/portfolio-composition"
import { CorrelationMatrix } from "@/components/risk/correlation-matrix"
import { CorrelationAlerts } from "@/components/risk/correlation-alerts"

export const metadata = {
  title: "Risk Analytics | EIMAS",
  description: "Portfolio risk metrics and correlation analysis",
}

export default function RiskPage() {
  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-balance">Risk Analytics</h1>
        <p className="text-muted-foreground">Monitor portfolio risk metrics and asset correlations</p>
      </div>

      <Suspense fallback={<div>Loading...</div>}>
        <RiskMetrics />
      </Suspense>

      <div className="grid gap-6 lg:grid-cols-2">
        <Suspense fallback={<div>Loading...</div>}>
          <PortfolioComposition />
        </Suspense>

        <Suspense fallback={<div>Loading...</div>}>
          <CorrelationAlerts />
        </Suspense>
      </div>

      <Suspense fallback={<div>Loading...</div>}>
        <CorrelationMatrix />
      </Suspense>
    </div>
  )
}
