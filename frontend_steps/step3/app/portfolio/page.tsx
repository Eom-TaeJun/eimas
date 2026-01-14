"use client"

import { PortfolioSummary } from "@/components/portfolio-summary"
import { PositionsTable } from "@/components/positions-table"
import { TradeHistory } from "@/components/trade-history"
import { PaperTradeForm } from "@/components/paper-trade-form"

export default function PortfolioPage() {
  return (
    <div className="container mx-auto py-8 px-4">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2">Portfolio</h1>
        <p className="text-muted-foreground">Manage your paper trading portfolio and track your performance</p>
      </div>

      <div className="grid gap-6">
        <PortfolioSummary />

        <div className="grid gap-6 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <PositionsTable />
          </div>
          <div>
            <PaperTradeForm />
          </div>
        </div>

        <TradeHistory />
      </div>
    </div>
  )
}
