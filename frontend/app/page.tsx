"use client"

import { Navbar } from "@/components/Navbar"
import { MetricsGrid } from "@/components/MetricsGrid"
import { ChartsSection } from "@/components/charts"
import { SignalsTable } from "@/components/SignalsTable"
import { FREDLiquidityDashboard } from "@/components/FREDLiquidityDashboard"
import { EventFeed } from "@/components/EventFeed"
import { StablecoinMonitor } from "@/components/StablecoinMonitor"
import { CryptoStressTest } from "@/components/CryptoStressTest"
import { VolumeAnomalies } from "@/components/VolumeAnomalies"
import { useEffect, useState } from "react"
import { FileText } from "lucide-react"
import { Button } from "@/components/ui/button"

function ReportButton() {
  const [reportUrl, setReportUrl] = useState<string | null>(null);

  useEffect(() => {
    fetch("http://localhost:8000/api/reports/latest")
      .then(res => res.json())
      .then(data => {
        if (data.url) setReportUrl(`http://localhost:8000${data.url}`);
      })
      .catch(err => console.error("Failed to fetch report URL", err));
  }, []);

  if (!reportUrl) return null;

  return (
    <Button
      variant="outline"
      onClick={() => window.open(reportUrl, "_blank")}
      className="bg-[#238636] text-white border-none hover:bg-[#2ea043] flex gap-2"
    >
      <FileText className="w-4 h-4" />
      View Latest Report
    </Button>
  );
}

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-[#0d1117]">
      <Navbar />
      <main className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          <div>
            <div className="flex justify-between items-center mb-2">
              <h1 className="text-3xl font-bold text-white">Dashboard</h1>
              <ReportButton />
            </div>
            <p className="text-gray-400">Economic Intelligence Multi-Agent System</p>
          </div>

          {/* Main Status & Key Metrics */}
          <MetricsGrid />

          {/* FRED Liquidity Dashboard */}
          <FREDLiquidityDashboard />

          {/* Charts & Analytics */}
          <ChartsSection />

          {/* Stablecoin & Crypto Monitoring */}
          <StablecoinMonitor />

          <CryptoStressTest />

          {/* Market Events & Anomalies */}
          <EventFeed />

          <VolumeAnomalies />

          {/* Signals Table */}
          <SignalsTable />
        </div>
      </main>
    </div>
  )
}
