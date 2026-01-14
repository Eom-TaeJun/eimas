import { Navbar } from "@/components/Navbar"
import { MetricsGrid } from "@/components/MetricsGrid"
import { ChartsSection } from "@/components/charts"
import { SignalsTable } from "@/components/SignalsTable"
import { FREDLiquidityDashboard } from "@/components/FREDLiquidityDashboard"
import { EventFeed } from "@/components/EventFeed"
import { StablecoinMonitor } from "@/components/StablecoinMonitor"
import { CryptoStressTest } from "@/components/CryptoStressTest"
import { VolumeAnomalies } from "@/components/VolumeAnomalies"

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-[#0d1117]">
      <Navbar />
      <main className="mx-auto max-w-[1400px] px-4 sm:px-6 lg:px-8 py-8">
        <div className="space-y-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Dashboard</h1>
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
