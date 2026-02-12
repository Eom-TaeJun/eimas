"use client"

import { Navbar } from "@/components/Navbar"
import { TabbedDashboard } from "@/components/TabbedDashboard"
import { ExportReportDialog } from "@/components/ExportReportDialog"
import { useEffect, useState } from "react"
import useSWR from "swr"
import { FileText, Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { fetchLatestAnalysis } from "@/lib/api"
import type { EIMASAnalysis } from "@/lib/types"

function ReportButtons() {
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [showExport, setShowExport] = useState(false);

  const { data: analysis } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
  });

  useEffect(() => {
    fetch("http://localhost:8000/api/reports/latest")
      .then(res => res.json())
      .then(data => {
        if (data.url) setReportUrl(`http://localhost:8000${data.url}`);
      })
      .catch(err => console.error("Failed to fetch report URL", err));
  }, []);

  return (
    <div className="flex items-center gap-3">
      <Button
        variant="outline"
        onClick={() => setShowExport(true)}
        className="bg-[#161b22] border-[#30363d] text-gray-300 hover:bg-[#238636] hover:text-white flex gap-2"
      >
        <Download className="w-4 h-4" />
        Export Report
      </Button>

      {reportUrl && (
        <Button
          variant="outline"
          onClick={() => window.open(reportUrl, "_blank")}
          className="bg-[#238636] text-white border-none hover:bg-[#2ea043] flex gap-2"
        >
          <FileText className="w-4 h-4" />
          View Latest Report
        </Button>
      )}

      {showExport && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="max-w-3xl w-full">
            <ExportReportDialog data={analysis} onClose={() => setShowExport(false)} />
          </div>
        </div>
      )}
    </div>
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
              <div>
                <h1 className="text-3xl font-bold text-white">Dashboard</h1>
                <p className="text-gray-400 mt-1">Economic Intelligence Multi-Agent System</p>
              </div>
              <ReportButtons />
            </div>
          </div>

          {/* New Tabbed Dashboard */}
          <TabbedDashboard />
        </div>
      </main>
    </div>
  )
}
