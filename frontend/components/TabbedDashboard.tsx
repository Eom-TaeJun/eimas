"use client";

import { useState } from "react";
import useSWR from "swr";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  LayoutDashboard,
  LineChart,
  Brain,
  AlertTriangle,
  TrendingUp,
  FileText,
  Activity
} from "lucide-react";
import { MetricsGrid } from "@/components/MetricsGrid";
import { ChartsSection } from "@/components/charts";
import { SignalsTable } from "@/components/SignalsTable";
import { FREDLiquidityDashboard } from "@/components/FREDLiquidityDashboard";
import { EventFeed } from "@/components/EventFeed";
import { StablecoinMonitor } from "@/components/StablecoinMonitor";
import { CryptoStressTest } from "@/components/CryptoStressTest";
import { VolumeAnomalies } from "@/components/VolumeAnomalies";
import { ReasoningChainDisplay } from "@/components/ReasoningChainDisplay";
import { fetchLatestAnalysis } from "@/lib/api";
import type { EIMASAnalysis } from "@/lib/types";

type Tab = "overview" | "analytics" | "reasoning" | "risk" | "signals" | "events";

interface TabConfig {
  id: Tab;
  label: string;
  icon: React.ReactNode;
  badge?: string;
}

const tabs: TabConfig[] = [
  { id: "overview", label: "Overview", icon: <LayoutDashboard className="w-4 h-4" /> },
  { id: "analytics", label: "Analytics", icon: <LineChart className="w-4 h-4" /> },
  { id: "reasoning", label: "AI Reasoning", icon: <Brain className="w-4 h-4" />, badge: "New" },
  { id: "risk", label: "Risk Analysis", icon: <AlertTriangle className="w-4 h-4" /> },
  { id: "signals", label: "Signals", icon: <TrendingUp className="w-4 h-4" /> },
  { id: "events", label: "Events", icon: <Activity className="w-4 h-4" /> },
];

export function TabbedDashboard() {
  const [activeTab, setActiveTab] = useState<Tab>("overview");

  // Fetch latest EIMAS analysis for reasoning chain
  const { data, error } = useSWR<EIMASAnalysis>("latest-analysis", fetchLatestAnalysis, {
    refreshInterval: 5000,
    dedupingInterval: 1000,
  });

  return (
    <div className="space-y-6">
      {/* Tab Navigation */}
      <Card className="bg-[#0d1117] border-[#30363d]">
        <CardContent className="p-0">
          <div className="flex overflow-x-auto">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-4 border-b-2 transition-all whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-[#238636] text-white bg-[#161b22]"
                    : "border-transparent text-gray-400 hover:text-white hover:bg-[#161b22]/50"
                }`}
              >
                {tab.icon}
                <span className="font-medium">{tab.label}</span>
                {tab.badge && (
                  <Badge variant="outline" className="text-xs bg-green-500/10 text-green-400 border-green-500/20">
                    {tab.badge}
                  </Badge>
                )}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Tab Content */}
      <div className="min-h-[600px]">
        {activeTab === "overview" && (
          <div className="space-y-6">
            <MetricsGrid />
            <FREDLiquidityDashboard />
          </div>
        )}

        {activeTab === "analytics" && (
          <div className="space-y-6">
            <ChartsSection />
          </div>
        )}

        {activeTab === "reasoning" && (
          <div className="space-y-6">
            <ReasoningChainDisplay data={data} />
          </div>
        )}

        {activeTab === "risk" && (
          <div className="space-y-6">
            <StablecoinMonitor />
            <CryptoStressTest />
          </div>
        )}

        {activeTab === "signals" && (
          <div className="space-y-6">
            <SignalsTable />
            <VolumeAnomalies />
          </div>
        )}

        {activeTab === "events" && (
          <div className="space-y-6">
            <EventFeed />
          </div>
        )}
      </div>
    </div>
  );
}
