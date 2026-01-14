"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import useSWR from "swr";
import { Server, Activity, Clock, CheckCircle, XCircle } from "lucide-react";
import { useEffect, useState } from "react";

interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  last_analysis_id: string | null;
}

const fetchHealth = async (): Promise<HealthResponse> => {
  const response = await fetch("http://localhost:8000/health");
  if (!response.ok) {
    throw new Error("Health check failed");
  }
  return response.json();
};

export function SystemStatusDashboard() {
  const [lastAnalysisTime, setLastAnalysisTime] = useState<string>("");

  // Poll health endpoint every 5 seconds
  const { data: health, error } = useSWR<HealthResponse>(
    "health-check",
    fetchHealth,
    {
      refreshInterval: 5000,
      shouldRetryOnError: true,
      errorRetryCount: 3,
    }
  );

  const isOnline = !error && health?.status === "healthy";

  // Calculate time since last analysis
  useEffect(() => {
    if (!health?.timestamp) return;

    const updateRelativeTime = () => {
      const now = new Date();
      const analysisTime = new Date(health.timestamp);
      const diffMs = now.getTime() - analysisTime.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      const diffSecs = Math.floor((diffMs % 60000) / 1000);

      if (diffMins > 0) {
        setLastAnalysisTime(`${diffMins}m ${diffSecs}s ago`);
      } else {
        setLastAnalysisTime(`${diffSecs}s ago`);
      }
    };

    updateRelativeTime();
    const interval = setInterval(updateRelativeTime, 1000);
    return () => clearInterval(interval);
  }, [health?.timestamp]);

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-white flex items-center gap-2">
            <Activity className="w-5 h-5" />
            System Status
          </CardTitle>
          {isOnline ? (
            <Badge className="bg-green-500/10 text-green-400 border-green-500/20">
              <CheckCircle className="w-3 h-3 mr-1" />
              Online
            </Badge>
          ) : (
            <Badge className="bg-red-500/10 text-red-400 border-red-500/20">
              <XCircle className="w-3 h-3 mr-1" />
              Offline
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Backend Server */}
          <div className="flex items-center justify-between p-3 bg-[#161b22] rounded-lg border border-[#30363d]">
            <div className="flex items-center gap-3">
              <Server className={`w-5 h-5 ${isOnline ? "text-green-400" : "text-red-400"}`} />
              <div>
                <div className="text-sm font-medium text-white">FastAPI Backend</div>
                <div className="text-xs text-gray-400">
                  {isOnline ? `v${health?.version || "2.0.0"}` : "Connection Failed"}
                </div>
              </div>
            </div>
            <div className={`w-3 h-3 rounded-full ${isOnline ? "bg-green-400 animate-pulse" : "bg-red-400"}`}></div>
          </div>

          {/* Analysis Engine */}
          <div className="flex items-center justify-between p-3 bg-[#161b22] rounded-lg border border-[#30363d]">
            <div className="flex items-center gap-3">
              <Activity className={`w-5 h-5 ${health?.last_analysis_id ? "text-blue-400" : "text-gray-400"}`} />
              <div>
                <div className="text-sm font-medium text-white">Analysis Engine</div>
                <div className="text-xs text-gray-400">
                  {health?.last_analysis_id ? "Running" : "Idle"}
                </div>
              </div>
            </div>
            <div className={`w-3 h-3 rounded-full ${health?.last_analysis_id ? "bg-blue-400 animate-pulse" : "bg-gray-600"}`}></div>
          </div>

          {/* Last Analysis */}
          <div className="flex items-center justify-between p-3 bg-[#161b22] rounded-lg border border-[#30363d]">
            <div className="flex items-center gap-3">
              <Clock className="w-5 h-5 text-yellow-400" />
              <div>
                <div className="text-sm font-medium text-white">Last Analysis</div>
                <div className="text-xs text-gray-400">
                  {lastAnalysisTime || "No recent analysis"}
                </div>
              </div>
            </div>
          </div>

          {/* Status Message */}
          <div className="mt-4 p-3 bg-[#161b22] rounded-lg border border-[#30363d]">
            <p className="text-xs text-gray-400">
              {isOnline ? (
                <>
                  ✓ Backend is running. Frontend continues to operate even if backend stops.
                </>
              ) : (
                <>
                  ⚠️ Backend connection lost. Dashboard displays cached data. Start FastAPI server to resume.
                </>
              )}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
