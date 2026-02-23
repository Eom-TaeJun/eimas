"use client";

import { Card, CardContent } from "@/components/ui/card";
import { Loader2, TrendingUp, Brain, LineChart, Shield } from "lucide-react";

interface LoadingStateProps {
  message?: string;
  type?: "default" | "analysis" | "charts" | "reasoning";
}

const loadingIcons = {
  default: Loader2,
  analysis: TrendingUp,
  charts: LineChart,
  reasoning: Brain,
};

const loadingMessages = {
  default: "Loading...",
  analysis: "Analyzing market data...",
  charts: "Generating visualizations...",
  reasoning: "Processing AI reasoning chain...",
};

export function LoadingState({ message, type = "default" }: LoadingStateProps) {
  const Icon = loadingIcons[type];
  const defaultMessage = loadingMessages[type];

  return (
    <Card className="bg-surface border-border">
      <CardContent className="p-12">
        <div className="flex flex-col items-center justify-center text-center">
          <Icon className="w-12 h-12 text-purple-400 animate-spin mb-4" />
          <h3 className="text-lg font-semibold text-white mb-2">{message || defaultMessage}</h3>
          <p className="text-sm text-gray-400">This may take a few moments</p>

          {/* Loading Progress Indicators */}
          <div className="mt-6 w-full max-w-md">
            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>Fetching latest data</span>
                <Shield className="w-4 h-4 animate-pulse" />
              </div>
              <div className="w-full bg-gray-700 rounded-full h-1.5">
                <div className="bg-purple-500 h-1.5 rounded-full animate-pulse" style={{ width: "75%" }}></div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function SkeletonCard() {
  return (
    <Card className="bg-surface border-border">
      <CardContent className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-700 rounded w-1/3"></div>
          <div className="space-y-2">
            <div className="h-4 bg-gray-700 rounded"></div>
            <div className="h-4 bg-gray-700 rounded w-5/6"></div>
            <div className="h-4 bg-gray-700 rounded w-4/6"></div>
          </div>
          <div className="h-32 bg-gray-700 rounded"></div>
        </div>
      </CardContent>
    </Card>
  );
}

export function GridSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {Array.from({ length: count }).map((_, i) => (
        <SkeletonCard key={i} />
      ))}
    </div>
  );
}
