"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckCircle, AlertTriangle } from "lucide-react";

interface ConsensusComparisonChartProps {
  full_mode: "BULLISH" | "BEARISH" | "NEUTRAL";
  reference_mode: "BULLISH" | "BEARISH" | "NEUTRAL";
  modes_agree: boolean;
}

const POSITION_STYLES = {
  BULLISH: {
    bg: "bg-green-900/30",
    border: "border-green-600",
    text: "text-green-400",
    icon: "text-green-400",
  },
  BEARISH: {
    bg: "bg-red-900/30",
    border: "border-red-600",
    text: "text-red-400",
    icon: "text-red-400",
  },
  NEUTRAL: {
    bg: "bg-yellow-900/30",
    border: "border-yellow-600",
    text: "text-yellow-400",
    icon: "text-yellow-400",
  },
};

export function ConsensusComparisonChart({
  full_mode,
  reference_mode,
  modes_agree,
}: ConsensusComparisonChartProps) {
  const fullStyle = POSITION_STYLES[full_mode];
  const refStyle = POSITION_STYLES[reference_mode];

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <CardTitle className="text-white">AI Debate Consensus</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Agreement Indicator */}
          <div className="flex items-center justify-center gap-2 mb-6">
            {modes_agree ? (
              <>
                <CheckCircle className="w-6 h-6 text-green-400" />
                <span className="text-green-400 font-semibold">
                  Modes in Agreement
                </span>
              </>
            ) : (
              <>
                <AlertTriangle className="w-6 h-6 text-yellow-400" />
                <span className="text-yellow-400 font-semibold">
                  Divergent Opinions
                </span>
              </>
            )}
          </div>

          {/* Mode Badges */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Full Mode */}
            <div
              className={`p-6 rounded-lg border-2 ${fullStyle.bg} ${fullStyle.border} transition-all duration-300 hover:scale-105`}
            >
              <div className="text-sm text-gray-400 mb-2">Full Mode (365d)</div>
              <div className={`text-2xl font-bold ${fullStyle.text}`}>
                {full_mode}
              </div>
            </div>

            {/* Reference Mode */}
            <div
              className={`p-6 rounded-lg border-2 ${refStyle.bg} ${refStyle.border} transition-all duration-300 hover:scale-105`}
            >
              <div className="text-sm text-gray-400 mb-2">
                Reference Mode (90d)
              </div>
              <div className={`text-2xl font-bold ${refStyle.text}`}>
                {reference_mode}
              </div>
            </div>
          </div>

          {/* Interpretation */}
          <div className="mt-4 p-4 bg-[#161b22] rounded-lg border border-[#30363d]">
            <p className="text-sm text-gray-400">
              {modes_agree ? (
                <>
                  Both long-term and short-term perspectives align on a{" "}
                  <span className={fullStyle.text}>{full_mode.toLowerCase()}</span>{" "}
                  outlook, suggesting strong conviction.
                </>
              ) : (
                <>
                  Long-term outlook is{" "}
                  <span className={fullStyle.text}>{full_mode.toLowerCase()}</span>{" "}
                  while short-term is{" "}
                  <span className={refStyle.text}>
                    {reference_mode.toLowerCase()}
                  </span>
                  , indicating tactical divergence.
                </>
              )}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
