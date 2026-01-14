"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AlertTriangle } from "lucide-react";
import { useState } from "react";

export interface AssetRisk {
  ticker: string;
  weight: number; // Portfolio weight (0-1)
  riskScore: number; // Risk score (0-100)
  liquidityScore?: number; // Liquidity score (0-100)
  bubbleRisk?: number; // Bubble risk score (0-100)
}

interface RiskHeatmapProps {
  assets?: AssetRisk[];
}

const getRiskColor = (riskScore: number): string => {
  if (riskScore > 70) return "bg-red-600";
  if (riskScore > 50) return "bg-orange-500";
  if (riskScore > 30) return "bg-yellow-500";
  return "bg-green-600";
};

const getRiskLevel = (riskScore: number): string => {
  if (riskScore > 70) return "HIGH";
  if (riskScore > 50) return "MEDIUM";
  if (riskScore > 30) return "LOW";
  return "VERY LOW";
};

const getTextColor = (riskScore: number): string => {
  return riskScore > 30 ? "text-white" : "text-gray-900";
};

export function RiskHeatmap({ assets }: RiskHeatmapProps) {
  const [hoveredAsset, setHoveredAsset] = useState<AssetRisk | null>(null);

  // If no assets provided, show empty state
  if (!assets || assets.length === 0) {
    return (
      <Card className="bg-[#0d1117] border-[#30363d]">
        <CardHeader>
          <CardTitle className="text-white flex items-center gap-2">
            <AlertTriangle className="w-5 h-5" />
            Portfolio Risk Distribution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-[400px]">
            <p className="text-gray-400">No risk data available</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Sort by weight descending
  const sortedAssets = [...assets].sort((a, b) => b.weight - a.weight);

  // Calculate grid sizes based on weight
  const getGridSize = (weight: number): string => {
    // Map weight to grid cell count (1-4)
    if (weight > 0.3) return "col-span-2 row-span-2"; // 4 cells
    if (weight > 0.15) return "col-span-2 row-span-1"; // 2 cells
    if (weight > 0.08) return "col-span-1 row-span-1"; // 1 cell
    return "col-span-1 row-span-1"; // 1 cell
  };

  const getFontSize = (weight: number): string => {
    if (weight > 0.3) return "text-2xl";
    if (weight > 0.15) return "text-xl";
    if (weight > 0.08) return "text-base";
    return "text-sm";
  };

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <CardTitle className="text-white flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          Portfolio Risk Distribution
        </CardTitle>
        <p className="text-xs text-gray-400 mt-2">
          Size = Portfolio weight, Color = Risk level. Hover for details.
        </p>
      </CardHeader>
      <CardContent>
        {/* Treemap Grid */}
        <div className="grid grid-cols-4 gap-2 min-h-[400px]">
          {sortedAssets.map((asset) => (
            <div
              key={asset.ticker}
              className={`${getGridSize(asset.weight)} ${getRiskColor(asset.riskScore)} ${getTextColor(asset.riskScore)} rounded-lg p-4 cursor-pointer transition-all duration-200 hover:scale-105 hover:shadow-xl flex flex-col items-center justify-center`}
              onMouseEnter={() => setHoveredAsset(asset)}
              onMouseLeave={() => setHoveredAsset(null)}
            >
              <div className={`font-bold ${getFontSize(asset.weight)}`}>
                {asset.ticker}
              </div>
              <div className="text-xs opacity-90 mt-1">
                {(asset.weight * 100).toFixed(1)}%
              </div>
              <div className="text-xs opacity-75 mt-1">
                Risk: {asset.riskScore.toFixed(0)}
              </div>
            </div>
          ))}
        </div>

        {/* Hovered Asset Details */}
        {hoveredAsset && (
          <div className="mt-4 p-4 bg-[#161b22] rounded-lg border border-[#30363d]">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-lg font-bold text-white">{hoveredAsset.ticker}</h4>
              <Badge className={`${getRiskColor(hoveredAsset.riskScore)} text-white border-none`}>
                {getRiskLevel(hoveredAsset.riskScore)}
              </Badge>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-400">Portfolio Weight:</span>
                <p className="text-white font-semibold">
                  {(hoveredAsset.weight * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <span className="text-gray-400">Risk Score:</span>
                <p className="text-white font-semibold">
                  {hoveredAsset.riskScore.toFixed(1)} / 100
                </p>
              </div>
              {hoveredAsset.liquidityScore !== undefined && (
                <div>
                  <span className="text-gray-400">Liquidity:</span>
                  <p className="text-white font-semibold">
                    {hoveredAsset.liquidityScore.toFixed(1)} / 100
                  </p>
                </div>
              )}
              {hoveredAsset.bubbleRisk !== undefined && (
                <div>
                  <span className="text-gray-400">Bubble Risk:</span>
                  <p className="text-white font-semibold">
                    {hoveredAsset.bubbleRisk.toFixed(1)} / 100
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Risk Legend */}
        <div className="mt-4 flex items-center justify-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-green-600 rounded"></div>
            <span className="text-xs text-gray-400">Very Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-yellow-500 rounded"></div>
            <span className="text-xs text-gray-400">Low</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-orange-500 rounded"></div>
            <span className="text-xs text-gray-400">Medium</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-red-600 rounded"></div>
            <span className="text-xs text-gray-400">High</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
