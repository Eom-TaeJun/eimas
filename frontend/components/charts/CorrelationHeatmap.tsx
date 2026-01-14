"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useState } from "react";

interface CorrelationHeatmapProps {
  tickers?: string[];
  correlationMatrix?: number[][];
}

// Generate mock correlation data (will be replaced with real data from backend)
const generateMockCorrelation = (tickers: string[]): number[][] => {
  const n = tickers.length;
  const matrix: number[][] = [];

  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 1.0; // Perfect self-correlation
      } else if (i < j) {
        // Random correlation between -1 and 1
        // Group similar assets with higher correlation
        const isSimilar =
          (tickers[i].startsWith('X') && tickers[j].startsWith('X')) || // Sector ETFs
          (['SPY', 'QQQ', 'DIA', 'IWM'].includes(tickers[i]) &&
           ['SPY', 'QQQ', 'DIA', 'IWM'].includes(tickers[j])); // Equity indices

        matrix[i][j] = isSimilar
          ? 0.6 + Math.random() * 0.3  // High correlation
          : -0.3 + Math.random() * 0.6; // Lower correlation
      } else {
        matrix[i][j] = matrix[j][i]; // Symmetric
      }
    }
  }

  return matrix;
};

const getCorrelationColor = (value: number): string => {
  // Blue (negative) -> White (zero) -> Red (positive)
  if (value > 0.7) return "bg-red-700";
  if (value > 0.4) return "bg-red-500";
  if (value > 0.1) return "bg-red-300";
  if (value > -0.1) return "bg-gray-300";
  if (value > -0.4) return "bg-blue-300";
  if (value > -0.7) return "bg-blue-500";
  return "bg-blue-700";
};

const getTextColor = (value: number): string => {
  return Math.abs(value) > 0.5 ? "text-white" : "text-gray-800";
};

export function CorrelationHeatmap({
  tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'HYG', 'XLF', 'XLE', 'XLV'],
  correlationMatrix
}: CorrelationHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number } | null>(null);

  // Use provided matrix or generate mock data
  const matrix = correlationMatrix || generateMockCorrelation(tickers);

  return (
    <Card className="bg-[#0d1117] border-[#30363d]">
      <CardHeader>
        <CardTitle className="text-white">Asset Correlation Heatmap</CardTitle>
        <p className="text-xs text-gray-400 mt-2">
          Hover over cells to see correlation values. Red = positive, Blue = negative.
        </p>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Header row with ticker names */}
            <div className="flex">
              <div className="w-16"></div> {/* Empty corner */}
              {tickers.map((ticker, i) => (
                <div
                  key={`header-${i}`}
                  className="w-16 h-8 flex items-center justify-center text-xs font-medium text-gray-300"
                >
                  {ticker}
                </div>
              ))}
            </div>

            {/* Matrix rows */}
            {tickers.map((ticker, i) => (
              <div key={`row-${i}`} className="flex">
                {/* Row label */}
                <div className="w-16 h-16 flex items-center justify-center text-xs font-medium text-gray-300 border-r border-[#30363d]">
                  {ticker}
                </div>

                {/* Correlation cells */}
                {tickers.map((_, j) => (
                  <div
                    key={`cell-${i}-${j}`}
                    className={`w-16 h-16 flex items-center justify-center text-xs font-bold border border-[#30363d] cursor-pointer transition-all duration-200 ${getCorrelationColor(matrix[i][j])} ${getTextColor(matrix[i][j])} hover:scale-110 hover:z-10 hover:shadow-lg`}
                    onMouseEnter={() => setHoveredCell({ i, j })}
                    onMouseLeave={() => setHoveredCell(null)}
                  >
                    {i === j ? "1.0" : matrix[i][j].toFixed(2)}
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* Tooltip showing hovered correlation */}
        {hoveredCell && (
          <div className="mt-4 p-3 bg-[#161b22] rounded-lg border border-[#30363d]">
            <p className="text-sm text-white">
              <span className="font-bold">{tickers[hoveredCell.i]}</span>
              {" ↔ "}
              <span className="font-bold">{tickers[hoveredCell.j]}</span>
              {": "}
              <span className={matrix[hoveredCell.i][hoveredCell.j] > 0 ? "text-red-400" : "text-blue-400"}>
                {matrix[hoveredCell.i][hoveredCell.j].toFixed(3)}
              </span>
            </p>
            <p className="text-xs text-gray-400 mt-1">
              {Math.abs(matrix[hoveredCell.i][hoveredCell.j]) > 0.7
                ? "Strong correlation"
                : Math.abs(matrix[hoveredCell.i][hoveredCell.j]) > 0.4
                ? "Moderate correlation"
                : "Weak correlation"}
            </p>
          </div>
        )}

        {/* Color legend */}
        <div className="mt-4 flex items-center justify-center gap-2">
          <span className="text-xs text-gray-400">Negative</span>
          <div className="flex gap-1">
            <div className="w-8 h-4 bg-blue-700"></div>
            <div className="w-8 h-4 bg-blue-500"></div>
            <div className="w-8 h-4 bg-blue-300"></div>
            <div className="w-8 h-4 bg-gray-300"></div>
            <div className="w-8 h-4 bg-red-300"></div>
            <div className="w-8 h-4 bg-red-500"></div>
            <div className="w-8 h-4 bg-red-700"></div>
          </div>
          <span className="text-xs text-gray-400">Positive</span>
        </div>
      </CardContent>
    </Card>
  );
}
