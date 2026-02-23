"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Info, ZoomIn, Download } from "lucide-react";
import { Button } from "@/components/ui/button";

interface EnhancedCorrelationHeatmapProps {
  tickers?: string[];
  correlationMatrix?: number[][];
}

// Generate mock correlation data with clustering
const generateMockCorrelation = (tickers: string[]): number[][] => {
  const n = tickers.length;
  const matrix: number[][] = [];

  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 1.0;
      } else if (i < j) {
        // Group similar assets with higher correlation
        const isSimilar =
          (tickers[i].startsWith("X") && tickers[j].startsWith("X")) ||
          (["SPY", "QQQ", "DIA", "IWM"].includes(tickers[i]) &&
            ["SPY", "QQQ", "DIA", "IWM"].includes(tickers[j]));

        matrix[i][j] = isSimilar
          ? 0.6 + Math.random() * 0.3
          : -0.3 + Math.random() * 0.6;
      } else {
        matrix[i][j] = matrix[j][i];
      }
    }
  }

  return matrix;
};

const getCorrelationColor = (value: number, hovered: boolean): string => {
  const intensity = hovered ? "600" : "500";
  if (value > 0.7) return `bg-red-${intensity}`;
  if (value > 0.4) return `bg-red-${parseInt(intensity) - 100}`;
  if (value > 0.1) return "bg-red-300";
  if (value > -0.1) return "bg-gray-300";
  if (value > -0.4) return "bg-blue-300";
  if (value > -0.7) return `bg-blue-${parseInt(intensity) - 100}`;
  return `bg-blue-${intensity}`;
};

const getTextColor = (value: number): string => {
  return Math.abs(value) > 0.5 ? "text-white" : "text-gray-800";
};

const getCorrelationStrength = (value: number): string => {
  const abs = Math.abs(value);
  if (abs > 0.8) return "Very Strong";
  if (abs > 0.6) return "Strong";
  if (abs > 0.4) return "Moderate";
  if (abs > 0.2) return "Weak";
  return "Very Weak";
};

export function EnhancedCorrelationHeatmap({
  tickers = ["SPY", "QQQ", "TLT", "GLD", "HYG", "XLF", "XLE", "XLV", "IWM", "DIA"],
  correlationMatrix,
}: EnhancedCorrelationHeatmapProps) {
  const [hoveredCell, setHoveredCell] = useState<{ i: number; j: number } | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ i: number; j: number } | null>(null);
  const [filterThreshold, setFilterThreshold] = useState<number>(0);

  const matrix = correlationMatrix || generateMockCorrelation(tickers);

  // Calculate correlation statistics
  const getAllCorrelations = () => {
    const correlations: number[] = [];
    for (let i = 0; i < matrix.length; i++) {
      for (let j = i + 1; j < matrix.length; j++) {
        correlations.push(matrix[i][j]);
      }
    }
    return correlations;
  };

  const allCorrelations = getAllCorrelations();
  const avgCorrelation = allCorrelations.reduce((a, b) => a + b, 0) / allCorrelations.length;
  const maxCorrelation = Math.max(...allCorrelations);
  const minCorrelation = Math.min(...allCorrelations);
  const highCorrelations = allCorrelations.filter((c) => Math.abs(c) > 0.7).length;

  const handleCellClick = (i: number, j: number) => {
    if (i === j) return; // Don't select diagonal
    setSelectedCell(selectedCell?.i === i && selectedCell?.j === j ? null : { i, j });
  };

  const exportData = () => {
    const csv = [
      ["", ...tickers].join(","),
      ...matrix.map((row, i) => [tickers[i], ...row.map((v) => v.toFixed(3))].join(",")),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "correlation_matrix.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const displayedTickers = tickers.slice(0, 12); // Limit for readability
  const displayedMatrix = matrix.slice(0, 12).map((row) => row.slice(0, 12));

  return (
    <Card className="bg-surface border-border">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-white flex items-center gap-2">
              Asset Correlation Matrix
              <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                Interactive
              </Badge>
            </CardTitle>
            <p className="text-xs text-gray-400 mt-1">
              Click cells for details • Hover for quick view • Red = positive, Blue = negative
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={exportData}
              className="bg-surface-card border-border text-gray-300 hover:bg-secondary hover:text-white"
            >
              <Download className="w-4 h-4 mr-1" />
              Export CSV
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Statistics Grid */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400 mb-1">Avg Correlation</div>
            <div className="text-lg font-bold text-white">{avgCorrelation.toFixed(3)}</div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400 mb-1">Max (Positive)</div>
            <div className="text-lg font-bold text-red-400">{maxCorrelation.toFixed(3)}</div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400 mb-1">Min (Negative)</div>
            <div className="text-lg font-bold text-blue-400">{minCorrelation.toFixed(3)}</div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400 mb-1">Strong Pairs</div>
            <div className="text-lg font-bold text-white">
              {highCorrelations} <span className="text-xs text-gray-400">(&gt;0.7)</span>
            </div>
          </div>
        </div>

        {/* Filter Controls */}
        <div className="mb-4 flex items-center gap-4">
          <label className="text-sm text-gray-400">
            Filter: |correlation| &gt;
            <input
              type="range"
              min="0"
              max="0.9"
              step="0.1"
              value={filterThreshold}
              onChange={(e) => setFilterThreshold(parseFloat(e.target.value))}
              className="ml-2 mr-2"
            />
            <span className="text-white font-mono">{filterThreshold.toFixed(1)}</span>
          </label>
        </div>

        {/* Heatmap */}
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            {/* Header row */}
            <div className="flex">
              <div className="w-16"></div>
              {displayedTickers.map((ticker, i) => (
                <div
                  key={`header-${i}`}
                  className="w-16 h-8 flex items-center justify-center text-xs font-medium text-gray-300"
                >
                  {ticker}
                </div>
              ))}
            </div>

            {/* Matrix rows */}
            {displayedTickers.map((ticker, i) => (
              <div key={`row-${i}`} className="flex">
                <div className="w-16 h-16 flex items-center justify-center text-xs font-medium text-gray-300 border-r border-border">
                  {ticker}
                </div>

                {displayedTickers.map((_, j) => {
                  const value = displayedMatrix[i][j];
                  const isHovered = hoveredCell?.i === i && hoveredCell?.j === j;
                  const isSelected = selectedCell?.i === i && selectedCell?.j === j;
                  const isFiltered = Math.abs(value) < filterThreshold && i !== j;

                  return (
                    <div
                      key={`cell-${i}-${j}`}
                      className={`w-16 h-16 flex items-center justify-center text-xs font-bold border border-border transition-all duration-200 ${
                        isFiltered
                          ? "bg-gray-800 opacity-30"
                          : getCorrelationColor(value, isHovered || isSelected)
                      } ${getTextColor(value)} ${
                        i !== j ? "cursor-pointer hover:scale-110 hover:z-10 hover:shadow-lg" : ""
                      } ${isSelected ? "ring-2 ring-yellow-400 scale-110 z-20" : ""}`}
                      onMouseEnter={() => !isFiltered && setHoveredCell({ i, j })}
                      onMouseLeave={() => setHoveredCell(null)}
                      onClick={() => !isFiltered && handleCellClick(i, j)}
                    >
                      {i === j ? "1.0" : value.toFixed(2)}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        {/* Selected Cell Details */}
        {selectedCell && (
          <div className="mt-6 p-4 bg-surface-card rounded-lg border-2 border-secondary">
            <div className="flex items-start justify-between mb-3">
              <div>
                <h4 className="text-white font-bold text-lg mb-1">
                  {displayedTickers[selectedCell.i]} ↔ {displayedTickers[selectedCell.j]}
                </h4>
                <Badge
                  variant="outline"
                  className={
                    displayedMatrix[selectedCell.i][selectedCell.j] > 0
                      ? "bg-red-500/10 text-red-400 border-red-500/20"
                      : "bg-blue-500/10 text-blue-400 border-blue-500/20"
                  }
                >
                  {getCorrelationStrength(displayedMatrix[selectedCell.i][selectedCell.j])} Correlation
                </Badge>
              </div>
              <div className="text-3xl font-bold text-white">
                {displayedMatrix[selectedCell.i][selectedCell.j].toFixed(3)}
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-gray-400 mb-1">Interpretation:</div>
                <p className="text-white">
                  {Math.abs(displayedMatrix[selectedCell.i][selectedCell.j]) > 0.7
                    ? `Strong ${
                        displayedMatrix[selectedCell.i][selectedCell.j] > 0 ? "positive" : "negative"
                      } relationship. These assets tend to move ${
                        displayedMatrix[selectedCell.i][selectedCell.j] > 0 ? "together" : "inversely"
                      }.`
                    : Math.abs(displayedMatrix[selectedCell.i][selectedCell.j]) > 0.4
                    ? `Moderate ${
                        displayedMatrix[selectedCell.i][selectedCell.j] > 0 ? "positive" : "negative"
                      } relationship. Some connection in price movements.`
                    : "Weak relationship. Assets move relatively independently."}
                </p>
              </div>
              <div>
                <div className="text-gray-400 mb-1">Portfolio Impact:</div>
                <p className="text-white">
                  {Math.abs(displayedMatrix[selectedCell.i][selectedCell.j]) > 0.7
                    ? displayedMatrix[selectedCell.i][selectedCell.j] > 0
                      ? "⚠️ High correlation increases concentration risk. Consider diversification."
                      : "✅ Negative correlation provides natural hedging benefits."
                    : "✅ Low correlation improves portfolio diversification."}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Hover Tooltip */}
        {hoveredCell && !selectedCell && (
          <div className="mt-4 p-3 bg-surface-card rounded-lg border border-border">
            <p className="text-sm text-white">
              <span className="font-bold">{displayedTickers[hoveredCell.i]}</span>
              {" ↔ "}
              <span className="font-bold">{displayedTickers[hoveredCell.j]}</span>
              {": "}
              <span
                className={
                  displayedMatrix[hoveredCell.i][hoveredCell.j] > 0 ? "text-red-400" : "text-blue-400"
                }
              >
                {displayedMatrix[hoveredCell.i][hoveredCell.j].toFixed(3)}
              </span>
              <span className="text-gray-400 ml-2">
                ({getCorrelationStrength(displayedMatrix[hoveredCell.i][hoveredCell.j])})
              </span>
            </p>
          </div>
        )}

        {/* Color Legend */}
        <div className="mt-6 flex items-center justify-center gap-2">
          <span className="text-xs text-gray-400">Strong Negative</span>
          <div className="flex gap-1">
            <div className="w-8 h-4 bg-blue-600"></div>
            <div className="w-8 h-4 bg-blue-400"></div>
            <div className="w-8 h-4 bg-blue-300"></div>
            <div className="w-8 h-4 bg-gray-300"></div>
            <div className="w-8 h-4 bg-red-300"></div>
            <div className="w-8 h-4 bg-red-400"></div>
            <div className="w-8 h-4 bg-red-600"></div>
          </div>
          <span className="text-xs text-gray-400">Strong Positive</span>
        </div>
      </CardContent>
    </Card>
  );
}
