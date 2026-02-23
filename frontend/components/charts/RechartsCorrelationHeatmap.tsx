"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { Download } from "lucide-react";
import { Button } from "@/components/ui/button";

interface RechartsCorrelationHeatmapProps {
  tickers?: string[];
  correlationMatrix?: number[][];
}

// Convert 2D matrix to scatter plot data format
const matrixToScatterData = (
  matrix: number[][],
  tickers: string[]
): Array<{ x: number; y: number; z: number; value: number; pair: string }> => {
  const data: Array<{ x: number; y: number; z: number; value: number; pair: string }> = [];

  for (let i = 0; i < matrix.length; i++) {
    for (let j = 0; j < matrix[i].length; j++) {
      data.push({
        x: j,
        y: matrix.length - 1 - i, // Flip y-axis for top-to-bottom rendering
        z: Math.abs(matrix[i][j]) * 1000, // Size by absolute value
        value: matrix[i][j],
        pair: `${tickers[i]} ↔ ${tickers[j]}`,
      });
    }
  }

  return data;
};

// Get color based on correlation value
const getCorrelationColor = (value: number): string => {
  if (value > 0.7) return "#dc2626"; // red-600
  if (value > 0.4) return "#f87171"; // red-400
  if (value > 0.1) return "#fca5a5"; // red-300
  if (value > -0.1) return "#9ca3af"; // gray-400
  if (value > -0.4) return "#93c5fd"; // blue-300
  if (value > -0.7) return "#60a5fa"; // blue-400
  return "#2563eb"; // blue-600
};

const getCorrelationStrength = (value: number): string => {
  const abs = Math.abs(value);
  if (abs > 0.8) return "Very Strong";
  if (abs > 0.6) return "Strong";
  if (abs > 0.4) return "Moderate";
  if (abs > 0.2) return "Weak";
  return "Very Weak";
};

export function RechartsCorrelationHeatmap({
  tickers = ["SPY", "QQQ", "TLT", "GLD", "HYG", "XLF"],
  correlationMatrix,
}: RechartsCorrelationHeatmapProps) {
  const [selectedCell, setSelectedCell] = useState<{
    i: number;
    j: number;
    value: number;
  } | null>(null);

  // Generate mock data if no matrix provided
  const matrix =
    correlationMatrix ||
    tickers.map((_, i) =>
      tickers.map((_, j) => {
        if (i === j) return 1.0;
        const base = Math.random() * 1.4 - 0.7;
        return Math.max(-1, Math.min(1, base));
      })
    );

  const scatterData = matrixToScatterData(matrix, tickers);

  // Calculate statistics
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
  const avgCorrelation =
    allCorrelations.reduce((a, b) => a + b, 0) / allCorrelations.length;
  const maxCorrelation = Math.max(...allCorrelations);
  const minCorrelation = Math.min(...allCorrelations);
  const strongCorrelations = allCorrelations.filter((c) => Math.abs(c) > 0.7).length;

  const exportData = () => {
    const csv = [
      ["", ...tickers].join(","),
      ...matrix.map((row, i) => [tickers[i], ...row.map((v) => v.toFixed(3))].join(",")),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `correlation_matrix_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload[0]) {
      const data = payload[0].payload;
      return (
        <div className="bg-surface-card border border-border p-3 rounded-md shadow-xl">
          <p className="text-xs text-gray-400 mb-1">{data.pair}</p>
          <p className="text-lg font-bold text-white">{data.value.toFixed(3)}</p>
          <p className="text-xs text-gray-400 mt-1">{getCorrelationStrength(data.value)}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="bg-surface border-border">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-white flex items-center gap-2">
              Asset Correlation Matrix
              <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                Recharts
              </Badge>
            </CardTitle>
            <p className="text-xs text-gray-400 mt-1">
              Interactive heatmap • Hover for details • Red = positive, Blue = negative
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={exportData}
            className="bg-surface-card border-border text-gray-300 hover:bg-secondary hover:text-white"
          >
            <Download className="w-4 h-4 mr-1" />
            CSV
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Statistics */}
        <div className="grid grid-cols-4 gap-3 mb-6">
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400">Avg</div>
            <div className="text-xl font-bold text-white">{avgCorrelation.toFixed(3)}</div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400">Max</div>
            <div className="text-xl font-bold text-green-400">{maxCorrelation.toFixed(3)}</div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400">Min</div>
            <div className="text-xl font-bold text-blue-400">{minCorrelation.toFixed(3)}</div>
          </div>
          <div className="bg-surface-card rounded-lg p-3 border border-border">
            <div className="text-xs text-gray-400">Strong (|r| &gt; 0.7)</div>
            <div className="text-xl font-bold text-orange-400">{strongCorrelations}</div>
          </div>
        </div>

        {/* Heatmap */}
        <div className="bg-surface-card rounded-lg p-4 border border-border">
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart
              margin={{ top: 20, right: 20, bottom: 60, left: 60 }}
            >
              <XAxis
                type="number"
                dataKey="x"
                domain={[-0.5, tickers.length - 0.5]}
                ticks={tickers.map((_, i) => i)}
                tickFormatter={(value) => tickers[value] || ""}
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                interval={0}
                angle={-45}
                textAnchor="end"
                height={80}
              />
              <YAxis
                type="number"
                dataKey="y"
                domain={[-0.5, tickers.length - 0.5]}
                ticks={tickers.map((_, i) => tickers.length - 1 - i)}
                tickFormatter={(value) => tickers[tickers.length - 1 - value] || ""}
                tick={{ fill: "#9ca3af", fontSize: 12 }}
                interval={0}
                width={60}
              />
              <ZAxis type="number" dataKey="z" range={[200, 800]} />
              <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3" }} />
              <Scatter data={scatterData} shape="square">
                {scatterData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={getCorrelationColor(entry.value)} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Color Legend */}
        <div className="mt-6 flex items-center justify-center gap-2">
          <span className="text-xs text-gray-400">Strong Negative</span>
          <div className="flex gap-1">
            <div className="w-10 h-5 rounded" style={{ backgroundColor: "#2563eb" }}></div>
            <div className="w-10 h-5 rounded" style={{ backgroundColor: "#60a5fa" }}></div>
            <div className="w-10 h-5 rounded" style={{ backgroundColor: "#93c5fd" }}></div>
            <div className="w-10 h-5 rounded" style={{ backgroundColor: "#9ca3af" }}></div>
            <div className="w-10 h-5 rounded" style={{ backgroundColor: "#fca5a5" }}></div>
            <div className="w-10 h-5 rounded" style={{ backgroundColor: "#f87171" }}></div>
            <div className="w-10 h-5 rounded" style={{ backgroundColor: "#dc2626" }}></div>
          </div>
          <span className="text-xs text-gray-400">Strong Positive</span>
        </div>

        {/* Selected Cell Info */}
        {selectedCell && (
          <div className="mt-4 p-4 bg-blue-500/10 rounded-lg border border-blue-500/20">
            <div className="text-sm text-blue-400 font-medium">Selected Pair</div>
            <div className="text-white mt-1">
              {tickers[selectedCell.i]} ↔ {tickers[selectedCell.j]}
            </div>
            <div className="text-2xl font-bold text-white mt-2">
              {selectedCell.value.toFixed(4)}
            </div>
            <div className="text-xs text-gray-400 mt-1">
              {getCorrelationStrength(selectedCell.value)} Correlation
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
