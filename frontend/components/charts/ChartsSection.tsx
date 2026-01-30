"use client";

import useSWR from "swr";
import { fetchLatestAnalysis } from "@/lib/api";
import type { EIMASAnalysis } from "@/lib/types";
import { PortfolioChart } from "./PortfolioChart";
import { GMMProbabilityChart } from "./GMMProbabilityChart";
import { RiskBreakdownChart } from "./RiskBreakdownChart";
import { ConsensusComparisonChart } from "./ConsensusComparisonChart";
import { SystemStatusDashboard } from "./SystemStatusDashboard";
import { CorrelationHeatmap } from "./CorrelationHeatmap";
import { RiskHeatmap, AssetRisk } from "./RiskHeatmap";
import { ArkAnalysisDashboard } from "./ArkAnalysisDashboard";
import { DebateSchoolCards } from "./DebateSchoolCards";
import { MarketSentimentGauge } from "./MarketSentimentGauge";
import { SignalsPieChart } from "./SignalsPieChart";
import { VolumeAnomalyScatter } from "./VolumeAnomalyScatter";
import { CryptoRiskGauge } from "./CryptoRiskGauge";
import { MarketRegimeRadar } from "./MarketRegimeRadar";

export function ChartsSection() {
  // Fetch latest EIMAS analysis every 5 seconds (same as MetricsGrid)
  const { data, error } = useSWR<EIMASAnalysis>(
    "latest-analysis",
    fetchLatestAnalysis,
    {
      refreshInterval: 5000,
      dedupingInterval: 1000,
    }
  );

  if (error) {
    return (
      <section className="mt-8">
        <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-4">
          <p className="text-red-400">Failed to load chart data</p>
        </div>
      </section>
    );
  }

  if (!data) {
    return (
      <section className="mt-8">
        <h2 className="text-2xl font-bold text-white mb-6">Analytics Charts</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[...Array(4)].map((_, i) => (
            <div
              key={i}
              className="bg-[#0d1117] border border-[#30363d] rounded-lg p-8"
            >
              <div className="animate-pulse space-y-2">
                <div className="h-8 bg-gray-700 rounded"></div>
                <div className="h-4 bg-gray-700 rounded w-2/3"></div>
              </div>
            </div>
          ))}
        </div>
      </section>
    );
  }

  // Check if we have the required data
  const hasPortfolioData =
    data.portfolio_weights && Object.keys(data.portfolio_weights).length > 0;
  const hasRiskData =
    data.base_risk_score !== undefined &&
    data.microstructure_adjustment !== undefined &&
    data.bubble_risk_adjustment !== undefined;
  const hasDebateData =
    data.full_mode_position &&
    data.reference_mode_position &&
    data.modes_agree !== undefined;

  // GMM probabilities
  const gmmProbabilities = data.regime?.gmm_probabilities || {
    Bull: 0.33,
    Neutral: 0.34,
    Bear: 0.33,
  };

  // Prepare correlation data
  const correlationTickers = data.correlation_tickers || undefined;
  const correlationMatrix = data.correlation_matrix || undefined;

  // Prepare risk heatmap data
  const riskAssets: AssetRisk[] | undefined = hasPortfolioData
    ? Object.entries(data.portfolio_weights).map(([ticker, weight]) => {
      const liquidityScore = data.market_quality?.liquidity_scores?.[ticker];
      const bubbleRiskTicker = data.bubble_risk?.risk_tickers?.find(
        (r: any) => r.ticker === ticker
      );
      const bubbleRisk = bubbleRiskTicker?.bubble_score || 0;

      // Calculate composite risk score
      const baseRisk = data.base_risk_score || 50;
      const microAdj = data.microstructure_adjustment || 0;
      const bubbleAdj = data.bubble_risk_adjustment || 0;

      // Normalize to 0-100 scale per asset
      let riskScore = baseRisk;
      if (liquidityScore !== undefined) {
        riskScore += (50 - liquidityScore) / 5; // Lower liquidity = higher risk
      }
      riskScore += bubbleRisk;
      riskScore = Math.max(0, Math.min(100, riskScore)); // Clamp to 0-100

      return {
        ticker,
        weight,
        riskScore,
        liquidityScore,
        bubbleRisk,
      };
    })
    : undefined;

  return (
    <section className="mt-8 space-y-6">
      <h2 className="text-2xl font-bold text-white mb-6">Analytics Charts</h2>

      {/* Row 0: System Status Dashboard */}
      <SystemStatusDashboard />

      {/* Row 0.5: New Visualizations - Sentiment & HFT */}
      {data.sentiment_analysis && data.hft_microstructure && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <MarketSentimentGauge
            sentiment={data.sentiment_analysis}
            hft={data.hft_microstructure}
          />
          {data.debate_consensus?.enhanced && (
            <DebateSchoolCards data={data.debate_consensus.enhanced} />
          )}
        </div>
      )}

      {/* Row 0.7: ARK Analysis */}
      {data.ark_analysis && (
        <ArkAnalysisDashboard data={data.ark_analysis} />
      )}

      {/* Row 1: Portfolio + GMM Probabilities */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {hasPortfolioData ? (
          <div className="lg:col-span-2">
            <PortfolioChart weights={data.portfolio_weights} />
          </div>
        ) : (
          <div className="lg:col-span-2 bg-[#0d1117] border border-[#30363d] rounded-lg p-8 flex items-center justify-center">
            <p className="text-gray-400">No portfolio data available</p>
          </div>
        )}

        <div className="lg:col-span-1">
          <GMMProbabilityChart probabilities={gmmProbabilities} />
        </div>
      </div>

      {/* Row 2: Risk Breakdown + Consensus Comparison */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {hasRiskData ? (
          <RiskBreakdownChart
            base_risk={data.base_risk_score}
            microstructure_adj={data.microstructure_adjustment}
            bubble_adj={data.bubble_risk_adjustment}
            final_risk={data.risk_score}
          />
        ) : (
          <div className="bg-[#0d1117] border border-[#30363d] rounded-lg p-8 flex items-center justify-center">
            <p className="text-gray-400">No risk data available</p>
          </div>
        )}

        {hasDebateData ? (
          <ConsensusComparisonChart
            full_mode={
              data.full_mode_position as "BULLISH" | "BEARISH" | "NEUTRAL"
            }
            reference_mode={
              data.reference_mode_position as "BULLISH" | "BEARISH" | "NEUTRAL"
            }
            modes_agree={data.modes_agree}
          />
        ) : (
          <div className="bg-[#0d1117] border border-[#30363d] rounded-lg p-8 flex items-center justify-center">
            <p className="text-gray-400">No debate data available</p>
          </div>
        )}
      </div>

      {/* Row 3: Correlation Heatmap + Risk Heatmap */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <CorrelationHeatmap
          tickers={correlationTickers}
          correlationMatrix={correlationMatrix}
        />
        <RiskHeatmap assets={riskAssets} />
      </div>

      {/* Row 6: Volume Anomalies & Risk Gauges */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <VolumeAnomalyScatter />
        </div>
        <div className="space-y-6">
          <CryptoRiskGauge />
          <div className="h-[250px]">
            <SignalsPieChart />
          </div>
        </div>
      </div>

      {/* Row 7: Market Regime Radar */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MarketRegimeRadar />
        {/* Placeholder for future expansion */}
      </div>
    </section>
  );
}
