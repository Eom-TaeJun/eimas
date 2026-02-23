"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Brain,
  MessageSquare,
  TrendingUp,
  Shield,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Download,
  ArrowRight,
} from "lucide-react";
import { useState } from "react";
import type { EIMASAnalysis } from "@/lib/types";

interface ReasoningChainDisplayProps {
  data?: EIMASAnalysis;
}

export function ReasoningChainDisplay({ data }: ReasoningChainDisplayProps) {
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([0]));

  if (!data) {
    return (
      <Card className="bg-surface border-border p-8">
        <div className="text-center text-gray-400">
          <Brain className="w-16 h-16 mx-auto mb-4 text-purple-400 animate-pulse" />
          <h3 className="text-xl font-bold text-white mb-2">Loading AI Reasoning...</h3>
          <p>Fetching multi-agent analysis results</p>
        </div>
      </Card>
    );
  }

  const toggleStep = (index: number) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSteps(newExpanded);
  };

  const expandAll = () => {
    setExpandedSteps(new Set(data.reasoning_chain?.map((_, i) => i) || []));
  };

  const collapseAll = () => {
    setExpandedSteps(new Set([0]));
  };

  const exportReasoning = () => {
    const content = JSON.stringify(
      {
        timestamp: data.timestamp,
        final_recommendation: data.final_recommendation,
        confidence: data.confidence,
        reasoning_chain: data.reasoning_chain,
        debate_consensus: data.debate_consensus,
      },
      null,
      2
    );

    const blob = new Blob([content], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `eimas-reasoning-${data.timestamp}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getRecommendationColor = (rec: string) => {
    switch (rec) {
      case "BULLISH":
      case "BUY":
        return "text-green-400 bg-green-500/10 border-green-500/20";
      case "BEARISH":
      case "SELL":
        return "text-red-400 bg-red-500/10 border-red-500/20";
      default:
        return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <Card className="bg-surface border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-white flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                AI Reasoning Chain
                <Badge variant="outline" className="text-xs bg-purple-500/10 text-purple-400 border-purple-500/20">
                  {data.reasoning_chain?.length || 0} Steps
                </Badge>
              </CardTitle>
              <p className="text-sm text-gray-400 mt-1">
                Multi-agent decision-making process and consensus building
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={expandAll}
                className="bg-surface-card border-border text-gray-300 hover:bg-secondary hover:text-white"
              >
                <ChevronDown className="w-4 h-4 mr-1" />
                Expand All
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={collapseAll}
                className="bg-surface-card border-border text-gray-300 hover:bg-secondary hover:text-white"
              >
                <ChevronUp className="w-4 h-4 mr-1" />
                Collapse All
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={exportReasoning}
                className="bg-surface-card border-border text-gray-300 hover:bg-secondary hover:text-white"
              >
                <Download className="w-4 h-4 mr-1" />
                Export
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Final Consensus Summary */}
      <Card className={`bg-surface-card border-2 ${getRecommendationColor(data.final_recommendation)}`}>
        <CardContent className="pt-6">
          <div className="flex items-start gap-4">
            <CheckCircle className="w-8 h-8 text-green-400 flex-shrink-0 mt-1" />
            <div className="flex-1">
              <h3 className="text-xl font-bold text-white mb-2">Final Consensus</h3>
              <div className="flex items-center gap-3 mb-4">
                <Badge variant="outline" className={`text-lg px-4 py-1 ${getRecommendationColor(data.final_recommendation)}`}>
                  {data.final_recommendation}
                </Badge>
                <span className="text-gray-300">
                  Confidence: <span className="font-bold text-white">{(data.confidence * 100).toFixed(0)}%</span>
                </span>
              </div>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Full Mode Position:</span>
                  <Badge variant="outline" className={`ml-2 ${getRecommendationColor(data.full_mode_position)}`}>
                    {data.full_mode_position}
                  </Badge>
                </div>
                <div>
                  <span className="text-gray-400">Reference Mode Position:</span>
                  <Badge variant="outline" className={`ml-2 ${getRecommendationColor(data.reference_mode_position)}`}>
                    {data.reference_mode_position}
                  </Badge>
                </div>
              </div>
              {data.modes_agree !== undefined && (
                <div className="mt-3 flex items-center gap-2">
                  {data.modes_agree ? (
                    <>
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span className="text-green-400 text-sm">Modes in agreement</span>
                    </>
                  ) : (
                    <>
                      <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                      <span className="text-yellow-400 text-sm">Mode disagreement detected</span>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Reasoning Chain Steps */}
      {data.reasoning_chain && data.reasoning_chain.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-bold text-white">Reasoning Steps</h3>
          {data.reasoning_chain.map((step, index) => (
            <Card key={index} className="bg-surface border-border">
              <CardContent className="p-0">
                <button
                  onClick={() => toggleStep(index)}
                  className="w-full p-4 flex items-center justify-between hover:bg-surface-card transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="flex items-center justify-center w-8 h-8 rounded-full bg-purple-500/10 text-purple-400 font-bold">
                      {index + 1}
                    </div>
                    <div className="text-left">
                      <div className="font-semibold text-white">{step.agent}</div>
                      <div className="text-sm text-gray-400">{step.output_summary}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className="bg-blue-500/10 text-blue-400 border-blue-500/20">
                      {step.confidence}% confidence
                    </Badge>
                    {expandedSteps.has(index) ? (
                      <ChevronUp className="w-5 h-5 text-gray-400" />
                    ) : (
                      <ChevronDown className="w-5 h-5 text-gray-400" />
                    )}
                  </div>
                </button>

                {expandedSteps.has(index) && (
                  <div className="p-6 pt-0 border-t border-border">
                    <div className="space-y-4">
                      {step.key_factors && step.key_factors.length > 0 && (
                        <div>
                          <h4 className="text-sm font-semibold text-gray-300 mb-2">Key Factors:</h4>
                          <ul className="space-y-2">
                            {step.key_factors.map((factor, i) => (
                              <li key={i} className="flex items-start gap-2 text-sm text-gray-400">
                                <ArrowRight className="w-4 h-4 text-purple-400 flex-shrink-0 mt-0.5" />
                                <span>{factor}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Enhanced Debate Section */}
      {data.debate_consensus?.enhanced && (
        <Card className="bg-surface border-border">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <MessageSquare className="w-5 h-5 text-blue-400" />
              Multi-Agent Debate Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Interpretation */}
            {data.debate_consensus.enhanced.interpretation && (
              <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Economic School Interpretation</h4>
                <div className="bg-surface-card rounded-lg p-4 border border-border">
                  <div className="mb-3">
                    <span className="text-gray-400 text-sm">Recommended Action: </span>
                    <Badge
                      variant="outline"
                      className={getRecommendationColor(
                        data.debate_consensus.enhanced.interpretation.recommended_action
                      )}
                    >
                      {data.debate_consensus.enhanced.interpretation.recommended_action}
                    </Badge>
                  </div>

                  {data.debate_consensus.enhanced.interpretation.school_interpretations && (
                    <div className="space-y-3 mt-4">
                      {data.debate_consensus.enhanced.interpretation.school_interpretations.map((school, i) => (
                        <div key={i} className="bg-surface rounded-lg p-3 border border-border">
                          <div className="flex items-center justify-between mb-2">
                            <span className="font-semibold text-white">{school.school} School</span>
                            <Badge variant="outline" className={getRecommendationColor(school.stance)}>
                              {school.stance}
                            </Badge>
                          </div>
                          <ul className="space-y-1">
                            {school.reasoning.map((reason, j) => (
                              <li key={j} className="text-sm text-gray-400 flex items-start gap-2">
                                <span className="text-purple-400">•</span>
                                <span>{reason}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  )}

                  {data.debate_consensus.enhanced.interpretation.consensus_points && (
                    <div className="mt-4">
                      <div className="text-sm font-semibold text-gray-300 mb-2">Consensus Points:</div>
                      <ul className="space-y-1">
                        {data.debate_consensus.enhanced.interpretation.consensus_points.map((point, i) => (
                          <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                            <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                            <span>{point}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {data.debate_consensus.enhanced.interpretation.divergence_points && (
                    <div className="mt-4">
                      <div className="text-sm font-semibold text-gray-300 mb-2">Divergence Points:</div>
                      <ul className="space-y-1">
                        {data.debate_consensus.enhanced.interpretation.divergence_points.map((point, i) => (
                          <li key={i} className="text-sm text-gray-400 flex items-start gap-2">
                            <AlertTriangle className="w-4 h-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                            <span>{point}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Methodology */}
            {data.debate_consensus.enhanced.methodology && (
              <div>
                <h4 className="text-sm font-semibold text-gray-300 mb-3">Selected Methodology</h4>
                <div className="bg-surface-card rounded-lg p-4 border border-border">
                  <div className="font-semibold text-white mb-2">
                    {data.debate_consensus.enhanced.methodology.selected_methodology}
                  </div>
                  <div className="text-sm text-gray-400">
                    {data.debate_consensus.enhanced.methodology.rationale}
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Verification Results */}
      {data.debate_consensus?.verification && (
        <Card className="bg-surface border-border">
          <CardHeader>
            <CardTitle className="text-white flex items-center gap-2">
              <Shield className="w-5 h-5 text-green-400" />
              Verification & Quality Check
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-surface-card rounded-lg p-4 border border-border">
                <div className="text-sm text-gray-400 mb-1">Overall Score</div>
                <div className="text-2xl font-bold text-white">
                  {data.debate_consensus.verification.overall_score?.toFixed(0)}/100
                </div>
              </div>
              <div className="bg-surface-card rounded-lg p-4 border border-border">
                <div className="text-sm text-gray-400 mb-1">Status</div>
                <Badge
                  variant="outline"
                  className={
                    data.debate_consensus.verification.passed
                      ? "bg-green-500/10 text-green-400 border-green-500/20"
                      : "bg-red-500/10 text-red-400 border-red-500/20"
                  }
                >
                  {data.debate_consensus.verification.passed ? "✅ Passed" : "❌ Failed"}
                </Badge>
              </div>
              <div className="bg-surface-card rounded-lg p-4 border border-border">
                <div className="text-sm text-gray-400 mb-1">Hallucination Risk</div>
                <div className="text-2xl font-bold text-white">
                  {(data.debate_consensus.verification.hallucination_risk * 100).toFixed(0)}%
                </div>
              </div>
            </div>

            {data.debate_consensus.verification.warnings &&
              data.debate_consensus.verification.warnings.length > 0 && (
                <div className="mt-4">
                  <div className="text-sm font-semibold text-gray-300 mb-2">Warnings:</div>
                  <ul className="space-y-1">
                    {data.debate_consensus.verification.warnings.map((warning, i) => (
                      <li key={i} className="text-sm text-yellow-400 flex items-start gap-2">
                        <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                        <span>{warning}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
