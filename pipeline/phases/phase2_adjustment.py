#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 2: Risk Adjustment

Purpose:
    Risk adjustment based on sentiment, bubble, and microstructure analysis

Input:
    - market_data: Dict[str, Any]
    - result: EIMASResult

Output:
    - result: EIMASResult (adjusted risk_score)

Functions:
    - analyze_sentiment_bubble: Sentiment and bubble analysis
    - apply_extended_data_adjustment: Apply extended data adjustments
    - analyze_institutional_frameworks: Analyze institutional frameworks
    - run_adaptive_portfolio_phase: Run adaptive portfolio optimization

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

from typing import Any, Dict

from lib.bubble_framework import FiveStageBubbleFramework
from lib.fomc_analyzer import FOMCDotPlotAnalyzer
from lib.gap_analyzer import MarketModelGapAnalyzer
from pipeline.analyzers import analyze_bubble_risk, analyze_sentiment, run_adaptive_portfolio
from pipeline.schemas import BubbleRiskMetrics, EIMASResult


def analyze_sentiment_bubble(result: EIMASResult, market_data: Dict[str, Any], quick_mode: bool):
    """[Phase 2.3] Run bubble risk (full) and sentiment (always)."""
    if not quick_mode:
        try:
            bubble_res = analyze_bubble_risk(market_data)
            if bubble_res:
                result.bubble_risk = BubbleRiskMetrics(**bubble_res)
        except Exception as e:
            print(f"⚠️ Bubble Risk Error: {e}")

    try:
        result.sentiment_analysis = analyze_sentiment()
    except Exception as e:
        print(f"⚠️ Sentiment Error: {e}")


def apply_extended_data_adjustment(result: EIMASResult):
    """Apply risk score adjustments from extended data sentiment overlays."""
    if not result.extended_data:
        return

    ext = result.extended_data
    adjustment = 0.0
    details = []

    pcr = ext.get("put_call_ratio", {})
    if pcr.get("ratio", 0) > 0:
        ratio = pcr["ratio"]
        if ratio > 1.0:
            adjustment -= 5
            details.append(f"PCR={ratio:.2f} (Fear) -> -5")
        elif ratio < 0.7:
            adjustment += 5
            details.append(f"PCR={ratio:.2f} (Greed) -> +5")

    fng = ext.get("crypto_fng", {})
    if fng.get("value", 0) > 0:
        val = fng["value"]
        if val < 25:
            adjustment -= 3
            details.append(f"Crypto F&G={val} (Fear) -> -3")
        elif val > 75:
            adjustment += 5
            details.append(f"Crypto F&G={val} (Greed) -> +5")

    news = ext.get("news_sentiment", {})
    label = news.get("label", "")
    if label == "Bearish":
        adjustment -= 3
        details.append("News=Bearish -> -3")
    elif label == "Bullish":
        adjustment += 2
        details.append("News=Bullish -> +2")

    credit = ext.get("credit_spreads", {})
    interp = credit.get("interpretation", "")
    if interp == "Risk OFF":
        adjustment += 3
        details.append("Credit=Risk OFF -> +3")
    elif interp == "Risk ON":
        adjustment -= 2
        details.append("Credit=Risk ON -> -2")

    krw = ext.get("korea_risk", {})
    status = krw.get("status", "")
    if "Overheated" in status:
        adjustment += 5
        details.append("KRW=Overheated -> +5")
    elif "Volatile" in status:
        adjustment += 3
        details.append("KRW=Volatile -> +3")

    adjustment = max(-15, min(15, adjustment))
    if adjustment != 0:
        result.extended_data_adjustment = adjustment
        old_risk = result.risk_score
        result.risk_score = max(1.0, min(100, result.risk_score + adjustment))
        print(f"      ✓ Extended Data Adjustment: {adjustment:+.0f} ({old_risk:.1f} -> {result.risk_score:.1f})")
        if details:
            print(f"        Details: {', '.join(details)}")
        if result.risk_score < 5:
            warning = f"⚠️ Extremely Low Risk ({result.risk_score:.1f}/100) - Verify market conditions"
            result.warnings.append(warning)
            print(f"      {warning}")


def analyze_institutional_frameworks(result: EIMASResult, market_data: Dict[str, Any], quick_mode: bool):
    """Run institutional framework analyses (Bubble/Gap/FOMC)."""
    print("\n[Phase 2.Institutional] Running Institutional Frameworks...")

    try:
        bubble_fw = FiveStageBubbleFramework()
        bubble_result = bubble_fw.analyze(market_data, sector="tech")
        result.bubble_framework = bubble_result.to_dict()
        print(f"      ✓ 5-Stage Bubble: {bubble_result.stage} (Score: {bubble_result.total_score:.1f}/100)")
    except Exception as e:
        print(f"      ⚠️ 5-Stage Bubble Error: {e}")

    try:
        gap_analyzer = MarketModelGapAnalyzer()
        gap_result = gap_analyzer.analyze()
        result.gap_analysis = gap_result.to_dict()
        print(f"      ✓ Market-Model Gap: {gap_result.overall_signal} ({gap_result.opportunity[:40]}...)")
    except Exception as e:
        print(f"      ⚠️ Gap Analysis Error: {e}")

    if not quick_mode:
        try:
            fomc_analyzer = FOMCDotPlotAnalyzer()
            fomc_result = fomc_analyzer.analyze("2026")
            result.fomc_analysis = fomc_result.to_dict()
            print(
                f"      ✓ FOMC Analysis: {fomc_result.stance} "
                f"(Uncertainty: {fomc_result.policy_uncertainty_index:.0f}/100)"
            )
        except Exception as e:
            print(f"      ⚠️ FOMC Analysis Error: {e}")


def run_adaptive_portfolio_phase(result: EIMASResult, regime_res: Any, quick_mode: bool):
    """[Phase 2.4] Run adaptive portfolio only in full mode."""
    if not quick_mode:
        try:
            result.adaptive_portfolios = run_adaptive_portfolio(regime_res)
        except Exception as e:
            print(f"⚠️ Adaptive Portfolio Error: {e}")
