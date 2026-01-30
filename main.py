#!/usr/bin/env python3
"""
EIMAS - Economic Intelligence Multi-Agent System
=================================================
통합 실행 파이프라인 (Unified Main Entry Point)
Unified execution pipeline for comprehensive market analysis.

================================================================================
ARCHITECTURE OVERVIEW | 아키텍처 개요
================================================================================

This is the main entry point that orchestrates all EIMAS components:
    - Data Collection: Multi-source data gathering (FRED, Market, Crypto)
    - Analysis Engine: Quantitative analysis (Regime, Risk, Microstructure)
    - AI Debate: Multi-agent debate for consensus building
    - Report Generation: AI-powered narrative reports
    - Validation: Fact-checking and whitening

================================================================================
PIPELINE FLOW | 파이프라인 흐름
================================================================================

    [CLI] main()
           │
           ▼
    run_integrated_pipeline()
           │
           ├─► [Phase 1] _collect_data()        # 데이터 수집 (FRED, Market, Crypto)
           │                                    # Data collection from multiple sources
           │
           ├─► [Phase 2] _analyze_basic()       # 기본 분석 (Regime, Events, Risk)
           │        └─► _analyze_enhanced()     # 고급 분석 (HFT, GARCH, DTW, etc.)
           │        └─► _analyze_sentiment_bubble()  # 센티먼트 & 버블 분석
           │
           ├─► [Phase 3] _run_debate()          # AI 에이전트 토론 (Multi-LLM)
           │                                    # Dual-mode debate with consensus
           │
           ├─► [Phase 4] _run_realtime()        # 실시간 스트리밍 (Optional)
           │                                    # VPIN/OFI stream analysis
           │
           ├─► [Phase 5] _save_results()        # 결과 저장 (unified JSON)
           │                                    # Save to outputs/eimas_*.json
           │
           ├─► [Phase 6] _generate_report()     # AI 리포트 생성 (Optional)
           │                                    # LLM-powered narrative report
           │
           ├─► [Phase 7] _validate_report()     # Whitening & Fact Check
           │                                    # Data quality validation
           │
           └─► [Phase 8] _run_ai_validation()   # Multi-LLM 검증 (--full only)
                                                # Cross-LLM consensus check

================================================================================
USAGE | 사용법
================================================================================

    python main.py              # 기본 분석 (버블/센티먼트 포함)
                                # Default analysis with bubble/sentiment
    
    python main.py --short      # Quick 분석 (버블/DTW 제외)
                                # Quick mode - skip heavy computations
    
    python main.py --full       # 전체 기능 (AI Validation 포함, API 비용 발생)
                                # Full mode - includes Multi-LLM validation
    
    python main.py --realtime   # 실시간 스트리밍 포함
                                # Include real-time streaming
    
    python main.py --full -r    # 전체 기능 + 실시간
                                # Full mode with real-time streaming

================================================================================
OUTPUT | 출력물
================================================================================

    outputs/
    ├── eimas_YYYYMMDD_HHMMSS.json   # Unified analysis results
    ├── eimas_YYYYMMDD.md            # Markdown summary
    └── reports/                     # AI-generated reports

================================================================================
"""


import asyncio
import argparse
from datetime import datetime
from typing import Dict, List, Any

# ============================================================================
# Pipeline Imports
# ============================================================================
from pipeline import (
    EIMASResult,
    # Phase 1: Collectors
    collect_fred_data, collect_market_data, collect_crypto_data, collect_market_indicators,
    # Phase 2: Analyzers
    detect_regime, detect_events, analyze_liquidity, analyze_critical_path,
    analyze_etf_flow, generate_explanation, analyze_genius_act, analyze_theme_etf,
    analyze_shock_propagation, optimize_portfolio_mst, analyze_volume_anomalies,
    track_events_with_news, run_adaptive_portfolio,
    analyze_bubble_risk, analyze_sentiment,  # NEW: 2026-01-29
    # Phase 3: Debate
    run_dual_mode_debate,
    # Phase 4: Realtime
    run_realtime_stream,
    # Phase 5: Storage
    save_result_json, save_result_md, save_to_trading_db, save_to_event_db,
    # Phase 6-7: Report & Validation
    generate_ai_report, run_whitening_check, run_fact_check,
    # Phase 8: AI Validation (--full only)
    run_ai_validation
)
from pipeline.schemas import BubbleRiskMetrics
from pipeline.analyzers import (
    analyze_hft_microstructure, analyze_volatility_garch,
    analyze_information_flow, calculate_proof_of_index,
    enhance_portfolio_with_systemic_similarity,
    detect_outliers_with_dbscan, analyze_dtw_similarity, analyze_ark_trades
)
from lib.extended_data_sources import ExtendedDataCollector
from lib.bubble_framework import FiveStageBubbleFramework
from lib.gap_analyzer import MarketModelGapAnalyzer
from lib.fomc_analyzer import FOMCDotPlotAnalyzer


# ============================================================================
# Phase Helper Functions
# ============================================================================

async def _collect_data(result: EIMASResult, quick_mode: bool) -> Dict:
    """[Phase 1] 데이터 수집: FRED, Market, Crypto, Extended"""
    print("\n[Phase 1] Collecting Data...")
    
    result.fred_summary = collect_fred_data()
    
    ext_collector = ExtendedDataCollector()
    result.extended_data = await ext_collector.collect_all()
    
    market_data = collect_market_data(lookback_days=90 if quick_mode else 365)
    result.market_data_count = len(market_data)
    
    collect_crypto_data()
    if not quick_mode:
        collect_market_indicators()
    
    return market_data


def _analyze_basic(result: EIMASResult, market_data: Dict) -> List:
    """[Phase 2.1] 기본 분석: Regime Detection, Event Detection, Risk Score"""
    print("\n[Phase 2] Analyzing Market...")
    
    regime_res = detect_regime()
    result.regime = regime_res.to_dict()
    
    events = detect_events(result.fred_summary, market_data)
    result.events_detected = [e.to_dict() for e in events]
    
    try:
        cp_res = analyze_critical_path(market_data)
        result.risk_score = cp_res.risk_score
        result.base_risk_score = cp_res.risk_score
    except Exception as e:
        print(f"⚠️ Critical Path Error: {e}")
    
    return events, regime_res


def _analyze_enhanced(result: EIMASResult, market_data: Dict, quick_mode: bool):
    """[Phase 2.2] 고급 분석: HFT, GARCH, DTW, DBSCAN, Liquidity, etc."""
    print("\n[Phase 2.Enhanced] Running Advanced Metrics...")
    
    # Always run (quick or full)
    _safe_call(lambda: setattr(result, 'hft_microstructure', analyze_hft_microstructure(market_data)), "HFT")
    _safe_call(lambda: setattr(result, 'garch_volatility', analyze_volatility_garch(market_data)), "GARCH")
    _safe_call(lambda: setattr(result, 'information_flow', analyze_information_flow(market_data)), "Info Flow")
    _safe_call(lambda: setattr(result, 'proof_of_index', calculate_proof_of_index(market_data)), "PoI")
    _safe_call(lambda: setattr(result, 'ark_analysis', analyze_ark_trades()), "ARK")
    _safe_call(lambda: enhance_portfolio_with_systemic_similarity(market_data), "Systemic Sim")
    
    # Full mode only
    if not quick_mode:
        _safe_call(lambda: setattr(result, 'dtw_similarity', analyze_dtw_similarity(market_data)), "DTW")
        _safe_call(lambda: setattr(result, 'dbscan_outliers', detect_outliers_with_dbscan(market_data)), "DBSCAN")
        _safe_call(lambda: _set_liquidity(result), "Liquidity")
        _safe_call(lambda: setattr(result, 'etf_flow_result', analyze_etf_flow().to_dict()), "ETF Flow")
        _safe_call(lambda: _set_genius_act(result), "Genius Act")
        _safe_call(lambda: setattr(result, 'theme_etf_analysis', analyze_theme_etf().to_dict()), "Theme ETF")
        _safe_call(lambda: setattr(result, 'shock_propagation', analyze_shock_propagation(market_data).to_dict()), "Shock")
        _safe_call(lambda: setattr(result, 'portfolio_weights', optimize_portfolio_mst(market_data).weights), "Portfolio")
        _safe_call(lambda: setattr(result, 'volume_anomalies', analyze_volume_anomalies(market_data)), "Volume")


def _set_liquidity(result):
    liq_res = analyze_liquidity()
    result.liquidity_signal = liq_res.signal
    result.liquidity_analysis = liq_res.to_dict()


def _set_genius_act(result):
    ga_res = analyze_genius_act()
    result.genius_act_signals = ga_res.signals
    result.genius_act_regime = ga_res.regime


def _apply_extended_data_adjustment(result: EIMASResult):
    """
    [Phase 2.Extended] Extended Data 기반 리스크 조정

    조정 로직:
    - Put/Call Ratio: >1.0 (공포) → -5, <0.7 (탐욕) → +5
    - Crypto F&G: <25 (Extreme Fear) → -3, >75 (Extreme Greed) → +5
    - News Sentiment: Bearish → -3, Bullish → +2
    - Credit Spreads: Risk OFF → +3, Risk ON → -2
    - KRW Risk: Overheated → +5, Volatile → +3

    총 조정 범위: -13 ~ +23 (clamp to ±15)
    """
    if not result.extended_data:
        return

    ext = result.extended_data
    adjustment = 0.0
    details = []

    # 1. Put/Call Ratio (옵션 센티먼트)
    pcr = ext.get('put_call_ratio', {})
    if pcr.get('ratio', 0) > 0:
        ratio = pcr['ratio']
        if ratio > 1.0:  # 공포/헤징 → 리스크 인식 높음 → 조심
            adjustment -= 5
            details.append(f"PCR={ratio:.2f} (Fear) → -5")
        elif ratio < 0.7:  # 탐욕 → 과열 신호
            adjustment += 5
            details.append(f"PCR={ratio:.2f} (Greed) → +5")

    # 2. Crypto Fear & Greed
    fng = ext.get('crypto_fng', {})
    if fng.get('value', 0) > 0:
        val = fng['value']
        if val < 25:  # Extreme Fear
            adjustment -= 3
            details.append(f"Crypto F&G={val} (Fear) → -3")
        elif val > 75:  # Extreme Greed
            adjustment += 5
            details.append(f"Crypto F&G={val} (Greed) → +5")

    # 3. News Sentiment
    news = ext.get('news_sentiment', {})
    label = news.get('label', '')
    if label == 'Bearish':
        adjustment -= 3
        details.append("News=Bearish → -3")
    elif label == 'Bullish':
        adjustment += 2
        details.append("News=Bullish → +2")

    # 4. Credit Spreads (HYG/IEF 리스크 비율)
    credit = ext.get('credit_spreads', {})
    interp = credit.get('interpretation', '')
    if interp == 'Risk OFF':
        adjustment += 3
        details.append("Credit=Risk OFF → +3")
    elif interp == 'Risk ON':
        adjustment -= 2
        details.append("Credit=Risk ON → -2")

    # 5. KRW Risk (한국 시장 리스크)
    krw = ext.get('korea_risk', {})
    status = krw.get('status', '')
    if 'Overheated' in status:
        adjustment += 5
        details.append("KRW=Overheated → +5")
    elif 'Volatile' in status:
        adjustment += 3
        details.append("KRW=Volatile → +3")

    # Clamp to ±15
    adjustment = max(-15, min(15, adjustment))

    # Apply adjustment
    if adjustment != 0:
        result.extended_data_adjustment = adjustment
        old_risk = result.risk_score
        result.risk_score = max(0, min(100, result.risk_score + adjustment))
        print(f"      ✓ Extended Data Adjustment: {adjustment:+.0f} ({old_risk:.1f} → {result.risk_score:.1f})")
        if details:
            print(f"        Details: {', '.join(details)}")


def _analyze_sentiment_bubble(result: EIMASResult, market_data: Dict, quick_mode: bool):
    """[Phase 2.3] 센티먼트 & 버블 분석"""
    # Bubble (full mode only)
    if not quick_mode:
        try:
            bubble_res = analyze_bubble_risk(market_data)
            if bubble_res:
                result.bubble_risk = BubbleRiskMetrics(**bubble_res)
        except Exception as e:
            print(f"⚠️ Bubble Risk Error: {e}")

    # Sentiment (always)
    try:
        result.sentiment_analysis = analyze_sentiment()
    except Exception as e:
        print(f"⚠️ Sentiment Error: {e}")


def _analyze_institutional_frameworks(result: EIMASResult, market_data: Dict, quick_mode: bool):
    """
    [Phase 2.Institutional] 기관급 분석 프레임워크

    JP Morgan, Goldman Sachs 방법론 기반:
    1. 5-Stage Bubble Framework
    2. Market-Model Gap Analysis
    3. FOMC Dot Plot Analysis
    """
    print("\n[Phase 2.Institutional] Running Institutional Frameworks...")

    # 1. 5-Stage Bubble Framework (JP Morgan WM)
    try:
        bubble_fw = FiveStageBubbleFramework()
        bubble_result = bubble_fw.analyze(market_data, sector='tech')
        result.bubble_framework = bubble_result.to_dict()
        print(f"      ✓ 5-Stage Bubble: {bubble_result.stage} (Score: {bubble_result.total_score:.1f}/100)")
    except Exception as e:
        print(f"      ⚠️ 5-Stage Bubble Error: {e}")

    # 2. Gap Analyzer (Goldman Sachs)
    try:
        gap_analyzer = MarketModelGapAnalyzer()
        gap_result = gap_analyzer.analyze()
        result.gap_analysis = gap_result.to_dict()
        print(f"      ✓ Market-Model Gap: {gap_result.overall_signal} ({gap_result.opportunity[:40]}...)")
    except Exception as e:
        print(f"      ⚠️ Gap Analysis Error: {e}")

    # 3. FOMC Dot Plot Analyzer (JP Morgan AM) - full mode only
    if not quick_mode:
        try:
            fomc_analyzer = FOMCDotPlotAnalyzer()
            fomc_result = fomc_analyzer.analyze('2026')
            result.fomc_analysis = fomc_result.to_dict()
            print(f"      ✓ FOMC Analysis: {fomc_result.stance} (Uncertainty: {fomc_result.policy_uncertainty_index:.0f}/100)")
        except Exception as e:
            print(f"      ⚠️ FOMC Analysis Error: {e}")


async def _run_debate(result: EIMASResult, market_data: Dict):
    """[Phase 3] AI 에이전트 토론: Dual Mode Debate"""
    print("\n[Phase 3] Running AI Debate...")
    try:
        debate_res = await run_dual_mode_debate(market_data, extended_data=result.extended_data)
        result.full_mode_position = debate_res.full_mode_position
        result.reference_mode_position = debate_res.reference_mode_position
        result.modes_agree = debate_res.modes_agree
        result.final_recommendation = debate_res.final_recommendation
        result.confidence = debate_res.confidence
        result.risk_level = debate_res.risk_level
        result.warnings.extend(debate_res.warnings)
        result.reasoning_chain = debate_res.reasoning_chain
        
        if debate_res.enhanced_debate:
            result.debate_consensus['enhanced'] = debate_res.enhanced_debate
        if debate_res.verification:
            result.debate_consensus['verification'] = debate_res.verification
        if debate_res.metadata:
            result.debate_consensus['metadata'] = debate_res.metadata
    except Exception as e:
        print(f"⚠️ Debate Error: {e}")


def _run_adaptive_portfolio(result: EIMASResult, regime_res, quick_mode: bool):
    """[Phase 2.4] 적응형 포트폴리오 배분"""
    if not quick_mode:
        try:
            result.adaptive_portfolios = run_adaptive_portfolio(regime_res)
        except Exception as e:
            print(f"⚠️ Adaptive Portfolio Error: {e}")


async def _run_realtime(result: EIMASResult, enable: bool, duration: int):
    """[Phase 4] 실시간 스트리밍 (선택)"""
    if enable:
        print("\n[Phase 4] Realtime Streaming...")
        signals = await run_realtime_stream(duration=duration)
        result.realtime_signals = [s.to_dict() for s in signals]
        save_to_trading_db(signals)


def _save_results(result: EIMASResult, events: List) -> str:
    """[Phase 5] 결과 저장: unified JSON (eimas_*.json)"""
    print("\n[Phase 5] Saving Results...")
    save_to_event_db(events)
    output_file = save_result_json(result)
    save_result_md(result)
    return output_file


async def _generate_report(result: EIMASResult, market_data: Dict, generate: bool) -> str:
    """[Phase 6] AI 리포트 생성 (선택)"""
    if not generate:
        return ""
    
    print("\n[Phase 6] Generating Report...")
    try:
        ai_report = await generate_ai_report(result, market_data)
        result.ai_report = ai_report.to_dict() if hasattr(ai_report, 'to_dict') else ai_report.__dict__
        save_result_json(result)
        return ai_report.content
    except Exception as e:
        print(f"⚠️ Report Generation Error: {e}")
        return ""


async def _validate_report(result: EIMASResult, report_content: str, generate: bool):
    """[Phase 7] 리포트 검증: Whitening & Fact Check"""
    if not generate:
        return
    
    print("\n" + "=" * 50)
    print("PHASE 7: VALIDATION")
    print("=" * 50)
    
    try:
        result.whitening_summary = run_whitening_check(result)
        if report_content:
            result.fact_check_grade = await run_fact_check(report_content)
        save_result_json(result)
    except Exception as e:
        print(f"⚠️ Validation Error: {e}")


def _run_ai_validation_phase(result: EIMASResult, full_mode: bool):
    """[Phase 8] Multi-LLM 검증 (--full only, API 비용 발생)"""
    if not full_mode:
        return
    
    print("\n" + "=" * 50)
    print("PHASE 8: AI VALIDATION (Multi-LLM)")
    print("=" * 50)
    
    try:
        result.validation_loop_result = run_ai_validation(result.to_dict())
        save_result_json(result)
    except Exception as e:
        print(f"⚠️ AI Validation Error: {e}")


def _safe_call(fn, name: str):
    """안전한 함수 호출 (에러 시 경고만)"""
    try:
        fn()
    except Exception as e:
        print(f"⚠️ {name} Error: {e}")


# ============================================================================
# Main Pipeline
# ============================================================================

async def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    generate_report: bool = False,
    full_mode: bool = False
) -> EIMASResult:
    """
    Execute the EIMAS integrated analysis pipeline.
    EIMAS 통합 분석 파이프라인 실행.
    
    This function orchestrates the complete analysis workflow from data collection
    through AI-powered report generation and validation.
    
    Pipeline Phases:
        Phase 1: Data collection (FRED macroeconomic, market prices, crypto)
        Phase 2: Quantitative analysis (regime detection, risk scoring, etc.)
        Phase 3: AI agent debate with dual-mode consensus
        Phase 4: Real-time streaming analysis (optional)
        Phase 5: Result storage (JSON, database)
        Phase 6: AI report generation (optional)
        Phase 7: Report validation (whitening, fact-check)
        Phase 8: Multi-LLM cross-validation (full mode only)
    
    Args:
        enable_realtime (bool): Enable real-time VPIN/OFI streaming.
                               실시간 스트리밍 활성화. Default: False.
        realtime_duration (int): Duration in seconds for streaming.
                                스트리밍 지속 시간 (초). Default: 30.
        quick_mode (bool): Skip heavy computations (bubble, DTW, DBSCAN).
                          버블/DTW 등 고비용 분석 제외. Default: False.
        generate_report (bool): Generate AI-powered narrative report.
                               AI 리포트 생성 여부. Default: False.
        full_mode (bool): Include Multi-LLM validation (incurs API costs).
                         Multi-LLM 검증 포함 (API 비용 발생). Default: False.
    
    Returns:
        EIMASResult: Comprehensive analysis result object containing:
            - fred_summary: Macroeconomic data summary
            - regime: Current market regime classification
            - risk_score: Composite risk score (0-100)
            - debate results: AI agent consensus
            - And many more analysis outputs...
    
    Raises:
        No exceptions are raised; errors are logged with warnings.
        모든 에러는 경고로 출력되며, 예외는 발생하지 않습니다.
    
    Example:
        >>> result = await run_integrated_pipeline(quick_mode=True)
        >>> print(f"Risk Score: {result.risk_score}")
    """
    start_time = datetime.now()
    result = EIMASResult(timestamp=start_time.isoformat())

    print("=" * 70)
    print("  EIMAS - Integrated Analysis Pipeline")
    print("=" * 70)

    # Phase 1-2: Data & Analysis
    market_data = await _collect_data(result, quick_mode)
    events, regime_res = _analyze_basic(result, market_data)
    _analyze_enhanced(result, market_data, quick_mode)
    _analyze_sentiment_bubble(result, market_data, quick_mode)
    _apply_extended_data_adjustment(result)  # PCR, Sentiment, Credit 기반 리스크 조정
    _analyze_institutional_frameworks(result, market_data, quick_mode)  # JP Morgan, Goldman Sachs 프레임워크
    _run_adaptive_portfolio(result, regime_res, quick_mode)

    # Phase 3-4: Debate & Realtime
    await _run_debate(result, market_data)
    await _run_realtime(result, enable_realtime, realtime_duration)
    
    # Phase 5-8: Storage, Report, Validation
    output_file = _save_results(result, events)
    report_content = await _generate_report(result, market_data, generate_report)
    await _validate_report(result, report_content, generate_report)
    _run_ai_validation_phase(result, full_mode)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print(f"EIMAS PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"Output: {output_file}")
    print("=" * 70)

    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description='EIMAS Modular Pipeline')
    parser.add_argument('--realtime', '-r', action='store_true', help='Enable realtime stream')
    parser.add_argument('--duration', '-d', type=int, default=30, help='Stream duration (seconds)')
    parser.add_argument('--short', '-s', action='store_true', help='Quick analysis (no bubble/DTW)')
    parser.add_argument('--quick', '-q', action='store_true', help='Alias for --short')
    parser.add_argument('--full', '-f', action='store_true', help='All features (AI Validation, costs API)')
    
    args = parser.parse_args()
    
    is_short = args.short or args.quick
    
    asyncio.run(run_integrated_pipeline(
        enable_realtime=args.realtime,
        realtime_duration=args.duration,
        quick_mode=is_short,
        generate_report=not is_short,
        full_mode=args.full
    ))


if __name__ == "__main__":
    main()
