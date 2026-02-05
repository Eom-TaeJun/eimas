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
import numpy as np

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
    run_allocation_engine, run_rebalancing_policy,  # NEW: 2026-02-02
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
from lib.operational_engine import OperationalEngine  # Operational Report generation
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

# Quick Mode AI Agents (2026-02-04)
from lib.quick_agents import QuickOrchestrator

# Portfolio Theory Modules (2026-02-04)
from lib.backtest import BacktestEngine, BacktestConfig
from lib.trading_db import TradingDB
from lib.performance_attribution import BrinsonAttribution, InformationRatio, ActiveShare
from lib.tactical_allocation import TacticalAssetAllocator
from lib.stress_test import StressTestEngine, generate_stress_test_report

# Korea & Strategic Allocation (2026-02-04)
from pipeline.korea_integration import (
    collect_korea_assets,
    calculate_fair_values,
    calculate_strategic_allocation,
    generate_allocation_summary
)


# ============================================================================
# Phase Helper Functions
# ============================================================================

async def _collect_data(result: EIMASResult, quick_mode: bool) -> Dict:
    """[Phase 1] 데이터 수집: FRED, Market, Crypto, Extended, Korea"""
    print("\n[Phase 1] Collecting Data...")

    result.fred_summary = collect_fred_data()

    ext_collector = ExtendedDataCollector()
    result.extended_data = await ext_collector.collect_all()

    market_data = collect_market_data(lookback_days=90 if quick_mode else 365)
    result.market_data_count = len(market_data)

    collect_crypto_data()
    if not quick_mode:
        collect_market_indicators()

    # [NEW] 한국 자산 수집 (Quick/Full 모두 포함)
    print("\n[Phase 1.4] Collecting Korea Assets...")
    korea_result = collect_korea_assets(
        lookback_days=90 if quick_mode else 365,
        use_parallel=True
    )
    result.korea_data = korea_result['data']
    result.korea_summary = korea_result['summary']
    print(f"  ✓ Korea assets: {korea_result['summary'].get('total_assets', 0)} collected")

    # Store korea data in market_data for downstream use
    market_data['korea_data'] = korea_result['data']

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

    # NEW: Fair Value & Strategic Allocation (Quick/Full 모두 실행)
    _safe_call(lambda: _calculate_strategic_allocation(result, market_data, quick_mode), "Strategic Allocation")

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
        # NEW: 비중 산출 엔진 (2026-02-02)
        _safe_call(lambda: _set_allocation_result(result, market_data), "Allocation Engine")
        # NEW: 전술적 자산배분 (2026-02-04)
        _safe_call(lambda: _run_tactical_allocation(result), "Tactical Allocation")


def _set_allocation_result(result: EIMASResult, market_data: Dict):
    """[Phase 2.11-2.12] 비중 산출 및 리밸런싱 정책 평가"""
    # 기존 portfolio_weights를 current_weights로 사용
    current_weights = result.portfolio_weights if result.portfolio_weights else None

    alloc_result = run_allocation_engine(
        market_data=market_data,
        strategy="risk_parity",  # 기본 Risk Parity 전략
        current_weights=current_weights
    )

    result.allocation_result = alloc_result.get('allocation_result', {})
    result.allocation_strategy = alloc_result.get('allocation_strategy', 'risk_parity')
    result.allocation_config = alloc_result.get('allocation_config', {})

    # 리밸런싱 결정
    if alloc_result.get('rebalance_decision'):
        result.rebalance_decision = alloc_result['rebalance_decision']
    elif current_weights and alloc_result.get('allocation_result', {}).get('weights'):
        # 리밸런싱 평가 별도 실행
        result.rebalance_decision = run_rebalancing_policy(
            current_weights=current_weights,
            target_weights=alloc_result['allocation_result']['weights']
        )

    # 경고 추가
    if alloc_result.get('warnings'):
        result.warnings.extend(alloc_result['warnings'])


def _set_liquidity(result):
    liq_res = analyze_liquidity()
    result.liquidity_signal = liq_res.signal
    result.liquidity_analysis = liq_res.to_dict()


def _set_genius_act(result):
    ga_res = analyze_genius_act()
    result.genius_act_signals = ga_res.signals
    result.genius_act_regime = ga_res.regime


def _calculate_strategic_allocation(result: EIMASResult, market_data: Dict, quick_mode: bool):
    """
    [Phase 2.13] Fair Value & 전략적 자산배분

    Quick Mode: Fed Model + 주식/채권 비중
    Full Mode: 종합 Fair Value + 글로벌 배분
    """
    print("\n[Phase 2.13] Calculating Fair Values & Strategic Allocation...")

    try:
        # 1. Bond yields from FRED summary
        bond_yields = {
            'us_10y': 0.042,  # Default
            'korea_10y': 0.035  # Default
        }

        # Try to get actual yields from FRED
        if hasattr(result, 'fred_summary') and result.fred_summary:
            # US 10Y from FRED (DGS10)
            if 'DGS10' in result.fred_summary:
                bond_yields['us_10y'] = result.fred_summary['DGS10'] / 100

        # 2. Fair Value calculation
        mode = 'quick' if quick_mode else 'comprehensive'
        fair_value_results = calculate_fair_values(
            market_data=market_data,
            bond_yields=bond_yields,
            mode=mode
        )

        result.fair_value_analysis = fair_value_results
        print(f"  ✓ Fair Value: SPX {'✓' if 'spx' in fair_value_results else '✗'}, KOSPI {'✓' if 'kospi' in fair_value_results else '✗'}")

        # 3. Market stats (estimated from recent data)
        market_stats = {
            'stock_return': 0.08,
            'bond_return': 0.04,
            'stock_vol': 0.16,
            'bond_vol': 0.06,
            'correlation': 0.1,
            'kospi_return': 0.06,
            'kospi_vol': 0.20,
            'us_korea_corr': 0.6
        }

        # Estimate from market_data if available
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_returns = market_data['SPY']['Close'].pct_change().dropna()
            if len(spy_returns) > 20:
                market_stats['stock_return'] = spy_returns.mean() * 252
                market_stats['stock_vol'] = spy_returns.std() * np.sqrt(252)

        # 4. Strategic Allocation (Evidence-Based in Full mode)
        include_korea = 'korea_data' in market_data and market_data['korea_data']

        allocation_results = calculate_strategic_allocation(
            fair_value_results=fair_value_results,
            market_stats=market_stats,
            risk_tolerance='moderate',
            include_korea=include_korea,
            use_evidence_based=not quick_mode  # Full 모드에서만 증거 기반
        )

        result.strategic_allocation = allocation_results

        # Summary
        summary = generate_allocation_summary(allocation_results)
        print(summary)

    except Exception as e:
        print(f"  ✗ Strategic Allocation Error: {e}")
        import traceback
        traceback.print_exc()


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
        # Floor of 1.0 prevents economically unrealistic zero risk
        result.risk_score = max(1.0, min(100, result.risk_score + adjustment))
        print(f"      ✓ Extended Data Adjustment: {adjustment:+.0f} ({old_risk:.1f} → {result.risk_score:.1f})")
        if details:
            print(f"        Details: {', '.join(details)}")

        # Warn if risk is extremely low
        if result.risk_score < 5:
            warning = f"⚠️ Extremely Low Risk ({result.risk_score:.1f}/100) - Verify market conditions"
            result.warnings.append(warning)
            print(f"      {warning}")


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
        # NEW: 기관 투자자 종합 분석 저장
        if debate_res.institutional_analysis:
            result.institutional_analysis = debate_res.institutional_analysis
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


def _generate_operational_report(result: EIMASResult, current_weights: Dict = None):
    """[Phase 4.5] Operational Report 생성 (decision governance, rebalance plan)"""
    print("\n[Phase 4.5] Generating Operational Report...")

    try:
        from lib.operational_engine import (
            OperationalEngine, get_indicator_classification, get_input_validation,
            get_operational_controls, get_audit_metadata, get_approval_status
        )

        engine = OperationalEngine()

        # Use existing EIMAS results as input
        eimas_data = result.to_dict()

        # Use portfolio_weights as current weights if not provided
        if current_weights is None:
            current_weights = result.portfolio_weights or {}

        # Generate operational report
        op_report = engine.process(eimas_data, current_weights)

        # Store in result
        result.operational_report = op_report.to_dict()

        # ================================================================
        # Populate NEW enhanced operational fields (2026-02-03)
        # ================================================================

        # 1. Input Validation
        result.input_validation = get_input_validation(eimas_data)

        # 2. Indicator Classification (CORE vs AUX)
        result.indicator_classification = get_indicator_classification(eimas_data)

        # 3. Trade Plan (from rebalance plan or rebalance_decision)
        rebal_plan = op_report.rebalance_plan
        if rebal_plan.should_execute and rebal_plan.trades:
            result.trade_plan = [t.to_dict() for t in rebal_plan.trades if t.action != "HOLD"]
        elif result.rebalance_decision and result.rebalance_decision.get('trade_plan'):
            result.trade_plan = result.rebalance_decision['trade_plan']

        # 4. Operational Controls
        result.operational_controls = get_operational_controls(
            eimas_data,
            rebalance_decision=result.rebalance_decision,
            constraint_result=op_report.constraint_repair.to_dict()
        )

        # 5. Audit Metadata
        result.audit_metadata = get_audit_metadata(eimas_data)

        # 6. Approval Status
        result.approval_status = get_approval_status(
            eimas_data,
            rebalance_plan=op_report.rebalance_plan.to_dict()
        )

        # ================================================================
        # Failsafe Logic: Constraint Violations → HOLD
        # ================================================================
        if not op_report.constraint_repair.constraints_satisfied:
            # Check if there are serious constraint violations
            violations = op_report.constraint_repair.violations_found
            has_severe_violation = any(
                v.severity == "SEVERE" for v in violations
            )

            if op_report.constraint_repair.force_hold or has_severe_violation:
                result.failsafe_status = {
                    'triggered': True,
                    'reason': 'CONSTRAINT_VIOLATION',
                    'fallback_action': 'HOLD',
                    'original_recommendation': result.final_recommendation,
                    'violations': [v.to_dict() for v in violations]
                }
                result.final_recommendation = "HOLD"
                result.warnings.append("⚠️ Constraint Violation - Forced to HOLD")
                print(f"      ⚠️ FAILSAFE TRIGGERED: Constraint Violation → HOLD")
            else:
                # Constraints violated but repaired - just log warning
                result.failsafe_status = {
                    'triggered': False,
                    'reason': 'Constraints violated but partially repaired',
                    'fallback_action': None,
                    'original_recommendation': result.final_recommendation
                }
                result.warnings.append(f"⚠️ {len(violations)} constraint violation(s) detected")
        else:
            # No constraint violations
            if not result.failsafe_status:
                result.failsafe_status = {
                    'triggered': False,
                    'reason': None,
                    'fallback_action': None,
                    'original_recommendation': result.final_recommendation
                }

        # Update final_recommendation to match operational decision
        # (only if failsafe not triggered)
        if not result.failsafe_status.get('triggered', False):
            if op_report.decision_policy.final_stance:
                # Map HOLD/BULLISH/BEARISH to EIMAS recommendation format
                stance = op_report.decision_policy.final_stance
                if stance == "HOLD":
                    result.final_recommendation = "HOLD"
                elif stance == "BULLISH":
                    result.final_recommendation = "BULLISH"
                elif stance == "BEARISH":
                    result.final_recommendation = "BEARISH"
                else:
                    result.final_recommendation = "NEUTRAL"

        print(f"      ✓ Decision: {op_report.decision_policy.final_stance}")
        print(f"      ✓ Constraints: {'SATISFIED' if op_report.constraint_repair.constraints_satisfied else 'VIOLATED'}")
        print(f"      ✓ Rebalance: {'EXECUTE' if op_report.rebalance_plan.should_execute else 'NOT EXECUTED'}")
        print(f"      ✓ Failsafe: {'TRIGGERED' if result.failsafe_status.get('triggered') else 'NOT TRIGGERED'}")

    except Exception as e:
        import traceback
        print(f"      ⚠️ Operational Report Error: {e}")
        traceback.print_exc()
        result.operational_report = {"error": str(e)}
        result.failsafe_status = {
            'triggered': False,
            'reason': f'Error in operational report: {e}',
            'fallback_action': None,
            'original_recommendation': result.final_recommendation
        }


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

        # CRITICAL: If validation REJECTS, force failsafe to HOLD
        if result.validation_loop_result.get('final_result') == "REJECT":
            result.failsafe_status = {
                'triggered': True,
                'reason': 'AI Validation REJECT',
                'fallback_action': 'HOLD',
                'original_recommendation': result.final_recommendation
            }
            result.final_recommendation = "HOLD"
            result.warnings.append("⚠️ AI Validation REJECT - Forced to HOLD")
            print(f"      ⚠️ FAILSAFE TRIGGERED: AI Validation REJECT → HOLD")

        save_result_json(result)
    except Exception as e:
        print(f"⚠️ AI Validation Error: {e}")


def _run_quick_validation(
    result: EIMASResult,
    market_data: Dict,
    output_file: str,
    market_focus: str = None
):
    """[Phase 8.5] Quick Mode AI Validation (KOSPI/SPX 분리)"""
    if not market_focus:
        return

    print("\n" + "=" * 70)
    print(f"PHASE 8.5: QUICK MODE AI VALIDATION ({market_focus} Focus)")
    print("=" * 70)

    try:
        orchestrator = QuickOrchestrator()

        # Run validation
        quick_result = orchestrator.run_quick_validation(
            full_json_path=output_file,
            market_data=market_data
        )

        # Store result
        result.quick_validation = quick_result

        # Extract key findings
        final_val = quick_result.get('final_validation', {})
        final_rec = final_val.get('final_recommendation', 'N/A')
        confidence = final_val.get('confidence', 0)
        validation_result = final_val.get('validation_result', 'N/A')

        # Print summary
        print(f"\n📊 Quick Validation Summary ({market_focus}):")
        print(f"   • Validation Result: {validation_result}")
        print(f"   • Final Recommendation: {final_rec}")
        print(f"   • Confidence: {confidence*100:.0f}%")

        # Agent consensus
        agent_consensus = final_val.get('agent_consensus', {})
        agreement_level = agent_consensus.get('agreement_level', 'N/A')
        print(f"   • Agent Agreement: {agreement_level}")

        # Full vs Quick comparison
        comparison = final_val.get('full_vs_quick_comparison', {})
        alignment = comparison.get('alignment', 'N/A')
        print(f"   • Full vs Quick: {alignment}")

        # Market sentiment (KOSPI or SPX focus)
        market_sentiment = quick_result.get('market_sentiment', {})
        if market_focus == 'KOSPI':
            kospi_sent = market_sentiment.get('kospi_sentiment', {})
            sent = kospi_sent.get('sentiment', 'N/A')
            sent_conf = kospi_sent.get('confidence', 0)
            print(f"   • KOSPI Sentiment: {sent} ({sent_conf*100:.0f}%)")
        elif market_focus == 'SPX':
            spx_sent = market_sentiment.get('spx_sentiment', {})
            sent = spx_sent.get('sentiment', 'N/A')
            sent_conf = spx_sent.get('confidence', 0)
            print(f"   • SPX Sentiment: {sent} ({sent_conf*100:.0f}%)")

        # Risk warnings
        risk_warnings = final_val.get('risk_warnings', [])
        if risk_warnings:
            print(f"\n⚠️ Risk Warnings ({len(risk_warnings)}):")
            for i, warning in enumerate(risk_warnings[:3], 1):
                print(f"   {i}. {warning}")

        # Save Quick validation result
        import json
        from pathlib import Path
        quick_output = Path(output_file).parent / f"quick_validation_{market_focus.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(quick_output, 'w', encoding='utf-8') as f:
            json.dump(quick_result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Quick validation saved: {quick_output}")

    except Exception as e:
        print(f"⚠️ Quick Validation Error: {e}")
        import traceback
        traceback.print_exc()


def _safe_call(fn, name: str):
    """안전한 함수 호출 (에러 시 경고만)"""
    try:
        fn()
    except Exception as e:
        print(f"⚠️ {name} Error: {e}")


# ============================================================================
# Portfolio Theory Module Functions (2026-02-04)
# ============================================================================

def _run_backtest(result: EIMASResult, market_data: Dict, enable: bool):
    """[Phase 6.1] 백테스팅 엔진 (Optional)"""
    if not enable:
        return

    print("\n[Phase 6.1] Running Backtest Engine...")

    try:
        import pandas as pd
        from datetime import timedelta

        # Convert market_data to prices DataFrame
        if not market_data or len(market_data) == 0:
            print("⚠️ No market data for backtest")
            return

        # Get price data from market_data dict
        tickers = list(market_data.keys())
        prices_dict = {}

        for ticker in tickers:
            data = market_data[ticker]
            if isinstance(data, pd.DataFrame) and 'close' in data.columns:
                prices_dict[ticker] = data['close']

        if not prices_dict:
            print("⚠️ No valid price data for backtest")
            return

        prices = pd.DataFrame(prices_dict)
        prices = prices.dropna()

        if len(prices) < 252:
            print(f"⚠️ Insufficient data for backtest: {len(prices)} days")
            return

        # Configure backtest (5 years or all available)
        start_date = str(prices.index[252])  # Skip first year for indicators
        end_date = str(prices.index[-1])

        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            rebalance_frequency='monthly',
            transaction_cost_bps=10,
            initial_capital=1_000_000
        )

        # Define allocation strategy (use current portfolio weights or equal weight)
        def allocation_strategy(prices_window):
            if result.portfolio_weights:
                # Use existing portfolio weights
                return result.portfolio_weights
            else:
                # Equal weight fallback
                n = len(prices_window.columns)
                return {ticker: 1/n for ticker in prices_window.columns}

        # TradingCostModel 생성
        from lib.rebalancing_policy import TradingCostModel
        cost_model = TradingCostModel()

        # Run backtest (v2.1: drift tracking + benchmark + TradingCostModel)
        engine = BacktestEngine(config, cost_model=cost_model)
        backtest_result = engine.run(prices, allocation_strategy, benchmark='SPY')

        # Store metrics
        metrics = backtest_result.metrics
        result.backtest_metrics = metrics.to_dict()

        # --- Period metrics에서 OVERALL alpha/beta 추출 ---
        overall_alpha = 0.0
        overall_bm_return = 0.0
        for pm in backtest_result.period_metrics:
            if pm['period_type'] == 'OVERALL':
                overall_alpha = pm.get('alpha') or 0.0
                overall_bm_return = pm.get('benchmark_return') or 0.0
                break

        # DB 저장
        db = TradingDB()

        # 1. backtest_runs (기존 테이블 + v2.1 컬럼)
        db_payload = {
            'strategy_name': 'EIMAS_Portfolio',
            'start_date': metrics.start_date,
            'end_date': metrics.end_date,
            'initial_capital': config.initial_capital,
            'final_capital': config.initial_capital * (1 + metrics.total_return),
            'total_return': metrics.total_return,
            'annual_return': metrics.annualized_return,
            'benchmark_return': overall_bm_return,
            'alpha': overall_alpha,
            'volatility': metrics.annualized_volatility,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_duration': metrics.max_drawdown_duration,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'total_trades': metrics.num_trades,
            'winning_trades': int(metrics.win_rate * metrics.num_periods),
            'losing_trades': metrics.num_periods - int(metrics.win_rate * metrics.num_periods),
            'win_rate': metrics.win_rate,
            'avg_win': metrics.avg_win,
            'avg_loss': metrics.avg_loss,
            'profit_factor': metrics.profit_factor,
            'avg_holding_days': 30,  # monthly rebalance
            'total_commission': metrics.total_transaction_costs,
            'total_slippage': 0.0,
            'total_short_cost': 0.0,
            'parameters': {
                'rebalance_frequency': config.rebalance_frequency,
                'transaction_cost_bps': config.transaction_cost_bps,
                'initial_capital': config.initial_capital,
                'benchmark': 'SPY',
                'cost_model': 'TradingCostModel',
            },
            'trades': []  # ticker-level entry/exit는 snapshots에서 복원 가능
        }
        result.backtest_run_id = db.save_backtest_run(db_payload)
        run_id = result.backtest_run_id

        # 2. backtest_daily_nav (일별 NAV + ticker attribution)
        db.save_backtest_daily_nav(run_id, backtest_result.daily_nav_records)

        # 3. backtest_snapshots (리밸런싱 스냅샷)
        db.save_backtest_snapshots(run_id, backtest_result.snapshot_records)

        # 4. backtest_period_metrics (기간별 성과 분해)
        db.save_backtest_period_metrics(run_id, backtest_result.period_metrics)

        # v2.1 컬럼 업데이트 (git_commit, benchmark, cost_model)
        import subprocess, sqlite3
        try:
            git_hash = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=str(Path(__file__).parent),
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            git_hash = None

        conn = sqlite3.connect(db.db_path)
        conn.execute("""
            UPDATE backtest_runs
            SET git_commit = ?, benchmark = 'SPY', cost_model = 'TradingCostModel'
            WHERE id = ?
        """, (git_hash, run_id))
        conn.commit()
        conn.close()

        print(f"  ✅ Backtest Complete:")
        print(f"     Sharpe: {metrics.sharpe_ratio:.2f}")
        print(f"     Max DD: {metrics.max_drawdown*100:.1f}%")
        print(f"     VaR 95%: {metrics.var_95*100:.2f}%")
        print(f"     Alpha: {overall_alpha*100:.2f}%")
        print(f"     Snapshots: {len(backtest_result.snapshot_records)}")
        print(f"     DB Saved: Run ID {run_id}")

    except Exception as e:
        print(f"⚠️ Backtest Error: {e}")
        import traceback
        traceback.print_exc()


def _run_performance_attribution(result: EIMASResult, enable: bool):
    """[Phase 6.2] 성과 귀속 분석 (Optional)"""
    if not enable:
        return

    print("\n[Phase 6.2] Running Performance Attribution...")

    try:
        # Need portfolio weights and benchmark weights
        portfolio_weights = result.portfolio_weights
        if not portfolio_weights:
            print("⚠️ No portfolio weights for attribution")
            return

        # Use 60/40 as default benchmark
        benchmark_weights = {}
        equity_tickers = ['SPY', 'QQQ', 'IWM']
        bond_tickers = ['TLT', 'IEF']

        # Distribute 60% to equities, 40% to bonds
        for ticker in portfolio_weights.keys():
            if ticker in equity_tickers:
                benchmark_weights[ticker] = 0.60 / len([t for t in portfolio_weights if t in equity_tickers])
            elif ticker in bond_tickers:
                benchmark_weights[ticker] = 0.40 / len([t for t in portfolio_weights if t in bond_tickers])
            else:
                benchmark_weights[ticker] = 0.0

        # Normalize
        total = sum(benchmark_weights.values())
        if total > 0:
            benchmark_weights = {k: v/total for k, v in benchmark_weights.items()}

        # Use assumed returns (would need actual returns from market data)
        # This is a simplified example - real implementation would calculate from prices
        portfolio_returns = {ticker: 0.10 for ticker in portfolio_weights.keys()}
        benchmark_returns = {ticker: 0.08 for ticker in benchmark_weights.keys()}

        # Brinson Attribution
        brinson = BrinsonAttribution()
        attribution = brinson.compute(
            portfolio_weights, portfolio_returns,
            benchmark_weights, benchmark_returns
        )

        result.performance_attribution = attribution.to_dict()

        # Active Share
        active_share = ActiveShare.compute(portfolio_weights, benchmark_weights)
        result.performance_attribution['active_share'] = active_share

        print(f"  ✅ Attribution Complete:")
        print(f"     Excess Return: {attribution.excess_return*100:.2f}%")
        print(f"     Allocation Effect: {attribution.allocation_effect*100:.2f}%")
        print(f"     Active Share: {active_share*100:.1f}%")

    except Exception as e:
        print(f"⚠️ Attribution Error: {e}")
        import traceback
        traceback.print_exc()


def _run_tactical_allocation(result: EIMASResult):
    """[Phase 2.11] 전술적 자산배분 (Always run if portfolio exists)"""
    if not result.portfolio_weights:
        return

    print("\n[Phase 2.11] Running Tactical Asset Allocation...")

    try:
        # Asset class mapping
        asset_class_mapping = {
            'SPY': 'equity', 'QQQ': 'equity', 'IWM': 'equity', 'DIA': 'equity',
            'TLT': 'bond', 'IEF': 'bond', 'SHY': 'bond', 'HYG': 'bond',
            'GLD': 'commodity', 'DBC': 'commodity',
            'BTC-USD': 'crypto', 'ETH-USD': 'crypto'
        }

        # Get current regime
        regime = result.regime.get('regime', 'Neutral')
        regime_confidence = result.regime.get('confidence', 0.5)

        # Tactical allocator
        taa = TacticalAssetAllocator(
            strategic_weights=result.portfolio_weights,
            asset_class_mapping=asset_class_mapping,
            max_tilt_pct=0.15
        )

        # Compute tactical weights
        tactical_weights = taa.compute_tactical_weights(
            regime=regime,
            confidence=regime_confidence
        )

        result.tactical_weights = tactical_weights

        # Calculate adjustment
        total_adjustment = sum(abs(tactical_weights[t] - result.portfolio_weights[t])
                              for t in tactical_weights)

        print(f"  ✅ Tactical Allocation Complete:")
        print(f"     Regime: {regime}")
        print(f"     Total Adjustment: {total_adjustment*100:.1f}%")

    except Exception as e:
        print(f"⚠️ Tactical Allocation Error: {e}")
        import traceback
        traceback.print_exc()


def _run_stress_test(result: EIMASResult, enable: bool):
    """[Phase 6.3] 스트레스 테스트 (Optional)"""
    if not enable:
        return

    print("\n[Phase 6.3] Running Stress Testing...")

    try:
        portfolio_weights = result.portfolio_weights
        if not portfolio_weights:
            print("⚠️ No portfolio weights for stress test")
            return

        # Stress test engine
        engine = StressTestEngine(
            portfolio_weights=portfolio_weights,
            portfolio_value=1_000_000
        )

        # Run historical scenarios
        historical_results = engine.run_all_historical()

        # Run hypothetical scenarios
        hypothetical_results = engine.run_all_hypothetical()

        # Extreme scenario
        extreme_result = engine.extreme_scenario("severe")

        # Store results
        result.stress_test_results = {
            'historical': [r.to_dict() for r in historical_results],
            'hypothetical': [r.to_dict() for r in hypothetical_results],
            'extreme': extreme_result.to_dict()
        }

        # Find worst case
        all_results = historical_results + hypothetical_results + [extreme_result]
        worst_case = max(all_results, key=lambda r: r.loss_pct)

        print(f"  ✅ Stress Test Complete:")
        print(f"     Scenarios Tested: {len(all_results)}")
        print(f"     Worst Case: {worst_case.scenario_name} ({worst_case.loss_pct*100:.1f}%)")

    except Exception as e:
        print(f"⚠️ Stress Test Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# Main Pipeline
# ============================================================================

async def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    generate_report: bool = False,
    full_mode: bool = False,
    enable_backtest: bool = False,
    enable_attribution: bool = False,
    enable_stress_test: bool = False,
    quick_validation_mode: str = None
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
        Phase 8.5: Quick mode AI validation (KOSPI/SPX 분리, --quick1/--quick2)

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
        quick_validation_mode (str): Quick AI validation market focus ('KOSPI' or 'SPX').
                                    Quick 모드 검증 시장 (KOSPI/SPX). Default: None.
    
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

    # Phase 4.5: Operational Report (decision governance, rebalance)
    _generate_operational_report(result)

    # Phase 5: Storage
    output_file = _save_results(result, events)

    # Phase 6: Portfolio Theory Modules (Optional, 2026-02-04)
    _run_backtest(result, market_data, enable_backtest)
    _run_performance_attribution(result, enable_attribution)
    _run_stress_test(result, enable_stress_test)

    # Phase 7: AI Report Generation
    report_content = await _generate_report(result, market_data, generate_report)

    # Phase 8: Validation
    await _validate_report(result, report_content, generate_report)
    _run_ai_validation_phase(result, full_mode)

    # Phase 8.5: Quick Mode AI Validation (KOSPI/SPX 분리)
    _run_quick_validation(result, market_data, output_file, quick_validation_mode)

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

    # Quick Mode AI Validation (2026-02-04)
    parser.add_argument('--quick1', action='store_true', help='Quick AI validation (KOSPI focus)')
    parser.add_argument('--quick2', action='store_true', help='Quick AI validation (SPX focus)')

    # Portfolio Theory Modules (2026-02-04)
    parser.add_argument('--backtest', action='store_true', help='Run backtest engine (5-year historical)')
    parser.add_argument('--attribution', action='store_true', help='Performance attribution (Brinson)')
    parser.add_argument('--stress-test', action='store_true', help='Stress testing (historical + hypothetical)')

    args = parser.parse_args()

    is_short = args.short or args.quick

    # Determine market focus
    market_focus = None
    if args.quick1:
        market_focus = 'KOSPI'
    elif args.quick2:
        market_focus = 'SPX'

    asyncio.run(run_integrated_pipeline(
        enable_realtime=args.realtime,
        realtime_duration=args.duration,
        quick_mode=is_short,
        generate_report=not is_short,
        full_mode=args.full,
        enable_backtest=args.backtest,
        enable_attribution=args.attribution,
        enable_stress_test=args.stress_test,
        quick_validation_mode=market_focus
    ))


if __name__ == "__main__":
    main()
