#!/usr/bin/env python3
"""
EIMAS Pipeline Debate
======================
Phase 3: 멀티 에이전트 토론 모듈
"""

import asyncio
from typing import Dict, List, Any
import pandas as pd

# EIMAS 라이브러리
from agents.orchestrator import MetaOrchestrator
from lib.dual_mode_analyzer import DualModeAnalyzer, ModeResult, AnalysisMode
from lib.data_collector import DataManager
from pipeline.schemas import DebateResult

async def run_single_mode(mode_name: str, lookback: int, query: str, 
                          market_data: Dict[str, pd.DataFrame] = None) -> ModeResult:
    """단일 모드 분석 실행"""
    print(f"\n[{mode_name}] Running mode analysis (lookback={lookback})...")
    
    # 데이터가 없으면 수집 (Reference mode용)
    if market_data is None:
        dm = DataManager(lookback_days=lookback)
        tickers_config = {
            'market': [
                {'ticker': 'SPY'}, {'ticker': 'QQQ'}, {'ticker': 'IWM'},
                {'ticker': 'TLT'}, {'ticker': 'GLD'}
            ]
        }
        market_data, _ = dm.collect_all(tickers_config)

    try:
        orchestrator = MetaOrchestrator(verbose=False)
        result = await orchestrator.run_with_debate(query, market_data)
        
        # 결과 파싱
        consensus_by_topic = result.get('debate', {}).get('consensus', {})
        analysis_data = result.get('analysis', {})
        
        # 신뢰도 계산
        confidence = 0.5
        if consensus_by_topic:
            confidences = [c.get('confidence', 0.5) for c in consensus_by_topic.values()]
            confidence = sum(confidences) / len(confidences)
            
        # 포지션 결정
        regime = analysis_data.get('current_regime', 'NEUTRAL')
        risk_score = analysis_data.get('total_risk_score', 50)
        regime_conf = analysis_data.get('regime_confidence', 50) / 100
        
        position = 'NEUTRAL'
        if regime == 'BULL' and risk_score < 30:
            position = 'BULLISH'
            confidence = max(confidence, regime_conf)
        elif regime == 'BEAR' or risk_score > 70:
            position = 'BEARISH'
            confidence = max(confidence, regime_conf)
            
        print(f"      ✓ {mode_name} Position: {position} (Conf: {confidence:.0%})")
        
        return ModeResult(
            mode=AnalysisMode.FULL if mode_name == 'FULL' else AnalysisMode.REFERENCE,
            consensus=None,
            confidence=confidence,
            position=position,
            dissent_count=0, # 실제 dissent 로직은 orchestrator 내부에 있으나 여기선 간소화
            has_strong_dissent=False
        )
        
    except Exception as e:
        print(f"      ✗ {mode_name} mode error: {e}")
        return ModeResult(
            mode=AnalysisMode.FULL if mode_name == 'FULL' else AnalysisMode.REFERENCE,
            consensus=None, confidence=0.5, position="NEUTRAL",
            dissent_count=0, has_strong_dissent=False
        )

async def run_dual_mode_debate(market_data: Dict[str, pd.DataFrame], 
                               lookback_full: int = 365, 
                               lookback_ref: int = 90) -> DebateResult:
    """듀얼 모드 토론 실행"""
    print("\n" + "=" * 50)
    print("PHASE 3: MULTI-AGENT DEBATE")
    print("=" * 50)
    
    query = "Analyze current market conditions, risks, and generate trading signals"
    
    # 1. Full Mode
    full_result = await run_single_mode('FULL', lookback_full, query, market_data)
    
    # 2. Reference Mode (데이터 별도 수집)
    ref_result = await run_single_mode('REFERENCE', lookback_ref, query, None)
    
    # 3. Compare
    print("\n[3.3] Comparing modes...")
    analyzer = DualModeAnalyzer()
    comparison = analyzer.compare_modes(full_result, ref_result)
    
    warnings = []
    if not comparison.positions_agree:
        warnings.append(f"Mode divergence: FULL={full_result.position}, REF={ref_result.position}")
        
    print(f"      ✓ Modes Agree: {'Yes' if comparison.positions_agree else 'NO'}")
    print(f"      ✓ Final Recommendation: {comparison.recommended_action}")
    
    return DebateResult(
        full_mode_position=full_result.position,
        reference_mode_position=ref_result.position,
        modes_agree=comparison.positions_agree,
        final_recommendation=comparison.recommended_action,
        confidence=(full_result.confidence + ref_result.confidence) / 2,
        risk_level=comparison.risk_level,
        dissent_records=[],
        warnings=warnings
    )

def extract_consensus(debate_result: DebateResult) -> Dict[str, Any]:
    """합의 결과 추출"""
    return {
        'action': debate_result.final_recommendation,
        'confidence': debate_result.confidence,
        'risk': debate_result.risk_level
    }
