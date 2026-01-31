#!/usr/bin/env python3
"""
EIMAS Pipeline - Debate Module
===============================

Purpose:
    Phase 3 멀티 에이전트 토론 담당 (Multi-Agent Debate)

Functions:
    - run_dual_mode_debate(market_data) -> DebateResult
    - extract_consensus(debate_result) -> Dict

Dependencies:
    - agents.orchestrator
    - lib.dual_mode_analyzer

Example:
    from pipeline.debate import run_dual_mode_debate
    result = await run_dual_mode_debate(market_data)
    print(result.final_recommendation)
"""

import asyncio
from typing import Dict, List, Any, Tuple
import pandas as pd

# EIMAS 라이브러리
from agents.orchestrator import MetaOrchestrator
from lib.dual_mode_analyzer import DualModeAnalyzer, ModeResult, AnalysisMode
from pipeline.schemas import DebateResult

async def run_single_mode(mode_name: str, lookback: int, query: str,
                          market_data: Dict[str, pd.DataFrame]) -> Tuple[ModeResult, Dict[str, Any]]:
    """
    단일 모드 분석 실행

    Returns:
        Tuple[ModeResult, Dict]: (ModeResult, full orchestrator result)
    """
    print(f"\n[{mode_name}] Running mode analysis (lookback={lookback})...")

    # 데이터 필터링 (이미 수집된 데이터에서 lookback 기간만큼 자르기)
    filtered_data = {}
    if market_data:
        for ticker, df in market_data.items():
            if not df.empty:
                # 최근 n일 데이터만 슬라이싱
                filtered_data[ticker] = df.tail(lookback)
    else:
        print(f"      ⚠ No market data provided for {mode_name}")
        filtered_data = {}

    try:
        orchestrator = MetaOrchestrator(verbose=False)
        result = await orchestrator.run_with_debate(query, filtered_data)

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

        mode_result = ModeResult(
            mode=AnalysisMode.FULL if mode_name == 'FULL' else AnalysisMode.REFERENCE,
            consensus=None,
            confidence=confidence,
            position=position,
            dissent_count=0,
            has_strong_dissent=False
        )

        return mode_result, result

    except Exception as e:
        print(f"      ✗ {mode_name} mode error: {e}")
        mode_result = ModeResult(
            mode=AnalysisMode.FULL if mode_name == 'FULL' else AnalysisMode.REFERENCE,
            consensus=None, confidence=0.5, position="NEUTRAL",
            dissent_count=0, has_strong_dissent=False
        )
        return mode_result, {}

async def run_dual_mode_debate(market_data: Dict[str, pd.DataFrame],
                               lookback_full: int = 365,
                               lookback_ref: int = 90,
                               extended_data: Dict[str, Any] = None) -> DebateResult:
    """
    듀얼 모드 토론 실행

    Phase 2-3 Enhanced:
    - Full orchestrator result에서 enhanced_debate, reasoning_chain 추출
    - 7개 에이전트 기여도 추적
    - Multi-LLM 토론 결과 포함
    - Extended Data (PCR, F&G, Credit Spreads 등) 컨텍스트 제공
    """
    print("\n" + "=" * 50)
    print("PHASE 3: MULTI-AGENT DEBATE")
    print("=" * 50)

    # Build query with extended data context
    query = "Analyze current market conditions, risks, and generate trading signals"
    if extended_data:
        ext_context = _build_extended_context(extended_data)
        if ext_context:
            query += f"\n\nExtended Market Context:\n{ext_context}"

    # 1. Full Mode (enhanced results 추출용)
    full_result, full_orch_result = await run_single_mode('FULL', lookback_full, query, market_data)

    # 2. Reference Mode (비교용)
    ref_result, _ = await run_single_mode('REFERENCE', lookback_ref, query, market_data)

    # 3. Compare
    print("\n[3.3] Comparing modes...")
    analyzer = DualModeAnalyzer()
    comparison = analyzer.compare_modes(full_result, ref_result)

    warnings = []
    if not comparison.positions_agree:
        warnings.append(f"Mode divergence: FULL={full_result.position}, REF={ref_result.position}")

    print(f"      ✓ Modes Agree: {'Yes' if comparison.positions_agree else 'NO'}")
    print(f"      ✓ Final Recommendation: {comparison.recommended_action}")

    # 4. Extract Enhanced Results from FULL mode orchestrator result
    print("\n[3.4] Extracting enhanced results...")

    enhanced_debate = full_orch_result.get('debate', {}).get('enhanced_debate', {})
    reasoning_chain = full_orch_result.get('reasoning_chain', [])
    agent_outputs = full_orch_result.get('agent_outputs', {})
    verification = full_orch_result.get('verification', {})
    metadata = full_orch_result.get('metadata', {})
    institutional_analysis = full_orch_result.get('institutional_analysis', {})

    # Display enhanced results summary
    interp = enhanced_debate.get('interpretation', {})
    method = enhanced_debate.get('methodology', {})

    if interp:
        print(f"      ✓ Interpretation: {interp.get('recommended_action', 'N/A')} (Schools: Monetarist/Keynesian/Austrian)")
    if method:
        print(f"      ✓ Methodology: {method.get('selected_methodology', 'N/A')}")
    print(f"      ✓ Reasoning Chain: {len(reasoning_chain)} steps tracked")

    if verification:
        v_score = verification.get('overall_score', 'N/A')
        v_passed = verification.get('passed', 'N/A')
        print(f"      ✓ Verification: Score={v_score}, Passed={v_passed}")

    if metadata:
        print(f"      ✓ Agents: {metadata.get('num_agents', 'N/A')}, Debates: {metadata.get('total_debates', 'N/A')}")

    # Display institutional analysis if available
    if institutional_analysis:
        narrative = institutional_analysis.get('narrative', '')
        methods = institutional_analysis.get('methodology_applied', [])
        print(f"      ✓ Institutional Methods: {', '.join(methods[:3])}")
        if narrative:
            print(f"      ✓ Narrative: {narrative[:80]}...")

    return DebateResult(
        full_mode_position=full_result.position,
        reference_mode_position=ref_result.position,
        modes_agree=comparison.positions_agree,
        final_recommendation=comparison.recommended_action,
        confidence=(full_result.confidence + ref_result.confidence) / 2,
        risk_level=comparison.risk_level,
        dissent_records=[],
        warnings=warnings,
        # Phase 2-3 Enhanced Fields
        enhanced_debate=enhanced_debate,
        reasoning_chain=reasoning_chain,
        agent_outputs=agent_outputs,
        verification=verification,
        metadata=metadata,
        institutional_analysis=institutional_analysis
    )

def extract_consensus(debate_result: DebateResult) -> Dict[str, Any]:
    """합의 결과 추출"""
    return {
        'action': debate_result.final_recommendation,
        'confidence': debate_result.confidence,
        'risk': debate_result.risk_level,
        # Enhanced fields
        'interpretation': debate_result.enhanced_debate.get('interpretation', {}),
        'methodology': debate_result.enhanced_debate.get('methodology', {}),
        'reasoning_steps': len(debate_result.reasoning_chain),
        'verification_passed': debate_result.verification.get('passed', None)
    }


def _build_extended_context(extended_data: Dict[str, Any]) -> str:
    """
    Extended Data를 AI 토론용 자연어 컨텍스트로 변환

    Args:
        extended_data: ExtendedDataCollector.collect_all() 결과

    Returns:
        AI 에이전트가 이해할 수 있는 자연어 요약
    """
    if not extended_data:
        return ""

    lines = []

    # 1. Options Sentiment
    pcr = extended_data.get('put_call_ratio', {})
    if pcr.get('ratio', 0) > 0:
        lines.append(f"- Options: Put/Call Ratio {pcr['ratio']:.2f} → {pcr.get('sentiment', 'N/A')}")

    # 2. Valuation
    fund = extended_data.get('fundamentals', {})
    if fund.get('earnings_yield', 0) > 0:
        lines.append(f"- Valuation: S&P500 Earnings Yield {fund['earnings_yield']:.2f}% (PE {fund.get('pe_ratio', 0):.1f})")

    # 3. Digital Liquidity
    stable = extended_data.get('digital_liquidity', {})
    if stable.get('total_mcap', 0) > 0:
        lines.append(f"- Digital Liquidity: Stablecoin Market Cap ${stable['total_mcap']/1e9:.1f}B")

    # 4. Credit Market
    credit = extended_data.get('credit_spreads', {})
    if credit.get('interpretation'):
        lines.append(f"- Credit Market: {credit['interpretation']} (HYG/IEF 5d change: {credit.get('change_5d', 0):.2f}%)")

    # 5. Crypto Sentiment
    fng = extended_data.get('crypto_fng', {})
    if fng.get('value', 0) > 0:
        lines.append(f"- Crypto Sentiment: Fear & Greed Index {fng['value']} ({fng.get('classification', 'N/A')})")

    # 6. DeFi Health
    tvl = extended_data.get('defi_tvl', {})
    if tvl.get('total_tvl', 0) > 0:
        lines.append(f"- DeFi Health: Total TVL ${tvl['total_tvl']/1e9:.1f}B")

    # 7. Short Interest
    depth = extended_data.get('market_depth', {})
    spy_short = depth.get('SPY_short_float', 0)
    if spy_short and spy_short > 0:
        lines.append(f"- Short Interest: SPY {spy_short*100:.2f}%, TSLA {depth.get('TSLA_short_float', 0)*100:.2f}%")

    # 8. News Sentiment
    news = extended_data.get('news_sentiment', {})
    if news.get('label'):
        lines.append(f"- News Sentiment: {news['label']} (Score: {news.get('score', 0):.2f}, Headline: '{news.get('top_headline', '')[:50]}...')")

    # 9. Emerging Market Risk (KRW)
    krw = extended_data.get('korea_risk', {})
    if krw.get('status'):
        lines.append(f"- EM Risk (KRW): {krw['status']} (Volatility: {krw.get('volatility', 0):.2f}%)")

    return "\n".join(lines) if lines else ""
