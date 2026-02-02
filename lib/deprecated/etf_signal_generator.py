#!/usr/bin/env python3
"""
ETF Signal Generator
=====================
ETF Flow 분석 결과를 Signal-Action 프레임워크와 연동

ETFFlowAnalyzer의 결과를 EnhancedSignal로 변환하여
SignalActionMapper에서 사용할 수 있게 함
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from lib.etf_flow_analyzer import (
    ETFFlowAnalyzer,
    MarketRegimeResult,
    FlowComparison,
    SectorRotationResult,
    MarketSentiment,
    CyclePhase,
)
from core.signal_action import (
    EnhancedSignal,
    PositionDirection,
    SignalActionMapper,
    RiskProfile,
    Action,
)


class ETFSignalGenerator:
    """
    ETF Flow 분석 결과 → EnhancedSignal 변환기

    변환 규칙:
    1. Flow Comparison → 개별 자산 신호
    2. Sector Rotation → 섹터 ETF 신호
    3. Market Regime → 시장 전체 신호
    """

    # 신호 유형별 기본 신뢰도
    BASE_CONFIDENCE = {
        "comparison": 0.60,
        "sector_rotation": 0.55,
        "market_regime": 0.65,
    }

    def __init__(self, analyzer: ETFFlowAnalyzer):
        """
        Args:
            analyzer: ETFFlowAnalyzer 인스턴스
        """
        self.analyzer = analyzer

    def generate_comparison_signals(self, comparisons: List[FlowComparison]) -> List[EnhancedSignal]:
        """
        비교 분석 결과 → EnhancedSignal 변환

        각 비교 쌍에서 선도 ETF에 대한 신호 생성
        """
        signals = []

        for comp in comparisons:
            if comp.signal == "NEUTRAL":
                continue

            # 방향 결정
            if comp.signal == "A_LEADING":
                ticker = comp.etf_a
                direction = PositionDirection.LONG
                z_score = comp.ratio_z_score
            else:  # B_LEADING
                ticker = comp.etf_b
                direction = PositionDirection.LONG
                z_score = -comp.ratio_z_score

            # 신뢰도 계산
            base_conf = self.BASE_CONFIDENCE["comparison"]
            strength_boost = min(comp.strength * 0.2, 0.2)
            spread_boost = min(abs(comp.spread_20d) * 0.02, 0.15)
            confidence = min(base_conf + strength_boost + spread_boost, 0.90)

            signal = EnhancedSignal(
                signal_id="",
                type="etf_flow",
                ticker=ticker,
                name=f"{comp.pair_name} - {ticker}",
                indicator="relative_strength",
                value=comp.spread_20d,
                threshold=3.0,  # 3% 스프레드 기준
                z_score=z_score,
                level="ALERT" if abs(z_score) > 1.5 else "WARNING",
                description=comp.interpretation,
                confidence=round(confidence, 2),
                direction=direction,
                horizon="short",
                source="etf_flow_analyzer",
                metadata={
                    "pair_name": comp.pair_name,
                    "spread_1d": comp.spread_1d,
                    "spread_5d": comp.spread_5d,
                    "spread_20d": comp.spread_20d,
                }
            )
            signals.append(signal)

        return signals

    def generate_sector_signals(self, sector_result: SectorRotationResult) -> List[EnhancedSignal]:
        """
        섹터 로테이션 결과 → EnhancedSignal 변환

        선도 섹터에 LONG, 후행 섹터에 SHORT 신호
        """
        signals = []

        if sector_result.confidence < 0.4:
            return signals

        # 섹터 이름 매핑
        sector_names = {
            "XLK": "Technology", "XLF": "Financials", "XLV": "Healthcare",
            "XLE": "Energy", "XLI": "Industrials", "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples", "XLU": "Utilities", "XLB": "Materials",
            "XLRE": "Real Estate", "XLC": "Communication Services"
        }

        # 선도 섹터 → LONG
        for ticker in sector_result.leading_sectors[:2]:  # 상위 2개만
            rank = sector_result.sector_rankings.get(ticker, 6)
            confidence = self.BASE_CONFIDENCE["sector_rotation"] + (6 - rank) * 0.05
            confidence = min(confidence, 0.80)

            signal = EnhancedSignal(
                signal_id="",
                type="sector_rotation",
                ticker=ticker,
                name=f"Sector Leader - {sector_names.get(ticker, ticker)}",
                indicator="sector_rank",
                value=float(rank),
                threshold=3.0,
                z_score=2.0 - rank * 0.3,  # 순위가 높을수록 Z-score 높음
                level="ALERT" if rank <= 2 else "WARNING",
                description=f"{sector_result.interpretation} - {sector_names.get(ticker, ticker)} 선도",
                confidence=round(confidence, 2),
                direction=PositionDirection.LONG,
                horizon="short",
                source="etf_flow_analyzer",
                metadata={
                    "cycle_phase": sector_result.cycle_phase.value,
                    "offensive_score": sector_result.offensive_score,
                    "defensive_score": sector_result.defensive_score,
                }
            )
            signals.append(signal)

        # 후행 섹터 → SHORT (약한 신호)
        for ticker in sector_result.lagging_sectors[:1]:  # 하위 1개만
            rank = sector_result.sector_rankings.get(ticker, 6)
            confidence = self.BASE_CONFIDENCE["sector_rotation"] + 0.05

            signal = EnhancedSignal(
                signal_id="",
                type="sector_rotation",
                ticker=ticker,
                name=f"Sector Laggard - {sector_names.get(ticker, ticker)}",
                indicator="sector_rank",
                value=float(rank),
                threshold=8.0,
                z_score=-(rank - 6) * 0.3,  # 순위가 낮을수록 음의 Z-score
                level="WARNING",
                description=f"{sector_result.interpretation} - {sector_names.get(ticker, ticker)} 후행",
                confidence=round(confidence, 2),
                direction=PositionDirection.SHORT,
                horizon="short",
                source="etf_flow_analyzer",
                metadata={
                    "cycle_phase": sector_result.cycle_phase.value,
                }
            )
            signals.append(signal)

        return signals

    def generate_regime_signals(self, regime: MarketRegimeResult) -> List[EnhancedSignal]:
        """
        시장 레짐 결과 → EnhancedSignal 변환

        전체 시장 방향 신호 생성
        """
        signals = []

        # Risk Appetite 기반 시장 신호
        if regime.sentiment == MarketSentiment.RISK_ON and regime.risk_appetite_score > 60:
            signal = EnhancedSignal(
                signal_id="",
                type="market_regime",
                ticker="SPY",
                name="Market Risk-On Signal",
                indicator="risk_appetite",
                value=regime.risk_appetite_score,
                threshold=60.0,
                z_score=(regime.risk_appetite_score - 50) / 15,
                level="ALERT",
                description=f"Risk-On: Risk Appetite {regime.risk_appetite_score:.0f}/100, Breadth {regime.breadth_score:.0f}%",
                confidence=round(min(self.BASE_CONFIDENCE["market_regime"] + regime.confidence * 0.2, 0.85), 2),
                direction=PositionDirection.LONG,
                horizon="short",
                source="etf_flow_analyzer",
                regime_aligned=True,
                metadata={
                    "sentiment": regime.sentiment.value,
                    "breadth_score": regime.breadth_score,
                    "equity_bond_spread": regime.equity_bond_spread,
                }
            )
            signals.append(signal)

        elif regime.sentiment == MarketSentiment.RISK_OFF and regime.risk_appetite_score < 40:
            signal = EnhancedSignal(
                signal_id="",
                type="market_regime",
                ticker="TLT",  # 국채 ETF
                name="Market Risk-Off Signal",
                indicator="risk_appetite",
                value=regime.risk_appetite_score,
                threshold=40.0,
                z_score=(50 - regime.risk_appetite_score) / 15,
                level="ALERT",
                description=f"Risk-Off: Risk Appetite {regime.risk_appetite_score:.0f}/100, 안전자산 선호",
                confidence=round(min(self.BASE_CONFIDENCE["market_regime"] + regime.confidence * 0.2, 0.85), 2),
                direction=PositionDirection.LONG,
                horizon="short",
                source="etf_flow_analyzer",
                regime_aligned=True,
                metadata={
                    "sentiment": regime.sentiment.value,
                    "hy_treasury_spread": regime.hy_treasury_spread,
                }
            )
            signals.append(signal)

        # HY-Treasury 스프레드 기반 신용 신호
        if abs(regime.hy_treasury_spread) > 3:
            if regime.hy_treasury_spread > 3:
                # HY 강세 → 신용 위험 감소
                ticker = "HYG"
                direction = PositionDirection.LONG
                desc = f"신용 스프레드 축소: HY-Treasury +{regime.hy_treasury_spread:.1f}%"
            else:
                # Treasury 강세 → 안전자산 선호
                ticker = "TLT"
                direction = PositionDirection.LONG
                desc = f"안전자산 선호: HY-Treasury {regime.hy_treasury_spread:.1f}%"

            signal = EnhancedSignal(
                signal_id="",
                type="credit_spread",
                ticker=ticker,
                name=f"Credit Spread Signal - {ticker}",
                indicator="hy_treasury_spread",
                value=regime.hy_treasury_spread,
                threshold=3.0,
                z_score=regime.hy_treasury_spread / 2,
                level="ALERT" if abs(regime.hy_treasury_spread) > 5 else "WARNING",
                description=desc,
                confidence=round(0.65 + abs(regime.hy_treasury_spread) * 0.02, 2),
                direction=direction,
                horizon="short",
                source="etf_flow_analyzer",
            )
            signals.append(signal)

        return signals

    def generate_all_signals(self, analysis_result: Dict[str, Any]) -> List[EnhancedSignal]:
        """
        전체 분석 결과 → EnhancedSignal 목록 변환

        Args:
            analysis_result: ETFFlowAnalyzer.run_full_analysis() 결과

        Returns:
            EnhancedSignal 목록
        """
        all_signals = []

        # 1. Comparison 신호
        if 'comparisons' in analysis_result:
            comparisons = [FlowComparison(**c) for c in analysis_result['comparisons']]
            all_signals.extend(self.generate_comparison_signals(comparisons))

        # 2. Sector Rotation 신호
        if 'sector_rotation' in analysis_result:
            sr = analysis_result['sector_rotation']
            sector_result = SectorRotationResult(
                cycle_phase=CyclePhase(sr['cycle_phase']),
                leading_sectors=sr['leading_sectors'],
                lagging_sectors=sr['lagging_sectors'],
                sector_rankings=sr['sector_rankings'],
                offensive_score=sr['offensive_score'],
                defensive_score=sr['defensive_score'],
                confidence=sr['confidence'],
                interpretation=sr['interpretation']
            )
            all_signals.extend(self.generate_sector_signals(sector_result))

        # 3. Market Regime 신호
        if 'market_regime' in analysis_result:
            mr = analysis_result['market_regime']
            regime_result = MarketRegimeResult(
                sentiment=MarketSentiment(mr['sentiment']),
                style_rotation=mr['style_rotation'],
                cycle_phase=CyclePhase(mr['cycle_phase']),
                risk_appetite_score=mr['risk_appetite_score'],
                breadth_score=mr['breadth_score'],
                growth_value_spread=mr['growth_value_spread'],
                large_small_spread=mr['large_small_spread'],
                us_global_spread=mr['us_global_spread'],
                equity_bond_spread=mr['equity_bond_spread'],
                hy_treasury_spread=mr['hy_treasury_spread'],
                signals=mr['signals'],
                warnings=mr['warnings'],
                confidence=mr['confidence']
            )
            all_signals.extend(self.generate_regime_signals(regime_result))

        return all_signals


def run_integrated_analysis(risk_profile: RiskProfile = None) -> Dict[str, Any]:
    """
    ETF Flow 분석 → Signal 생성 → Action 매핑 통합 실행

    Args:
        risk_profile: 사용자 리스크 프로파일 (기본: Moderate)

    Returns:
        {
            "etf_analysis": ETF 분석 결과,
            "signals": 생성된 신호 목록,
            "actions": 권고 액션 목록,
            "summary": 요약
        }
    """
    print("=" * 70)
    print("Integrated ETF Flow → Signal → Action Analysis")
    print("=" * 70)

    # 1. ETF Flow 분석
    print("\n[1] Running ETF Flow Analysis...")
    analyzer = ETFFlowAnalyzer(lookback_days=90)
    etf_result = analyzer.run_full_analysis()

    # 2. Signal 생성
    print("\n[2] Generating Signals...")
    signal_generator = ETFSignalGenerator(analyzer)
    signals = signal_generator.generate_all_signals(etf_result)
    print(f"    Generated {len(signals)} signals")

    # 3. Signal → Action 매핑
    print("\n[3] Mapping Signals to Actions...")
    if risk_profile is None:
        risk_profile = RiskProfile.moderate()

    # VIX 설정 (ETF 분석에서 추정)
    vix_estimate = 100 - etf_result['market_regime']['risk_appetite_score']
    vix_estimate = max(12, min(40, vix_estimate * 0.5))  # 대략적 추정

    mapper = SignalActionMapper(risk_profile=risk_profile)
    mapper.set_regime(vix=vix_estimate)

    actions = mapper.process_signals(signals)
    print(f"    Generated {len(actions)} actions")

    # 4. 결과 요약
    summary = {
        "market_sentiment": etf_result['market_regime']['sentiment'],
        "cycle_phase": etf_result['market_regime']['cycle_phase'],
        "risk_appetite": etf_result['market_regime']['risk_appetite_score'],
        "total_signals": len(signals),
        "total_actions": len(actions),
        "action_summary": mapper.get_summary(),
    }

    return {
        "etf_analysis": etf_result,
        "signals": [s.to_dict() for s in signals],
        "actions": [a.to_dict() for a in actions],
        "summary": summary,
    }


def print_integrated_results(result: Dict[str, Any]):
    """통합 분석 결과 출력"""
    print("\n" + "=" * 70)
    print("INTEGRATED ANALYSIS RESULTS")
    print("=" * 70)

    summary = result['summary']
    print(f"\n[Market Overview]")
    print(f"  Sentiment: {summary['market_sentiment']}")
    print(f"  Cycle Phase: {summary['cycle_phase']}")
    print(f"  Risk Appetite: {summary['risk_appetite']:.1f}/100")

    print(f"\n[Signals Generated: {summary['total_signals']}]")
    for sig in result['signals'][:5]:  # 상위 5개만
        conf = sig['confidence']
        print(f"  • {sig['ticker']:6s} {sig['direction']:5s} "
              f"Conf:{conf:.0%} - {sig['description'][:40]}...")

    print(f"\n[Actions Recommended: {summary['total_actions']}]")
    for act in result['actions']:
        print(f"  → {act['ticker']:6s} {act['action_type']:15s} "
              f"Size:{act['position_size']:.1%} {act['direction']:5s}")

    if result['actions']:
        print(f"\n[Action Summary]")
        action_sum = summary['action_summary']
        print(f"  Risk Profile: {action_sum['risk_profile']}")
        print(f"  Current Regime: {action_sum['current_regime']}")
        print(f"  Avg Position Size: {action_sum['avg_position_size']:.1%}")

    # 경고 출력
    warnings = result['etf_analysis']['market_regime'].get('warnings', [])
    if warnings:
        print(f"\n[⚠ Warnings]")
        for warn in warnings:
            print(f"  • {warn}")

    print("\n" + "=" * 70)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # 통합 분석 실행
    result = run_integrated_analysis()

    # 결과 출력
    print_integrated_results(result)

    print("\nIntegration Test Complete!")
