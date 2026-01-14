#!/usr/bin/env python3
"""
Dual Mode Analyzer
==================
두 가지 분석 모드를 병렬로 실행하고 결과를 비교

모드:
- FULL: 2024-2025 역사적 데이터를 주요 입력으로 사용 (기존)
- REFERENCE: 역사적 데이터를 참고용으로만 사용 (새로운 방식)

사용법:
    analyzer = DualModeAnalyzer()
    results = await analyzer.run_both_modes(topic, context)
    comparison = analyzer.compare_results(results)
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import (
    AnalysisMode,
    HistoricalDataConfig,
    Consensus,
    AgentOpinion,
)


@dataclass
class ModeResult:
    """단일 모드 분석 결과"""
    mode: AnalysisMode
    consensus: Optional[Consensus]
    confidence: float
    position: str
    dissent_count: int
    has_strong_dissent: bool
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DualModeComparison:
    """두 모드 비교 결과"""
    full_result: ModeResult
    reference_result: ModeResult

    # 비교 분석
    positions_agree: bool = False
    confidence_delta: float = 0.0
    recommended_action: str = "HOLD"
    recommendation_reason: str = ""
    risk_level: str = "MEDIUM"

    def to_dict(self) -> Dict:
        return {
            'full_mode': {
                'position': self.full_result.position,
                'confidence': self.full_result.confidence,
                'has_strong_dissent': self.full_result.has_strong_dissent,
            },
            'reference_mode': {
                'position': self.reference_result.position,
                'confidence': self.reference_result.confidence,
                'has_strong_dissent': self.reference_result.has_strong_dissent,
            },
            'comparison': {
                'positions_agree': self.positions_agree,
                'confidence_delta': self.confidence_delta,
                'recommended_action': self.recommended_action,
                'recommendation_reason': self.recommendation_reason,
                'risk_level': self.risk_level,
            }
        }


class DualModeAnalyzer:
    """
    두 가지 분석 모드를 병렬 실행하고 비교

    핵심 원칙:
    1. 두 모드가 동의 → 높은 신뢰도
    2. 두 모드가 불일치 → 주의 필요 (Regime 변화 가능성)
    3. REFERENCE 모드가 반대 → 역사적 패턴 무효화 가능성
    """

    def __init__(self):
        self.full_config = HistoricalDataConfig(mode=AnalysisMode.FULL)
        self.reference_config = HistoricalDataConfig(mode=AnalysisMode.REFERENCE)

    def apply_historical_weight(
        self,
        historical_signal: float,
        realtime_signal: float,
        config: HistoricalDataConfig
    ) -> float:
        """
        역사적/실시간 신호에 가중치 적용

        Args:
            historical_signal: 역사적 데이터 기반 신호 (-1 to 1)
            realtime_signal: 실시간 데이터 기반 신호 (-1 to 1)
            config: 가중치 설정

        Returns:
            가중 평균 신호 (-1 to 1)
        """
        combined = (
            historical_signal * config.historical_weight +
            realtime_signal * config.realtime_weight
        )
        return max(-1.0, min(1.0, combined))

    def compare_modes(
        self,
        full_result: ModeResult,
        reference_result: ModeResult
    ) -> DualModeComparison:
        """
        두 모드 결과 비교 및 최종 권고 생성

        비교 로직:
        1. 두 모드 일치 + 높은 신뢰도 → 강한 신호
        2. 두 모드 일치 + 낮은 신뢰도 → 약한 신호
        3. 두 모드 불일치 → Regime 변화 가능, 주의
        4. REFERENCE만 반대 → 역사적 패턴 의문
        5. 강한 반대의견 존재 → 추가 경고
        """
        comparison = DualModeComparison(
            full_result=full_result,
            reference_result=reference_result
        )

        # 포지션 일치 여부
        comparison.positions_agree = (full_result.position == reference_result.position)
        comparison.confidence_delta = abs(full_result.confidence - reference_result.confidence)

        # 권고 생성
        if comparison.positions_agree:
            avg_confidence = (full_result.confidence + reference_result.confidence) / 2

            if avg_confidence >= 0.7:
                comparison.recommended_action = full_result.position
                comparison.recommendation_reason = f"Both modes agree with high confidence ({avg_confidence:.0%})"
                comparison.risk_level = "LOW"
            elif avg_confidence >= 0.5:
                comparison.recommended_action = full_result.position
                comparison.recommendation_reason = f"Both modes agree with moderate confidence ({avg_confidence:.0%})"
                comparison.risk_level = "MEDIUM"
            else:
                comparison.recommended_action = "HOLD"
                comparison.recommendation_reason = f"Low confidence despite agreement ({avg_confidence:.0%})"
                comparison.risk_level = "MEDIUM"
        else:
            # 모드 불일치 - 중요한 경고 신호
            comparison.recommended_action = "HOLD"
            comparison.risk_level = "HIGH"

            if reference_result.confidence > full_result.confidence:
                comparison.recommendation_reason = (
                    f"MODE DIVERGENCE: FULL={full_result.position} vs REF={reference_result.position}. "
                    f"Reference mode has higher confidence - historical patterns may be outdated."
                )
            else:
                comparison.recommendation_reason = (
                    f"MODE DIVERGENCE: FULL={full_result.position} vs REF={reference_result.position}. "
                    f"Possible regime change - proceed with caution."
                )

        # 강한 반대의견 경고
        if full_result.has_strong_dissent or reference_result.has_strong_dissent:
            comparison.risk_level = "HIGH"
            comparison.recommendation_reason += " ⚠️ STRONG DISSENT EXISTS."

        return comparison

    def generate_dual_report(self, comparison: DualModeComparison) -> str:
        """
        두 모드 비교 리포트 생성
        """
        report = []
        report.append("=" * 60)
        report.append("DUAL MODE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        # FULL 모드 결과
        report.append("📊 FULL MODE (Historical Data = 70%)")
        report.append(f"   Position: {comparison.full_result.position}")
        report.append(f"   Confidence: {comparison.full_result.confidence:.0%}")
        report.append(f"   Dissent Count: {comparison.full_result.dissent_count}")
        if comparison.full_result.has_strong_dissent:
            report.append("   ⚠️ STRONG DISSENT EXISTS")
        report.append("")

        # REFERENCE 모드 결과
        report.append("🔍 REFERENCE MODE (Historical Data = 20%)")
        report.append(f"   Position: {comparison.reference_result.position}")
        report.append(f"   Confidence: {comparison.reference_result.confidence:.0%}")
        report.append(f"   Dissent Count: {comparison.reference_result.dissent_count}")
        if comparison.reference_result.has_strong_dissent:
            report.append("   ⚠️ STRONG DISSENT EXISTS")
        report.append("")

        # 비교 분석
        report.append("📋 COMPARISON")
        report.append(f"   Modes Agree: {'✓' if comparison.positions_agree else '✗'}")
        report.append(f"   Confidence Delta: {comparison.confidence_delta:.0%}")
        report.append(f"   Risk Level: {comparison.risk_level}")
        report.append("")

        # 최종 권고
        report.append("🎯 RECOMMENDATION")
        report.append(f"   Action: {comparison.recommended_action}")
        report.append(f"   Reason: {comparison.recommendation_reason}")
        report.append("")

        report.append("=" * 60)

        return "\n".join(report)


def create_mock_results() -> Tuple[ModeResult, ModeResult]:
    """테스트용 목 결과 생성"""
    full_result = ModeResult(
        mode=AnalysisMode.FULL,
        consensus=None,
        confidence=0.75,
        position="BULLISH",
        dissent_count=1,
        has_strong_dissent=False,
        warnings=[]
    )

    reference_result = ModeResult(
        mode=AnalysisMode.REFERENCE,
        consensus=None,
        confidence=0.60,
        position="NEUTRAL",  # 다른 결과!
        dissent_count=2,
        has_strong_dissent=True,
        warnings=["Regime change detected"]
    )

    return full_result, reference_result


if __name__ == "__main__":
    print("=== Dual Mode Analyzer Test ===\n")

    analyzer = DualModeAnalyzer()

    # 테스트 1: 모드 불일치 시나리오
    print("Test 1: Mode Divergence Scenario")
    print("-" * 40)
    full_result, reference_result = create_mock_results()
    comparison = analyzer.compare_modes(full_result, reference_result)
    print(analyzer.generate_dual_report(comparison))

    # 테스트 2: 모드 일치 시나리오
    print("\nTest 2: Mode Agreement Scenario")
    print("-" * 40)
    reference_result.position = "BULLISH"
    reference_result.confidence = 0.70
    reference_result.has_strong_dissent = False
    comparison2 = analyzer.compare_modes(full_result, reference_result)
    print(analyzer.generate_dual_report(comparison2))

    # 가중치 테스트
    print("\nTest 3: Weight Application")
    print("-" * 40)

    historical_signal = 0.8  # 역사적 데이터: 강한 매수
    realtime_signal = -0.3   # 실시간 데이터: 약한 매도

    full_combined = analyzer.apply_historical_weight(
        historical_signal, realtime_signal, analyzer.full_config
    )
    ref_combined = analyzer.apply_historical_weight(
        historical_signal, realtime_signal, analyzer.reference_config
    )

    print(f"Historical Signal: {historical_signal:+.2f}")
    print(f"Realtime Signal:   {realtime_signal:+.2f}")
    print(f"FULL Mode Combined:      {full_combined:+.2f} (hist=70%)")
    print(f"REFERENCE Mode Combined: {ref_combined:+.2f} (hist=20%)")
