#!/usr/bin/env python3
"""
Operational - Enumerations
===========================

운영 엔진에서 사용하는 Enum 타입 정의

Enums:
    - FinalStance: 최종 투자 스탠스 (BULLISH/BEARISH/HOLD)
    - ReasonCode: 스탠스 결정 근거 코드
    - TriggerType: 신호 트리거 유형
    - SignalType: 신호 타입 (BUY/SELL/HOLD/NEUTRAL)

Design Philosophy:
    - Explicit over implicit (명시적 열거형으로 가독성 향상)
    - Type safety (타입 안전성 확보)
    - Auditable decisions (감사 가능한 결정 추적)
"""

from enum import Enum


class FinalStance(Enum):
    """최종 투자 스탠스"""
    BULLISH = "BULLISH"   # 적극 매수 (신뢰도 높음)
    BEARISH = "BEARISH"   # 적극 매도 (위험 높음)
    HOLD = "HOLD"         # 유지 (불확실성 높음)


class ReasonCode(Enum):
    """스탠스 결정 근거 코드"""
    # BULLISH 근거
    HIGH_CONFIDENCE_BULLISH = "HIGH_CONFIDENCE_BULLISH"  # 높은 신뢰도 강세
    LOW_RISK_BULLISH = "LOW_RISK_BULLISH"                # 낮은 리스크 강세
    REGIME_BULLISH = "REGIME_BULLISH"                    # 강세장 레짐

    # BEARISH 근거
    HIGH_CONFIDENCE_BEARISH = "HIGH_CONFIDENCE_BEARISH"  # 높은 신뢰도 약세
    HIGH_RISK_BEARISH = "HIGH_RISK_BEARISH"              # 높은 리스크 약세
    REGIME_BEARISH = "REGIME_BEARISH"                    # 약세장 레짐

    # HOLD 근거
    LOW_CONFIDENCE = "LOW_CONFIDENCE"                    # 낮은 신뢰도
    CONFLICTING_SIGNALS = "CONFLICTING_SIGNALS"          # 신호 충돌
    REGIME_TRANSITION = "REGIME_TRANSITION"              # 레짐 전환기
    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"              # 데이터 부족
    HUMAN_OVERRIDE = "HUMAN_OVERRIDE"                    # 수동 개입


class TriggerType(Enum):
    """신호 트리거 유형"""
    REGIME_CHANGE = "REGIME_CHANGE"      # 레짐 변화
    RISK_SPIKE = "RISK_SPIKE"            # 리스크 급증
    CONFIDENCE_DROP = "CONFIDENCE_DROP"  # 신뢰도 급락
    SIGNAL_CONFLICT = "SIGNAL_CONFLICT"  # 신호 충돌
    THRESHOLD_BREACH = "THRESHOLD_BREACH"  # 임계값 위반


class SignalType(Enum):
    """신호 타입"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"
