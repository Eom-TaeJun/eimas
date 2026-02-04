#!/usr/bin/env python3
"""
Bubble Detection - Enums
============================================================

Bubble warning levels and risk signal types

Economic Foundation:
    - "Bubbles for Fama" (Greenwood et al. 2019)
    - JP Morgan 5-Stage Bubble Framework
"""

from enum import Enum


class BubbleWarningLevel(str, Enum):
    """버블 경고 수준"""
    NONE = "NONE"           # 버블 징후 없음
    WATCH = "WATCH"         # 관찰 필요 (Run-up만 충족)
    WARNING = "WARNING"     # 경고 (Run-up + 1개 위험 신호)
    DANGER = "DANGER"       # 위험 (Run-up + 2개 이상 위험 신호)


class RiskSignalType(str, Enum):
    """위험 신호 유형"""
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"     # 변동성 급등
    SHARE_ISSUANCE = "SHARE_ISSUANCE"         # 주식 발행 증가
    PRICE_ACCELERATION = "PRICE_ACCELERATION"  # 가격 가속화
    VOLUME_SURGE = "VOLUME_SURGE"             # 거래량 급증
    # JP Morgan 5-Stage Framework (NEW)
    PARADIGM_SHIFT = "PARADIGM_SHIFT"         # 패러다임 전환 (새 기술/산업)
    CREDIT_AVAILABILITY = "CREDIT_AVAILABILITY"  # 신용 가용성 확대
    LEVERAGE_GAP = "LEVERAGE_GAP"             # 레버리지/밸류에이션 괴리
    SPECULATIVE_LOOP = "SPECULATIVE_LOOP"     # 투기적 피드백 루프


class JPMorganBubbleStage(str, Enum):
    """JP Morgan 5단계 버블 프레임워크"""
    STAGE_1_PARADIGM = "Stage 1: Paradigm Shift"        # 패러다임 전환
    STAGE_2_CREDIT = "Stage 2: Credit Availability"    # 신용 가용성
    STAGE_3_LEVERAGE = "Stage 3: Leverage Gap"         # 레버리지 괴리
    STAGE_4_SPECULATION = "Stage 4: Speculative Loop"  # 투기적 피드백
    STAGE_5_COLLAPSE = "Stage 5: Collapse Trigger"     # 붕괴 트리거
    NONE = "No Bubble Stage"

