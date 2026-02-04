#!/usr/bin/env python3
"""
ETF Flow Analyzer - Enums
============================================================

Market sentiment and rotation enums
"""

from enum import Enum


class MarketSentiment(str, Enum):
    """시장 심리 상태"""
    RISK_ON = "risk_on"           # 위험 선호
    RISK_OFF = "risk_off"         # 위험 회피
    NEUTRAL = "neutral"           # 중립
    MIXED = "mixed"               # 혼조


class StyleRotation(str, Enum):
    """스타일 로테이션 상태"""
    GROWTH_LEADING = "growth_leading"    # 성장주 우위
    VALUE_LEADING = "value_leading"      # 가치주 우위
    LARGE_CAP_LEADING = "large_cap"      # 대형주 우위
    SMALL_CAP_LEADING = "small_cap"      # 소형주 우위
    NEUTRAL = "neutral"


class CyclePhase(str, Enum):
    """경기 사이클 단계 (섹터 로테이션 기반)"""
    EARLY_EXPANSION = "early_expansion"   # 초기 확장 (소비재, 금융 강세)
    MID_EXPANSION = "mid_expansion"       # 중기 확장 (기술, 산업재 강세)
    LATE_EXPANSION = "late_expansion"     # 후기 확장 (에너지, 원자재 강세)
    RECESSION = "recession"               # 침체 (방어주, 유틸리티 강세)
    UNCERTAIN = "uncertain"


