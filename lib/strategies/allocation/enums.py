#!/usr/bin/env python3
"""Allocation Strategy - Enums"""
from enum import Enum

class AllocationStrategy(Enum):
    """배분 전략 유형"""
    MVO_MAX_SHARPE = "mvo_max_sharpe"       # 샤프 비율 최대화
    MVO_MIN_VARIANCE = "mvo_min_variance"   # 최소 분산
    MVO_MAX_RETURN = "mvo_max_return"       # 수익률 최대화 (주어진 변동성)
    RISK_PARITY = "risk_parity"             # 동일 리스크 기여도
    HRP = "hrp"                             # 계층적 리스크 패리티
    EQUAL_WEIGHT = "equal_weight"           # 균등 배분
    INVERSE_VOLATILITY = "inverse_vol"      # 변동성 역수 비중
    BLACK_LITTERMAN = "black_litterman"     # Black-Litterman (views 필요)

