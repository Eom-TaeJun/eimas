#!/usr/bin/env python3
"""Rebalancing Strategy - Enums"""
from enum import Enum

class RebalanceFrequency(Enum):
    """리밸런싱 주기"""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    NEVER = "never"  # Buy & Hold


class RebalancePolicy(Enum):
    """리밸런싱 정책 유형"""
    PERIODIC = "periodic"           # 정기 리밸런싱
    THRESHOLD = "threshold"         # 편차 기반
    HYBRID = "hybrid"               # 정기 + 편차
    TACTICAL = "tactical"           # 시그널 기반 전술적

