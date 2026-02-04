#!/usr/bin/env python3
"""Allocation Strategy - Data Schemas"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from .enums import AllocationStrategy

@dataclass
class AllocationConstraints:
    """
    배분 제약 조건

    Attributes:
        min_weight: 최소 비중 (기본 0)
        max_weight: 최대 비중 (기본 1)
        sum_to_one: 비중 합 = 1 강제 (기본 True)
        long_only: 공매도 금지 (기본 True)
        max_turnover: 최대 회전율 (리밸런싱용)
        sector_limits: 섹터별 최대 비중 {섹터명: 최대비중}
        asset_limits: 자산별 비중 제한 {자산명: (min, max)}
    """
    min_weight: float = 0.0
    max_weight: float = 1.0
    sum_to_one: bool = True
    long_only: bool = True
    max_turnover: Optional[float] = None
    sector_limits: Optional[Dict[str, float]] = None
    asset_limits: Optional[Dict[str, Tuple[float, float]]] = None


@dataclass
class AllocationResult:
    """
    배분 결과

    Attributes:
        weights: 자산별 비중 {ticker: weight}
        strategy: 사용된 전략
        expected_return: 기대 수익률 (연환산)
        expected_volatility: 기대 변동성 (연환산)
        sharpe_ratio: 샤프 비율
        risk_contributions: 자산별 리스크 기여도
        diversification_ratio: 분산화 비율
        effective_n: 실효 자산 수 (1/sum(w^2))
        optimization_status: 최적화 상태 ("SUCCESS" | "FALLBACK_EQUAL_WEIGHT" | "FALLBACK_PREVIOUS" | "ANALYTICAL")
        is_fallback: 최적화 실패 시 fallback 사용 여부
        fallback_reason: fallback 사유 (최적화 실패 원인)
        metadata: 추가 메타데이터
    """
    weights: Dict[str, float]
    strategy: str
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    diversification_ratio: float = 1.0
    effective_n: float = 1.0
    optimization_status: str = "SUCCESS"
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.weights)

