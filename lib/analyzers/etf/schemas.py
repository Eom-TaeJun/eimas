#!/usr/bin/env python3
"""
ETF Flow Analyzer - Data Schemas
============================================================

Data classes for ETF flow analysis
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .enums import MarketSentiment, StyleRotation, CyclePhase


@dataclass
class ETFData:
    """단일 ETF 데이터"""
    ticker: str
    name: str
    category: str
    current_price: float
    change_1d: float           # 1일 수익률
    change_5d: float           # 5일 수익률
    change_20d: float          # 20일 수익률
    change_60d: float          # 60일 수익률
    volume_ratio: float        # 거래량 비율 (vs 20일 평균)
    rsi: float                 # RSI (14일)
    relative_strength: float   # 상대 강도 (vs SPY)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ETF 상세 정보 (NEW)
    expense_ratio: Optional[float] = None      # 비용비율 (%)
    dividend_yield: Optional[float] = None     # 배당수익률 (%)
    total_assets: Optional[float] = None       # AUM (억 달러)
    holdings_count: Optional[int] = None       # 보유 종목 수
    pe_ratio: Optional[float] = None           # P/E Ratio
    beta: Optional[float] = None               # 베타 (시장 대비 변동성)
    ytd_return: Optional[float] = None         # YTD 수익률 (%)
    three_year_return: Optional[float] = None  # 3년 수익률 (연환산, %)

    def to_dict(self) -> Dict:
        return asdict(self)

    def get_info_summary(self) -> str:
        """ETF 상세 정보 요약 문자열"""
        parts = []
        if self.expense_ratio is not None:
            parts.append(f"비용비율: {self.expense_ratio:.2f}%")
        if self.dividend_yield is not None:
            parts.append(f"배당수익률: {self.dividend_yield:.2f}%")
        if self.total_assets is not None:
            parts.append(f"AUM: ${self.total_assets:.1f}B")
        if self.pe_ratio is not None:
            parts.append(f"P/E: {self.pe_ratio:.1f}")
        if self.beta is not None:
            parts.append(f"Beta: {self.beta:.2f}")
        return " | ".join(parts) if parts else "상세 정보 없음"


@dataclass
class FlowComparison:
    """ETF 쌍 비교 결과"""
    pair_name: str             # 예: "Growth vs Value"
    etf_a: str                 # 예: VUG
    etf_b: str                 # 예: VTV
    ratio_current: float       # 현재 비율 (A/B)
    ratio_20d_avg: float       # 20일 평균 비율
    ratio_z_score: float       # 비율의 Z-score
    spread_1d: float           # 1일 스프레드 (A - B)
    spread_5d: float           # 5일 스프레드
    spread_20d: float          # 20일 스프레드
    signal: str                # "A_LEADING", "B_LEADING", "NEUTRAL"
    strength: float            # 신호 강도 (0-1)
    interpretation: str        # 해석

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SectorRotationResult:
    """섹터 로테이션 분석 결과"""
    cycle_phase: CyclePhase
    leading_sectors: List[str]
    lagging_sectors: List[str]
    sector_rankings: Dict[str, int]  # {섹터: 순위}
    offensive_score: float     # 공격적 섹터 점수 (0-100)
    defensive_score: float     # 방어적 섹터 점수 (0-100)
    confidence: float          # 신뢰도 (0-1)
    interpretation: str

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['cycle_phase'] = self.cycle_phase.value
        return data


@dataclass
class MarketRegimeResult:
    """시장 레짐 분석 결과"""
    sentiment: MarketSentiment
    style_rotation: StyleRotation
    cycle_phase: CyclePhase
    risk_appetite_score: float      # 0-100, 높을수록 위험 선호
    breadth_score: float            # 시장 폭 점수 (0-100)

    # 세부 지표
    growth_value_spread: float
    large_small_spread: float
    us_global_spread: float
    equity_bond_spread: float
    hy_treasury_spread: float       # HY vs Treasury 스프레드

    # 신호
    signals: List[str]
    warnings: List[str]
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['sentiment'] = self.sentiment.value
        data['style_rotation'] = self.style_rotation.value
        data['cycle_phase'] = self.cycle_phase.value
        return data


# ============================================================================
# ETF Flow Analyzer
# ============================================================================
