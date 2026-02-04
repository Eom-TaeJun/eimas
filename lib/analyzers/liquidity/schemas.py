#!/usr/bin/env python3
"""Liquidity Analyzer - Data Schemas"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class LiquidityImpactResult:
    """유동성 → 시장 영향 분석 결과"""
    timestamp: datetime

    # RRP 영향
    rrp_to_spy_lag: int = 0          # RRP가 SPY에 선행하는 일수
    rrp_to_spy_pvalue: float = 1.0
    rrp_to_spy_significant: bool = False

    rrp_to_vix_lag: int = 0
    rrp_to_vix_pvalue: float = 1.0
    rrp_to_vix_significant: bool = False

    # TGA 영향
    tga_to_spy_lag: int = 0
    tga_to_spy_pvalue: float = 1.0
    tga_to_spy_significant: bool = False

    # Net Liquidity 영향
    net_liq_to_spy_lag: int = 0
    net_liq_to_spy_pvalue: float = 1.0
    net_liq_to_spy_significant: bool = False

    # 핵심 드라이버
    key_drivers: List[str] = field(default_factory=list)
    critical_path: Optional[str] = None

    # 전체 결과
    granger_results: List[GrangerTestResult] = field(default_factory=list)
    network_stats: Dict[str, Any] = field(default_factory=dict)

    # 해석
    interpretation: str = ""
    trading_signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    signal_confidence: float = 0.0


@dataclass
class LiquidityCorrelation:
    """유동성-시장 상관관계"""
    variable1: str
    variable2: str
    correlation: float
    rolling_corr_mean: float
    rolling_corr_std: float
    is_significant: bool
    correlation_regime: str  # "high_positive", "low", "negative"


# ============================================================================
# Liquidity Market Analyzer
# ============================================================================
@dataclass
class AssetClassLag:
    """자산 클래스별 동적 시차 결과"""
    asset_class: str                    # equity, fixed_income, real_estate, commodity, crypto
    optimal_lag_days: int               # 최적 시차 (일)
    optimal_lag_weeks: float            # 최적 시차 (주)
    p_value: float
    r_squared: float
    is_significant: bool
    lag_confidence: str                 # HIGH, MEDIUM, LOW
    economic_interpretation: str


@dataclass
class RegimeConditionalLag:
    """레짐별 시차 분석 결과"""
    regime: str                         # BULL, BEAR, NEUTRAL, CRISIS
    avg_lag_days: float
    lag_volatility: float               # 시차 변동성
    sample_size: int
    interpretation: str


@dataclass
class DynamicLagResult:
    """동적 시차 분석 종합 결과"""
    timestamp: datetime

    # 자산 클래스별 시차
    asset_class_lags: List[AssetClassLag] = field(default_factory=list)

    # 레짐별 시차
    regime_lags: List[RegimeConditionalLag] = field(default_factory=list)

    # Cross-asset 시차 구조
    lag_matrix: Optional[pd.DataFrame] = None

    # 핵심 인사이트
    fastest_response_asset: str = ""
    slowest_response_asset: str = ""
    current_regime_lag: int = 0

    # 투자 시사점
