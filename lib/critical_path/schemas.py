#!/usr/bin/env python3
"""
Critical Path - Data Schemas & Helper Functions
================================================

데이터클래스 및 유틸리티 함수 모음

Dataclasses:
    - RiskAppetiteUncertaintyResult: 리스크 선호도/불확실성 분석 결과
    - RegimeResult: 레짐 분석 결과 (BULL/BEAR/TRANSITION/CRISIS)
    - SpilloverEdge: 자산간 충격 전이 경로
    - SpilloverResult: Spillover 네트워크 분석 결과
    - CryptoSentimentResult: 암호화폐 심리 분석 결과
    - CriticalPathResult: Critical Path 분석 종합 결과
    - StressMultiplierResult: 스트레스 레짐 승수 결과

Helper Functions:
    - calculate_rolling_zscore(): 롤링 Z-score 계산
    - calculate_realized_volatility(): 실현 변동성 계산
    - normalize_to_score(): 0-100 정규화

Economic Foundation:
    - Bekaert et al. (2013): VIX 분해 (Uncertainty vs Risk Appetite)
    - Maheu & McCurdy: 레짐 스위칭 모델
    - Boeckelmann: 충격 전이(spillover) 네트워크
    - Longin-Solnik: 위기 시 상관관계 증가
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field


# ============================================================================
# Helper Functions
# ============================================================================

def calculate_rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    롤링 Z-score 계산

    경제학적 의미:
    - Z-score = (현재값 - N일 평균) / N일 표준편차
    - |Z| > 2: 통계적으로 이상치 (95% 신뢰구간 벗어남)
    - Mean Reversion: 극단적 Z-score는 평균 회귀 경향

    Args:
        series: 시계열 데이터
        window: 롤링 윈도우 크기 (일)

    Returns:
        Z-score 시계열
    """
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std()
    z_score = (series - mean) / std.replace(0, np.nan)
    return z_score.fillna(0)


def calculate_realized_volatility(prices: pd.Series, window: int = 20) -> float:
    """
    실현 변동성 계산 (연율화)

    경제학적 의미:
    - 실현 변동성 = 과거 N일간 수익률의 표준편차 × √252
    - 높은 변동성 = 높은 불확실성 = 높은 리스크

    Args:
        prices: 가격 시계열
        window: 계산 윈도우 (일)

    Returns:
        연율화된 실현 변동성 (%)
    """
    returns = prices.pct_change().dropna()
    if len(returns) < window:
        window = len(returns)
    if window == 0:
        return 0.0
    return returns.tail(window).std() * np.sqrt(252) * 100


def normalize_to_score(value: float, min_val: float, max_val: float) -> float:
    """
    값을 0-100 스코어로 정규화

    Args:
        value: 정규화할 값
        min_val: 최소값 (0점)
        max_val: 최대값 (100점)

    Returns:
        0-100 사이의 정규화된 스코어

    Note:
        - Median bias: min/max가 같으면 50점 반환
        - Clamping: [0, 100] 범위로 제한
    """
    if max_val == min_val:
        return 50.0  # 기본값
    normalized = (value - min_val) / (max_val - min_val) * 100
    return max(0.0, min(100.0, normalized))


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RiskAppetiteUncertaintyResult:
    """
    리스크 애퍼타이트와 불확실성 분석 결과

    경제학적 의미:
    - risk_appetite_score: 0-100, 높을수록 위험 선호 (투자자들이 위험을 감수하려는 의지)
    - uncertainty_score: 0-100, 높을수록 불확실 (시장의 예측 불가능성)
    - market_state: 두 지수의 조합으로 시장 상태 해석

    Reference:
        Bekaert et al. (2013) "The VRP and the Cross-Section of Expected Returns"
    """
    timestamp: str
    risk_appetite_score: float      # 0-100, 높을수록 위험 선호
    uncertainty_score: float        # 0-100, 높을수록 불확실
    risk_appetite_level: str        # "LOW", "MEDIUM", "HIGH"
    uncertainty_level: str          # "LOW", "MEDIUM", "HIGH"
    market_state: str               # "NORMAL", "SPECULATIVE", "STAGNANT", "CRISIS", "MIXED"
    components: Dict                 # 개별 지표 값들
    interpretation: str           # 해석 텍스트

    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)


@dataclass
class RegimeResult:
    """
    레짐 분석 결과

    경제학적 의미:
    - current_regime: 현재 시장 국면 (BULL/BEAR/TRANSITION/CRISIS)
    - regime_confidence: 레짐 판단의 확신도 (0-100%)
    - transition_probability: 레짐 전환 확률 (0-100%)
    - thresholds: 현재 레짐에 맞는 임계값 세트 (레짐별로 다름)

    Reference:
        Maheu & McCurdy: Markov Switching Models
    """
    timestamp: str
    current_regime: str           # "BULL", "BEAR", "TRANSITION", "CRISIS"
    regime_confidence: float      # 0-100%
    transition_probability: float  # 레짐 전환 확률 (0-100%)
    transition_direction: str      # "BULL_TO_BEAR", "BEAR_TO_BULL", "STABLE", "UNCERTAIN"
    thresholds: Dict               # 현재 레짐에 맞는 임계값 세트
    ma_status: Dict                # MA 상태 정보
    interpretation: str            # 해석 텍스트

    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)


@dataclass
class SpilloverEdge:
    """
    자산간 충격 전이(spillover) 경로

    경제학적 의미:
    - source: 충격이 발생한 자산 (위험 진원지)
    - target: 충격이 전이될 자산
    - edge_type: 전이 방향 (POSITIVE: 같은 방향, NEGATIVE: 반대 방향)
    - adjusted_lag: 레짐에 따라 조정된 시차 (위기 시 단축)

    Reference:
        Boeckelmann: Spillover Networks in Financial Markets
    """
    source: str                    # 출발 노드 (예: "TLT")
    target: str                    # 도착 노드 (예: "QQQ")
    edge_type: str                 # "POSITIVE", "NEGATIVE"
    base_lag: int                  # 기본 시차 (일)
    adjusted_lag: int              # 레짐 조정된 시차
    signal_strength: float         # 신호 강도 (0-100)
    is_active: bool               # 현재 활성화 여부
    source_move: float            # 소스 자산 움직임 (%)
    expected_target_move: str     # 예상 타겟 방향 ("UP", "DOWN")
    theory_note: str              # 경제학적 설명
    category: str = ""            # 경로 카테고리: 'liquidity', 'volatility', 'credit', 'concentration', 'rotation'

    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)


@dataclass
class SpilloverResult:
    """
    Spillover 네트워크 분석 결과

    경제학적 의미:
    - active_paths: 현재 활성화된 충격 전이 경로들
    - risk_score: 전이 위험 점수 (활성 경로 수와 강도 기반)
    - primary_risk_source: 가장 많은 경로의 소스가 되는 자산 (위험 진원지)
    """
    timestamp: str
    active_paths: List[SpilloverEdge]    # 활성화된 경로들
    risk_score: float                     # 전이 위험 점수 (0-100)
    primary_risk_source: str              # 주요 위험 진원지
    expected_impacts: Dict[str, str]      # 자산별 예상 영향
    interpretation: str                  # 해석 텍스트

    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        d = asdict(self)
        d['active_paths'] = [edge.to_dict() for edge in self.active_paths]
        return d


@dataclass
class CryptoSentimentResult:
    """
    암호화폐 심리 분석 결과

    경제학적 의미:
    - sentiment_score: 암호화폐 시장 심리 점수 (0-100)
    - btc_spy_correlation: BTC-주식 상관관계 (레짐에 따라 다름)
    - correlation_regime: 상관관계 기반 레짐 (DECOUPLED/COUPLED/CRISIS_COUPLED)
    - is_leading_indicator: 선행지표로 작동 중인지 여부
    - risk_contribution: 전체 위험도에 기여하는 비중 (위기 시 증가)
    - causality_analysis: Granger Causality 검정 결과 (인과관계 방향성)

    Reference:
        IMF (2021): Crypto-Asset Cross-Border Flows
    """
    timestamp: str
    sentiment_score: float             # 0-100
    sentiment_level: str               # "EXTREME_FEAR", "FEAR", "NEUTRAL", "GREED", "EXTREME_GREED"
    btc_spy_correlation: float         # 20일 롤링 상관관계
    correlation_regime: str            # "DECOUPLED", "COUPLED", "CRISIS_COUPLED"
    is_leading_indicator: bool         # 선행지표로 작동 중인지
    leading_signal: Optional[str]      # "RISK_OFF_WARNING", "RISK_ON_SIGNAL", None
    risk_contribution: float           # 전체 위험도에 기여하는 비중 (0-20%)
    components: Dict                   # 개별 지표 값들
    interpretation: str                # 해석 텍스트
    causality_analysis: Dict = field(default_factory=lambda: {})  # Granger Causality 검정 결과

    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return asdict(self)


@dataclass
class CriticalPathResult:
    """
    Critical Path 분석 종합 결과

    경제학적 의미:
    - total_risk_score: 전체 시장 위험도 (0-100)
    - path_contributions: 경로별 위험 기여도 (합계 = total_risk_score)
    - primary_risk_path: 가장 큰 기여도를 가진 경로 (위험 진원지)
    """
    timestamp: str

    # 전체 위험도
    total_risk_score: float           # 0-100
    risk_level: str                   # "LOW", "MEDIUM", "HIGH", "CRITICAL"

    # 레짐 정보
    current_regime: str
    regime_confidence: float
    transition_probability: float

    # 경로별 기여도 (raw scores, 절대값)
    path_contributions: Dict[str, float]
    # 예: {"liquidity": 25, "concentration": 22, "credit": 10, ...}

    # 경로별 분포 (100% 정규화, 시각화용)
    path_distribution: Dict[str, float]
    # 예: {"liquidity": 35.2%, "concentration": 30.1%, ...}

    # 하위 모듈 결과
    risk_appetite_result: 'RiskAppetiteUncertaintyResult'
    regime_result: 'RegimeResult'
    spillover_result: 'SpilloverResult'
    crypto_result: 'CryptoSentimentResult'

    # 해석 및 경고
    primary_risk_path: str            # 가장 큰 기여도 경로
    active_warnings: List[str]        # 활성화된 경고 목록
    interpretation: str               # 종합 해석

    def to_dict(self) -> Dict:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        d = asdict(self)
        d['risk_appetite_result'] = self.risk_appetite_result.to_dict()
        d['regime_result'] = self.regime_result.to_dict()
        d['spillover_result'] = self.spillover_result.to_dict()
        d['crypto_result'] = self.crypto_result.to_dict()
        return d


@dataclass
class StressMultiplierResult:
    """
    스트레스 레짐 승수 결과

    경제학적 의미:
    - base_multiplier: 기본 레짐 승수
    - correlation_adjustment: 위기 시 상관관계 증가 (Longin-Solnik)
    - volatility_scaling: 변동성 스케일링
    - contagion_factor: 전염 가속 계수
    - final_multiplier: 최종 승수

    Reference:
        Longin & Solnik (2001): "Extreme Correlation of International Equity Markets"
    """
    timestamp: str
    base_multiplier: float           # 기본 레짐 승수
    correlation_adjustment: float    # 상관관계 조정 (Longin-Solnik)
    volatility_scaling: float        # 변동성 스케일링
    contagion_factor: float          # 전염 가속 계수
    final_multiplier: float          # 최종 승수
    regime: str                      # 현재 레짐
    methodology_notes: str           # 방법론 설명
    academic_references: List[str]   # 학술 참고문헌
