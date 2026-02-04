#!/usr/bin/env python3
"""
Bubble Detection - Data Schemas
============================================================

Data classes for bubble detection results

Economic Foundation:
    - "Bubbles for Fama" (Greenwood et al. 2019)
    - Run-up threshold: 100% cumulative return over 2 years
    - Volatility spike: 2σ above historical mean
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .enums import BubbleWarningLevel, RiskSignalType, JPMorganBubbleStage


@dataclass
class RunUpResult:
    """Run-up 분석 결과"""
    cumulative_return: float      # 누적 수익률 (예: 1.5 = 150%)
    is_runup: bool                # Run-up 조건 충족 여부
    start_price: float            # 시작 가격
    end_price: float              # 종료 가격
    period_days: int              # 분석 기간 (일)
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            'cumulative_return': self.cumulative_return,
            'cumulative_return_pct': f"{self.cumulative_return * 100:.1f}%",
            'is_runup': self.is_runup,
            'start_price': self.start_price,
            'end_price': self.end_price,
            'period_days': self.period_days,
            'interpretation': self.interpretation
        }


@dataclass
class VolatilityResult:
    """변동성 분석 결과"""
    current_volatility: float     # 현재 변동성 (연율화)
    historical_mean: float        # 과거 평균 변동성
    historical_std: float         # 과거 변동성 표준편차
    zscore: float                 # z-score
    is_spike: bool                # 변동성 급등 여부
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            'current_volatility': self.current_volatility,
            'historical_mean': self.historical_mean,
            'zscore': self.zscore,
            'is_spike': self.is_spike,
            'interpretation': self.interpretation
        }


@dataclass
class IssuanceResult:
    """주식 발행 분석 결과"""
    current_shares: Optional[float]       # 현재 발행주식수
    previous_shares: Optional[float]      # 이전 발행주식수 (추정)
    change_rate: Optional[float]          # 변화율
    is_increasing: bool                   # 증가 여부
    data_available: bool                  # 데이터 가용성
    interpretation: str
    data_source: str = "unknown"          # 데이터 출처: sharesOutstanding, balance_sheet, marketCap_estimate
    is_estimated: bool = False            # 추정치 여부 (marketCap/price로 계산된 경우)

    def to_dict(self) -> Dict:
        return {
            'current_shares': self.current_shares,
            'change_rate': self.change_rate,
            'is_increasing': self.is_increasing,
            'data_available': self.data_available,
            'interpretation': self.interpretation,
            'data_source': self.data_source,
            'is_estimated': self.is_estimated
        }


@dataclass
class RiskSignal:
    """개별 위험 신호"""
    signal_type: RiskSignalType
    severity: float               # 심각도 (0-1)
    description: str
    evidence: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            'type': self.signal_type.value,
            'severity': self.severity,
            'description': self.description,
            'evidence': self.evidence
        }


class JPMorganFrameworkResult:
    """JP Morgan 5단계 버블 프레임워크 분석 결과"""
    current_stage: 'JPMorganBubbleStage'
    stage_scores: Dict[str, float]  # 각 단계별 점수 (0-100)
    historical_comparison: str      # 역사적 비교 (철도, 닷컴 등)
    key_indicators: List[str]
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            'current_stage': self.current_stage.value,
            'stage_scores': self.stage_scores,
            'historical_comparison': self.historical_comparison,
            'key_indicators': self.key_indicators,
            'interpretation': self.interpretation
        }


class BubbleDetectionResult:
    """버블 탐지 결과"""
    ticker: str
    timestamp: str

    # 핵심 분석 결과
    runup: RunUpResult
    volatility: VolatilityResult
    issuance: IssuanceResult

    # 종합 판단
    bubble_warning_level: BubbleWarningLevel
    risk_signals: List[RiskSignal]
    risk_score: float              # 0-100

    # JP Morgan Framework (NEW)
    jpmorgan_framework: Optional[JPMorganFrameworkResult] = None

    # 추가 정보
    company_name: str = ""
    sector: str = ""
    market_cap: Optional[float] = None

    def to_dict(self) -> Dict:
        result = {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'company_name': self.company_name,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'runup': self.runup.to_dict(),
            'volatility': self.volatility.to_dict(),
            'issuance': self.issuance.to_dict(),
            'bubble_warning_level': self.bubble_warning_level.value,
            'risk_signals': [s.to_dict() for s in self.risk_signals],
            'risk_score': self.risk_score
        }
        # JP Morgan Framework (NEW)
        if self.jpmorgan_framework:
            result['jpmorgan_framework'] = self.jpmorgan_framework.to_dict()
        return result

    def get_summary(self) -> str:
        """결과 요약 문자열"""
        lines = [
            f"=== Bubble Detection: {self.ticker} ===",
            f"Company: {self.company_name}",
            f"Warning Level: {self.bubble_warning_level.value}",
            f"Risk Score: {self.risk_score:.1f}/100",
            "",
            f"Run-up: {self.runup.cumulative_return * 100:.1f}% ({self.runup.interpretation})",
            f"Volatility: z-score {self.volatility.zscore:.2f} ({self.volatility.interpretation})",
            f"Issuance: {self.issuance.interpretation}",
        ]

        # JP Morgan Framework (NEW)
        if self.jpmorgan_framework:
            lines.append("")
            lines.append("=== JP Morgan 5-Stage Framework ===")
            lines.append(f"Current Stage: {self.jpmorgan_framework.current_stage.value}")
            lines.append(f"Historical Comparison: {self.jpmorgan_framework.historical_comparison}")
            lines.append(f"Interpretation: {self.jpmorgan_framework.interpretation}")
            if self.jpmorgan_framework.key_indicators:
                lines.append("Key Indicators:")
                for indicator in self.jpmorgan_framework.key_indicators:
                    lines.append(f"  - {indicator}")

        if self.risk_signals:
            lines.append("")
            lines.append("Risk Signals:")
            for signal in self.risk_signals:
                lines.append(f"  - [{signal.signal_type.value}] {signal.description}")

        return "\n".join(lines)


# =============================================================================
# Bubble Detector
# =============================================================================

@dataclass
class StageResult:
    """개별 단계 평가 결과"""
    stage: str
    passed: bool  # True = 버블 신호 없음, False = 버블 신호
    score: float  # 0-20 (낮을수록 건전)
    evidence: str
    data: Dict[str, Any] = field(default_factory=dict)


class BubbleFrameworkResult:
    """5단계 버블 평가 종합 결과"""
    timestamp: str
    sector: str
    total_score: float  # 0-100
    stage: str  # NO_BUBBLE, EARLY_FORMATION, BUBBLE_BUILDING, LATE_STAGE, IMMINENT_POP
    stage_results: List[StageResult] = field(default_factory=list)
    warning_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'sector': self.sector,
            'total_score': round(self.total_score, 1),
            'stage': self.stage,
            'stage_results': [
                {
                    'stage': sr.stage,
                    'passed': sr.passed,
                    'score': round(sr.score, 1),
                    'evidence': sr.evidence
                }
                for sr in self.stage_results
            ],
            'warning_flags': self.warning_flags
        }

