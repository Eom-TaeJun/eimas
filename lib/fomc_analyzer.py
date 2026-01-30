#!/usr/bin/env python3
"""
FOMC Dot Plot Analyzer
=======================
JP Morgan Asset Management의 연준 점도표 분석 방법론 구현

연준 위원들의 금리 전망 분포를 분석하여:
1. 정책 불확실성 정량화
2. 매파/비둘기파 비율 측정
3. 시나리오별 금리 경로 도출

핵심 인사이트:
"단순히 금리 전망의 중앙값을 따르지 않고 위원들 간의 이견과
정책 불확실성을 리스크 프리미엄으로 환산"

Usage:
    from lib.fomc_analyzer import FOMCDotPlotAnalyzer

    analyzer = FOMCDotPlotAnalyzer()
    result = analyzer.analyze()
    print(result.policy_uncertainty_index)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


@dataclass
class DotPlotProjection:
    """개별 위원의 금리 전망"""
    year: int
    rate: float


@dataclass
class FOMCAnalysisResult:
    """FOMC 점도표 분석 결과"""
    timestamp: str
    meeting_date: str

    # 2026년 전망
    median_rate_2026: float
    mean_rate_2026: float
    std_rate_2026: float
    min_rate_2026: float
    max_rate_2026: float

    # 장기 전망
    median_rate_longer_run: float

    # 분포 분석
    hawkish_count: int      # 중앙값 위 위원 수
    dovish_count: int       # 중앙값 아래 위원 수
    neutral_count: int      # 중앙값 위원 수
    total_members: int

    # 불확실성 지수
    policy_uncertainty_index: float  # 0-100
    dispersion_index: float          # 표준편차 기반

    # 시나리오
    base_path: List[float]      # 중앙값 기반
    hawkish_path: List[float]   # 상위 25%
    dovish_path: List[float]    # 하위 25%

    # 해석
    stance: str  # HAWKISH, DOVISH, BALANCED
    interpretation: str

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'meeting_date': self.meeting_date,
            '2026_projections': {
                'median': self.median_rate_2026,
                'mean': round(self.mean_rate_2026, 2),
                'std': round(self.std_rate_2026, 2),
                'range': [self.min_rate_2026, self.max_rate_2026]
            },
            'longer_run': self.median_rate_longer_run,
            'member_distribution': {
                'hawkish': self.hawkish_count,
                'neutral': self.neutral_count,
                'dovish': self.dovish_count,
                'total': self.total_members
            },
            'uncertainty': {
                'policy_uncertainty_index': round(self.policy_uncertainty_index, 1),
                'dispersion_index': round(self.dispersion_index, 2)
            },
            'scenarios': {
                'base': self.base_path,
                'hawkish': self.hawkish_path,
                'dovish': self.dovish_path
            },
            'stance': self.stance,
            'interpretation': self.interpretation
        }


class FOMCDotPlotAnalyzer:
    """
    FOMC 점도표 분석기

    2024년 12월 FOMC 기준 점도표 데이터 내장.
    실시간 업데이트는 Fed 웹사이트 스크래핑 또는 API 필요.
    """

    # 2024년 12월 FOMC 점도표 (19명 위원)
    # 실제 데이터: https://www.federalreserve.gov/monetarypolicy/fomcprojtabl20241218.htm
    DOT_PLOT_DEC_2024 = {
        '2025': [
            4.125, 4.125, 4.125, 4.125,  # 4명
            3.875, 3.875, 3.875, 3.875, 3.875,  # 5명
            3.625, 3.625, 3.625,  # 3명
            3.375, 3.375, 3.375, 3.375,  # 4명
            3.125, 3.125, 3.125  # 3명
        ],
        '2026': [
            4.125, 4.125,  # 2명
            3.875, 3.875, 3.875, 3.875,  # 4명
            3.625, 3.625, 3.625,  # 3명
            3.375, 3.375, 3.375, 3.375, 3.375,  # 5명
            3.125, 3.125, 3.125,  # 3명
            2.875, 2.875  # 2명
        ],
        '2027': [
            3.875, 3.875,  # 2명
            3.625, 3.625, 3.625,  # 3명
            3.375, 3.375, 3.375, 3.375,  # 4명
            3.125, 3.125, 3.125, 3.125, 3.125,  # 5명
            2.875, 2.875, 2.875, 2.875,  # 4명
            2.625  # 1명
        ],
        'longer_run': [
            3.25, 3.25,  # 2명
            3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # 8명
            2.875, 2.875, 2.875,  # 3명
            2.75, 2.75, 2.75,  # 3명
            2.5, 2.5, 2.5  # 3명
        ]
    }

    def __init__(self, dot_plot_data: Dict[str, List[float]] = None):
        """
        Args:
            dot_plot_data: 커스텀 점도표 데이터 (없으면 내장 데이터 사용)
        """
        self.dot_plot = dot_plot_data or self.DOT_PLOT_DEC_2024
        self.meeting_date = "2024-12-18"  # 최신 FOMC

    def analyze(self, target_year: str = '2026') -> FOMCAnalysisResult:
        """
        점도표 분석 실행

        Args:
            target_year: 분석 대상 연도 ('2025', '2026', '2027', 'longer_run')

        Returns:
            FOMCAnalysisResult: 분석 결과
        """
        dots = self.dot_plot.get(target_year, self.dot_plot['2026'])
        longer_run = self.dot_plot.get('longer_run', [3.0])

        # 기본 통계
        median = np.median(dots)
        mean = np.mean(dots)
        std = np.std(dots)
        min_rate = min(dots)
        max_rate = max(dots)
        total = len(dots)

        # 위원 분포
        hawkish = sum(1 for d in dots if d > median)
        dovish = sum(1 for d in dots if d < median)
        neutral = sum(1 for d in dots if d == median)

        # 정책 불확실성 지수 (0-100)
        # 표준편차가 클수록 불확실성 높음
        # 역사적으로 0.3-0.5가 일반적
        uncertainty = min(std / 0.3 * 50, 100)

        # 분산 지수 (IQR 기반)
        q75, q25 = np.percentile(dots, [75, 25])
        iqr = q75 - q25
        dispersion = iqr

        # 시나리오별 경로
        current_rate = 4.5  # 현재 Fed Funds Rate
        years = ['2025', '2026', '2027']

        base_path = [current_rate]
        hawkish_path = [current_rate]
        dovish_path = [current_rate]

        for year in years:
            year_dots = self.dot_plot.get(year, [3.5])
            base_path.append(round(np.median(year_dots), 3))
            hawkish_path.append(round(np.percentile(year_dots, 75), 3))
            dovish_path.append(round(np.percentile(year_dots, 25), 3))

        # 스탠스 판단
        lr_median = np.median(longer_run)
        if median > lr_median + 0.25:
            stance = 'HAWKISH'
            interpretation = f"{target_year}년 금리 전망({median:.2f}%)이 장기 균형({lr_median:.2f}%)을 상회. 위원들이 긴축적."
        elif median < lr_median - 0.25:
            stance = 'DOVISH'
            interpretation = f"{target_year}년 금리 전망({median:.2f}%)이 장기 균형({lr_median:.2f}%)을 하회. 위원들이 완화적."
        else:
            stance = 'BALANCED'
            interpretation = f"{target_year}년 금리 전망({median:.2f}%)이 장기 균형({lr_median:.2f}%)에 근접. 중립적."

        # 불확실성 추가 해석
        if uncertainty > 60:
            interpretation += f" 정책 불확실성 높음 (지수: {uncertainty:.0f})."
        elif uncertainty > 40:
            interpretation += f" 정책 불확실성 중간 (지수: {uncertainty:.0f})."

        return FOMCAnalysisResult(
            timestamp=datetime.now().isoformat(),
            meeting_date=self.meeting_date,
            median_rate_2026=median,
            mean_rate_2026=mean,
            std_rate_2026=std,
            min_rate_2026=min_rate,
            max_rate_2026=max_rate,
            median_rate_longer_run=lr_median,
            hawkish_count=hawkish,
            dovish_count=dovish,
            neutral_count=neutral,
            total_members=total,
            policy_uncertainty_index=uncertainty,
            dispersion_index=dispersion,
            base_path=base_path,
            hawkish_path=hawkish_path,
            dovish_path=dovish_path,
            stance=stance,
            interpretation=interpretation
        )

    def get_rate_path_scenarios(self) -> Dict[str, List[Tuple[str, float]]]:
        """
        시나리오별 금리 경로 반환

        Returns:
            {
                'base': [('2025Q1', 4.25), ('2025Q2', 4.0), ...],
                'hawkish': [...],
                'dovish': [...]
            }
        """
        result = self.analyze()

        # 분기별 보간 (단순 선형)
        def interpolate_quarterly(annual_path: List[float]) -> List[Tuple[str, float]]:
            quarterly = []
            years = ['2024', '2025', '2026', '2027']
            for i in range(len(annual_path) - 1):
                start = annual_path[i]
                end = annual_path[i + 1]
                step = (end - start) / 4
                for q in range(4):
                    rate = start + step * q
                    quarterly.append((f"{years[i]}Q{q+1}", round(rate, 3)))
            return quarterly

        return {
            'base': interpolate_quarterly(result.base_path),
            'hawkish': interpolate_quarterly(result.hawkish_path),
            'dovish': interpolate_quarterly(result.dovish_path)
        }

    def calculate_term_premium_impact(self, result: FOMCAnalysisResult = None) -> Dict[str, float]:
        """
        정책 불확실성이 채권 시장에 미치는 영향 추정

        불확실성 → 텀 프리미엄 → 장기 금리에 영향

        Returns:
            {
                'term_premium_contribution': 0.15,  # 텀 프리미엄 기여분 (%)
                'yield_curve_impact': 'steepening',
                'bond_vol_forecast': 'elevated'
            }
        """
        if result is None:
            result = self.analyze()

        # 불확실성 지수 기반 텀 프리미엄 추정
        # 불확실성 50 → 약 0.15% 텀 프리미엄
        term_premium = result.policy_uncertainty_index / 100 * 0.3

        # 커브 영향
        if result.stance == 'HAWKISH' and result.policy_uncertainty_index > 50:
            curve_impact = 'steepening'
        elif result.stance == 'DOVISH':
            curve_impact = 'flattening'
        else:
            curve_impact = 'stable'

        # 채권 변동성
        if result.policy_uncertainty_index > 60:
            vol_forecast = 'elevated'
        elif result.policy_uncertainty_index > 40:
            vol_forecast = 'moderate'
        else:
            vol_forecast = 'low'

        return {
            'term_premium_contribution': round(term_premium, 3),
            'yield_curve_impact': curve_impact,
            'bond_vol_forecast': vol_forecast
        }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FOMC Dot Plot Analyzer Test")
    print("=" * 60)

    analyzer = FOMCDotPlotAnalyzer()

    # 2026년 분석
    result = analyzer.analyze('2026')

    print(f"\nMeeting Date: {result.meeting_date}")
    print(f"\n--- 2026 Rate Projections ---")
    print(f"Median: {result.median_rate_2026:.3f}%")
    print(f"Mean: {result.mean_rate_2026:.3f}%")
    print(f"Std Dev: {result.std_rate_2026:.3f}%")
    print(f"Range: {result.min_rate_2026:.3f}% - {result.max_rate_2026:.3f}%")

    print(f"\n--- Member Distribution ({result.total_members} members) ---")
    print(f"Hawkish (above median): {result.hawkish_count}")
    print(f"Neutral (at median): {result.neutral_count}")
    print(f"Dovish (below median): {result.dovish_count}")

    print(f"\n--- Uncertainty Metrics ---")
    print(f"Policy Uncertainty Index: {result.policy_uncertainty_index:.1f}/100")
    print(f"Dispersion (IQR): {result.dispersion_index:.3f}%")

    print(f"\n--- Stance ---")
    print(f"Stance: {result.stance}")
    print(f"Interpretation: {result.interpretation}")

    print(f"\n--- Rate Path Scenarios ---")
    print(f"Base (median):    {' → '.join([f'{r:.2f}%' for r in result.base_path])}")
    print(f"Hawkish (75th):   {' → '.join([f'{r:.2f}%' for r in result.hawkish_path])}")
    print(f"Dovish (25th):    {' → '.join([f'{r:.2f}%' for r in result.dovish_path])}")

    # 텀 프리미엄 영향
    tp_impact = analyzer.calculate_term_premium_impact(result)
    print(f"\n--- Bond Market Impact ---")
    print(f"Term Premium Contribution: {tp_impact['term_premium_contribution']:.3f}%")
    print(f"Yield Curve: {tp_impact['yield_curve_impact']}")
    print(f"Bond Volatility: {tp_impact['bond_vol_forecast']}")

    print("\n--- JSON Output ---")
    import json
    print(json.dumps(result.to_dict(), indent=2))
