#!/usr/bin/env python3
"""
Critical Path - Stress Regime Multiplier
=========================================

스트레스 레짐 승수 계산 모듈

Economic Foundation:
    Longin & Solnik (2001): "Extreme Correlation of International Equity Markets"

    핵심 개념:
    - 위기 시 상관관계 증가 (tail dependence)
    - 변동성 클러스터링 (GARCH 효과)
    - 전염 가속 계수 (contagion factor)
    - 레짐별 승수 적용

Classes:
    - StressRegimeMultiplier: 스트레스 승수 계산

Returns:
    StressMultiplierResult: 최종 승수 및 방법론 설명
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import schemas from same package
from .schemas import StressMultiplierResult


class StressRegimeMultiplier:
    """
    스트레스 레짐 승수 계산기 (Elicit Report Enhancement)

    학술적 근거:
    - Longin & Solnik (2001): 극단적 시장에서 상관관계 비대칭 발견
    - Forbes & Rigobon (2002): 위기 시 "contagion" vs "interdependence" 구분
    - Elicit Report: 위기 시 상관관계 61.4% 증가 확인

    Perplexity 검증 결과:
    - 학술적 합의: 스트레스 기간에 상관관계 증가 (confirmatory bias 주의)
    - Forbes-Rigobon 조정: 변동성 증가로 인한 spurious correlation 보정 필요
    - 실무적 함의: 분산 효과 감소 → 리스크 과소평가 방지
    """

    # 레짐별 기본 승수 (기존 로직 기반)
    BASE_MULTIPLIERS = {
        'BULL': 0.8,
        'NEUTRAL': 1.0,
        'TRANSITION': 1.0,
        'BEAR': 1.2,
        'CRISIS': 1.5
    }

    # 상관관계 증가 계수 (Elicit: 61.4% 증가)
    CRISIS_CORRELATION_INCREASE = 0.614

    # VIX 임계값 (스트레스 레벨 결정)
    VIX_THRESHOLDS = {
        'normal': 20,
        'elevated': 25,
        'stress': 30,
        'crisis': 40
    }

    def __init__(
        self,
        correlation_window: int = 60,
        volatility_window: int = 20
    ):
        self.correlation_window = correlation_window
        self.volatility_window = volatility_window

    def calculate_multiplier(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_regime: str,
        vix_level: Optional[float] = None
    ) -> StressMultiplierResult:
        """
        스트레스 레짐 승수 계산

        Parameters:
        -----------
        market_data : Dict[str, DataFrame]
            시장 데이터 (SPY, QQQ 등)
        current_regime : str
            현재 시장 레짐 (BULL/BEAR/NEUTRAL/CRISIS)
        vix_level : float (optional)
            현재 VIX 레벨

        Returns:
        --------
        StressMultiplierResult
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 1. 기본 레짐 승수
        base_multiplier = self.BASE_MULTIPLIERS.get(current_regime.upper(), 1.0)

        # 2. VIX 기반 변동성 스케일링
        if vix_level is None:
            vix_data = market_data.get('^VIX') or market_data.get('VIX')
            if vix_data is not None and not vix_data.empty and 'Close' in vix_data.columns:
                vix_level = float(vix_data['Close'].iloc[-1])
            else:
                vix_level = 20.0  # 기본값

        volatility_scaling = self._calculate_volatility_scaling(vix_level)

        # 3. 상관관계 조정 (Longin-Solnik / Forbes-Rigobon)
        correlation_adjustment = self._calculate_correlation_adjustment(
            market_data, current_regime, vix_level
        )

        # 4. 전염 가속 계수
        contagion_factor = self._calculate_contagion_factor(
            market_data, current_regime
        )

        # 5. 최종 승수 계산
        # 공식: Final = Base × (1 + VolScaling) × (1 + CorrAdj) × (1 + Contagion)
        final_multiplier = (
            base_multiplier
            * (1 + volatility_scaling)
            * (1 + correlation_adjustment)
            * (1 + contagion_factor)
        )

        # 상한선 (과도한 승수 방지)
        final_multiplier = min(final_multiplier, 3.0)

        # 방법론 설명
        methodology_notes = self._generate_methodology_notes(
            base_multiplier, volatility_scaling,
            correlation_adjustment, contagion_factor, vix_level
        )

        return StressMultiplierResult(
            timestamp=timestamp,
            base_multiplier=base_multiplier,
            correlation_adjustment=correlation_adjustment,
            volatility_scaling=volatility_scaling,
            contagion_factor=contagion_factor,
            final_multiplier=final_multiplier,
            regime=current_regime,
            methodology_notes=methodology_notes,
            academic_references=[
                "Longin & Solnik (2001): Extreme Correlation of International Equity Markets",
                "Forbes & Rigobon (2002): No Contagion, Only Interdependence",
                "Elicit Report (2026): 61.4% correlation increase during stress"
            ]
        )

    def _calculate_volatility_scaling(self, vix_level: float) -> float:
        """VIX 기반 변동성 스케일링 계산"""
        # 정상 VIX (20) 대비 초과분에 비례하여 스케일링
        if vix_level <= self.VIX_THRESHOLDS['normal']:
            return 0.0
        elif vix_level <= self.VIX_THRESHOLDS['elevated']:
            return (vix_level - 20) / 100  # 0~5% 추가
        elif vix_level <= self.VIX_THRESHOLDS['stress']:
            return (vix_level - 20) / 50   # 0~20% 추가
        elif vix_level <= self.VIX_THRESHOLDS['crisis']:
            return (vix_level - 20) / 40   # 0~50% 추가
        else:
            return 0.5 + (vix_level - 40) / 100  # 50%+ 추가

    def _calculate_correlation_adjustment(
        self,
        market_data: Dict[str, pd.DataFrame],
        regime: str,
        vix_level: float
    ) -> float:
        """
        상관관계 조정 계산 (Longin-Solnik / Forbes-Rigobon)

        핵심 아이디어:
        - 위기 시 자산 간 상관관계가 비선형적으로 증가
        - Elicit Report: 평균 61.4% 상관관계 증가 관측
        - Forbes-Rigobon: 변동성 증가로 인한 spurious correlation 보정 필요
        """
        # 스트레스 레벨 결정
        if regime.upper() in ['CRISIS'] or vix_level > 35:
            stress_level = 'CRISIS'
        elif regime.upper() in ['BEAR'] or vix_level > 25:
            stress_level = 'STRESS'
        else:
            stress_level = 'NORMAL'

        # 상관관계 조정값
        if stress_level == 'CRISIS':
            # Elicit 61.4% × 0.7 (Forbes-Rigobon 보정)
            raw_adjustment = self.CRISIS_CORRELATION_INCREASE * 0.7
        elif stress_level == 'STRESS':
            raw_adjustment = self.CRISIS_CORRELATION_INCREASE * 0.4
        else:
            raw_adjustment = 0.0

        # 실제 상관관계 변화 측정 (데이터 있으면)
        try:
            empirical_adj = self._measure_empirical_correlation_change(market_data)
            if empirical_adj is not None:
                # 이론값과 실증값의 가중 평균
                return 0.6 * raw_adjustment + 0.4 * empirical_adj
        except Exception:
            pass

        return raw_adjustment

    def _measure_empirical_correlation_change(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Optional[float]:
        """실제 상관관계 변화 측정"""
        # SPY-QQQ, SPY-TLT 등 주요 자산 쌍의 롤링 상관관계 변화
        spy_data = market_data.get('SPY')
        qqq_data = market_data.get('QQQ')
        tlt_data = market_data.get('TLT')

        if spy_data is None or qqq_data is None:
            return None

        try:
            # 수익률 계산
            spy_ret = spy_data['Close'].pct_change().dropna()
            qqq_ret = qqq_data['Close'].pct_change().dropna()

            # 최근 상관관계 vs 장기 상관관계
            if len(spy_ret) < self.correlation_window:
                return None

            short_window = min(20, len(spy_ret) // 2)
            long_corr = spy_ret.tail(self.correlation_window).corr(
                qqq_ret.tail(self.correlation_window)
            )
            short_corr = spy_ret.tail(short_window).corr(
                qqq_ret.tail(short_window)
            )

            # 상관관계 변화율
            if long_corr != 0:
                return (short_corr - long_corr) / abs(long_corr)
            return 0.0
        except Exception:
            return None

    def _calculate_contagion_factor(
        self,
        market_data: Dict[str, pd.DataFrame],
        regime: str
    ) -> float:
        """
        전염 가속 계수 계산

        위기 시 자산 간 충격 전파 속도가 가속화됨을 반영
        """
        if regime.upper() not in ['BEAR', 'CRISIS']:
            return 0.0

        # 섹터 ETF 동조화 측정
        sector_etfs = ['XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP']
        available_sectors = [s for s in sector_etfs if s in market_data]

        if len(available_sectors) < 3:
            # 데이터 부족 시 레짐 기반 기본값
            return 0.1 if regime.upper() == 'BEAR' else 0.2

        try:
            returns = {}
            for sector in available_sectors:
                if 'Close' in market_data[sector].columns:
                    returns[sector] = market_data[sector]['Close'].pct_change().dropna()

            if len(returns) < 3:
                return 0.1

            returns_df = pd.DataFrame(returns).dropna()
            if len(returns_df) < 20:
                return 0.1

            # 상관관계 행렬
            corr_matrix = returns_df.tail(20).corr()

            # 평균 상관관계 (대각선 제외)
            n = len(corr_matrix)
            if n < 2:
                return 0.1

            off_diag = corr_matrix.values[~np.eye(n, dtype=bool)]
            avg_corr = np.mean(off_diag)

            # 높은 동조화 = 높은 전염 가속
            # 평균 상관관계 > 0.7이면 전염 가속
            if avg_corr > 0.8:
                return 0.3
            elif avg_corr > 0.7:
                return 0.2
            elif avg_corr > 0.5:
                return 0.1
            return 0.0
        except Exception:
            return 0.1

    def _generate_methodology_notes(
        self,
        base: float, vol: float, corr: float, contagion: float, vix: float
    ) -> str:
        """방법론 설명 생성"""
        notes = []
        notes.append(f"기본 레짐 승수: {base:.2f}")

        if vol > 0:
            notes.append(f"변동성 스케일링: +{vol*100:.1f}% (VIX={vix:.1f})")

        if corr > 0:
            notes.append(
                f"상관관계 조정: +{corr*100:.1f}% "
                f"(Longin-Solnik/Forbes-Rigobon 기반)"
            )

        if contagion > 0:
            notes.append(f"전염 가속: +{contagion*100:.1f}% (섹터 동조화)")

        return " | ".join(notes)

    def apply_to_risk_score(
        self,
        base_risk_score: float,
        multiplier_result: StressMultiplierResult
    ) -> Tuple[float, str]:
        """
        리스크 점수에 스트레스 승수 적용

        Parameters:
        -----------
        base_risk_score : float
            기본 리스크 점수 (0-100)
        multiplier_result : StressMultiplierResult
            스트레스 승수 결과

        Returns:
        --------
        Tuple[adjusted_score, explanation]
        """
        adjusted_score = base_risk_score * multiplier_result.final_multiplier
        adjusted_score = min(100.0, adjusted_score)  # 상한 100

        explanation = (
            f"Base: {base_risk_score:.1f} × Multiplier: {multiplier_result.final_multiplier:.2f} "
            f"= Adjusted: {adjusted_score:.1f}"
        )

        return adjusted_score, explanation


