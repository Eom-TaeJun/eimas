#!/usr/bin/env python3
"""
Shock Propagation - Granger Causality
============================================================

Granger causality testing for time series

Economic Foundation:
    - Granger (1969): "Investigating Causal Relations"
    - VAR model-based causality testing
    - Null hypothesis: X does not Granger-cause Y

Class:
    - GrangerCausalityAnalyzer: Granger causality tester
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import logging

from .schemas import GrangerResult

logger = logging.getLogger(__name__)

# Optional statsmodels import
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    grangercausalitytests = None


class GrangerCausalityAnalyzer:
    """
    Granger Causality 검정

    "X가 Y를 Granger-cause 한다" =
    "X의 과거 정보가 Y의 예측에 통계적으로 유의한 기여를 한다"

    주의: Granger Causality ≠ True Causality
    하지만 예측적 관계의 좋은 프록시
    """

    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level

    def test(
        self,
        source: pd.Series,
        target: pd.Series,
        source_name: str = "source",
        target_name: str = "target"
    ) -> GrangerResult:
        """
        Granger Causality 검정

        H0: source does not Granger-cause target
        H1: source Granger-causes target
        """
        # 데이터 준비
        df = pd.DataFrame({'target': target, 'source': source}).dropna()

        if len(df) < self.max_lag * 3:
            return GrangerResult(
                source=source_name,
                target=target_name,
                optimal_lag=0,
                f_statistic=0.0,
                p_value=1.0,
                strength=CausalityStrength.NONE,
                is_significant=False
            )

        if not STATSMODELS_AVAILABLE:
            # Simplified fallback using correlation
            return self._simplified_test(df, source_name, target_name)

        try:
            # Granger Causality 검정
            result = grangercausalitytests(
                df[['target', 'source']],
                maxlag=self.max_lag,
                verbose=False
            )

            # 최적 lag 및 p-value 추출
            best_lag = 1
            best_pvalue = 1.0
            best_fstat = 0.0

            for lag in range(1, self.max_lag + 1):
                if lag in result:
                    # F-test 결과 사용
                    ftest = result[lag][0]['ssr_ftest']
                    pvalue = ftest[1]
                    fstat = ftest[0]

                    if pvalue < best_pvalue:
                        best_pvalue = pvalue
                        best_lag = lag
                        best_fstat = fstat

            # 강도 판정
            if best_pvalue < 0.01:
                strength = CausalityStrength.STRONG
            elif best_pvalue < 0.05:
                strength = CausalityStrength.MODERATE
            elif best_pvalue < 0.10:
                strength = CausalityStrength.WEAK
            else:
                strength = CausalityStrength.NONE

            return GrangerResult(
                source=source_name,
                target=target_name,
                optimal_lag=best_lag,
                f_statistic=best_fstat,
                p_value=best_pvalue,
                strength=strength,
                is_significant=(best_pvalue < self.significance_level)
            )

        except Exception as e:
            # 에러 시 fallback
            return self._simplified_test(df, source_name, target_name)

    def _simplified_test(
        self,
        df: pd.DataFrame,
        source_name: str,
        target_name: str
    ) -> GrangerResult:
        """Simplified causality test using lagged correlation"""
        lead_lag = LeadLagAnalyzer(max_lag=self.max_lag)
        result = lead_lag.analyze(
            df['source'], df['target'],
            source_name, target_name
        )

        # 상관관계 기반 유사 p-value 계산
        n = len(df)
        r = result.max_correlation
        if abs(r) > 0:
            t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
            # 근사 p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if SCIPY_AVAILABLE else 0.05
        else:
            t_stat = 0
            p_value = 1.0

        if p_value < 0.01:
            strength = CausalityStrength.STRONG
        elif p_value < 0.05:
            strength = CausalityStrength.MODERATE
        elif p_value < 0.10:
            strength = CausalityStrength.WEAK
        else:
            strength = CausalityStrength.NONE

        return GrangerResult(
            source=source_name,
            target=target_name,
            optimal_lag=abs(result.optimal_lag),
            f_statistic=t_stat**2,
            p_value=p_value,
            strength=strength,
            is_significant=(p_value < self.significance_level)
        )

    def test_bidirectional(
        self,
        series1: pd.Series,
        series2: pd.Series,
        name1: str,
        name2: str
    ) -> Tuple[GrangerResult, GrangerResult]:
        """양방향 Granger Causality 검정"""
        result_1to2 = self.test(series1, series2, name1, name2)
        result_2to1 = self.test(series2, series1, name2, name1)

        # 양방향 여부 업데이트
        if result_1to2.is_significant and result_2to1.is_significant:
            result_1to2.bidirectional = True
            result_2to1.bidirectional = True

        return result_1to2, result_2to1


# ============================================================================
# Shock Propagation Graph
# ============================================================================
