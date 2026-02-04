#!/usr/bin/env python3
"""
Causality Analysis - Granger Causality
============================================================

Granger causality testing for time series

Economic Foundation:
    - Granger (1969) "Investigating Causal Relations by Econometric Models"
    - Tests whether one time series helps predict another

Class:
    - GrangerCausalityAnalyzer: Granger causality testing engine
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from .schemas import GrangerTestResult

logger = logging.getLogger(__name__)

# Optional statsmodels import
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels not available, Granger tests will use fallback")


class GrangerCausalityAnalyzer:
    """
    Granger Causality 분석기

    모든 변수 쌍에 대해 Granger Causality 검정 수행
    유의미한 선행 관계 식별
    """

    def __init__(
        self,
        max_lag: int = 10,
        significance_level: float = 0.05,
        min_observations: int = 50
    ):
        """
        Parameters:
        -----------
        max_lag : int
            검정할 최대 래그
        significance_level : float
            유의수준 (기본 5%)
        min_observations : int
            최소 관측치 수
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required. Install with: pip install statsmodels")

        self.max_lag = max_lag
        self.significance_level = significance_level
        self.min_observations = min_observations

    def test_stationarity(
        self,
        series: pd.Series,
        significance: float = 0.05
    ) -> Tuple[bool, float]:
        """
        ADF 검정으로 정상성 테스트

        Returns:
        --------
        (is_stationary, p_value)
        """
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            p_value = result[1]
            return p_value < significance, p_value
        except Exception:
            return False, 1.0

    def make_stationary(
        self,
        data: pd.DataFrame,
        max_diff: int = 2
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        차분으로 정상성 확보

        Returns:
        --------
        (stationary_data, diff_orders)
        """
        stationary_data = data.copy()
        diff_orders = {}

        for col in data.columns:
            series = data[col].dropna()
            is_stationary, _ = self.test_stationarity(series)

            diff_order = 0
            while not is_stationary and diff_order < max_diff:
                series = series.diff().dropna()
                diff_order += 1
                is_stationary, _ = self.test_stationarity(series)

            diff_orders[col] = diff_order
            if diff_order > 0:
                stationary_data[col] = data[col].diff(diff_order)

        return stationary_data.dropna(), diff_orders

    def test_granger_causality(
        self,
        data: pd.DataFrame,
        cause: str,
        effect: str,
        max_lag: Optional[int] = None
    ) -> GrangerTestResult:
        """
        단일 변수 쌍에 대한 Granger Causality 검정

        Parameters:
        -----------
        data : DataFrame
            시계열 데이터
        cause : str
            원인 변수명
        effect : str
            결과 변수명
        max_lag : int
            최대 래그 (None이면 self.max_lag 사용)

        Returns:
        --------
        GrangerTestResult
        """
        max_lag = max_lag or self.max_lag

        # 데이터 준비
        test_data = data[[effect, cause]].dropna()

        if len(test_data) < self.min_observations:
            return GrangerTestResult(
                cause=cause,
                effect=effect,
                optimal_lag=0,
                p_value=1.0,
                f_statistic=0.0,
                is_significant=False,
                direction=CausalDirection.NO_CAUSALITY,
                lead_time_days=0
            )

        try:
            # Granger 검정 수행
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = grangercausalitytests(
                    test_data,
                    maxlag=max_lag,
                    verbose=False
                )

            # 최적 래그 찾기 (가장 낮은 p-value)
            best_lag = 1
            best_pvalue = 1.0
            best_fstat = 0.0

            for lag in range(1, max_lag + 1):
                if lag in results:
                    # F-test 결과 사용
                    f_test = results[lag][0]['ssr_ftest']
                    p_value = f_test[1]
                    f_stat = f_test[0]

                    if p_value < best_pvalue:
                        best_pvalue = p_value
                        best_lag = lag
                        best_fstat = f_stat

            is_significant = best_pvalue < self.significance_level

            return GrangerTestResult(
                cause=cause,
                effect=effect,
                optimal_lag=best_lag,
                p_value=best_pvalue,
                f_statistic=best_fstat,
                is_significant=is_significant,
                direction=CausalDirection.X_TO_Y if is_significant else CausalDirection.NO_CAUSALITY,
                lead_time_days=best_lag
            )

        except Exception:
            return GrangerTestResult(
                cause=cause,
                effect=effect,
                optimal_lag=0,
                p_value=1.0,
                f_statistic=0.0,
                is_significant=False,
                direction=CausalDirection.NO_CAUSALITY,
                lead_time_days=0
            )

    def test_all_pairs(
        self,
        data: pd.DataFrame,
        variables: Optional[List[str]] = None,
        make_stationary: bool = True
    ) -> List[GrangerTestResult]:
        """
        모든 변수 쌍에 대해 Granger Causality 검정

        Parameters:
        -----------
        data : DataFrame
            시계열 데이터
        variables : List[str]
            검정할 변수 목록 (None이면 모든 컬럼)
        make_stationary : bool
            정상성 변환 여부

        Returns:
        --------
        List[GrangerTestResult]
            유의미한 결과만 포함
        """
        if variables is None:
            variables = list(data.columns)

        # 정상성 확보
        if make_stationary:
            data, _ = self.make_stationary(data[variables])

        results = []

        for cause in variables:
            for effect in variables:
                if cause == effect:
                    continue

                result = self.test_granger_causality(data, cause, effect)
                if result.is_significant:
                    results.append(result)

        # p-value 순으로 정렬
        results.sort(key=lambda x: x.p_value)

        return results

    def get_bidirectional_relationships(
        self,
        results: List[GrangerTestResult]
    ) -> List[Tuple[str, str]]:
        """양방향 인과관계 식별"""
        pairs = {}
        for r in results:
            key = tuple(sorted([r.cause, r.effect]))
            if key not in pairs:
                pairs[key] = []
            pairs[key].append(r)

        bidirectional = []
        for key, rs in pairs.items():
            if len(rs) == 2:  # 양방향
                bidirectional.append(key)

        return bidirectional


# ============================================================================
# Causal Network Builder
# ============================================================================

