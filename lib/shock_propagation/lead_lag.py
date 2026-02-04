#!/usr/bin/env python3
"""
Shock Propagation - Lead-Lag Analysis
============================================================

Cross-correlation based lead-lag relationship detection

Economic Foundation:
    - Lead-lag relationships in time series
    - Cross-correlation function (CCF)

Class:
    - LeadLagAnalyzer: Lead-lag relationship analyzer
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
import logging

from .schemas import LeadLagResult

logger = logging.getLogger(__name__)


class LeadLagAnalyzer:
    """
    Lead-Lag 관계 분석

    Cross-correlation at multiple lags를 통해
    어떤 시계열이 다른 시계열을 선행하는지 탐색
    """

    def __init__(self, max_lag: int = 20):
        self.max_lag = max_lag

    def analyze(
        self,
        source: pd.Series,
        target: pd.Series,
        source_name: str = "source",
        target_name: str = "target"
    ) -> LeadLagResult:
        """
        Lead-Lag 분석 수행

        Args:
            source: 소스 시계열
            target: 타겟 시계열

        Returns:
            LeadLagResult
        """
        # 데이터 정렬 및 결측치 처리
        df = pd.DataFrame({'source': source, 'target': target}).dropna()

        if len(df) < self.max_lag * 2:
            return LeadLagResult(
                source=source_name,
                target=target_name,
                optimal_lag=0,
                max_correlation=0.0,
                correlation_at_zero=df['source'].corr(df['target']),
                is_leading=False,
                confidence=0.0
            )

        correlations = {}

        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag > 0:
                # source가 target을 lead (source[t] → target[t+lag])
                corr = df['source'].iloc[:-lag].corr(df['target'].iloc[lag:])
            elif lag < 0:
                # target이 source를 lead
                corr = df['source'].iloc[-lag:].corr(df['target'].iloc[:lag])
            else:
                corr = df['source'].corr(df['target'])

            correlations[lag] = corr if not np.isnan(corr) else 0.0

        # 최대 상관관계 lag 찾기
        optimal_lag = max(correlations, key=lambda k: abs(correlations[k]))
        max_corr = correlations[optimal_lag]
        zero_corr = correlations[0]

        # 신뢰도: 최적 lag의 상관관계가 0보다 얼마나 높은지
        confidence = abs(max_corr) - abs(zero_corr)
        confidence = max(0, min(1, confidence * 5))  # 0-1 정규화

        return LeadLagResult(
            source=source_name,
            target=target_name,
            optimal_lag=optimal_lag,
            max_correlation=max_corr,
            correlation_at_zero=zero_corr,
            is_leading=(optimal_lag > 0 and max_corr > 0) or (optimal_lag < 0 and max_corr < 0),
            confidence=confidence
        )

    def analyze_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        모든 변수 쌍에 대한 Lead-Lag 매트릭스 생성

        Returns:
            DataFrame with optimal lags (양수 = row가 column을 lead)
        """
        columns = data.columns.tolist()
        n = len(columns)

        lag_matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=columns,
            columns=columns
        )

        for i, source in enumerate(columns):
            for j, target in enumerate(columns):
                if i != j:
                    result = self.analyze(
                        data[source], data[target],
                        source, target
                    )
                    lag_matrix.loc[source, target] = result.optimal_lag

        return lag_matrix


# ============================================================================
# Granger Causality
# ============================================================================
