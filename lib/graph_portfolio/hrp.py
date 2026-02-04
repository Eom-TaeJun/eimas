#!/usr/bin/env python3
"""
Graph Portfolio - HRP (Hierarchical Risk Parity)
============================================================

계층적 리스크 패리티

Economic Foundation:
    - Lopez de Prado (2016): "Building Diversified Portfolios"
    - Recursive bisection for risk allocation
    - Correlation-based hierarchical clustering

Class:
    - HierarchicalRiskParity: HRP 엔진
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Scipy imports
try:
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity (HRP) 구현

    Lopez de Prado (2016) 기반:
    1. 상관관계 기반 계층적 클러스터링
    2. Quasi-diagonal 정렬
    3. Recursive bisection으로 가중치 배분

    장점:
    - 공분산 행렬 역행렬 불필요
    - 추정 오차에 강건
    - 다양화 효과 극대화
    """

    def __init__(self):
        self.weights = None
        self.linkage_matrix = None
        self.sorted_indices = None

    def fit(
        self,
        returns: pd.DataFrame,
        covariance: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        HRP 가중치 계산

        Args:
            returns: 자산 수익률
            covariance: 공분산 행렬 (없으면 계산)

        Returns:
            {asset: weight}
        """
        assets = returns.columns.tolist()
        n = len(assets)

        if n == 1:
            return {assets[0]: 1.0}

        # 공분산 & 상관관계 행렬
        if covariance is None:
            covariance = returns.cov()
        correlation = returns.corr()

        # 1. 거리 행렬 계산
        distance = np.sqrt(2 * (1 - correlation))

        # 2. 계층적 클러스터링 (Ward)
        if SCIPY_AVAILABLE:
            condensed_dist = squareform(distance.values, checks=False)
            self.linkage_matrix = linkage(condensed_dist, method='ward')

            # 3. Quasi-diagonal 정렬
            self.sorted_indices = self._get_quasi_diag(self.linkage_matrix)
        else:
            # Fallback: 변동성 기반 정렬
            volatilities = np.sqrt(np.diag(covariance))
            self.sorted_indices = np.argsort(volatilities)

        sorted_assets = [assets[i] for i in self.sorted_indices]

        # 4. Recursive bisection
        weights = self._recursive_bisection(
            covariance,
            sorted_assets
        )

        self.weights = weights
        return weights

    def _get_quasi_diag(self, link: np.ndarray) -> List[int]:
        """Quasi-diagonal 순서 추출"""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]

        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])

        return sort_ix.tolist()

    def _recursive_bisection(
        self,
        covariance: pd.DataFrame,
        sorted_assets: List[str]
    ) -> Dict[str, float]:
        """
        Recursive bisection으로 가중치 계산

        핵심 아이디어:
        - 클러스터를 반으로 나누며 분산 역가중치 적용
        """
        weights = pd.Series(1.0, index=sorted_assets)
        clusters = [sorted_assets]

        while len(clusters) > 0:
            # 클러스터를 반으로 분할
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # 각 하위 클러스터의 분산 계산
                var_left = self._get_cluster_var(covariance, left)
                var_right = self._get_cluster_var(covariance, right)

                # 역분산 가중치
                alpha = 1 - var_left / (var_left + var_right + 1e-10)

                # 가중치 업데이트
                weights[left] *= alpha
                weights[right] *= (1 - alpha)

                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights.to_dict()

    def _get_cluster_var(self, covariance: pd.DataFrame, assets: List[str]) -> float:
        """클러스터 분산 계산 (역분산 가중)"""
        cov_slice = covariance.loc[assets, assets]

        # 역분산 가중치
        ivp = 1 / np.diag(cov_slice)
        ivp /= ivp.sum()

        # 클러스터 분산
        cluster_var = np.dot(ivp, np.dot(cov_slice, ivp))
        return cluster_var


