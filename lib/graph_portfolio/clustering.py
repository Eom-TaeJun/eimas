from __future__ import annotations
#!/usr/bin/env python3
"""
Graph Portfolio - Asset Clustering
============================================================

자산 클러스터링 (Community Detection)

Economic Foundation:
    - Louvain Algorithm: Blondel et al. (2008)
    - Graph modularity optimization
    - Hierarchical clustering for asset grouping

Class:
    - AssetClusterer: 자산 클러스터링 엔진
"""

from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd
import numpy as np
import networkx as nx
import logging

from .enums import ClusteringMethod
from .schemas import ClusterInfo

if TYPE_CHECKING:
    from .network import CorrelationNetwork

logger = logging.getLogger(__name__)

# Optional imports
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import community as community_louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


class AssetClusterer:
    """
    자산 클러스터링 엔진

    N → ∞ 문제 해결의 첫 단계:
    무한한 자산을 유사 그룹으로 묶어 차원 축소
    """

    def __init__(
        self,
        method: ClusteringMethod = ClusteringMethod.LOUVAIN,
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 2
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.labels_ = None
        self.cluster_centers_ = None

    def fit(
        self,
        network: CorrelationNetwork,
        returns: Optional[pd.DataFrame] = None
    ) -> Dict[int, List[str]]:
        """
        클러스터링 수행

        Args:
            network: CorrelationNetwork 객체
            returns: 수익률 데이터 (K-means용)

        Returns:
            {cluster_id: [asset_list]}
        """
        assets = list(network.graph.nodes)
        n_assets = len(assets)

        if n_assets < 2:
            return {0: assets}

        # 클러스터 수 자동 결정 (지정 안 된 경우)
        if self.n_clusters is None:
            # 경험적 규칙: sqrt(N/2) 또는 최소 2개
            self.n_clusters = max(2, min(int(np.sqrt(n_assets / 2)), 20))

        if self.method == ClusteringMethod.LOUVAIN and LOUVAIN_AVAILABLE:
            clusters = self._louvain_clustering(network)
        elif self.method == ClusteringMethod.KMEANS and SKLEARN_AVAILABLE:
            clusters = self._kmeans_clustering(network, returns)
        elif self.method == ClusteringMethod.GAUSSIAN_MIXTURE and SKLEARN_AVAILABLE:
            clusters = self._gmm_clustering(network, returns)
        elif self.method == ClusteringMethod.HIERARCHICAL and SCIPY_AVAILABLE:
            clusters = self._hierarchical_clustering(network)
        else:
            # Fallback: 단순 상관관계 기반
            clusters = self._simple_correlation_clustering(network)

        # 너무 작은 클러스터 병합
        clusters = self._merge_small_clusters(clusters, network)

        return clusters

    def _louvain_clustering(self, network: CorrelationNetwork) -> Dict[int, List[str]]:
        """
        Louvain 알고리즘 기반 커뮤니티 탐지

        경제학적 의미:
        - 모듈성(Modularity) 최대화
        - 자연스러운 시장 섹터/테마 발견
        """
        partition = community_louvain.best_partition(network.graph, weight='weight')

        clusters = {}
        for asset, cluster_id in partition.items():
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(asset)

        return clusters

    def _kmeans_clustering(
        self,
        network: CorrelationNetwork,
        returns: Optional[pd.DataFrame]
    ) -> Dict[int, List[str]]:
        """
        K-means 클러스터링

        특징 공간:
        - 상관관계 거리
        - 변동성
        - 거래량 특성
        """
        assets = list(network.graph.nodes)

        # 특징 벡터 구성
        features = []
        for asset in assets:
            feat = [
                network.graph.nodes[asset].get('volatility', 0),
                network.graph.nodes[asset].get('mean_return', 0),
                network.graph.nodes[asset].get('sharpe', 0)
            ]

            # 상관관계 벡터 추가
            if network.correlation_matrix is not None:
                corr_vec = network.correlation_matrix.loc[asset].values
                feat.extend(corr_vec[:min(10, len(corr_vec))])  # 상위 10개만

            features.append(feat)

        features = np.array(features)

        # 정규화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-means
        n_clusters = min(self.n_clusters, len(assets) - 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        self.labels_ = labels
        self.cluster_centers_ = kmeans.cluster_centers_

        clusters = {}
        for i, asset in enumerate(assets):
            cluster_id = int(labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(asset)

        return clusters

    def _gmm_clustering(
        self,
        network: CorrelationNetwork,
        returns: Optional[pd.DataFrame]
    ) -> Dict[int, List[str]]:
        """
        Gaussian Mixture Model 클러스터링

        장점: 확률적 소속 (soft clustering)
        """
        assets = list(network.graph.nodes)

        # 거리 행렬을 특징으로 사용
        if network.distance_matrix is not None:
            features = network.distance_matrix.values
        else:
            features = network.correlation_matrix.values

        n_clusters = min(self.n_clusters, len(assets) - 1)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        labels = gmm.fit_predict(features)

        self.labels_ = labels

        clusters = {}
        for i, asset in enumerate(assets):
            cluster_id = int(labels[i])
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(asset)

        return clusters

    def _hierarchical_clustering(self, network: CorrelationNetwork) -> Dict[int, List[str]]:
        """계층적 클러스터링"""
        assets = list(network.graph.nodes)

        # 거리 행렬
        dist_matrix = network.distance_matrix.values

        # 압축 거리 행렬
        condensed_dist = squareform(dist_matrix, checks=False)

        # Ward linkage
        Z = linkage(condensed_dist, method='ward')

        # 클러스터 할당
        labels = fcluster(Z, t=self.n_clusters, criterion='maxclust')

        self.labels_ = labels

        clusters = {}
        for i, asset in enumerate(assets):
            cluster_id = int(labels[i]) - 1  # fcluster는 1부터 시작
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(asset)

        return clusters

    def _simple_correlation_clustering(self, network: CorrelationNetwork) -> Dict[int, List[str]]:
        """단순 상관관계 기반 클러스터링 (fallback)"""
        assets = list(network.graph.nodes)
        corr = network.correlation_matrix

        clusters = {}
        assigned = set()
        cluster_id = 0

        for asset in assets:
            if asset in assigned:
                continue

            # 높은 상관관계 자산 찾기
            cluster = [asset]
            assigned.add(asset)

            for other in assets:
                if other not in assigned:
                    if abs(corr.loc[asset, other]) > 0.7:  # 높은 상관관계
                        cluster.append(other)
                        assigned.add(other)

            clusters[cluster_id] = cluster
            cluster_id += 1

        return clusters

    def _merge_small_clusters(
        self,
        clusters: Dict[int, List[str]],
        network: CorrelationNetwork
    ) -> Dict[int, List[str]]:
        """
        작은 클러스터를 가장 가까운 클러스터에 병합

        최적화: 행렬 슬라이싱으로 O(k²×m²) → O(k²) + 행렬 연산
        """
        small_clusters = [cid for cid, assets in clusters.items()
                         if len(assets) < self.min_cluster_size]

        if not small_clusters or network.correlation_matrix is None:
            return clusters

        # === 벡터화 준비: 자산 → 인덱스 매핑 ===
        corr_matrix = network.correlation_matrix.values
        all_assets = list(network.correlation_matrix.columns)
        asset_to_idx = {a: i for i, a in enumerate(all_assets)}

        for small_cid in small_clusters:
            if small_cid not in clusters:
                continue

            small_assets = clusters[small_cid]
            # 작은 클러스터의 인덱스 추출
            small_indices = [asset_to_idx[a] for a in small_assets if a in asset_to_idx]

            if not small_indices:
                continue

            best_target = None
            best_similarity = -1

            for target_cid, target_assets in clusters.items():
                if target_cid == small_cid:
                    continue
                if len(target_assets) < self.min_cluster_size:
                    continue

                # 타겟 클러스터의 인덱스 추출
                target_indices = [asset_to_idx[a] for a in target_assets if a in asset_to_idx]

                if not target_indices:
                    continue

                # === 벡터화: 행렬 슬라이싱으로 평균 상관관계 계산 ===
                # np.ix_로 서브행렬 추출
                sub_corr = corr_matrix[np.ix_(small_indices, target_indices)]
                avg_corr = np.abs(sub_corr).mean()

                if avg_corr > best_similarity:
                    best_similarity = avg_corr
                    best_target = target_cid

            # 병합
            if best_target is not None:
                clusters[best_target].extend(small_assets)
                del clusters[small_cid]

        # 클러스터 ID 재정렬
        new_clusters = {}
        for new_id, (old_id, assets) in enumerate(clusters.items()):
            new_clusters[new_id] = assets

        return new_clusters


