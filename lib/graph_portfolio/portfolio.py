#!/usr/bin/env python3
"""
Graph Portfolio - Main Portfolio Optimizer
============================================================

Graph-Clustered HRP 포트폴리오 최적화

Economic Foundation:
    - Combines: Network → Clustering → Representative → HRP
    - MST systemic risk analysis (Mantegna 1999)
    - Handles infinite asset universe (N → ∞)

Class:
    - GraphClusteredPortfolio: Main optimization engine
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .enums import ClusteringMethod, RepresentativeMethod
from .schemas import ClusterInfo, PortfolioAllocation
from .network import CorrelationNetwork
from .clustering import AssetClusterer
from .representative import RepresentativeSelector
from .hrp import HierarchicalRiskParity

logger = logging.getLogger(__name__)


class GraphClusteredPortfolio:
    """
    Graph-Clustered HRP 통합 클래스

    무한 에셋 환경에서의 포트폴리오 최적화:
    1. 상관관계 네트워크 구축
    2. 커뮤니티 탐지로 클러스터링
    3. 대표 자산 선정
    4. HRP로 가중치 계산
    """

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        clustering_method: ClusteringMethod = ClusteringMethod.LOUVAIN,
        representative_method: RepresentativeMethod = RepresentativeMethod.CENTRALITY,
        max_representatives_per_cluster: int = 2,
        min_cluster_size: int = 2
    ):
        self.correlation_threshold = correlation_threshold
        self.clustering_method = clustering_method
        self.representative_method = representative_method
        self.max_representatives = max_representatives_per_cluster
        self.min_cluster_size = min_cluster_size

        # Components
        self.network = None
        self.clusterer = None
        self.selector = None
        self.hrp = None

        # Results
        self.clusters = None
        self.representatives = None
        self.reduced_universe = None

    def fit(
        self,
        returns: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None
    ) -> PortfolioAllocation:
        """
        전체 파이프라인 실행

        Args:
            returns: 자산 수익률 DataFrame (columns = assets)
            volumes: 거래량 DataFrame (선택)

        Returns:
            PortfolioAllocation 결과
        """
        n_original = len(returns.columns)
        print(f"[GC-HRP] Starting with {n_original} assets")

        # Step 1: 상관관계 네트워크 구축
        print("[GC-HRP] Building correlation network...")
        self.network = CorrelationNetwork(
            correlation_threshold=self.correlation_threshold,
            use_volume_weight=(volumes is not None)
        )
        self.network.build_from_returns(returns, volumes)

        # Step 2: 클러스터링
        print("[GC-HRP] Clustering assets...")
        self.clusterer = AssetClusterer(
            method=self.clustering_method,
            min_cluster_size=self.min_cluster_size
        )
        self.clusters = self.clusterer.fit(self.network, returns)
        print(f"[GC-HRP] Found {len(self.clusters)} clusters")

        # Step 3: 대표 자산 선정
        print("[GC-HRP] Selecting representatives...")
        self.selector = RepresentativeSelector(
            method=self.representative_method,
            max_representatives=self.max_representatives
        )
        self.representatives = self.selector.select(self.clusters, self.network, volumes)

        # 축소된 유니버스
        self.reduced_universe = []
        for reps in self.representatives.values():
            self.reduced_universe.extend(reps)
        self.reduced_universe = list(set(self.reduced_universe))
        print(f"[GC-HRP] Reduced to {len(self.reduced_universe)} representatives")

        # Step 4: HRP 가중치 계산
        print("[GC-HRP] Calculating HRP weights...")
        self.hrp = HierarchicalRiskParity()
        reduced_returns = returns[self.reduced_universe]
        rep_weights = self.hrp.fit(reduced_returns)

        # Step 5: 전체 가중치 계산 (대표 가중치를 클러스터 멤버에 분배)
        full_weights = self._distribute_weights(rep_weights, returns)

        # Step 6: 리스크 기여도 계산
        risk_contributions = self._calculate_risk_contributions(returns, full_weights)

        # Step 7: 포트폴리오 메트릭 계산
        metrics = self._calculate_portfolio_metrics(returns, full_weights)

        # 클러스터 정보 구성
        cluster_infos = self._build_cluster_infos(returns, volumes)

        # Step 8: MST 시스템 리스크 분석
        print("[GC-HRP] Analyzing systemic risk via MST...")
        mst_analysis = self.network.identify_systemic_risk_nodes(top_n=3)
        print(f"[GC-HRP] Top systemic risk nodes: {[n.ticker for n in mst_analysis.systemic_risk_nodes]}")

        return PortfolioAllocation(
            timestamp=datetime.now().isoformat(),
            weights=full_weights,
            cluster_weights={cid: sum(full_weights.get(a, 0) for a in assets)
                           for cid, assets in self.clusters.items()},
            risk_contributions=risk_contributions,
            expected_volatility=metrics['volatility'],
            diversification_ratio=metrics['diversification_ratio'],
            effective_n=metrics['effective_n'],
            methodology=f"GC-HRP ({self.clustering_method.value})",
            clusters=cluster_infos,
            mst_analysis=mst_analysis
        )

    def _distribute_weights(
        self,
        rep_weights: Dict[str, float],
        returns: pd.DataFrame
    ) -> Dict[str, float]:
        """
        대표 가중치를 클러스터 멤버에 분배

        방식: 클러스터 내 역분산 가중
        """
        full_weights = {}

        for cluster_id, assets in self.clusters.items():
            # 클러스터 내 대표 자산 가중치 합
            reps = self.representatives[cluster_id]
            cluster_weight = sum(rep_weights.get(r, 0) for r in reps)

            if cluster_weight == 0:
                # 균등 배분
                for asset in assets:
                    full_weights[asset] = 1.0 / len(returns.columns)
                continue

            # 클러스터 내 역분산 가중치
            variances = []
            for asset in assets:
                if asset in returns.columns:
                    var = returns[asset].var()
                    variances.append((asset, 1 / (var + 1e-10)))
                else:
                    variances.append((asset, 0))

            total_inv_var = sum(v[1] for v in variances)

            for asset, inv_var in variances:
                asset_weight = cluster_weight * (inv_var / (total_inv_var + 1e-10))
                full_weights[asset] = asset_weight

        # 정규화
        total = sum(full_weights.values())
        if total > 0:
            full_weights = {k: v / total for k, v in full_weights.items()}

        return full_weights

    def _calculate_risk_contributions(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """리스크 기여도 계산"""
        assets = [a for a in weights.keys() if a in returns.columns]
        w = np.array([weights[a] for a in assets])
        cov = returns[assets].cov().values

        # 포트폴리오 변동성
        port_var = np.dot(w, np.dot(cov, w))
        port_vol = np.sqrt(port_var)

        if port_vol == 0:
            return {a: 1/len(assets) for a in assets}

        # Marginal Risk Contribution
        mrc = np.dot(cov, w) / port_vol

        # Risk Contribution = w * MRC
        rc = w * mrc
        rc_pct = rc / rc.sum()

        return {assets[i]: float(rc_pct[i]) for i in range(len(assets))}

    def _calculate_portfolio_metrics(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """포트폴리오 메트릭 계산"""
        assets = [a for a in weights.keys() if a in returns.columns]
        w = np.array([weights[a] for a in assets])
        cov = returns[assets].cov().values

        # 포트폴리오 변동성 (연율화)
        port_var = np.dot(w, np.dot(cov, w))
        port_vol = np.sqrt(port_var) * np.sqrt(252)

        # 개별 변동성 가중평균
        individual_vols = np.sqrt(np.diag(cov)) * np.sqrt(252)
        weighted_avg_vol = np.dot(w, individual_vols)

        # Diversification Ratio
        div_ratio = weighted_avg_vol / (port_vol + 1e-10)

        # Effective N (HHI의 역수)
        hhi = np.sum(w ** 2)
        effective_n = 1 / (hhi + 1e-10)

        return {
            'volatility': port_vol,
            'diversification_ratio': div_ratio,
            'effective_n': effective_n
        }

    def _build_cluster_infos(
        self,
        returns: pd.DataFrame,
        volumes: Optional[pd.DataFrame]
    ) -> List[ClusterInfo]:
        """클러스터 정보 구성"""
        infos = []

        for cluster_id, assets in self.clusters.items():
            # 클러스터 내 평균 상관관계
            if len(assets) > 1 and self.network.correlation_matrix is not None:
                cluster_corr = self.network.correlation_matrix.loc[assets, assets]
                mask = np.triu(np.ones_like(cluster_corr, dtype=bool), k=1)
                intra_corr = cluster_corr.where(mask).stack().mean()
            else:
                intra_corr = 1.0

            # 대표 자산
            rep = self.representatives[cluster_id][0] if self.representatives[cluster_id] else assets[0]

            # 거래량 합계
            total_vol = 0
            if volumes is not None:
                for asset in assets:
                    if asset in volumes.columns:
                        total_vol += volumes[asset].sum()

            # 평균 변동성
            avg_vol = np.mean([
                returns[a].std() * np.sqrt(252)
                for a in assets if a in returns.columns
            ])

            infos.append(ClusterInfo(
                cluster_id=cluster_id,
                assets=assets,
                representative=rep,
                intra_correlation=float(intra_corr),
                size=len(assets),
                total_volume=total_vol,
                avg_volatility=avg_vol
            ))

        return infos

    def get_cluster_summary(self) -> pd.DataFrame:
        """클러스터 요약 DataFrame"""
        if self.clusters is None:
            return pd.DataFrame()

        data = []
        for cluster_id, assets in self.clusters.items():
            reps = self.representatives.get(cluster_id, [])
            data.append({
                'cluster_id': cluster_id,
                'n_assets': len(assets),
                'representatives': ', '.join(reps),
                'all_assets': ', '.join(assets[:5]) + ('...' if len(assets) > 5 else '')
            })

        return pd.DataFrame(data)


