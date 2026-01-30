#!/usr/bin/env python3
"""
Hierarchical Risk Parity (HRP) Portfolio Optimizer
===================================================

역행렬 없이 대규모 자산 배분을 수행하는 HRP 알고리즘 구현

경제학적 근거:
1. 특이점(Singularity) 문제:
   - 자산 개수 N이 커지면 공분산 행렬 Sigma의 역행렬 Sigma^-1을 구할 수 없음
   - 전통적 Markowitz MVO(Mean-Variance Optimization)는 N > T (자산 > 기간)일 때 실패
   - 토큰화 시대에 N -> infinity이므로 역행렬 기반 방법론은 사용 불가

2. HRP(Hierarchical Risk Parity) 해결책:
   - Marco Lopez de Prado (2016) 제안
   - 역행렬을 사용하지 않음 -> 특이점 문제 해결
   - 계층적 군집화로 유사 자산 그룹화
   - 분산(Variance) 역비례로 위험 균등 배분

3. 알고리즘 단계:
   Step 1: 상관관계 기반 거리 행렬 계산 (d = sqrt(0.5 * (1 - rho)))
   Step 2: 계층적 군집화 (Hierarchical Clustering)
   Step 3: 준 대각화 (Quasi-Diagonalization) - 유사 자산 인접 배치
   Step 4: 재귀적 이분할 (Recursive Bisection) - 분산 역비례 배분

4. 장점:
   - 추정 오차에 강건 (Robust)
   - 과적합 방지
   - 직관적인 자산 그룹화
   - 대규모 자산에 확장 가능

Usage:
    optimizer = HRPOptimizer()
    weights = optimizer.optimize(returns_df)
    optimizer.plot_dendrogram()
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list, fcluster
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger('eimas.hrp_optimizer')


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClusterInfo:
    """
    클러스터 정보

    경제학적 의미:
    - 상관관계가 높은 자산들의 그룹
    - 같은 클러스터 내 자산은 분산투자 효과 제한적
    - 다른 클러스터 간 자산이 진정한 분산투자
    """
    cluster_id: int
    assets: List[str]
    avg_correlation: float
    total_weight: float
    description: str = ""


@dataclass
class HRPResult:
    """
    HRP 최적화 결과

    경제학적 의미:
    - weights: 위험 균등 배분된 자산별 비중
    - clusters: 상관관계 기반 자산 군집
    - effective_n: 유효 자산 수 (분산 척도)
    - diversification_ratio: 분산투자 비율
    """
    weights: Dict[str, float]
    clusters: List[ClusterInfo]
    linkage_matrix: np.ndarray
    sorted_assets: List[str]

    # 포트폴리오 특성
    expected_volatility: float = 0.0
    diversification_ratio: float = 0.0
    effective_n: float = 0.0

    # 메타데이터
    method: str = "HRP"
    timestamp: datetime = field(default_factory=datetime.now)
    commentary: str = ""


# =============================================================================
# HRP Core Algorithm
# =============================================================================

class HRPOptimizer:
    """
    Hierarchical Risk Parity 최적화기

    핵심 알고리즘:
    1. 상관관계 -> 거리 행렬 변환: d_ij = sqrt(0.5 * (1 - rho_ij))
    2. Ward 연결법으로 계층적 군집화
    3. 군집 순서대로 자산 재배열 (준 대각화)
    4. 재귀적 이분할로 분산 역비례 배분

    경제학적 의미:
    - 역행렬 불필요 -> 특이점 문제 해결
    - 상관관계 구조 반영한 자연스러운 그룹화
    - 분산(위험) 기여도 균등화
    """

    def __init__(
        self,
        linkage_method: str = "ward",
        n_clusters: int = None,
        min_weight: float = 0.01,
        max_weight: float = 0.30,
        verbose: bool = False
    ):
        """
        Args:
            linkage_method: 연결법 ("ward", "single", "complete", "average")
            n_clusters: 목표 클러스터 수 (None이면 자동)
            min_weight: 최소 비중
            max_weight: 최대 비중
            verbose: 상세 로깅
        """
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.verbose = verbose

        # 결과 저장
        self._linkage_matrix: Optional[np.ndarray] = None
        self._sorted_assets: Optional[List[str]] = None
        self._returns: Optional[pd.DataFrame] = None
        self._cov: Optional[pd.DataFrame] = None
        self._corr: Optional[pd.DataFrame] = None

    def _log(self, msg: str):
        """로깅"""
        if self.verbose:
            logger.info(msg)
            print(f"[HRP] {msg}")

    def optimize(self, returns: pd.DataFrame) -> HRPResult:
        """
        HRP 포트폴리오 최적화 수행

        경제학적 근거:
        - 역행렬 없이 공분산 구조를 활용한 위험 배분
        - Marco Lopez de Prado (2016) 알고리즘

        Args:
            returns: 수익률 DataFrame (rows=dates, cols=assets)

        Returns:
            HRPResult: 최적화 결과
        """
        self._log(f"Starting HRP optimization with {len(returns.columns)} assets...")
        self._returns = returns

        # Step 1: 공분산/상관관계 행렬 계산
        self._cov = returns.cov()
        self._corr = returns.corr()
        assets = list(returns.columns)

        self._log("Step 1: Computing correlation-based distance matrix")
        dist_matrix = self._correlation_to_distance(self._corr)

        # Step 2: 계층적 군집화
        self._log("Step 2: Hierarchical clustering")
        self._linkage_matrix = self._hierarchical_clustering(dist_matrix)

        # Step 3: 준 대각화 (자산 순서 재배열)
        self._log("Step 3: Quasi-diagonalization")
        sorted_indices = self._quasi_diagonalization(self._linkage_matrix)
        self._sorted_assets = [assets[i] for i in sorted_indices]

        # Step 4: 재귀적 이분할로 비중 계산
        self._log("Step 4: Recursive bisection for weight allocation")
        weights = self._recursive_bisection(
            self._cov,
            self._sorted_assets
        )

        # 비중 제약 적용
        weights = self._apply_weight_constraints(weights)

        # 클러스터 정보 추출
        clusters = self._extract_clusters(assets)

        # 포트폴리오 특성 계산
        port_vol = self._calculate_portfolio_volatility(weights)
        div_ratio = self._calculate_diversification_ratio(weights)
        eff_n = self._calculate_effective_n(weights)

        # 코멘터리 생성
        commentary = self._generate_commentary(weights, clusters, port_vol, div_ratio)

        result = HRPResult(
            weights=weights,
            clusters=clusters,
            linkage_matrix=self._linkage_matrix,
            sorted_assets=self._sorted_assets,
            expected_volatility=port_vol,
            diversification_ratio=div_ratio,
            effective_n=eff_n,
            commentary=commentary
        )

        self._log(f"Optimization complete. Effective N: {eff_n:.1f}")
        return result

    def _correlation_to_distance(self, corr: pd.DataFrame) -> np.ndarray:
        """
        상관관계를 거리로 변환

        경제학적 의미:
        d = sqrt(0.5 * (1 - rho))
        - rho = 1 (완전 상관) -> d = 0 (거리 0)
        - rho = 0 (무상관) -> d = 0.707
        - rho = -1 (역상관) -> d = 1 (최대 거리)
        """
        dist = np.sqrt(0.5 * (1 - corr.values))
        np.fill_diagonal(dist, 0)  # 대각선은 0
        return dist

    def _hierarchical_clustering(self, dist_matrix: np.ndarray) -> np.ndarray:
        """
        계층적 군집화 수행

        경제학적 의미:
        - Ward 방법: 클러스터 내 분산 최소화
        - 유사한 자산끼리 먼저 병합
        """
        # 거리 행렬을 condensed form으로 변환
        condensed_dist = squareform(dist_matrix, checks=False)

        # 연결 행렬 계산
        linkage_matrix = linkage(condensed_dist, method=self.linkage_method)

        return linkage_matrix

    def _quasi_diagonalization(self, linkage_matrix: np.ndarray) -> List[int]:
        """
        준 대각화: 덴드로그램 순서대로 자산 재배열

        경제학적 의미:
        - 유사한 자산이 인접하게 배치
        - 공분산 행렬이 블록 대각 형태에 가까워짐
        """
        return list(leaves_list(linkage_matrix))

    def _recursive_bisection(
        self,
        cov: pd.DataFrame,
        sorted_assets: List[str]
    ) -> Dict[str, float]:
        """
        재귀적 이분할로 비중 계산

        경제학적 의미:
        - 각 단계에서 두 클러스터의 분산 역비례로 배분
        - 분산이 큰 클러스터에 적은 비중 -> 위험 균등화
        - 역행렬 불필요!
        """
        weights = pd.Series(1.0, index=sorted_assets)
        clusters = [sorted_assets]

        while clusters:
            # 각 클러스터를 이분할
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                # 클러스터를 반으로 분할
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                # 각 서브클러스터의 분산 계산
                var_left = self._cluster_variance(cov, left)
                var_right = self._cluster_variance(cov, right)

                # 분산 역비례로 비중 배분
                # alpha = 1 - var_left / (var_left + var_right)
                # -> var_left가 크면 alpha 작음 -> right에 더 많은 비중
                total_var = var_left + var_right
                if total_var > 0:
                    alpha = 1 - var_left / total_var
                else:
                    alpha = 0.5

                # 비중 업데이트
                weights[left] *= alpha
                weights[right] *= (1 - alpha)

                # 서브클러스터 추가
                if len(left) > 1:
                    new_clusters.append(left)
                if len(right) > 1:
                    new_clusters.append(right)

            clusters = new_clusters

        return weights.to_dict()

    def _cluster_variance(self, cov: pd.DataFrame, assets: List[str]) -> float:
        """
        클러스터 분산 계산 (Inverse Variance 방식)

        경제학적 의미:
        - 역분산 가중 포트폴리오의 분산
        - 개별 자산 분산의 역수로 가중
        """
        cluster_cov = cov.loc[assets, assets]

        # 개별 자산 분산의 역수 (Inverse Variance)
        inv_var = 1 / np.diag(cluster_cov)
        inv_var_weights = inv_var / inv_var.sum()

        # 클러스터 분산 = w' * Cov * w
        cluster_var = np.dot(inv_var_weights, np.dot(cluster_cov, inv_var_weights))

        return cluster_var

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """비중 제약 적용"""
        total = sum(weights.values())

        # 정규화
        weights = {k: v / total for k, v in weights.items()}

        # 최소/최대 비중 적용
        for asset in weights:
            weights[asset] = max(self.min_weight, min(self.max_weight, weights[asset]))

        # 재정규화
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return weights

    def _extract_clusters(self, assets: List[str]) -> List[ClusterInfo]:
        """클러스터 정보 추출"""
        if self._linkage_matrix is None:
            return []

        # 클러스터 수 결정
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = max(2, len(assets) // 3)
            n_clusters = min(n_clusters, len(assets) - 1)

        # 클러스터 할당
        cluster_labels = fcluster(
            self._linkage_matrix,
            t=n_clusters,
            criterion='maxclust'
        )

        clusters = []
        for cluster_id in range(1, n_clusters + 1):
            cluster_assets = [
                assets[i] for i in range(len(assets))
                if cluster_labels[i] == cluster_id
            ]

            if not cluster_assets:
                continue

            # 클러스터 내 평균 상관관계
            if len(cluster_assets) > 1:
                cluster_corr = self._corr.loc[cluster_assets, cluster_assets]
                # 대각선 제외한 평균
                mask = np.ones(cluster_corr.shape, dtype=bool)
                np.fill_diagonal(mask, False)
                avg_corr = cluster_corr.values[mask].mean()
            else:
                avg_corr = 1.0

            clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                assets=cluster_assets,
                avg_correlation=avg_corr,
                total_weight=0.0,  # 나중에 채움
                description=f"Cluster {cluster_id}: {len(cluster_assets)} assets"
            ))

        return clusters

    def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """포트폴리오 변동성 계산"""
        w = np.array([weights.get(a, 0) for a in self._cov.columns])
        port_var = np.dot(w, np.dot(self._cov.values, w))
        return np.sqrt(port_var * 252)  # 연율화

    def _calculate_diversification_ratio(self, weights: Dict[str, float]) -> float:
        """
        분산투자 비율 계산

        경제학적 의미:
        DR = (가중 평균 개별 변동성) / (포트폴리오 변동성)
        DR > 1이면 분산투자 효과 있음
        """
        w = np.array([weights.get(a, 0) for a in self._cov.columns])

        # 개별 자산 변동성
        asset_vols = np.sqrt(np.diag(self._cov.values))

        # 가중 평균 변동성
        weighted_avg_vol = np.dot(w, asset_vols)

        # 포트폴리오 변동성
        port_var = np.dot(w, np.dot(self._cov.values, w))
        port_vol = np.sqrt(port_var)

        if port_vol > 0:
            return weighted_avg_vol / port_vol
        return 1.0

    def _calculate_effective_n(self, weights: Dict[str, float]) -> float:
        """
        유효 자산 수 계산 (Herfindahl Index 역수)

        경제학적 의미:
        - Effective N = 1 / sum(w_i^2)
        - 모든 비중이 동일하면 Effective N = N
        - 집중도가 높으면 Effective N < N
        """
        w = np.array(list(weights.values()))
        hhi = np.sum(w ** 2)
        if hhi > 0:
            return 1 / hhi
        return len(weights)

    def _generate_commentary(
        self,
        weights: Dict[str, float],
        clusters: List[ClusterInfo],
        port_vol: float,
        div_ratio: float
    ) -> str:
        """포트폴리오 코멘터리 생성"""
        lines = []

        lines.append("=" * 60)
        lines.append("HRP PORTFOLIO OPTIMIZATION RESULT")
        lines.append("(Hierarchical Risk Parity - No Matrix Inversion)")
        lines.append("=" * 60)
        lines.append("")

        # 핵심 지표
        lines.append(f"Portfolio Volatility (Ann.): {port_vol:.1%}")
        lines.append(f"Diversification Ratio: {div_ratio:.2f}")
        lines.append(f"Effective N: {self._calculate_effective_n(weights):.1f}")
        lines.append("")

        # Top 10 비중
        lines.append("-" * 40)
        lines.append("TOP 10 ALLOCATIONS")
        lines.append("-" * 40)
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        for asset, weight in sorted_weights[:10]:
            lines.append(f"  {asset:<12}: {weight:>6.1%}")
        lines.append("")

        # 클러스터 정보
        if clusters:
            lines.append("-" * 40)
            lines.append("CLUSTER ANALYSIS")
            lines.append("-" * 40)
            for cluster in clusters:
                cluster_weight = sum(weights.get(a, 0) for a in cluster.assets)
                lines.append(f"  Cluster {cluster.cluster_id}: "
                           f"{len(cluster.assets)} assets, "
                           f"avg_corr={cluster.avg_correlation:.2f}, "
                           f"weight={cluster_weight:.1%}")
                # 클러스터 내 주요 자산
                cluster_assets_weights = [(a, weights.get(a, 0)) for a in cluster.assets]
                cluster_assets_weights.sort(key=lambda x: x[1], reverse=True)
                top_in_cluster = cluster_assets_weights[:3]
                for a, w in top_in_cluster:
                    lines.append(f"    - {a}: {w:.1%}")

        return "\n".join(lines)

    def plot_dendrogram(
        self,
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Asset Clustering Dendrogram"
    ):
        """
        덴드로그램 시각화

        경제학적 의미:
        - 수직 거리가 클수록 자산 간 차이가 큼
        - 일찍 병합될수록 유사한 자산
        """
        try:
            import matplotlib.pyplot as plt

            if self._linkage_matrix is None or self._sorted_assets is None:
                print("No clustering result. Run optimize() first.")
                return

            fig, ax = plt.subplots(figsize=figsize)

            dendrogram(
                self._linkage_matrix,
                labels=list(self._returns.columns),
                leaf_rotation=90,
                ax=ax
            )

            ax.set_title(title)
            ax.set_xlabel("Assets")
            ax.set_ylabel("Distance (sqrt(0.5*(1-corr)))")

            plt.tight_layout()
            plt.savefig('outputs/hrp_dendrogram.png', dpi=150)
            plt.close()

            print("Dendrogram saved to outputs/hrp_dendrogram.png")

        except ImportError:
            print("matplotlib not available for plotting")


# =============================================================================
# Alternative: K-Means + Risk Parity
# =============================================================================

class KMeansRiskParity:
    """
    K-Means 군집화 + Risk Parity 조합

    경제학적 의미:
    - K-Means로 자산을 K개 그룹으로 분류
    - 각 그룹 내에서 동일 비중
    - 그룹 간에는 Risk Parity (분산 역비례)

    HRP 대비 장점:
    - 클러스터 수를 명시적으로 제어 가능
    - 더 균등한 클러스터 크기
    """

    def __init__(
        self,
        n_clusters: int = 5,
        min_weight: float = 0.02,
        max_weight: float = 0.25,
        verbose: bool = False
    ):
        self.n_clusters = n_clusters
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[KMeans-RP] {msg}")

    def optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        K-Means + Risk Parity 최적화

        Args:
            returns: 수익률 DataFrame

        Returns:
            자산별 비중 딕셔너리
        """
        self._log(f"Optimizing with {self.n_clusters} clusters...")

        assets = list(returns.columns)
        corr = returns.corr()

        # Step 1: 상관관계 기반 특성 추출
        features = corr.values  # 각 자산의 상관관계 벡터를 특성으로 사용

        # Step 2: K-Means 군집화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Step 3: 클러스터별 분산 계산
        cov = returns.cov()
        cluster_vars = {}

        for cluster_id in range(self.n_clusters):
            cluster_assets = [
                assets[i] for i in range(len(assets))
                if cluster_labels[i] == cluster_id
            ]
            if cluster_assets:
                cluster_cov = cov.loc[cluster_assets, cluster_assets]
                # 동일 비중 가정 시 클러스터 분산
                n = len(cluster_assets)
                w = np.ones(n) / n
                cluster_var = np.dot(w, np.dot(cluster_cov.values, w))
                cluster_vars[cluster_id] = (cluster_assets, cluster_var)

        # Step 4: Risk Parity로 클러스터 간 비중 배분
        total_inv_var = sum(1/v[1] for v in cluster_vars.values() if v[1] > 0)

        weights = {}
        for cluster_id, (cluster_assets, cluster_var) in cluster_vars.items():
            if cluster_var > 0:
                cluster_weight = (1 / cluster_var) / total_inv_var
            else:
                cluster_weight = 1 / len(cluster_vars)

            # 클러스터 내 동일 비중
            asset_weight = cluster_weight / len(cluster_assets)
            for asset in cluster_assets:
                weights[asset] = asset_weight

        # 비중 제약 적용
        for asset in weights:
            weights[asset] = max(self.min_weight, min(self.max_weight, weights[asset]))

        # 재정규화
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}

        self._log(f"Optimization complete. {len(weights)} assets allocated.")
        return weights


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import yfinance as yf
    from datetime import datetime, timedelta

    print("Testing HRP Optimizer...")
    print()

    # 테스트 데이터 수집
    tickers = ["SPY", "QQQ", "TLT", "GLD", "IWM", "EFA", "USO", "UUP"]

    print("1. Fetching test data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    price_data = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                price_data[ticker] = data['Close']
        except Exception as e:
            print(f"  {ticker}: Failed - {e}")

    prices = pd.DataFrame(price_data)
    returns = np.log(prices / prices.shift(1)).dropna()
    print(f"   Returns shape: {returns.shape}")

    # HRP 최적화
    print("\n2. Running HRP optimization...")
    hrp = HRPOptimizer(verbose=True)
    result = hrp.optimize(returns)

    print("\n3. Results:")
    print(result.commentary)

    # 덴드로그램
    print("\n4. Generating dendrogram...")
    hrp.plot_dendrogram()

    # K-Means Risk Parity 비교
    print("\n5. Comparing with K-Means Risk Parity...")
    kmeans_rp = KMeansRiskParity(n_clusters=3, verbose=True)
    kmeans_weights = kmeans_rp.optimize(returns)

    print("\nK-Means RP Weights:")
    for asset, weight in sorted(kmeans_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  {asset}: {weight:.1%}")
