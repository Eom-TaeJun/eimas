"""
Graph-Clustered Hierarchical Risk Parity (GC-HRP)
==================================================
무한 에셋(N → ∞) 환경에서의 포트폴리오 최적화 엔진

경제학적 배경:
- 토큰화로 자산 수가 무한대로 확장되는 환경
- 공분산 행렬의 역행렬(Σ⁻¹) 특이점 문제 해결
- 전통적 Mean-Variance 대신 Graph + Clustering + HRP 접근

핵심 아이디어:
1. Correlation Network: 자산 간 상관관계를 그래프로 표현
2. Community Detection: 유사 자산을 클러스터로 묶음
3. Representative Selection: 클러스터당 대표 자산 선정
4. HRP: 축소된 유니버스에서 계층적 리스크 배분

References:
- Lopez de Prado (2016): "Building Diversified Portfolios that Outperform Out-of-Sample"
- Louvain Algorithm: Blondel et al. (2008)
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# Optional imports with fallbacks
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not available. Clustering will use fallback methods.")

try:
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    csr_matrix = None
    minimum_spanning_tree = None
    print("[WARN] scipy not available. HRP will use simplified method.")

try:
    import community as community_louvain  # python-louvain
    LOUVAIN_AVAILABLE = True
except ImportError:
    LOUVAIN_AVAILABLE = False


class ClusteringMethod(Enum):
    """클러스터링 방법론"""
    KMEANS = "kmeans"
    GAUSSIAN_MIXTURE = "gmm"
    LOUVAIN = "louvain"  # Graph-based
    HIERARCHICAL = "hierarchical"


class RepresentativeMethod(Enum):
    """대표 자산 선정 방법"""
    CENTRALITY = "centrality"      # 그래프 중심성
    VOLUME = "volume"              # 거래량 기반
    LIQUIDITY = "liquidity"        # 유동성 기반
    SHARPE = "sharpe"              # 샤프 비율 기반


@dataclass
class ClusterInfo:
    """클러스터 정보"""
    cluster_id: int
    assets: List[str]
    representative: str
    intra_correlation: float  # 클러스터 내 평균 상관관계
    size: int
    total_volume: float = 0.0
    avg_volatility: float = 0.0


@dataclass
class SystemicRiskNode:
    """시스템 리스크 유발 노드 정보 (v2 - Eigenvector 제거)"""
    ticker: str
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    composite_score: float  # 종합 중심성 점수
    mst_connections: int    # MST에서의 연결 수
    risk_interpretation: str
    # v2: eigenvector_centrality 제거됨 (트리 구조에서 비효율적)


@dataclass
class MSTAnalysisResult:
    """MST 분석 결과"""
    timestamp: str
    n_nodes: int
    n_edges: int
    total_mst_weight: float
    avg_distance: float
    systemic_risk_nodes: List[SystemicRiskNode]
    mst_edges: List[Tuple[str, str, float]]  # (node1, node2, distance)
    risk_summary: str


@dataclass
class PortfolioAllocation:
    """포트폴리오 배분 결과"""
    timestamp: str
    weights: Dict[str, float]
    cluster_weights: Dict[int, float]
    risk_contributions: Dict[str, float]
    expected_volatility: float
    diversification_ratio: float
    effective_n: float  # 실효 자산 수
    methodology: str
    clusters: List[ClusterInfo] = field(default_factory=list)
    mst_analysis: Optional[MSTAnalysisResult] = None  # MST 시스템 리스크 분석

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['clusters'] = [asdict(c) for c in self.clusters]
        if self.mst_analysis:
            result['mst_analysis'] = {
                'timestamp': self.mst_analysis.timestamp,
                'n_nodes': self.mst_analysis.n_nodes,
                'n_edges': self.mst_analysis.n_edges,
                'total_mst_weight': self.mst_analysis.total_mst_weight,
                'avg_distance': self.mst_analysis.avg_distance,
                'systemic_risk_nodes': [
                    {
                        'ticker': n.ticker,
                        'degree_centrality': n.degree_centrality,
                        'betweenness_centrality': n.betweenness_centrality,
                        'closeness_centrality': n.closeness_centrality,
                        'composite_score': n.composite_score,
                        'mst_connections': n.mst_connections,
                        'risk_interpretation': n.risk_interpretation
                    }
                    for n in self.mst_analysis.systemic_risk_nodes
                ],
                'risk_summary': self.mst_analysis.risk_summary
            }
        return result


class CorrelationNetwork:
    """
    상관관계 네트워크 구축

    경제학적 의미:
    - 노드: 개별 자산
    - 엣지: 상관관계 (threshold 이상만)
    - 엣지 가중치: |correlation| 또는 1 - |correlation| (거리)
    """

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        use_volume_weight: bool = True
    ):
        self.threshold = correlation_threshold
        self.use_volume_weight = use_volume_weight
        self.graph = nx.Graph()
        self.correlation_matrix = None
        self.distance_matrix = None

    def build_from_returns(
        self,
        returns: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None
    ) -> nx.Graph:
        """
        수익률 데이터에서 상관관계 네트워크 구축

        Args:
            returns: 자산별 수익률 (columns = assets)
            volumes: 거래량 데이터 (정보 비대칭 가중치용)

        Returns:
            NetworkX Graph
        """
        # 상관관계 행렬 계산
        self.correlation_matrix = returns.corr()

        # 거리 행렬: d = sqrt(2 * (1 - corr))
        # 이는 상관관계를 유클리드 거리로 변환
        self.distance_matrix = np.sqrt(2 * (1 - self.correlation_matrix))

        assets = returns.columns.tolist()

        # 그래프 구축
        self.graph = nx.Graph()

        # 노드 추가 (자산 속성 포함)
        for asset in assets:
            asset_returns = returns[asset].dropna()

            node_attrs = {
                'volatility': asset_returns.std() * np.sqrt(252),
                'mean_return': asset_returns.mean() * 252,
                'sharpe': (asset_returns.mean() * 252) / (asset_returns.std() * np.sqrt(252) + 1e-10)
            }

            if volumes is not None and asset in volumes.columns:
                node_attrs['avg_volume'] = volumes[asset].mean()
                # 거래량 급증 비율 (정보 비대칭 프록시)
                vol_ma = volumes[asset].rolling(20).mean()
                node_attrs['volume_surge_ratio'] = (volumes[asset] / vol_ma).mean()

            self.graph.add_node(asset, **node_attrs)

        # === 벡터화된 엣지 추가 (O(n^2) 반복문 제거) ===
        n = len(assets)
        corr_array = self.correlation_matrix.values
        dist_array = self.distance_matrix.values

        # Threshold 초과하는 상관관계 찾기 (벡터화)
        abs_corr = np.abs(corr_array)
        # 상삼각 마스크 (중복 방지: i < j)
        upper_tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        # threshold 조건과 결합
        edge_mask = (abs_corr >= self.threshold) & upper_tri_mask

        # 유효한 엣지 인덱스 추출
        rows, cols = np.where(edge_mask)

        # 거래량 가중치 사전 계산 (벡터화)
        vol_weights = None
        if self.use_volume_weight and volumes is not None:
            vol_means = np.array([
                volumes[asset].mean() if asset in volumes.columns else 1.0
                for asset in assets
            ])
            # 모든 페어에 대한 가중치 행렬 (대칭)
            vol_i, vol_j = np.meshgrid(vol_means, vol_means, indexing='ij')
            vol_weights = 1 + np.log1p((vol_i + vol_j) / 2) * 0.1

        # 배치 엣지 생성
        edges = []
        for idx in range(len(rows)):
            i, j = rows[idx], cols[idx]
            weight = abs_corr[i, j]
            if vol_weights is not None:
                weight *= vol_weights[i, j]

            edges.append((
                assets[i],
                assets[j],
                {
                    'correlation': corr_array[i, j],
                    'distance': dist_array[i, j],
                    'weight': weight
                }
            ))

        # 배치 엣지 추가 (단일 호출)
        self.graph.add_edges_from(edges)

        return self.graph

    def get_adjacency_matrix(self) -> pd.DataFrame:
        """인접 행렬 반환"""
        return nx.to_pandas_adjacency(self.graph)

    def calculate_centrality(self) -> Dict[str, Dict[str, float]]:
        """
        노드 중심성 계산

        경제학적 의미:
        - degree: 많은 자산과 연결 = 시장 전반과 동조
        - eigenvector/pagerank: 중요한 자산과 연결 = 핵심 자산
        - betweenness: 클러스터 간 연결 = 분산화에 중요
        """
        if len(self.graph.nodes) == 0:
            return {}

        centrality = {}

        degree = nx.degree_centrality(self.graph)

        # Eigenvector centrality - disconnected graph에서는 PageRank 사용
        try:
            if nx.is_connected(self.graph):
                eigenvector = nx.eigenvector_centrality_numpy(self.graph)
            else:
                # Disconnected graph: PageRank 사용 (모든 그래프에서 작동)
                eigenvector = nx.pagerank(self.graph, weight='weight')
        except Exception:
            # Fallback: degree centrality 사용
            eigenvector = degree.copy()

        betweenness = nx.betweenness_centrality(self.graph)

        for node in self.graph.nodes:
            centrality[node] = {
                'degree': degree.get(node, 0),
                'eigenvector': eigenvector.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'composite': (degree.get(node, 0) + eigenvector.get(node, 0) + betweenness.get(node, 0)) / 3
            }

        return centrality

    def build_mst(self) -> nx.Graph:
        """
        최소 신장 트리(MST) 구축

        경제학적 의미:
        - MST는 모든 자산을 연결하는 최소 거리(= 최대 상관관계) 경로
        - 시장 충격 전파의 핵심 경로를 나타냄
        - Mantegna (1999) "Hierarchical structure in financial markets"

        Distance: d = sqrt(2 * (1 - rho))
        - rho = 1 (완전 상관) → d = 0
        - rho = 0 (무상관) → d = sqrt(2) ≈ 1.414
        - rho = -1 (역상관) → d = 2

        최적화: scipy.sparse.csgraph.minimum_spanning_tree 사용
        - O(E log V) 복잡도
        - 희소 행렬 기반으로 메모리 효율적
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not computed. Call build_from_returns first.")

        assets = list(self.graph.nodes)

        # scipy 사용 가능하면 최적화된 버전 사용
        if SCIPY_AVAILABLE and minimum_spanning_tree is not None:
            return self._build_mst_scipy(assets)
        else:
            return self._build_mst_legacy(assets)

    def _build_mst_scipy(self, assets: List[str]) -> nx.Graph:
        """
        scipy 기반 최적화된 MST 구축 (O(E log V))
        """
        n = len(assets)
        dist_array = self.distance_matrix.values.copy()
        corr_array = self.correlation_matrix.values

        # 대각선을 0으로 설정 (자기 자신과의 거리)
        np.fill_diagonal(dist_array, 0)

        # 희소 행렬로 변환 후 MST 계산
        dist_sparse = csr_matrix(dist_array)
        mst_sparse = minimum_spanning_tree(dist_sparse)

        # NetworkX 그래프로 변환
        mst = nx.Graph()
        for i, asset in enumerate(assets):
            mst.add_node(asset)

        # 희소 행렬에서 엣지 추출 (COO 형식 사용)
        mst_coo = mst_sparse.tocoo()
        for i, j, weight in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            if weight > 0:
                mst.add_edge(
                    assets[i],
                    assets[j],
                    weight=weight,
                    correlation=corr_array[i, j]
                )

        return mst

    def _build_mst_legacy(self, assets: List[str]) -> nx.Graph:
        """
        기존 방식 MST 구축 (scipy 없을 때 fallback)
        """
        n = len(assets)

        # 완전 연결 그래프 생성 (MST 계산용)
        complete_graph = nx.Graph()
        for asset in assets:
            complete_graph.add_node(asset)

        # 엣지 배치 생성
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((
                    assets[i],
                    assets[j],
                    {
                        'weight': self.distance_matrix.iloc[i, j],
                        'correlation': self.correlation_matrix.iloc[i, j]
                    }
                ))
        complete_graph.add_edges_from(edges)

        # Prim/Kruskal 알고리즘으로 MST 계산
        mst = nx.minimum_spanning_tree(complete_graph, weight='weight')

        return mst

    # v2 가중치 (API 검증 결과 반영 - Eigenvector 제거)
    CENTRALITY_WEIGHTS = {
        'betweenness': 0.45,  # 충격 전파 핵심 (최고 가중치)
        'degree': 0.35,       # 허브 식별
        'closeness': 0.20,    # 정보 흐름 속도
        # 'eigenvector': 0.0  # 트리 구조에서 비효율적 → 제거
    }

    def identify_systemic_risk_nodes(
        self,
        top_n: int = 3,
        adaptive: bool = True
    ) -> MSTAnalysisResult:
        """
        MST 기반 시스템 리스크 유발 노드 식별 (v2 - API 검증 반영)

        경제학적 의미:
        - MST에서 높은 중심성 = 시장 전체와의 강한 연결성
        - 이 노드의 충격은 빠르게 전파됨
        - 포트폴리오에서 주의 깊게 모니터링 필요

        중심성 지표 (v2 가중치):
        1. Betweenness Centrality (45%): 최단 경로 상의 위치 - 충격 전파 핵심
        2. Degree Centrality (35%): 직접 연결된 자산 수 - 허브 식별
        3. Closeness Centrality (20%): 모든 노드까지의 평균 거리
        * Eigenvector Centrality: 제거됨 (트리 구조에서 비효율적)

        Args:
            top_n: 기본 노드 수
            adaptive: True면 포트폴리오 크기에 따라 동적 조정
        """
        mst = self.build_mst()
        assets = list(mst.nodes)
        n_assets = len(assets)

        # 적응형 노드 선택 (v2)
        if adaptive:
            top_n = self._adaptive_node_selection(n_assets, top_n)

        if n_assets < top_n:
            top_n = n_assets

        # MST에서 중심성 계산 (v2 - Eigenvector 제거)
        degree_cent = nx.degree_centrality(mst)
        betweenness_cent = nx.betweenness_centrality(mst)
        closeness_cent = nx.closeness_centrality(mst)

        # 종합 점수 계산 (v2 가중치)
        composite_scores = {}
        for asset in assets:
            composite_scores[asset] = (
                self.CENTRALITY_WEIGHTS['betweenness'] * betweenness_cent.get(asset, 0) +
                self.CENTRALITY_WEIGHTS['degree'] * degree_cent.get(asset, 0) +
                self.CENTRALITY_WEIGHTS['closeness'] * closeness_cent.get(asset, 0)
            )

        # 상위 N개 노드 선정
        sorted_nodes = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        top_nodes = sorted_nodes[:top_n]

        # SystemicRiskNode 객체 생성 (v2 - Eigenvector 제거)
        systemic_nodes = []
        for ticker, score in top_nodes:
            mst_connections = mst.degree(ticker)

            # 리스크 해석 생성 (v2 - 더 상세한 기준)
            betweenness = betweenness_cent.get(ticker, 0)
            degree = degree_cent.get(ticker, 0)
            closeness = closeness_cent.get(ticker, 0)

            if betweenness > 0.20:
                interpretation = "Critical bridge - Primary shock transmission channel"
            elif betweenness > 0.10:
                interpretation = "High betweenness - Important bridge node for shock propagation"
            elif degree > 0.15:
                interpretation = "Hub node - Connected to many assets, concentration risk"
            elif closeness > 0.5:
                interpretation = "Central position - Quick contagion risk to periphery"
            elif degree > 0.08:
                interpretation = "Moderate hub - Some concentration risk"
            else:
                interpretation = "Moderate systemic importance"

            systemic_nodes.append(SystemicRiskNode(
                ticker=ticker,
                degree_centrality=round(degree, 4),
                betweenness_centrality=round(betweenness, 4),
                closeness_centrality=round(closeness, 4),
                composite_score=round(score, 4),
                mst_connections=mst_connections,
                risk_interpretation=interpretation
            ))

        # MST 엣지 정보
        mst_edges = [
            (u, v, round(mst[u][v]['weight'], 4))
            for u, v in mst.edges()
        ]

        # 총 MST 가중치 및 평균 거리
        total_weight = sum(mst[u][v]['weight'] for u, v in mst.edges())
        avg_distance = total_weight / len(mst_edges) if mst_edges else 0

        # 리스크 요약
        top_tickers = [node.ticker for node in systemic_nodes]
        risk_summary = (
            f"Top {top_n} systemic risk nodes identified: {', '.join(top_tickers)}. "
            f"Average MST distance: {avg_distance:.3f} (lower = higher correlation). "
            f"These assets act as key transmission channels for market shocks."
        )

        return MSTAnalysisResult(
            timestamp=datetime.now().isoformat(),
            n_nodes=n_assets,
            n_edges=len(mst_edges),
            total_mst_weight=round(total_weight, 4),
            avg_distance=round(avg_distance, 4),
            systemic_risk_nodes=systemic_nodes,
            mst_edges=mst_edges,
            risk_summary=risk_summary
        )

    def _adaptive_node_selection(self, n_assets: int, default_n: int = 3) -> int:
        """
        포트폴리오 크기에 따른 적응형 시스템 리스크 노드 수 결정

        v2 로직 (API 검증 결과 반영):
        - 고정 3개 대신 포트폴리오 크기에 비례
        - 최소 3개, 최대 10개
        - sqrt(N) 기반 휴리스틱

        Args:
            n_assets: 총 자산 수
            default_n: 기본값

        Returns:
            선택할 시스템 리스크 노드 수
        """
        if n_assets <= 10:
            return min(3, n_assets)
        elif n_assets <= 30:
            return 3
        elif n_assets <= 100:
            # sqrt(N) 기반
            return max(3, min(5, int(np.sqrt(n_assets))))
        elif n_assets <= 500:
            return max(5, min(8, int(np.sqrt(n_assets) * 0.7)))
        else:
            # 대규모 포트폴리오
            return min(10, int(np.sqrt(n_assets) * 0.5))

    def rolling_mst_analysis(
        self,
        returns: pd.DataFrame,
        window: int = 252,
        step: int = 21,
        top_n: int = 3
    ) -> List[Dict]:
        """
        롤링 윈도우 MST 분석 (v2 - API 검증 결과 반영)

        시간에 따른 시스템 리스크 노드 변화 추적
        - 위기 전후 MST 구조 변화 감지
        - 시스템 리스크 노드의 지속성/전환 모니터링

        Args:
            returns: 수익률 DataFrame
            window: 롤링 윈도우 크기 (기본 252일 = 1년)
            step: 스텝 크기 (기본 21일 = 1개월)
            top_n: 추적할 상위 노드 수

        Returns:
            시간별 MST 분석 결과 리스트
        """
        results = []
        n_periods = len(returns)

        if n_periods < window:
            print(f"[WARN] Data length ({n_periods}) < window ({window}). Using full data.")
            window = n_periods

        for start in range(0, n_periods - window + 1, step):
            end = start + window
            window_returns = returns.iloc[start:end]
            window_date = returns.index[end - 1] if hasattr(returns.index[end-1], 'strftime') else str(end)

            # 윈도우 데이터로 네트워크 구축
            temp_network = CorrelationNetwork(
                correlation_threshold=self.threshold,
                use_volume_weight=False
            )
            temp_network.build_from_returns(window_returns)

            # MST 분석
            try:
                mst_result = temp_network.identify_systemic_risk_nodes(top_n=top_n, adaptive=False)

                results.append({
                    'period_end': str(window_date),
                    'start_idx': start,
                    'end_idx': end,
                    'top_nodes': [n.ticker for n in mst_result.systemic_risk_nodes],
                    'avg_distance': mst_result.avg_distance,
                    'top_scores': {n.ticker: n.composite_score for n in mst_result.systemic_risk_nodes}
                })
            except Exception as e:
                print(f"[WARN] MST analysis failed for period {start}-{end}: {e}")
                continue

        # 노드 지속성 분석
        if results:
            node_frequency = {}
            for r in results:
                for node in r['top_nodes']:
                    node_frequency[node] = node_frequency.get(node, 0) + 1

            # 가장 빈번한 시스템 리스크 노드
            persistent_nodes = sorted(node_frequency.items(), key=lambda x: x[1], reverse=True)[:5]

            print(f"\n[Rolling MST] Analyzed {len(results)} periods")
            print(f"[Rolling MST] Most persistent systemic risk nodes:")
            for node, freq in persistent_nodes:
                pct = freq / len(results) * 100
                print(f"  - {node}: {freq}/{len(results)} periods ({pct:.1f}%)")

        return results


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


class RepresentativeSelector:
    """
    클러스터 대표 자산 선정

    경제학적 의미:
    - 각 클러스터를 대표하는 자산 1-3개 선정
    - N=10,000 → N'=50~100으로 차원 축소
    """

    def __init__(
        self,
        method: RepresentativeMethod = RepresentativeMethod.CENTRALITY,
        max_representatives: int = 3
    ):
        self.method = method
        self.max_representatives = max_representatives

    def select(
        self,
        clusters: Dict[int, List[str]],
        network: CorrelationNetwork,
        volumes: Optional[pd.DataFrame] = None
    ) -> Dict[int, List[str]]:
        """
        각 클러스터에서 대표 자산 선정

        Returns:
            {cluster_id: [representative_assets]}
        """
        representatives = {}
        centrality = network.calculate_centrality()

        for cluster_id, assets in clusters.items():
            if len(assets) == 1:
                representatives[cluster_id] = assets
                continue

            # 자산별 점수 계산
            scores = {}
            for asset in assets:
                if self.method == RepresentativeMethod.CENTRALITY:
                    scores[asset] = centrality.get(asset, {}).get('eigenvector', 0)

                elif self.method == RepresentativeMethod.VOLUME:
                    if volumes is not None and asset in volumes.columns:
                        scores[asset] = volumes[asset].mean()
                    else:
                        scores[asset] = network.graph.nodes[asset].get('avg_volume', 0)

                elif self.method == RepresentativeMethod.SHARPE:
                    scores[asset] = network.graph.nodes[asset].get('sharpe', 0)

                elif self.method == RepresentativeMethod.LIQUIDITY:
                    # 유동성 = 거래량 / 변동성
                    vol = network.graph.nodes[asset].get('avg_volume', 1)
                    volatility = network.graph.nodes[asset].get('volatility', 1)
                    scores[asset] = vol / (volatility + 1e-10)

            # 상위 자산 선정
            sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            n_reps = min(self.max_representatives, len(sorted_assets))
            representatives[cluster_id] = [a[0] for a in sorted_assets[:n_reps]]

        return representatives


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


# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_data(n_assets: int = 100, n_days: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """테스트용 샘플 데이터 생성"""
    np.random.seed(42)

    # 팩터 기반 수익률 생성
    n_factors = 5
    factor_returns = np.random.randn(n_days, n_factors) * 0.01

    # 자산별 팩터 로딩
    loadings = np.random.randn(n_assets, n_factors)

    # 자산 수익률 = 팩터 수익률 × 로딩 + 고유 수익률
    idiosyncratic = np.random.randn(n_days, n_assets) * 0.02
    returns = np.dot(factor_returns, loadings.T) + idiosyncratic

    # 거래량 (로그정규분포)
    volumes = np.exp(np.random.randn(n_days, n_assets) + 10)

    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    assets = [f'ASSET_{i:03d}' for i in range(n_assets)]

    returns_df = pd.DataFrame(returns, index=dates, columns=assets)
    volumes_df = pd.DataFrame(volumes, index=dates, columns=assets)

    return returns_df, volumes_df


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Graph-Clustered HRP Test")
    print("=" * 60)

    # 샘플 데이터 생성
    print("\n1. Generating sample data (100 assets, 252 days)...")
    returns, volumes = create_sample_data(n_assets=100, n_days=252)
    print(f"   Returns shape: {returns.shape}")
    print(f"   Volumes shape: {volumes.shape}")

    # GC-HRP 실행
    print("\n2. Running Graph-Clustered HRP...")
    gc_hrp = GraphClusteredPortfolio(
        correlation_threshold=0.3,
        clustering_method=ClusteringMethod.KMEANS if SKLEARN_AVAILABLE else ClusteringMethod.HIERARCHICAL,
        representative_method=RepresentativeMethod.CENTRALITY,
        max_representatives_per_cluster=2
    )

    allocation = gc_hrp.fit(returns, volumes)

    # 결과 출력
    print("\n3. Results:")
    print(f"   Methodology: {allocation.methodology}")
    print(f"   Number of clusters: {len(allocation.clusters)}")
    print(f"   Expected Volatility: {allocation.expected_volatility:.2%}")
    print(f"   Diversification Ratio: {allocation.diversification_ratio:.2f}")
    print(f"   Effective N: {allocation.effective_n:.1f}")

    print("\n4. Top 10 Weights:")
    sorted_weights = sorted(allocation.weights.items(), key=lambda x: x[1], reverse=True)
    for asset, weight in sorted_weights[:10]:
        print(f"   {asset}: {weight:.2%}")

    print("\n5. Cluster Summary:")
    summary = gc_hrp.get_cluster_summary()
    print(summary.to_string())

    # MST 시스템 리스크 분석 출력 (v2 - Eigenvector 제거)
    print("\n6. MST Systemic Risk Analysis (v2):")
    if allocation.mst_analysis:
        mst = allocation.mst_analysis
        print(f"   MST Nodes: {mst.n_nodes}, Edges: {mst.n_edges}")
        print(f"   Average Distance: {mst.avg_distance:.4f}")
        print(f"\n   Top Systemic Risk Nodes (Adaptive Selection):")
        print("   " + "-" * 65)
        print(f"   {'Ticker':<12} {'Between':>10} {'Degree':>8} {'Close':>8} {'Score':>8}")
        print("   " + "-" * 65)
        for node in mst.systemic_risk_nodes:
            print(f"   {node.ticker:<12} {node.betweenness_centrality:>10.4f} "
                  f"{node.degree_centrality:>8.4f} {node.closeness_centrality:>8.4f} "
                  f"{node.composite_score:>8.4f}")
            print(f"   └─ {node.risk_interpretation}")
        print("   " + "-" * 65)
        print(f"\n   Centrality Weights (v2): Between=45%, Degree=35%, Close=20%")
        print(f"   Risk Summary: {mst.risk_summary}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
