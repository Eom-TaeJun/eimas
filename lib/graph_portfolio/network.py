#!/usr/bin/env python3
"""
Graph Portfolio - Correlation Network
============================================================

상관관계 네트워크 구축

Economic Foundation:
    - Mantegna (1999): "Hierarchical structure in financial markets"
    - Distance metric: d = sqrt(2 * (1 - ρ))
    - MST for systemic risk analysis

Class:
    - CorrelationNetwork: 상관관계 그래프 구축 및 MST 분석
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import logging

from .schemas import SystemicRiskNode, MSTAnalysisResult, OutlierDetectionResult

logger = logging.getLogger(__name__)

# Scipy imports
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    csr_matrix = None
    minimum_spanning_tree = None


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

    def compute_systemic_similarity(self) -> pd.DataFrame:
        """
        Systemic Similarity 계산 (D_bar matrix)

        경제학적 배경 (eco1.docx):
        - 단순 correlation → distance 변환을 넘어서
        - 각 자산 쌍이 다른 모든 자산과 어떻게 관계되어 있는지 고려
        - "Systemic Similarity" = 자산 간 상호작용 강도 정량화

        수식:
            D_bar[i,j] = sqrt(sum_k (D[k,i] - D[k,j])²)

        해석:
        - D_bar[i,j] = 0: 자산 i와 j가 모든 다른 자산과 동일한 거리 관계
                          → 시스템적으로 매우 유사 (대체재)
        - D_bar[i,j] 큼: 자산 i와 j가 다른 자산과 다른 관계
                          → 시스템적으로 상이 (보완재)

        활용:
        - HRP (Hierarchical Risk Parity) 고도화
        - 클러스터링 품질 향상 (MST, Hierarchical Clustering)
        - 포트폴리오 분산화 효과 정량화

        Returns:
            d_bar: Systemic Similarity 행렬 (n × n DataFrame)
                   인덱스/컬럼 = 자산 티커
                   값 = 시스템적 유사도 (낮을수록 유사)

        References:
            De Prado, M. L. (2016). Building Diversified Portfolios that
            Outperform Out of Sample. Journal of Portfolio Management, 42(4).

        Example:
            >>> network = CorrelationNetwork()
            >>> network.build_from_returns(returns_df)
            >>> d_bar = network.compute_systemic_similarity()
            >>> print(f"Most similar pair: {d_bar.min().min():.3f}")
            >>> print(f"Most dissimilar pair: {d_bar.max().max():.3f}")

        Notes:
            - distance_matrix가 먼저 계산되어야 함 (build_from_returns 호출 필수)
            - O(n³) 복잡도이므로 대규모 자산(>500)에서는 주의
            - 대각선 값은 0 (자기 자신과의 유사도)
        """
        if self.distance_matrix is None:
            raise ValueError(
                "Distance matrix not computed yet. "
                "Call build_from_returns() first."
            )

        # Distance 행렬을 numpy array로 변환
        D = self.distance_matrix.values  # (n, n)
        n = D.shape[0]

        # Systemic Similarity 행렬 초기화
        D_bar = np.zeros((n, n))

        # D_bar[i,j] = sqrt(sum_k (D[k,i] - D[k,j])²)
        for i in range(n):
            for j in range(n):
                if i == j:
                    D_bar[i, j] = 0.0  # 자기 자신과의 유사도
                else:
                    # 모든 k에 대해 (D[k,i] - D[k,j])² 합산
                    diff = D[:, i] - D[:, j]  # (n,) array
                    D_bar[i, j] = np.sqrt(np.sum(diff ** 2))

        # DataFrame으로 변환 (인덱스/컬럼 = 자산 티커)
        assets = self.distance_matrix.index.tolist()
        d_bar_df = pd.DataFrame(
            D_bar,
            index=assets,
            columns=assets
        )

        return d_bar_df

    def detect_outliers_dbscan(
        self,
        eps: float = 0.5,
        min_samples: int = 3,
        metric: str = 'precomputed'
    ) -> OutlierDetectionResult:
        """
        DBSCAN 기반 이상치 탐지 (밀도 기반 클러스터링)

        경제학적 배경 (금융경제정리.docx):
        - DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        - 밀도가 낮은 자산 = 다른 자산들과 상관관계 패턴이 다름
        - 노이즈 포인트 (label=-1) = 포트폴리오 품질 저하 요인
        - 이상치 제거로 HRP 클러스터링 품질 향상

        알고리즘:
        1. 거리 행렬 기반 DBSCAN 실행
        2. 각 자산을 클러스터 또는 노이즈로 분류
        3. 노이즈 포인트 = 이상치 (outlier)

        Parameters:
            eps: float
                DBSCAN epsilon 파라미터 (이웃 반경)
                - 작을수록 엄격한 이상치 탐지
                - 권장: 0.3-0.7 (거리 행렬 스케일)
            min_samples: int
                최소 클러스터 크기
                - 권장: 3-5
            metric: str
                거리 메트릭 (기본값: 'precomputed')
                - 거리 행렬을 직접 사용

        Returns:
            OutlierDetectionResult:
                - n_outliers: 이상치 개수
                - outlier_tickers: 이상치 자산 리스트
                - normal_tickers: 정상 자산 리스트
                - cluster_labels: 각 자산의 클러스터 ID

        Example:
            >>> network = CorrelationNetwork()
            >>> network.build_from_returns(returns_df)
            >>> result = network.detect_outliers_dbscan(eps=0.5, min_samples=3)
            >>> print(f"Outliers: {result.n_outliers}/{result.n_total_assets}")
            >>> print(f"Outlier tickers: {result.outlier_tickers}")

        Notes:
            - distance_matrix가 먼저 계산되어야 함
            - 이상치 제거 후 HRP 재실행 권장
            - eps 값은 데이터셋에 따라 튜닝 필요

        References:
            Ester, M., et al. (1996). "A density-based algorithm for discovering
            clusters in large spatial databases with noise." KDD-96.
        """
        if self.distance_matrix is None:
            raise ValueError(
                "Distance matrix not computed yet. "
                "Call build_from_returns() first."
            )

        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn not available. "
                "Install with: pip install scikit-learn"
            )

        from sklearn.cluster import DBSCAN

        # 거리 행렬 추출
        distance_array = self.distance_matrix.values
        assets = self.distance_matrix.index.tolist()
        n_assets = len(assets)

        # DBSCAN 실행
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )
        labels = dbscan.fit_predict(distance_array)

        # 이상치 식별 (label == -1)
        outlier_mask = (labels == -1)
        outlier_tickers = [assets[i] for i in range(n_assets) if outlier_mask[i]]
        normal_tickers = [assets[i] for i in range(n_assets) if not outlier_mask[i]]

        # 클러스터 레이블 매핑
        cluster_labels = {
            assets[i]: int(labels[i])
            for i in range(n_assets)
        }

        # 유효 클러스터 개수 (노이즈 제외)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # 이상치 비율
        outlier_ratio = len(outlier_tickers) / n_assets if n_assets > 0 else 0.0

        # 해석
        if outlier_ratio == 0:
            interpretation = "NONE: 이상치 없음 (모든 자산이 밀집 클러스터 형성)"
        elif outlier_ratio < 0.05:
            interpretation = f"LOW: {outlier_ratio:.1%}의 자산이 이상치 (정상 범위)"
        elif outlier_ratio < 0.15:
            interpretation = f"MEDIUM: {outlier_ratio:.1%}의 자산이 이상치 (검토 권장)"
        else:
            interpretation = f"HIGH: {outlier_ratio:.1%}의 자산이 이상치 (eps 파라미터 재조정 필요)"

        return OutlierDetectionResult(
            timestamp=datetime.now().isoformat(),
            n_total_assets=n_assets,
            n_outliers=len(outlier_tickers),
            outlier_ratio=outlier_ratio,
            outlier_tickers=outlier_tickers,
            normal_tickers=normal_tickers,
            cluster_labels=cluster_labels,
            n_clusters=n_clusters,
            eps=eps,
            min_samples=min_samples,
            interpretation=interpretation
        )

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


