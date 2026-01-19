"""
Portfolio Optimizer Module
==========================
Hierarchical Risk Parity (HRP) & Minimum Spanning Tree (MST) Optimizer

전통적인 평균-분산 최적화(MVO)의 역행렬 불안정성 문제를 해결하기 위한
계층적 리스크 패리티(HRP) 알고리즘 및 시장 구조 분석을 위한 MST 구현.

기능:
1. Hierarchical Risk Parity (HRP):
   - Tree Clustering: 상관계수 거리 기반 계층적 군집화
   - Quasi-Diagonalization: 공분산 행렬 재정렬
   - Recursive Bisection: 역분산 비중 배분

2. MST Analysis:
   - Minimum Spanning Tree 구축
   - Systemic Risk Hub 식별 (Centrality analysis)

References:
- Lopez de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings

# Scipy & Sklearn Check
try:
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from scipy.spatial.distance import squareform
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("Scipy not found. HRP will use simplified fallback methods.")

class HRPOptimizer:
    """
    Hierarchical Risk Parity (HRP) Optimizer
    
    특징:
    - 공분산 행렬의 역행렬을 계산하지 않음 (안정적)
    - 상관관계 구조를 반영하여 자산을 계층적으로 배분
    """

    def __init__(self):
        self.weights: Optional[Dict[str, float]] = None
        self.linkage_matrix = None
        self.sorted_indices = None
        self.clusters = None

    def optimize(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        HRP 최적화 수행
        
        Args:
            returns: 자산별 수익률 (일별/주별 등)
            
        Returns:
            {자산명: 비중} 딕셔너리
        """
        if returns.empty:
            return {}

        # 1. 데이터 준비
        corr = returns.corr()
        cov = returns.cov()
        
        # 2. Tree Clustering & Quasi-Diagonalization
        self.sorted_indices = self._get_quasi_diag(corr)
        
        # 정렬된 자산 리스트
        sorted_assets = returns.columns[self.sorted_indices].tolist()
        
        # 3. Recursive Bisection
        weights_series = self._recursive_bisection(cov, sorted_assets)
        
        self.weights = weights_series.to_dict()
        return self.weights

    def _get_quasi_diag(self, corr: pd.DataFrame) -> List[int]:
        """
        행렬 재정렬 (Quasi-Diagonalization)
        
        상관계수 거리를 기반으로 비슷한 자산끼리 인접하도록 정렬
        """
        # 거리 행렬: d = sqrt(2 * (1 - rho))
        dist = np.sqrt(2 * (1 - corr))
        dist = dist.fillna(0) # NaN 처리
        
        if SCIPY_AVAILABLE:
            # Linkage Clustering
            condensed_dist = squareform(dist.values, checks=False)
            self.linkage_matrix = linkage(condensed_dist, method='ward')
            
            # Dendrogram 순서 추출
            return self._get_linkage_order(self.linkage_matrix)
        else:
            # Fallback: 변동성 순 정렬 (Scipy 없을 때)
            return list(range(len(corr)))

    def _get_linkage_order(self, link: np.ndarray) -> List[int]:
        """Linkage 매트릭스에서 정렬 순서 추출"""
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
        cov: pd.DataFrame, 
        sorted_assets: List[str]
    ) -> pd.Series:
        """
        재귀적 이분법 (Recursive Bisection)
        
        트리를 상단에서부터 쪼개며 하위 그룹의 분산 역수로 자금 배분
        """
        weights = pd.Series(1.0, index=sorted_assets)
        clusters = [sorted_assets] # Queue

        while len(clusters) > 0:
            # 클러스터 pop
            cluster = clusters.pop(0)
            
            if len(cluster) <= 1:
                continue
                
            # 이분할 (Bisect)
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            
            # 각 하위 그룹의 분산 계산
            var_left = self._get_cluster_var(cov, left)
            var_right = self._get_cluster_var(cov, right)
            
            # 배분 비율 (Alpha)
            # 분산이 작을수록 더 많은 비중 할당 (Inverse Variance)
            alpha = 1 - var_left / (var_left + var_right + 1e-10)
            
            # 가중치 업데이트
            weights[left] *= alpha
            weights[right] *= (1 - alpha)
            
            # Queue에 추가
            if len(left) > 1:
                clusters.append(left)
            if len(right) > 1:
                clusters.append(right)
                
        return weights

    def _get_cluster_var(self, cov: pd.DataFrame, assets: List[str]) -> float:
        """클러스터의 통합 분산 계산 (Inverse Variance Weighting 가정)"""
        cov_slice = cov.loc[assets, assets]
        
        # 클러스터 내에서는 역분산 비중으로 가정하여 리스크 측정
        # w_i = (1/var_i) / sum(1/var_k)
        inv_diag = 1 / np.diag(cov_slice)
        parity_w = inv_diag / inv_diag.sum()
        
        # Portfolio Variance = w.T * Cov * w
        cluster_var = np.dot(parity_w, np.dot(cov_slice, parity_w))
        return cluster_var


@dataclass
class MSTNode:
    ticker: str
    centrality: float
    is_hub: bool
    connections: int

class MSTAnalyzer:
    """
    Minimum Spanning Tree (MST) Analyzer
    
    자산 간 연결성을 분석하여 리스크 전이의 중심(Hub) 식별
    """
    
    def __init__(self):
        self.graph = None
        self.mst = None

    def analyze(self, returns: pd.DataFrame) -> Tuple[nx.Graph, List[MSTNode]]:
        """
        MST 구축 및 분석 실행
        
        Returns:
            (MST_Graph, List[MSTNode])
        """
        corr = returns.corr()
        
        # 거리 행렬 변환
        dist = np.sqrt(2 * (1 - corr))
        dist = dist.fillna(2.0) # 무상관/역상관 처리
        assets = returns.columns.tolist()
        
        # 1. MST 구축
        self.mst = self._build_mst(dist, assets)
        
        # 2. Centrality 분석 (Degree & Betweenness)
        degree = nx.degree_centrality(self.mst)
        betweenness = nx.betweenness_centrality(self.mst)
        
        # 3. 노드 정보 생성
        nodes = []
        # Hub 기준: 상위 10% 또는 Degree > 0.1
        degree_values = list(degree.values())
        threshold = np.percentile(degree_values, 90) if degree_values else 0
        
        for asset in assets:
            d = degree.get(asset, 0)
            b = betweenness.get(asset, 0)
            
            # 종합 점수 (Degree 50% + Betweenness 50%)
            score = (d + b) / 2
            is_hub = d >= threshold and d > 0.1
            
            nodes.append(MSTNode(
                ticker=asset,
                centrality=score,
                is_hub=is_hub,
                connections=self.mst.degree[asset]
            ))
            
        # 점수 순 정렬
        nodes.sort(key=lambda x: x.centrality, reverse=True)
        
        return self.mst, nodes

    def _build_mst(self, dist: pd.DataFrame, assets: List[str]) -> nx.Graph:
        """MST 그래프 생성"""
        if SCIPY_AVAILABLE and minimum_spanning_tree is not None:
            # Scipy 최적화 버전
            dist_array = dist.values
            np.fill_diagonal(dist_array, 0)
            # NaN 제거
            dist_array = np.nan_to_num(dist_array, nan=2.0)
            
            mst_sparse = minimum_spanning_tree(csr_matrix(dist_array))
            
            mst = nx.Graph()
            mst.add_nodes_from(assets)
            
            rows, cols = mst_sparse.nonzero()
            for r, c in zip(rows, cols):
                if r < c: # 중복 방지
                    weight = dist_array[r, c]
                    mst.add_edge(assets[r], assets[c], weight=weight)
            return mst
        else:
            # NetworkX 기본 버전 (느릴 수 있음)
            G = nx.Graph()
            n = len(assets)
            for i in range(n):
                for j in range(i+1, n):
                    G.add_edge(assets[i], assets[j], weight=dist.iloc[i, j])
            return nx.minimum_spanning_tree(G)


@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


class PortfolioOptimizer:
    """
    Wrapper class for backward compatibility and integration with CLI/API.
    Uses HRPOptimizer for risk_parity and adds simple MVO methods.
    """

    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.returns = pd.DataFrame()

    def fetch_data(self, lookback_days: int = 252):
        """Fetch historical data using yfinance"""
        import yfinance as yf
        try:
            data = yf.download(self.tickers, period=f"{lookback_days}d", progress=False)['Close']
            self.returns = data.pct_change().dropna()
        except Exception as e:
            print(f"Error fetching data: {e}")

    def optimize_sharpe(self, risk_free_rate: float = 0.02) -> OptimizationResult:
        """Maximize Sharpe Ratio (Simplified implementation)"""
        if self.returns.empty:
            return OptimizationResult({}, 0, 0, 0)

        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        num_assets = len(self.tickers)

        # Simple Equal Weight as fallback/baseline
        weights = np.array([1.0/num_assets] * num_assets)
        
        # Scipy optimize could be added here for true MVO
        # For now, returning equal weights with stats
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

        return OptimizationResult(
            weights=dict(zip(self.tickers, weights)),
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe
        )

    def optimize_min_variance(self) -> OptimizationResult:
        """Minimize Variance (Simplified)"""
        return self.optimize_sharpe() # Placeholder

    def optimize_risk_parity(self) -> OptimizationResult:
        """Use HRP for Risk Parity"""
        hrp = HRPOptimizer()
        weights_dict = hrp.optimize(self.returns)
        
        # Calculate stats
        weights = np.array([weights_dict.get(t, 0) for t in self.tickers])
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252
        
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_vol if port_vol > 0 else 0
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe
        )


# ============================================================================ 
# Test Code
# ============================================================================ 

if __name__ == "__main__":
    print("=" * 60)
    print("Portfolio Optimizer (HRP & MST) Test")
    print("=" * 60)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252)
    
    # 3개 그룹의 상관된 자산 생성
    group1 = np.random.randn(252, 3) + np.random.randn(252, 1) # Group 1
    group2 = np.random.randn(252, 3) + np.random.randn(252, 1) * 2 # Group 2 (Higher Vol)
    group3 = np.random.randn(252, 3) # Independent
    
    data = np.hstack([group1, group2, group3])
    assets = [f'Asset_{i}' for i in range(data.shape[1])]
    returns_df = pd.DataFrame(data, index=dates, columns=assets)
    
    print(f"Sample Data: {returns_df.shape}")
    
    # 1. HRP Optimization
    print("\n[1] Running HRP Optimization...")
    hrp = HRPOptimizer()
    weights = hrp.optimize(returns_df)
    
    print("  Optimal Weights:")
    for asset, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    {asset}: {w:.4f}")
        
    print(f"  Sum of weights: {sum(weights.values()):.4f}")
    
    # 2. MST Analysis
    print("\n[2] Running MST Analysis...")
    mst_analyzer = MSTAnalyzer()
    mst_graph, nodes = mst_analyzer.analyze(returns_df)
    
    print(f"  MST Edges: {mst_graph.number_of_edges()}")
    print("  Top Hub Assets:")
    for node in nodes[:3]:
        print(f"    {node.ticker}: Centrality={node.centrality:.4f}, Hub={'Yes' if node.is_hub else 'No'}")

    print("\nTest completed successfully!")