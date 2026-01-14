"""
Causal Network Analysis Module

Granger Causality 기반 인과관계 네트워크 구축 및 경로 추출
ECON_AI_AGENT_SYSTEM.md Section 4 - Critical Path Discovery 구현

기존 critical_path.py와 독립적으로 네트워크 분석 기능 제공

Author: EIMAS Team
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import pandas as pd

# 통계 라이브러리
try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    from statsmodels.tsa.api import VAR
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

# 네트워크 라이브러리
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    warnings.warn("networkx not available. Install with: pip install networkx")


# ============================================================================
# Data Classes
# ============================================================================

class CausalDirection(Enum):
    """인과 방향"""
    X_TO_Y = "x_causes_y"       # X → Y
    Y_TO_X = "y_causes_x"       # Y → X
    BIDIRECTIONAL = "bidirectional"  # X ↔ Y
    NO_CAUSALITY = "no_causality"


@dataclass
class GrangerTestResult:
    """Granger Causality 검정 결과"""
    cause: str                  # 원인 변수
    effect: str                 # 결과 변수
    optimal_lag: int            # 최적 래그
    p_value: float              # p-value (F-test)
    f_statistic: float          # F-statistic
    is_significant: bool        # 유의미 여부
    direction: CausalDirection
    lead_time_days: int         # 선행 일수


@dataclass
class CausalEdge:
    """인과관계 엣지"""
    source: str                 # 원인 노드
    target: str                 # 결과 노드
    weight: float               # 영향력 크기 (β 계수 또는 F-stat)
    lag: int                    # 래그 (일)
    p_value: float
    confidence: float           # 신뢰도


@dataclass
class CausalPath:
    """인과 경로"""
    nodes: List[str]            # 경로 노드들 [A, B, C, D]
    edges: List[CausalEdge]     # 경로 엣지들
    total_lag: int              # 전체 래그
    path_strength: float        # 경로 강도 (엣지 가중치의 곱)
    description: str


@dataclass
class NetworkAnalysisResult:
    """네트워크 분석 결과"""
    target_variable: str
    all_paths: List[CausalPath]
    critical_path: Optional[CausalPath]  # 가장 중요한 경로
    key_drivers: List[str]          # 핵심 선행 지표
    network_stats: Dict[str, Any]   # 네트워크 통계
    granger_results: List[GrangerTestResult]
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Granger Causality Analyzer
# ============================================================================

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

class CausalNetworkBuilder:
    """
    인과관계 네트워크 구축

    Granger Causality 결과를 바탕으로 방향성 그래프 생성
    """

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx is required. Install with: pip install networkx")

        self.graph = nx.DiGraph()

    def build_network(
        self,
        granger_results: List[GrangerTestResult],
        weight_by: str = "f_statistic"  # "f_statistic" or "inverse_pvalue"
    ) -> 'nx.DiGraph':
        """
        Granger 결과로부터 네트워크 구축

        Parameters:
        -----------
        granger_results : List[GrangerTestResult]
            Granger Causality 검정 결과
        weight_by : str
            엣지 가중치 기준

        Returns:
        --------
        nx.DiGraph
            방향성 그래프
        """
        self.graph = nx.DiGraph()

        for result in granger_results:
            if not result.is_significant:
                continue

            # 가중치 계산
            if weight_by == "f_statistic":
                weight = result.f_statistic
            else:
                weight = 1.0 / max(result.p_value, 1e-10)

            # 엣지 추가
            self.graph.add_edge(
                result.cause,
                result.effect,
                weight=weight,
                lag=result.optimal_lag,
                p_value=result.p_value,
                f_stat=result.f_statistic
            )

        return self.graph

    def get_network_stats(self) -> Dict[str, Any]:
        """네트워크 통계 계산"""
        if not self.graph.nodes():
            return {}

        stats = {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_dag": nx.is_directed_acyclic_graph(self.graph),
        }

        # 중심성 지표
        try:
            stats["in_degree_centrality"] = nx.in_degree_centrality(self.graph)
            stats["out_degree_centrality"] = nx.out_degree_centrality(self.graph)
            stats["pagerank"] = nx.pagerank(self.graph)
        except Exception:
            pass

        return stats

    def get_key_drivers(
        self,
        target: str,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        타겟 변수에 영향을 주는 핵심 드라이버 식별

        Parameters:
        -----------
        target : str
            타겟 변수
        top_n : int
            반환할 드라이버 수

        Returns:
        --------
        List[Tuple[str, float]]
            (드라이버명, 영향력) 튜플 리스트
        """
        if target not in self.graph:
            return []

        # 직접 연결된 선행 변수
        predecessors = list(self.graph.predecessors(target))

        drivers = []
        for pred in predecessors:
            edge_data = self.graph.get_edge_data(pred, target)
            weight = edge_data.get('weight', 1.0) if edge_data else 1.0
            drivers.append((pred, weight))

        # 가중치 순 정렬
        drivers.sort(key=lambda x: x[1], reverse=True)

        return drivers[:top_n]

    def find_all_paths(
        self,
        source: str,
        target: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """
        소스에서 타겟까지의 모든 경로 찾기

        Parameters:
        -----------
        source : str
            시작 노드
        target : str
            도착 노드
        max_length : int
            최대 경로 길이

        Returns:
        --------
        List[List[str]]
            경로 목록
        """
        if source not in self.graph or target not in self.graph:
            return []

        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source,
                target,
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXError:
            return []

    def get_critical_paths_to_target(
        self,
        target: str,
        max_paths: int = 10
    ) -> List[CausalPath]:
        """
        타겟까지의 중요 경로들 추출

        Parameters:
        -----------
        target : str
            타겟 변수
        max_paths : int
            반환할 최대 경로 수

        Returns:
        --------
        List[CausalPath]
            중요도 순 경로 리스트
        """
        if target not in self.graph:
            return []

        all_paths = []

        # 모든 노드에서 타겟까지의 경로
        for source in self.graph.nodes():
            if source == target:
                continue

            paths = self.find_all_paths(source, target)

            for path in paths:
                if len(path) < 2:
                    continue

                # 엣지 정보 수집
                edges = []
                total_lag = 0
                path_strength = 1.0

                for i in range(len(path) - 1):
                    edge_data = self.graph.get_edge_data(path[i], path[i+1])
                    if edge_data:
                        edge = CausalEdge(
                            source=path[i],
                            target=path[i+1],
                            weight=edge_data.get('weight', 1.0),
                            lag=edge_data.get('lag', 1),
                            p_value=edge_data.get('p_value', 0.05),
                            confidence=1 - edge_data.get('p_value', 0.05)
                        )
                        edges.append(edge)
                        total_lag += edge.lag
                        path_strength *= edge.weight

                causal_path = CausalPath(
                    nodes=path,
                    edges=edges,
                    total_lag=total_lag,
                    path_strength=path_strength,
                    description=f"{' → '.join(path)}"
                )
                all_paths.append(causal_path)

        # 강도 순 정렬
        all_paths.sort(key=lambda x: x.path_strength, reverse=True)

        return all_paths[:max_paths]

    def get_visualization_data(self) -> Dict[str, Any]:
        """시각화용 네트워크 데이터"""
        nodes = []
        for node in self.graph.nodes():
            nodes.append({
                "id": node,
                "label": node,
                "in_degree": self.graph.in_degree(node),
                "out_degree": self.graph.out_degree(node)
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "source": u,
                "target": v,
                "weight": data.get('weight', 1),
                "lag": data.get('lag', 1)
            })

        return {
            "nodes": nodes,
            "edges": edges
        }


# ============================================================================
# Integrated Analyzer
# ============================================================================

class CausalNetworkAnalyzer:
    """
    인과관계 네트워크 분석 통합 클래스

    Granger Causality 분석 → 네트워크 구축 → 핵심 경로 추출
    """

    def __init__(
        self,
        max_lag: int = 10,
        significance_level: float = 0.05
    ):
        self.granger_analyzer = GrangerCausalityAnalyzer(
            max_lag=max_lag,
            significance_level=significance_level
        )
        self.network_builder = CausalNetworkBuilder()

    def analyze(
        self,
        data: pd.DataFrame,
        target_variable: str,
        variables: Optional[List[str]] = None,
        make_stationary: bool = True,
        max_paths: int = 10
    ) -> NetworkAnalysisResult:
        """
        전체 분석 실행

        Parameters:
        -----------
        data : DataFrame
            시계열 데이터
        target_variable : str
            분석 타겟 변수
        variables : List[str]
            분석할 변수 목록
        make_stationary : bool
            정상성 변환 여부
        max_paths : int
            반환할 최대 경로 수

        Returns:
        --------
        NetworkAnalysisResult
            분석 결과
        """
        # 1. Granger Causality 분석
        granger_results = self.granger_analyzer.test_all_pairs(
            data,
            variables=variables,
            make_stationary=make_stationary
        )

        # 2. 네트워크 구축
        self.network_builder.build_network(granger_results)

        # 3. 네트워크 통계
        network_stats = self.network_builder.get_network_stats()

        # 4. 핵심 드라이버 식별
        key_drivers_raw = self.network_builder.get_key_drivers(target_variable)
        key_drivers = [d[0] for d in key_drivers_raw]

        # 5. Critical Paths 추출
        all_paths = self.network_builder.get_critical_paths_to_target(
            target_variable,
            max_paths=max_paths
        )

        # 가장 중요한 경로
        critical_path = all_paths[0] if all_paths else None

        return NetworkAnalysisResult(
            target_variable=target_variable,
            all_paths=all_paths,
            critical_path=critical_path,
            key_drivers=key_drivers,
            network_stats=network_stats,
            granger_results=granger_results
        )

    def get_network(self) -> 'nx.DiGraph':
        """현재 네트워크 그래프 반환"""
        return self.network_builder.graph

    def get_visualization_data(self) -> Dict[str, Any]:
        """시각화용 데이터 반환"""
        return self.network_builder.get_visualization_data()


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Causal Network Analysis Module Demo")
    print("=" * 60)

    # 샘플 데이터 생성
    np.random.seed(42)
    n = 500

    # X1 → X2 → Y 관계 시뮬레이션
    X1 = np.random.randn(n).cumsum()
    X2 = np.zeros(n)
    Y = np.zeros(n)

    for i in range(3, n):
        X2[i] = 0.5 * X1[i-2] + 0.3 * X2[i-1] + np.random.randn() * 0.5
        Y[i] = 0.4 * X2[i-1] + 0.2 * Y[i-1] + np.random.randn() * 0.5

    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'Y': Y,
        'Noise': np.random.randn(n).cumsum()
    })

    print(f"\n[Sample Data]")
    print(f"  Shape: {data.shape}")
    print(f"  Simulated relationship: X1 → X2 → Y")

    # 분석 실행
    analyzer = CausalNetworkAnalyzer(max_lag=5, significance_level=0.05)
    result = analyzer.analyze(
        data=data,
        target_variable='Y',
        make_stationary=True
    )

    print(f"\n[Granger Causality Results]")
    for gr in result.granger_results[:5]:
        print(f"  {gr.cause} → {gr.effect}: lag={gr.optimal_lag}, p={gr.p_value:.4f}")

    print(f"\n[Key Drivers for Y]")
    for driver in result.key_drivers:
        print(f"  - {driver}")

    print(f"\n[Critical Paths to Y]")
    for path in result.all_paths[:3]:
        print(f"  {path.description}")
        print(f"    Total lag: {path.total_lag}, Strength: {path.path_strength:.4f}")

    if result.critical_path:
        print(f"\n[Most Critical Path]")
        print(f"  {result.critical_path.description}")

    print(f"\n[Network Stats]")
    for key, value in result.network_stats.items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
