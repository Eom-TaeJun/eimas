#!/usr/bin/env python3
"""
Shock Propagation - Main Graph
============================================================

Shock propagation network construction and analysis

Economic Foundation:
    - Network topology: Asset correlation networks
    - Shock transmission: Path-based propagation
    - Systemic risk: Critical node identification

Class:
    - ShockPropagationGraph: Main shock propagation analyzer
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
import logging

from .enums import NodeLayer, CausalityStrength
from .schemas import EconomicEdge, ShockPath, NodeAnalysis, PropagationAnalysis
from .lead_lag import LeadLagAnalyzer
from .granger import GrangerCausalityAnalyzer

logger = logging.getLogger(__name__)


class ShockPropagationGraph:
    """
    충격 전파 그래프

    경제 시스템에서 충격이 어떻게 전파되는지를
    방향성 그래프(DAG)로 모델링
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_lag: int = 20,
        enforce_layer_order: bool = True
    ):
        self.significance_level = significance_level
        self.max_lag = max_lag
        self.enforce_layer_order = enforce_layer_order

        self.graph = nx.DiGraph()
        self.lead_lag_analyzer = LeadLagAnalyzer(max_lag)
        self.granger_analyzer = GrangerCausalityAnalyzer(max_lag, significance_level)

        # 결과 저장
        self.lead_lag_results: Dict[Tuple[str, str], LeadLagResult] = {}
        self.granger_results: Dict[Tuple[str, str], GrangerResult] = {}

    def build_from_data(
        self,
        data: pd.DataFrame,
        min_observations: int = 60
    ) -> nx.DiGraph:
        """
        시계열 데이터에서 충격 전파 그래프 구축

        Args:
            data: 시계열 DataFrame (columns = variables)
            min_observations: 최소 관측치 수

        Returns:
            NetworkX DiGraph
        """
        # 충분한 데이터가 있는 변수만 선택
        valid_columns = [
            col for col in data.columns
            if data[col].dropna().shape[0] >= min_observations
        ]

        if len(valid_columns) < 2:
            print("[SPG] Not enough valid columns")
            return self.graph

        print(f"[SPG] Building graph with {len(valid_columns)} variables")

        # 노드 추가
        for col in valid_columns:
            layer = get_node_layer(col)
            self.graph.add_node(
                col,
                layer=layer.value,
                layer_name=layer.name
            )

        # 엣지 추가 (Granger Causality 기반)
        n_edges = 0
        for i, source in enumerate(valid_columns):
            for j, target in enumerate(valid_columns):
                if i == j:
                    continue

                # Lead-Lag 분석
                ll_result = self.lead_lag_analyzer.analyze(
                    data[source], data[target],
                    source, target
                )
                self.lead_lag_results[(source, target)] = ll_result

                # Granger Causality 검정
                gc_result = self.granger_analyzer.test(
                    data[source], data[target],
                    source, target
                )
                self.granger_results[(source, target)] = gc_result

                # 유의미한 인과관계만 엣지로 추가
                if gc_result.is_significant:
                    # Layer 순서 강제 (선택적)
                    source_layer = get_node_layer(source)
                    target_layer = get_node_layer(target)

                    if self.enforce_layer_order and source_layer.value > target_layer.value:
                        # 역방향 (하위 레이어 → 상위 레이어)
                        # 피드백 루프로 표시
                        edge_type = "feedback"
                    else:
                        edge_type = "propagation"

                    self.graph.add_edge(
                        source, target,
                        lag=gc_result.optimal_lag,
                        strength=gc_result.strength.value,
                        p_value=gc_result.p_value,
                        correlation=ll_result.max_correlation,
                        edge_type=edge_type
                    )
                    n_edges += 1

        print(f"[SPG] Added {n_edges} significant edges")
        return self.graph

    def find_critical_path(
        self,
        source: str,
        max_depth: int = 10
    ) -> Optional[ShockPath]:
        """
        특정 소스에서 시작하는 최장 충격 전파 경로 탐색

        Critical Path = 가장 긴 전파 체인
        """
        if source not in self.graph:
            return None

        def dfs(node: str, path: List[str], total_lag: int, visited: Set[str]) -> Tuple[List[str], int]:
            """DFS로 최장 경로 탐색"""
            best_path = path
            best_lag = total_lag

            if len(path) >= max_depth:
                return best_path, best_lag

            for neighbor in self.graph.successors(node):
                if neighbor not in visited:
                    edge_data = self.graph.edges[node, neighbor]
                    new_lag = total_lag + edge_data.get('lag', 1)
                    new_path = path + [neighbor]

                    result_path, result_lag = dfs(
                        neighbor,
                        new_path,
                        new_lag,
                        visited | {neighbor}
                    )

                    if len(result_path) > len(best_path):
                        best_path = result_path
                        best_lag = result_lag

            return best_path, best_lag

        path, total_lag = dfs(source, [source], 0, {source})

        if len(path) <= 1:
            return None

        # 누적 충격 강도 계산
        cumulative_impact = 1.0
        for i in range(len(path) - 1):
            edge_data = self.graph.edges[path[i], path[i+1]]
            corr = abs(edge_data.get('correlation', 0.5))
            cumulative_impact *= corr

        # 병목 노드 탐색 (betweenness 가장 높은 노드)
        betweenness = nx.betweenness_centrality(self.graph)
        path_nodes = path[1:-1]  # 시작/끝 제외
        bottleneck = max(path_nodes, key=lambda n: betweenness.get(n, 0)) if path_nodes else None

        return ShockPath(
            source=source,
            path=path,
            total_lag=total_lag,
            cumulative_impact=cumulative_impact,
            bottleneck=bottleneck
        )

    def find_all_critical_paths(self, top_n: int = 5) -> List[ShockPath]:
        """모든 소스에서 시작하는 Critical Path 탐색"""
        paths = []

        for node in self.graph.nodes:
            layer = get_node_layer(node)
            # Policy/Liquidity 레이어만 소스로
            if layer in [NodeLayer.POLICY, NodeLayer.LIQUIDITY]:
                path = self.find_critical_path(node)
                if path and len(path.path) > 2:
                    paths.append(path)

        # 경로 길이 기준 정렬
        paths.sort(key=lambda p: len(p.path), reverse=True)
        return paths[:top_n]

    def analyze_nodes(self) -> List[NodeAnalysis]:
        """모든 노드 분석"""
        analyses = []

        betweenness = nx.betweenness_centrality(self.graph)

        for node in self.graph.nodes:
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)

            # 평균 선행 시간 계산
            out_edges = self.graph.out_edges(node, data=True)
            avg_lead_time = np.mean([
                e[2].get('lag', 0) for e in out_edges
            ]) if out_edges else 0

            # 역할 판정
            leading_score = out_deg - in_deg
            if leading_score > 2:
                role = "LEADING"
            elif leading_score < -2:
                role = "LAGGING"
            elif betweenness.get(node, 0) > 0.1:
                role = "BRIDGE"
            elif in_deg == 0 and out_deg == 0:
                role = "ISOLATED"
            else:
                role = "NEUTRAL"

            analyses.append(NodeAnalysis(
                node=node,
                layer=get_node_layer(node),
                in_degree=in_deg,
                out_degree=out_deg,
                leading_score=leading_score,
                betweenness=betweenness.get(node, 0),
                avg_lead_time=avg_lead_time,
                role=role
            ))

        return analyses

    def get_leading_indicators(self, top_n: int = 5) -> List[str]:
        """선행 지표 추출"""
        analyses = self.analyze_nodes()
        leading = [a for a in analyses if a.role == "LEADING"]
        leading.sort(key=lambda a: a.leading_score, reverse=True)
        return [a.node for a in leading[:top_n]]

    def get_lagging_indicators(self, top_n: int = 5) -> List[str]:
        """후행 지표 추출"""
        analyses = self.analyze_nodes()
        lagging = [a for a in analyses if a.role == "LAGGING"]
        lagging.sort(key=lambda a: a.leading_score)
        return [a.node for a in lagging[:top_n]]

    def get_bridge_nodes(self) -> List[str]:
        """브릿지 노드 (전파 중개자) 추출"""
        analyses = self.analyze_nodes()
        return [a.node for a in analyses if a.role == "BRIDGE"]

    def run_full_analysis(self, data: pd.DataFrame) -> PropagationAnalysis:
        """전체 분석 실행"""
        print("[SPG] Running full analysis...")

        # 그래프 구축
        self.build_from_data(data)

        # 노드 분석
        node_analyses = self.analyze_nodes()

        # 엣지 정보
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'lag': d.get('lag', 0),
                'strength': d.get('strength', 'none'),
                'p_value': d.get('p_value', 1.0),
                'correlation': d.get('correlation', 0),
                'edge_type': d.get('edge_type', 'unknown')
            })

        # Critical Paths
        critical_paths = self.find_all_critical_paths()

        return PropagationAnalysis(
            timestamp=datetime.now().isoformat(),
            nodes=node_analyses,
            edges=edges,
            critical_paths=critical_paths,
            leading_indicators=self.get_leading_indicators(),
            lagging_indicators=self.get_lagging_indicators(),
            bridge_nodes=self.get_bridge_nodes()
        )

    def to_adjacency_matrix(self) -> pd.DataFrame:
        """인접 행렬 반환 (lag 값)"""
        nodes = list(self.graph.nodes)
        n = len(nodes)

        matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=nodes,
            columns=nodes
        )

        for u, v, d in self.graph.edges(data=True):
            matrix.loc[u, v] = d.get('lag', 1)

        return matrix

    def visualize_text(self) -> str:
        """텍스트 기반 시각화"""
        lines = ["=" * 60, "Shock Propagation Graph", "=" * 60, ""]

        # 레이어별 노드
        layers = {layer: [] for layer in NodeLayer}
        for node in self.graph.nodes:
            layer = get_node_layer(node)
            layers[layer].append(node)

        for layer in NodeLayer:
            if layers[layer]:
                lines.append(f"[{layer.name}]")
                lines.append(f"  {', '.join(layers[layer])}")
                lines.append("")

        # 주요 엣지
        lines.append("Key Edges (p < 0.05):")
        for u, v, d in self.graph.edges(data=True):
            if d.get('p_value', 1) < 0.05:
                lag = d.get('lag', 0)
                corr = d.get('correlation', 0)
                lines.append(f"  {u} --[lag={lag}, r={corr:.2f}]--> {v}")

        # Critical Paths
        lines.append("")
        lines.append("Critical Paths:")
        for path in self.find_all_critical_paths(3):
            path_str = " → ".join(path.path)
            lines.append(f"  {path_str} (lag={path.total_lag}d)")

        return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_macro_data(n_days: int = 500) -> pd.DataFrame:
    """테스트용 거시경제 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # 기본 노이즈
    noise = np.random.randn(n_days) * 0.01

    # Fed Funds (기본 드라이버)
    fed_funds = np.cumsum(np.random.randn(n_days) * 0.001) + 4.5

    # DXY (Fed Funds에 2일 후행)
    dxy = pd.Series(fed_funds).shift(2).fillna(method='bfill').values * 20 + 100 + noise * 2

    # VIX (DXY에 3일 후행)
    vix = pd.Series(dxy).shift(3).fillna(method='bfill').values * 0.2 + 15 + np.abs(noise) * 10

    # SPY (VIX에 1일 후행, 역상관)
    spy = 500 - pd.Series(vix).shift(1).fillna(method='bfill').values * 2 + noise * 50

    # TLT (Fed Funds에 5일 후행)
    tlt = 100 - pd.Series(fed_funds).shift(5).fillna(method='bfill').values * 2 + noise * 20

    # GLD (VIX와 양상관, 2일 후행)
    gld = pd.Series(vix).shift(2).fillna(method='bfill').values * 5 + 1800 + noise * 50

    return pd.DataFrame({
        'FED_FUNDS': fed_funds,
        'DXY': dxy,
        'VIX': vix,
        'SPY': spy,
        'TLT': tlt,
        'GLD': gld
    }, index=dates)


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Shock Propagation Graph Test")
    print("=" * 60)

    # 샘플 데이터 생성
    print("\n1. Generating sample macro data (500 days)...")
    data = create_sample_macro_data(500)
    print(f"   Variables: {data.columns.tolist()}")

    # SPG 구축
    print("\n2. Building Shock Propagation Graph...")
    spg = ShockPropagationGraph(
        significance_level=0.05,
        max_lag=10,
        enforce_layer_order=True
    )

    analysis = spg.run_full_analysis(data)

    # 결과 출력
    print("\n3. Results:")
    print(f"   Nodes: {len(analysis.nodes)}")
    print(f"   Edges: {len(analysis.edges)}")
    print(f"   Leading Indicators: {analysis.leading_indicators}")
    print(f"   Lagging Indicators: {analysis.lagging_indicators}")
    print(f"   Bridge Nodes: {analysis.bridge_nodes}")

    print("\n4. Critical Paths:")
    for path in analysis.critical_paths:
        path_str = " → ".join(path.path)
        print(f"   {path_str}")
        print(f"      Total Lag: {path.total_lag} days")
        print(f"      Cumulative Impact: {path.cumulative_impact:.2%}")

    print("\n5. Node Roles:")
    for node in analysis.nodes:
        print(f"   {node.node}: {node.role} (in={node.in_degree}, out={node.out_degree})")

    print("\n" + spg.visualize_text())

    print("\n" + "=" * 60)
    print("Test completed successfully!")
