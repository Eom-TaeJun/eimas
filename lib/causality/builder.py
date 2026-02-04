#!/usr/bin/env python3
"""
Causality Analysis - Network Builder
============================================================

Builds causal networks from data

Class:
    - CausalNetworkBuilder: Constructs causal networks
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import logging

from .enums import CausalDirection
from .schemas import GrangerTestResult, CausalEdge, CausalityPath
# Alias for backward compatibility
CausalPath = CausalityPath

logger = logging.getLogger(__name__)


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

