#!/usr/bin/env python3
"""
Causality Graph Engine
=======================
그래프 이론 기반 인과관계 네트워크 분석

소스 이론: "경제학은 인과관계(Causality)... 팔란티어의 온톨로지처럼
노드와 엣지로 설명해야 한다"

핵심 기능:
1. 다중 병목점(Bottleneck) 탐지 및 연결
2. Granger Causality 기반 시계열 인과관계
3. 충격 전파(Shock Propagation) 시뮬레이션
4. LLM 기반 Narrative 생성

Usage:
    from lib.causality_graph import CausalityGraphEngine

    engine = CausalityGraphEngine()
    engine.build_from_market_data(market_data)
    engine.add_supply_chain_layer(supply_chain_graph)

    narratives = await engine.generate_narratives()
"""

import asyncio
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger('eimas.causality_graph')


# =============================================================================
# Enums and Data Classes
# =============================================================================

class EdgeType(Enum):
    """엣지 유형"""
    SUPPLY_DEPENDENCY = "supply_dependency"      # 공급망 의존성
    PRICE_CORRELATION = "price_correlation"      # 가격 상관관계
    GRANGER_CAUSALITY = "granger_causality"      # Granger 인과관계
    SECTOR_LINKAGE = "sector_linkage"            # 섹터 연결
    EXTERNAL_SHOCK = "external_shock"            # 외부 충격


class NodeType(Enum):
    """노드 유형"""
    COMPANY = "company"
    SECTOR = "sector"
    COMMODITY = "commodity"
    MACRO_INDICATOR = "macro_indicator"
    EXTERNAL_EVENT = "external_event"


@dataclass
class CausalNode:
    """인과관계 그래프 노드"""
    id: str
    name: str
    node_type: NodeType
    layer: str = ""                              # 공급망 레이어

    # 메트릭
    centrality_score: float = 0.0                # 중심성 점수
    pagerank: float = 0.0                        # PageRank
    is_bottleneck: bool = False                  # 병목점 여부
    criticality: float = 0.0                     # 중요도 (0-1)

    # 시장 데이터
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0
    volume_ratio: float = 1.0

    # 메타데이터
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.node_type.value,
            'layer': self.layer,
            'centrality': self.centrality_score,
            'pagerank': self.pagerank,
            'is_bottleneck': self.is_bottleneck,
            'criticality': self.criticality,
            'price_change_1d': self.price_change_1d,
            'volume_ratio': self.volume_ratio
        }


@dataclass
class CausalEdge:
    """인과관계 그래프 엣지"""
    source: str
    target: str
    edge_type: EdgeType
    weight: float = 1.0                          # 연결 강도
    lag: int = 0                                 # 시차 (일)
    direction: str = "forward"                   # forward, backward, bidirectional
    confidence: float = 0.8                      # 신뢰도

    # Granger Causality 결과
    granger_pvalue: float = 1.0                  # p-value (낮을수록 유의)
    granger_fstat: float = 0.0                   # F-statistic

    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'type': self.edge_type.value,
            'weight': self.weight,
            'lag': self.lag,
            'direction': self.direction,
            'confidence': self.confidence,
            'granger_pvalue': self.granger_pvalue
        }


@dataclass
class CausalityPath:
    """인과관계 경로"""
    nodes: List[str]
    edges: List[CausalEdge]
    total_weight: float = 0.0
    total_lag: int = 0
    path_type: str = ""                          # supply_chain, correlation, mixed
    narrative: str = ""                          # LLM 생성 내러티브

    def to_string(self) -> str:
        return " → ".join(self.nodes)


@dataclass
class CausalityInsight:
    """인과관계 인사이트"""
    path: CausalityPath
    insight_type: str                            # bottleneck_risk, shock_propagation, hub_influence
    severity: str                                # low, medium, high, critical
    confidence: float
    narrative: str
    affected_assets: List[str]
    recommended_action: str = ""

    def to_dict(self) -> Dict:
        return {
            'path': self.path.to_string(),
            'type': self.insight_type,
            'severity': self.severity,
            'confidence': self.confidence,
            'narrative': self.narrative,
            'affected_assets': self.affected_assets,
            'action': self.recommended_action
        }


# =============================================================================
# Causality Graph Engine
# =============================================================================

class CausalityGraphEngine:
    """
    인과관계 그래프 엔진

    팔란티어 온톨로지 스타일의 노드-엣지 기반 인과관계 분석
    """

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: LLM 사용 여부 (Claude API)
        """
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.use_llm = use_llm
        self._api_config = None

    def _get_api_config(self):
        """API 설정 lazy loading"""
        if self._api_config is None:
            try:
                from core.config import APIConfig
                self._api_config = APIConfig()
            except:
                self._api_config = None
        return self._api_config

    # =========================================================================
    # Graph Construction
    # =========================================================================

    def add_node(self, node: CausalNode):
        """노드 추가"""
        self.nodes[node.id] = node
        self.graph.add_node(
            node.id,
            **node.to_dict()
        )

    def add_edge(self, edge: CausalEdge):
        """엣지 추가"""
        self.edges.append(edge)
        self.graph.add_edge(
            edge.source,
            edge.target,
            **edge.to_dict()
        )

    def build_from_supply_chain(
        self,
        supply_chain_layers: Dict[str, List[str]],
        stock_info: Dict[str, Dict] = None
    ):
        """
        공급망 데이터로 그래프 구축

        Args:
            supply_chain_layers: {'equipment': ['ASML', ...], 'manufacturer': ['TSM', ...]}
            stock_info: 종목별 추가 정보 {'ASML': {'name': 'ASML Holding', ...}}
        """
        stock_info = stock_info or {}

        # 레이어 순서 정의
        layer_order = ['raw_material', 'component', 'equipment', 'manufacturer',
                       'integrator', 'distribution', 'end_user']

        # 노드 추가
        for layer, tickers in supply_chain_layers.items():
            for ticker in tickers:
                info = stock_info.get(ticker, {})
                node = CausalNode(
                    id=ticker,
                    name=info.get('name', ticker),
                    node_type=NodeType.COMPANY,
                    layer=layer,
                    metadata={'sector': info.get('sector', '')}
                )
                self.add_node(node)

        # 레이어 간 엣지 추가 (순차적 의존성)
        for i, current_layer in enumerate(layer_order):
            if current_layer not in supply_chain_layers:
                continue

            # 다음 레이어 찾기
            for next_layer in layer_order[i+1:]:
                if next_layer in supply_chain_layers:
                    # 현재 레이어 → 다음 레이어 연결
                    for source in supply_chain_layers[current_layer]:
                        for target in supply_chain_layers[next_layer]:
                            edge = CausalEdge(
                                source=source,
                                target=target,
                                edge_type=EdgeType.SUPPLY_DEPENDENCY,
                                weight=0.7,
                                direction="forward"
                            )
                            self.add_edge(edge)
                    break

    def build_from_market_data(
        self,
        market_data: Dict[str, pd.DataFrame],
        correlation_threshold: float = 0.6,
        granger_pvalue_threshold: float = 0.05
    ):
        """
        시장 데이터로 상관관계/인과관계 엣지 추가

        Args:
            market_data: {ticker: DataFrame with 'Close' column}
            correlation_threshold: 상관관계 임계값
            granger_pvalue_threshold: Granger 인과관계 p-value 임계값
        """
        # 수익률 계산
        returns = {}
        for ticker, df in market_data.items():
            if 'Close' in df.columns and len(df) > 20:
                ret = df['Close'].pct_change().dropna()
                if len(ret) > 20:
                    returns[ticker] = ret

                    # 노드가 없으면 추가
                    if ticker not in self.nodes:
                        node = CausalNode(
                            id=ticker,
                            name=ticker,
                            node_type=NodeType.COMPANY,
                            price_change_1d=float(ret.iloc[-1] * 100) if len(ret) > 0 else 0,
                            price_change_5d=float(ret.iloc[-5:].sum() * 100) if len(ret) >= 5 else 0
                        )
                        self.add_node(node)

        if len(returns) < 2:
            return

        # 수익률 DataFrame
        returns_df = pd.DataFrame(returns).dropna()

        if len(returns_df) < 30:
            return

        # 상관관계 계산
        corr_matrix = returns_df.corr()

        # 상관관계 기반 엣지 추가
        tickers = list(returns.keys())
        for i, ticker1 in enumerate(tickers):
            for ticker2 in tickers[i+1:]:
                corr = corr_matrix.loc[ticker1, ticker2]
                if abs(corr) >= correlation_threshold:
                    edge = CausalEdge(
                        source=ticker1,
                        target=ticker2,
                        edge_type=EdgeType.PRICE_CORRELATION,
                        weight=abs(corr),
                        direction="bidirectional",
                        confidence=abs(corr)
                    )
                    self.add_edge(edge)

        # Granger Causality 테스트 (상관관계 높은 쌍만)
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            for edge in [e for e in self.edges if e.edge_type == EdgeType.PRICE_CORRELATION]:
                if edge.source in returns_df.columns and edge.target in returns_df.columns:
                    try:
                        data = returns_df[[edge.target, edge.source]].dropna()
                        if len(data) > 30:
                            result = grangercausalitytests(data, maxlag=5, verbose=False)

                            # 최적 lag의 p-value
                            min_pvalue = min(result[lag][0]['ssr_ftest'][1]
                                           for lag in range(1, 6))

                            if min_pvalue < granger_pvalue_threshold:
                                # Granger 인과관계 엣지 추가
                                granger_edge = CausalEdge(
                                    source=edge.source,
                                    target=edge.target,
                                    edge_type=EdgeType.GRANGER_CAUSALITY,
                                    weight=1 - min_pvalue,
                                    granger_pvalue=min_pvalue,
                                    direction="forward",
                                    confidence=1 - min_pvalue
                                )
                                self.add_edge(granger_edge)
                    except:
                        pass
        except ImportError:
            logger.warning("statsmodels not installed, skipping Granger causality")

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def compute_centrality_metrics(self):
        """중심성 지표 계산"""
        if len(self.graph.nodes()) == 0:
            return

        # Betweenness Centrality
        betweenness = nx.betweenness_centrality(self.graph)

        # PageRank
        try:
            pagerank = nx.pagerank(self.graph, max_iter=100)
        except:
            pagerank = {n: 1/len(self.graph.nodes()) for n in self.graph.nodes()}

        # Degree Centrality
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())

        # 노드 업데이트
        for node_id, node in self.nodes.items():
            if node_id in betweenness:
                node.centrality_score = betweenness[node_id]
            if node_id in pagerank:
                node.pagerank = pagerank[node_id]

            # 종합 중요도 계산
            node.criticality = (
                0.4 * betweenness.get(node_id, 0) +
                0.3 * pagerank.get(node_id, 0) +
                0.2 * (in_degree.get(node_id, 0) / max(1, max(in_degree.values()))) +
                0.1 * (out_degree.get(node_id, 0) / max(1, max(out_degree.values())))
            )

    def identify_bottlenecks(self, top_n: int = 5) -> List[CausalNode]:
        """
        병목점 식별

        병목점 기준:
        1. 높은 Betweenness Centrality (많은 경로가 통과)
        2. 낮은 대체 가능성 (unique한 연결)
        3. 공급망 상류 위치 (equipment, manufacturer)
        """
        self.compute_centrality_metrics()

        # 상류 레이어 가중치
        upstream_layers = {'raw_material', 'component', 'equipment', 'manufacturer'}

        candidates = []
        for node_id, node in self.nodes.items():
            # 상류 레이어 보너스
            layer_bonus = 0.2 if node.layer in upstream_layers else 0

            # 병목 점수 계산
            bottleneck_score = (
                node.criticality +
                layer_bonus +
                (0.1 if self.graph.in_degree(node_id) == 1 else 0)  # 단일 공급자
            )

            candidates.append((node, bottleneck_score))

        # 상위 N개 선택
        candidates.sort(key=lambda x: x[1], reverse=True)
        bottlenecks = []

        for node, score in candidates[:top_n]:
            node.is_bottleneck = True
            bottlenecks.append(node)

        return bottlenecks

    def find_critical_paths(
        self,
        source: str = None,
        max_paths: int = 5
    ) -> List[CausalityPath]:
        """
        주요 인과관계 경로 탐색

        Args:
            source: 시작 노드 (None이면 모든 병목점에서 시작)
            max_paths: 최대 경로 수
        """
        paths = []

        # 시작 노드 결정
        if source:
            start_nodes = [source] if source in self.nodes else []
        else:
            # 병목점에서 시작
            start_nodes = [n.id for n in self.nodes.values() if n.is_bottleneck]

        if not start_nodes:
            start_nodes = list(self.nodes.keys())[:3]

        # 각 시작점에서 경로 탐색
        for start in start_nodes:
            # 모든 도달 가능한 노드로의 경로
            for target in self.graph.nodes():
                if target != start and nx.has_path(self.graph, start, target):
                    try:
                        # 최단 경로
                        shortest = nx.shortest_path(self.graph, start, target)

                        if len(shortest) >= 2:
                            # 경로의 엣지 수집
                            path_edges = []
                            total_weight = 0
                            total_lag = 0

                            for i in range(len(shortest) - 1):
                                edge_data = self.graph.get_edge_data(shortest[i], shortest[i+1])
                                if edge_data:
                                    path_edges.append(CausalEdge(
                                        source=shortest[i],
                                        target=shortest[i+1],
                                        edge_type=EdgeType(edge_data.get('type', 'supply_dependency')),
                                        weight=edge_data.get('weight', 0.5)
                                    ))
                                    total_weight += edge_data.get('weight', 0.5)
                                    total_lag += edge_data.get('lag', 0)

                            paths.append(CausalityPath(
                                nodes=shortest,
                                edges=path_edges,
                                total_weight=total_weight,
                                total_lag=total_lag
                            ))
                    except:
                        pass

        # 가중치 기준 정렬
        paths.sort(key=lambda p: p.total_weight, reverse=True)

        return paths[:max_paths]

    def simulate_shock_propagation(
        self,
        shock_node: str,
        shock_magnitude: float = -0.10,  # -10% 하락
        decay_factor: float = 0.7
    ) -> Dict[str, float]:
        """
        충격 전파 시뮬레이션

        Args:
            shock_node: 충격 발생 노드
            shock_magnitude: 충격 크기 (-0.10 = 10% 하락)
            decay_factor: 전파 감쇠율

        Returns:
            {node_id: expected_impact} 형태의 영향 추정
        """
        if shock_node not in self.graph.nodes():
            return {}

        impacts = {shock_node: shock_magnitude}
        visited = {shock_node}
        queue = [(shock_node, shock_magnitude, 0)]  # (node, impact, depth)

        while queue:
            current, current_impact, depth = queue.pop(0)

            if depth > 5:  # 최대 5단계 전파
                continue

            # 후속 노드로 전파
            for successor in self.graph.successors(current):
                if successor not in visited:
                    visited.add(successor)

                    edge_data = self.graph.get_edge_data(current, successor)
                    edge_weight = edge_data.get('weight', 0.5) if edge_data else 0.5

                    # 감쇠된 충격
                    propagated_impact = current_impact * decay_factor * edge_weight

                    if abs(propagated_impact) > 0.01:  # 1% 이상만
                        impacts[successor] = propagated_impact
                        queue.append((successor, propagated_impact, depth + 1))

        return impacts

    # =========================================================================
    # Narrative Generation
    # =========================================================================

    async def generate_insights(
        self,
        max_insights: int = 5
    ) -> List[CausalityInsight]:
        """
        인과관계 인사이트 생성

        Rule-based + LLM 혼합 방식
        """
        insights = []

        # 1. 병목점 리스크 인사이트
        bottlenecks = self.identify_bottlenecks(top_n=3)
        for bn in bottlenecks:
            # 충격 전파 시뮬레이션
            impacts = self.simulate_shock_propagation(bn.id)
            affected = [n for n, impact in impacts.items() if abs(impact) > 0.03]

            severity = "high" if len(affected) > 5 else "medium" if len(affected) > 2 else "low"

            path = CausalityPath(
                nodes=[bn.id] + affected[:3],
                edges=[],
                path_type="shock_propagation"
            )

            narrative = self._generate_bottleneck_narrative(bn, impacts)

            insights.append(CausalityInsight(
                path=path,
                insight_type="bottleneck_risk",
                severity=severity,
                confidence=bn.criticality,
                narrative=narrative,
                affected_assets=affected,
                recommended_action=f"Monitor {bn.id} for supply chain disruption signals"
            ))

        # 2. 허브 노드 영향력 인사이트
        hub_nodes = sorted(self.nodes.values(), key=lambda n: n.pagerank, reverse=True)[:2]
        for hub in hub_nodes:
            if hub.id in [bn.id for bn in bottlenecks]:
                continue  # 이미 병목점으로 처리됨

            upstream = list(self.graph.predecessors(hub.id))[:3]
            downstream = list(self.graph.successors(hub.id))[:3]

            path = CausalityPath(
                nodes=upstream + [hub.id] + downstream,
                edges=[],
                path_type="hub_influence"
            )

            narrative = self._generate_hub_narrative(hub, upstream, downstream)

            insights.append(CausalityInsight(
                path=path,
                insight_type="hub_influence",
                severity="medium",
                confidence=hub.pagerank,
                narrative=narrative,
                affected_assets=upstream + downstream,
                recommended_action=f"Track {hub.id} as key network node"
            ))

        # 3. 주요 경로 인사이트
        critical_paths = self.find_critical_paths(max_paths=2)
        for cp in critical_paths:
            if len(cp.nodes) >= 3:
                narrative = self._generate_path_narrative(cp)

                insights.append(CausalityInsight(
                    path=cp,
                    insight_type="critical_path",
                    severity="medium",
                    confidence=cp.total_weight / len(cp.nodes) if cp.nodes else 0.5,
                    narrative=narrative,
                    affected_assets=cp.nodes
                ))

        # LLM으로 내러티브 향상 (옵션)
        if self.use_llm:
            insights = await self._enhance_with_llm(insights)

        return insights[:max_insights]

    def _generate_bottleneck_narrative(
        self,
        bottleneck: CausalNode,
        impacts: Dict[str, float]
    ) -> str:
        """병목점 내러티브 생성 (Rule-based)"""
        affected_count = len([i for i in impacts.values() if abs(i) > 0.03])
        max_impact = max(abs(i) for i in impacts.values()) if impacts else 0

        layer_desc = {
            'equipment': '반도체 장비',
            'manufacturer': '파운드리/제조',
            'integrator': '칩 설계',
            'end_user': '최종 사용자'
        }

        narrative = (
            f"**Path:** External Shock → {bottleneck.id} (Bottleneck) → "
            f"{affected_count} downstream nodes affected.\n"
            f"**Insight:** {bottleneck.name}는 {layer_desc.get(bottleneck.layer, bottleneck.layer)} "
            f"레이어의 핵심 병목점으로, 이 노드에 충격 발생 시 최대 {max_impact:.1%}의 영향이 "
            f"공급망 전체에 전파됨. Criticality Score: {bottleneck.criticality:.2f}."
        )

        return narrative

    def _generate_hub_narrative(
        self,
        hub: CausalNode,
        upstream: List[str],
        downstream: List[str]
    ) -> str:
        """허브 노드 내러티브 생성"""
        narrative = (
            f"**Path:** {', '.join(upstream[:2])} → {hub.id} (Hub) → {', '.join(downstream[:2])}.\n"
            f"**Insight:** {hub.name}는 PageRank {hub.pagerank:.3f}로 네트워크 중심에 위치. "
            f"{len(upstream)}개 상류 노드와 {len(downstream)}개 하류 노드를 연결하는 허브로, "
            f"이 종목의 변동은 양방향으로 전파됨 (Bidirectional Causality)."
        )

        return narrative

    def _generate_path_narrative(self, path: CausalityPath) -> str:
        """경로 내러티브 생성"""
        path_str = " → ".join(path.nodes)

        narrative = (
            f"**Path:** {path_str}\n"
            f"**Insight:** 총 {len(path.nodes)}개 노드를 연결하는 인과관계 경로. "
            f"경로 가중치: {path.total_weight:.2f}. "
            f"첫 번째 노드의 변동이 순차적으로 마지막 노드까지 전파됨."
        )

        return narrative

    async def _enhance_with_llm(
        self,
        insights: List[CausalityInsight]
    ) -> List[CausalityInsight]:
        """LLM으로 내러티브 향상"""
        api_config = self._get_api_config()
        if not api_config:
            return insights

        try:
            client = api_config.get_client('claude')

            # 그래프 요약 생성
            graph_summary = {
                'nodes': len(self.nodes),
                'edges': len(self.edges),
                'bottlenecks': [n.id for n in self.nodes.values() if n.is_bottleneck],
                'top_pagerank': sorted(
                    [(n.id, n.pagerank) for n in self.nodes.values()],
                    key=lambda x: x[1], reverse=True
                )[:5]
            }

            prompt = f"""You are an expert economist analyzing supply chain causality networks.

Graph Summary:
- Nodes: {graph_summary['nodes']}
- Edges: {graph_summary['edges']}
- Key Bottlenecks: {', '.join(graph_summary['bottlenecks'])}
- Top PageRank: {graph_summary['top_pagerank']}

Current Insights (enhance these with more economic context):
{[i.narrative for i in insights[:3]]}

For each insight, provide:
1. A refined narrative with economic theory references
2. Specific risk scenarios
3. Actionable recommendations

Format as JSON array with 'narrative', 'risk_scenario', 'recommendation' keys."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )

            import json
            response_text = response.content[0].text.strip()

            # JSON 파싱 시도
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            enhanced = json.loads(response_text)

            # 인사이트 업데이트
            for i, insight in enumerate(insights):
                if i < len(enhanced):
                    insight.narrative = enhanced[i].get('narrative', insight.narrative)
                    insight.recommended_action = enhanced[i].get('recommendation', insight.recommended_action)

        except Exception as e:
            logger.warning(f"LLM enhancement failed: {e}")

        return insights

    # =========================================================================
    # Narrative Report Generation (for integrated_*.md)
    # =========================================================================

    def generate_report_narrative(
        self,
        external_shock: str = "Market Shock",
        include_shock_sim: bool = True
    ) -> str:
        """
        리포트용 자연어 Narrative 생성

        Source 요구사항:
        1. Supply Chain Narrative: Bottleneck → Hub 관계 설명
        2. Economic Interpretation: 방향성(→) 명시
        3. Fallback: 데이터 부족시 명시적 메시지

        Returns:
            자연어 인과관계 설명 (Markdown 형식)
        """
        lines = []

        # 데이터 충분성 검사
        if len(self.nodes) < 3 or len(self.edges) < 2:
            return "Not enough correlation data to build causality chain yet."

        # 병목점 및 허브 식별
        bottlenecks = self.identify_bottlenecks(top_n=5)
        hub_nodes = sorted(
            [n for n in self.nodes.values() if not n.is_bottleneck],
            key=lambda n: n.pagerank,
            reverse=True
        )[:3]

        if not bottlenecks:
            return "Not enough correlation data to build causality chain yet."

        # ===== Section 1: Supply Chain Causality Flow =====
        lines.append("### Supply Chain Causality Flow")
        lines.append("")
        lines.append(f"**External Shock:** {external_shock}")
        lines.append("")

        # 병목점 → 허브 관계 설명
        lines.append("**Propagation Path (전파 경로):**")
        lines.append("```")

        # 주요 전파 경로 생성
        critical_paths = self.find_critical_paths(max_paths=3)
        if critical_paths:
            for i, path in enumerate(critical_paths, 1):
                path_str = " → ".join(path.nodes)
                lines.append(f"[Path {i}] {external_shock} → {path_str}")

                # 경로 해석
                if len(path.nodes) >= 2:
                    source_node = self.nodes.get(path.nodes[0])
                    target_node = self.nodes.get(path.nodes[-1])

                    if source_node and target_node:
                        lines.append(f"         ({source_node.layer or 'upstream'} → {target_node.layer or 'downstream'})")
        else:
            # 병목점 기반 단순 경로
            for bn in bottlenecks[:3]:
                downstream = list(self.graph.successors(bn.id))[:2]
                if downstream:
                    path_str = f"{external_shock} → {bn.id} (Bottleneck) → {' → '.join(downstream)}"
                    lines.append(path_str)

        lines.append("```")
        lines.append("")

        # ===== Section 2: Bottleneck Analysis =====
        lines.append("### Bottleneck Nodes (병목 지점)")
        lines.append("")

        for bn in bottlenecks[:5]:
            downstream_count = self.graph.out_degree(bn.id)
            upstream_count = self.graph.in_degree(bn.id)

            layer_desc = {
                'equipment': 'Equipment (장비)',
                'manufacturer': 'Manufacturer (제조)',
                'integrator': 'Integrator (설계)',
                'end_user': 'End User (최종수요)'
            }.get(bn.layer, bn.layer or 'Core')

            lines.append(f"- **{bn.id}** [{layer_desc}]")
            lines.append(f"  - Criticality Score: {bn.criticality:.2f}")
            lines.append(f"  - Upstream Dependencies: {upstream_count} nodes")
            lines.append(f"  - Downstream Impact: {downstream_count} nodes")

        lines.append("")

        # ===== Section 3: Hub Nodes =====
        if hub_nodes:
            lines.append("### Hub Nodes (핵심 허브)")
            lines.append("")

            for hub in hub_nodes[:3]:
                upstream = list(self.graph.predecessors(hub.id))[:3]
                downstream = list(self.graph.successors(hub.id))[:3]

                lines.append(f"- **{hub.id}** (PageRank: {hub.pagerank:.3f})")
                if upstream:
                    lines.append(f"  - Receives from: {', '.join(upstream)}")
                if downstream:
                    lines.append(f"  - Flows to: {', '.join(downstream)}")

            lines.append("")

        # ===== Section 4: Shock Propagation Simulation =====
        if include_shock_sim and bottlenecks:
            lines.append("### Shock Propagation Simulation")
            lines.append("")

            # 가장 중요한 병목점에서 충격 시뮬레이션
            main_bottleneck = bottlenecks[0]
            impacts = self.simulate_shock_propagation(
                main_bottleneck.id,
                shock_magnitude=-0.10,
                decay_factor=0.7
            )

            if len(impacts) > 1:
                lines.append(f"**Scenario:** {main_bottleneck.id} experiences -10% shock")
                lines.append("")
                lines.append("| Node | Expected Impact | Propagation Depth |")
                lines.append("|------|-----------------|-------------------|")

                # 영향 순으로 정렬
                sorted_impacts = sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)
                for node_id, impact in sorted_impacts[:8]:
                    # 깊이 계산 (간단 버전)
                    try:
                        depth = nx.shortest_path_length(self.graph, main_bottleneck.id, node_id)
                    except:
                        depth = 0

                    lines.append(f"| {node_id} | {impact:+.1%} | {depth} |")

                lines.append("")

                # 내러티브 요약
                heavily_affected = [n for n, i in sorted_impacts if abs(i) > 0.03 and n != main_bottleneck.id]
                if heavily_affected:
                    lines.append(f"**Economic Interpretation:**")
                    lines.append(f"{main_bottleneck.id}에서 -10% 충격 발생 시, "
                               f"{', '.join(heavily_affected[:3])}에 순차적으로 전파됨. "
                               f"총 {len(heavily_affected)}개 노드가 3% 이상 영향받음.")
            else:
                lines.append("Insufficient network connectivity for shock simulation.")

            lines.append("")

        # ===== Section 5: Causality Chains (Event → Node → Impact) =====
        lines.append("### Causality Chains (인과관계 체인)")
        lines.append("")

        # Granger Causality 엣지 기반 체인
        granger_edges = [e for e in self.edges if e.edge_type == EdgeType.GRANGER_CAUSALITY]

        if granger_edges:
            lines.append("**Statistically Significant Causality (Granger Test):**")
            for edge in sorted(granger_edges, key=lambda e: e.granger_pvalue)[:5]:
                lines.append(f"- {edge.source} → {edge.target} (p-value: {edge.granger_pvalue:.3f})")
            lines.append("")

        # Supply Chain 기반 체인
        supply_edges = [e for e in self.edges if e.edge_type == EdgeType.SUPPLY_DEPENDENCY]
        if supply_edges:
            lines.append("**Supply Chain Dependencies:**")

            # 레이어별 그룹화
            upstream_nodes = [n for n in self.nodes.values() if n.layer in ['equipment', 'manufacturer']]
            downstream_nodes = [n for n in self.nodes.values() if n.layer in ['integrator', 'end_user']]

            if upstream_nodes and downstream_nodes:
                up_str = ', '.join([n.id for n in upstream_nodes[:3]])
                down_str = ', '.join([n.id for n in downstream_nodes[:3]])
                lines.append(f"- Upstream ({up_str}) → Downstream ({down_str})")
                lines.append(f"- Total supply chain edges: {len(supply_edges)}")

            lines.append("")

        # 종합 내러티브
        lines.append("---")
        lines.append("")
        lines.append("**Summary:**")

        if bottlenecks and hub_nodes:
            bn_names = ', '.join([b.id for b in bottlenecks[:3]])
            hub_names = ', '.join([h.id for h in hub_nodes[:2]])
            lines.append(f"Network analysis identified **{bn_names}** as critical bottlenecks "
                        f"and **{hub_names}** as central hub nodes. "
                        f"Disruption at bottleneck nodes will propagate through the network "
                        f"following the paths outlined above.")
        else:
            lines.append("Network structure analyzed. Monitor identified nodes for early warning signals.")

        return "\n".join(lines)

    def get_critical_path(self, source: str = None) -> List[str]:
        """
        Critical Path 반환 (리포트용 간단 API)

        Returns:
            노드 ID 리스트 (경로 순서)
        """
        paths = self.find_critical_paths(source=source, max_paths=1)
        if paths:
            return paths[0].nodes
        return []

    def simulate_shock(
        self,
        node: str,
        magnitude: float = -0.10
    ) -> Dict[str, float]:
        """
        Shock Simulation 간단 API (리포트용)

        Args:
            node: 충격 발생 노드
            magnitude: 충격 크기 (기본 -10%)

        Returns:
            {node_id: impact} 딕셔너리
        """
        return self.simulate_shock_propagation(node, magnitude)

    # =========================================================================
    # Export
    # =========================================================================

    def to_dict(self) -> Dict:
        """그래프를 딕셔너리로 변환"""
        return {
            'nodes': [n.to_dict() for n in self.nodes.values()],
            'edges': [e.to_dict() for e in self.edges],
            'stats': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'bottleneck_count': sum(1 for n in self.nodes.values() if n.is_bottleneck)
            }
        }

    def to_markdown(self, insights: List[CausalityInsight] = None) -> str:
        """마크다운 형식으로 내보내기"""
        lines = ["## Causality Network Analysis", ""]

        # 그래프 통계
        lines.append("### Network Statistics")
        lines.append(f"- **Nodes**: {len(self.nodes)}")
        lines.append(f"- **Edges**: {len(self.edges)}")

        bottlenecks = [n for n in self.nodes.values() if n.is_bottleneck]
        if bottlenecks:
            lines.append(f"- **Bottlenecks**: {', '.join([b.id for b in bottlenecks])}")

        lines.append("")

        # 인사이트
        if insights:
            lines.append("### Causality Insights")
            lines.append("")

            for i, insight in enumerate(insights, 1):
                lines.append(f"**[{i}] {insight.insight_type.replace('_', ' ').title()}** "
                           f"(Severity: {insight.severity.upper()}, Confidence: {insight.confidence:.0%})")
                lines.append("")
                lines.append(insight.narrative)
                lines.append("")
                if insight.recommended_action:
                    lines.append(f"*Recommendation: {insight.recommended_action}*")
                lines.append("")
                lines.append("---")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 60)
        print("Causality Graph Engine Test")
        print("=" * 60)

        engine = CausalityGraphEngine(use_llm=False)

        # 공급망 데이터로 그래프 구축
        supply_chain = {
            'equipment': ['ASML', 'AMAT', 'LRCX'],
            'manufacturer': ['TSM', 'INTC'],
            'integrator': ['NVDA', 'AMD', 'AVGO'],
            'end_user': ['MSFT', 'GOOGL', 'AMZN']
        }

        engine.build_from_supply_chain(supply_chain)

        print(f"\n[1] Graph Built")
        print(f"    Nodes: {len(engine.nodes)}")
        print(f"    Edges: {len(engine.edges)}")

        # 병목점 식별
        bottlenecks = engine.identify_bottlenecks(top_n=3)
        print(f"\n[2] Bottlenecks Identified")
        for bn in bottlenecks:
            print(f"    → {bn.id} (Layer: {bn.layer}, Criticality: {bn.criticality:.3f})")

        # 충격 전파 시뮬레이션
        print(f"\n[3] Shock Propagation (ASML -10%)")
        impacts = engine.simulate_shock_propagation('ASML', -0.10)
        for node, impact in sorted(impacts.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"    → {node}: {impact:+.1%}")

        # 인사이트 생성
        print(f"\n[4] Generating Insights...")
        insights = await engine.generate_insights(max_insights=3)

        for i, insight in enumerate(insights, 1):
            print(f"\n    [{i}] {insight.insight_type}")
            print(f"        Path: {insight.path.to_string()[:50]}...")
            print(f"        Severity: {insight.severity}")
            print(f"        Narrative: {insight.narrative[:100]}...")

        print("\n" + "=" * 60)
        print("Test Complete!")
        print("=" * 60)

    asyncio.run(test())
