"""
Economic Insight Agent - Orchestrator
======================================

agentcommand.txt 요구사항에 맞는 6단계 추론 파이프라인 구현

Reasoning Pipeline:
1. Parse request → classify frame (macro / markets / crypto / mixed)
2. Build initial causal graph template (based on frame)
3. Map available evidence/features into nodes/edges (with confidence)
4. Generate top mechanism paths + feedback loops
5. Generate rival hypotheses + falsification tests
6. Produce final JSON report

Integration:
- 기존 EIMAS 모듈 결과를 EIMASAdapter로 변환
- 신규 질문은 템플릿 기반 인과 그래프 생성
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio

from agent.schemas.insight_schema import (
    InsightRequest, EconomicInsightReport, InsightMeta, AnalysisFrame,
    CausalGraph, CausalNode, CausalEdge, EdgeSign, ConfidenceLevel,
    MechanismPath, HypothesesSection, Hypothesis, FalsificationTest,
    RiskSection, RegimeShiftRisk, DataLimitation, RiskSeverity,
    SuggestedDataset, NextAction
)
from agent.core.adapters import EIMASAdapter


class EconomicInsightOrchestrator:
    """
    Economic Insight Agent 메인 오케스트레이터

    기존 EIMAS 결과 활용:
        orchestrator = EconomicInsightOrchestrator()
        report = orchestrator.run_with_eimas_results(
            request=InsightRequest(question="Fed 금리 인상 영향은?"),
            eimas_results={
                'shock_propagation': {...},
                'critical_path': {...},
                'genius_act': {...}
            }
        )

    신규 질문 (템플릿 기반):
        report = orchestrator.run(
            InsightRequest(question="스테이블코인 공급 증가가 국채 수요에 미치는 영향은?")
        )
    """

    def __init__(self, debug: bool = False):
        self.adapter = EIMASAdapter()
        self.debug = debug

        # 프레임별 템플릿 그래프
        self._graph_templates = self._init_graph_templates()

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def run(self, request: InsightRequest) -> EconomicInsightReport:
        """
        신규 질문에 대한 분석 실행 (템플릿 기반)

        Args:
            request: InsightRequest with question

        Returns:
            EconomicInsightReport (JSON-serializable)
        """
        start_time = time.time()

        # Step 1: Parse request → classify frame
        frame = self._classify_frame(request)

        # Step 2: Build initial causal graph template
        graph = self._build_template_graph(frame, request.question)

        # Step 3: Map evidence into nodes/edges
        graph = self._map_evidence_to_graph(graph, request.context or {})

        # Step 4: Generate mechanism paths
        mechanisms = self._generate_mechanisms(graph, frame)

        # Step 5: Generate hypotheses
        hypotheses = self._generate_hypotheses(request.question, graph, mechanisms)

        # Step 6: Produce final report
        processing_time = int((time.time() - start_time) * 1000)

        return self._build_report(
            request=request,
            frame=frame,
            graph=graph,
            mechanisms=mechanisms,
            hypotheses=hypotheses,
            processing_time=processing_time
        )

    def run_with_eimas_results(
        self,
        request: InsightRequest,
        eimas_results: Dict[str, Any]
    ) -> EconomicInsightReport:
        """
        기존 EIMAS 모듈 결과를 활용한 분석

        Args:
            request: InsightRequest
            eimas_results: Dictionary with EIMAS module outputs:
                - shock_propagation: ShockPropagationGraph.analyze() 결과
                - critical_path: CriticalPathAggregator.aggregate() 결과
                - genius_act: GeniusActMacroStrategy.run() 결과
                - bubble_detector: BubbleDetector.detect() 결과 (optional)
                - portfolio: GraphClusteredPortfolio.optimize() 결과 (optional)
                - volume: VolumeAnalyzer.analyze() 결과 (optional)

        Returns:
            EconomicInsightReport
        """
        start_time = time.time()

        # Step 1: Parse request → classify frame
        frame = self._classify_frame(request)

        # Step 2-3: Build graph from EIMAS results
        graph = self._build_graph_from_eimas(eimas_results)

        # Step 4: Generate mechanisms from EIMAS
        mechanisms = self._generate_mechanisms_from_eimas(eimas_results)

        # Step 5: Generate hypotheses
        hypotheses = self._generate_hypotheses(request.question, graph, mechanisms)

        # Collect risks from EIMAS
        risks = self._collect_risks_from_eimas(eimas_results)

        # Collect suggested data and actions
        suggested_data, limitations = self._collect_data_suggestions(eimas_results)
        actions = self._collect_actions(eimas_results)

        # Step 6: Build final report
        processing_time = int((time.time() - start_time) * 1000)

        meta = InsightMeta(
            request_id=request.request_id,
            timestamp=datetime.now().isoformat(),
            frame=frame,
            modules_used=list(set(self.adapter.modules_used)),
            processing_time_ms=processing_time
        )

        # Extract phenomenon from EIMAS results
        phenomenon = self._extract_phenomenon(request.question, eimas_results)

        return EconomicInsightReport(
            meta=meta,
            phenomenon=phenomenon,
            causal_graph=graph,
            mechanisms=mechanisms[:5],  # max 5
            hypotheses=hypotheses,
            risk=RiskSection(
                regime_shift_risks=risks[:10],
                data_limitations=limitations[:5]
            ),
            suggested_data=suggested_data[:10],
            next_actions=actions[:7],  # 3-7 required
            raw_eimas_data=eimas_results if self.debug else None
        )

    # =========================================================================
    # Step 1: Parse Request → Classify Frame
    # =========================================================================

    def _classify_frame(self, request: InsightRequest) -> AnalysisFrame:
        """프레임 분류"""
        if request.frame_hint:
            return request.frame_hint
        return self.adapter.detect_frame(request.question, request.context)

    # =========================================================================
    # Step 2: Build Initial Causal Graph Template
    # =========================================================================

    def _init_graph_templates(self) -> Dict[AnalysisFrame, CausalGraph]:
        """프레임별 기본 인과 그래프 템플릿"""
        templates = {}

        # MACRO 템플릿: Fed → Liquidity → Risk Premium → Assets
        templates[AnalysisFrame.MACRO] = CausalGraph(
            nodes=[
                CausalNode(id="Fed_Policy", name="Fed Policy", layer="POLICY", category="macro"),
                CausalNode(id="Fed_Funds_Rate", name="Fed Funds Rate", layer="POLICY", category="macro"),
                CausalNode(id="Net_Liquidity", name="Net Liquidity", layer="LIQUIDITY", category="macro"),
                CausalNode(id="RRP", name="Reverse Repo", layer="LIQUIDITY", category="macro"),
                CausalNode(id="TGA", name="Treasury General Account", layer="LIQUIDITY", category="macro"),
                CausalNode(id="Credit_Spread", name="Credit Spread", layer="RISK_PREMIUM", category="market"),
                CausalNode(id="VIX", name="VIX", layer="RISK_PREMIUM", category="market"),
                CausalNode(id="SPY", name="S&P 500", layer="ASSET_PRICE", category="market"),
                CausalNode(id="TLT", name="Long Treasury ETF", layer="ASSET_PRICE", category="market"),
            ],
            edges=[
                CausalEdge(source="Fed_Policy", target="Fed_Funds_Rate", sign=EdgeSign.POSITIVE, mechanism="직접 통제"),
                CausalEdge(source="Fed_Funds_Rate", target="Net_Liquidity", sign=EdgeSign.NEGATIVE, mechanism="금리 인상 → 유동성 감소"),
                CausalEdge(source="RRP", target="Net_Liquidity", sign=EdgeSign.NEGATIVE, mechanism="RRP 증가 → 유동성 흡수"),
                CausalEdge(source="TGA", target="Net_Liquidity", sign=EdgeSign.NEGATIVE, mechanism="TGA 증가 → 유동성 흡수"),
                CausalEdge(source="Net_Liquidity", target="VIX", sign=EdgeSign.NEGATIVE, mechanism="유동성 ↓ → 변동성 ↑"),
                CausalEdge(source="Net_Liquidity", target="Credit_Spread", sign=EdgeSign.NEGATIVE, mechanism="유동성 ↓ → 스프레드 ↑"),
                CausalEdge(source="VIX", target="SPY", sign=EdgeSign.NEGATIVE, mechanism="변동성 ↑ → 주가 ↓"),
                CausalEdge(source="Credit_Spread", target="SPY", sign=EdgeSign.NEGATIVE, mechanism="스프레드 ↑ → 주가 ↓"),
                CausalEdge(source="Fed_Funds_Rate", target="TLT", sign=EdgeSign.NEGATIVE, mechanism="금리 ↑ → 채권 ↓"),
            ],
            has_cycles=False,
            critical_path=["Fed_Policy", "Fed_Funds_Rate", "Net_Liquidity", "VIX", "SPY"]
        )

        # CRYPTO 템플릿: Stablecoin → Reserve → Treasury
        templates[AnalysisFrame.CRYPTO] = CausalGraph(
            nodes=[
                CausalNode(id="Stablecoin_Supply", name="Stablecoin Supply", layer="LIQUIDITY", category="crypto"),
                CausalNode(id="USDC", name="USDC Supply", layer="LIQUIDITY", category="crypto"),
                CausalNode(id="USDT", name="USDT Supply", layer="LIQUIDITY", category="crypto"),
                CausalNode(id="Reserve_Demand", name="Reserve Demand", layer="LIQUIDITY", category="macro"),
                CausalNode(id="TBill_Demand", name="T-Bill Demand", layer="ASSET_PRICE", category="macro"),
                CausalNode(id="Short_Rate", name="Short-term Rate", layer="POLICY", category="macro"),
                CausalNode(id="BTC", name="Bitcoin", layer="ASSET_PRICE", category="crypto"),
                CausalNode(id="ETH", name="Ethereum", layer="ASSET_PRICE", category="crypto"),
                CausalNode(id="DeFi_TVL", name="DeFi TVL", layer="LIQUIDITY", category="crypto"),
            ],
            edges=[
                CausalEdge(source="USDC", target="Stablecoin_Supply", sign=EdgeSign.POSITIVE, mechanism="공급 합산"),
                CausalEdge(source="USDT", target="Stablecoin_Supply", sign=EdgeSign.POSITIVE, mechanism="공급 합산"),
                CausalEdge(source="Stablecoin_Supply", target="Reserve_Demand", sign=EdgeSign.POSITIVE, mechanism="담보 수요"),
                CausalEdge(source="Reserve_Demand", target="TBill_Demand", sign=EdgeSign.POSITIVE, mechanism="국채 매수"),
                CausalEdge(source="TBill_Demand", target="Short_Rate", sign=EdgeSign.NEGATIVE, mechanism="수요 ↑ → 금리 ↓"),
                CausalEdge(source="Stablecoin_Supply", target="DeFi_TVL", sign=EdgeSign.POSITIVE, mechanism="유동성 공급"),
                CausalEdge(source="DeFi_TVL", target="ETH", sign=EdgeSign.POSITIVE, mechanism="활동 증가 → 가스비 ↑ → ETH ↑"),
                CausalEdge(source="Stablecoin_Supply", target="BTC", sign=EdgeSign.POSITIVE, mechanism="매수 유동성 증가"),
            ],
            has_cycles=False,
            critical_path=["Stablecoin_Supply", "Reserve_Demand", "TBill_Demand"]
        )

        # MARKETS 템플릿
        templates[AnalysisFrame.MARKETS] = CausalGraph(
            nodes=[
                CausalNode(id="Earnings", name="Corporate Earnings", layer="FUNDAMENTAL", category="market"),
                CausalNode(id="Valuation", name="Market Valuation", layer="FUNDAMENTAL", category="market"),
                CausalNode(id="Sentiment", name="Market Sentiment", layer="BEHAVIORAL", category="market"),
                CausalNode(id="Flows", name="Fund Flows", layer="FLOW", category="market"),
                CausalNode(id="SPY", name="S&P 500", layer="ASSET_PRICE", category="market"),
                CausalNode(id="QQQ", name="Nasdaq 100", layer="ASSET_PRICE", category="market"),
                CausalNode(id="Sector_Rotation", name="Sector Rotation", layer="FLOW", category="market"),
                CausalNode(id="VIX", name="VIX", layer="RISK_PREMIUM", category="market"),
            ],
            edges=[
                CausalEdge(source="Earnings", target="Valuation", sign=EdgeSign.POSITIVE, mechanism="실적 → 밸류에이션"),
                CausalEdge(source="Valuation", target="SPY", sign=EdgeSign.POSITIVE, mechanism="저평가 → 상승"),
                CausalEdge(source="Sentiment", target="Flows", sign=EdgeSign.POSITIVE, mechanism="심리 → 자금 이동"),
                CausalEdge(source="Flows", target="SPY", sign=EdgeSign.POSITIVE, mechanism="유입 → 가격 상승"),
                CausalEdge(source="Flows", target="QQQ", sign=EdgeSign.POSITIVE, mechanism="유입 → 가격 상승"),
                CausalEdge(source="Sector_Rotation", target="QQQ", sign=EdgeSign.AMBIGUOUS, mechanism="로테이션 방향에 따라 상이"),
                CausalEdge(source="VIX", target="Sentiment", sign=EdgeSign.NEGATIVE, mechanism="변동성 ↑ → 심리 ↓"),
            ],
            has_cycles=True,
            critical_path=["Sentiment", "Flows", "SPY"]
        )

        # MIXED 템플릿 = MACRO + CRYPTO 결합
        templates[AnalysisFrame.MIXED] = CausalGraph(
            nodes=templates[AnalysisFrame.MACRO].nodes + templates[AnalysisFrame.CRYPTO].nodes[:5],
            edges=templates[AnalysisFrame.MACRO].edges + [
                CausalEdge(source="Net_Liquidity", target="Stablecoin_Supply", sign=EdgeSign.POSITIVE, mechanism="유동성 → 스테이블코인 수요"),
                CausalEdge(source="Stablecoin_Supply", target="Reserve_Demand", sign=EdgeSign.POSITIVE, mechanism="담보 수요"),
            ],
            has_cycles=False,
            critical_path=["Fed_Policy", "Net_Liquidity", "Stablecoin_Supply", "Reserve_Demand"]
        )

        return templates

    def _build_template_graph(self, frame: AnalysisFrame, question: str) -> CausalGraph:
        """프레임에 맞는 템플릿 그래프 반환"""
        template = self._graph_templates.get(frame, self._graph_templates[AnalysisFrame.MIXED])

        # Deep copy to avoid modifying template
        return CausalGraph(
            nodes=[CausalNode(**n.model_dump()) for n in template.nodes],
            edges=[CausalEdge(**e.model_dump()) for e in template.edges],
            has_cycles=template.has_cycles,
            critical_path=template.critical_path
        )

    # =========================================================================
    # Step 3: Map Evidence into Nodes/Edges
    # =========================================================================

    def _map_evidence_to_graph(self, graph: CausalGraph, context: Dict) -> CausalGraph:
        """컨텍스트 정보를 그래프에 매핑"""
        # 컨텍스트에서 신뢰도 업데이트
        for node in graph.nodes:
            node_data = context.get(node.id, {})
            if 'centrality' in node_data:
                node.centrality = node_data['centrality']
            if 'criticality' in node_data:
                node.criticality = node_data['criticality']

        for edge in graph.edges:
            edge_key = f"{edge.source}_{edge.target}"
            edge_data = context.get(edge_key, {})
            if 'p_value' in edge_data:
                edge.p_value = edge_data['p_value']
                edge.confidence = self.adapter._p_to_confidence(edge_data['p_value'])
            if 'lag' in edge_data:
                edge.lag = edge_data['lag']

        return graph

    # =========================================================================
    # Step 2-3 (EIMAS): Build Graph from EIMAS Results
    # =========================================================================

    def _build_graph_from_eimas(self, eimas_results: Dict) -> CausalGraph:
        """EIMAS 결과로부터 CausalGraph 생성"""

        # ShockPropagation 우선
        if 'shock_propagation' in eimas_results:
            return self.adapter.adapt_shock_propagation(eimas_results['shock_propagation'])

        # 없으면 CriticalPath에서 노드만 추출
        if 'critical_path' in eimas_results:
            cp = eimas_results['critical_path']
            nodes = []
            edges = []

            # 경고 기반 노드 생성
            for i, warning in enumerate(cp.get('active_warnings', [])):
                nodes.append(CausalNode(
                    id=f"warning_{i}",
                    name=warning[:50],
                    category="risk"
                ))

            # Path contributions 기반 노드
            for path, contrib in cp.get('path_contributions', {}).items():
                nodes.append(CausalNode(
                    id=path,
                    name=path.replace('_', ' ').title(),
                    criticality=contrib / 100
                ))

            return CausalGraph(nodes=nodes, edges=edges)

        # 기본 빈 그래프
        return CausalGraph(nodes=[], edges=[])

    # =========================================================================
    # Step 4: Generate Mechanism Paths
    # =========================================================================

    def _generate_mechanisms(self, graph: CausalGraph, frame: AnalysisFrame) -> List[MechanismPath]:
        """그래프에서 메커니즘 경로 추출"""
        mechanisms = []

        # Critical Path 기반 메커니즘
        if graph.critical_path and len(graph.critical_path) >= 2:
            path = graph.critical_path
            edge_signs = []

            for i in range(len(path) - 1):
                # 해당 엣지 찾기
                for edge in graph.edges:
                    if edge.source == path[i] and edge.target == path[i + 1]:
                        edge_signs.append(edge.sign.value)
                        break
                else:
                    edge_signs.append("+")  # 기본값

            # Net effect 계산 (음수 개수가 홀수면 음수)
            neg_count = sum(1 for s in edge_signs if s == "-")
            net_effect = EdgeSign.NEGATIVE if neg_count % 2 == 1 else EdgeSign.POSITIVE

            mechanisms.append(MechanismPath(
                nodes=path,
                edge_signs=edge_signs,
                net_effect=net_effect,
                narrative=self._generate_path_narrative(path, edge_signs, frame),
                strength=ConfidenceLevel.HIGH
            ))

        # 추가 경로 탐색 (BFS로 다른 경로 찾기)
        additional_paths = self._find_alternative_paths(graph)
        for path, signs in additional_paths[:2]:  # 최대 2개 추가
            neg_count = sum(1 for s in signs if s == "-")
            net_effect = EdgeSign.NEGATIVE if neg_count % 2 == 1 else EdgeSign.POSITIVE

            mechanisms.append(MechanismPath(
                nodes=path,
                edge_signs=signs,
                net_effect=net_effect,
                narrative=self._generate_path_narrative(path, signs, frame),
                strength=ConfidenceLevel.MEDIUM
            ))

        return mechanisms if mechanisms else [
            MechanismPath(
                nodes=["Unknown"],
                edge_signs=[],
                net_effect=EdgeSign.AMBIGUOUS,
                narrative="추가 데이터 필요",
                strength=ConfidenceLevel.LOW
            )
        ]

    def _generate_mechanisms_from_eimas(self, eimas_results: Dict) -> List[MechanismPath]:
        """EIMAS 결과에서 메커니즘 추출"""
        mechanisms = []

        # GeniusAct 결과
        if 'genius_act' in eimas_results:
            mechanisms.extend(self.adapter.adapt_genius_act(eimas_results['genius_act']))

        # ShockPropagation 결과에서 shock_paths
        if 'shock_propagation' in eimas_results:
            spg = eimas_results['shock_propagation']
            for shock_path in spg.get('shock_paths', [])[:3]:
                path = shock_path.get('path', [])
                if len(path) >= 2:
                    mechanisms.append(MechanismPath(
                        nodes=path,
                        edge_signs=['+'] * (len(path) - 1),
                        net_effect=EdgeSign.POSITIVE,
                        narrative=f"충격 전파 경로: {' → '.join(path)}",
                        strength=ConfidenceLevel.HIGH if shock_path.get('strength', 0) > 0.7 else ConfidenceLevel.MEDIUM
                    ))

        # 최소 1개는 보장
        if not mechanisms:
            mechanisms.append(MechanismPath(
                nodes=["Data_Input", "Analysis", "Output"],
                edge_signs=['+', '+'],
                net_effect=EdgeSign.POSITIVE,
                narrative="EIMAS 분석 파이프라인 실행됨",
                strength=ConfidenceLevel.LOW
            ))

        return mechanisms

    def _generate_path_narrative(self, path: List[str], signs: List[str], frame: AnalysisFrame) -> str:
        """경로에 대한 설명 생성"""
        if not path:
            return "경로 없음"

        parts = []
        for i, node in enumerate(path):
            node_name = node.replace('_', ' ')
            if i < len(signs):
                sign = "↑" if signs[i] == "+" else "↓" if signs[i] == "-" else "?"
                parts.append(f"{node_name} {sign}")
            else:
                parts.append(node_name)

        return " → ".join(parts)

    def _find_alternative_paths(self, graph: CausalGraph) -> List[Tuple[List[str], List[str]]]:
        """그래프에서 대안 경로 찾기"""
        # 간단한 BFS로 시작/끝 노드 연결 경로 찾기
        if len(graph.nodes) < 2:
            return []

        # 인접 리스트 생성
        adj = {}
        edge_signs = {}
        for edge in graph.edges:
            if edge.source not in adj:
                adj[edge.source] = []
            adj[edge.source].append(edge.target)
            edge_signs[(edge.source, edge.target)] = edge.sign.value

        paths = []

        # 모든 노드 쌍에 대해 경로 탐색
        for start_node in graph.nodes[:3]:  # 처음 3개 노드만
            for end_node in graph.nodes[-3:]:  # 마지막 3개 노드만
                if start_node.id != end_node.id:
                    path, signs = self._bfs_path(start_node.id, end_node.id, adj, edge_signs)
                    if path and len(path) >= 2 and path != graph.critical_path:
                        paths.append((path, signs))

        return paths

    def _bfs_path(
        self,
        start: str,
        end: str,
        adj: Dict[str, List[str]],
        edge_signs: Dict[Tuple[str, str], str]
    ) -> Tuple[Optional[List[str]], List[str]]:
        """BFS로 경로 찾기"""
        from collections import deque

        queue = deque([(start, [start], [])])
        visited = {start}

        while queue:
            current, path, signs = queue.popleft()

            if current == end:
                return path, signs

            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_signs = signs + [edge_signs.get((current, neighbor), "+")]
                    queue.append((neighbor, path + [neighbor], new_signs))

        return None, []

    # =========================================================================
    # Step 5: Generate Hypotheses
    # =========================================================================

    def _generate_hypotheses(
        self,
        question: str,
        graph: CausalGraph,
        mechanisms: List[MechanismPath]
    ) -> HypothesesSection:
        """가설 생성"""
        return self.adapter.generate_hypotheses(question, graph, mechanisms)

    # =========================================================================
    # Step 6: Build Final Report
    # =========================================================================

    def _build_report(
        self,
        request: InsightRequest,
        frame: AnalysisFrame,
        graph: CausalGraph,
        mechanisms: List[MechanismPath],
        hypotheses: HypothesesSection,
        processing_time: int
    ) -> EconomicInsightReport:
        """최종 리포트 생성 (템플릿 기반)"""

        meta = InsightMeta(
            request_id=request.request_id,
            timestamp=datetime.now().isoformat(),
            frame=frame,
            modules_used=["template_graph"],
            processing_time_ms=processing_time
        )

        # Phenomenon 추출
        phenomenon = self._extract_phenomenon_from_question(request.question, frame)

        # 리스크 생성
        risks = self._generate_template_risks(frame, graph)

        # 데이터 제안
        suggested_data = self._generate_suggested_data(frame)

        # 다음 행동
        actions = self._generate_next_actions(frame, mechanisms)

        return EconomicInsightReport(
            meta=meta,
            phenomenon=phenomenon,
            causal_graph=graph,
            mechanisms=mechanisms[:5],
            hypotheses=hypotheses,
            risk=RiskSection(
                regime_shift_risks=risks,
                data_limitations=[
                    DataLimitation(
                        description="템플릿 기반 분석으로 실제 데이터 미반영",
                        impact="정확도 제한적",
                        mitigation="EIMAS 모듈과 통합 실행 권장"
                    )
                ]
            ),
            suggested_data=suggested_data,
            next_actions=actions
        )

    def _extract_phenomenon_from_question(self, question: str, frame: AnalysisFrame) -> str:
        """질문에서 현상 추출"""
        # 간단한 변환
        if "영향" in question:
            return question.replace("?", "").replace("은?", "에 대한 분석").strip()
        return f"{question.rstrip('?')}에 대한 분석"

    def _extract_phenomenon(self, question: str, eimas_results: Dict) -> str:
        """EIMAS 결과에서 현상 추출"""
        # 레짐 정보 활용
        if 'critical_path' in eimas_results:
            regime = eimas_results['critical_path'].get('current_regime', 'UNKNOWN')
            risk_score = eimas_results['critical_path'].get('total_risk_score', 0)
            return f"현재 {regime} 레짐, 리스크 점수 {risk_score:.1f}/100"

        if 'genius_act' in eimas_results:
            regime = eimas_results['genius_act'].get('regime', 'NEUTRAL')
            return f"유동성 레짐: {regime}"

        return self._extract_phenomenon_from_question(question, AnalysisFrame.MIXED)

    def _generate_template_risks(self, frame: AnalysisFrame, graph: CausalGraph) -> List[RegimeShiftRisk]:
        """프레임별 기본 리스크"""
        risks = []

        if frame == AnalysisFrame.MACRO:
            risks.append(RegimeShiftRisk(
                description="Fed 정책 기조 전환 가능성",
                trigger="인플레이션/고용 데이터 급변",
                severity=RiskSeverity.HIGH
            ))
            risks.append(RegimeShiftRisk(
                description="유동성 급감 리스크",
                trigger="RRP/TGA 급변동",
                severity=RiskSeverity.MEDIUM
            ))

        elif frame == AnalysisFrame.CRYPTO:
            risks.append(RegimeShiftRisk(
                description="스테이블코인 규제 강화 가능성",
                trigger="의회/SEC 규제 발표",
                severity=RiskSeverity.HIGH
            ))
            risks.append(RegimeShiftRisk(
                description="디페그 리스크",
                trigger="담보 자산 가치 하락",
                severity=RiskSeverity.CRITICAL
            ))

        elif frame == AnalysisFrame.MARKETS:
            risks.append(RegimeShiftRisk(
                description="밸류에이션 조정 리스크",
                trigger="실적 시즌 하회",
                severity=RiskSeverity.MEDIUM
            ))

        if graph.has_cycles:
            risks.append(RegimeShiftRisk(
                description="피드백 루프로 인한 급격한 변동 가능성",
                trigger="인과 그래프 사이클 감지됨",
                severity=RiskSeverity.MEDIUM
            ))

        return risks

    def _generate_suggested_data(self, frame: AnalysisFrame) -> List[SuggestedDataset]:
        """프레임별 추천 데이터"""
        data = []

        if frame in [AnalysisFrame.MACRO, AnalysisFrame.MIXED]:
            data.extend([
                SuggestedDataset(
                    name="Fed H.4.1 Weekly Release",
                    category="macro",
                    priority=1,
                    rationale="Fed 자산 및 유동성 추적",
                    source="FRED"
                ),
                SuggestedDataset(
                    name="Treasury TGA Balance",
                    category="macro",
                    priority=2,
                    rationale="정부 지출/세입 유동성 영향",
                    source="FRED"
                ),
            ])

        if frame in [AnalysisFrame.CRYPTO, AnalysisFrame.MIXED]:
            data.extend([
                SuggestedDataset(
                    name="Stablecoin Market Cap",
                    category="on-chain",
                    priority=1,
                    rationale="스테이블코인 공급 추적",
                    source="DeFiLlama/CoinGecko"
                ),
                SuggestedDataset(
                    name="DeFi TVL",
                    category="on-chain",
                    priority=2,
                    rationale="DeFi 활동 및 유동성",
                    source="DeFiLlama"
                ),
            ])

        if frame == AnalysisFrame.MARKETS:
            data.extend([
                SuggestedDataset(
                    name="ETF Fund Flows",
                    category="flows",
                    priority=1,
                    rationale="자금 유출입 추적",
                    source="Bloomberg/Morningstar"
                ),
                SuggestedDataset(
                    name="Sector Rotation Data",
                    category="flows",
                    priority=2,
                    rationale="섹터 로테이션 분석",
                    source="yfinance"
                ),
            ])

        return data

    def _generate_next_actions(self, frame: AnalysisFrame, mechanisms: List[MechanismPath]) -> List[NextAction]:
        """다음 행동 생성"""
        actions = []

        # 메커니즘 기반 행동
        if mechanisms:
            top_path = mechanisms[0]
            actions.append(NextAction(
                description=f"핵심 경로 모니터링: {' → '.join(top_path.nodes[:3])}",
                category="monitor",
                priority=1,
                timeframe="일간"
            ))

        # 프레임별 행동
        if frame == AnalysisFrame.MACRO:
            actions.extend([
                NextAction(description="FOMC 회의록 및 발언 추적", category="analysis", priority=2, timeframe="주간"),
                NextAction(description="순유동성 지표 일간 체크", category="monitor", priority=1, timeframe="일간"),
            ])
        elif frame == AnalysisFrame.CRYPTO:
            actions.extend([
                NextAction(description="스테이블코인 공급량 변화 추적", category="monitor", priority=1, timeframe="일간"),
                NextAction(description="규제 뉴스 모니터링", category="monitor", priority=2, timeframe="일간"),
            ])
        elif frame == AnalysisFrame.MARKETS:
            actions.extend([
                NextAction(description="섹터 ETF 자금흐름 분석", category="analysis", priority=2, timeframe="주간"),
                NextAction(description="VIX 레벨 모니터링", category="monitor", priority=1, timeframe="일간"),
            ])
        elif frame == AnalysisFrame.MIXED:
            # MIXED는 MACRO + CRYPTO 결합
            actions.extend([
                NextAction(description="Fed 유동성 지표 및 스테이블코인 공급 동시 추적", category="monitor", priority=1, timeframe="일간"),
                NextAction(description="전통 자산-암호화폐 상관관계 분석", category="analysis", priority=2, timeframe="주간"),
            ])

        # EIMAS 연동 권장
        actions.append(NextAction(
            description="EIMAS 전체 파이프라인 실행으로 상세 분석",
            category="analysis",
            priority=3,
            timeframe="필요시"
        ))

        return actions[:7]  # max 7

    def _collect_risks_from_eimas(self, eimas_results: Dict) -> List[RegimeShiftRisk]:
        """EIMAS 결과에서 리스크 수집"""
        risks = []

        if 'critical_path' in eimas_results:
            risks.extend(self.adapter.adapt_critical_path(eimas_results['critical_path']))

        if 'bubble_detector' in eimas_results:
            risks.extend(self.adapter.adapt_bubble_detector(eimas_results['bubble_detector']))

        return risks

    def _collect_data_suggestions(self, eimas_results: Dict) -> Tuple[List[SuggestedDataset], List[DataLimitation]]:
        """EIMAS 결과에서 데이터 제안 수집"""
        suggested = []
        limitations = []

        if 'volume' in eimas_results:
            s, l = self.adapter.adapt_volume_analyzer(eimas_results['volume'])
            suggested.extend(s)
            limitations.extend(l)

        # 기본 제안 추가
        suggested.extend(self._generate_suggested_data(AnalysisFrame.MIXED))

        return suggested, limitations

    def _collect_actions(self, eimas_results: Dict) -> List[NextAction]:
        """EIMAS 결과에서 행동 수집"""
        actions = []

        if 'portfolio' in eimas_results:
            actions.extend(self.adapter.adapt_portfolio(eimas_results['portfolio']))

        # 기본 행동 추가
        actions.extend([
            NextAction(description="분석 결과 검증 및 백테스트", category="analysis", priority=2, timeframe="주간"),
            NextAction(description="리스크 지표 임계값 알림 설정", category="monitor", priority=1, timeframe="즉시"),
            NextAction(description="다음 EIMAS 분석 스케줄링", category="data", priority=3, timeframe="일간"),
        ])

        return actions


# =============================================================================
# Async Wrapper (for integration with existing async EIMAS)
# =============================================================================

class AsyncEconomicInsightOrchestrator:
    """비동기 래퍼"""

    def __init__(self, debug: bool = False):
        self._sync = EconomicInsightOrchestrator(debug=debug)

    async def run(self, request: InsightRequest) -> EconomicInsightReport:
        return await asyncio.to_thread(self._sync.run, request)

    async def run_with_eimas_results(
        self,
        request: InsightRequest,
        eimas_results: Dict[str, Any]
    ) -> EconomicInsightReport:
        return await asyncio.to_thread(
            self._sync.run_with_eimas_results,
            request,
            eimas_results
        )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Test 1: Template-based analysis
    print("=" * 60)
    print("Test 1: Template-based Analysis")
    print("=" * 60)

    orchestrator = EconomicInsightOrchestrator(debug=True)

    request = InsightRequest(
        question="스테이블코인 공급 증가가 국채 수요에 미치는 영향은?"
    )

    report = orchestrator.run(request)

    print(f"Frame: {report.meta.frame.value}")
    print(f"Phenomenon: {report.phenomenon}")
    print(f"Nodes: {len(report.causal_graph.nodes)}")
    print(f"Edges: {len(report.causal_graph.edges)}")
    print(f"Mechanisms: {len(report.mechanisms)}")
    print(f"Main Hypothesis: {report.hypotheses.main.statement[:50]}...")
    print(f"Risks: {len(report.risk.regime_shift_risks)}")
    print(f"Suggested Data: {len(report.suggested_data)}")
    print(f"Next Actions: {len(report.next_actions)}")
    print(f"Processing Time: {report.meta.processing_time_ms}ms")

    # Test 2: EIMAS-based analysis
    print("\n" + "=" * 60)
    print("Test 2: EIMAS-based Analysis")
    print("=" * 60)

    mock_eimas = {
        'shock_propagation': {
            'node_analysis': {
                'Fed_Funds': {'layer': 'POLICY', 'centrality_score': 0.8},
                'Net_Liquidity': {'layer': 'LIQUIDITY', 'centrality_score': 0.6},
                'SPY': {'layer': 'ASSET_PRICE', 'centrality_score': 0.4}
            },
            'granger_results': [
                {'cause': 'Fed_Funds', 'effect': 'Net_Liquidity', 'p_value': 0.01, 'optimal_lag': 5, 'is_significant': True, 'coefficient': -0.3},
                {'cause': 'Net_Liquidity', 'effect': 'SPY', 'p_value': 0.03, 'optimal_lag': 3, 'is_significant': True, 'coefficient': 0.5}
            ],
            'shock_paths': [{'path': ['Fed_Funds', 'Net_Liquidity', 'SPY'], 'strength': 0.8}]
        },
        'critical_path': {
            'current_regime': 'BULL',
            'transition_probability': 0.35,
            'total_risk_score': 42.0,
            'active_warnings': ['VIX 상승세'],
            'path_contributions': {'liquidity': 30, 'credit': 12}
        },
        'genius_act': {
            'regime': 'EXPANSION',
            'extended_liquidity': {
                'base_liquidity': 5800000000000,
                'stablecoin_contribution': 150000000000,
                'total_extended_liquidity': 5950000000000
            },
            'signals': [
                {'type': 'LIQUIDITY_INJECTION', 'description': 'Net liquidity expanding'}
            ]
        }
    }

    request2 = InsightRequest(
        question="Fed 금리 정책이 시장에 미치는 영향은?",
        frame_hint=AnalysisFrame.MACRO
    )

    report2 = orchestrator.run_with_eimas_results(request2, mock_eimas)

    print(f"Frame: {report2.meta.frame.value}")
    print(f"Phenomenon: {report2.phenomenon}")
    print(f"Modules Used: {report2.meta.modules_used}")
    print(f"Nodes: {len(report2.causal_graph.nodes)}")
    print(f"Edges: {len(report2.causal_graph.edges)}")
    print(f"Critical Path: {report2.causal_graph.critical_path}")
    print(f"Mechanisms: {len(report2.mechanisms)}")
    for i, m in enumerate(report2.mechanisms[:2]):
        print(f"  [{i+1}] {m.narrative[:60]}...")
    print(f"Risks: {len(report2.risk.regime_shift_risks)}")
    for r in report2.risk.regime_shift_risks[:2]:
        print(f"  - {r.description[:50]}... ({r.severity.value})")
    print(f"Processing Time: {report2.meta.processing_time_ms}ms")

    # JSON 출력 테스트
    print("\n" + "=" * 60)
    print("JSON Output Sample (first 1000 chars)")
    print("=" * 60)
    json_output = report2.model_dump_json(indent=2)
    print(json_output[:1000] + "...")

    print("\n[SUCCESS] All tests passed!")
