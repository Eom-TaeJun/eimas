"""
EIMAS Module Adapters
=====================

기존 EIMAS 모듈 출력을 Economic Insight Schema로 변환

Supported Modules:
- ShockPropagationGraph → CausalGraph
- CriticalPathAggregator → RegimeShiftRisk
- GeniusActMacroStrategy → MechanismPath
- VolumeAnalyzer → DataLimitation, SuggestedDataset
- GraphClusteredPortfolio → NextAction
- BubbleDetector → RegimeShiftRisk
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Any, Optional
from datetime import datetime

from agent.schemas.insight_schema import (
    CausalGraph, CausalNode, CausalEdge, EdgeSign, ConfidenceLevel,
    MechanismPath, RegimeShiftRisk, RiskSeverity, DataLimitation,
    SuggestedDataset, NextAction, HypothesesSection, Hypothesis, FalsificationTest,
    AnalysisFrame, InsightMeta
)


class EIMASAdapter:
    """
    기존 EIMAS 모듈 결과를 Economic Insight Schema로 변환하는 어댑터

    Usage:
        adapter = EIMASAdapter()
        causal_graph = adapter.adapt_shock_propagation(spg_result)
        risks = adapter.adapt_critical_path(cp_result)
    """

    def __init__(self):
        self.modules_used = []

    # =========================================================================
    # ShockPropagationGraph → CausalGraph
    # =========================================================================

    def adapt_shock_propagation(self, spg_result: Dict) -> CausalGraph:
        """
        ShockPropagationGraph 결과를 CausalGraph로 변환

        기존 모듈: lib/shock_propagation_graph.py
        - PropagationAnalysis: lead_lag_results, granger_results, shock_paths
        """
        self.modules_used.append("shock_propagation_graph")

        nodes = []
        edges = []
        node_ids = set()

        # 1. 노드 추출 (node_analysis에서)
        node_analysis = spg_result.get('node_analysis', {})
        for node_id, analysis in node_analysis.items():
            nodes.append(CausalNode(
                id=node_id,
                name=self._format_node_name(node_id),
                layer=analysis.get('layer'),
                category=self._infer_category(node_id),
                centrality=analysis.get('centrality_score'),
                criticality=analysis.get('is_bottleneck', False) and 1.0 or 0.0
            ))
            node_ids.add(node_id)

        # 2. Granger 인과관계 엣지
        granger_results = spg_result.get('granger_results', [])
        for gr in granger_results:
            if gr.get('is_significant'):
                source = gr.get('cause', gr.get('source'))
                target = gr.get('effect', gr.get('target'))

                # 노드가 없으면 추가
                for nid in [source, target]:
                    if nid and nid not in node_ids:
                        nodes.append(CausalNode(
                            id=nid,
                            name=self._format_node_name(nid),
                            category=self._infer_category(nid)
                        ))
                        node_ids.add(nid)

                edges.append(CausalEdge(
                    source=source,
                    target=target,
                    sign=self._infer_sign(gr),
                    lag=gr.get('optimal_lag'),
                    p_value=gr.get('p_value'),
                    confidence=self._p_to_confidence(gr.get('p_value', 1)),
                    mechanism=f"Granger Causality (lag={gr.get('optimal_lag')})"
                ))

        # 3. Lead-Lag 관계 엣지
        lead_lag = spg_result.get('lead_lag_results', [])
        for ll in lead_lag:
            if ll.get('leader') and ll.get('follower'):
                edges.append(CausalEdge(
                    source=ll['leader'],
                    target=ll['follower'],
                    sign=EdgeSign.POSITIVE if ll.get('correlation', 0) > 0 else EdgeSign.NEGATIVE,
                    lag=ll.get('lag_days'),
                    confidence=ConfidenceLevel.HIGH if abs(ll.get('correlation', 0)) > 0.7 else ConfidenceLevel.MEDIUM
                ))

        # 4. Critical Path 추출
        critical_path = None
        shock_paths = spg_result.get('shock_paths', [])
        if shock_paths:
            top_path = shock_paths[0]
            critical_path = top_path.get('path', [])

        # 5. 사이클 감지
        has_cycles = self._detect_cycles(edges)

        return CausalGraph(
            nodes=nodes,
            edges=edges,
            has_cycles=has_cycles,
            critical_path=critical_path
        )

    # =========================================================================
    # CriticalPathAggregator → RegimeShiftRisk
    # =========================================================================

    def adapt_critical_path(self, cp_result: Dict) -> List[RegimeShiftRisk]:
        """
        CriticalPathAggregator 결과를 RegimeShiftRisk로 변환

        기존 모듈: lib/critical_path.py
        - total_risk_score, current_regime, transition_probability, active_warnings
        """
        self.modules_used.append("critical_path")

        risks = []

        # 1. 레짐 전환 리스크
        trans_prob = cp_result.get('transition_probability', 0)
        if trans_prob > 0.3:
            current_regime = cp_result.get('current_regime', 'UNKNOWN')
            risks.append(RegimeShiftRisk(
                description=f"현재 {current_regime} 레짐에서 전환 가능성",
                trigger=f"전환 확률 {trans_prob:.1%} (임계값 30% 초과)",
                probability=trans_prob,
                severity=RiskSeverity.CRITICAL if trans_prob > 0.7 else
                        RiskSeverity.HIGH if trans_prob > 0.5 else
                        RiskSeverity.MEDIUM,
                source_module="critical_path"
            ))

        # 2. 고위험 점수
        risk_score = cp_result.get('total_risk_score', 0)
        if risk_score > 60:
            risks.append(RegimeShiftRisk(
                description=f"종합 리스크 점수 {risk_score:.1f}/100 (고위험)",
                trigger="CriticalPath 리스크 스코어 60 초과",
                severity=RiskSeverity.HIGH if risk_score > 70 else RiskSeverity.MEDIUM,
                source_module="critical_path"
            ))

        # 3. 활성 경고들
        for warning in cp_result.get('active_warnings', [])[:5]:
            risks.append(RegimeShiftRisk(
                description=warning,
                trigger="임계값 위반",
                severity=RiskSeverity.MEDIUM,
                source_module="critical_path"
            ))

        # 4. Path Contribution 기반 리스크
        path_contrib = cp_result.get('path_contributions', {})
        for path, contrib in path_contrib.items():
            if contrib > 25:  # 25% 이상 기여
                risks.append(RegimeShiftRisk(
                    description=f"{path} 경로 리스크 기여도 {contrib:.1f}%",
                    trigger=f"{path} 경로 압력 증가",
                    severity=RiskSeverity.HIGH if contrib > 40 else RiskSeverity.MEDIUM,
                    source_module="critical_path"
                ))

        return risks

    # =========================================================================
    # GeniusActMacroStrategy → MechanismPath
    # =========================================================================

    def adapt_genius_act(self, ga_result: Dict) -> List[MechanismPath]:
        """
        GeniusActMacroStrategy 결과를 MechanismPath로 변환

        기존 모듈: lib/genius_act_macro.py
        - Extended Liquidity: M = B + S*B*
        - Stablecoin Signals
        """
        self.modules_used.append("genius_act_macro")

        paths = []

        # 1. 확장 유동성 경로 (M = B + S*B*)
        ext_liq = ga_result.get('extended_liquidity', {})
        if ext_liq:
            base_b = ext_liq.get('base_liquidity', 0)
            stable_contrib = ext_liq.get('stablecoin_contribution', 0)
            total_m = ext_liq.get('total_extended_liquidity', 0)

            paths.append(MechanismPath(
                nodes=['Fed_BS', 'RRP', 'TGA', 'Net_Liquidity', 'Stablecoin_Supply', 'Extended_M'],
                edge_signs=['+', '-', '-', '+', '+'],
                net_effect=EdgeSign.POSITIVE,
                narrative=f"확장 유동성 모델: M={total_m/1e9:.0f}B = B({base_b/1e9:.0f}B) + S*B*({stable_contrib/1e9:.0f}B). "
                         f"Fed 자산에서 RRP/TGA 차감 후 스테이블코인 기여도 합산",
                strength=ConfidenceLevel.HIGH
            ))

        # 2. 스테이블코인 시그널 경로
        for signal in ga_result.get('signals', []):
            signal_type = signal.get('type', '')

            if 'STABLECOIN' in signal_type:
                direction = 'SURGE' in signal_type
                paths.append(MechanismPath(
                    nodes=['Stablecoin_Supply', 'Reserve_Requirement', 'Treasury_Demand'],
                    edge_signs=['+' if direction else '-', '+'],
                    net_effect=EdgeSign.POSITIVE if direction else EdgeSign.NEGATIVE,
                    narrative=f"스테이블코인 {'증가' if direction else '감소'} → "
                             f"준비금 수요 {'증가' if direction else '감소'} → "
                             f"국채 수요 영향 ({signal.get('description', '')})",
                    strength=ConfidenceLevel.MEDIUM
                ))

            elif 'LIQUIDITY' in signal_type:
                paths.append(MechanismPath(
                    nodes=['Fed_Policy', 'Net_Liquidity', 'Risk_Assets'],
                    edge_signs=['+' if 'INJECTION' in signal_type else '-', '+'],
                    net_effect=EdgeSign.POSITIVE if 'INJECTION' in signal_type else EdgeSign.NEGATIVE,
                    narrative=f"Fed 유동성 {'공급' if 'INJECTION' in signal_type else '흡수'} → "
                             f"순유동성 변화 → 위험자산 영향",
                    strength=ConfidenceLevel.HIGH
                ))

        # 3. 레짐 기반 경로
        regime = ga_result.get('regime', 'NEUTRAL')
        if regime == 'EXPANSION':
            paths.append(MechanismPath(
                nodes=['Liquidity_Expansion', 'Credit_Availability', 'Asset_Prices'],
                edge_signs=['+', '+'],
                net_effect=EdgeSign.POSITIVE,
                narrative="유동성 확장 레짐: 신용 가용성 증가 → 자산 가격 상승 압력",
                strength=ConfidenceLevel.HIGH
            ))
        elif regime == 'CONTRACTION':
            paths.append(MechanismPath(
                nodes=['Liquidity_Contraction', 'Credit_Tightening', 'Asset_Prices'],
                edge_signs=['-', '-'],
                net_effect=EdgeSign.NEGATIVE,
                narrative="유동성 축소 레짐: 신용 긴축 → 자산 가격 하락 압력",
                strength=ConfidenceLevel.HIGH
            ))

        return paths

    # =========================================================================
    # VolumeAnalyzer → Signals + Data Limitations
    # =========================================================================

    def adapt_volume_analyzer(self, va_result: Dict) -> tuple:
        """
        VolumeAnalyzer 결과를 분석

        Returns:
            (suggested_data: List[SuggestedDataset], limitations: List[DataLimitation])
        """
        self.modules_used.append("volume_analyzer")

        suggested = []
        limitations = []

        # 1. 이상 거래량 감지 시 추가 데이터 제안
        anomalies = va_result.get('anomalies', [])
        for anomaly in anomalies[:3]:
            ticker = anomaly.get('ticker')
            anomaly_type = anomaly.get('anomaly_type')

            if anomaly_type in ['ABNORMAL_SURGE', 'EXTREME_SURGE']:
                suggested.append(SuggestedDataset(
                    name=f"{ticker} 기관 거래 데이터",
                    category="flows",
                    priority=1,
                    rationale=f"{ticker} 비정상 거래량 탐지 ({anomaly.get('z_score', 0):.1f}σ) - 정보 비대칭 가능성",
                    source="13F filings / Bloomberg Terminal"
                ))

            if anomaly.get('information_type') == 'PRIVATE_INFO':
                limitations.append(DataLimitation(
                    description=f"{ticker} 내부자 거래 가능성",
                    impact="공개 정보만으로 원인 파악 불가",
                    mitigation="SEC Form 4 제출 모니터링"
                ))

        # 2. 데이터 커버리지 한계
        if va_result.get('coverage', {}).get('missing_tickers'):
            limitations.append(DataLimitation(
                description=f"거래량 데이터 미수집 종목: {len(va_result['coverage']['missing_tickers'])}개",
                impact="포트폴리오 전체 유동성 평가 불완전",
                mitigation="Bloomberg/Refinitiv 데이터 보완"
            ))

        return suggested, limitations

    # =========================================================================
    # BubbleDetector → RegimeShiftRisk
    # =========================================================================

    def adapt_bubble_detector(self, bd_result: Dict) -> List[RegimeShiftRisk]:
        """
        BubbleDetector 결과를 RegimeShiftRisk로 변환

        기존 모듈: lib/bubble_detector.py
        - Greenwood-Shleifer (2019) 방법론
        """
        self.modules_used.append("bubble_detector")

        risks = []

        overall_status = bd_result.get('overall_status', 'NONE')

        if overall_status in ['WARNING', 'DANGER']:
            risks.append(RegimeShiftRisk(
                description=f"버블 리스크 레벨: {overall_status}",
                trigger="Greenwood-Shleifer 2년 run-up > 100%",
                probability=0.7 if overall_status == 'DANGER' else 0.4,
                severity=RiskSeverity.CRITICAL if overall_status == 'DANGER' else RiskSeverity.HIGH,
                source_module="bubble_detector"
            ))

        # 개별 종목 리스크
        for ticker_risk in bd_result.get('risk_tickers', [])[:3]:
            ticker = ticker_risk.get('ticker')
            run_up = ticker_risk.get('run_up_pct', 0)

            if run_up > 100:
                risks.append(RegimeShiftRisk(
                    description=f"{ticker} 버블 신호: 2년 수익률 {run_up:.0f}%",
                    trigger=f"Run-up {run_up:.0f}% > 100% 임계값",
                    probability=min(run_up / 200, 0.9),
                    severity=RiskSeverity.HIGH,
                    source_module="bubble_detector"
                ))

        return risks

    # =========================================================================
    # GraphClusteredPortfolio → NextAction
    # =========================================================================

    def adapt_portfolio(self, gcp_result: Dict) -> List[NextAction]:
        """
        GraphClusteredPortfolio 결과를 NextAction으로 변환

        기존 모듈: lib/graph_clustered_portfolio.py
        """
        self.modules_used.append("graph_clustered_portfolio")

        actions = []

        # 1. 포트폴리오 리밸런싱 제안
        weights = gcp_result.get('weights', {})
        if weights:
            top_holdings = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
            actions.append(NextAction(
                description=f"HRP 포트폴리오 리밸런싱: 상위 5 비중 - {', '.join([f'{t}:{w:.1%}' for t, w in top_holdings])}",
                category="trade",
                priority=2,
                timeframe="주간"
            ))

        # 2. 시스템 리스크 노드 모니터링
        systemic_nodes = gcp_result.get('systemic_risk_nodes', [])
        if systemic_nodes:
            actions.append(NextAction(
                description=f"시스템 리스크 허브 모니터링: {', '.join([n.get('ticker', n) for n in systemic_nodes[:3]])}",
                category="monitor",
                priority=1,
                timeframe="일간"
            ))

        # 3. 클러스터 분산
        clusters = gcp_result.get('clusters', {})
        if len(clusters) < 3:
            actions.append(NextAction(
                description="포트폴리오 클러스터 분산 부족 - 새 자산군 탐색 필요",
                category="analysis",
                priority=3,
                timeframe="월간"
            ))

        return actions

    # =========================================================================
    # Hypothesis Generation (신규 로직)
    # =========================================================================

    def generate_hypotheses(
        self,
        phenomenon: str,
        causal_graph: CausalGraph,
        mechanisms: List[MechanismPath]
    ) -> HypothesesSection:
        """
        현상과 인과 그래프를 바탕으로 가설 생성

        주요 가설 + 대안 가설 + 반증 테스트
        """
        # 1. 주요 가설 (최강 경로 기반)
        main_path = mechanisms[0] if mechanisms else None
        main_hypothesis = Hypothesis(
            statement=f"{phenomenon}은(는) {main_path.narrative.split('→')[0] if main_path else '다양한 요인'}에 의해 주도된다",
            supporting_evidence=[
                f"Mechanism Path: {' → '.join(main_path.nodes)}" if main_path else "Unknown",
                f"Net Effect: {main_path.net_effect.value if main_path else '?'}",
                f"Confidence: {main_path.strength.value if main_path else 'low'}"
            ],
            confidence=ConfidenceLevel.HIGH if main_path and main_path.strength == ConfidenceLevel.HIGH else ConfidenceLevel.MEDIUM
        )

        # 2. 대안 가설들
        rivals = []
        for i, path in enumerate(mechanisms[1:4], 1):
            rivals.append(Hypothesis(
                statement=f"대안 {i}: {path.narrative.split('.')[0]}",
                supporting_evidence=[f"Path: {' → '.join(path.nodes[:3])}..."],
                confidence=path.strength
            ))

        # 추가 대안: 외부 요인
        if causal_graph.has_cycles:
            rivals.append(Hypothesis(
                statement="피드백 루프로 인해 단일 원인 특정이 어려움",
                supporting_evidence=["그래프에 사이클 존재"],
                confidence=ConfidenceLevel.LOW
            ))

        # 3. 반증 테스트
        falsification_tests = []

        # 주요 경로 반증
        if main_path:
            first_node = main_path.nodes[0] if main_path.nodes else "원인"
            last_node = main_path.nodes[-1] if main_path.nodes else "결과"

            falsification_tests.append(FalsificationTest(
                description=f"{first_node} 변화 시 {last_node} 반응 확인",
                data_required=[first_node, last_node, "control_variables"],
                expected_if_true=f"{first_node} {main_path.edge_signs[0] if main_path.edge_signs else '+'} 변화 → {last_node} 동방향 변화",
                expected_if_false=f"{last_node} 무반응 또는 역방향 반응"
            ))

        # Granger 테스트
        falsification_tests.append(FalsificationTest(
            description="Granger Causality 역방향 테스트",
            data_required=["time_series_data", "lag_structure"],
            expected_if_true="원인 → 결과 방향만 유의",
            expected_if_false="역방향도 유의하거나 둘 다 무의미"
        ))

        return HypothesesSection(
            main=main_hypothesis,
            rivals=rivals,
            falsification_tests=falsification_tests
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _format_node_name(self, node_id: str) -> str:
        """노드 ID를 읽기 쉬운 이름으로 변환"""
        return node_id.replace('_', ' ').title()

    def _infer_category(self, node_id: str) -> str:
        """노드 ID에서 카테고리 추론"""
        node_lower = node_id.lower()

        if any(x in node_lower for x in ['fed', 'treasury', 'gdp', 'cpi', 'inflation', 'employment']):
            return 'macro'
        elif any(x in node_lower for x in ['spy', 'qqq', 'vix', 'bond', 'equity', 'yield']):
            return 'market'
        elif any(x in node_lower for x in ['btc', 'eth', 'stable', 'usdc', 'usdt', 'defi', 'crypto']):
            return 'crypto'
        elif any(x in node_lower for x in ['tech', 'energy', 'health', 'financial', 'sector']):
            return 'sector'
        else:
            return 'other'

    def _infer_sign(self, granger_result: Dict) -> EdgeSign:
        """Granger 결과에서 부호 추론"""
        coef = granger_result.get('coefficient', granger_result.get('correlation', 0))
        if coef > 0:
            return EdgeSign.POSITIVE
        elif coef < 0:
            return EdgeSign.NEGATIVE
        else:
            return EdgeSign.AMBIGUOUS

    def _p_to_confidence(self, p_value: float) -> ConfidenceLevel:
        """p-value를 신뢰도로 변환"""
        if p_value < 0.01:
            return ConfidenceLevel.HIGH
        elif p_value < 0.05:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _detect_cycles(self, edges: List[CausalEdge]) -> bool:
        """엣지 리스트에서 사이클 감지 (DFS)"""
        # 인접 리스트 생성
        adj = {}
        for edge in edges:
            if edge.source not in adj:
                adj[edge.source] = []
            adj[edge.source].append(edge.target)

        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in adj.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in adj:
            if node not in visited:
                if dfs(node):
                    return True

        return False

    def detect_frame(self, question: str, context: Dict = None) -> AnalysisFrame:
        """질문에서 분석 프레임 자동 감지"""
        q_lower = question.lower()

        crypto_keywords = ['스테이블', 'stable', 'crypto', 'btc', 'eth', 'defi', '암호화폐', '코인']
        macro_keywords = ['fed', '금리', 'rate', 'inflation', '인플레', 'gdp', '고용', 'employment']
        market_keywords = ['spy', '주식', 'stock', 'equity', 'bond', '채권', 'yield']

        crypto_score = sum(1 for kw in crypto_keywords if kw in q_lower)
        macro_score = sum(1 for kw in macro_keywords if kw in q_lower)
        market_score = sum(1 for kw in market_keywords if kw in q_lower)

        if crypto_score > 0 and (macro_score > 0 or market_score > 0):
            return AnalysisFrame.MIXED
        elif crypto_score > max(macro_score, market_score):
            return AnalysisFrame.CRYPTO
        elif macro_score > market_score:
            return AnalysisFrame.MACRO
        elif market_score > 0:
            return AnalysisFrame.MARKETS
        else:
            return AnalysisFrame.MIXED

    def create_meta(self, request_id: str, frame: AnalysisFrame) -> InsightMeta:
        """메타 정보 생성"""
        return InsightMeta(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            frame=frame,
            modules_used=list(set(self.modules_used))
        )


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    adapter = EIMASAdapter()

    # Mock ShockPropagation result
    mock_spg = {
        'node_analysis': {
            'Fed_Funds': {'layer': 'POLICY', 'centrality_score': 0.8},
            'Net_Liquidity': {'layer': 'LIQUIDITY', 'centrality_score': 0.6},
            'SPY': {'layer': 'ASSET_PRICE', 'centrality_score': 0.4}
        },
        'granger_results': [
            {'cause': 'Fed_Funds', 'effect': 'Net_Liquidity', 'p_value': 0.01, 'optimal_lag': 5, 'is_significant': True},
            {'cause': 'Net_Liquidity', 'effect': 'SPY', 'p_value': 0.03, 'optimal_lag': 3, 'is_significant': True}
        ],
        'shock_paths': [{'path': ['Fed_Funds', 'Net_Liquidity', 'SPY']}]
    }

    graph = adapter.adapt_shock_propagation(mock_spg)
    print("Causal Graph:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Critical Path: {graph.critical_path}")
    print(f"  Has Cycles: {graph.has_cycles}")

    # Mock CriticalPath result
    mock_cp = {
        'current_regime': 'BULL',
        'transition_probability': 0.45,
        'total_risk_score': 55.0,
        'active_warnings': ['VIX elevated', 'Credit spread widening'],
        'path_contributions': {'liquidity': 35, 'credit': 20}
    }

    risks = adapter.adapt_critical_path(mock_cp)
    print(f"\nRegime Risks: {len(risks)}")
    for r in risks:
        print(f"  - {r.description} ({r.severity.value})")

    print(f"\nModules Used: {adapter.modules_used}")
