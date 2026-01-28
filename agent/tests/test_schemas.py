"""
Test: Schema Validity
=====================

agentcommand.txt: "Add unit tests for: schema validity"
"""

import pytest
import json
from pydantic import ValidationError

from agent.schemas.insight_schema import (
    InsightRequest, EconomicInsightReport, InsightMeta, AnalysisFrame,
    CausalGraph, CausalNode, CausalEdge, EdgeSign, ConfidenceLevel,
    MechanismPath, HypothesesSection, Hypothesis, FalsificationTest,
    RiskSection, RegimeShiftRisk, DataLimitation, RiskSeverity,
    SuggestedDataset, NextAction
)


class TestInsightRequest:
    """InsightRequest 스키마 테스트"""

    def test_minimal_request(self):
        """최소 필수 필드만으로 생성"""
        request = InsightRequest(question="Fed 금리 영향?")
        assert request.question == "Fed 금리 영향?"
        assert request.request_id is not None  # 자동 생성
        assert request.frame_hint is None
        assert request.context is None

    def test_full_request(self):
        """모든 필드로 생성"""
        request = InsightRequest(
            request_id="test-123",
            question="스테이블코인 공급 증가 영향?",
            frame_hint=AnalysisFrame.CRYPTO,
            context={"usdc_change": "+5%"}
        )
        assert request.request_id == "test-123"
        assert request.frame_hint == AnalysisFrame.CRYPTO

    def test_missing_question_fails(self):
        """question 필수 검증"""
        with pytest.raises(ValidationError):
            InsightRequest()

    def test_json_serialization(self):
        """JSON 직렬화 테스트"""
        request = InsightRequest(question="테스트")
        json_str = request.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["question"] == "테스트"


class TestCausalGraph:
    """CausalGraph 스키마 테스트"""

    def test_empty_graph(self):
        """빈 그래프 생성 가능"""
        graph = CausalGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.has_cycles is False

    def test_node_creation(self):
        """노드 생성"""
        node = CausalNode(
            id="fed_funds",
            name="Fed Funds Rate",
            layer="POLICY",
            category="macro",
            centrality=0.8
        )
        assert node.id == "fed_funds"
        assert node.centrality == 0.8

    def test_edge_creation(self):
        """엣지 생성"""
        edge = CausalEdge(
            source="fed_funds",
            target="net_liquidity",
            sign=EdgeSign.NEGATIVE,
            lag=5,
            p_value=0.01,
            confidence=ConfidenceLevel.HIGH,
            mechanism="금리 인상 → 유동성 감소"
        )
        assert edge.sign == EdgeSign.NEGATIVE
        assert edge.lag == 5

    def test_graph_with_nodes_and_edges(self):
        """노드와 엣지를 포함한 그래프"""
        graph = CausalGraph(
            nodes=[
                CausalNode(id="A", name="Node A"),
                CausalNode(id="B", name="Node B"),
            ],
            edges=[
                CausalEdge(source="A", target="B", sign=EdgeSign.POSITIVE),
            ],
            has_cycles=False,
            critical_path=["A", "B"]
        )
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.critical_path == ["A", "B"]


class TestMechanismPath:
    """MechanismPath 스키마 테스트"""

    def test_valid_mechanism(self):
        """유효한 메커니즘"""
        path = MechanismPath(
            nodes=["Fed", "Liquidity", "SPY"],
            edge_signs=["-", "+"],
            net_effect=EdgeSign.NEGATIVE,
            narrative="Fed 긴축 → 유동성 감소 → 주가 하락"
        )
        assert len(path.nodes) == 3
        assert len(path.edge_signs) == 2
        assert path.net_effect == EdgeSign.NEGATIVE

    def test_path_id_auto_generated(self):
        """path_id 자동 생성"""
        path1 = MechanismPath(nodes=["A"], edge_signs=[], net_effect=EdgeSign.POSITIVE, narrative="Test")
        path2 = MechanismPath(nodes=["B"], edge_signs=[], net_effect=EdgeSign.POSITIVE, narrative="Test")
        assert path1.path_id != path2.path_id


class TestHypotheses:
    """Hypotheses 스키마 테스트"""

    def test_hypothesis_creation(self):
        """가설 생성"""
        hyp = Hypothesis(
            statement="Fed 긴축이 주가 하락을 유발한다",
            supporting_evidence=["Granger Causality 유의"],
            confidence=ConfidenceLevel.HIGH
        )
        assert hyp.statement
        assert len(hyp.supporting_evidence) == 1

    def test_falsification_test(self):
        """반증 테스트"""
        test = FalsificationTest(
            description="Fed 긴축 시 주가 상승 확인",
            data_required=["fed_funds", "spy_returns"],
            expected_if_true="음의 상관관계",
            expected_if_false="양의 상관관계 또는 무상관"
        )
        assert len(test.data_required) == 2

    def test_hypotheses_section(self):
        """가설 섹션"""
        section = HypothesesSection(
            main=Hypothesis(statement="Main hypothesis", supporting_evidence=[], confidence=ConfidenceLevel.MEDIUM),
            rivals=[
                Hypothesis(statement="Rival 1", supporting_evidence=[], confidence=ConfidenceLevel.LOW),
            ],
            falsification_tests=[]
        )
        assert section.main.statement == "Main hypothesis"
        assert len(section.rivals) == 1


class TestRiskSection:
    """RiskSection 스키마 테스트"""

    def test_regime_shift_risk(self):
        """레짐 변화 리스크"""
        risk = RegimeShiftRisk(
            description="Bull → Bear 전환 가능성",
            trigger="VIX 40 초과",
            probability=0.4,
            severity=RiskSeverity.HIGH
        )
        assert risk.probability == 0.4
        assert risk.severity == RiskSeverity.HIGH

    def test_data_limitation(self):
        """데이터 한계"""
        limit = DataLimitation(
            description="실시간 데이터 미지원",
            impact="지연 분석만 가능",
            mitigation="Bloomberg Terminal 연동"
        )
        assert limit.mitigation is not None


class TestActions:
    """NextAction, SuggestedDataset 테스트"""

    def test_next_action(self):
        """다음 행동"""
        action = NextAction(
            description="FOMC 회의록 분석",
            category="analysis",
            priority=1,
            timeframe="주간"
        )
        assert action.priority == 1

    def test_priority_bounds(self):
        """우선순위 범위 검증"""
        with pytest.raises(ValidationError):
            NextAction(description="Test", category="test", priority=0)  # min 1

        with pytest.raises(ValidationError):
            NextAction(description="Test", category="test", priority=6)  # max 5

    def test_suggested_dataset(self):
        """추천 데이터셋"""
        data = SuggestedDataset(
            name="FRED Fed Balance Sheet",
            category="macro",
            priority=1,
            rationale="Fed 유동성 추적",
            source="FRED API"
        )
        assert data.category == "macro"


class TestEconomicInsightReport:
    """EconomicInsightReport 최종 스키마 테스트"""

    def test_full_report_creation(self):
        """전체 리포트 생성"""
        report = EconomicInsightReport(
            meta=InsightMeta(
                request_id="test-1",
                frame=AnalysisFrame.MACRO,
                modules_used=["test"]
            ),
            phenomenon="Fed 금리 인상으로 시장 변동성 증가",
            causal_graph=CausalGraph(
                nodes=[CausalNode(id="A", name="A")],
                edges=[]
            ),
            mechanisms=[
                MechanismPath(nodes=["A"], edge_signs=[], net_effect=EdgeSign.POSITIVE, narrative="Test")
            ],
            hypotheses=HypothesesSection(
                main=Hypothesis(statement="Test", supporting_evidence=[], confidence=ConfidenceLevel.MEDIUM),
                rivals=[],
                falsification_tests=[]
            ),
            risk=RiskSection(regime_shift_risks=[], data_limitations=[]),
            suggested_data=[
                SuggestedDataset(name="Test", category="macro", priority=1, rationale="Test")
            ],
            next_actions=[
                NextAction(description="Action 1", category="monitor", priority=1),
                NextAction(description="Action 2", category="analysis", priority=2),
                NextAction(description="Action 3", category="data", priority=3),
            ]
        )

        # Required keys 검증
        dump = report.model_dump()
        required_keys = ['meta', 'phenomenon', 'causal_graph', 'mechanisms',
                        'hypotheses', 'risk', 'suggested_data', 'next_actions']
        for key in required_keys:
            assert key in dump

    def test_min_mechanisms(self):
        """최소 1개 메커니즘 필요"""
        with pytest.raises(ValidationError):
            EconomicInsightReport(
                meta=InsightMeta(request_id="test", frame=AnalysisFrame.MACRO),
                phenomenon="Test",
                causal_graph=CausalGraph(),
                mechanisms=[],  # empty - should fail
                hypotheses=HypothesesSection(
                    main=Hypothesis(statement="Test", supporting_evidence=[], confidence=ConfidenceLevel.LOW),
                    rivals=[],
                    falsification_tests=[]
                ),
                risk=RiskSection(regime_shift_risks=[], data_limitations=[]),
                suggested_data=[],
                next_actions=[
                    NextAction(description="1", category="a", priority=1),
                    NextAction(description="2", category="a", priority=2),
                    NextAction(description="3", category="a", priority=3),
                ]
            )

    def test_next_actions_bounds(self):
        """next_actions 개수 제한 (3-7)"""
        base_kwargs = dict(
            meta=InsightMeta(request_id="test", frame=AnalysisFrame.MACRO),
            phenomenon="Test",
            causal_graph=CausalGraph(),
            mechanisms=[MechanismPath(nodes=["A"], edge_signs=[], net_effect=EdgeSign.POSITIVE, narrative="Test")],
            hypotheses=HypothesesSection(
                main=Hypothesis(statement="Test", supporting_evidence=[], confidence=ConfidenceLevel.LOW),
                rivals=[],
                falsification_tests=[]
            ),
            risk=RiskSection(regime_shift_risks=[], data_limitations=[]),
            suggested_data=[]
        )

        # Too few (2)
        with pytest.raises(ValidationError):
            EconomicInsightReport(
                **base_kwargs,
                next_actions=[
                    NextAction(description="1", category="a", priority=1),
                    NextAction(description="2", category="a", priority=2),
                ]
            )

        # Too many (8)
        with pytest.raises(ValidationError):
            EconomicInsightReport(
                **base_kwargs,
                next_actions=[NextAction(description=str(i), category="a", priority=1) for i in range(8)]
            )

    def test_json_roundtrip(self):
        """JSON 직렬화/역직렬화"""
        report = EconomicInsightReport(
            meta=InsightMeta(request_id="test", frame=AnalysisFrame.CRYPTO, modules_used=["test"]),
            phenomenon="Test phenomenon",
            causal_graph=CausalGraph(nodes=[CausalNode(id="A", name="A")], edges=[]),
            mechanisms=[MechanismPath(nodes=["A"], edge_signs=[], net_effect=EdgeSign.POSITIVE, narrative="Test")],
            hypotheses=HypothesesSection(
                main=Hypothesis(statement="Main", supporting_evidence=[], confidence=ConfidenceLevel.HIGH),
                rivals=[],
                falsification_tests=[]
            ),
            risk=RiskSection(regime_shift_risks=[], data_limitations=[]),
            suggested_data=[SuggestedDataset(name="D", category="macro", priority=1, rationale="R")],
            next_actions=[
                NextAction(description="1", category="a", priority=1),
                NextAction(description="2", category="a", priority=2),
                NextAction(description="3", category="a", priority=3),
            ]
        )

        json_str = report.model_dump_json()
        parsed = json.loads(json_str)

        # Roundtrip
        report2 = EconomicInsightReport(**parsed)
        assert report2.meta.request_id == report.meta.request_id
        assert report2.phenomenon == report.phenomenon


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
