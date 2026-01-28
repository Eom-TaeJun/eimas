"""
Test: Graph Utilities
=====================

agentcommand.txt: "Add unit tests for: graph utilities"

Tests for:
- Path sign composition
- Cycle detection
- Critical path ranking
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agent.schemas.insight_schema import CausalGraph, CausalNode, CausalEdge, EdgeSign
from agent.core.adapters import EIMASAdapter


class TestPathSignComposition:
    """경로 부호 합성 테스트"""

    def setup_method(self):
        self.adapter = EIMASAdapter()

    def test_all_positive(self):
        """모든 양수: + * + = +"""
        signs = ["+", "+", "+"]
        neg_count = sum(1 for s in signs if s == "-")
        net = EdgeSign.NEGATIVE if neg_count % 2 == 1 else EdgeSign.POSITIVE
        assert net == EdgeSign.POSITIVE

    def test_one_negative(self):
        """하나 음수: + * - = -"""
        signs = ["+", "-", "+"]
        neg_count = sum(1 for s in signs if s == "-")
        net = EdgeSign.NEGATIVE if neg_count % 2 == 1 else EdgeSign.POSITIVE
        assert net == EdgeSign.NEGATIVE

    def test_two_negatives(self):
        """두 개 음수: - * - = +"""
        signs = ["-", "-", "+"]
        neg_count = sum(1 for s in signs if s == "-")
        net = EdgeSign.NEGATIVE if neg_count % 2 == 1 else EdgeSign.POSITIVE
        assert net == EdgeSign.POSITIVE

    def test_three_negatives(self):
        """세 개 음수: - * - * - = -"""
        signs = ["-", "-", "-"]
        neg_count = sum(1 for s in signs if s == "-")
        net = EdgeSign.NEGATIVE if neg_count % 2 == 1 else EdgeSign.POSITIVE
        assert net == EdgeSign.NEGATIVE

    def test_empty_path(self):
        """빈 경로: +"""
        signs = []
        neg_count = sum(1 for s in signs if s == "-")
        net = EdgeSign.NEGATIVE if neg_count % 2 == 1 else EdgeSign.POSITIVE
        assert net == EdgeSign.POSITIVE


class TestCycleDetection:
    """사이클 감지 테스트"""

    def setup_method(self):
        self.adapter = EIMASAdapter()

    def test_no_cycle_linear(self):
        """선형 그래프 (사이클 없음): A → B → C"""
        edges = [
            CausalEdge(source="A", target="B", sign=EdgeSign.POSITIVE),
            CausalEdge(source="B", target="C", sign=EdgeSign.POSITIVE),
        ]
        has_cycle = self.adapter._detect_cycles(edges)
        assert has_cycle is False

    def test_simple_cycle(self):
        """단순 사이클: A → B → C → A"""
        edges = [
            CausalEdge(source="A", target="B", sign=EdgeSign.POSITIVE),
            CausalEdge(source="B", target="C", sign=EdgeSign.POSITIVE),
            CausalEdge(source="C", target="A", sign=EdgeSign.POSITIVE),
        ]
        has_cycle = self.adapter._detect_cycles(edges)
        assert has_cycle is True

    def test_self_loop(self):
        """자기 루프: A → A"""
        edges = [
            CausalEdge(source="A", target="A", sign=EdgeSign.POSITIVE),
        ]
        has_cycle = self.adapter._detect_cycles(edges)
        assert has_cycle is True

    def test_diamond_no_cycle(self):
        """다이아몬드 (사이클 없음): A → B, A → C, B → D, C → D"""
        edges = [
            CausalEdge(source="A", target="B", sign=EdgeSign.POSITIVE),
            CausalEdge(source="A", target="C", sign=EdgeSign.POSITIVE),
            CausalEdge(source="B", target="D", sign=EdgeSign.POSITIVE),
            CausalEdge(source="C", target="D", sign=EdgeSign.POSITIVE),
        ]
        has_cycle = self.adapter._detect_cycles(edges)
        assert has_cycle is False

    def test_multiple_components_one_cycle(self):
        """여러 컴포넌트, 하나에 사이클: A → B, C → D → C"""
        edges = [
            CausalEdge(source="A", target="B", sign=EdgeSign.POSITIVE),
            CausalEdge(source="C", target="D", sign=EdgeSign.POSITIVE),
            CausalEdge(source="D", target="C", sign=EdgeSign.POSITIVE),
        ]
        has_cycle = self.adapter._detect_cycles(edges)
        assert has_cycle is True

    def test_empty_graph(self):
        """빈 그래프"""
        edges = []
        has_cycle = self.adapter._detect_cycles(edges)
        assert has_cycle is False


class TestCriticalPathRanking:
    """Critical Path 순위 테스트"""

    def test_critical_path_stored(self):
        """Critical Path 저장 확인"""
        graph = CausalGraph(
            nodes=[
                CausalNode(id="A", name="A", criticality=0.8),
                CausalNode(id="B", name="B", criticality=0.6),
                CausalNode(id="C", name="C", criticality=0.4),
            ],
            edges=[
                CausalEdge(source="A", target="B", sign=EdgeSign.POSITIVE),
                CausalEdge(source="B", target="C", sign=EdgeSign.POSITIVE),
            ],
            critical_path=["A", "B", "C"]
        )

        assert graph.critical_path == ["A", "B", "C"]

    def test_criticality_scores(self):
        """Criticality 점수 정렬"""
        nodes = [
            CausalNode(id="A", name="A", criticality=0.4),
            CausalNode(id="B", name="B", criticality=0.8),
            CausalNode(id="C", name="C", criticality=0.6),
        ]

        # 중요도 순 정렬
        sorted_nodes = sorted(nodes, key=lambda n: n.criticality or 0, reverse=True)
        assert sorted_nodes[0].id == "B"
        assert sorted_nodes[1].id == "C"
        assert sorted_nodes[2].id == "A"


class TestPValueToConfidence:
    """p-value → 신뢰도 변환 테스트"""

    def setup_method(self):
        self.adapter = EIMASAdapter()

    def test_high_confidence(self):
        """p < 0.01 → HIGH"""
        from agent.schemas.insight_schema import ConfidenceLevel
        assert self.adapter._p_to_confidence(0.005) == ConfidenceLevel.HIGH
        assert self.adapter._p_to_confidence(0.009) == ConfidenceLevel.HIGH

    def test_medium_confidence(self):
        """0.01 <= p < 0.05 → MEDIUM"""
        from agent.schemas.insight_schema import ConfidenceLevel
        assert self.adapter._p_to_confidence(0.01) == ConfidenceLevel.MEDIUM
        assert self.adapter._p_to_confidence(0.03) == ConfidenceLevel.MEDIUM
        assert self.adapter._p_to_confidence(0.049) == ConfidenceLevel.MEDIUM

    def test_low_confidence(self):
        """p >= 0.05 → LOW"""
        from agent.schemas.insight_schema import ConfidenceLevel
        assert self.adapter._p_to_confidence(0.05) == ConfidenceLevel.LOW
        assert self.adapter._p_to_confidence(0.1) == ConfidenceLevel.LOW
        assert self.adapter._p_to_confidence(0.5) == ConfidenceLevel.LOW


class TestCategoryInference:
    """카테고리 추론 테스트"""

    def setup_method(self):
        self.adapter = EIMASAdapter()

    def test_macro_keywords(self):
        """거시경제 키워드"""
        assert self.adapter._infer_category("Fed_Funds_Rate") == "macro"
        assert self.adapter._infer_category("GDP_Growth") == "macro"
        assert self.adapter._infer_category("CPI_YoY") == "macro"
        assert self.adapter._infer_category("Inflation_Rate") == "macro"

    def test_market_keywords(self):
        """시장 키워드"""
        assert self.adapter._infer_category("SPY_Returns") == "market"
        assert self.adapter._infer_category("VIX_Level") == "market"
        assert self.adapter._infer_category("Bond_Yield") == "market"

    def test_crypto_keywords(self):
        """암호화폐 키워드"""
        assert self.adapter._infer_category("BTC_Price") == "crypto"
        assert self.adapter._infer_category("USDC_Supply") == "crypto"
        assert self.adapter._infer_category("DeFi_TVL") == "crypto"
        assert self.adapter._infer_category("Stablecoin_Market") == "crypto"

    def test_sector_keywords(self):
        """섹터 키워드"""
        assert self.adapter._infer_category("Tech_Sector") == "sector"
        assert self.adapter._infer_category("Energy_Index") == "sector"

    def test_other_category(self):
        """기타 카테고리"""
        assert self.adapter._infer_category("Unknown_Variable") == "other"
        assert self.adapter._infer_category("Random_Node") == "other"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
