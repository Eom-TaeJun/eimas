"""
Test: Orchestrator
==================

Integration tests for EconomicInsightOrchestrator
"""

import pytest

from agent.schemas.insight_schema import (
    InsightRequest, EconomicInsightReport, AnalysisFrame
)
from agent.core.orchestrator import EconomicInsightOrchestrator


class TestTemplateBasedAnalysis:
    """템플릿 기반 분석 테스트"""

    def setup_method(self):
        self.orchestrator = EconomicInsightOrchestrator(debug=True)

    def test_macro_question(self):
        """거시경제 질문"""
        request = InsightRequest(
            question="Fed 금리 인상이 시장에 미치는 영향은?"
        )
        report = self.orchestrator.run(request)

        assert report.meta.frame == AnalysisFrame.MACRO
        assert len(report.causal_graph.nodes) > 0
        assert len(report.mechanisms) >= 1
        assert len(report.next_actions) >= 3

    def test_crypto_question(self):
        """암호화폐 질문"""
        request = InsightRequest(
            question="스테이블코인 공급 증가가 BTC 가격에 미치는 영향은?"
        )
        report = self.orchestrator.run(request)

        assert report.meta.frame == AnalysisFrame.CRYPTO
        assert len(report.causal_graph.nodes) > 0

    def test_market_question(self):
        """시장 질문"""
        request = InsightRequest(
            question="VIX 급등이 주식 가격에 미치는 영향은?"
        )
        report = self.orchestrator.run(request)

        assert report.meta.frame == AnalysisFrame.MARKETS

    def test_mixed_question(self):
        """복합 질문"""
        request = InsightRequest(
            question="Fed 정책이 비트코인과 주식시장에 동시에 미치는 영향은?"
        )
        report = self.orchestrator.run(request)

        assert report.meta.frame == AnalysisFrame.MIXED

    def test_frame_hint_override(self):
        """frame_hint로 프레임 지정"""
        request = InsightRequest(
            question="일반적인 시장 분석",
            frame_hint=AnalysisFrame.CRYPTO
        )
        report = self.orchestrator.run(request)

        # frame_hint가 우선
        assert report.meta.frame == AnalysisFrame.CRYPTO

    def test_report_json_valid(self):
        """리포트 JSON 유효성"""
        request = InsightRequest(question="테스트 질문")
        report = self.orchestrator.run(request)

        # JSON 직렬화 가능해야 함
        json_str = report.model_dump_json()
        assert len(json_str) > 0
        assert '"meta"' in json_str
        assert '"phenomenon"' in json_str
        assert '"causal_graph"' in json_str

    def test_processing_time_recorded(self):
        """처리 시간 기록"""
        request = InsightRequest(question="처리 시간 테스트")
        report = self.orchestrator.run(request)

        assert report.meta.processing_time_ms is not None
        assert report.meta.processing_time_ms >= 0


class TestEIMASIntegration:
    """EIMAS 모듈 통합 테스트"""

    def setup_method(self):
        self.orchestrator = EconomicInsightOrchestrator(debug=True)

    def test_with_shock_propagation(self):
        """ShockPropagation 결과 통합"""
        request = InsightRequest(question="충격 전파 분석")

        mock_eimas = {
            'shock_propagation': {
                'node_analysis': {
                    'Fed_Funds': {'layer': 'POLICY', 'centrality_score': 0.8},
                    'SPY': {'layer': 'ASSET_PRICE', 'centrality_score': 0.4}
                },
                'granger_results': [
                    {'cause': 'Fed_Funds', 'effect': 'SPY', 'p_value': 0.02,
                     'optimal_lag': 3, 'is_significant': True, 'coefficient': -0.3}
                ],
                'shock_paths': [{'path': ['Fed_Funds', 'SPY'], 'strength': 0.7}]
            }
        }

        report = self.orchestrator.run_with_eimas_results(request, mock_eimas)

        assert "shock_propagation_graph" in report.meta.modules_used
        assert len(report.causal_graph.nodes) >= 2
        assert len(report.causal_graph.edges) >= 1

    def test_with_critical_path(self):
        """CriticalPath 결과 통합"""
        request = InsightRequest(question="리스크 분석")

        mock_eimas = {
            'critical_path': {
                'current_regime': 'BULL',
                'transition_probability': 0.45,
                'total_risk_score': 55.0,
                'active_warnings': ['VIX 상승세', 'Credit spread 확대'],
                'path_contributions': {'liquidity': 35}
            }
        }

        report = self.orchestrator.run_with_eimas_results(request, mock_eimas)

        # 리스크 추출 확인
        assert len(report.risk.regime_shift_risks) > 0

    def test_with_genius_act(self):
        """GeniusAct 결과 통합"""
        request = InsightRequest(question="유동성 분석")

        mock_eimas = {
            'genius_act': {
                'regime': 'EXPANSION',
                'extended_liquidity': {
                    'base_liquidity': 5800000000000,
                    'stablecoin_contribution': 150000000000,
                    'total_extended_liquidity': 5950000000000
                },
                'signals': [{'type': 'LIQUIDITY_INJECTION', 'description': 'Test'}]
            }
        }

        report = self.orchestrator.run_with_eimas_results(request, mock_eimas)

        assert "genius_act_macro" in report.meta.modules_used
        assert len(report.mechanisms) >= 1

    def test_with_bubble_detector(self):
        """BubbleDetector 결과 통합"""
        request = InsightRequest(question="버블 분석")

        mock_eimas = {
            'bubble_detector': {
                'overall_status': 'WARNING',
                'risk_tickers': [
                    {'ticker': 'NVDA', 'run_up_pct': 150}
                ]
            }
        }

        report = self.orchestrator.run_with_eimas_results(request, mock_eimas)

        # 버블 리스크 추출
        bubble_risks = [r for r in report.risk.regime_shift_risks
                       if 'bubble_detector' in (r.source_module or '')]
        assert len(bubble_risks) > 0

    def test_combined_eimas_results(self):
        """여러 EIMAS 모듈 결합"""
        request = InsightRequest(question="종합 분석")

        mock_eimas = {
            'shock_propagation': {
                'node_analysis': {'A': {'layer': 'TEST'}},
                'granger_results': [],
                'shock_paths': []
            },
            'critical_path': {
                'current_regime': 'NEUTRAL',
                'transition_probability': 0.2,
                'total_risk_score': 30.0,
                'active_warnings': []
            },
            'genius_act': {
                'regime': 'NEUTRAL',
                'signals': []
            }
        }

        report = self.orchestrator.run_with_eimas_results(request, mock_eimas)

        # 여러 모듈 사용 확인
        assert len(report.meta.modules_used) >= 2


class TestOutputRequirements:
    """agentcommand.txt 출력 요구사항 테스트"""

    def setup_method(self):
        self.orchestrator = EconomicInsightOrchestrator()

    def test_required_top_level_keys(self):
        """필수 top-level 키 확인"""
        request = InsightRequest(question="테스트")
        report = self.orchestrator.run(request)
        dump = report.model_dump()

        required = ['meta', 'phenomenon', 'causal_graph', 'mechanisms',
                   'hypotheses', 'risk', 'suggested_data', 'next_actions']

        for key in required:
            assert key in dump, f"Missing required key: {key}"

    def test_meta_structure(self):
        """meta 구조 확인"""
        request = InsightRequest(question="테스트")
        report = self.orchestrator.run(request)

        assert report.meta.request_id
        assert report.meta.timestamp
        assert report.meta.frame in AnalysisFrame

    def test_causal_graph_structure(self):
        """causal_graph 구조 확인"""
        request = InsightRequest(question="Fed 정책 영향", frame_hint=AnalysisFrame.MACRO)
        report = self.orchestrator.run(request)

        assert hasattr(report.causal_graph, 'nodes')
        assert hasattr(report.causal_graph, 'edges')
        assert isinstance(report.causal_graph.nodes, list)
        assert isinstance(report.causal_graph.edges, list)

    def test_mechanisms_structure(self):
        """mechanisms 구조 확인"""
        request = InsightRequest(question="테스트")
        report = self.orchestrator.run(request)

        assert 1 <= len(report.mechanisms) <= 5

        for mech in report.mechanisms:
            assert hasattr(mech, 'nodes')
            assert hasattr(mech, 'edge_signs')
            assert hasattr(mech, 'narrative')

    def test_hypotheses_structure(self):
        """hypotheses 구조 확인"""
        request = InsightRequest(question="테스트")
        report = self.orchestrator.run(request)

        assert report.hypotheses.main is not None
        assert report.hypotheses.main.statement
        assert isinstance(report.hypotheses.rivals, list)
        assert isinstance(report.hypotheses.falsification_tests, list)

    def test_next_actions_count(self):
        """next_actions 개수 (3-7)"""
        request = InsightRequest(question="테스트")
        report = self.orchestrator.run(request)

        assert 3 <= len(report.next_actions) <= 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
