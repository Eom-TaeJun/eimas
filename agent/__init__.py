"""
Economic Insight Agent
======================

agentcommand.txt 기반 인과적, 설명 가능한 경제 분석 에이전트

기존 EIMAS 모듈과 통합하여 JSON-first 출력 생성

Usage:
    # 템플릿 기반 분석
    from agent import EconomicInsightOrchestrator, InsightRequest

    orchestrator = EconomicInsightOrchestrator()
    request = InsightRequest(question="스테이블코인 공급 증가가 국채 수요에 미치는 영향은?")
    report = orchestrator.run(request)
    print(report.model_dump_json(indent=2))

    # EIMAS 결과 활용
    eimas_results = {...}  # main.py 실행 결과
    report = orchestrator.run_with_eimas_results(request, eimas_results)

CLI:
    python -m agent.cli --question "Fed 금리 인상 영향은?"
    python -m agent.cli --with-eimas --question "현재 시장 분석"
"""

from agent.schemas.insight_schema import (
    # Request
    InsightRequest,
    AnalysisFrame,

    # Output
    EconomicInsightReport,
    InsightMeta,

    # Graph
    CausalGraph,
    CausalNode,
    CausalEdge,
    EdgeSign,

    # Mechanisms
    MechanismPath,

    # Hypotheses
    HypothesesSection,
    Hypothesis,
    FalsificationTest,

    # Risk
    RiskSection,
    RegimeShiftRisk,
    DataLimitation,
    RiskSeverity,

    # Actions
    SuggestedDataset,
    NextAction,

    # Adapters (functions)
    from_shock_propagation,
    from_critical_path,
    from_genius_act,
)

from agent.core.adapters import EIMASAdapter
from agent.core.orchestrator import EconomicInsightOrchestrator, AsyncEconomicInsightOrchestrator

__all__ = [
    # Request
    'InsightRequest',
    'AnalysisFrame',

    # Output
    'EconomicInsightReport',
    'InsightMeta',

    # Graph
    'CausalGraph',
    'CausalNode',
    'CausalEdge',
    'EdgeSign',

    # Mechanisms
    'MechanismPath',

    # Hypotheses
    'HypothesesSection',
    'Hypothesis',
    'FalsificationTest',

    # Risk
    'RiskSection',
    'RegimeShiftRisk',
    'DataLimitation',
    'RiskSeverity',

    # Actions
    'SuggestedDataset',
    'NextAction',

    # Adapters
    'EIMASAdapter',
    'from_shock_propagation',
    'from_critical_path',
    'from_genius_act',

    # Orchestrators
    'EconomicInsightOrchestrator',
    'AsyncEconomicInsightOrchestrator',
]
