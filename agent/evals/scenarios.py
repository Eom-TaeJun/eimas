"""
Economic Insight Agent - Eval Scenarios
========================================

5-10 scenario prompts with expected JSON shape checks
agentcommand.txt 요구사항: "Provide a small eval harness with 5-10 scenario prompts"
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from agent.schemas.insight_schema import AnalysisFrame


@dataclass
class EvalScenario:
    """평가 시나리오"""
    id: str
    name: str
    question: str
    frame: AnalysisFrame
    context: Optional[Dict[str, Any]] = None

    # Expected outputs
    expected_min_nodes: int = 3
    expected_min_edges: int = 2
    expected_min_mechanisms: int = 1
    expected_frame: Optional[AnalysisFrame] = None
    expected_keywords_in_phenomenon: List[str] = None

    def __post_init__(self):
        if self.expected_keywords_in_phenomenon is None:
            self.expected_keywords_in_phenomenon = []
        if self.expected_frame is None:
            self.expected_frame = self.frame


# =============================================================================
# Eval Scenarios (10개)
# =============================================================================

SCENARIOS: List[EvalScenario] = [
    # Scenario 1: Stablecoin → Treasury
    EvalScenario(
        id="S01",
        name="Stablecoin-Treasury Channel",
        question="스테이블코인 공급 증가가 국채 수요에 미치는 영향은?",
        frame=AnalysisFrame.CRYPTO,
        expected_min_nodes=5,
        expected_min_edges=3,
        expected_keywords_in_phenomenon=["스테이블코인", "국채"]
    ),

    # Scenario 2: Fed Rate Hike
    EvalScenario(
        id="S02",
        name="Fed Rate Policy Impact",
        question="Fed 금리 인상이 주식시장에 미치는 영향은?",
        frame=AnalysisFrame.MACRO,
        expected_min_nodes=5,
        expected_min_edges=4,
        expected_keywords_in_phenomenon=["Fed", "금리", "주식"]
    ),

    # Scenario 3: Liquidity Transmission
    EvalScenario(
        id="S03",
        name="Liquidity Transmission Mechanism",
        question="Fed 자산 축소가 시장 유동성에 어떻게 전파되나?",
        frame=AnalysisFrame.MACRO,
        expected_min_nodes=4,
        expected_keywords_in_phenomenon=["유동성", "Fed"]
    ),

    # Scenario 4: Crypto-Macro Correlation
    EvalScenario(
        id="S04",
        name="Crypto-Macro Correlation",
        question="순유동성 변화가 비트코인 가격에 미치는 영향은?",
        frame=AnalysisFrame.MIXED,
        expected_min_nodes=4,
        expected_keywords_in_phenomenon=["유동성", "비트코인"]
    ),

    # Scenario 5: Sector Rotation
    EvalScenario(
        id="S05",
        name="Sector Rotation Analysis",
        question="현재 섹터 로테이션이 기술주에 미치는 영향은?",
        frame=AnalysisFrame.MARKETS,
        expected_min_nodes=4,
        expected_min_edges=3,
        expected_keywords_in_phenomenon=["섹터", "기술"]
    ),

    # Scenario 6: DeFi TVL Impact
    EvalScenario(
        id="S06",
        name="DeFi TVL and ETH",
        question="DeFi TVL 증가가 ETH 가격에 미치는 영향은?",
        frame=AnalysisFrame.CRYPTO,
        expected_min_nodes=3,
        expected_keywords_in_phenomenon=["DeFi", "ETH"]
    ),

    # Scenario 7: VIX Transmission
    EvalScenario(
        id="S07",
        name="VIX Risk Transmission",
        question="VIX 급등이 자산 가격에 미치는 전파 경로는?",
        frame=AnalysisFrame.MARKETS,
        expected_min_nodes=4,
        expected_keywords_in_phenomenon=["VIX", "자산"]
    ),

    # Scenario 8: RRP Liquidity
    EvalScenario(
        id="S08",
        name="RRP Liquidity Drain",
        question="RRP 증가가 순유동성에 미치는 영향은?",
        frame=AnalysisFrame.MACRO,
        expected_min_nodes=3,
        expected_keywords_in_phenomenon=["RRP", "유동성"]
    ),

    # Scenario 9: Credit Spread
    EvalScenario(
        id="S09",
        name="Credit Spread Widening",
        question="신용 스프레드 확대가 하이일드 채권과 주식에 미치는 영향은?",
        frame=AnalysisFrame.MARKETS,
        expected_min_nodes=4,
        expected_keywords_in_phenomenon=["스프레드", "채권"]
    ),

    # Scenario 10: Full Integration
    EvalScenario(
        id="S10",
        name="Full Macro-Crypto Integration",
        question="Fed 정책 변화가 전통 금융과 암호화폐 시장에 동시에 미치는 영향은?",
        frame=AnalysisFrame.MIXED,
        expected_min_nodes=6,
        expected_min_edges=5,
        expected_min_mechanisms=2,
        expected_keywords_in_phenomenon=["Fed", "암호화폐"]
    ),
]


def get_scenario(scenario_id: str) -> Optional[EvalScenario]:
    """시나리오 ID로 조회"""
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    return None


def get_all_scenarios() -> List[EvalScenario]:
    """모든 시나리오 반환"""
    return SCENARIOS
