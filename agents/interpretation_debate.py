"""
Interpretation Debate Agent - 경제학파별 결과 해석 토론

분석 결과를 다양한 경제학파 관점에서 해석하고 토론
ECON_AI_AGENT_SYSTEM.md Phase 2 구현

경제학파:
- Monetarist (통화주의): 통화량, 금리, 인플레이션 중심
- Keynesian (케인즈): 총수요, 재정정책, 고용 중심
- Austrian (오스트리아): 시장 자율, 정부 개입 비판, 사이클 중심

Author: EIMAS Team
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import json
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentConfig
from core.schemas import AgentRole, AgentRequest, AgentResponse
from core.multi_llm_debate import MultiLLMDebate, DebateResult

# ============================================================================
# Economic Schools and Data Classes
# ============================================================================

class EconomicSchool(Enum):
    """경제학파"""
    MONETARIST = "monetarist"       # 통화주의
    KEYNESIAN = "keynesian"         # 케인즈
    AUSTRIAN = "austrian"           # 오스트리아


@dataclass
class AnalysisResult:
    """분석 결과 (해석 대상)"""
    topic: str
    methodology: str
    key_findings: List[str]
    statistics: Dict[str, Any]
    predictions: Dict[str, float]
    confidence: float
    raw_data_summary: str = ""


@dataclass
class InterpretationConsensus:
    """해석 합의 결과"""
    topic: str
    consensus_points: List[str]         # 모든 학파가 동의하는 점
    divergence_points: List[str]        # 학파별 다른 해석
    school_interpretations: List[Dict]
    recommended_action: str
    risk_factors: List[str]
    confidence: float
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Economic School Prompts (도메인 지식 주입)
# ============================================================================

SCHOOL_SYSTEM_PROMPTS = {
    EconomicSchool.MONETARIST: """
# Role: Monetarist Economist

## Core Beliefs (Milton Friedman)
1. "Inflation is always and everywhere a monetary phenomenon"
2. Money supply (M2) determines long-term price levels.
3. Fed policy (rates/QT) is the primary economic driver.

## Analysis Framework
- M2 Growth vs Inflation
- Real Rates = Nominal - Inflation Expectations
- Taylor Rule deviations

## Interpretation Bias
- Sensitive to Fed policy errors
- Focus on inflation risks
- Warns against excessive money supply
""",

    EconomicSchool.KEYNESIAN: """
# Role: Keynesian Economist

## Core Beliefs (John Maynard Keynes)
1. Aggregate Demand determines output (Y = C + I + G + NX).
2. Markets can remain irrational; government intervention is needed.
3. Fiscal policy is effective during liquidity traps.

## Analysis Framework
- Output Gap (Actual - Potential GDP)
- Employment/Unemployment
- Consumer Sentiment
- Fiscal Multipliers

## Interpretation Bias
- Focus on growth and employment
- Supports fiscal stimulus during downturns
- Cautious about premature tightening
""",

    EconomicSchool.AUSTRIAN: """
# Role: Austrian Economist

## Core Beliefs (Hayek, Mises)
1. Market prices coordinate dispersed knowledge.
2. Artificial low rates cause malinvestment bubbles.
3. Recessions are necessary cleansing processes.

## Analysis Framework
- Natural Rate vs Market Rate
- Credit Expansion & Leverage
- Capital Structure Distortion

## Interpretation Bias
- Warns of bubbles and moral hazard
- Critical of government intervention
- Focus on long-term structural health
"""
}


# ============================================================================
# Interpretation Debate Agent
# ============================================================================

class InterpretationDebateAgent(BaseAgent):
    """
    경제학파별 해석 토론 에이전트

    MultiLLMDebate 엔진을 사용하여 3개 학파(모델) 간 토론을 진행
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None
    ):
        if config is None:
            config = AgentConfig(
                name="InterpretationDebateAgent",
                role=AgentRole.ANALYSIS,
                model="multi-ai"
            )
        super().__init__(config)

        self.debate_engine = MultiLLMDebate(verbose=self.config.verbose)

    async def _execute(self, request: AgentRequest) -> Any:
        """토론 실행"""
        analysis_result = request.context.get('analysis_result')
        if not analysis_result:
            # Try to build from raw context
            analysis_result = AnalysisResult(
                topic=request.instruction,
                methodology="General",
                key_findings=[],
                statistics={},
                predictions={},
                confidence=0.5
            )

        consensus = await self.interpret_results(
            analysis_result=analysis_result,
            additional_context=request.context
        )

        return {
            "content": consensus,
            "summary": consensus.summary,
            "recommended_action": consensus.recommended_action
        }

    async def interpret_results(
        self,
        analysis_result: AnalysisResult,
        additional_context: Dict[str, Any] = None
    ) -> InterpretationConsensus:
        """
        분석 결과 해석 토론 실행

        Parameters:
        -----------
        analysis_result : AnalysisResult
            해석할 분석 결과
        additional_context : Dict
            추가 컨텍스트

        Returns:
        --------
        InterpretationConsensus
            해석 합의 결과
        """
        # Prepare context for debate engine
        context = {
            'topic': analysis_result.topic,
            'key_findings': analysis_result.key_findings,
            'statistics': analysis_result.statistics,
            'predictions': analysis_result.predictions,
            'school_prompts': {k.value: v for k, v in SCHOOL_SYSTEM_PROMPTS.items()},
            **(additional_context or {})
        }

        # Run Debate
        debate_result: DebateResult = await self.debate_engine.run_debate(
            topic=f"Economic Interpretation of: {analysis_result.topic}",
            context=context,
            max_rounds=3
        )

        # Process results
        return InterpretationConsensus(
            topic=analysis_result.topic,
            consensus_points=debate_result.consensus_points,
            divergence_points=[d['point'] for d in debate_result.dissent_points if 'point' in d],
            school_interpretations=self._extract_school_interpretations(debate_result),
            recommended_action=debate_result.consensus_position,
            risk_factors=[r for r in debate_result.dissent_points if 'risk' in r], # Simplified
            confidence=(debate_result.consensus_confidence[0] + debate_result.consensus_confidence[1]) / 200.0,
            summary=f"Consensus: {debate_result.consensus_position}. {len(debate_result.consensus_points)} consensus points."
        )

    def _extract_school_interpretations(self, debate_result: DebateResult) -> List[Dict]:
        """토론 결과에서 학파별 해석 추출"""
        # MultiLLMDebate captures positions in transcript
        # Need to parse transcript or model_contributions
        interpretations = []
        
        # Round 1 has initial positions
        for entry in debate_result.transcript:
            if entry.get('round') == 1:
                positions = entry.get('content', {})
                for model, pos in positions.items():
                    interpretations.append({
                        'school': pos.get('role', model),
                        'stance': pos.get('stance'),
                        'reasoning': pos.get('reasoning')
                    })
        return interpretations

    async def form_opinion(self, topic: str, context: Dict[str, Any]) -> "AgentOpinion":
        """BaseAgent 인터페이스"""
        from core.schemas import AgentOpinion, OpinionStrength

        # Create dummy analysis result from context
        ar = AnalysisResult(
            topic=topic,
            methodology="Context Analysis",
            key_findings=[f"Context key: {k}" for k in context.keys()][:3],
            statistics={},
            predictions={},
            confidence=0.5
        )
        
        consensus = await self.interpret_results(ar, context)

        return AgentOpinion(
            agent_role=self.config.role,
            topic=topic,
            position=consensus.recommended_action,
            confidence=consensus.confidence,
            strength=OpinionStrength.NEUTRAL, # Default
            evidence=consensus.consensus_points[:3],
            reasoning=consensus.summary
        )

# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("Interpretation Debate Agent Demo (Multi-LLM)")
        print("=" * 60)

        agent = InterpretationDebateAgent()
        
        # Mock analysis result
        result = AnalysisResult(
            topic="Rising Bond Yields",
            methodology="Trend Analysis",
            key_findings=["10Y Yield up 20bps", "Inflation expectations stable"],
            statistics={"r2": 0.8},
            predictions={"next_month": 4.5},
            confidence=0.8
        )
        
        print(f"Interpreting: {result.topic}")
        consensus = await agent.interpret_results(result)
        
        print(f"\nConsensus: {consensus.recommended_action}")
        print(f"Summary: {consensus.summary}")

    asyncio.run(demo())
