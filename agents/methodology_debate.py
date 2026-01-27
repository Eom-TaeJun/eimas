"""
Methodology Debate Agent - 분석 방법론 토론

Multi-AI 토론을 통해 최적의 분석 방법론 결정
ECON_AI_AGENT_SYSTEM.md Phase 2 구현

Author: EIMAS Team
"""

import asyncio
from dataclasses import dataclass, field, asdict
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
# Methodology Types and Data Classes
# ============================================================================

class MethodologyType(Enum):
    """분석 방법론 유형"""
    LASSO = "LASSO"                         # 변수 선택
    POST_LASSO_OLS = "POST_LASSO_OLS"      # 통계 추론
    VAR = "VAR"                             # Vector Autoregression
    IRF = "IRF"                             # Impulse Response Function
    GRANGER = "GRANGER"                     # Granger Causality
    GARCH = "GARCH"                         # 변동성 모델링
    ML_ENSEMBLE = "ML_ENSEMBLE"             # 머신러닝 앙상블
    BAYESIAN = "BAYESIAN"                   # 베이지안 추론
    HYBRID = "HYBRID"                       # 하이브리드 접근


class ResearchGoal(Enum):
    """연구 목표"""
    VARIABLE_SELECTION = "variable_selection"   # 핵심 변수 식별
    FORECASTING = "forecasting"                 # 예측
    CAUSAL_INFERENCE = "causal_inference"       # 인과관계 분석
    VOLATILITY_MODELING = "volatility"          # 변동성 분석
    DYNAMIC_RELATIONSHIP = "dynamic"            # 동적 관계 분석
    INTERPRETATION = "interpretation"           # 해석/설명


@dataclass
class DataSummary:
    """데이터 요약 정보"""
    n_observations: int
    n_variables: int
    time_range: str                     # "2020-01 to 2024-12"
    frequency: str                      # "daily", "weekly", "monthly"
    missing_ratio: float
    stationarity: Dict[str, bool]       # 변수별 정상성
    variable_categories: Dict[str, List[str]]  # 카테고리별 변수 목록
    key_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodologyDecision:
    """최종 방법론 결정"""
    selected_methodology: str
    components: List[str]                   # HYBRID인 경우 구성 요소
    pipeline: List[str]                     # 실행 순서
    parameters: Dict[str, Any]
    validation: str
    confidence: float
    rationale: str
    dissenting_views: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Methodology Knowledge Base
# ============================================================================

METHODOLOGY_KNOWLEDGE = {
    MethodologyType.LASSO: {
        "name": "LASSO (Least Absolute Shrinkage and Selection Operator)",
        "use_case": "변수 선택, 고차원 데이터",
        "pros": ["Sparsity 유도", "과적합 방지", "핵심 변수 식별"],
        "cons": ["비선형 관계 미포착", "그룹화된 변수 중 하나만 선택"],
        "requirements": ["정규화된 변수", "충분한 관측치"],
        "best_for": ResearchGoal.VARIABLE_SELECTION
    },
    MethodologyType.POST_LASSO_OLS: {
        "name": "Post-LASSO OLS with HAC Standard Errors",
        "use_case": "통계적 추론, 계수 해석",
        "pros": ["p-value 제공", "시계열 특성 보정", "해석 가능"],
        "cons": ["LASSO 선택에 의존", "선형성 가정"],
        "requirements": ["LASSO 선택 결과", "HAC 보정"],
        "best_for": ResearchGoal.INTERPRETATION
    },
    MethodologyType.VAR: {
        "name": "Vector Autoregression",
        "use_case": "다변량 시계열 분석, 동적 관계",
        "pros": ["변수 간 상호작용", "예측", "IRF 분석 가능"],
        "cons": ["변수 많으면 불안정", "차원의 저주"],
        "requirements": ["정상성", "적절한 래그 선택"],
        "best_for": ResearchGoal.DYNAMIC_RELATIONSHIP
    },
    MethodologyType.IRF: {
        "name": "Impulse Response Function",
        "use_case": "충격 전파 분석",
        "pros": ["시간에 따른 영향 추적", "정책 시뮬레이션"],
        "cons": ["VAR 추정에 의존", "식별 문제"],
        "requirements": ["VAR 모델", "충격 식별"],
        "best_for": ResearchGoal.CAUSAL_INFERENCE
    },
    MethodologyType.GRANGER: {
        "name": "Granger Causality Test",
        "use_case": "선행 관계 검정",
        "pros": ["시간적 선후 관계", "통계적 검정"],
        "cons": ["상관 ≠ 인과", "bivariate 한계"],
        "requirements": ["정상성", "적절한 래그"],
        "best_for": ResearchGoal.CAUSAL_INFERENCE
    },
    MethodologyType.GARCH: {
        "name": "Generalized Autoregressive Conditional Heteroskedasticity",
        "use_case": "변동성 모델링, 리스크 분석",
        "pros": ["변동성 클러스터링", "조건부 분산"],
        "cons": ["복잡성", "수렴 문제"],
        "requirements": ["고빈도 데이터", "ARCH 효과"],
        "best_for": ResearchGoal.VOLATILITY_MODELING
    },
    MethodologyType.ML_ENSEMBLE: {
        "name": "Machine Learning Ensemble (RF, XGBoost, etc.)",
        "use_case": "예측, 비선형 관계",
        "pros": ["높은 예측력", "비선형 포착", "변수 중요도"],
        "cons": ["해석 어려움", "과적합 위험"],
        "requirements": ["충분한 데이터", "적절한 검증"],
        "best_for": ResearchGoal.FORECASTING
    }
}

GOAL_METHOD_MAPPING = {
    ResearchGoal.VARIABLE_SELECTION: [
        MethodologyType.LASSO,
        MethodologyType.POST_LASSO_OLS
    ],
    ResearchGoal.FORECASTING: [
        MethodologyType.ML_ENSEMBLE,
        MethodologyType.VAR
    ],
    ResearchGoal.CAUSAL_INFERENCE: [
        MethodologyType.GRANGER,
        MethodologyType.VAR,
        MethodologyType.IRF
    ],
    ResearchGoal.VOLATILITY_MODELING: [
        MethodologyType.GARCH
    ],
    ResearchGoal.DYNAMIC_RELATIONSHIP: [
        MethodologyType.VAR,
        MethodologyType.IRF,
        MethodologyType.GRANGER
    ],
    ResearchGoal.INTERPRETATION: [
        MethodologyType.POST_LASSO_OLS,
        MethodologyType.LASSO
    ]
}


# ============================================================================
# Methodology Debate Agent
# ============================================================================

class MethodologyDebateAgent(BaseAgent):
    """
    방법론 토론 에이전트

    MultiLLMDebate 엔진을 사용하여 최적의 분석 방법론을 결정
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None
    ):
        if config is None:
            config = AgentConfig(
                name="MethodologyDebateAgent",
                role=AgentRole.ANALYSIS,
                model="multi-ai"
            )
        super().__init__(config)

        self.debate_engine = MultiLLMDebate(verbose=self.config.verbose)

    async def _execute(self, request: Any) -> Any:
        """토론 실행"""
        # Parse inputs
        research_question = request.context.get('research_question') or request.instruction
        research_goal_str = request.context.get('research_goal', 'variable_selection')
        try:
            research_goal = ResearchGoal(research_goal_str)
        except ValueError:
            research_goal = ResearchGoal.VARIABLE_SELECTION
            
        data_summary = request.context.get('data_summary')
        research_context = request.context.get('research_context', '')

        decision = await self.debate_methodology(
            research_question=research_question,
            research_goal=research_goal,
            data_summary=data_summary,
            research_context=research_context
        )

        return {
            "content": decision,
            "selected_methodology": decision.selected_methodology,
            "rationale": decision.rationale,
            "confidence": decision.confidence
        }

    async def debate_methodology(
        self,
        research_question: str,
        research_goal: ResearchGoal,
        data_summary: Optional[DataSummary] = None,
        research_context: str = ""
    ) -> MethodologyDecision:
        """
        방법론 토론 실행

        Parameters:
        -----------
        research_question : str
            연구 질문
        research_goal : ResearchGoal
            연구 목표 (변수선택, 예측, 인과분석 등)
        data_summary : DataSummary
            데이터 요약 정보
        research_context : str
            추가 컨텍스트

        Returns:
        --------
        MethodologyDecision
            최종 방법론 결정
        """
        # Prepare Knowledge Base
        knowledge_str = "\n".join([
            f"- {m.value}: {k['use_case']} (Pros: {', '.join(k['pros'])})" 
            for m, k in METHODOLOGY_KNOWLEDGE.items()
        ])
        
        # Prepare context for debate engine
        context = {
            'research_question': research_question,
            'research_goal': research_goal.value,
            'data_summary': asdict(data_summary) if data_summary else {},
            'research_context': research_context,
            'available_methodologies': knowledge_str,
            'recommended_for_goal': [m.value for m in GOAL_METHOD_MAPPING.get(research_goal, [])]
        }

        # Run Debate
        debate_result = await self.debate_engine.run_debate(
            topic=f"Optimal Methodology for: {research_question}",
            context=context,
            max_rounds=3
        )

        # Parse final decision from consensus
        return MethodologyDecision(
            selected_methodology=debate_result.consensus_position,
            components=debate_result.consensus_points, # Simplified mapping
            pipeline=[debate_result.consensus_position], # Simplified
            parameters={}, # Would need deeper parsing
            validation="Time Series Cross Validation", # Default
            confidence=(debate_result.consensus_confidence[0] + debate_result.consensus_confidence[1]) / 200.0,
            rationale=f"Consensus reached on {debate_result.consensus_position}. " + " ".join(debate_result.consensus_points[:2]),
            dissenting_views=[d.get('point', '') for d in debate_result.dissent_points]
        )

    async def form_opinion(self, topic: str, context: Dict[str, Any]) -> "AgentOpinion":
        """BaseAgent 인터페이스"""
        from core.schemas import AgentOpinion, OpinionStrength
        from dataclasses import asdict

        decision = await self.debate_methodology(
            research_question=topic,
            research_goal=ResearchGoal.VARIABLE_SELECTION, # Default
            data_summary=None,
            research_context=str(context)
        )

        return AgentOpinion(
            agent_role=self.config.role,
            topic=topic,
            position=decision.selected_methodology,
            confidence=decision.confidence,
            strength=OpinionStrength.AGREE,
            evidence=decision.components[:3],
            reasoning=decision.rationale
        )

    def get_recommended_methods(
        self,
        goal: ResearchGoal
    ) -> List[MethodologyType]:
        """연구 목표에 맞는 추천 방법론"""
        return GOAL_METHOD_MAPPING.get(goal, [MethodologyType.LASSO])


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    from dataclasses import asdict
    async def demo():
        print("=" * 60)
        print("Methodology Debate Agent Demo")
        print("=" * 60)

        # 샘플 데이터 요약
        data_summary = DataSummary(
            n_observations=1000,
            n_variables=50,
            time_range="2020-01 to 2024-12",
            frequency="daily",
            missing_ratio=0.02,
            stationarity={"SPY": True, "GDP": False},
            variable_categories={
                "equity": ["SPY", "QQQ", "IWM"],
                "bonds": ["TLT", "IEF", "HYG"],
                "rates": ["DGS2", "DGS10", "FEDFUNDS"]
            }
        )

        print(f"\n[Data Summary]")
        print(f"  Observations: {data_summary.n_observations}")
        
        agent = MethodologyDebateAgent()
        print("\nStarting debate...")
        decision = await agent.debate_methodology(
            research_question="Forecasting Fed Funds Rate",
            research_goal=ResearchGoal.FORECASTING,
            data_summary=data_summary
        )
        
        print(f"\nSelected: {decision.selected_methodology}")
        print(f"Rationale: {decision.rationale}")

    asyncio.run(demo())
