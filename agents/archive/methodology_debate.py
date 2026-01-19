"""
Methodology Debate Agent - 분석 방법론 토론

Multi-AI 토론을 통해 최적의 분석 방법론 결정
ECON_AI_AGENT_SYSTEM.md Phase 2 구현

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
sys.path.append('..')
from core.archive.debate_framework import (
    DebateFramework,
    DebateParticipant,
    DebateConfig,
    Opinion,
    ConsensusType,
    AIProvider,
    get_default_participants
)
from agents.base_agent import BaseAgent, AgentConfig
from core.schemas import AgentRole


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
class MethodologyProposal:
    """방법론 제안"""
    primary_method: MethodologyType
    secondary_methods: List[MethodologyType] = field(default_factory=list)
    preprocessing: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    validation_strategy: str = "time_series_cv"
    rationale: str = ""
    expected_insights: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    confidence: float = 0.7


@dataclass
class MethodologyDecision:
    """최종 방법론 결정"""
    selected_methodology: MethodologyType
    components: List[MethodologyType]       # HYBRID인 경우 구성 요소
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
# Methodology Debate Implementation
# ============================================================================

class MethodologyDebate(DebateFramework[MethodologyProposal]):
    """
    방법론 선택을 위한 Multi-AI 토론

    Round 1: 각 AI가 방법론 제안
    Round 2: 상호 비판 (장단점 지적)
    Round 3: 합의 또는 하이브리드 도출
    """

    PROPOSAL_PROMPT_TEMPLATE = """
# 방법론 제안 요청

## 연구 주제
{research_question}

## 연구 목표
{research_goal}

## 데이터 요약
- 관측치 수: {n_observations}
- 변수 수: {n_variables}
- 기간: {time_range}
- 빈도: {frequency}
- 변수 카테고리: {variable_categories}

## 기존 연구 컨텍스트
{research_context}

## 사용 가능한 방법론
{available_methods}

---

## 요청사항

위 컨텍스트를 고려하여 최적의 분석 방법론을 제안해주세요.

### 응답 형식 (JSON)
```json
{{
    "primary_method": "LASSO | VAR | GRANGER | ...",
    "secondary_methods": ["method1", "method2"],
    "preprocessing": ["standardization", "differencing", ...],
    "hyperparameters": {{
        "key": "value"
    }},
    "validation_strategy": "time_series_cv | walk_forward | ...",
    "rationale": "선택 이유 상세 설명",
    "expected_insights": ["예상 인사이트 1", "예상 인사이트 2"],
    "limitations": ["한계점 1", "한계점 2"],
    "confidence": 0.8
}}
```

특히 다음을 고려해주세요:
1. 연구 목표에 가장 적합한 방법론인가?
2. 데이터 특성(크기, 빈도)에 맞는가?
3. 경제학적 해석이 가능한가?
4. 과적합 위험은 어떻게 관리하는가?
"""

    CRITIQUE_PROMPT_TEMPLATE = """
# 방법론 비판 요청

## 비판 대상 제안
- 제안자: {proposer}
- 주요 방법론: {primary_method}
- 근거: {rationale}

## 다른 제안들
{other_proposals}

## 연구 컨텍스트
목표: {research_goal}
데이터: {n_observations} 관측치, {n_variables} 변수

---

위 제안에 대해 비판적으로 검토해주세요:

1. **장점**: 이 접근법의 강점은 무엇인가?
2. **약점**: 어떤 한계가 있는가?
3. **대안**: 더 나은 접근법이 있는가?
4. **보완점**: 어떻게 개선할 수 있는가?

비판의 심각도를 표시해주세요: minor / moderate / major
"""

    def __init__(
        self,
        participants: Optional[List[DebateParticipant]] = None,
        config: Optional[DebateConfig] = None
    ):
        if participants is None:
            participants = get_default_participants()

        if config is None:
            config = DebateConfig(
                max_rounds=3,
                enable_rebuttal=True,
                consensus_threshold=0.7
            )

        super().__init__(participants, config)

    def get_proposal_prompt(self, context: Dict[str, Any]) -> str:
        """제안 프롬프트 생성"""
        data_summary: DataSummary = context.get('data_summary')

        # 사용 가능한 방법론 목록
        methods_desc = []
        for method, info in METHODOLOGY_KNOWLEDGE.items():
            methods_desc.append(
                f"- **{method.value}**: {info['use_case']}\n"
                f"  장점: {', '.join(info['pros'][:2])}\n"
                f"  단점: {', '.join(info['cons'][:2])}"
            )

        return self.PROPOSAL_PROMPT_TEMPLATE.format(
            research_question=context.get('research_question', 'Not specified'),
            research_goal=context.get('research_goal', ResearchGoal.VARIABLE_SELECTION).value,
            n_observations=data_summary.n_observations if data_summary else 'N/A',
            n_variables=data_summary.n_variables if data_summary else 'N/A',
            time_range=data_summary.time_range if data_summary else 'N/A',
            frequency=data_summary.frequency if data_summary else 'N/A',
            variable_categories=str(data_summary.variable_categories) if data_summary else 'N/A',
            research_context=context.get('research_context', ''),
            available_methods='\n'.join(methods_desc)
        )

    def get_critique_prompt(
        self,
        target_opinion: Opinion,
        all_opinions: List[Opinion],
        context: Dict[str, Any]
    ) -> str:
        """비판 프롬프트 생성"""
        proposal: MethodologyProposal = target_opinion.content

        # 다른 제안 요약
        other_summaries = []
        for op in all_opinions:
            if op.participant != target_opinion.participant and op.content:
                other_summaries.append(
                    f"- {op.participant}: {op.content.primary_method.value}"
                )

        data_summary: DataSummary = context.get('data_summary')

        return self.CRITIQUE_PROMPT_TEMPLATE.format(
            proposer=target_opinion.participant,
            primary_method=proposal.primary_method.value if proposal else 'N/A',
            rationale=proposal.rationale if proposal else 'N/A',
            other_proposals='\n'.join(other_summaries) if other_summaries else 'None',
            research_goal=context.get('research_goal', ResearchGoal.VARIABLE_SELECTION).value,
            n_observations=data_summary.n_observations if data_summary else 'N/A',
            n_variables=data_summary.n_variables if data_summary else 'N/A'
        )

    def parse_proposal(self, response: str, participant: str) -> MethodologyProposal:
        """AI 응답을 MethodologyProposal로 파싱"""
        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 블록 없으면 전체에서 찾기
            json_match = re.search(r'\{[\s\S]*\}', response)
            json_str = json_match.group(0) if json_match else "{}"

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본값
            return MethodologyProposal(
                primary_method=MethodologyType.LASSO,
                rationale=response[:500]
            )

        # MethodologyType 파싱
        primary = data.get('primary_method', 'LASSO')
        try:
            primary_method = MethodologyType[primary.upper()]
        except KeyError:
            primary_method = MethodologyType.LASSO

        secondary = []
        for m in data.get('secondary_methods', []):
            try:
                secondary.append(MethodologyType[m.upper()])
            except KeyError:
                continue

        return MethodologyProposal(
            primary_method=primary_method,
            secondary_methods=secondary,
            preprocessing=data.get('preprocessing', []),
            hyperparameters=data.get('hyperparameters', {}),
            validation_strategy=data.get('validation_strategy', 'time_series_cv'),
            rationale=data.get('rationale', ''),
            expected_insights=data.get('expected_insights', []),
            limitations=data.get('limitations', []),
            confidence=float(data.get('confidence', 0.7))
        )

    def evaluate_consensus(self, opinions: List[Opinion]) -> ConsensusType:
        """합의 여부 평가"""
        if not opinions:
            return ConsensusType.NO_CONSENSUS

        # 주요 방법론 추출
        methods = []
        for op in opinions:
            if op.content:
                methods.append(op.content.primary_method)

        if not methods:
            return ConsensusType.NO_CONSENSUS

        # 만장일치 체크
        if len(set(methods)) == 1:
            return ConsensusType.UNANIMOUS

        # 다수결 체크
        from collections import Counter
        method_counts = Counter(methods)
        most_common = method_counts.most_common(1)[0]

        if most_common[1] / len(methods) >= self.config.consensus_threshold:
            return ConsensusType.MAJORITY

        # 하이브리드 필요
        return ConsensusType.HYBRID

    def merge_proposals(self, opinions: List[Opinion]) -> MethodologyProposal:
        """여러 제안을 하이브리드로 병합"""
        all_methods = set()
        all_preprocessing = set()
        all_insights = []
        all_limitations = []

        for op in opinions:
            if op.content:
                all_methods.add(op.content.primary_method)
                all_methods.update(op.content.secondary_methods)
                all_preprocessing.update(op.content.preprocessing)
                all_insights.extend(op.content.expected_insights)
                all_limitations.extend(op.content.limitations)

        methods_list = list(all_methods)

        # 하이브리드 파이프라인 구성
        # 우선순위: 변수선택 → 인과분석 → 예측
        pipeline_order = [
            MethodologyType.LASSO,
            MethodologyType.GRANGER,
            MethodologyType.VAR,
            MethodologyType.IRF,
            MethodologyType.ML_ENSEMBLE
        ]

        ordered_methods = [m for m in pipeline_order if m in methods_list]
        remaining = [m for m in methods_list if m not in ordered_methods]
        ordered_methods.extend(remaining)

        return MethodologyProposal(
            primary_method=MethodologyType.HYBRID,
            secondary_methods=ordered_methods,
            preprocessing=list(all_preprocessing),
            hyperparameters={},
            validation_strategy="time_series_cv",
            rationale="Multi-AI 토론을 통한 하이브리드 접근법",
            expected_insights=list(set(all_insights))[:5],
            limitations=list(set(all_limitations))[:5],
            confidence=0.75
        )


# ============================================================================
# Methodology Debate Agent (High-level wrapper)
# ============================================================================

class MethodologyDebateAgent(BaseAgent):
    """
    방법론 토론 에이전트

    사용법:
    ```python
    agent = MethodologyDebateAgent()
    decision = await agent.debate_methodology(
        research_question="Fed 금리 예측의 핵심 변수는?",
        research_goal=ResearchGoal.VARIABLE_SELECTION,
        data_summary=data_summary
    )
    ```
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        participants: Optional[List[DebateParticipant]] = None
    ):
        if config is None:
            config = AgentConfig(
                name="MethodologyDebateAgent",
                role=AgentRole.ANALYSIS,
                model="multi-ai"
            )
        super().__init__(config)

        self.debate = MethodologyDebate(participants)

    async def _execute(self, request: Any) -> MethodologyDecision:
        """토론 실행"""
        return await self.debate_methodology(
            research_question=request.get('research_question', ''),
            research_goal=request.get('research_goal', ResearchGoal.VARIABLE_SELECTION),
            data_summary=request.get('data_summary'),
            research_context=request.get('research_context', '')
        )

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
            추가 컨텍스트 (기존 연구, 도메인 지식 등)

        Returns:
        --------
        MethodologyDecision
            최종 방법론 결정
        """
        context = {
            'research_question': research_question,
            'research_goal': research_goal,
            'data_summary': data_summary,
            'research_context': research_context
        }

        # 토론 실행
        result = await self.debate.run_debate(
            topic=f"Methodology for: {research_question}",
            context=context
        )

        # MethodologyDecision으로 변환
        final_proposal: MethodologyProposal = result.final_decision

        if final_proposal.primary_method == MethodologyType.HYBRID:
            components = final_proposal.secondary_methods
            pipeline = [m.value for m in components]
        else:
            components = [final_proposal.primary_method] + final_proposal.secondary_methods
            pipeline = [final_proposal.primary_method.value]
            pipeline.extend([m.value for m in final_proposal.secondary_methods])

        return MethodologyDecision(
            selected_methodology=final_proposal.primary_method,
            components=components,
            pipeline=pipeline,
            parameters=final_proposal.hyperparameters,
            validation=final_proposal.validation_strategy,
            confidence=result.confidence,
            rationale=final_proposal.rationale,
            dissenting_views=result.dissenting_views
        )

    async def form_opinion(self, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """BaseAgent 인터페이스 구현"""
        decision = await self.debate_methodology(
            research_question=topic,
            research_goal=context.get('goal', ResearchGoal.VARIABLE_SELECTION),
            data_summary=context.get('data_summary'),
            research_context=context.get('context', '')
        )

        return {
            "topic": topic,
            "stance": decision.selected_methodology.value,
            "confidence": decision.confidence,
            "reasoning": decision.rationale,
            "evidence": decision.pipeline,
            "metadata": {
                "components": [m.value for m in decision.components],
                "validation": decision.validation
            }
        }

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
        print(f"  Variables: {data_summary.n_variables}")
        print(f"  Period: {data_summary.time_range}")

        # 추천 방법론 확인
        print(f"\n[Recommended Methods by Goal]")
        for goal in ResearchGoal:
            methods = GOAL_METHOD_MAPPING.get(goal, [])
            print(f"  {goal.value}: {[m.value for m in methods]}")

        print("\n" + "=" * 60)
        print("Full debate requires API keys.")
        print("Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY")
        print("=" * 60)

    asyncio.run(demo())
