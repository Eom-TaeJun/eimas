"""
Interpretation Debate Agent - 경제학파별 결과 해석 토론

분석 결과를 다양한 경제학파 관점에서 해석하고 토론
ECON_AI_AGENT_SYSTEM.md Phase 4 구현

경제학파:
- Monetarist (통화주의): 통화량, 금리, 인플레이션 중심
- Keynesian (케인즈): 총수요, 재정정책, 고용 중심
- Austrian (오스트리아): 시장 자율, 정부 개입 비판, 사이클 중심
- Technical (기술적 분석): 가격/거래량 패턴, 모멘텀 중심

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
    AIProvider
)
from agents.base_agent import BaseAgent, AgentConfig
from core.schemas import AgentRole


# ============================================================================
# Economic Schools and Data Classes
# ============================================================================

class EconomicSchool(Enum):
    """경제학파"""
    MONETARIST = "monetarist"       # 통화주의
    KEYNESIAN = "keynesian"         # 케인즈
    AUSTRIAN = "austrian"           # 오스트리아
    TECHNICAL = "technical"         # 기술적 분석


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
class SchoolInterpretation:
    """학파별 해석"""
    school: EconomicSchool
    interpretation: str
    key_points: List[str]
    policy_implications: List[str]
    risk_assessment: str
    confidence: float
    supporting_theory: str
    counter_arguments: List[str] = field(default_factory=list)


@dataclass
class InterpretationConsensus:
    """해석 합의 결과"""
    topic: str
    consensus_points: List[str]         # 모든 학파가 동의하는 점
    divergence_points: List[str]        # 학파별 다른 해석
    school_interpretations: List[SchoolInterpretation]
    recommended_action: str
    risk_factors: List[str]
    confidence: float
    summary: str
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# Economic School Prompts (도메인 지식 주입)
# ============================================================================

SCHOOL_SYSTEM_PROMPTS = {
    EconomicSchool.MONETARIST: """
# 당신의 역할: Monetarist Economist (통화주의 경제학자)

## 핵심 신념 (Milton Friedman 기반)
1. "Inflation is always and everywhere a monetary phenomenon"
   - 인플레이션은 항상 통화 현상
   - 통화량 증가 → 물가 상승 (장기)

2. 통화정책의 중요성
   - Fed의 금리/통화량 결정이 경제의 핵심
   - 재정정책보다 통화정책이 더 효과적

3. 장기 균형
   - 단기: 통화량 → 실물경제 영향
   - 장기: 통화량 → 물가만 영향 (화폐 중립성)

## 분석 프레임워크
- M2 성장률 vs 인플레이션
- 실질금리 = 명목금리 - 기대인플레이션
- 테일러 룰: 적정 금리 수준 판단
- 통화 유통속도 (V = PY/M)

## 주요 지표
- M2, M1 통화량
- 연준 금리 (Fed Funds Rate)
- 기대 인플레이션 (BEI, TIPS spread)
- 실질 금리

## 해석 성향
- Fed 정책 변화에 민감
- 인플레이션 리스크 강조
- 통화량 과잉 시 경고
""",

    EconomicSchool.KEYNESIAN: """
# 당신의 역할: Keynesian Economist (케인즈 경제학자)

## 핵심 신념 (John Maynard Keynes 기반)
1. 총수요가 경제를 결정
   - Y = C + I + G + NX
   - 수요 부족 시 불황 지속 가능

2. 정부 개입의 필요성
   - 시장은 스스로 균형을 찾지 못할 수 있음
   - 재정정책으로 경기 조절 가능
   - 승수효과: 정부지출 1원 → GDP 1원 이상 증가

3. 유동성 함정
   - 금리가 0에 가까우면 통화정책 무력
   - 재정정책이 더 효과적

## 분석 프레임워크
- 총수요 구성요소 분석
- 산출 갭 (실제 GDP - 잠재 GDP)
- 고용/실업률 중시
- 승수효과 크기

## 주요 지표
- GDP 성장률, 구성요소
- 실업률, 고용 지표
- 소비자/기업 심리
- 정부 지출, 재정적자

## 해석 성향
- 고용과 성장 중시
- 경기 침체 시 재정 확대 지지
- 긴축 정책에 신중
""",

    EconomicSchool.AUSTRIAN: """
# 당신의 역할: Austrian Economist (오스트리아 학파 경제학자)

## 핵심 신념 (Hayek, Mises 기반)
1. 시장 가격의 정보 기능
   - 가격은 분산된 지식을 집약
   - 정부 개입은 가격 신호 왜곡

2. 경기 순환 이론 (Austrian Business Cycle Theory)
   - 인위적 저금리 → 과잉 투자 → 버블
   - 버블 붕괴는 불가피, 자연스러운 조정

3. 정부 개입 비판
   - 경기 부양책은 문제를 미룰 뿐
   - 장기적으로 더 큰 위기 초래

## 분석 프레임워크
- 자연 이자율 vs 시장 이자율
- 신용 확장 속도
- 자본재 vs 소비재 가격
- 마진부채, 레버리지 수준

## 주요 지표
- 신용 성장률
- 자산 가격 (주식, 부동산)
- 금/은 가격 (실물 화폐)
- 기업 부채, 가계 부채

## 해석 성향
- 버블 형성에 경고
- 구조조정의 필요성 강조
- 장기적 시각, 단기 부양 비판
""",

    EconomicSchool.TECHNICAL: """
# 당신의 역할: Technical Analyst (기술적 분석가)

## 핵심 신념
1. 가격에 모든 정보가 반영
   - 펀더멘털은 이미 가격에 포함
   - 가격 패턴 분석이 핵심

2. 역사는 반복된다
   - 시장 심리는 패턴을 형성
   - 과거 패턴으로 미래 예측 가능

3. 추세의 지속성
   - 추세는 반전 신호 전까지 지속
   - 추세 추종이 기본 전략

## 분석 프레임워크
- 가격 패턴 (Head & Shoulders, Double Top 등)
- 이동평균선 (20일, 50일, 200일)
- 모멘텀 지표 (RSI, MACD, Stochastic)
- 거래량 분석

## 주요 지표
- 가격 이동평균
- RSI (과매수/과매도)
- MACD (추세 전환)
- 거래량, OBV
- 변동성 (ATR, Bollinger Bands)

## 해석 성향
- 차트 패턴 중시
- 명확한 진입/청산 기준
- 손절 라인 설정
- 추세 추종
"""
}


# ============================================================================
# School-Specific Participants
# ============================================================================

def create_school_participants() -> List[DebateParticipant]:
    """경제학파별 토론 참여자 생성"""
    return [
        DebateParticipant(
            name="Monetarist",
            provider=AIProvider.CLAUDE,
            role="통화주의 관점",
            system_prompt=SCHOOL_SYSTEM_PROMPTS[EconomicSchool.MONETARIST]
        ),
        DebateParticipant(
            name="Keynesian",
            provider=AIProvider.OPENAI,
            role="케인즈 관점",
            system_prompt=SCHOOL_SYSTEM_PROMPTS[EconomicSchool.KEYNESIAN]
        ),
        DebateParticipant(
            name="Austrian",
            provider=AIProvider.GEMINI,
            role="오스트리아 학파 관점",
            system_prompt=SCHOOL_SYSTEM_PROMPTS[EconomicSchool.AUSTRIAN]
        ),
        DebateParticipant(
            name="Technical",
            provider=AIProvider.CLAUDE,  # 다른 Claude 인스턴스
            role="기술적 분석 관점",
            system_prompt=SCHOOL_SYSTEM_PROMPTS[EconomicSchool.TECHNICAL]
        )
    ]


# ============================================================================
# Interpretation Debate Implementation
# ============================================================================

class InterpretationDebate(DebateFramework[SchoolInterpretation]):
    """
    경제학파별 해석 토론

    분석 결과를 4개 학파 관점에서 해석하고 토론
    """

    INTERPRETATION_PROMPT = """
# 분석 결과 해석 요청

## 분석 주제
{topic}

## 사용된 방법론
{methodology}

## 주요 발견
{key_findings}

## 통계 결과
{statistics}

## 예측
{predictions}

---

## 당신의 역할
당신은 {school_name} 관점의 경제학자입니다.

위 분석 결과를 당신의 학파 관점에서 해석해주세요.

### 응답 형식 (JSON)
```json
{{
    "interpretation": "전체 해석 (2-3 문단)",
    "key_points": ["핵심 포인트 1", "핵심 포인트 2", "핵심 포인트 3"],
    "policy_implications": ["정책적 함의 1", "정책적 함의 2"],
    "risk_assessment": "리스크 평가",
    "confidence": 0.8,
    "supporting_theory": "뒷받침하는 이론/개념"
}}
```

특히 다음을 포함해주세요:
1. 당신 학파의 이론적 프레임워크로 결과 설명
2. 다른 학파와 다르게 볼 수 있는 부분
3. 투자/정책 관점에서의 시사점
"""

    CRITIQUE_PROMPT = """
# 해석 비판 요청

## 비판 대상
학파: {target_school}
해석: {interpretation}
핵심 포인트: {key_points}

## 당신의 학파: {your_school}

---

위 해석에 대해 당신의 학파 관점에서 비판해주세요:

1. 동의하는 부분은?
2. 동의하지 않는 부분과 그 이유는?
3. 누락된 중요한 관점은?
4. 대안적 해석은?

비판 심각도: minor / moderate / major
"""

    def __init__(
        self,
        participants: Optional[List[DebateParticipant]] = None,
        config: Optional[DebateConfig] = None
    ):
        if participants is None:
            participants = create_school_participants()

        if config is None:
            config = DebateConfig(
                max_rounds=3,
                enable_rebuttal=True,
                consensus_threshold=0.6  # 경제학파는 의견 다를 수 있음
            )

        super().__init__(participants, config)

    def get_proposal_prompt(self, context: Dict[str, Any]) -> str:
        """해석 요청 프롬프트"""
        result: AnalysisResult = context.get('analysis_result')
        school = context.get('current_school', 'Economist')

        return self.INTERPRETATION_PROMPT.format(
            topic=result.topic if result else 'N/A',
            methodology=result.methodology if result else 'N/A',
            key_findings='\n'.join([f"- {f}" for f in result.key_findings]) if result else 'N/A',
            statistics=json.dumps(result.statistics, indent=2) if result else '{}',
            predictions=json.dumps(result.predictions, indent=2) if result else '{}',
            school_name=school
        )

    def get_critique_prompt(
        self,
        target_opinion: Opinion,
        all_opinions: List[Opinion],
        context: Dict[str, Any]
    ) -> str:
        """비판 프롬프트"""
        interpretation: SchoolInterpretation = target_opinion.content

        return self.CRITIQUE_PROMPT.format(
            target_school=target_opinion.participant,
            interpretation=interpretation.interpretation if interpretation else 'N/A',
            key_points=', '.join(interpretation.key_points) if interpretation else 'N/A',
            your_school=context.get('critic_school', 'Economist')
        )

    def parse_proposal(self, response: str, participant: str) -> SchoolInterpretation:
        """응답을 SchoolInterpretation으로 파싱"""
        # JSON 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            json_str = json_match.group(0) if json_match else "{}"

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            data = {}

        # 학파 매핑
        school_map = {
            'Monetarist': EconomicSchool.MONETARIST,
            'Keynesian': EconomicSchool.KEYNESIAN,
            'Austrian': EconomicSchool.AUSTRIAN,
            'Technical': EconomicSchool.TECHNICAL
        }

        return SchoolInterpretation(
            school=school_map.get(participant, EconomicSchool.MONETARIST),
            interpretation=data.get('interpretation', response[:500]),
            key_points=data.get('key_points', []),
            policy_implications=data.get('policy_implications', []),
            risk_assessment=data.get('risk_assessment', ''),
            confidence=float(data.get('confidence', 0.7)),
            supporting_theory=data.get('supporting_theory', '')
        )

    def evaluate_consensus(self, opinions: List[Opinion]) -> ConsensusType:
        """합의 평가 - 경제학파는 합의 어려움"""
        if not opinions:
            return ConsensusType.NO_CONSENSUS

        # 해석들의 핵심 포인트 비교
        all_key_points = []
        for op in opinions:
            if op.content:
                all_key_points.extend(op.content.key_points)

        # 공통 키워드 찾기 (단순화)
        if len(set(all_key_points)) < len(all_key_points) * 0.5:
            return ConsensusType.MAJORITY  # 어느 정도 공통점

        return ConsensusType.HYBRID  # 대부분 다른 해석

    def merge_proposals(self, opinions: List[Opinion]) -> SchoolInterpretation:
        """해석들을 종합"""
        all_points = []
        all_implications = []
        all_risks = []

        for op in opinions:
            if op.content:
                all_points.extend(op.content.key_points)
                all_implications.extend(op.content.policy_implications)
                all_risks.append(op.content.risk_assessment)

        # 중복 제거
        unique_points = list(set(all_points))[:5]
        unique_implications = list(set(all_implications))[:5]

        return SchoolInterpretation(
            school=EconomicSchool.MONETARIST,  # 대표로 설정
            interpretation="다양한 경제학파 관점을 종합한 해석",
            key_points=unique_points,
            policy_implications=unique_implications,
            risk_assessment=' | '.join(all_risks[:3]),
            confidence=0.7,
            supporting_theory="Multi-school synthesis"
        )


# ============================================================================
# Interpretation Debate Agent
# ============================================================================

class InterpretationDebateAgent(BaseAgent):
    """
    경제학파별 해석 토론 에이전트

    사용법:
    ```python
    agent = InterpretationDebateAgent()
    consensus = await agent.interpret_results(analysis_result)
    ```
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        participants: Optional[List[DebateParticipant]] = None
    ):
        if config is None:
            config = AgentConfig(
                name="InterpretationDebateAgent",
                role=AgentRole.ANALYSIS,
                model="multi-ai"
            )
        super().__init__(config)

        self.debate = InterpretationDebate(participants)

    async def _execute(self, request: Any) -> InterpretationConsensus:
        """토론 실행"""
        return await self.interpret_results(
            analysis_result=request.get('analysis_result'),
            additional_context=request.get('context', {})
        )

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
        context = {
            'analysis_result': analysis_result,
            **(additional_context or {})
        }

        # 토론 실행
        result = await self.debate.run_debate(
            topic=f"Interpretation of: {analysis_result.topic}",
            context=context
        )

        # 합의점/분열점 추출
        consensus_points, divergence_points = self._extract_consensus_divergence(
            result.opinions
        )

        # 학파별 해석 수집
        school_interpretations = [
            op.content for op in result.opinions
            if op.content
        ]

        # 추천 행동 도출
        recommended_action = self._derive_recommendation(school_interpretations)

        # 리스크 요인 수집
        risk_factors = []
        for interp in school_interpretations:
            if interp.risk_assessment:
                risk_factors.append(f"[{interp.school.value}] {interp.risk_assessment}")

        return InterpretationConsensus(
            topic=analysis_result.topic,
            consensus_points=consensus_points,
            divergence_points=divergence_points,
            school_interpretations=school_interpretations,
            recommended_action=recommended_action,
            risk_factors=risk_factors[:5],
            confidence=result.confidence,
            summary=result.summary
        )

    def _extract_consensus_divergence(
        self,
        opinions: List[Opinion]
    ) -> tuple[List[str], List[str]]:
        """합의점과 분열점 추출"""
        all_points = {}

        for op in opinions:
            if op.content:
                for point in op.content.key_points:
                    # 단순화된 키워드 매칭
                    key = point[:50].lower()
                    if key not in all_points:
                        all_points[key] = []
                    all_points[key].append((op.participant, point))

        consensus = []
        divergence = []

        for key, occurrences in all_points.items():
            if len(occurrences) >= 3:  # 3개 이상 학파가 동의
                consensus.append(occurrences[0][1])
            elif len(occurrences) == 1:  # 1개 학파만 주장
                school, point = occurrences[0]
                divergence.append(f"[{school}] {point}")

        return consensus[:5], divergence[:5]

    def _derive_recommendation(
        self,
        interpretations: List[SchoolInterpretation]
    ) -> str:
        """추천 행동 도출"""
        if not interpretations:
            return "데이터 부족으로 추천 불가"

        # 신뢰도 가중 평균으로 종합
        bullish_score = 0
        bearish_score = 0
        total_weight = 0

        for interp in interpretations:
            weight = interp.confidence

            # 간단한 sentiment 분석
            text = (interp.interpretation + ' '.join(interp.key_points)).lower()

            bullish_words = ['growth', 'expansion', 'positive', 'bullish', 'increase', 'recovery']
            bearish_words = ['recession', 'contraction', 'negative', 'bearish', 'decline', 'risk']

            bullish_count = sum(1 for w in bullish_words if w in text)
            bearish_count = sum(1 for w in bearish_words if w in text)

            bullish_score += bullish_count * weight
            bearish_score += bearish_count * weight
            total_weight += weight

        if total_weight == 0:
            return "중립 유지 권고"

        net_score = (bullish_score - bearish_score) / total_weight

        if net_score > 1:
            return "위험 선호 확대 가능, 주식 비중 확대 고려"
        elif net_score > 0:
            return "온건한 낙관, 현재 포지션 유지"
        elif net_score > -1:
            return "신중한 접근, 리스크 관리 강화"
        else:
            return "방어적 포지션, 현금/채권 비중 확대"

    async def form_opinion(self, topic: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """BaseAgent 인터페이스"""
        result = AnalysisResult(
            topic=topic,
            methodology=context.get('methodology', 'Unknown'),
            key_findings=context.get('findings', []),
            statistics=context.get('statistics', {}),
            predictions=context.get('predictions', {}),
            confidence=context.get('confidence', 0.7)
        )

        consensus = await self.interpret_results(result)

        return {
            "topic": topic,
            "stance": consensus.recommended_action,
            "confidence": consensus.confidence,
            "reasoning": consensus.summary,
            "evidence": consensus.consensus_points,
            "metadata": {
                "schools_participated": [i.school.value for i in consensus.school_interpretations],
                "divergence_count": len(consensus.divergence_points)
            }
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def get_school_prompt(school: EconomicSchool) -> str:
    """특정 학파의 시스템 프롬프트 반환"""
    return SCHOOL_SYSTEM_PROMPTS.get(school, "")


def create_sample_analysis_result() -> AnalysisResult:
    """테스트용 샘플 분석 결과"""
    return AnalysisResult(
        topic="2025년 Fed 금리 전망",
        methodology="LASSO + Post-LASSO OLS",
        key_findings=[
            "Core PCE가 가장 유의미한 선행 지표",
            "실업률과 금리 간 강한 음의 상관관계",
            "금융 스트레스 지수 상승 시 금리 인하 확률 증가"
        ],
        statistics={
            "R_squared": 0.72,
            "selected_variables": 8,
            "p_value_threshold": 0.05
        },
        predictions={
            "rate_2025Q1": 4.25,
            "rate_2025Q2": 4.00,
            "rate_2025Q4": 3.75
        },
        confidence=0.75
    )


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("Interpretation Debate Agent Demo")
        print("=" * 60)

        # 샘플 분석 결과
        result = create_sample_analysis_result()

        print(f"\n[Analysis Result]")
        print(f"  Topic: {result.topic}")
        print(f"  Method: {result.methodology}")
        print(f"  Findings:")
        for f in result.key_findings:
            print(f"    - {f}")
        print(f"  Predictions: {result.predictions}")

        print(f"\n[Economic School Prompts Available]")
        for school in EconomicSchool:
            prompt = SCHOOL_SYSTEM_PROMPTS[school]
            print(f"  - {school.value}: {len(prompt)} chars")

        print(f"\n[Participants]")
        participants = create_school_participants()
        for p in participants:
            print(f"  - {p.name} ({p.provider.value}): {p.role}")

        print("\n" + "=" * 60)
        print("Full debate requires API keys.")
        print("Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY")
        print("=" * 60)

    asyncio.run(demo())
