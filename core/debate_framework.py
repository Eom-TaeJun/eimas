"""
Debate Framework - Multi-AI 토론 프로세스 일반화

모든 토론 기반 에이전트의 기반 프레임워크
Round 1: 의견 제시 → Round 2: 상호 비판 → Round 3: 합의/하이브리드 도출

Author: EIMAS Team
"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
import json


# ============================================================================
# Enums and Data Classes
# ============================================================================

class DebatePhase(Enum):
    """토론 단계"""
    PROPOSAL = "proposal"           # Round 1: 의견 제시
    CRITIQUE = "critique"           # Round 2: 상호 비판
    REBUTTAL = "rebuttal"          # Round 2.5: 반박 (optional)
    CONSENSUS = "consensus"         # Round 3: 합의 도출
    COMPLETED = "completed"


class ConsensusType(Enum):
    """합의 유형"""
    UNANIMOUS = "unanimous"         # 만장일치
    MAJORITY = "majority"           # 다수결
    HYBRID = "hybrid"              # 하이브리드 (여러 의견 통합)
    NO_CONSENSUS = "no_consensus"  # 합의 불가


class AIProvider(Enum):
    """AI 제공자"""
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"
    PERPLEXITY = "perplexity"


@dataclass
class DebateParticipant:
    """토론 참여자"""
    name: str
    provider: AIProvider
    role: str = ""                  # 역할/관점 (예: "Monetarist", "Conservative")
    model: str = ""
    api_key: Optional[str] = None
    system_prompt: str = ""

    def __post_init__(self):
        if not self.model:
            default_models = {
                AIProvider.CLAUDE: "claude-sonnet-4-20250514",
                AIProvider.OPENAI: "gpt-4o",
                AIProvider.GEMINI: "gemini-1.5-pro",
                AIProvider.PERPLEXITY: "llama-3.1-sonar-large-128k-online"
            }
            self.model = default_models.get(self.provider, "")


@dataclass
class Opinion:
    """토론 의견"""
    participant: str
    provider: AIProvider
    content: Any                    # 제안 내용 (구조화된 데이터)
    reasoning: str                  # 근거
    confidence: float               # 신뢰도 (0-1)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Critique:
    """비판"""
    critic: str                     # 비판자
    target: str                     # 비판 대상
    points: List[str]               # 비판 포인트
    severity: str                   # "minor", "moderate", "major"
    counter_evidence: List[str] = field(default_factory=list)


@dataclass
class Rebuttal:
    """반박"""
    participant: str
    responding_to: str              # 누구의 비판에 대한 반박인지
    points: List[str]
    revised_position: Optional[Any] = None  # 수정된 입장 (있는 경우)


@dataclass
class DebateResult:
    """토론 결과"""
    topic: str
    consensus_type: ConsensusType
    final_decision: Any             # 최종 결정 (구조화된 데이터)
    confidence: float
    opinions: List[Opinion] = field(default_factory=list)
    critiques: List[Critique] = field(default_factory=list)
    rebuttals: List[Rebuttal] = field(default_factory=list)
    summary: str = ""
    dissenting_views: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DebateConfig:
    """토론 설정"""
    max_rounds: int = 3
    enable_rebuttal: bool = True
    consensus_threshold: float = 0.7    # 70% 이상 동의시 합의
    timeout_seconds: int = 120
    parallel_execution: bool = True


# ============================================================================
# AI Client Wrapper
# ============================================================================

class AIClient:
    """AI API 클라이언트 래퍼"""

    def __init__(self, participant: DebateParticipant):
        self.participant = participant
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self):
        api_key = self.participant.api_key

        if self.participant.provider == AIProvider.CLAUDE:
            from anthropic import Anthropic
            return Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

        elif self.participant.provider == AIProvider.OPENAI:
            from openai import OpenAI
            return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        elif self.participant.provider == AIProvider.GEMINI:
            import google.generativeai as genai
            genai.configure(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
            return genai.GenerativeModel(self.participant.model)

        elif self.participant.provider == AIProvider.PERPLEXITY:
            from openai import OpenAI
            return OpenAI(
                api_key=api_key or os.getenv("PERPLEXITY_API_KEY"),
                base_url="https://api.perplexity.ai"
            )

        raise ValueError(f"Unknown provider: {self.participant.provider}")

    async def complete(self, prompt: str, system_prompt: str = "") -> str:
        """AI 응답 생성"""
        system = system_prompt or self.participant.system_prompt

        if self.participant.provider == AIProvider.CLAUDE:
            response = await asyncio.to_thread(
                lambda: self.client.messages.create(
                    model=self.participant.model,
                    max_tokens=2500,
                    system=system if system else None,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.content[0].text

        elif self.participant.provider == AIProvider.OPENAI:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=self.participant.model,
                    max_tokens=2500,
                    messages=messages
                )
            )
            return response.choices[0].message.content

        elif self.participant.provider == AIProvider.GEMINI:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = await asyncio.to_thread(
                lambda: self.client.generate_content(full_prompt)
            )
            return response.text

        elif self.participant.provider == AIProvider.PERPLEXITY:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=self.participant.model,
                    max_tokens=2500,
                    messages=messages
                )
            )
            return response.choices[0].message.content

        raise ValueError(f"Unknown provider: {self.participant.provider}")


# ============================================================================
# Base Debate Framework
# ============================================================================

T = TypeVar('T')  # 제안 타입

class DebateFramework(Generic[T], ABC):
    """
    Multi-AI 토론 프레임워크 기본 클래스

    사용법:
    1. 이 클래스를 상속
    2. parse_proposal, evaluate_consensus, merge_proposals 구현
    3. run_debate() 호출

    토론 프로세스:
    Round 1: 각 AI가 의견 제시 (proposal)
    Round 2: 상호 비판 (critique)
    Round 2.5: 반박 (rebuttal) - optional
    Round 3: 합의 도출 (consensus)
    """

    def __init__(
        self,
        participants: List[DebateParticipant],
        config: Optional[DebateConfig] = None
    ):
        self.participants = participants
        self.config = config or DebateConfig()
        self.clients: Dict[str, AIClient] = {
            p.name: AIClient(p) for p in participants
        }
        self.current_phase = DebatePhase.PROPOSAL

    # ========================================================================
    # Abstract Methods (구현 필요)
    # ========================================================================

    @abstractmethod
    def get_proposal_prompt(self, context: Dict[str, Any]) -> str:
        """제안 요청 프롬프트 생성"""
        pass

    @abstractmethod
    def get_critique_prompt(
        self,
        target_opinion: Opinion,
        all_opinions: List[Opinion],
        context: Dict[str, Any]
    ) -> str:
        """비판 요청 프롬프트 생성"""
        pass

    @abstractmethod
    def parse_proposal(self, response: str, participant: str) -> T:
        """AI 응답을 구조화된 제안으로 파싱"""
        pass

    @abstractmethod
    def evaluate_consensus(self, opinions: List[Opinion]) -> ConsensusType:
        """합의 여부 평가"""
        pass

    @abstractmethod
    def merge_proposals(self, opinions: List[Opinion]) -> T:
        """여러 제안을 하나로 병합 (하이브리드 생성)"""
        pass

    # ========================================================================
    # Core Debate Methods
    # ========================================================================

    async def run_debate(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> DebateResult:
        """전체 토론 실행"""

        # Round 1: 의견 제시
        self.current_phase = DebatePhase.PROPOSAL
        opinions = await self._gather_proposals(context)

        # Round 2: 상호 비판
        self.current_phase = DebatePhase.CRITIQUE
        critiques = await self._gather_critiques(opinions, context)

        # Round 2.5: 반박 (optional)
        rebuttals = []
        if self.config.enable_rebuttal and critiques:
            self.current_phase = DebatePhase.REBUTTAL
            rebuttals = await self._gather_rebuttals(opinions, critiques, context)

            # 반박 후 의견 업데이트
            opinions = self._update_opinions_after_rebuttal(opinions, rebuttals)

        # Round 3: 합의 도출
        self.current_phase = DebatePhase.CONSENSUS
        consensus_type = self.evaluate_consensus(opinions)
        final_decision = await self._reach_consensus(opinions, consensus_type)

        # 결과 생성
        self.current_phase = DebatePhase.COMPLETED

        return DebateResult(
            topic=topic,
            consensus_type=consensus_type,
            final_decision=final_decision,
            confidence=self._calculate_confidence(opinions, consensus_type),
            opinions=opinions,
            critiques=critiques,
            rebuttals=rebuttals,
            summary=self._generate_summary(opinions, consensus_type, final_decision),
            dissenting_views=self._extract_dissenting_views(opinions, final_decision)
        )

    async def _gather_proposals(
        self,
        context: Dict[str, Any]
    ) -> List[Opinion]:
        """Round 1: 모든 참여자로부터 제안 수집"""
        prompt = self.get_proposal_prompt(context)

        if self.config.parallel_execution:
            # 병렬 실행
            tasks = [
                self._get_proposal(p.name, prompt)
                for p in self.participants
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # 순차 실행
            results = []
            for p in self.participants:
                result = await self._get_proposal(p.name, prompt)
                results.append(result)

        # 예외 필터링
        opinions = [r for r in results if isinstance(r, Opinion)]
        return opinions

    async def _get_proposal(
        self,
        participant_name: str,
        prompt: str
    ) -> Opinion:
        """단일 참여자로부터 제안 받기"""
        client = self.clients[participant_name]
        participant = next(p for p in self.participants if p.name == participant_name)

        try:
            response = await client.complete(prompt)
            proposal = self.parse_proposal(response, participant_name)

            return Opinion(
                participant=participant_name,
                provider=participant.provider,
                content=proposal,
                reasoning=self._extract_reasoning(response),
                confidence=self._extract_confidence(response),
                evidence=self._extract_evidence(response)
            )
        except Exception as e:
            # 실패 시 기본 의견 반환
            return Opinion(
                participant=participant_name,
                provider=participant.provider,
                content=None,
                reasoning=f"Error: {str(e)}",
                confidence=0.0
            )

    async def _gather_critiques(
        self,
        opinions: List[Opinion],
        context: Dict[str, Any]
    ) -> List[Critique]:
        """Round 2: 상호 비판 수집"""
        critiques = []

        for opinion in opinions:
            # 다른 모든 참여자가 이 의견을 비판
            other_participants = [
                p for p in self.participants
                if p.name != opinion.participant
            ]

            for critic in other_participants:
                prompt = self.get_critique_prompt(opinion, opinions, context)

                try:
                    client = self.clients[critic.name]
                    response = await client.complete(prompt)
                    critique = self._parse_critique(
                        response, critic.name, opinion.participant
                    )
                    critiques.append(critique)
                except Exception:
                    continue

        return critiques

    async def _gather_rebuttals(
        self,
        opinions: List[Opinion],
        critiques: List[Critique],
        context: Dict[str, Any]
    ) -> List[Rebuttal]:
        """Round 2.5: 반박 수집"""
        rebuttals = []

        for opinion in opinions:
            # 이 참여자에 대한 비판 수집
            relevant_critiques = [
                c for c in critiques
                if c.target == opinion.participant
            ]

            if not relevant_critiques:
                continue

            prompt = self._get_rebuttal_prompt(opinion, relevant_critiques, context)

            try:
                client = self.clients[opinion.participant]
                response = await client.complete(prompt)
                rebuttal = self._parse_rebuttal(
                    response, opinion.participant, relevant_critiques
                )
                rebuttals.append(rebuttal)
            except Exception:
                continue

        return rebuttals

    async def _reach_consensus(
        self,
        opinions: List[Opinion],
        consensus_type: ConsensusType
    ) -> T:
        """Round 3: 합의 도출"""
        if consensus_type == ConsensusType.UNANIMOUS:
            # 만장일치: 첫 번째 의견 사용
            return opinions[0].content

        elif consensus_type == ConsensusType.MAJORITY:
            # 다수결: 가장 많은 지지를 받은 의견
            return self._get_majority_opinion(opinions)

        elif consensus_type == ConsensusType.HYBRID:
            # 하이브리드: 여러 의견 병합
            return self.merge_proposals(opinions)

        else:
            # 합의 불가: 가장 높은 신뢰도 의견
            best_opinion = max(opinions, key=lambda o: o.confidence)
            return best_opinion.content

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _extract_reasoning(self, response: str) -> str:
        """응답에서 근거 추출"""
        # 간단한 추출 - 하위 클래스에서 오버라이드 가능
        if "rationale:" in response.lower():
            idx = response.lower().find("rationale:")
            return response[idx:idx+500]
        return response[:500]

    def _extract_confidence(self, response: str) -> float:
        """응답에서 신뢰도 추출"""
        import re
        # confidence: 0.8 형태 찾기
        match = re.search(r'confidence[:\s]+([0-9.]+)', response.lower())
        if match:
            return min(float(match.group(1)), 1.0)
        return 0.7  # 기본값

    def _extract_evidence(self, response: str) -> List[str]:
        """응답에서 근거 목록 추출"""
        evidence = []
        lines = response.split('\n')
        for line in lines:
            if line.strip().startswith('-') or line.strip().startswith('•'):
                evidence.append(line.strip().lstrip('-•').strip())
        return evidence[:5]

    def _parse_critique(
        self,
        response: str,
        critic: str,
        target: str
    ) -> Critique:
        """비판 응답 파싱"""
        points = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                points.append(line.lstrip('0123456789.-) '))

        severity = "moderate"
        if "major" in response.lower() or "critical" in response.lower():
            severity = "major"
        elif "minor" in response.lower():
            severity = "minor"

        return Critique(
            critic=critic,
            target=target,
            points=points[:5],
            severity=severity
        )

    def _get_rebuttal_prompt(
        self,
        opinion: Opinion,
        critiques: List[Critique],
        context: Dict[str, Any]
    ) -> str:
        """반박 프롬프트 생성"""
        critique_summary = "\n".join([
            f"- {c.critic}: {'; '.join(c.points[:2])}"
            for c in critiques
        ])

        return f"""
Your proposal has received the following critiques:

{critique_summary}

Please respond to these critiques:
1. Address valid points and explain how they can be mitigated
2. Defend your position where the criticism is unfounded
3. If needed, revise your proposal

Original proposal: {opinion.content}

Provide a structured rebuttal.
"""

    def _parse_rebuttal(
        self,
        response: str,
        participant: str,
        critiques: List[Critique]
    ) -> Rebuttal:
        """반박 응답 파싱"""
        points = []
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                points.append(line[:200])

        critics = [c.critic for c in critiques]

        return Rebuttal(
            participant=participant,
            responding_to=", ".join(critics),
            points=points[:5]
        )

    def _update_opinions_after_rebuttal(
        self,
        opinions: List[Opinion],
        rebuttals: List[Rebuttal]
    ) -> List[Opinion]:
        """반박 후 의견 업데이트"""
        for rebuttal in rebuttals:
            if rebuttal.revised_position:
                for opinion in opinions:
                    if opinion.participant == rebuttal.participant:
                        opinion.content = rebuttal.revised_position
                        break
        return opinions

    def _get_majority_opinion(self, opinions: List[Opinion]) -> T:
        """다수 의견 추출"""
        # 가장 높은 신뢰도 의견 반환 (단순화)
        best = max(opinions, key=lambda o: o.confidence)
        return best.content

    def _calculate_confidence(
        self,
        opinions: List[Opinion],
        consensus_type: ConsensusType
    ) -> float:
        """전체 신뢰도 계산"""
        if not opinions:
            return 0.0

        avg_confidence = sum(o.confidence for o in opinions) / len(opinions)

        # 합의 유형에 따른 보정
        multipliers = {
            ConsensusType.UNANIMOUS: 1.2,
            ConsensusType.MAJORITY: 1.0,
            ConsensusType.HYBRID: 0.9,
            ConsensusType.NO_CONSENSUS: 0.7
        }
        multiplier = multipliers.get(consensus_type, 1.0)

        return min(avg_confidence * multiplier, 1.0)

    def _generate_summary(
        self,
        opinions: List[Opinion],
        consensus_type: ConsensusType,
        final_decision: T
    ) -> str:
        """토론 요약 생성"""
        participant_names = [o.participant for o in opinions]
        return (
            f"토론 참여자: {', '.join(participant_names)}. "
            f"합의 유형: {consensus_type.value}. "
            f"최종 결정: {str(final_decision)[:100]}..."
        )

    def _extract_dissenting_views(
        self,
        opinions: List[Opinion],
        final_decision: T
    ) -> List[str]:
        """반대 의견 추출"""
        dissenting = []
        for opinion in opinions:
            if opinion.content != final_decision and opinion.confidence > 0.5:
                dissenting.append(
                    f"{opinion.participant}: {str(opinion.content)[:100]}"
                )
        return dissenting


# ============================================================================
# Utility: Default Participants
# ============================================================================

def get_default_participants(
    include_perplexity: bool = False
) -> List[DebateParticipant]:
    """기본 토론 참여자 생성"""
    participants = [
        DebateParticipant(
            name="Claude",
            provider=AIProvider.CLAUDE,
            role="Analytical & Theory-focused",
            system_prompt="You are an analytical AI focusing on economic theory and rigorous methodology."
        ),
        DebateParticipant(
            name="OpenAI",
            provider=AIProvider.OPENAI,
            role="Creative & Practical",
            system_prompt="You are a creative AI focusing on practical applications and novel approaches."
        ),
        DebateParticipant(
            name="Gemini",
            provider=AIProvider.GEMINI,
            role="Data-driven & Technical",
            system_prompt="You are a data-driven AI focusing on quantitative analysis and technical implementation."
        )
    ]

    if include_perplexity:
        participants.append(
            DebateParticipant(
                name="Perplexity",
                provider=AIProvider.PERPLEXITY,
                role="Research & Current Events",
                system_prompt="You are a research-focused AI with access to current information and recent developments."
            )
        )

    return participants


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("DebateFramework module loaded successfully")
    print(f"Available phases: {[p.value for p in DebatePhase]}")
    print(f"Consensus types: {[c.value for c in ConsensusType]}")
    print(f"Default participants: {[p.name for p in get_default_participants()]}")
