"""
AI Validation Agents - 적응형 에이전트 검증 시스템

4가지 AI를 활용한 투자 전략 검증:
1. Claude (Anthropic) - 깊은 분석, 논리 검증
2. Perplexity - 실시간 시장 정보, 뉴스 검증
3. Gemini (Google) - 다각적 관점, 패턴 인식
4. GPT (OpenAI) - 종합 판단, 리스크 평가

각 AI가 독립적으로 검증 후 합의 도출
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from enum import Enum
import traceback

# API 클라이언트
try:
    from openai import OpenAI
    import anthropic
    import google.generativeai as genai
except ImportError as e:
    print(f"[ValidationAgents] Import warning: {e}")


# ============================================================================
# 검증 결과 구조
# ============================================================================

class ValidationResult(Enum):
    APPROVE = "APPROVE"           # 승인
    REJECT = "REJECT"             # 거부
    MODIFY = "MODIFY"             # 수정 필요
    NEEDS_INFO = "NEEDS_INFO"     # 추가 정보 필요


@dataclass
class AIValidation:
    """개별 AI 검증 결과"""
    ai_name: str
    model: str
    timestamp: str

    # 검증 결과
    result: ValidationResult
    confidence: float           # 0-100%
    reasoning: str              # 판단 근거

    # 수정 제안
    suggested_changes: List[Dict] = field(default_factory=list)
    # [{field: str, current: any, suggested: any, reason: str}]

    # 추가 인사이트
    market_insights: str = ""   # 시장 관련 인사이트
    risk_warnings: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

    # 메타
    response_time_ms: int = 0
    tokens_used: int = 0
    raw_response: str = ""


@dataclass
class ConsensusResult:
    """AI 합의 결과"""
    timestamp: str

    # 개별 결과
    validations: Dict[str, AIValidation] = field(default_factory=dict)

    # 합의
    final_result: ValidationResult = ValidationResult.APPROVE
    consensus_confidence: float = 0.0
    agreement_ratio: float = 0.0  # 동의 비율

    # 통합 제안
    merged_suggestions: List[Dict] = field(default_factory=list)
    key_concerns: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)

    # 요약
    summary: str = ""


# ============================================================================
# 프롬프트 템플릿
# ============================================================================

VALIDATION_PROMPT = """
You are an expert financial analyst validating an AI trading agent's decision.

## Agent Decision to Validate

**Agent Type**: {agent_type}
**Decision**: {action}
**Risk Level**: {risk_level}/100
**Rationale**: {rationale}

## Current Market Condition

- **Regime**: {regime}
- **Risk Score**: {market_risk}/100
- **VIX**: {vix}
- **Volatility**: {volatility}
- **Liquidity Signal**: {liquidity}
- **VPIN Alert**: {vpin_alert}

## Proposed Allocation

{allocation}

## Your Task

Evaluate this decision and respond in JSON format:

```json
{{
    "result": "APPROVE" | "REJECT" | "MODIFY",
    "confidence": 0-100,
    "reasoning": "Your detailed reasoning",
    "suggested_changes": [
        {{"field": "risk_level", "current": X, "suggested": Y, "reason": "..."}}
    ],
    "risk_warnings": ["warning1", "warning2"],
    "opportunities": ["opportunity1"],
    "market_insights": "Your insights about current market"
}}
```

Consider:
1. Is the risk level appropriate for the market condition?
2. Is the asset allocation diversified enough?
3. Are there any obvious risks being ignored?
4. Does the strategy make sense for short-term trading?
"""

PERPLEXITY_MARKET_PROMPT = """
Given the current market conditions and this trading decision, provide real-time validation.

Agent Decision: {action} with {risk_level}% risk level
Market: {regime}, VIX at {vix}, Risk Score {market_risk}/100

Search for:
1. Current market sentiment and news
2. Any breaking events that affect this decision
3. Expert opinions on similar strategies
4. Recent performance of proposed assets

Respond with validation in JSON format:
{{
    "result": "APPROVE" | "REJECT" | "MODIFY",
    "confidence": 0-100,
    "reasoning": "Based on current market data...",
    "market_insights": "Latest relevant information",
    "risk_warnings": [],
    "opportunities": []
}}
"""


# ============================================================================
# 베이스 검증 에이전트
# ============================================================================

class BaseValidationAgent(ABC):
    """검증 에이전트 베이스 클래스"""

    def __init__(self, name: str, model: str):
        self.name = name
        self.model = model
        self.client = None

    @abstractmethod
    def _init_client(self):
        """클라이언트 초기화"""
        pass

    @abstractmethod
    async def validate(
        self,
        agent_decision: Dict,
        market_condition: Dict
    ) -> AIValidation:
        """검증 수행"""
        pass

    def _parse_json_response(self, text: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            # JSON 블록 추출
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                text = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                text = text[start:end].strip()

            return json.loads(text)
        except json.JSONDecodeError:
            # 기본 응답 구조
            return {
                "result": "NEEDS_INFO",
                "confidence": 50,
                "reasoning": text[:500],
                "suggested_changes": [],
                "risk_warnings": [],
                "opportunities": [],
                "market_insights": ""
            }

    def _build_prompt(self, agent_decision: Dict, market_condition: Dict) -> str:
        """프롬프트 생성"""
        allocation_str = "\n".join([
            f"- {ticker}: {weight:.1%}"
            for ticker, weight in agent_decision.get('allocations', {}).items()
        ])

        return VALIDATION_PROMPT.format(
            agent_type=agent_decision.get('agent_type', 'unknown'),
            action=agent_decision.get('action', 'unknown'),
            risk_level=agent_decision.get('risk_level', 50),
            rationale=agent_decision.get('rationale', 'N/A'),
            regime=market_condition.get('regime', 'Unknown'),
            market_risk=market_condition.get('risk_score', 50),
            vix=market_condition.get('vix_level', 20),
            volatility=market_condition.get('volatility', 'Medium'),
            liquidity=market_condition.get('liquidity_signal', 'NEUTRAL'),
            vpin_alert=market_condition.get('vpin_alert', False),
            allocation=allocation_str or "No specific allocation"
        )


# ============================================================================
# Claude 검증 에이전트
# ============================================================================

class ClaudeValidationAgent(BaseValidationAgent):
    """Claude 기반 검증 에이전트 - 깊은 분석 (Opus 4.5)"""

    def __init__(self):
        # Claude Opus 4.5 - 최고 성능 모델 (2025년 11월 버전)
        super().__init__("Claude", "claude-opus-4-5-20251101")
        self._init_client()

    def _init_client(self):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)

    async def validate(
        self,
        agent_decision: Dict,
        market_condition: Dict
    ) -> AIValidation:
        if not self.client:
            return self._error_validation("Claude API key not configured")

        start_time = datetime.now()
        prompt = self._build_prompt(agent_decision, market_condition)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.2,
                system="You are a senior quantitative analyst specializing in risk management and portfolio optimization. Be critical and thorough in your analysis.",
                messages=[{"role": "user", "content": prompt}]
            )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            raw_text = response.content[0].text
            parsed = self._parse_json_response(raw_text)

            return AIValidation(
                ai_name=self.name,
                model=self.model,
                timestamp=datetime.now().isoformat(),
                result=ValidationResult[parsed.get('result', 'APPROVE')],
                confidence=parsed.get('confidence', 70),
                reasoning=parsed.get('reasoning', ''),
                suggested_changes=parsed.get('suggested_changes', []),
                market_insights=parsed.get('market_insights', ''),
                risk_warnings=parsed.get('risk_warnings', []),
                opportunities=parsed.get('opportunities', []),
                response_time_ms=int(elapsed),
                tokens_used=response.usage.output_tokens if hasattr(response, 'usage') else 0,
                raw_response=raw_text
            )

        except Exception as e:
            return self._error_validation(str(e))

    def _error_validation(self, error: str) -> AIValidation:
        return AIValidation(
            ai_name=self.name,
            model=self.model,
            timestamp=datetime.now().isoformat(),
            result=ValidationResult.NEEDS_INFO,
            confidence=0,
            reasoning=f"Error: {error}",
            raw_response=error
        )


# ============================================================================
# Perplexity 검증 에이전트
# ============================================================================

class PerplexityValidationAgent(BaseValidationAgent):
    """Perplexity 기반 검증 에이전트 - 실시간 시장 정보"""

    def __init__(self):
        # Sonar Pro - 실시간 검색 + 분석 (reasoning 모델은 금융조언 거부)
        super().__init__("Perplexity", "sonar-pro")
        self._init_client()

    def _init_client(self):
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if api_key:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )

    async def validate(
        self,
        agent_decision: Dict,
        market_condition: Dict
    ) -> AIValidation:
        if not self.client:
            return self._error_validation("Perplexity API key not configured")

        start_time = datetime.now()

        prompt = PERPLEXITY_MARKET_PROMPT.format(
            action=agent_decision.get('action', 'unknown'),
            risk_level=agent_decision.get('risk_level', 50),
            regime=market_condition.get('regime', 'Unknown'),
            vix=market_condition.get('vix_level', 20),
            market_risk=market_condition.get('risk_score', 50)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial market analyst with access to real-time data. Validate trading decisions based on current market conditions."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            raw_text = response.choices[0].message.content
            parsed = self._parse_json_response(raw_text)

            return AIValidation(
                ai_name=self.name,
                model=self.model,
                timestamp=datetime.now().isoformat(),
                result=ValidationResult[parsed.get('result', 'APPROVE')],
                confidence=parsed.get('confidence', 70),
                reasoning=parsed.get('reasoning', ''),
                suggested_changes=parsed.get('suggested_changes', []),
                market_insights=parsed.get('market_insights', ''),
                risk_warnings=parsed.get('risk_warnings', []),
                opportunities=parsed.get('opportunities', []),
                response_time_ms=int(elapsed),
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0,
                raw_response=raw_text
            )

        except Exception as e:
            return self._error_validation(str(e))

    def _error_validation(self, error: str) -> AIValidation:
        return AIValidation(
            ai_name=self.name,
            model=self.model,
            timestamp=datetime.now().isoformat(),
            result=ValidationResult.NEEDS_INFO,
            confidence=0,
            reasoning=f"Error: {error}",
            raw_response=error
        )


# ============================================================================
# Gemini 검증 에이전트
# ============================================================================

class GeminiValidationAgent(BaseValidationAgent):
    """Gemini 기반 검증 에이전트 - 다각적 관점 (Gemini 2.5 Pro)"""

    def __init__(self):
        # Gemini 2.5 Pro - 최신 실험 모델 (Thinking 포함)
        super().__init__("Gemini", "gemini-2.5-pro-exp-03-25")
        self._init_client()

    def _init_client(self):
        api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)

    async def validate(
        self,
        agent_decision: Dict,
        market_condition: Dict
    ) -> AIValidation:
        if not self.client:
            return self._error_validation("Gemini API key not configured")

        start_time = datetime.now()
        prompt = self._build_prompt(agent_decision, market_condition)

        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            raw_text = response.text
            parsed = self._parse_json_response(raw_text)

            return AIValidation(
                ai_name=self.name,
                model=self.model,
                timestamp=datetime.now().isoformat(),
                result=ValidationResult[parsed.get('result', 'APPROVE')],
                confidence=parsed.get('confidence', 70),
                reasoning=parsed.get('reasoning', ''),
                suggested_changes=parsed.get('suggested_changes', []),
                market_insights=parsed.get('market_insights', ''),
                risk_warnings=parsed.get('risk_warnings', []),
                opportunities=parsed.get('opportunities', []),
                response_time_ms=int(elapsed),
                raw_response=raw_text
            )

        except Exception as e:
            return self._error_validation(str(e))

    def _error_validation(self, error: str) -> AIValidation:
        return AIValidation(
            ai_name=self.name,
            model=self.model,
            timestamp=datetime.now().isoformat(),
            result=ValidationResult.NEEDS_INFO,
            confidence=0,
            reasoning=f"Error: {error}",
            raw_response=error
        )


# ============================================================================
# OpenAI GPT 검증 에이전트
# ============================================================================

class GPTValidationAgent(BaseValidationAgent):
    """GPT 기반 검증 에이전트 - 종합 판단 (o1 Reasoning)"""

    def __init__(self):
        # OpenAI o1 - 최고 성능 reasoning 모델
        super().__init__("GPT", "o1")
        self._init_client()

    def _init_client(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.client = OpenAI(api_key=api_key)

    async def validate(
        self,
        agent_decision: Dict,
        market_condition: Dict
    ) -> AIValidation:
        if not self.client:
            return self._error_validation("OpenAI API key not configured")

        start_time = datetime.now()
        base_prompt = self._build_prompt(agent_decision, market_condition)

        # o1 모델은 system message 미지원 - 프롬프트에 역할 포함
        prompt = f"""You are an expert portfolio manager and risk analyst. Provide critical and balanced assessment of trading decisions.

{base_prompt}"""

        try:
            # o1 모델: temperature, max_tokens 미지원 → max_completion_tokens 사용
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=4000  # o1은 reasoning tokens도 필요
            )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            raw_text = response.choices[0].message.content
            parsed = self._parse_json_response(raw_text)

            return AIValidation(
                ai_name=self.name,
                model=self.model,
                timestamp=datetime.now().isoformat(),
                result=ValidationResult[parsed.get('result', 'APPROVE')],
                confidence=parsed.get('confidence', 70),
                reasoning=parsed.get('reasoning', ''),
                suggested_changes=parsed.get('suggested_changes', []),
                market_insights=parsed.get('market_insights', ''),
                risk_warnings=parsed.get('risk_warnings', []),
                opportunities=parsed.get('opportunities', []),
                response_time_ms=int(elapsed),
                tokens_used=response.usage.total_tokens if hasattr(response, 'usage') else 0,
                raw_response=raw_text
            )

        except Exception as e:
            return self._error_validation(str(e))

    def _error_validation(self, error: str) -> AIValidation:
        return AIValidation(
            ai_name=self.name,
            model=self.model,
            timestamp=datetime.now().isoformat(),
            result=ValidationResult.NEEDS_INFO,
            confidence=0,
            reasoning=f"Error: {error}",
            raw_response=error
        )


# ============================================================================
# 합의 엔진
# ============================================================================

class ConsensusEngine:
    """AI 검증 결과 합의 도출"""

    def __init__(self):
        self.weights = {
            'Claude': 0.30,      # 깊은 분석
            'Perplexity': 0.25,  # 실시간 정보
            'GPT': 0.25,         # 종합 판단
            'Gemini': 0.20       # 다각적 관점
        }

    def reach_consensus(self, validations: Dict[str, AIValidation]) -> ConsensusResult:
        """합의 도출"""
        timestamp = datetime.now().isoformat()

        if not validations:
            return ConsensusResult(
                timestamp=timestamp,
                summary="No validations available"
            )

        # 1. 결과별 점수 계산
        result_scores = {
            ValidationResult.APPROVE: 0,
            ValidationResult.MODIFY: 0,
            ValidationResult.REJECT: 0,
            ValidationResult.NEEDS_INFO: 0
        }

        total_weight = 0
        weighted_confidence = 0

        for ai_name, validation in validations.items():
            weight = self.weights.get(ai_name, 0.2)
            total_weight += weight

            # 결과에 가중치 적용
            result_scores[validation.result] += weight * (validation.confidence / 100)

            # 가중 신뢰도
            weighted_confidence += weight * validation.confidence

        # 2. 최종 결과 결정
        best_result = max(result_scores.items(), key=lambda x: x[1])
        final_result = best_result[0]

        # 3. 동의 비율 계산
        agreed_count = sum(
            1 for v in validations.values()
            if v.result == final_result
        )
        agreement_ratio = agreed_count / len(validations) if validations else 0

        # 4. 평균 신뢰도
        consensus_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

        # 5. 제안 사항 병합
        all_suggestions = []
        all_warnings = []
        all_opportunities = []

        for validation in validations.values():
            all_suggestions.extend(validation.suggested_changes)
            all_warnings.extend(validation.risk_warnings)
            all_opportunities.extend(validation.opportunities)

        # 중복 제거
        unique_warnings = list(set(all_warnings))
        unique_opportunities = list(set(all_opportunities))

        # 6. 요약 생성
        summary = self._generate_summary(
            final_result, agreement_ratio, consensus_confidence,
            unique_warnings, validations
        )

        # 7. 액션 아이템 생성
        action_items = self._generate_action_items(
            final_result, all_suggestions, unique_warnings
        )

        return ConsensusResult(
            timestamp=timestamp,
            validations=validations,
            final_result=final_result,
            consensus_confidence=consensus_confidence,
            agreement_ratio=agreement_ratio,
            merged_suggestions=all_suggestions[:10],  # 상위 10개
            key_concerns=unique_warnings[:5],
            action_items=action_items,
            summary=summary
        )

    def _generate_summary(
        self,
        result: ValidationResult,
        agreement: float,
        confidence: float,
        warnings: List[str],
        validations: Dict
    ) -> str:
        """요약 생성"""
        parts = []

        # 결과
        if result == ValidationResult.APPROVE:
            parts.append(f"✓ 승인 (동의율 {agreement:.0%}, 신뢰도 {confidence:.0f}%)")
        elif result == ValidationResult.REJECT:
            parts.append(f"✗ 거부 (동의율 {agreement:.0%}, 신뢰도 {confidence:.0f}%)")
        elif result == ValidationResult.MODIFY:
            parts.append(f"⚠ 수정 필요 (동의율 {agreement:.0%}, 신뢰도 {confidence:.0f}%)")
        else:
            parts.append(f"? 추가 정보 필요 (동의율 {agreement:.0%})")

        # 주요 우려사항
        if warnings:
            parts.append(f"주요 우려: {warnings[0]}")

        # AI별 판단
        ai_results = [f"{k}: {v.result.value}" for k, v in validations.items()]
        parts.append(f"AI 판단: {', '.join(ai_results)}")

        return " | ".join(parts)

    def _generate_action_items(
        self,
        result: ValidationResult,
        suggestions: List[Dict],
        warnings: List[str]
    ) -> List[str]:
        """액션 아이템 생성"""
        items = []

        if result == ValidationResult.REJECT:
            items.append("즉시 전략 재검토 필요")
            items.append("리스크 레벨 하향 조정 검토")

        elif result == ValidationResult.MODIFY:
            for s in suggestions[:3]:
                items.append(f"{s.get('field', 'Unknown')} 수정: {s.get('reason', 'N/A')}")

        if warnings:
            items.append(f"리스크 모니터링: {warnings[0]}")

        return items


# ============================================================================
# 검증 매니저
# ============================================================================

class ValidationAgentManager:
    """검증 에이전트 통합 관리"""

    def __init__(self):
        self.agents = {}
        self.consensus_engine = ConsensusEngine()

        # 사용 가능한 에이전트 초기화
        self._init_agents()

    def _init_agents(self):
        """에이전트 초기화"""
        # Claude
        if os.getenv('ANTHROPIC_API_KEY'):
            self.agents['Claude'] = ClaudeValidationAgent()
            print("[Validation] Claude agent initialized")

        # Perplexity
        if os.getenv('PERPLEXITY_API_KEY'):
            self.agents['Perplexity'] = PerplexityValidationAgent()
            print("[Validation] Perplexity agent initialized")

        # Gemini
        if os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY'):
            self.agents['Gemini'] = GeminiValidationAgent()
            print("[Validation] Gemini agent initialized")

        # GPT
        if os.getenv('OPENAI_API_KEY'):
            self.agents['GPT'] = GPTValidationAgent()
            print("[Validation] GPT agent initialized")

        if not self.agents:
            print("[Validation] Warning: No API keys configured!")

    async def validate_decision(
        self,
        agent_decision: Dict,
        market_condition: Dict
    ) -> ConsensusResult:
        """결정 검증 (모든 AI 병렬 실행)"""
        if not self.agents:
            return ConsensusResult(
                timestamp=datetime.now().isoformat(),
                summary="No validation agents available"
            )

        # 병렬 검증
        tasks = []
        agent_names = []

        for name, agent in self.agents.items():
            tasks.append(agent.validate(agent_decision, market_condition))
            agent_names.append(name)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 수집
        validations = {}
        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                validations[name] = AIValidation(
                    ai_name=name,
                    model="error",
                    timestamp=datetime.now().isoformat(),
                    result=ValidationResult.NEEDS_INFO,
                    confidence=0,
                    reasoning=f"Error: {str(result)}"
                )
            else:
                validations[name] = result

        # 합의 도출
        return self.consensus_engine.reach_consensus(validations)

    def validate_all(
        self,
        agent_decision: Dict,
        market_condition: Dict
    ) -> ConsensusResult:
        """동기 래퍼: validate_decision을 동기적으로 호출"""
        import nest_asyncio
        nest_asyncio.apply()

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.validate_decision(agent_decision, market_condition)
            )
        finally:
            loop.close()

    def get_report(self, consensus: ConsensusResult) -> str:
        """검증 리포트 생성"""
        lines = []
        lines.append("=" * 70)
        lines.append("AI VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {consensus.timestamp}")
        lines.append("")

        # 최종 결과
        lines.append("[CONSENSUS]")
        lines.append(f"  Result: {consensus.final_result.value}")
        lines.append(f"  Confidence: {consensus.consensus_confidence:.1f}%")
        lines.append(f"  Agreement: {consensus.agreement_ratio:.0%}")
        lines.append(f"  Summary: {consensus.summary}")
        lines.append("")

        # 개별 AI 결과
        lines.append("[INDIVIDUAL VALIDATIONS]")
        for name, v in consensus.validations.items():
            lines.append(f"\n  [{name}] - {v.model}")
            lines.append(f"    Result: {v.result.value} ({v.confidence:.0f}%)")
            lines.append(f"    Time: {v.response_time_ms}ms")
            lines.append(f"    Reasoning: {v.reasoning[:100]}...")
            if v.risk_warnings:
                lines.append(f"    Warnings: {v.risk_warnings[:2]}")
        lines.append("")

        # 주요 우려사항
        if consensus.key_concerns:
            lines.append("[KEY CONCERNS]")
            for i, concern in enumerate(consensus.key_concerns, 1):
                lines.append(f"  {i}. {concern}")
            lines.append("")

        # 액션 아이템
        if consensus.action_items:
            lines.append("[ACTION ITEMS]")
            for i, item in enumerate(consensus.action_items, 1):
                lines.append(f"  {i}. {item}")
            lines.append("")

        # 수정 제안
        if consensus.merged_suggestions:
            lines.append("[SUGGESTED CHANGES]")
            for s in consensus.merged_suggestions[:5]:
                lines.append(f"  - {s.get('field', 'N/A')}: {s.get('reason', 'N/A')}")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 피드백 검증 에이전트 (수정 제안 검증)
# ============================================================================

FEEDBACK_PROMPT = """
You are validating MODIFICATION SUGGESTIONS made by other AI agents.

## Original Decision
- Agent: {agent_type}
- Action: {action}
- Risk Level: {risk_level}/100

## Market Context
- Regime: {regime}
- Risk Score: {market_risk}/100
- VIX: {vix}

## Suggested Modifications to Validate
{suggestions}

## Your Task

Evaluate each suggestion and respond in JSON:

```json
{{
    "overall_verdict": "APPROVE" | "REJECT" | "PARTIAL",
    "confidence": 0-100,
    "suggestion_verdicts": [
        {{"field": "...", "verdict": "APPROVE/REJECT", "reason": "..."}}
    ],
    "final_recommendations": [
        {{"field": "...", "value": ..., "reason": "..."}}
    ],
    "rationale": "Overall reasoning for your decision"
}}
```

Be critical. Only approve suggestions that are:
1. Economically sound
2. Consistent with market conditions
3. Actionable and specific
"""


@dataclass
class FeedbackResult:
    """피드백 검증 결과"""
    ai_name: str
    model: str
    timestamp: str

    overall_verdict: str  # APPROVE, REJECT, PARTIAL
    confidence: float
    suggestion_verdicts: List[Dict]  # 개별 제안 검증
    final_recommendations: List[Dict]  # 최종 권고
    rationale: str

    response_time_ms: int = 0
    raw_response: str = ""


@dataclass
class FeedbackLoopResult:
    """피드백 루프 최종 결과"""
    timestamp: str
    rounds_completed: int
    max_rounds: int

    # 최종 결정
    final_decision: Dict
    original_decision: Dict

    # 변경 이력
    modification_history: List[Dict] = field(default_factory=list)

    # AI 피드백
    feedback_results: Dict[str, FeedbackResult] = field(default_factory=dict)

    # 합의
    consensus_reached: bool = False
    final_confidence: float = 0.0

    summary: str = ""


class FeedbackValidationAgent:
    """
    피드백 검증 에이전트 - 수정 제안 검증

    Claude와 Perplexity가 다른 AI들의 수정 제안을 검증
    """

    def __init__(self):
        self.claude_agent = None
        self.perplexity_agent = None
        self._init_agents()

    def _init_agents(self):
        """피드백 에이전트 초기화 (Claude + Perplexity만)"""
        if os.getenv('ANTHROPIC_API_KEY'):
            self.claude_agent = ClaudeValidationAgent()
            print("[Feedback] Claude feedback agent initialized")

        if os.getenv('PERPLEXITY_API_KEY'):
            self.perplexity_agent = PerplexityValidationAgent()
            print("[Feedback] Perplexity feedback agent initialized")

    def _build_feedback_prompt(
        self,
        original_decision: Dict,
        market_condition: Dict,
        suggestions: List[Dict]
    ) -> str:
        """피드백 프롬프트 생성"""
        # 제안 포맷팅
        suggestion_text = ""
        for i, s in enumerate(suggestions, 1):
            suggestion_text += f"\n{i}. **{s.get('field', 'Unknown')}**\n"
            suggestion_text += f"   - Current: {s.get('current', 'N/A')}\n"
            suggestion_text += f"   - Suggested: {s.get('suggested', 'N/A')}\n"
            suggestion_text += f"   - Reason: {s.get('reason', 'N/A')}\n"

        return FEEDBACK_PROMPT.format(
            agent_type=original_decision.get('agent_type', 'unknown'),
            action=original_decision.get('action', 'unknown'),
            risk_level=original_decision.get('risk_level', 50),
            regime=market_condition.get('regime', 'Unknown'),
            market_risk=market_condition.get('risk_score', 50),
            vix=market_condition.get('vix_level', 20),
            suggestions=suggestion_text if suggestion_text else "No suggestions provided"
        )

    async def _validate_with_claude(
        self,
        prompt: str
    ) -> FeedbackResult:
        """Claude로 피드백 검증"""
        if not self.claude_agent or not self.claude_agent.client:
            return self._error_result("Claude", "API not configured")

        start_time = datetime.now()

        try:
            response = self.claude_agent.client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=2000,
                temperature=0.1,
                system="You are a senior risk manager validating AI-generated trading suggestions. Be critical and precise.",
                messages=[{"role": "user", "content": prompt}]
            )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            raw_text = response.content[0].text
            parsed = self._parse_response(raw_text)

            return FeedbackResult(
                ai_name="Claude",
                model="claude-opus-4-5-20251101",
                timestamp=datetime.now().isoformat(),
                overall_verdict=parsed.get('overall_verdict', 'PARTIAL'),
                confidence=parsed.get('confidence', 50),
                suggestion_verdicts=parsed.get('suggestion_verdicts', []),
                final_recommendations=parsed.get('final_recommendations', []),
                rationale=parsed.get('rationale', ''),
                response_time_ms=int(elapsed),
                raw_response=raw_text
            )

        except Exception as e:
            return self._error_result("Claude", str(e))

    async def _validate_with_perplexity(
        self,
        prompt: str
    ) -> FeedbackResult:
        """Perplexity로 피드백 검증 (실시간 데이터 기반)"""
        if not self.perplexity_agent or not self.perplexity_agent.client:
            return self._error_result("Perplexity", "API not configured")

        start_time = datetime.now()

        # Perplexity용 프롬프트 (실시간 검색 강조)
        enhanced_prompt = f"""
{prompt}

Additionally, search for:
1. Recent market news that might affect these suggestions
2. Current analyst consensus on mentioned assets
3. Any recent volatility events or warnings
"""

        try:
            response = self.perplexity_agent.client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {
                        "role": "system",
                        "content": "You are validating trading suggestions with real-time market data. Be factual and cite sources."
                    },
                    {"role": "user", "content": enhanced_prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            raw_text = response.choices[0].message.content
            parsed = self._parse_response(raw_text)

            return FeedbackResult(
                ai_name="Perplexity",
                model="sonar-pro",
                timestamp=datetime.now().isoformat(),
                overall_verdict=parsed.get('overall_verdict', 'PARTIAL'),
                confidence=parsed.get('confidence', 50),
                suggestion_verdicts=parsed.get('suggestion_verdicts', []),
                final_recommendations=parsed.get('final_recommendations', []),
                rationale=parsed.get('rationale', ''),
                response_time_ms=int(elapsed),
                raw_response=raw_text
            )

        except Exception as e:
            return self._error_result("Perplexity", str(e))

    def _parse_response(self, text: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            # JSON 블록 추출
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                # JSON 객체 직접 찾기
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]

            return json.loads(json_str)
        except:
            return {
                'overall_verdict': 'PARTIAL',
                'confidence': 50,
                'rationale': text[:500]
            }

    def _error_result(self, ai_name: str, error: str) -> FeedbackResult:
        """에러 결과 생성"""
        return FeedbackResult(
            ai_name=ai_name,
            model="error",
            timestamp=datetime.now().isoformat(),
            overall_verdict="PARTIAL",
            confidence=0,
            suggestion_verdicts=[],
            final_recommendations=[],
            rationale=f"Error: {error}"
        )

    async def validate_suggestions(
        self,
        original_decision: Dict,
        market_condition: Dict,
        suggestions: List[Dict]
    ) -> Dict[str, FeedbackResult]:
        """수정 제안 검증 (Claude + Perplexity 병렬)"""
        prompt = self._build_feedback_prompt(
            original_decision, market_condition, suggestions
        )

        # 병렬 실행
        tasks = []
        if self.claude_agent:
            tasks.append(self._validate_with_claude(prompt))
        if self.perplexity_agent:
            tasks.append(self._validate_with_perplexity(prompt))

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        feedback_results = {}
        for result in results:
            if isinstance(result, FeedbackResult):
                feedback_results[result.ai_name] = result
            elif isinstance(result, Exception):
                print(f"[Feedback] Error: {result}")

        return feedback_results

    def merge_feedback(
        self,
        feedback_results: Dict[str, FeedbackResult]
    ) -> Tuple[str, float, List[Dict]]:
        """피드백 결과 병합"""
        if not feedback_results:
            return "PARTIAL", 50.0, []

        # 가중치
        weights = {'Claude': 0.55, 'Perplexity': 0.45}

        verdict_scores = {'APPROVE': 0, 'REJECT': 0, 'PARTIAL': 0}
        total_confidence = 0
        total_weight = 0
        all_recommendations = []

        for name, result in feedback_results.items():
            weight = weights.get(name, 0.5)
            verdict_scores[result.overall_verdict] += weight
            total_confidence += result.confidence * weight
            total_weight += weight
            all_recommendations.extend(result.final_recommendations)

        # 최종 verdict
        final_verdict = max(verdict_scores, key=verdict_scores.get)
        avg_confidence = total_confidence / total_weight if total_weight > 0 else 50

        # 중복 제거된 권고
        seen = set()
        unique_recs = []
        for rec in all_recommendations:
            key = rec.get('field', '')
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)

        return final_verdict, avg_confidence, unique_recs


# ============================================================================
# 검증 루프 매니저
# ============================================================================

class ValidationLoopManager:
    """
    검증 피드백 루프 관리

    Flow:
    1. Adaptive Agent → 결정
    2. Validation Agents (4개) → 검증 + 수정 제안
    3. Feedback Agents (Claude + Perplexity) → 수정 제안 검증
    4. 합의 도달 시 → 최종 결정 반환
    5. 미합의 시 → 수정 적용 후 재검증 (최대 3라운드)
    """

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.validation_manager = ValidationAgentManager()
        self.feedback_agent = FeedbackValidationAgent()
        print(f"[Loop] ValidationLoopManager initialized (max_rounds={max_rounds})")

    async def run_validation_loop(
        self,
        original_decision: Dict,
        market_condition: Dict
    ) -> FeedbackLoopResult:
        """전체 검증 루프 실행"""
        timestamp = datetime.now().isoformat()
        current_decision = original_decision.copy()
        modification_history = []

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n[Loop] === Round {round_num}/{self.max_rounds} ===")

            # Step 1: 4개 AI 검증
            print("[Loop] Step 1: Running validation agents...")
            consensus = await self.validation_manager.validate_decision(
                current_decision, market_condition
            )

            # 승인이면 즉시 종료
            if consensus.final_result == ValidationResult.APPROVE:
                print(f"[Loop] ✓ Decision APPROVED at round {round_num}")
                return FeedbackLoopResult(
                    timestamp=timestamp,
                    rounds_completed=round_num,
                    max_rounds=self.max_rounds,
                    final_decision=current_decision,
                    original_decision=original_decision,
                    modification_history=modification_history,
                    consensus_reached=True,
                    final_confidence=consensus.consensus_confidence,
                    summary=f"✓ 승인 (Round {round_num}, 신뢰도 {consensus.consensus_confidence:.0f}%)"
                )

            # 거부면 원본 유지하고 종료
            if consensus.final_result == ValidationResult.REJECT:
                print(f"[Loop] ✗ Decision REJECTED at round {round_num}")
                return FeedbackLoopResult(
                    timestamp=timestamp,
                    rounds_completed=round_num,
                    max_rounds=self.max_rounds,
                    final_decision=original_decision,  # 원본 반환
                    original_decision=original_decision,
                    modification_history=modification_history,
                    consensus_reached=False,
                    final_confidence=consensus.consensus_confidence,
                    summary=f"✗ 거부됨 - 원본 결정 유지 (Round {round_num})"
                )

            # Step 2: 수정 제안이 있으면 피드백 검증
            suggestions = consensus.merged_suggestions
            if not suggestions:
                print("[Loop] No suggestions to validate")
                break

            print(f"[Loop] Step 2: Validating {len(suggestions)} suggestions...")
            feedback_results = await self.feedback_agent.validate_suggestions(
                current_decision, market_condition, suggestions
            )

            # Step 3: 피드백 병합
            verdict, confidence, final_recs = self.feedback_agent.merge_feedback(
                feedback_results
            )

            print(f"[Loop] Feedback verdict: {verdict} ({confidence:.0f}%)")

            # Step 4: 수정 적용
            if verdict == "APPROVE" and final_recs:
                print(f"[Loop] Applying {len(final_recs)} modifications...")

                for rec in final_recs:
                    field = rec.get('field', '')
                    value = rec.get('value')

                    # 수정 이력 기록
                    modification_history.append({
                        'round': round_num,
                        'field': field,
                        'old_value': current_decision.get(field) or current_decision.get('allocations', {}).get(field),
                        'new_value': value,
                        'reason': rec.get('reason', '')
                    })

                    # 실제 수정 적용
                    if field == 'risk_level':
                        current_decision['risk_level'] = value
                    elif field in current_decision.get('allocations', {}):
                        current_decision['allocations'][field] = value

                print(f"[Loop] Modifications applied, continuing to next round...")

            elif verdict == "REJECT":
                print("[Loop] Suggestions rejected, keeping current decision")
                break

            else:
                print("[Loop] Partial approval, applying conservative adjustments...")
                # 부분 승인: 보수적으로 리스크만 10% 감소
                if 'risk_level' in current_decision:
                    old_risk = current_decision['risk_level']
                    new_risk = max(30, old_risk - 10)
                    modification_history.append({
                        'round': round_num,
                        'field': 'risk_level',
                        'old_value': old_risk,
                        'new_value': new_risk,
                        'reason': 'Partial approval - conservative adjustment'
                    })
                    current_decision['risk_level'] = new_risk

        # 루프 완료
        return FeedbackLoopResult(
            timestamp=timestamp,
            rounds_completed=self.max_rounds,
            max_rounds=self.max_rounds,
            final_decision=current_decision,
            original_decision=original_decision,
            modification_history=modification_history,
            feedback_results=feedback_results if 'feedback_results' in dir() else {},
            consensus_reached=len(modification_history) > 0,
            final_confidence=confidence if 'confidence' in dir() else 50.0,
            summary=f"루프 완료 ({len(modification_history)}개 수정 적용)"
        )

    def get_loop_report(self, result: FeedbackLoopResult) -> str:
        """루프 결과 리포트 생성"""
        lines = []
        lines.append("=" * 70)
        lines.append("VALIDATION LOOP REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append(f"Rounds: {result.rounds_completed}/{result.max_rounds}")
        lines.append(f"Consensus: {'Yes' if result.consensus_reached else 'No'}")
        lines.append(f"Final Confidence: {result.final_confidence:.1f}%")
        lines.append("")

        # 결정 비교
        lines.append("[DECISION COMPARISON]")
        lines.append(f"  Original Risk: {result.original_decision.get('risk_level')}")
        lines.append(f"  Final Risk: {result.final_decision.get('risk_level')}")
        lines.append("")

        # 수정 이력
        if result.modification_history:
            lines.append("[MODIFICATION HISTORY]")
            for mod in result.modification_history:
                lines.append(f"  Round {mod['round']}: {mod['field']}")
                lines.append(f"    {mod['old_value']} → {mod['new_value']}")
                lines.append(f"    Reason: {mod['reason']}")
            lines.append("")

        # 피드백 결과
        if result.feedback_results:
            lines.append("[FEEDBACK RESULTS]")
            for name, fb in result.feedback_results.items():
                lines.append(f"  [{name}] {fb.overall_verdict} ({fb.confidence:.0f}%)")
                lines.append(f"    {fb.rationale[:100]}...")
            lines.append("")

        lines.append(f"[SUMMARY] {result.summary}")
        lines.append("=" * 70)

        return "\n".join(lines)


# ============================================================================
# 테스트
# ============================================================================

async def test_validation():
    """단일 라운드 검증 테스트"""
    print("AI Validation Agents Test (Single Round)")
    print("=" * 60)

    manager = ValidationAgentManager()

    if not manager.agents:
        print("No agents available. Check API keys.")
        return

    print(f"\nActive agents: {list(manager.agents.keys())}")

    agent_decision = {
        'agent_type': 'aggressive',
        'action': 'AGGRESSIVE_ENTRY',
        'risk_level': 85,
        'rationale': 'Bull market with low VIX, high opportunity score',
        'allocations': {
            'TQQQ': 0.25,
            'SOXL': 0.20,
            'BTC-USD': 0.20,
            'QQQ': 0.20,
            'GLD': 0.15
        }
    }

    market_condition = {
        'regime': 'Bull (Low Vol)',
        'risk_score': 15,
        'vix_level': 14.5,
        'volatility': 'Low',
        'liquidity_signal': 'RISK_ON',
        'vpin_alert': False
    }

    print("\n[Testing Single Round Validation...]")
    print(f"Decision: {agent_decision['action']}")
    print(f"Risk Level: {agent_decision['risk_level']}")

    consensus = await manager.validate_decision(agent_decision, market_condition)
    print("\n" + manager.get_report(consensus))


async def test_validation_loop():
    """피드백 루프 검증 테스트"""
    print("\n" + "=" * 70)
    print("VALIDATION LOOP TEST (with Feedback)")
    print("=" * 70)

    loop_manager = ValidationLoopManager(max_rounds=3)

    agent_decision = {
        'agent_type': 'aggressive',
        'action': 'AGGRESSIVE_ENTRY',
        'risk_level': 85,
        'rationale': 'Bull market with low VIX, high opportunity score',
        'allocations': {
            'TQQQ': 0.25,
            'SOXL': 0.20,
            'BTC-USD': 0.20,
            'QQQ': 0.20,
            'GLD': 0.15
        }
    }

    market_condition = {
        'regime': 'Bull (Low Vol)',
        'risk_score': 15,
        'vix_level': 14.5,
        'volatility': 'Low',
        'liquidity_signal': 'RISK_ON',
        'vpin_alert': False
    }

    print(f"\nOriginal Decision: {agent_decision['action']}")
    print(f"Original Risk Level: {agent_decision['risk_level']}")
    print("\nRunning validation loop...")

    result = await loop_manager.run_validation_loop(agent_decision, market_condition)

    print("\n" + loop_manager.get_loop_report(result))

    # 결과 요약
    print("\n[FINAL COMPARISON]")
    print(f"  Risk Level: {agent_decision['risk_level']} → {result.final_decision.get('risk_level')}")
    print(f"  Rounds: {result.rounds_completed}/{result.max_rounds}")
    print(f"  Modifications: {len(result.modification_history)}")


async def main():
    """메인 테스트 실행"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        # 피드백 루프 테스트
        await test_validation_loop()
    else:
        # 단일 라운드 테스트
        await test_validation()


if __name__ == "__main__":
    asyncio.run(main())
