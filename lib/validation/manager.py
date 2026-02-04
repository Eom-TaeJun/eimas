#!/usr/bin/env python3
"""
Validation - Agent Manager
============================================================

Manages multiple validation agents

Class:
    - ValidationAgentManager: Coordinates all validation agents
"""

from typing import List, Dict, Optional
import logging

from .base import BaseValidationAgent
from .claude import ClaudeValidationAgent
from .perplexity import PerplexityValidationAgent
from .gemini import GeminiValidationAgent
from .gpt import GPTValidationAgent
from .consensus import ConsensusEngine
from .schemas import AIValidation, ConsensusResult

logger = logging.getLogger(__name__)


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


