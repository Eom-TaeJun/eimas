#!/usr/bin/env python3
"""
Validation - ClaudeValidationAgent
============================================================

ClaudeValidationAgent implementation

Uses claude API for validation
"""

from typing import Dict, Optional
import logging

from .base import BaseValidationAgent
from .schemas import AIValidation
from .enums import ValidationResult

logger = logging.getLogger(__name__)


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
