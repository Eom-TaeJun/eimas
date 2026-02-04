#!/usr/bin/env python3
"""
Validation - GPTValidationAgent
============================================================

GPTValidationAgent implementation

Uses gpt API for validation
"""

from typing import Dict, Optional
import logging

from .base import BaseValidationAgent
from .schemas import AIValidation
from .enums import ValidationResult

logger = logging.getLogger(__name__)


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
