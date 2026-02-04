#!/usr/bin/env python3
"""
Validation - GeminiValidationAgent
============================================================

GeminiValidationAgent implementation

Uses gemini API for validation
"""

from typing import Dict, Optional
import logging

from .base import BaseValidationAgent
from .schemas import AIValidation
from .enums import ValidationResult

logger = logging.getLogger(__name__)


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
