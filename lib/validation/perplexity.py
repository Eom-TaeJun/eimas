#!/usr/bin/env python3
"""
Validation - PerplexityValidationAgent
============================================================

PerplexityValidationAgent implementation

Uses perplexity API for validation
"""

from typing import Dict, Optional
import logging

from .base import BaseValidationAgent
from .schemas import AIValidation
from .enums import ValidationResult

logger = logging.getLogger(__name__)


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
