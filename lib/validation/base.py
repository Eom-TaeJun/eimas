#!/usr/bin/env python3
"""
Validation - Base Agent
============================================================

Abstract base class for validation agents

Class:
    - BaseValidationAgent: ABC for all validation agents
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging

from .schemas import AIValidation
from .enums import ValidationResult

logger = logging.getLogger(__name__)


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
