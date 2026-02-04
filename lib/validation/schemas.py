#!/usr/bin/env python3
"""
Validation - Data Schemas
============================================================

Validation result schemas

Contains:
    - AIValidation: Single AI validation result
    - ConsensusResult: Multi-agent consensus
    - FeedbackResult: Feedback loop result
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .enums import ValidationResult


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

