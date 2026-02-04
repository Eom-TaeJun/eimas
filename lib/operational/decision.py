#!/usr/bin/env python3
"""
Operational - Decision Policy
============================================================

의사결정 정책 및 규칙
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging

# Import from same package
from .config import OperationalConfig
from .enums import FinalStance, ReasonCode, TriggerType, SignalType

logger = logging.getLogger(__name__)


@dataclass
class DecisionPolicy:
    """
    결정 거버넌스 결과

    명시적, 순서화된 규칙을 적용하여 최종 스탠스 결정
    """
    final_stance: str = "HOLD"
    reason_codes: List[str] = field(default_factory=list)
    applied_rules: List[str] = field(default_factory=list)

    # 입력 요약
    regime_input: str = "NEUTRAL"
    risk_score_input: float = 50.0
    confidence_input: float = 0.5
    agent_consensus_input: str = "NEUTRAL"
    modes_agree_input: bool = True
    constraint_status_input: str = "OK"
    client_profile_status_input: str = "COMPLETE"

    # 결정 과정
    rule_evaluation_log: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DecisionInputs:
    """결정 함수 입력"""
    regime: str                           # BULL/BEAR/NEUTRAL
    regime_confidence: float              # 0-1
    risk_score: float                     # 0-100
    agent_consensus: str                  # BULLISH/BEARISH/NEUTRAL
    full_mode_position: str               # BULLISH/BEARISH/NEUTRAL
    reference_mode_position: str          # BULLISH/BEARISH/NEUTRAL
    modes_agree: bool
    confidence: float                     # 0-1
    constraint_status: str                # OK/VIOLATED/REPAIRED/UNREPAIRED
    client_profile_status: str = "COMPLETE"  # COMPLETE/PARTIAL/MISSING
    data_quality: str = "COMPLETE"        # COMPLETE/PARTIAL/DEGRADED


def resolve_decision(
    inputs: DecisionInputs,
    config: OperationalConfig
) -> DecisionPolicy:
    """
    결정 해결 함수 (Decision Governance)

    명시적, 순서화된 규칙 적용:
    1. 데이터 품질 검사
    2. 제약 상태 검사
    3. 신뢰도 검사
    4. 에이전트 합의 검사
    5. 레짐-스탠스 일관성 검사
    6. 리스크 기반 조정
    7. 최종 스탠스 결정

    Args:
        inputs: 결정에 필요한 모든 입력
        config: 운영 설정

    Returns:
        DecisionPolicy with final stance and reason codes
    """
    policy = DecisionPolicy(
        regime_input=inputs.regime,
        risk_score_input=inputs.risk_score,
        confidence_input=inputs.confidence,
        agent_consensus_input=inputs.agent_consensus,
        modes_agree_input=inputs.modes_agree,
        constraint_status_input=inputs.constraint_status,
        client_profile_status_input=inputs.client_profile_status,
    )

    # Rule evaluation log
    rules_log = []

    # =================================================================
    # Rule 0: Client Profile Check (최우선)
    # =================================================================
    rule0 = {
        'rule': 'RULE_0_CLIENT_PROFILE',
        'condition': 'client_profile_status == MISSING',
        'input': inputs.client_profile_status,
        'result': None
    }
    if inputs.client_profile_status == "MISSING":
        rule0['result'] = 'TRIGGERED -> HOLD'
        rules_log.append(rule0)
        policy.final_stance = FinalStance.HOLD.value
        policy.reason_codes.append(ReasonCode.CLIENT_PROFILE_MISSING.value)
        policy.applied_rules.append("RULE_0: Client profile missing -> HOLD")
        policy.rule_evaluation_log = rules_log
        return policy
    rule0['result'] = 'PASSED'
    rules_log.append(rule0)

    # =================================================================
    # Rule 1: Data Quality Check
    # =================================================================
    rule1 = {
        'rule': 'RULE_1_DATA_QUALITY',
        'condition': f'data_quality == DEGRADED',
        'input': inputs.data_quality,
        'result': None
    }
    if inputs.data_quality == "DEGRADED":
        rule1['result'] = 'TRIGGERED -> HOLD'
        rules_log.append(rule1)
        policy.final_stance = FinalStance.HOLD.value
        policy.reason_codes.append(ReasonCode.DATA_QUALITY_ISSUE.value)
        policy.applied_rules.append("RULE_1: Data quality degraded -> HOLD")
        policy.rule_evaluation_log = rules_log
        return policy
    rule1['result'] = 'PASSED'
    rules_log.append(rule1)

    # =================================================================
    # Rule 2: Constraint Status Check
    # =================================================================
    rule2 = {
        'rule': 'RULE_2_CONSTRAINT_STATUS',
        'condition': f'constraint_status == UNREPAIRED',
        'input': inputs.constraint_status,
        'result': None
    }
    if inputs.constraint_status == "UNREPAIRED":
        rule2['result'] = 'TRIGGERED -> HOLD'
        rules_log.append(rule2)
        policy.final_stance = FinalStance.HOLD.value
        policy.reason_codes.append(ReasonCode.CONSTRAINT_VIOLATION_UNREPAIRED.value)
        policy.applied_rules.append("RULE_2: Constraint violation unrepaired -> HOLD")
        policy.rule_evaluation_log = rules_log
        return policy
    rule2['result'] = 'PASSED'
    rules_log.append(rule2)

    # =================================================================
    # Rule 3: Low Confidence Check
    # =================================================================
    rule3 = {
        'rule': 'RULE_3_LOW_CONFIDENCE',
        'condition': f'confidence < {config.confidence_threshold_low}',
        'input': inputs.confidence,
        'result': None
    }
    if inputs.confidence < config.confidence_threshold_low:
        rule3['result'] = 'TRIGGERED -> HOLD'
        rules_log.append(rule3)
        policy.final_stance = FinalStance.HOLD.value
        policy.reason_codes.append(ReasonCode.LOW_CONFIDENCE.value)
        policy.applied_rules.append(f"RULE_3: Confidence {inputs.confidence:.2f} < {config.confidence_threshold_low} -> HOLD")
        policy.rule_evaluation_log = rules_log
        return policy
    rule3['result'] = 'PASSED'
    rules_log.append(rule3)

    # =================================================================
    # Rule 4: Agent Conflict Check
    # =================================================================
    rule4 = {
        'rule': 'RULE_4_AGENT_CONFLICT',
        'condition': 'modes_agree == False AND confidence < high_threshold',
        'input': f'modes_agree={inputs.modes_agree}, confidence={inputs.confidence}',
        'result': None
    }
    if not inputs.modes_agree and inputs.confidence < config.confidence_threshold_high:
        rule4['result'] = 'TRIGGERED -> HOLD'
        rules_log.append(rule4)
        policy.final_stance = FinalStance.HOLD.value
        policy.reason_codes.append(ReasonCode.AGENT_CONFLICT.value)
        policy.applied_rules.append("RULE_4: Agent modes disagree with moderate confidence -> HOLD")
        policy.rule_evaluation_log = rules_log
        return policy
    rule4['result'] = 'PASSED'
    rules_log.append(rule4)

    # =================================================================
    # Rule 5: Regime-Stance Consistency Check
    # =================================================================
    rule5 = {
        'rule': 'RULE_5_REGIME_STANCE_CONSISTENCY',
        'condition': 'regime contradicts agent_consensus',
        'input': f'regime={inputs.regime}, consensus={inputs.agent_consensus}',
        'result': None
    }
    regime_stance_mismatch = (
        (inputs.regime == "BULL" and inputs.agent_consensus == "BEARISH") or
        (inputs.regime == "BEAR" and inputs.agent_consensus == "BULLISH")
    )
    if regime_stance_mismatch and inputs.confidence < config.confidence_threshold_high:
        rule5['result'] = 'TRIGGERED -> HOLD'
        rules_log.append(rule5)
        policy.final_stance = FinalStance.HOLD.value
        policy.reason_codes.append(ReasonCode.REGIME_STANCE_MISMATCH.value)
        policy.applied_rules.append("RULE_5: Regime contradicts consensus with moderate confidence -> HOLD")
        policy.rule_evaluation_log = rules_log
        return policy
    rule5['result'] = 'PASSED'
    rules_log.append(rule5)

    # =================================================================
    # Rule 6: High Risk Override
    # =================================================================
    rule6 = {
        'rule': 'RULE_6_HIGH_RISK_OVERRIDE',
        'condition': f'risk_score >= {config.risk_score_high}',
        'input': inputs.risk_score,
        'result': None
    }
    if inputs.risk_score >= config.risk_score_high:
        rule6['result'] = 'TRIGGERED -> reduce stance'
        rules_log.append(rule6)
        # 고위험 환경에서는 방어적 스탠스
        if inputs.agent_consensus == "BULLISH":
            policy.final_stance = FinalStance.NEUTRAL.value
            policy.reason_codes.append(ReasonCode.HIGH_RISK_BEAR.value)
            policy.applied_rules.append(f"RULE_6: High risk ({inputs.risk_score:.1f}) overrides BULLISH -> NEUTRAL")
        else:
            policy.final_stance = FinalStance.BEARISH.value
            policy.reason_codes.append(ReasonCode.HIGH_RISK_BEAR.value)
            policy.applied_rules.append(f"RULE_6: High risk ({inputs.risk_score:.1f}) -> BEARISH")
        policy.rule_evaluation_log = rules_log
        return policy
    rule6['result'] = 'PASSED'
    rules_log.append(rule6)

    # =================================================================
    # Rule 7: Determine Final Stance from Inputs
    # =================================================================
    rule7 = {
        'rule': 'RULE_7_DETERMINE_STANCE',
        'condition': 'All checks passed, use regime + consensus',
        'input': f'regime={inputs.regime}, consensus={inputs.agent_consensus}, conf={inputs.confidence}',
        'result': None
    }

    # High confidence scenario
    if inputs.confidence >= config.confidence_threshold_high:
        if inputs.regime == "BULL" and inputs.agent_consensus in ["BULLISH", "NEUTRAL"]:
            policy.final_stance = FinalStance.BULLISH.value
            policy.reason_codes.append(ReasonCode.BULL_REGIME_HIGH_CONF.value)
            rule7['result'] = 'BULLISH (high confidence bull)'
        elif inputs.regime == "BEAR" and inputs.agent_consensus in ["BEARISH", "NEUTRAL"]:
            policy.final_stance = FinalStance.BEARISH.value
            policy.reason_codes.append(ReasonCode.BEAR_REGIME_HIGH_CONF.value)
            rule7['result'] = 'BEARISH (high confidence bear)'
        elif inputs.agent_consensus == "BULLISH":
            policy.final_stance = FinalStance.BULLISH.value
            policy.reason_codes.append(ReasonCode.BULL_REGIME_CONSENSUS.value)
            rule7['result'] = 'BULLISH (consensus)'
        elif inputs.agent_consensus == "BEARISH":
            policy.final_stance = FinalStance.BEARISH.value
            policy.reason_codes.append(ReasonCode.BEAR_REGIME_CONSENSUS.value)
            rule7['result'] = 'BEARISH (consensus)'
        else:
            policy.final_stance = FinalStance.NEUTRAL.value
            policy.reason_codes.append(ReasonCode.NEUTRAL_REGIME.value)
            rule7['result'] = 'NEUTRAL'
    # Moderate confidence
    else:
        if inputs.regime == "BULL" and inputs.modes_agree and inputs.agent_consensus == "BULLISH":
            policy.final_stance = FinalStance.BULLISH.value
            policy.reason_codes.append(ReasonCode.BULL_REGIME_CONSENSUS.value)
            rule7['result'] = 'BULLISH (regime + consensus agree)'
        elif inputs.regime == "BEAR" and inputs.modes_agree and inputs.agent_consensus == "BEARISH":
            policy.final_stance = FinalStance.BEARISH.value
            policy.reason_codes.append(ReasonCode.BEAR_REGIME_CONSENSUS.value)
            rule7['result'] = 'BEARISH (regime + consensus agree)'
        elif inputs.risk_score <= config.risk_score_low:
            policy.final_stance = FinalStance.NEUTRAL.value
            policy.reason_codes.append(ReasonCode.LOW_RISK_BULL.value)
            rule7['result'] = 'NEUTRAL (low risk)'
        else:
            policy.final_stance = FinalStance.HOLD.value
            policy.reason_codes.append(ReasonCode.MIXED_SIGNALS.value)
            rule7['result'] = 'HOLD (mixed signals)'

    policy.applied_rules.append(f"RULE_7: {rule7['result']}")
    rules_log.append(rule7)
    policy.rule_evaluation_log = rules_log

    return policy

