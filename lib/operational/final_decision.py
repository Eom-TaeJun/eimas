#!/usr/bin/env python3
"""
Operational - Final Decision
============================================================

최종 결정 해석 및 통합
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
class FinalDecisionInputs:
    """
    resolve_final_decision 함수의 입력

    기존 EIMAS 출력만 사용 (새로운 시그널 생성 금지)
    """
    regime_signal: str                    # BULL / BEAR / NEUTRAL
    regime_confidence: float              # 0.0 - 1.0
    canonical_risk_score: float           # 0 - 100
    agent_consensus_stance: str           # BULLISH / BEARISH / NEUTRAL
    agent_consensus_confidence: float     # 0.0 - 1.0
    constraint_status: str                # OK / VIOLATED / REPAIRED / UNREPAIRED
    client_profile_status: str            # COMPLETE / PARTIAL / MISSING


@dataclass
class FinalDecisionResult:
    """
    resolve_final_decision 함수의 출력

    decision_policy 섹션에 출력
    """
    final_stance: str                     # BULLISH / BEARISH / NEUTRAL / HOLD
    reason_codes: List[str]               # 결정 사유 코드 목록
    applied_rules: List[str]              # 적용된 규칙 설명
    rule_evaluation_log: List[Dict]       # 규칙 평가 상세 로그

    # 입력 요약 (감사 추적용)
    inputs_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'final_stance': self.final_stance,
            'reason_codes': self.reason_codes,
            'applied_rules': self.applied_rules,
            'rule_evaluation_log': self.rule_evaluation_log,
            'inputs_summary': self.inputs_summary,
        }

    def to_markdown(self) -> str:
        """decision_policy 섹션 마크다운 생성"""
        lines = []
        lines.append("## decision_policy")
        lines.append("")
        lines.append(f"### Final Stance: **{self.final_stance}**")
        lines.append("")

        lines.append("### Inputs")
        for key, value in self.inputs_summary.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

        lines.append("### Applied Rules (in order)")
        for i, rule in enumerate(self.applied_rules, 1):
            lines.append(f"{i}. {rule}")
        lines.append("")

        lines.append("### Reason Codes")
        for code in self.reason_codes:
            lines.append(f"- `{code}`")
        lines.append("")

        lines.append("### Rule Evaluation Log")
        lines.append("| # | Rule | Condition | Input | Result |")
        lines.append("|---|------|-----------|-------|--------|")
        for i, log in enumerate(self.rule_evaluation_log, 1):
            lines.append(f"| {i} | {log['rule']} | {log['condition']} | {log['input']} | {log['result']} |")
        lines.append("")

        return "\n".join(lines)


def resolve_final_decision(
    regime_signal: str,
    regime_confidence: float,
    canonical_risk_score: float,
    agent_consensus_stance: str,
    agent_consensus_confidence: float,
    constraint_status: str,
    client_profile_status: str,
    config: Optional[OperationalConfig] = None
) -> FinalDecisionResult:
    """
    Decision Governance Layer

    명시적, 순서화된 규칙을 적용하여 최종 스탠스 결정.
    기존 EIMAS 출력만 사용하며 새로운 시그널을 생성하지 않음.

    ============================================================================
    RULE ORDER (순서대로 평가, 첫 번째 트리거에서 종료)
    ============================================================================

    RULE 1: Client Profile Check
            - client_profile_status == "MISSING" → HOLD
            - 고객 프로필 없이는 투자 결정 불가

    RULE 2: Constraint Status Check
            - constraint_status == "UNREPAIRED" → HOLD
            - 수정 불가능한 제약 위반 시 거래 중단

    RULE 3: Low Confidence Check
            - agent_consensus_confidence < threshold_low (0.50) → HOLD
            - 신뢰도가 임계값 미달 시 보수적 접근

    RULE 4: High Risk Override
            - canonical_risk_score >= threshold_high (70) → 방어적 스탠스
            - BULLISH → NEUTRAL로 하향 조정
            - 그 외 → BEARISH

    RULE 5: Regime-Stance Consistency Check
            - regime와 consensus가 충돌 + 중간 신뢰도 → HOLD
            - (BULL + BEARISH) 또는 (BEAR + BULLISH) 조합

    RULE 6: High Confidence Decision
            - confidence >= threshold_high (0.70) → consensus 따름
            - 높은 신뢰도에서는 에이전트 합의 신뢰

    RULE 7: Moderate Confidence with Agreement
            - regime + consensus 일치 + modes_agree → 해당 방향
            - 신호 정렬 시에만 포지션 취함

    RULE 8: Default to HOLD
            - 위 규칙 모두 해당 없음 → HOLD
            - 불확실한 상황에서는 보수적 접근

    ============================================================================

    Args:
        regime_signal: 시장 레짐 (BULL/BEAR/NEUTRAL)
        regime_confidence: 레짐 신뢰도 (0-1)
        canonical_risk_score: 공식 리스크 점수 (0-100)
        agent_consensus_stance: 에이전트 합의 스탠스 (BULLISH/BEARISH/NEUTRAL)
        agent_consensus_confidence: 에이전트 합의 신뢰도 (0-1)
        constraint_status: 제약 상태 (OK/VIOLATED/REPAIRED/UNREPAIRED)
        client_profile_status: 고객 프로필 상태 (COMPLETE/PARTIAL/MISSING)
        config: 운영 설정 (Optional)

    Returns:
        FinalDecisionResult with final_stance, reason_codes, applied_rules
    """
    if config is None:
        config = OperationalConfig()

    # Initialize result
    result = FinalDecisionResult(
        final_stance="HOLD",
        reason_codes=[],
        applied_rules=[],
        rule_evaluation_log=[],
        inputs_summary={
            'regime_signal': regime_signal,
            'regime_confidence': f"{regime_confidence:.2f}",
            'canonical_risk_score': f"{canonical_risk_score:.1f}",
            'agent_consensus_stance': agent_consensus_stance,
            'agent_consensus_confidence': f"{agent_consensus_confidence:.2f}",
            'constraint_status': constraint_status,
            'client_profile_status': client_profile_status,
        }
    )

    rules_log = []

    # =========================================================================
    # RULE 1: Client Profile Check (최우선)
    # =========================================================================
    rule1 = {
        'rule': 'RULE_1_CLIENT_PROFILE',
        'condition': 'client_profile_status == "MISSING"',
        'input': client_profile_status,
        'result': None
    }
    if client_profile_status == "MISSING":
        rule1['result'] = 'TRIGGERED → HOLD'
        rules_log.append(rule1)
        result.final_stance = FinalStance.HOLD.value
        result.reason_codes.append(ReasonCode.CLIENT_PROFILE_MISSING.value)
        result.applied_rules.append("RULE_1: Client profile missing → HOLD")
        result.rule_evaluation_log = rules_log
        return result
    rule1['result'] = 'PASSED'
    rules_log.append(rule1)

    # =========================================================================
    # RULE 2: Constraint Status Check
    # =========================================================================
    rule2 = {
        'rule': 'RULE_2_CONSTRAINT_STATUS',
        'condition': 'constraint_status == "UNREPAIRED"',
        'input': constraint_status,
        'result': None
    }
    if constraint_status == "UNREPAIRED":
        rule2['result'] = 'TRIGGERED → HOLD'
        rules_log.append(rule2)
        result.final_stance = FinalStance.HOLD.value
        result.reason_codes.append(ReasonCode.CONSTRAINT_VIOLATION_UNREPAIRED.value)
        result.applied_rules.append("RULE_2: Constraints unrepaired → HOLD")
        result.rule_evaluation_log = rules_log
        return result
    rule2['result'] = 'PASSED'
    rules_log.append(rule2)

    # =========================================================================
    # RULE 3: Low Confidence Check
    # =========================================================================
    rule3 = {
        'rule': 'RULE_3_LOW_CONFIDENCE',
        'condition': f'agent_consensus_confidence < {config.confidence_threshold_low}',
        'input': f'{agent_consensus_confidence:.2f}',
        'result': None
    }
    if agent_consensus_confidence < config.confidence_threshold_low:
        rule3['result'] = 'TRIGGERED → HOLD'
        rules_log.append(rule3)
        result.final_stance = FinalStance.HOLD.value
        result.reason_codes.append(ReasonCode.LOW_CONFIDENCE.value)
        result.applied_rules.append(
            f"RULE_3: Confidence {agent_consensus_confidence:.2f} < {config.confidence_threshold_low} → HOLD"
        )
        result.rule_evaluation_log = rules_log
        return result
    rule3['result'] = 'PASSED'
    rules_log.append(rule3)

    # =========================================================================
    # RULE 4: High Risk Override
    # =========================================================================
    rule4 = {
        'rule': 'RULE_4_HIGH_RISK',
        'condition': f'canonical_risk_score >= {config.risk_score_high}',
        'input': f'{canonical_risk_score:.1f}',
        'result': None
    }
    if canonical_risk_score >= config.risk_score_high:
        if agent_consensus_stance == "BULLISH":
            rule4['result'] = 'TRIGGERED → NEUTRAL (downgrade from BULLISH)'
            result.final_stance = FinalStance.NEUTRAL.value
            result.reason_codes.append(ReasonCode.HIGH_RISK_BEAR.value)
            result.applied_rules.append(
                f"RULE_4: High risk ({canonical_risk_score:.1f}) overrides BULLISH → NEUTRAL"
            )
        else:
            rule4['result'] = 'TRIGGERED → BEARISH'
            result.final_stance = FinalStance.BEARISH.value
            result.reason_codes.append(ReasonCode.HIGH_RISK_BEAR.value)
            result.applied_rules.append(
                f"RULE_4: High risk ({canonical_risk_score:.1f}) → BEARISH"
            )
        rules_log.append(rule4)
        result.rule_evaluation_log = rules_log
        return result
    rule4['result'] = 'PASSED'
    rules_log.append(rule4)

    # =========================================================================
    # RULE 5: Regime-Stance Consistency Check
    # =========================================================================
    regime_stance_conflict = (
        (regime_signal == "BULL" and agent_consensus_stance == "BEARISH") or
        (regime_signal == "BEAR" and agent_consensus_stance == "BULLISH")
    )
    rule5 = {
        'rule': 'RULE_5_REGIME_STANCE_CONSISTENCY',
        'condition': 'regime contradicts consensus + mid confidence',
        'input': f'regime={regime_signal}, consensus={agent_consensus_stance}, conf={agent_consensus_confidence:.2f}',
        'result': None
    }
    if regime_stance_conflict and agent_consensus_confidence < config.confidence_threshold_high:
        rule5['result'] = 'TRIGGERED → HOLD (conflict with moderate confidence)'
        rules_log.append(rule5)
        result.final_stance = FinalStance.HOLD.value
        result.reason_codes.append(ReasonCode.REGIME_STANCE_MISMATCH.value)
        result.applied_rules.append("RULE_5: Regime contradicts consensus with moderate confidence → HOLD")
        result.rule_evaluation_log = rules_log
        return result
    rule5['result'] = 'PASSED'
    rules_log.append(rule5)

    # =========================================================================
    # RULE 6: High Confidence Decision
    # =========================================================================
    rule6 = {
        'rule': 'RULE_6_HIGH_CONFIDENCE',
        'condition': f'agent_consensus_confidence >= {config.confidence_threshold_high}',
        'input': f'{agent_consensus_confidence:.2f}',
        'result': None
    }
    if agent_consensus_confidence >= config.confidence_threshold_high:
        if agent_consensus_stance == "BULLISH":
            rule6['result'] = 'TRIGGERED → BULLISH (high confidence)'
            result.final_stance = FinalStance.BULLISH.value
            result.reason_codes.append(ReasonCode.BULL_REGIME_HIGH_CONF.value)
            result.applied_rules.append("RULE_6: High confidence consensus → BULLISH")
        elif agent_consensus_stance == "BEARISH":
            rule6['result'] = 'TRIGGERED → BEARISH (high confidence)'
            result.final_stance = FinalStance.BEARISH.value
            result.reason_codes.append(ReasonCode.BEAR_REGIME_HIGH_CONF.value)
            result.applied_rules.append("RULE_6: High confidence consensus → BEARISH")
        else:
            rule6['result'] = 'TRIGGERED → NEUTRAL (high confidence neutral)'
            result.final_stance = FinalStance.NEUTRAL.value
            result.reason_codes.append(ReasonCode.NEUTRAL_REGIME.value)
            result.applied_rules.append("RULE_6: High confidence consensus → NEUTRAL")
        rules_log.append(rule6)
        result.rule_evaluation_log = rules_log
        return result
    rule6['result'] = 'PASSED (confidence below high threshold)'
    rules_log.append(rule6)

    # =========================================================================
    # RULE 7: Moderate Confidence with Regime-Consensus Agreement
    # =========================================================================
    regime_consensus_agree = (
        (regime_signal == "BULL" and agent_consensus_stance == "BULLISH") or
        (regime_signal == "BEAR" and agent_consensus_stance == "BEARISH") or
        (regime_signal == "NEUTRAL" and agent_consensus_stance == "NEUTRAL")
    )
    rule7 = {
        'rule': 'RULE_7_REGIME_CONSENSUS_AGREEMENT',
        'condition': 'regime aligns with consensus',
        'input': f'regime={regime_signal}, consensus={agent_consensus_stance}',
        'result': None
    }
    if regime_consensus_agree:
        if regime_signal == "BULL":
            rule7['result'] = 'TRIGGERED → BULLISH (regime + consensus aligned)'
            result.final_stance = FinalStance.BULLISH.value
            result.reason_codes.append(ReasonCode.BULL_REGIME_CONSENSUS.value)
            result.applied_rules.append("RULE_7: Regime and consensus aligned → BULLISH")
        elif regime_signal == "BEAR":
            rule7['result'] = 'TRIGGERED → BEARISH (regime + consensus aligned)'
            result.final_stance = FinalStance.BEARISH.value
            result.reason_codes.append(ReasonCode.BEAR_REGIME_CONSENSUS.value)
            result.applied_rules.append("RULE_7: Regime and consensus aligned → BEARISH")
        else:
            rule7['result'] = 'TRIGGERED → NEUTRAL (regime + consensus aligned)'
            result.final_stance = FinalStance.NEUTRAL.value
            result.reason_codes.append(ReasonCode.NEUTRAL_REGIME.value)
            result.applied_rules.append("RULE_7: Regime and consensus aligned → NEUTRAL")
        rules_log.append(rule7)
        result.rule_evaluation_log = rules_log
        return result
    rule7['result'] = 'PASSED (no alignment)'
    rules_log.append(rule7)

    # =========================================================================
    # RULE 8: Default to HOLD
    # =========================================================================
    rule8 = {
        'rule': 'RULE_8_DEFAULT_HOLD',
        'condition': 'No rules triggered',
        'input': 'N/A',
        'result': 'TRIGGERED → HOLD (default)'
    }
    rules_log.append(rule8)
    result.final_stance = FinalStance.HOLD.value
    result.reason_codes.append(ReasonCode.DEFAULT_HOLD.value)
    result.applied_rules.append("RULE_8: No clear signal → HOLD (default)")
    result.rule_evaluation_log = rules_log

    return result

