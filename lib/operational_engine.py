"""
Operational Engine
==================
기존 EIMAS 결과를 운영 가능한 형태로 변환하는 엔진

핵심 원칙:
- 새로운 시그널, 추정치, 시장 뷰 생성 금지
- 기존 계산된 결과만 사용
- 결정론적이고 감사 가능한 출력
- 하드코딩된 값 없음 (config에서 로드)

기능:
1. Decision Governance - 최종 스탠스 결정
2. Risk Score Consistency - 단일 공식 리스크 점수
3. Rebalancing Execution Plan - 거래 리스트 생성
4. Constraint Repair - 제약 위반 자동 수정
5. Report Structure - 표준화된 리포트 섹션
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (하드코딩 대신 config에서 로드)
# =============================================================================

@dataclass
class OperationalConfig:
    """운영 설정 (하드코딩 없이 config로 관리)"""
    # Decision Governance Thresholds
    confidence_threshold_high: float = 0.70
    confidence_threshold_low: float = 0.50
    risk_score_high: float = 70.0
    risk_score_low: float = 30.0

    # Rebalancing Thresholds
    turnover_cap: float = 0.30
    min_trade_size: float = 0.01
    human_approval_threshold: float = 0.20  # 20% 이상 변화 시 승인 필요

    # Asset Class Bounds (constraint repair용)
    equity_min: float = 0.0
    equity_max: float = 1.0
    bond_min: float = 0.0
    bond_max: float = 1.0
    cash_min: float = 0.0
    cash_max: float = 0.20
    commodity_min: float = 0.0
    commodity_max: float = 0.20
    crypto_min: float = 0.0
    crypto_max: float = 0.10

    # Trading Cost Model
    commission_rate: float = 0.001
    spread_cost: float = 0.0005
    market_impact: float = 0.001

    @classmethod
    def from_dict(cls, data: Dict) -> 'OperationalConfig':
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Enums
# =============================================================================

class FinalStance(Enum):
    """최종 투자 스탠스"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    HOLD = "HOLD"


class ReasonCode(Enum):
    """결정 사유 코드"""
    # Bullish reasons
    BULL_REGIME_HIGH_CONF = "BULL_REGIME_HIGH_CONFIDENCE"
    BULL_REGIME_CONSENSUS = "BULL_REGIME_WITH_CONSENSUS"
    LOW_RISK_BULL = "LOW_RISK_BULLISH_ENVIRONMENT"

    # Bearish reasons
    BEAR_REGIME_HIGH_CONF = "BEAR_REGIME_HIGH_CONFIDENCE"
    BEAR_REGIME_CONSENSUS = "BEAR_REGIME_WITH_CONSENSUS"
    HIGH_RISK_BEAR = "HIGH_RISK_BEARISH_ENVIRONMENT"

    # Neutral reasons
    NEUTRAL_REGIME = "NEUTRAL_REGIME"
    MIXED_SIGNALS = "MIXED_SIGNALS"

    # Hold reasons
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    AGENT_CONFLICT = "AGENT_CONFLICT_UNRESOLVED"
    DATA_QUALITY_ISSUE = "DATA_QUALITY_ISSUE"
    CONSTRAINT_VIOLATION_UNREPAIRED = "CONSTRAINT_VIOLATION_UNREPAIRED"
    REGIME_STANCE_MISMATCH = "REGIME_STANCE_MISMATCH"
    CLIENT_PROFILE_MISSING = "CLIENT_PROFILE_MISSING"
    DEFAULT_HOLD = "DEFAULT_HOLD"


class TriggerType(Enum):
    """리밸런싱 트리거 유형"""
    DRIFT = "DRIFT"                       # 편차 임계값 초과
    REGIME_CHANGE = "REGIME_CHANGE"       # 레짐 변화
    CONSTRAINT_REPAIR = "CONSTRAINT_REPAIR"  # 제약 위반 수정
    SCHEDULED = "SCHEDULED"               # 정기 리밸런싱
    MANUAL = "MANUAL"                     # 수동 요청


# =============================================================================
# Score Definitions
# =============================================================================

@dataclass
class ScoreDefinitions:
    """
    리스크 점수 정의 (단일 공식 점수 + 서브스코어)

    Canonical Risk Score = Base + Microstructure Adj + Bubble Adj + Extended Adj
    """
    # Canonical (공식) Risk Score
    canonical_risk_score: float = 0.0

    # Sub-scores (명시적으로 문서화)
    base_risk_score: float = 0.0           # CriticalPathAggregator 기본 점수
    microstructure_adjustment: float = 0.0  # 시장 미세구조 조정 (±10)
    bubble_risk_adjustment: float = 0.0     # 버블 리스크 가산 (+0~15)
    extended_data_adjustment: float = 0.0   # PCR/Sentiment/Credit 조정 (±15)

    # Score interpretation
    risk_level: str = "MEDIUM"             # LOW/MEDIUM/HIGH

    def to_dict(self) -> Dict:
        return {
            'canonical_risk_score': self.canonical_risk_score,
            'sub_scores': {
                'base_risk_score': self.base_risk_score,
                'microstructure_adjustment': self.microstructure_adjustment,
                'bubble_risk_adjustment': self.bubble_risk_adjustment,
                'extended_data_adjustment': self.extended_data_adjustment,
            },
            'formula': 'canonical = base + microstructure_adj + bubble_adj + extended_adj',
            'risk_level': self.risk_level,
        }


# =============================================================================
# Decision Policy
# =============================================================================

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
    )

    # Rule evaluation log
    rules_log = []

    # =================================================================
    # Rule 1: Data Quality Check (최우선)
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


# =============================================================================
# Decision Governance Layer (resolve_final_decision)
# =============================================================================

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


# =============================================================================
# Constraint Repair
# =============================================================================

# Asset class mapping (should be loaded from config in production)
ASSET_CLASS_MAP = {
    'SPY': 'equity', 'QQQ': 'equity', 'IWM': 'equity', 'DIA': 'equity',
    'VTI': 'equity', 'VOO': 'equity', 'XLK': 'equity', 'XLF': 'equity',
    'XLV': 'equity', 'XLE': 'equity', 'XLI': 'equity', 'XLY': 'equity',
    'XLP': 'equity', 'XLU': 'equity', 'XLB': 'equity', 'XLRE': 'equity',
    'VNQ': 'equity', 'COIN': 'equity',
    'TLT': 'bond', 'IEF': 'bond', 'SHY': 'bond', 'LQD': 'bond',
    'HYG': 'bond', 'AGG': 'bond', 'BND': 'bond', 'GOVT': 'bond', 'UUP': 'bond',
    'GLD': 'commodity', 'SLV': 'commodity', 'USO': 'commodity',
    'DBC': 'commodity', 'PAXG-USD': 'commodity',
    'BTC-USD': 'crypto', 'ETH-USD': 'crypto', 'ONDO-USD': 'crypto',
    'SHV': 'cash', 'BIL': 'cash', 'SGOV': 'cash',
}


@dataclass
class ConstraintViolation:
    """제약 위반 정보"""
    asset_class: str
    violation_type: str  # BELOW_MIN / ABOVE_MAX
    current_value: float
    limit_value: float
    delta: float


@dataclass
class ConstraintRepairResult:
    """제약 수정 결과"""
    original_weights: Dict[str, float]
    repaired_weights: Dict[str, float]
    violations_found: List[ConstraintViolation]
    repair_actions: List[str]
    repair_successful: bool
    constraints_satisfied: bool

    def to_dict(self) -> Dict:
        return {
            'original_weights': self.original_weights,
            'repaired_weights': self.repaired_weights,
            'violations_found': [asdict(v) for v in self.violations_found],
            'repair_actions': self.repair_actions,
            'repair_successful': self.repair_successful,
            'constraints_satisfied': self.constraints_satisfied,
        }


def repair_constraints(
    target_weights: Dict[str, float],
    config: OperationalConfig,
    asset_class_map: Dict[str, str] = None
) -> ConstraintRepairResult:
    """
    제약 위반 자동 수정

    1. 자산군별 합계 계산
    2. 위반 식별
    3. 위반 자산군 비중 조정
    4. 정규화
    5. 제약 충족 확인

    Args:
        target_weights: 목표 비중
        config: 운영 설정
        asset_class_map: 자산→자산군 매핑

    Returns:
        ConstraintRepairResult
    """
    if asset_class_map is None:
        asset_class_map = ASSET_CLASS_MAP

    result = ConstraintRepairResult(
        original_weights=target_weights.copy(),
        repaired_weights={},
        violations_found=[],
        repair_actions=[],
        repair_successful=True,
        constraints_satisfied=True,
    )

    # 자산군별 합계
    class_weights = {'equity': 0.0, 'bond': 0.0, 'cash': 0.0, 'commodity': 0.0, 'crypto': 0.0}
    class_assets = {'equity': [], 'bond': [], 'cash': [], 'commodity': [], 'crypto': []}

    for asset, weight in target_weights.items():
        asset_class = asset_class_map.get(asset, 'equity')
        if asset_class in class_weights:
            class_weights[asset_class] += weight
            class_assets[asset_class].append(asset)

    # 제약 정의
    constraints = {
        'equity': (config.equity_min, config.equity_max),
        'bond': (config.bond_min, config.bond_max),
        'cash': (config.cash_min, config.cash_max),
        'commodity': (config.commodity_min, config.commodity_max),
        'crypto': (config.crypto_min, config.crypto_max),
    }

    # 위반 식별
    for asset_class, (min_bound, max_bound) in constraints.items():
        current = class_weights[asset_class]
        if current < min_bound:
            result.violations_found.append(ConstraintViolation(
                asset_class=asset_class,
                violation_type="BELOW_MIN",
                current_value=current,
                limit_value=min_bound,
                delta=min_bound - current
            ))
        elif current > max_bound:
            result.violations_found.append(ConstraintViolation(
                asset_class=asset_class,
                violation_type="ABOVE_MAX",
                current_value=current,
                limit_value=max_bound,
                delta=current - max_bound
            ))

    if not result.violations_found:
        result.repaired_weights = target_weights.copy()
        result.repair_actions.append("No violations found - no repair needed")
        return result

    # 비중 수정
    repaired = target_weights.copy()

    for violation in result.violations_found:
        assets_in_class = class_assets[violation.asset_class]

        if not assets_in_class:
            result.repair_actions.append(f"Cannot repair {violation.asset_class}: no assets in class")
            result.repair_successful = False
            continue

        if violation.violation_type == "ABOVE_MAX":
            # 초과분을 비례적으로 감소
            excess = violation.delta
            current_class_weight = sum(repaired.get(a, 0) for a in assets_in_class)
            if current_class_weight > 0:
                scale = (current_class_weight - excess) / current_class_weight
                for asset in assets_in_class:
                    if asset in repaired:
                        old_w = repaired[asset]
                        repaired[asset] = old_w * scale
                        result.repair_actions.append(
                            f"Reduced {asset}: {old_w:.3f} -> {repaired[asset]:.3f} ({violation.asset_class} above max)"
                        )

        elif violation.violation_type == "BELOW_MIN":
            # 부족분을 다른 자산군에서 차감하여 보충
            shortfall = violation.delta
            # 가장 큰 비중의 자산군에서 차감
            other_classes = [c for c in class_weights if c != violation.asset_class and class_weights[c] > 0]
            if other_classes:
                largest_class = max(other_classes, key=lambda c: class_weights[c])
                donor_assets = class_assets[largest_class]
                donor_total = sum(repaired.get(a, 0) for a in donor_assets)

                if donor_total > shortfall:
                    # 비례적으로 차감
                    for asset in donor_assets:
                        if asset in repaired:
                            reduction = repaired[asset] * (shortfall / donor_total)
                            repaired[asset] -= reduction

                    # 수혜 자산군에 균등 분배
                    if assets_in_class:
                        per_asset = shortfall / len(assets_in_class)
                        for asset in assets_in_class:
                            if asset in repaired:
                                repaired[asset] += per_asset
                            else:
                                repaired[asset] = per_asset

                    result.repair_actions.append(
                        f"Transferred {shortfall:.3f} from {largest_class} to {violation.asset_class}"
                    )
                else:
                    result.repair_actions.append(
                        f"Cannot fully repair {violation.asset_class}: insufficient donor weight"
                    )
                    result.repair_successful = False
            else:
                result.repair_actions.append(
                    f"Cannot repair {violation.asset_class}: no donor classes available"
                )
                result.repair_successful = False

    # 정규화
    total = sum(repaired.values())
    if total > 0:
        repaired = {k: v / total for k, v in repaired.items()}

    # 음수 비중 제거
    repaired = {k: max(0, v) for k, v in repaired.items()}
    total = sum(repaired.values())
    if total > 0:
        repaired = {k: v / total for k, v in repaired.items()}

    result.repaired_weights = repaired

    # 수정 후 제약 충족 확인
    new_class_weights = {'equity': 0.0, 'bond': 0.0, 'cash': 0.0, 'commodity': 0.0, 'crypto': 0.0}
    for asset, weight in repaired.items():
        asset_class = asset_class_map.get(asset, 'equity')
        if asset_class in new_class_weights:
            new_class_weights[asset_class] += weight

    for asset_class, (min_bound, max_bound) in constraints.items():
        if new_class_weights[asset_class] < min_bound - 0.001 or new_class_weights[asset_class] > max_bound + 0.001:
            result.constraints_satisfied = False
            result.repair_actions.append(
                f"Post-repair: {asset_class} = {new_class_weights[asset_class]:.3f}, "
                f"bounds = [{min_bound:.3f}, {max_bound:.3f}] - NOT SATISFIED"
            )

    if result.constraints_satisfied:
        result.repair_actions.append("All constraints satisfied after repair")

    return result


# =============================================================================
# Rebalancing Execution Plan
# =============================================================================

@dataclass
class TradeItem:
    """개별 거래 항목"""
    ticker: str
    current_weight: float
    target_weight: float
    delta_weight: float
    action: str  # BUY / SELL / HOLD
    estimated_cost: float


@dataclass
class RebalancePlan:
    """리밸런싱 실행 계획"""
    should_execute: bool
    trigger_type: str
    trigger_reason: str

    # Trade list
    trades: List[TradeItem]

    # Summary
    total_turnover: float
    total_estimated_cost: float
    buy_count: int
    sell_count: int
    hold_count: int

    # Approval
    requires_human_approval: bool
    approval_reason: str = ""

    def to_dict(self) -> Dict:
        return {
            'should_execute': self.should_execute,
            'trigger_type': self.trigger_type,
            'trigger_reason': self.trigger_reason,
            'trades': [asdict(t) for t in self.trades],
            'summary': {
                'total_turnover': self.total_turnover,
                'total_estimated_cost': self.total_estimated_cost,
                'buy_count': self.buy_count,
                'sell_count': self.sell_count,
                'hold_count': self.hold_count,
            },
            'requires_human_approval': self.requires_human_approval,
            'approval_reason': self.approval_reason,
        }


def generate_rebalance_plan(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    trigger_type: TriggerType,
    trigger_reason: str,
    config: OperationalConfig
) -> RebalancePlan:
    """
    리밸런싱 실행 계획 생성

    Args:
        current_weights: 현재 비중
        target_weights: 목표 비중
        trigger_type: 트리거 유형
        trigger_reason: 트리거 사유
        config: 운영 설정

    Returns:
        RebalancePlan
    """
    trades = []
    total_turnover = 0.0
    total_cost = 0.0
    buy_count = 0
    sell_count = 0
    hold_count = 0

    all_tickers = set(current_weights.keys()) | set(target_weights.keys())

    for ticker in sorted(all_tickers):
        current = current_weights.get(ticker, 0.0)
        target = target_weights.get(ticker, 0.0)
        delta = target - current

        # 최소 거래 규모 필터
        if abs(delta) < config.min_trade_size:
            action = "HOLD"
            delta = 0.0
            hold_count += 1
        elif delta > 0:
            action = "BUY"
            buy_count += 1
        else:
            action = "SELL"
            sell_count += 1

        # 비용 계산 (선형 모델)
        trade_value = abs(delta)
        cost = trade_value * (config.commission_rate + config.spread_cost)
        cost += trade_value * config.market_impact * (trade_value ** 0.5)  # 시장 충격

        trades.append(TradeItem(
            ticker=ticker,
            current_weight=current,
            target_weight=target,
            delta_weight=delta,
            action=action,
            estimated_cost=cost
        ))

        total_turnover += abs(delta)
        total_cost += cost

    # Turnover는 편도 합계 / 2
    total_turnover = total_turnover / 2

    # Turnover cap 적용
    should_execute = total_turnover > 0
    if total_turnover > config.turnover_cap:
        scale = config.turnover_cap / total_turnover
        for trade in trades:
            trade.delta_weight *= scale
            trade.estimated_cost *= scale
        total_turnover = config.turnover_cap
        total_cost *= scale

    # Human approval 필요 여부
    requires_approval = total_turnover >= config.human_approval_threshold
    approval_reason = ""
    if requires_approval:
        approval_reason = f"Turnover {total_turnover:.1%} >= approval threshold {config.human_approval_threshold:.1%}"

    return RebalancePlan(
        should_execute=should_execute,
        trigger_type=trigger_type.value,
        trigger_reason=trigger_reason,
        trades=trades,
        total_turnover=total_turnover,
        total_estimated_cost=total_cost,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
        requires_human_approval=requires_approval,
        approval_reason=approval_reason,
    )


# =============================================================================
# Operational Report
# =============================================================================

@dataclass
class AuditMetadata:
    """감사 메타데이터"""
    timestamp: str
    system_version: str = "EIMAS v2.2.2"
    config_hash: str = ""
    input_data_hash: str = ""
    deterministic: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OperationalReport:
    """
    운영 리포트 (표준화된 섹션)

    필수 섹션:
    - decision_policy
    - score_definitions
    - allocation
    - constraint_repair
    - rebalance_plan
    - audit_metadata
    """
    # 필수 섹션
    decision_policy: DecisionPolicy
    score_definitions: ScoreDefinitions
    allocation: Dict[str, float]
    constraint_repair: ConstraintRepairResult
    rebalance_plan: RebalancePlan
    audit_metadata: AuditMetadata

    # 원본 데이터 참조
    raw_inputs: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'decision_policy': self.decision_policy.to_dict(),
            'score_definitions': self.score_definitions.to_dict(),
            'allocation': self.allocation,
            'constraint_repair': self.constraint_repair.to_dict(),
            'rebalance_plan': self.rebalance_plan.to_dict(),
            'audit_metadata': self.audit_metadata.to_dict(),
        }

    def to_markdown(self) -> str:
        """마크다운 리포트 생성"""
        md = []
        md.append("# Operational Report")
        md.append(f"**Generated:** {self.audit_metadata.timestamp}")
        md.append(f"**System:** {self.audit_metadata.system_version}")
        md.append("")

        # 1. Decision Policy
        md.append("## 1. Decision Policy")
        dp = self.decision_policy
        md.append(f"### Final Stance: **{dp.final_stance}**")
        md.append("")
        md.append("#### Inputs")
        md.append(f"- Regime: {dp.regime_input}")
        md.append(f"- Risk Score: {dp.risk_score_input:.1f}")
        md.append(f"- Confidence: {dp.confidence_input:.2f}")
        md.append(f"- Agent Consensus: {dp.agent_consensus_input}")
        md.append(f"- Modes Agree: {dp.modes_agree_input}")
        md.append(f"- Constraint Status: {dp.constraint_status_input}")
        md.append("")
        md.append("#### Applied Rules")
        for rule in dp.applied_rules:
            md.append(f"- {rule}")
        md.append("")
        md.append("#### Reason Codes")
        for code in dp.reason_codes:
            md.append(f"- `{code}`")
        md.append("")
        md.append("#### Rule Evaluation Log")
        md.append("| Rule | Condition | Input | Result |")
        md.append("|------|-----------|-------|--------|")
        for log in dp.rule_evaluation_log:
            md.append(f"| {log['rule']} | {log['condition']} | {log['input']} | {log['result']} |")
        md.append("")

        # 2. Score Definitions
        md.append("## 2. Score Definitions")
        sd = self.score_definitions
        md.append(f"### Canonical Risk Score: **{sd.canonical_risk_score:.1f}** ({sd.risk_level})")
        md.append("")
        md.append("#### Sub-Scores")
        md.append(f"- Base Risk Score: {sd.base_risk_score:.1f}")
        md.append(f"- Microstructure Adjustment: {sd.microstructure_adjustment:+.1f}")
        md.append(f"- Bubble Risk Adjustment: {sd.bubble_risk_adjustment:+.1f}")
        md.append(f"- Extended Data Adjustment: {sd.extended_data_adjustment:+.1f}")
        md.append("")
        md.append("#### Formula")
        md.append("```")
        md.append("canonical = base + microstructure_adj + bubble_adj + extended_adj")
        md.append(f"         = {sd.base_risk_score:.1f} + ({sd.microstructure_adjustment:+.1f}) + ({sd.bubble_risk_adjustment:+.1f}) + ({sd.extended_data_adjustment:+.1f})")
        md.append(f"         = {sd.canonical_risk_score:.1f}")
        md.append("```")
        md.append("")

        # 3. Allocation
        md.append("## 3. Allocation")
        md.append("| Asset | Weight |")
        md.append("|-------|--------|")
        for ticker, weight in sorted(self.allocation.items(), key=lambda x: x[1], reverse=True):
            if weight >= 0.01:
                md.append(f"| {ticker} | {weight:.1%} |")
        md.append("")

        # 4. Constraint Repair
        md.append("## 4. Constraint Repair")
        cr = self.constraint_repair
        if cr.violations_found:
            md.append("### Violations Found")
            md.append("| Asset Class | Type | Current | Limit | Delta |")
            md.append("|-------------|------|---------|-------|-------|")
            for v in cr.violations_found:
                md.append(f"| {v.asset_class} | {v.violation_type} | {v.current_value:.1%} | {v.limit_value:.1%} | {v.delta:.1%} |")
            md.append("")
            md.append("### Repair Actions")
            for action in cr.repair_actions:
                md.append(f"- {action}")
            md.append("")
            md.append(f"### Repair Successful: **{cr.repair_successful}**")
            md.append(f"### Constraints Satisfied: **{cr.constraints_satisfied}**")
        else:
            md.append("No constraint violations found.")
        md.append("")

        # 5. Rebalance Plan
        md.append("## 5. Rebalance Plan")
        rp = self.rebalance_plan
        md.append(f"### Execute: **{rp.should_execute}**")
        md.append(f"- Trigger Type: {rp.trigger_type}")
        md.append(f"- Trigger Reason: {rp.trigger_reason}")
        md.append(f"- Total Turnover: {rp.total_turnover:.2%}")
        md.append(f"- Total Estimated Cost: {rp.total_estimated_cost:.4f}")
        md.append(f"- Trades: {rp.buy_count} buys, {rp.sell_count} sells, {rp.hold_count} holds")
        md.append("")
        if rp.requires_human_approval:
            md.append(f"### ⚠️ Human Approval Required")
            md.append(f"Reason: {rp.approval_reason}")
            md.append("")
        md.append("### Trade List")
        md.append("| Ticker | Current | Target | Delta | Action | Cost |")
        md.append("|--------|---------|--------|-------|--------|------|")
        for t in sorted(rp.trades, key=lambda x: abs(x.delta_weight), reverse=True):
            if abs(t.delta_weight) >= 0.005:
                md.append(f"| {t.ticker} | {t.current_weight:.1%} | {t.target_weight:.1%} | {t.delta_weight:+.1%} | {t.action} | {t.estimated_cost:.4f} |")
        md.append("")

        # 6. Audit Metadata
        md.append("## 6. Audit Metadata")
        am = self.audit_metadata
        md.append(f"- Timestamp: {am.timestamp}")
        md.append(f"- System Version: {am.system_version}")
        md.append(f"- Deterministic: {am.deterministic}")
        md.append("")
        md.append("---")
        md.append("*This report uses only computed results from the system. No new signals or estimates were generated.*")

        return "\n".join(md)


# =============================================================================
# Main Engine
# =============================================================================

class OperationalEngine:
    """
    운영 엔진

    EIMAS 결과를 운영 가능한 형태로 변환

    Example:
        >>> engine = OperationalEngine()
        >>> report = engine.process(eimas_result, current_weights)
        >>> print(report.to_markdown())
    """

    def __init__(self, config: Optional[OperationalConfig] = None):
        self.config = config or OperationalConfig()

    def process(
        self,
        eimas_result: Dict,
        current_weights: Optional[Dict[str, float]] = None
    ) -> OperationalReport:
        """
        EIMAS 결과를 운영 리포트로 변환

        Args:
            eimas_result: EIMAS JSON 결과
            current_weights: 현재 포트폴리오 비중 (리밸런싱용)

        Returns:
            OperationalReport
        """
        # 1. Score Definitions
        score_defs = self._extract_scores(eimas_result)

        # 2. Extract allocation
        allocation = self._extract_allocation(eimas_result)

        # 3. Constraint Repair
        constraint_result = repair_constraints(allocation, self.config)

        # 4. Decision Inputs
        decision_inputs = self._extract_decision_inputs(
            eimas_result, score_defs, constraint_result
        )

        # 5. Resolve Decision
        decision = resolve_decision(decision_inputs, self.config)

        # 6. Rebalance Plan
        rebalance = self._generate_rebalance_plan(
            eimas_result, current_weights, constraint_result, decision
        )

        # 7. Audit Metadata
        audit = AuditMetadata(
            timestamp=datetime.now().isoformat(),
            system_version="EIMAS v2.2.2",
        )

        return OperationalReport(
            decision_policy=decision,
            score_definitions=score_defs,
            allocation=constraint_result.repaired_weights if constraint_result.repair_successful else allocation,
            constraint_repair=constraint_result,
            rebalance_plan=rebalance,
            audit_metadata=audit,
            raw_inputs={'eimas_result': eimas_result},
        )

    def _extract_scores(self, data: Dict) -> ScoreDefinitions:
        """리스크 점수 추출"""
        base = data.get('base_risk_score', data.get('risk_score', 50.0))
        micro_adj = data.get('microstructure_adjustment', 0.0)
        bubble_adj = data.get('bubble_risk_adjustment', 0.0)
        ext_adj = data.get('extended_data_adjustment', 0.0)

        canonical = base + micro_adj + bubble_adj + ext_adj
        canonical = max(0, min(100, canonical))

        if canonical < 30:
            risk_level = "LOW"
        elif canonical < 70:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return ScoreDefinitions(
            canonical_risk_score=canonical,
            base_risk_score=base,
            microstructure_adjustment=micro_adj,
            bubble_risk_adjustment=bubble_adj,
            extended_data_adjustment=ext_adj,
            risk_level=risk_level,
        )

    def _extract_allocation(self, data: Dict) -> Dict[str, float]:
        """배분 비중 추출"""
        # allocation_result가 있으면 사용
        alloc_result = data.get('allocation_result', {})
        if alloc_result and 'weights' in alloc_result:
            return alloc_result['weights']

        # portfolio_weights 사용
        return data.get('portfolio_weights', {})

    def _extract_decision_inputs(
        self,
        data: Dict,
        scores: ScoreDefinitions,
        constraint_result: ConstraintRepairResult
    ) -> DecisionInputs:
        """결정 입력 추출"""
        regime_data = data.get('regime', {})
        if isinstance(regime_data, dict):
            regime = regime_data.get('regime', 'NEUTRAL')
            # Extract regime part (e.g., "Bull (Low Vol)" -> "BULL")
            if 'Bull' in regime or 'BULL' in regime:
                regime = 'BULL'
            elif 'Bear' in regime or 'BEAR' in regime:
                regime = 'BEAR'
            else:
                regime = 'NEUTRAL'
            regime_confidence = regime_data.get('confidence', 0.5)
        else:
            regime = 'NEUTRAL'
            regime_confidence = 0.5

        # Constraint status
        if not constraint_result.violations_found:
            constraint_status = "OK"
        elif constraint_result.constraints_satisfied:
            constraint_status = "REPAIRED"
        else:
            constraint_status = "UNREPAIRED"

        # Data quality
        mq = data.get('market_quality', {})
        data_quality = mq.get('data_quality', 'COMPLETE') if isinstance(mq, dict) else 'COMPLETE'

        return DecisionInputs(
            regime=regime,
            regime_confidence=regime_confidence,
            risk_score=scores.canonical_risk_score,
            agent_consensus=data.get('full_mode_position', 'NEUTRAL'),
            full_mode_position=data.get('full_mode_position', 'NEUTRAL'),
            reference_mode_position=data.get('reference_mode_position', 'NEUTRAL'),
            modes_agree=data.get('modes_agree', True),
            confidence=data.get('confidence', 0.5),
            constraint_status=constraint_status,
            data_quality=data_quality,
        )

    def _generate_rebalance_plan(
        self,
        data: Dict,
        current_weights: Optional[Dict[str, float]],
        constraint_result: ConstraintRepairResult,
        decision: DecisionPolicy
    ) -> RebalancePlan:
        """리밸런싱 계획 생성"""
        target = constraint_result.repaired_weights if constraint_result.repair_successful else {}

        if not current_weights:
            current_weights = data.get('portfolio_weights', {})

        if not target or not current_weights:
            return RebalancePlan(
                should_execute=False,
                trigger_type=TriggerType.MANUAL.value,
                trigger_reason="No weights available",
                trades=[],
                total_turnover=0.0,
                total_estimated_cost=0.0,
                buy_count=0,
                sell_count=0,
                hold_count=0,
                requires_human_approval=False,
            )

        # Determine trigger type
        rebal_decision = data.get('rebalance_decision', {})
        reason = rebal_decision.get('reason', 'Manual request')

        if 'Threshold exceeded' in reason or 'drift' in reason.lower():
            trigger_type = TriggerType.DRIFT
        elif constraint_result.violations_found:
            trigger_type = TriggerType.CONSTRAINT_REPAIR
        elif 'regime' in reason.lower():
            trigger_type = TriggerType.REGIME_CHANGE
        elif 'Periodic' in reason or 'scheduled' in reason.lower():
            trigger_type = TriggerType.SCHEDULED
        else:
            trigger_type = TriggerType.MANUAL

        # If decision is HOLD due to issues, don't execute
        if decision.final_stance == "HOLD" and any(
            code in [ReasonCode.DATA_QUALITY_ISSUE.value,
                    ReasonCode.CONSTRAINT_VIOLATION_UNREPAIRED.value,
                    ReasonCode.LOW_CONFIDENCE.value]
            for code in decision.reason_codes
        ):
            return RebalancePlan(
                should_execute=False,
                trigger_type=trigger_type.value,
                trigger_reason=f"Blocked by decision policy: {decision.reason_codes}",
                trades=[],
                total_turnover=0.0,
                total_estimated_cost=0.0,
                buy_count=0,
                sell_count=0,
                hold_count=0,
                requires_human_approval=False,
            )

        return generate_rebalance_plan(
            current_weights=current_weights,
            target_weights=target,
            trigger_type=trigger_type,
            trigger_reason=reason,
            config=self.config,
        )


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Operational Engine Test")
    print("=" * 70)

    # Mock EIMAS result
    mock_eimas = {
        "timestamp": "2026-02-02T23:00:00",
        "regime": {
            "regime": "Bull (Low Vol)",
            "confidence": 0.75,
        },
        "risk_score": 45.0,
        "base_risk_score": 40.0,
        "microstructure_adjustment": 5.0,
        "bubble_risk_adjustment": 0.0,
        "extended_data_adjustment": 0.0,
        "final_recommendation": "BULLISH",
        "confidence": 0.66,
        "full_mode_position": "BULLISH",
        "reference_mode_position": "BULLISH",
        "modes_agree": True,
        "market_quality": {"data_quality": "COMPLETE"},
        "allocation_result": {
            "weights": {
                "SPY": 0.35,
                "TLT": 0.20,
                "GLD": 0.18,
                "BTC-USD": 0.12,  # Exceeds crypto max (10%)
                "QQQ": 0.10,
                "USO": 0.05,
            }
        },
        "rebalance_decision": {
            "should_rebalance": True,
            "reason": "Threshold exceeded: max drift 8.5% >= 5.00%",
        }
    }

    # Current weights
    current = {
        "SPY": 0.30,
        "TLT": 0.25,
        "GLD": 0.15,
        "BTC-USD": 0.05,
        "QQQ": 0.15,
        "USO": 0.10,
    }

    # Create engine and process
    engine = OperationalEngine()
    report = engine.process(mock_eimas, current)

    # Print markdown report
    print(report.to_markdown())

    # Test constraint violation scenario
    print("\n" + "=" * 70)
    print("Test 2: Constraint Violation (Crypto > 10%)")
    print("=" * 70)

    mock_eimas_violation = mock_eimas.copy()
    mock_eimas_violation['allocation_result'] = {
        "weights": {
            "SPY": 0.30,
            "TLT": 0.20,
            "GLD": 0.10,
            "BTC-USD": 0.25,  # 25% crypto - violation!
            "ETH-USD": 0.10,  # Total crypto = 35%
            "QQQ": 0.05,
        }
    }

    report2 = engine.process(mock_eimas_violation, current)
    print(f"\nConstraint violations found: {len(report2.constraint_repair.violations_found)}")
    print(f"Repair successful: {report2.constraint_repair.repair_successful}")
    print(f"Repair actions: {report2.constraint_repair.repair_actions}")

    # Test low confidence scenario
    print("\n" + "=" * 70)
    print("Test 3: Low Confidence -> HOLD")
    print("=" * 70)

    mock_low_conf = mock_eimas.copy()
    mock_low_conf['confidence'] = 0.35

    report3 = engine.process(mock_low_conf, current)
    print(f"\nFinal Stance: {report3.decision_policy.final_stance}")
    print(f"Reason Codes: {report3.decision_policy.reason_codes}")
    print(f"Rebalance Execute: {report3.rebalance_plan.should_execute}")

    print("\nTest completed successfully!")

    # ==========================================================================
    # Test resolve_final_decision
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Test 4: resolve_final_decision - High Confidence BULLISH")
    print("=" * 70)

    decision1 = resolve_final_decision(
        regime_signal="BULL",
        regime_confidence=0.85,
        canonical_risk_score=45.0,
        agent_consensus_stance="BULLISH",
        agent_consensus_confidence=0.75,
        constraint_status="OK",
        client_profile_status="COMPLETE"
    )
    print(decision1.to_markdown())

    print("\n" + "=" * 70)
    print("Test 5: resolve_final_decision - Client Profile MISSING")
    print("=" * 70)

    decision2 = resolve_final_decision(
        regime_signal="BULL",
        regime_confidence=0.85,
        canonical_risk_score=45.0,
        agent_consensus_stance="BULLISH",
        agent_consensus_confidence=0.75,
        constraint_status="OK",
        client_profile_status="MISSING"
    )
    print(f"Final Stance: {decision2.final_stance}")
    print(f"Reason Codes: {decision2.reason_codes}")
    print(f"Applied Rules: {decision2.applied_rules}")

    print("\n" + "=" * 70)
    print("Test 6: resolve_final_decision - Low Confidence")
    print("=" * 70)

    decision3 = resolve_final_decision(
        regime_signal="BULL",
        regime_confidence=0.85,
        canonical_risk_score=45.0,
        agent_consensus_stance="BULLISH",
        agent_consensus_confidence=0.35,
        constraint_status="OK",
        client_profile_status="COMPLETE"
    )
    print(f"Final Stance: {decision3.final_stance}")
    print(f"Reason Codes: {decision3.reason_codes}")

    print("\n" + "=" * 70)
    print("Test 7: resolve_final_decision - High Risk Override")
    print("=" * 70)

    decision4 = resolve_final_decision(
        regime_signal="BULL",
        regime_confidence=0.85,
        canonical_risk_score=75.0,
        agent_consensus_stance="BULLISH",
        agent_consensus_confidence=0.65,
        constraint_status="OK",
        client_profile_status="COMPLETE"
    )
    print(f"Final Stance: {decision4.final_stance}")
    print(f"Reason Codes: {decision4.reason_codes}")

    print("\n" + "=" * 70)
    print("Test 8: resolve_final_decision - Regime-Stance Conflict")
    print("=" * 70)

    decision5 = resolve_final_decision(
        regime_signal="BEAR",
        regime_confidence=0.75,
        canonical_risk_score=55.0,
        agent_consensus_stance="BULLISH",
        agent_consensus_confidence=0.60,
        constraint_status="OK",
        client_profile_status="COMPLETE"
    )
    print(f"Final Stance: {decision5.final_stance}")
    print(f"Reason Codes: {decision5.reason_codes}")

    print("\n" + "=" * 70)
    print("All resolve_final_decision tests completed!")
    print("=" * 70)
