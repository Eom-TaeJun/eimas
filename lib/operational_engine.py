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


class SignalType(Enum):
    """시그널 유형 (Signal Hierarchy)"""
    CORE = "CORE"                         # 의사결정에 직접 사용
    AUXILIARY = "AUXILIARY"               # 설명/참고용, 의사결정에 미사용


# =============================================================================
# Signal Hierarchy
# =============================================================================

"""
Signal Hierarchy - 시그널 계층 구조

============================================================================
CORE SIGNALS (의사결정에 직접 사용)
============================================================================
Core signals directly influence final stance and rebalancing decisions.
These are the ONLY signals used in decision rules.

| Signal Name              | Source                    | Used In          |
|--------------------------|---------------------------|------------------|
| regime_signal            | RegimeDetector            | RULE_5, RULE_7   |
| regime_confidence        | RegimeDetector            | All rules        |
| canonical_risk_score     | ScoreDefinitions          | RULE_4           |
| agent_consensus_stance   | MetaOrchestrator          | RULE_5, RULE_6, 7|
| agent_consensus_confidence | MetaOrchestrator        | RULE_3, RULE_6   |
| constraint_status        | ConstraintRepair          | RULE_2           |
| client_profile_status    | ClientConfig              | RULE_1           |

============================================================================
AUXILIARY SIGNALS (설명/참고용, 의사결정에 미사용)
============================================================================
Auxiliary signals provide context and explanation but NEVER override Core logic.
These are clearly labeled in reports.

| Signal Name              | Source                    | Purpose          |
|--------------------------|---------------------------|------------------|
| aux_base_risk_score      | CriticalPathAggregator    | Risk breakdown   |
| aux_microstructure_adj   | MicrostructureAnalyzer    | Risk breakdown   |
| aux_bubble_risk_adj      | BubbleDetector            | Risk breakdown   |
| volatility_score         | RegimeDetector            | Market context   |
| liquidity_score          | MicrostructureAnalyzer    | Market context   |
| credit_spread            | CreditAnalyzer            | Market context   |
| sentiment_score          | SentimentAnalyzer         | Market context   |
| news_sentiment           | NewsCollector             | Market context   |
| sector_rotation          | ETFFlowAnalyzer           | Allocation hint  |

============================================================================
IMPORTANT
============================================================================
- Only CORE signals are used in decision rules
- AUXILIARY signals are for transparency and audit only
- AUXILIARY signals NEVER override CORE logic
- All signals must be classified before use
"""


# Core signal names (used for validation)
CORE_SIGNALS = frozenset([
    'regime_signal',
    'regime_confidence',
    'canonical_risk_score',
    'agent_consensus_stance',
    'agent_consensus_confidence',
    'constraint_status',
    'client_profile_status',
])

# Auxiliary signal names (for documentation)
AUXILIARY_SIGNALS = frozenset([
    'aux_base_risk_score',
    'aux_microstructure_adjustment',
    'aux_bubble_risk_adjustment',
    'aux_extended_data_adjustment',
    'volatility_score',
    'liquidity_score',
    'credit_spread',
    'sentiment_score',
    'news_sentiment',
    'sector_rotation',
    'full_mode_position',
    'reference_mode_position',
    'modes_agree',
])


@dataclass
class SignalClassification:
    """개별 시그널 분류"""
    name: str
    signal_type: str              # CORE / AUXILIARY
    value: Any
    source: str
    used_in_rules: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'signal_type': self.signal_type,
            'value': str(self.value) if not isinstance(self.value, (int, float, bool, str)) else self.value,
            'source': self.source,
            'used_in_rules': self.used_in_rules,
            'description': self.description,
        }


@dataclass
class SignalHierarchyReport:
    """
    시그널 계층 구조 리포트

    Documents which signals are Core vs Auxiliary and ensures
    only Core signals influence decisions.
    """
    core_signals: List[SignalClassification] = field(default_factory=list)
    auxiliary_signals: List[SignalClassification] = field(default_factory=list)
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'core_signals': [s.to_dict() for s in self.core_signals],
            'auxiliary_signals': [s.to_dict() for s in self.auxiliary_signals],
            'validation': {
                'passed': self.validation_passed,
                'errors': self.validation_errors,
            }
        }

    def to_markdown(self) -> str:
        """Generate signal_hierarchy section for report"""
        lines = []
        lines.append("## signal_hierarchy")
        lines.append("")

        # Validation status
        if self.validation_passed:
            lines.append("**Validation: PASSED** ✓")
            lines.append("")
            lines.append("Only Core signals were used in decision rules.")
        else:
            lines.append("**Validation: FAILED** ✗")
            lines.append("")
            for error in self.validation_errors:
                lines.append(f"- ⚠️ {error}")
        lines.append("")

        # Core Signals
        lines.append("### Core Signals (Used in Decision Rules)")
        lines.append("")
        lines.append("| Signal | Value | Source | Used In |")
        lines.append("|--------|-------|--------|---------|")
        for s in self.core_signals:
            rules = ", ".join(s.used_in_rules) if s.used_in_rules else "—"
            value_str = f"{s.value:.2f}" if isinstance(s.value, float) else str(s.value)
            lines.append(f"| {s.name} | {value_str} | {s.source} | {rules} |")
        lines.append("")

        # Auxiliary Signals
        lines.append("### Auxiliary Signals (Context Only)")
        lines.append("")
        lines.append("**Note:** These signals are for transparency only. They do NOT influence decisions.")
        lines.append("")
        lines.append("| Signal | Value | Source | Description |")
        lines.append("|--------|-------|--------|-------------|")
        for s in self.auxiliary_signals:
            value_str = f"{s.value:.2f}" if isinstance(s.value, float) else str(s.value)
            lines.append(f"| {s.name} | {value_str} | {s.source} | {s.description} |")
        lines.append("")

        return "\n".join(lines)


# =============================================================================
# HOLD Policy (Default Stance)
# =============================================================================

"""
HOLD Policy - HOLD as Valid Strategic Outcome

============================================================================
HOLD is NOT "missing output" - it is a deliberate, strategic decision
to maintain current positions when conditions are uncertain or unfavorable.
============================================================================

EXPLICIT HOLD CONDITIONS (ordered by priority):

| Priority | Condition                    | Reason Code                      |
|----------|------------------------------|----------------------------------|
| 1        | Client profile missing       | CLIENT_PROFILE_MISSING           |
| 2        | Constraints unrepaired       | CONSTRAINT_VIOLATION_UNREPAIRED  |
| 3        | Low confidence (< 0.50)      | LOW_CONFIDENCE                   |
| 4        | Agent conflict unresolved    | AGENT_CONFLICT                   |
| 5        | Regime-stance mismatch       | REGIME_STANCE_MISMATCH           |
| 6        | Data quality degraded        | DATA_QUALITY_ISSUE               |
| 7        | Mixed signals (no consensus) | MIXED_SIGNALS                    |
| 8        | Default (no clear direction) | DEFAULT_HOLD                     |

============================================================================
HOLD IMPLICATIONS
============================================================================
- No rebalancing executed
- Current portfolio maintained
- Action blocked until conditions improve
- Explicitly documented in report

============================================================================
EXITING HOLD
============================================================================
To exit HOLD, ALL blocking conditions must be resolved:
- Client profile must be complete
- Constraints must be satisfied or repaired
- Confidence must exceed threshold
- Agent consensus must be achieved
- Regime and stance must align
"""


@dataclass
class HoldCondition:
    """HOLD 조건 상세"""
    priority: int
    condition_name: str
    is_triggered: bool
    reason_code: str
    current_value: Any
    threshold: Any = None
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            'priority': self.priority,
            'condition_name': self.condition_name,
            'is_triggered': self.is_triggered,
            'reason_code': self.reason_code,
            'current_value': str(self.current_value),
            'threshold': str(self.threshold) if self.threshold else None,
            'description': self.description,
        }


@dataclass
class HoldPolicyReport:
    """
    HOLD 정책 리포트

    Documents whether HOLD was selected and why.
    Treats HOLD as a valid strategic outcome.
    """
    is_hold: bool = False
    hold_conditions: List[HoldCondition] = field(default_factory=list)
    triggered_conditions: List[str] = field(default_factory=list)
    primary_reason: str = ""
    exit_requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'is_hold': self.is_hold,
            'hold_conditions': [c.to_dict() for c in self.hold_conditions],
            'triggered_conditions': self.triggered_conditions,
            'primary_reason': self.primary_reason,
            'exit_requirements': self.exit_requirements,
        }

    def to_markdown(self) -> str:
        """Generate hold_policy section for report"""
        lines = []
        lines.append("## hold_policy")
        lines.append("")

        if self.is_hold:
            lines.append("### Status: **HOLD ACTIVE** ⏸️")
            lines.append("")
            lines.append(f"**Primary Reason:** {self.primary_reason}")
            lines.append("")
            lines.append("HOLD is a valid strategic decision to maintain current positions.")
            lines.append("")
        else:
            lines.append("### Status: **NOT IN HOLD** ✓")
            lines.append("")
            lines.append("All HOLD conditions passed. Active stance permitted.")
            lines.append("")

        # Condition Evaluation
        lines.append("### HOLD Condition Evaluation")
        lines.append("")
        lines.append("| # | Condition | Value | Threshold | Status |")
        lines.append("|---|-----------|-------|-----------|--------|")
        for c in self.hold_conditions:
            status = "⏸️ TRIGGERED" if c.is_triggered else "✓ PASSED"
            threshold_str = str(c.threshold) if c.threshold else "—"
            lines.append(f"| {c.priority} | {c.condition_name} | {c.current_value} | {threshold_str} | {status} |")
        lines.append("")

        if self.is_hold and self.exit_requirements:
            lines.append("### Exit Requirements")
            lines.append("")
            lines.append("To exit HOLD, the following must be resolved:")
            lines.append("")
            for req in self.exit_requirements:
                lines.append(f"- [ ] {req}")
            lines.append("")

        return "\n".join(lines)


def evaluate_hold_conditions(
    regime_signal: str,
    regime_confidence: float,
    canonical_risk_score: float,
    agent_consensus_stance: str,
    agent_consensus_confidence: float,
    constraint_status: str,
    client_profile_status: str,
    data_quality: str = "COMPLETE",
    modes_agree: bool = True,
    config: Optional['OperationalConfig'] = None
) -> HoldPolicyReport:
    """
    Evaluate all HOLD conditions and generate report.

    HOLD is selected when ANY condition is triggered.
    Returns a report documenting all evaluated conditions.
    """
    if config is None:
        config = OperationalConfig()

    report = HoldPolicyReport()
    conditions = []

    # Priority 1: Client profile missing
    cond1 = HoldCondition(
        priority=1,
        condition_name="Client Profile",
        is_triggered=(client_profile_status == "MISSING"),
        reason_code=ReasonCode.CLIENT_PROFILE_MISSING.value,
        current_value=client_profile_status,
        threshold="COMPLETE or PARTIAL",
        description="Client profile required for personalized allocation"
    )
    conditions.append(cond1)

    # Priority 2: Constraints unrepaired
    cond2 = HoldCondition(
        priority=2,
        condition_name="Constraint Status",
        is_triggered=(constraint_status == "UNREPAIRED"),
        reason_code=ReasonCode.CONSTRAINT_VIOLATION_UNREPAIRED.value,
        current_value=constraint_status,
        threshold="OK or REPAIRED",
        description="Asset class constraints must be satisfied"
    )
    conditions.append(cond2)

    # Priority 3: Low confidence
    cond3 = HoldCondition(
        priority=3,
        condition_name="Consensus Confidence",
        is_triggered=(agent_consensus_confidence < config.confidence_threshold_low),
        reason_code=ReasonCode.LOW_CONFIDENCE.value,
        current_value=f"{agent_consensus_confidence:.2f}",
        threshold=f">= {config.confidence_threshold_low:.2f}",
        description="Minimum confidence required for action"
    )
    conditions.append(cond3)

    # Priority 4: Agent conflict
    cond4 = HoldCondition(
        priority=4,
        condition_name="Agent Agreement",
        is_triggered=(not modes_agree and agent_consensus_confidence < config.confidence_threshold_high),
        reason_code=ReasonCode.AGENT_CONFLICT.value,
        current_value=f"agree={modes_agree}, conf={agent_consensus_confidence:.2f}",
        threshold=f"agree=True OR conf>={config.confidence_threshold_high:.2f}",
        description="Agents must agree or have high confidence"
    )
    conditions.append(cond4)

    # Priority 5: Regime-stance mismatch
    regime_stance_conflict = (
        (regime_signal == "BULL" and agent_consensus_stance == "BEARISH") or
        (regime_signal == "BEAR" and agent_consensus_stance == "BULLISH")
    )
    cond5 = HoldCondition(
        priority=5,
        condition_name="Regime-Stance Alignment",
        is_triggered=(regime_stance_conflict and agent_consensus_confidence < config.confidence_threshold_high),
        reason_code=ReasonCode.REGIME_STANCE_MISMATCH.value,
        current_value=f"regime={regime_signal}, stance={agent_consensus_stance}",
        threshold="Aligned OR high confidence",
        description="Regime and consensus must align or have high confidence"
    )
    conditions.append(cond5)

    # Priority 6: Data quality
    cond6 = HoldCondition(
        priority=6,
        condition_name="Data Quality",
        is_triggered=(data_quality == "DEGRADED"),
        reason_code=ReasonCode.DATA_QUALITY_ISSUE.value,
        current_value=data_quality,
        threshold="COMPLETE or PARTIAL",
        description="Data quality must be acceptable"
    )
    conditions.append(cond6)

    report.hold_conditions = conditions

    # Check for any triggered conditions
    triggered = [c for c in conditions if c.is_triggered]
    report.triggered_conditions = [c.reason_code for c in triggered]

    if triggered:
        report.is_hold = True
        report.primary_reason = triggered[0].reason_code  # Highest priority

        # Build exit requirements
        for c in triggered:
            if c.condition_name == "Client Profile":
                report.exit_requirements.append("Provide complete client profile")
            elif c.condition_name == "Constraint Status":
                report.exit_requirements.append("Repair or relax constraint violations")
            elif c.condition_name == "Consensus Confidence":
                report.exit_requirements.append(f"Increase confidence above {config.confidence_threshold_low:.0%}")
            elif c.condition_name == "Agent Agreement":
                report.exit_requirements.append("Resolve agent disagreement or increase confidence")
            elif c.condition_name == "Regime-Stance Alignment":
                report.exit_requirements.append("Wait for regime-stance alignment or increase confidence")
            elif c.condition_name == "Data Quality":
                report.exit_requirements.append("Restore data quality to COMPLETE or PARTIAL")

    return report


# =============================================================================
# Score Definitions (Unified Risk Scoring)
# =============================================================================

# Risk level thresholds (fixed scale 0-100)
RISK_THRESHOLD_LOW = 30.0      # 0-30: LOW risk
RISK_THRESHOLD_HIGH = 70.0     # 70-100: HIGH risk
                               # 30-70: MEDIUM risk


@dataclass
class AuxiliaryRiskMetric:
    """
    보조 리스크 지표 (auxiliary, not used for decisions)

    이 지표들은 정보 제공 목적으로만 사용되며,
    의사결정에는 canonical_risk_score만 사용됨.
    """
    name: str                  # 지표 이름
    value: float               # 지표 값
    unit: str                  # 단위 (points, %, bps 등)
    source: str                # 출처 모듈
    description: str           # 설명

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'source': self.source,
            'description': self.description,
        }


@dataclass
class ScoreDefinitions:
    """
    통합 리스크 점수 정의 (Unified Risk Scoring)

    ============================================================================
    CANONICAL RISK SCORE (의사결정용 공식 점수)
    ============================================================================

    Scale: 0 - 100 (고정)
    Formula: canonical = base + microstructure_adj + bubble_adj + extended_adj
    Clamping: min(100, max(0, canonical))

    Interpretation Bands:
    ---------------------
    | Range   | Level  | Action Guidance                              |
    |---------|--------|----------------------------------------------|
    | 0-30    | LOW    | 공격적 포지션 허용                           |
    | 30-70   | MEDIUM | 표준 리스크 관리 적용                        |
    | 70-100  | HIGH   | 방어적 스탠스, 포지션 축소 고려              |

    Decision Rule Usage:
    --------------------
    - RULE_4 (High Risk Override): canonical_risk_score >= 70 → 방어적 스탠스
    - resolve_final_decision(): canonical_risk_score만 참조

    ============================================================================
    AUXILIARY SUB-SCORES (정보 제공용, 의사결정에 미사용)
    ============================================================================

    | Sub-Score                | Range   | Source                    |
    |--------------------------|---------|---------------------------|
    | base_risk_score          | 0-100   | CriticalPathAggregator    |
    | microstructure_adjustment| ±10     | DailyMicrostructureAnalyzer|
    | bubble_risk_adjustment   | +0~15   | BubbleDetector            |
    | extended_data_adjustment | ±15     | PCR/Sentiment/Credit      |

    Note: Sub-scores are ONLY for transparency and audit.
          All decision logic uses canonical_risk_score exclusively.
    """

    # =========================================================================
    # CANONICAL RISK SCORE (의사결정용 - 이 값만 decision rules에서 사용)
    # =========================================================================
    canonical_risk_score: float = 50.0     # Scale: 0-100, clamped

    # =========================================================================
    # INTERPRETATION (canonical_risk_score 기반)
    # =========================================================================
    risk_level: str = "MEDIUM"             # LOW (0-30) / MEDIUM (30-70) / HIGH (70-100)
    risk_level_threshold_low: float = field(default=RISK_THRESHOLD_LOW, repr=False)
    risk_level_threshold_high: float = field(default=RISK_THRESHOLD_HIGH, repr=False)

    # =========================================================================
    # AUXILIARY SUB-SCORES (정보 제공용 - 의사결정에 사용 금지)
    # =========================================================================
    # Note: These are labeled "auxiliary" to emphasize they are not for decisions

    # Base score from CriticalPathAggregator
    aux_base_risk_score: float = 50.0
    aux_base_source: str = "CriticalPathAggregator"

    # Microstructure adjustment (market quality)
    aux_microstructure_adjustment: float = 0.0
    aux_microstructure_source: str = "DailyMicrostructureAnalyzer"

    # Bubble risk overlay
    aux_bubble_risk_adjustment: float = 0.0
    aux_bubble_source: str = "BubbleDetector"

    # Extended data adjustment (PCR, sentiment, credit)
    aux_extended_data_adjustment: float = 0.0
    aux_extended_source: str = "ExtendedDataCollector"

    # =========================================================================
    # OPTIONAL: Additional auxiliary metrics (for information only)
    # =========================================================================
    auxiliary_metrics: List[AuxiliaryRiskMetric] = field(default_factory=list)

    def __post_init__(self):
        """Calculate risk_level from canonical_risk_score"""
        self.risk_level = self._interpret_risk_level(self.canonical_risk_score)

    def _interpret_risk_level(self, score: float) -> str:
        """
        Interpret canonical risk score into risk level.

        Fixed bands (not configurable to ensure consistency):
        - LOW: 0-30
        - MEDIUM: 30-70
        - HIGH: 70-100
        """
        if score < self.risk_level_threshold_low:
            return "LOW"
        elif score >= self.risk_level_threshold_high:
            return "HIGH"
        else:
            return "MEDIUM"

    @classmethod
    def from_components(
        cls,
        base_risk_score: float,
        microstructure_adjustment: float = 0.0,
        bubble_risk_adjustment: float = 0.0,
        extended_data_adjustment: float = 0.0,
        auxiliary_metrics: List[AuxiliaryRiskMetric] = None
    ) -> 'ScoreDefinitions':
        """
        Create ScoreDefinitions from component scores.

        Computes canonical_risk_score = base + micro + bubble + extended,
        clamped to [0, 100].
        """
        canonical = base_risk_score + microstructure_adjustment + bubble_risk_adjustment + extended_data_adjustment
        canonical = max(0.0, min(100.0, canonical))

        return cls(
            canonical_risk_score=canonical,
            aux_base_risk_score=base_risk_score,
            aux_microstructure_adjustment=microstructure_adjustment,
            aux_bubble_risk_adjustment=bubble_risk_adjustment,
            aux_extended_data_adjustment=extended_data_adjustment,
            auxiliary_metrics=auxiliary_metrics or [],
        )

    def to_dict(self) -> Dict:
        """Export as dictionary for JSON serialization"""
        return {
            # Canonical score (for decisions)
            'canonical_risk_score': self.canonical_risk_score,
            'risk_level': self.risk_level,
            'scale': {
                'min': 0,
                'max': 100,
                'interpretation': {
                    'LOW': f'0 - {self.risk_level_threshold_low}',
                    'MEDIUM': f'{self.risk_level_threshold_low} - {self.risk_level_threshold_high}',
                    'HIGH': f'{self.risk_level_threshold_high} - 100',
                }
            },
            # Auxiliary sub-scores (for transparency only)
            'auxiliary_sub_scores': {
                'base_risk_score': {
                    'value': self.aux_base_risk_score,
                    'source': self.aux_base_source,
                    'note': 'AUXILIARY - not used for decisions'
                },
                'microstructure_adjustment': {
                    'value': self.aux_microstructure_adjustment,
                    'source': self.aux_microstructure_source,
                    'note': 'AUXILIARY - not used for decisions'
                },
                'bubble_risk_adjustment': {
                    'value': self.aux_bubble_risk_adjustment,
                    'source': self.aux_bubble_source,
                    'note': 'AUXILIARY - not used for decisions'
                },
                'extended_data_adjustment': {
                    'value': self.aux_extended_data_adjustment,
                    'source': self.aux_extended_source,
                    'note': 'AUXILIARY - not used for decisions'
                },
            },
            'formula': 'canonical = base + microstructure_adj + bubble_adj + extended_adj',
            'additional_auxiliary_metrics': [m.to_dict() for m in self.auxiliary_metrics],
        }

    def to_markdown(self) -> str:
        """
        Generate score_definitions section for report.

        This section documents the unified risk scoring system.
        """
        lines = []
        lines.append("## score_definitions")
        lines.append("")

        # Canonical Score
        lines.append("### Canonical Risk Score")
        lines.append("")
        lines.append(f"**Score: {self.canonical_risk_score:.1f} / 100**")
        lines.append(f"**Level: {self.risk_level}**")
        lines.append("")

        # Scale interpretation
        lines.append("### Scale Interpretation")
        lines.append("")
        lines.append("| Range | Level | Description |")
        lines.append("|-------|-------|-------------|")
        lines.append(f"| 0 - {self.risk_level_threshold_low:.0f} | LOW | 공격적 포지션 허용 |")
        lines.append(f"| {self.risk_level_threshold_low:.0f} - {self.risk_level_threshold_high:.0f} | MEDIUM | 표준 리스크 관리 |")
        lines.append(f"| {self.risk_level_threshold_high:.0f} - 100 | HIGH | 방어적 스탠스 권고 |")
        lines.append("")

        # Formula
        lines.append("### Formula")
        lines.append("")
        lines.append("```")
        lines.append("canonical_risk_score = base + microstructure_adj + bubble_adj + extended_adj")
        lines.append(f"                     = {self.aux_base_risk_score:.1f} + ({self.aux_microstructure_adjustment:+.1f}) + ({self.aux_bubble_risk_adjustment:+.1f}) + ({self.aux_extended_data_adjustment:+.1f})")
        lines.append(f"                     = {self.canonical_risk_score:.1f}")
        lines.append("```")
        lines.append("")

        # Auxiliary sub-scores
        lines.append("### Auxiliary Sub-Scores")
        lines.append("")
        lines.append("**Note:** These sub-scores are for transparency only. Decision rules use ONLY `canonical_risk_score`.")
        lines.append("")
        lines.append("| Sub-Score | Value | Source | Purpose |")
        lines.append("|-----------|-------|--------|---------|")
        lines.append(f"| base_risk_score | {self.aux_base_risk_score:.1f} | {self.aux_base_source} | 기본 리스크 |")
        lines.append(f"| microstructure_adj | {self.aux_microstructure_adjustment:+.1f} | {self.aux_microstructure_source} | 시장 미세구조 |")
        lines.append(f"| bubble_risk_adj | {self.aux_bubble_risk_adjustment:+.1f} | {self.aux_bubble_source} | 버블 리스크 |")
        lines.append(f"| extended_data_adj | {self.aux_extended_data_adjustment:+.1f} | {self.aux_extended_source} | PCR/감성/신용 |")
        lines.append("")

        # Additional auxiliary metrics
        if self.auxiliary_metrics:
            lines.append("### Additional Auxiliary Metrics")
            lines.append("")
            lines.append("| Metric | Value | Unit | Source |")
            lines.append("|--------|-------|------|--------|")
            for m in self.auxiliary_metrics:
                lines.append(f"| {m.name} | {m.value:.2f} | {m.unit} | {m.source} |")
            lines.append("")

        # Decision usage note
        lines.append("### Usage in Decision Rules")
        lines.append("")
        lines.append("```")
        lines.append("RULE_4 (High Risk Override):")
        lines.append(f"  IF canonical_risk_score >= {self.risk_level_threshold_high:.0f}")
        lines.append("  THEN apply defensive stance")
        lines.append("")
        lines.append("All other rules reference canonical_risk_score exclusively.")
        lines.append("Sub-scores are NEVER used in decision logic.")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    # Backward compatibility aliases
    @property
    def base_risk_score(self) -> float:
        """Alias for aux_base_risk_score (backward compatibility)"""
        return self.aux_base_risk_score

    @property
    def microstructure_adjustment(self) -> float:
        """Alias for aux_microstructure_adjustment (backward compatibility)"""
        return self.aux_microstructure_adjustment

    @property
    def bubble_risk_adjustment(self) -> float:
        """Alias for aux_bubble_risk_adjustment (backward compatibility)"""
        return self.aux_bubble_risk_adjustment

    @property
    def extended_data_adjustment(self) -> float:
        """Alias for aux_extended_data_adjustment (backward compatibility)"""
        return self.aux_extended_data_adjustment


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
    """
    제약 위반 상세 정보

    Documents:
    - Which asset class violated constraints
    - Type of violation (ABOVE_MAX / BELOW_MIN)
    - Current weight vs limit
    - Delta (amount of violation)
    """
    asset_class: str
    violation_type: str              # BELOW_MIN / ABOVE_MAX
    current_value: float             # Actual weight before repair
    limit_value: float               # Constraint limit (min or max)
    delta: float                     # |current - limit|
    severity: str = "MINOR"          # MINOR (<5%), MODERATE (5-10%), SEVERE (>10%)

    def __post_init__(self):
        """Calculate severity based on delta"""
        if self.delta >= 0.10:
            self.severity = "SEVERE"
        elif self.delta >= 0.05:
            self.severity = "MODERATE"
        else:
            self.severity = "MINOR"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AssetClassWeights:
    """자산군별 비중 (before/after comparison)"""
    asset_class: str
    original_weight: float
    repaired_weight: float
    delta: float
    min_bound: float
    max_bound: float
    status: str                      # OK / VIOLATED / REPAIRED / STILL_VIOLATED

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ConstraintRepairResult:
    """
    제약 수정 결과 (Automatic Constraint Repair)

    ============================================================================
    REPAIR PROCESS
    ============================================================================
    1. Identify violations (ABOVE_MAX or BELOW_MIN)
    2. For ABOVE_MAX: proportionally reduce weights in violating class
    3. For BELOW_MIN: transfer from largest class to deficient class
    4. Normalize weights to sum to 1.0
    5. Verify all constraints satisfied

    ============================================================================
    REPAIR OUTCOMES
    ============================================================================
    - repair_successful=True, constraints_satisfied=True: All fixed
    - repair_successful=True, constraints_satisfied=False: Partial fix
    - repair_successful=False: Could not repair (force HOLD)

    ============================================================================
    FORCE HOLD CONDITIONS
    ============================================================================
    - No donor class available for BELOW_MIN
    - Insufficient donor weight
    - Circular constraint dependencies
    """

    # Original and repaired weights (per-asset)
    original_weights: Dict[str, float]
    repaired_weights: Dict[str, float]

    # Violations found
    violations_found: List[ConstraintViolation] = field(default_factory=list)

    # Repair actions taken
    repair_actions: List[str] = field(default_factory=list)

    # Outcome flags
    repair_successful: bool = True           # Was repair attempted successfully?
    constraints_satisfied: bool = True       # Are all constraints satisfied after repair?
    force_hold: bool = False                 # Should force HOLD due to failed repair?
    force_hold_reason: str = ""              # Why HOLD is forced

    # Asset class comparison (before/after)
    asset_class_comparison: List[AssetClassWeights] = field(default_factory=list)

    # Verification results
    verification_passed: bool = True
    verification_details: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'original_weights': self.original_weights,
            'repaired_weights': self.repaired_weights,
            'violations_found': [v.to_dict() for v in self.violations_found],
            'repair_actions': self.repair_actions,
            'repair_successful': self.repair_successful,
            'constraints_satisfied': self.constraints_satisfied,
            'force_hold': self.force_hold,
            'force_hold_reason': self.force_hold_reason,
            'asset_class_comparison': [c.to_dict() for c in self.asset_class_comparison],
            'verification': {
                'passed': self.verification_passed,
                'details': self.verification_details,
            }
        }

    def to_markdown(self) -> str:
        """
        Generate constraint_repair section for report.
        """
        lines = []
        lines.append("## constraint_repair")
        lines.append("")

        # Status Summary
        lines.append("### Repair Status")
        lines.append("")
        if not self.violations_found:
            lines.append("**Status: NO VIOLATIONS** ✓")
            lines.append("")
            lines.append("All asset class constraints are satisfied. No repair needed.")
            lines.append("")
            return "\n".join(lines)

        if self.constraints_satisfied:
            lines.append("**Status: REPAIRED** ✓")
            lines.append("")
            lines.append("Constraints were violated but successfully repaired.")
        elif self.force_hold:
            lines.append("**Status: REPAIR FAILED → FORCE HOLD** ✗")
            lines.append("")
            lines.append(f"**Reason:** {self.force_hold_reason}")
            lines.append("")
            lines.append("⚠️ **Action blocked.** Manual intervention required.")
        else:
            lines.append("**Status: PARTIALLY REPAIRED** ⚠️")
            lines.append("")
            lines.append("Some constraints could not be fully satisfied.")
        lines.append("")

        # Violations Found
        lines.append("### Violations Found")
        lines.append("")
        lines.append("| Asset Class | Type | Current | Limit | Delta | Severity |")
        lines.append("|-------------|------|---------|-------|-------|----------|")
        for v in self.violations_found:
            type_icon = "▲" if v.violation_type == "ABOVE_MAX" else "▼"
            severity_icon = {"MINOR": "🟡", "MODERATE": "🟠", "SEVERE": "🔴"}.get(v.severity, "")
            lines.append(f"| {v.asset_class} | {type_icon} {v.violation_type} | {v.current_value:.1%} | {v.limit_value:.1%} | {v.delta:.1%} | {severity_icon} {v.severity} |")
        lines.append("")

        # Asset Class Comparison (Before/After)
        if self.asset_class_comparison:
            lines.append("### Asset Class Comparison (Before → After)")
            lines.append("")
            lines.append("| Asset Class | Original | Repaired | Change | Bounds | Status |")
            lines.append("|-------------|----------|----------|--------|--------|--------|")
            for c in self.asset_class_comparison:
                status_icon = {"OK": "✓", "REPAIRED": "✓", "VIOLATED": "✗", "STILL_VIOLATED": "✗"}.get(c.status, "")
                lines.append(f"| {c.asset_class} | {c.original_weight:.1%} | {c.repaired_weight:.1%} | {c.delta:+.1%} | [{c.min_bound:.0%}-{c.max_bound:.0%}] | {status_icon} {c.status} |")
            lines.append("")

        # Repair Actions
        lines.append("### Repair Actions Taken")
        lines.append("")
        for i, action in enumerate(self.repair_actions, 1):
            lines.append(f"{i}. {action}")
        lines.append("")

        # Verification
        lines.append("### Verification")
        lines.append("")
        if self.verification_passed:
            lines.append("**Result: PASSED** ✓")
        else:
            lines.append("**Result: FAILED** ✗")
        lines.append("")
        for detail in self.verification_details:
            lines.append(f"- {detail}")
        lines.append("")

        # Weight Changes (top changes)
        if self.original_weights and self.repaired_weights:
            lines.append("### Weight Changes (Top 10)")
            lines.append("")
            lines.append("| Asset | Original | Repaired | Change |")
            lines.append("|-------|----------|----------|--------|")

            all_assets = set(self.original_weights.keys()) | set(self.repaired_weights.keys())
            changes = []
            for asset in all_assets:
                orig = self.original_weights.get(asset, 0.0)
                rep = self.repaired_weights.get(asset, 0.0)
                delta = rep - orig
                if abs(delta) > 0.001:
                    changes.append((asset, orig, rep, delta))

            changes.sort(key=lambda x: abs(x[3]), reverse=True)
            for asset, orig, rep, delta in changes[:10]:
                lines.append(f"| {asset} | {orig:.2%} | {rep:.2%} | {delta:+.2%} |")
            lines.append("")

        return "\n".join(lines)


def repair_constraints(
    target_weights: Dict[str, float],
    config: OperationalConfig,
    asset_class_map: Dict[str, str] = None
) -> ConstraintRepairResult:
    """
    자동 제약 수정 (Automatic Constraint Repair)

    ============================================================================
    REPAIR PROCESS
    ============================================================================
    1. Calculate asset class totals
    2. Identify constraint violations
    3. For ABOVE_MAX: proportionally reduce weights in violating class
    4. For BELOW_MIN: transfer from largest class to deficient class
    5. Normalize weights to sum to 1.0
    6. Remove negative weights
    7. Verify all constraints satisfied after repair

    ============================================================================
    FORCE HOLD CONDITIONS
    ============================================================================
    - No assets in violating class
    - No donor class available for BELOW_MIN
    - Insufficient donor weight
    - Constraints still violated after repair

    Args:
        target_weights: 목표 비중
        config: 운영 설정
        asset_class_map: 자산→자산군 매핑

    Returns:
        ConstraintRepairResult with full documentation
    """
    if asset_class_map is None:
        asset_class_map = ASSET_CLASS_MAP

    result = ConstraintRepairResult(
        original_weights=target_weights.copy(),
        repaired_weights={},
    )

    # Step 1: Calculate asset class totals
    class_weights = {'equity': 0.0, 'bond': 0.0, 'cash': 0.0, 'commodity': 0.0, 'crypto': 0.0}
    class_assets = {'equity': [], 'bond': [], 'cash': [], 'commodity': [], 'crypto': []}

    for asset, weight in target_weights.items():
        asset_class = asset_class_map.get(asset, 'other')
        if asset_class in class_weights:
            class_weights[asset_class] += weight
            class_assets[asset_class].append(asset)

    original_class_weights = class_weights.copy()

    # Step 2: Define constraints
    constraints = {
        'equity': (config.equity_min, config.equity_max),
        'bond': (config.bond_min, config.bond_max),
        'cash': (config.cash_min, config.cash_max),
        'commodity': (config.commodity_min, config.commodity_max),
        'crypto': (config.crypto_min, config.crypto_max),
    }

    # Step 3: Identify violations
    for asset_class, (min_bound, max_bound) in constraints.items():
        current = class_weights[asset_class]
        if current < min_bound - 0.0001:  # Small tolerance
            result.violations_found.append(ConstraintViolation(
                asset_class=asset_class,
                violation_type="BELOW_MIN",
                current_value=current,
                limit_value=min_bound,
                delta=min_bound - current
            ))
        elif current > max_bound + 0.0001:  # Small tolerance
            result.violations_found.append(ConstraintViolation(
                asset_class=asset_class,
                violation_type="ABOVE_MAX",
                current_value=current,
                limit_value=max_bound,
                delta=current - max_bound
            ))

    # No violations - return early
    if not result.violations_found:
        result.repaired_weights = target_weights.copy()
        result.repair_actions.append("No violations found - no repair needed")
        result.verification_passed = True
        result.verification_details.append("All asset class weights within bounds")

        # Build asset class comparison
        for ac, (min_b, max_b) in constraints.items():
            result.asset_class_comparison.append(AssetClassWeights(
                asset_class=ac,
                original_weight=original_class_weights[ac],
                repaired_weight=original_class_weights[ac],
                delta=0.0,
                min_bound=min_b,
                max_bound=max_b,
                status="OK"
            ))
        return result

    result.repair_actions.append(f"Found {len(result.violations_found)} constraint violation(s)")

    # Step 4: Perform repairs
    repaired = target_weights.copy()
    force_hold_reasons = []

    for violation in result.violations_found:
        assets_in_class = class_assets[violation.asset_class]

        if not assets_in_class:
            msg = f"Cannot repair {violation.asset_class}: no assets in class"
            result.repair_actions.append(msg)
            force_hold_reasons.append(msg)
            result.repair_successful = False
            continue

        if violation.violation_type == "ABOVE_MAX":
            # Proportionally reduce weights in violating class
            excess = violation.delta
            current_class_weight = sum(repaired.get(a, 0) for a in assets_in_class)

            if current_class_weight > 0:
                scale = (current_class_weight - excess) / current_class_weight
                result.repair_actions.append(
                    f"Reducing {violation.asset_class} by {excess:.1%} (scale factor: {scale:.3f})"
                )
                for asset in assets_in_class:
                    if asset in repaired:
                        old_w = repaired[asset]
                        repaired[asset] = old_w * scale
                        result.repair_actions.append(
                            f"  {asset}: {old_w:.3f} → {repaired[asset]:.3f}"
                        )

        elif violation.violation_type == "BELOW_MIN":
            # Transfer from largest class to deficient class
            shortfall = violation.delta
            result.repair_actions.append(
                f"Need to add {shortfall:.1%} to {violation.asset_class}"
            )

            # Find donor class (largest available)
            other_classes = [
                c for c in class_weights
                if c != violation.asset_class and class_weights[c] > shortfall
            ]

            if not other_classes:
                msg = f"Cannot repair {violation.asset_class}: no donor class with sufficient weight"
                result.repair_actions.append(msg)
                force_hold_reasons.append(msg)
                result.repair_successful = False
                continue

            largest_class = max(other_classes, key=lambda c: class_weights[c])
            donor_assets = class_assets[largest_class]
            donor_total = sum(repaired.get(a, 0) for a in donor_assets)

            if donor_total > shortfall:
                # Proportionally reduce donor class
                result.repair_actions.append(
                    f"Transferring {shortfall:.1%} from {largest_class} to {violation.asset_class}"
                )
                for asset in donor_assets:
                    if asset in repaired:
                        reduction = repaired[asset] * (shortfall / donor_total)
                        repaired[asset] -= reduction
                        result.repair_actions.append(
                            f"  {asset}: reduced by {reduction:.3f}"
                        )

                # Distribute to recipient class
                if assets_in_class:
                    per_asset = shortfall / len(assets_in_class)
                    for asset in assets_in_class:
                        if asset in repaired:
                            repaired[asset] += per_asset
                        else:
                            repaired[asset] = per_asset
                        result.repair_actions.append(
                            f"  {asset}: increased by {per_asset:.3f}"
                        )

                # Update class weights for next iteration
                class_weights[largest_class] -= shortfall
                class_weights[violation.asset_class] += shortfall
            else:
                msg = f"Cannot fully repair {violation.asset_class}: insufficient donor weight ({donor_total:.1%} < {shortfall:.1%})"
                result.repair_actions.append(msg)
                force_hold_reasons.append(msg)
                result.repair_successful = False

    # Step 5: Normalize weights
    total = sum(repaired.values())
    if total > 0:
        repaired = {k: v / total for k, v in repaired.items()}
        result.repair_actions.append(f"Normalized weights (sum was {total:.3f})")

    # Step 6: Remove negative weights
    negative_assets = [k for k, v in repaired.items() if v < 0]
    if negative_assets:
        result.repair_actions.append(f"Removing negative weights: {negative_assets}")
        repaired = {k: max(0, v) for k, v in repaired.items()}
        total = sum(repaired.values())
        if total > 0:
            repaired = {k: v / total for k, v in repaired.items()}

    result.repaired_weights = repaired

    # Step 7: Verify constraints after repair
    new_class_weights = {'equity': 0.0, 'bond': 0.0, 'cash': 0.0, 'commodity': 0.0, 'crypto': 0.0}
    for asset, weight in repaired.items():
        asset_class = asset_class_map.get(asset, 'other')
        if asset_class in new_class_weights:
            new_class_weights[asset_class] += weight

    result.verification_passed = True
    still_violated = []

    for asset_class, (min_bound, max_bound) in constraints.items():
        new_weight = new_class_weights[asset_class]
        orig_weight = original_class_weights[asset_class]

        # Determine status
        was_violated = any(
            v.asset_class == asset_class for v in result.violations_found
        )

        if new_weight < min_bound - 0.001 or new_weight > max_bound + 0.001:
            status = "STILL_VIOLATED"
            still_violated.append(asset_class)
            result.verification_passed = False
            result.verification_details.append(
                f"✗ {asset_class}: {new_weight:.1%} not in [{min_bound:.0%}, {max_bound:.0%}]"
            )
        elif was_violated:
            status = "REPAIRED"
            result.verification_details.append(
                f"✓ {asset_class}: repaired to {new_weight:.1%}"
            )
        else:
            status = "OK"
            result.verification_details.append(
                f"✓ {asset_class}: {new_weight:.1%} (no change needed)"
            )

        result.asset_class_comparison.append(AssetClassWeights(
            asset_class=asset_class,
            original_weight=orig_weight,
            repaired_weight=new_weight,
            delta=new_weight - orig_weight,
            min_bound=min_bound,
            max_bound=max_bound,
            status=status
        ))

    # Set final status
    result.constraints_satisfied = len(still_violated) == 0

    if still_violated:
        result.repair_actions.append(
            f"Constraints still violated after repair: {still_violated}"
        )

    if result.constraints_satisfied:
        result.repair_actions.append("✓ All constraints satisfied after repair")
    else:
        result.repair_actions.append("✗ Some constraints could not be satisfied")

    # Determine if HOLD should be forced
    if not result.repair_successful or not result.constraints_satisfied:
        result.force_hold = True
        if force_hold_reasons:
            result.force_hold_reason = "; ".join(force_hold_reasons)
        elif still_violated:
            result.force_hold_reason = f"Constraints still violated: {', '.join(still_violated)}"
        else:
            result.force_hold_reason = "Repair unsuccessful"

    return result


# =============================================================================
# Rebalancing Execution Plan
# =============================================================================

@dataclass
class CostBreakdown:
    """거래 비용 상세"""
    commission: float = 0.0      # 수수료
    spread: float = 0.0          # 스프레드
    market_impact: float = 0.0   # 시장 충격
    total: float = 0.0           # 총 비용

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TradeItem:
    """
    개별 거래 항목 (Execution Plan 단위)

    Per-asset information:
    - current vs target weight
    - delta (weight difference)
    - action (BUY/SELL/HOLD)
    - cost breakdown
    """
    ticker: str
    asset_class: str                       # equity/bond/commodity/crypto/cash
    current_weight: float
    target_weight: float
    delta_weight: float
    delta_pct: float                       # delta as % of current (for drift)
    action: str                            # BUY / SELL / HOLD
    cost_breakdown: CostBreakdown
    estimated_cost: float                  # total cost (for backward compat)
    priority: int = 0                      # execution priority (1=highest)

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'asset_class': self.asset_class,
            'current_weight': self.current_weight,
            'target_weight': self.target_weight,
            'delta_weight': self.delta_weight,
            'delta_pct': self.delta_pct,
            'action': self.action,
            'cost_breakdown': self.cost_breakdown.to_dict(),
            'estimated_cost': self.estimated_cost,
            'priority': self.priority,
        }


@dataclass
class AssetClassSummary:
    """자산군별 요약"""
    asset_class: str
    current_weight: float
    target_weight: float
    delta_weight: float
    trade_count: int
    total_cost: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RebalancePlan:
    """
    리밸런싱 실행 계획 (Execution Plan)

    ============================================================================
    TRIGGER TYPES
    ============================================================================
    - DRIFT: 목표 대비 편차가 임계값 초과
    - REGIME_CHANGE: 시장 레짐 변화 감지
    - CONSTRAINT_REPAIR: 제약 위반 수정 필요
    - SCHEDULED: 정기 리밸런싱 (월간/분기)
    - MANUAL: 수동 요청

    ============================================================================
    EXECUTION STATUS
    ============================================================================
    - should_execute=True: 거래 실행 권고
    - should_execute=False: 거래 보류 (not_executed_reason 참조)

    ============================================================================
    HUMAN APPROVAL
    ============================================================================
    - requires_human_approval=True when:
      - Total turnover >= approval threshold (default 20%)
      - Large single-asset trade
      - Cross-asset-class rebalancing
    """

    # Required fields (no defaults) must come first
    should_execute: bool                   # Whether to execute rebalancing
    trigger_type: str                      # DRIFT/REGIME_CHANGE/CONSTRAINT_REPAIR/SCHEDULED/MANUAL
    trigger_reason: str                    # Detailed trigger description

    # Optional fields with defaults
    not_executed_reason: str = ""          # Why not executed (if should_execute=False)

    # Trade list (per-asset)
    trades: List[TradeItem] = field(default_factory=list)

    # Asset class summary
    asset_class_summary: List[AssetClassSummary] = field(default_factory=list)

    # Aggregate summary
    total_turnover: float = 0.0            # One-way turnover (sum of |delta|/2)
    total_estimated_cost: float = 0.0      # Total trading cost
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0

    # Cost breakdown (aggregate)
    total_commission: float = 0.0
    total_spread: float = 0.0
    total_market_impact: float = 0.0

    # Approval requirements
    requires_human_approval: bool = False
    approval_reason: str = ""
    approval_checklist: List[str] = field(default_factory=list)

    # Execution metadata
    turnover_cap_applied: bool = False
    original_turnover: float = 0.0         # Before cap

    def to_dict(self) -> Dict:
        return {
            'execution': {
                'should_execute': self.should_execute,
                'not_executed_reason': self.not_executed_reason,
            },
            'trigger': {
                'type': self.trigger_type,
                'reason': self.trigger_reason,
            },
            'trades': [t.to_dict() for t in self.trades],
            'asset_class_summary': [s.to_dict() for s in self.asset_class_summary],
            'summary': {
                'total_turnover': self.total_turnover,
                'total_estimated_cost': self.total_estimated_cost,
                'buy_count': self.buy_count,
                'sell_count': self.sell_count,
                'hold_count': self.hold_count,
            },
            'cost_breakdown': {
                'commission': self.total_commission,
                'spread': self.total_spread,
                'market_impact': self.total_market_impact,
                'total': self.total_estimated_cost,
            },
            'approval': {
                'requires_human_approval': self.requires_human_approval,
                'approval_reason': self.approval_reason,
                'approval_checklist': self.approval_checklist,
            },
            'metadata': {
                'turnover_cap_applied': self.turnover_cap_applied,
                'original_turnover': self.original_turnover,
            },
        }

    def to_markdown(self) -> str:
        """
        Generate rebalance_plan section for report.
        """
        lines = []
        lines.append("## rebalance_plan")
        lines.append("")

        # Execution Status
        lines.append("### Execution Status")
        lines.append("")
        if self.should_execute:
            lines.append(f"**Status: EXECUTE** ✓")
        else:
            lines.append(f"**Status: NOT EXECUTED** ✗")
            lines.append(f"**Reason: {self.not_executed_reason}**")
        lines.append("")

        # Trigger Classification
        lines.append("### Trigger Classification")
        lines.append("")
        lines.append(f"| Field | Value |")
        lines.append(f"|-------|-------|")
        lines.append(f"| Type | `{self.trigger_type}` |")
        lines.append(f"| Reason | {self.trigger_reason} |")
        lines.append("")

        # Trigger type explanation
        trigger_explanations = {
            'DRIFT': '목표 대비 편차가 임계값 초과',
            'REGIME_CHANGE': '시장 레짐 변화 감지 (Bull↔Bear↔Neutral)',
            'CONSTRAINT_REPAIR': '자산군 제약 위반 수정',
            'SCHEDULED': '정기 리밸런싱 (월간/분기)',
            'MANUAL': '수동 요청에 의한 리밸런싱',
        }
        explanation = trigger_explanations.get(self.trigger_type, 'N/A')
        lines.append(f"*Trigger: {explanation}*")
        lines.append("")

        if not self.should_execute:
            lines.append("---")
            lines.append("*Rebalancing not executed. See reason above.*")
            lines.append("")
            return "\n".join(lines)

        # Summary Statistics
        lines.append("### Summary Statistics")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Turnover | {self.total_turnover:.2%} |")
        lines.append(f"| Total Estimated Cost | {self.total_estimated_cost:.4f} |")
        lines.append(f"| BUY Orders | {self.buy_count} |")
        lines.append(f"| SELL Orders | {self.sell_count} |")
        lines.append(f"| HOLD (no trade) | {self.hold_count} |")
        if self.turnover_cap_applied:
            lines.append(f"| Turnover Cap Applied | Yes (from {self.original_turnover:.2%}) |")
        lines.append("")

        # Cost Breakdown
        lines.append("### Cost Breakdown")
        lines.append("")
        lines.append(f"| Cost Component | Amount |")
        lines.append(f"|----------------|--------|")
        lines.append(f"| Commission | {self.total_commission:.6f} |")
        lines.append(f"| Spread | {self.total_spread:.6f} |")
        lines.append(f"| Market Impact | {self.total_market_impact:.6f} |")
        lines.append(f"| **Total** | **{self.total_estimated_cost:.6f}** |")
        lines.append("")

        # Asset Class Summary
        if self.asset_class_summary:
            lines.append("### Asset Class Summary")
            lines.append("")
            lines.append("| Asset Class | Current | Target | Delta | Trades | Cost |")
            lines.append("|-------------|---------|--------|-------|--------|------|")
            for s in sorted(self.asset_class_summary, key=lambda x: abs(x.delta_weight), reverse=True):
                lines.append(f"| {s.asset_class} | {s.current_weight:.1%} | {s.target_weight:.1%} | {s.delta_weight:+.1%} | {s.trade_count} | {s.total_cost:.4f} |")
            lines.append("")

        # Trade List (per-asset)
        lines.append("### Trade List")
        lines.append("")
        lines.append("| # | Ticker | Class | Current | Target | Delta | Action | Cost |")
        lines.append("|---|--------|-------|---------|--------|-------|--------|------|")

        # Sort by priority, then by absolute delta
        sorted_trades = sorted(
            [t for t in self.trades if t.action != "HOLD"],
            key=lambda x: (-x.priority if x.priority else 0, -abs(x.delta_weight))
        )

        for i, t in enumerate(sorted_trades[:20], 1):  # Top 20 trades
            lines.append(f"| {i} | {t.ticker} | {t.asset_class} | {t.current_weight:.2%} | {t.target_weight:.2%} | {t.delta_weight:+.2%} | {t.action} | {t.estimated_cost:.4f} |")

        if len(sorted_trades) > 20:
            lines.append(f"| ... | *{len(sorted_trades) - 20} more trades* | | | | | | |")
        lines.append("")

        # Human Approval Section
        lines.append("### Human Approval")
        lines.append("")
        if self.requires_human_approval:
            lines.append("**⚠️ HUMAN APPROVAL REQUIRED**")
            lines.append("")
            lines.append(f"Reason: {self.approval_reason}")
            lines.append("")
            if self.approval_checklist:
                lines.append("Approval Checklist:")
                for item in self.approval_checklist:
                    lines.append(f"- [ ] {item}")
                lines.append("")
        else:
            lines.append("No human approval required. Auto-execution permitted.")
            lines.append("")

        return "\n".join(lines)


def generate_rebalance_plan(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    trigger_type: TriggerType,
    trigger_reason: str,
    config: OperationalConfig,
    asset_class_map: Dict[str, str] = None
) -> RebalancePlan:
    """
    리밸런싱 실행 계획 생성 (Enhanced Execution Plan)

    Computes:
    1. Per-asset weight differences (current vs target)
    2. Trade list with delta weights and cost breakdown
    3. Asset class summary
    4. Trigger classification
    5. Human approval requirements

    Args:
        current_weights: 현재 비중
        target_weights: 목표 비중
        trigger_type: 트리거 유형 (DRIFT/REGIME_CHANGE/CONSTRAINT_REPAIR/SCHEDULED/MANUAL)
        trigger_reason: 트리거 사유
        config: 운영 설정
        asset_class_map: 자산→자산군 매핑 (optional)

    Returns:
        RebalancePlan with full execution details
    """
    if asset_class_map is None:
        asset_class_map = ASSET_CLASS_MAP

    trades = []
    total_turnover = 0.0
    total_commission = 0.0
    total_spread = 0.0
    total_market_impact = 0.0
    buy_count = 0
    sell_count = 0
    hold_count = 0

    # Asset class aggregation
    class_data = {}  # asset_class -> {current, target, trades, cost}

    all_tickers = set(current_weights.keys()) | set(target_weights.keys())
    priority_counter = 1

    for ticker in sorted(all_tickers):
        current = current_weights.get(ticker, 0.0)
        target = target_weights.get(ticker, 0.0)
        delta = target - current
        asset_class = asset_class_map.get(ticker, 'other')

        # Initialize asset class data
        if asset_class not in class_data:
            class_data[asset_class] = {
                'current': 0.0, 'target': 0.0, 'trades': 0, 'cost': 0.0
            }
        class_data[asset_class]['current'] += current
        class_data[asset_class]['target'] += target

        # Calculate delta percentage (drift)
        delta_pct = (delta / current * 100) if current > 0 else (100.0 if delta > 0 else 0.0)

        # Determine action
        original_delta = delta
        if abs(delta) < config.min_trade_size:
            action = "HOLD"
            delta = 0.0
            hold_count += 1
        elif delta > 0:
            action = "BUY"
            buy_count += 1
            class_data[asset_class]['trades'] += 1
        else:
            action = "SELL"
            sell_count += 1
            class_data[asset_class]['trades'] += 1

        # Cost calculation (linear model with breakdown)
        trade_value = abs(delta)
        commission = trade_value * config.commission_rate
        spread = trade_value * config.spread_cost
        market_impact = trade_value * config.market_impact * (trade_value ** 0.5)
        total_cost_item = commission + spread + market_impact

        cost_breakdown = CostBreakdown(
            commission=commission,
            spread=spread,
            market_impact=market_impact,
            total=total_cost_item
        )

        # Assign priority based on delta magnitude
        priority = 0
        if action != "HOLD":
            priority = priority_counter
            priority_counter += 1

        trades.append(TradeItem(
            ticker=ticker,
            asset_class=asset_class,
            current_weight=current,
            target_weight=target,
            delta_weight=delta,
            delta_pct=delta_pct,
            action=action,
            cost_breakdown=cost_breakdown,
            estimated_cost=total_cost_item,
            priority=priority
        ))

        # Aggregate totals
        total_turnover += abs(original_delta)
        total_commission += commission
        total_spread += spread
        total_market_impact += market_impact
        class_data[asset_class]['cost'] += total_cost_item

    # One-way turnover (sum of |delta| / 2)
    original_turnover = total_turnover / 2
    total_turnover = original_turnover

    # Apply turnover cap
    turnover_cap_applied = False
    if total_turnover > config.turnover_cap:
        turnover_cap_applied = True
        scale = config.turnover_cap / total_turnover

        for trade in trades:
            trade.delta_weight *= scale
            trade.cost_breakdown.commission *= scale
            trade.cost_breakdown.spread *= scale
            trade.cost_breakdown.market_impact *= scale
            trade.cost_breakdown.total *= scale
            trade.estimated_cost *= scale

        total_turnover = config.turnover_cap
        total_commission *= scale
        total_spread *= scale
        total_market_impact *= scale

        for ac in class_data:
            class_data[ac]['cost'] *= scale

    total_cost = total_commission + total_spread + total_market_impact

    # Build asset class summary
    asset_class_summary = []
    for ac, data in sorted(class_data.items(), key=lambda x: abs(x[1]['target'] - x[1]['current']), reverse=True):
        asset_class_summary.append(AssetClassSummary(
            asset_class=ac,
            current_weight=data['current'],
            target_weight=data['target'],
            delta_weight=data['target'] - data['current'],
            trade_count=data['trades'],
            total_cost=data['cost']
        ))

    # Re-sort trades by absolute delta (priority)
    trades_with_action = [t for t in trades if t.action != "HOLD"]
    trades_with_action.sort(key=lambda x: abs(x.delta_weight), reverse=True)
    for i, t in enumerate(trades_with_action, 1):
        t.priority = i

    # Determine execution status
    should_execute = total_turnover > 0

    # Human approval requirements
    requires_approval = False
    approval_reasons = []
    approval_checklist = []

    if total_turnover >= config.human_approval_threshold:
        requires_approval = True
        approval_reasons.append(f"Total turnover {total_turnover:.1%} >= threshold {config.human_approval_threshold:.1%}")
        approval_checklist.append("Verify portfolio manager authorization")

    # Check for large single trades (> 10% of portfolio)
    large_trades = [t for t in trades if abs(t.delta_weight) > 0.10]
    if large_trades:
        requires_approval = True
        approval_reasons.append(f"{len(large_trades)} trade(s) exceed 10% of portfolio")
        approval_checklist.append(f"Review large trades: {', '.join(t.ticker for t in large_trades)}")

    # Check for cross-asset-class rebalancing
    cross_class_changes = sum(1 for s in asset_class_summary if abs(s.delta_weight) > 0.05)
    if cross_class_changes >= 3:
        requires_approval = True
        approval_reasons.append(f"Cross-asset-class rebalancing ({cross_class_changes} classes affected)")
        approval_checklist.append("Review asset allocation strategy alignment")

    if requires_approval:
        approval_checklist.extend([
            "Confirm market conditions are favorable",
            "Check for upcoming events (earnings, FOMC, etc.)",
            "Verify trading liquidity"
        ])

    approval_reason = "; ".join(approval_reasons) if approval_reasons else ""

    return RebalancePlan(
        should_execute=should_execute,
        not_executed_reason="" if should_execute else "No trades required (all deltas below minimum)",
        trigger_type=trigger_type.value,
        trigger_reason=trigger_reason,
        trades=trades,
        asset_class_summary=asset_class_summary,
        total_turnover=total_turnover,
        total_estimated_cost=total_cost,
        buy_count=buy_count,
        sell_count=sell_count,
        hold_count=hold_count,
        total_commission=total_commission,
        total_spread=total_spread,
        total_market_impact=total_market_impact,
        requires_human_approval=requires_approval,
        approval_reason=approval_reason,
        approval_checklist=approval_checklist,
        turnover_cap_applied=turnover_cap_applied,
        original_turnover=original_turnover,
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
    1. signal_hierarchy - Core vs Auxiliary signal classification
    2. hold_policy - HOLD conditions and status
    3. decision_policy - Final stance and rules applied
    4. score_definitions - Unified risk scoring
    5. allocation - Target weights
    6. constraint_repair - Constraint violations and repairs
    7. rebalance_plan - Execution plan
    8. audit_metadata - Audit trail
    """
    # Required sections
    decision_policy: DecisionPolicy
    score_definitions: ScoreDefinitions
    allocation: Dict[str, float]
    constraint_repair: ConstraintRepairResult
    rebalance_plan: RebalancePlan
    audit_metadata: AuditMetadata

    # New sections for signal hierarchy and HOLD policy
    signal_hierarchy: SignalHierarchyReport = field(default_factory=SignalHierarchyReport)
    hold_policy: HoldPolicyReport = field(default_factory=HoldPolicyReport)

    # Raw inputs for audit
    raw_inputs: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'signal_hierarchy': self.signal_hierarchy.to_dict(),
            'hold_policy': self.hold_policy.to_dict(),
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

        # 1. Signal Hierarchy (use signal_hierarchy.to_markdown())
        signal_md = self.signal_hierarchy.to_markdown()
        signal_md = signal_md.replace("## signal_hierarchy", "## 1. signal_hierarchy", 1)
        md.append(signal_md)
        md.append("")

        # 2. HOLD Policy (use hold_policy.to_markdown())
        hold_md = self.hold_policy.to_markdown()
        hold_md = hold_md.replace("## hold_policy", "## 2. hold_policy", 1)
        md.append(hold_md)
        md.append("")

        # 3. Decision Policy
        md.append("## 3. decision_policy")
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

        # 4. Score Definitions (use unified score_definitions.to_markdown())
        score_md = self.score_definitions.to_markdown()
        score_md = score_md.replace("## score_definitions", "## 4. score_definitions", 1)
        md.append(score_md)
        md.append("")

        # 5. Allocation
        md.append("## 5. allocation")
        md.append("| Asset | Weight |")
        md.append("|-------|--------|")
        for ticker, weight in sorted(self.allocation.items(), key=lambda x: x[1], reverse=True):
            if weight >= 0.01:
                md.append(f"| {ticker} | {weight:.1%} |")
        md.append("")

        # 6. Constraint Repair (use constraint_repair.to_markdown())
        constraint_md = self.constraint_repair.to_markdown()
        constraint_md = constraint_md.replace("## constraint_repair", "## 6. constraint_repair", 1)
        md.append(constraint_md)
        md.append("")

        # 7. Rebalance Plan (use rebalance_plan.to_markdown())
        rebalance_md = self.rebalance_plan.to_markdown()
        rebalance_md = rebalance_md.replace("## rebalance_plan", "## 7. rebalance_plan", 1)
        md.append(rebalance_md)
        md.append("")

        # 8. Audit Metadata
        md.append("## 8. audit_metadata")
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

        # 7. Signal Hierarchy Report
        signal_hierarchy = self._build_signal_hierarchy(
            eimas_result, score_defs, decision_inputs
        )

        # 8. HOLD Policy Evaluation
        hold_policy = evaluate_hold_conditions(
            regime_signal=decision_inputs.regime,
            regime_confidence=decision_inputs.regime_confidence,
            canonical_risk_score=score_defs.canonical_risk_score,
            agent_consensus_stance=decision_inputs.agent_consensus,
            agent_consensus_confidence=decision_inputs.confidence,
            constraint_status=decision_inputs.constraint_status,
            client_profile_status=eimas_result.get('client_profile_status', 'COMPLETE'),
            data_quality=decision_inputs.data_quality,
            modes_agree=decision_inputs.modes_agree,
            config=self.config,
        )

        # 9. Audit Metadata
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
            signal_hierarchy=signal_hierarchy,
            hold_policy=hold_policy,
            raw_inputs={'eimas_result': eimas_result},
        )

    def _extract_scores(self, data: Dict) -> ScoreDefinitions:
        """
        리스크 점수 추출 (Unified Risk Scoring)

        Uses ScoreDefinitions.from_components() to ensure:
        - canonical_risk_score is computed consistently
        - risk_level is derived from canonical score
        - sub-scores are labeled as auxiliary
        """
        base = data.get('base_risk_score', data.get('risk_score', 50.0))
        micro_adj = data.get('microstructure_adjustment', 0.0)
        bubble_adj = data.get('bubble_risk_adjustment', 0.0)
        ext_adj = data.get('extended_data_adjustment', 0.0)

        # Extract any additional auxiliary metrics
        auxiliary_metrics = []

        # Add volatility if available
        if 'volatility_score' in data:
            auxiliary_metrics.append(AuxiliaryRiskMetric(
                name='volatility_score',
                value=data['volatility_score'],
                unit='points',
                source='RegimeDetector',
                description='시장 변동성 지표 (auxiliary)'
            ))

        # Add liquidity score if available
        market_quality = data.get('market_quality', {})
        if isinstance(market_quality, dict) and 'avg_liquidity_score' in market_quality:
            auxiliary_metrics.append(AuxiliaryRiskMetric(
                name='liquidity_score',
                value=market_quality['avg_liquidity_score'],
                unit='points',
                source='DailyMicrostructureAnalyzer',
                description='시장 유동성 지표 (auxiliary)'
            ))

        # Add credit spread if available
        if 'credit_spread' in data:
            auxiliary_metrics.append(AuxiliaryRiskMetric(
                name='credit_spread',
                value=data['credit_spread'],
                unit='bps',
                source='CreditAnalyzer',
                description='신용 스프레드 (auxiliary)'
            ))

        return ScoreDefinitions.from_components(
            base_risk_score=base,
            microstructure_adjustment=micro_adj,
            bubble_risk_adjustment=bubble_adj,
            extended_data_adjustment=ext_adj,
            auxiliary_metrics=auxiliary_metrics,
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
        """
        리밸런싱 계획 생성 (Enhanced Execution Plan)

        When rebalancing is NOT executed, explicitly states why.
        """
        target = constraint_result.repaired_weights if constraint_result.repair_successful else {}

        if not current_weights:
            current_weights = data.get('portfolio_weights', {})

        # Case 1: No weights available
        if not target or not current_weights:
            return RebalancePlan(
                should_execute=False,
                not_executed_reason="No portfolio weights available (current or target)",
                trigger_type=TriggerType.MANUAL.value,
                trigger_reason="No weights available",
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

        # Case 2: Decision is HOLD due to issues - block rebalancing
        hold_reason_codes = [
            ReasonCode.DATA_QUALITY_ISSUE.value,
            ReasonCode.CONSTRAINT_VIOLATION_UNREPAIRED.value,
            ReasonCode.LOW_CONFIDENCE.value,
            ReasonCode.CLIENT_PROFILE_MISSING.value,
            ReasonCode.AGENT_CONFLICT.value,
            ReasonCode.REGIME_STANCE_MISMATCH.value,
        ]

        if decision.final_stance == "HOLD":
            blocking_codes = [c for c in decision.reason_codes if c in hold_reason_codes]
            if blocking_codes:
                # Map codes to human-readable reasons
                code_explanations = {
                    ReasonCode.DATA_QUALITY_ISSUE.value: "Data quality degraded",
                    ReasonCode.CONSTRAINT_VIOLATION_UNREPAIRED.value: "Constraints could not be repaired",
                    ReasonCode.LOW_CONFIDENCE.value: "Confidence below threshold",
                    ReasonCode.CLIENT_PROFILE_MISSING.value: "Client profile missing",
                    ReasonCode.AGENT_CONFLICT.value: "Agent conflict unresolved",
                    ReasonCode.REGIME_STANCE_MISMATCH.value: "Regime contradicts consensus",
                }
                explanations = [code_explanations.get(c, c) for c in blocking_codes]

                return RebalancePlan(
                    should_execute=False,
                    not_executed_reason=f"Decision policy HOLD: {'; '.join(explanations)}",
                    trigger_type=trigger_type.value,
                    trigger_reason=f"Blocked by decision policy (stance=HOLD, codes={blocking_codes})",
                )

            # HOLD for other reasons (mixed signals, default) - still block
            return RebalancePlan(
                should_execute=False,
                not_executed_reason=f"Decision policy HOLD: {', '.join(decision.reason_codes)}",
                trigger_type=trigger_type.value,
                trigger_reason=f"Blocked by HOLD stance (reasons: {decision.reason_codes})",
            )

        # Case 3: Normal rebalancing - generate full execution plan
        return generate_rebalance_plan(
            current_weights=current_weights,
            target_weights=target,
            trigger_type=trigger_type,
            trigger_reason=reason,
            config=self.config,
        )

    def _build_signal_hierarchy(
        self,
        data: Dict,
        scores: ScoreDefinitions,
        inputs: DecisionInputs
    ) -> SignalHierarchyReport:
        """
        Build Signal Hierarchy Report classifying signals as CORE or AUXILIARY.

        Core signals are directly used in decision rules.
        Auxiliary signals are for context and transparency only.
        """
        core_signals = []
        auxiliary_signals = []

        # === CORE SIGNALS ===
        # These are used in resolve_decision() rules

        # 1. Regime signal
        core_signals.append(SignalClassification(
            name='regime_signal',
            signal_type=SignalType.CORE.value,
            value=inputs.regime,
            source='RegimeDetector',
            used_in_rules=['R1', 'R2', 'R5'],
            description='Current market regime (BULL/BEAR/NEUTRAL)'
        ))

        # 2. Regime confidence
        core_signals.append(SignalClassification(
            name='regime_confidence',
            signal_type=SignalType.CORE.value,
            value=inputs.regime_confidence,
            source='RegimeDetector',
            used_in_rules=['R1', 'R2'],
            description='Confidence in regime classification'
        ))

        # 3. Canonical risk score
        core_signals.append(SignalClassification(
            name='canonical_risk_score',
            signal_type=SignalType.CORE.value,
            value=scores.canonical_risk_score,
            source='ScoreDefinitions (unified)',
            used_in_rules=['R3', 'R6'],
            description='Unified risk score (0-100)'
        ))

        # 4. Agent consensus stance
        core_signals.append(SignalClassification(
            name='agent_consensus_stance',
            signal_type=SignalType.CORE.value,
            value=inputs.agent_consensus,
            source='MetaOrchestrator',
            used_in_rules=['R4', 'R5', 'R7'],
            description='Final consensus from agent debate'
        ))

        # 5. Agent consensus confidence
        core_signals.append(SignalClassification(
            name='agent_consensus_confidence',
            signal_type=SignalType.CORE.value,
            value=inputs.confidence,
            source='MetaOrchestrator',
            used_in_rules=['R3', 'R6'],
            description='Confidence in agent consensus'
        ))

        # 6. Constraint status
        core_signals.append(SignalClassification(
            name='constraint_status',
            signal_type=SignalType.CORE.value,
            value=inputs.constraint_status,
            source='ConstraintRepair',
            used_in_rules=['R7'],
            description='Status of asset class constraints'
        ))

        # 7. Client profile status (assumed from data)
        client_status = data.get('client_profile_status', 'COMPLETE')
        core_signals.append(SignalClassification(
            name='client_profile_status',
            signal_type=SignalType.CORE.value,
            value=client_status,
            source='ClientProfile',
            used_in_rules=['R7'],
            description='Client profile availability'
        ))

        # === AUXILIARY SIGNALS ===
        # These are for context only, never used in decision rules

        # 1. Base risk score (component of canonical)
        auxiliary_signals.append(SignalClassification(
            name='aux_base_risk_score',
            signal_type=SignalType.AUXILIARY.value,
            value=scores.base_risk_score,
            source='CriticalPathAggregator',
            description='Base risk from critical path analysis (auxiliary)'
        ))

        # 2. Microstructure adjustment
        auxiliary_signals.append(SignalClassification(
            name='aux_microstructure_adjustment',
            signal_type=SignalType.AUXILIARY.value,
            value=scores.microstructure_adjustment,
            source='MicrostructureAnalyzer',
            description='Market quality adjustment (auxiliary)'
        ))

        # 3. Bubble risk adjustment
        auxiliary_signals.append(SignalClassification(
            name='aux_bubble_risk_adjustment',
            signal_type=SignalType.AUXILIARY.value,
            value=scores.bubble_risk_adjustment,
            source='BubbleDetector',
            description='Bubble risk overlay (auxiliary)'
        ))

        # 4. Full mode position (component of consensus)
        auxiliary_signals.append(SignalClassification(
            name='full_mode_position',
            signal_type=SignalType.AUXILIARY.value,
            value=inputs.full_mode_position,
            source='MetaOrchestrator (full mode)',
            description='Position from full analysis mode (auxiliary)'
        ))

        # 5. Reference mode position (component of consensus)
        auxiliary_signals.append(SignalClassification(
            name='reference_mode_position',
            signal_type=SignalType.AUXILIARY.value,
            value=inputs.reference_mode_position,
            source='MetaOrchestrator (reference mode)',
            description='Position from reference mode (auxiliary)'
        ))

        # 6. Modes agree
        auxiliary_signals.append(SignalClassification(
            name='modes_agree',
            signal_type=SignalType.AUXILIARY.value,
            value=inputs.modes_agree,
            source='DualModeAnalyzer',
            description='Whether full and reference modes agree (auxiliary)'
        ))

        # Add auxiliary metrics from ScoreDefinitions
        for metric in scores.auxiliary_metrics:
            auxiliary_signals.append(SignalClassification(
                name=metric.name,
                signal_type=SignalType.AUXILIARY.value,
                value=metric.value,
                source=metric.source,
                description=f'{metric.description} (auxiliary)'
            ))

        # Validation: ensure only core signals are in CORE_SIGNALS set
        validation_passed = True
        validation_errors = []
        for sig in core_signals:
            if sig.name not in CORE_SIGNALS:
                validation_passed = False
                validation_errors.append(f"Signal '{sig.name}' classified as CORE but not in CORE_SIGNALS set")

        return SignalHierarchyReport(
            core_signals=core_signals,
            auxiliary_signals=auxiliary_signals,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
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
