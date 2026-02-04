#!/usr/bin/env python3
"""
Operational - Hold Policy
============================================================

HOLD 정책 평가 및 조건
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

