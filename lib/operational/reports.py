#!/usr/bin/env python3
"""
Operational - Reports
============================================================

감사 메타데이터 및 운영 리포트
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging

# Import from same package
from .config import OperationalConfig
from .enums import FinalStance, ReasonCode, TriggerType, SignalType
from .signal import SignalHierarchyReport
from .hold_policy import HoldPolicyReport
from .risk_metrics import ScoreDefinitions
from .decision import DecisionPolicy
from .constraints import ConstraintRepairResult
from .rebalance import RebalancePlan

logger = logging.getLogger(__name__)


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
        md.append(f"### final_stance: **{dp.final_stance}**")
        md.append("")
        # Explicit constraint and client profile status (required by rubric)
        constraints_ok = dp.constraint_status_input in ("OK", "REPAIRED")
        client_profile = dp.client_profile_status_input
        md.append(f"- **constraints_ok**: {constraints_ok}")
        md.append(f"- **client_profile**: {client_profile}")
        md.append("")
        md.append("#### Inputs")
        md.append(f"- Regime: {dp.regime_input}")
        md.append(f"- Risk Score: {dp.risk_score_input:.1f}")
        md.append(f"- Confidence: {dp.confidence_input:.2f}")
        md.append(f"- Agent Consensus: {dp.agent_consensus_input}")
        md.append(f"- Modes Agree: {dp.modes_agree_input}")
        md.append(f"- Constraint Status: {dp.constraint_status_input}")
        md.append("")
        md.append("#### applied_rules")
        for rule in dp.applied_rules:
            md.append(f"- {rule}")
        md.append("")
        md.append("#### reason_codes")
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

