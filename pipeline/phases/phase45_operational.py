#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 4.5: Operational Reporting

Purpose:
    Operational report generation

Input:
    - result: EIMASResult
    - current_weights: Dict[str, float] | None

Output:
    - result: EIMASResult (operational_report)

Functions:
    - generate_operational_report: Operational report generator

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

from typing import Dict, Optional

from lib.adapters import generate_operational_bundle
from pipeline.schemas import EIMASResult


def generate_operational_report(result: EIMASResult, current_weights: Optional[Dict] = None):
    """[Phase 4.5] Build operational governance report and apply failsafe policy."""
    print("\n[Phase 4.5] Generating Operational Report...")

    try:
        eimas_data = result.to_dict()
        if current_weights is None:
            current_weights = result.portfolio_weights or {}

        op_bundle = generate_operational_bundle(
            result_data=eimas_data,
            current_weights=current_weights,
            rebalance_decision=result.rebalance_decision,
        )
        op_report = op_bundle["op_report"]

        result.operational_report = op_bundle.get("operational_report", {})
        result.input_validation = op_bundle.get("input_validation", {})
        result.indicator_classification = op_bundle.get("indicator_classification", {})

        rebal_plan = getattr(op_report, "rebalance_plan", None)
        if rebal_plan and rebal_plan.should_execute and rebal_plan.trades:
            result.trade_plan = [t.to_dict() for t in rebal_plan.trades if t.action != "HOLD"]
        elif result.rebalance_decision and result.rebalance_decision.get("trade_plan"):
            result.trade_plan = result.rebalance_decision["trade_plan"]

        result.operational_controls = op_bundle.get("operational_controls", {})
        result.audit_metadata = op_bundle.get("audit_metadata", {})
        result.approval_status = op_bundle.get("approval_status", {})

        constraint_repair = getattr(op_report, "constraint_repair", None)
        if constraint_repair is not None and not constraint_repair.constraints_satisfied:
            violations = constraint_repair.violations_found
            has_severe_violation = any(v.severity == "SEVERE" for v in violations)

            if constraint_repair.force_hold or has_severe_violation:
                result.failsafe_status = {
                    "triggered": True,
                    "reason": "CONSTRAINT_VIOLATION",
                    "fallback_action": "HOLD",
                    "original_recommendation": result.final_recommendation,
                    "violations": [v.to_dict() for v in violations],
                }
                result.final_recommendation = "HOLD"
                result.warnings.append("⚠️ Constraint Violation - Forced to HOLD")
                print("      ⚠️ FAILSAFE TRIGGERED: Constraint Violation -> HOLD")
            else:
                result.failsafe_status = {
                    "triggered": False,
                    "reason": "Constraints violated but partially repaired",
                    "fallback_action": None,
                    "original_recommendation": result.final_recommendation,
                }
                result.warnings.append(f"⚠️ {len(violations)} constraint violation(s) detected")
        else:
            if not result.failsafe_status:
                result.failsafe_status = {
                    "triggered": False,
                    "reason": None,
                    "fallback_action": None,
                    "original_recommendation": result.final_recommendation,
                }

        if not result.failsafe_status.get("triggered", False):
            decision_policy = getattr(op_report, "decision_policy", None)
            if decision_policy and decision_policy.final_stance:
                stance = decision_policy.final_stance
                if stance == "HOLD":
                    result.final_recommendation = "HOLD"
                elif stance == "BULLISH":
                    result.final_recommendation = "BULLISH"
                elif stance == "BEARISH":
                    result.final_recommendation = "BEARISH"
                else:
                    result.final_recommendation = "NEUTRAL"

        decision_text = getattr(getattr(op_report, "decision_policy", None), "final_stance", "N/A")
        constraints_ok = getattr(getattr(op_report, "constraint_repair", None), "constraints_satisfied", False)
        rebalance_exec = getattr(getattr(op_report, "rebalance_plan", None), "should_execute", False)
        print(f"      ✓ Decision: {decision_text}")
        print(f"      ✓ Constraints: {'SATISFIED' if constraints_ok else 'VIOLATED'}")
        print(f"      ✓ Rebalance: {'EXECUTE' if rebalance_exec else 'NOT EXECUTED'}")
        print(f"      ✓ Failsafe: {'TRIGGERED' if result.failsafe_status.get('triggered') else 'NOT TRIGGERED'}")

    except Exception as e:
        import traceback

        print(f"      ⚠️ Operational Report Error: {e}")
        traceback.print_exc()
        result.operational_report = {"error": str(e)}
        result.failsafe_status = {
            "triggered": False,
            "reason": f"Error in operational report: {e}",
            "fallback_action": None,
            "original_recommendation": result.final_recommendation,
        }
