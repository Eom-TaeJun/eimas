"""Execution backend adapter.

This module decouples EIMAS full pipeline from a hard dependency on
`lib.operational_engine` so execution logic can be externalized to a
separate project (`execution_intelligence`) in phases.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

Bundle = Dict[str, Any]


def _local_bundle(
    result_data: Dict[str, Any],
    current_weights: Optional[Dict[str, float]] = None,
    rebalance_decision: Optional[Dict[str, Any]] = None,
) -> Bundle:
    """Use in-repo monolithic execution backend (current default)."""
    from lib.operational_engine import (
        OperationalEngine,
        get_indicator_classification,
        get_input_validation,
        get_operational_controls,
        get_audit_metadata,
        get_approval_status,
    )

    engine = OperationalEngine()
    op_report = engine.process(result_data, current_weights or {})

    constraint_payload = (
        op_report.constraint_repair.to_dict() if hasattr(op_report, "constraint_repair") else {}
    )
    rebalance_payload = (
        op_report.rebalance_plan.to_dict() if hasattr(op_report, "rebalance_plan") else {}
    )

    return {
        "op_report": op_report,
        "operational_report": op_report.to_dict(),
        "input_validation": get_input_validation(result_data),
        "indicator_classification": get_indicator_classification(result_data),
        "operational_controls": get_operational_controls(
            result_data,
            rebalance_decision=rebalance_decision,
            constraint_result=constraint_payload,
        ),
        "audit_metadata": get_audit_metadata(result_data),
        "approval_status": get_approval_status(
            result_data,
            rebalance_plan=rebalance_payload,
        ),
    }


def _load_external_builder() -> Optional[Callable[..., Bundle]]:
    """Try loading external execution backend from sibling project."""
    external_root = Path(__file__).resolve().parents[3] / "execution_intelligence"
    if not external_root.exists():
        return None

    external_path = str(external_root)
    if external_path not in sys.path:
        sys.path.insert(0, external_path)

    try:
        from execution_intelligence.bridge import generate_operational_bundle as external_builder

        return external_builder
    except Exception:
        return None


def generate_operational_bundle(
    result_data: Dict[str, Any],
    current_weights: Optional[Dict[str, float]] = None,
    rebalance_decision: Optional[Dict[str, Any]] = None,
) -> Bundle:
    """Return operational artifacts from selected backend.

    Backend selection:
    - `EIMAS_EXECUTION_BACKEND=external`: try external project first.
    - otherwise: local backend.

    Any external failure falls back to local backend to preserve full-mode runtime.
    """
    backend = os.getenv("EIMAS_EXECUTION_BACKEND", "local").strip().lower()

    if backend == "external":
        external_builder = _load_external_builder()
        if external_builder is not None:
            try:
                return external_builder(
                    result_data=result_data,
                    current_weights=current_weights,
                    rebalance_decision=rebalance_decision,
                )
            except Exception:
                # Deterministic fallback for full-mode stability.
                pass

    return _local_bundle(
        result_data=result_data,
        current_weights=current_weights,
        rebalance_decision=rebalance_decision,
    )
