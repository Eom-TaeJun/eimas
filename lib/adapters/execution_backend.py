"""Execution backend adapter.

This module decouples EIMAS full pipeline from a hard dependency on
`lib.operational_engine` so execution logic can be externalized to a
separate project (`execution_intelligence`) in phases.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

Bundle = Dict[str, Any]
logger = logging.getLogger(__name__)


def _local_package_bundle(
    result_data: Dict[str, Any],
    current_weights: Optional[Dict[str, float]] = None,
    rebalance_decision: Optional[Dict[str, Any]] = None,
) -> Bundle:
    """Use modular in-repo operational package backend."""
    from lib.operational import OperationalEngine
    from lib.operational.utils import (
        get_approval_status,
        get_audit_metadata,
        get_indicator_classification,
        get_input_validation,
        get_operational_controls,
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
        "backend_source": "local_package",
    }


def _local_monolith_bundle(
    result_data: Dict[str, Any],
    current_weights: Optional[Dict[str, float]] = None,
    rebalance_decision: Optional[Dict[str, Any]] = None,
) -> Bundle:
    """Use in-repo monolithic execution backend as compatibility fallback."""
    from lib.operational_engine import (
        OperationalEngine,
        get_approval_status,
        get_audit_metadata,
        get_indicator_classification,
        get_input_validation,
        get_operational_controls,
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
        "backend_source": "local_monolith",
    }


def _local_bundle(
    result_data: Dict[str, Any],
    current_weights: Optional[Dict[str, float]] = None,
    rebalance_decision: Optional[Dict[str, Any]] = None,
) -> Bundle:
    """Resolve local backend with package-first policy and monolith fallback."""
    local_backend = os.getenv("EIMAS_LOCAL_OPERATIONAL_BACKEND", "package").strip().lower()

    if local_backend == "monolith":
        return _local_monolith_bundle(
            result_data=result_data,
            current_weights=current_weights,
            rebalance_decision=rebalance_decision,
        )

    try:
        return _local_package_bundle(
            result_data=result_data,
            current_weights=current_weights,
            rebalance_decision=rebalance_decision,
        )
    except Exception as e:
        if local_backend == "package_strict":
            raise
        logger.warning(
            "Local operational package failed (%s); falling back to monolith backend.",
            e.__class__.__name__,
        )
        fallback = _local_monolith_bundle(
            result_data=result_data,
            current_weights=current_weights,
            rebalance_decision=rebalance_decision,
        )
        if isinstance(fallback, dict):
            fallback["backend_fallback_reason"] = f"local_package_error:{e.__class__.__name__}"
        return fallback


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
                bundle = external_builder(
                    result_data=result_data,
                    current_weights=current_weights,
                    rebalance_decision=rebalance_decision,
                )
                if isinstance(bundle, dict):
                    bundle.setdefault("backend_source", "external")
                return bundle
            except Exception as e:
                # Deterministic fallback for full-mode stability.
                logger.warning(
                    "External execution backend failed (%s); falling back to local backend.",
                    e.__class__.__name__,
                )
                local_fallback = _local_bundle(
                    result_data=result_data,
                    current_weights=current_weights,
                    rebalance_decision=rebalance_decision,
                )
                if isinstance(local_fallback, dict):
                    local_fallback.setdefault(
                        "backend_fallback_reason",
                        f"external_backend_error:{e.__class__.__name__}",
                    )
                return local_fallback

    return _local_bundle(
        result_data=result_data,
        current_weights=current_weights,
        rebalance_decision=rebalance_decision,
    )
