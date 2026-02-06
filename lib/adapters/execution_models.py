"""Execution model adapter.

This module decouples EIMAS from direct imports of execution-related model
modules so they can be externalized to `execution_intelligence` gradually.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ModelExports = Dict[str, Any]


def _local_exports() -> ModelExports:
    from lib.allocation_engine import (
        AllocationConstraints,
        AllocationEngine,
        AllocationResult,
        AllocationStrategy,
    )
    from lib.rebalancing_policy import (
        AssetClassBounds,
        RebalanceConfig,
        RebalanceDecision,
        RebalanceFrequency,
        RebalancePolicy,
        RebalancingPolicy,
        TradingCostModel,
    )

    return {
        "AllocationConstraints": AllocationConstraints,
        "AllocationEngine": AllocationEngine,
        "AllocationResult": AllocationResult,
        "AllocationStrategy": AllocationStrategy,
        "AssetClassBounds": AssetClassBounds,
        "RebalanceConfig": RebalanceConfig,
        "RebalanceDecision": RebalanceDecision,
        "RebalanceFrequency": RebalanceFrequency,
        "RebalancePolicy": RebalancePolicy,
        "RebalancingPolicy": RebalancingPolicy,
        "TradingCostModel": TradingCostModel,
    }


def _external_exports() -> Optional[ModelExports]:
    external_root = Path(__file__).resolve().parents[3] / "execution_intelligence"
    if not external_root.exists():
        return None

    external_path = str(external_root)
    if external_path not in sys.path:
        sys.path.insert(0, external_path)

    try:
        from execution_intelligence.models import (
            AllocationConstraints,
            AllocationEngine,
            AllocationResult,
            AllocationStrategy,
            AssetClassBounds,
            RebalanceConfig,
            RebalanceDecision,
            RebalanceFrequency,
            RebalancePolicy,
            RebalancingPolicy,
            TradingCostModel,
        )
    except Exception:
        return None

    return {
        "AllocationConstraints": AllocationConstraints,
        "AllocationEngine": AllocationEngine,
        "AllocationResult": AllocationResult,
        "AllocationStrategy": AllocationStrategy,
        "AssetClassBounds": AssetClassBounds,
        "RebalanceConfig": RebalanceConfig,
        "RebalanceDecision": RebalanceDecision,
        "RebalanceFrequency": RebalanceFrequency,
        "RebalancePolicy": RebalancePolicy,
        "RebalancingPolicy": RebalancingPolicy,
        "TradingCostModel": TradingCostModel,
    }


def _resolve_exports() -> ModelExports:
    backend = os.getenv("EIMAS_EXECUTION_BACKEND", "local").strip().lower()
    if backend == "external":
        external = _external_exports()
        if external is not None:
            return external

    return _local_exports()


_EXPORTS = _resolve_exports()

AllocationConstraints = _EXPORTS["AllocationConstraints"]
AllocationEngine = _EXPORTS["AllocationEngine"]
AllocationResult = _EXPORTS["AllocationResult"]
AllocationStrategy = _EXPORTS["AllocationStrategy"]
AssetClassBounds = _EXPORTS["AssetClassBounds"]
RebalanceConfig = _EXPORTS["RebalanceConfig"]
RebalanceDecision = _EXPORTS["RebalanceDecision"]
RebalanceFrequency = _EXPORTS["RebalanceFrequency"]
RebalancePolicy = _EXPORTS["RebalancePolicy"]
RebalancingPolicy = _EXPORTS["RebalancingPolicy"]
TradingCostModel = _EXPORTS["TradingCostModel"]

__all__ = [
    "AllocationConstraints",
    "AllocationEngine",
    "AllocationResult",
    "AllocationStrategy",
    "AssetClassBounds",
    "RebalanceConfig",
    "RebalanceDecision",
    "RebalanceFrequency",
    "RebalancePolicy",
    "RebalancingPolicy",
    "TradingCostModel",
]
