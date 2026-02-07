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
    from lib.stress_test import StressTestEngine, generate_stress_test_report
    from lib.tactical_allocation import (
        MomentumOverlay,
        TacticalAssetAllocator,
        VolatilityTargeting,
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
        "StressTestEngine": StressTestEngine,
        "TacticalAssetAllocator": TacticalAssetAllocator,
        "TradingCostModel": TradingCostModel,
        "VolatilityTargeting": VolatilityTargeting,
        "MomentumOverlay": MomentumOverlay,
        "generate_stress_test_report": generate_stress_test_report,
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
            StressTestEngine,
            TacticalAssetAllocator,
            TradingCostModel,
            generate_stress_test_report,
        )
        from execution_intelligence.models.tactical_allocation import (
            MomentumOverlay,
            VolatilityTargeting,
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
        "StressTestEngine": StressTestEngine,
        "TacticalAssetAllocator": TacticalAssetAllocator,
        "TradingCostModel": TradingCostModel,
        "VolatilityTargeting": VolatilityTargeting,
        "MomentumOverlay": MomentumOverlay,
        "generate_stress_test_report": generate_stress_test_report,
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
StressTestEngine = _EXPORTS["StressTestEngine"]
TacticalAssetAllocator = _EXPORTS["TacticalAssetAllocator"]
TradingCostModel = _EXPORTS["TradingCostModel"]
VolatilityTargeting = _EXPORTS["VolatilityTargeting"]
MomentumOverlay = _EXPORTS["MomentumOverlay"]
generate_stress_test_report = _EXPORTS["generate_stress_test_report"]

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
    "StressTestEngine",
    "TacticalAssetAllocator",
    "TradingCostModel",
    "VolatilityTargeting",
    "MomentumOverlay",
    "generate_stress_test_report",
]
