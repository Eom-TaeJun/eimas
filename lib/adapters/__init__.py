"""Adapter layer for optional externalized domains."""

from .execution_backend import generate_operational_bundle
from .execution_models import (
    AllocationConstraints,
    AllocationEngine,
    AllocationResult,
    AllocationStrategy,
    AssetClassBounds,
    MomentumOverlay,
    RebalanceConfig,
    RebalanceDecision,
    RebalanceFrequency,
    RebalancePolicy,
    RebalancingPolicy,
    StressTestEngine,
    TacticalAssetAllocator,
    TradingCostModel,
    VolatilityTargeting,
    generate_stress_test_report,
)

__all__ = [
    "generate_operational_bundle",
    "AllocationConstraints",
    "AllocationEngine",
    "AllocationResult",
    "AllocationStrategy",
    "AssetClassBounds",
    "MomentumOverlay",
    "RebalanceConfig",
    "RebalanceDecision",
    "RebalanceFrequency",
    "RebalancePolicy",
    "RebalancingPolicy",
    "StressTestEngine",
    "TacticalAssetAllocator",
    "TradingCostModel",
    "VolatilityTargeting",
    "generate_stress_test_report",
]
