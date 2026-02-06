"""Adapter layer for optional externalized domains."""

from .execution_backend import generate_operational_bundle
from .execution_models import (
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

__all__ = [
    "generate_operational_bundle",
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
