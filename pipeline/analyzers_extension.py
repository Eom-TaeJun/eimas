#!/usr/bin/env python3
"""
EIMAS Pipeline - Analyzers Extension (DEPRECATED)
==================================================

⚠️ DEPRECATION NOTICE ⚠️

This module is DEPRECATED and maintained only for backward compatibility.

Canonical implementations have been moved to:
    - pipeline.analyzers

Please update your imports to use the canonical module:
    OLD: from pipeline.analyzers_extension import analyze_volume_anomalies
    NEW: from pipeline.analyzers import analyze_volume_anomalies

This shim will be removed in a future release.

Policy:
    - ADR: docs/architecture/ADV_004_ANALYZERS_EXTENSION_POLICY_V1.md
    - Work Order: work_orders/GEN-202.md
    - Decommission Date: 2026-02-06

Rationale:
    - Drift detected between extension and canonical implementations
    - Import errors found (AdaptivePortfolioManager path mismatch)
    - Single source of truth: pipeline/analyzers.py facade
"""

import warnings

# Issue deprecation warning on module import
warnings.warn(
    "pipeline.analyzers_extension is deprecated. "
    "Use 'from pipeline.analyzers import ...' instead. "
    "This module will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2
)

# ============================================================================
# Re-export canonical implementations from analyzers facade
# ============================================================================
from pipeline.analyzers import (
    analyze_volume_anomalies,
    track_events_with_news,  # async
    run_adaptive_portfolio,
)

# ============================================================================
# Public API (3 functions - deprecated)
# ============================================================================
__all__ = [
    "analyze_volume_anomalies",
    "track_events_with_news",
    "run_adaptive_portfolio",
]

# ============================================================================
# Module Metadata
# ============================================================================
__version__ = "0.0.0-deprecated"
__status__ = "DEPRECATED"
__canonical__ = "pipeline.analyzers"
__decommission_date__ = "2026-02-06"
__work_order__ = "GEN-202"
