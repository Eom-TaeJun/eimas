#!/usr/bin/env python3
"""
EIMAS Pipeline - Analyzers Facade Module
=========================================

Purpose:
    Public API facade for Phase 2 market analysis functions.

    This module re-exports all analyzer functions from specialized shard modules:
    - analyzers_core: Core market analysis (regime, events, liquidity, critical path)
    - analyzers_advanced: Advanced analysis (genius act, theme ETF, shock propagation, MST)
    - analyzers_quant: Quantitative analysis (HFT, GARCH, information flow, DTW, ARK)
    - analyzers_sentiment: Sentiment & bubble analysis
    - analyzers_governance: AI validation, allocation, rebalancing

Architecture:
    - ADR: docs/architecture/ADV_002_ANALYZERS_SPLIT_OWNERSHIP_V1.md
    - Work Order: work_orders/GEN-201.md
    - Migration Date: 2026-02-06

Compatibility:
    All existing import paths remain valid:
    - from pipeline.analyzers import detect_regime
    - from pipeline import detect_regime (via __init__.py)

Example:
    >>> from pipeline.analyzers import detect_regime, run_allocation_engine
    >>> regime = detect_regime('SPY')
    >>> allocation = run_allocation_engine(market_data)
"""

# ============================================================================
# Core Market Analysis (6 functions)
# ============================================================================
from pipeline.analyzers_core import (
    detect_regime,
    detect_events,
    analyze_liquidity,
    analyze_critical_path,
    analyze_etf_flow,
    generate_explanation,
)

# ============================================================================
# Advanced Thematic/Graph Analysis (7 functions)
# ============================================================================
from pipeline.analyzers_advanced import (
    analyze_genius_act,
    analyze_theme_etf,
    analyze_shock_propagation,
    optimize_portfolio_mst,
    analyze_volume_anomalies,
    track_events_with_news,  # async
    run_adaptive_portfolio,
)

# ============================================================================
# Quant & Microstructure Analysis (8 functions)
# ============================================================================
from pipeline.analyzers_quant import (
    analyze_hft_microstructure,
    analyze_volatility_garch,
    analyze_information_flow,
    calculate_proof_of_index,
    enhance_portfolio_with_systemic_similarity,
    detect_outliers_with_dbscan,
    analyze_dtw_similarity,
    analyze_ark_trades,
)

# ============================================================================
# Sentiment & Bubble Analysis (2 functions)
# ============================================================================
from pipeline.analyzers_sentiment import (
    analyze_bubble_risk,
    analyze_sentiment,
)

# ============================================================================
# Governance & Execution (3 functions)
# ============================================================================
from pipeline.analyzers_governance import (
    run_ai_validation,
    run_allocation_engine,
    run_rebalancing_policy,
)

# ============================================================================
# Public API (26 functions)
# ============================================================================
__all__ = [
    # Core (6)
    "detect_regime",
    "detect_events",
    "analyze_liquidity",
    "analyze_critical_path",
    "analyze_etf_flow",
    "generate_explanation",
    # Advanced (7)
    "analyze_genius_act",
    "analyze_theme_etf",
    "analyze_shock_propagation",
    "optimize_portfolio_mst",
    "analyze_volume_anomalies",
    "track_events_with_news",
    "run_adaptive_portfolio",
    # Quant (8)
    "analyze_hft_microstructure",
    "analyze_volatility_garch",
    "analyze_information_flow",
    "calculate_proof_of_index",
    "enhance_portfolio_with_systemic_similarity",
    "detect_outliers_with_dbscan",
    "analyze_dtw_similarity",
    "analyze_ark_trades",
    # Sentiment (2)
    "analyze_bubble_risk",
    "analyze_sentiment",
    # Governance (3)
    "run_ai_validation",
    "run_allocation_engine",
    "run_rebalancing_policy",
]

# ============================================================================
# Module Metadata
# ============================================================================
__version__ = "2.0.0"
__author__ = "EIMAS Team"
__description__ = "Facade module for EIMAS market analysis functions"
__migration_date__ = "2026-02-06"
__work_order__ = "GEN-201"
