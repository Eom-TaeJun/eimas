#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 2: Basic Analysis

Purpose:
    Basic analysis including regime detection, events, liquidity, and critical path

Input:
    - market_data: Dict[str, Any]
    - result: EIMASResult

Output:
    - (events, regime_res)

Functions:
    - analyze_basic: Basic analysis orchestrator

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

from typing import Dict, Any, List, Tuple

from pipeline.analyzers import detect_regime, detect_events, analyze_critical_path
from pipeline.schemas import EIMASResult


def analyze_basic(result: EIMASResult, market_data: Dict[str, Any]) -> Tuple[List[Any], Any]:
    """[Phase 2.1] Regime/events/risk baseline analysis."""
    print("\n[Phase 2] Analyzing Market...")

    regime_res = detect_regime()
    result.regime = regime_res.to_dict()

    events = detect_events(result.fred_summary, market_data)
    result.events_detected = [e.to_dict() for e in events]

    try:
        cp_res = analyze_critical_path(market_data)
        result.risk_score = cp_res.risk_score
        result.base_risk_score = cp_res.risk_score
    except Exception as e:
        print(f"⚠️ Critical Path Error: {e}")

    return events, regime_res
