#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 3: AI Debate

Purpose:
    AI agent debate orchestration

Input:
    - result: EIMASResult
    - market_data: Dict[str, Any]

Output:
    - result: EIMASResult (full_mode_position, reference_mode_position, modes_agree)

Functions:
    - run_debate: AI debate orchestrator

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

from typing import Any, Dict

from pipeline.debate import run_dual_mode_debate
from pipeline.risk_utils import derive_risk_level
from pipeline.schemas import EIMASResult


async def run_debate(result: EIMASResult, market_data: Dict[str, Any]):
    """[Phase 3] Execute dual-mode debate and write consensus fields."""
    print("\n[Phase 3] Running AI Debate...")
    try:
        debate_res = await run_dual_mode_debate(market_data, extended_data=result.extended_data)
        result.full_mode_position = debate_res.full_mode_position
        result.reference_mode_position = debate_res.reference_mode_position
        result.modes_agree = debate_res.modes_agree
        result.final_recommendation = debate_res.final_recommendation
        result.confidence = debate_res.confidence
        result.debate_consensus["dual_mode_risk_level"] = debate_res.risk_level
        result.debate_consensus["canonical_risk_level"] = derive_risk_level(result.risk_score)
        result.risk_level = result.debate_consensus["canonical_risk_level"]
        result.warnings.extend(debate_res.warnings)
        result.reasoning_chain = debate_res.reasoning_chain

        if debate_res.enhanced_debate:
            result.debate_consensus["enhanced"] = debate_res.enhanced_debate
        if debate_res.verification:
            result.debate_consensus["verification"] = debate_res.verification
        if debate_res.metadata:
            result.debate_consensus["metadata"] = debate_res.metadata
        if debate_res.institutional_analysis:
            result.institutional_analysis = debate_res.institutional_analysis
    except Exception as e:
        print(f"⚠️ Debate Error: {e}")
