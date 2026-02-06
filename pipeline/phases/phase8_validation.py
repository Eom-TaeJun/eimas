#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 8: Validation

Purpose:
    Multi-LLM and Quick validation of results

Input:
    - result: EIMASResult
    - full_mode / market_focus flags
    - market_data
    - output_file / output_dir

Output:
    - result: EIMASResult (validation_results)

Functions:
    - run_ai_validation_phase: AI validation orchestrator
    - run_quick_validation: Quick validation runner

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from lib.quick_agents import QuickOrchestrator
from pipeline.analyzers import run_ai_validation
from pipeline.schemas import EIMASResult
from pipeline.storage import save_result_json


def run_ai_validation_phase(
    result: EIMASResult,
    full_mode: bool,
    output_dir: Path,
    output_file: str = "",
):
    """[Phase 8] Multi-LLM validation with HOLD failsafe on REJECT."""
    if not full_mode:
        return

    print("\n" + "=" * 50)
    print("PHASE 8: AI VALIDATION (Multi-LLM)")
    print("=" * 50)

    try:
        result.validation_loop_result = run_ai_validation(result.to_dict())

        if result.validation_loop_result.get("final_result") == "REJECT":
            result.failsafe_status = {
                "triggered": True,
                "reason": "AI Validation REJECT",
                "fallback_action": "HOLD",
                "original_recommendation": result.final_recommendation,
            }
            result.final_recommendation = "HOLD"
            result.warnings.append("⚠️ AI Validation REJECT - Forced to HOLD")
            print("      ⚠️ FAILSAFE TRIGGERED: AI Validation REJECT -> HOLD")

        save_result_json(result, output_dir=output_dir, output_file=output_file or None)
    except Exception as e:
        print(f"⚠️ AI Validation Error: {e}")


def run_quick_validation(
    result: EIMASResult,
    market_data: Dict[str, Any],
    output_file: str,
    market_focus: str = None,
):
    """[Phase 8.5] Quick-mode validation with market-specific summary."""
    if not market_focus:
        return

    print("\n" + "=" * 70)
    print(f"PHASE 8.5: QUICK MODE AI VALIDATION ({market_focus} Focus)")
    print("=" * 70)

    try:
        orchestrator = QuickOrchestrator()
        quick_result = orchestrator.run_quick_validation(
            full_json_path=output_file,
            market_data=market_data,
        )

        result.quick_validation = quick_result

        final_val = quick_result.get("final_validation", {})
        final_rec = final_val.get("final_recommendation", "N/A")
        confidence = final_val.get("confidence", 0)
        validation_result = final_val.get("validation_result", "N/A")

        print(f"\n[Quick Validation Summary] {market_focus}")
        print(f"   - Validation Result: {validation_result}")
        print(f"   - Final Recommendation: {final_rec}")
        print(f"   - Confidence: {confidence*100:.0f}%")

        agent_consensus = final_val.get("agent_consensus", {})
        agreement_level = agent_consensus.get("agreement_level", "N/A")
        print(f"   - Agent Agreement: {agreement_level}")

        comparison = final_val.get("full_vs_quick_comparison", {})
        alignment = comparison.get("alignment", "N/A")
        print(f"   - Full vs Quick: {alignment}")

        market_sentiment = quick_result.get("market_sentiment", {})
        if market_focus == "KOSPI":
            primary = market_sentiment.get("kospi_sentiment", {})
            secondary = market_sentiment.get("spx_sentiment", {})
            print(
                f"   - KOSPI Sentiment: {primary.get('sentiment', 'N/A')} ({primary.get('confidence', 0)*100:.0f}%)"
            )
            print(
                f"   - SPX (ref): {secondary.get('sentiment', 'N/A')} ({secondary.get('confidence', 0)*100:.0f}%)"
            )
        elif market_focus == "SPX":
            primary = market_sentiment.get("spx_sentiment", {})
            secondary = market_sentiment.get("kospi_sentiment", {})
            print(
                f"   - SPX Sentiment: {primary.get('sentiment', 'N/A')} ({primary.get('confidence', 0)*100:.0f}%)"
            )
            print(
                f"   - KOSPI (ref): {secondary.get('sentiment', 'N/A')} ({secondary.get('confidence', 0)*100:.0f}%)"
            )

        alt_assets = quick_result.get("alternative_assets", {})
        crypto_rec = alt_assets.get("crypto_assessment", {}).get("recommendation", "N/A")
        commodity_rec = alt_assets.get("commodity_assessment", {}).get("recommendation", "N/A")
        print(f"   - Crypto: {crypto_rec}, Commodity: {commodity_rec}")

        success_rate = quick_result.get("success_rate", 1.0)
        print(f"   - Agent Success: {success_rate*100:.0f}% ({int(success_rate*4)}/4 agents)")

        risk_warnings = final_val.get("risk_warnings", [])
        if risk_warnings:
            other_market = "KOSPI" if market_focus == "SPX" else "SPX"
            filtered_warnings = [
                w for w in risk_warnings if other_market.lower() not in w.lower() or "divergence" in w.lower()
            ]
            if filtered_warnings:
                print(f"\n⚠️ Risk Warnings ({len(filtered_warnings)}):")
                for i, warning in enumerate(filtered_warnings[:3], 1):
                    print(f"   {i}. {warning}")

        quick_output = Path(output_file).parent / (
            f"quick_validation_{market_focus.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(quick_output, "w", encoding="utf-8") as f:
            json.dump(quick_result, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Quick validation saved: {quick_output}")

    except Exception as e:
        print(f"⚠️ Quick Validation Error: {e}")
        import traceback

        traceback.print_exc()
