#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 7: AI Report Generation

Purpose:
    AI report generation and report validation workflow

Input:
    - result: EIMASResult
    - market_data: Dict[str, Any]
    - generate: bool
    - output_dir: Path
    - output_file: str (optional, canonical artifact path)

Output:
    - report_content: str

Functions:
    - generate_report: AI report generator
    - validate_report: Whitening + fact-check validator

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

from pathlib import Path
from typing import Dict, Any

from pipeline.schemas import EIMASResult
from pipeline.report import generate_ai_report, run_whitening_check, run_fact_check
from pipeline.storage import save_result_json


async def generate_report(
    result: EIMASResult,
    market_data: Dict[str, Any],
    generate: bool,
    output_dir: Path,
    output_file: str = "",
) -> str:
    """[Phase 7] Generate AI report and persist updated result."""
    if not generate:
        return ""

    print("\n[Phase 7] Generating Report...")
    try:
        ai_report = await generate_ai_report(result, market_data)
        result.ai_report = ai_report.to_dict() if hasattr(ai_report, "to_dict") else ai_report.__dict__
        save_result_json(result, output_dir=output_dir, output_file=output_file or None)
        return ai_report.content
    except Exception as e:
        print(f"⚠️ Report Generation Error: {e}")
        return ""


async def validate_report(
    result: EIMASResult,
    report_content: str,
    generate: bool,
    output_dir: Path,
    output_file: str = "",
):
    """[Phase 7.x] Validate generated report and persist updated result."""
    if not generate:
        return

    print("\n" + "=" * 50)
    print("PHASE 7: VALIDATION")
    print("=" * 50)

    try:
        result.whitening_summary = run_whitening_check(result)
        if report_content:
            result.fact_check_grade = await run_fact_check(report_content)
        save_result_json(result, output_dir=output_dir, output_file=output_file or None)
    except Exception as e:
        print(f"⚠️ Validation Error: {e}")
