#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 5: Result Storage

Purpose:
    Result storage to JSON and database

Input:
    - result: EIMASResult
    - events: List[Event]
    - output_dir: Path

Output:
    - output_file: str

Functions:
    - save_results: Result storage orchestrator

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

from pathlib import Path
from typing import List

from pipeline.schemas import EIMASResult, Event
from pipeline.storage import save_result_json, save_result_md, save_to_event_db


def save_results(result: EIMASResult, events: List[Event], output_dir: Path) -> str:
    """[Phase 5] Save events + unified JSON/MD artifacts."""
    print("\n[Phase 5] Saving Results...")
    save_to_event_db(events)
    output_file = save_result_json(result, output_dir=output_dir)
    save_result_md(result, output_dir=output_dir)
    return output_file
