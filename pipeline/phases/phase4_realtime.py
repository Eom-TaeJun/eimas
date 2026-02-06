#!/usr/bin/env python3
"""
EIMAS Pipeline - Phase 4: Realtime Streaming

Purpose:
    Realtime streaming of analysis results (optional)

Input:
    - result: EIMASResult
    - enable: bool
    - duration: int

Output:
    - None (side effect: streaming output)

Functions:
    - run_realtime: Realtime streaming orchestrator

Architecture:
    - ADR: docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md
    - Stage: M2 (Logic migrated from main.py)
"""

from pipeline.realtime import run_realtime_stream
from pipeline.schemas import EIMASResult
from pipeline.storage import save_to_trading_db


async def run_realtime(result: EIMASResult, enable: bool, duration: int):
    """[Phase 4] Optionally run realtime stream and persist signals."""
    if enable:
        print("\n[Phase 4] Realtime Streaming...")
        signals = await run_realtime_stream(duration=duration)
        result.realtime_signals = [s.to_dict() for s in signals]
        save_to_trading_db(signals)
