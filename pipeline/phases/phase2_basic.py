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

import os
import socket
from datetime import datetime
from typing import Dict, Any, List, Tuple

from pipeline.analyzers import detect_regime, detect_events, analyze_critical_path
from pipeline.schemas import EIMASResult, RegimeResult


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_hosts(value: str, fallback: list[str]) -> list[str]:
    hosts = [item.strip() for item in value.split(",") if item.strip()]
    return hosts or fallback


def _is_network_available(hosts: list[str]) -> bool:
    for host in hosts:
        try:
            socket.getaddrinfo(host, 443)
            return True
        except OSError:
            continue
    return False


def analyze_basic(result: EIMASResult, market_data: Dict[str, Any]) -> Tuple[List[Any], Any]:
    """[Phase 2.1] Regime/events/risk baseline analysis."""
    print("\n[Phase 2] Analyzing Market...")

    skip_regime = _env_flag("EIMAS_SKIP_REGIME_DETECTION", default=False)
    regime_fail_fast = _env_flag("EIMAS_REGIME_FAIL_FAST_NETWORK", default=False)
    regime_reason = ""
    if skip_regime:
        regime_reason = "EIMAS_SKIP_REGIME_DETECTION"
    elif regime_fail_fast:
        hosts = _resolve_hosts(
            os.getenv(
                "EIMAS_REGIME_NETWORK_PROBE_HOSTS",
                "guce.yahoo.com,query1.finance.yahoo.com",
            ),
            ["guce.yahoo.com", "query1.finance.yahoo.com"],
        )
        if not _is_network_available(hosts):
            regime_reason = f"dns_unavailable:{','.join(hosts)}"

    if regime_reason:
        print(f"      i Regime detection skip ({regime_reason})")
        regime_res = RegimeResult(
            timestamp=datetime.now().isoformat(),
            regime="Transition",
            trend="Neutral",
            volatility="Unknown",
            confidence=0.5,
            description=f"Regime detection skipped: {regime_reason}",
            strategy="Conservative positioning until data recovers",
        )
    else:
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
