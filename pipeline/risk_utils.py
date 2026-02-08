#!/usr/bin/env python3
"""
Pipeline risk utility helpers.

`risk_level` in canonical outputs is derived from `risk_score` to keep
decision payloads and presentation artifacts consistent.
"""

from __future__ import annotations

from typing import Any

LOW_RISK_THRESHOLD = 30.0
HIGH_RISK_THRESHOLD = 70.0


def derive_risk_level(score: Any) -> str:
    """Return LOW/MEDIUM/HIGH from a numeric 0-100 risk score."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "MEDIUM"

    if value < LOW_RISK_THRESHOLD:
        return "LOW"
    if value >= HIGH_RISK_THRESHOLD:
        return "HIGH"
    return "MEDIUM"
