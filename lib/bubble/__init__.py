#!/usr/bin/env python3
"""
Bubble Detection Package
============================================================

Bubble detection and risk assessment framework

Public API:
    - BubbleDetector: Main bubble detector
    - FiveStageBubbleFramework: JP Morgan 5-stage framework
    - BubbleDetectionResult, BubbleWarningLevel: Result types
    - quick_bubble_check, scan_for_bubbles: Utility functions

Economic Foundation:
    - "Bubbles for Fama" (Greenwood, Shleifer, You 2019)
    - JP Morgan 5-Stage Bubble Framework

Usage:
    from lib.bubble import BubbleDetector
    
    detector = BubbleDetector()
    result = detector.analyze(ticker='TSLA')
    print(result.bubble_warning_level)
"""

from .detector import BubbleDetector
from .framework import FiveStageBubbleFramework
from .utils import quick_bubble_check, scan_for_bubbles
from .schemas import (
    BubbleDetectionResult,
    RunUpResult,
    VolatilityResult,
    IssuanceResult,
    RiskSignal,
    JPMorganFrameworkResult,
    StageResult,
    BubbleFrameworkResult,
)
from .enums import (
    BubbleWarningLevel,
    RiskSignalType,
    JPMorganBubbleStage,
)

__all__ = [
    "BubbleDetector",
    "FiveStageBubbleFramework",
    "quick_bubble_check",
    "scan_for_bubbles",
    "BubbleDetectionResult",
    "RunUpResult",
    "VolatilityResult",
    "IssuanceResult",
    "RiskSignal",
    "JPMorganFrameworkResult",
    "StageResult",
    "BubbleFrameworkResult",
    "BubbleWarningLevel",
    "RiskSignalType",
    "JPMorganBubbleStage",
]
