#!/usr/bin/env python3
"""
lib/bubble_detector.py — Shim (패키지 호환 레이어)
=====================================================
실제 구현은 lib/bubble/ 패키지로 이동됨.
이 파일은 기존 import 경로 호환성 유지용입니다.

    from lib.bubble_detector import BubbleDetector  # 기존 경로 (유지)
    from lib.bubble import BubbleDetector            # 신규 경로 (권장)
"""
from lib.bubble import (
    BubbleDetector,
    BubbleDetectionResult,
    BubbleWarningLevel,
    RiskSignalType,
    quick_bubble_check,
    scan_for_bubbles,
)

__all__ = [
    "BubbleDetector",
    "BubbleDetectionResult",
    "BubbleWarningLevel",
    "RiskSignalType",
    "quick_bubble_check",
    "scan_for_bubbles",
]
