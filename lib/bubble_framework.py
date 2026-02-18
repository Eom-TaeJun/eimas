#!/usr/bin/env python3
"""
lib/bubble_framework.py — Shim (패키지 호환 레이어)
=====================================================
실제 구현은 lib/bubble/ 패키지로 이동됨.
이 파일은 기존 import 경로 호환성 유지용입니다.

    from lib.bubble_framework import FiveStageBubbleFramework  # 기존 경로 (유지)
    from lib.bubble import FiveStageBubbleFramework            # 신규 경로 (권장)
"""
from lib.bubble import (
    FiveStageBubbleFramework,
    JPMorganFrameworkResult,
    StageResult,
    BubbleFrameworkResult,
    JPMorganBubbleStage,
)

__all__ = [
    "FiveStageBubbleFramework",
    "JPMorganFrameworkResult",
    "StageResult",
    "BubbleFrameworkResult",
    "JPMorganBubbleStage",
]
