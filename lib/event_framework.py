#!/usr/bin/env python3
"""
lib/event_framework.py — Shim (패키지 호환 레이어)
====================================================
실제 구현은 lib/event_framework/ 패키지로 이동됨.
이 파일은 기존 import 경로 호환성 유지용입니다.

    from lib.event_framework import QuantitativeEventDetector  # 기존 경로 (유지)
    from lib.event_framework import EventFramework             # 신규 경로 (권장)
"""
from lib.event_framework import (
    EventFramework,
    QuantitativeEventDetector,
    EventImpactAnalyzer,
    CalendarEventManager,
    EarningsCalendar,
    Event,
    EventImpact,
    EventType,
    AssetClass,
    EventImportance,
    EventTiming,
)

__all__ = [
    "EventFramework",
    "QuantitativeEventDetector",
    "EventImpactAnalyzer",
    "CalendarEventManager",
    "EarningsCalendar",
    "Event",
    "EventImpact",
    "EventType",
    "AssetClass",
    "EventImportance",
    "EventTiming",
]
