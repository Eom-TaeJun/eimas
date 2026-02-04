#!/usr/bin/env python3
"""
Event Framework Package
============================================================

Event detection, calendar management, and impact analysis

Public API:
    - EventFramework: Main event framework
    - QuantitativeEventDetector: Statistical event detector
    - EventImpactAnalyzer: Impact analyzer
    - CalendarEventManager: Calendar manager
    - Event, EventImpact: Data classes

Economic Foundation:
    - Event study methodology (Fama et al. 1969)
    - Market anomalies detection
    - Impact assessment with statistical testing

Usage:
    from lib.event_framework import EventFramework

    framework = EventFramework()
    events = framework.detect_events(market_data)
"""

from .framework import EventFramework
from .detector import QuantitativeEventDetector
from .impact import EventImpactAnalyzer
from .calendar import CalendarEventManager, EarningsCalendar
from .schemas import Event, EventImpact
from .enums import EventType, AssetClass, EventImportance, EventTiming

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
