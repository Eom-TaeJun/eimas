#!/usr/bin/env python3
"""
Event Framework - Main Framework
============================================================

Main event detection and management framework

Class:
    - EventFramework: Integrates detector, calendar, and impact analysis
"""

from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime
import logging

from .schemas import Event, EventImpact
from .detector import QuantitativeEventDetector
from .calendar import CalendarEventManager

logger = logging.getLogger(__name__)


class EventFramework:
    """통합 이벤트 프레임워크"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.calendar_manager = CalendarEventManager()
        self.quantitative_detector = QuantitativeEventDetector()
        self.earnings_calendar = EarningsCalendar()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[EventFramework] {msg}")

    def get_all_upcoming_events(
        self,
        days_ahead: int = 7,
        tickers: List[str] = None
    ) -> List[Event]:
        """모든 예정된 이벤트 조회"""
        events = []

        # 1. 경제 이벤트
        self._log("Fetching economic calendar events...")
        econ_events = self.calendar_manager.get_upcoming_events(
            days_ahead=days_ahead,
            min_importance=3  # HIGH 이상
        )
        events.extend(econ_events)
        self._log(f"  Found {len(econ_events)} economic events")

        # 2. 실적 발표
        if tickers:
            self._log("Fetching earnings calendar...")
            earnings_events = self.earnings_calendar.get_upcoming_earnings(
                tickers=tickers,
                days_ahead=days_ahead
            )
            events.extend(earnings_events)
            self._log(f"  Found {len(earnings_events)} earnings events")

        return sorted(events, key=lambda e: e.timestamp)

    def detect_market_events(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> List[Event]:
        """시장 데이터에서 이벤트 감지"""
        all_events = []

        for ticker, df in data.items():
            if df.empty:
                continue

            self._log(f"Detecting events for {ticker}...")
            events = self.quantitative_detector.detect_all(df, ticker=ticker)
            all_events.extend(events)
            self._log(f"  Found {len(events)} events")

        return all_events

    def analyze_event_context(
        self,
        date: datetime,
        ticker: str = None,
        lookback_days: int = 5,
        lookahead_days: int = 5
    ) -> Dict[str, Any]:
        """이벤트 컨텍스트 분석"""
        context = {
            "date": date,
            "ticker": ticker,
            "scheduled_events": [],
            "detected_events": [],
            "pre_event_setup": {},
            "post_event_reaction": {}
        }

        # 해당 날짜 예정 이벤트
        context["scheduled_events"] = self.calendar_manager.get_events_for_date(date)

        # 주변 이벤트
        nearby_events = self.calendar_manager.get_upcoming_events(
            days_ahead=lookahead_days
        )
        context["nearby_events"] = nearby_events

        # 다음 주요 이벤트까지 일수
        context["days_to_fomc"] = self.calendar_manager.days_to_next_event(EventType.FOMC)
        context["days_to_cpi"] = self.calendar_manager.days_to_next_event(EventType.CPI)
        context["days_to_nfp"] = self.calendar_manager.days_to_next_event(EventType.NFP)

        return context

    def get_event_summary(self, days_ahead: int = 7) -> str:
        """이벤트 요약 출력"""
        events = self.get_all_upcoming_events(days_ahead=days_ahead)

        lines = [
            "=" * 60,
            "EIMAS Event Framework - Upcoming Events",
            "=" * 60,
            ""
        ]

        # 날짜별 그룹화
        by_date = {}
        for event in events:
            date_str = event.timestamp.strftime("%Y-%m-%d (%a)")
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(event)

        for date_str, date_events in sorted(by_date.items()):
            lines.append(f"\n{date_str}")
            lines.append("-" * 40)
            for event in date_events:
                importance_icon = "🔴" if event.importance == EventImportance.CRITICAL else "🟡"
                lines.append(f"  {importance_icon} {event.name}")

        # 다음 주요 이벤트
        lines.append("\n" + "=" * 60)
        lines.append("Days to Next Major Events:")
        lines.append(f"  FOMC: {self.calendar_manager.days_to_next_event(EventType.FOMC)} days")
        lines.append(f"  CPI:  {self.calendar_manager.days_to_next_event(EventType.CPI)} days")
        lines.append(f"  NFP:  {self.calendar_manager.days_to_next_event(EventType.NFP)} days")
        lines.append("=" * 60)

        return "\n".join(lines)


# ============================================================================
# Event Impact Analyzer
# ============================================================================
