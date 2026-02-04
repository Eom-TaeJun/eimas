#!/usr/bin/env python3
"""
Event Framework - Calendar Management
============================================================

Economic calendar and earnings event management

Classes:
    - CalendarEventManager: Economic calendar manager
    - EarningsCalendar: Earnings event calendar
"""

from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

from .schemas import Event
from .enums import EventType, EventImportance

logger = logging.getLogger(__name__)


class CalendarEventManager:
    """예정된 이벤트 관리"""

    def __init__(self):
        self.calendar = ECONOMIC_CALENDAR_2025_2026

    def get_upcoming_events(
        self,
        days_ahead: int = 7,
        event_types: List[EventType] = None,
        min_importance: int = 1
    ) -> List[Event]:
        """다가오는 이벤트 조회"""
        events = []
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)

        for event_type_str, event_list in self.calendar.items():
            # 필터링
            if event_types:
                try:
                    et = EventType(event_type_str)
                    if et not in event_types:
                        continue
                except ValueError:
                    continue

            for event_data in event_list:
                event_date = datetime.strptime(event_data["date"], "%Y-%m-%d")

                if today <= event_date <= end_date:
                    if event_data["importance"] >= min_importance:
                        events.append(Event(
                            event_id=f"{event_type_str}_{event_data['date']}",
                            event_type=EventType(event_type_str),
                            asset_class=AssetClass.INDEX,
                            timestamp=event_date,
                            timing=EventTiming.SCHEDULED,
                            importance=EventImportance(event_data["importance"]),
                            name=event_data["name"],
                            source="economic_calendar"
                        ))

        return sorted(events, key=lambda e: e.timestamp)

    def get_events_for_date(self, date: datetime) -> List[Event]:
        """특정 날짜의 이벤트"""
        date_str = date.strftime("%Y-%m-%d")
        events = []

        for event_type_str, event_list in self.calendar.items():
            for event_data in event_list:
                if event_data["date"] == date_str:
                    events.append(Event(
                        event_id=f"{event_type_str}_{date_str}",
                        event_type=EventType(event_type_str),
                        asset_class=AssetClass.INDEX,
                        timestamp=datetime.strptime(date_str, "%Y-%m-%d"),
                        timing=EventTiming.SCHEDULED,
                        importance=EventImportance(event_data["importance"]),
                        name=event_data["name"],
                        source="economic_calendar"
                    ))

        return events

    def days_to_next_event(self, event_type: EventType) -> int:
        """다음 특정 이벤트까지 일수"""
        today = datetime.now()

        if event_type.value not in self.calendar:
            return -1

        for event_data in self.calendar[event_type.value]:
            event_date = datetime.strptime(event_data["date"], "%Y-%m-%d")
            if event_date > today:
                return (event_date - today).days

        return -1


# ============================================================================
# Earnings Calendar
# ============================================================================

class EarningsCalendar:
    """실적 발표 캘린더"""

    def __init__(self):
        pass

    def get_upcoming_earnings(
        self,
        tickers: List[str],
        days_ahead: int = 14
    ) -> List[Event]:
        """다가오는 실적 발표 조회"""
        events = []
        today = datetime.now()

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                calendar = stock.calendar

                if calendar is not None and 'Earnings Date' in calendar:
                    earnings_dates = calendar['Earnings Date']
                    if isinstance(earnings_dates, list) and earnings_dates:
                        earnings_date = earnings_dates[0]
                        if isinstance(earnings_date, datetime):
                            if today <= earnings_date <= today + timedelta(days=days_ahead):
                                events.append(Event(
                                    event_id=f"earnings_{ticker}_{earnings_date.date()}",
                                    event_type=EventType.EARNINGS,
                                    asset_class=AssetClass.EQUITY,
                                    timestamp=earnings_date,
                                    timing=EventTiming.SCHEDULED,
                                    importance=EventImportance.HIGH,
                                    name=f"{ticker} Earnings",
                                    ticker=ticker,
                                    source="yfinance"
                                ))
            except Exception:
                pass

        return sorted(events, key=lambda e: e.timestamp)


# ============================================================================
# Unified Event Framework
# ============================================================================
