#!/usr/bin/env python3
"""
EIMAS Economic Calendar
=======================
Track major economic events and data releases.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import os
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class EventImportance(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventCategory(Enum):
    FED = "fed"
    EMPLOYMENT = "employment"
    INFLATION = "inflation"
    GDP = "gdp"
    HOUSING = "housing"
    MANUFACTURING = "manufacturing"
    CONSUMER = "consumer"
    TRADE = "trade"
    OTHER = "other"


@dataclass
class EconomicEvent:
    date: datetime
    time: str
    name: str
    category: EventCategory
    importance: EventImportance
    previous: Optional[str] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None
    country: str = "US"


# Key economic events (manually curated)
MAJOR_EVENTS = [
    # Federal Reserve
    {"name": "FOMC Rate Decision", "category": "fed", "importance": "critical"},
    {"name": "FOMC Minutes", "category": "fed", "importance": "high"},
    {"name": "Fed Chair Speech", "category": "fed", "importance": "high"},
    {"name": "Beige Book", "category": "fed", "importance": "medium"},

    # Employment
    {"name": "Non-Farm Payrolls", "category": "employment", "importance": "critical"},
    {"name": "Unemployment Rate", "category": "employment", "importance": "critical"},
    {"name": "Initial Jobless Claims", "category": "employment", "importance": "high"},
    {"name": "ADP Employment", "category": "employment", "importance": "medium"},
    {"name": "JOLTS Job Openings", "category": "employment", "importance": "medium"},

    # Inflation
    {"name": "CPI (YoY)", "category": "inflation", "importance": "critical"},
    {"name": "Core CPI (YoY)", "category": "inflation", "importance": "critical"},
    {"name": "PPI (YoY)", "category": "inflation", "importance": "high"},
    {"name": "PCE Price Index", "category": "inflation", "importance": "critical"},
    {"name": "Core PCE", "category": "inflation", "importance": "critical"},

    # GDP
    {"name": "GDP (QoQ)", "category": "gdp", "importance": "critical"},
    {"name": "GDP Price Index", "category": "gdp", "importance": "medium"},

    # Manufacturing
    {"name": "ISM Manufacturing PMI", "category": "manufacturing", "importance": "high"},
    {"name": "ISM Services PMI", "category": "manufacturing", "importance": "high"},
    {"name": "Industrial Production", "category": "manufacturing", "importance": "medium"},
    {"name": "Durable Goods Orders", "category": "manufacturing", "importance": "medium"},

    # Consumer
    {"name": "Retail Sales", "category": "consumer", "importance": "high"},
    {"name": "Consumer Confidence", "category": "consumer", "importance": "medium"},
    {"name": "Michigan Consumer Sentiment", "category": "consumer", "importance": "medium"},

    # Housing
    {"name": "Housing Starts", "category": "housing", "importance": "medium"},
    {"name": "Existing Home Sales", "category": "housing", "importance": "medium"},
    {"name": "New Home Sales", "category": "housing", "importance": "medium"},
    {"name": "Case-Shiller Home Price", "category": "housing", "importance": "medium"},

    # Trade
    {"name": "Trade Balance", "category": "trade", "importance": "medium"},
]


class EconomicCalendar:
    """Track economic events and data releases"""

    def __init__(self):
        self.api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        self.events: List[EconomicEvent] = []

    def _fetch_from_perplexity(self, query: str) -> str:
        """Fetch calendar data using Perplexity"""
        if not self.api_key:
            return ""

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": query}],
                        "temperature": 0.1
                    }
                )
                return response.json()['choices'][0]['message']['content']
        except Exception:
            return ""

    def get_upcoming_events(self, days: int = 7) -> List[EconomicEvent]:
        """Get upcoming economic events"""
        print("\n" + "=" * 60)
        print("EIMAS Economic Calendar")
        print("=" * 60)

        # Try to get real dates from Perplexity
        prompt = f"""List the major US economic data releases and Fed events for the next {days} days.
Include: date, time (EST), event name, previous value, forecast.
Focus on: FOMC, CPI, PPI, PCE, NFP, GDP, ISM PMI, Retail Sales."""

        content = self._fetch_from_perplexity(prompt)

        # For demo, generate sample events based on typical schedule
        events = self._generate_sample_events(days)

        return events

    def _generate_sample_events(self, days: int) -> List[EconomicEvent]:
        """Generate sample events (in production, this would parse real data)"""
        events = []
        today = datetime.now()

        # Simulate typical monthly schedule
        sample_schedule = [
            (1, "Initial Jobless Claims", "employment", "high", "8:30 AM"),
            (2, "ISM Manufacturing PMI", "manufacturing", "high", "10:00 AM"),
            (3, "Retail Sales", "consumer", "high", "8:30 AM"),
            (4, "CPI (YoY)", "inflation", "critical", "8:30 AM"),
            (5, "FOMC Rate Decision", "fed", "critical", "2:00 PM"),
            (6, "Non-Farm Payrolls", "employment", "critical", "8:30 AM"),
            (7, "PCE Price Index", "inflation", "critical", "8:30 AM"),
        ]

        for day_offset, name, category, importance, time in sample_schedule:
            if day_offset <= days:
                event_date = today + timedelta(days=day_offset)
                events.append(EconomicEvent(
                    date=event_date,
                    time=time,
                    name=name,
                    category=EventCategory(category),
                    importance=EventImportance(importance),
                    previous="--",
                    forecast="--"
                ))

        return events

    def get_events_by_importance(self, importance: EventImportance) -> List[EconomicEvent]:
        """Filter events by importance"""
        upcoming = self.get_upcoming_events(14)
        return [e for e in upcoming if e.importance == importance]

    def get_fed_events(self) -> List[EconomicEvent]:
        """Get Fed-related events"""
        upcoming = self.get_upcoming_events(30)
        return [e for e in upcoming if e.category == EventCategory.FED]

    def get_market_moving_events(self) -> List[EconomicEvent]:
        """Get events likely to move markets"""
        upcoming = self.get_upcoming_events(7)
        return [e for e in upcoming if e.importance in [EventImportance.CRITICAL, EventImportance.HIGH]]

    def print_calendar(self, events: List[EconomicEvent] = None):
        """Print economic calendar"""
        if events is None:
            events = self.get_upcoming_events(7)

        print("\n" + "=" * 80)
        print("Upcoming Economic Events")
        print("=" * 80)

        print(f"\n{'Date':<12} {'Time':<10} {'Event':<30} {'Importance':<12} {'Category':<15}")
        print("-" * 80)

        for e in events:
            date_str = e.date.strftime("%Y-%m-%d")
            importance_icon = "🔴" if e.importance == EventImportance.CRITICAL else "🟡" if e.importance == EventImportance.HIGH else "⚪"
            print(f"{date_str:<12} {e.time:<10} {e.name:<30} {importance_icon} {e.importance.value:<10} {e.category.value:<15}")

        # Highlight critical events
        critical = [e for e in events if e.importance == EventImportance.CRITICAL]
        if critical:
            print("\n⚠️  CRITICAL EVENTS THIS WEEK:")
            for e in critical:
                print(f"   {e.date.strftime('%a %m/%d')} {e.time}: {e.name}")

        print("=" * 80)

    def get_fomc_schedule(self) -> List[Dict]:
        """Get FOMC meeting schedule for the year"""
        # 2025 FOMC Schedule (typical)
        fomc_dates = [
            {"date": "2025-01-28/29", "type": "meeting"},
            {"date": "2025-03-18/19", "type": "meeting", "projections": True},
            {"date": "2025-05-06/07", "type": "meeting"},
            {"date": "2025-06-17/18", "type": "meeting", "projections": True},
            {"date": "2025-07-29/30", "type": "meeting"},
            {"date": "2025-09-16/17", "type": "meeting", "projections": True},
            {"date": "2025-11-04/05", "type": "meeting"},
            {"date": "2025-12-16/17", "type": "meeting", "projections": True},
        ]
        return fomc_dates


class EventImpactAnalyzer:
    """Analyze historical impact of economic events"""

    def __init__(self):
        pass

    def analyze_event_impact(self, event_name: str) -> Dict:
        """Analyze typical market impact of an event"""
        # Historical impact data (simplified)
        impact_data = {
            "Non-Farm Payrolls": {
                "typical_move": "0.5-1.5%",
                "affected_sectors": ["Financials", "Consumer Discretionary"],
                "direction_factor": "Better = bullish for stocks, bearish for bonds",
                "typical_duration": "1-2 hours volatility"
            },
            "CPI (YoY)": {
                "typical_move": "0.5-2.0%",
                "affected_sectors": ["Tech", "Growth", "Bonds"],
                "direction_factor": "Higher = bearish (rate hike fears)",
                "typical_duration": "Day-long impact"
            },
            "FOMC Rate Decision": {
                "typical_move": "1.0-3.0%",
                "affected_sectors": ["All sectors", "Bonds", "Dollar"],
                "direction_factor": "Dovish = bullish, Hawkish = bearish",
                "typical_duration": "Multi-day repositioning"
            },
            "PCE Price Index": {
                "typical_move": "0.3-1.0%",
                "affected_sectors": ["Tech", "Growth"],
                "direction_factor": "Fed's preferred inflation measure",
                "typical_duration": "Morning session"
            },
        }

        return impact_data.get(event_name, {
            "typical_move": "Unknown",
            "affected_sectors": ["General market"],
            "direction_factor": "Data dependent",
            "typical_duration": "Variable"
        })


if __name__ == "__main__":
    calendar = EconomicCalendar()
    events = calendar.get_upcoming_events(7)
    calendar.print_calendar(events)

    # Show FOMC schedule
    print("\nFOMC Meeting Schedule 2025:")
    for meeting in calendar.get_fomc_schedule():
        proj = " (with SEP)" if meeting.get('projections') else ""
        print(f"  {meeting['date']}{proj}")

    # Analyze impact
    analyzer = EventImpactAnalyzer()
    print("\nNFP Impact Analysis:")
    impact = analyzer.analyze_event_impact("Non-Farm Payrolls")
    for k, v in impact.items():
        print(f"  {k}: {v}")
