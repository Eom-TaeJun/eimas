#!/usr/bin/env python3
"""
Event Framework - Data Schemas
============================================================

Event and impact data classes

Economic Foundation:
    - Event study methodology (Fama et al. 1969)
    - Market anomalies detection

Contains:
    - Event: Event data class
    - EventImpact: Impact assessment result
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from .enums import EventType, AssetClass, EventImportance, EventTiming


class Event:
    """이벤트 기본 클래스"""
    event_id: str
    event_type: EventType
    asset_class: AssetClass
    timestamp: datetime
    timing: EventTiming
    importance: EventImportance

    # 상세 정보
    name: str
    description: str = ""
    ticker: Optional[str] = None

    # 수치 정보
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    surprise: Optional[float] = None  # actual - expected

    # 메타데이터
    source: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_surprise(self) -> bool:
        """서프라이즈 여부"""
        if self.surprise is None:
            return False
        return abs(self.surprise) > 0.1  # 10% 이상 차이

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "asset_class": self.asset_class.value,
            "timestamp": self.timestamp.isoformat(),
            "timing": self.timing.value,
            "importance": self.importance.value,
            "name": self.name,
            "ticker": self.ticker,
            "surprise": self.surprise,
            "confidence": self.confidence
        }


@dataclass
class EventImpact:
    """이벤트 임팩트"""
    event: Event

    # 가격 영향
    price_change_pct: float = 0.0
    price_change_1d: float = 0.0
    price_change_5d: float = 0.0

    # 변동성 영향
    volatility_before: float = 0.0
    volatility_after: float = 0.0
    volatility_change: float = 0.0

    # 거래량 영향
    volume_ratio: float = 1.0  # vs 평균

    # 지속 시간
    impact_duration_hours: float = 0.0

    # 방향성
    direction: str = "neutral"  # bullish, bearish, neutral


# ============================================================================
# Economic Calendar (실제 날짜)
# ============================================================================

# 2025-2026 주요 경제 이벤트 캘린더
ECONOMIC_CALENDAR_2025_2026 = {
    # FOMC 회의 (2025)
    "fomc": [
        {"date": "2025-01-29", "name": "FOMC Meeting", "importance": 4},
        {"date": "2025-03-19", "name": "FOMC Meeting + SEP", "importance": 4},
        {"date": "2025-05-07", "name": "FOMC Meeting", "importance": 4},
        {"date": "2025-06-18", "name": "FOMC Meeting + SEP", "importance": 4},
        {"date": "2025-07-30", "name": "FOMC Meeting", "importance": 4},
        {"date": "2025-09-17", "name": "FOMC Meeting + SEP", "importance": 4},
        {"date": "2025-11-05", "name": "FOMC Meeting", "importance": 4},
        {"date": "2025-12-17", "name": "FOMC Meeting + SEP", "importance": 4},
        # 2026
        {"date": "2026-01-28", "name": "FOMC Meeting", "importance": 4},
        {"date": "2026-03-18", "name": "FOMC Meeting + SEP", "importance": 4},
    ],

    # CPI 발표 (매월 둘째주 화~수)
    "cpi": [
        {"date": "2025-01-15", "name": "CPI Dec", "importance": 4},
        {"date": "2025-02-12", "name": "CPI Jan", "importance": 4},
        {"date": "2025-03-12", "name": "CPI Feb", "importance": 4},
        {"date": "2025-04-10", "name": "CPI Mar", "importance": 4},
        {"date": "2025-05-13", "name": "CPI Apr", "importance": 4},
        {"date": "2025-06-11", "name": "CPI May", "importance": 4},
        {"date": "2025-07-11", "name": "CPI Jun", "importance": 4},
        {"date": "2025-08-12", "name": "CPI Jul", "importance": 4},
        {"date": "2025-09-11", "name": "CPI Aug", "importance": 4},
        {"date": "2025-10-10", "name": "CPI Sep", "importance": 4},
        {"date": "2025-11-13", "name": "CPI Oct", "importance": 4},
        {"date": "2025-12-10", "name": "CPI Nov", "importance": 4},
        # 2026
        {"date": "2026-01-14", "name": "CPI Dec 2025", "importance": 4},
        {"date": "2026-02-11", "name": "CPI Jan", "importance": 4},
        {"date": "2026-03-11", "name": "CPI Feb", "importance": 4},
    ],

    # NFP (매월 첫째주 금요일)
    "nfp": [
        {"date": "2025-01-10", "name": "NFP Dec", "importance": 4},
        {"date": "2025-02-07", "name": "NFP Jan", "importance": 4},
        {"date": "2025-03-07", "name": "NFP Feb", "importance": 4},
        {"date": "2025-04-04", "name": "NFP Mar", "importance": 4},
        {"date": "2025-05-02", "name": "NFP Apr", "importance": 4},
        {"date": "2025-06-06", "name": "NFP May", "importance": 4},
        {"date": "2025-07-03", "name": "NFP Jun", "importance": 4},
        {"date": "2025-08-01", "name": "NFP Jul", "importance": 4},
        {"date": "2025-09-05", "name": "NFP Aug", "importance": 4},
        {"date": "2025-10-03", "name": "NFP Sep", "importance": 4},
        {"date": "2025-11-07", "name": "NFP Oct", "importance": 4},
        {"date": "2025-12-05", "name": "NFP Nov", "importance": 4},
        # 2026
        {"date": "2026-01-09", "name": "NFP Dec 2025", "importance": 4},
        {"date": "2026-02-06", "name": "NFP Jan", "importance": 4},
        {"date": "2026-03-06", "name": "NFP Feb", "importance": 4},
    ],

    # PCE (매월 마지막주)
    "pce": [
        {"date": "2025-01-31", "name": "PCE Dec", "importance": 4},
        {"date": "2025-02-28", "name": "PCE Jan", "importance": 4},
        {"date": "2025-03-28", "name": "PCE Feb", "importance": 4},
        {"date": "2025-04-30", "name": "PCE Mar", "importance": 4},
        {"date": "2025-05-30", "name": "PCE Apr", "importance": 4},
        {"date": "2025-06-27", "name": "PCE May", "importance": 4},
        {"date": "2025-07-31", "name": "PCE Jun", "importance": 4},
        {"date": "2025-08-29", "name": "PCE Jul", "importance": 4},
        {"date": "2025-09-26", "name": "PCE Aug", "importance": 4},
        {"date": "2025-10-31", "name": "PCE Sep", "importance": 4},
        {"date": "2025-11-26", "name": "PCE Oct", "importance": 4},
        {"date": "2025-12-23", "name": "PCE Nov", "importance": 4},
        # 2026
        {"date": "2026-01-30", "name": "PCE Dec 2025", "importance": 4},
        {"date": "2026-02-27", "name": "PCE Jan", "importance": 4},
        {"date": "2026-03-27", "name": "PCE Feb", "importance": 4},
    ],

    # GDP (분기별)
    "gdp": [
        {"date": "2025-01-30", "name": "GDP Q4 Advance", "importance": 4},
        {"date": "2025-02-27", "name": "GDP Q4 Second", "importance": 3},
        {"date": "2025-03-27", "name": "GDP Q4 Final", "importance": 3},
        {"date": "2025-04-30", "name": "GDP Q1 Advance", "importance": 4},
        {"date": "2025-05-29", "name": "GDP Q1 Second", "importance": 3},
        {"date": "2025-06-26", "name": "GDP Q1 Final", "importance": 3},
        {"date": "2025-07-30", "name": "GDP Q2 Advance", "importance": 4},
        {"date": "2025-08-28", "name": "GDP Q2 Second", "importance": 3},
        {"date": "2025-09-25", "name": "GDP Q2 Final", "importance": 3},
        {"date": "2025-10-30", "name": "GDP Q3 Advance", "importance": 4},
        {"date": "2025-11-27", "name": "GDP Q3 Second", "importance": 3},
        {"date": "2025-12-22", "name": "GDP Q3 Final", "importance": 3},
        # 2026
        {"date": "2026-01-29", "name": "GDP Q4 Advance", "importance": 4},
        {"date": "2026-02-26", "name": "GDP Q4 Second", "importance": 3},
        {"date": "2026-03-26", "name": "GDP Q4 Final", "importance": 3},
    ],
}


# ============================================================================
# Quantitative Event Detector
# ============================================================================
