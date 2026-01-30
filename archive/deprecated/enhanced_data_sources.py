"""
Enhanced Data Sources - 확장 데이터 수집 모듈

기존 data_collector.py를 보완하는 추가 데이터 소스:
- CME FedWatch: Fed 금리 기대
- Economic Calendar: 주요 경제지표 발표 일정
- Enhanced FRED: 추가 거시경제 지표
- Sentiment Data: 시장 심리 지표

Author: EIMAS Team
"""

import os
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import warnings
import json

import pandas as pd
import numpy as np

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from fredapi import Fred
    FREDAPI_AVAILABLE = True
except ImportError:
    FREDAPI_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ============================================================================
# Data Classes
# ============================================================================

class DataFrequency(Enum):
    """데이터 빈도"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    REALTIME = "realtime"


@dataclass
class FedWatchData:
    """CME FedWatch 데이터"""
    meeting_date: datetime
    days_to_meeting: int
    current_rate_bp: int            # 현재 금리 (bp)
    expected_rate_bp: float         # 기대 금리 (bp)
    probabilities: Dict[int, float] # 금리별 확률
    rate_change_expected: float     # 기대 변화폭 (bp)
    uncertainty_index: float        # 불확실성 지수
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EconomicEvent:
    """경제 이벤트"""
    date: datetime
    event_name: str
    country: str
    importance: str             # "high", "medium", "low"
    previous: Optional[float]
    forecast: Optional[float]
    actual: Optional[float]
    currency: str = "USD"


@dataclass
class SentimentData:
    """시장 심리 데이터"""
    date: datetime
    fear_greed_index: float     # 0-100
    put_call_ratio: float
    vix_level: float
    vix_term_structure: str     # "contango", "backwardation"
    retail_sentiment: float     # -1 to 1
    institutional_flow: float   # -1 to 1


# ============================================================================
# FRED Enhanced Indicators
# ============================================================================

# 확장된 FRED 지표 목록
FRED_INDICATORS = {
    # 금리
    "rates": {
        "DFF": "Fed Funds Rate",
        "DGS1MO": "1-Month Treasury",
        "DGS3MO": "3-Month Treasury",
        "DGS6MO": "6-Month Treasury",
        "DGS1": "1-Year Treasury",
        "DGS2": "2-Year Treasury",
        "DGS5": "5-Year Treasury",
        "DGS10": "10-Year Treasury",
        "DGS30": "30-Year Treasury",
        "DFII10": "10-Year TIPS (Real)",
    },

    # 스프레드
    "spreads": {
        "T10Y2Y": "10Y-2Y Spread",
        "T10Y3M": "10Y-3M Spread",
        "BAMLH0A0HYM2": "High Yield Spread",
        "BAMLC0A0CM": "Investment Grade Spread",
        "TEDRATE": "TED Spread",
    },

    # 인플레이션
    "inflation": {
        "CPIAUCSL": "CPI All Items",
        "CPILFESL": "Core CPI",
        "PCEPI": "PCE",
        "PCEPILFE": "Core PCE",
        "T5YIE": "5-Year Breakeven",
        "T10YIE": "10-Year Breakeven",
        "MICH": "Michigan Inflation Expectations",
    },

    # 고용
    "employment": {
        "UNRATE": "Unemployment Rate",
        "PAYEMS": "Nonfarm Payrolls",
        "ICSA": "Initial Claims",
        "CCSA": "Continuing Claims",
        "JTSJOL": "Job Openings (JOLTS)",
        "LNS14000006": "Employment-Population Ratio",
    },

    # 경제활동
    "activity": {
        "GDPC1": "Real GDP",
        "INDPRO": "Industrial Production",
        "RSXFS": "Retail Sales ex Auto",
        "HOUST": "Housing Starts",
        "PERMIT": "Building Permits",
        "UMCSENT": "Consumer Sentiment",
        "DGORDER": "Durable Goods Orders",
    },

    # 통화/신용
    "money_credit": {
        "M2SL": "M2 Money Supply",
        "TOTCI": "Commercial & Industrial Loans",
        "DRTSCILM": "Bank Lending Standards",
        "BOGZ1FL073164003Q": "Household Debt Service",
    },

    # 금융 상황
    "financial_conditions": {
        "NFCI": "Chicago Fed Financial Conditions",
        "STLFSI4": "St. Louis Financial Stress",
        "VIXCLS": "VIX",
        "DTWEXBGS": "Trade Weighted Dollar",
    },

    # 글로벌
    "global": {
        "GFDEGDQ188S": "Debt to GDP",
        "IR3TIB01DEM156N": "Germany 3M Rate",
        "IRLTLT01JPM156N": "Japan 10Y Rate",
    }
}

# 주요 FOMC 날짜 (2025년)
FOMC_DATES_2025 = [
    datetime(2025, 1, 29),
    datetime(2025, 3, 19),
    datetime(2025, 5, 7),
    datetime(2025, 6, 18),
    datetime(2025, 7, 30),
    datetime(2025, 9, 17),
    datetime(2025, 11, 5),
    datetime(2025, 12, 17),
]


# ============================================================================
# CME FedWatch Collector
# ============================================================================

class CMEFedWatchCollector:
    """
    CME FedWatch 데이터 수집

    Fed Funds Futures에서 금리 기대 추출
    """

    def __init__(self):
        self.fomc_dates = FOMC_DATES_2025
        self._cache: Dict[str, FedWatchData] = {}
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=1)

    def get_next_fomc_date(self) -> Optional[datetime]:
        """다음 FOMC 회의 날짜"""
        now = datetime.now()
        for date in self.fomc_dates:
            if date > now:
                return date
        return None

    def get_days_to_meeting(self, meeting_date: datetime) -> int:
        """FOMC까지 남은 일수"""
        return (meeting_date - datetime.now()).days

    async def fetch_from_futures(self) -> Optional[FedWatchData]:
        """
        Fed Funds Futures에서 금리 기대 추출

        Note: 실제 구현은 CME 데이터 피드 또는 Yahoo Finance의
        Fed Funds Futures (ZQ) 데이터를 사용
        """
        next_fomc = self.get_next_fomc_date()
        if not next_fomc:
            return None

        try:
            if not YFINANCE_AVAILABLE:
                return self._get_fallback_data(next_fomc)

            # Fed Funds Futures 심볼 (예: ZQF25 = 2025년 1월)
            # 실제로는 해당 월의 정확한 심볼 필요
            current_rate = await self._get_current_fed_rate()

            # 간단한 시뮬레이션 (실제는 futures 가격에서 계산)
            expected_rate = current_rate  # 실제 구현 시 futures에서 추출

            return FedWatchData(
                meeting_date=next_fomc,
                days_to_meeting=self.get_days_to_meeting(next_fomc),
                current_rate_bp=int(current_rate * 100),
                expected_rate_bp=expected_rate * 100,
                probabilities=self._calculate_probabilities(current_rate, expected_rate),
                rate_change_expected=(expected_rate - current_rate) * 100,
                uncertainty_index=self._calculate_uncertainty()
            )

        except Exception as e:
            warnings.warn(f"FedWatch fetch failed: {e}")
            return self._get_fallback_data(next_fomc)

    async def _get_current_fed_rate(self) -> float:
        """현재 Fed Funds Rate 가져오기"""
        try:
            if FREDAPI_AVAILABLE:
                api_key = os.getenv('FRED_API_KEY')
                if api_key:
                    fred = Fred(api_key=api_key)
                    rate = fred.get_series('DFF').iloc[-1]
                    return float(rate)

            # Fallback: 대략적인 현재 금리 (2025년 기준)
            return 4.50  # 예시값

        except Exception:
            return 4.50

    def _calculate_probabilities(
        self,
        current: float,
        expected: float
    ) -> Dict[int, float]:
        """금리별 확률 계산 (단순화)"""
        # 실제 구현은 options 가격에서 추출
        diff = expected - current

        # 25bp 단위로 확률 분포
        base = int(current * 4) / 4  # 가장 가까운 25bp

        probs = {}
        for i in range(-2, 3):  # -50bp ~ +50bp
            rate = base + i * 0.25
            rate_bp = int(rate * 100)

            # 간단한 정규분포 가정
            distance = abs(rate - expected)
            prob = max(0, 1 - distance * 4)  # 단순화
            probs[rate_bp] = round(prob, 3)

        # 정규화
        total = sum(probs.values())
        if total > 0:
            probs = {k: round(v / total, 3) for k, v in probs.items()}

        return probs

    def _calculate_uncertainty(self) -> float:
        """불확실성 지수 계산"""
        # 실제 구현은 options implied volatility에서 추출
        # 여기서는 VIX 기반 프록시 사용
        try:
            if YFINANCE_AVAILABLE:
                vix = yf.Ticker("^VIX")
                hist = vix.history(period="1d")
                if not hist.empty:
                    vix_level = hist['Close'].iloc[-1]
                    # 정규화: VIX 15 = 0.3, VIX 30 = 0.7
                    return min(max((vix_level - 10) / 30, 0), 1)
        except Exception:
            pass

        return 0.5  # 기본값

    def _get_fallback_data(self, next_fomc: datetime) -> FedWatchData:
        """Fallback 데이터"""
        return FedWatchData(
            meeting_date=next_fomc,
            days_to_meeting=self.get_days_to_meeting(next_fomc),
            current_rate_bp=450,
            expected_rate_bp=450,
            probabilities={425: 0.2, 450: 0.6, 475: 0.2},
            rate_change_expected=0,
            uncertainty_index=0.5
        )

    def get_all_meeting_expectations(self) -> List[FedWatchData]:
        """모든 FOMC 회의에 대한 기대 (동기 버전)"""
        results = []
        now = datetime.now()

        for date in self.fomc_dates:
            if date > now:
                results.append(self._get_fallback_data(date))

        return results


# ============================================================================
# Enhanced FRED Collector
# ============================================================================

class EnhancedFREDCollector:
    """
    확장된 FRED 데이터 수집

    카테고리별로 구조화된 거시경제 지표 수집
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.fred = None

        if self.api_key and FREDAPI_AVAILABLE:
            try:
                self.fred = Fred(api_key=self.api_key)
            except Exception as e:
                warnings.warn(f"FRED init failed: {e}")

    def is_available(self) -> bool:
        return self.fred is not None

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.Series]:
        """단일 시리즈 수집"""
        if not self.is_available():
            return None

        try:
            kwargs = {}
            if start_date:
                kwargs['observation_start'] = start_date
            if end_date:
                kwargs['observation_end'] = end_date

            return self.fred.get_series(series_id, **kwargs)
        except Exception as e:
            warnings.warn(f"FRED {series_id} failed: {e}")
            return None

    def get_category(
        self,
        category: str,
        start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        카테고리별 지표 수집

        Parameters:
        -----------
        category : str
            "rates", "spreads", "inflation", "employment", etc.
        start_date : str
            시작 날짜 (YYYY-MM-DD)

        Returns:
        --------
        DataFrame
            카테고리 내 모든 지표
        """
        if category not in FRED_INDICATORS:
            return pd.DataFrame()

        indicators = FRED_INDICATORS[category]
        dfs = []

        for series_id, name in indicators.items():
            series = self.get_series(series_id, start_date=start_date)
            if series is not None:
                series.name = series_id
                dfs.append(series)

        if dfs:
            return pd.concat(dfs, axis=1)
        return pd.DataFrame()

    def get_all_indicators(
        self,
        start_date: Optional[str] = None,
        categories: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        전체 지표 수집

        Returns:
        --------
        Dict[str, DataFrame]
            카테고리별 DataFrame
        """
        if categories is None:
            categories = list(FRED_INDICATORS.keys())

        results = {}
        for category in categories:
            df = self.get_category(category, start_date)
            if not df.empty:
                results[category] = df

        return results

    def get_summary_stats(self, lookback_days: int = 30) -> Dict[str, Any]:
        """최근 N일 요약 통계"""
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

        stats = {}

        # 주요 지표들의 최신값과 변화
        key_indicators = {
            "DFF": "Fed Funds Rate",
            "DGS10": "10Y Treasury",
            "T10Y2Y": "Yield Curve",
            "BAMLH0A0HYM2": "HY Spread",
            "UNRATE": "Unemployment",
            "CPIAUCSL": "CPI YoY",
        }

        for series_id, name in key_indicators.items():
            series = self.get_series(series_id, start_date=start_date)
            if series is not None and len(series) > 0:
                stats[name] = {
                    "current": float(series.iloc[-1]),
                    "change": float(series.iloc[-1] - series.iloc[0]) if len(series) > 1 else 0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                }

        return stats


# ============================================================================
# Economic Calendar
# ============================================================================

class EconomicCalendar:
    """
    경제 이벤트 캘린더

    주요 경제지표 발표 일정 관리
    """

    # 주요 이벤트 (정적 데이터 - 실제로는 API에서 가져옴)
    RECURRING_EVENTS = {
        "FOMC": {
            "description": "Federal Reserve Policy Decision",
            "importance": "high",
            "frequency": "6 weeks",
            "dates": FOMC_DATES_2025
        },
        "NFP": {
            "description": "Nonfarm Payrolls",
            "importance": "high",
            "frequency": "monthly (first Friday)",
        },
        "CPI": {
            "description": "Consumer Price Index",
            "importance": "high",
            "frequency": "monthly (around 10th)",
        },
        "GDP": {
            "description": "Gross Domestic Product",
            "importance": "high",
            "frequency": "quarterly",
        },
        "PCE": {
            "description": "Personal Consumption Expenditures",
            "importance": "high",
            "frequency": "monthly",
        },
        "Retail Sales": {
            "description": "Retail Sales",
            "importance": "medium",
            "frequency": "monthly",
        },
        "ISM Manufacturing": {
            "description": "ISM Manufacturing PMI",
            "importance": "medium",
            "frequency": "monthly (first business day)",
        },
        "Consumer Confidence": {
            "description": "Consumer Confidence Index",
            "importance": "medium",
            "frequency": "monthly",
        }
    }

    def __init__(self):
        self.events_cache: List[EconomicEvent] = []

    def get_upcoming_events(
        self,
        days_ahead: int = 7,
        importance: Optional[str] = None
    ) -> List[EconomicEvent]:
        """
        향후 N일간의 이벤트

        Parameters:
        -----------
        days_ahead : int
            앞으로 며칠
        importance : str
            "high", "medium", "low" 필터

        Returns:
        --------
        List[EconomicEvent]
        """
        now = datetime.now()
        end = now + timedelta(days=days_ahead)

        events = []

        # FOMC 날짜 추가
        for date in FOMC_DATES_2025:
            if now <= date <= end:
                events.append(EconomicEvent(
                    date=date,
                    event_name="FOMC Meeting",
                    country="US",
                    importance="high",
                    previous=None,
                    forecast=None,
                    actual=None
                ))

        # 중요도 필터
        if importance:
            events = [e for e in events if e.importance == importance]

        # 날짜순 정렬
        events.sort(key=lambda x: x.date)

        return events

    def get_next_high_impact_event(self) -> Optional[EconomicEvent]:
        """다음 고영향 이벤트"""
        events = self.get_upcoming_events(days_ahead=30, importance="high")
        return events[0] if events else None


# ============================================================================
# Sentiment Collector
# ============================================================================

class SentimentCollector:
    """
    시장 심리 데이터 수집

    VIX, Put/Call Ratio, Fear & Greed Index 등
    """

    def __init__(self):
        pass

    async def collect_sentiment(self) -> Optional[SentimentData]:
        """현재 심리 데이터 수집"""
        try:
            vix_level = await self._get_vix()
            vix_structure = await self._get_vix_term_structure()
            put_call = await self._get_put_call_ratio()

            # Fear & Greed Index (0-100)
            # VIX 기반 계산 (단순화)
            fear_greed = 100 - min(vix_level * 2, 100)

            return SentimentData(
                date=datetime.now(),
                fear_greed_index=fear_greed,
                put_call_ratio=put_call,
                vix_level=vix_level,
                vix_term_structure=vix_structure,
                retail_sentiment=0.0,  # 별도 소스 필요
                institutional_flow=0.0  # 별도 소스 필요
            )

        except Exception as e:
            warnings.warn(f"Sentiment collection failed: {e}")
            return None

    async def _get_vix(self) -> float:
        """VIX 레벨"""
        if not YFINANCE_AVAILABLE:
            return 20.0  # 기본값

        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
        except Exception:
            pass

        return 20.0

    async def _get_vix_term_structure(self) -> str:
        """VIX 기간구조"""
        if not YFINANCE_AVAILABLE:
            return "unknown"

        try:
            vix = yf.Ticker("^VIX")
            vix3m = yf.Ticker("^VIX3M")

            vix_hist = vix.history(period="1d")
            vix3m_hist = vix3m.history(period="1d")

            if not vix_hist.empty and not vix3m_hist.empty:
                spot = vix_hist['Close'].iloc[-1]
                forward = vix3m_hist['Close'].iloc[-1]

                if forward > spot:
                    return "contango"
                else:
                    return "backwardation"

        except Exception:
            pass

        return "unknown"

    async def _get_put_call_ratio(self) -> float:
        """Put/Call Ratio"""
        # CBOE Put/Call Ratio (실제로는 별도 API 필요)
        # 여기서는 기본값 반환
        return 0.9  # 기본값


# ============================================================================
# Unified Data Manager
# ============================================================================

class EnhancedDataManager:
    """
    통합 데이터 관리자

    모든 데이터 소스를 통합 관리
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_collector = EnhancedFREDCollector(fred_api_key)
        self.fedwatch_collector = CMEFedWatchCollector()
        self.calendar = EconomicCalendar()
        self.sentiment_collector = SentimentCollector()

    async def get_market_context(self) -> Dict[str, Any]:
        """
        현재 시장 컨텍스트 수집

        Returns:
        --------
        Dict with:
            - fedwatch: FedWatch data
            - sentiment: Sentiment data
            - upcoming_events: List of events
            - fred_summary: FRED summary stats
        """
        context = {}

        # 1. FedWatch
        try:
            context['fedwatch'] = await self.fedwatch_collector.fetch_from_futures()
        except Exception:
            context['fedwatch'] = None

        # 2. Sentiment
        try:
            context['sentiment'] = await self.sentiment_collector.collect_sentiment()
        except Exception:
            context['sentiment'] = None

        # 3. Upcoming Events
        context['upcoming_events'] = self.calendar.get_upcoming_events(days_ahead=14)

        # 4. FRED Summary
        if self.fred_collector.is_available():
            context['fred_summary'] = self.fred_collector.get_summary_stats()
        else:
            context['fred_summary'] = {}

        return context

    def get_fred_data(
        self,
        categories: Optional[List[str]] = None,
        start_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """FRED 데이터 수집"""
        return self.fred_collector.get_all_indicators(
            start_date=start_date,
            categories=categories
        )


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("Enhanced Data Sources Demo")
        print("=" * 60)

        # 1. FRED Indicators
        print(f"\n[Available FRED Indicators]")
        for category, indicators in FRED_INDICATORS.items():
            print(f"  {category}: {len(indicators)} indicators")

        # 2. FOMC Dates
        print(f"\n[FOMC Dates 2025]")
        for date in FOMC_DATES_2025[:3]:
            print(f"  - {date.strftime('%Y-%m-%d')}")
        print(f"  ... and {len(FOMC_DATES_2025) - 3} more")

        # 3. FedWatch
        print(f"\n[CME FedWatch]")
        collector = CMEFedWatchCollector()
        next_fomc = collector.get_next_fomc_date()
        if next_fomc:
            print(f"  Next FOMC: {next_fomc.strftime('%Y-%m-%d')}")
            print(f"  Days to meeting: {collector.get_days_to_meeting(next_fomc)}")

        # 4. Economic Calendar
        print(f"\n[Upcoming High-Impact Events]")
        calendar = EconomicCalendar()
        events = calendar.get_upcoming_events(days_ahead=30, importance="high")
        for event in events[:3]:
            print(f"  - {event.date.strftime('%Y-%m-%d')}: {event.event_name}")

        # 5. Enhanced Data Manager
        print(f"\n[Enhanced Data Manager]")
        manager = EnhancedDataManager()
        context = await manager.get_market_context()

        if context.get('fedwatch'):
            fw = context['fedwatch']
            print(f"  FedWatch - Current Rate: {fw.current_rate_bp}bp")
            print(f"  FedWatch - Expected: {fw.expected_rate_bp}bp")

        if context.get('sentiment'):
            sent = context['sentiment']
            print(f"  VIX: {sent.vix_level:.1f}")
            print(f"  Fear/Greed: {sent.fear_greed_index:.0f}")

        print("\n" + "=" * 60)

    asyncio.run(demo())
