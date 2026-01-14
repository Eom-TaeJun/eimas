"""
Event Tracker - 거래량/가격 이상 시점 역추적 및 뉴스 연결

Flow:
1. VolumeAnalyzer로 이상 탐지
2. 이상 시점(timestamp) 추출
3. NewsCorrelator로 해당 시점 뉴스 검색
4. 이벤트-가격 영향 평가

사용 예시:
    tracker = EventTracker()
    events = tracker.track_anomaly_events(['NVDA', 'TSLA'], days=30)
    for e in events:
        print(f"{e.ticker}: {e.anomaly_type} at {e.timestamp}")
        print(f"  News: {e.news_summary}")
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import yfinance as yf

# 내부 모듈
try:
    from lib.volume_analyzer import VolumeAnalyzer, VolumeAnomaly
    from lib.news_correlator import NewsCorrelator
    from lib.sentiment_analyzer import SentimentAnalyzer
except ImportError:
    from volume_analyzer import VolumeAnalyzer, VolumeAnomaly
    from news_correlator import NewsCorrelator
    from sentiment_analyzer import SentimentAnalyzer

# Perplexity API
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class TrackedEvent:
    """역추적된 이벤트"""
    ticker: str
    timestamp: str
    anomaly_type: str  # VOLUME_SPIKE, PRICE_SURGE, VOLATILITY_BURST

    # 이상 수치
    volume_zscore: float = 0.0
    price_change_pct: float = 0.0
    volatility_zscore: float = 0.0

    # 뉴스/이벤트
    news_found: bool = False
    news_query: str = ""
    news_summary: str = ""
    news_sources: List[str] = field(default_factory=list)

    # 영향 평가
    event_type: str = ""  # earnings, M&A, product, regulatory, macro
    sentiment: str = ""  # positive, negative, neutral
    impact_score: float = 0.0  # -100 to +100

    # 메타
    search_time_ms: int = 0
    raw_news: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EventTrackingResult:
    """이벤트 추적 결과"""
    timestamp: str
    tickers_analyzed: List[str]
    anomalies_found: int
    events_matched: int

    tracked_events: List[TrackedEvent] = field(default_factory=list)

    # 요약
    top_events: List[Dict] = field(default_factory=list)
    event_type_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['tracked_events'] = [e.to_dict() for e in self.tracked_events]
        return d


class EventTracker:
    """
    이벤트 역추적기

    거래량/가격 이상 → 시점 역추적 → 뉴스/이벤트 연결
    """

    # 이상 탐지 임계값
    VOLUME_ZSCORE_THRESHOLD = 2.0
    PRICE_CHANGE_THRESHOLD = 3.0  # %
    VOLATILITY_ZSCORE_THRESHOLD = 2.5

    # 이벤트 유형 키워드
    EVENT_KEYWORDS = {
        'earnings': ['earnings', 'revenue', 'profit', 'EPS', 'quarterly', 'guidance', 'beat', 'miss'],
        'M&A': ['acquisition', 'merger', 'buyout', 'takeover', 'deal', 'acquire'],
        'product': ['launch', 'release', 'announce', 'new product', 'unveil', 'introduce'],
        'regulatory': ['SEC', 'FDA', 'FTC', 'lawsuit', 'fine', 'regulation', 'investigation', 'approve'],
        'macro': ['Fed', 'interest rate', 'inflation', 'GDP', 'employment', 'tariff', 'trade war'],
        'analyst': ['upgrade', 'downgrade', 'price target', 'rating', 'analyst'],
        'insider': ['insider', 'CEO', 'CFO', 'executive', 'resign', 'appoint']
    }

    def __init__(self, use_perplexity: bool = True):
        self.volume_analyzer = VolumeAnalyzer(verbose=False)
        self.news_correlator = NewsCorrelator()

        # Perplexity 클라이언트
        self.perplexity_client = None
        if use_perplexity and OpenAI:
            api_key = os.getenv('PERPLEXITY_API_KEY')
            if api_key:
                self.perplexity_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.perplexity.ai"
                )
                print("[EventTracker] Perplexity API initialized")

    def detect_anomalies(
        self,
        tickers: List[str],
        days: int = 30,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[Dict]:
        """이상 시점 탐지"""
        anomalies = []

        # 데이터 수집
        if market_data is None:
            market_data = {}
            for ticker in tickers:
                try:
                    df = yf.download(ticker, period=f"{days}d", progress=False)
                    if len(df) > 5:
                        market_data[ticker] = df
                except Exception as e:
                    print(f"[EventTracker] Error fetching {ticker}: {e}")

        # 각 티커별 이상 탐지
        for ticker, df in market_data.items():
            ticker_anomalies = self._detect_ticker_anomalies(ticker, df)
            anomalies.extend(ticker_anomalies)

        # 시간순 정렬
        anomalies.sort(key=lambda x: x['timestamp'], reverse=True)

        return anomalies

    def _detect_ticker_anomalies(self, ticker: str, df: pd.DataFrame) -> List[Dict]:
        """단일 티커 이상 탐지"""
        anomalies = []

        if len(df) < 10:
            return anomalies

        # DataFrame 복사 (경고 방지)
        df = df.copy()

        # Volume 컬럼 처리 (MultiIndex인 경우)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 거래량 z-score
        vol_mean = float(df['Volume'].mean())
        vol_std = float(df['Volume'].std())
        if vol_std > 0:
            df['vol_zscore'] = (df['Volume'] - vol_mean) / vol_std
        else:
            df['vol_zscore'] = 0

        # 가격 변화율
        df['price_change'] = df['Close'].pct_change() * 100

        # 변동성 (20일 롤링)
        df['volatility'] = df['Close'].pct_change().rolling(20).std() * 100
        volatility_mean = float(df['volatility'].mean()) if not df['volatility'].isna().all() else 0
        volatility_std = float(df['volatility'].std()) if not df['volatility'].isna().all() else 0
        if volatility_std > 0:
            df['vol_z'] = (df['volatility'] - volatility_mean) / volatility_std
        else:
            df['vol_z'] = 0

        # 이상 탐지
        for idx, row in df.iterrows():
            anomaly_types = []

            if abs(row.get('vol_zscore', 0)) >= self.VOLUME_ZSCORE_THRESHOLD:
                anomaly_types.append('VOLUME_SPIKE')

            if abs(row.get('price_change', 0)) >= self.PRICE_CHANGE_THRESHOLD:
                anomaly_types.append('PRICE_SURGE' if row['price_change'] > 0 else 'PRICE_DROP')

            if abs(row.get('vol_z', 0)) >= self.VOLATILITY_ZSCORE_THRESHOLD:
                anomaly_types.append('VOLATILITY_BURST')

            if anomaly_types:
                anomalies.append({
                    'ticker': ticker,
                    'timestamp': idx.strftime('%Y-%m-%d'),
                    'anomaly_types': anomaly_types,
                    'volume_zscore': float(row.get('vol_zscore', 0)),
                    'price_change_pct': float(row.get('price_change', 0)),
                    'volatility_zscore': float(row.get('vol_z', 0)),
                    'close_price': float(row['Close']),
                    'volume': float(row['Volume'])
                })

        return anomalies

    async def search_news_for_anomaly(
        self,
        ticker: str,
        date: str,
        anomaly_type: str
    ) -> Dict:
        """이상 시점에 대한 뉴스 검색"""
        start_time = datetime.now()

        # 검색 쿼리 생성
        query = f"{ticker} stock {date}"
        if 'VOLUME' in anomaly_type:
            query += " trading volume spike"
        elif 'PRICE' in anomaly_type:
            query += " price movement"
        elif 'VOLATILITY' in anomaly_type:
            query += " volatility news"

        result = {
            'query': query,
            'found': False,
            'summary': '',
            'sources': [],
            'event_type': '',
            'sentiment': 'neutral',
            'raw': ''
        }

        # Perplexity로 검색
        if self.perplexity_client:
            try:
                response = self.perplexity_client.chat.completions.create(
                    model="sonar-pro",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a financial news analyst. Find and summarize news events that explain stock price or volume movements."
                        },
                        {
                            "role": "user",
                            "content": f"""
Search for news about {ticker} around {date} that might explain a {anomaly_type.lower().replace('_', ' ')}.

Respond in JSON format:
{{
    "found": true/false,
    "event_type": "earnings|M&A|product|regulatory|macro|analyst|insider|other",
    "headline": "main headline",
    "summary": "2-3 sentence summary of what happened",
    "sentiment": "positive|negative|neutral",
    "impact_score": -100 to +100 (estimated price impact),
    "sources": ["source1", "source2"]
}}
"""
                        }
                    ],
                    max_tokens=500,
                    temperature=0.1
                )

                raw_text = response.choices[0].message.content
                result['raw'] = raw_text

                # JSON 파싱
                parsed = self._parse_json_response(raw_text)
                if parsed:
                    result['found'] = parsed.get('found', False)
                    result['summary'] = f"{parsed.get('headline', '')}: {parsed.get('summary', '')}"
                    result['event_type'] = parsed.get('event_type', 'other')
                    result['sentiment'] = parsed.get('sentiment', 'neutral')
                    result['impact_score'] = parsed.get('impact_score', 0)
                    result['sources'] = parsed.get('sources', [])

            except Exception as e:
                result['error'] = str(e)

        # 캐시된 뉴스 검색 (fallback)
        if not result['found']:
            try:
                cached_news = self.news_correlator.search_news(
                    query=f"{ticker} {date}",
                    time_window="1d"
                )
                if cached_news:
                    result['summary'] = cached_news[:500]
                    result['found'] = True
            except:
                pass

        result['search_time_ms'] = int((datetime.now() - start_time).total_seconds() * 1000)

        return result

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """JSON 응답 파싱"""
        try:
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]

            return json.loads(json_str)
        except:
            return None

    def _classify_event_type(self, news_text: str) -> str:
        """뉴스 텍스트로 이벤트 유형 분류"""
        news_lower = news_text.lower()

        scores = {}
        for event_type, keywords in self.EVENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in news_lower)
            if score > 0:
                scores[event_type] = score

        if scores:
            return max(scores, key=scores.get)
        return 'other'

    async def track_anomaly_events(
        self,
        tickers: List[str],
        days: int = 30,
        max_events: int = 10
    ) -> EventTrackingResult:
        """전체 이벤트 추적 파이프라인"""
        timestamp = datetime.now().isoformat()

        # 1. 이상 탐지
        print(f"[EventTracker] Detecting anomalies for {len(tickers)} tickers...")
        anomalies = self.detect_anomalies(tickers, days)
        print(f"[EventTracker] Found {len(anomalies)} anomalies")

        # 2. 상위 이상에 대해 뉴스 검색
        tracked_events = []
        anomalies_to_track = anomalies[:max_events]

        for i, anomaly in enumerate(anomalies_to_track):
            print(f"[EventTracker] Searching news for {anomaly['ticker']} ({i+1}/{len(anomalies_to_track)})...")

            news_result = await self.search_news_for_anomaly(
                ticker=anomaly['ticker'],
                date=anomaly['timestamp'],
                anomaly_type=anomaly['anomaly_types'][0]
            )

            event = TrackedEvent(
                ticker=anomaly['ticker'],
                timestamp=anomaly['timestamp'],
                anomaly_type=','.join(anomaly['anomaly_types']),
                volume_zscore=anomaly['volume_zscore'],
                price_change_pct=anomaly['price_change_pct'],
                volatility_zscore=anomaly['volatility_zscore'],
                news_found=news_result['found'],
                news_query=news_result['query'],
                news_summary=news_result['summary'],
                news_sources=news_result.get('sources', []),
                event_type=news_result.get('event_type', ''),
                sentiment=news_result.get('sentiment', 'neutral'),
                impact_score=news_result.get('impact_score', 0),
                search_time_ms=news_result.get('search_time_ms', 0),
                raw_news=news_result.get('raw', '')
            )

            tracked_events.append(event)

        # 3. 이벤트 유형 분포
        event_types = {}
        for e in tracked_events:
            if e.event_type:
                event_types[e.event_type] = event_types.get(e.event_type, 0) + 1

        # 4. 영향도 순 정렬
        top_events = sorted(
            [e.to_dict() for e in tracked_events if e.news_found],
            key=lambda x: abs(x.get('impact_score', 0)),
            reverse=True
        )[:5]

        return EventTrackingResult(
            timestamp=timestamp,
            tickers_analyzed=tickers,
            anomalies_found=len(anomalies),
            events_matched=sum(1 for e in tracked_events if e.news_found),
            tracked_events=tracked_events,
            top_events=top_events,
            event_type_distribution=event_types
        )

    def get_report(self, result: EventTrackingResult) -> str:
        """리포트 생성"""
        lines = []
        lines.append("=" * 70)
        lines.append("EVENT TRACKING REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append(f"Tickers: {', '.join(result.tickers_analyzed)}")
        lines.append(f"Anomalies Found: {result.anomalies_found}")
        lines.append(f"Events Matched: {result.events_matched}")
        lines.append("")

        if result.event_type_distribution:
            lines.append("[EVENT TYPE DISTRIBUTION]")
            for et, count in result.event_type_distribution.items():
                lines.append(f"  {et}: {count}")
            lines.append("")

        if result.tracked_events:
            lines.append("[TRACKED EVENTS]")
            for e in result.tracked_events[:10]:
                lines.append(f"\n  [{e.ticker}] {e.timestamp}")
                lines.append(f"    Anomaly: {e.anomaly_type}")
                lines.append(f"    Volume Z: {e.volume_zscore:.2f}, Price: {e.price_change_pct:+.2f}%")
                if e.news_found:
                    lines.append(f"    Event: {e.event_type} ({e.sentiment})")
                    lines.append(f"    News: {e.news_summary[:100]}...")
                    lines.append(f"    Impact: {e.impact_score:+.0f}")
                else:
                    lines.append(f"    News: Not found")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 테스트
# ============================================================================

async def test_event_tracker():
    """이벤트 추적 테스트"""
    print("Event Tracker Test")
    print("=" * 60)

    tracker = EventTracker(use_perplexity=True)

    # 테스트 티커
    tickers = ['NVDA', 'TSLA', 'AAPL']

    result = await tracker.track_anomaly_events(
        tickers=tickers,
        days=30,
        max_events=5
    )

    print(tracker.get_report(result))

    # JSON 저장
    with open('outputs/event_tracking_test.json', 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print("\nSaved to outputs/event_tracking_test.json")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_event_tracker())
