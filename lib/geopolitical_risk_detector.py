#!/usr/bin/env python3
"""
Geopolitical Risk & Black Swan Detector
========================================
지정학적 리스크와 블랙스완 이벤트 실시간 감지 시스템

핵심 기능:
1. 뉴스 API를 통한 실시간 이벤트 감지
2. 키워드 기반 위험 분류
3. 심각도(Severity) 평가
4. 시장 영향 예측

경제학적 배경:
- Black Swan Events (Nassim Taleb): 예측 불가능하지만 극심한 영향
- Geopolitical Risk Index (Caldara & Iacoviello 2018)
- News-driven Market Reactions
- Event Study Methodology

Data Sources:
- NewsAPI (newsapi.org)
- Google News RSS
- Twitter/X API (optional)
- GDELT Project (optional)
"""

import sys
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import requests
import re
from collections import Counter

# Optional imports
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("[WARN] BeautifulSoup not available. HTML parsing will be limited.")


class RiskCategory(str, Enum):
    """리스크 카테고리"""
    WAR = "war"                          # 전쟁
    TERRORISM = "terrorism"              # 테러
    COUP = "coup"                        # 쿠데타
    SANCTIONS = "sanctions"              # 경제 제재
    PANDEMIC = "pandemic"                # 팬데믹
    NATURAL_DISASTER = "natural_disaster"  # 자연재해
    FINANCIAL_CRISIS = "financial_crisis"  # 금융위기
    POLITICAL_CRISIS = "political_crisis"  # 정치 위기
    CYBER_ATTACK = "cyber_attack"        # 사이버 공격
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """심각도"""
    LOW = "low"          # 1-3: 경미한 영향
    MEDIUM = "medium"    # 4-6: 중간 영향
    HIGH = "high"        # 7-9: 심각한 영향
    CRITICAL = "critical"  # 10: 블랙스완 수준


@dataclass
class GeopoliticalEvent:
    """지정학적 이벤트"""
    id: str
    title: str
    description: str
    category: RiskCategory
    severity: Severity
    severity_score: int  # 1-10
    confidence: float    # 0-1
    timestamp: str
    source: str
    url: Optional[str] = None
    affected_regions: List[str] = field(default_factory=list)
    affected_assets: List[str] = field(default_factory=list)  # 영향받을 자산 (SPY, GLD, OIL 등)
    market_impact_estimate: str = ""  # 예상 시장 영향
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'category': self.category.value,
            'severity': self.severity.value,
            'severity_score': self.severity_score,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'source': self.source,
            'url': self.url,
            'affected_regions': self.affected_regions,
            'affected_assets': self.affected_assets,
            'market_impact_estimate': self.market_impact_estimate,
            'keywords': self.keywords
        }


@dataclass
class RiskAlert:
    """리스크 알림"""
    alert_id: str
    event: GeopoliticalEvent
    triggered_at: str
    alert_message: str
    recommended_actions: List[str] = field(default_factory=list)


# ============================================================================
# Keyword Dictionaries (키워드 사전)
# ============================================================================

RISK_KEYWORDS = {
    RiskCategory.WAR: [
        'war', 'warfare', 'military invasion', 'armed conflict', 'missile strike',
        'bombing', 'airstrike', 'ground offensive', 'naval blockade',
        'troops deploy', 'mobilization', 'declaration of war', 'ceasefire breakdown'
    ],
    RiskCategory.TERRORISM: [
        'terrorism', 'terrorist attack', 'bombing', 'hostage',
        'suicide bomber', 'extremist', 'ISIS', 'Al-Qaeda', 'mass shooting'
    ],
    RiskCategory.COUP: [
        'coup', 'military coup', 'overthrow', 'government collapse',
        'revolution', 'uprising', 'martial law', 'state of emergency'
    ],
    RiskCategory.SANCTIONS: [
        'sanctions', 'economic sanctions', 'trade embargo', 'export ban',
        'asset freeze', 'financial restrictions', 'SWIFT ban'
    ],
    RiskCategory.PANDEMIC: [
        'pandemic', 'epidemic', 'outbreak', 'virus', 'disease',
        'lockdown', 'quarantine', 'infection rate', 'WHO declares'
    ],
    RiskCategory.NATURAL_DISASTER: [
        'earthquake', 'tsunami', 'hurricane', 'typhoon', 'flood',
        'wildfire', 'volcano', 'landslide', 'tornado', 'drought'
    ],
    RiskCategory.FINANCIAL_CRISIS: [
        'financial crisis', 'bank collapse', 'default', 'bankruptcy',
        'debt crisis', 'market crash', 'recession', 'depression',
        'credit crunch', 'liquidity crisis'
    ],
    RiskCategory.POLITICAL_CRISIS: [
        'political crisis', 'impeachment', 'resignation', 'scandal',
        'corruption', 'election fraud', 'constitutional crisis'
    ],
    RiskCategory.CYBER_ATTACK: [
        'cyber attack', 'hacking', 'ransomware', 'data breach',
        'infrastructure hack', 'DDoS attack', 'supply chain attack'
    ]
}

# 심각도 증폭 키워드 (이 단어가 포함되면 심각도 +2)
SEVERITY_AMPLIFIERS = [
    'nuclear', 'atomic', 'WMD', 'massive', 'catastrophic',
    'unprecedented', 'global', 'systemic', 'collapse', 'meltdown'
]

# 지역별 시장 영향
REGION_ASSET_MAPPING = {
    'russia': ['RSX', 'XLE'],  # Russia → Energy
    'china': ['FXI', 'MCHI', 'KWEB'],
    'middle east': ['XLE', 'USO', 'OIL'],  # Oil
    'europe': ['EZU', 'VGK'],
    'usa': ['SPY', 'QQQ'],
    'japan': ['EWJ'],
    'korea': ['EWY'],
    'taiwan': ['EWT', 'TSM'],
}


class GeopoliticalRiskDetector:
    """
    지정학적 리스크 감지기

    주요 기능:
    - 뉴스 API를 통한 실시간 감시
    - 키워드 기반 리스크 분류
    - 심각도 평가
    - 시장 영향 예측
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        check_interval: int = 300,  # 5분
        lookback_hours: int = 24,
        verbose: bool = True
    ):
        """
        Args:
            newsapi_key: NewsAPI 키 (newsapi.org에서 발급)
            check_interval: 체크 간격 (초)
            lookback_hours: 과거 몇 시간의 뉴스를 확인할지
            verbose: 로그 출력 여부
        """
        self.newsapi_key = newsapi_key or os.getenv('NEWSAPI_KEY')
        self.check_interval = check_interval
        self.lookback_hours = lookback_hours
        self.verbose = verbose

        self.logger = self._setup_logger()

        # 최근 감지된 이벤트 (중복 방지)
        self.recent_events: Dict[str, GeopoliticalEvent] = {}

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("GeopoliticalRiskDetector")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def fetch_news(self, query: str = "war OR terrorism OR crisis") -> List[Dict]:
        """
        NewsAPI에서 뉴스 가져오기

        Args:
            query: 검색 쿼리

        Returns:
            뉴스 리스트
        """
        if not self.newsapi_key:
            self.logger.warning("NewsAPI key not set. Using fallback RSS method.")
            return self._fetch_google_news_rss()

        url = "https://newsapi.org/v2/everything"

        # 시간 범위
        from_time = (datetime.now() - timedelta(hours=self.lookback_hours)).isoformat()

        params = {
            'q': query,
            'from': from_time,
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': self.newsapi_key,
            'pageSize': 100
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'ok':
                self.logger.info(f"Fetched {len(data['articles'])} articles from NewsAPI")
                return data['articles']
            else:
                self.logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []

        except Exception as e:
            self.logger.error(f"Error fetching news: {e}")
            return []

    def _fetch_google_news_rss(self) -> List[Dict]:
        """
        Google News RSS를 통한 뉴스 가져오기 (Fallback)

        Note: BeautifulSoup 필요
        """
        if not BEAUTIFULSOUP_AVAILABLE:
            self.logger.warning("BeautifulSoup not available. Cannot fetch RSS.")
            return []

        # Google News RSS (geopolitics)
        rss_url = "https://news.google.com/rss/search?q=geopolitics+OR+war+OR+crisis&hl=en-US&gl=US&ceid=US:en"

        try:
            response = requests.get(rss_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')

            articles = []
            for item in items[:50]:  # 최대 50개
                title = item.title.text if item.title else ""
                description = item.description.text if item.description else ""
                link = item.link.text if item.link else ""
                pub_date = item.pubDate.text if item.pubDate else datetime.now().isoformat()

                articles.append({
                    'title': title,
                    'description': description,
                    'url': link,
                    'publishedAt': pub_date,
                    'source': {'name': 'Google News'}
                })

            self.logger.info(f"Fetched {len(articles)} articles from Google News RSS")
            return articles

        except Exception as e:
            self.logger.error(f"Error fetching Google News RSS: {e}")
            return []

    def analyze_event(self, article: Dict) -> Optional[GeopoliticalEvent]:
        """
        뉴스 기사 분석하여 지정학적 이벤트 추출

        Args:
            article: 뉴스 기사 딕셔너리

        Returns:
            GeopoliticalEvent 또는 None
        """
        title = article.get('title', '')
        description = article.get('description', '') or article.get('content', '')
        text = f"{title} {description}".lower()

        # 1. 카테고리 분류
        category, confidence = self._classify_risk(text)

        if category == RiskCategory.UNKNOWN or confidence < 0.3:
            return None  # 관련 없는 뉴스

        # 2. 심각도 평가
        severity_score = self._calculate_severity(text, category)

        # 심각도 레벨
        if severity_score >= 10:
            severity = Severity.CRITICAL
        elif severity_score >= 7:
            severity = Severity.HIGH
        elif severity_score >= 4:
            severity = Severity.MEDIUM
        else:
            severity = Severity.LOW

        # 3. 영향받는 지역 추출
        affected_regions = self._extract_regions(text)

        # 4. 영향받는 자산 예측
        affected_assets = self._predict_affected_assets(category, affected_regions)

        # 5. 시장 영향 예측
        market_impact = self._predict_market_impact(category, severity_score)

        # 6. 키워드 추출
        keywords = self._extract_keywords(text, category)

        event = GeopoliticalEvent(
            id=f"geo_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{abs(hash(title)) % 10000}",
            title=title,
            description=description[:500],
            category=category,
            severity=severity,
            severity_score=severity_score,
            confidence=confidence,
            timestamp=article.get('publishedAt', datetime.now().isoformat()),
            source=article.get('source', {}).get('name', 'Unknown'),
            url=article.get('url'),
            affected_regions=affected_regions,
            affected_assets=affected_assets,
            market_impact_estimate=market_impact,
            keywords=keywords
        )

        return event

    def _classify_risk(self, text: str) -> Tuple[RiskCategory, float]:
        """
        텍스트를 분석하여 리스크 카테고리 분류

        Returns:
            (카테고리, 신뢰도)
        """
        scores = {}

        for category, keywords in RISK_KEYWORDS.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text:
                    score += 1

            if score > 0:
                scores[category] = score

        if not scores:
            return RiskCategory.UNKNOWN, 0.0

        # 최고 점수 카테고리
        best_category = max(scores, key=scores.get)
        max_score = scores[best_category]

        # 신뢰도 계산 (0-1)
        confidence = min(1.0, max_score / 3)  # 3개 이상 매칭되면 100% 신뢰

        return best_category, confidence

    def _calculate_severity(self, text: str, category: RiskCategory) -> int:
        """
        심각도 점수 계산 (1-10)

        기준:
        - 카테고리별 기본 점수
        - 증폭 키워드 (+2)
        - 다수 키워드 매칭 (+1 per keyword)
        """
        # 카테고리별 기본 점수
        base_scores = {
            RiskCategory.WAR: 8,
            RiskCategory.TERRORISM: 6,
            RiskCategory.COUP: 7,
            RiskCategory.SANCTIONS: 5,
            RiskCategory.PANDEMIC: 7,
            RiskCategory.NATURAL_DISASTER: 5,
            RiskCategory.FINANCIAL_CRISIS: 8,
            RiskCategory.POLITICAL_CRISIS: 4,
            RiskCategory.CYBER_ATTACK: 6,
            RiskCategory.UNKNOWN: 3
        }

        score = base_scores.get(category, 3)

        # 증폭 키워드 체크
        for amplifier in SEVERITY_AMPLIFIERS:
            if amplifier in text:
                score += 2

        # 키워드 매칭 개수
        keyword_count = sum(1 for kw in RISK_KEYWORDS.get(category, []) if kw in text)
        score += min(keyword_count, 3)  # 최대 +3

        return min(10, max(1, score))

    def _extract_regions(self, text: str) -> List[str]:
        """텍스트에서 지역 추출"""
        regions = []

        region_keywords = {
            'russia': ['russia', 'moscow', 'kremlin', 'putin'],
            'china': ['china', 'beijing', 'xi jinping'],
            'middle east': ['middle east', 'iran', 'iraq', 'syria', 'israel', 'gaza', 'saudi'],
            'europe': ['europe', 'eu', 'european union', 'nato'],
            'usa': ['united states', 'us', 'america', 'washington'],
            'japan': ['japan', 'tokyo'],
            'korea': ['korea', 'seoul', 'pyongyang'],
            'taiwan': ['taiwan', 'taipei'],
        }

        for region, keywords in region_keywords.items():
            if any(kw in text for kw in keywords):
                regions.append(region)

        return regions

    def _predict_affected_assets(
        self,
        category: RiskCategory,
        regions: List[str]
    ) -> List[str]:
        """영향받을 자산 예측"""
        assets = set()

        # 지역 기반
        for region in regions:
            if region in REGION_ASSET_MAPPING:
                assets.update(REGION_ASSET_MAPPING[region])

        # 카테고리 기반
        if category == RiskCategory.WAR:
            assets.update(['XLE', 'GLD', 'VIX'])  # Energy, Gold, Volatility
        elif category == RiskCategory.FINANCIAL_CRISIS:
            assets.update(['SPY', 'TLT', 'GLD'])  # Equities down, Bonds/Gold up
        elif category == RiskCategory.PANDEMIC:
            assets.update(['XLV', 'TLT', 'ZOOM'])  # Healthcare, Bonds
        elif category == RiskCategory.CYBER_ATTACK:
            assets.update(['HACK', 'XLK'])  # Cybersecurity, Tech

        return list(assets)

    def _predict_market_impact(self, category: RiskCategory, severity: int) -> str:
        """시장 영향 예측"""
        if severity >= 8:
            return "MAJOR NEGATIVE: Flight to safety (Gold, Bonds up, Equities down)"
        elif severity >= 6:
            return "MODERATE NEGATIVE: Increased volatility, sector rotation"
        elif severity >= 4:
            return "MINOR NEGATIVE: Short-term pullback possible"
        else:
            return "MINIMAL: Localized impact only"

    def _extract_keywords(self, text: str, category: RiskCategory) -> List[str]:
        """관련 키워드 추출"""
        keywords = []
        for kw in RISK_KEYWORDS.get(category, []):
            if kw in text:
                keywords.append(kw)
        return keywords[:5]  # 최대 5개

    def scan_for_risks(self) -> List[GeopoliticalEvent]:
        """
        실시간 리스크 스캔

        Returns:
            감지된 이벤트 리스트
        """
        self.logger.info("Starting geopolitical risk scan...")

        # 뉴스 가져오기
        articles = self.fetch_news()

        events = []
        for article in articles:
            event = self.analyze_event(article)

            if event:
                # 중복 체크 (같은 제목)
                if event.title not in self.recent_events:
                    events.append(event)
                    self.recent_events[event.title] = event

        self.logger.info(f"Detected {len(events)} geopolitical events")

        return events

    def generate_alerts(self, events: List[GeopoliticalEvent]) -> List[RiskAlert]:
        """
        이벤트를 기반으로 알림 생성

        Args:
            events: 감지된 이벤트 리스트

        Returns:
            알림 리스트
        """
        alerts = []

        for event in events:
            # 심각도가 HIGH 이상인 경우만 알림
            if event.severity in [Severity.HIGH, Severity.CRITICAL]:
                alert = RiskAlert(
                    alert_id=f"alert_{event.id}",
                    event=event,
                    triggered_at=datetime.now().isoformat(),
                    alert_message=self._generate_alert_message(event),
                    recommended_actions=self._recommend_actions(event)
                )
                alerts.append(alert)

        return alerts

    def _generate_alert_message(self, event: GeopoliticalEvent) -> str:
        """알림 메시지 생성"""
        return (
            f"🚨 {event.severity.value.upper()} GEOPOLITICAL RISK DETECTED\n"
            f"Category: {event.category.value.upper()}\n"
            f"Title: {event.title}\n"
            f"Severity: {event.severity_score}/10\n"
            f"Affected Regions: {', '.join(event.affected_regions)}\n"
            f"Affected Assets: {', '.join(event.affected_assets)}\n"
            f"Market Impact: {event.market_impact_estimate}"
        )

    def _recommend_actions(self, event: GeopoliticalEvent) -> List[str]:
        """권장 조치 생성"""
        actions = []

        if event.severity == Severity.CRITICAL:
            actions.append("IMMEDIATE: Reduce equity exposure")
            actions.append("Increase cash position")
            actions.append("Buy hedges (VIX, TLT, GLD)")
        elif event.severity == Severity.HIGH:
            actions.append("Monitor positions closely")
            actions.append("Consider defensive sectors (XLP, XLU)")
            actions.append("Trim winners, keep dry powder")
        elif event.severity == Severity.MEDIUM:
            actions.append("Stay informed")
            actions.append("Review portfolio risk")

        # 카테고리별 특수 권장사항
        if event.category == RiskCategory.WAR:
            actions.append("Consider energy (XLE) and defense (ITA) positions")
        elif event.category == RiskCategory.FINANCIAL_CRISIS:
            actions.append("Avoid financials (XLF), prefer quality bonds (TLT)")
        elif event.category == RiskCategory.PANDEMIC:
            actions.append("Healthcare (XLV) and remote work (ZOOM) may benefit")

        return actions


# Test code
if __name__ == "__main__":
    print("=" * 70)
    print("Testing GeopoliticalRiskDetector")
    print("=" * 70)

    # Note: NewsAPI 키가 필요합니다
    # 무료 키: https://newsapi.org/
    detector = GeopoliticalRiskDetector(verbose=True)

    print("\n[1] Scanning for geopolitical risks...")
    events = detector.scan_for_risks()

    print(f"\n[2] Found {len(events)} events")

    if events:
        print("\nTop 3 events by severity:")
        sorted_events = sorted(events, key=lambda e: e.severity_score, reverse=True)
        for i, event in enumerate(sorted_events[:3], 1):
            print(f"\n--- Event {i} ---")
            print(f"Title: {event.title}")
            print(f"Category: {event.category.value}")
            print(f"Severity: {event.severity.value} ({event.severity_score}/10)")
            print(f"Regions: {', '.join(event.affected_regions) or 'N/A'}")
            print(f"Assets: {', '.join(event.affected_assets) or 'N/A'}")
            print(f"Impact: {event.market_impact_estimate}")

    print("\n[3] Generating alerts...")
    alerts = detector.generate_alerts(events)

    print(f"\nGenerated {len(alerts)} alerts")

    if alerts:
        print("\n" + "=" * 70)
        print("ALERTS")
        print("=" * 70)
        for alert in alerts:
            print(f"\n{alert.alert_message}")
            print("\nRecommended Actions:")
            for action in alert.recommended_actions:
                print(f"  - {action}")

    print("\n" + "=" * 70)
