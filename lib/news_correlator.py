"""
News Correlator - 이상 탐지 시간 기반 뉴스 검색 및 귀인

이상 탐지 → 시간 클러스터링 → 뉴스 검색 → 원인 귀인
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class Anomaly:
    """단일 이상 탐지 기록"""
    timestamp: datetime  # UTC
    asset: str
    anomaly_type: str  # volume_explosion, volatility_spike, price_shock
    value: float
    details: dict = field(default_factory=dict)


@dataclass
class AnomalyCluster:
    """시간적으로 근접한 이상들의 클러스터"""
    cluster_id: str
    start_time: datetime
    end_time: datetime
    anomalies: list[Anomaly]
    severity_score: float = 0.0
    affected_assets: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.affected_assets = list(set(a.asset for a in self.anomalies))
        self._calculate_severity()

    def _calculate_severity(self):
        """심각도 점수 계산"""
        score = 0.0
        for a in self.anomalies:
            if a.anomaly_type == 'volume_explosion':
                score += min(a.value / 5.0, 3.0)  # 5배 = 1점, 최대 3점
            elif a.anomaly_type == 'volatility_spike':
                score += min(a.value / 2.0, 3.0)  # 2σ = 1점, 최대 3점
            elif a.anomaly_type == 'price_shock':
                score += min(abs(a.value) / 2.0, 3.0)  # 2% = 1점, 최대 3점

        # 다중 자산 보너스
        score *= (1 + 0.2 * (len(self.affected_assets) - 1))
        self.severity_score = round(score, 2)


@dataclass
class NewsResult:
    """뉴스 검색 결과"""
    headline: str
    source: str
    language: str
    search_query: str
    relevance_score: float = 0.0


@dataclass
class EventAttribution:
    """이상-뉴스 귀인 결과"""
    cluster: AnomalyCluster
    news: list[NewsResult]
    confidence_score: float
    summary: str


class NewsCorrelator:
    """이상 탐지와 뉴스를 시간 기반으로 연결"""

    # 국가별 키워드 매핑
    COUNTRY_KEYWORDS = {
        'ko': ['korea', 'korean', 'samsung', 'kospi', 'kosdaq', 'hyundai', 'sk', 'seoul',
               'north korea', 'pyongyang', 'kim jong'],
        'zh': ['china', 'chinese', 'taiwan', 'xi jinping', 'beijing', 'shanghai',
               'hong kong', 'alibaba', 'tencent', 'csi', 'hang seng'],
        'ja': ['japan', 'japanese', 'nikkei', 'tokyo', 'yen', 'boj', 'kishida',
               'sony', 'toyota', 'softbank'],
        'de': ['germany', 'german', 'dax', 'bundesbank', 'ecb', 'frankfurt'],
        'es': ['venezuela', 'maduro', 'mexico', 'brazil', 'latin america', 'peso']
    }

    # 언어별 검색 쿼리 템플릿
    LANGUAGE_TEMPLATES = {
        'en': {
            'market': '{asset} market news {date}',
            'breaking': 'breaking news financial markets {date}',
            'geopolitical': 'geopolitical news world events {date}'
        },
        'ko': {
            'market': '{asset} 시장 뉴스 {date}',
            'breaking': '속보 금융시장 {date}',
            'geopolitical': '국제 정세 뉴스 {date}'
        },
        'zh': {
            'market': '{asset} 市场新闻 {date}',
            'breaking': '突发新闻 金融市场 {date}',
            'geopolitical': '国际局势 新闻 {date}'
        },
        'ja': {
            'market': '{asset} 市場ニュース {date}',
            'breaking': '速報 金融市場 {date}',
            'geopolitical': '国際情勢 ニュース {date}'
        }
    }

    # 자산별 표시 이름 (검색용)
    ASSET_NAMES = {
        'BTC': {'en': 'Bitcoin', 'ko': '비트코인', 'zh': '比特币', 'ja': 'ビットコイン'},
        'ETH': {'en': 'Ethereum', 'ko': '이더리움', 'zh': '以太坊', 'ja': 'イーサリアム'},
        'SPY': {'en': 'S&P 500', 'ko': 'S&P 500', 'zh': '标普500', 'ja': 'S&P 500'},
        'QQQ': {'en': 'Nasdaq', 'ko': '나스닥', 'zh': '纳斯达克', 'ja': 'ナスダック'},
        'GLD': {'en': 'Gold', 'ko': '금', 'zh': '黄金', 'ja': '金'},
        'CL=F': {'en': 'Oil', 'ko': '원유', 'zh': '原油', 'ja': '原油'},
        'GC=F': {'en': 'Gold futures', 'ko': '금 선물', 'zh': '黄金期货', 'ja': '金先物'},
        'DX-Y.NYB': {'en': 'US Dollar', 'ko': '달러', 'zh': '美元', 'ja': 'ドル'},
    }

    # 심각도 임계값 (이 이상이어야 뉴스 검색)
    SEVERITY_THRESHOLD = 1.5

    # 클러스터링 윈도우 (분)
    CLUSTER_WINDOW_MINUTES = 30

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'volatile', 'realtime.db'
        )

        # Perplexity API 클라이언트
        api_key = os.getenv('PERPLEXITY_API_KEY')
        if api_key:
            self.perplexity_client = OpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )
        else:
            self.perplexity_client = None
            print("⚠️ PERPLEXITY_API_KEY not set")

        self._init_db()

    def _init_db(self):
        """귀인 결과 저장 테이블 생성"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS event_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id TEXT UNIQUE,
                anomaly_start TEXT,
                anomaly_end TEXT,
                affected_assets TEXT,
                severity_score REAL,
                news_results TEXT,
                confidence_score REAL,
                summary TEXT,
                languages_searched TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 검색 캐시 테이블 (중복 검색 방지)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_key TEXT UNIQUE,
                result TEXT,
                searched_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

    def cluster_anomalies(self, anomalies: list[Anomaly]) -> list[AnomalyCluster]:
        """시간적으로 근접한 이상들을 클러스터링"""
        if not anomalies:
            return []

        # 시간순 정렬
        sorted_anomalies = sorted(anomalies, key=lambda x: x.timestamp)

        clusters = []
        current_cluster = [sorted_anomalies[0]]

        for anomaly in sorted_anomalies[1:]:
            # 이전 이상과의 시간 차이
            time_diff = (anomaly.timestamp - current_cluster[-1].timestamp).total_seconds() / 60

            if time_diff <= self.CLUSTER_WINDOW_MINUTES:
                current_cluster.append(anomaly)
            else:
                # 새 클러스터 시작
                if current_cluster:
                    cluster_id = f"cluster_{current_cluster[0].timestamp.strftime('%Y%m%d_%H%M')}"
                    clusters.append(AnomalyCluster(
                        cluster_id=cluster_id,
                        start_time=current_cluster[0].timestamp,
                        end_time=current_cluster[-1].timestamp,
                        anomalies=current_cluster
                    ))
                current_cluster = [anomaly]

        # 마지막 클러스터
        if current_cluster:
            cluster_id = f"cluster_{current_cluster[0].timestamp.strftime('%Y%m%d_%H%M')}"
            clusters.append(AnomalyCluster(
                cluster_id=cluster_id,
                start_time=current_cluster[0].timestamp,
                end_time=current_cluster[-1].timestamp,
                anomalies=current_cluster
            ))

        return clusters

    def detect_relevant_languages(self, news_text: str) -> list[str]:
        """뉴스 텍스트에서 관련 국가/언어 감지"""
        languages = ['en']  # 영어는 항상 포함
        text_lower = news_text.lower()

        for lang, keywords in self.COUNTRY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if lang not in languages:
                        languages.append(lang)
                    break

        return languages

    def _get_cache_key(self, query: str, time_window: str) -> str:
        """캐시 키 생성"""
        return f"{query}_{time_window}"

    def _check_cache(self, cache_key: str) -> Optional[str]:
        """캐시 확인"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT result FROM search_cache WHERE search_key = ?',
            (cache_key,)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def _save_cache(self, cache_key: str, result: str):
        """캐시 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO search_cache (search_key, result) VALUES (?, ?)',
            (cache_key, result)
        )
        conn.commit()
        conn.close()

    def search_news(self, query: str, time_window: str, language: str = 'en') -> Optional[str]:
        """Perplexity API로 뉴스 검색"""
        if not self.perplexity_client:
            return None

        cache_key = self._get_cache_key(query, time_window)
        cached = self._check_cache(cache_key)
        if cached:
            print(f"  📦 캐시 사용: {query[:50]}...")
            return cached

        try:
            # 언어별 시스템 프롬프트
            system_prompts = {
                'en': "You are a financial news analyst. Find and summarize breaking news related to the query. Focus on events that could impact financial markets. Be concise.",
                'ko': "당신은 금융 뉴스 분석가입니다. 쿼리와 관련된 속보를 찾아 요약하세요. 금융 시장에 영향을 줄 수 있는 이벤트에 집중하세요. 간결하게 작성하세요.",
                'zh': "你是一名金融新闻分析师。查找并总结与查询相关的突发新闻。关注可能影响金融市场的事件。请简洁。",
                'ja': "あなたは金融ニュースアナリストです。クエリに関連するニュースを見つけて要約してください。金融市場に影響を与える可能性のあるイベントに焦点を当ててください。簡潔に。"
            }

            response = self.perplexity_client.chat.completions.create(
                model="sonar",
                messages=[
                    {"role": "system", "content": system_prompts.get(language, system_prompts['en'])},
                    {"role": "user", "content": f"{query}\n\nTime window: {time_window}"}
                ],
                max_tokens=500
            )

            result = response.choices[0].message.content
            self._save_cache(cache_key, result)
            return result

        except Exception as e:
            print(f"  ❌ 검색 실패: {e}")
            return None

    def correlate_cluster(self, cluster: AnomalyCluster) -> Optional[EventAttribution]:
        """클러스터에 대해 뉴스 검색 및 귀인"""

        if cluster.severity_score < self.SEVERITY_THRESHOLD:
            print(f"  ⏭️ 심각도 부족 ({cluster.severity_score} < {self.SEVERITY_THRESHOLD})")
            return None

        print(f"\n🔍 클러스터 분석: {cluster.cluster_id}")
        print(f"   시간: {cluster.start_time} ~ {cluster.end_time} UTC")
        print(f"   자산: {cluster.affected_assets}")
        print(f"   심각도: {cluster.severity_score}")

        # 검색 시간 윈도우 (비대칭: 전 1시간 ~ 후 3시간)
        search_start = cluster.start_time - timedelta(hours=1)
        search_end = cluster.end_time + timedelta(hours=3)
        time_window = f"{search_start.strftime('%Y-%m-%d %H:%M')} to {search_end.strftime('%Y-%m-%d %H:%M')} UTC"
        date_str = cluster.start_time.strftime('%Y-%m-%d')

        news_results = []
        languages_searched = []

        # Phase 1: 영어로 글로벌 개요 검색
        print("\n  📰 Phase 1: 영어 글로벌 검색")

        # 자산 특정 검색
        for asset in cluster.affected_assets[:3]:  # 최대 3개 자산
            asset_name = self.ASSET_NAMES.get(asset, {}).get('en', asset)
            query = f"{asset_name} market news breaking {date_str}"
            result = self.search_news(query, time_window, 'en')
            if result:
                news_results.append(NewsResult(
                    headline=result[:200],
                    source='perplexity',
                    language='en',
                    search_query=query
                ))

        # 지정학적 검색
        query = f"breaking news geopolitical events financial markets {date_str}"
        result = self.search_news(query, time_window, 'en')
        if result:
            news_results.append(NewsResult(
                headline=result[:200],
                source='perplexity',
                language='en',
                search_query=query
            ))

        languages_searched.append('en')

        # Phase 2: 관련 국가 감지 및 해당 언어로 상세 검색
        combined_news = ' '.join(n.headline for n in news_results)
        relevant_langs = self.detect_relevant_languages(combined_news)

        for lang in relevant_langs:
            if lang == 'en':
                continue

            print(f"\n  🌍 Phase 2: {lang.upper()} 상세 검색")

            templates = self.LANGUAGE_TEMPLATES.get(lang, self.LANGUAGE_TEMPLATES['en'])

            # 시장 뉴스 검색
            for asset in cluster.affected_assets[:2]:
                asset_name = self.ASSET_NAMES.get(asset, {}).get(lang, asset)
                query = templates['market'].format(asset=asset_name, date=date_str)
                result = self.search_news(query, time_window, lang)
                if result:
                    news_results.append(NewsResult(
                        headline=result[:200],
                        source='perplexity',
                        language=lang,
                        search_query=query
                    ))

            # 지정학 검색
            query = templates['geopolitical'].format(date=date_str)
            result = self.search_news(query, time_window, lang)
            if result:
                news_results.append(NewsResult(
                    headline=result[:200],
                    source='perplexity',
                    language=lang,
                    search_query=query
                ))

            languages_searched.append(lang)

        # 신뢰도 계산
        confidence = self._calculate_confidence(cluster, news_results)

        # 요약 생성
        summary = self._generate_summary(cluster, news_results)

        attribution = EventAttribution(
            cluster=cluster,
            news=news_results,
            confidence_score=confidence,
            summary=summary
        )

        # DB 저장
        self._save_attribution(attribution, languages_searched)

        return attribution

    def _calculate_confidence(self, cluster: AnomalyCluster, news: list[NewsResult]) -> float:
        """귀인 신뢰도 계산"""
        if not news:
            return 0.0

        # 기본 점수: 뉴스 개수
        score = min(len(news) * 0.15, 0.6)

        # 다국어 보너스
        languages = set(n.language for n in news)
        score += len(languages) * 0.1

        # 심각도 보너스
        score += min(cluster.severity_score * 0.05, 0.2)

        return min(round(score, 2), 1.0)

    def _generate_summary(self, cluster: AnomalyCluster, news: list[NewsResult]) -> str:
        """귀인 결과 요약 생성"""
        if not news:
            return "뉴스를 찾지 못함"

        assets = ', '.join(cluster.affected_assets)
        time_str = cluster.start_time.strftime('%Y-%m-%d %H:%M UTC')

        summary_parts = [
            f"[{time_str}] {assets} 이상 감지",
            f"심각도: {cluster.severity_score}",
            f"관련 뉴스 {len(news)}건 발견:",
        ]

        for i, n in enumerate(news[:3], 1):
            summary_parts.append(f"  {i}. [{n.language.upper()}] {n.headline[:100]}...")

        return '\n'.join(summary_parts)

    def _save_attribution(self, attr: EventAttribution, languages: list[str]):
        """귀인 결과 DB 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO event_attribution
            (cluster_id, anomaly_start, anomaly_end, affected_assets,
             severity_score, news_results, confidence_score, summary, languages_searched)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            attr.cluster.cluster_id,
            attr.cluster.start_time.isoformat(),
            attr.cluster.end_time.isoformat(),
            json.dumps(attr.cluster.affected_assets),
            attr.cluster.severity_score,
            json.dumps([{
                'headline': n.headline,
                'source': n.source,
                'language': n.language,
                'query': n.search_query
            } for n in attr.news]),
            attr.confidence_score,
            attr.summary,
            json.dumps(languages)
        ))

        conn.commit()
        conn.close()
        print(f"  💾 저장됨: {attr.cluster.cluster_id}")

    def process_recent_anomalies(self, hours_back: int = 24) -> list[EventAttribution]:
        """최근 이상들을 처리하고 뉴스와 연결"""

        # volatile DB에서 최근 이상 로드
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()

        cursor.execute('''
            SELECT timestamp, ticker, event_type, value, metadata_json
            FROM detected_events
            WHERE timestamp > ?
            ORDER BY timestamp
        ''', (cutoff,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("최근 이상 없음")
            return []

        # Anomaly 객체로 변환
        anomalies = []
        for row in rows:
            try:
                anomalies.append(Anomaly(
                    timestamp=datetime.fromisoformat(row[0]),
                    asset=row[1],
                    anomaly_type=row[2],
                    value=float(row[3]) if row[3] else 0,
                    details=json.loads(row[4]) if row[4] else {}
                ))
            except Exception as e:
                print(f"  ⚠️ 파싱 오류: {e}")

        print(f"\n📊 {len(anomalies)}개 이상 로드됨")

        # 클러스터링
        clusters = self.cluster_anomalies(anomalies)
        print(f"📦 {len(clusters)}개 클러스터 생성")

        # 각 클러스터에 대해 뉴스 검색
        attributions = []
        for cluster in clusters:
            attr = self.correlate_cluster(cluster)
            if attr:
                attributions.append(attr)

        return attributions

    def generate_report(self, attributions: list[EventAttribution]) -> str:
        """귀인 결과 리포트 생성"""
        if not attributions:
            return "귀인 결과 없음"

        lines = [
            "# 이상 탐지-뉴스 귀인 리포트",
            f"> 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "---",
            ""
        ]

        for i, attr in enumerate(attributions, 1):
            lines.extend([
                f"## {i}. {attr.cluster.cluster_id}",
                "",
                f"**시간**: {attr.cluster.start_time} ~ {attr.cluster.end_time} UTC",
                f"**자산**: {', '.join(attr.cluster.affected_assets)}",
                f"**심각도**: {attr.cluster.severity_score}",
                f"**신뢰도**: {attr.confidence_score}",
                "",
                "### 관련 뉴스",
                ""
            ])

            for j, news in enumerate(attr.news, 1):
                lines.append(f"{j}. [{news.language.upper()}] {news.headline}")

            lines.extend(["", "---", ""])

        return '\n'.join(lines)


# 주말용 추가 자산 수집기
class WeekendAssetCollector:
    """일요일 저녁부터 거래되는 선물/FX 수집"""

    WEEKEND_ASSETS = {
        'CL=F': 'WTI Crude Oil Futures',
        'GC=F': 'Gold Futures',
        'SI=F': 'Silver Futures',
        'DX-Y.NYB': 'US Dollar Index',
        'EURUSD=X': 'EUR/USD',
        'USDJPY=X': 'USD/JPY',
        'GBPUSD=X': 'GBP/USD',
    }

    # 이상 탐지 임계값
    THRESHOLDS = {
        'price_change_pct': 1.5,  # 1.5% 이상 변동
        'volume_ratio': 3.0,      # 평균 대비 3배 이상
    }

    def __init__(self):
        import yfinance as yf
        self.yf = yf

    def collect_and_detect(self) -> list[Anomaly]:
        """주말 자산 수집 및 이상 탐지"""
        anomalies = []

        for symbol, name in self.WEEKEND_ASSETS.items():
            try:
                ticker = self.yf.Ticker(symbol)

                # 최근 5일 데이터
                hist = ticker.history(period='5d', interval='1h')
                if hist.empty:
                    continue

                latest = hist.iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else latest['Close']

                # 가격 변화율
                pct_change = ((latest['Close'] - prev_close) / prev_close) * 100

                # 거래량 비율
                avg_volume = hist['Volume'].mean()
                volume_ratio = latest['Volume'] / avg_volume if avg_volume > 0 else 0

                # 이상 탐지
                if abs(pct_change) >= self.THRESHOLDS['price_change_pct']:
                    anomalies.append(Anomaly(
                        timestamp=datetime.utcnow(),
                        asset=symbol,
                        anomaly_type='price_shock',
                        value=pct_change,
                        details={'name': name, 'close': latest['Close']}
                    ))
                    print(f"  🚨 {symbol}: 가격 변동 {pct_change:+.2f}%")

                if volume_ratio >= self.THRESHOLDS['volume_ratio']:
                    anomalies.append(Anomaly(
                        timestamp=datetime.utcnow(),
                        asset=symbol,
                        anomaly_type='volume_explosion',
                        value=volume_ratio,
                        details={'name': name, 'volume': latest['Volume']}
                    ))
                    print(f"  🚨 {symbol}: 거래량 {volume_ratio:.1f}배")

            except Exception as e:
                print(f"  ⚠️ {symbol} 수집 실패: {e}")

        return anomalies


if __name__ == '__main__':
    print("=" * 60)
    print("News Correlator - 이상 탐지-뉴스 귀인 시스템")
    print("=" * 60)

    # 테스트: 샘플 이상 생성
    test_anomalies = [
        Anomaly(
            timestamp=datetime(2026, 1, 3, 6, 15),
            asset='BTC',
            anomaly_type='volume_explosion',
            value=9.2
        ),
        Anomaly(
            timestamp=datetime(2026, 1, 3, 6, 30),
            asset='ETH',
            anomaly_type='volume_explosion',
            value=12.5
        ),
        Anomaly(
            timestamp=datetime(2026, 1, 3, 7, 0),
            asset='BTC',
            anomaly_type='volatility_spike',
            value=6.4
        ),
    ]

    correlator = NewsCorrelator()

    # 클러스터링
    clusters = correlator.cluster_anomalies(test_anomalies)
    print(f"\n클러스터 {len(clusters)}개 생성")

    for cluster in clusters:
        print(f"\n클러스터: {cluster.cluster_id}")
        print(f"  시간: {cluster.start_time} ~ {cluster.end_time}")
        print(f"  자산: {cluster.affected_assets}")
        print(f"  심각도: {cluster.severity_score}")

        # 뉴스 검색 (API 키 있으면)
        if correlator.perplexity_client:
            attr = correlator.correlate_cluster(cluster)
            if attr:
                print(f"\n{attr.summary}")
