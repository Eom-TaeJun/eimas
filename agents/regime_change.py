"""
Regime Change Detection Module

거래량 급변 → 뉴스 검색 → 영향력 평가 → 레짐 변화 결정
ECON_AI_AGENT_SYSTEM.md Section 5.2 구현

Author: EIMAS Team
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# ============================================================================
# Data Classes
# ============================================================================

class NewsSentiment(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class NewsType(Enum):
    CONTRACT = "CONTRACT"        # 대형 계약 체결
    PARTNERSHIP = "PARTNERSHIP"  # 전략적 제휴
    MA = "MA"                    # 인수합병
    EARNINGS = "EARNINGS"        # 실적 발표
    PRODUCT = "PRODUCT"          # 신제품/기술
    REGULATORY = "REGULATORY"    # 규제 변화
    SCANDAL = "SCANDAL"          # 스캔들/소송
    MANAGEMENT = "MANAGEMENT"    # 경영진 변화
    MACRO = "MACRO"              # 거시경제 영향
    OTHER = "OTHER"


class ImpactDuration(Enum):
    TEMPORARY = "TEMPORARY"      # 일시적 (1-2주)
    MEDIUM_TERM = "MEDIUM_TERM"  # 중기적 (1-6개월)
    STRUCTURAL = "STRUCTURAL"    # 구조적 변화


@dataclass
class VolumeEvent:
    """거래량 급변 이벤트"""
    date: datetime
    ticker: str
    volume_zscore: float
    volume_multiple: float  # 평균 대비 배수
    price_change: float     # 당일 가격 변화율
    direction: str          # "UP" or "DOWN"

    def __str__(self) -> str:
        return f"VolumeEvent({self.ticker} on {self.date.strftime('%Y-%m-%d')}: {self.volume_multiple:.1f}x, {self.price_change:+.1%})"


@dataclass
class NewsItem:
    """뉴스 아이템"""
    headline: str
    source: str
    published_date: datetime
    content: str
    url: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class NewsSearchResult:
    """뉴스 검색 결과"""
    found: bool
    news_items: List[NewsItem] = field(default_factory=list)
    primary_news: Optional[NewsItem] = None
    search_date: Optional[datetime] = None
    message: str = ""


@dataclass
class NewsClassification:
    """뉴스 분류 결과"""
    sentiment: NewsSentiment
    sentiment_score: float  # -1.0 ~ +1.0
    news_type: NewsType
    duration: ImpactDuration
    is_regime_change: bool
    regime_change_reason: str
    confidence: float
    key_points: List[str]


@dataclass
class ImpactOpinion:
    """AI의 영향력 평가 의견"""
    agent_name: str
    valuation_impact: Dict[str, Any]
    growth_impact: Dict[str, Any]
    risk_impact: Dict[str, Any]
    price_impact: Dict[str, Any]
    similar_case_comparison: Dict[str, Any]
    reasoning: str


@dataclass
class SimilarCase:
    """과거 유사 사례"""
    company: str
    event_description: str
    event_date: datetime
    price_impact: float
    recovery_days: int
    applicability: float  # 현재 케이스와의 유사도


@dataclass
class ImpactAssessment:
    """영향력 평가 결과"""
    magnitude: float        # 예상 가격 영향 (%)
    duration: ImpactDuration
    valuation_impact: Dict[str, Any]
    confidence: float
    similar_cases: List[SimilarCase]
    debate_summary: str
    opinions: List[ImpactOpinion] = field(default_factory=list)


@dataclass
class RegimeDefinition:
    """레짐 정의"""
    name: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    characteristics: Dict[str, Any]
    valuation_metrics: Dict[str, float]


@dataclass
class RegimeChangeResult:
    """레짐 변화 결정 결과"""
    is_regime_change: bool
    change_date: Optional[datetime] = None
    before_regime: Optional[RegimeDefinition] = None
    after_regime: Optional[RegimeDefinition] = None
    analysis_instruction: str = ""
    reason: str = ""
    recommendation: str = ""


# ============================================================================
# Step 1: Volume Breakout Detector
# ============================================================================

class VolumeBreakoutDetector:
    """
    거래량 급변 탐지

    구조 변화 시 정보를 가진 투자자들이 먼저 움직임
    거래량 급증 = 새로운 정보가 시장에 반영 중
    3시그마 이상 = 통계적으로 유의미한 이벤트
    """

    def __init__(
        self,
        lookback: int = 60,
        threshold_sigma: float = 3.0,
        min_volume_multiple: float = 2.0
    ):
        self.lookback = lookback
        self.threshold_sigma = threshold_sigma
        self.min_volume_multiple = min_volume_multiple

    def detect(
        self,
        data: pd.DataFrame,
        ticker: str,
        lookback: Optional[int] = None,
        threshold_sigma: Optional[float] = None
    ) -> List[VolumeEvent]:
        """
        거래량이 평균 대비 N시그마 이상 급증한 날 탐지

        Parameters:
        -----------
        data : DataFrame
            'Volume', 'Close' 컬럼 필요. index는 datetime
        ticker : str
            종목 코드
        lookback : int
            이동평균 기간 (기본: 60일)
        threshold_sigma : float
            Z-score 임계값 (기본: 3.0)

        Returns:
        --------
        List[VolumeEvent]
            탐지된 거래량 급변 이벤트 리스트
        """
        lookback = lookback or self.lookback
        threshold_sigma = threshold_sigma or self.threshold_sigma

        if 'Volume' not in data.columns or 'Close' not in data.columns:
            raise ValueError("DataFrame must contain 'Volume' and 'Close' columns")

        volume = data['Volume'].astype(float)
        close = data['Close'].astype(float)

        # Rolling statistics
        rolling_mean = volume.rolling(lookback).mean()
        rolling_std = volume.rolling(lookback).std()

        # Z-score 계산
        z_score = (volume - rolling_mean) / rolling_std

        # Volume multiple
        volume_multiple = volume / rolling_mean

        # 급증 날짜 탐지 (Z-score와 배수 조건 모두 충족)
        mask = (z_score > threshold_sigma) & (volume_multiple > self.min_volume_multiple)
        breakout_indices = data.index[mask].tolist()

        events = []
        for idx in breakout_indices:
            # 전일 대비 가격 변화 계산
            idx_pos = data.index.get_loc(idx)
            if idx_pos > 0:
                prev_close = close.iloc[idx_pos - 1]
                curr_close = close.iloc[idx_pos]
                price_change = (curr_close / prev_close) - 1
            else:
                price_change = 0.0

            event = VolumeEvent(
                date=idx if isinstance(idx, datetime) else pd.to_datetime(idx),
                ticker=ticker,
                volume_zscore=float(z_score.loc[idx]),
                volume_multiple=float(volume_multiple.loc[idx]),
                price_change=float(price_change),
                direction="UP" if price_change > 0 else "DOWN"
            )
            events.append(event)

        return events

    def get_volume_stats(
        self,
        data: pd.DataFrame,
        window: int = 60
    ) -> pd.DataFrame:
        """거래량 통계 반환 (디버깅/시각화용)"""
        volume = data['Volume'].astype(float)

        stats = pd.DataFrame(index=data.index)
        stats['volume'] = volume
        stats['rolling_mean'] = volume.rolling(window).mean()
        stats['rolling_std'] = volume.rolling(window).std()
        stats['z_score'] = (volume - stats['rolling_mean']) / stats['rolling_std']
        stats['multiple'] = volume / stats['rolling_mean']

        return stats


# ============================================================================
# Step 2: News Search Agent (Perplexity)
# ============================================================================

class NewsSearchAgent:
    """
    Perplexity API를 활용한 뉴스 검색
    거래량 급변 날짜에 무슨 일이 있었는지 파악
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Parameters:
        -----------
        api_key : str
            Perplexity API key. None이면 환경변수에서 로드
        """
        import os
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Perplexity client"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.perplexity.ai"
            )
        return self._client

    async def search_news_for_event(
        self,
        ticker: str,
        company_name: str,
        event_date: datetime,
        days_range: int = 3
    ) -> NewsSearchResult:
        """
        특정 날짜 전후 관련 뉴스 검색

        검색 전략:
        1. 기업명 + 날짜로 직접 검색
        2. 티커 + 주요 키워드 (계약, 인수, 실적, 스캔들)
        3. 산업 전체 뉴스 (섹터 이벤트일 수 있음)
        """
        start_date = event_date - timedelta(days=days_range)
        end_date = event_date + timedelta(days=days_range)
        date_str = event_date.strftime('%Y-%m-%d')
        month_str = event_date.strftime('%B %Y')

        # 검색 쿼리 구성
        search_query = f"""
        Find news about {company_name} ({ticker}) around {date_str}.

        Focus on:
        1. Major announcements, contracts, partnerships
        2. Earnings reports or guidance
        3. Management changes
        4. Regulatory news
        5. Any significant events that could explain unusual trading volume

        Time range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}

        Provide specific news headlines and brief summaries.
        """

        try:
            # Perplexity API 호출 (synchronous wrapper for async)
            response = await asyncio.to_thread(
                self._call_perplexity,
                search_query
            )

            # 응답 파싱
            news_items = self._parse_news_response(response, company_name)

            if not news_items:
                return NewsSearchResult(
                    found=False,
                    message="관련 뉴스 없음 - 거래량 급증이 노이즈일 가능성"
                )

            return NewsSearchResult(
                found=True,
                news_items=news_items,
                primary_news=news_items[0] if news_items else None,
                search_date=event_date
            )

        except Exception as e:
            return NewsSearchResult(
                found=False,
                message=f"뉴스 검색 실패: {str(e)}"
            )

    def _call_perplexity(self, query: str) -> str:
        """Perplexity API 호출"""
        response = self.client.chat.completions.create(
            model="llama-3.1-sonar-small-128k-online",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial news researcher. Find relevant news articles and provide structured summaries."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content

    def _parse_news_response(
        self,
        response: str,
        company_name: str
    ) -> List[NewsItem]:
        """Perplexity 응답을 NewsItem 리스트로 파싱"""
        # 간단한 파싱 로직 - 실제로는 더 정교한 파싱 필요
        news_items = []

        # 응답을 줄 단위로 분석
        lines = response.split('\n')
        current_headline = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_headline:
                    news_items.append(NewsItem(
                        headline=current_headline,
                        source="Perplexity Search",
                        published_date=datetime.now(),
                        content=' '.join(current_content),
                        relevance_score=self._calculate_relevance(
                            current_headline, company_name
                        )
                    ))
                    current_headline = None
                    current_content = []
                continue

            # 헤드라인 감지 (숫자로 시작하거나 대문자로 시작)
            if line[0].isdigit() or (line[0].isupper() and ':' in line[:50]):
                if current_headline:
                    news_items.append(NewsItem(
                        headline=current_headline,
                        source="Perplexity Search",
                        published_date=datetime.now(),
                        content=' '.join(current_content),
                        relevance_score=self._calculate_relevance(
                            current_headline, company_name
                        )
                    ))
                current_headline = line.lstrip('0123456789. ')
                current_content = []
            else:
                current_content.append(line)

        # 마지막 아이템 추가
        if current_headline:
            news_items.append(NewsItem(
                headline=current_headline,
                source="Perplexity Search",
                published_date=datetime.now(),
                content=' '.join(current_content),
                relevance_score=self._calculate_relevance(
                    current_headline, company_name
                )
            ))

        # 관련성 순 정렬
        news_items.sort(key=lambda x: x.relevance_score, reverse=True)

        return news_items[:5]  # 상위 5개만 반환

    def _calculate_relevance(self, text: str, company_name: str) -> float:
        """텍스트와 기업명의 관련성 점수 계산"""
        text_lower = text.lower()
        company_lower = company_name.lower()

        score = 0.0

        # 기업명 포함 여부
        if company_lower in text_lower:
            score += 0.5

        # 핵심 키워드 포함 여부
        keywords = ['contract', 'partnership', 'acquisition', 'earnings',
                   'deal', 'agreement', 'revenue', 'profit', 'announce']
        for keyword in keywords:
            if keyword in text_lower:
                score += 0.1

        return min(score, 1.0)

    async def search_similar_cases(
        self,
        news_type: NewsType,
        industry: str,
        market_cap: float
    ) -> List[SimilarCase]:
        """과거 유사 사례 검색"""
        query = f"""
        Find historical cases similar to:
        - Event type: {news_type.value}
        - Industry: {industry}
        - Company size: market cap around ${market_cap/1e9:.1f}B

        Provide:
        1. Company name
        2. Brief event description
        3. Date of event
        4. Stock price impact (%)
        5. Recovery time (days)

        List 3-5 relevant cases.
        """

        try:
            response = await asyncio.to_thread(
                self._call_perplexity,
                query
            )
            return self._parse_similar_cases(response)
        except Exception:
            return []

    def _parse_similar_cases(self, response: str) -> List[SimilarCase]:
        """유사 사례 응답 파싱"""
        # 간단한 파싱 - 실제로는 더 정교하게 구현
        cases = []
        # TODO: 정교한 파싱 로직 구현
        return cases


# ============================================================================
# Step 3: News Classification Agent (Claude)
# ============================================================================

class NewsClassificationAgent:
    """
    Claude를 활용한 뉴스 분류 및 성격 판단
    좋은 뉴스인지, 나쁜 뉴스인지, 얼마나 중요한지 판단
    """

    CLASSIFICATION_PROMPT = """
당신은 금융 뉴스 분석 전문가입니다.

## 분석 대상
기업: {company_name} ({ticker})
날짜: {event_date}
거래량 변화: 평균 대비 {volume_multiple:.1f}배 (Z-score: {volume_zscore:.1f})
가격 변화: {price_change:+.1%}

## 관련 뉴스
{news_content}

## 분석 요청

1. **뉴스 성격 판단**
   - POSITIVE (호재) / NEGATIVE (악재) / NEUTRAL (중립)
   - 판단 근거

2. **뉴스 유형 분류**
   - CONTRACT: 대형 계약 체결
   - PARTNERSHIP: 전략적 제휴
   - MA: 인수합병
   - EARNINGS: 실적 발표
   - PRODUCT: 신제품/기술
   - REGULATORY: 규제 변화
   - SCANDAL: 스캔들/소송
   - MANAGEMENT: 경영진 변화
   - MACRO: 거시경제 영향
   - OTHER: 기타

3. **영향 지속성 판단**
   - TEMPORARY: 일시적 이벤트 (1-2주 내 소멸)
   - MEDIUM_TERM: 중기적 영향 (1-6개월)
   - STRUCTURAL: 구조적 변화 (기업 가치 재평가 필요)

4. **구조적 변화 여부**
   - 이 이벤트로 인해 기업의 "본질적 가치 구조"가 변하는가?
   - 예: 삼성전자-테슬라 계약 → 메모리 기업에서 EV 생태계 기업으로

## 출력 형식 (JSON)
```json
{{
    "sentiment": "POSITIVE | NEGATIVE | NEUTRAL",
    "sentiment_score": 0.0,
    "news_type": "CONTRACT | PARTNERSHIP | ...",
    "duration": "TEMPORARY | MEDIUM_TERM | STRUCTURAL",
    "is_regime_change": false,
    "regime_change_reason": "...",
    "confidence": 0.0,
    "key_points": ["핵심 포인트 1", "핵심 포인트 2"]
}}
```
"""

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    async def classify_news(
        self,
        news_result: NewsSearchResult,
        volume_event: VolumeEvent,
        company_info: Dict[str, Any]
    ) -> NewsClassification:
        """뉴스를 분류하고 성격 판단"""

        # 뉴스 내용 포맷팅
        news_content = self._format_news(news_result.news_items)

        prompt = self.CLASSIFICATION_PROMPT.format(
            company_name=company_info.get('name', 'Unknown'),
            ticker=company_info.get('ticker', 'Unknown'),
            event_date=volume_event.date.strftime('%Y-%m-%d'),
            volume_multiple=volume_event.volume_multiple,
            volume_zscore=volume_event.volume_zscore,
            price_change=volume_event.price_change,
            news_content=news_content
        )

        try:
            response = await asyncio.to_thread(
                self._call_claude,
                prompt
            )
            return self._parse_classification(response)
        except Exception as e:
            # 기본값 반환
            return NewsClassification(
                sentiment=NewsSentiment.NEUTRAL,
                sentiment_score=0.0,
                news_type=NewsType.OTHER,
                duration=ImpactDuration.TEMPORARY,
                is_regime_change=False,
                regime_change_reason=f"분류 실패: {str(e)}",
                confidence=0.0,
                key_points=[]
            )

    def _call_claude(self, prompt: str) -> str:
        """Claude API 호출"""
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text

    def _format_news(self, news_items: List[NewsItem]) -> str:
        """뉴스 아이템을 문자열로 포맷"""
        if not news_items:
            return "관련 뉴스 없음"

        formatted = []
        for i, news in enumerate(news_items, 1):
            formatted.append(f"{i}. **{news.headline}**")
            formatted.append(f"   {news.content[:500]}...")
            formatted.append("")

        return '\n'.join(formatted)

    def _parse_classification(self, response: str) -> NewsClassification:
        """Claude 응답을 NewsClassification으로 파싱"""
        import json
        import re

        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 블록 없으면 전체 응답에서 찾기
            json_str = response

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본값
            return NewsClassification(
                sentiment=NewsSentiment.NEUTRAL,
                sentiment_score=0.0,
                news_type=NewsType.OTHER,
                duration=ImpactDuration.TEMPORARY,
                is_regime_change=False,
                regime_change_reason="JSON 파싱 실패",
                confidence=0.0,
                key_points=[]
            )

        return NewsClassification(
            sentiment=NewsSentiment[data.get('sentiment', 'NEUTRAL')],
            sentiment_score=float(data.get('sentiment_score', 0.0)),
            news_type=NewsType[data.get('news_type', 'OTHER')],
            duration=ImpactDuration[data.get('duration', 'TEMPORARY')],
            is_regime_change=bool(data.get('is_regime_change', False)),
            regime_change_reason=str(data.get('regime_change_reason', '')),
            confidence=float(data.get('confidence', 0.0)),
            key_points=list(data.get('key_points', []))
        )


# ============================================================================
# Step 4: Impact Assessment Debate (Multi-AI)
# ============================================================================

class ImpactAssessmentDebate:
    """
    여러 AI가 토론하여 영향력 크기 추정

    판단 기준:
    1. 거래량 변화 폭 (클수록 시장 반응 강함)
    2. 과거 유사 사례 (역사적 선례)
    3. 경제학 이론 (밸류에이션, 성장률 영향)
    """

    ASSESSMENT_PROMPT = """
## 이벤트 분석
기업: {company_name}
뉴스 유형: {news_type}
뉴스 성격: {sentiment} ({sentiment_score:+.1f})
거래량: 평균 대비 {volume_multiple:.1f}배

## 과거 유사 사례
{similar_cases}

## 경제학적 분석 요청

1. **밸류에이션 영향**
   - 이 이벤트가 기업 가치에 미치는 영향은?
   - P/E, P/S 배수 변화 예상?

2. **성장률 영향**
   - 매출/이익 성장률에 미치는 영향?
   - 일시적 vs 지속적?

3. **리스크 프리미엄 영향**
   - 기업 리스크가 증가/감소하는가?
   - 할인율 변화?

4. **과거 사례와 비교**
   - 유사 사례에서 주가는 어떻게 움직였는가?
   - 이번 케이스와의 차이점은?

5. **적정 주가 영향 추정**
   - 현재가 대비 +X% ~ +Y% 또는 -X% ~ -Y%

## 출력 (JSON)
```json
{{
    "valuation_impact": {{
        "pe_change": "+2x ~ +4x",
        "reason": "리레이팅 사유"
    }},
    "growth_impact": {{
        "revenue_boost": "+5% ~ +10% annually",
        "duration": "3-5 years"
    }},
    "risk_impact": {{
        "direction": "DECREASE or INCREASE",
        "discount_rate_change": "-0.5%"
    }},
    "price_impact": {{
        "range": "+15% ~ +25%",
        "base_case": 0.20,
        "confidence": 0.7
    }},
    "similar_case_comparison": {{
        "case": "유사 사례명",
        "outcome": "+35% in 3 months",
        "applicability": 0.8
    }},
    "reasoning": "전체 분석 근거"
}}
```
"""

    def __init__(
        self,
        claude_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None
    ):
        import os
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GOOGLE_API_KEY")

        self.perplexity_agent = NewsSearchAgent()

    async def assess_impact(
        self,
        volume_event: VolumeEvent,
        news_classification: NewsClassification,
        company_info: Dict[str, Any]
    ) -> ImpactAssessment:
        """Multi-AI 토론으로 영향력 추정"""

        # 1. 과거 유사 사례 검색 (Perplexity)
        similar_cases = await self.perplexity_agent.search_similar_cases(
            news_type=news_classification.news_type,
            industry=company_info.get('industry', 'Unknown'),
            market_cap=company_info.get('market_cap', 0)
        )

        # 2. 컨텍스트 구성
        context = {
            "volume_event": volume_event,
            "news": news_classification,
            "company": company_info,
            "similar_cases": similar_cases
        }

        # 3. 각 AI가 영향력 추정 (병렬 실행)
        opinions = await asyncio.gather(
            self._claude_assess(context),
            self._openai_assess(context),
            self._gemini_assess(context),
            return_exceptions=True
        )

        # 예외 필터링
        valid_opinions = [
            op for op in opinions
            if isinstance(op, ImpactOpinion)
        ]

        if not valid_opinions:
            # 모든 AI 실패 시 기본값
            return ImpactAssessment(
                magnitude=0.0,
                duration=news_classification.duration,
                valuation_impact={},
                confidence=0.0,
                similar_cases=similar_cases,
                debate_summary="모든 AI 평가 실패",
                opinions=[]
            )

        # 4. 토론 및 합의
        debate_result = await self._run_debate(valid_opinions)

        return ImpactAssessment(
            magnitude=debate_result['consensus_magnitude'],
            duration=debate_result['consensus_duration'],
            valuation_impact=debate_result['valuation_change'],
            confidence=debate_result['confidence'],
            similar_cases=similar_cases,
            debate_summary=debate_result['summary'],
            opinions=valid_opinions
        )

    async def _claude_assess(self, context: Dict) -> ImpactOpinion:
        """Claude의 영향력 평가 (경제학 이론 중심)"""
        from anthropic import Anthropic

        prompt = self.ASSESSMENT_PROMPT.format(
            company_name=context['company'].get('name', 'Unknown'),
            news_type=context['news'].news_type.value,
            sentiment=context['news'].sentiment.value,
            sentiment_score=context['news'].sentiment_score,
            volume_multiple=context['volume_event'].volume_multiple,
            similar_cases=self._format_similar_cases(context['similar_cases'])
        )

        client = Anthropic(api_key=self.claude_api_key)
        response = await asyncio.to_thread(
            lambda: client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
        )

        return self._parse_opinion(response.content[0].text, "Claude")

    async def _openai_assess(self, context: Dict) -> ImpactOpinion:
        """OpenAI의 영향력 평가"""
        from openai import OpenAI

        prompt = self.ASSESSMENT_PROMPT.format(
            company_name=context['company'].get('name', 'Unknown'),
            news_type=context['news'].news_type.value,
            sentiment=context['news'].sentiment.value,
            sentiment_score=context['news'].sentiment_score,
            volume_multiple=context['volume_event'].volume_multiple,
            similar_cases=self._format_similar_cases(context['similar_cases'])
        )

        client = OpenAI(api_key=self.openai_api_key)
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
        )

        return self._parse_opinion(response.choices[0].message.content, "OpenAI")

    async def _gemini_assess(self, context: Dict) -> ImpactOpinion:
        """Gemini의 영향력 평가"""
        import google.generativeai as genai

        prompt = self.ASSESSMENT_PROMPT.format(
            company_name=context['company'].get('name', 'Unknown'),
            news_type=context['news'].news_type.value,
            sentiment=context['news'].sentiment.value,
            sentiment_score=context['news'].sentiment_score,
            volume_multiple=context['volume_event'].volume_multiple,
            similar_cases=self._format_similar_cases(context['similar_cases'])
        )

        genai.configure(api_key=self.gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')

        response = await asyncio.to_thread(
            lambda: model.generate_content(prompt)
        )

        return self._parse_opinion(response.text, "Gemini")

    def _format_similar_cases(self, cases: List[SimilarCase]) -> str:
        """유사 사례를 문자열로 포맷"""
        if not cases:
            return "유사 사례 없음"

        formatted = []
        for case in cases:
            formatted.append(
                f"- {case.company}: {case.event_description} "
                f"({case.event_date.strftime('%Y-%m')}) "
                f"→ {case.price_impact:+.1%}"
            )
        return '\n'.join(formatted)

    def _parse_opinion(self, response: str, agent_name: str) -> ImpactOpinion:
        """AI 응답을 ImpactOpinion으로 파싱"""
        import json
        import re

        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # 파싱 실패 시 기본값
            data = {
                "price_impact": {"base_case": 0.0, "confidence": 0.0},
                "valuation_impact": {},
                "growth_impact": {},
                "risk_impact": {},
                "similar_case_comparison": {},
                "reasoning": response[:500]
            }

        return ImpactOpinion(
            agent_name=agent_name,
            valuation_impact=data.get('valuation_impact', {}),
            growth_impact=data.get('growth_impact', {}),
            risk_impact=data.get('risk_impact', {}),
            price_impact=data.get('price_impact', {}),
            similar_case_comparison=data.get('similar_case_comparison', {}),
            reasoning=data.get('reasoning', '')
        )

    async def _run_debate(
        self,
        opinions: List[ImpactOpinion]
    ) -> Dict[str, Any]:
        """AI들의 의견을 종합하여 합의 도출"""

        # 각 AI의 base_case 추출
        magnitudes = []
        for op in opinions:
            base_case = op.price_impact.get('base_case', 0)
            if isinstance(base_case, str):
                # "+20%" 같은 문자열 처리
                try:
                    base_case = float(base_case.strip('%+')) / 100
                except ValueError:
                    base_case = 0.0
            magnitudes.append(float(base_case))

        if not magnitudes:
            return {
                'consensus_magnitude': 0.0,
                'consensus_duration': ImpactDuration.TEMPORARY,
                'valuation_change': {},
                'confidence': 0.0,
                'summary': "의견 없음"
            }

        avg_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)

        # 불일치 체크 (10%p 이상)
        if std_magnitude > 0.10 and len(opinions) > 1:
            # 추가 토론 필요 - 중간값 사용
            consensus = np.median(magnitudes)
        else:
            consensus = avg_magnitude

        # 신뢰도 계산
        if avg_magnitude != 0:
            confidence = max(0, 1 - (std_magnitude / abs(avg_magnitude)))
        else:
            confidence = 0.5

        # Duration 집계
        durations = [op.risk_impact.get('duration', 'MEDIUM_TERM') for op in opinions]
        if 'STRUCTURAL' in str(durations):
            consensus_duration = ImpactDuration.STRUCTURAL
        elif 'MEDIUM_TERM' in str(durations):
            consensus_duration = ImpactDuration.MEDIUM_TERM
        else:
            consensus_duration = ImpactDuration.TEMPORARY

        # 요약 생성
        summary_parts = []
        for op in opinions:
            base = op.price_impact.get('base_case', 'N/A')
            summary_parts.append(f"{op.agent_name}: {base}")

        return {
            'consensus_magnitude': float(consensus),
            'consensus_duration': consensus_duration,
            'valuation_change': opinions[0].valuation_impact if opinions else {},
            'confidence': float(confidence),
            'summary': f"합의: {consensus:+.1%} | " + ' | '.join(summary_parts)
        }


# ============================================================================
# Step 5: Regime Change Decision
# ============================================================================

class RegimeChangeDecision:
    """최종 레짐 변화 결정 및 데이터 분리"""

    def __init__(
        self,
        min_confidence: float = 0.6,
        min_magnitude: float = 0.15  # 15% 이상
    ):
        self.min_confidence = min_confidence
        self.min_magnitude = min_magnitude

    async def decide_regime_change(
        self,
        volume_event: VolumeEvent,
        news_classification: NewsClassification,
        impact_assessment: ImpactAssessment
    ) -> RegimeChangeResult:
        """구조적 변화 여부 최종 결정"""

        # 구조적 변화 조건
        is_structural = (
            news_classification.is_regime_change and
            news_classification.duration == ImpactDuration.STRUCTURAL and
            impact_assessment.confidence > self.min_confidence and
            abs(impact_assessment.magnitude) > self.min_magnitude
        )

        if is_structural:
            before_regime = self._define_before_regime(volume_event.ticker)
            after_regime = self._define_after_regime(
                volume_event.ticker,
                news_classification,
                impact_assessment
            )

            analysis_instruction = f"""
[레짐 변화 확정]
날짜: {volume_event.date.strftime('%Y-%m-%d')}
이유: {news_classification.regime_change_reason}
영향: {impact_assessment.magnitude:+.1%}

분석 지침:
1. {volume_event.date.strftime('%Y-%m-%d')} 이전 데이터는 과거 레짐으로 분류
2. 이후 데이터만 현재 분석에 사용
3. 밸류에이션 기준: {impact_assessment.valuation_impact}
4. 새로운 핵심 드라이버 반영
"""

            return RegimeChangeResult(
                is_regime_change=True,
                change_date=volume_event.date,
                before_regime=before_regime,
                after_regime=after_regime,
                analysis_instruction=analysis_instruction
            )
        else:
            reasons = []
            if not news_classification.is_regime_change:
                reasons.append("뉴스 분류: 구조적 변화 아님")
            if news_classification.duration != ImpactDuration.STRUCTURAL:
                reasons.append(f"지속성: {news_classification.duration.value}")
            if impact_assessment.confidence <= self.min_confidence:
                reasons.append(f"신뢰도 부족: {impact_assessment.confidence:.0%}")
            if abs(impact_assessment.magnitude) <= self.min_magnitude:
                reasons.append(f"영향 미미: {impact_assessment.magnitude:+.1%}")

            return RegimeChangeResult(
                is_regime_change=False,
                reason="일시적 이벤트로 판단 - " + ', '.join(reasons),
                recommendation="전체 데이터 계속 사용"
            )

    def _define_before_regime(self, ticker: str) -> RegimeDefinition:
        """이전 레짐 정의"""
        return RegimeDefinition(
            name=f"{ticker}_before",
            start_date=None,
            end_date=None,
            characteristics={
                "description": "구조 변화 이전 상태",
                "data_usage": "참고용으로만 사용"
            },
            valuation_metrics={}
        )

    def _define_after_regime(
        self,
        ticker: str,
        news: NewsClassification,
        impact: ImpactAssessment
    ) -> RegimeDefinition:
        """새로운 레짐 정의"""
        return RegimeDefinition(
            name=f"{ticker}_after_{news.news_type.value}",
            start_date=None,
            end_date=None,
            characteristics={
                "trigger": news.news_type.value,
                "sentiment": news.sentiment.value,
                "key_change": news.regime_change_reason,
                "expected_duration": news.duration.value
            },
            valuation_metrics={
                "expected_impact": impact.magnitude,
                "confidence": impact.confidence
            }
        )


# ============================================================================
# Integrated Pipeline
# ============================================================================

class RegimeChangeDetectionPipeline:
    """
    거래량 급변 → 뉴스 검색 → 영향력 평가 → 레짐 변화 결정
    전체 파이프라인 통합
    """

    def __init__(self, verbose: bool = True):
        self.volume_detector = VolumeBreakoutDetector()
        self.news_agent = NewsSearchAgent()
        self.classification_agent = NewsClassificationAgent()
        self.impact_debate = ImpactAssessmentDebate()
        self.regime_decision = RegimeChangeDecision()
        self.verbose = verbose

    def log(self, message: str):
        """로깅 (verbose 모드일 때만)"""
        if self.verbose:
            print(message)

    async def run(
        self,
        ticker: str,
        data: pd.DataFrame,
        company_info: Dict[str, Any]
    ) -> List[RegimeChangeResult]:
        """전체 파이프라인 실행"""
        results = []

        # Step 1: 거래량 급변 탐지
        volume_events = self.volume_detector.detect(data, ticker)
        self.log(f"[Step 1] {len(volume_events)}개 거래량 급변 탐지됨")

        for event in volume_events:
            self.log(f"\n[분석 중] {event.date.strftime('%Y-%m-%d')} - "
                    f"거래량 {event.volume_multiple:.1f}x, "
                    f"가격 {event.price_change:+.1%}")

            # Step 2: 뉴스 검색
            self.log("  [Step 2] 뉴스 검색 중...")
            news_result = await self.news_agent.search_news_for_event(
                ticker=ticker,
                company_name=company_info.get('name', ticker),
                event_date=event.date
            )

            if not news_result.found:
                self.log(f"  → 관련 뉴스 없음 - 노이즈로 판단")
                continue

            self.log(f"  → 뉴스 발견: {news_result.primary_news.headline[:50]}...")

            # Step 3: 뉴스 분류
            self.log("  [Step 3] 뉴스 분류 중...")
            classification = await self.classification_agent.classify_news(
                news_result=news_result,
                volume_event=event,
                company_info=company_info
            )

            self.log(f"  → 분류: {classification.news_type.value}, "
                    f"{classification.sentiment.value}")
            self.log(f"  → 지속성: {classification.duration.value}")

            if classification.duration == ImpactDuration.TEMPORARY:
                self.log(f"  → 일시적 이벤트 - 레짐 변화 아님")
                continue

            # Step 4: 영향력 평가 (Multi-AI Debate)
            self.log("  [Step 4] 영향력 평가 중 (AI 토론)...")
            impact = await self.impact_debate.assess_impact(
                volume_event=event,
                news_classification=classification,
                company_info=company_info
            )

            self.log(f"  → 예상 영향: {impact.magnitude:+.1%} "
                    f"(신뢰도: {impact.confidence:.0%})")
            self.log(f"  → {impact.debate_summary}")

            # Step 5: 레짐 변화 결정
            self.log("  [Step 5] 레짐 변화 결정 중...")
            regime_result = await self.regime_decision.decide_regime_change(
                volume_event=event,
                news_classification=classification,
                impact_assessment=impact
            )

            if regime_result.is_regime_change:
                self.log(f"  ✅ 레짐 변화 확정!")
                self.log(f"     Before: {regime_result.before_regime.name}")
                self.log(f"     After: {regime_result.after_regime.name}")
            else:
                self.log(f"  ❌ 레짐 변화 아님: {regime_result.reason}")

            results.append(regime_result)

        return results

    def run_sync(
        self,
        ticker: str,
        data: pd.DataFrame,
        company_info: Dict[str, Any]
    ) -> List[RegimeChangeResult]:
        """동기 실행 (asyncio.run 래퍼)"""
        return asyncio.run(self.run(ticker, data, company_info))


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    import yfinance as yf

    async def demo():
        print("=" * 60)
        print("Regime Change Detection Pipeline Demo")
        print("=" * 60)

        # 1. 데이터 준비
        ticker = "005930.KS"  # 삼성전자
        company_info = {
            "name": "Samsung Electronics",
            "ticker": ticker,
            "industry": "Semiconductors",
            "market_cap": 400e9  # $400B
        }

        print(f"\n[데이터 로드] {ticker}")
        data = yf.download(ticker, period="1y", progress=False)
        print(f"  → {len(data)} 거래일 로드됨")

        # 2. 파이프라인 실행
        pipeline = RegimeChangeDetectionPipeline(verbose=True)

        # 거래량 탐지만 테스트
        print("\n[Step 1 테스트] 거래량 급변 탐지")
        events = pipeline.volume_detector.detect(data, ticker)
        print(f"  → {len(events)}개 이벤트 탐지됨")

        for event in events[:3]:  # 상위 3개만 출력
            print(f"     {event}")

        # 전체 파이프라인 실행 (API 키 필요)
        # results = await pipeline.run(ticker, data, company_info)

        print("\n" + "=" * 60)
        print("Demo 완료 (전체 파이프라인은 API 키 필요)")
        print("=" * 60)

    asyncio.run(demo())
