#!/usr/bin/env python3
"""
EIMAS Sentiment & Options Analyzer
===================================
센티먼트 및 옵션 시장 분석

주요 기능:
1. Fear & Greed Index 수집
2. 뉴스 센티먼트 (제목 기반)
3. 소셜 미디어 트렌드 추정
4. VIX Term Structure 분석 (NEW)
5. Put/Call Ratio 분석 (NEW)
6. IV Percentile 계산 (NEW)
7. 통합 센티먼트 점수

Usage:
    from lib.sentiment_analyzer import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    result = analyzer.analyze()
    print(result)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import re
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

# 센티먼트 키워드 (간단한 사전 기반)
BULLISH_KEYWORDS = [
    'rally', 'surge', 'soar', 'jump', 'gain', 'rise', 'up', 'bull',
    'growth', 'beat', 'outperform', 'record', 'high', 'positive',
    'optimistic', 'strong', 'boost', 'upgrade', 'buy', 'bullish',
]

BEARISH_KEYWORDS = [
    'crash', 'plunge', 'fall', 'drop', 'decline', 'down', 'bear',
    'loss', 'miss', 'underperform', 'low', 'negative', 'pessimistic',
    'weak', 'cut', 'downgrade', 'sell', 'bearish', 'fear', 'recession',
]

# Fear & Greed 레벨
FEAR_GREED_LEVELS = {
    (0, 25): 'extreme_fear',
    (25, 45): 'fear',
    (45, 55): 'neutral',
    (55, 75): 'greed',
    (75, 100): 'extreme_greed',
}


class VIXStructure(str, Enum):
    """VIX 기간구조"""
    CONTANGO = "contango"           # 정상 (미래 VIX > 현재)
    BACKWARDATION = "backwardation"  # 역전 (현재 VIX > 미래) - 공포
    FLAT = "flat"


# ============================================================================
# Data Classes
# ============================================================================

class SentimentLevel(str, Enum):
    """센티먼트 레벨"""
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class FearGreedData:
    """Fear & Greed 데이터"""
    value: int
    level: SentimentLevel
    previous_close: int
    week_ago: int
    month_ago: int
    year_ago: int
    timestamp: datetime


@dataclass
class NewsSentiment:
    """뉴스 센티먼트"""
    headline: str
    source: str
    sentiment_score: float  # -1 to 1
    sentiment: str  # bullish, bearish, neutral
    keywords: List[str]
    timestamp: datetime


@dataclass
class SocialSentiment:
    """소셜 센티먼트"""
    platform: str  # reddit, twitter, stocktwits
    ticker: str
    mentions: int
    sentiment_score: float
    bullish_pct: float
    bearish_pct: float
    trending: bool


@dataclass
class VIXTermStructureResult:
    """VIX 기간구조 분석 결과"""
    vix_spot: float
    vix_3m: float
    structure: VIXStructure
    contango_ratio: float   # VIX3M / VIX - 1 (양수면 contango)
    percentile: float       # VIX 백분위 (과거 1년)
    signal: str
    interpretation: str


@dataclass
class PutCallRatioResult:
    """Put/Call Ratio 분석 결과"""
    ticker: str
    expiry_date: str
    put_volume: int
    call_volume: int
    put_call_ratio: float
    put_oi: int
    call_oi: int
    put_call_oi_ratio: float
    signal: str              # "BULLISH", "BEARISH", "NEUTRAL"
    interpretation: str


@dataclass
class IVPercentileResult:
    """IV Percentile 결과"""
    ticker: str
    current_iv: float
    iv_percentile: float     # 0-100
    hv_20: float             # 20일 Historical Volatility
    iv_hv_ratio: float       # IV / HV
    signal: str
    interpretation: str


@dataclass
class CompositeSentiment:
    """통합 센티먼트"""
    score: float  # -100 to 100
    level: SentimentLevel
    fear_greed: Optional[FearGreedData]
    news_sentiment: float
    social_sentiment: float
    contrarian_signal: str  # buy, sell, hold
    confidence: float


@dataclass
class SentimentAnalysisResult:
    """분석 결과"""
    timestamp: datetime
    composite: CompositeSentiment
    news_items: List[NewsSentiment]
    social_data: List[SocialSentiment]
    market_context: Dict[str, Any]
    signals: List[str]
    summary: str


# ============================================================================
# Sentiment Analyzer
# ============================================================================

class SentimentAnalyzer:
    """센티먼트 분석기"""

    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers or ['SPY', 'QQQ']
        self.fear_greed: Optional[FearGreedData] = None

    def fetch_fear_greed_index(self) -> Optional[FearGreedData]:
        """Fear & Greed Index 수집"""
        try:
            # CNN Fear & Greed API (비공식)
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()

                current = data.get('fear_and_greed', {})
                value = int(current.get('score', 50))

                # 레벨 판단
                level = SentimentLevel.NEUTRAL
                for (low, high), lv in FEAR_GREED_LEVELS.items():
                    if low <= value < high:
                        level = SentimentLevel(lv)
                        break

                self.fear_greed = FearGreedData(
                    value=value,
                    level=level,
                    previous_close=int(current.get('previous_close', value)),
                    week_ago=int(data.get('fear_and_greed_historical', {}).get('one_week_ago', value)),
                    month_ago=int(data.get('fear_and_greed_historical', {}).get('one_month_ago', value)),
                    year_ago=int(data.get('fear_and_greed_historical', {}).get('one_year_ago', value)),
                    timestamp=datetime.now(),
                )
                return self.fear_greed

        except Exception as e:
            print(f"Fear & Greed fetch error: {e}")

        # 기본값
        self.fear_greed = FearGreedData(
            value=50,
            level=SentimentLevel.NEUTRAL,
            previous_close=50,
            week_ago=50,
            month_ago=50,
            year_ago=50,
            timestamp=datetime.now(),
        )
        return self.fear_greed

    def analyze_text_sentiment(self, text: str) -> Tuple[float, str, List[str]]:
        """텍스트 센티먼트 분석 (키워드 기반)"""
        text_lower = text.lower()

        bullish_count = 0
        bearish_count = 0
        found_keywords = []

        for word in BULLISH_KEYWORDS:
            if word in text_lower:
                bullish_count += 1
                found_keywords.append(f"+{word}")

        for word in BEARISH_KEYWORDS:
            if word in text_lower:
                bearish_count += 1
                found_keywords.append(f"-{word}")

        total = bullish_count + bearish_count
        if total == 0:
            return 0.0, 'neutral', []

        score = (bullish_count - bearish_count) / total
        sentiment = 'bullish' if score > 0.2 else 'bearish' if score < -0.2 else 'neutral'

        return score, sentiment, found_keywords

    def fetch_news_sentiment(self, ticker: str = 'SPY') -> List[NewsSentiment]:
        """뉴스 센티먼트 수집 (yfinance 뉴스)"""
        import yfinance as yf

        news_items = []

        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            for item in news[:10]:
                title = item.get('title', '')
                score, sentiment, keywords = self.analyze_text_sentiment(title)

                news_items.append(NewsSentiment(
                    headline=title,
                    source=item.get('publisher', 'Unknown'),
                    sentiment_score=score,
                    sentiment=sentiment,
                    keywords=keywords,
                    timestamp=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                ))

        except Exception as e:
            print(f"News fetch error: {e}")

        return news_items

    def estimate_social_sentiment(self, ticker: str) -> SocialSentiment:
        """소셜 센티먼트 추정 (시뮬레이션)"""
        # 실제로는 Reddit/Twitter API 사용
        # 여기서는 Fear & Greed와 VIX 기반으로 추정

        fg_score = self.fear_greed.value if self.fear_greed else 50

        # Fear & Greed를 -1~1 스케일로 변환
        normalized = (fg_score - 50) / 50

        # 약간의 랜덤 노이즈 추가
        noise = np.random.uniform(-0.1, 0.1)
        sentiment_score = max(-1, min(1, normalized + noise))

        bullish_pct = (sentiment_score + 1) / 2 * 100
        bearish_pct = 100 - bullish_pct

        return SocialSentiment(
            platform='estimated',
            ticker=ticker,
            mentions=int(abs(sentiment_score) * 1000 + 500),  # 추정 언급 수
            sentiment_score=sentiment_score,
            bullish_pct=bullish_pct,
            bearish_pct=bearish_pct,
            trending=abs(sentiment_score) > 0.3,
        )

    def calculate_composite_sentiment(
        self,
        fear_greed: Optional[FearGreedData],
        news: List[NewsSentiment],
        social: List[SocialSentiment],
    ) -> CompositeSentiment:
        """통합 센티먼트 계산"""
        # Fear & Greed (40% 가중치)
        fg_score = fear_greed.value if fear_greed else 50
        fg_normalized = (fg_score - 50) * 2  # -100 to 100

        # 뉴스 센티먼트 (30% 가중치)
        if news:
            news_avg = np.mean([n.sentiment_score for n in news])
            news_normalized = news_avg * 100
        else:
            news_normalized = 0

        # 소셜 센티먼트 (30% 가중치)
        if social:
            social_avg = np.mean([s.sentiment_score for s in social])
            social_normalized = social_avg * 100
        else:
            social_normalized = 0

        # 가중 평균
        composite_score = (
            fg_normalized * 0.4 +
            news_normalized * 0.3 +
            social_normalized * 0.3
        )

        # 레벨 판단
        if composite_score <= -40:
            level = SentimentLevel.EXTREME_FEAR
        elif composite_score <= -10:
            level = SentimentLevel.FEAR
        elif composite_score <= 10:
            level = SentimentLevel.NEUTRAL
        elif composite_score <= 40:
            level = SentimentLevel.GREED
        else:
            level = SentimentLevel.EXTREME_GREED

        # Contrarian 시그널
        if composite_score <= -40:
            contrarian = 'buy'  # Extreme fear = buying opportunity
            confidence = abs(composite_score) / 100
        elif composite_score >= 40:
            contrarian = 'sell'  # Extreme greed = selling opportunity
            confidence = abs(composite_score) / 100
        else:
            contrarian = 'hold'
            confidence = 0.5

        return CompositeSentiment(
            score=composite_score,
            level=level,
            fear_greed=fear_greed,
            news_sentiment=news_normalized,
            social_sentiment=social_normalized,
            contrarian_signal=contrarian,
            confidence=confidence,
        )

    # ========================================================================
    # Options Analysis Methods (NEW)
    # ========================================================================

    def analyze_vix_term_structure(self) -> Optional[VIXTermStructureResult]:
        """VIX 기간구조 분석"""
        import yfinance as yf

        try:
            # VIX 관련 티커 데이터 수집
            tickers = ["^VIX", "^VIX3M"]
            data = yf.download(tickers, period="1y", progress=False)

            if isinstance(data.columns, pd.MultiIndex):
                vix_data = data['Close']['^VIX']
                vix3m_data = data['Close']['^VIX3M']
            else:
                vix_data = data['Close']
                vix3m_data = None

            vix_spot = float(vix_data.iloc[-1])

            # VIX 백분위 계산
            vix_percentile = (vix_data < vix_spot).sum() / len(vix_data) * 100

            # VIX3M 처리
            if vix3m_data is not None and not vix3m_data.empty:
                vix_3m = float(vix3m_data.iloc[-1])
            else:
                vix_3m = vix_spot * 1.05  # 추정값

            # Contango Ratio
            contango_ratio = (vix_3m / vix_spot) - 1

            # 구조 판단
            if contango_ratio > 0.05:
                structure = VIXStructure.CONTANGO
                signal = "NORMAL"
                interpretation = "정상적인 Contango 상태. 시장 안정."
            elif contango_ratio < -0.05:
                structure = VIXStructure.BACKWARDATION
                signal = "WARNING"
                interpretation = "Backwardation 상태. 단기 불안정성 증가."
            else:
                structure = VIXStructure.FLAT
                signal = "NEUTRAL"
                interpretation = "기간구조 평탄. 방향성 불명확."

            # VIX 수준 추가 해석
            if vix_spot > 30:
                interpretation += f" VIX {vix_spot:.0f}로 높은 공포 상태."
            elif vix_spot < 15:
                interpretation += f" VIX {vix_spot:.0f}로 과도한 안심."

            return VIXTermStructureResult(
                vix_spot=round(vix_spot, 2),
                vix_3m=round(vix_3m, 2),
                structure=structure,
                contango_ratio=round(contango_ratio, 4),
                percentile=round(vix_percentile, 1),
                signal=signal,
                interpretation=interpretation
            )

        except Exception as e:
            print(f"VIX analysis error: {e}")
            return None

    def analyze_put_call_ratio(self, ticker: str = "SPY") -> Optional[PutCallRatioResult]:
        """Put/Call Ratio 분석"""
        import yfinance as yf

        try:
            stock = yf.Ticker(ticker)
            options_dates = stock.options

            if not options_dates:
                return None

            # 가장 가까운 만기일 (최소 7일 이상)
            today = datetime.now()
            target_expiry = None

            for expiry in options_dates:
                exp_date = datetime.strptime(expiry, "%Y-%m-%d")
                if (exp_date - today).days >= 7:
                    target_expiry = expiry
                    break

            if not target_expiry:
                target_expiry = options_dates[0]

            chain = stock.option_chain(target_expiry)

            # Volume 기반
            put_volume = int(chain.puts['volume'].fillna(0).sum())
            call_volume = int(chain.calls['volume'].fillna(0).sum())

            # Open Interest 기반
            put_oi = int(chain.puts['openInterest'].fillna(0).sum())
            call_oi = int(chain.calls['openInterest'].fillna(0).sum())

            # Ratio 계산
            pc_ratio = put_volume / call_volume if call_volume > 0 else 1.0
            pc_oi_ratio = put_oi / call_oi if call_oi > 0 else 1.0

            # 신호 판단 (역발상)
            if pc_ratio < 0.7:
                signal = "BEARISH"  # 역발상: 낙관 과다 = 조정 가능
                interpretation = f"P/C {pc_ratio:.2f}로 과도한 낙관. 조정 가능성."
            elif pc_ratio > 1.0:
                signal = "BULLISH"  # 역발상: 비관 과다 = 반등 가능
                interpretation = f"P/C {pc_ratio:.2f}로 과도한 비관. 반등 가능성."
            else:
                signal = "NEUTRAL"
                interpretation = f"P/C {pc_ratio:.2f}로 중립적."

            return PutCallRatioResult(
                ticker=ticker,
                expiry_date=target_expiry,
                put_volume=put_volume,
                call_volume=call_volume,
                put_call_ratio=round(pc_ratio, 3),
                put_oi=put_oi,
                call_oi=call_oi,
                put_call_oi_ratio=round(pc_oi_ratio, 3),
                signal=signal,
                interpretation=interpretation
            )

        except Exception as e:
            print(f"Put/Call ratio error: {e}")
            return None

    def calculate_iv_percentile(self, ticker: str = "SPY") -> Optional[IVPercentileResult]:
        """IV Percentile 계산"""
        import yfinance as yf
        import pandas as pd

        try:
            stock = yf.Ticker(ticker)

            # 과거 1년 가격 데이터로 HV 계산
            hist = stock.history(period="1y")
            if hist.empty:
                return None

            # 20일 Historical Volatility
            returns = hist['Close'].pct_change()
            hv_20 = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100)

            # VIX를 IV 프록시로 사용
            vix_data = yf.download("^VIX", period="1y", progress=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_series = vix_data['Close']['^VIX']
            else:
                vix_series = vix_data['Close']

            current_vix = float(vix_series.iloc[-1])
            iv_percentile = float((vix_series < current_vix).sum() / len(vix_series) * 100)

            # IV/HV Ratio
            iv_hv_ratio = current_vix / hv_20 if hv_20 > 0 else 1.0

            # 신호 판단
            if iv_percentile > 80:
                signal = "HIGH_IV"
                interpretation = f"IV 상위 {100-iv_percentile:.0f}%. 옵션 매도 유리."
            elif iv_percentile < 20:
                signal = "LOW_IV"
                interpretation = f"IV 하위 {iv_percentile:.0f}%. 옵션 매수 유리."
            else:
                signal = "NORMAL"
                interpretation = f"IV 정상 범위 ({iv_percentile:.0f}%)."

            return IVPercentileResult(
                ticker=ticker,
                current_iv=round(current_vix, 2),
                iv_percentile=round(iv_percentile, 1),
                hv_20=round(hv_20, 2),
                iv_hv_ratio=round(iv_hv_ratio, 2),
                signal=signal,
                interpretation=interpretation
            )

        except Exception as e:
            print(f"IV percentile error: {e}")
            return None

    def get_market_context(self) -> Dict[str, Any]:
        """시장 컨텍스트"""
        import yfinance as yf

        context = {}

        try:
            # VIX
            vix = yf.Ticker('^VIX')
            vix_hist = vix.history(period='5d')
            if len(vix_hist) > 0:
                context['vix'] = float(vix_hist['Close'].iloc[-1])
                context['vix_change'] = float(
                    (vix_hist['Close'].iloc[-1] / vix_hist['Close'].iloc[0] - 1) * 100
                )

            # SPY
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(period='5d')
            if len(spy_hist) > 0:
                context['spy_price'] = float(spy_hist['Close'].iloc[-1])
                context['spy_change'] = float(
                    (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
                )

        except Exception as e:
            print(f"Market context error: {e}")

        return context

    def generate_signals(
        self,
        composite: CompositeSentiment,
        context: Dict[str, Any],
    ) -> List[str]:
        """시그널 생성"""
        signals = []

        # Contrarian 시그널
        if composite.contrarian_signal == 'buy':
            signals.append(f"Extreme fear detected - Contrarian BUY signal (Confidence: {composite.confidence:.0%})")
        elif composite.contrarian_signal == 'sell':
            signals.append(f"Extreme greed detected - Contrarian SELL signal (Confidence: {composite.confidence:.0%})")

        # Fear & Greed 변화
        if composite.fear_greed:
            fg = composite.fear_greed
            if fg.value < fg.week_ago - 10:
                signals.append(f"Fear increased significantly (Now: {fg.value}, Week ago: {fg.week_ago})")
            elif fg.value > fg.week_ago + 10:
                signals.append(f"Greed increased significantly (Now: {fg.value}, Week ago: {fg.week_ago})")

        # VIX 기반
        vix = context.get('vix', 20)
        if vix > 30:
            signals.append(f"VIX elevated at {vix:.1f} - Market fear")
        elif vix < 15:
            signals.append(f"VIX low at {vix:.1f} - Complacency warning")

        return signals

    def analyze(self, include_options: bool = True) -> SentimentAnalysisResult:
        """전체 분석 실행"""
        print(f"\n{'='*50}")
        print("EIMAS Sentiment & Options Analysis")
        print('='*50)

        # Fear & Greed
        print("Fetching Fear & Greed Index...")
        fear_greed = self.fetch_fear_greed_index()
        print(f"  Value: {fear_greed.value} ({fear_greed.level.value})")

        # 뉴스
        print("Analyzing news sentiment...")
        all_news = []
        for ticker in self.tickers[:2]:
            news = self.fetch_news_sentiment(ticker)
            all_news.extend(news)
        print(f"  Collected {len(all_news)} news items")

        # 소셜
        print("Estimating social sentiment...")
        social_data = []
        for ticker in self.tickers:
            social = self.estimate_social_sentiment(ticker)
            social_data.append(social)

        # 시장 컨텍스트
        context = self.get_market_context()

        # 옵션 분석 (NEW)
        if include_options:
            print("Analyzing VIX term structure...")
            vix_structure = self.analyze_vix_term_structure()
            if vix_structure:
                context['vix_structure'] = vix_structure
                print(f"  VIX: {vix_structure.vix_spot} ({vix_structure.structure.value})")

            print("Analyzing Put/Call ratio...")
            put_call = self.analyze_put_call_ratio("SPY")
            if put_call:
                context['put_call_ratio'] = put_call
                print(f"  P/C Ratio: {put_call.put_call_ratio:.2f} ({put_call.signal})")

            print("Calculating IV percentile...")
            iv_pct = self.calculate_iv_percentile("SPY")
            if iv_pct:
                context['iv_percentile'] = iv_pct
                print(f"  IV Percentile: {iv_pct.iv_percentile:.0f}%")

        # 통합
        composite = self.calculate_composite_sentiment(fear_greed, all_news, social_data)
        print(f"  Composite: {composite.score:.1f} ({composite.level.value})")

        # 시그널
        signals = self.generate_signals(composite, context)

        # 옵션 관련 시그널 추가
        if include_options and 'vix_structure' in context:
            vs = context['vix_structure']
            if vs.structure == VIXStructure.BACKWARDATION:
                signals.append(f"VIX Backwardation: 단기 변동성 스파이크 경고")
            if vs.vix_spot > 25:
                signals.append(f"VIX {vs.vix_spot:.0f}: 높은 공포 수준")

        if include_options and 'put_call_ratio' in context:
            pcr = context['put_call_ratio']
            if pcr.signal == "BULLISH":
                signals.append(f"P/C Ratio {pcr.put_call_ratio:.2f}: 역발상 매수 기회")
            elif pcr.signal == "BEARISH":
                signals.append(f"P/C Ratio {pcr.put_call_ratio:.2f}: 역발상 주의")

        # 요약
        summary = self._generate_summary(composite, context, signals)

        return SentimentAnalysisResult(
            timestamp=datetime.now(),
            composite=composite,
            news_items=all_news,
            social_data=social_data,
            market_context=context,
            signals=signals,
            summary=summary,
        )

    def _generate_summary(
        self,
        composite: CompositeSentiment,
        context: Dict[str, Any],
        signals: List[str],
    ) -> str:
        """요약 생성"""
        lines = [
            f"Composite Sentiment: {composite.score:.1f} ({composite.level.value})",
            f"Fear & Greed: {composite.fear_greed.value if composite.fear_greed else 'N/A'}",
            f"News Sentiment: {composite.news_sentiment:.1f}",
            f"Social Sentiment: {composite.social_sentiment:.1f}",
            f"Contrarian Signal: {composite.contrarian_signal.upper()}",
        ]

        if context:
            lines.append(f"\nMarket Context:")
            if 'vix' in context:
                lines.append(f"  VIX: {context['vix']:.1f} ({context.get('vix_change', 0):+.1f}%)")
            if 'spy_price' in context:
                lines.append(f"  SPY: ${context['spy_price']:.2f} ({context.get('spy_change', 0):+.1f}%)")

        if signals:
            lines.append("\nSignals:")
            for s in signals:
                lines.append(f"  • {s}")

        return "\n".join(lines)

    def print_result(self, result: SentimentAnalysisResult):
        """결과 출력"""
        print("\n" + result.summary)
        print("="*50)


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_sentiment_check() -> Dict[str, Any]:
    """빠른 센티먼트 체크"""
    analyzer = SentimentAnalyzer()
    fg = analyzer.fetch_fear_greed_index()

    return {
        'fear_greed': fg.value if fg else 50,
        'level': fg.level.value if fg else 'neutral',
        'timestamp': datetime.now().isoformat(),
    }


def get_contrarian_signal() -> str:
    """Contrarian 시그널"""
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze()
    return result.composite.contrarian_signal


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    analyzer = SentimentAnalyzer(['SPY', 'QQQ'])
    result = analyzer.analyze()
    analyzer.print_result(result)

    # Quick check
    print("\n\nQuick Sentiment Check:")
    quick = quick_sentiment_check()
    print(f"Fear & Greed: {quick['fear_greed']} ({quick['level']})")
