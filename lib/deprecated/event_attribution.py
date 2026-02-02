#!/usr/bin/env python3
"""
EIMAS Event Attribution
=======================
감지된 이벤트의 원인 역추적 및 분석

주요 기능:
1. 시장 이벤트 감지 → 원인 추론
2. 뉴스/경제지표 매칭
3. 크로스-에셋 상관관계 분석
4. AI 기반 원인 분석

사용법:
    from lib.event_attribution import EventAttributor

    attributor = EventAttributor()
    report = attributor.analyze_recent_events()
    attributor.print_report(report)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import os
import httpx
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Known Causes Database
# ============================================================================

# 자산별 주요 영향 요인
ASSET_DRIVERS = {
    "GLD": {
        "name": "Gold",
        "primary_drivers": ["dollar_strength", "real_rates", "inflation_expectations", "geopolitical"],
        "inverse_correlated": ["DXY", "US10Y"],
        "positive_correlated": ["TLT", "VIX"],
        "typical_catalysts": [
            "Fed rate decision / hawkish-dovish pivot",
            "Inflation data (CPI, PCE)",
            "Dollar strength/weakness",
            "Geopolitical tension (war, sanctions)",
            "Central bank gold purchases",
            "Real yield changes",
        ]
    },
    "SPY": {
        "name": "S&P 500",
        "primary_drivers": ["earnings", "rates", "economy", "sentiment"],
        "inverse_correlated": ["VIX"],
        "positive_correlated": ["QQQ", "IWM"],
        "typical_catalysts": [
            "Fed policy / rate expectations",
            "Earnings season",
            "Economic data (GDP, Jobs)",
            "Inflation data",
            "Geopolitical events",
            "Sector rotation",
        ]
    },
    "QQQ": {
        "name": "NASDAQ 100",
        "primary_drivers": ["tech_earnings", "rates", "growth_sentiment"],
        "inverse_correlated": ["TLT", "VIX"],
        "positive_correlated": ["SPY"],
        "typical_catalysts": [
            "Tech mega-cap earnings",
            "Rate expectations (growth vs value)",
            "AI/Tech sentiment",
            "Bond yields",
        ]
    },
    "TLT": {
        "name": "20+ Year Treasury",
        "primary_drivers": ["rates", "inflation", "fed_policy"],
        "inverse_correlated": ["US10Y"],
        "positive_correlated": ["GLD"],
        "typical_catalysts": [
            "Fed rate decision",
            "Inflation data",
            "Treasury auctions",
            "Flight to safety",
            "Economic slowdown signals",
        ]
    },
    "VIX": {
        "name": "Volatility Index",
        "primary_drivers": ["uncertainty", "fear", "hedging"],
        "inverse_correlated": ["SPY"],
        "positive_correlated": [],
        "typical_catalysts": [
            "Market selloff",
            "Geopolitical shock",
            "Unexpected economic data",
            "Policy uncertainty",
            "Earnings surprises",
        ]
    }
}

# 날짜별 알려진 이벤트 (수동 데이터)
KNOWN_EVENTS_2025 = {
    "2025-12-29": [
        {"type": "market", "description": "Year-end rebalancing, low liquidity"},
    ],
    "2025-12-30": [
        {"type": "market", "description": "Year-end positioning, tax-loss selling end"},
    ],
    "2025-12-31": [
        {"type": "market", "description": "New Year's Eve, half-day trading"},
    ],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EventAttribution:
    """이벤트 원인 분석 결과"""
    event_date: str
    ticker: str
    event_type: str
    event_description: str

    # 가격 데이터
    price_change: float = 0.0
    volume_ratio: float = 1.0

    # 원인 분석
    likely_causes: List[str] = field(default_factory=list)
    correlated_moves: Dict[str, float] = field(default_factory=dict)
    known_events: List[str] = field(default_factory=list)

    # AI 분석 (선택적)
    ai_analysis: str = ""

    # 신뢰도
    confidence: float = 0.0


@dataclass
class AttributionReport:
    """역추적 리포트"""
    generated_at: str
    period_start: str
    period_end: str
    total_events: int
    attributions: List[EventAttribution] = field(default_factory=list)
    summary: str = ""


# ============================================================================
# Event Attributor
# ============================================================================

class EventAttributor:
    """이벤트 원인 역추적"""

    def __init__(self, use_ai: bool = False, verbose: bool = True):
        self.use_ai = use_ai
        self.verbose = verbose
        self.perplexity_key = os.environ.get("PERPLEXITY_API_KEY", "")
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"[EventAttributor] {msg}")

    def _get_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """가격 데이터 조회"""
        cache_key = f"{ticker}_{start}_{end}"
        if cache_key not in self._price_cache:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            self._price_cache[cache_key] = df
        return self._price_cache[cache_key]

    def _analyze_correlations(
        self,
        event_date: datetime,
        primary_ticker: str,
        lookback: int = 5
    ) -> Dict[str, float]:
        """관련 자산 동시 움직임 분석"""
        correlations = {}

        # 비교 대상 결정
        ticker_info = ASSET_DRIVERS.get(primary_ticker, {})
        related = ticker_info.get("inverse_correlated", []) + ticker_info.get("positive_correlated", [])

        if not related:
            related = ["SPY", "TLT", "GLD", "^VIX"]

        start = (event_date - timedelta(days=lookback)).strftime("%Y-%m-%d")
        end = (event_date + timedelta(days=2)).strftime("%Y-%m-%d")

        for ticker in related:
            try:
                df = self._get_prices(ticker, start, end)
                if not df.empty and len(df) >= 2:
                    # 이벤트 날짜 수익률
                    date_strs = [d.strftime("%Y-%m-%d") for d in df.index]
                    event_str = event_date.strftime("%Y-%m-%d")

                    if event_str in date_strs:
                        idx = date_strs.index(event_str)
                        if idx > 0:
                            ret = (df['Close'].iloc[idx] / df['Close'].iloc[idx-1] - 1) * 100
                            correlations[ticker] = round(ret, 2)
            except Exception:
                pass

        return correlations

    def _infer_causes(
        self,
        ticker: str,
        event_type: str,
        price_change: float,
        correlations: Dict[str, float],
        event_date: str
    ) -> List[str]:
        """원인 추론"""
        causes = []

        # 자산별 기본 드라이버
        ticker_info = ASSET_DRIVERS.get(ticker, {})
        typical = ticker_info.get("typical_catalysts", [])

        # 알려진 이벤트 확인
        known = KNOWN_EVENTS_2025.get(event_date, [])
        for k in known:
            causes.append(f"[Known] {k['description']}")

        # 상관관계 기반 추론
        if ticker == "GLD":
            # 달러와 역상관
            dxy_move = correlations.get("DXY", correlations.get("UUP", 0))
            if abs(dxy_move) > 0.3:
                if (price_change > 0 and dxy_move < 0) or (price_change < 0 and dxy_move > 0):
                    causes.append(f"Dollar {'weakness' if dxy_move < 0 else 'strength'} ({dxy_move:+.1f}%)")

            # VIX와 정상관
            vix_move = correlations.get("^VIX", 0)
            if vix_move > 5:
                causes.append(f"Risk-off sentiment (VIX {vix_move:+.1f}%)")

        elif ticker == "SPY":
            vix_move = correlations.get("^VIX", 0)
            if vix_move > 10:
                causes.append(f"Fear spike (VIX {vix_move:+.1f}%)")

            # 섹터 분산
            qqq_move = correlations.get("QQQ", 0)
            iwm_move = correlations.get("IWM", 0)
            if abs(qqq_move - iwm_move) > 1:
                if qqq_move > iwm_move:
                    causes.append("Growth outperforming Value")
                else:
                    causes.append("Value outperforming Growth")

        # 변동성 급등
        if "Volatility" in event_type:
            causes.append("Increased uncertainty / hedging demand")

        # 거래량 급등
        if "Volume" in event_type:
            causes.append("Large institutional flow / rebalancing")

        # 가격 급변
        if "Price Shock" in event_type:
            if abs(price_change) > 3:
                causes.append("Significant news event or liquidity shock")
            else:
                causes.append("Moderate sentiment shift")

        # 날짜 기반 추론
        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        if event_dt.month == 12 and event_dt.day >= 20:
            causes.append("Year-end dynamics (low liquidity, rebalancing)")
        if event_dt.weekday() == 0:  # 월요일
            causes.append("Weekend gap / news accumulation")
        if event_dt.day <= 5:
            causes.append("Month-start rebalancing")

        # 기본 드라이버 추가
        if not causes and typical:
            causes.append(f"[Possible] {typical[0]}")

        return causes[:5]  # 상위 5개

    def _query_perplexity(self, query: str) -> str:
        """Perplexity API로 원인 검색"""
        if not self.perplexity_key:
            return ""

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": query}],
                        "temperature": 0.1
                    }
                )
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"API Error: {e}"

    def attribute_event(
        self,
        event_date: str,
        ticker: str,
        event_type: str,
        event_description: str,
        use_ai: bool = None
    ) -> EventAttribution:
        """단일 이벤트 원인 분석"""
        if use_ai is None:
            use_ai = self.use_ai

        event_dt = datetime.strptime(event_date, "%Y-%m-%d")

        # 가격 데이터
        start = (event_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end = (event_dt + timedelta(days=2)).strftime("%Y-%m-%d")
        prices = self._get_prices(ticker, start, end)

        price_change = 0.0
        volume_ratio = 1.0

        if not prices.empty:
            date_strs = [d.strftime("%Y-%m-%d") for d in prices.index]
            if event_date in date_strs:
                idx = date_strs.index(event_date)
                if idx > 0:
                    price_change = (prices['Close'].iloc[idx] / prices['Close'].iloc[idx-1] - 1) * 100
                    if 'Volume' in prices.columns:
                        avg_vol = prices['Volume'].iloc[max(0,idx-5):idx].mean()
                        if avg_vol > 0:
                            volume_ratio = prices['Volume'].iloc[idx] / avg_vol

        # 상관관계 분석
        correlations = self._analyze_correlations(event_dt, ticker)

        # 원인 추론
        causes = self._infer_causes(ticker, event_type, price_change, correlations, event_date)

        # 알려진 이벤트
        known_events = [e['description'] for e in KNOWN_EVENTS_2025.get(event_date, [])]

        # AI 분석 (선택적)
        ai_analysis = ""
        if use_ai and self.perplexity_key:
            query = f"What happened to {ASSET_DRIVERS.get(ticker, {}).get('name', ticker)} on {event_date}? Brief financial news summary."
            ai_analysis = self._query_perplexity(query)

        return EventAttribution(
            event_date=event_date,
            ticker=ticker,
            event_type=event_type,
            event_description=event_description,
            price_change=price_change,
            volume_ratio=volume_ratio,
            likely_causes=causes,
            correlated_moves=correlations,
            known_events=known_events,
            ai_analysis=ai_analysis,
            confidence=min(0.3 + len(causes) * 0.15, 0.9)
        )

    def analyze_recent_events(
        self,
        days_back: int = 14,
        tickers: List[str] = None
    ) -> AttributionReport:
        """최근 이벤트 분석"""
        from lib.event_framework import QuantitativeEventDetector

        if tickers is None:
            tickers = ["SPY", "QQQ", "GLD", "TLT", "IWM"]

        self._log(f"Analyzing events for {tickers} over last {days_back} days...")

        detector = QuantitativeEventDetector()
        all_events = []

        for ticker in tickers:
            df = yf.download(ticker, period="3mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)

            if not df.empty:
                events = detector.detect_all(df, ticker=ticker)
                all_events.extend(events)

        # 최근 N일 필터
        cutoff = datetime.now() - timedelta(days=days_back)
        recent = [e for e in all_events if e.timestamp >= cutoff]
        recent.sort(key=lambda e: e.timestamp, reverse=True)

        self._log(f"  Found {len(recent)} events")

        # 각 이벤트 분석
        attributions = []
        for event in recent[:20]:  # 상위 20개
            attr = self.attribute_event(
                event_date=event.timestamp.strftime("%Y-%m-%d"),
                ticker=event.ticker,
                event_type=event.event_type.value,
                event_description=event.name
            )
            attributions.append(attr)

        # 요약 생성
        if attributions:
            most_active = {}
            for attr in attributions:
                most_active[attr.ticker] = most_active.get(attr.ticker, 0) + 1

            summary = f"Most active: {max(most_active, key=most_active.get)} ({max(most_active.values())} events)"
        else:
            summary = "No significant events detected"

        return AttributionReport(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            period_start=(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
            period_end=datetime.now().strftime("%Y-%m-%d"),
            total_events=len(recent),
            attributions=attributions,
            summary=summary
        )

    def print_report(self, report: AttributionReport):
        """리포트 출력"""
        print("\n" + "=" * 70)
        print("EVENT ATTRIBUTION REPORT")
        print("=" * 70)

        print(f"\nGenerated: {report.generated_at}")
        print(f"Period: {report.period_start} ~ {report.period_end}")
        print(f"Total Events: {report.total_events}")
        print(f"Summary: {report.summary}")

        if not report.attributions:
            print("\nNo events to analyze.")
            return

        print("\n" + "-" * 70)
        print("DETAILED ANALYSIS")
        print("-" * 70)

        for i, attr in enumerate(report.attributions, 1):
            print(f"\n[{i}] {attr.event_date} | {attr.ticker} | {attr.event_description}")
            print(f"    Price Change: {attr.price_change:+.2f}% | Volume: {attr.volume_ratio:.1f}x avg")

            if attr.likely_causes:
                print("    Likely Causes:")
                for cause in attr.likely_causes:
                    print(f"      • {cause}")

            if attr.correlated_moves:
                moves = ", ".join([f"{k}: {v:+.1f}%" for k, v in attr.correlated_moves.items()])
                print(f"    Correlated Moves: {moves}")

            if attr.ai_analysis:
                print(f"    AI Analysis: {attr.ai_analysis[:200]}...")

            print(f"    Confidence: {attr.confidence*100:.0f}%")

        print("\n" + "=" * 70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    attributor = EventAttributor(use_ai=False, verbose=True)

    # 최근 이벤트 분석
    report = attributor.analyze_recent_events(days_back=14)
    attributor.print_report(report)
