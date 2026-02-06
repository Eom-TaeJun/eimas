#!/usr/bin/env python3
"""
EIMAS Pipeline - Analyzers Sentiment Module

Purpose: Sentiment and bubble analysis functions (Phase 2.22-2.23)
Functions: analyze_bubble_risk, analyze_sentiment
"""

from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

# Schemas
from pipeline.exceptions import get_logger, log_error

logger = get_logger("analyzers.sentiment")

def analyze_bubble_risk(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    버블 리스크 분석 (Bubbles for Fama 기반)

    기능:
    - 주요 자산의 Run-up 체크 (2년 누적 수익률 > 100%)
    - 변동성 급등 탐지
    - 주식 발행 증가 분석

    References:
    - Greenwood, Shleifer & You (2019)
    """
    print("\n[2.22] Bubble Risk Analysis (Bubbles for Fama)...")
    try:
        from lib.bubble_detector import BubbleDetector

        detector = BubbleDetector()
        results = {
            'overall_status': 'NONE',
            'risk_tickers': [],
            'highest_risk_ticker': '',
            'highest_risk_score': 0.0,
            'methodology_notes': 'Bubbles for Fama (2019)'
        }

        # 주요 자산 분석
        tickers_to_check = ['SPY', 'QQQ', 'IWM', 'ARKK']
        risk_tickers = []
        highest_score = 0.0
        highest_ticker = ''

        for ticker in tickers_to_check:
            if ticker in market_data and not market_data[ticker].empty:
                df = market_data[ticker]
                if 'Close' in df.columns:
                    try:
                        # analyze()는 DataFrame을 기대함 (Close 컬럼 포함)
                        detection = detector.analyze(ticker, price_data=df)
                        if detection.bubble_warning_level.value != "NONE":
                            runup_val = 0.0
                            if detection.runup and hasattr(detection.runup, 'cumulative_return'):
                                runup_val = detection.runup.cumulative_return
                            risk_tickers.append({
                                'ticker': ticker,
                                'warning_level': detection.bubble_warning_level.value,
                                'risk_score': detection.risk_score,
                                'runup': runup_val
                            })
                            if detection.risk_score > highest_score:
                                highest_score = detection.risk_score
                                highest_ticker = ticker
                    except Exception as ex:
                        logger.warning(f"Bubble detection failed for {ticker}: {ex}")

        # 전체 상태 결정
        if highest_score >= 70:
            results['overall_status'] = 'DANGER'
        elif highest_score >= 50:
            results['overall_status'] = 'WARNING'
        elif highest_score >= 30:
            results['overall_status'] = 'WATCH'

        results['risk_tickers'] = risk_tickers
        results['highest_risk_ticker'] = highest_ticker
        results['highest_risk_score'] = highest_score

        print(f"      ✓ Bubble Status: {results['overall_status']}")
        if risk_tickers:
            print(f"      ⚠ Risk Tickers: {', '.join([t['ticker'] for t in risk_tickers])}")

        return results
    except Exception as e:
        log_error(logger, "Bubble risk analysis failed", e)
        return {}


def analyze_sentiment() -> Dict[str, Any]:
    """
    센티먼트 분석 (Fear & Greed, VIX 구조, 뉴스)

    기능:
    - Fear & Greed Index 수집
    - VIX Term Structure 분석
    - 뉴스 센티먼트 분석

    References:
    - CNN Fear & Greed Index
    """
    print("\n[2.23] Sentiment Analysis...")
    try:
        from lib.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        results = {}

        # Fear & Greed
        try:
            fg = analyzer.fetch_fear_greed_index()
            if fg:
                results['fear_greed'] = {
                    'value': fg.value,
                    'level': fg.level.value,
                    'previous_close': fg.previous_close,
                    'week_ago': fg.week_ago
                }
                print(f"      ✓ Fear & Greed: {fg.value} ({fg.level.value})")
        except Exception as ex:
            logger.warning(f"Fear & Greed fetch failed: {ex}")

        # VIX Term Structure
        try:
            vix = analyzer.analyze_vix_term_structure()
            if vix:
                results['vix_structure'] = {
                    'vix_spot': vix.vix_spot,
                    'structure': vix.structure.value,
                    'contango_ratio': vix.contango_ratio,
                    'signal': vix.signal
                }
                print(f"      ✓ VIX Structure: {vix.structure.value} (Signal: {vix.signal})")
        except Exception as ex:
            logger.warning(f"VIX structure analysis failed: {ex}")

        # News Sentiment
        try:
            news = analyzer.fetch_news_sentiment('SPY')
            if news:
                avg_score = sum(n.sentiment_score for n in news) / len(news)
                results['news_sentiment'] = {
                    'avg_score': avg_score,
                    'count': len(news),
                    'overall': 'BULLISH' if avg_score > 0.2 else 'BEARISH' if avg_score < -0.2 else 'NEUTRAL'
                }
                print(f"      ✓ News Sentiment: {results['news_sentiment']['overall']} (n={len(news)})")
        except Exception as ex:
            logger.warning(f"News sentiment fetch failed: {ex}")

        return results
    except Exception as e:
        log_error(logger, "Sentiment analysis failed", e)
        return {}
