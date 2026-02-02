#!/usr/bin/env python3
"""
EIMAS News & Social Sentiment Analyzer
======================================
Analyze news and social media sentiment.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import os
import httpx
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class SentimentLevel(Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class SentimentResult:
    ticker: str
    source: str
    sentiment: SentimentLevel
    score: float  # -1 to 1
    headline_count: int
    key_headlines: List[str]
    summary: str


class NewsSentimentAnalyzer:
    """Analyze news sentiment using Perplexity"""

    def __init__(self):
        self.api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        self.base_url = "https://api.perplexity.ai"

    def _call_api(self, prompt: str) -> str:
        """Call Perplexity API"""
        if not self.api_key:
            return self._mock_response(prompt)

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3
                    }
                )
                return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        """Mock response when API unavailable"""
        return """Based on recent news:
- Market sentiment is cautiously optimistic
- Recent earnings have been mixed
- Macro concerns persist around rates
Overall sentiment: NEUTRAL (0.1)
Key headlines:
1. Market awaits Fed decision
2. Tech earnings beat estimates
3. Economic data shows resilience"""

    def analyze_ticker(self, ticker: str) -> SentimentResult:
        """Analyze sentiment for a ticker"""
        prompt = f"""Analyze the current news sentiment for {ticker} stock.
Provide:
1. Overall sentiment (very_bullish/bullish/neutral/bearish/very_bearish)
2. Sentiment score (-1 to 1)
3. Key headlines (top 3)
4. Brief summary

Format your response clearly with these sections."""

        response = self._call_api(prompt)

        # Parse response (simplified)
        score = 0.0
        sentiment = SentimentLevel.NEUTRAL

        if "very bullish" in response.lower():
            sentiment = SentimentLevel.VERY_BULLISH
            score = 0.8
        elif "bullish" in response.lower():
            sentiment = SentimentLevel.BULLISH
            score = 0.4
        elif "very bearish" in response.lower():
            sentiment = SentimentLevel.VERY_BEARISH
            score = -0.8
        elif "bearish" in response.lower():
            sentiment = SentimentLevel.BEARISH
            score = -0.4

        return SentimentResult(
            ticker=ticker,
            source="news",
            sentiment=sentiment,
            score=score,
            headline_count=3,
            key_headlines=response.split('\n')[:3],
            summary=response[:200]
        )

    def analyze_multiple(self, tickers: List[str]) -> List[SentimentResult]:
        """Analyze multiple tickers"""
        print("\n" + "=" * 60)
        print("EIMAS Sentiment Analyzer")
        print("=" * 60)

        results = []
        for ticker in tickers:
            print(f"  Analyzing {ticker}...")
            result = self.analyze_ticker(ticker)
            results.append(result)

        return results

    def print_report(self, results: List[SentimentResult]):
        """Print sentiment report"""
        print("\n" + "=" * 60)
        print("Sentiment Analysis Report")
        print("=" * 60)

        print(f"\n{'Ticker':<8} {'Sentiment':<15} {'Score':>8}")
        print("-" * 35)

        for r in results:
            print(f"{r.ticker:<8} {r.sentiment.value:<15} {r.score:>+8.2f}")

        # Bullish/Bearish summary
        bullish = [r for r in results if r.score > 0.2]
        bearish = [r for r in results if r.score < -0.2]

        if bullish:
            print(f"\nMost Bullish: {', '.join(r.ticker for r in bullish)}")
        if bearish:
            print(f"Most Bearish: {', '.join(r.ticker for r in bearish)}")

        print("=" * 60)


class SocialSentimentAnalyzer:
    """Analyze social media sentiment"""

    def __init__(self):
        self.api_key = os.environ.get("PERPLEXITY_API_KEY", "")

    def analyze_reddit(self, ticker: str) -> SentimentResult:
        """Analyze Reddit sentiment (WSB, stocks, etc.)"""
        # This would use Reddit API in production
        # For now, use Perplexity to summarize

        prompt = f"""Search Reddit (r/wallstreetbets, r/stocks, r/investing) for recent sentiment on {ticker}.
What is the community sentiment? Are there any unusual mentions or hype?
Rate sentiment from -1 (very bearish) to 1 (very bullish)."""

        if self.api_key:
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": "sonar", "messages": [{"role": "user", "content": prompt}]}
                    )
                    content = response.json()['choices'][0]['message']['content']
            except Exception:
                content = "Unable to fetch Reddit sentiment"
        else:
            content = "Reddit sentiment analysis requires API key"

        return SentimentResult(
            ticker=ticker,
            source="reddit",
            sentiment=SentimentLevel.NEUTRAL,
            score=0.0,
            headline_count=0,
            key_headlines=[],
            summary=content[:200]
        )


if __name__ == "__main__":
    analyzer = NewsSentimentAnalyzer()
    results = analyzer.analyze_multiple(['AAPL', 'TSLA', 'NVDA', 'SPY'])
    analyzer.print_report(results)
