#!/usr/bin/env python3
"""
News-based Event Generator for EIMAS
=====================================
Generates events based on:
1. AI debate topics and macro analysis results
2. Real-time news from APIs (Perplexity)
3. Sharp price movements and sector signals
4. Default: macro sentiment, crypto news

Author: EIMAS Team
Created: 2026-02-14
"""

import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class NewsEvent:
    """News-based event"""
    event_type: str
    title: str
    description: str
    severity: str  # LOW, MEDIUM, HIGH
    source: str
    timestamp: str
    related_tickers: List[str]
    category: str  # macro, crypto, sector, ticker


class NewsEventGenerator:
    """
    Generates events from news APIs and market signals
    """

    def __init__(self):
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

    def generate_events(
        self,
        ai_debate_topics: Optional[List[str]] = None,
        macro_analysis: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        portfolio_tickers: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate events from multiple sources

        Args:
            ai_debate_topics: Topics from AI debate consensus
            macro_analysis: Results from macro analysis
            market_data: Market price and volume data
            portfolio_tickers: Current portfolio tickers

        Returns:
            List of event dictionaries
        """
        events = []

        # 1. Generate macro sentiment events (default)
        macro_events = self._generate_macro_events(macro_analysis)
        events.extend(macro_events)

        # 2. Generate crypto news events (default)
        crypto_events = self._generate_crypto_events()
        events.extend(crypto_events)

        # 3. Generate AI debate topic events
        if ai_debate_topics:
            debate_events = self._generate_debate_topic_events(ai_debate_topics)
            events.extend(debate_events)

        # 4. Detect sharp price movements
        if market_data:
            price_events = self._detect_sharp_movements(market_data, portfolio_tickers or [])
            events.extend(price_events)

        # 5. Detect sector signals
        if market_data:
            sector_events = self._detect_sector_signals(market_data)
            events.extend(sector_events)

        return events

    def _generate_macro_events(self, macro_analysis: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate events from macro environment analysis"""
        events = []

        if not macro_analysis:
            return events

        try:
            # Check Fed liquidity conditions
            net_liquidity = macro_analysis.get("net_liquidity")
            if net_liquidity:
                if net_liquidity > 6000:
                    events.append({
                        "event_type": "LIQUIDITY_INJECTION",
                        "title": "High Net Liquidity Environment",
                        "description": f"Fed net liquidity at ${net_liquidity:.1f}B, above $6T threshold. Supportive for risk assets.",
                        "severity": "MEDIUM",
                        "source": "FRED",
                        "timestamp": datetime.now().isoformat(),
                        "related_tickers": ["SPY", "QQQ"],
                        "category": "macro",
                    })
                elif net_liquidity < 5000:
                    events.append({
                        "event_type": "LIQUIDITY_STRESS",
                        "title": "Low Net Liquidity Environment",
                        "description": f"Fed net liquidity at ${net_liquidity:.1f}B, below $5T. Potential headwind for risk assets.",
                        "severity": "HIGH",
                        "source": "FRED",
                        "timestamp": datetime.now().isoformat(),
                        "related_tickers": ["SPY", "TLT"],
                        "category": "macro",
                    })

            # Check treasury yields
            treasury_10y = macro_analysis.get("treasury_10y")
            if treasury_10y:
                if treasury_10y > 4.5:
                    events.append({
                        "event_type": "RATE_STRESS",
                        "title": "Elevated Treasury Yields",
                        "description": f"10Y Treasury yield at {treasury_10y:.2f}%, above 4.5%. Higher borrowing costs may pressure equities.",
                        "severity": "MEDIUM",
                        "source": "FRED",
                        "timestamp": datetime.now().isoformat(),
                        "related_tickers": ["TLT", "HYG"],
                        "category": "macro",
                    })

        except Exception as e:
            print(f"Warning: Error generating macro events: {e}")

        return events

    def _generate_crypto_events(self) -> List[Dict[str, Any]]:
        """Generate crypto news events using Perplexity API"""
        events = []

        if not self.perplexity_api_key:
            return events

        try:
            # Query for crypto news
            query = "Latest Bitcoin Ethereum cryptocurrency news today developments regulations"

            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.perplexity_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sonar",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{query}. Summarize in 2-3 sentences the most important news today."
                        }
                    ],
                    "max_tokens": 200,
                    "temperature": 0.2,
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                if content:
                    events.append({
                        "event_type": "NEWS_SURGE",
                        "title": "Crypto Market News Update",
                        "description": content[:200],
                        "severity": "LOW",
                        "source": "Perplexity",
                        "timestamp": datetime.now().isoformat(),
                        "related_tickers": ["BTC-USD", "ETH-USD"],
                        "category": "crypto",
                    })

        except Exception as e:
            print(f"Warning: Error fetching crypto news: {e}")

        return events

    def _generate_debate_topic_events(self, topics: List[str]) -> List[Dict[str, Any]]:
        """Generate events based on AI debate consensus topics"""
        events = []

        if not self.perplexity_api_key or not topics:
            return events

        try:
            # Take first 2 topics to avoid rate limits
            for topic in topics[:2]:
                query = f"{topic} latest news developments impact"

                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"{query}. Provide 1-2 sentence summary of recent developments."
                            }
                        ],
                        "max_tokens": 150,
                        "temperature": 0.2,
                    },
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                    if content:
                        events.append({
                            "event_type": "ANALYST_REVISION",
                            "title": f"Update: {topic[:50]}",
                            "description": content[:200],
                            "severity": "MEDIUM",
                            "source": "Perplexity",
                            "timestamp": datetime.now().isoformat(),
                            "related_tickers": [],
                            "category": "debate_topic",
                        })

        except Exception as e:
            print(f"Warning: Error generating debate topic events: {e}")

        return events

    def _detect_sharp_movements(
        self, market_data: Dict[str, Any], portfolio_tickers: List[str]
    ) -> List[Dict[str, Any]]:
        """Detect sharp price movements (>5% daily change)"""
        events = []

        try:
            for ticker, data in market_data.items():
                if not isinstance(data, pd.DataFrame) or data.empty:
                    continue

                # Calculate daily returns
                if "Close" in data.columns and len(data) >= 2:
                    last_close = data["Close"].iloc[-1]
                    prev_close = data["Close"].iloc[-2]
                    pct_change = ((last_close - prev_close) / prev_close) * 100

                    # Detect significant moves
                    if abs(pct_change) >= 5.0:
                        direction = "surged" if pct_change > 0 else "plunged"
                        severity = "HIGH" if abs(pct_change) >= 10 else "MEDIUM"

                        events.append({
                            "event_type": "PRICE_SHOCK",
                            "title": f"{ticker} {direction} {abs(pct_change):.1f}%",
                            "description": f"{ticker} moved {pct_change:+.2f}% today. Current price: ${last_close:.2f}",
                            "severity": severity,
                            "source": "Market Data",
                            "timestamp": datetime.now().isoformat(),
                            "related_tickers": [ticker],
                            "category": "ticker",
                        })

        except Exception as e:
            print(f"Warning: Error detecting sharp movements: {e}")

        return events

    def _detect_sector_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect sector-wide signals from ETF movements"""
        events = []

        sector_etfs = {
            "XLF": "Financials",
            "XLE": "Energy",
            "XLV": "Healthcare",
            "XLK": "Technology",
            "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples",
            "XLI": "Industrials",
            "XLB": "Materials",
            "XLU": "Utilities",
            "XLRE": "Real Estate",
        }

        try:
            sector_moves = []

            for ticker, sector_name in sector_etfs.items():
                if ticker not in market_data:
                    continue

                data = market_data[ticker]
                if not isinstance(data, pd.DataFrame) or data.empty or "Close" not in data.columns:
                    continue

                if len(data) >= 2:
                    last_close = data["Close"].iloc[-1]
                    prev_close = data["Close"].iloc[-2]
                    pct_change = ((last_close - prev_close) / prev_close) * 100

                    sector_moves.append({
                        "ticker": ticker,
                        "sector": sector_name,
                        "pct_change": pct_change,
                    })

            # Find top movers
            if sector_moves:
                sector_moves.sort(key=lambda x: abs(x["pct_change"]), reverse=True)
                top_movers = sector_moves[:3]

                for mover in top_movers:
                    if abs(mover["pct_change"]) >= 2.0:
                        direction = "outperforming" if mover["pct_change"] > 0 else "underperforming"

                        events.append({
                            "event_type": "FLOW_REVERSAL",
                            "title": f"{mover['sector']} Sector {direction.capitalize()}",
                            "description": f"{mover['sector']} ({mover['ticker']}) {direction} with {mover['pct_change']:+.2f}% move.",
                            "severity": "LOW",
                            "source": "Market Data",
                            "timestamp": datetime.now().isoformat(),
                            "related_tickers": [mover["ticker"]],
                            "category": "sector",
                        })

        except Exception as e:
            print(f"Warning: Error detecting sector signals: {e}")

        return events


def generate_news_events(
    ai_debate_topics: Optional[List[str]] = None,
    macro_analysis: Optional[Dict[str, Any]] = None,
    market_data: Optional[Dict[str, Any]] = None,
    portfolio_tickers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate news events

    Returns:
        List of event dictionaries compatible with EIMAS event framework
    """
    generator = NewsEventGenerator()
    return generator.generate_events(
        ai_debate_topics=ai_debate_topics,
        macro_analysis=macro_analysis,
        market_data=market_data,
        portfolio_tickers=portfolio_tickers,
    )


if __name__ == "__main__":
    # Test
    generator = NewsEventGenerator()

    # Test macro events
    mock_macro = {
        "net_liquidity": 5700,
        "treasury_10y": 4.2,
    }

    events = generator.generate_events(
        macro_analysis=mock_macro,
        ai_debate_topics=["Federal Reserve policy outlook", "AI chip demand"],
    )

    print(f"Generated {len(events)} events:")
    for event in events:
        print(f"\n- {event['title']}")
        print(f"  Category: {event['category']}, Severity: {event['severity']}")
        print(f"  {event['description'][:100]}...")
