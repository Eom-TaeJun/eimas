"""
Market Sentiment Agent
======================
시장 정서 분석 에이전트 (Claude API)

Purpose:
- KOSPI 전용 시장 정서 분석
- SPX 전용 시장 정서 분석
- 두 시장 간 정서 비교 및 디커플링 감지

Economic Foundation:
- Behavioral Finance (Kahneman & Tversky)
- Market Sentiment Indicators (Baker & Wurgler 2006)
- Fear & Greed Index
- Put/Call Ratio Analysis
"""

import anthropic
from typing import Dict, Optional
import json
import logging

logger = logging.getLogger(__name__)


class MarketSentimentAgent:
    """시장 정서 분석 에이전트 (KOSPI/SPX 분리)"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Anthropic API key (None이면 환경변수 사용)
        """
        import os
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"

    def analyze_market_sentiment(
        self,
        kospi_data: Dict,
        spx_data: Dict,
        market_context: Dict
    ) -> Dict:
        """
        시장 정서 분석 (KOSPI + SPX 분리)

        Args:
            kospi_data: KOSPI 시장 데이터
            spx_data: SPX 시장 데이터
            market_context: 전체 시장 컨텍스트

        Returns:
            {
                'kospi_sentiment': {
                    'sentiment': 'BULLISH' | 'NEUTRAL' | 'BEARISH',
                    'confidence': 0.0-1.0,
                    'key_factors': [...],
                    'risk_factors': [...]
                },
                'spx_sentiment': {...},
                'comparison': {
                    'divergence': 'ALIGNED' | 'MILD' | 'STRONG',
                    'implications': str,
                    'correlation_regime': str
                },
                'overall_assessment': str
            }
        """
        logger.info("Analyzing market sentiment (KOSPI + SPX)...")

        # Prepare prompt
        prompt = self._build_sentiment_prompt(kospi_data, spx_data, market_context)

        try:
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2500,
                temperature=0.3,
                system=self._get_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Parse response
            response_text = message.content[0].text
            result = self._parse_sentiment_response(response_text)

            logger.info(f"KOSPI Sentiment: {result.get('kospi_sentiment', {}).get('sentiment', 'N/A')}")
            logger.info(f"SPX Sentiment: {result.get('spx_sentiment', {}).get('sentiment', 'N/A')}")

            return result

        except Exception as e:
            logger.error(f"Market sentiment analysis failed: {e}")
            return {
                'kospi_sentiment': {'sentiment': 'ERROR', 'error': str(e)},
                'spx_sentiment': {'sentiment': 'ERROR', 'error': str(e)},
                'comparison': {'divergence': 'ERROR'},
                'overall_assessment': f"Failed to analyze sentiment: {e}"
            }

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 (행동경제학 전문가)"""
        return """You are a behavioral finance specialist and market sentiment analyst with expertise in:
- Behavioral Finance (Kahneman & Tversky)
- Market Sentiment Indicators (Baker & Wurgler 2006)
- Fear & Greed Index interpretation
- Put/Call Ratio analysis
- Cross-market sentiment comparison
- Korean market characteristics (retail investor dominance, FX sensitivity)
- US market characteristics (institutional dominance, global risk-off flows)

Your task is to analyze market sentiment separately for KOSPI and SPX, then compare:
1. KOSPI-specific sentiment drivers (FX, retail flows, Samsung/Hynix)
2. SPX-specific sentiment drivers (Fed policy, tech sector, credit spreads)
3. Divergence analysis (Are the two markets decoupling? Why?)
4. Cross-market implications (What does KOSPI sentiment tell us about SPX and vice versa?)

Output in JSON format with keys: kospi_sentiment, spx_sentiment, comparison, overall_assessment.
"""

    def _build_sentiment_prompt(
        self,
        kospi_data: Dict,
        spx_data: Dict,
        context: Dict
    ) -> str:
        """정서 분석 프롬프트 구성"""
        prompt_parts = []

        # 1. KOSPI Market Data
        prompt_parts.append("## KOSPI Market Data\n")
        prompt_parts.append(self._format_market_data(kospi_data, "KOSPI"))

        # 2. SPX Market Data
        prompt_parts.append("\n## SPX Market Data\n")
        prompt_parts.append(self._format_market_data(spx_data, "SPX"))

        # 3. Global Context
        prompt_parts.append("\n## Global Market Context\n")
        prompt_parts.append(f"- Overall Regime: {context.get('regime', 'Unknown')}")
        prompt_parts.append(f"- Risk Score: {context.get('risk_score', 'N/A')}/100")
        prompt_parts.append(f"- VIX: {context.get('vix', 'N/A')}")
        if 'fear_greed_index' in context:
            prompt_parts.append(f"- Fear & Greed Index: {context['fear_greed_index']}")
        if 'liquidity_signal' in context:
            prompt_parts.append(f"- Liquidity Signal: {context['liquidity_signal']}")

        # 4. Analysis Questions
        prompt_parts.append("\n## Analysis Questions\n")
        prompt_parts.append("### KOSPI-Specific:")
        prompt_parts.append("1. What is the current sentiment for Korean equities?")
        prompt_parts.append("2. Key drivers: FX (USDKRW), Samsung/SK Hynix, foreign flows?")
        prompt_parts.append("3. Retail investor sentiment vs institutional flows?")
        prompt_parts.append("4. Sector rotation signals (bank vs tech vs export)?")

        prompt_parts.append("\n### SPX-Specific:")
        prompt_parts.append("1. What is the current sentiment for US equities?")
        prompt_parts.append("2. Key drivers: Fed policy, tech sector, credit spreads?")
        prompt_parts.append("3. Risk-on vs risk-off positioning?")
        prompt_parts.append("4. Market breadth and participation?")

        prompt_parts.append("\n### Cross-Market:")
        prompt_parts.append("1. Are KOSPI and SPX sentiments aligned or diverging?")
        prompt_parts.append("2. If diverging, what explains the decoupling?")
        prompt_parts.append("3. Which market is leading/lagging?")
        prompt_parts.append("4. What are the portfolio implications?")

        # 5. Required Output Format
        prompt_parts.append("\n## Required Output Format\n")
        prompt_parts.append("""```json
{
  "kospi_sentiment": {
    "sentiment": "BULLISH" | "NEUTRAL" | "BEARISH",
    "confidence": 0.0-1.0,
    "key_factors": [
      "Factor 1 (e.g., Strong Samsung earnings)",
      "Factor 2 (e.g., Weak USDKRW)"
    ],
    "risk_factors": [
      "Risk 1",
      "Risk 2"
    ],
    "sector_rotation": "Tech-led" | "Defensive" | "Cyclical"
  },
  "spx_sentiment": {
    "sentiment": "BULLISH" | "NEUTRAL" | "BEARISH",
    "confidence": 0.0-1.0,
    "key_factors": [...],
    "risk_factors": [...],
    "market_breadth": "Strong" | "Weak" | "Mixed"
  },
  "comparison": {
    "divergence": "ALIGNED" | "MILD" | "STRONG",
    "divergence_explanation": "Why are they aligned/diverging?",
    "implications": "What should investors do?",
    "correlation_regime": "HIGH" | "MEDIUM" | "LOW"
  },
  "overall_assessment": "Comprehensive summary of both markets and their interplay..."
}
```""")

        return "\n".join(prompt_parts)

    def _format_market_data(self, data: Dict, market_name: str) -> str:
        """시장 데이터 포맷팅"""
        lines = []

        # Helper function to safely format numeric values
        def safe_format(value, format_spec):
            if isinstance(value, (int, float)):
                return f"{value:{format_spec}}"
            else:
                return str(value)

        # Price & Returns
        if 'current_price' in data:
            lines.append(f"- Current Price: {safe_format(data['current_price'], '.2f')}")
        if 'ytd_return' in data:
            lines.append(f"- YTD Return: {safe_format(data['ytd_return'], '.2f')}%")
        if 'momentum' in data:
            lines.append(f"- Momentum: {safe_format(data['momentum'], '.2f')}%")

        # Volatility
        if 'volatility' in data:
            lines.append(f"- Volatility: {safe_format(data['volatility'], '.2f')}%")

        # Sentiment Indicators
        if 'fear_greed' in data:
            lines.append(f"- Fear & Greed: {data['fear_greed']}")
        if 'put_call_ratio' in data:
            lines.append(f"- Put/Call Ratio: {safe_format(data['put_call_ratio'], '.2f')}")

        # Volume & Flows
        if 'volume_trend' in data:
            lines.append(f"- Volume Trend: {data['volume_trend']}")
        if 'foreign_flow' in data:
            lines.append(f"- Foreign Flow: {data['foreign_flow']}")

        # Valuation
        if 'pe_ratio' in data:
            lines.append(f"- P/E Ratio: {safe_format(data['pe_ratio'], '.1f')}")
        if 'fair_value_gap' in data:
            lines.append(f"- Fair Value Gap: {safe_format(data['fair_value_gap'], '.1f')}%")

        # Sector/Constituent Info
        if 'sector_leaders' in data:
            lines.append(f"- Sector Leaders: {', '.join(data['sector_leaders'])}")
        if 'top_performers' in data:
            lines.append(f"- Top Performers: {', '.join(data['top_performers'])}")

        return "\n".join(lines) if lines else "- No data available"

    def _parse_sentiment_response(self, response: str) -> Dict:
        """응답 파싱"""
        try:
            # Extract JSON from markdown code block if present
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            result = json.loads(json_str)

            # Validate structure
            if 'kospi_sentiment' not in result or 'spx_sentiment' not in result:
                raise ValueError("Missing kospi_sentiment or spx_sentiment")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return raw response as assessment
            return {
                'kospi_sentiment': {'sentiment': 'NEUTRAL', 'confidence': 0.5},
                'spx_sentiment': {'sentiment': 'NEUTRAL', 'confidence': 0.5},
                'comparison': {'divergence': 'UNKNOWN'},
                'overall_assessment': response,
                'parse_error': str(e)
            }

    def analyze_divergence(
        self,
        kospi_sentiment: str,
        spx_sentiment: str,
        correlation: float
    ) -> Dict:
        """
        시장 간 정서 괴리도 분석

        Args:
            kospi_sentiment: 'BULLISH' | 'NEUTRAL' | 'BEARISH'
            spx_sentiment: 'BULLISH' | 'NEUTRAL' | 'BEARISH'
            correlation: 최근 상관계수

        Returns:
            {
                'divergence_level': 'ALIGNED' | 'MILD' | 'STRONG',
                'warning': bool,
                'explanation': str
            }
        """
        sentiment_map = {'BULLISH': 1, 'NEUTRAL': 0, 'BEARISH': -1}
        kospi_score = sentiment_map.get(kospi_sentiment, 0)
        spx_score = sentiment_map.get(spx_sentiment, 0)

        sentiment_diff = abs(kospi_score - spx_score)

        # Divergence classification
        if sentiment_diff == 0:
            divergence = 'ALIGNED'
            warning = False
            explanation = "KOSPI and SPX sentiments are aligned"
        elif sentiment_diff == 1:
            divergence = 'MILD'
            warning = correlation < 0.4  # Warning if low correlation
            explanation = "Moderate divergence between KOSPI and SPX"
        else:  # sentiment_diff == 2
            divergence = 'STRONG'
            warning = True
            explanation = "Strong divergence - KOSPI and SPX moving in opposite directions"

        return {
            'divergence_level': divergence,
            'warning': warning,
            'explanation': explanation,
            'sentiment_diff': sentiment_diff,
            'correlation': correlation
        }


if __name__ == "__main__":
    # Test sentiment analysis
    agent = MarketSentimentAgent()

    test_kospi_data = {
        'current_price': 2580.5,
        'ytd_return': 12.3,
        'volatility': 18.5,
        'momentum': 5.2,
        'foreign_flow': 'Buying',
        'sector_leaders': ['Semiconductors', 'Battery'],
        'pe_ratio': 11.2,
        'fair_value_gap': -8.5
    }

    test_spx_data = {
        'current_price': 4850.2,
        'ytd_return': 15.7,
        'volatility': 14.2,
        'momentum': 3.8,
        'market_breadth': 'Strong',
        'sector_leaders': ['Technology', 'Communication Services'],
        'pe_ratio': 21.5,
        'fair_value_gap': 5.2
    }

    test_context = {
        'regime': 'Bull (Low Vol)',
        'risk_score': 35.2,
        'vix': 13.5,
        'fear_greed_index': 72,
        'liquidity_signal': 'POSITIVE'
    }

    result = agent.analyze_market_sentiment(
        test_kospi_data,
        test_spx_data,
        test_context
    )

    print("\n=== Market Sentiment Analysis ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
