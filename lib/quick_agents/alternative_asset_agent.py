"""
Alternative Asset Agent
=======================
대체자산 판단 에이전트 (Perplexity API)

Purpose:
- 크립토 자산 (BTC, ETH, Stablecoin) 시장 분석
- 금/원자재 시장 분석
- RWA (Real World Assets) 토큰화 자산 분석
- 대체자산 간 상관관계 및 포트폴리오 역할 평가

Economic Foundation:
- Gorton & Rouwenhorst (2006): Facts and Fantasies about Commodity Futures
- Baur & Lucey (2010): Gold as Safe Haven
- Genius Act: Stablecoin-Liquidity Nexus
- RWA Tokenization: BlackRock BUIDL, Ondo Finance
"""

import requests
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class AlternativeAssetAgent:
    """대체자산 판단 에이전트"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Perplexity API key (None이면 환경변수 사용)
        """
        import os
        self.api_key = api_key or os.getenv('PERPLEXITY_API_KEY')
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not found")

        self.api_url = "https://api.perplexity.ai/chat/completions"
        self.model = "llama-3.1-sonar-large-128k-online"  # Online search model

    def analyze_alternative_assets(
        self,
        crypto_data: Dict,
        commodity_data: Dict,
        market_context: Dict
    ) -> Dict:
        """
        대체자산 분석

        Args:
            crypto_data: 크립토 자산 데이터 (BTC, ETH, stablecoins)
            commodity_data: 원자재 데이터 (Gold, Oil, etc.)
            market_context: 시장 컨텍스트

        Returns:
            {
                'crypto_assessment': {
                    'recommendation': 'BULLISH' | 'NEUTRAL' | 'BEARISH',
                    'btc_outlook': {...},
                    'eth_outlook': {...},
                    'stablecoin_regime': {...},
                    'key_catalysts': [...]
                },
                'commodity_assessment': {
                    'gold_role': 'SAFE_HAVEN' | 'INFLATION_HEDGE' | 'NEUTRAL',
                    'energy_outlook': {...},
                    'diversification_benefit': str
                },
                'rwa_assessment': {
                    'tokenization_trend': 'ACCELERATING' | 'STABLE' | 'SLOWING',
                    'key_products': [...],
                    'institutional_adoption': str
                },
                'portfolio_role': {
                    'recommended_allocation': {...},
                    'correlation_benefits': str,
                    'risk_considerations': [...]
                },
                'overall_assessment': str,
                'citations': [...]
            }
        """
        logger.info("Analyzing alternative assets with Perplexity API...")

        # Build query
        query = self._build_alternative_asset_query(
            crypto_data,
            commodity_data,
            market_context
        )

        try:
            # Call Perplexity API
            response = self._call_perplexity_api(query)

            # Parse response
            result = self._parse_alternative_asset_response(response)

            logger.info(f"Crypto Recommendation: {result.get('crypto_assessment', {}).get('recommendation', 'N/A')}")
            logger.info(f"Gold Role: {result.get('commodity_assessment', {}).get('gold_role', 'N/A')}")

            return result

        except Exception as e:
            logger.error(f"Alternative asset analysis failed: {e}")
            return {
                'crypto_assessment': {'recommendation': 'ERROR', 'error': str(e)},
                'commodity_assessment': {'gold_role': 'ERROR'},
                'rwa_assessment': {'tokenization_trend': 'ERROR'},
                'portfolio_role': {'recommended_allocation': {}},
                'overall_assessment': f"Failed to analyze alternative assets: {e}"
            }

    def _build_alternative_asset_query(
        self,
        crypto: Dict,
        commodity: Dict,
        context: Dict
    ) -> str:
        """Perplexity 쿼리 구성"""
        # Extract key data
        btc_price = crypto.get('btc_price', 'N/A')
        btc_return = crypto.get('btc_ytd_return', 'N/A')
        eth_price = crypto.get('eth_price', 'N/A')
        stablecoin_supply = crypto.get('stablecoin_supply_change', 'N/A')

        gold_price = commodity.get('gold_price', 'N/A')
        gold_return = commodity.get('gold_ytd_return', 'N/A')

        regime = context.get('regime', 'Unknown')
        risk_score = context.get('risk_score', 'N/A')
        liquidity_signal = context.get('liquidity_signal', 'N/A')

        query = f"""
Based on latest market research and institutional reports in 2024-2026:

## Current Market Data

### Crypto Assets:
- Bitcoin: ${btc_price}, YTD Return: {btc_return}%
- Ethereum: ${eth_price}
- Stablecoin Supply Change: {stablecoin_supply}%

### Commodities:
- Gold: ${gold_price}, YTD Return: {gold_return}%

### Macro Context:
- Market Regime: {regime}
- Risk Score: {risk_score}/100
- Liquidity Signal: {liquidity_signal}

## Analysis Questions:

### 1. Crypto Market (Bitcoin, Ethereum):
- What is the current institutional sentiment toward BTC/ETH?
- What do recent BlackRock, Fidelity, Grayscale reports say?
- Are spot ETF flows supporting the bullish case?
- What are the key catalysts for 2025-2026 (halving, regulatory clarity, etc.)?
- What are the main downside risks?

### 2. Stablecoin Market:
- Is stablecoin supply expanding or contracting?
- What does Genius Act framework suggest about liquidity conditions?
- Are USDC, USDT, DAI supply trends aligned with risk appetite?

### 3. Gold & Commodities:
- Is gold acting as a safe haven or inflation hedge?
- What do central bank gold purchases indicate?
- How are commodities positioned in current regime?

### 4. RWA (Real World Assets) Tokenization:
- What is the latest on BlackRock BUIDL, Ondo Finance OUSG, Franklin Templeton FOBXX?
- Is institutional adoption of tokenized treasuries accelerating?
- What are the regulatory developments?

### 5. Portfolio Implications:
- Should investors allocate to crypto/gold in current regime?
- What is the optimal allocation range (if any)?
- What are the correlation benefits vs traditional 60/40?
- What are the key risk considerations?

Please provide evidence-based analysis with specific references to recent institutional research, regulatory developments, and market data.
"""
        return query.strip()

    def _call_perplexity_api(self, query: str) -> Dict:
        """Perplexity API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a quantitative analyst specializing in alternative assets (crypto, commodities, RWA). Provide evidence-based analysis with specific references to institutional research."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2500,
            "search_domain_filter": ["coindesk.com", "theblock.co", "bloomberg.com", "ft.com"],
            "return_citations": True
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    def _parse_alternative_asset_response(self, api_response: Dict) -> Dict:
        """응답 파싱"""
        try:
            content = api_response['choices'][0]['message']['content']
            citations = api_response.get('citations', [])

            # Analyze response content
            result = {
                'crypto_assessment': self._extract_crypto_assessment(content),
                'commodity_assessment': self._extract_commodity_assessment(content),
                'rwa_assessment': self._extract_rwa_assessment(content),
                'portfolio_role': self._extract_portfolio_role(content),
                'overall_assessment': content,
                'citations': citations
            }

            return result

        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Perplexity response: {e}")
            return {
                'crypto_assessment': {'recommendation': 'ERROR', 'error': str(e)},
                'commodity_assessment': {'gold_role': 'ERROR'},
                'rwa_assessment': {'tokenization_trend': 'ERROR'},
                'portfolio_role': {},
                'overall_assessment': str(api_response)
            }

    def _extract_crypto_assessment(self, content: str) -> Dict:
        """크립토 평가 추출"""
        content_lower = content.lower()

        # Recommendation heuristic
        bullish_signals = ['bullish', 'positive', 'strong inflows', 'institutional buying', 'breakout']
        bearish_signals = ['bearish', 'negative', 'outflows', 'regulatory risk', 'breakdown']

        bullish_count = sum(1 for sig in bullish_signals if sig in content_lower)
        bearish_count = sum(1 for sig in bearish_signals if sig in content_lower)

        if bullish_count > bearish_count + 1:
            recommendation = 'BULLISH'
        elif bearish_count > bullish_count + 1:
            recommendation = 'BEARISH'
        else:
            recommendation = 'NEUTRAL'

        # Extract key catalysts
        catalysts = []
        catalyst_keywords = ['etf', 'halving', 'regulation', 'adoption', 'institutional']
        lines = content.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in catalyst_keywords):
                catalysts.append(line.strip())

        return {
            'recommendation': recommendation,
            'key_catalysts': catalysts[:3],
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count
        }

    def _extract_commodity_assessment(self, content: str) -> Dict:
        """원자재 평가 추출"""
        content_lower = content.lower()

        # Gold role heuristic
        if 'safe haven' in content_lower or 'flight to quality' in content_lower:
            gold_role = 'SAFE_HAVEN'
        elif 'inflation hedge' in content_lower or 'inflation protection' in content_lower:
            gold_role = 'INFLATION_HEDGE'
        else:
            gold_role = 'NEUTRAL'

        # Diversification benefit
        diversification_keywords = ['diversification', 'uncorrelated', 'low correlation', 'hedge']
        diversification_benefit = 'Yes' if any(kw in content_lower for kw in diversification_keywords) else 'Limited'

        return {
            'gold_role': gold_role,
            'diversification_benefit': diversification_benefit
        }

    def _extract_rwa_assessment(self, content: str) -> Dict:
        """RWA 평가 추출"""
        content_lower = content.lower()

        # Tokenization trend
        accel_keywords = ['accelerating', 'growing', 'expanding', 'increasing adoption']
        slow_keywords = ['slowing', 'declining', 'regulatory headwinds']

        if any(kw in content_lower for kw in accel_keywords):
            trend = 'ACCELERATING'
        elif any(kw in content_lower for kw in slow_keywords):
            trend = 'SLOWING'
        else:
            trend = 'STABLE'

        # Key products mentioned
        rwa_products = []
        product_keywords = ['blackrock buidl', 'ondo', 'franklin templeton', 'ousg', 'fobxx', 'paxg']
        for keyword in product_keywords:
            if keyword in content_lower:
                rwa_products.append(keyword.upper())

        return {
            'tokenization_trend': trend,
            'key_products': rwa_products
        }

    def _extract_portfolio_role(self, content: str) -> Dict:
        """포트폴리오 역할 추출"""
        content_lower = content.lower()

        # Recommended allocation heuristic
        allocation = {}

        # Crypto allocation
        if 'no crypto' in content_lower or 'avoid crypto' in content_lower:
            allocation['crypto'] = '0%'
        elif 'small allocation' in content_lower or '1-5%' in content_lower:
            allocation['crypto'] = '1-5%'
        elif 'moderate' in content_lower:
            allocation['crypto'] = '5-10%'
        else:
            allocation['crypto'] = 'Not specified'

        # Gold allocation
        if 'gold' in content_lower:
            if '5-10%' in content_lower:
                allocation['gold'] = '5-10%'
            elif '10%' in content_lower:
                allocation['gold'] = '10%'
            else:
                allocation['gold'] = 'Consider allocation'

        # Correlation benefits
        if 'low correlation' in content_lower or 'diversification' in content_lower:
            correlation_benefits = 'Provides diversification benefits'
        elif 'high correlation' in content_lower or 'no benefit' in content_lower:
            correlation_benefits = 'Limited diversification'
        else:
            correlation_benefits = 'Mixed correlation regime'

        # Risk considerations
        risk_keywords = ['volatility', 'regulatory', 'liquidity', 'custody', 'tax']
        risk_considerations = []
        lines = content.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in risk_keywords):
                risk_considerations.append(line.strip())

        return {
            'recommended_allocation': allocation,
            'correlation_benefits': correlation_benefits,
            'risk_considerations': risk_considerations[:3]
        }


if __name__ == "__main__":
    # Test alternative asset analysis
    agent = AlternativeAssetAgent()

    test_crypto_data = {
        'btc_price': 67500,
        'btc_ytd_return': 145.3,
        'eth_price': 3850,
        'eth_ytd_return': 92.1,
        'stablecoin_supply_change': 8.5
    }

    test_commodity_data = {
        'gold_price': 2385,
        'gold_ytd_return': 18.2,
        'oil_price': 82.5
    }

    test_context = {
        'regime': 'Bull (Low Vol)',
        'risk_score': 25.3,
        'liquidity_signal': 'POSITIVE'
    }

    result = agent.analyze_alternative_assets(
        test_crypto_data,
        test_commodity_data,
        test_context
    )

    print("\n=== Alternative Asset Analysis ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
