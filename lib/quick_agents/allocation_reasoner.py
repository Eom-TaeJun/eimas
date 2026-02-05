"""
Allocation Reasoner Agent
==========================
자산배분 논리 분석 에이전트 (Perplexity API)

Purpose:
- 배분 결정의 근거 설명
- 계산식의 학술적 타당성 검증
- 최신 학계/업계 의견 조회

Economic Foundation:
- Ilmanen (2011): Expected Returns
- Damodaran (2023): Equity Risk Premium
- AQR Capital Management research
- Research Affiliates insights
"""

import requests
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class AllocationReasoner:
    """자산배분 논리 분석 에이전트"""

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
        self.model = "sonar-pro"  # Perplexity online search model

    def analyze_reasoning(
        self,
        allocation_decision: Dict,
        calculation_details: List[str],
        market_context: Dict
    ) -> Dict:
        """
        배분 논리 분석

        Args:
            allocation_decision: 배분 결정 (weights, metrics)
            calculation_details: 계산 과정 상세
            market_context: 시장 컨텍스트

        Returns:
            {
                'reasoning_quality': 'STRONG' | 'MODERATE' | 'WEAK',
                'academic_support': {...},
                'alternative_views': [...],
                'risk_factors': [...],
                'overall_assessment': str
            }
        """
        logger.info("Analyzing allocation reasoning with Perplexity API...")

        # Build query for Perplexity
        query = self._build_reasoning_query(
            allocation_decision,
            calculation_details,
            market_context
        )

        try:
            # Call Perplexity API
            response = self._call_perplexity_api(query)

            # Parse response
            result = self._parse_reasoning_response(response)

            logger.info(f"Reasoning quality: {result.get('reasoning_quality', 'N/A')}")

            return result

        except Exception as e:
            logger.error(f"Reasoning analysis failed: {e}")
            return {
                'reasoning_quality': 'ERROR',
                'error': str(e),
                'overall_assessment': f"Failed to analyze reasoning: {e}"
            }

    def _build_reasoning_query(
        self,
        decision: Dict,
        details: List[str],
        context: Dict
    ) -> str:
        """Perplexity 쿼리 구성"""
        stock_pct = decision.get('stock', 0) * 100
        bond_pct = decision.get('bond', 0) * 100
        regime = context.get('regime', 'Unknown')
        risk_score = context.get('risk_score', 'N/A')

        query = f"""
Based on latest academic research and industry best practices in 2024-2026:

Asset Allocation Decision:
- Stock: {stock_pct:.1f}%
- Bond: {bond_pct:.1f}%
- Expected Return: {decision.get('expected_return', 0)*100:.2f}%
- Sharpe Ratio: {decision.get('sharpe_ratio', 0):.2f}

Market Context:
- Regime: {regime}
- Risk Score: {risk_score}/100

Questions:
1. Is this stock/bond allocation ratio appropriate for the current market regime?
2. What do recent academic papers (2020-2026) say about optimal asset allocation in similar conditions?
3. What are the main risk factors to consider?
4. Are there alternative allocation strategies that might be more suitable?
5. What do leading asset managers (AQR, Bridgewater, BlackRock) recommend for similar market conditions?

Please provide evidence-based analysis with specific references to recent research.
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
                    "content": "You are a quantitative researcher specializing in asset allocation and portfolio theory. Provide evidence-based analysis with specific references."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.2,
            "max_tokens": 2000
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    def _parse_reasoning_response(self, api_response: Dict) -> Dict:
        """응답 파싱"""
        try:
            content = api_response['choices'][0]['message']['content']
            citations = api_response.get('citations', [])

            # Analyze response content
            result = {
                'reasoning_quality': self._assess_reasoning_quality(content),
                'academic_support': self._extract_academic_support(content, citations),
                'alternative_views': self._extract_alternatives(content),
                'risk_factors': self._extract_risk_factors(content),
                'overall_assessment': content,
                'citations': citations
            }

            return result

        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse Perplexity response: {e}")
            return {
                'reasoning_quality': 'ERROR',
                'error': str(e),
                'overall_assessment': str(api_response)
            }

    def _assess_reasoning_quality(self, content: str) -> str:
        """논리 품질 평가"""
        # Simple heuristic based on response
        content_lower = content.lower()

        strong_indicators = [
            'research shows', 'studies indicate', 'according to',
            'empirical evidence', 'academic', 'published'
        ]
        weak_indicators = [
            'might', 'could be', 'possibly', 'unclear'
        ]

        strong_count = sum(1 for ind in strong_indicators if ind in content_lower)
        weak_count = sum(1 for ind in weak_indicators if ind in content_lower)

        if strong_count >= 3:
            return 'STRONG'
        elif weak_count >= 3:
            return 'WEAK'
        else:
            return 'MODERATE'

    def _extract_academic_support(self, content: str, citations: List) -> Dict:
        """학술적 근거 추출"""
        return {
            'num_citations': len(citations),
            'key_findings': [c for c in citations[:3]],  # Top 3 citations
            'has_recent_research': any('202' in str(c) for c in citations)
        }

    def _extract_alternatives(self, content: str) -> List[str]:
        """대안 전략 추출"""
        alternatives = []

        # Look for alternative strategies mentioned
        alt_keywords = [
            'alternative', 'instead', 'better', 'preferable',
            'consider', 'another approach'
        ]

        lines = content.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in alt_keywords):
                alternatives.append(line.strip())

        return alternatives[:3]  # Top 3 alternatives

    def _extract_risk_factors(self, content: str) -> List[str]:
        """리스크 요인 추출"""
        risk_factors = []

        risk_keywords = [
            'risk', 'concern', 'warning', 'caution',
            'drawback', 'limitation', 'downside'
        ]

        lines = content.split('\n')
        for line in lines:
            if any(kw in line.lower() for kw in risk_keywords):
                risk_factors.append(line.strip())

        return risk_factors[:5]  # Top 5 risks


if __name__ == "__main__":
    # Test reasoning
    reasoner = AllocationReasoner()

    test_decision = {
        'stock': 0.67,
        'bond': 0.33,
        'expected_return': 0.0818,
        'sharpe_ratio': 0.77
    }

    test_details = [
        "Base Return: 8.00%",
        "Regime Adjustment (Bull): +0.00%",
        "Risk Score Adjustment: +1.50%"
    ]

    test_context = {
        'regime': 'Bull (Low Vol)',
        'risk_score': 12.6
    }

    result = reasoner.analyze_reasoning(test_decision, test_details, test_context)

    print("\n=== Allocation Reasoning Analysis ===")
    print(json.dumps(result, indent=2))
