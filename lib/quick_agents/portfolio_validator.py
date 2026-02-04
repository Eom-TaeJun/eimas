"""
Portfolio Validator Agent
=========================
포트폴리오 이론 검증 에이전트 (Claude API)

Purpose:
- Full JSON의 자산배분 결과 검증
- 경제학 이론 (Markowitz, Black-Litterman, Risk Parity) 적합성 확인
- 제약 조건 위반 여부 체크

Economic Foundation:
- Markowitz (1952): Mean-Variance Optimization
- Black-Litterman (1992): Bayesian Portfolio
- Qian (2005): Risk Parity
- DeMiguel et al. (2009): Optimal vs Naive Diversification
"""

import anthropic
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class PortfolioValidator:
    """포트폴리오 검증 에이전트"""

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

    def validate_portfolio(
        self,
        allocation_result: Dict,
        market_context: Dict,
        constraints: Optional[Dict] = None
    ) -> Dict:
        """
        포트폴리오 배분 검증

        Args:
            allocation_result: 자산배분 결과
            market_context: 시장 컨텍스트 (regime, risk, etc.)
            constraints: 제약 조건 (min/max weights, turnover cap 등)

        Returns:
            {
                'validation_result': 'PASS' | 'WARNING' | 'FAIL',
                'theory_compliance': {...},
                'risk_assessment': {...},
                'recommendations': [...],
                'reasoning': str
            }
        """
        logger.info("Validating portfolio allocation with Claude API...")

        # Prepare prompt
        prompt = self._build_validation_prompt(
            allocation_result,
            market_context,
            constraints
        )

        try:
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,  # Lower temperature for analytical tasks
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
            result = self._parse_validation_response(response_text)

            logger.info(f"Validation result: {result.get('validation_result', 'N/A')}")

            return result

        except Exception as e:
            logger.error(f"Portfolio validation failed: {e}")
            return {
                'validation_result': 'ERROR',
                'error': str(e),
                'reasoning': f"Failed to validate portfolio: {e}"
            }

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 (경제학 전문가 역할)"""
        return """You are a senior portfolio manager and quantitative researcher with expertise in:
- Modern Portfolio Theory (Markowitz 1952)
- Black-Litterman model (1992)
- Risk Parity (Qian 2005)
- Factor models (Fama-French)
- Behavioral finance

Your task is to validate asset allocation decisions based on:
1. Economic theory compliance
2. Risk-return trade-offs
3. Market regime appropriateness
4. Constraint satisfaction
5. Practical implementation feasibility

Provide analytical, evidence-based assessments with clear reasoning.
Output in JSON format with keys: validation_result, theory_compliance, risk_assessment, recommendations, reasoning."""

    def _build_validation_prompt(
        self,
        allocation: Dict,
        context: Dict,
        constraints: Optional[Dict]
    ) -> str:
        """검증 프롬프트 구성"""
        prompt_parts = []

        # 1. Allocation summary
        prompt_parts.append("## Asset Allocation to Validate\n")
        prompt_parts.append(json.dumps(allocation, indent=2))

        # 2. Market context
        prompt_parts.append("\n## Market Context\n")
        prompt_parts.append(f"- Regime: {context.get('regime', 'Unknown')}")
        prompt_parts.append(f"- Risk Score: {context.get('risk_score', 'N/A')}/100")
        prompt_parts.append(f"- Volatility: {context.get('volatility', 'N/A')}")
        if 'bubble_status' in context:
            prompt_parts.append(f"- Bubble Risk: {context['bubble_status']}")
        if 'sector_rotation' in context:
            prompt_parts.append(f"- Sector Rotation: {context['sector_rotation']}")

        # 3. Constraints (if any)
        if constraints:
            prompt_parts.append("\n## Constraints\n")
            prompt_parts.append(json.dumps(constraints, indent=2))

        # 4. Validation questions
        prompt_parts.append("\n## Validation Questions\n")
        prompt_parts.append("1. Does this allocation align with Modern Portfolio Theory principles?")
        prompt_parts.append("2. Is the stock/bond ratio appropriate for the current market regime?")
        prompt_parts.append("3. Are there any constraint violations?")
        prompt_parts.append("4. Does the risk-adjusted return (Sharpe ratio) seem reasonable?")
        prompt_parts.append("5. Are there any red flags or concerns?")

        prompt_parts.append("\n## Required Output Format\n")
        prompt_parts.append("""```json
{
  "validation_result": "PASS" | "WARNING" | "FAIL",
  "theory_compliance": {
    "markowitz_mvo": "compliant" | "non-compliant",
    "risk_parity": "appropriate" | "inappropriate",
    "diversification": "adequate" | "inadequate"
  },
  "risk_assessment": {
    "overall_risk": "low" | "medium" | "high",
    "concerns": ["concern1", "concern2"],
    "sharpe_adequacy": "good" | "acceptable" | "poor"
  },
  "recommendations": [
    "Recommendation 1",
    "Recommendation 2"
  ],
  "reasoning": "Detailed explanation of the validation result..."
}
```""")

        return "\n".join(prompt_parts)

    def _parse_validation_response(self, response: str) -> Dict:
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
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return raw response as reasoning
            return {
                'validation_result': 'WARNING',
                'reasoning': response,
                'parse_error': str(e)
            }


if __name__ == "__main__":
    # Test validation
    validator = PortfolioValidator()

    test_allocation = {
        'stock': 0.67,
        'bond': 0.33,
        'expected_return': 0.0818,
        'volatility': 0.1066,
        'sharpe_ratio': 0.77
    }

    test_context = {
        'regime': 'Bull (Low Vol)',
        'risk_score': 12.6,
        'volatility': 0.16,
        'bubble_status': 'NONE'
    }

    result = validator.validate_portfolio(test_allocation, test_context)

    print("\n=== Portfolio Validation Result ===")
    print(json.dumps(result, indent=2))
