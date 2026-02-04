"""
Final Validator Agent
=====================
최종 검증 에이전트 (Claude API)

Purpose:
- 모든 Quick Agent 의견 종합
- Full 모드 진단과 Quick 모드 검증 결과 비교
- 최종 투자 권고 및 신뢰도 평가
- 리스크 경고 및 행동 권고

Economic Foundation:
- 다중 전문가 의견 집계 (Wisdom of Crowds)
- Bayesian 신뢰도 업데이트
- Risk-Adjusted Decision Making
"""

import anthropic
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class FinalValidator:
    """최종 검증 에이전트"""

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

    def validate_and_synthesize(
        self,
        full_mode_result: Dict,
        portfolio_validation: Dict,
        allocation_reasoning: Dict,
        market_sentiment: Dict,
        alternative_assets: Dict
    ) -> Dict:
        """
        최종 검증 및 종합

        Args:
            full_mode_result: Full 모드 결과 (JSON)
            portfolio_validation: 포트폴리오 검증 결과
            allocation_reasoning: 자산배분 논리 분석
            market_sentiment: 시장 정서 분석 (KOSPI + SPX)
            alternative_assets: 대체자산 분석

        Returns:
            {
                'validation_result': 'PASS' | 'CAUTION' | 'FAIL',
                'final_recommendation': 'BULLISH' | 'NEUTRAL' | 'BEARISH',
                'confidence': 0.0-1.0,
                'agent_consensus': {
                    'portfolio_vote': str,
                    'reasoning_vote': str,
                    'sentiment_vote': str,
                    'alternatives_vote': str,
                    'agreement_level': 'HIGH' | 'MEDIUM' | 'LOW'
                },
                'full_vs_quick_comparison': {
                    'alignment': 'ALIGNED' | 'DIVERGENT',
                    'key_differences': [...],
                    'confidence_adjustment': str
                },
                'risk_warnings': [...],
                'action_items': [...],
                'final_assessment': str
            }
        """
        logger.info("Running final validation and synthesis...")

        # Prepare prompt
        prompt = self._build_final_validation_prompt(
            full_mode_result,
            portfolio_validation,
            allocation_reasoning,
            market_sentiment,
            alternative_assets
        )

        try:
            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=3000,
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
            result = self._parse_final_validation_response(response_text)

            logger.info(f"Final Recommendation: {result.get('final_recommendation', 'N/A')}")
            logger.info(f"Confidence: {result.get('confidence', 0)*100:.0f}%")

            return result

        except Exception as e:
            logger.error(f"Final validation failed: {e}")
            return {
                'validation_result': 'ERROR',
                'final_recommendation': 'NEUTRAL',
                'confidence': 0.0,
                'error': str(e),
                'final_assessment': f"Failed to complete final validation: {e}"
            }

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트 (최종 검증 역할)"""
        return """You are a senior investment committee chairman synthesizing recommendations from multiple specialist agents.

Your task:
1. Review Full mode diagnosis (comprehensive EIMAS analysis)
2. Review Quick mode validation results from 4 specialist agents:
   - Portfolio Validator (economic theory compliance)
   - Allocation Reasoner (academic research support)
   - Market Sentiment Agent (KOSPI + SPX sentiment)
   - Alternative Assets Agent (crypto, gold, RWA analysis)
3. Identify agreements and disagreements among agents
4. Compare Full mode vs Quick mode conclusions
5. Assign final recommendation with confidence level
6. Flag risk warnings and suggest action items

Decision Framework:
- If all agents agree and align with Full mode → HIGH confidence
- If most agents agree (3/4) → MEDIUM confidence
- If agents disagree significantly → LOW confidence, recommend CAUTION
- If Quick mode strongly contradicts Full mode → Investigate discrepancy

Output in JSON format with keys: validation_result, final_recommendation, confidence, agent_consensus, full_vs_quick_comparison, risk_warnings, action_items, final_assessment.
"""

    def _build_final_validation_prompt(
        self,
        full_mode: Dict,
        portfolio_val: Dict,
        reasoning: Dict,
        sentiment: Dict,
        alternatives: Dict
    ) -> str:
        """최종 검증 프롬프트 구성"""
        prompt_parts = []

        # 1. Full Mode Summary
        prompt_parts.append("## Full Mode Diagnosis (EIMAS)\n")
        prompt_parts.append(f"- Final Recommendation: {full_mode.get('final_recommendation', 'N/A')}")
        prompt_parts.append(f"- Confidence: {full_mode.get('confidence', 0)*100:.0f}%")
        prompt_parts.append(f"- Market Regime: {full_mode.get('regime', {}).get('regime', 'N/A')}")
        prompt_parts.append(f"- Risk Score: {full_mode.get('risk_score', 'N/A')}/100")
        prompt_parts.append(f"- Risk Level: {full_mode.get('risk_level', 'N/A')}")

        if 'full_mode_position' in full_mode:
            prompt_parts.append(f"- Full Mode Position: {full_mode['full_mode_position']}")
        if 'reference_mode_position' in full_mode:
            prompt_parts.append(f"- Reference Mode Position: {full_mode['reference_mode_position']}")
            prompt_parts.append(f"- Modes Agree: {full_mode.get('modes_agree', False)}")

        # 2. Agent Validation Results
        prompt_parts.append("\n## Quick Mode Agent Validations\n")

        # Portfolio Validator
        prompt_parts.append("### 1. Portfolio Validator (Economic Theory):")
        pv_result = portfolio_val.get('validation_result', 'N/A')
        pv_theory = portfolio_val.get('theory_compliance', {})
        prompt_parts.append(f"- Result: {pv_result}")
        prompt_parts.append(f"- Markowitz MVO: {pv_theory.get('markowitz_mvo', 'N/A')}")
        prompt_parts.append(f"- Risk Parity: {pv_theory.get('risk_parity', 'N/A')}")
        prompt_parts.append(f"- Diversification: {pv_theory.get('diversification', 'N/A')}")

        # Allocation Reasoner
        prompt_parts.append("\n### 2. Allocation Reasoner (Academic Research):")
        ar_quality = reasoning.get('reasoning_quality', 'N/A')
        ar_support = reasoning.get('academic_support', {})
        prompt_parts.append(f"- Reasoning Quality: {ar_quality}")
        prompt_parts.append(f"- Citations: {ar_support.get('num_citations', 0)}")
        prompt_parts.append(f"- Recent Research: {ar_support.get('has_recent_research', False)}")

        # Market Sentiment
        prompt_parts.append("\n### 3. Market Sentiment Agent:")
        kospi_sent = sentiment.get('kospi_sentiment', {})
        spx_sent = sentiment.get('spx_sentiment', {})
        comparison = sentiment.get('comparison', {})
        prompt_parts.append(f"- KOSPI Sentiment: {kospi_sent.get('sentiment', 'N/A')} (Confidence: {kospi_sent.get('confidence', 0)*100:.0f}%)")
        prompt_parts.append(f"- SPX Sentiment: {spx_sent.get('sentiment', 'N/A')} (Confidence: {spx_sent.get('confidence', 0)*100:.0f}%)")
        prompt_parts.append(f"- Divergence: {comparison.get('divergence', 'N/A')}")

        # Alternative Assets
        prompt_parts.append("\n### 4. Alternative Assets Agent:")
        crypto_assess = alternatives.get('crypto_assessment', {})
        commodity_assess = alternatives.get('commodity_assessment', {})
        rwa_assess = alternatives.get('rwa_assessment', {})
        prompt_parts.append(f"- Crypto Recommendation: {crypto_assess.get('recommendation', 'N/A')}")
        prompt_parts.append(f"- Gold Role: {commodity_assess.get('gold_role', 'N/A')}")
        prompt_parts.append(f"- RWA Trend: {rwa_assess.get('tokenization_trend', 'N/A')}")

        # 3. Validation Questions
        prompt_parts.append("\n## Final Validation Questions\n")
        prompt_parts.append("1. Do the 4 Quick mode agents agree with each other?")
        prompt_parts.append("2. Does Quick mode validation align with Full mode diagnosis?")
        prompt_parts.append("3. Are there any red flags or contradictions?")
        prompt_parts.append("4. What is the appropriate confidence level?")
        prompt_parts.append("5. What are the key risk warnings?")
        prompt_parts.append("6. What should investors do (action items)?")

        # 4. Required Output Format
        prompt_parts.append("\n## Required Output Format\n")
        prompt_parts.append("""```json
{
  "validation_result": "PASS" | "CAUTION" | "FAIL",
  "final_recommendation": "BULLISH" | "NEUTRAL" | "BEARISH",
  "confidence": 0.0-1.0,
  "agent_consensus": {
    "portfolio_vote": "PASS/WARNING/FAIL",
    "reasoning_vote": "STRONG/MODERATE/WEAK",
    "sentiment_vote": "BULLISH/NEUTRAL/BEARISH (KOSPI), BULLISH/NEUTRAL/BEARISH (SPX)",
    "alternatives_vote": "BULLISH/NEUTRAL/BEARISH",
    "agreement_level": "HIGH" | "MEDIUM" | "LOW",
    "disagreements": ["Agent X says...", "Agent Y says..."]
  },
  "full_vs_quick_comparison": {
    "alignment": "ALIGNED" | "DIVERGENT",
    "full_recommendation": "...",
    "quick_recommendation": "...",
    "key_differences": ["Difference 1", "Difference 2"],
    "confidence_adjustment": "Increased/Decreased/Unchanged"
  },
  "risk_warnings": [
    "Warning 1: High market concentration",
    "Warning 2: Regulatory uncertainty",
    "Warning 3: Divergent sentiment (KOSPI vs SPX)"
  ],
  "action_items": [
    "Action 1: Monitor Fed policy closely",
    "Action 2: Consider hedging with gold",
    "Action 3: Review portfolio allocation"
  ],
  "final_assessment": "Comprehensive final assessment synthesizing all inputs..."
}
```""")

        return "\n".join(prompt_parts)

    def _parse_final_validation_response(self, response: str) -> Dict:
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

            # Validate required keys
            required_keys = ['validation_result', 'final_recommendation', 'confidence']
            for key in required_keys:
                if key not in result:
                    logger.warning(f"Missing required key: {key}")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return partial result with raw response
            return {
                'validation_result': 'CAUTION',
                'final_recommendation': 'NEUTRAL',
                'confidence': 0.5,
                'final_assessment': response,
                'parse_error': str(e)
            }

    def calculate_agent_agreement_score(
        self,
        agent_votes: List[str]
    ) -> Dict:
        """
        에이전트 합의도 계산

        Args:
            agent_votes: 각 에이전트의 투표 (예: ['BULLISH', 'BULLISH', 'NEUTRAL', 'BULLISH'])

        Returns:
            {
                'agreement_score': 0.0-1.0,
                'agreement_level': 'HIGH' | 'MEDIUM' | 'LOW',
                'majority_vote': str
            }
        """
        if not agent_votes:
            return {'agreement_score': 0.0, 'agreement_level': 'LOW', 'majority_vote': 'NEUTRAL'}

        # Count votes
        vote_counts = {}
        for vote in agent_votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        # Find majority
        majority_vote = max(vote_counts, key=vote_counts.get)
        majority_count = vote_counts[majority_vote]

        # Calculate agreement score
        agreement_score = majority_count / len(agent_votes)

        # Classify agreement level
        if agreement_score >= 0.75:
            agreement_level = 'HIGH'
        elif agreement_score >= 0.5:
            agreement_level = 'MEDIUM'
        else:
            agreement_level = 'LOW'

        return {
            'agreement_score': agreement_score,
            'agreement_level': agreement_level,
            'majority_vote': majority_vote,
            'vote_distribution': vote_counts
        }


if __name__ == "__main__":
    # Test final validation
    validator = FinalValidator()

    test_full_mode = {
        'final_recommendation': 'BULLISH',
        'confidence': 0.72,
        'regime': {'regime': 'Bull (Low Vol)'},
        'risk_score': 28.5,
        'risk_level': 'MEDIUM',
        'full_mode_position': 'BULLISH',
        'reference_mode_position': 'BULLISH',
        'modes_agree': True
    }

    test_portfolio_val = {
        'validation_result': 'PASS',
        'theory_compliance': {
            'markowitz_mvo': 'compliant',
            'risk_parity': 'appropriate',
            'diversification': 'adequate'
        }
    }

    test_reasoning = {
        'reasoning_quality': 'STRONG',
        'academic_support': {
            'num_citations': 7,
            'has_recent_research': True
        }
    }

    test_sentiment = {
        'kospi_sentiment': {'sentiment': 'BULLISH', 'confidence': 0.65},
        'spx_sentiment': {'sentiment': 'BULLISH', 'confidence': 0.70},
        'comparison': {'divergence': 'ALIGNED'}
    }

    test_alternatives = {
        'crypto_assessment': {'recommendation': 'BULLISH'},
        'commodity_assessment': {'gold_role': 'SAFE_HAVEN'},
        'rwa_assessment': {'tokenization_trend': 'ACCELERATING'}
    }

    result = validator.validate_and_synthesize(
        test_full_mode,
        test_portfolio_val,
        test_reasoning,
        test_sentiment,
        test_alternatives
    )

    print("\n=== Final Validation Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
