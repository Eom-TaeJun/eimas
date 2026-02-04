#!/usr/bin/env python3
"""
Validation - Feedback Loop
============================================================

Feedback validation with iterative refinement

Class:
    - FeedbackValidationAgent: Iterative validation with feedback
"""

from typing import Dict, List, Optional, Tuple
import logging

from .base import BaseValidationAgent
from .schemas import AIValidation, FeedbackResult, FeedbackLoopResult
from .enums import ValidationResult

logger = logging.getLogger(__name__)


class FeedbackValidationAgent:
    """
    피드백 검증 에이전트 - 수정 제안 검증

    Claude와 Perplexity가 다른 AI들의 수정 제안을 검증
    """

    def __init__(self):
        self.claude_agent = None
        self.perplexity_agent = None
        self._init_agents()

    def _init_agents(self):
        """피드백 에이전트 초기화 (Claude + Perplexity만)"""
        if os.getenv('ANTHROPIC_API_KEY'):
            self.claude_agent = ClaudeValidationAgent()
            print("[Feedback] Claude feedback agent initialized")

        if os.getenv('PERPLEXITY_API_KEY'):
            self.perplexity_agent = PerplexityValidationAgent()
            print("[Feedback] Perplexity feedback agent initialized")

    def _build_feedback_prompt(
        self,
        original_decision: Dict,
        market_condition: Dict,
        suggestions: List[Dict]
    ) -> str:
        """피드백 프롬프트 생성"""
        # 제안 포맷팅
        suggestion_text = ""
        for i, s in enumerate(suggestions, 1):
            suggestion_text += f"\n{i}. **{s.get('field', 'Unknown')}**\n"
            suggestion_text += f"   - Current: {s.get('current', 'N/A')}\n"
            suggestion_text += f"   - Suggested: {s.get('suggested', 'N/A')}\n"
            suggestion_text += f"   - Reason: {s.get('reason', 'N/A')}\n"

        return FEEDBACK_PROMPT.format(
            agent_type=original_decision.get('agent_type', 'unknown'),
            action=original_decision.get('action', 'unknown'),
            risk_level=original_decision.get('risk_level', 50),
            regime=market_condition.get('regime', 'Unknown'),
            market_risk=market_condition.get('risk_score', 50),
            vix=market_condition.get('vix_level', 20),
            suggestions=suggestion_text if suggestion_text else "No suggestions provided"
        )

    async def _validate_with_claude(
        self,
        prompt: str
    ) -> FeedbackResult:
        """Claude로 피드백 검증"""
        if not self.claude_agent or not self.claude_agent.client:
            return self._error_result("Claude", "API not configured")

        start_time = datetime.now()

        try:
            response = self.claude_agent.client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=2000,
                temperature=0.1,
                system="You are a senior risk manager validating AI-generated trading suggestions. Be critical and precise.",
                messages=[{"role": "user", "content": prompt}]
            )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            raw_text = response.content[0].text
            parsed = self._parse_response(raw_text)

            return FeedbackResult(
                ai_name="Claude",
                model="claude-opus-4-5-20251101",
                timestamp=datetime.now().isoformat(),
                overall_verdict=parsed.get('overall_verdict', 'PARTIAL'),
                confidence=parsed.get('confidence', 50),
                suggestion_verdicts=parsed.get('suggestion_verdicts', []),
                final_recommendations=parsed.get('final_recommendations', []),
                rationale=parsed.get('rationale', ''),
                response_time_ms=int(elapsed),
                raw_response=raw_text
            )

        except Exception as e:
            return self._error_result("Claude", str(e))

    async def _validate_with_perplexity(
        self,
        prompt: str
    ) -> FeedbackResult:
        """Perplexity로 피드백 검증 (실시간 데이터 기반)"""
        if not self.perplexity_agent or not self.perplexity_agent.client:
            return self._error_result("Perplexity", "API not configured")

        start_time = datetime.now()

        # Perplexity용 프롬프트 (실시간 검색 강조)
        enhanced_prompt = f"""
{prompt}

Additionally, search for:
1. Recent market news that might affect these suggestions
2. Current analyst consensus on mentioned assets
3. Any recent volatility events or warnings
"""

        try:
            response = self.perplexity_agent.client.chat.completions.create(
                model="sonar-pro",
                messages=[
                    {
                        "role": "system",
                        "content": "You are validating trading suggestions with real-time market data. Be factual and cite sources."
                    },
                    {"role": "user", "content": enhanced_prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )

            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            raw_text = response.choices[0].message.content
            parsed = self._parse_response(raw_text)

            return FeedbackResult(
                ai_name="Perplexity",
                model="sonar-pro",
                timestamp=datetime.now().isoformat(),
                overall_verdict=parsed.get('overall_verdict', 'PARTIAL'),
                confidence=parsed.get('confidence', 50),
                suggestion_verdicts=parsed.get('suggestion_verdicts', []),
                final_recommendations=parsed.get('final_recommendations', []),
                rationale=parsed.get('rationale', ''),
                response_time_ms=int(elapsed),
                raw_response=raw_text
            )

        except Exception as e:
            return self._error_result("Perplexity", str(e))

    def _parse_response(self, text: str) -> Dict:
        """JSON 응답 파싱"""
        try:
            # JSON 블록 추출
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                # JSON 객체 직접 찾기
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]

            return json.loads(json_str)
        except:
            return {
                'overall_verdict': 'PARTIAL',
                'confidence': 50,
                'rationale': text[:500]
            }

    def _error_result(self, ai_name: str, error: str) -> FeedbackResult:
        """에러 결과 생성"""
        return FeedbackResult(
            ai_name=ai_name,
            model="error",
            timestamp=datetime.now().isoformat(),
            overall_verdict="PARTIAL",
            confidence=0,
            suggestion_verdicts=[],
            final_recommendations=[],
            rationale=f"Error: {error}"
        )

    async def validate_suggestions(
        self,
        original_decision: Dict,
        market_condition: Dict,
        suggestions: List[Dict]
    ) -> Dict[str, FeedbackResult]:
        """수정 제안 검증 (Claude + Perplexity 병렬)"""
        prompt = self._build_feedback_prompt(
            original_decision, market_condition, suggestions
        )

        # 병렬 실행
        tasks = []
        if self.claude_agent:
            tasks.append(self._validate_with_claude(prompt))
        if self.perplexity_agent:
            tasks.append(self._validate_with_perplexity(prompt))

        if not tasks:
            return {}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        feedback_results = {}
        for result in results:
            if isinstance(result, FeedbackResult):
                feedback_results[result.ai_name] = result
            elif isinstance(result, Exception):
                print(f"[Feedback] Error: {result}")

        return feedback_results

    def merge_feedback(
        self,
        feedback_results: Dict[str, FeedbackResult]
    ) -> Tuple[str, float, List[Dict]]:
        """피드백 결과 병합"""
        if not feedback_results:
            return "PARTIAL", 50.0, []

        # 가중치
        weights = {'Claude': 0.55, 'Perplexity': 0.45}

        verdict_scores = {'APPROVE': 0, 'REJECT': 0, 'PARTIAL': 0}
        total_confidence = 0
        total_weight = 0
        all_recommendations = []

        for name, result in feedback_results.items():
            weight = weights.get(name, 0.5)
            verdict_scores[result.overall_verdict] += weight
            total_confidence += result.confidence * weight
            total_weight += weight
            all_recommendations.extend(result.final_recommendations)

        # 최종 verdict
        final_verdict = max(verdict_scores, key=verdict_scores.get)
        avg_confidence = total_confidence / total_weight if total_weight > 0 else 50

        # 중복 제거된 권고
        seen = set()
        unique_recs = []
        for rec in all_recommendations:
            key = rec.get('field', '')
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)

        return final_verdict, avg_confidence, unique_recs


# ============================================================================
# 검증 루프 매니저
# ============================================================================
