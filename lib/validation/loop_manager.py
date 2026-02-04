#!/usr/bin/env python3
"""
Validation - Loop Manager
============================================================

Manages validation feedback loops

Class:
    - ValidationLoopManager: Orchestrates validation iterations
"""

from typing import Dict, List, Optional
import logging

from .manager import ValidationAgentManager
from .feedback import FeedbackValidationAgent
from .schemas import ConsensusResult, FeedbackLoopResult

logger = logging.getLogger(__name__)


class ValidationLoopManager:
    """
    검증 피드백 루프 관리

    Flow:
    1. Adaptive Agent → 결정
    2. Validation Agents (4개) → 검증 + 수정 제안
    3. Feedback Agents (Claude + Perplexity) → 수정 제안 검증
    4. 합의 도달 시 → 최종 결정 반환
    5. 미합의 시 → 수정 적용 후 재검증 (최대 3라운드)
    """

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.validation_manager = ValidationAgentManager()
        self.feedback_agent = FeedbackValidationAgent()
        print(f"[Loop] ValidationLoopManager initialized (max_rounds={max_rounds})")

    async def run_validation_loop(
        self,
        original_decision: Dict,
        market_condition: Dict
    ) -> FeedbackLoopResult:
        """전체 검증 루프 실행"""
        timestamp = datetime.now().isoformat()
        current_decision = original_decision.copy()
        modification_history = []

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n[Loop] === Round {round_num}/{self.max_rounds} ===")

            # Step 1: 4개 AI 검증
            print("[Loop] Step 1: Running validation agents...")
            consensus = await self.validation_manager.validate_decision(
                current_decision, market_condition
            )

            # 승인이면 즉시 종료
            if consensus.final_result == ValidationResult.APPROVE:
                print(f"[Loop] ✓ Decision APPROVED at round {round_num}")
                return FeedbackLoopResult(
                    timestamp=timestamp,
                    rounds_completed=round_num,
                    max_rounds=self.max_rounds,
                    final_decision=current_decision,
                    original_decision=original_decision,
                    modification_history=modification_history,
                    consensus_reached=True,
                    final_confidence=consensus.consensus_confidence,
                    summary=f"✓ 승인 (Round {round_num}, 신뢰도 {consensus.consensus_confidence:.0f}%)"
                )

            # 거부면 원본 유지하고 종료
            if consensus.final_result == ValidationResult.REJECT:
                print(f"[Loop] ✗ Decision REJECTED at round {round_num}")
                return FeedbackLoopResult(
                    timestamp=timestamp,
                    rounds_completed=round_num,
                    max_rounds=self.max_rounds,
                    final_decision=original_decision,  # 원본 반환
                    original_decision=original_decision,
                    modification_history=modification_history,
                    consensus_reached=False,
                    final_confidence=consensus.consensus_confidence,
                    summary=f"✗ 거부됨 - 원본 결정 유지 (Round {round_num})"
                )

            # Step 2: 수정 제안이 있으면 피드백 검증
            suggestions = consensus.merged_suggestions
            if not suggestions:
                print("[Loop] No suggestions to validate")
                break

            print(f"[Loop] Step 2: Validating {len(suggestions)} suggestions...")
            feedback_results = await self.feedback_agent.validate_suggestions(
                current_decision, market_condition, suggestions
            )

            # Step 3: 피드백 병합
            verdict, confidence, final_recs = self.feedback_agent.merge_feedback(
                feedback_results
            )

            print(f"[Loop] Feedback verdict: {verdict} ({confidence:.0f}%)")

            # Step 4: 수정 적용
            if verdict == "APPROVE" and final_recs:
                print(f"[Loop] Applying {len(final_recs)} modifications...")

                for rec in final_recs:
                    field = rec.get('field', '')
                    value = rec.get('value')

                    # 수정 이력 기록
                    modification_history.append({
                        'round': round_num,
                        'field': field,
                        'old_value': current_decision.get(field) or current_decision.get('allocations', {}).get(field),
                        'new_value': value,
                        'reason': rec.get('reason', '')
                    })

                    # 실제 수정 적용
                    if field == 'risk_level':
                        current_decision['risk_level'] = value
                    elif field in current_decision.get('allocations', {}):
                        current_decision['allocations'][field] = value

                print(f"[Loop] Modifications applied, continuing to next round...")

            elif verdict == "REJECT":
                print("[Loop] Suggestions rejected, keeping current decision")
                break

            else:
                print("[Loop] Partial approval, applying conservative adjustments...")
                # 부분 승인: 보수적으로 리스크만 10% 감소
                if 'risk_level' in current_decision:
                    old_risk = current_decision['risk_level']
                    new_risk = max(30, old_risk - 10)
                    modification_history.append({
                        'round': round_num,
                        'field': 'risk_level',
                        'old_value': old_risk,
                        'new_value': new_risk,
                        'reason': 'Partial approval - conservative adjustment'
                    })
                    current_decision['risk_level'] = new_risk

        # 루프 완료
        return FeedbackLoopResult(
            timestamp=timestamp,
            rounds_completed=self.max_rounds,
            max_rounds=self.max_rounds,
            final_decision=current_decision,
            original_decision=original_decision,
            modification_history=modification_history,
            feedback_results=feedback_results if 'feedback_results' in dir() else {},
            consensus_reached=len(modification_history) > 0,
            final_confidence=confidence if 'confidence' in dir() else 50.0,
            summary=f"루프 완료 ({len(modification_history)}개 수정 적용)"
        )

    def get_loop_report(self, result: FeedbackLoopResult) -> str:
        """루프 결과 리포트 생성"""
        lines = []
        lines.append("=" * 70)
        lines.append("VALIDATION LOOP REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {result.timestamp}")
        lines.append(f"Rounds: {result.rounds_completed}/{result.max_rounds}")
        lines.append(f"Consensus: {'Yes' if result.consensus_reached else 'No'}")
        lines.append(f"Final Confidence: {result.final_confidence:.1f}%")
        lines.append("")

        # 결정 비교
        lines.append("[DECISION COMPARISON]")
        lines.append(f"  Original Risk: {result.original_decision.get('risk_level')}")
        lines.append(f"  Final Risk: {result.final_decision.get('risk_level')}")
        lines.append("")

        # 수정 이력
        if result.modification_history:
            lines.append("[MODIFICATION HISTORY]")
            for mod in result.modification_history:
                lines.append(f"  Round {mod['round']}: {mod['field']}")
                lines.append(f"    {mod['old_value']} → {mod['new_value']}")
                lines.append(f"    Reason: {mod['reason']}")
            lines.append("")

        # 피드백 결과
        if result.feedback_results:
            lines.append("[FEEDBACK RESULTS]")
            for name, fb in result.feedback_results.items():
                lines.append(f"  [{name}] {fb.overall_verdict} ({fb.confidence:.0f}%)")
                lines.append(f"    {fb.rationale[:100]}...")
            lines.append("")

        lines.append(f"[SUMMARY] {result.summary}")
        lines.append("=" * 70)

        return "\n".join(lines)


# ============================================================================
# 테스트
# ============================================================================

async def test_validation():
    """단일 라운드 검증 테스트"""
    print("AI Validation Agents Test (Single Round)")
    print("=" * 60)

    manager = ValidationAgentManager()

    if not manager.agents:
        print("No agents available. Check API keys.")
        return

    print(f"\nActive agents: {list(manager.agents.keys())}")

    agent_decision = {
        'agent_type': 'aggressive',
        'action': 'AGGRESSIVE_ENTRY',
        'risk_level': 85,
        'rationale': 'Bull market with low VIX, high opportunity score',
        'allocations': {
            'TQQQ': 0.25,
            'SOXL': 0.20,
            'BTC-USD': 0.20,
            'QQQ': 0.20,
            'GLD': 0.15
        }
    }

    market_condition = {
        'regime': 'Bull (Low Vol)',
        'risk_score': 15,
        'vix_level': 14.5,
        'volatility': 'Low',
        'liquidity_signal': 'RISK_ON',
        'vpin_alert': False
    }

    print("\n[Testing Single Round Validation...]")
    print(f"Decision: {agent_decision['action']}")
    print(f"Risk Level: {agent_decision['risk_level']}")

    consensus = await manager.validate_decision(agent_decision, market_condition)
    print("\n" + manager.get_report(consensus))


async def test_validation_loop():
    """피드백 루프 검증 테스트"""
    print("\n" + "=" * 70)
    print("VALIDATION LOOP TEST (with Feedback)")
    print("=" * 70)

    loop_manager = ValidationLoopManager(max_rounds=3)

    agent_decision = {
        'agent_type': 'aggressive',
        'action': 'AGGRESSIVE_ENTRY',
        'risk_level': 85,
        'rationale': 'Bull market with low VIX, high opportunity score',
        'allocations': {
            'TQQQ': 0.25,
            'SOXL': 0.20,
            'BTC-USD': 0.20,
            'QQQ': 0.20,
            'GLD': 0.15
        }
    }

    market_condition = {
        'regime': 'Bull (Low Vol)',
        'risk_score': 15,
        'vix_level': 14.5,
        'volatility': 'Low',
        'liquidity_signal': 'RISK_ON',
        'vpin_alert': False
    }

    print(f"\nOriginal Decision: {agent_decision['action']}")
    print(f"Original Risk Level: {agent_decision['risk_level']}")
    print("\nRunning validation loop...")

    result = await loop_manager.run_validation_loop(agent_decision, market_condition)

    print("\n" + loop_manager.get_loop_report(result))

    # 결과 요약
    print("\n[FINAL COMPARISON]")
    print(f"  Risk Level: {agent_decision['risk_level']} → {result.final_decision.get('risk_level')}")
    print(f"  Rounds: {result.rounds_completed}/{result.max_rounds}")
    print(f"  Modifications: {len(result.modification_history)}")


async def main():
    """메인 테스트 실행"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        # 피드백 루프 테스트
        await test_validation_loop()
    else:
        # 단일 라운드 테스트
        await test_validation()


