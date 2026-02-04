#!/usr/bin/env python3
"""
Validation - Consensus Engine
============================================================

Multi-agent consensus algorithm

Class:
    - ConsensusEngine: Aggregates multiple AI validations
"""

from typing import List, Dict
import logging

from .schemas import AIValidation, ConsensusResult
from .enums import ValidationResult

logger = logging.getLogger(__name__)


class ConsensusEngine:
    """AI 검증 결과 합의 도출"""

    def __init__(self):
        self.weights = {
            'Claude': 0.30,      # 깊은 분석
            'Perplexity': 0.25,  # 실시간 정보
            'GPT': 0.25,         # 종합 판단
            'Gemini': 0.20       # 다각적 관점
        }

    def reach_consensus(self, validations: Dict[str, AIValidation]) -> ConsensusResult:
        """합의 도출"""
        timestamp = datetime.now().isoformat()

        if not validations:
            return ConsensusResult(
                timestamp=timestamp,
                summary="No validations available"
            )

        # 1. 결과별 점수 계산
        result_scores = {
            ValidationResult.APPROVE: 0,
            ValidationResult.MODIFY: 0,
            ValidationResult.REJECT: 0,
            ValidationResult.NEEDS_INFO: 0
        }

        total_weight = 0
        weighted_confidence = 0

        for ai_name, validation in validations.items():
            weight = self.weights.get(ai_name, 0.2)
            total_weight += weight

            # 결과에 가중치 적용
            result_scores[validation.result] += weight * (validation.confidence / 100)

            # 가중 신뢰도
            weighted_confidence += weight * validation.confidence

        # 2. 최종 결과 결정
        best_result = max(result_scores.items(), key=lambda x: x[1])
        final_result = best_result[0]

        # 3. 동의 비율 계산
        agreed_count = sum(
            1 for v in validations.values()
            if v.result == final_result
        )
        agreement_ratio = agreed_count / len(validations) if validations else 0

        # 4. 평균 신뢰도
        consensus_confidence = weighted_confidence / total_weight if total_weight > 0 else 0

        # 5. 제안 사항 병합
        all_suggestions = []
        all_warnings = []
        all_opportunities = []

        for validation in validations.values():
            all_suggestions.extend(validation.suggested_changes)
            all_warnings.extend(validation.risk_warnings)
            all_opportunities.extend(validation.opportunities)

        # 중복 제거
        unique_warnings = list(set(all_warnings))
        unique_opportunities = list(set(all_opportunities))

        # 6. 요약 생성
        summary = self._generate_summary(
            final_result, agreement_ratio, consensus_confidence,
            unique_warnings, validations
        )

        # 7. 액션 아이템 생성
        action_items = self._generate_action_items(
            final_result, all_suggestions, unique_warnings
        )

        return ConsensusResult(
            timestamp=timestamp,
            validations=validations,
            final_result=final_result,
            consensus_confidence=consensus_confidence,
            agreement_ratio=agreement_ratio,
            merged_suggestions=all_suggestions[:10],  # 상위 10개
            key_concerns=unique_warnings[:5],
            action_items=action_items,
            summary=summary
        )

    def _generate_summary(
        self,
        result: ValidationResult,
        agreement: float,
        confidence: float,
        warnings: List[str],
        validations: Dict
    ) -> str:
        """요약 생성"""
        parts = []

        # 결과
        if result == ValidationResult.APPROVE:
            parts.append(f"✓ 승인 (동의율 {agreement:.0%}, 신뢰도 {confidence:.0f}%)")
        elif result == ValidationResult.REJECT:
            parts.append(f"✗ 거부 (동의율 {agreement:.0%}, 신뢰도 {confidence:.0f}%)")
        elif result == ValidationResult.MODIFY:
            parts.append(f"⚠ 수정 필요 (동의율 {agreement:.0%}, 신뢰도 {confidence:.0f}%)")
        else:
            parts.append(f"? 추가 정보 필요 (동의율 {agreement:.0%})")

        # 주요 우려사항
        if warnings:
            parts.append(f"주요 우려: {warnings[0]}")

        # AI별 판단
        ai_results = [f"{k}: {v.result.value}" for k, v in validations.items()]
        parts.append(f"AI 판단: {', '.join(ai_results)}")

        return " | ".join(parts)

    def _generate_action_items(
        self,
        result: ValidationResult,
        suggestions: List[Dict],
        warnings: List[str]
    ) -> List[str]:
        """액션 아이템 생성"""
        items = []

        if result == ValidationResult.REJECT:
            items.append("즉시 전략 재검토 필요")
            items.append("리스크 레벨 하향 조정 검토")

        elif result == ValidationResult.MODIFY:
            for s in suggestions[:3]:
                items.append(f"{s.get('field', 'Unknown')} 수정: {s.get('reason', 'N/A')}")

        if warnings:
            items.append(f"리스크 모니터링: {warnings[0]}")

        return items


# ============================================================================
# 검증 매니저
# ============================================================================
