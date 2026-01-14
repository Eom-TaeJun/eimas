#!/usr/bin/env python3
"""
Multi-Agent System - Debate Protocol
=====================================
에이전트 간 토론 프로토콜 (Rule-based)

경제학적 의미:
- 시장은 다양한 의견을 종합 → 에이전트도 동일한 프로세스
- 85% 일관성 임계값: 과도한 수렴 방지 (군집사고 회피)
- 5% 수정 임계값: 교착 상태 감지 (비생산적 토론 차단)
- 최대 3라운드: 효율성과 철저함의 균형
"""

import sys
import os
from typing import List, Dict, Optional, Tuple
from collections import Counter
import numpy as np
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.schemas import (
    AgentOpinion,
    Consensus,
    Conflict,
    AgentRole,
    OpinionStrength,
)


class DebateProtocol:
    """
    에이전트 간 토론 조율 프로토콜

    핵심 기능:
    1. 의견 충돌 식별
    2. 최대 3라운드 토론 진행
    3. 일관성 ≥85% 시 조기 종료
    4. 수정 폭 <5% 시 라운드 스킵
    5. 합의 도출

    방법론: Rule-based (No LLM calls for revision)
    """

    def __init__(
        self,
        max_rounds: int = 3,
        consistency_threshold: float = 85.0,
        modification_threshold: float = 5.0,
        preserve_dissent: bool = True,  # 소수의견 보호 모드
        record_dissent: bool = True      # 반대의견 기록 여부
    ):
        """
        Args:
            max_rounds: 최대 토론 라운드 (기본 3)
            consistency_threshold: 조기 종료 일관성 임계값 (기본 85%)
            modification_threshold: 스킵 수정 임계값 (기본 5%)
            preserve_dissent: True면 소수의견을 다수에 강제 수렴시키지 않음
            record_dissent: True면 반대의견을 별도 기록
        """
        self.max_rounds = max_rounds
        self.consistency_threshold = consistency_threshold
        self.modification_threshold = modification_threshold
        self.preserve_dissent = preserve_dissent
        self.record_dissent = record_dissent
        self.dissent_records: List[Dict] = []  # 반대의견 기록

    def run_debate(
        self,
        opinions: List[AgentOpinion],
        topic: str
    ) -> Consensus:
        """
        전체 토론 프로세스 실행

        프로세스:
        1. 초기 일관성 확인 → ≥85%면 조기 종료
        2. 충돌 식별
        3. 최대 3라운드 토론
        4. 각 라운드 후 일관성/수정폭 확인
        5. 합의 도출

        Args:
            opinions: 초기 의견 리스트
            topic: 토론 주제

        Returns:
            Consensus 객체
        """
        if not opinions:
            raise ValueError("At least one opinion required for debate")

        # 모든 의견이 같은 주제인지 확인
        for op in opinions:
            if op.topic != topic:
                raise ValueError(f"Opinion topic '{op.topic}' doesn't match debate topic '{topic}'")

        # 초기 일관성 확인
        initial_consistency = self.calculate_consistency(opinions)
        if initial_consistency >= self.consistency_threshold:
            print(f"   [Debate] Early termination - initial consistency {initial_consistency:.1f}% >= {self.consistency_threshold}%")
            return self.reach_consensus(opinions, debate_rounds=0)

        # 충돌 식별
        conflicts = self.identify_conflicts(opinions)
        if not conflicts:
            print(f"   [Debate] No conflicts detected - reaching consensus")
            return self.reach_consensus(opinions, debate_rounds=0)

        print(f"   [Debate] Starting debate on '{topic}' - {len(conflicts)} conflicts, consistency={initial_consistency:.1f}%")

        # 토론 라운드
        current_opinions = opinions
        for round_num in range(1, self.max_rounds + 1):
            print(f"   [Debate] Round {round_num}/{self.max_rounds}")

            # 라운드 실행 (이전 의견 저장)
            previous_opinions = current_opinions.copy()
            current_opinions = self.run_debate_round(current_opinions, conflicts, round_num)

            # 일관성 확인
            consistency = self.calculate_consistency(current_opinions)
            print(f"   [Debate] Round {round_num} consistency: {consistency:.1f}%")

            if consistency >= self.consistency_threshold:
                print(f"   [Debate] Early termination - consistency {consistency:.1f}% >= {self.consistency_threshold}%")
                return self.reach_consensus(current_opinions, debate_rounds=round_num)

            # 수정폭 확인 (라운드 1 이후)
            if round_num > 1:
                mod_delta = self.calculate_modification_delta(previous_opinions, current_opinions)
                print(f"   [Debate] Round {round_num} modification delta: {mod_delta:.1f}%")

                if mod_delta < self.modification_threshold:
                    print(f"   [Debate] Modification delta {mod_delta:.1f}% < {self.modification_threshold}% - gridlock detected")
                    return self.reach_consensus(current_opinions, debate_rounds=round_num)

            # 충돌 재평가 (다음 라운드용)
            conflicts = self.identify_conflicts(current_opinions)

        # 최대 라운드 도달
        print(f"   [Debate] Max rounds reached - reaching consensus")
        return self.reach_consensus(current_opinions, debate_rounds=self.max_rounds)

    def identify_conflicts(self, opinions: List[AgentOpinion]) -> List[Conflict]:
        """
        의견 간 충돌 식별

        충돌 유형:
        1. Directional: 반대 입장 (severity 가장 높음)
        2. Magnitude: 신뢰도 차이 > 0.3

        Args:
            opinions: 의견 리스트

        Returns:
            충돌 리스트 (severity 기준 정렬)
        """
        conflicts = []

        # 1. 입장 분석 (position 기반)
        positions_count = Counter(op.position for op in opinions)

        # Directional conflict: 다수 입장과 반대되는 소수 입장
        if len(positions_count) > 1:
            modal_position = positions_count.most_common(1)[0][0]
            minority_opinions = [op for op in opinions if op.position != modal_position]

            if minority_opinions:
                conflicts.append(Conflict(
                    agents=[op.agent_role for op in minority_opinions],
                    topic=opinions[0].topic,
                    positions={op.agent_role: op.position for op in opinions},
                    severity=len(minority_opinions) / len(opinions),  # 소수 비율 = severity
                    resolution_needed=True
                ))

        # 2. Magnitude conflict: 신뢰도 분산이 큼
        confidences = [op.confidence for op in opinions]
        confidence_std = np.std(confidences)

        if confidence_std > 0.3:
            # 가장 낮은 신뢰도와 가장 높은 신뢰도 간 충돌
            min_conf_op = min(opinions, key=lambda op: op.confidence)
            max_conf_op = max(opinions, key=lambda op: op.confidence)

            conflicts.append(Conflict(
                agents=[min_conf_op.agent_role, max_conf_op.agent_role],
                topic=opinions[0].topic,
                positions={
                    min_conf_op.agent_role: f"{min_conf_op.position} (low confidence: {min_conf_op.confidence:.2f})",
                    max_conf_op.agent_role: f"{max_conf_op.position} (high confidence: {max_conf_op.confidence:.2f})"
                },
                severity=confidence_std,
                resolution_needed=True
            ))

        # Severity 기준 정렬
        conflicts.sort(key=lambda c: c.severity, reverse=True)

        return conflicts

    def run_debate_round(
        self,
        opinions: List[AgentOpinion],
        conflicts: List[Conflict],
        round_num: int
    ) -> List[AgentOpinion]:
        """
        단일 토론 라운드 실행 (Rule-based revision)

        수정 규칙:
        1. 소수 의견 + 낮은 신뢰도 → 다수 의견으로 이동 (신뢰도 +10%)
        2. 높은 신뢰도 충돌 → 신뢰도 -10% (불확실성 증가)
        3. 중간 신뢰도 → 유지

        Args:
            opinions: 현재 의견 리스트
            conflicts: 식별된 충돌 리스트
            round_num: 현재 라운드 번호

        Returns:
            수정된 의견 리스트
        """
        if not conflicts:
            return opinions

        # 다수 입장 파악
        positions_count = Counter(op.position for op in opinions)
        modal_position = positions_count.most_common(1)[0][0] if positions_count else None

        # 평균 신뢰도
        avg_confidence = np.mean([op.confidence for op in opinions])

        revised_opinions = []

        for op in opinions:
            # 새 의견 객체 생성 (수정용)
            new_op = AgentOpinion(
                agent_role=op.agent_role,
                topic=op.topic,
                position=op.position,
                strength=op.strength,
                confidence=op.confidence,
                evidence=op.evidence.copy(),
                caveats=op.caveats.copy(),
                key_metrics=op.key_metrics.copy(),
                timestamp=datetime.now().isoformat()
            )

            # Rule 1: 소수 의견 처리 (preserve_dissent에 따라 다름)
            is_minority = (modal_position and op.position != modal_position)
            is_low_confidence = (op.confidence < avg_confidence)

            if is_minority and is_low_confidence:
                if self.preserve_dissent:
                    # 소수의견 보호: 위치 유지, 신뢰도만 소폭 조정
                    new_op.confidence = max(0.3, op.confidence - 0.05)  # 최소 0.3 유지
                    new_op.caveats.append(f"Round {round_num}: Minority view preserved (dissenting)")
                    print(f"      [{op.agent_role.value}] DISSENT PRESERVED: {op.position} (conf: {op.confidence:.2f} → {new_op.confidence:.2f})")

                    # 반대의견 기록
                    if self.record_dissent:
                        self.dissent_records.append({
                            'round': round_num,
                            'agent': op.agent_role.value,
                            'position': op.position,
                            'confidence': op.confidence,
                            'evidence': op.evidence,
                            'reason': 'minority_low_confidence'
                        })
                else:
                    # 기존 동작: 다수 입장으로 이동 (위험)
                    new_op.position = modal_position
                    new_op.confidence = min(1.0, op.confidence + 0.10)
                    new_op.caveats.append(f"Revised in round {round_num}: moved to majority position")
                    print(f"      [{op.agent_role.value}] Moved to majority (confidence {op.confidence:.2f} → {new_op.confidence:.2f})")

            # Rule 1.5: 소수 의견 + 높은 신뢰도 → 강력한 반대의견 기록
            elif is_minority and op.confidence >= avg_confidence:
                # 높은 신뢰도의 소수의견은 특별히 기록 (중요한 경고 신호)
                if self.record_dissent:
                    self.dissent_records.append({
                        'round': round_num,
                        'agent': op.agent_role.value,
                        'position': op.position,
                        'confidence': op.confidence,
                        'evidence': op.evidence,
                        'reason': 'high_confidence_dissent',
                        'warning': 'STRONG DISSENT - Consider carefully'
                    })
                    print(f"      [{op.agent_role.value}] ⚠️ STRONG DISSENT: {op.position} (conf: {op.confidence:.2f})")

                # 신뢰도 소폭 감소하지만 의견은 유지
                new_op.confidence = max(0.4, op.confidence - 0.05)
                new_op.caveats.append(f"Round {round_num}: Strong minority dissent maintained")

            # Rule 2: 높은 신뢰도지만 충돌 존재 → 신뢰도 감소
            elif op.confidence > avg_confidence and conflicts:
                new_op.confidence = max(0.0, op.confidence - 0.10)  # 신뢰도 -10%
                new_op.caveats.append(f"Revised in round {round_num}: reduced confidence due to conflicts")
                print(f"      [{op.agent_role.value}] Reduced confidence (conflict) ({op.confidence:.2f} → {new_op.confidence:.2f})")

            # Rule 3: 중간 신뢰도 → 소폭 증가 (토론을 통한 확신)
            else:
                new_op.confidence = min(1.0, op.confidence + 0.05)  # 신뢰도 +5%
                new_op.caveats.append(f"Revised in round {round_num}: minor confidence increase")

            revised_opinions.append(new_op)

        return revised_opinions

    def calculate_consistency(self, opinions: List[AgentOpinion]) -> float:
        """
        의견 일관성 점수 계산

        공식:
        consistency = 0.4 × stance_consistency
                    + 0.3 × confidence_convergence
                    + 0.3 × metric_alignment

        Args:
            opinions: 의견 리스트

        Returns:
            일관성 점수 (0-100)
        """
        if not opinions:
            return 0.0

        # 1. Stance consistency (40%)
        positions = [op.position for op in opinions]
        position_counts = Counter(positions)
        modal_count = position_counts.most_common(1)[0][1] if position_counts else 0
        stance_consistency = (modal_count / len(opinions)) * 100

        # 2. Confidence convergence (30%)
        confidences = [op.confidence for op in opinions]
        confidence_std = np.std(confidences)
        confidence_convergence = max(0, 100 - (confidence_std * 100))  # 낮은 std = 높은 수렴

        # 3. Metric alignment (30%)
        # key_metrics 간 상관관계 계산
        if opinions[0].key_metrics:
            # 모든 key_metrics가 공통 키를 가지고 있는지 확인
            common_keys = set(opinions[0].key_metrics.keys())
            for op in opinions[1:]:
                common_keys &= set(op.key_metrics.keys())

            if common_keys:
                # 공통 키에 대해 값 추출
                metric_vectors = []
                for op in opinions:
                    vector = [op.key_metrics[key] for key in sorted(common_keys)]
                    metric_vectors.append(vector)

                # 벡터 간 평균 상관관계 계산
                correlations = []
                for i in range(len(metric_vectors)):
                    for j in range(i + 1, len(metric_vectors)):
                        corr = np.corrcoef(metric_vectors[i], metric_vectors[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)

                if correlations:
                    metric_alignment = (np.mean(correlations) + 1) * 50  # -1~1 → 0~100
                else:
                    metric_alignment = 50.0  # 기본값
            else:
                metric_alignment = 50.0  # 공통 키 없음
        else:
            metric_alignment = 50.0  # key_metrics 없음

        # 가중 합산
        consistency = (
            0.4 * stance_consistency +
            0.3 * confidence_convergence +
            0.3 * metric_alignment
        )

        return consistency

    def calculate_modification_delta(
        self,
        previous_opinions: List[AgentOpinion],
        current_opinions: List[AgentOpinion]
    ) -> float:
        """
        라운드 간 수정 폭 계산

        측정:
        1. Position changes (50%): 입장 변경 비율
        2. Confidence deltas (30%): 신뢰도 변화 평균
        3. Evidence changes (20%): 근거 변경 비율

        Args:
            previous_opinions: 이전 라운드 의견
            current_opinions: 현재 라운드 의견

        Returns:
            수정 비율 (0-100)
        """
        if len(previous_opinions) != len(current_opinions):
            raise ValueError("Opinion lists must have same length")

        # Agent role 기준으로 매칭
        prev_dict = {op.agent_role: op for op in previous_opinions}
        curr_dict = {op.agent_role: op for op in current_opinions}

        position_changes = 0
        confidence_deltas = []
        evidence_changes = 0

        for role in prev_dict:
            if role not in curr_dict:
                continue

            prev_op = prev_dict[role]
            curr_op = curr_dict[role]

            # Position change
            if prev_op.position != curr_op.position:
                position_changes += 1

            # Confidence delta
            conf_delta = abs(curr_op.confidence - prev_op.confidence)
            confidence_deltas.append(conf_delta)

            # Evidence change (새로운 근거 추가 여부)
            if len(curr_op.evidence) != len(prev_op.evidence):
                evidence_changes += 1

        n = len(prev_dict)

        # 각 측정값 정규화
        position_change_pct = (position_changes / n) * 100 if n > 0 else 0
        confidence_delta_pct = (np.mean(confidence_deltas) * 100) if confidence_deltas else 0
        evidence_change_pct = (evidence_changes / n) * 100 if n > 0 else 0

        # 가중 합산
        modification_delta = (
            0.5 * position_change_pct +
            0.3 * confidence_delta_pct +
            0.2 * evidence_change_pct
        )

        return modification_delta

    def reach_consensus(
        self,
        opinions: List[AgentOpinion],
        debate_rounds: int
    ) -> Consensus:
        """
        최종 합의 도출

        방법:
        1. Modal position (최빈 입장) 선택
        2. Weighted confidence (신뢰도 가중평균)
        3. Supporting/dissenting agents 분류
        4. Compromises 기록

        Args:
            opinions: 최종 의견 리스트
            debate_rounds: 거친 토론 라운드 수

        Returns:
            Consensus 객체
        """
        if not opinions:
            raise ValueError("At least one opinion required for consensus")

        topic = opinions[0].topic

        # 1. Modal position
        positions = [op.position for op in opinions]
        position_counts = Counter(positions)
        final_position = position_counts.most_common(1)[0][0]

        # 2. Average confidence (가중치 제거 - 자기강화 바이어스 방지)
        supporting_opinions = [op for op in opinions if op.position == final_position]
        if supporting_opinions:
            weighted_confidence = np.mean([op.confidence for op in supporting_opinions])
        else:
            weighted_confidence = 0.5

        # 3. Supporting/dissenting agents
        supporting_agents = [op.agent_role for op in opinions if op.position == final_position]
        dissenting_agents = [op.agent_role for op in opinions if op.position != final_position]

        # 4. Compromises (caveats에서 추출)
        compromises = []
        for op in opinions:
            for caveat in op.caveats:
                if "Revised in round" in caveat:
                    compromises.append(f"{op.agent_role.value}: {caveat}")

        # Unique compromises
        compromises = list(set(compromises))

        # 5. 반대의견 상세 포함
        dissent_details = self.dissent_records.copy() if self.record_dissent else []
        has_strong_dissent = any(
            d.get('reason') == 'high_confidence_dissent'
            for d in dissent_details
        )

        # 강력한 반대의견 있으면 신뢰도 조정 (경고)
        if has_strong_dissent:
            weighted_confidence = max(0.5, weighted_confidence - 0.1)
            compromises.append("⚠️ STRONG DISSENT EXISTS - Review dissent_details")

        return Consensus(
            topic=topic,
            final_position=final_position,
            confidence=weighted_confidence,
            supporting_agents=supporting_agents,
            dissenting_agents=dissenting_agents,
            compromises=compromises,
            debate_rounds=debate_rounds,
            timestamp=datetime.now().isoformat(),
            dissent_details=dissent_details,
            has_strong_dissent=has_strong_dissent
        )


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("DebateProtocol Test")
    print("="*60)

    # 테스트 의견 생성
    opinions = [
        AgentOpinion(
            agent_role=AgentRole.ANALYSIS,
            topic="market_outlook",
            position="Bullish - markets showing strength",
            strength=OpinionStrength.AGREE,
            confidence=0.75,
            evidence=["Risk score: 25/100", "Regime: EXPANSION"],
            key_metrics={"risk_score": 25.0, "regime_confidence": 80.0}
        ),
        AgentOpinion(
            agent_role=AgentRole.FORECAST,
            topic="market_outlook",
            position="Bearish - headwinds ahead",
            strength=OpinionStrength.DISAGREE,
            confidence=0.65,
            evidence=["Fed tightening", "Yield curve inversion"],
            key_metrics={"risk_score": 65.0, "regime_confidence": 70.0}
        ),
        AgentOpinion(
            agent_role=AgentRole.RESEARCH,
            topic="market_outlook",
            position="Bullish - markets showing strength",
            strength=OpinionStrength.AGREE,
            confidence=0.80,
            evidence=["Strong earnings", "Low VIX"],
            key_metrics={"risk_score": 30.0, "regime_confidence": 85.0}
        ),
    ]

    # 토론 프로토콜 실행
    protocol = DebateProtocol(
        max_rounds=3,
        consistency_threshold=85.0,
        modification_threshold=5.0
    )

    consensus = protocol.run_debate(opinions, "market_outlook")

    # 결과 출력
    print("\n" + "="*60)
    print("CONSENSUS RESULT")
    print("="*60)
    print(f"Topic: {consensus.topic}")
    print(f"Final Position: {consensus.final_position}")
    print(f"Confidence: {consensus.confidence:.2%}")
    print(f"Debate Rounds: {consensus.debate_rounds}")
    print(f"\nSupporting Agents: {[a.value for a in consensus.supporting_agents]}")
    print(f"Dissenting Agents: {[a.value for a in consensus.dissenting_agents]}")
    print(f"\nCompromises: {len(consensus.compromises)}")
    for comp in consensus.compromises[:3]:
        print(f"  - {comp}")
