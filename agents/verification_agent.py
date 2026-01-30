#!/usr/bin/env python3
"""
Verification Agent
==================
멀티에이전트 토론 결과를 검증하는 에이전트

핵심 기능:
1. Hallucination Detection: 잘못된 정보 생성 감지
2. Sycophancy Detection: 동조 편향 감지
3. Logical Consistency Check: 논리적 일관성 검증
4. Data Cross-Verification: 수치 데이터 교차 검증
5. Opinion Diversity Assessment: 의견 다양성 평가

경제학적 배경:
- Information Asymmetry 해소
- Groupthink 방지 (Janis 1972)
- Wisdom of Crowds (Surowiecki 2004)
- Adversarial Review (Peer Review Process)
"""

import sys
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base_agent import BaseAgent, AgentConfig
from core.schemas import AgentRequest, AgentRole, AgentOpinion, OpinionStrength


@dataclass
class VerificationResult:
    """검증 결과"""
    overall_score: float  # 0-100
    hallucination_risk: float  # 0-100, 높을수록 위험
    sycophancy_risk: float  # 0-100, 높을수록 위험
    logical_consistency: float  # 0-100, 높을수록 일관성 있음
    data_accuracy: float  # 0-100, 높을수록 정확
    opinion_diversity: float  # 0-100, 높을수록 다양
    issues_found: List[str]
    warnings: List[str]
    passed: bool
    timestamp: str


@dataclass
class HallucinationCheck:
    """Hallucination 체크 결과"""
    contains_hallucination: bool
    confidence: float  # 0-100
    problematic_statements: List[str]
    reasoning: str


@dataclass
class SycophancyCheck:
    """Sycophancy 체크 결과"""
    has_sycophancy: bool
    agreement_rate: float  # 0-100, 높을수록 동조 경향
    dissent_count: int
    independent_opinions: int
    reasoning: str


class VerificationAgent(BaseAgent):
    """
    검증 에이전트

    역할:
    - 멀티에이전트 토론 결과의 신뢰성 검증
    - Hallucination과 Sycophancy 리스크 평가
    - 논리적 일관성과 데이터 정확성 확인
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """초기화"""
        if config is None:
            config = AgentConfig(
                name="VerificationAgent",
                role=AgentRole.VERIFICATION,
                verbose=True,
                custom_config={
                    'hallucination_threshold': 30.0,  # 30% 이상이면 경고
                    'sycophancy_threshold': 80.0,  # 80% 이상 동조시 경고
                    'min_diversity_score': 40.0,  # 최소 다양성 점수
                    'data_tolerance': 10.0,  # 데이터 허용 오차 (%)
                }
            )
        super().__init__(config)

        self.hallucination_threshold = config.custom_config.get('hallucination_threshold', 30.0)
        self.sycophancy_threshold = config.custom_config.get('sycophancy_threshold', 80.0)
        self.min_diversity_score = config.custom_config.get('min_diversity_score', 40.0)
        self.data_tolerance = config.custom_config.get('data_tolerance', 10.0)

    def _normalize_opinions(self, opinions: List) -> List[AgentOpinion]:
        """
        Normalize opinions to AgentOpinion objects.
        Handles both dict and AgentOpinion inputs.
        """
        normalized = []
        for op in opinions:
            if isinstance(op, AgentOpinion):
                normalized.append(op)
            elif isinstance(op, dict):
                # Convert dict to AgentOpinion
                role_val = op.get('agent_role', 'ANALYSIS')
                if isinstance(role_val, str):
                    try:
                        role = AgentRole(role_val)
                    except ValueError:
                        role = AgentRole.ANALYSIS
                else:
                    role = role_val
                    
                strength_val = op.get('strength', 'NEUTRAL')
                if isinstance(strength_val, str):
                    try:
                        strength = OpinionStrength(strength_val)
                    except ValueError:
                        strength = OpinionStrength.NEUTRAL
                else:
                    strength = strength_val
                    
                normalized.append(AgentOpinion(
                    agent_role=role,
                    topic=op.get('topic', 'unknown'),
                    position=op.get('position', 'unknown'),
                    confidence=op.get('confidence', 0.5),
                    strength=strength,
                    evidence=op.get('evidence', []),
                    caveats=op.get('caveats', []),
                    key_metrics=op.get('key_metrics', {})
                ))
            else:
                self.logger.warning(f"Unknown opinion type: {type(op)}")
        return normalized

    async def _execute(self, request: AgentRequest) -> Dict[str, Any]:
        """
        검증 실행

        Args:
            request: AgentRequest with context containing:
                - debate_results: 토론 결과
                - opinions: 각 에이전트 의견 리스트
                - market_data: 원본 시장 데이터 (교차 검증용)

        Returns:
            검증 결과 딕셔너리
        """
        context = request.context or {}
        debate_results = context.get('debate_results', {})
        opinions_raw = context.get('opinions', [])
        market_data = context.get('market_data', {})
        
        # Normalize opinions (handle both dict and AgentOpinion)
        opinions = self._normalize_opinions(opinions_raw)

        self.logger.info(f"Verifying debate with {len(opinions)} agent opinions")

        # 1. Hallucination 검증
        hallucination_check = self._check_hallucination(opinions, market_data)

        # 2. Sycophancy 검증
        sycophancy_check = self._check_sycophancy(opinions)

        # 3. 논리적 일관성 검증
        logical_consistency = self._check_logical_consistency(opinions, debate_results)

        # 4. 데이터 정확성 검증
        data_accuracy = self._check_data_accuracy(opinions, market_data)

        # 5. 의견 다양성 평가
        opinion_diversity = self._assess_opinion_diversity(opinions)

        # 종합 평가
        verification_result = self._synthesize_results(
            hallucination_check,
            sycophancy_check,
            logical_consistency,
            data_accuracy,
            opinion_diversity
        )

        return {
            'verification_result': asdict(verification_result),
            'hallucination_check': asdict(hallucination_check),
            'sycophancy_check': asdict(sycophancy_check),
            'reasoning': self._generate_reasoning(verification_result)
        }

    def _check_hallucination(
        self,
        opinions: List[AgentOpinion],
        market_data: Dict[str, Any]
    ) -> HallucinationCheck:
        """
        Hallucination 검증

        검증 항목:
        1. 수치 데이터가 원본과 일치하는가?
        2. 존재하지 않는 지표를 언급하는가?
        3. 논리적으로 불가능한 주장을 하는가?
        """
        problematic_statements = []
        hallucination_score = 0.0
        total_checks = 0

        # 원본 데이터에서 허용 가능한 키워드 추출
        valid_tickers = set(market_data.get('tickers', []))
        valid_metrics = set(market_data.keys())

        for opinion in opinions:
            content = f"{opinion.position} {' '.join(opinion.evidence)}".lower()

            # Check 1: 존재하지 않는 티커 언급
            mentioned_tickers = self._extract_tickers(content)
            for ticker in mentioned_tickers:
                total_checks += 1
                if ticker not in valid_tickers:
                    problematic_statements.append(
                        f"Unknown ticker '{ticker}' mentioned by {opinion.agent_role.value}"
                    )
                    hallucination_score += 1

            # Check 2: 수치 데이터 일치 여부
            numeric_claims = self._extract_numeric_claims(content)
            for claim in numeric_claims:
                total_checks += 1
                if not self._verify_numeric_claim(claim, market_data):
                    problematic_statements.append(
                        f"Unverified numeric claim: '{claim}' by {opinion.agent_role.value}"
                    )
                    hallucination_score += 1

            # Check 3: 논리적 모순 (예: "VIX가 5인데 높은 변동성")
            contradictions = self._detect_contradictions(content, market_data)
            if contradictions:
                total_checks += 1
                problematic_statements.extend(contradictions)
                hallucination_score += len(contradictions)

        # 점수 계산 (0-100, 높을수록 위험)
        hallucination_risk = (hallucination_score / max(total_checks, 1)) * 100

        return HallucinationCheck(
            contains_hallucination=hallucination_risk > self.hallucination_threshold,
            confidence=100 - hallucination_risk,
            problematic_statements=problematic_statements,
            reasoning=f"Found {len(problematic_statements)} potential hallucinations out of {total_checks} checks"
        )

    def _check_sycophancy(self, opinions: List[AgentOpinion]) -> SycophancyCheck:
        """
        Sycophancy 검증

        검증 항목:
        1. 의견의 다양성이 충분한가?
        2. 강한 반대 의견이 있는가?
        3. 동일한 근거를 반복 인용하는가?
        """
        if len(opinions) < 2:
            return SycophancyCheck(
                has_sycophancy=False,
                agreement_rate=0.0,
                dissent_count=0,
                independent_opinions=len(opinions),
                reasoning="Not enough opinions to assess sycophancy"
            )

        # 의견 강도 분석
        positions = {}
        for opinion in opinions:
            topic = opinion.topic
            position = opinion.position.lower()

            if topic not in positions:
                positions[topic] = []
            positions[topic].append(position)

        # 동조율 계산
        agreement_rates = []
        dissent_count = 0

        for topic, stances in positions.items():
            if len(stances) < 2:
                continue

            # 가장 많은 입장
            from collections import Counter
            position_counts = Counter(stances)
            most_common_count = position_counts.most_common(1)[0][1]

            # 동조율 = 다수 의견 비율
            agreement_rate = (most_common_count / len(stances)) * 100
            agreement_rates.append(agreement_rate)

            # 반대 의견 카운트 (소수 의견)
            dissent_count += len(stances) - most_common_count

        avg_agreement_rate = np.mean(agreement_rates) if agreement_rates else 0.0
        independent_opinions = dissent_count

        has_sycophancy = avg_agreement_rate > self.sycophancy_threshold

        return SycophancyCheck(
            has_sycophancy=has_sycophancy,
            agreement_rate=avg_agreement_rate,
            dissent_count=dissent_count,
            independent_opinions=independent_opinions,
            reasoning=f"Average agreement rate: {avg_agreement_rate:.1f}%, {dissent_count} dissenting opinions"
        )

    def _check_logical_consistency(
        self,
        opinions: List[AgentOpinion],
        debate_results: Dict[str, Any]
    ) -> float:
        """
        논리적 일관성 검증

        검증 항목:
        1. 의견 간 모순이 없는가?
        2. 근거가 결론을 뒷받침하는가?
        3. 시계열 논리가 맞는가?

        Returns:
            일관성 점수 (0-100)
        """
        consistency_score = 100.0

        # 상반된 주장 감지
        contradictions = 0
        for i, opinion1 in enumerate(opinions):
            for opinion2 in opinions[i+1:]:
                if opinion1.topic == opinion2.topic:
                    # 같은 주제에 대해 정반대 입장
                    if self._are_contradictory(opinion1.position, opinion2.position):
                        # 근거가 충분하면 healthy debate, 없으면 논리적 모순
                        if opinion1.confidence < 0.60 or opinion2.confidence < 0.60:
                            contradictions += 1

        # 모순 페널티
        consistency_score -= contradictions * 10

        return max(0.0, min(100.0, consistency_score))

    def _check_data_accuracy(
        self,
        opinions: List[AgentOpinion],
        market_data: Dict[str, Any]
    ) -> float:
        """
        데이터 정확성 검증

        Returns:
            정확성 점수 (0-100)
        """
        if not market_data:
            return 50.0  # 검증 불가능

        total_claims = 0
        accurate_claims = 0

        for opinion in opinions:
            numeric_claims = self._extract_numeric_claims(f"{opinion.position} {' '.join(opinion.evidence)}")
            for claim in numeric_claims:
                total_claims += 1
                if self._verify_numeric_claim(claim, market_data, tolerance=self.data_tolerance):
                    accurate_claims += 1

        if total_claims == 0:
            return 100.0  # 수치 주장이 없으면 정확

        accuracy = (accurate_claims / total_claims) * 100
        return accuracy

    def _assess_opinion_diversity(self, opinions: List[AgentOpinion]) -> float:
        """
        의견 다양성 평가

        다양성 지표:
        1. 서로 다른 입장의 개수
        2. 신뢰도 분포의 표준편차
        3. 독립적인 근거의 개수

        Returns:
            다양성 점수 (0-100)
        """
        if len(opinions) < 2:
            return 0.0

        # 1. 입장 다양성
        positions_list = [op.position for op in opinions]
        unique_positions = len(set(positions_list))
        position_diversity = (unique_positions / len(positions_list)) * 100

        # 2. 신뢰도 분포
        confidences = [op.confidence for op in opinions]
        confidence_std = np.std(confidences)
        confidence_diversity = min(confidence_std, 50.0) * 2  # 0-100 스케일

        # 3. 근거 다양성 (간단한 키워드 중복 체크)
        all_evidence = " ".join([" ".join(op.evidence) for op in opinions])
        words = re.findall(r'\b\w+\b', all_evidence.lower())
        unique_ratio = len(set(words)) / max(len(words), 1)
        evidence_diversity = unique_ratio * 100

        # 종합 점수
        diversity_score = (
            position_diversity * 0.4 +
            confidence_diversity * 0.3 +
            evidence_diversity * 0.3
        )

        return diversity_score

    def _synthesize_results(
        self,
        hallucination_check: HallucinationCheck,
        sycophancy_check: SycophancyCheck,
        logical_consistency: float,
        data_accuracy: float,
        opinion_diversity: float
    ) -> VerificationResult:
        """종합 평가"""
        issues_found = []
        warnings = []

        # Issue 수집
        if hallucination_check.contains_hallucination:
            issues_found.append(f"Hallucination risk detected ({len(hallucination_check.problematic_statements)} issues)")
            for stmt in hallucination_check.problematic_statements[:3]:  # 최대 3개만
                warnings.append(f"⚠️  {stmt}")

        if sycophancy_check.has_sycophancy:
            issues_found.append(f"Sycophancy detected (agreement rate: {sycophancy_check.agreement_rate:.1f}%)")
            warnings.append(f"⚠️  High agreement rate ({sycophancy_check.agreement_rate:.1f}%), only {sycophancy_check.dissent_count} dissenting opinions")

        if logical_consistency < 70:
            issues_found.append(f"Low logical consistency ({logical_consistency:.1f})")
            warnings.append(f"⚠️  Logical inconsistencies detected")

        if data_accuracy < 80:
            issues_found.append(f"Low data accuracy ({data_accuracy:.1f})")
            warnings.append(f"⚠️  Some numeric claims could not be verified")

        if opinion_diversity < self.min_diversity_score:
            issues_found.append(f"Low opinion diversity ({opinion_diversity:.1f})")
            warnings.append(f"⚠️  Lack of diverse perspectives")

        # 종합 점수 계산
        overall_score = (
            (100 - hallucination_check.confidence) * 0.3 +  # Hallucination 리스크 (역)
            (100 - sycophancy_check.agreement_rate) * 0.2 +  # Sycophancy 리스크 (역)
            logical_consistency * 0.2 +
            data_accuracy * 0.15 +
            opinion_diversity * 0.15
        )

        passed = (
            not hallucination_check.contains_hallucination and
            not sycophancy_check.has_sycophancy and
            logical_consistency >= 70 and
            data_accuracy >= 80 and
            opinion_diversity >= self.min_diversity_score
        )

        return VerificationResult(
            overall_score=overall_score,
            hallucination_risk=(100 - hallucination_check.confidence),
            sycophancy_risk=sycophancy_check.agreement_rate,
            logical_consistency=logical_consistency,
            data_accuracy=data_accuracy,
            opinion_diversity=opinion_diversity,
            issues_found=issues_found,
            warnings=warnings,
            passed=passed,
            timestamp=datetime.now().isoformat()
        )

    def _generate_reasoning(self, result: VerificationResult) -> str:
        """검증 결과 요약 생성"""
        if result.passed:
            return (
                f"✅ Verification PASSED (Score: {result.overall_score:.1f}/100)\n"
                f"   - Hallucination Risk: {result.hallucination_risk:.1f}%\n"
                f"   - Sycophancy Risk: {result.sycophancy_risk:.1f}%\n"
                f"   - Logical Consistency: {result.logical_consistency:.1f}%\n"
                f"   - Data Accuracy: {result.data_accuracy:.1f}%\n"
                f"   - Opinion Diversity: {result.opinion_diversity:.1f}%"
            )
        else:
            issues = "\n   - ".join(result.issues_found)
            return (
                f"❌ Verification FAILED (Score: {result.overall_score:.1f}/100)\n"
                f"Issues found:\n   - {issues}\n\n"
                f"Metrics:\n"
                f"   - Hallucination Risk: {result.hallucination_risk:.1f}%\n"
                f"   - Sycophancy Risk: {result.sycophancy_risk:.1f}%\n"
                f"   - Logical Consistency: {result.logical_consistency:.1f}%\n"
                f"   - Data Accuracy: {result.data_accuracy:.1f}%\n"
                f"   - Opinion Diversity: {result.opinion_diversity:.1f}%"
            )

    # Helper methods

    def _extract_tickers(self, text: str) -> List[str]:
        """텍스트에서 티커 추출 (대문자 2-5자)"""
        pattern = r'\b[A-Z]{2,5}\b'
        return re.findall(pattern, text.upper())

    def _extract_numeric_claims(self, text: str) -> List[str]:
        """수치 주장 추출 (예: "VIX is 15.2", "RRP at $5B")"""
        # 간단한 패턴: "X is/at/= Y" 형태
        pattern = r'(\w+)\s+(?:is|at|=|:)\s+([\d.,]+\s*[%$BMK]*)'
        matches = re.findall(pattern, text, re.IGNORECASE)
        return [f"{m[0]} {m[1]}" for m in matches]

    def _verify_numeric_claim(
        self,
        claim: str,
        market_data: Dict[str, Any],
        tolerance: float = 10.0
    ) -> bool:
        """수치 주장 검증 (tolerance % 이내면 OK)"""
        # 단순 구현: 키워드 매칭
        # 실제로는 더 정교한 NLP 필요
        claim_lower = claim.lower()

        for key, value in market_data.items():
            if key.lower() in claim_lower:
                try:
                    # 수치 추출
                    numbers = re.findall(r'[\d.]+', claim)
                    if numbers:
                        claimed_value = float(numbers[0])
                        actual_value = float(value) if isinstance(value, (int, float)) else None

                        if actual_value is not None:
                            diff_pct = abs(claimed_value - actual_value) / actual_value * 100
                            return diff_pct <= tolerance
                except (ValueError, ZeroDivisionError):
                    pass

        return True  # 검증 불가능하면 일단 통과 (보수적)

    def _detect_contradictions(self, text: str, market_data: Dict[str, Any]) -> List[str]:
        """논리적 모순 감지"""
        contradictions = []
        text_lower = text.lower()

        # 예시: "VIX가 10인데 높은 변동성" → 모순
        if 'vix' in market_data:
            vix = market_data['vix']
            if vix < 15 and ('high volatility' in text_lower or 'elevated volatility' in text_lower):
                contradictions.append(f"Claims high volatility but VIX is {vix} (low)")
            elif vix > 30 and ('low volatility' in text_lower or 'calm market' in text_lower):
                contradictions.append(f"Claims low volatility but VIX is {vix} (high)")

        return contradictions

    def _are_contradictory(self, stance1: str, stance2: str) -> bool:
        """두 입장이 정반대인지 확인"""
        opposites = {
            'bullish': 'bearish',
            'positive': 'negative',
            'buy': 'sell',
            'increase': 'decrease',
            'expansion': 'contraction'
        }

        s1 = stance1.lower()
        s2 = stance2.lower()

        for key, value in opposites.items():
            if (key in s1 and value in s2) or (value in s1 and key in s2):
                return True

        return False

    async def form_opinion(self, topic: str, context: Dict[str, Any]) -> AgentOpinion:
        """의견 형성 (검증 결과 기반)"""
        verification_result = context.get('verification_result', {})

        if not verification_result:
            return AgentOpinion(
                agent_role=self.config.role,
                topic=topic,
                position="unknown",
                confidence=0.0,
                strength=OpinionStrength.NEUTRAL,
                evidence=["No verification data available"]
            )

        passed = verification_result.get('passed', False)
        overall_score = verification_result.get('overall_score', 0.0)

        position = "approved" if passed else "rejected"
        confidence = overall_score / 100.0  # 0-100 → 0.0-1.0

        return AgentOpinion(
            agent_role=self.config.role,
            topic=topic,
            position=position,
            confidence=confidence,
            strength=OpinionStrength.STRONG_AGREE if abs(overall_score - 50) > 30 else OpinionStrength.NEUTRAL,
            evidence=[
                f"Verification score: {overall_score:.1f}/100",
                f"Issues found: {len(verification_result.get('issues_found', []))}",
                f"Passed: {passed}"
            ]
        )


# Test code
if __name__ == "__main__":
    import asyncio
    from core.schemas import AgentRequest, TaskPriority

    async def test_verification_agent():
        """Test VerificationAgent"""
        print("=" * 60)
        print("Testing VerificationAgent")
        print("=" * 60)

        # Mock opinions
        opinions = [
            AgentOpinion(
                agent_role=AgentRole.ANALYSIS,
                topic="market_outlook",
                position="bullish",
                confidence=0.85,
                strength=OpinionStrength.STRONG_AGREE,
                evidence=["SPY is at 694, strong momentum", "SPY price trend", "Low VIX"]
            ),
            AgentOpinion(
                agent_role=AgentRole.FORECAST,
                topic="market_outlook",
                position="bullish",
                confidence=0.90,
                strength=OpinionStrength.STRONG_AGREE,
                evidence=["Fed Funds rate stable, positive outlook", "Fed data", "Economic indicators"]
            ),
            AgentOpinion(
                agent_role=AgentRole.RESEARCH,
                topic="market_outlook",
                position="neutral",
                confidence=0.60,
                strength=OpinionStrength.NEUTRAL,
                evidence=["Mixed signals from different sectors", "Sector rotation", "Volume data"]
            )
        ]

        # Mock market data
        market_data = {
            'tickers': ['SPY', 'QQQ', 'TLT', 'GLD'],
            'vix': 14.5,
            'spy_price': 694.0,
            'fed_funds_rate': 4.33,
        }

        # Create agent
        agent = VerificationAgent()

        # Create request
        request = AgentRequest(
            task_id="test_verification",
            role=AgentRole.VERIFICATION,
            instruction="Verify the multi-agent debate results",
            context={
                'opinions': opinions,
                'market_data': market_data,
                'debate_results': {}
            },
            priority=TaskPriority.HIGH
        )

        # Execute
        response = await agent.execute(request)

        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"Status: {response.status}")
        print(f"Confidence: {response.confidence:.1f}%")
        print(f"\nReasoning:\n{response.reasoning}")

        result = response.result.get('verification_result', {})
        print(f"\nOverall Score: {result.get('overall_score', 0):.1f}/100")
        print(f"Passed: {result.get('passed', False)}")

        if result.get('warnings'):
            print("\nWarnings:")
            for warning in result['warnings']:
                print(f"  {warning}")

        print("=" * 60)

    asyncio.run(test_verification_agent())
