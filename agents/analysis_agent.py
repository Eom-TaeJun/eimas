#!/usr/bin/env python3
"""
Multi-Agent System - Analysis Agent
====================================
Wraps CriticalPathAggregator for agent framework integration

경제학적 의미:
- 기존 검증된 분석 로직 재사용 (Wrapper pattern)
- Rule-based opinion formation (객관적 지표 → 주관적 의견 변환)
- 정량적 분석 + 정성적 해석의 결합
"""

from typing import Dict, Any
from datetime import datetime

from agents.base_agent import BaseAgent, AgentConfig
from core.schemas import AgentRequest, AgentResponse, AgentOpinion, AgentRole, OpinionStrength

try:
    from lib.critical_path import CriticalPathAggregator, CriticalPathResult
    CRITICAL_PATH_AVAILABLE = True
except ImportError as e:
    CRITICAL_PATH_AVAILABLE = False
    print(f"Warning: CriticalPathAggregator not available - {e}")
    CriticalPathAggregator = None
    CriticalPathResult = None


class AnalysisAgent(BaseAgent):
    """
    Critical Path 분석 에이전트

    역할:
    1. 기존 CriticalPathAggregator.analyze() 실행 (Delegation)
    2. 결과를 에이전트 프레임워크 형식으로 변환
    3. 토론용 의견 형성 (rule-based)

    재사용:
    - RiskAppetiteUncertaintyIndex (Bekaert et al.)
    - EnhancedRegimeDetector
    - SpilloverNetwork (23 경로)
    - CryptoSentimentBlock (Granger Causality)
    """

    def __init__(self):
        """초기화"""
        config = AgentConfig(
            name="CriticalPathAnalyst",
            role=AgentRole.ANALYSIS,
            max_retries=2,
            timeout=120,
            verbose=True
        )
        super().__init__(config)

        if not CRITICAL_PATH_AVAILABLE:
            self.logger.error("CriticalPathAggregator not available!")
            self.aggregator = None
        else:
            self.aggregator = CriticalPathAggregator()
            self.logger.info("CriticalPathAggregator initialized")

    async def _execute(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Critical Path 분석 실행

        Input (from request.context):
            - market_data: Dict[str, pd.DataFrame] (OHLCV)

        Output:
            - critical_path_result: CriticalPathResult (serialized)
            - total_risk_score: float (0-100)
            - current_regime: str
            - path_contributions: Dict[str, float]
            - active_warnings: List[str]
            - confidence: float (0-1, from regime_confidence)

        Args:
            request: AgentRequest with market_data in context

        Returns:
            Dict with analysis results
        """
        if not self.aggregator:
            raise RuntimeError("CriticalPathAggregator not initialized")

        # Extract market_data
        market_data = request.context.get('market_data')
        if not market_data:
            raise ValueError("market_data not found in request context")

        self.logger.info(f"Running critical path analysis on {len(market_data)} tickers")

        # Run analysis
        result: CriticalPathResult = self.aggregator.analyze(market_data)

        self.logger.info(
            f"Analysis complete: Risk={result.total_risk_score:.1f}, "
            f"Regime={result.current_regime}"
        )

        # Package results
        return {
            'critical_path_result': result.to_dict(),  # Full result
            'total_risk_score': result.total_risk_score,
            'risk_level': result.risk_level,
            'current_regime': result.current_regime,
            'regime_confidence': result.regime_confidence,
            'transition_probability': result.transition_probability,
            'path_contributions': result.path_contributions,
            'path_distribution': result.path_distribution,
            'primary_risk_path': result.primary_risk_path,
            'active_warnings': result.active_warnings,
            'confidence': result.regime_confidence / 100.0,  # 0-1 scale
            'reasoning': self._generate_reasoning(result)
        }

    async def form_opinion(
        self,
        topic: str,
        context: Dict[str, Any]
    ) -> AgentOpinion:
        """
        주제에 대한 의견 형성 (Rule-based)

        지원 주제:
        - "market_outlook": 전체 시장 전망
        - "primary_risk": 주요 위험 요인
        - "regime_stability": 레짐 안정성
        - "crypto_correlation": 암호화폐 상관관계
        - "bubble_assessment": JP Morgan 5단계 버블 평가 (NEW)
        - "institutional_view": 기관 투자자 관점 종합 (NEW)

        Args:
            topic: 의견 주제
            context: CriticalPathResult 포함

        Returns:
            AgentOpinion 객체
        """
        # Extract CriticalPathResult from context
        if 'critical_path_result' in context:
            result_dict = context['critical_path_result']
            # Reconstruct from dict (simplified - use key fields)
            total_risk_score = result_dict.get('total_risk_score', 50.0)
            current_regime = result_dict.get('current_regime', 'NEUTRAL')
            regime_confidence = result_dict.get('regime_confidence', 50.0)
            transition_probability = result_dict.get('transition_probability', 0.5)
            primary_risk_path = result_dict.get('primary_risk_path', 'unknown')
            path_contributions = result_dict.get('path_contributions', {})
            active_warnings = result_dict.get('active_warnings', [])
        else:
            # Fallback to individual fields
            total_risk_score = context.get('total_risk_score', 50.0)
            current_regime = context.get('current_regime', 'NEUTRAL')
            regime_confidence = context.get('regime_confidence', 50.0)
            transition_probability = context.get('transition_probability', 0.5)
            primary_risk_path = context.get('primary_risk_path', 'unknown')
            path_contributions = context.get('path_contributions', {})
            active_warnings = context.get('active_warnings', [])

        # Route to topic-specific handler
        if transition_probability > 1.0:
            transition_probability /= 100.0

        if topic == "market_outlook":
            return self._opinion_market_outlook(
                total_risk_score, current_regime, regime_confidence,
                active_warnings, path_contributions
            )
        elif topic == "primary_risk":
            return self._opinion_primary_risk(
                primary_risk_path, path_contributions, total_risk_score
            )
        elif topic == "regime_stability":
            return self._opinion_regime_stability(
                current_regime, regime_confidence, transition_probability
            )
        elif topic == "crypto_correlation":
            # Would use crypto_result from context
            crypto_result = context.get('critical_path_result', {}).get('crypto_result', {})
            return self._opinion_crypto_correlation(crypto_result)
        elif topic == "bubble_assessment":
            # JP Morgan 5단계 버블 프레임워크 평가 (NEW)
            bubble_data = context.get('bubble_risk', {})
            return self._opinion_bubble_assessment(bubble_data)
        elif topic == "institutional_view":
            # 기관 투자자 관점 종합 (NEW)
            return self._opinion_institutional_view(
                total_risk_score, current_regime, regime_confidence,
                active_warnings, context
            )
        else:
            # Default: neutral opinion
            return AgentOpinion(
                agent_role=self.config.role,
                topic=topic,
                position=f"No specific analysis for topic: {topic}",
                strength=OpinionStrength.NEUTRAL,
                confidence=0.5,
                evidence=[],
                key_metrics={}
            )

    # ============================================================
    # Opinion Formation Helpers (Rule-based)
    # ============================================================

    def _opinion_market_outlook(
        self,
        total_risk_score: float,
        current_regime: str,
        regime_confidence: float,
        active_warnings: list,
        path_contributions: dict
    ) -> AgentOpinion:
        """
        시장 전망 의견 형성

        Mapping Rules:
        - risk < 30 AND regime in [BULL, EXPANSION] → Bullish
        - risk > 60 OR regime in [BEAR, CRISIS] → Bearish
        - else → Neutral
        """
        bullish_regimes = ['BULL', 'EXPANSION', 'BULLISH']
        bearish_regimes = ['BEAR', 'CONTRACTION', 'BEARISH', 'CRISIS']

        if total_risk_score < 30 and current_regime in bullish_regimes:
            position = "Bullish market conditions"
            strength = OpinionStrength.AGREE
            confidence = regime_confidence / 100.0
        elif total_risk_score > 60 or current_regime in bearish_regimes:
            position = "Bearish market conditions"
            strength = OpinionStrength.DISAGREE
            confidence = regime_confidence / 100.0
        else:
            position = "Neutral/Mixed market conditions"
            strength = OpinionStrength.NEUTRAL
            confidence = max(0.5, regime_confidence / 150.0)  # Lower confidence for neutral

        # Evidence
        evidence = [
            f"Total risk score: {total_risk_score:.1f}/100",
            f"Current regime: {current_regime} (confidence: {regime_confidence:.1f}%)",
        ]
        evidence.extend(active_warnings[:3])  # Top 3 warnings

        # Key metrics
        key_metrics = {
            'total_risk_score': total_risk_score,
            'regime_confidence': regime_confidence,
            **{f'path_{k}': v for k, v in list(path_contributions.items())[:5]}
        }

        # Caveats
        caveats = []
        if regime_confidence < 70:
            caveats.append(f"Low regime confidence ({regime_confidence:.1f}%) suggests uncertainty")
        if len(active_warnings) > 5:
            caveats.append(f"Multiple active warnings ({len(active_warnings)}) indicate elevated risk")

        return AgentOpinion(
            agent_role=self.config.role,
            topic="market_outlook",
            position=position,
            strength=strength,
            confidence=confidence,
            evidence=evidence,
            caveats=caveats,
            key_metrics=key_metrics
        )

    def _opinion_primary_risk(
        self,
        primary_risk_path: str,
        path_contributions: dict,
        total_risk_score: float
    ) -> AgentOpinion:
        """
        주요 위험 요인 의견 형성

        Position: Primary risk path name
        Confidence: Based on contribution magnitude
        """
        contribution = path_contributions.get(primary_risk_path, 0.0)

        position = f"Primary risk: {primary_risk_path}"

        # Confidence based on contribution magnitude
        if contribution > 40:
            strength = OpinionStrength.STRONG_AGREE
            confidence = 0.9
        elif contribution > 25:
            strength = OpinionStrength.AGREE
            confidence = 0.75
        else:
            strength = OpinionStrength.NEUTRAL
            confidence = 0.6

        evidence = [
            f"{primary_risk_path} contributes {contribution:.1f}% to total risk",
            f"Total risk score: {total_risk_score:.1f}/100"
        ]

        # Add top 3 contributors
        sorted_paths = sorted(path_contributions.items(), key=lambda x: x[1], reverse=True)
        for path, contrib in sorted_paths[:3]:
            evidence.append(f"{path}: {contrib:.1f}%")

        key_metrics = {
            'primary_contribution': contribution,
            **{k: v for k, v in sorted_paths[:5]}
        }

        return AgentOpinion(
            agent_role=self.config.role,
            topic="primary_risk",
            position=position,
            strength=strength,
            confidence=confidence,
            evidence=evidence,
            key_metrics=key_metrics
        )

    def _opinion_regime_stability(
        self,
        current_regime: str,
        regime_confidence: float,
        transition_probability: float
    ) -> AgentOpinion:
        """
        레짐 안정성 의견 형성

        Position: Stable vs Transitioning
        Confidence: Based on transition_probability
        """
        if transition_probability > 0.5:
            position = f"Regime unstable - high transition risk ({transition_probability:.1%})"
            strength = OpinionStrength.DISAGREE  # Disagree with stability
            confidence = transition_probability
        elif transition_probability > 0.3:
            position = f"Regime moderately stable - some transition risk"
            strength = OpinionStrength.NEUTRAL
            confidence = 0.7
        else:
            position = f"Regime stable - low transition risk"
            strength = OpinionStrength.AGREE  # Agree with stability
            confidence = 1.0 - transition_probability

        evidence = [
            f"Current regime: {current_regime}",
            f"Regime confidence: {regime_confidence:.1f}%",
            f"Transition probability: {transition_probability:.1%}"
        ]

        key_metrics = {
            'regime_confidence': regime_confidence,
            'transition_probability': transition_probability
        }

        caveats = []
        if regime_confidence < 60:
            caveats.append("Low regime confidence reduces stability assessment reliability")

        return AgentOpinion(
            agent_role=self.config.role,
            topic="regime_stability",
            position=position,
            strength=strength,
            confidence=confidence,
            evidence=evidence,
            caveats=caveats,
            key_metrics=key_metrics
        )

    def _opinion_crypto_correlation(self, crypto_result: dict) -> AgentOpinion:
        """
        암호화폐 상관관계 의견 형성

        (Simplified - would use full crypto_result in production)
        """
        # Placeholder - would analyze BTC-SPY correlation, Granger causality
        correlation = crypto_result.get('btc_spy_correlation', 0.0)

        if abs(correlation) > 0.7:
            position = f"High crypto-equity correlation ({correlation:.2f})"
            strength = OpinionStrength.AGREE
            confidence = 0.8
        elif abs(correlation) < 0.3:
            position = f"Low crypto-equity correlation - decoupled"
            strength = OpinionStrength.DISAGREE
            confidence = 0.7
        else:
            position = f"Moderate crypto-equity correlation"
            strength = OpinionStrength.NEUTRAL
            confidence = 0.6

        evidence = [
            f"BTC-SPY correlation: {correlation:.2f}"
        ]

        key_metrics = {
            'btc_spy_correlation': correlation
        }

        return AgentOpinion(
            agent_role=self.config.role,
            topic="crypto_correlation",
            position=position,
            strength=strength,
            confidence=confidence,
            evidence=evidence,
            key_metrics=key_metrics
        )

    def _opinion_bubble_assessment(self, bubble_data: dict) -> AgentOpinion:
        """
        JP Morgan 5단계 버블 프레임워크 기반 의견 형성 (NEW)

        Reference: docs/INSTITUTIONAL_REFERENCE.md
        - Stage 1: 패러다임 전환
        - Stage 2: 신용 가용성
        - Stage 3: 레버리지/밸류에이션 괴리
        - Stage 4: 투기적 피드백 루프
        - Stage 5: 붕괴 트리거
        """
        overall_status = bubble_data.get('overall_status', 'NONE')
        highest_risk_ticker = bubble_data.get('highest_risk_ticker', 'N/A')
        highest_risk_score = bubble_data.get('highest_risk_score', 0)
        risk_tickers = bubble_data.get('risk_tickers', [])

        # JP Morgan 프레임워크 데이터 추출
        jpmorgan_stage = 'None'
        historical_comparison = 'No comparison'
        if risk_tickers:
            for rt in risk_tickers:
                jpmorgan = rt.get('jpmorgan_framework', {})
                if jpmorgan:
                    jpmorgan_stage = jpmorgan.get('current_stage', 'None')
                    historical_comparison = jpmorgan.get('historical_comparison', 'No comparison')
                    break

        # 의견 형성
        if overall_status == 'DANGER':
            position = f"DANGER: {highest_risk_ticker} shows Stage 4 speculative loop ({historical_comparison})"
            strength = OpinionStrength.STRONG_DISAGREE
            confidence = 0.9
        elif overall_status == 'WARNING':
            position = f"WARNING: Elevated bubble risk in {highest_risk_ticker} ({jpmorgan_stage})"
            strength = OpinionStrength.DISAGREE
            confidence = 0.75
        elif overall_status == 'WATCH':
            position = f"WATCH: Monitor {highest_risk_ticker} for bubble formation"
            strength = OpinionStrength.NEUTRAL
            confidence = 0.6
        else:
            position = "No significant bubble risk detected"
            strength = OpinionStrength.AGREE
            confidence = 0.7

        evidence = [
            f"Overall Status: {overall_status}",
            f"JP Morgan Stage: {jpmorgan_stage}",
            f"Historical Comparison: {historical_comparison}",
            f"Highest Risk: {highest_risk_ticker} ({highest_risk_score:.1f})"
        ]

        key_metrics = {
            'bubble_status': overall_status,
            'highest_risk_score': highest_risk_score,
            'jpmorgan_stage': jpmorgan_stage
        }

        caveats = []
        if 'Dot-com' in historical_comparison:
            caveats.append("Warning: Historical parallel to 1990s dot-com bubble detected")
        if jpmorgan_stage and 'Stage 4' in str(jpmorgan_stage):
            caveats.append("Critical: Speculative feedback loop in progress")

        return AgentOpinion(
            agent_role=self.config.role,
            topic="bubble_assessment",
            position=position,
            strength=strength,
            confidence=confidence,
            evidence=evidence,
            caveats=caveats,
            key_metrics=key_metrics
        )

    def _opinion_institutional_view(
        self,
        total_risk_score: float,
        current_regime: str,
        regime_confidence: float,
        active_warnings: list,
        context: dict
    ) -> AgentOpinion:
        """
        기관 투자자 관점 종합 의견 형성 (NEW)

        References:
        - JP Morgan Asset Management: Time-series Shock Modeling, Mag7 vs Russell 493
        - Goldman Sachs: Gap-Bridging (시장 내재 vs 예측), AI 버블 DCF 검증
        - Berkshire Hathaway: 보수적 추정, Fair Value Hierarchy

        Key Methodology:
        1. 정량적 검증 필수 (회귀분석, DCF, 시나리오)
        2. 구조적 vs 순환적 분리
        3. 가치사슬 매핑
        4. 리스크 프리미엄 정량화
        """
        # Gap-Bridging: 시장 내재 기대 vs 분석 결과 비교
        bubble_data = context.get('bubble_risk', {})
        market_quality = context.get('market_quality', {})

        # 기관 투자자 체크리스트
        institutional_checks = []
        confidence_adjustments = []

        # 1. DCF/펀더멘털 검증 (Berkshire 방식)
        # Risk score < 30이면 낙관적 → 펀더멘털 검증 필요
        if total_risk_score < 30 and current_regime in ['BULL', 'EXPANSION']:
            institutional_checks.append("✓ Low risk + Bull regime: Fundamentals supportive")
            confidence_adjustments.append(+0.05)
        elif total_risk_score > 60:
            institutional_checks.append("⚠ High risk: Conservative positioning recommended (Berkshire approach)")
            confidence_adjustments.append(-0.10)

        # 2. 버블 리스크 오버레이 (JP Morgan 5단계)
        bubble_status = bubble_data.get('overall_status', 'NONE')
        if bubble_status in ['WARNING', 'DANGER']:
            institutional_checks.append(f"⚠ Bubble risk: {bubble_status} (JP Morgan Framework)")
            confidence_adjustments.append(-0.15)
        else:
            institutional_checks.append("✓ No bubble warning (JP Morgan Framework)")
            confidence_adjustments.append(+0.05)

        # 3. 시장 미세구조 품질 (Goldman approach)
        avg_liquidity = market_quality.get('avg_liquidity_score', 50)
        if avg_liquidity < 30:
            institutional_checks.append("⚠ Low liquidity: Execution risk elevated")
            confidence_adjustments.append(-0.10)
        elif avg_liquidity > 70:
            institutional_checks.append("✓ High liquidity: Good execution conditions")
            confidence_adjustments.append(+0.05)

        # 4. 레짐 신뢰도 (GMM Entropy 기반)
        if regime_confidence < 60:
            institutional_checks.append("⚠ Low regime confidence: Mixed signals")
            confidence_adjustments.append(-0.10)
        elif regime_confidence > 80:
            institutional_checks.append("✓ High regime confidence: Clear directional signal")
            confidence_adjustments.append(+0.05)

        # 최종 신뢰도 계산
        base_confidence = 0.65
        final_confidence = max(0.3, min(0.95, base_confidence + sum(confidence_adjustments)))

        # 포지션 결정
        bullish_signals = sum(1 for c in institutional_checks if '✓' in c)
        bearish_signals = sum(1 for c in institutional_checks if '⚠' in c)

        if bullish_signals > bearish_signals and total_risk_score < 50:
            position = "Institutional View: CONSTRUCTIVE - Fundamentals support risk-on"
            strength = OpinionStrength.AGREE
        elif bearish_signals > bullish_signals or total_risk_score > 60:
            position = "Institutional View: CAUTIOUS - Defensive positioning recommended"
            strength = OpinionStrength.DISAGREE
        else:
            position = "Institutional View: NEUTRAL - Balanced approach, await clarity"
            strength = OpinionStrength.NEUTRAL

        evidence = institutional_checks + [
            f"Base confidence: {base_confidence:.0%}",
            f"Adjustment: {sum(confidence_adjustments):+.0%}",
            f"Final confidence: {final_confidence:.0%}"
        ]

        key_metrics = {
            'institutional_bullish_signals': bullish_signals,
            'institutional_bearish_signals': bearish_signals,
            'confidence_adjustment': sum(confidence_adjustments),
            'liquidity_score': avg_liquidity,
            'regime_confidence': regime_confidence
        }

        caveats = []
        if len(active_warnings) > 5:
            caveats.append(f"Multiple active warnings ({len(active_warnings)}) require monitoring")
        if bubble_status == 'DANGER':
            caveats.append("CRITICAL: JP Morgan Stage 4/5 bubble risk - extreme caution")

        return AgentOpinion(
            agent_role=self.config.role,
            topic="institutional_view",
            position=position,
            strength=strength,
            confidence=final_confidence,
            evidence=evidence,
            caveats=caveats,
            key_metrics=key_metrics
        )

    # ============================================================
    # Helper Methods
    # ============================================================

    def _generate_reasoning(self, result: CriticalPathResult) -> str:
        """
        Generate reasoning text from CriticalPathResult

        Args:
            result: CriticalPathResult object

        Returns:
            Reasoning text
        """
        reasoning_parts = [
            f"Critical Path Analysis completed with {result.risk_level} risk level.",
            f"Total risk score: {result.total_risk_score:.1f}/100.",
            f"Current regime: {result.current_regime} (confidence: {result.regime_confidence:.1f}%).",
            f"Primary risk pathway: {result.primary_risk_path}.",
        ]

        if result.active_warnings:
            reasoning_parts.append(f"Active warnings: {len(result.active_warnings)}.")

        return " ".join(reasoning_parts)


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    import asyncio

    async def test_analysis_agent():
        """Test with mock data"""
        print("="*60)
        print("AnalysisAgent Test")
        print("="*60)

        if not CRITICAL_PATH_AVAILABLE:
            print("ERROR: CriticalPathAggregator not available")
            return

        agent = AnalysisAgent()

        # Mock request (would need real market_data in production)
        print("\nNote: Full test requires real market_data from collectors.DataManager")
        print("This is a basic initialization test only.\n")

        # Test opinion formation with mock context
        mock_context = {
            'total_risk_score': 45.0,
            'current_regime': 'TRANSITION',
            'regime_confidence': 65.0,
            'transition_probability': 0.35,
            'primary_risk_path': 'liquidity',
            'path_contributions': {
                'liquidity': 30.0,
                'credit': 20.0,
                'volatility': 15.0
            },
            'active_warnings': [
                'Credit spread widening',
                'VIX elevated'
            ]
        }

        opinion = await agent.form_opinion("market_outlook", mock_context)

        print("Opinion Generated:")
        print(f"  Topic: {opinion.topic}")
        print(f"  Position: {opinion.position}")
        print(f"  Strength: {opinion.strength.value}")
        print(f"  Confidence: {opinion.confidence:.2%}")
        print(f"  Evidence: {opinion.evidence}")
        print(f"  Key Metrics: {opinion.key_metrics}")

    asyncio.run(test_analysis_agent())
