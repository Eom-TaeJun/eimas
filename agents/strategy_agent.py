"""
Strategy Agent - Trading Recommendations

Critical Path 분석 결과를 바탕으로 매매 전략 제안
ECON_AI_AGENT_SYSTEM.md Section 4 구현

Author: EIMAS Team
"""

import asyncio
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from agents.base_agent import BaseAgent, AgentConfig
from core.schemas import AgentRequest, AgentResponse, AgentRole
from lib.portfolio_optimizer import HRPOptimizer


# ============================================================================
# Enums and Data Classes
# ============================================================================

class SignalStrength(Enum):
    """신호 강도"""
    STRONG = "STRONG"           # 강한 신호
    MODERATE = "MODERATE"       # 중간 신호
    WEAK = "WEAK"               # 약한 신호


class ActionType(Enum):
    """행동 유형"""
    STRONG_BUY = "STRONG_BUY"       # 적극 매수
    BUY = "BUY"                     # 매수
    HOLD = "HOLD"                   # 보유
    SELL = "SELL"                   # 매도
    STRONG_SELL = "STRONG_SELL"    # 적극 매도
    AVOID = "AVOID"                # 회피


class TimeHorizon(Enum):
    """투자 기간"""
    SHORT = "short"      # 1개월 이내
    MEDIUM = "medium"    # 1-6개월
    LONG = "long"        # 6개월 이상


class RiskLevel(Enum):
    """리스크 수준"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


@dataclass
class MarketState:
    """현재 시장 상태"""
    regime: str                    # expansion, contraction, transition
    risk_level: RiskLevel
    volatility_percentile: float   # 0-100
    trend_direction: str           # bullish, bearish, neutral
    key_indicators: Dict[str, float]


@dataclass
class CriticalPathState:
    """Critical Path 상태"""
    primary_drivers: List[str]      # 주요 선행 지표
    current_signals: Dict[str, str] # 지표별 현재 신호
    propagation_time: int           # 평균 전파 시간 (일)
    confidence: float


@dataclass
class TradeRecommendation:
    """개별 매매 추천"""
    asset: str
    action: ActionType
    strength: SignalStrength
    confidence: float
    target_allocation: float        # 목표 비중 (%)
    entry_range: Tuple[float, float]  # 진입 가격대
    stop_loss: float                # 손절가
    take_profit: float              # 익절가
    time_horizon: TimeHorizon
    risk_reward_ratio: float
    reasoning: str
    triggers: List[str]             # 진입/청산 조건


@dataclass
class PortfolioStrategy:
    """포트폴리오 전략"""
    timestamp: datetime
    market_state: MarketState
    critical_path: CriticalPathState
    recommendations: List[TradeRecommendation]
    overall_stance: str             # defensive, neutral, aggressive
    cash_allocation: float          # 현금 비중 권고
    hedging_recommendations: List[str]
    risk_warnings: List[str]
    key_events_to_watch: List[str]
    summary: str


# ============================================================================
# Strategy Agent
# ============================================================================

class StrategyAgent(BaseAgent):
    """
    매매 전략 제안 에이전트

    역할:
    - Critical Path 분석 결과 해석
    - Top-Down 상태 진단 통합
    - 자산별 매수/매도/보유 추천
    - 리스크 관리 권고
    """

    # 경제학적 규칙 기반 전략 매핑
    REGIME_STRATEGY_MAP = {
        "expansion_early": {
            "stance": "aggressive",
            "equity_bias": 0.7,
            "bond_bias": 0.2,
            "cash_bias": 0.1,
            "preferred_sectors": ["technology", "consumer_discretionary", "industrials"],
            "avoid_sectors": ["utilities", "consumer_staples"]
        },
        "expansion_late": {
            "stance": "neutral",
            "equity_bias": 0.5,
            "bond_bias": 0.35,
            "cash_bias": 0.15,
            "preferred_sectors": ["energy", "materials", "financials"],
            "avoid_sectors": ["technology", "consumer_discretionary"]
        },
        "contraction": {
            "stance": "defensive",
            "equity_bias": 0.3,
            "bond_bias": 0.5,
            "cash_bias": 0.2,
            "preferred_sectors": ["utilities", "healthcare", "consumer_staples"],
            "avoid_sectors": ["technology", "industrials", "materials"]
        },
        "transition": {
            "stance": "cautious",
            "equity_bias": 0.4,
            "bond_bias": 0.4,
            "cash_bias": 0.2,
            "preferred_sectors": [],
            "avoid_sectors": []
        }
    }

    # Fed 정책 영향 매핑
    FED_POLICY_IMPACT = {
        "hawkish": {
            "bond_action": ActionType.SELL,
            "duration_preference": "short",
            "equity_sectors_hurt": ["technology", "real_estate", "utilities"],
            "equity_sectors_benefit": ["financials", "energy"]
        },
        "dovish": {
            "bond_action": ActionType.BUY,
            "duration_preference": "long",
            "equity_sectors_hurt": ["financials"],
            "equity_sectors_benefit": ["technology", "real_estate", "utilities"]
        },
        "neutral": {
            "bond_action": ActionType.HOLD,
            "duration_preference": "medium",
            "equity_sectors_hurt": [],
            "equity_sectors_benefit": []
        }
    }

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        api_key: Optional[str] = None
    ):
        if config is None:
            config = AgentConfig(
                name="StrategyAgent",
                role=AgentRole.STRATEGY,
                model="claude-sonnet"
            )
        super().__init__(config)

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    async def _execute(self, request: AgentRequest) -> Any:
        """
        전략 수립 실행

        Args:
            request: AgentRequest containing:
                - market_state: MarketState (in context)
                - critical_path: CriticalPathState (in context)
                - portfolio: Optional[Dict] (current portfolio in context)
                - market_data: Optional[pd.DataFrame] for HRP (in context)

        Returns:
            AgentResponse containing PortfolioStrategy
        """
        market_state = request.context.get('market_state')
        critical_path = request.context.get('critical_path')
        current_portfolio = request.context.get('portfolio', {})
        risk_tolerance = request.context.get('risk_tolerance', 'moderate')
        market_data = request.context.get('market_data')

        # 1. GC-HRP로 최적 가중치 계산 (Quantitative)
        hrp_weights = {}
        if market_data is not None and not market_data.empty:
            try:
                optimizer = HRPOptimizer()
                hrp_weights = optimizer.optimize(market_data.pct_change().dropna())
            except Exception as e:
                self.logger.warning(f"HRP Optimization failed: {e}")

        # 2. 시장 상태 기반 기본 전략 결정
        base_strategy = self._determine_base_strategy(market_state)

        # 3. Critical Path 신호 해석
        path_signals = self._interpret_critical_path(critical_path)

        # 4. 자산별 추천 생성
        recommendations = await self._generate_recommendations(
            market_state,
            critical_path,
            base_strategy,
            path_signals,
            risk_tolerance
        )

        # 5. 리스크 경고 생성
        risk_warnings = self._generate_risk_warnings(market_state, critical_path)

        # 6. 주요 관찰 이벤트
        events_to_watch = self._identify_key_events(critical_path)

        # 7. 종합 요약
        summary = await self._generate_strategy_summary(
            market_state, recommendations, risk_warnings
        )

        strategy = PortfolioStrategy(
            timestamp=datetime.now(),
            market_state=market_state,
            critical_path=critical_path,
            recommendations=recommendations,
            overall_stance=base_strategy['stance'],
            cash_allocation=base_strategy.get('cash_bias', 0.15),
            hedging_recommendations=self._generate_hedging_advice(market_state),
            risk_warnings=risk_warnings,
            key_events_to_watch=events_to_watch,
            summary=summary
        )

        # Return dict matching AgentResponse content expectation or the object itself
        # Wrapper logic in orchestrator handles AgentResponse creation usually, 
        # but BaseAgent._execute typically returns the content dict.
        return {
            "content": strategy,
            "hrp_weights": hrp_weights,
            "reasoning": summary,
            "recommendations": [asdict(r) for r in recommendations] if hasattr(recommendations[0], 'asdict') else str(recommendations)
        }

    def _determine_base_strategy(self, market_state: MarketState) -> Dict:
        """시장 상태 기반 기본 전략 결정"""
        regime = market_state.regime.lower()

        # 레짐 매핑
        if 'expansion' in regime and 'early' in regime:
            strategy_key = 'expansion_early'
        elif 'expansion' in regime:
            strategy_key = 'expansion_late'
        elif 'contraction' in regime:
            strategy_key = 'contraction'
        else:
            strategy_key = 'transition'

        base = self.REGIME_STRATEGY_MAP[strategy_key].copy()

        # 변동성에 따른 조정
        if market_state.volatility_percentile > 80:
            base['cash_bias'] = min(base['cash_bias'] * 1.5, 0.4)
            base['equity_bias'] = max(base['equity_bias'] * 0.8, 0.2)

        # 리스크 수준에 따른 조정
        if market_state.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            base['stance'] = 'defensive'
            base['cash_bias'] = min(base['cash_bias'] * 1.3, 0.35)

        return base

    def _interpret_critical_path(
        self,
        critical_path: CriticalPathState
    ) -> Dict[str, Any]:
        """Critical Path 신호 해석"""
        signals = {
            'overall_direction': 'neutral',
            'strength': SignalStrength.MODERATE,
            'time_to_impact': critical_path.propagation_time,
            'driver_signals': {}
        }

        # 각 드라이버 신호 해석
        bullish_count = 0
        bearish_count = 0

        for driver in critical_path.primary_drivers:
            signal = critical_path.current_signals.get(driver, 'neutral')
            signals['driver_signals'][driver] = signal

            if signal in ['positive', 'bullish', 'up']:
                bullish_count += 1
            elif signal in ['negative', 'bearish', 'down']:
                bearish_count += 1

        # 전체 방향 결정
        total = len(critical_path.primary_drivers)
        if total > 0:
            if bullish_count > total * 0.6:
                signals['overall_direction'] = 'bullish'
                signals['strength'] = SignalStrength.STRONG if bullish_count > total * 0.8 else SignalStrength.MODERATE
            elif bearish_count > total * 0.6:
                signals['overall_direction'] = 'bearish'
                signals['strength'] = SignalStrength.STRONG if bearish_count > total * 0.8 else SignalStrength.MODERATE

        return signals

    async def _generate_recommendations(
        self,
        market_state: MarketState,
        critical_path: CriticalPathState,
        base_strategy: Dict,
        path_signals: Dict,
        risk_tolerance: str
    ) -> List[TradeRecommendation]:
        """자산별 매매 추천 생성"""
        recommendations = []

        # 1. 주식 추천
        equity_rec = self._generate_equity_recommendation(
            market_state, base_strategy, path_signals, risk_tolerance
        )
        if equity_rec:
            recommendations.append(equity_rec)

        # 2. 채권 추천
        bond_rec = self._generate_bond_recommendation(
            market_state, base_strategy, risk_tolerance
        )
        if bond_rec:
            recommendations.append(bond_rec)

        # 3. 섹터별 추천 (선호 섹터)
        for sector in base_strategy.get('preferred_sectors', [])[:3]:
            sector_rec = self._generate_sector_recommendation(
                sector, ActionType.BUY, market_state, path_signals
            )
            if sector_rec:
                recommendations.append(sector_rec)

        # 4. 회피 섹터
        for sector in base_strategy.get('avoid_sectors', [])[:2]:
            avoid_rec = self._generate_sector_recommendation(
                sector, ActionType.AVOID, market_state, path_signals
            )
            if avoid_rec:
                recommendations.append(avoid_rec)

        return recommendations

    def _generate_equity_recommendation(
        self,
        market_state: MarketState,
        base_strategy: Dict,
        path_signals: Dict,
        risk_tolerance: str
    ) -> Optional[TradeRecommendation]:
        """주식 전체 추천"""
        direction = path_signals['overall_direction']
        strength = path_signals['strength']

        # 기본 액션 결정
        if direction == 'bullish':
            if strength == SignalStrength.STRONG:
                action = ActionType.STRONG_BUY
            else:
                action = ActionType.BUY
        elif direction == 'bearish':
            if strength == SignalStrength.STRONG:
                action = ActionType.STRONG_SELL
            else:
                action = ActionType.SELL
        else:
            action = ActionType.HOLD

        # 리스크 허용도에 따른 조정
        if risk_tolerance == 'conservative':
            if action in [ActionType.STRONG_BUY, ActionType.BUY]:
                action = ActionType.HOLD
            target_allocation = base_strategy['equity_bias'] * 0.7
        elif risk_tolerance == 'aggressive':
            target_allocation = base_strategy['equity_bias'] * 1.2
        else:
            target_allocation = base_strategy['equity_bias']

        target_allocation = min(target_allocation, 0.8)  # 최대 80%

        return TradeRecommendation(
            asset="US_EQUITY",
            action=action,
            strength=strength,
            confidence=path_signals.get('confidence', 0.7),
            target_allocation=target_allocation,
            entry_range=(0.0, 0.0),  # 지수 수준에서는 의미 없음
            stop_loss=0.0,
            take_profit=0.0,
            time_horizon=TimeHorizon.MEDIUM,
            risk_reward_ratio=1.0,
            reasoning=self._generate_reasoning('equity', market_state, path_signals),
            triggers=self._generate_triggers(direction)
        )

    def _generate_bond_recommendation(
        self,
        market_state: MarketState,
        base_strategy: Dict,
        risk_tolerance: str
    ) -> Optional[TradeRecommendation]:
        """채권 추천"""
        # 금리 전망 기반
        rate_trend = market_state.key_indicators.get('rate_trend', 'stable')

        if rate_trend == 'falling':
            action = ActionType.BUY
            duration = "long"
        elif rate_trend == 'rising':
            action = ActionType.SELL
            duration = "short"
        else:
            action = ActionType.HOLD
            duration = "medium"

        target_allocation = base_strategy.get('bond_bias', 0.3)

        return TradeRecommendation(
            asset="US_BONDS",
            action=action,
            strength=SignalStrength.MODERATE,
            confidence=0.6,
            target_allocation=target_allocation,
            entry_range=(0.0, 0.0),
            stop_loss=0.0,
            take_profit=0.0,
            time_horizon=TimeHorizon.MEDIUM,
            risk_reward_ratio=1.0,
            reasoning=f"금리 전망: {rate_trend}, 선호 듀레이션: {duration}",
            triggers=[f"10Y Treasury yield {rate_trend}"]
        )

    def _generate_sector_recommendation(
        self,
        sector: str,
        action: ActionType,
        market_state: MarketState,
        path_signals: Dict
    ) -> Optional[TradeRecommendation]:
        """섹터별 추천"""
        sector_names = {
            'technology': 'Technology',
            'healthcare': 'Healthcare',
            'financials': 'Financials',
            'energy': 'Energy',
            'materials': 'Materials',
            'industrials': 'Industrials',
            'utilities': 'Utilities',
            'consumer_discretionary': 'Consumer Discretionary',
            'consumer_staples': 'Consumer Staples',
            'real_estate': 'Real Estate'
        }

        display_name = sector_names.get(sector, sector.title())

        return TradeRecommendation(
            asset=f"SECTOR_{sector.upper()}",
            action=action,
            strength=SignalStrength.MODERATE,
            confidence=0.6,
            target_allocation=0.1 if action == ActionType.BUY else 0.0,
            entry_range=(0.0, 0.0),
            stop_loss=0.0,
            take_profit=0.0,
            time_horizon=TimeHorizon.MEDIUM,
            risk_reward_ratio=1.0,
            reasoning=f"{display_name} 섹터 - 현재 레짐에서 {'선호' if action == ActionType.BUY else '회피'}",
            triggers=[f"{display_name} sector rotation signal"]
        )

    def _generate_risk_warnings(
        self,
        market_state: MarketState,
        critical_path: CriticalPathState
    ) -> List[str]:
        """리스크 경고 생성"""
        warnings = []

        # 변동성 경고
        if market_state.volatility_percentile > 80:
            warnings.append(
                f"높은 변동성 환경 (상위 {market_state.volatility_percentile:.0f}%ile) - "
                "포지션 사이즈 축소 권고"
            )

        # 리스크 수준 경고
        if market_state.risk_level == RiskLevel.VERY_HIGH:
            warnings.append(
                "매우 높은 리스크 환경 - 신규 진입 자제, 헷지 강화 필요"
            )
        elif market_state.risk_level == RiskLevel.HIGH:
            warnings.append(
                "높은 리스크 환경 - 손절 라인 타이트하게 설정"
            )

        # Critical Path 경고
        if critical_path.confidence < 0.5:
            warnings.append(
                "Critical Path 신호 불명확 - 추가 확인 필요, 관망 권고"
            )

        # 전파 시간 경고
        if critical_path.propagation_time < 7:
            warnings.append(
                f"신호 전파 시간 {critical_path.propagation_time}일 - "
                "빠른 시장 반응 예상, 민첩한 대응 필요"
            )

        return warnings

    def _generate_hedging_advice(self, market_state: MarketState) -> List[str]:
        """헷지 권고"""
        advice = []

        if market_state.risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            advice.append("VIX 콜옵션 또는 VIX ETF 매수 고려")
            advice.append("포트폴리오 풋옵션 구매 검토")

        if market_state.volatility_percentile > 70:
            advice.append("현금 비중 확대")
            advice.append("저베타 주식으로 포트폴리오 재조정")

        if market_state.trend_direction == 'bearish':
            advice.append("인버스 ETF 소규모 편입 고려")

        return advice if advice else ["현재 추가 헷지 불필요"]

    def _identify_key_events(self, critical_path: CriticalPathState) -> List[str]:
        """주요 관찰 이벤트 식별"""
        events = []

        # Primary drivers 기반 이벤트
        driver_events = {
            'fed_policy': 'FOMC 회의 및 Fed 발언',
            'employment': '고용보고서 (NFP)',
            'inflation': 'CPI/PCE 발표',
            'gdp': 'GDP 발표',
            'earnings': '기업 실적 시즌',
            'credit_spread': '신용 스프레드 변화',
            'yield_curve': '수익률 곡선 변화'
        }

        for driver in critical_path.primary_drivers:
            if driver.lower() in driver_events:
                events.append(driver_events[driver.lower()])

        # 기본 이벤트
        if not events:
            events = [
                "FOMC 정책 결정",
                "주요 경제지표 발표",
                "기업 실적 발표"
            ]

        return events

    def _generate_reasoning(
        self,
        asset_type: str,
        market_state: MarketState,
        path_signals: Dict
    ) -> str:
        """추천 이유 생성"""
        direction = path_signals['overall_direction']
        regime = market_state.regime

        if asset_type == 'equity':
            if direction == 'bullish':
                return (f"레짐: {regime}, Critical Path 신호 강세. "
                       f"선행 지표들이 긍정적 방향을 가리킴. "
                       f"신호 전파까지 약 {path_signals['time_to_impact']}일 예상.")
            elif direction == 'bearish':
                return (f"레짐: {regime}, Critical Path 신호 약세. "
                       f"선행 지표들이 부정적 방향을 가리킴. "
                       f"리스크 관리 강화 필요.")
            else:
                return (f"레짐: {regime}, Critical Path 신호 불명확. "
                       f"명확한 방향성 확인까지 관망 권고.")

        return f"시장 상태: {regime}"

    def _generate_triggers(self, direction: str) -> List[str]:
        """진입/청산 트리거 생성"""
        if direction == 'bullish':
            return [
                "VIX 하락 및 20 이하 유지",
                "주요 지수 20일 이평선 상향 돌파",
                "Credit spread 축소 확인"
            ]
        elif direction == 'bearish':
            return [
                "VIX 30 돌파",
                "주요 지수 50일 이평선 하향 이탈",
                "Credit spread 급등"
            ]
        else:
            return [
                "명확한 방향성 신호 확인 후 진입",
                "변동성 안정화 확인"
            ]

    async def _generate_strategy_summary(
        self,
        market_state: MarketState,
        recommendations: List[TradeRecommendation],
        risk_warnings: List[str]
    ) -> str:
        """전략 요약 생성"""
        buy_recs = [r for r in recommendations if r.action in [ActionType.BUY, ActionType.STRONG_BUY]]
        sell_recs = [r for r in recommendations if r.action in [ActionType.SELL, ActionType.STRONG_SELL]]

        summary_parts = []

        # 시장 상태
        summary_parts.append(
            f"현재 시장 레짐: {market_state.regime}, "
            f"리스크 수준: {market_state.risk_level.value}"
        )

        # 추천 요약
        if buy_recs:
            assets = ', '.join([r.asset for r in buy_recs[:3]])
            summary_parts.append(f"매수 권고: {assets}")

        if sell_recs:
            assets = ', '.join([r.asset for r in sell_recs[:3]])
            summary_parts.append(f"매도/회피 권고: {assets}")

        # 리스크 요약
        if risk_warnings:
            summary_parts.append(f"주요 경고: {risk_warnings[0]}")

        return ' | '.join(summary_parts)

    async def form_opinion(self, topic: str, context: Dict[str, Any]) -> "AgentOpinion":
        """
        특정 주제에 대한 의견 형성

        Returns:
            AgentOpinion 표준화된 의견 객체
        """
        from core.schemas import AgentOpinion, OpinionStrength

        # 컨텍스트에서 시장 상태 추출
        risk_score = context.get('total_risk_score', 50.0)
        regime = context.get('current_regime', 'NEUTRAL')
        risk_level = context.get('risk_level', 'MEDIUM')

        # 전략적 스탠스 결정
        stance, position = self._determine_strategic_stance(topic, risk_score, regime, risk_level)
        confidence = self._calculate_strategic_confidence(context)

        # OpinionStrength 결정
        if stance in ['aggressive', 'BULLISH']:
            if confidence > 0.7:
                strength = OpinionStrength.STRONG_AGREE
            else:
                strength = OpinionStrength.AGREE
        elif stance in ['defensive', 'BEARISH']:
            if confidence > 0.7:
                strength = OpinionStrength.STRONG_DISAGREE
            else:
                strength = OpinionStrength.DISAGREE
        else:
            strength = OpinionStrength.NEUTRAL

        # 증거 수집
        evidence = self._gather_strategic_evidence(context)

        return AgentOpinion(
            agent_role=self.config.role,
            topic=topic,
            position=position,
            strength=strength,
            confidence=confidence,
            evidence=evidence,
            caveats=self._generate_caveats(context),
            key_metrics={
                'risk_score': risk_score,
                'suggested_equity_weight': self._suggest_equity_weight(risk_score, regime),
                'suggested_cash_weight': self._suggest_cash_weight(risk_score)
            }
        )

    def _determine_strategic_stance(
        self,
        topic: str,
        risk_score: float,
        regime: str,
        risk_level: str
    ) -> Tuple[str, str]:
        """전략적 스탠스와 포지션 문장 결정"""
        regime_upper = regime.upper() if isinstance(regime, str) else 'NEUTRAL'

        if topic == "market_outlook":
            if risk_score < 40 and regime_upper in ['BULL', 'EXPANSION', 'EXPANSION_EARLY']:
                return "aggressive", f"Bullish outlook: Low risk ({risk_score:.1f}) in {regime} regime supports equity overweight"
            elif risk_score > 60 or regime_upper in ['BEAR', 'CONTRACTION']:
                return "defensive", f"Bearish outlook: Elevated risk ({risk_score:.1f}) warrants defensive positioning"
            else:
                return "neutral", f"Neutral outlook: Balanced risk ({risk_score:.1f}) in {regime} regime"

        elif topic == "primary_risk":
            if risk_score > 70:
                return "defensive", f"High risk environment ({risk_score:.1f}/100): Prioritize capital preservation"
            elif risk_score < 30:
                return "aggressive", f"Low risk environment ({risk_score:.1f}/100): Opportunities for alpha generation"
            else:
                return "neutral", f"Moderate risk ({risk_score:.1f}/100): Balanced approach recommended"

        elif topic == "regime_stability":
            trans_prob = context.get('transition_probability', 0) if 'context' in dir() else 0
            if trans_prob > 0.5:
                return "defensive", f"High regime transition risk ({trans_prob:.1%}): Position for volatility"
            else:
                return "neutral", f"Stable regime expected: Current positioning appropriate"

        return "neutral", f"Strategy assessment for {topic}: Balanced approach"

    def _calculate_strategic_confidence(self, context: Dict[str, Any]) -> float:
        """전략 신뢰도 계산"""
        base_confidence = 0.5

        # 레짐 신뢰도 반영
        regime_conf = context.get('regime_confidence', 50)
        if regime_conf > 1:  # 0-100 스케일인 경우
            regime_conf /= 100
        base_confidence += regime_conf * 0.2

        # 전이 확률이 낮으면 신뢰도 증가
        trans_prob = context.get('transition_probability', 0.5)
        if trans_prob > 1:
            trans_prob /= 100
        base_confidence += (1 - trans_prob) * 0.1

        return min(max(base_confidence, 0.3), 0.85)

    def _gather_strategic_evidence(self, context: Dict[str, Any]) -> List[str]:
        """전략적 증거 수집"""
        evidence = []

        risk_score = context.get('total_risk_score')
        if risk_score is not None:
            evidence.append(f"Risk Score: {risk_score:.1f}/100")

        regime = context.get('current_regime')
        if regime:
            evidence.append(f"Market Regime: {regime}")

        path_contributions = context.get('path_contributions', {})
        if path_contributions:
            top_path = max(path_contributions.items(), key=lambda x: x[1], default=(None, 0))
            if top_path[0]:
                evidence.append(f"Primary Risk Path: {top_path[0]} ({top_path[1]:.1f}%)")

        warnings = context.get('active_warnings', [])
        if warnings:
            evidence.append(f"Active Warnings: {len(warnings)} detected")

        return evidence if evidence else ["Insufficient data for detailed evidence"]

    def _generate_caveats(self, context: Dict[str, Any]) -> List[str]:
        """주의사항 생성"""
        caveats = []

        trans_prob = context.get('transition_probability', 0)
        if trans_prob > 1:
            trans_prob /= 100
        if trans_prob > 0.3:
            caveats.append(f"Regime transition risk elevated ({trans_prob:.1%})")

        warnings = context.get('active_warnings', [])
        if len(warnings) > 3:
            caveats.append(f"Multiple risk signals active ({len(warnings)})")

        if not caveats:
            caveats.append("Standard market conditions apply")

        return caveats

    def _suggest_equity_weight(self, risk_score: float, regime: str) -> float:
        """권장 주식 비중"""
        regime_upper = regime.upper() if isinstance(regime, str) else 'NEUTRAL'

        base_weight = 0.6  # 기본 60%

        # 리스크 기반 조정
        if risk_score > 60:
            base_weight -= 0.15
        elif risk_score < 40:
            base_weight += 0.1

        # 레짐 기반 조정
        if regime_upper in ['BULL', 'EXPANSION', 'EXPANSION_EARLY']:
            base_weight += 0.1
        elif regime_upper in ['BEAR', 'CONTRACTION']:
            base_weight -= 0.15

        return min(max(base_weight, 0.3), 0.8)

    def _suggest_cash_weight(self, risk_score: float) -> float:
        """권장 현금 비중"""
        if risk_score > 70:
            return 0.25
        elif risk_score > 50:
            return 0.15
        else:
            return 0.1


# ============================================================================
# Utility Functions
# ============================================================================

def create_market_state_from_data(
    indicators: Dict[str, float],
    volatility: float,
    trend: str
) -> MarketState:
    """데이터에서 MarketState 생성"""
    # 레짐 결정
    gdp_growth = indicators.get('gdp_growth', 0)
    unemployment = indicators.get('unemployment', 5)

    if gdp_growth > 2 and unemployment < 5:
        regime = "expansion_early"
    elif gdp_growth > 0:
        regime = "expansion_late"
    elif gdp_growth < 0:
        regime = "contraction"
    else:
        regime = "transition"

    # 리스크 수준 결정
    vix = indicators.get('vix', 20)
    if vix > 35:
        risk_level = RiskLevel.VERY_HIGH
    elif vix > 25:
        risk_level = RiskLevel.HIGH
    elif vix > 18:
        risk_level = RiskLevel.MODERATE
    else:
        risk_level = RiskLevel.LOW

    return MarketState(
        regime=regime,
        risk_level=risk_level,
        volatility_percentile=volatility,
        trend_direction=trend,
        key_indicators=indicators
    )


def create_critical_path_state(
    drivers: List[str],
    signals: Dict[str, str],
    propagation_days: int = 21,
    confidence: float = 0.7
) -> CriticalPathState:
    """CriticalPathState 생성 헬퍼"""
    return CriticalPathState(
        primary_drivers=drivers,
        current_signals=signals,
        propagation_time=propagation_days,
        confidence=confidence
    )


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("Strategy Agent Demo")
        print("=" * 60)

        # 1. 샘플 시장 상태 생성
        market_state = create_market_state_from_data(
            indicators={
                'gdp_growth': 2.5,
                'unemployment': 4.2,
                'inflation': 3.5,
                'vix': 18,
                'rate_trend': 'stable'
            },
            volatility=45,
            trend='bullish'
        )
        print(f"\n[Market State]")
        print(f"  Regime: {market_state.regime}")
        print(f"  Risk Level: {market_state.risk_level.value}")
        print(f"  Trend: {market_state.trend_direction}")

        # 2. Critical Path 상태
        critical_path = create_critical_path_state(
            drivers=['fed_policy', 'employment', 'credit_spread'],
            signals={
                'fed_policy': 'neutral',
                'employment': 'positive',
                'credit_spread': 'stable'
            },
            propagation_days=14,
            confidence=0.75
        )
        print(f"\n[Critical Path]")
        print(f"  Primary Drivers: {critical_path.primary_drivers}")
        print(f"  Signals: {critical_path.current_signals}")

        # 3. 전략 수립
        agent = StrategyAgent()
        strategy = await agent._execute({
            'market_state': market_state,
            'critical_path': critical_path,
            'risk_tolerance': 'moderate'
        })

        print(f"\n[Strategy Output]")
        print(f"  Overall Stance: {strategy.overall_stance}")
        print(f"  Cash Allocation: {strategy.cash_allocation:.0%}")

        print(f"\n[Recommendations]")
        for rec in strategy.recommendations[:5]:
            print(f"  - {rec.asset}: {rec.action.value} "
                  f"(confidence: {rec.confidence:.0%})")

        print(f"\n[Risk Warnings]")
        for warning in strategy.risk_warnings:
            print(f"  - {warning}")

        print(f"\n[Events to Watch]")
        for event in strategy.key_events_to_watch:
            print(f"  - {event}")

        print(f"\n[Summary]")
        print(f"  {strategy.summary}")

        print("\n" + "=" * 60)

    asyncio.run(demo())
