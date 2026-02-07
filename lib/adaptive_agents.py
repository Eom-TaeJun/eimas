"""
Adaptive Risk Agents - 시장 상황 적응형 투자 에이전트

핵심 변경:
1. 고정된 배분이 아닌 실시간 시장 상황 기반 동적 배분
2. 각 에이전트가 독자적으로 리스크 수준 조정
3. 단기 목표 집중 - 빠른 전략 전환
4. 실시간 시그널(VPIN, 레짐, 유동성)에 반응
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    yf = None
    pd = None


# ============================================================================
# 시장 상태 분석
# ============================================================================

@dataclass
class MarketCondition:
    """실시간 시장 상태"""
    timestamp: str

    # 레짐
    regime: str                    # Bull/Bear/Neutral
    regime_confidence: float       # 0-100
    trend: str                     # Uptrend/Downtrend/Sideways
    volatility: str                # Low/Medium/High

    # 리스크 지표
    risk_score: float              # 0-100
    vix_level: float               # VIX 수치
    liquidity_signal: str          # RISK_ON/RISK_OFF/NEUTRAL

    # 실시간 시그널
    vpin_alert: bool = False       # VPIN 경고
    vpin_level: float = 0.0        # VPIN 값
    spread_widening: bool = False  # 스프레드 확대

    # 거시
    fed_liquidity_trend: str = "stable"  # expanding/contracting/stable
    yield_curve: str = "normal"          # normal/inverted/flat

    # 버블 리스크 (v2.1.3)
    bubble_status: str = "NONE"          # NONE/WATCH/WARNING/DANGER

    def urgency_score(self) -> float:
        """긴급성 점수 (0-100) - 높을수록 빠른 대응 필요"""
        score = 0

        # 변동성
        if self.volatility == "High":
            score += 30
        elif self.volatility == "Medium":
            score += 15

        # VPIN 경고
        if self.vpin_alert:
            score += 25
        elif self.vpin_level > 0.5:
            score += 15

        # 리스크
        if self.risk_score > 60:
            score += 20
        elif self.risk_score > 40:
            score += 10

        # 스프레드
        if self.spread_widening:
            score += 15

        return min(score, 100)

    def opportunity_score(self) -> float:
        """기회 점수 (0-100) - 높을수록 진입 기회"""
        score = 50  # 기본

        # 강세장
        if "Bull" in self.regime:
            score += 25
        elif "Bear" in self.regime:
            score -= 25

        # 저위험
        if self.risk_score < 20:
            score += 20
        elif self.risk_score > 50:
            score -= 15

        # 유동성
        if self.liquidity_signal == "RISK_ON":
            score += 15
        elif self.liquidity_signal == "RISK_OFF":
            score -= 15

        # 저변동성
        if self.volatility == "Low":
            score += 10
        elif self.volatility == "High":
            score -= 10

        return max(0, min(score, 100))


# ============================================================================
# 자산 정의 (동적 리스크 점수)
# ============================================================================

class AssetClass(Enum):
    EQUITY = "equity"
    BOND = "bond"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    CASH = "cash"


@dataclass
class Asset:
    """자산 정보 (동적 속성 포함)"""
    ticker: str
    name: str
    asset_class: AssetClass
    base_risk: int              # 기본 리스크 (0-100)

    # 동적 속성 (시장 상황에 따라 조정)
    current_risk: int = 0       # 현재 리스크 (조정 후)
    momentum_score: float = 0   # 모멘텀 (-100 ~ +100)
    liquidity_score: float = 100  # 유동성 (0-100)
    correlation_to_spy: float = 1.0  # SPY 상관관계

    def adjusted_risk(self, market: MarketCondition) -> int:
        """시장 상황 반영 조정 리스크"""
        risk = self.base_risk

        # 고변동성 환경에서 리스크 증가
        if market.volatility == "High":
            risk = int(risk * 1.3)
        elif market.volatility == "Low":
            risk = int(risk * 0.85)

        # VPIN 경고 시 크립토 리스크 급등
        if market.vpin_alert and self.asset_class == AssetClass.CRYPTO:
            risk = int(risk * 1.5)

        # 약세장에서 주식 리스크 증가
        if "Bear" in market.regime and self.asset_class == AssetClass.EQUITY:
            risk = int(risk * 1.2)

        return min(risk, 100)


# 자산 유니버스 (동적)
def create_asset_universe() -> Dict[str, Asset]:
    """자산 유니버스 생성"""
    return {
        # === 주식 - 미국 ===
        "SPY": Asset("SPY", "S&P 500 ETF", AssetClass.EQUITY, 45),
        "QQQ": Asset("QQQ", "Nasdaq 100 ETF", AssetClass.EQUITY, 55),
        "IWM": Asset("IWM", "Russell 2000 ETF", AssetClass.EQUITY, 65),
        "DIA": Asset("DIA", "Dow Jones ETF", AssetClass.EQUITY, 40),

        # === 주식 - 레버리지 ===
        "TQQQ": Asset("TQQQ", "3x Nasdaq ETF", AssetClass.EQUITY, 90),
        "SOXL": Asset("SOXL", "3x Semiconductor ETF", AssetClass.EQUITY, 95),
        "UPRO": Asset("UPRO", "3x S&P 500 ETF", AssetClass.EQUITY, 88),

        # === 주식 - 인버스 ===
        "SQQQ": Asset("SQQQ", "3x Inverse Nasdaq", AssetClass.EQUITY, 92),
        "SH": Asset("SH", "Inverse S&P 500", AssetClass.EQUITY, 60),
        "PSQ": Asset("PSQ", "Inverse Nasdaq", AssetClass.EQUITY, 62),

        # === 주식 - 섹터 ===
        "XLF": Asset("XLF", "Financial Sector", AssetClass.EQUITY, 50),
        "XLE": Asset("XLE", "Energy Sector", AssetClass.EQUITY, 60),
        "XLV": Asset("XLV", "Healthcare Sector", AssetClass.EQUITY, 35),
        "XLK": Asset("XLK", "Technology Sector", AssetClass.EQUITY, 55),
        "XLU": Asset("XLU", "Utilities Sector", AssetClass.EQUITY, 30),

        # === 주식 - 중동 ===
        "KSA": Asset("KSA", "Saudi Arabia ETF", AssetClass.EQUITY, 60),
        "UAE": Asset("UAE", "UAE ETF", AssetClass.EQUITY, 55),
        "TUR": Asset("TUR", "Turkey ETF", AssetClass.EQUITY, 70),

        # === 채권 ===
        "TLT": Asset("TLT", "20+ Year Treasury", AssetClass.BOND, 35),
        "IEF": Asset("IEF", "7-10 Year Treasury", AssetClass.BOND, 25),
        "SHY": Asset("SHY", "1-3 Year Treasury", AssetClass.BOND, 10),
        "HYG": Asset("HYG", "High Yield Corporate", AssetClass.BOND, 45),
        "LQD": Asset("LQD", "Investment Grade Corp", AssetClass.BOND, 28),
        "TIP": Asset("TIP", "TIPS (Inflation)", AssetClass.BOND, 20),
        "EMB": Asset("EMB", "Emerging Market Bond", AssetClass.BOND, 50),

        # === 암호화폐 ===
        "BTC-USD": Asset("BTC-USD", "Bitcoin", AssetClass.CRYPTO, 80),
        "ETH-USD": Asset("ETH-USD", "Ethereum", AssetClass.CRYPTO, 85),
        "SOL-USD": Asset("SOL-USD", "Solana", AssetClass.CRYPTO, 90),
        "ONDO-USD": Asset("ONDO-USD", "Ondo (RWA)", AssetClass.CRYPTO, 75),

        # === 원자재 ===
        "GLD": Asset("GLD", "Gold", AssetClass.COMMODITY, 25),
        "SLV": Asset("SLV", "Silver", AssetClass.COMMODITY, 45),
        "USO": Asset("USO", "Oil", AssetClass.COMMODITY, 65),
        "UNG": Asset("UNG", "Natural Gas", AssetClass.COMMODITY, 75),

        # === 현금 ===
        "BIL": Asset("BIL", "1-3 Month T-Bill", AssetClass.CASH, 3),
        "SHV": Asset("SHV", "Short Treasury", AssetClass.CASH, 5),
    }


# ============================================================================
# 적응형 에이전트 베이스
# ============================================================================

@dataclass
class TradeOrder:
    """거래 주문"""
    order_id: str
    timestamp: str
    agent_id: str
    ticker: str
    action: str          # BUY / SELL / SHORT / COVER
    quantity: float
    target_price: float
    order_type: str      # MARKET / LIMIT / STOP
    rationale: str
    urgency: str         # IMMEDIATE / NORMAL / LOW

    # 실행 정보
    status: str = "PENDING"  # PENDING / FILLED / CANCELLED
    fill_price: float = 0.0
    fill_time: str = ""
    slippage: float = 0.0


@dataclass
class AgentState:
    """에이전트 상태"""
    agent_id: str
    agent_type: str

    # 자본
    initial_capital: float
    current_cash: float

    # 포지션
    positions: Dict[str, Dict] = field(default_factory=dict)
    # {ticker: {quantity, entry_price, entry_time, unrealized_pnl}}

    # 성과
    realized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0

    # 현재 전략 상태
    current_risk_level: float = 50.0  # 0-100 동적 조정
    current_strategy: str = "neutral"
    last_rebalance: str = ""

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        """포트폴리오 총 가치"""
        position_value = sum(
            pos['quantity'] * prices.get(ticker, pos['entry_price'])
            for ticker, pos in self.positions.items()
        )
        return self.current_cash + position_value

    def exposure_by_class(self) -> Dict[str, float]:
        """자산 클래스별 노출도"""
        universe = create_asset_universe()
        exposure = {ac.value: 0.0 for ac in AssetClass}

        for ticker, pos in self.positions.items():
            if ticker in universe:
                asset = universe[ticker]
                exposure[asset.asset_class.value] += pos['quantity'] * pos['entry_price']

        return exposure


class AdaptiveAgent(ABC):
    """적응형 에이전트 베이스 클래스"""

    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        initial_capital: float = 100000.0,
        base_risk_tolerance: float = 50.0  # 기본 리스크 허용도
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.base_risk_tolerance = base_risk_tolerance
        self.universe = create_asset_universe()

        self.state = AgentState(
            agent_id=agent_id,
            agent_type=agent_type,
            initial_capital=initial_capital,
            current_cash=initial_capital,
            current_risk_level=base_risk_tolerance
        )

        self.order_history: List[TradeOrder] = []
        self.decision_log: List[Dict] = []

    @abstractmethod
    def personality_modifier(self, market: MarketCondition) -> Dict:
        """
        에이전트 성격에 따른 수정자 반환

        Returns:
            {
                'risk_adjustment': float (-30 ~ +30),
                'urgency_threshold': float (0-100),
                'opportunity_threshold': float (0-100),
                'preferred_classes': List[AssetClass],
                'avoid_classes': List[AssetClass]
            }
        """
        pass

    def calculate_dynamic_risk_level(self, market: MarketCondition) -> float:
        """시장 상황 기반 동적 리스크 레벨 계산"""
        base = self.base_risk_tolerance
        mods = self.personality_modifier(market)

        # 시장 상황 조정
        adjustment = 0

        # 기회 점수 높으면 리스크 증가 허용
        opp_score = market.opportunity_score()
        if opp_score > 70:
            adjustment += 15
        elif opp_score < 30:
            adjustment -= 20

        # 긴급성 높으면 리스크 감소
        urg_score = market.urgency_score()
        if urg_score > 60:
            adjustment -= 15
        elif urg_score > 40:
            adjustment -= 5

        # VIX 레벨
        if market.vix_level > 30:
            adjustment -= 20
        elif market.vix_level > 20:
            adjustment -= 10
        elif market.vix_level < 15:
            adjustment += 10

        # 에이전트 성격 반영
        adjustment += mods.get('risk_adjustment', 0)

        new_level = base + adjustment
        self.state.current_risk_level = max(10, min(90, new_level))

        return self.state.current_risk_level

    def select_assets(self, market: MarketCondition) -> Dict[str, float]:
        """
        시장 상황과 리스크 레벨에 맞는 자산 선택

        Returns:
            {ticker: weight} 딕셔너리
        """
        risk_level = self.calculate_dynamic_risk_level(market)
        mods = self.personality_modifier(market)

        preferred = mods.get('preferred_classes', [])
        avoid = mods.get('avoid_classes', [])

        candidates = []

        for ticker, asset in self.universe.items():
            # 피해야 할 자산 클래스 제외
            if asset.asset_class in avoid:
                continue

            # 조정된 리스크 계산
            adj_risk = asset.adjusted_risk(market)

            # 현재 리스크 레벨에 맞는 자산만 선택
            risk_tolerance_range = 25
            if abs(adj_risk - risk_level) > risk_tolerance_range:
                # 리스크 허용 범위 밖
                if adj_risk > risk_level + risk_tolerance_range:
                    continue

            # 선호 클래스 가산점
            score = 100 - abs(adj_risk - risk_level)
            if asset.asset_class in preferred:
                score += 20

            candidates.append((ticker, asset, score, adj_risk))

        # 점수 기준 정렬
        candidates.sort(key=lambda x: -x[2])

        # 상위 자산 선택 및 비중 배분
        allocations = {}
        selected = candidates[:8]  # 최대 8개

        if not selected:
            # 안전 자산으로 폴백
            return {"BIL": 0.5, "SHY": 0.3, "GLD": 0.2}

        total_score = sum(s[2] for s in selected)

        for ticker, asset, score, adj_risk in selected:
            weight = score / total_score
            # 단일 자산 최대 비중 제한
            weight = min(weight, 0.30)
            allocations[ticker] = weight

        # 정규화
        total = sum(allocations.values())
        if total > 0:
            allocations = {k: v/total for k, v in allocations.items()}

        return allocations

    def analyze_and_decide(
        self,
        market: MarketCondition,
        current_prices: Dict[str, float]
    ) -> Tuple[str, List[TradeOrder], str]:
        """
        시장 분석 및 거래 결정

        Returns:
            (action, orders, rationale)
        """
        mods = self.personality_modifier(market)
        urg_threshold = mods.get('urgency_threshold', 50)
        opp_threshold = mods.get('opportunity_threshold', 50)

        urgency = market.urgency_score()
        opportunity = market.opportunity_score()
        risk_level = self.calculate_dynamic_risk_level(market)

        # 결정 로직
        action = "HOLD"
        orders = []
        rationale_parts = []

        rationale_parts.append(f"Risk Level: {risk_level:.0f}")
        rationale_parts.append(f"Urgency: {urgency:.0f}")
        rationale_parts.append(f"Opportunity: {opportunity:.0f}")

        # === 긴급 탈출 ===
        if urgency > urg_threshold + 20:
            action = "EMERGENCY_EXIT"
            rationale_parts.append("긴급 탈출 - 리스크 급등")

            # 모든 위험 포지션 청산
            for ticker, pos in self.state.positions.items():
                asset = self.universe.get(ticker)
                if asset and asset.base_risk > 40:
                    orders.append(self._create_order(
                        ticker, "SELL", pos['quantity'],
                        current_prices.get(ticker, pos['entry_price']),
                        "MARKET", "긴급 청산", "IMMEDIATE"
                    ))

        # === 방어 모드 ===
        elif urgency > urg_threshold:
            action = "DEFENSIVE"
            rationale_parts.append("방어 모드 전환")

            # 고위험 자산 축소, 안전 자산 확대
            target = self._defensive_allocation(market)
            orders = self._generate_rebalance_orders(target, current_prices)

        # === 공격 기회 ===
        elif opportunity > opp_threshold + 20:
            action = "AGGRESSIVE_ENTRY"
            rationale_parts.append("적극 진입 기회")

            target = self.select_assets(market)
            orders = self._generate_rebalance_orders(target, current_prices)

        # === 기회 포착 ===
        elif opportunity > opp_threshold:
            action = "ENTRY"
            rationale_parts.append("진입 기회")

            target = self.select_assets(market)
            orders = self._generate_rebalance_orders(target, current_prices)

        # === 유지 ===
        else:
            action = "HOLD"
            rationale_parts.append("현 포지션 유지")

            # 경미한 리밸런싱 검토
            target = self.select_assets(market)
            orders = self._generate_rebalance_orders(target, current_prices, threshold=0.05)

        # 전략 상태 업데이트
        self.state.current_strategy = action
        self.state.last_rebalance = market.timestamp

        # 로그 저장
        self.decision_log.append({
            'timestamp': market.timestamp,
            'action': action,
            'risk_level': risk_level,
            'urgency': urgency,
            'opportunity': opportunity,
            'orders_count': len(orders),
            'rationale': " | ".join(rationale_parts)
        })

        return action, orders, " | ".join(rationale_parts)

    def _defensive_allocation(self, market: MarketCondition) -> Dict[str, float]:
        """방어적 배분"""
        # 기본 방어 포트폴리오
        base = {
            "BIL": 0.25,
            "SHY": 0.20,
            "TIP": 0.15,
            "GLD": 0.20,
            "XLU": 0.10,
            "XLV": 0.10
        }

        # 인버스 추가 여부 (에이전트별 다름)
        mods = self.personality_modifier(market)
        if mods.get('use_inverse', False) and "Bear" in market.regime:
            base["SH"] = 0.15
            base["BIL"] -= 0.10
            base["GLD"] -= 0.05

        return base

    def _generate_rebalance_orders(
        self,
        target: Dict[str, float],
        prices: Dict[str, float],
        threshold: float = 0.02
    ) -> List[TradeOrder]:
        """리밸런싱 주문 생성"""
        orders = []
        portfolio_value = self.state.portfolio_value(prices)

        for ticker, target_weight in target.items():
            price = prices.get(ticker, 0)
            if price <= 0:
                continue

            target_value = portfolio_value * target_weight
            current_pos = self.state.positions.get(ticker, {})
            current_value = current_pos.get('quantity', 0) * price

            diff = target_value - current_value
            diff_pct = abs(diff) / portfolio_value if portfolio_value > 0 else 0

            # 임계값 이상만 거래
            if diff_pct < threshold:
                continue

            if diff > 100:  # $100 이상 매수
                qty = diff / price
                orders.append(self._create_order(
                    ticker, "BUY", qty, price, "MARKET",
                    f"Target {target_weight:.1%}", "NORMAL"
                ))
            elif diff < -100:  # $100 이상 매도
                qty = abs(diff) / price
                orders.append(self._create_order(
                    ticker, "SELL", qty, price, "MARKET",
                    f"Target {target_weight:.1%}", "NORMAL"
                ))

        return orders

    def _create_order(
        self,
        ticker: str,
        action: str,
        quantity: float,
        price: float,
        order_type: str,
        rationale: str,
        urgency: str
    ) -> TradeOrder:
        """주문 생성"""
        order_id = f"{self.agent_id}_{ticker}_{datetime.now().strftime('%H%M%S%f')}"

        return TradeOrder(
            order_id=order_id,
            timestamp=datetime.now().isoformat(),
            agent_id=self.agent_id,
            ticker=ticker,
            action=action,
            quantity=round(quantity, 4),
            target_price=price,
            order_type=order_type,
            rationale=rationale,
            urgency=urgency
        )

    def execute_orders(self, orders: List[TradeOrder], prices: Dict[str, float]) -> List[TradeOrder]:
        """주문 실행 (시뮬레이션)"""
        executed = []

        for order in orders:
            # 실행 가격 (슬리피지 포함)
            base_price = prices.get(order.ticker, order.target_price)
            slippage = np.random.uniform(0.0005, 0.002)  # 0.05% ~ 0.2%

            if order.action == "BUY":
                fill_price = base_price * (1 + slippage)
            else:
                fill_price = base_price * (1 - slippage)

            order.status = "FILLED"
            order.fill_price = fill_price
            order.fill_time = datetime.now().isoformat()
            order.slippage = slippage

            # 포지션 업데이트
            self._update_position(order)

            executed.append(order)
            self.order_history.append(order)
            self.state.total_trades += 1

        return executed

    def _update_position(self, order: TradeOrder):
        """포지션 업데이트"""
        ticker = order.ticker

        if order.action == "BUY":
            if ticker in self.state.positions:
                pos = self.state.positions[ticker]
                total_qty = pos['quantity'] + order.quantity
                avg_price = (pos['quantity'] * pos['entry_price'] +
                            order.quantity * order.fill_price) / total_qty
                pos['quantity'] = total_qty
                pos['entry_price'] = avg_price
            else:
                self.state.positions[ticker] = {
                    'quantity': order.quantity,
                    'entry_price': order.fill_price,
                    'entry_time': order.fill_time,
                    'unrealized_pnl': 0
                }
            self.state.current_cash -= order.quantity * order.fill_price

        elif order.action == "SELL":
            if ticker in self.state.positions:
                pos = self.state.positions[ticker]

                # 실현 손익
                realized = (order.fill_price - pos['entry_price']) * order.quantity
                self.state.realized_pnl += realized
                if realized > 0:
                    self.state.winning_trades += 1

                pos['quantity'] -= order.quantity
                if pos['quantity'] < 0.0001:
                    del self.state.positions[ticker]

            self.state.current_cash += order.quantity * order.fill_price


# ============================================================================
# 구체적 에이전트 구현
# ============================================================================

class AggressiveAdaptiveAgent(AdaptiveAgent):
    """
    공격적 적응형 에이전트

    특징:
    - 기회에 빠르게 반응
    - 레버리지 활용 가능
    - 높은 리스크 허용
    - 하지만 시장 악화 시 빠르게 방어 전환
    """

    def __init__(self, agent_id: str = "aggressive_01", initial_capital: float = 100000.0):
        super().__init__(
            agent_id=agent_id,
            agent_type="aggressive",
            initial_capital=initial_capital,
            base_risk_tolerance=70.0
        )

    def personality_modifier(self, market: MarketCondition) -> Dict:
        mods = {
            'risk_adjustment': 15,  # 기본적으로 리스크 추가
            'urgency_threshold': 60,  # 높은 긴급성에서만 반응
            'opportunity_threshold': 40,  # 낮은 기회에도 진입
            'preferred_classes': [AssetClass.EQUITY, AssetClass.CRYPTO],
            'avoid_classes': [],
            'use_inverse': True
        }

        # 강세장에서 더 공격적
        if "Bull" in market.regime:
            mods['risk_adjustment'] = 25
            mods['opportunity_threshold'] = 30

        # 약세장에서도 인버스로 기회 포착
        elif "Bear" in market.regime:
            mods['risk_adjustment'] = 0
            mods['preferred_classes'] = [AssetClass.COMMODITY, AssetClass.BOND]

        return mods


class BalancedAdaptiveAgent(AdaptiveAgent):
    """
    균형 적응형 에이전트

    특징:
    - 시장 상황에 비례하여 대응
    - 다양한 자산 클래스 활용
    - 중간 리스크 허용
    - 안정적 성과 추구
    """

    def __init__(self, agent_id: str = "balanced_01", initial_capital: float = 100000.0):
        super().__init__(
            agent_id=agent_id,
            agent_type="balanced",
            initial_capital=initial_capital,
            base_risk_tolerance=50.0
        )

    def personality_modifier(self, market: MarketCondition) -> Dict:
        mods = {
            'risk_adjustment': 0,
            'urgency_threshold': 50,
            'opportunity_threshold': 50,
            'preferred_classes': [AssetClass.EQUITY, AssetClass.BOND, AssetClass.COMMODITY],
            'avoid_classes': [],
            'use_inverse': False
        }

        # 시장 상황에 따라 조정
        if "Bull" in market.regime and market.risk_score < 30:
            mods['risk_adjustment'] = 10
            mods['opportunity_threshold'] = 40
        elif "Bear" in market.regime:
            mods['risk_adjustment'] = -10
            mods['preferred_classes'] = [AssetClass.BOND, AssetClass.COMMODITY, AssetClass.CASH]
        elif market.volatility == "High":
            mods['risk_adjustment'] = -15
            mods['urgency_threshold'] = 40

        return mods


class ConservativeAdaptiveAgent(AdaptiveAgent):
    """
    보수적 적응형 에이전트

    특징:
    - 자본 보존 최우선
    - 위험 신호에 민감하게 반응
    - 낮은 리스크 자산 선호
    - 천천히 기회 포착
    """

    def __init__(self, agent_id: str = "conservative_01", initial_capital: float = 100000.0):
        super().__init__(
            agent_id=agent_id,
            agent_type="conservative",
            initial_capital=initial_capital,
            base_risk_tolerance=30.0
        )

    def personality_modifier(self, market: MarketCondition) -> Dict:
        mods = {
            'risk_adjustment': -10,
            'urgency_threshold': 35,  # 낮은 긴급성에도 반응
            'opportunity_threshold': 65,  # 높은 기회에서만 진입
            'preferred_classes': [AssetClass.BOND, AssetClass.CASH, AssetClass.COMMODITY],
            'avoid_classes': [AssetClass.CRYPTO],
            'use_inverse': False
        }

        # 리스크 점수가 높으면 더 보수적
        if market.risk_score > 40:
            mods['risk_adjustment'] = -20
            mods['avoid_classes'].append(AssetClass.EQUITY)

        # 매우 좋은 환경에서만 주식 허용
        if "Bull" in market.regime and market.risk_score < 20 and market.volatility == "Low":
            mods['risk_adjustment'] = 5
            mods['avoid_classes'] = [AssetClass.CRYPTO]

        return mods


# ============================================================================
# 에이전트 매니저
# ============================================================================

class AdaptiveAgentManager:
    """적응형 에이전트 통합 관리"""

    def __init__(self, initial_capital: float = 100000.0):
        self.agents = {
            'aggressive': AggressiveAdaptiveAgent(initial_capital=initial_capital),
            'balanced': BalancedAdaptiveAgent(initial_capital=initial_capital),
            'conservative': ConservativeAdaptiveAgent(initial_capital=initial_capital)
        }

        self.db_path = Path(__file__).parent.parent / "data" / "adaptive_trades.db"
        self._init_db()

    def _init_db(self):
        """DB 초기화"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT UNIQUE,
            timestamp TEXT,
            agent_id TEXT,
            ticker TEXT,
            action TEXT,
            quantity REAL,
            target_price REAL,
            order_type TEXT,
            rationale TEXT,
            urgency TEXT,
            status TEXT,
            fill_price REAL,
            fill_time TEXT,
            slippage REAL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            agent_id TEXT,
            agent_type TEXT,
            action TEXT,
            risk_level REAL,
            urgency REAL,
            opportunity REAL,
            orders_count INTEGER,
            rationale TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            agent_id TEXT,
            portfolio_value REAL,
            cash REAL,
            positions_json TEXT,
            realized_pnl REAL,
            unrealized_pnl REAL,
            current_risk_level REAL,
            current_strategy TEXT
        )
        """)

        conn.commit()
        conn.close()

    def run_all(
        self,
        market: MarketCondition,
        prices: Dict[str, float],
        persist: bool = True,
    ) -> Dict[str, Dict]:
        """모든 에이전트 실행"""
        results = {}
        conn = None
        cursor = None
        if persist:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

        try:
            for name, agent in self.agents.items():
                action, orders, rationale = agent.analyze_and_decide(market, prices)

                # 주문 실행
                executed = agent.execute_orders(orders, prices)

                # DB 저장 (batched transaction when persist=True)
                if persist:
                    self._save_decision(agent, action, market, len(orders), rationale, cursor=cursor)
                    for order in executed:
                        self._save_order(order, cursor=cursor)
                    self._save_snapshot(agent, prices, cursor=cursor)

                results[name] = {
                    'action': action,
                    'orders': len(executed),
                    'risk_level': agent.state.current_risk_level,
                    'portfolio_value': agent.state.portfolio_value(prices),
                    'rationale': rationale
                }
            if conn is not None:
                conn.commit()
        finally:
            if conn is not None:
                conn.close()

        return results

    def _save_order(self, order: TradeOrder, cursor=None):
        own_conn = None
        if cursor is None:
            own_conn = sqlite3.connect(str(self.db_path))
            cursor = own_conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO orders VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            order.order_id, order.timestamp, order.agent_id, order.ticker,
            order.action, order.quantity, order.target_price, order.order_type,
            order.rationale, order.urgency, order.status, order.fill_price,
            order.fill_time, order.slippage
        ))
        if own_conn is not None:
            own_conn.commit()
            own_conn.close()

    def _save_decision(self, agent, action, market, orders_count, rationale, cursor=None):
        own_conn = None
        if cursor is None:
            own_conn = sqlite3.connect(str(self.db_path))
            cursor = own_conn.cursor()
        cursor.execute("""
        INSERT INTO decisions VALUES (NULL,?,?,?,?,?,?,?,?,?)
        """, (
            market.timestamp, agent.agent_id, agent.agent_type, action,
            agent.state.current_risk_level, market.urgency_score(),
            market.opportunity_score(), orders_count, rationale
        ))
        if own_conn is not None:
            own_conn.commit()
            own_conn.close()

    def _save_snapshot(self, agent, prices, cursor=None):
        own_conn = None
        if cursor is None:
            own_conn = sqlite3.connect(str(self.db_path))
            cursor = own_conn.cursor()

        # 미실현 손익 계산
        unrealized = sum(
            (prices.get(t, p['entry_price']) - p['entry_price']) * p['quantity']
            for t, p in agent.state.positions.items()
        )

        cursor.execute("""
        INSERT INTO snapshots VALUES (NULL,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now().isoformat(),
            agent.agent_id,
            agent.state.portfolio_value(prices),
            agent.state.current_cash,
            json.dumps(agent.state.positions),
            agent.state.realized_pnl,
            unrealized,
            agent.state.current_risk_level,
            agent.state.current_strategy
        ))
        if own_conn is not None:
            own_conn.commit()
            own_conn.close()

    def get_comparison(self, prices: Dict[str, float]) -> str:
        """에이전트 비교"""
        lines = ["=" * 70]
        lines.append("ADAPTIVE AGENTS - REAL-TIME COMPARISON")
        lines.append("=" * 70)

        for name, agent in self.agents.items():
            pv = agent.state.portfolio_value(prices)
            ret = (pv / agent.state.initial_capital - 1) * 100

            lines.append(f"\n[{name.upper()}]")
            lines.append(f"  Strategy: {agent.state.current_strategy}")
            lines.append(f"  Risk Level: {agent.state.current_risk_level:.0f}/100")
            lines.append(f"  Portfolio: ${pv:,.2f} ({ret:+.2f}%)")
            lines.append(f"  Positions: {len(agent.state.positions)}")
            lines.append(f"  Win Rate: {agent.state.winning_trades}/{agent.state.total_trades}")

            if agent.state.positions:
                lines.append("  Holdings:")
                for ticker, pos in list(agent.state.positions.items())[:3]:
                    price = prices.get(ticker, pos['entry_price'])
                    pnl_pct = (price / pos['entry_price'] - 1) * 100
                    lines.append(f"    - {ticker}: {pnl_pct:+.2f}%")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    print("Adaptive Agents Test")
    print("=" * 60)

    # 시장 상태 시뮬레이션
    market = MarketCondition(
        timestamp=datetime.now().isoformat(),
        regime="Bull (Low Vol)",
        regime_confidence=75,
        trend="Uptrend",
        volatility="Low",
        risk_score=15,
        vix_level=14.5,
        liquidity_signal="RISK_ON",
        fed_liquidity_trend="expanding"
    )

    print(f"\n[Market Condition]")
    print(f"  Regime: {market.regime}")
    print(f"  Risk Score: {market.risk_score}")
    print(f"  Urgency: {market.urgency_score():.0f}")
    print(f"  Opportunity: {market.opportunity_score():.0f}")

    # 가격 데이터
    prices = {
        "SPY": 595.0, "QQQ": 520.0, "TLT": 88.0, "GLD": 240.0,
        "BTC-USD": 95000.0, "BIL": 91.5, "TQQQ": 85.0, "SOXL": 35.0,
        "IEF": 95.0, "TIP": 108.0, "XLV": 145.0, "SHY": 82.0
    }

    # 에이전트 매니저
    manager = AdaptiveAgentManager(initial_capital=100000)

    print("\n[Running Agents...]")
    results = manager.run_all(market, prices)

    for name, result in results.items():
        print(f"\n  [{name}]")
        print(f"    Action: {result['action']}")
        print(f"    Risk Level: {result['risk_level']:.0f}")
        print(f"    Orders: {result['orders']}")

    # 비교 리포트
    print("\n" + manager.get_comparison(prices))
