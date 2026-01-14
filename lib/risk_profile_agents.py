"""
Risk Profile Agents - 리스크 성향별 투자 에이전트

3가지 리스크 프로파일:
- RiskLover: 고위험 고수익 추구, 레버리지/변동성 자산 선호
- RiskNeutral: 균형 잡힌 접근, 시장 중립적 판단
- RiskAverse: 자본 보존 우선, 방어적 자산 선호

각 에이전트는:
1. 시장 진입/탈출 결정
2. 자산 선택 및 비중 결정
3. 결정 근거 제공
4. 가상 거래 실행 및 손익 추적
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import yfinance as yf
import pandas as pd
import numpy as np

# ============================================================================
# 데이터 구조
# ============================================================================

class RiskProfile(Enum):
    LOVER = "risk_lover"      # 고위험 고수익
    NEUTRAL = "risk_neutral"  # 균형
    AVERSE = "risk_averse"    # 보수적


class Action(Enum):
    ENTER = "ENTER"     # 시장 진입
    EXIT = "EXIT"       # 시장 탈출
    HOLD = "HOLD"       # 유지
    REBALANCE = "REBALANCE"  # 리밸런싱


class AssetClass(Enum):
    EQUITY = "equity"           # 주식
    BOND = "bond"               # 채권
    CRYPTO = "crypto"           # 암호화폐
    COMMODITY = "commodity"     # 원자재
    CASH = "cash"               # 현금


@dataclass
class Position:
    """포지션 정보"""
    ticker: str
    asset_class: AssetClass
    quantity: float
    entry_price: float
    entry_date: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class Trade:
    """거래 정보"""
    trade_id: str
    timestamp: str
    agent_type: str
    action: str          # BUY / SELL
    ticker: str
    asset_class: str
    quantity: float
    price: float
    total_value: float
    rationale: str
    executed: bool = False
    execution_price: float = 0.0
    execution_time: str = ""


@dataclass
class AgentDecision:
    """에이전트 결정"""
    timestamp: str
    agent_type: str
    action: Action
    confidence: float           # 0-100%
    rationale: str
    allocations: Dict[str, float]  # {ticker: weight}
    trades: List[Trade] = field(default_factory=list)
    market_view: str = ""       # 시장 전망
    risk_assessment: str = ""   # 리스크 평가


# ============================================================================
# 자산 유니버스
# ============================================================================

ASSET_UNIVERSE = {
    # 주식 - 미국
    "equity_us": {
        "SPY": {"name": "S&P 500 ETF", "risk": 50, "class": AssetClass.EQUITY},
        "QQQ": {"name": "Nasdaq 100 ETF", "risk": 60, "class": AssetClass.EQUITY},
        "IWM": {"name": "Russell 2000 ETF", "risk": 70, "class": AssetClass.EQUITY},
        "TQQQ": {"name": "3x Nasdaq ETF", "risk": 95, "class": AssetClass.EQUITY},
        "SOXL": {"name": "3x Semiconductor ETF", "risk": 98, "class": AssetClass.EQUITY},
    },
    # 주식 - 섹터
    "equity_sector": {
        "XLF": {"name": "Financial Sector", "risk": 55, "class": AssetClass.EQUITY},
        "XLE": {"name": "Energy Sector", "risk": 65, "class": AssetClass.EQUITY},
        "XLV": {"name": "Healthcare Sector", "risk": 40, "class": AssetClass.EQUITY},
        "XLK": {"name": "Technology Sector", "risk": 60, "class": AssetClass.EQUITY},
    },
    # 주식 - 중동 (신규)
    "equity_mena": {
        "KSA": {"name": "Saudi Arabia ETF", "risk": 65, "class": AssetClass.EQUITY},
        "UAE": {"name": "UAE ETF", "risk": 60, "class": AssetClass.EQUITY},
        "GULF": {"name": "Gulf States ETF", "risk": 55, "class": AssetClass.EQUITY},
        "TUR": {"name": "Turkey ETF", "risk": 75, "class": AssetClass.EQUITY},
    },
    # 채권
    "bond": {
        "TLT": {"name": "20+ Year Treasury", "risk": 35, "class": AssetClass.BOND},
        "IEF": {"name": "7-10 Year Treasury", "risk": 25, "class": AssetClass.BOND},
        "SHY": {"name": "1-3 Year Treasury", "risk": 10, "class": AssetClass.BOND},
        "HYG": {"name": "High Yield Corporate", "risk": 50, "class": AssetClass.BOND},
        "LQD": {"name": "Investment Grade Corp", "risk": 30, "class": AssetClass.BOND},
        "TIP": {"name": "TIPS (Inflation)", "risk": 20, "class": AssetClass.BOND},
        "EMB": {"name": "Emerging Market Bond", "risk": 55, "class": AssetClass.BOND},
    },
    # 암호화폐
    "crypto": {
        "BTC-USD": {"name": "Bitcoin", "risk": 85, "class": AssetClass.CRYPTO},
        "ETH-USD": {"name": "Ethereum", "risk": 88, "class": AssetClass.CRYPTO},
        "SOL-USD": {"name": "Solana", "risk": 92, "class": AssetClass.CRYPTO},
        "ONDO-USD": {"name": "Ondo (RWA)", "risk": 80, "class": AssetClass.CRYPTO},
    },
    # 원자재
    "commodity": {
        "GLD": {"name": "Gold", "risk": 30, "class": AssetClass.COMMODITY},
        "SLV": {"name": "Silver", "risk": 50, "class": AssetClass.COMMODITY},
        "USO": {"name": "Oil", "risk": 70, "class": AssetClass.COMMODITY},
        "DBA": {"name": "Agriculture", "risk": 55, "class": AssetClass.COMMODITY},
    },
    # 현금 등가물
    "cash": {
        "BIL": {"name": "1-3 Month T-Bill", "risk": 5, "class": AssetClass.CASH},
        "SHV": {"name": "Short Treasury", "risk": 5, "class": AssetClass.CASH},
    }
}


def get_all_tickers() -> Dict[str, Dict]:
    """전체 티커 리스트 반환"""
    all_tickers = {}
    for category, tickers in ASSET_UNIVERSE.items():
        all_tickers.update(tickers)
    return all_tickers


# ============================================================================
# 베이스 에이전트
# ============================================================================

class BaseRiskAgent(ABC):
    """리스크 프로파일 에이전트 기본 클래스"""

    def __init__(
        self,
        profile: RiskProfile,
        initial_capital: float = 100000.0,
        max_position_pct: float = 0.25
    ):
        self.profile = profile
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.decision_history: List[AgentDecision] = []

    @property
    @abstractmethod
    def risk_tolerance(self) -> Tuple[int, int]:
        """허용 리스크 범위 (min, max)"""
        pass

    @property
    @abstractmethod
    def preferred_assets(self) -> List[str]:
        """선호 자산 카테고리"""
        pass

    @abstractmethod
    def calculate_target_allocation(
        self,
        market_data: Dict,
        regime: Dict,
        risk_score: float
    ) -> Dict[str, float]:
        """목표 배분 계산"""
        pass

    def decide(
        self,
        market_data: Dict,
        regime: Dict,
        risk_score: float,
        signals: List[Dict]
    ) -> AgentDecision:
        """
        투자 결정 수행

        Args:
            market_data: 시장 데이터 (티커별 DataFrame)
            regime: 레짐 정보 {regime, trend, volatility, confidence}
            risk_score: 리스크 점수 (0-100)
            signals: 시그널 리스트

        Returns:
            AgentDecision: 결정 내용
        """
        timestamp = datetime.now().isoformat()

        # 1. 시장 상황 평가
        market_view = self._assess_market(regime, risk_score)

        # 2. 진입/탈출 결정
        action, confidence = self._decide_action(regime, risk_score, signals)

        # 3. 목표 배분 계산
        allocations = self.calculate_target_allocation(market_data, regime, risk_score)

        # 4. 거래 생성
        trades = self._generate_trades(allocations, market_data)

        # 5. 근거 작성
        rationale = self._generate_rationale(action, regime, risk_score, allocations)

        decision = AgentDecision(
            timestamp=timestamp,
            agent_type=self.profile.value,
            action=action,
            confidence=confidence,
            rationale=rationale,
            allocations=allocations,
            trades=trades,
            market_view=market_view,
            risk_assessment=f"Risk Score: {risk_score:.1f}/100"
        )

        self.decision_history.append(decision)
        return decision

    def _assess_market(self, regime: Dict, risk_score: float) -> str:
        """시장 상황 평가"""
        regime_str = regime.get('regime', 'Unknown')
        trend = regime.get('trend', 'Unknown')
        vol = regime.get('volatility', 'Unknown')

        if 'Bull' in regime_str and risk_score < 30:
            return "강세장, 저위험 환경 - 적극적 포지션 가능"
        elif 'Bull' in regime_str and risk_score >= 30:
            return "강세장이나 리스크 상승 - 선별적 진입"
        elif 'Bear' in regime_str:
            return "약세장 - 방어적 포지션 권장"
        else:
            return "혼조세 - 관망 또는 중립 유지"

    def _decide_action(
        self,
        regime: Dict,
        risk_score: float,
        signals: List[Dict]
    ) -> Tuple[Action, float]:
        """진입/탈출 결정"""
        regime_str = regime.get('regime', 'Unknown')
        confidence_base = regime.get('confidence', 50)

        # 기본 로직 (서브클래스에서 오버라이드)
        if 'Bull' in regime_str:
            return Action.ENTER, min(confidence_base + 10, 100)
        elif 'Bear' in regime_str:
            return Action.EXIT, min(confidence_base + 10, 100)
        else:
            return Action.HOLD, confidence_base

    def _generate_trades(
        self,
        allocations: Dict[str, float],
        market_data: Dict
    ) -> List[Trade]:
        """거래 생성"""
        trades = []
        all_tickers = get_all_tickers()

        for ticker, target_weight in allocations.items():
            if ticker not in all_tickers:
                continue

            asset_info = all_tickers[ticker]

            # 현재 가격 조회
            current_price = self._get_current_price(ticker, market_data)
            if current_price <= 0:
                continue

            # 목표 금액 계산
            target_value = self.current_capital * target_weight

            # 현재 포지션
            current_position = self.positions.get(ticker)
            current_value = 0
            if current_position:
                current_value = current_position.quantity * current_price

            # 차이 계산
            diff_value = target_value - current_value

            if abs(diff_value) < 100:  # $100 미만 무시
                continue

            # 거래 생성
            action = "BUY" if diff_value > 0 else "SELL"
            quantity = abs(diff_value) / current_price

            trade = Trade(
                trade_id=f"{self.profile.value}_{ticker}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.now().isoformat(),
                agent_type=self.profile.value,
                action=action,
                ticker=ticker,
                asset_class=asset_info['class'].value,
                quantity=round(quantity, 4),
                price=current_price,
                total_value=abs(diff_value),
                rationale=f"Target weight: {target_weight:.1%}, Current: {current_value/self.current_capital:.1%}"
            )
            trades.append(trade)

        return trades

    def _get_current_price(self, ticker: str, market_data: Dict) -> float:
        """현재 가격 조회"""
        if ticker in market_data:
            df = market_data[ticker]
            if hasattr(df, 'iloc') and len(df) > 0:
                return float(df['Close'].iloc[-1])

        # yfinance에서 직접 조회
        try:
            data = yf.download(ticker, period='1d', progress=False)
            if len(data) > 0:
                return float(data['Close'].iloc[-1])
        except:
            pass

        return 0.0

    def _generate_rationale(
        self,
        action: Action,
        regime: Dict,
        risk_score: float,
        allocations: Dict[str, float]
    ) -> str:
        """결정 근거 생성"""
        parts = []

        # 1. 시장 상황
        parts.append(f"[시장] {regime.get('regime', 'Unknown')}, Risk {risk_score:.1f}/100")

        # 2. 행동 이유
        if action == Action.ENTER:
            parts.append(f"[결정] 시장 진입 - {self.profile.value} 전략 실행")
        elif action == Action.EXIT:
            parts.append(f"[결정] 시장 탈출 - 리스크 회피")
        else:
            parts.append(f"[결정] 포지션 유지")

        # 3. 주요 배분
        top_allocs = sorted(allocations.items(), key=lambda x: -x[1])[:3]
        alloc_str = ", ".join([f"{t}:{w:.1%}" for t, w in top_allocs])
        parts.append(f"[배분] {alloc_str}")

        return " | ".join(parts)

    def execute_trades(self, trades: List[Trade]) -> List[Trade]:
        """거래 실행 (가상)"""
        executed = []
        for trade in trades:
            # 가격 재조회 (실행 가격)
            try:
                data = yf.download(trade.ticker, period='1d', progress=False)
                if len(data) > 0:
                    exec_price = float(data['Close'].iloc[-1])
                else:
                    exec_price = trade.price
            except:
                exec_price = trade.price

            trade.executed = True
            trade.execution_price = exec_price
            trade.execution_time = datetime.now().isoformat()

            # 포지션 업데이트
            self._update_position(trade)

            executed.append(trade)
            self.trade_history.append(trade)

        return executed

    def _update_position(self, trade: Trade):
        """포지션 업데이트"""
        ticker = trade.ticker
        all_tickers = get_all_tickers()

        if trade.action == "BUY":
            if ticker in self.positions:
                # 기존 포지션에 추가
                pos = self.positions[ticker]
                total_qty = pos.quantity + trade.quantity
                avg_price = (pos.quantity * pos.entry_price + trade.quantity * trade.execution_price) / total_qty
                pos.quantity = total_qty
                pos.entry_price = avg_price
            else:
                # 새 포지션
                self.positions[ticker] = Position(
                    ticker=ticker,
                    asset_class=AssetClass(trade.asset_class),
                    quantity=trade.quantity,
                    entry_price=trade.execution_price,
                    entry_date=trade.execution_time,
                    current_price=trade.execution_price
                )
            self.current_capital -= trade.total_value

        elif trade.action == "SELL":
            if ticker in self.positions:
                pos = self.positions[ticker]
                pos.quantity -= trade.quantity
                if pos.quantity <= 0.0001:
                    del self.positions[ticker]
                self.current_capital += trade.total_value

    def calculate_pnl(self, market_data: Dict = None) -> Dict:
        """손익 계산"""
        total_unrealized = 0
        total_realized = 0
        position_pnls = {}

        # 미실현 손익
        for ticker, pos in self.positions.items():
            current_price = self._get_current_price(ticker, market_data or {})
            if current_price > 0:
                pos.current_price = current_price
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
                pos.unrealized_pnl_pct = (current_price / pos.entry_price - 1) * 100
                total_unrealized += pos.unrealized_pnl
                position_pnls[ticker] = {
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': current_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'unrealized_pnl_pct': pos.unrealized_pnl_pct
                }

        # 실현 손익 (완료된 매도 거래)
        for trade in self.trade_history:
            if trade.action == "SELL" and trade.executed:
                # 단순화: 실현 손익은 별도 계산 필요
                pass

        # 포트폴리오 가치
        portfolio_value = self.current_capital + sum(
            pos.quantity * pos.current_price for pos in self.positions.values()
        )

        return {
            'agent_type': self.profile.value,
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'portfolio_value': portfolio_value,
            'total_return': (portfolio_value / self.initial_capital - 1) * 100,
            'unrealized_pnl': total_unrealized,
            'realized_pnl': total_realized,
            'position_count': len(self.positions),
            'positions': position_pnls
        }


# ============================================================================
# 리스크 프로파일별 에이전트
# ============================================================================

class RiskLoverAgent(BaseRiskAgent):
    """
    Risk Lover - 공격적 투자자

    특징:
    - 고위험 고수익 추구
    - 레버리지 ETF, 암호화폐 선호
    - 강세장에서 적극적 진입
    - 손실 감내 높음
    """

    def __init__(self, initial_capital: float = 100000.0):
        super().__init__(
            profile=RiskProfile.LOVER,
            initial_capital=initial_capital,
            max_position_pct=0.40  # 단일 포지션 최대 40%
        )

    @property
    def risk_tolerance(self) -> Tuple[int, int]:
        return (60, 100)  # 리스크 60-100 자산 선호

    @property
    def preferred_assets(self) -> List[str]:
        return ["equity_us", "crypto", "equity_sector"]

    def calculate_target_allocation(
        self,
        market_data: Dict,
        regime: Dict,
        risk_score: float
    ) -> Dict[str, float]:
        """공격적 배분"""
        allocations = {}
        regime_str = regime.get('regime', 'Unknown')

        if 'Bull' in regime_str and risk_score < 40:
            # 강세장 + 저위험: 최대 공격
            allocations = {
                "TQQQ": 0.25,      # 3x 나스닥
                "SOXL": 0.20,      # 3x 반도체
                "BTC-USD": 0.20,  # 비트코인
                "ETH-USD": 0.15,  # 이더리움
                "QQQ": 0.15,      # 나스닥
                "BIL": 0.05       # 현금
            }
        elif 'Bull' in regime_str:
            # 강세장 + 리스크 있음: 중간 공격
            allocations = {
                "QQQ": 0.30,
                "BTC-USD": 0.20,
                "SOXL": 0.15,
                "IWM": 0.15,
                "GLD": 0.10,
                "BIL": 0.10
            }
        elif 'Bear' in regime_str:
            # 약세장: 방어적이지만 여전히 공격 옵션 보유
            allocations = {
                "BIL": 0.30,
                "GLD": 0.25,
                "TLT": 0.20,
                "BTC-USD": 0.15,  # 약세장에서도 크립토 보유
                "QQQ": 0.10
            }
        else:
            # 혼조: 균형
            allocations = {
                "SPY": 0.25,
                "QQQ": 0.20,
                "BTC-USD": 0.15,
                "GLD": 0.15,
                "TLT": 0.15,
                "BIL": 0.10
            }

        return allocations

    def _decide_action(
        self,
        regime: Dict,
        risk_score: float,
        signals: List[Dict]
    ) -> Tuple[Action, float]:
        """공격적 결정"""
        regime_str = regime.get('regime', 'Unknown')

        if 'Bull' in regime_str:
            # 강세장이면 무조건 진입
            return Action.ENTER, 85
        elif 'Bear' in regime_str and risk_score > 60:
            # 약세 + 고위험이면 탈출
            return Action.EXIT, 70
        elif 'Bear' in regime_str:
            # 약세지만 기회 탐색
            return Action.HOLD, 50
        else:
            return Action.ENTER, 60  # 혼조에서도 진입 시도


class RiskNeutralAgent(BaseRiskAgent):
    """
    Risk Neutral - 균형 투자자

    특징:
    - 위험 대비 수익 최적화
    - 분산 투자 중시
    - 시장 상황에 따른 유연한 대응
    - 감정 배제, 규칙 기반
    """

    def __init__(self, initial_capital: float = 100000.0):
        super().__init__(
            profile=RiskProfile.NEUTRAL,
            initial_capital=initial_capital,
            max_position_pct=0.25
        )

    @property
    def risk_tolerance(self) -> Tuple[int, int]:
        return (30, 70)  # 중간 리스크

    @property
    def preferred_assets(self) -> List[str]:
        return ["equity_us", "bond", "commodity"]

    def calculate_target_allocation(
        self,
        market_data: Dict,
        regime: Dict,
        risk_score: float
    ) -> Dict[str, float]:
        """균형 배분"""
        allocations = {}
        regime_str = regime.get('regime', 'Unknown')

        if 'Bull' in regime_str and risk_score < 30:
            # 강세 + 저위험: 60/40 주식 중심
            allocations = {
                "SPY": 0.25,
                "QQQ": 0.20,
                "XLF": 0.10,
                "TLT": 0.15,
                "LQD": 0.10,
                "GLD": 0.10,
                "BIL": 0.10
            }
        elif 'Bull' in regime_str:
            # 강세 + 리스크: 50/50
            allocations = {
                "SPY": 0.20,
                "QQQ": 0.15,
                "TLT": 0.20,
                "HYG": 0.15,
                "GLD": 0.15,
                "BIL": 0.15
            }
        elif 'Bear' in regime_str:
            # 약세: 30/70 채권 중심
            allocations = {
                "TLT": 0.25,
                "IEF": 0.20,
                "GLD": 0.20,
                "SPY": 0.10,
                "LQD": 0.15,
                "BIL": 0.10
            }
        else:
            # 혼조: 전통적 60/40
            allocations = {
                "SPY": 0.30,
                "TLT": 0.20,
                "LQD": 0.15,
                "GLD": 0.15,
                "IEF": 0.10,
                "BIL": 0.10
            }

        return allocations

    def _decide_action(
        self,
        regime: Dict,
        risk_score: float,
        signals: List[Dict]
    ) -> Tuple[Action, float]:
        """균형적 결정"""
        regime_str = regime.get('regime', 'Unknown')
        confidence_base = regime.get('confidence', 50)

        if 'Bull' in regime_str and risk_score < 40:
            return Action.ENTER, min(confidence_base + 15, 90)
        elif 'Bear' in regime_str and risk_score > 50:
            return Action.EXIT, min(confidence_base + 10, 80)
        elif risk_score > 70:
            return Action.EXIT, 75
        else:
            return Action.HOLD, confidence_base


class RiskAverseAgent(BaseRiskAgent):
    """
    Risk Averse - 보수적 투자자

    특징:
    - 자본 보존 최우선
    - 채권, 현금 중심
    - 낮은 변동성 선호
    - 손실 회피
    """

    def __init__(self, initial_capital: float = 100000.0):
        super().__init__(
            profile=RiskProfile.AVERSE,
            initial_capital=initial_capital,
            max_position_pct=0.20
        )

    @property
    def risk_tolerance(self) -> Tuple[int, int]:
        return (0, 40)  # 저위험만

    @property
    def preferred_assets(self) -> List[str]:
        return ["bond", "cash", "commodity"]

    def calculate_target_allocation(
        self,
        market_data: Dict,
        regime: Dict,
        risk_score: float
    ) -> Dict[str, float]:
        """보수적 배분"""
        allocations = {}
        regime_str = regime.get('regime', 'Unknown')

        if 'Bull' in regime_str and risk_score < 20:
            # 강세 + 매우 저위험: 약간의 주식 노출
            allocations = {
                "TLT": 0.20,
                "IEF": 0.20,
                "TIP": 0.15,
                "GLD": 0.15,
                "SPY": 0.15,
                "XLV": 0.05,  # 헬스케어 (방어)
                "BIL": 0.10
            }
        elif 'Bull' in regime_str:
            # 강세: 채권 중심
            allocations = {
                "TLT": 0.20,
                "IEF": 0.20,
                "LQD": 0.15,
                "TIP": 0.15,
                "GLD": 0.15,
                "BIL": 0.15
            }
        elif 'Bear' in regime_str:
            # 약세: 최대 방어
            allocations = {
                "BIL": 0.30,
                "SHY": 0.25,
                "TIP": 0.20,
                "GLD": 0.20,
                "IEF": 0.05
            }
        else:
            # 혼조: 방어적
            allocations = {
                "IEF": 0.25,
                "TIP": 0.20,
                "GLD": 0.20,
                "LQD": 0.15,
                "BIL": 0.20
            }

        return allocations

    def _decide_action(
        self,
        regime: Dict,
        risk_score: float,
        signals: List[Dict]
    ) -> Tuple[Action, float]:
        """보수적 결정"""
        regime_str = regime.get('regime', 'Unknown')

        if risk_score > 40:
            # 리스크 40 이상이면 무조건 탈출
            return Action.EXIT, 90
        elif 'Bear' in regime_str:
            return Action.EXIT, 85
        elif 'Bull' in regime_str and risk_score < 20:
            return Action.ENTER, 60
        else:
            return Action.HOLD, 50


# ============================================================================
# 에이전트 매니저
# ============================================================================

class RiskAgentManager:
    """에이전트 통합 관리"""

    def __init__(self, initial_capital: float = 100000.0):
        self.agents = {
            RiskProfile.LOVER: RiskLoverAgent(initial_capital),
            RiskProfile.NEUTRAL: RiskNeutralAgent(initial_capital),
            RiskProfile.AVERSE: RiskAverseAgent(initial_capital)
        }
        self.db_path = Path(__file__).parent.parent / "data" / "paper_trades.db"
        self._init_db()

    def _init_db(self):
        """DB 초기화"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 거래 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE,
            timestamp TEXT,
            agent_type TEXT,
            action TEXT,
            ticker TEXT,
            asset_class TEXT,
            quantity REAL,
            price REAL,
            total_value REAL,
            rationale TEXT,
            executed INTEGER,
            execution_price REAL,
            execution_time TEXT
        )
        """)

        # 결정 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            agent_type TEXT,
            action TEXT,
            confidence REAL,
            rationale TEXT,
            allocations_json TEXT,
            market_view TEXT,
            risk_assessment TEXT
        )
        """)

        # 일별 PnL 테이블
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_pnl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            agent_type TEXT,
            portfolio_value REAL,
            daily_return REAL,
            cumulative_return REAL,
            unrealized_pnl REAL,
            position_count INTEGER,
            UNIQUE(date, agent_type)
        )
        """)

        conn.commit()
        conn.close()

    def run_all(
        self,
        market_data: Dict,
        regime: Dict,
        risk_score: float,
        signals: List[Dict] = None
    ) -> Dict[str, AgentDecision]:
        """모든 에이전트 실행"""
        signals = signals or []
        decisions = {}

        for profile, agent in self.agents.items():
            decision = agent.decide(market_data, regime, risk_score, signals)
            decisions[profile.value] = decision

            # 거래 실행
            if decision.trades:
                agent.execute_trades(decision.trades)

            # DB 저장
            self._save_decision(decision)
            for trade in decision.trades:
                self._save_trade(trade)

        return decisions

    def _save_decision(self, decision: AgentDecision):
        """결정 저장"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO decisions (
            timestamp, agent_type, action, confidence,
            rationale, allocations_json, market_view, risk_assessment
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision.timestamp,
            decision.agent_type,
            decision.action.value,
            decision.confidence,
            decision.rationale,
            json.dumps(decision.allocations),
            decision.market_view,
            decision.risk_assessment
        ))

        conn.commit()
        conn.close()

    def _save_trade(self, trade: Trade):
        """거래 저장"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
        INSERT OR REPLACE INTO trades (
            trade_id, timestamp, agent_type, action, ticker,
            asset_class, quantity, price, total_value, rationale,
            executed, execution_price, execution_time
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.trade_id, trade.timestamp, trade.agent_type,
            trade.action, trade.ticker, trade.asset_class,
            trade.quantity, trade.price, trade.total_value,
            trade.rationale, 1 if trade.executed else 0,
            trade.execution_price, trade.execution_time
        ))

        conn.commit()
        conn.close()

    def save_daily_pnl(self, market_data: Dict = None):
        """일별 PnL 저장"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        date = datetime.now().strftime("%Y-%m-%d")

        for profile, agent in self.agents.items():
            pnl = agent.calculate_pnl(market_data)

            cursor.execute("""
            INSERT OR REPLACE INTO daily_pnl (
                date, agent_type, portfolio_value, daily_return,
                cumulative_return, unrealized_pnl, position_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date,
                profile.value,
                pnl['portfolio_value'],
                0,  # daily_return은 이전일 대비 계산 필요
                pnl['total_return'],
                pnl['unrealized_pnl'],
                pnl['position_count']
            ))

        conn.commit()
        conn.close()

    def get_comparison_report(self, market_data: Dict = None) -> str:
        """에이전트 비교 리포트"""
        lines = []
        lines.append("=" * 70)
        lines.append("RISK PROFILE AGENTS - COMPARISON REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        for profile, agent in self.agents.items():
            pnl = agent.calculate_pnl(market_data)
            lines.append(f"\n[{profile.value.upper()}]")
            lines.append("-" * 40)
            lines.append(f"  Portfolio Value: ${pnl['portfolio_value']:,.2f}")
            lines.append(f"  Total Return: {pnl['total_return']:+.2f}%")
            lines.append(f"  Unrealized P&L: ${pnl['unrealized_pnl']:+,.2f}")
            lines.append(f"  Positions: {pnl['position_count']}")

            if pnl['positions']:
                lines.append("  Top Positions:")
                sorted_pos = sorted(
                    pnl['positions'].items(),
                    key=lambda x: -abs(x[1]['unrealized_pnl'])
                )[:3]
                for ticker, pos in sorted_pos:
                    lines.append(
                        f"    - {ticker}: {pos['unrealized_pnl_pct']:+.2f}% "
                        f"(${pos['unrealized_pnl']:+,.2f})"
                    )

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    print("Risk Profile Agents Test")
    print("=" * 50)

    # 에이전트 매니저 생성
    manager = RiskAgentManager(initial_capital=100000)

    # 테스트 데이터
    regime = {
        'regime': 'Bull (Low Vol)',
        'trend': 'Uptrend',
        'volatility': 'Low',
        'confidence': 75
    }
    risk_score = 15.0

    # 시장 데이터 수집
    print("\nFetching market data...")
    tickers = ['SPY', 'QQQ', 'TLT', 'GLD', 'BTC-USD']
    market_data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period='5d', progress=False)
            if len(df) > 0:
                market_data[ticker] = df
        except:
            pass

    print(f"Collected {len(market_data)} tickers")

    # 모든 에이전트 실행
    print("\nRunning agents...")
    decisions = manager.run_all(market_data, regime, risk_score)

    # 결과 출력
    for agent_type, decision in decisions.items():
        print(f"\n[{agent_type}]")
        print(f"  Action: {decision.action.value}")
        print(f"  Confidence: {decision.confidence:.0f}%")
        print(f"  Trades: {len(decision.trades)}")
        print(f"  Rationale: {decision.rationale[:80]}...")

    # 비교 리포트
    print("\n" + manager.get_comparison_report(market_data))
