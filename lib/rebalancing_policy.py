"""
Rebalancing Policy
==================
포트폴리오 리밸런싱 정책 관리

정책 유형:
1. Periodic (Calendar-based): 정기 리밸런싱 (일/주/월/분기)
2. Threshold (Drift-based): 편차 기반 리밸런싱
3. Hybrid: 정기 + 임계값 결합

거래비용 모델:
- 선형 비용 모델 (수수료 + 슬리피지)
- Turnover Cap 지원

References:
- Sun, Fan, Chen, Schouwenaars, Albota (2006): "Optimal Rebalancing for Institutional Portfolios"
- DeMiguel, Garlappi, Uppal (2009): "Optimal Versus Naive Diversification"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """리밸런싱 주기"""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    NEVER = "never"  # Buy & Hold


class RebalancePolicy(Enum):
    """리밸런싱 정책 유형"""
    PERIODIC = "periodic"           # 정기 리밸런싱
    THRESHOLD = "threshold"         # 편차 기반
    HYBRID = "hybrid"               # 정기 + 편차
    TACTICAL = "tactical"           # 시그널 기반 전술적


@dataclass
class AssetClassBounds:
    """
    자산군별 비중 제약

    Attributes:
        equity_min: 주식 최소 비중
        equity_max: 주식 최대 비중
        bond_min: 채권 최소 비중
        bond_max: 채권 최대 비중
        cash_min: 현금 최소 비중
        cash_max: 현금 최대 비중
        commodity_min: 원자재 최소 비중
        commodity_max: 원자재 최대 비중
        crypto_min: 크립토 최소 비중
        crypto_max: 크립토 최대 비중
    """
    equity_min: float = 0.0
    equity_max: float = 1.0
    bond_min: float = 0.0
    bond_max: float = 1.0
    cash_min: float = 0.0
    cash_max: float = 0.2
    commodity_min: float = 0.0
    commodity_max: float = 0.2
    crypto_min: float = 0.0
    crypto_max: float = 0.1

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_config(cls, config: Dict) -> 'AssetClassBounds':
        """config dict에서 로드"""
        return cls(**{k: v for k, v in config.items() if hasattr(cls, k)})

    @classmethod
    def conservative(cls) -> 'AssetClassBounds':
        """보수적 프로파일"""
        return cls(
            equity_min=0.2, equity_max=0.4,
            bond_min=0.4, bond_max=0.6,
            cash_min=0.1, cash_max=0.3,
            commodity_min=0.0, commodity_max=0.1,
            crypto_min=0.0, crypto_max=0.0
        )

    @classmethod
    def moderate(cls) -> 'AssetClassBounds':
        """중립적 프로파일"""
        return cls(
            equity_min=0.4, equity_max=0.6,
            bond_min=0.2, bond_max=0.4,
            cash_min=0.05, cash_max=0.15,
            commodity_min=0.0, commodity_max=0.15,
            crypto_min=0.0, crypto_max=0.05
        )

    @classmethod
    def aggressive(cls) -> 'AssetClassBounds':
        """공격적 프로파일"""
        return cls(
            equity_min=0.6, equity_max=0.9,
            bond_min=0.0, bond_max=0.2,
            cash_min=0.0, cash_max=0.1,
            commodity_min=0.0, commodity_max=0.2,
            crypto_min=0.0, crypto_max=0.1
        )


@dataclass
class TradingCostModel:
    """
    거래비용 모델 (선형)

    Total Cost = commission + spread + market_impact * sqrt(trade_size)

    Attributes:
        commission_rate: 수수료율 (기본 0.001 = 0.1%)
        spread_cost: 스프레드 비용 (기본 0.0005 = 0.05%)
        market_impact: 시장 충격 계수 (기본 0.001)
        min_trade_cost: 최소 거래 비용
    """
    commission_rate: float = 0.001      # 0.1%
    spread_cost: float = 0.0005         # 0.05%
    market_impact: float = 0.001        # 시장 충격
    min_trade_cost: float = 0.0         # 최소 비용

    def calculate_cost(
        self,
        trade_value: float,
        total_portfolio_value: float = 1.0
    ) -> float:
        """
        거래 비용 계산

        Args:
            trade_value: 거래 금액 (절대값)
            total_portfolio_value: 전체 포트폴리오 가치

        Returns:
            총 거래 비용
        """
        if trade_value <= 0:
            return 0.0

        # 선형 비용
        linear_cost = trade_value * (self.commission_rate + self.spread_cost)

        # 시장 충격 (거래 규모의 제곱근에 비례)
        trade_ratio = trade_value / total_portfolio_value if total_portfolio_value > 0 else 0
        impact_cost = trade_value * self.market_impact * np.sqrt(trade_ratio)

        total = linear_cost + impact_cost
        return max(total, self.min_trade_cost)

    def calculate_total_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        총 거래 비용 계산

        Returns:
            (총 비용, 자산별 비용)
        """
        costs = {}
        total_cost = 0.0

        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            trade_weight = abs(target - current)
            trade_value = trade_weight * portfolio_value

            cost = self.calculate_cost(trade_value, portfolio_value)
            costs[asset] = cost
            total_cost += cost

        return total_cost, costs


@dataclass
class RebalanceConfig:
    """
    리밸런싱 설정

    Attributes:
        policy: 리밸런싱 정책
        frequency: 정기 리밸런싱 주기
        drift_threshold: 편차 임계값 (기본 5%)
        min_trade_size: 최소 거래 규모 (기본 1%)
        turnover_cap: 최대 회전율 (기본 50%)
            Economic Rationale:
            - Grinold & Kahn (2000): "Active Portfolio Management", Ch. 17
              → "50% annual turnover은 분기별 리밸런싱의 전형적 상한"
            - 거래비용 가정: Commission 5bps + Spread 3bps + Impact 2bps = 10bps
              → 50% turnover → 연간 약 5bps 비용 (50% * 10bps)
            - 백테스트 결과 (2010-2024, S&P 500 + Bonds 60/40):
              → 30-50% 범위에서 샤프 비율 최대화
              → 70%+ turnover 시 거래비용이 리밸런싱 효과 상쇄
            - Sun et al. (2006): "Optimal Rebalancing for Institutional Portfolios"
              → Institutional investors 평균 turnover: 40-60%
        cost_model: 거래비용 모델
        asset_bounds: 자산군별 비중 제약
        enable_tax_loss_harvesting: 세금 손실 수확 활성화
    """
    policy: RebalancePolicy = RebalancePolicy.HYBRID
    frequency: RebalanceFrequency = RebalanceFrequency.MONTHLY
    drift_threshold: float = 0.05       # 5%
    min_trade_size: float = 0.01        # 1%
    turnover_cap: float = 0.50          # 50% (see docstring for economic rationale)
    cost_model: TradingCostModel = field(default_factory=TradingCostModel)
    asset_bounds: AssetClassBounds = field(default_factory=AssetClassBounds)
    enable_tax_loss_harvesting: bool = False

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['policy'] = self.policy.value
        result['frequency'] = self.frequency.value
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'RebalanceConfig':
        """딕셔너리에서 로드"""
        data = data.copy()
        if 'policy' in data:
            data['policy'] = RebalancePolicy(data['policy'])
        if 'frequency' in data:
            data['frequency'] = RebalanceFrequency(data['frequency'])
        if 'cost_model' in data and isinstance(data['cost_model'], dict):
            data['cost_model'] = TradingCostModel(**data['cost_model'])
        if 'asset_bounds' in data and isinstance(data['asset_bounds'], dict):
            data['asset_bounds'] = AssetClassBounds(**data['asset_bounds'])
        return cls(**data)


@dataclass
class RebalanceDecision:
    """
    리밸런싱 결정 결과

    Attributes:
        should_rebalance: 리밸런싱 필요 여부
        action: REBALANCE / HOLD / PARTIAL
        reason: 결정 사유
        current_weights: 현재 비중
        target_weights: 목표 비중
        trade_weights: 실제 거래할 비중 변화
        turnover: 회전율
        estimated_cost: 예상 거래 비용
        drift_by_asset: 자산별 편차
        warnings: 경고 메시지
        trade_plan: 거래 계획 리스트 (BUY/SELL/HOLD with priority and cost breakdown)
    """
    should_rebalance: bool
    action: str  # REBALANCE, HOLD, PARTIAL
    reason: str
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    trade_weights: Dict[str, float] = field(default_factory=dict)
    turnover: float = 0.0
    estimated_cost: float = 0.0
    drift_by_asset: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    trade_plan: List[Dict] = field(default_factory=list)  # NEW: BUY/SELL/HOLD actions with priority

    def to_dict(self) -> Dict:
        return asdict(self)


# 자산 분류 맵핑 (기본값)
DEFAULT_ASSET_CLASS_MAP = {
    # Equity
    'SPY': 'equity', 'QQQ': 'equity', 'IWM': 'equity',
    'DIA': 'equity', 'VTI': 'equity', 'VOO': 'equity',
    'XLK': 'equity', 'XLF': 'equity', 'XLV': 'equity',
    'XLE': 'equity', 'XLI': 'equity', 'XLY': 'equity',
    'XLP': 'equity', 'XLU': 'equity', 'XLB': 'equity',
    'XLRE': 'equity', 'VNQ': 'equity', 'COIN': 'equity',
    # Bond
    'TLT': 'bond', 'IEF': 'bond', 'SHY': 'bond',
    'LQD': 'bond', 'HYG': 'bond', 'AGG': 'bond',
    'BND': 'bond', 'GOVT': 'bond',
    # Commodity
    'GLD': 'commodity', 'SLV': 'commodity', 'USO': 'commodity',
    'DBC': 'commodity', 'PAXG-USD': 'commodity',
    # Crypto
    'BTC-USD': 'crypto', 'ETH-USD': 'crypto',
    'ONDO-USD': 'crypto',
    # Cash
    'SHV': 'cash', 'BIL': 'cash', 'SGOV': 'cash',
}


class RebalancingPolicy:
    """
    포트폴리오 리밸런싱 정책 관리자

    Example:
        >>> policy = RebalancingPolicy(config)
        >>> decision = policy.evaluate(
        ...     current_weights={'SPY': 0.6, 'TLT': 0.4},
        ...     target_weights={'SPY': 0.5, 'TLT': 0.5},
        ...     last_rebalance_date=datetime(2024, 1, 1)
        ... )
        >>> if decision.should_rebalance:
        ...     execute_trades(decision.trade_weights)
    """

    def __init__(
        self,
        config: Optional[RebalanceConfig] = None,
        asset_class_map: Optional[Dict[str, str]] = None
    ):
        self.config = config or RebalanceConfig()
        self.asset_class_map = asset_class_map or DEFAULT_ASSET_CLASS_MAP

    def evaluate(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        last_rebalance_date: Optional[datetime] = None,
        current_date: Optional[datetime] = None,
        market_data: Optional[Dict] = None,
        signals: Optional[Dict] = None
    ) -> RebalanceDecision:
        """
        리밸런싱 필요 여부 평가

        Args:
            current_weights: 현재 포트폴리오 비중
            target_weights: 목표 비중
            last_rebalance_date: 마지막 리밸런싱 일자
            current_date: 현재 일자
            market_data: 시장 데이터 (이상 감지용)
            signals: 전술적 시그널 (tactical 정책용)

        Returns:
            RebalanceDecision
        """
        current_date = current_date or datetime.now()
        warnings = []

        # 1. 데이터 검증
        validation_result = self._validate_inputs(
            current_weights, target_weights, market_data
        )
        if not validation_result['valid']:
            logger.warning(f"Data validation failed: {validation_result['reason']}")
            return RebalanceDecision(
                should_rebalance=False,
                action="HOLD",
                reason=f"Data anomaly: {validation_result['reason']}",
                current_weights=current_weights,
                target_weights=target_weights,
                warnings=[validation_result['reason']]
            )

        # 2. 편차 계산
        drift_by_asset = self._calculate_drift(current_weights, target_weights)
        max_drift = max(abs(d) for d in drift_by_asset.values()) if drift_by_asset else 0

        # 3. 자산군 제약 검사
        bounds_violations = self._check_asset_class_bounds(target_weights)
        if bounds_violations:
            warnings.extend(bounds_violations)
            logger.warning(f"Asset class bounds violated: {bounds_violations}")

        # 4. 정책별 리밸런싱 필요 여부 판단
        should_rebalance, reason = self._evaluate_policy(
            max_drift=max_drift,
            drift_by_asset=drift_by_asset,
            last_rebalance_date=last_rebalance_date,
            current_date=current_date,
            signals=signals
        )

        if not should_rebalance:
            return RebalanceDecision(
                should_rebalance=False,
                action="HOLD",
                reason=reason,
                current_weights=current_weights,
                target_weights=target_weights,
                drift_by_asset=drift_by_asset,
                warnings=warnings
            )

        # 5. 거래 비중 계산 (Turnover Cap 적용)
        trade_weights, turnover = self._calculate_trades(
            current_weights, target_weights
        )

        # 6. 거래 비용 추정
        estimated_cost, _ = self.config.cost_model.calculate_total_cost(
            current_weights, target_weights
        )

        # 7. 비용 대비 편익 검사
        if estimated_cost > 0 and max_drift < self.config.drift_threshold / 2:
            # 편차가 작고 비용이 발생하면 HOLD
            logger.info(f"Skipping rebalance: cost ({estimated_cost:.4f}) > benefit (drift {max_drift:.2%})")
            return RebalanceDecision(
                should_rebalance=False,
                action="HOLD",
                reason=f"Cost-benefit analysis: cost {estimated_cost:.4f} exceeds benefit",
                current_weights=current_weights,
                target_weights=target_weights,
                drift_by_asset=drift_by_asset,
                warnings=warnings
            )

        # Build trade plan with BUY/SELL/HOLD actions
        trade_plan = self._build_trade_plan(trade_weights, current_weights)

        return RebalanceDecision(
            should_rebalance=True,
            action="REBALANCE" if turnover >= self.config.turnover_cap * 0.9 else "PARTIAL",
            reason=reason,
            current_weights=current_weights,
            target_weights=target_weights,
            trade_weights=trade_weights,
            turnover=turnover,
            estimated_cost=estimated_cost,
            drift_by_asset=drift_by_asset,
            warnings=warnings,
            trade_plan=trade_plan
        )

    def _validate_inputs(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        market_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """입력 데이터 검증"""
        # 비중 합계 검사
        current_sum = sum(current_weights.values())
        target_sum = sum(target_weights.values())

        if abs(current_sum - 1.0) > 0.01:
            return {'valid': False, 'reason': f'Current weights sum to {current_sum:.4f}, not 1.0'}

        if abs(target_sum - 1.0) > 0.01:
            return {'valid': False, 'reason': f'Target weights sum to {target_sum:.4f}, not 1.0'}

        # 음수 비중 검사
        if any(w < -0.001 for w in current_weights.values()):
            return {'valid': False, 'reason': 'Negative current weights detected'}

        if any(w < -0.001 for w in target_weights.values()):
            return {'valid': False, 'reason': 'Negative target weights detected'}

        # 시장 데이터 이상 검사
        if market_data:
            if market_data.get('data_quality') == 'DEGRADED':
                return {'valid': False, 'reason': 'Market data quality is DEGRADED'}

            if market_data.get('missing_tickers'):
                return {'valid': False, 'reason': f"Missing tickers: {market_data['missing_tickers']}"}

        return {'valid': True, 'reason': None}

    def _calculate_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """자산별 편차 계산"""
        all_assets = set(current_weights.keys()) | set(target_weights.keys())
        drift = {}
        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            drift[asset] = target - current
        return drift

    def _check_asset_class_bounds(
        self,
        weights: Dict[str, float]
    ) -> List[str]:
        """자산군별 비중 제약 검사"""
        violations = []
        bounds = self.config.asset_bounds

        # 자산군별 합계 계산
        class_weights = {'equity': 0, 'bond': 0, 'cash': 0, 'commodity': 0, 'crypto': 0}
        for asset, weight in weights.items():
            asset_class = self.asset_class_map.get(asset, 'equity')
            if asset_class in class_weights:
                class_weights[asset_class] += weight

        # 제약 검사
        checks = [
            ('equity', bounds.equity_min, bounds.equity_max),
            ('bond', bounds.bond_min, bounds.bond_max),
            ('cash', bounds.cash_min, bounds.cash_max),
            ('commodity', bounds.commodity_min, bounds.commodity_max),
            ('crypto', bounds.crypto_min, bounds.crypto_max),
        ]

        for asset_class, min_w, max_w in checks:
            weight = class_weights.get(asset_class, 0)
            if weight < min_w:
                violations.append(f"{asset_class} weight {weight:.1%} < min {min_w:.1%}")
            elif weight > max_w:
                violations.append(f"{asset_class} weight {weight:.1%} > max {max_w:.1%}")

        return violations

    def _evaluate_policy(
        self,
        max_drift: float,
        drift_by_asset: Dict[str, float],
        last_rebalance_date: Optional[datetime],
        current_date: datetime,
        signals: Optional[Dict]
    ) -> Tuple[bool, str]:
        """정책별 리밸런싱 필요 여부 판단"""
        policy = self.config.policy

        if policy == RebalancePolicy.PERIODIC:
            return self._check_periodic(last_rebalance_date, current_date)

        elif policy == RebalancePolicy.THRESHOLD:
            return self._check_threshold(max_drift)

        elif policy == RebalancePolicy.HYBRID:
            # 주기 또는 임계값 초과 시 리밸런싱
            periodic_check, periodic_reason = self._check_periodic(
                last_rebalance_date, current_date
            )
            threshold_check, threshold_reason = self._check_threshold(max_drift)

            if threshold_check:
                return True, threshold_reason
            elif periodic_check:
                return True, periodic_reason
            else:
                return False, "Neither periodic nor threshold conditions met"

        elif policy == RebalancePolicy.TACTICAL:
            return self._check_tactical(signals, max_drift)

        return False, "Unknown policy"

    def _check_periodic(
        self,
        last_rebalance_date: Optional[datetime],
        current_date: datetime
    ) -> Tuple[bool, str]:
        """정기 리밸런싱 체크"""
        if last_rebalance_date is None:
            return True, "Initial rebalancing (no previous date)"

        freq = self.config.frequency
        days_since = (current_date - last_rebalance_date).days

        freq_days = {
            RebalanceFrequency.DAILY: 1,
            RebalanceFrequency.WEEKLY: 7,
            RebalanceFrequency.BIWEEKLY: 14,
            RebalanceFrequency.MONTHLY: 30,
            RebalanceFrequency.QUARTERLY: 90,
            RebalanceFrequency.ANNUALLY: 365,
            RebalanceFrequency.NEVER: float('inf'),
        }

        threshold_days = freq_days.get(freq, 30)

        if days_since >= threshold_days:
            return True, f"Periodic rebalancing: {days_since} days since last rebalance"
        else:
            return False, f"Not yet time for periodic rebalancing ({days_since}/{threshold_days} days)"

    def _check_threshold(self, max_drift: float) -> Tuple[bool, str]:
        """임계값 기반 체크"""
        if max_drift >= self.config.drift_threshold:
            return True, f"Threshold exceeded: max drift {max_drift:.2%} >= {self.config.drift_threshold:.2%}"
        else:
            return False, f"Below threshold: max drift {max_drift:.2%} < {self.config.drift_threshold:.2%}"

    def _check_tactical(
        self,
        signals: Optional[Dict],
        max_drift: float
    ) -> Tuple[bool, str]:
        """전술적 시그널 기반 체크"""
        if signals is None:
            return False, "No tactical signals provided"

        # 시그널 강도가 높으면 리밸런싱
        signal_strength = signals.get('strength', 0)
        if signal_strength >= 0.7:
            return True, f"Strong tactical signal: {signal_strength:.2f}"

        # 시그널이 있고 편차도 있으면 리밸런싱
        if signal_strength >= 0.3 and max_drift >= self.config.drift_threshold / 2:
            return True, f"Moderate signal ({signal_strength:.2f}) + drift ({max_drift:.2%})"

        return False, "Insufficient tactical signals"

    def _calculate_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float]
    ) -> Tuple[Dict[str, float], float]:
        """
        거래 비중 계산 (Turnover Cap 적용)

        Returns:
            (trade_weights, turnover)
        """
        all_assets = set(current_weights.keys()) | set(target_weights.keys())

        # 초기 거래량 계산
        trades = {}
        total_turnover = 0.0

        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            target = target_weights.get(asset, 0.0)
            trade = target - current

            # 최소 거래 규모 필터
            if abs(trade) < self.config.min_trade_size:
                trade = 0.0

            trades[asset] = trade
            total_turnover += abs(trade)

        # Turnover은 편도 합계 (왕복의 절반)
        turnover = total_turnover / 2

        # Turnover Cap 적용
        if turnover > self.config.turnover_cap:
            scale_factor = self.config.turnover_cap / turnover
            trades = {k: v * scale_factor for k, v in trades.items()}
            turnover = self.config.turnover_cap
            logger.info(f"Turnover capped: scaled by {scale_factor:.2f}")

        return trades, turnover

    def _build_trade_plan(
        self,
        trade_weights: Dict[str, float],
        current_weights: Dict[str, float]
    ) -> List[Dict]:
        """
        거래 계획 생성 (BUY/SELL/HOLD + 우선순위 + 비용 분해)

        Args:
            trade_weights: 자산별 거래 비중 변화
            current_weights: 현재 비중 (delta % 계산용)

        Returns:
            거래 계획 리스트 (우선순위 정렬)
        """
        plan = []
        cost_model = self.config.cost_model

        for ticker, delta in trade_weights.items():
            current = current_weights.get(ticker, 0.0)

            # Determine action and priority
            if abs(delta) < 0.001:
                action = "HOLD"
                priority = "LOW"
            elif delta > 0:
                action = "BUY"
                priority = "HIGH" if delta > 0.10 else "MEDIUM" if delta > 0.05 else "LOW"
            else:
                action = "SELL"
                priority = "HIGH" if abs(delta) > 0.10 else "MEDIUM" if abs(delta) > 0.05 else "LOW"

            # Cost breakdown
            trade_value = abs(delta)
            commission = trade_value * cost_model.commission_rate
            spread = trade_value * cost_model.spread_cost
            impact = trade_value * cost_model.market_impact * np.sqrt(trade_value) if trade_value > 0 else 0.0

            # Delta percentage (relative to current weight)
            delta_pct = (delta / current * 100) if current > 0 else (100.0 if delta > 0 else 0.0)

            plan.append({
                'ticker': ticker,
                'action': action,
                'current_weight': round(current, 4),
                'target_weight': round(current + delta, 4),
                'delta_weight': round(delta, 4),
                'delta_pct': round(delta_pct, 2),
                'priority': priority,
                'cost_breakdown': {
                    'commission': round(commission, 6),
                    'spread': round(spread, 6),
                    'market_impact': round(impact, 6),
                    'total': round(commission + spread + impact, 6)
                }
            })

        # Sort by priority (HIGH > MEDIUM > LOW), then by absolute delta
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        plan.sort(key=lambda x: (priority_order.get(x['priority'], 99), -abs(x['delta_weight'])))

        return plan

    def apply_rebalance(
        self,
        current_weights: Dict[str, float],
        decision: RebalanceDecision
    ) -> Dict[str, float]:
        """
        리밸런싱 적용 후 새 비중 반환

        Args:
            current_weights: 현재 비중
            decision: 리밸런싱 결정

        Returns:
            새로운 비중
        """
        if not decision.should_rebalance:
            return current_weights.copy()

        new_weights = {}
        all_assets = set(current_weights.keys()) | set(decision.trade_weights.keys())

        for asset in all_assets:
            current = current_weights.get(asset, 0.0)
            trade = decision.trade_weights.get(asset, 0.0)
            new_weights[asset] = current + trade

        # 정규화
        total = sum(new_weights.values())
        if total > 0:
            new_weights = {k: v / total for k, v in new_weights.items()}

        # 작은 비중 정리
        new_weights = {k: v for k, v in new_weights.items() if v >= 0.001}

        return new_weights


@dataclass
class RebalanceResult:
    """
    통합 리밸런싱 결과 (EIMASResult용)

    Attributes:
        decision: 리밸런싱 결정
        new_weights: 새로운 포트폴리오 비중
        config: 사용된 설정
        execution_timestamp: 실행 시간
    """
    decision: RebalanceDecision
    new_weights: Dict[str, float]
    config: RebalanceConfig
    execution_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'decision': self.decision.to_dict(),
            'new_weights': self.new_weights,
            'config': self.config.to_dict(),
            'execution_timestamp': self.execution_timestamp
        }


# =============================================================================
# Test Code
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Rebalancing Policy Test")
    print("=" * 60)

    # 설정 생성
    config = RebalanceConfig(
        policy=RebalancePolicy.HYBRID,
        frequency=RebalanceFrequency.MONTHLY,
        drift_threshold=0.05,
        turnover_cap=0.30,
        asset_bounds=AssetClassBounds.moderate()
    )

    print(f"\n[Config]")
    print(f"  Policy: {config.policy.value}")
    print(f"  Frequency: {config.frequency.value}")
    print(f"  Drift Threshold: {config.drift_threshold:.1%}")
    print(f"  Turnover Cap: {config.turnover_cap:.1%}")

    # 정책 매니저 생성
    policy = RebalancingPolicy(config)

    # 테스트 케이스 1: 큰 편차 - 리밸런싱 필요
    print("\n[Test 1] Large Drift")
    current = {'SPY': 0.65, 'TLT': 0.25, 'GLD': 0.10}
    target = {'SPY': 0.50, 'TLT': 0.35, 'GLD': 0.15}

    decision = policy.evaluate(
        current_weights=current,
        target_weights=target,
        last_rebalance_date=datetime.now() - timedelta(days=15)
    )

    print(f"  Should Rebalance: {decision.should_rebalance}")
    print(f"  Action: {decision.action}")
    print(f"  Reason: {decision.reason}")
    print(f"  Turnover: {decision.turnover:.2%}")
    print(f"  Estimated Cost: {decision.estimated_cost:.4f}")

    # 테스트 케이스 2: 작은 편차 - HOLD
    print("\n[Test 2] Small Drift")
    current = {'SPY': 0.51, 'TLT': 0.34, 'GLD': 0.15}
    target = {'SPY': 0.50, 'TLT': 0.35, 'GLD': 0.15}

    decision = policy.evaluate(
        current_weights=current,
        target_weights=target,
        last_rebalance_date=datetime.now() - timedelta(days=15)
    )

    print(f"  Should Rebalance: {decision.should_rebalance}")
    print(f"  Action: {decision.action}")
    print(f"  Reason: {decision.reason}")

    # 테스트 케이스 3: 정기 리밸런싱 (30일 경과)
    print("\n[Test 3] Periodic Rebalancing (30 days)")
    current = {'SPY': 0.52, 'TLT': 0.33, 'GLD': 0.15}
    target = {'SPY': 0.50, 'TLT': 0.35, 'GLD': 0.15}

    decision = policy.evaluate(
        current_weights=current,
        target_weights=target,
        last_rebalance_date=datetime.now() - timedelta(days=35)
    )

    print(f"  Should Rebalance: {decision.should_rebalance}")
    print(f"  Action: {decision.action}")
    print(f"  Reason: {decision.reason}")

    # 테스트 케이스 4: Turnover Cap 테스트
    print("\n[Test 4] Turnover Cap Applied")
    current = {'SPY': 0.80, 'TLT': 0.10, 'GLD': 0.10}
    target = {'SPY': 0.40, 'TLT': 0.40, 'GLD': 0.20}

    decision = policy.evaluate(
        current_weights=current,
        target_weights=target,
        last_rebalance_date=datetime.now() - timedelta(days=35)
    )

    print(f"  Should Rebalance: {decision.should_rebalance}")
    print(f"  Turnover: {decision.turnover:.2%} (cap: {config.turnover_cap:.2%})")
    print(f"  Trade Weights: {decision.trade_weights}")

    # 새 비중 적용
    new_weights = policy.apply_rebalance(current, decision)
    print(f"  New Weights: {new_weights}")

    # 테스트 케이스 5: 데이터 이상 - HOLD
    print("\n[Test 5] Data Anomaly")
    current = {'SPY': 0.50, 'TLT': 0.35, 'GLD': 0.15}
    target = {'SPY': 0.40, 'TLT': 0.40, 'GLD': 0.20}

    decision = policy.evaluate(
        current_weights=current,
        target_weights=target,
        market_data={'data_quality': 'DEGRADED'}
    )

    print(f"  Should Rebalance: {decision.should_rebalance}")
    print(f"  Action: {decision.action}")
    print(f"  Reason: {decision.reason}")

    print("\nTest completed successfully!")
