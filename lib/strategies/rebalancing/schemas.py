#!/usr/bin/env python3
"""Rebalancing Strategy - Data Schemas"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from .enums import RebalanceFrequency, RebalancePolicy

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
