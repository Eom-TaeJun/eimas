#!/usr/bin/env python3
"""
Operational - Configuration
============================

운영 설정 관리 모듈

Economic Foundation:
    하드코딩 없이 설정값을 관리하여 감사 가능성과 재현성 확보

Configuration Classes:
    - OperationalConfig: 의사결정, 리밸런싱, 제약 조건 임계값

Design Principles:
    - No hardcoded values (모든 값은 config로 관리)
    - Deterministic and auditable (결정론적이고 감사 가능)
    - Explainable thresholds (각 임계값의 경제학적 근거 명시)
"""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class OperationalConfig:
    """
    운영 설정 (하드코딩 없이 config로 관리)

    Economic Rationale:
        - confidence_threshold: Grinold & Kahn (2000) - IC > 0.05 유의미
        - turnover_cap: Trading cost optimization (30% = 3bps annual cost)
        - risk_score: VaR 95% 신뢰구간 기준
    """
    # Decision Governance Thresholds
    confidence_threshold_high: float = 0.70  # 70% 이상 → BULLISH/BEARISH
    confidence_threshold_low: float = 0.50   # 50% 미만 → HOLD
    risk_score_high: float = 70.0            # 70점 이상 → 위험 회피
    risk_score_low: float = 30.0             # 30점 이하 → 정상

    # Rebalancing Thresholds
    turnover_cap: float = 0.30               # 30% 최대 회전율
    min_trade_size: float = 0.01             # 1% 최소 거래 크기
    human_approval_threshold: float = 0.20   # 20% 이상 변화 시 승인 필요

    # Asset Class Bounds (constraint repair용)
    equity_min: float = 0.0
    equity_max: float = 1.0
    bond_min: float = 0.0
    bond_max: float = 1.0
    cash_min: float = 0.0
    cash_max: float = 0.20
    commodity_min: float = 0.0
    commodity_max: float = 0.20
    crypto_min: float = 0.0
    crypto_max: float = 0.10  # 크립토 최대 10%

    # Risk Limits
    max_single_asset_weight: float = 0.30    # 단일 자산 최대 30%
    max_sector_concentration: float = 0.40   # 섹터 집중도 최대 40%

    # Trading Cost Model
    commission_rate: float = 0.001
    spread_cost: float = 0.0005
    market_impact: float = 0.001

    @classmethod
    def from_dict(cls, data: Dict) -> 'OperationalConfig':
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    def to_dict(self) -> Dict:
        return asdict(self)


# Asset-class mapping shared across constraint repair and rebalance planning.
# Keep this close to config so classification policy is centralized.
ASSET_CLASS_MAP = {
    'SPY': 'equity', 'QQQ': 'equity', 'IWM': 'equity', 'DIA': 'equity',
    'VTI': 'equity', 'VOO': 'equity', 'XLK': 'equity', 'XLF': 'equity',
    'XLV': 'equity', 'XLE': 'equity', 'XLI': 'equity', 'XLY': 'equity',
    'XLP': 'equity', 'XLU': 'equity', 'XLB': 'equity', 'XLRE': 'equity',
    'VNQ': 'equity', 'COIN': 'equity',
    'TLT': 'bond', 'IEF': 'bond', 'SHY': 'bond', 'LQD': 'bond',
    'HYG': 'bond', 'AGG': 'bond', 'BND': 'bond', 'GOVT': 'bond', 'UUP': 'bond',
    'GLD': 'commodity', 'SLV': 'commodity', 'USO': 'commodity',
    'DBC': 'commodity', 'PAXG-USD': 'commodity',
    'BTC-USD': 'crypto', 'ETH-USD': 'crypto', 'ONDO-USD': 'crypto',
    'SHV': 'cash', 'BIL': 'cash', 'SGOV': 'cash',
}
