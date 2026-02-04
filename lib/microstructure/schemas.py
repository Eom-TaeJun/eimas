#!/usr/bin/env python3
"""
Microstructure - Data Schemas
============================================================

시장 미세구조 데이터 클래스 정의

Economic Foundation:
    - Order Book: Kyle (1985) Market microstructure model
    - Trade: Hasbrouck (1991) Information in prices
    - Amihud Lambda: Amihud (2002) Illiquidity measure
    - Roll Spread: Roll (1984) Bid-ask spread estimator
    - VPIN: Easley et al. (2012) Volume-Synchronized PIN

Contains:
    - OrderBookLevel, OrderBook, Trade
    - MicrostructureMetrics
    - AmihudLambdaResult, RollSpreadResult
    - VPINApproxResult, DailyMicrostructureResult
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OrderBookLevel:
    """호가 레벨"""
    price: float
    quantity: float
    side: str  # 'bid' or 'ask'


@dataclass
class OrderBook:
    """호가창 스냅샷"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # 매수호가 (높은 가격순)
    asks: List[OrderBookLevel]  # 매도호가 (낮은 가격순)

    @property
    def mid_price(self) -> float:
        """중간가"""
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    @property
    def spread(self) -> float:
        """스프레드"""
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0


@dataclass
class Trade:
    """체결 거래"""
    symbol: str
    timestamp: datetime
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'


@dataclass
class MicrostructureMetrics:
    """
    미세구조 메트릭 결과

    경제학적 의미:
    - OFI: Order Flow Imbalance (Cont et al. 2014)
    - VPIN: Volume-Synchronized PIN (Easley et al. 2012)
    - Depth Imbalance: Kyle (1985) market depth
    """
    timestamp: datetime
    symbol: str

    # OFI metrics
    ofi_value: float = 0.0
    ofi_cumulative: float = 0.0
    ofi_standardized: float = 0.0

    # VPIN metrics
    vpin: float = 0.0
    vpin_confidence: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    # Depth metrics
    depth_imbalance: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0

    # Volume metrics
    trade_volume: float = 0.0
    trade_count: int = 0
    volume_anomaly_score: float = 0.0

    # Price metrics
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0

    # Metadata
    data_quality: str = "UNKNOWN"  # GOOD, PARTIAL, MISSING
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'symbol': self.symbol,
            'ofi_value': self.ofi_value,
            'ofi_cumulative': self.ofi_cumulative,
            'ofi_standardized': self.ofi_standardized,
            'vpin': self.vpin,
            'vpin_confidence': self.vpin_confidence,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'depth_imbalance': self.depth_imbalance,
            'bid_depth': self.bid_depth,
            'ask_depth': self.ask_depth,
            'trade_volume': self.trade_volume,
            'trade_count': self.trade_count,
            'volume_anomaly_score': self.volume_anomaly_score,
            'mid_price': self.mid_price,
            'spread': self.spread,
            'spread_bps': self.spread_bps,
            'data_quality': self.data_quality,
            'warnings': self.warnings,
        }


@dataclass
class AmihudLambdaResult:
    """
    Amihud (2002) Illiquidity Measure

    Lambda = |return| / volume
    Higher lambda = lower liquidity (higher price impact per volume)
    """
    ticker: str
    lambda_value: float
    avg_daily_volume: float
    avg_abs_return: float
    n_days: int
    interpretation: str  # "LIQUID" / "MODERATE" / "ILLIQUID"


@dataclass
class RollSpreadResult:
    """
    Roll (1984) Bid-Ask Spread Estimator

    Spread = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    Using serial covariance of price changes to estimate spread
    """
    ticker: str
    roll_spread: float
    roll_spread_bps: float
    serial_covariance: float
    n_observations: int
    valid: bool
    note: str


@dataclass
class VPINApproxResult:
    """
    VPIN Approximation for Daily Data

    Simplified VPIN using volume-weighted buy/sell classification
    """
    ticker: str
    vpin_value: float
    buy_volume: float
    sell_volume: float
    total_volume: float
    n_days: int
    interpretation: str  # "LOW" / "MODERATE" / "HIGH"


@dataclass
class DailyMicrostructureResult:
    """
    Daily Microstructure Analysis Result (AMFL Chapter 19)

    Combines:
    - Amihud Lambda (illiquidity)
    - Roll Spread (bid-ask spread estimate)
    - VPIN Approximation (toxicity/informed trading)
    """
    timestamp: str
    tickers_analyzed: List[str]
    amihud_results: Dict[str, AmihudLambdaResult]
    roll_results: Dict[str, RollSpreadResult]
    vpin_results: Dict[str, VPINApproxResult]

    # Aggregate metrics
    avg_liquidity_score: float  # 0-100 (higher = more liquid)
    high_toxicity_tickers: List[str]  # VPIN > 50%
    illiquid_tickers: List[str]  # Liquidity score < 30

    data_quality: str  # "COMPLETE" / "PARTIAL" / "DEGRADED"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp,
            'tickers_analyzed': self.tickers_analyzed,
            'amihud_results': {
                k: {
                    'ticker': v.ticker,
                    'lambda_value': v.lambda_value,
                    'avg_daily_volume': v.avg_daily_volume,
                    'avg_abs_return': v.avg_abs_return,
                    'n_days': v.n_days,
                    'interpretation': v.interpretation
                }
                for k, v in self.amihud_results.items()
            },
            'roll_results': {
                k: {
                    'ticker': v.ticker,
                    'roll_spread': v.roll_spread,
                    'roll_spread_bps': v.roll_spread_bps,
                    'serial_covariance': v.serial_covariance,
                    'n_observations': v.n_observations,
                    'valid': v.valid,
                    'note': v.note
                }
                for k, v in self.roll_results.items()
            },
            'vpin_results': {
                k: {
                    'ticker': v.ticker,
                    'vpin_value': v.vpin_value,
                    'buy_volume': v.buy_volume,
                    'sell_volume': v.sell_volume,
                    'total_volume': v.total_volume,
                    'n_days': v.n_days,
                    'interpretation': v.interpretation
                }
                for k, v in self.vpin_results.items()
            },
            'avg_liquidity_score': self.avg_liquidity_score,
            'high_toxicity_tickers': self.high_toxicity_tickers,
            'illiquid_tickers': self.illiquid_tickers,
            'data_quality': self.data_quality,
            'warnings': self.warnings
        }
