#!/usr/bin/env python3
"""
Trading Cost Model
==================
거래 비용 및 슬리피지 모델 (백테스트 정확도 향상)

핵심 기능:
1. 슬리피지 모델 (Slippage)
2. 시장 충격 비용 (Market Impact)
3. Bid-Ask Spread
4. 수수료 (Commission)

경제학적 배경:
- Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions
- Market Microstructure Theory
- Square-root Impact Model
- Bid-Ask Bounce

백테스트에서 이 모델을 사용하지 않으면:
- 과도하게 낙관적인 수익률
- 실제 거래 시 예상보다 낮은 성과
- 고빈도 전략의 실현 불가능

References:
- Almgren & Chriss (2000): "Optimal Execution of Portfolio Transactions"
- Kissell & Glantz (2003): "Optimal Trading Strategies"
- Easley, López de Prado, O'Hara (2012): "Flow Toxicity and Liquidity"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging


class SlippageModel(str, Enum):
    """슬리피지 모델 타입"""
    FIXED = "fixed"                # 고정 슬리피지 (예: 0.05%)
    PROPORTIONAL = "proportional"  # 비례 슬리피지 (거래량 비례)
    SQUARE_ROOT = "square_root"    # Square-root impact (Almgren)
    VOLUME_BASED = "volume_based"  # 일평균 거래량 기반


class LiquidityTier(str, Enum):
    """유동성 등급"""
    VERY_HIGH = "very_high"  # SPY, QQQ
    HIGH = "high"            # 대형주
    MEDIUM = "medium"        # 중형주
    LOW = "low"              # 소형주
    VERY_LOW = "very_low"    # 마이크로캡


@dataclass
class TradingCostBreakdown:
    """거래 비용 상세"""
    ticker: str
    order_value: float       # 주문 금액
    commission: float        # 수수료
    bid_ask_spread: float    # Bid-Ask Spread 비용
    slippage: float          # 슬리피지
    market_impact: float     # 시장 충격
    total_cost: float        # 총 비용
    cost_bps: float          # 비용 (basis points, 1bp = 0.01%)
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BacktestAdjustment:
    """백테스트 조정 결과"""
    original_return: float
    adjusted_return: float
    total_cost_bps: float
    total_trades: int
    avg_cost_per_trade_bps: float
    improvement_realistic: bool  # 비용 감안 후에도 벤치마크 초과하는가


class TradingCostModel:
    """
    거래 비용 모델

    백테스트에 현실적인 거래 비용을 반영하여
    과도한 낙관주의를 방지합니다.

    비용 구성:
    1. Commission (수수료): 브로커 수수료
    2. Bid-Ask Spread: 매수가-매도가 차이
    3. Slippage: 의도한 가격과 실제 체결가 차이
    4. Market Impact: 대량 주문의 가격 충격

    총 비용 = Commission + Spread + Slippage + Impact
    """

    def __init__(
        self,
        commission_per_share: float = 0.0,  # Alpaca는 무료
        commission_min: float = 0.0,
        commission_pct: float = 0.0,  # 백분율 수수료
        slippage_model: SlippageModel = SlippageModel.SQUARE_ROOT,
        default_spread_bps: float = 5.0,  # 5 bps = 0.05%
        verbose: bool = True
    ):
        """
        Args:
            commission_per_share: 주당 수수료 (예: $0.005)
            commission_min: 최소 수수료
            commission_pct: 거래 금액 기반 수수료 (예: 0.001 = 0.1%)
            slippage_model: 슬리피지 모델 타입
            default_spread_bps: 기본 Bid-Ask Spread (basis points)
            verbose: 로그 출력
        """
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min
        self.commission_pct = commission_pct
        self.slippage_model = slippage_model
        self.default_spread_bps = default_spread_bps
        self.verbose = verbose

        self.logger = self._setup_logger()

        # 유동성 등급별 기본 Spread (bps)
        self.spread_by_liquidity = {
            LiquidityTier.VERY_HIGH: 1.0,   # SPY, QQQ: 1bp
            LiquidityTier.HIGH: 3.0,        # 대형주: 3bp
            LiquidityTier.MEDIUM: 8.0,      # 중형주: 8bp
            LiquidityTier.LOW: 20.0,        # 소형주: 20bp
            LiquidityTier.VERY_LOW: 50.0,   # 마이크로캡: 50bp
        }

        # 유동성 등급별 슬리피지 계수
        self.slippage_coef_by_liquidity = {
            LiquidityTier.VERY_HIGH: 0.5,
            LiquidityTier.HIGH: 1.0,
            LiquidityTier.MEDIUM: 2.0,
            LiquidityTier.LOW: 4.0,
            LiquidityTier.VERY_LOW: 8.0,
        }

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("TradingCostModel")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    # =========================================================================
    # Liquidity Classification
    # =========================================================================

    def classify_liquidity(
        self,
        ticker: str,
        avg_daily_volume: Optional[float] = None,
        avg_daily_dollar_volume: Optional[float] = None
    ) -> LiquidityTier:
        """
        유동성 등급 분류

        Args:
            ticker: 티커
            avg_daily_volume: 일평균 거래량 (주)
            avg_daily_dollar_volume: 일평균 거래 금액 ($)

        Returns:
            LiquidityTier
        """
        # 대표 ETF는 매우 높은 유동성
        mega_liquid = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'GLD', 'EEM', 'VTI']
        if ticker in mega_liquid:
            return LiquidityTier.VERY_HIGH

        # 거래 금액 기반 분류
        if avg_daily_dollar_volume:
            if avg_daily_dollar_volume > 1_000_000_000:  # $1B+
                return LiquidityTier.VERY_HIGH
            elif avg_daily_dollar_volume > 100_000_000:  # $100M+
                return LiquidityTier.HIGH
            elif avg_daily_dollar_volume > 10_000_000:   # $10M+
                return LiquidityTier.MEDIUM
            elif avg_daily_dollar_volume > 1_000_000:    # $1M+
                return LiquidityTier.LOW
            else:
                return LiquidityTier.VERY_LOW

        # 거래량 기반 분류 (fallback)
        if avg_daily_volume:
            if avg_daily_volume > 10_000_000:  # 1000만주+
                return LiquidityTier.VERY_HIGH
            elif avg_daily_volume > 1_000_000:  # 100만주+
                return LiquidityTier.HIGH
            elif avg_daily_volume > 100_000:    # 10만주+
                return LiquidityTier.MEDIUM
            elif avg_daily_volume > 10_000:     # 1만주+
                return LiquidityTier.LOW
            else:
                return LiquidityTier.VERY_LOW

        # 기본값: 중간
        return LiquidityTier.MEDIUM

    # =========================================================================
    # Cost Components
    # =========================================================================

    def calculate_commission(
        self,
        quantity: float,
        price: float
    ) -> float:
        """
        수수료 계산

        Args:
            quantity: 수량
            price: 가격

        Returns:
            수수료 ($)
        """
        order_value = quantity * price

        # 1. 주당 수수료
        commission = quantity * self.commission_per_share

        # 2. 백분율 수수료
        commission += order_value * self.commission_pct

        # 3. 최소 수수료
        commission = max(commission, self.commission_min)

        return commission

    def calculate_bid_ask_spread_cost(
        self,
        ticker: str,
        order_value: float,
        liquidity_tier: Optional[LiquidityTier] = None,
        spread_bps: Optional[float] = None
    ) -> float:
        """
        Bid-Ask Spread 비용 계산

        매수 시: Ask 가격으로 체결 (높음)
        매도 시: Bid 가격으로 체결 (낮음)

        평균적으로 Spread의 절반을 비용으로 부담

        Args:
            ticker: 티커
            order_value: 주문 금액
            liquidity_tier: 유동성 등급
            spread_bps: Spread (basis points)

        Returns:
            Spread 비용 ($)
        """
        if spread_bps is None:
            if liquidity_tier is None:
                liquidity_tier = self.classify_liquidity(ticker)

            spread_bps = self.spread_by_liquidity.get(liquidity_tier, self.default_spread_bps)

        # Spread의 절반을 비용으로 가정 (crossing the spread)
        cost = order_value * (spread_bps / 10000) * 0.5

        return cost

    def calculate_slippage(
        self,
        ticker: str,
        quantity: float,
        price: float,
        avg_daily_volume: Optional[float] = None,
        liquidity_tier: Optional[LiquidityTier] = None
    ) -> float:
        """
        슬리피지 계산

        슬리피지 = 의도한 가격과 실제 체결가 차이

        Args:
            ticker: 티커
            quantity: 수량
            price: 의도한 가격
            avg_daily_volume: 일평균 거래량
            liquidity_tier: 유동성 등급

        Returns:
            슬리피지 ($)
        """
        if liquidity_tier is None:
            liquidity_tier = self.classify_liquidity(ticker, avg_daily_volume=avg_daily_volume)

        order_value = quantity * price

        if self.slippage_model == SlippageModel.FIXED:
            # 고정 슬리피지 (예: 0.05%)
            slippage_bps = 5.0
            return order_value * (slippage_bps / 10000)

        elif self.slippage_model == SlippageModel.PROPORTIONAL:
            # 유동성 등급에 비례
            base_bps = 2.0
            coef = self.slippage_coef_by_liquidity.get(liquidity_tier, 1.0)
            slippage_bps = base_bps * coef
            return order_value * (slippage_bps / 10000)

        elif self.slippage_model == SlippageModel.SQUARE_ROOT:
            # Square-root impact model (Almgren & Chriss)
            # Impact ∝ sqrt(quantity / daily_volume)
            if avg_daily_volume is None or avg_daily_volume == 0:
                # Fallback to proportional
                coef = self.slippage_coef_by_liquidity.get(liquidity_tier, 1.0)
                slippage_bps = 3.0 * coef
                return order_value * (slippage_bps / 10000)

            participation_rate = quantity / avg_daily_volume
            impact_bps = 10.0 * np.sqrt(participation_rate)  # 10 bps per sqrt(participation)

            # 유동성 등급 조정
            coef = self.slippage_coef_by_liquidity.get(liquidity_tier, 1.0)
            impact_bps *= coef

            return order_value * (impact_bps / 10000)

        elif self.slippage_model == SlippageModel.VOLUME_BASED:
            # 거래량 대비 비율 기반
            if avg_daily_volume is None or avg_daily_volume == 0:
                slippage_bps = 5.0
                return order_value * (slippage_bps / 10000)

            participation_rate = quantity / avg_daily_volume

            if participation_rate < 0.01:  # < 1% ADV
                slippage_bps = 2.0
            elif participation_rate < 0.05:  # < 5% ADV
                slippage_bps = 5.0
            elif participation_rate < 0.10:  # < 10% ADV
                slippage_bps = 15.0
            else:  # >= 10% ADV (매우 높은 충격)
                slippage_bps = 50.0 * participation_rate

            return order_value * (slippage_bps / 10000)

        return 0.0

    def calculate_market_impact(
        self,
        ticker: str,
        quantity: float,
        price: float,
        avg_daily_volume: Optional[float] = None,
        liquidity_tier: Optional[LiquidityTier] = None
    ) -> float:
        """
        시장 충격 비용 계산

        대량 주문이 시장 가격을 움직이는 효과

        Linear Impact Model:
        Impact = α * (Q / V)
        - Q: 주문 수량
        - V: 일평균 거래량
        - α: 충격 계수

        Args:
            ticker: 티커
            quantity: 수량
            price: 가격
            avg_daily_volume: 일평균 거래량
            liquidity_tier: 유동성 등급

        Returns:
            시장 충격 비용 ($)
        """
        if avg_daily_volume is None or avg_daily_volume == 0:
            # 거래량 정보 없으면 보수적으로 0
            return 0.0

        if liquidity_tier is None:
            liquidity_tier = self.classify_liquidity(ticker, avg_daily_volume=avg_daily_volume)

        order_value = quantity * price
        participation_rate = quantity / avg_daily_volume

        # 참여율이 낮으면 충격 무시
        if participation_rate < 0.001:  # 0.1% 미만
            return 0.0

        # Linear impact: 참여율에 비례
        # 유동성이 낮을수록 충격 계수 증가
        impact_coef = {
            LiquidityTier.VERY_HIGH: 5.0,
            LiquidityTier.HIGH: 10.0,
            LiquidityTier.MEDIUM: 20.0,
            LiquidityTier.LOW: 40.0,
            LiquidityTier.VERY_LOW: 100.0,
        }.get(liquidity_tier, 20.0)

        impact_bps = impact_coef * participation_rate

        return order_value * (impact_bps / 10000)

    # =========================================================================
    # Total Cost Calculation
    # =========================================================================

    def calculate_total_cost(
        self,
        ticker: str,
        quantity: float,
        price: float,
        avg_daily_volume: Optional[float] = None,
        liquidity_tier: Optional[LiquidityTier] = None
    ) -> TradingCostBreakdown:
        """
        총 거래 비용 계산

        Args:
            ticker: 티커
            quantity: 수량
            price: 가격
            avg_daily_volume: 일평균 거래량
            liquidity_tier: 유동성 등급

        Returns:
            TradingCostBreakdown
        """
        order_value = quantity * price

        if liquidity_tier is None:
            liquidity_tier = self.classify_liquidity(ticker, avg_daily_volume=avg_daily_volume)

        # 각 비용 구성요소 계산
        commission = self.calculate_commission(quantity, price)
        spread_cost = self.calculate_bid_ask_spread_cost(ticker, order_value, liquidity_tier)
        slippage = self.calculate_slippage(ticker, quantity, price, avg_daily_volume, liquidity_tier)
        market_impact = self.calculate_market_impact(ticker, quantity, price, avg_daily_volume, liquidity_tier)

        total_cost = commission + spread_cost + slippage + market_impact
        cost_bps = (total_cost / order_value) * 10000 if order_value > 0 else 0.0

        breakdown = TradingCostBreakdown(
            ticker=ticker,
            order_value=order_value,
            commission=commission,
            bid_ask_spread=spread_cost,
            slippage=slippage,
            market_impact=market_impact,
            total_cost=total_cost,
            cost_bps=cost_bps,
            timestamp=datetime.now().isoformat()
        )

        return breakdown

    # =========================================================================
    # Backtest Integration
    # =========================================================================

    def adjust_backtest_returns(
        self,
        trades: List[Dict],
        initial_capital: float
    ) -> BacktestAdjustment:
        """
        백테스트 수익률을 거래 비용으로 조정

        Args:
            trades: 거래 리스트 [{'ticker': 'SPY', 'quantity': 100, 'price': 400, ...}, ...]
            initial_capital: 초기 자본

        Returns:
            BacktestAdjustment
        """
        total_cost = 0.0
        total_trades = len(trades)

        for trade in trades:
            ticker = trade.get('ticker', 'SPY')
            quantity = trade.get('quantity', 0)
            price = trade.get('price', 0)
            avg_daily_volume = trade.get('avg_daily_volume')

            cost_breakdown = self.calculate_total_cost(
                ticker=ticker,
                quantity=abs(quantity),
                price=price,
                avg_daily_volume=avg_daily_volume
            )

            total_cost += cost_breakdown.total_cost

        # 비용이 수익률에 미치는 영향
        total_cost_bps = (total_cost / initial_capital) * 10000
        avg_cost_per_trade_bps = total_cost_bps / total_trades if total_trades > 0 else 0.0

        self.logger.info(f"Total trading cost: ${total_cost:,.2f} ({total_cost_bps:.1f} bps)")
        self.logger.info(f"Avg cost per trade: {avg_cost_per_trade_bps:.2f} bps")

        # 조정된 수익률 계산은 외부에서 수행
        return BacktestAdjustment(
            original_return=0.0,  # 외부에서 설정
            adjusted_return=0.0,  # 외부에서 설정
            total_cost_bps=total_cost_bps,
            total_trades=total_trades,
            avg_cost_per_trade_bps=avg_cost_per_trade_bps,
            improvement_realistic=False  # 외부에서 설정
        )

    def estimate_cost_for_strategy(
        self,
        annual_turnover: float,
        avg_order_size: float,
        portfolio_value: float,
        ticker: str = "SPY"
    ) -> Dict[str, float]:
        """
        전략의 연간 거래 비용 추정

        Args:
            annual_turnover: 연간 회전율 (예: 2.0 = 200%)
            avg_order_size: 평균 주문 크기 ($)
            portfolio_value: 포트폴리오 가치 ($)
            ticker: 대표 티커

        Returns:
            비용 추정치
        """
        # 연간 거래 금액
        annual_trade_volume = portfolio_value * annual_turnover

        # 거래 횟수
        num_trades = annual_trade_volume / avg_order_size

        # 대표 거래의 비용
        sample_quantity = avg_order_size / 100  # 가정: 주가 $100
        sample_price = 100.0

        sample_cost = self.calculate_total_cost(
            ticker=ticker,
            quantity=sample_quantity,
            price=sample_price
        )

        # 연간 총 비용
        annual_cost = sample_cost.total_cost * num_trades
        annual_cost_bps = (annual_cost / portfolio_value) * 10000

        return {
            'annual_turnover': annual_turnover,
            'annual_trade_volume': annual_trade_volume,
            'num_trades': num_trades,
            'avg_cost_per_trade': sample_cost.total_cost,
            'avg_cost_per_trade_bps': sample_cost.cost_bps,
            'annual_total_cost': annual_cost,
            'annual_cost_bps': annual_cost_bps,
            'annual_cost_pct': annual_cost_bps / 100
        }


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing TradingCostModel")
    print("=" * 70)

    model = TradingCostModel(
        commission_per_share=0.0,  # Alpaca 무료
        slippage_model=SlippageModel.SQUARE_ROOT,
        verbose=True
    )

    print("\n[1] Calculate cost for SPY trade")
    cost = model.calculate_total_cost(
        ticker="SPY",
        quantity=100,
        price=450.0,
        avg_daily_volume=80_000_000
    )

    print(f"\nOrder: BUY 100 shares of SPY @ $450")
    print(f"Order Value: ${cost.order_value:,.2f}")
    print(f"  - Commission: ${cost.commission:.2f}")
    print(f"  - Bid-Ask Spread: ${cost.bid_ask_spread:.2f}")
    print(f"  - Slippage: ${cost.slippage:.2f}")
    print(f"  - Market Impact: ${cost.market_impact:.2f}")
    print(f"  - TOTAL COST: ${cost.total_cost:.2f} ({cost.cost_bps:.2f} bps)")

    print("\n[2] Compare liquidity tiers")
    tickers = [
        ('SPY', 80_000_000, LiquidityTier.VERY_HIGH),
        ('AAPL', 50_000_000, LiquidityTier.HIGH),
        ('SOXX', 5_000_000, LiquidityTier.MEDIUM),
        ('Small Cap', 100_000, LiquidityTier.LOW)
    ]

    for ticker, volume, tier in tickers:
        cost = model.calculate_total_cost(
            ticker=ticker,
            quantity=100,
            price=100.0,
            avg_daily_volume=volume,
            liquidity_tier=tier
        )
        print(f"{ticker:12} ({tier.value:12}): ${cost.total_cost:6.2f} ({cost.cost_bps:5.1f} bps)")

    print("\n[3] Estimate annual cost for strategy")
    estimate = model.estimate_cost_for_strategy(
        annual_turnover=2.0,      # 200% turnover
        avg_order_size=10000.0,   # $10K per trade
        portfolio_value=100000.0  # $100K portfolio
    )

    print(f"Annual Turnover: {estimate['annual_turnover']:.1%}")
    print(f"Number of Trades: {estimate['num_trades']:.0f}")
    print(f"Avg Cost per Trade: ${estimate['avg_cost_per_trade']:.2f} ({estimate['avg_cost_per_trade_bps']:.2f} bps)")
    print(f"Annual Total Cost: ${estimate['annual_total_cost']:,.2f}")
    print(f"Annual Cost Impact: {estimate['annual_cost_bps']:.1f} bps ({estimate['annual_cost_pct']:.2%})")

    print("\n[4] High-frequency strategy cost")
    hft_estimate = model.estimate_cost_for_strategy(
        annual_turnover=50.0,     # 5000% turnover (고빈도)
        avg_order_size=5000.0,
        portfolio_value=100000.0
    )

    print(f"Annual Turnover: {hft_estimate['annual_turnover']:.0%}")
    print(f"Number of Trades: {hft_estimate['num_trades']:.0f}")
    print(f"Annual Cost Impact: {hft_estimate['annual_cost_bps']:.1f} bps ({hft_estimate['annual_cost_pct']:.2%})")
    print("\n⚠️  High-frequency strategies are often unprofitable after costs!")

    print("\n" + "=" * 70)
