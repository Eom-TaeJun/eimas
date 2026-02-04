#!/usr/bin/env python3
"""
Genius Act - Decision Rules
============================================================

Rule-based decision logic for Genius Act strategy

Economic Foundation:
    - Liquidity-driven asset allocation
    - Threshold-based regime classification

Class:
    - GeniusActRules: Decision rules engine
"""

from typing import Dict, Optional, List
import logging

from .schemas import MacroSignal, LiquidityIndicators, StrategyPosition
from .enums import LiquidityRegime, SignalType

logger = logging.getLogger(__name__)


class GeniusActRules:
    """
    Genius Act 기반 규칙:
    1. 스테이블코인 발행자는 미국 국채/현금 담보 보유 필수
    2. 담보 요건이 국채 수요에 영향
    3. 스테이블코인 증가 → 국채 수요 증가
    """

    # 임계값 설정
    STABLECOIN_SURGE_THRESHOLD = 0.05       # 5% 주간 증가
    STABLECOIN_DRAIN_THRESHOLD = -0.03      # 3% 주간 감소
    RRP_DRAIN_THRESHOLD = 0.10              # 10% 월간 감소
    TGA_DRAIN_THRESHOLD = 0.15              # 15% 월간 감소
    LIQUIDITY_CHANGE_THRESHOLD = 0.02       # 2% 변화

    @staticmethod
    def check_stablecoin_signals(
        current: LiquidityIndicators,
        previous: LiquidityIndicators
    ) -> List[MacroSignal]:
        """스테이블코인 관련 시그널 체크"""
        signals = []

        # 총 스테이블코인 공급
        current_total = current.usdt_supply + current.usdc_supply + current.dai_supply
        previous_total = previous.usdt_supply + previous.usdc_supply + previous.dai_supply

        if previous_total > 0:
            change = (current_total - previous_total) / previous_total

            if change > GeniusActRules.STABLECOIN_SURGE_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.STABLECOIN_SURGE,
                    strength=min(change / 0.1, 1.0),  # 10% 증가 시 최대
                    description=f"스테이블코인 공급 {change*100:.1f}% 증가 - 국채 수요 상승 예상",
                    triggered_at=current.timestamp,
                    affected_assets=["TLT", "IEF", "SHY", "BTC-USD"],
                    confidence=0.75,
                    metadata={
                        "usdt_change": f"{(current.usdt_supply - previous.usdt_supply):.1f}B",
                        "usdc_change": f"{(current.usdc_supply - previous.usdc_supply):.1f}B",
                        "total_supply": f"{current_total:.1f}B"
                    }
                ))

                # 크립토 리스크온
                signals.append(MacroSignal(
                    signal_type=SignalType.CRYPTO_RISK_ON,
                    strength=min(change / 0.1, 1.0),
                    description="스테이블코인 유입 → 크립토 매수 대기 자금 증가",
                    triggered_at=current.timestamp,
                    affected_assets=["BTC-USD", "ETH-USD", "COIN", "MSTR"],
                    confidence=0.7
                ))

            elif change < GeniusActRules.STABLECOIN_DRAIN_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.STABLECOIN_DRAIN,
                    strength=abs(change) / 0.1,
                    description=f"스테이블코인 공급 {change*100:.1f}% 감소 - 크립토 자금 이탈",
                    triggered_at=current.timestamp,
                    affected_assets=["BTC-USD", "ETH-USD"],
                    confidence=0.7,
                    metadata={"drain_amount": f"{previous_total - current_total:.1f}B"}
                ))

                signals.append(MacroSignal(
                    signal_type=SignalType.CRYPTO_RISK_OFF,
                    strength=abs(change) / 0.1,
                    description="스테이블코인 이탈 → 크립토 매도 압력",
                    triggered_at=current.timestamp,
                    affected_assets=["BTC-USD", "ETH-USD", "COIN"],
                    confidence=0.65
                ))

        return signals

    @staticmethod
    def check_fed_liquidity_signals(
        current: LiquidityIndicators,
        previous: LiquidityIndicators
    ) -> List[MacroSignal]:
        """Fed 유동성 관련 시그널"""
        signals = []

        # 역레포 변화
        if previous.rrp_balance > 0:
            rrp_change = (current.rrp_balance - previous.rrp_balance) / previous.rrp_balance

            if rrp_change < -GeniusActRules.RRP_DRAIN_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.RRP_DRAIN,
                    strength=min(abs(rrp_change) / 0.2, 1.0),
                    description=f"역레포 {rrp_change*100:.1f}% 감소 → 시장 유동성 주입",
                    triggered_at=current.timestamp,
                    affected_assets=["SPY", "QQQ", "BTC-USD", "TLT"],
                    confidence=0.8,
                    metadata={
                        "rrp_drain": f"${abs(current.rrp_balance - previous.rrp_balance)*1000:.0f}B",
                        "remaining_rrp": f"${current.rrp_balance:.2f}T"
                    }
                ))

        # TGA 변화
        if previous.tga_balance > 0:
            tga_change = (current.tga_balance - previous.tga_balance) / previous.tga_balance

            if tga_change < -GeniusActRules.TGA_DRAIN_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.TGA_DRAIN,
                    strength=min(abs(tga_change) / 0.3, 1.0),
                    description=f"TGA {tga_change*100:.1f}% 감소 → 재정 지출로 유동성 공급",
                    triggered_at=current.timestamp,
                    affected_assets=["SPY", "IWM"],
                    confidence=0.75
                ))

        # 순 유동성 변화
        current_net = current.fed_balance_sheet - current.rrp_balance - current.tga_balance
        previous_net = previous.fed_balance_sheet - previous.rrp_balance - previous.tga_balance

        if previous_net > 0:
            net_change = (current_net - previous_net) / previous_net

            if net_change > GeniusActRules.LIQUIDITY_CHANGE_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.LIQUIDITY_INJECTION,
                    strength=min(net_change / 0.05, 1.0),
                    description=f"순 유동성 {net_change*100:.1f}% 증가 → 위험자산 강세",
                    triggered_at=current.timestamp,
                    affected_assets=["SPY", "QQQ", "BTC-USD", "HYG"],
                    confidence=0.85
                ))
            elif net_change < -GeniusActRules.LIQUIDITY_CHANGE_THRESHOLD:
                signals.append(MacroSignal(
                    signal_type=SignalType.LIQUIDITY_DRAIN,
                    strength=min(abs(net_change) / 0.05, 1.0),
                    description=f"순 유동성 {net_change*100:.1f}% 감소 → 위험자산 약세",
                    triggered_at=current.timestamp,
                    affected_assets=["SPY", "QQQ", "BTC-USD"],
                    confidence=0.8
                ))

        return signals

    @staticmethod
    def check_treasury_signals(
        current: LiquidityIndicators,
        previous: LiquidityIndicators,
        treasury_supply_change: float = 0  # 국채 발행량 변화
    ) -> List[MacroSignal]:
        """국채 관련 시그널"""
        signals = []

        # 스테이블코인 증가 → 국채 담보 수요
        stablecoin_current = current.usdt_supply + current.usdc_supply
        stablecoin_previous = previous.usdt_supply + previous.usdc_supply

        if stablecoin_previous > 0:
            sc_change = (stablecoin_current - stablecoin_previous) / stablecoin_previous

            # 스테이블코인 증가 시 국채 수요 증가
            if sc_change > 0.03:
                signals.append(MacroSignal(
                    signal_type=SignalType.TREASURY_DEMAND,
                    strength=min(sc_change / 0.1, 1.0),
                    description="스테이블코인 담보 요건 → 단기 국채 수요 증가",
                    triggered_at=current.timestamp,
                    affected_assets=["SHY", "BIL", "SGOV"],
                    confidence=0.7,
                    metadata={
                        "estimated_demand": f"${sc_change * stablecoin_current:.1f}B"
                    }
                ))

        # 국채 공급 증가 (국채 발행)
        if treasury_supply_change > 0.05:  # 5% 이상 발행 증가
            signals.append(MacroSignal(
                signal_type=SignalType.TREASURY_SUPPLY,
                strength=min(treasury_supply_change / 0.1, 1.0),
                description=f"국채 발행 {treasury_supply_change*100:.1f}% 증가 → 금리 상승 압력",
                triggered_at=current.timestamp,
                affected_assets=["TLT", "IEF", "TBT"],
                confidence=0.65
            ))

        return signals


# =============================================================================
# 매크로 전략 엔진
# =============================================================================

