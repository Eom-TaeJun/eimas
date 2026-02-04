#!/usr/bin/env python3
"""
Genius Act - Liquidity Models
============================================================

Fed liquidity calculation and monitoring

Economic Foundation:
    - Genius Act framework: Net Liquidity = Fed BS - RRP - TGA
    - Expanded model: M = B + S·B* (stablecoin contribution)

Classes:
    - ExtendedLiquidityModel: Liquidity calculation with stablecoin
    - LiquidityMonitor: Real-time liquidity monitoring
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .schemas import LiquidityIndicators
from .enums import LiquidityRegime

logger = logging.getLogger(__name__)


class ExtendedLiquidityModel:
    """
    확장 유동성 공식: M = B + S·B*

    M = 총 유효 유동성
    B = 기본 유동성 (Fed BS - RRP - TGA)
    S = 스테이블코인 승수 (발행량/담보비율 기반)
    B* = 스테이블코인 담보 자산 (국채 등)
    """

    def __init__(
        self,
        stablecoin_multiplier: float = 0.9,  # 담보비율 반영
        crypto_impact_factor: float = 0.1    # 크립토가 전통금융에 미치는 영향
    ):
        self.stablecoin_multiplier = stablecoin_multiplier
        self.crypto_impact_factor = crypto_impact_factor

    def calculate_base_liquidity(
        self,
        fed_bs: float,
        rrp: float,
        tga: float
    ) -> float:
        """기본 유동성 계산: B = Fed BS - RRP - TGA"""
        return fed_bs - rrp - tga

    def calculate_stablecoin_contribution(
        self,
        usdt_supply: float,
        usdc_supply: float,
        dai_supply: float
    ) -> float:
        """스테이블코인 기여도: S·B*"""
        total_stablecoin = usdt_supply + usdc_supply + dai_supply
        # 스테이블코인 담보는 대부분 국채 + 현금성 자산
        # S·B* = total_stablecoin * multiplier
        return total_stablecoin * self.stablecoin_multiplier / 1000  # 조 달러로 변환

    def calculate_total_liquidity(
        self,
        indicators: LiquidityIndicators
    ) -> Dict[str, float]:
        """총 유효 유동성: M = B + S·B*"""

        B = self.calculate_base_liquidity(
            indicators.fed_balance_sheet,
            indicators.rrp_balance,
            indicators.tga_balance
        )

        SB_star = self.calculate_stablecoin_contribution(
            indicators.usdt_supply,
            indicators.usdc_supply,
            indicators.dai_supply
        )

        M = B + SB_star * self.crypto_impact_factor

        return {
            "base_liquidity_B": B,
            "stablecoin_contribution_SBstar": SB_star,
            "total_liquidity_M": M,
            "stablecoin_share": SB_star / M if M > 0 else 0,
            "formula": f"M({M:.2f}) = B({B:.2f}) + S·B*({SB_star:.2f})"
        }


# =============================================================================
# Genius Act 규칙 엔진
# =============================================================================

class LiquidityMonitor:
    """유동성 모니터링"""

    def __init__(self):
        self.history: List[LiquidityIndicators] = []
        self.strategy = GeniusActMacroStrategy()

    def update(self, indicators: LiquidityIndicators):
        """지표 업데이트"""
        self.history.append(indicators)

    def get_trend(self, window: int = 5) -> Dict:
        """트렌드 분석"""
        if len(self.history) < window:
            return {"error": "Insufficient data"}

        recent = self.history[-window:]

        # 스테이블코인 트렌드
        sc_trend = []
        for h in recent:
            total = h.usdt_supply + h.usdc_supply + h.dai_supply
            sc_trend.append(total)

        sc_change = (sc_trend[-1] - sc_trend[0]) / sc_trend[0] if sc_trend[0] > 0 else 0

        # 유동성 트렌드
        liq_trend = []
        for h in recent:
            net = h.fed_balance_sheet - h.rrp_balance - h.tga_balance
            liq_trend.append(net)

        liq_change = (liq_trend[-1] - liq_trend[0]) / liq_trend[0] if liq_trend[0] > 0 else 0

        return {
            "stablecoin_trend": "UP" if sc_change > 0.02 else "DOWN" if sc_change < -0.02 else "FLAT",
            "stablecoin_change": f"{sc_change*100:.1f}%",
            "liquidity_trend": "UP" if liq_change > 0.01 else "DOWN" if liq_change < -0.01 else "FLAT",
            "liquidity_change": f"{liq_change*100:.1f}%",
            "window": f"{window} periods"
        }

    def get_alerts(self) -> List[str]:
        """경고 알림"""
        alerts = []

        if len(self.history) < 2:
            return alerts

        current = self.history[-1]
        previous = self.history[-2]

        # 역레포 고갈 경고
        if current.rrp_balance < 0.2:  # 2000억 달러 미만
            alerts.append("⚠️ 역레포 잔액 고갈 임박 - 유동성 완충재 부족")

        # TGA 급락
        if previous.tga_balance > 0:
            tga_change = (current.tga_balance - previous.tga_balance) / previous.tga_balance
            if tga_change < -0.2:
                alerts.append(f"📊 TGA {tga_change*100:.0f}% 급락 - 대규모 재정 지출")

        # 스테이블코인 급변
        current_sc = current.usdt_supply + current.usdc_supply
        previous_sc = previous.usdt_supply + previous.usdc_supply
        if previous_sc > 0:
            sc_change = (current_sc - previous_sc) / previous_sc
            if sc_change > 0.1:
                alerts.append(f"🚀 스테이블코인 {sc_change*100:.0f}% 급증 - 크립토 유입 가속")
            elif sc_change < -0.05:
                alerts.append(f"🔻 스테이블코인 {sc_change*100:.0f}% 급감 - 크립토 이탈")

        return alerts


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Genius Act Macro Strategy Test")
    print("=" * 60)

    # 테스트 데이터 (2023-2024 시나리오 시뮬레이션)
    previous = LiquidityIndicators(
        fed_balance_sheet=7.8,     # 7.8조 달러
        rrp_balance=1.5,           # 1.5조 달러
        tga_balance=0.5,           # 5000억 달러
        usdt_supply=80,            # 800억 달러
        usdc_supply=30,            # 300억 달러
        dai_supply=5,              # 50억 달러
        m2=20.5,
        dxy=103,
        timestamp=datetime(2024, 1, 1)
    )

    current = LiquidityIndicators(
        fed_balance_sheet=7.5,     # 3000억 QT
        rrp_balance=0.8,           # 7000억 감소 (역레포 drain)
        tga_balance=0.6,           # 1000억 증가
        usdt_supply=95,            # 150억 증가 (+18.75%)
        usdc_supply=35,            # 50억 증가
        dai_supply=5,
        m2=20.8,
        dxy=101,
        timestamp=datetime(2024, 6, 1)
    )

    # 전략 실행
    strategy = GeniusActMacroStrategy()
    result = strategy.analyze(current, previous)

    print("\n1. Liquidity Analysis:")
    print(f"   Formula: {result['liquidity']['formula']}")
    print(f"   Base Liquidity (B): ${result['liquidity']['base_liquidity_B']:.2f}T")
    print(f"   Stablecoin Contribution (S·B*): ${result['liquidity']['stablecoin_contribution_SBstar']:.3f}T")
    print(f"   Total Liquidity (M): ${result['liquidity']['total_liquidity_M']:.2f}T")

    print(f"\n2. Current Regime: {result['regime']}")

    print("\n3. Generated Signals:")
    for sig in result['signals']:
        print(f"   [{sig['type']}] {sig['description']}")
        print(f"      Strength: {sig['strength']}, Confidence: {sig['confidence']}")
        print(f"      Affected: {', '.join(sig['affected_assets'])}")

    print("\n4. Recommended Positions:")
    for pos in result['positions']:
        print(f"   {pos['direction']} {pos['asset']} ({pos['size']})")
        print(f"      Signal: {pos['signal']}")
        print(f"      Rationale: {pos['rationale']}")

    print(f"\n5. Summary: {result['summary']}")

    # 모니터링 테스트
    print("\n" + "=" * 60)
    print("Liquidity Monitor Test")
    print("=" * 60)

    monitor = LiquidityMonitor()
    monitor.update(previous)
    monitor.update(current)

    alerts = monitor.get_alerts()
    print("\nAlerts:")
    for alert in alerts:
        print(f"   {alert}")

