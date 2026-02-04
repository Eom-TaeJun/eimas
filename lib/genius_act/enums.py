#!/usr/bin/env python3
"""
Genius Act - Enumerations
============================================================

Liquidity regime and signal type enums
"""

from enum import Enum


class LiquidityRegime(Enum):
    """유동성 레짐"""
    EXPANSION = "EXPANSION"      # 유동성 확장
    CONTRACTION = "CONTRACTION"  # 유동성 수축
    NEUTRAL = "NEUTRAL"          # 중립


class SignalType(Enum):
    """시그널 타입"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class StablecoinCollateralType(Enum):
    """스테이블코인 담보 유형 (리스크 순서)"""
    TREASURY_CASH = "TREASURY_CASH"        # US Treasury + Cash (USDC)
    MIXED_RESERVE = "MIXED_RESERVE"        # Mixed reserves (USDT)
    CRYPTO_BACKED = "CRYPTO_BACKED"        # Crypto collateral (DAI)
    DERIVATIVE_HEDGE = "DERIVATIVE_HEDGE"  # Derivative hedging (USDe)
    ALGORITHMIC = "ALGORITHMIC"            # Algorithmic (high risk)
