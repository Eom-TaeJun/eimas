#!/usr/bin/env python3
"""
Shock Propagation - Enumerations
============================================================

Node layer and causality strength classifications
"""

from enum import Enum


class NodeLayer(Enum):
    """
    노드 레이어 (경제학적 계층)

    충격 전파 순서:
    POLICY → LIQUIDITY → RISK_PREMIUM → ASSET_PRICE
    """
    POLICY = 1        # Fed Funds, ECB Rate
    LIQUIDITY = 2     # RRP, TGA, M2, Stablecoin Supply
    RISK_PREMIUM = 3  # VIX, Credit Spread, HY Spread
    ASSET_PRICE = 4   # SPY, QQQ, BTC, Gold
    UNKNOWN = 5


class CausalityStrength(Enum):
    """인과관계 강도"""
    STRONG = "strong"      # p < 0.01
    MODERATE = "moderate"  # p < 0.05
    WEAK = "weak"          # p < 0.10
    NONE = "none"          # p >= 0.10


