#!/usr/bin/env python3
"""
Genius Act Package
============================================================

Fed liquidity-driven macro strategy with stablecoin risk assessment

Public API:
    - GeniusActMacroStrategy: Main strategy class
    - ExtendedLiquidityModel: Liquidity calculation
    - CryptoRiskEvaluator: Crypto and stablecoin risk
    - LiquidityRegime, SignalType: Enums

Economic Foundation:
    - Genius Act: Fed liquidity framework (BS - RRP - TGA)
    - Stablecoin risk: Multi-dimensional risk scoring
    - Expanded liquidity: M = B + S·B* (stablecoin contribution)

Usage:
    from lib.genius_act import GeniusActMacroStrategy

    strategy = GeniusActMacroStrategy()
    result = strategy.analyze(fred_data, market_data)
"""

from .strategy import GeniusActMacroStrategy
from .liquidity import ExtendedLiquidityModel, LiquidityMonitor
from .crypto_risk import CryptoRiskEvaluator
from .stablecoin_risk import MultiDimensionalRiskScore, StablecoinRiskProfile
from .data_collector import StablecoinDataCollector
from .enums import LiquidityRegime, SignalType, StablecoinCollateralType
from .schemas import MacroSignal, LiquidityIndicators, StrategyPosition

__all__ = [
    "GeniusActMacroStrategy",
    "ExtendedLiquidityModel",
    "LiquidityMonitor",
    "CryptoRiskEvaluator",
    "MultiDimensionalRiskScore",
    "StablecoinRiskProfile",
    "StablecoinDataCollector",
    "LiquidityRegime",
    "SignalType",
    "StablecoinCollateralType",
    "MacroSignal",
    "LiquidityIndicators",
    "StrategyPosition",
]
