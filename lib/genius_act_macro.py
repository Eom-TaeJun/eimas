#!/usr/bin/env python3
"""
lib/genius_act_macro.py — Shim (패키지 호환 레이어)
실제 구현은 lib/genius_act/ 패키지로 이동됨.

    from lib.genius_act_macro import GeniusActMacroStrategy  # 기존 경로 (유지)
    from lib.genius_act import GeniusActMacroStrategy        # 신규 경로 (권장)
"""
from lib.genius_act import (
    GeniusActMacroStrategy,
    ExtendedLiquidityModel,
    LiquidityMonitor,
    CryptoRiskEvaluator,
    MultiDimensionalRiskScore,
    StablecoinRiskProfile,
    StablecoinDataCollector,
    LiquidityRegime,
    SignalType,
    StablecoinCollateralType,
    MacroSignal,
    LiquidityIndicators,
    StrategyPosition,
)

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
