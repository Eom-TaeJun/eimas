#!/usr/bin/env python3
"""
Event Framework - Enumerations
============================================================

Event type, asset class, importance, and timing classifications
"""

from enum import Enum


class EventType(Enum):
    """이벤트 유형"""
    # 예정된 이벤트 (Calendar)
    FOMC = "fomc"
    CPI = "cpi"
    PPI = "ppi"
    NFP = "nfp"  # Non-Farm Payrolls
    GDP = "gdp"
    PCE = "pce"
    ISM = "ism"
    EARNINGS = "earnings"
    DIVIDEND = "dividend"

    # 시장 감지 이벤트 (Quantitative)
    VOLUME_SPIKE = "volume_spike"
    PRICE_SHOCK = "price_shock"
    VOLATILITY_SURGE = "volatility_surge"
    SPREAD_WIDENING = "spread_widening"
    FLOW_REVERSAL = "flow_reversal"
    OPTIONS_UNUSUAL = "options_unusual"

    # 유동성 이벤트 (Liquidity) - Alpha 핵심
    RRP_SURGE = "rrp_surge"           # RRP 급등 (유동성 흡수)
    RRP_DRAIN = "rrp_drain"           # RRP 급감 (유동성 방출) - 불리시
    TGA_BUILDUP = "tga_buildup"       # TGA 증가 (유동성 흡수) - 베어리시
    TGA_DRAWDOWN = "tga_drawdown"     # TGA 감소 (유동성 방출) - 불리시
    QT_ACCELERATION = "qt_acceleration"  # Fed 자산 축소 가속
    LIQUIDITY_STRESS = "liquidity_stress"  # Net Liquidity 급감
    LIQUIDITY_INJECTION = "liquidity_injection"  # 유동성 주입

    # 외부 감지 이벤트 (Qualitative)
    NEWS_SURGE = "news_surge"
    RATING_CHANGE = "rating_change"
    ANALYST_REVISION = "analyst_revision"
    INSIDER_ACTIVITY = "insider_activity"
    REGULATORY = "regulatory"
    GEOPOLITICAL = "geopolitical"


class AssetClass(Enum):
    """자산군"""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    INDEX = "index"


class EventImportance(Enum):
    """이벤트 중요도"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class EventTiming(Enum):
    """이벤트 타이밍"""
    SCHEDULED = "scheduled"      # 예정됨
    DETECTED = "detected"        # 감지됨 (실시간)
    HISTORICAL = "historical"    # 과거


# ============================================================================
# Data Classes
# ============================================================================

