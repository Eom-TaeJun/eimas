"""
EIMAS Collectors - 데이터 수집 모듈
====================================
Data collection modules for various sources.

Modules:
    - base: Base interface for all collectors
    - market: MarketDataCollector, DataManager
    - fred: FREDCollector for Federal Reserve data
    - crypto: CryptoCollector for cryptocurrency data
    - extended: ExtendedDataCollector for additional sources
"""

from lib.data_collector import (
    DataManager,
    MarketDataCollector,
    FREDDataCollector,
    CryptoDataCollector,
    UnifiedDataCollector
)
from lib.fred_collector import FREDCollector
from lib.crypto_collector import CryptoCollector
from lib.extended_data_sources import ExtendedDataCollector
from lib.intraday_collector import IntradayCollector
from lib.market_indicators import MarketIndicatorsCollector

__all__ = [
    # Core collectors
    'DataManager',
    'MarketDataCollector',
    'FREDDataCollector',
    'CryptoDataCollector',
    'UnifiedDataCollector',
    # Specialized collectors
    'FREDCollector',
    'CryptoCollector',
    'ExtendedDataCollector',
    'IntradayCollector',
    'MarketIndicatorsCollector',
]
