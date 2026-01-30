"""
EIMAS Database - 데이터베이스 인터페이스
======================================
Database interfaces for various data stores.

Modules:
    - trading_db: Trading signals and paper trade database
    - event_db: Economic event database
    - unified_store: Unified data storage
    - predictions_db: Prediction history database
"""

from lib.trading_db import TradingDB
from lib.event_db import EventDB
from lib.unified_data_store import UnifiedDataStore
from lib.predictions_db import PredictionsDB

__all__ = [
    'TradingDB',
    'EventDB', 
    'UnifiedDataStore',
    'PredictionsDB',
]
