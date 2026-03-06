"""
EIMAS Database - 데이터베이스 인터페이스
======================================
Database interfaces for various data stores.

Modules:
    - trading_db: Trading signals and paper trade database
    - event_db: Economic event database
    - ra_sql_store: RA analysis SQL storage (PostgreSQL)
"""

from .trading_db import TradingDB
from .event_db import EventDatabase as EventDB
from .ra_sql_store import (  # noqa: F401
    RAResearchSQLStore,
    save_ra_commentary_audit_log,
    save_backtest_metrics_to_sql,
    ingest_company_ra_analysis_to_sql,
)

__all__ = [
    'TradingDB',
    'EventDB',
    'RAResearchSQLStore',
    'save_ra_commentary_audit_log',
    'save_backtest_metrics_to_sql',
    'ingest_company_ra_analysis_to_sql',
]
