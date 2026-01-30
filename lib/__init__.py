"""
EIMAS Library Module
====================
경제 분석을 위한 핵심 라이브러리
Core library for economic analysis.

Submodules:
    - collectors: Data collection modules (FRED, Market, Crypto)
    - analyzers: Market analysis engines (Regime, Liquidity, Microstructure)
    - reports: Report generation modules
    - strategies: Portfolio optimization strategies
    - db: Database interfaces
    - utils: Utility functions

Examples:
    # Direct import (traditional)
    from lib.fred_collector import FREDCollector
    
    # Submodule import (new, organized)
    from lib.collectors import FREDCollector
    from lib.analyzers import RegimeDetector
"""

__version__ = "2.2.0"

# Submodule exports for organized access (with graceful fallback)
try:
    from . import collectors
except ImportError as e:
    collectors = None

try:
    from . import analyzers
except ImportError as e:
    analyzers = None

try:
    from . import reports
except ImportError as e:
    reports = None

try:
    from . import strategies
except ImportError as e:
    strategies = None

try:
    from . import db
except ImportError as e:
    db = None

try:
    from . import utils
except ImportError as e:
    utils = None

# Legacy direct exports (for backward compatibility)
try:
    from .fred_collector import FREDCollector, FRED_SERIES
except ImportError:
    pass

try:
    from .regime_detector import RegimeDetector, MarketRegime
except ImportError:
    pass

try:
    from .critical_path import CriticalPathAggregator
except ImportError:
    pass

try:
    from .final_report_agent import FinalReportAgent
except ImportError:
    pass

try:
    from .data_collector import DataManager
except ImportError:
    pass

__all__ = [
    '__version__',
    # Submodules
    'collectors',
    'analyzers',
    'reports',
    'strategies',
    'db',
    'utils',
    # Legacy exports
    'FREDCollector',
    'RegimeDetector',
    'CriticalPathAggregator',
    'FinalReportAgent',
    'DataManager',
]
