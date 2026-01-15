"""
EIMAS Library Module
====================
경제 분석을 위한 핵심 라이브러리

Note: Individual modules should be imported directly.
Example: from lib.fred_collector import FREDCollector
"""

# Core exports only - most modules are imported directly in main.py
__version__ = "2.1.2"

# Optional: Add commonly used modules here with try-except for safety
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

__all__ = [
    '__version__',
]
