"""
EIMAS API Routes
================

FastAPI routers for each endpoint category.
"""

from .analysis import router as analysis_router
from .regime import router as regime_router
from .debate import router as debate_router
from .health import router as health_router
from .report import router as report_router

__all__ = [
    'analysis_router',
    'regime_router',
    'debate_router',
    'health_router',
    'report_router'
]
