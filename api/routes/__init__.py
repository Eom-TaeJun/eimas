"""
EIMAS API Routes
================

FastAPI routers for each endpoint category.
"""

from .analysis import router as analysis_router

__all__ = [
    'analysis_router'
]
