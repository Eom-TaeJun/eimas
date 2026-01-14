"""
EIMAS Health Check Router
=========================

Health check endpoint for monitoring.
"""

from datetime import datetime
from fastapi import APIRouter

from ..models.responses import HealthResponse

router = APIRouter(tags=["Health"])

# Global state reference (set by server.py)
_state = {"last_analysis_id": None}


def set_state(state: dict):
    """Set state reference from server"""
    global _state
    _state = state


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns server status, version, and last analysis ID.
    """
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
        last_analysis_id=_state.get("last_analysis_id")
    )
