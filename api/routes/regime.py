"""
EIMAS Regime Router
====================

Regime detection result endpoint.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from ..models.responses import RegimeResponse, RegimeChangeDetail

router = APIRouter(tags=["Regime"])

# Global state reference (set by server.py)
_state: Dict[str, Any] = {"results": {}, "last_analysis_id": None}


def set_state(state: dict):
    """Set state reference from server"""
    global _state
    _state = state


def _extract_regime_changes(regime_context) -> List[RegimeChangeDetail]:
    """Extract regime change details from context"""
    changes = []

    if not regime_context or not hasattr(regime_context, 'regime_changes'):
        return changes

    for change in regime_context.regime_changes:
        detail = RegimeChangeDetail(
            is_regime_change=getattr(change, 'is_regime_change', False),
            change_date=getattr(change, 'change_date', None),
            before_regime=getattr(change, 'before_regime', None),
            after_regime=getattr(change, 'after_regime', None),
            analysis_instruction=getattr(change, 'analysis_instruction', ''),
            reason=getattr(change, 'reason', '')
        )
        changes.append(detail)

    return changes


@router.get("/regime", response_model=RegimeResponse)
async def get_regime(
    analysis_id: Optional[str] = Query(
        None,
        description="Analysis ID to retrieve regime for. Uses last analysis if not provided."
    )
):
    """
    Get regime detection results.

    Returns the regime context from a previous analysis,
    including regime type, changes detected, and context adjustments.

    Args:
        analysis_id: Optional analysis ID. Uses last analysis if not provided.

    Returns:
        RegimeResponse with regime detection results
    """
    # Determine which analysis to use
    target_id = analysis_id or _state.get("last_analysis_id")

    if not target_id:
        raise HTTPException(
            status_code=404,
            detail="No analysis found. Run an analysis first."
        )

    if target_id not in _state.get("results", {}):
        raise HTTPException(
            status_code=404,
            detail=f"Analysis {target_id} not found"
        )

    result = _state["results"][target_id]
    regime_context = getattr(result, 'regime_context', None)

    if not regime_context:
        return RegimeResponse(
            regime_type="stable",
            regime_aware=False,
            changes_detected=0,
            context_adjustment={"data_handling": "use_all_data"},
            data_split_date=None,
            confidence=0.0,
            changes=[]
        )

    # Extract data split date
    data_split = None
    if hasattr(regime_context, 'data_split_date') and regime_context.data_split_date:
        data_split = regime_context.data_split_date.isoformat() if hasattr(
            regime_context.data_split_date, 'isoformat'
        ) else str(regime_context.data_split_date)

    # Extract context adjustment
    ctx_adj = {}
    if hasattr(regime_context, 'context_adjustment'):
        ctx_adj = regime_context.context_adjustment if isinstance(
            regime_context.context_adjustment, dict
        ) else {}

    return RegimeResponse(
        regime_type=regime_context.regime_type.value if hasattr(
            regime_context.regime_type, 'value'
        ) else str(regime_context.regime_type),
        regime_aware=getattr(regime_context, 'regime_aware', False),
        changes_detected=len(getattr(regime_context, 'regime_changes', [])),
        context_adjustment=ctx_adj,
        data_split_date=data_split,
        confidence=getattr(regime_context, 'confidence', 0.0),
        changes=_extract_regime_changes(regime_context)
    )
