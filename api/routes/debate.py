"""
EIMAS Debate Router
===================

Methodology debate and interpretation results endpoint.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from ..models.responses import DebateResponse, SchoolOpinion, InterpretationSummary

router = APIRouter(tags=["Debate"])

# Global state reference (set by server.py)
_state: Dict[str, Any] = {"results": {}, "last_analysis_id": None}


def set_state(state: dict):
    """Set state reference from server"""
    global _state
    _state = state


def _extract_school_opinions(interpretation_result) -> List[SchoolOpinion]:
    """Extract school opinions from interpretation result"""
    opinions = []

    if not interpretation_result:
        return opinions

    # Check for school_opinions attribute
    if hasattr(interpretation_result, 'school_opinions'):
        for opinion in interpretation_result.school_opinions:
            school_op = SchoolOpinion(
                school=getattr(opinion, 'school', 'Unknown'),
                interpretation=getattr(opinion, 'interpretation', ''),
                key_points=getattr(opinion, 'key_points', []),
                confidence=getattr(opinion, 'confidence', 0.0)
            )
            opinions.append(school_op)

    # Also check for interpretations dict (alternate structure)
    elif hasattr(interpretation_result, 'interpretations'):
        interps = interpretation_result.interpretations
        if isinstance(interps, dict):
            for school, interp in interps.items():
                if isinstance(interp, dict):
                    school_op = SchoolOpinion(
                        school=school,
                        interpretation=interp.get('interpretation', ''),
                        key_points=interp.get('key_points', []),
                        confidence=interp.get('confidence', 0.0)
                    )
                else:
                    school_op = SchoolOpinion(
                        school=school,
                        interpretation=str(interp),
                        key_points=[],
                        confidence=0.0
                    )
                opinions.append(school_op)

    return opinions


def _extract_interpretation_summary(interpretation_result) -> Optional[InterpretationSummary]:
    """Extract consensus summary from interpretation result"""
    if not interpretation_result:
        return None

    # Check for consensus attribute
    if hasattr(interpretation_result, 'consensus'):
        consensus = interpretation_result.consensus
        return InterpretationSummary(
            recommended_action=getattr(consensus, 'recommended_action', ''),
            confidence=getattr(consensus, 'confidence', 0.0),
            summary=getattr(consensus, 'summary', '')
        )

    # Check for synthesis_summary
    if hasattr(interpretation_result, 'synthesis_summary'):
        return InterpretationSummary(
            recommended_action='',
            confidence=getattr(interpretation_result, 'confidence', 0.0),
            summary=interpretation_result.synthesis_summary
        )

    return None


@router.get("/debate", response_model=DebateResponse)
async def get_debate(
    analysis_id: Optional[str] = Query(
        None,
        description="Analysis ID to retrieve debate for. Uses last analysis if not provided."
    )
):
    """
    Get methodology debate and interpretation results.

    Returns the methodology selection debate results and
    economic school interpretation consensus.

    Args:
        analysis_id: Optional analysis ID. Uses last analysis if not provided.

    Returns:
        DebateResponse with debate and interpretation results
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

    # Extract methodology info
    methodology = getattr(result, 'methodology', None)
    methodology_selected = None
    consensus_points = []
    divergence_points = []

    if methodology:
        if hasattr(methodology, 'selected_methodology'):
            sel = methodology.selected_methodology
            methodology_selected = sel.value if hasattr(sel, 'value') else str(sel)
        elif hasattr(methodology, 'methodology'):
            sel = methodology.methodology
            methodology_selected = sel.value if hasattr(sel, 'value') else str(sel)

        if hasattr(methodology, 'consensus_points'):
            consensus_points = methodology.consensus_points or []

        if hasattr(methodology, 'divergence_points'):
            divergence_points = methodology.divergence_points or []

    # Extract interpretation info
    interpretation = None
    school_opinions = []

    # Check stages for interpretation result
    from pipeline.full_pipeline import PipelineStage

    stages = getattr(result, 'stages', {})
    if PipelineStage.INTERPRETATION in stages:
        interp_stage = stages[PipelineStage.INTERPRETATION]
        interpretation = getattr(interp_stage, 'result', None)
        school_opinions = _extract_school_opinions(interpretation)

    # Build topic from question
    topic = getattr(result, 'question', '') or "Economic Analysis"

    return DebateResponse(
        topic=topic,
        methodology_selected=methodology_selected,
        interpretation_consensus=_extract_interpretation_summary(interpretation),
        school_opinions=school_opinions,
        consensus_points=consensus_points,
        divergence_points=divergence_points
    )
