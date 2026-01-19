"""
EIMAS Debate Router
===================

[DEPRECATED] This router depends on the archived full_pipeline.

For debate/consensus information, use:
- GET /latest - Returns integrated analysis with debate results

The methodology debate and interpretation endpoints have been archived
along with pipeline/full_pipeline.py and agents/archive/ agents.
"""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Debate"])


@router.get("/debate")
async def get_debate_deprecated():
    """
    [DEPRECATED] Get methodology debate results.

    This endpoint has been deprecated because it depends on the archived
    full_pipeline.py which used TopDownOrchestrator, MethodologyDebateAgent,
    and InterpretationDebateAgent.

    For debate/consensus information, use GET /latest which returns the
    integrated analysis results including:
    - full_mode_position / reference_mode_position
    - modes_agree (whether modes reached consensus)
    - dissent_records (if any)
    - has_strong_dissent

    To restore this functionality, see:
    - pipeline/archive/full_pipeline.py
    - agents/archive/methodology_debate.py
    - agents/archive/interpretation_debate.py
    """
    raise HTTPException(
        status_code=410,  # Gone
        detail={
            "message": "This endpoint has been deprecated.",
            "reason": "Depends on archived full_pipeline.py",
            "alternative": "GET /latest - Returns integrated analysis with debate consensus",
            "fields_in_latest": [
                "full_mode_position",
                "reference_mode_position",
                "modes_agree",
                "dissent_records",
                "has_strong_dissent"
            ],
            "restore_info": "See pipeline/archive/ and agents/archive/"
        }
    )
