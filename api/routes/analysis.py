"""
EIMAS Analysis Router
=====================

Main analysis endpoint for running the full pipeline.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from pipeline.full_pipeline import (
    FullPipelineRunner,
    PipelineConfig,
    PipelineResult,
    PipelineStage
)
from agents.top_down_orchestrator import AnalysisLevel
from agents.methodology_debate import ResearchGoal

from ..models.requests import AnalysisRequest
from ..models.responses import AnalysisResponse, TopDownSummary

router = APIRouter(tags=["Analysis"])

# Global state reference (set by server.py)
_state: Dict[str, Any] = {"results": {}, "last_analysis_id": None}


def set_state(state: dict):
    """Set state reference from server"""
    global _state
    _state = state


def _get_level_from_string(level_str: str) -> AnalysisLevel:
    """Convert string to AnalysisLevel enum"""
    level_map = {
        "geopolitics": AnalysisLevel.GEOPOLITICS,
        "monetary": AnalysisLevel.MONETARY,
        "asset_class": AnalysisLevel.ASSET_CLASS,
        "sector": AnalysisLevel.SECTOR,
        "individual": AnalysisLevel.INDIVIDUAL
    }
    return level_map.get(level_str.lower(), AnalysisLevel.SECTOR)


def _get_goal_from_string(goal_str: str) -> ResearchGoal:
    """Convert string to ResearchGoal enum"""
    goal_map = {
        "variable_selection": ResearchGoal.VARIABLE_SELECTION,
        "forecasting": ResearchGoal.FORECASTING,
        "causal_inference": ResearchGoal.CAUSAL_INFERENCE,
        "volatility": ResearchGoal.VOLATILITY_MODELING,
        "dynamic": ResearchGoal.DYNAMIC_RELATIONSHIP,
        "interpretation": ResearchGoal.INTERPRETATION
    }
    return goal_map.get(goal_str.lower(), ResearchGoal.VARIABLE_SELECTION)


def _serialize_result(result: PipelineResult) -> Dict[str, Any]:
    """Serialize PipelineResult to JSON-compatible dict"""

    def convert(obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return {k: convert(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            return obj

    return convert(result)


@router.post("/analyze", response_model=AnalysisResponse)
async def run_analysis(request: AnalysisRequest):
    """
    Run full pipeline analysis.

    Executes the complete EIMAS pipeline:
    1. Data Collection
    2. Top-Down Analysis
    3. Regime Detection (Stage 2.5)
    4. Methodology Selection
    5. Core Analysis
    6. Interpretation
    7. Strategy Generation
    8. Synthesis

    Args:
        request: Analysis request with question, options, and data

    Returns:
        AnalysisResponse with results and status
    """
    global _state

    # Generate analysis ID
    analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _state["last_analysis_id"] = analysis_id

    try:
        # Parse skip stages
        skip_stages = []
        for stage_str in request.skip_stages:
            stage_name = stage_str.upper()
            if stage_name in PipelineStage.__members__:
                skip_stages.append(PipelineStage[stage_name])

        # Configure pipeline
        config = PipelineConfig(
            stop_at_level=_get_level_from_string(request.stop_at_level),
            research_goal=_get_goal_from_string(request.research_goal),
            skip_stages=skip_stages,
            verbose=False  # Disable logging for API
        )

        # Run pipeline
        runner = FullPipelineRunner(verbose=False, use_mock=request.use_mock)
        result = await runner.run(
            research_question=request.question,
            data=request.data,
            config=config
        )

        # Cache result
        _state["results"][analysis_id] = result

        # Build response
        regime_ctx = None
        if result.regime_context:
            regime_ctx = result.regime_context.to_dict()

        top_down_summary = None
        if result.top_down:
            top_down_summary = TopDownSummary(
                stance=result.top_down.final_stance.value,
                confidence=result.top_down.total_confidence,
                recommendation=result.top_down.final_recommendation
            )

        return AnalysisResponse(
            analysis_id=analysis_id,
            status=result.status.value,
            question=request.question,
            final_stance=result.top_down.final_stance.value if result.top_down else None,
            final_recommendation=result.final_recommendation,
            executive_summary=result.executive_summary,
            confidence=result.confidence,
            total_duration_seconds=result.total_duration_seconds,
            stages_completed=[s.value for s in result.stages.keys()],
            regime_context=regime_ctx,
            top_down_summary=top_down_summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{analysis_id}")
async def get_analysis(analysis_id: str):
    """
    Get analysis result by ID.

    Args:
        analysis_id: Analysis identifier

    Returns:
        Serialized PipelineResult
    """
    if analysis_id not in _state.get("results", {}):
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")

    result = _state["results"][analysis_id]
    return _serialize_result(result)


@router.get("/latest")
async def get_latest_result():
    """
    Get the latest integrated analysis result from outputs directory.

    Returns the most recent integrated_YYYYMMDD_HHMMSS.json file.
    This endpoint is used by the real-time dashboard for auto-polling.

    Returns:
        Latest analysis result as JSON
    """
    import json
    import glob
    import os
    from pathlib import Path

    # Find outputs directory
    outputs_dir = Path(__file__).parent.parent.parent / "outputs"

    if not outputs_dir.exists():
        raise HTTPException(status_code=404, detail="Outputs directory not found")

    # Get all integrated JSON files
    pattern = str(outputs_dir / "integrated_*.json")
    files = glob.glob(pattern)

    if not files:
        raise HTTPException(status_code=404, detail="No integrated results found")

    # Get the most recent file
    latest_file = max(files, key=os.path.getmtime)

    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)

        # Add metadata about the file
        data['_meta'] = {
            'source_file': os.path.basename(latest_file),
            'file_modified': datetime.fromtimestamp(os.path.getmtime(latest_file)).isoformat()
        }

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read result: {str(e)}")
