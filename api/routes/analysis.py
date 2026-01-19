"""
EIMAS Analysis Router
=====================

Analysis endpoints for the EIMAS API.

Active Endpoints:
- GET /latest: Get the most recent integrated analysis result (for dashboard)

Archived Endpoints:
- POST /analyze: Moved to pipeline/archive/ (used full_pipeline.py)
"""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Analysis"])


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


@router.post("/analyze")
async def run_analysis_deprecated():
    """
    [DEPRECATED] Full pipeline analysis endpoint.

    This endpoint has been archived. The full_pipeline.py and related agents
    have been moved to archive directories.

    For analysis, use:
    - CLI: python main_integrated.py --quick
    - Dashboard: GET /latest for cached results

    To restore this endpoint, see:
    - pipeline/archive/full_pipeline.py
    - agents/archive/
    """
    raise HTTPException(
        status_code=410,  # Gone
        detail={
            "message": "This endpoint has been deprecated and archived.",
            "alternatives": [
                "CLI: python main_integrated.py --quick",
                "Dashboard: GET /latest",
            ],
            "restore_info": "See pipeline/archive/ and agents/archive/"
        }
    )
