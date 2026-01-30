"""
EIMAS FastAPI Server
====================

Main server setup with CORS, routers, and middleware.
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

from typing import Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import analysis_router, regime_router, debate_router, health_router, report_router
from .routes.analysis import set_state as set_analysis_state
from .routes.regime import set_state as set_regime_state
from .routes.debate import set_state as set_debate_state
from .routes.health import set_state as set_health_state
from .routes.report import set_state as set_report_state

# Global state for caching analysis results
_global_state: Dict[str, Any] = {
    "results": {},
    "last_analysis_id": None
}


def _init_state():
    """Initialize shared state across routers (called at import time)"""
    set_analysis_state(_global_state)
    set_regime_state(_global_state)
    set_debate_state(_global_state)
    set_health_state(_global_state)
    set_report_state(_global_state)


# Initialize state immediately at import
_init_state()

# Create FastAPI app
app = FastAPI(
    title="EIMAS API",
    description="""
Economic Intelligence Multi-Agent System API

## Features
- **Full Pipeline Analysis**: Run complete EIMAS analysis pipeline
- **Regime Detection**: Stage 2.5 regime change detection
- **Methodology Debate**: Multi-agent debate for methodology selection
- **Economic Interpretation**: School-based economic interpretation

## Pipeline Stages
1. Data Collection
2. Top-Down Analysis
3. **Regime Check (Stage 2.5)** - NEW
4. Methodology Selection
5. Core Analysis
6. Interpretation
7. Strategy Generation
8. Synthesis
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Re-initialize shared state on startup (ensures consistency)"""
    _init_state()
    print("EIMAS API Server started - State initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("EIMAS API Server shutting down")


# Register routers with /api prefix
app.include_router(health_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(regime_router, prefix="/api")
app.include_router(debate_router, prefix="/api")
app.include_router(report_router, prefix="/api")


from fastapi.staticfiles import StaticFiles
import os

# Create reports directory if it doesn't exist
os.makedirs("outputs/reports", exist_ok=True)

# Mount reports directory
app.mount("/reports", StaticFiles(directory="outputs/reports"), name="reports")

@app.get("/api/reports/latest")
async def get_latest_report_url():
    """Get URL for the latest HTML report"""
    import glob
    import os
    
    # Find all HTML reports
    pattern = "outputs/reports/*.html"
    files = glob.glob(pattern)
    
    if not files:
        return {"url": None}
        
    # Get newest file
    latest_file = max(files, key=os.path.getmtime)
    filename = os.path.basename(latest_file)
    
    return {
        "url": f"/reports/{filename}",
        "filename": filename,
        "timestamp": os.path.getmtime(latest_file)
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "EIMAS API",
        "version": "2.0.0",
        "description": "Economic Intelligence Multi-Agent System",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "analyze": "POST /api/analyze",
            "regime": "GET /api/regime",
            "debate": "GET /api/debate",
            "health": "GET /api/health"
        }
    }


# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
