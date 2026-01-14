"""
EIMAS API Response Models
=========================

Pydantic models for API response serialization.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class RegimeChangeDetail(BaseModel):
    """Regime change detail"""
    is_regime_change: bool
    change_date: Optional[str] = None
    before_regime: Optional[str] = None
    after_regime: Optional[str] = None
    analysis_instruction: str = ""
    reason: str = ""


class TopDownSummary(BaseModel):
    """Top-Down analysis summary"""
    stance: Optional[str] = None
    confidence: float = 0.0
    recommendation: str = ""


class SchoolOpinion(BaseModel):
    """Economic school opinion"""
    school: str
    interpretation: str
    key_points: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class InterpretationSummary(BaseModel):
    """Interpretation consensus summary"""
    recommended_action: str = ""
    confidence: float = 0.0
    summary: str = ""


class AnalysisResponse(BaseModel):
    """Response body for POST /api/analyze"""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: str = Field(..., description="Pipeline status")
    question: str = Field(..., description="Original research question")

    # Top-Down results
    final_stance: Optional[str] = Field(None, description="Final market stance")
    final_recommendation: str = Field("", description="Final recommendation")
    executive_summary: str = Field("", description="Executive summary")
    confidence: float = Field(0.0, description="Overall confidence")

    # Timing
    total_duration_seconds: float = Field(0.0, description="Total execution time")
    stages_completed: List[str] = Field(default_factory=list, description="Completed stages")

    # Regime context
    regime_context: Optional[Dict[str, Any]] = Field(None, description="Regime detection results")

    # Top-Down summary
    top_down_summary: Optional[TopDownSummary] = Field(None, description="Top-Down analysis summary")

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_id": "analysis_20251227_143022",
                "status": "completed",
                "question": "Fed 금리 영향 분석",
                "final_stance": "NEUTRAL",
                "confidence": 0.75,
                "total_duration_seconds": 12.5,
                "stages_completed": ["data_collection", "top_down_analysis", "regime_check"]
            }
        }


class RegimeResponse(BaseModel):
    """Response body for GET /api/regime"""

    regime_type: str = Field(..., description="Regime type (expansion/contraction/transition/crisis/stable)")
    regime_aware: bool = Field(..., description="Whether regime change was detected")
    changes_detected: int = Field(0, description="Number of regime changes detected")
    context_adjustment: Dict[str, Any] = Field(default_factory=dict, description="Context adjustment instructions")
    data_split_date: Optional[str] = Field(None, description="Date to split data at")
    confidence: float = Field(0.0, description="Detection confidence")
    changes: List[RegimeChangeDetail] = Field(default_factory=list, description="Detected regime changes")

    class Config:
        json_schema_extra = {
            "example": {
                "regime_type": "stable",
                "regime_aware": False,
                "changes_detected": 0,
                "context_adjustment": {"data_handling": "use_all_data"},
                "confidence": 0.8
            }
        }


class DebateResponse(BaseModel):
    """Response body for GET /api/debate"""

    topic: str = Field(..., description="Debate topic")
    methodology_selected: Optional[str] = Field(None, description="Selected methodology")
    interpretation_consensus: Optional[InterpretationSummary] = Field(None, description="Interpretation consensus")
    school_opinions: List[SchoolOpinion] = Field(default_factory=list, description="Economic school opinions")
    consensus_points: List[str] = Field(default_factory=list, description="Points of consensus")
    divergence_points: List[str] = Field(default_factory=list, description="Points of divergence")

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "Fed rate policy impact",
                "methodology_selected": "LASSO",
                "consensus_points": ["Inflation moderating"],
                "divergence_points": ["Policy response timing"]
            }
        }


class HealthResponse(BaseModel):
    """Response body for GET /api/health"""

    status: str = Field(..., description="Server status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    last_analysis_id: Optional[str] = Field(None, description="Last analysis ID")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": "2025-12-27T15:30:00",
                "last_analysis_id": "analysis_20251227_143022"
            }
        }
