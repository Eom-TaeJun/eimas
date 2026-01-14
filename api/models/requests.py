"""
EIMAS API Request Models
========================

Pydantic models for API request validation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request body for POST /api/analyze"""

    question: str = Field(
        ...,
        description="Research question to analyze",
        min_length=1,
        max_length=1000,
        examples=["Fed 금리 정책이 2025년 시장에 미치는 영향은?"]
    )

    data: Optional[Dict[str, Any]] = Field(
        None,
        description="Market data (optional, uses mock data if not provided)"
    )

    use_mock: bool = Field(
        False,
        description="Use mock mode (no API calls). Default: False (uses real AI APIs)"
    )

    stop_at_level: str = Field(
        "sector",
        description="Top-down analysis depth level",
        pattern="^(geopolitics|monetary|asset_class|sector|individual)$"
    )

    research_goal: str = Field(
        "variable_selection",
        description="Research goal for methodology selection",
        pattern="^(variable_selection|forecasting|causal_inference|volatility|dynamic|interpretation)$"
    )

    skip_stages: List[str] = Field(
        default_factory=list,
        description="Pipeline stages to skip",
        examples=[["regime_check", "interpretation"]]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Fed 금리 인상이 주식 시장에 미치는 영향 분석",
                "use_mock": False,
                "stop_at_level": "sector",
                "research_goal": "variable_selection",
                "skip_stages": []
            }
        }
