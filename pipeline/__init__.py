"""
EIMAS Pipeline Module
=====================
전체 분석 파이프라인

Modules:
- full_pipeline: 데이터 수집 → 분석 → 해석 → 전략 전체 흐름
"""

from .full_pipeline import (
    # Main Runner
    FullPipelineRunner,

    # Data Classes
    PipelineResult,
    PipelineConfig,
    StageResult,

    # Enums
    PipelineStage,
    PipelineStatus,

    # Mock Data
    MockDataProvider,

    # Convenience Functions
    run_quick_analysis,
    print_result_summary
)

__all__ = [
    # Main Runner
    'FullPipelineRunner',

    # Data Classes
    'PipelineResult',
    'PipelineConfig',
    'StageResult',

    # Enums
    'PipelineStage',
    'PipelineStatus',

    # Mock Data
    'MockDataProvider',

    # Convenience Functions
    'run_quick_analysis',
    'print_result_summary',
]
