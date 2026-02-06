"""
EIMAS Pipeline - Phases Module

Phase orchestration modules for main.py separation.

Modules:
    - common: Shared utilities
    - phase1_collect: Data collection
    - phase2_basic: Basic analysis
    - phase2_enhanced: Enhanced analysis
    - phase2_adjustment: Risk adjustment
    - phase3_debate: AI debate
    - phase4_realtime: Realtime streaming
    - phase45_operational: Operational reporting
    - phase5_storage: Result storage
    - phase6_portfolio: Portfolio theory (backtest/attribution/stress)
    - phase7_report: AI report generation
    - phase8_validation: Multi-LLM & Quick validation
"""

from .phase1_collect import collect_data
from .phase2_basic import analyze_basic
from .phase2_enhanced import analyze_enhanced
from .phase2_adjustment import (
    analyze_sentiment_bubble,
    apply_extended_data_adjustment,
    analyze_institutional_frameworks,
    run_adaptive_portfolio_phase,
)
from .phase3_debate import run_debate
from .phase4_realtime import run_realtime
from .phase45_operational import generate_operational_report
from .phase5_storage import save_results
from .phase6_portfolio import (
    run_backtest,
    run_performance_attribution,
    run_tactical_allocation,
    run_stress_test,
)
from .phase7_report import generate_report, validate_report
from .phase8_validation import run_ai_validation_phase, run_quick_validation

__all__ = [
    "collect_data",
    "analyze_basic",
    "analyze_enhanced",
    "analyze_sentiment_bubble",
    "apply_extended_data_adjustment",
    "analyze_institutional_frameworks",
    "run_adaptive_portfolio_phase",
    "run_debate",
    "run_realtime",
    "generate_operational_report",
    "save_results",
    "run_backtest",
    "run_performance_attribution",
    "run_tactical_allocation",
    "run_stress_test",
    "generate_report",
    "validate_report",
    "run_ai_validation_phase",
    "run_quick_validation",
]

__version__ = "1.2.0"
__stage__ = "M2-expanded"
