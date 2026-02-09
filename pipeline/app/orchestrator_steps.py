"""
Phase execution helpers for the integrated pipeline orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

from pipeline.app.runtime import PhaseRuntimeTracker
from pipeline.phases.phase1_collect import collect_data as phase1_collect_data
from pipeline.phases.phase2_adjustment import (
    analyze_institutional_frameworks as phase2_analyze_institutional_frameworks,
    analyze_sentiment_bubble as phase2_analyze_sentiment_bubble,
    apply_extended_data_adjustment as phase2_apply_extended_data_adjustment,
    run_adaptive_portfolio_phase as phase2_run_adaptive_portfolio,
)
from pipeline.phases.phase2_basic import analyze_basic as phase2_analyze_basic
from pipeline.phases.phase2_enhanced import analyze_enhanced as phase2_analyze_enhanced
from pipeline.phases.phase3_debate import run_debate as phase3_run_debate
from pipeline.phases.phase45_operational import (
    generate_operational_report as phase45_generate_operational_report,
)
from pipeline.phases.phase46_paper_execution import (
    run_paper_execution as phase46_run_paper_execution,
)
from pipeline.phases.phase4_realtime import run_realtime as phase4_run_realtime
from pipeline.phases.phase5_storage import save_results as phase5_save_results
from pipeline.phases.phase6_portfolio import (
    run_backtest as phase6_run_backtest,
    run_performance_attribution as phase6_run_performance_attribution,
    run_stress_test as phase6_run_stress_test,
    run_tactical_allocation as phase6_run_tactical_allocation,
)
from pipeline.phases.phase7_report import (
    generate_report as phase7_generate_report,
    validate_report as phase7_validate_report,
)
from pipeline.phases.phase8_validation import (
    run_ai_validation_phase as phase8_run_ai_validation_phase,
    run_quick_validation as phase8_run_quick_validation,
)
from pipeline.phases.phase9_artifacts import export_artifacts as phase9_export_artifacts
from pipeline.risk_utils import derive_risk_level
from pipeline.schemas import EIMASResult


def _run_tactical_allocation(result: EIMASResult) -> Any:
    """Hook for phase2 enhanced tactical allocation injection."""
    return phase6_run_tactical_allocation(result)


def _run_phase2_enhanced(
    result: EIMASResult,
    market_data: Dict[str, Any],
    quick_mode: bool,
) -> Any:
    """Run phase2 enhanced with tactical allocation hook."""
    return phase2_analyze_enhanced(
        result,
        market_data,
        quick_mode,
        run_tactical_allocation_fn=_run_tactical_allocation,
    )


async def run_pipeline_phases(
    *,
    runtime: PhaseRuntimeTracker,
    result: EIMASResult,
    output_path: Path,
    quick_mode: bool,
    enable_realtime: bool,
    realtime_duration: int,
    enable_backtest: bool,
    enable_attribution: bool,
    enable_stress_test: bool,
    should_generate_report: bool,
    full_mode: bool,
    quick_validation_mode: str | None,
    enable_paper_auto: bool,
    paper_account: str,
    paper_capital: float,
    paper_poll_only: bool,
    paper_backtest: bool,
    paper_enforce_approval: bool,
) -> Tuple[str | None, Dict[str, Any]]:
    """
    Execute phase 1~9 flow.

    Returns:
        (output_file, market_data)
    """
    # Phase 1-2: Data & Analysis
    market_data = await runtime.run_async(
        "phase1_collect_data",
        phase1_collect_data,
        result,
        quick_mode,
    )
    events, regime_res = runtime.run_sync(
        "phase2_basic_analyze",
        phase2_analyze_basic,
        result,
        market_data,
    )
    runtime.run_sync(
        "phase2_enhanced_analyze",
        _run_phase2_enhanced,
        result,
        market_data,
        quick_mode,
    )
    runtime.run_sync(
        "phase2_sentiment_bubble",
        phase2_analyze_sentiment_bubble,
        result,
        market_data,
        quick_mode,
    )
    runtime.run_sync(
        "phase2_extended_adjustment",
        phase2_apply_extended_data_adjustment,
        result,
    )
    runtime.run_sync(
        "phase2_institutional_frameworks",
        phase2_analyze_institutional_frameworks,
        result,
        market_data,
        quick_mode,
    )
    runtime.run_sync(
        "phase2_adaptive_portfolio",
        phase2_run_adaptive_portfolio,
        result,
        regime_res,
        quick_mode,
    )

    # Phase 3-4: Debate & Realtime
    await runtime.run_async(
        "phase3_debate",
        phase3_run_debate,
        result,
        market_data,
    )
    await runtime.run_async(
        "phase4_realtime",
        phase4_run_realtime,
        result,
        enable_realtime,
        realtime_duration,
    )
    result.risk_level = derive_risk_level(result.risk_score)

    # Phase 4.5: Operational & Paper Execution
    runtime.run_sync(
        "phase45_operational_report",
        phase45_generate_operational_report,
        result,
    )
    runtime.run_sync(
        "phase46_paper_execution",
        phase46_run_paper_execution,
        result,
        enable=enable_paper_auto,
        account_name=paper_account,
        initial_capital=paper_capital,
        poll_only=paper_poll_only,
        run_backtest=paper_backtest,
        enforce_human_approval=paper_enforce_approval,
    )

    # Phase 5: Storage
    output_file = runtime.run_sync(
        "phase5_storage",
        phase5_save_results,
        result,
        events,
        output_path,
    )

    # Phase 6: Portfolio Modules
    runtime.run_sync(
        "phase6_backtest",
        phase6_run_backtest,
        result,
        market_data,
        enable_backtest,
    )
    runtime.run_sync(
        "phase6_performance_attribution",
        phase6_run_performance_attribution,
        result,
        enable_attribution,
    )
    runtime.run_sync(
        "phase6_stress_test",
        phase6_run_stress_test,
        result,
        enable_stress_test,
    )

    # Phase 7: AI Report Generation
    report_content = await runtime.run_async(
        "phase7_generate_report",
        phase7_generate_report,
        result,
        market_data,
        should_generate_report,
        output_path,
        output_file=output_file,
    )

    # Phase 8: Validation
    await runtime.run_async(
        "phase7_validate_report",
        phase7_validate_report,
        result,
        report_content,
        should_generate_report,
        output_path,
        output_file=output_file,
    )
    runtime.run_sync(
        "phase8_ai_validation",
        phase8_run_ai_validation_phase,
        result,
        full_mode,
        output_path,
        output_file=output_file,
    )

    # Phase 8.5 + 9
    runtime.run_sync(
        "phase85_quick_validation",
        phase8_run_quick_validation,
        result,
        market_data,
        output_file,
        quick_validation_mode,
    )
    artifact_export = runtime.run_sync(
        "phase9_artifact_export",
        phase9_export_artifacts,
        output_file,
        output_path,
        full_mode,
    )
    if artifact_export:
        result.audit_metadata["artifact_export"] = artifact_export

    return output_file, market_data

