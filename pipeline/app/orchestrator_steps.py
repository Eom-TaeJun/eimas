"""
Phase execution helpers for the integrated pipeline orchestrator.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

from pipeline.app.profiles import PipelineProfile
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


def _mark_profile_skip(result: EIMASResult, phase_name: str, reason: str) -> None:
    """Record profile-driven phase skip metadata in audit trail."""
    if not isinstance(result.audit_metadata, dict):
        result.audit_metadata = {}
    skips = result.audit_metadata.setdefault("profile_skips", [])
    if isinstance(skips, list):
        skips.append(
            {
                "phase": phase_name,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
            }
        )


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
    debate_full_lookback: int,
    debate_ref_lookback: int,
    debate_skip_reference: bool,
    pipeline_profile: PipelineProfile,
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
    if pipeline_profile.run_sentiment_bubble:
        runtime.run_sync(
            "phase2_sentiment_bubble",
            phase2_analyze_sentiment_bubble,
            result,
            market_data,
            quick_mode,
            skip_bubble=pipeline_profile.skip_bubble_analysis,
        )
    else:
        _mark_profile_skip(
            result,
            "phase2_sentiment_bubble",
            f"profile:{pipeline_profile.name}",
        )
        result.sentiment_analysis = {
            "skipped": True,
            "reason": f"profile:{pipeline_profile.name}",
        }

    runtime.run_sync(
        "phase2_extended_adjustment",
        phase2_apply_extended_data_adjustment,
        result,
    )
    if pipeline_profile.run_institutional_frameworks:
        runtime.run_sync(
            "phase2_institutional_frameworks",
            phase2_analyze_institutional_frameworks,
            result,
            market_data,
            quick_mode,
        )
    else:
        _mark_profile_skip(
            result,
            "phase2_institutional_frameworks",
            f"profile:{pipeline_profile.name}",
        )
        result.institutional_analysis = {
            "skipped": True,
            "reason": f"profile:{pipeline_profile.name}",
        }

    if pipeline_profile.run_adaptive_portfolio:
        runtime.run_sync(
            "phase2_adaptive_portfolio",
            phase2_run_adaptive_portfolio,
            result,
            regime_res,
            quick_mode,
        )
    else:
        _mark_profile_skip(
            result,
            "phase2_adaptive_portfolio",
            f"profile:{pipeline_profile.name}",
        )
        result.adaptive_portfolios = {
            "skipped": True,
            "reason": f"profile:{pipeline_profile.name}",
        }

    # Phase 3-4: Debate & Realtime
    if pipeline_profile.run_debate:
        await runtime.run_async(
            "phase3_debate",
            phase3_run_debate,
            result,
            market_data,
            quick_mode,
            debate_full_lookback,
            debate_ref_lookback,
            debate_skip_reference,
        )
    else:
        _mark_profile_skip(
            result,
            "phase3_debate",
            f"profile:{pipeline_profile.name}",
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
    effective_backtest = enable_backtest and pipeline_profile.run_backtest
    effective_attribution = enable_attribution and pipeline_profile.run_attribution
    effective_stress_test = enable_stress_test and pipeline_profile.run_stress_test

    if enable_backtest and not pipeline_profile.run_backtest:
        _mark_profile_skip(
            result,
            "phase6_backtest",
            f"profile:{pipeline_profile.name}",
        )
    if enable_attribution and not pipeline_profile.run_attribution:
        _mark_profile_skip(
            result,
            "phase6_performance_attribution",
            f"profile:{pipeline_profile.name}",
        )
    if enable_stress_test and not pipeline_profile.run_stress_test:
        _mark_profile_skip(
            result,
            "phase6_stress_test",
            f"profile:{pipeline_profile.name}",
        )

    runtime.run_sync(
        "phase6_backtest",
        phase6_run_backtest,
        result,
        market_data,
        effective_backtest,
    )
    runtime.run_sync(
        "phase6_performance_attribution",
        phase6_run_performance_attribution,
        result,
        effective_attribution,
    )
    runtime.run_sync(
        "phase6_stress_test",
        phase6_run_stress_test,
        result,
        effective_stress_test,
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
    if pipeline_profile.run_phase8_ai_validation:
        runtime.run_sync(
            "phase8_ai_validation",
            phase8_run_ai_validation_phase,
            result,
            full_mode,
            output_path,
            output_file=output_file,
        )
    else:
        _mark_profile_skip(
            result,
            "phase8_ai_validation",
            f"profile:{pipeline_profile.name}",
        )
        result.validation_loop_result = {
            "skipped": True,
            "reason": f"profile:{pipeline_profile.name}",
            "timestamp": datetime.now().isoformat(),
        }

    # Phase 8.5 + 9
    effective_quick_validation_mode = quick_validation_mode
    if quick_validation_mode and not pipeline_profile.run_phase85_quick_validation:
        effective_quick_validation_mode = None
        _mark_profile_skip(
            result,
            "phase85_quick_validation",
            f"profile:{pipeline_profile.name}",
        )
        result.quick_validation = {
            "skipped": True,
            "reason": f"profile:{pipeline_profile.name}",
            "timestamp": datetime.now().isoformat(),
        }

    runtime.run_sync(
        "phase85_quick_validation",
        phase8_run_quick_validation,
        result,
        market_data,
        output_file,
        effective_quick_validation_mode,
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
