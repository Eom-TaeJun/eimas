#!/usr/bin/env python3
"""
EIMAS - Economic Intelligence Multi-Agent System
=================================================
통합 실행 파이프라인 (Unified Main Entry Point)
Unified execution pipeline for comprehensive market analysis.

================================================================================
ARCHITECTURE OVERVIEW | 아키텍처 개요
================================================================================

This is the main entry point that orchestrates all EIMAS components:
    - Data Collection: Multi-source data gathering (FRED, Market, Crypto)
    - Analysis Engine: Quantitative analysis (Regime, Risk, Microstructure)
    - AI Debate: Multi-agent debate for consensus building
    - Report Generation: AI-powered narrative reports
    - Validation: Fact-checking and whitening

================================================================================
PIPELINE FLOW | 파이프라인 흐름
================================================================================

    [CLI] main()
           │
           ▼
    run_integrated_pipeline()
           │
           ├─► [Phase 1] _collect_data()        # 데이터 수집 (FRED, Market, Crypto)
           │                                    # Data collection from multiple sources
           │
           ├─► [Phase 2] _analyze_basic()       # 기본 분석 (Regime, Events, Risk)
           │        └─► _analyze_enhanced()     # 고급 분석 (HFT, GARCH, DTW, etc.)
           │        └─► _analyze_sentiment_bubble()  # 센티먼트 & 버블 분석
           │
           ├─► [Phase 3] _run_debate()          # AI 에이전트 토론 (Multi-LLM)
           │                                    # Dual-mode debate with consensus
           │
           ├─► [Phase 4] _run_realtime()        # 실시간 스트리밍 (Optional)
           │                                    # VPIN/OFI stream analysis
           │
           ├─► [Phase 5] _save_results()        # 결과 저장 (unified JSON)
           │                                    # Save to outputs/eimas_*.json
           │
           ├─► [Phase 6] _generate_report()     # AI 리포트 생성 (Optional)
           │                                    # LLM-powered narrative report
           │
           ├─► [Phase 7] _validate_report()     # Whitening & Fact Check
           │                                    # Data quality validation
           │
           └─► [Phase 8] _run_ai_validation()   # Multi-LLM 검증 (--full only)
                                                # Cross-LLM consensus check

================================================================================
USAGE | 사용법
================================================================================

    python main.py              # 기본 분석 (버블/센티먼트 포함)
                                # Default analysis with bubble/sentiment
    
    python main.py --short      # Quick 분석 (버블/DTW 제외)
                                # Quick mode - skip heavy computations
    
    python main.py --full       # 전체 기능 (AI Validation 포함, API 비용 발생)
                                # Full mode - includes Multi-LLM validation
    
    python main.py --realtime   # 실시간 스트리밍 포함
                                # Include real-time streaming
    
    python main.py --full -r    # 전체 기능 + 실시간
                                # Full mode with real-time streaming

================================================================================
OUTPUT | 출력물
================================================================================

    outputs/
    ├── eimas_YYYYMMDD_HHMMSS.json   # Unified analysis results
    ├── eimas_YYYYMMDD.md            # Markdown summary
    └── reports/                     # AI-generated reports

================================================================================
"""


import asyncio
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict

# Runtime cache bootstrap (must run before pipeline imports that may touch yfinance).
def _configure_yfinance_cache_dir() -> None:
    cache_dir = os.getenv("EIMAS_YFINANCE_CACHE_DIR", "/tmp/eimas_yfinance_cache").strip()
    if not cache_dir or cache_dir.lower() in {"off", "none", "disable", "false", "0"}:
        return

    target = Path(cache_dir).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[Init] yfinance cache setup skipped: {type(exc).__name__}: {exc}")
        return

    try:
        import yfinance as yf

        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(target))
    except Exception as exc:
        print(f"[Init] yfinance cache redirect failed: {type(exc).__name__}: {exc}")


_configure_yfinance_cache_dir()

# ============================================================================
# Pipeline Imports
# ============================================================================
from pipeline.schemas import EIMASResult

# Extracted phase handlers (M2 split)
from pipeline.phases.phase1_collect import collect_data as phase1_collect_data
from pipeline.phases.phase2_basic import analyze_basic as phase2_analyze_basic
from pipeline.phases.phase2_enhanced import analyze_enhanced as phase2_analyze_enhanced
from pipeline.phases.phase2_adjustment import (
    analyze_sentiment_bubble as phase2_analyze_sentiment_bubble,
    apply_extended_data_adjustment as phase2_apply_extended_data_adjustment,
    analyze_institutional_frameworks as phase2_analyze_institutional_frameworks,
    run_adaptive_portfolio_phase as phase2_run_adaptive_portfolio,
)
from pipeline.phases.phase3_debate import run_debate as phase3_run_debate
from pipeline.phases.phase4_realtime import run_realtime as phase4_run_realtime
from pipeline.phases.phase45_operational import (
    generate_operational_report as phase45_generate_operational_report,
)
from pipeline.phases.phase5_storage import save_results as phase5_save_results
from pipeline.phases.phase6_portfolio import (
    run_backtest as phase6_run_backtest,
    run_performance_attribution as phase6_run_performance_attribution,
    run_tactical_allocation as phase6_run_tactical_allocation,
    run_stress_test as phase6_run_stress_test,
)
from pipeline.phases.phase7_report import (
    generate_report as phase7_generate_report,
    validate_report as phase7_validate_report,
)
from pipeline.phases.phase8_validation import (
    run_ai_validation_phase as phase8_run_ai_validation_phase,
    run_quick_validation as phase8_run_quick_validation,
)
from pipeline.phases.phase9_artifacts import (
    export_artifacts as phase9_export_artifacts,
)
# ============================================================================
# Phase Helper Functions
# ============================================================================

# Phase handlers are now owned by `pipeline/phases/*`.
_collect_data = phase1_collect_data
_analyze_basic = phase2_analyze_basic

def _analyze_enhanced(result: EIMASResult, market_data: Dict, quick_mode: bool):
    """Delegation wrapper to allow phase2 module to call tactical allocation hook."""
    return phase2_analyze_enhanced(
        result,
        market_data,
        quick_mode,
        run_tactical_allocation_fn=_run_tactical_allocation,
    )

_apply_extended_data_adjustment = phase2_apply_extended_data_adjustment
_analyze_sentiment_bubble = phase2_analyze_sentiment_bubble
_analyze_institutional_frameworks = phase2_analyze_institutional_frameworks
_run_adaptive_portfolio = phase2_run_adaptive_portfolio
_run_debate = phase3_run_debate
_run_realtime = phase4_run_realtime
_generate_operational_report = phase45_generate_operational_report
_save_results = phase5_save_results
_run_backtest = phase6_run_backtest
_run_performance_attribution = phase6_run_performance_attribution
_run_tactical_allocation = phase6_run_tactical_allocation
_run_stress_test = phase6_run_stress_test
_generate_report = phase7_generate_report
_validate_report = phase7_validate_report
_run_ai_validation_phase = phase8_run_ai_validation_phase
_run_quick_validation = phase8_run_quick_validation
_export_artifacts = phase9_export_artifacts

# ============================================================================
# Main Pipeline
# ============================================================================

async def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    generate_report: bool = False,
    full_mode: bool = False,
    enable_backtest: bool = False,
    enable_attribution: bool = False,
    enable_stress_test: bool = False,
    quick_validation_mode: str = None,
    output_dir: str = "outputs",
    cron_mode: bool = False,
) -> EIMASResult:
    """
    Execute the EIMAS integrated analysis pipeline.
    EIMAS 통합 분석 파이프라인 실행.
    
    This function orchestrates the complete analysis workflow from data collection
    through AI-powered report generation and validation.
    
    Pipeline Phases:
        Phase 1: Data collection (FRED macroeconomic, market prices, crypto)
        Phase 2: Quantitative analysis (regime detection, risk scoring, etc.)
        Phase 3: AI agent debate with dual-mode consensus
        Phase 4: Real-time streaming analysis (optional)
        Phase 5: Result storage (JSON, database)
        Phase 6: AI report generation (optional)
        Phase 7: Report validation (whitening, fact-check)
        Phase 8: Multi-LLM cross-validation (full mode only)
        Phase 8.5: Quick mode AI validation (KOSPI/SPX 분리, --quick1/--quick2)

    Args:
        enable_realtime (bool): Enable real-time VPIN/OFI streaming.
                               실시간 스트리밍 활성화. Default: False.
        realtime_duration (int): Duration in seconds for streaming.
                                스트리밍 지속 시간 (초). Default: 30.
        quick_mode (bool): Skip heavy computations (bubble, DTW, DBSCAN).
                          버블/DTW 등 고비용 분석 제외. Default: False.
        generate_report (bool): Generate AI-powered narrative report.
                               AI 리포트 생성 여부. Default: False.
        full_mode (bool): Include Multi-LLM validation (incurs API costs).
                         Multi-LLM 검증 포함 (API 비용 발생). Default: False.
        quick_validation_mode (str): Quick AI validation market focus ('KOSPI' or 'SPX').
                                    Quick 모드 검증 시장 (KOSPI/SPX). Default: None.
        output_dir (str): Output directory for JSON/Markdown artifacts.
                          결과 저장 디렉토리. Default: 'outputs'.
        cron_mode (bool): Scheduled mode; skips AI report generation.
                          스케줄 모드일 때 AI 리포트 생성을 생략. Default: False.
    
    Returns:
        EIMASResult: Comprehensive analysis result object containing:
            - fred_summary: Macroeconomic data summary
            - regime: Current market regime classification
            - risk_score: Composite risk score (0-100)
            - debate results: AI agent consensus
            - And many more analysis outputs...
    
    Raises:
        No exceptions are raised; errors are logged with warnings.
        모든 에러는 경고로 출력되며, 예외는 발생하지 않습니다.
    
    Example:
        >>> result = await run_integrated_pipeline(quick_mode=True)
        >>> print(f"Risk Score: {result.risk_score}")
    """
    start_time = datetime.now()
    start_perf = perf_counter()
    result = EIMASResult(timestamp=start_time.isoformat())
    raw_output_path = Path(output_dir).expanduser()
    if raw_output_path.is_absolute():
        output_path = raw_output_path
    else:
        # Keep legacy behavior: relative output dirs are project-root relative.
        output_path = Path(__file__).resolve().parent / raw_output_path
    should_generate_report = generate_report and not cron_mode
    phase_timings: Dict[str, Dict[str, Any]] = {}
    result.pipeline_phase_timings = phase_timings

    def _format_error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"[:300]

    def _record_phase_timing(
        phase_name: str,
        started_at: float,
        status: str = "ok",
        error: str = "",
    ) -> None:
        elapsed = perf_counter() - started_at
        entry: Dict[str, Any] = {
            "duration_sec": round(elapsed, 3),
            "status": status,
        }
        if error:
            entry["error"] = error
        phase_timings[phase_name] = entry
        print(f"  [Timing] {phase_name}: {elapsed:.3f}s ({status})")

    def _run_timed_sync(
        phase_name: str,
        fn: Callable[..., Any],
        *args,
        **kwargs,
    ) -> Any:
        started = perf_counter()
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:
            _record_phase_timing(
                phase_name,
                started,
                status="error",
                error=_format_error(exc),
            )
            raise
        _record_phase_timing(phase_name, started, status="ok")
        return value

    async def _run_timed_async(
        phase_name: str,
        fn: Callable[..., Awaitable[Any]],
        *args,
        **kwargs,
    ) -> Any:
        started = perf_counter()
        try:
            value = await fn(*args, **kwargs)
        except Exception as exc:
            _record_phase_timing(
                phase_name,
                started,
                status="error",
                error=_format_error(exc),
            )
            raise
        _record_phase_timing(phase_name, started, status="ok")
        return value

    print("=" * 70)
    print("  EIMAS - Integrated Analysis Pipeline")
    print("=" * 70)
    print(f"  Output Dir: {output_path}")
    if cron_mode:
        print("  Cron Mode: Enabled (report generation skipped)")

    # Phase 1-2: Data & Analysis
    market_data = await _run_timed_async(
        "phase1_collect_data",
        _collect_data,
        result,
        quick_mode,
    )
    events, regime_res = _run_timed_sync(
        "phase2_basic_analyze",
        _analyze_basic,
        result,
        market_data,
    )
    _run_timed_sync(
        "phase2_enhanced_analyze",
        _analyze_enhanced,
        result,
        market_data,
        quick_mode,
    )
    _run_timed_sync(
        "phase2_sentiment_bubble",
        _analyze_sentiment_bubble,
        result,
        market_data,
        quick_mode,
    )
    _run_timed_sync(
        "phase2_extended_adjustment",
        _apply_extended_data_adjustment,
        result,
    )  # PCR, Sentiment, Credit 기반 리스크 조정
    _run_timed_sync(
        "phase2_institutional_frameworks",
        _analyze_institutional_frameworks,
        result,
        market_data,
        quick_mode,
    )  # JP Morgan, Goldman Sachs 프레임워크
    _run_timed_sync(
        "phase2_adaptive_portfolio",
        _run_adaptive_portfolio,
        result,
        regime_res,
        quick_mode,
    )

    # Phase 3-4: Debate & Realtime
    await _run_timed_async(
        "phase3_debate",
        _run_debate,
        result,
        market_data,
    )
    await _run_timed_async(
        "phase4_realtime",
        _run_realtime,
        result,
        enable_realtime,
        realtime_duration,
    )

    # Phase 4.5: Operational Report (decision governance, rebalance)
    _run_timed_sync(
        "phase45_operational_report",
        _generate_operational_report,
        result,
    )

    # Phase 5: Storage
    output_file = _run_timed_sync(
        "phase5_storage",
        _save_results,
        result,
        events,
        output_path,
    )

    # Phase 6: Portfolio Theory Modules (Optional, 2026-02-04)
    _run_timed_sync(
        "phase6_backtest",
        _run_backtest,
        result,
        market_data,
        enable_backtest,
    )
    _run_timed_sync(
        "phase6_performance_attribution",
        _run_performance_attribution,
        result,
        enable_attribution,
    )
    _run_timed_sync(
        "phase6_stress_test",
        _run_stress_test,
        result,
        enable_stress_test,
    )

    # Phase 7: AI Report Generation
    report_content = await _run_timed_async(
        "phase7_generate_report",
        _generate_report,
        result,
        market_data,
        should_generate_report,
        output_path,
        output_file=output_file,
    )

    # Phase 8: Validation
    await _run_timed_async(
        "phase7_validate_report",
        _validate_report,
        result,
        report_content,
        should_generate_report,
        output_path,
        output_file=output_file,
    )
    _run_timed_sync(
        "phase8_ai_validation",
        _run_ai_validation_phase,
        result,
        full_mode,
        output_path,
        output_file=output_file,
    )

    # Phase 8.5: Quick Mode AI Validation (KOSPI/SPX 분리)
    _run_timed_sync(
        "phase85_quick_validation",
        _run_quick_validation,
        result,
        market_data,
        output_file,
        quick_validation_mode,
    )
    artifact_export = _run_timed_sync(
        "phase9_artifact_export",
        _export_artifacts,
        output_file,
        output_path,
        full_mode,
    )
    if artifact_export:
        result.audit_metadata["artifact_export"] = artifact_export

    # Summary
    elapsed = perf_counter() - start_perf
    phase_timings["pipeline_total"] = {
        "duration_sec": round(elapsed, 3),
        "status": "ok",
    }
    result.pipeline_elapsed_sec = round(elapsed, 3)
    result.audit_metadata["pipeline_elapsed_sec"] = round(elapsed, 3)
    result.audit_metadata["pipeline_phase_count"] = len(phase_timings) - 1
    result.audit_metadata["pipeline_timing_recorded_at"] = datetime.now().isoformat()

    # Persist final snapshot so runtime timing fields are reflected in final JSON.
    if output_file:
        try:
            target_path = Path(output_file).expanduser()
            target_path.parent.mkdir(exist_ok=True, parents=True)
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            print(f"  Final snapshot updated: {target_path}")
        except Exception as exc:
            print(f"⚠️ Final snapshot update failed: {_format_error(exc)}")

    ranked = sorted(
        (
            (phase_name, meta)
            for phase_name, meta in phase_timings.items()
            if phase_name != "pipeline_total"
        ),
        key=lambda item: item[1].get("duration_sec", 0.0),
        reverse=True,
    )
    print("\n[Pipeline Timing Summary] Top 8")
    for phase_name, meta in ranked[:8]:
        print(
            f"  - {phase_name}: {meta.get('duration_sec', 0.0):.3f}s"
            f" ({meta.get('status', 'n/a')})"
        )

    print("\n" + "=" * 70)
    print(f"EIMAS PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"Output: {output_file}")
    print("=" * 70)

    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description='EIMAS Modular Pipeline')
    parser.add_argument('--realtime', '-r', action='store_true', help='Enable realtime stream')
    parser.add_argument('--duration', '-d', type=int, default=30, help='Stream duration (seconds)')
    parser.add_argument('--short', '-s', action='store_true', help='Quick analysis (no bubble/DTW)')
    parser.add_argument('--quick', '-q', action='store_true', help='Alias for --short')
    parser.add_argument('--full', '-f', action='store_true', help='All features (AI Validation, costs API)')

    # Quick Mode AI Validation (2026-02-04)
    parser.add_argument('--quick1', action='store_true', help='Quick AI validation (KOSPI focus)')
    parser.add_argument('--quick2', action='store_true', help='Quick AI validation (SPX focus)')

    # Portfolio Theory Modules (2026-02-04)
    parser.add_argument('--backtest', action='store_true', help='Run backtest engine (5-year historical)')
    parser.add_argument('--attribution', action='store_true', help='Performance attribution (Brinson)')
    parser.add_argument('--stress-test', action='store_true', help='Stress testing (historical + hypothetical)')
    parser.add_argument('--output-dir', default='outputs', help='Output directory for artifacts')
    parser.add_argument('--cron-mode', action='store_true', help='Scheduled mode (skip AI report generation)')

    args = parser.parse_args()

    is_short = args.short or args.quick

    # Determine market focus
    market_focus = None
    if args.quick1:
        market_focus = 'KOSPI'
    elif args.quick2:
        market_focus = 'SPX'

    asyncio.run(run_integrated_pipeline(
        enable_realtime=args.realtime,
        realtime_duration=args.duration,
        quick_mode=is_short,
        generate_report=not is_short,
        full_mode=args.full,
        enable_backtest=args.backtest,
        enable_attribution=args.attribution,
        enable_stress_test=args.stress_test,
        quick_validation_mode=market_focus,
        output_dir=args.output_dir,
        cron_mode=args.cron_mode,
    ))


if __name__ == "__main__":
    main()
