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
           ├─► [Phase 1] phase1_collect_data()  # 데이터 수집 (FRED, Market, Crypto)
           │                                    # Data collection from multiple sources
           │
           ├─► [Phase 2] phase2_* analyzers     # 기본/고급/보정 분석
           │
           ├─► [Phase 3] phase3_run_debate()    # AI 에이전트 토론 (Multi-LLM)
           │                                    # Dual-mode debate with consensus
           │
           ├─► [Phase 4] phase4_run_realtime()  # 실시간 스트리밍 (Optional)
           │                                    # VPIN/OFI stream analysis
           │
           ├─► [Phase 5] phase5_save_results()  # 결과 저장 (unified JSON)
           │                                    # Save to outputs/eimas_*.json
           │
           ├─► [Phase 6] phase6_* portfolio     # 백테스트/기여도/스트레스 (Optional)
           │
           ├─► [Phase 7] phase7_* report        # AI 리포트 생성/검증
           │
           ├─► [Phase 8] phase8_* validation    # Multi-LLM 검증/quick 검증
           │
           └─► [Phase 9] phase9_export_artifacts()

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
import logging
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _configure_network_failfast_logging() -> None:
    """Reduce noisy third-party network logs when fail-fast mode is enabled."""
    fail_fast_flags = (
        "EIMAS_FRED_FAIL_FAST_NETWORK",
        "EIMAS_EXTENDED_FAIL_FAST_NETWORK",
        "EIMAS_MARKET_DATA_FAIL_FAST_NETWORK",
        "EIMAS_CRYPTO_DATA_FAIL_FAST_NETWORK",
        "EIMAS_MARKET_INDICATORS_FAIL_FAST_NETWORK",
        "EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK",
        "EIMAS_DEBATE_FAIL_FAST_NETWORK",
        "EIMAS_REPORT_FAIL_FAST_NETWORK",
        "EIMAS_BUBBLE_FAIL_FAST_NETWORK",
        "EIMAS_SENTIMENT_FAIL_FAST_NETWORK",
        "EIMAS_ENHANCED_FAIL_FAST_NETWORK",
        "EIMAS_REGIME_FAIL_FAST_NETWORK",
    )

    quiet_yfinance = _env_flag("EIMAS_YFINANCE_QUIET", default=False)
    if not quiet_yfinance:
        quiet_yfinance = any(_env_flag(flag, default=False) for flag in fail_fast_flags)

    if not quiet_yfinance:
        return

    yfinance_logger = logging.getLogger("yfinance")
    yfinance_logger.setLevel(logging.CRITICAL)
    yfinance_logger.propagate = False
    if not yfinance_logger.handlers:
        yfinance_logger.addHandler(logging.NullHandler())


_configure_yfinance_cache_dir()
_configure_network_failfast_logging()

# ============================================================================
# Pipeline Imports
# ============================================================================
from pipeline.schemas import EIMASResult

from pipeline.app import (
    PhaseRuntimeTracker,
    pipeline_profile_choices,
    resolve_output_path,
    resolve_pipeline_profile,
    run_pipeline_phases,
    run_short_pipeline_phases,
)

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
    quick_validation_mode: str | None = None,
    output_dir: str = "outputs",
    cron_mode: bool = False,
    enable_paper_auto: bool = False,
    paper_account: str = "ra_auto",
    paper_capital: float = 100000.0,
    paper_poll_only: bool = False,
    paper_backtest: bool = False,
    paper_enforce_approval: bool = False,
    debate_full_lookback: int = 365,
    debate_ref_lookback: int = 90,
    debate_skip_reference: bool = False,
    pipeline_profile_name: str = "legacy",
    use_parallel: bool = False,
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
        Phase 4.5: Operational report + paper execution
        Phase 5: Result storage (JSON, database)
        Phase 6: Portfolio modules (backtest/attribution/stress, optional)
        Phase 7: AI report generation + validation
        Phase 8: Multi-LLM cross-validation (full mode only)
        Phase 8.5: Quick mode AI validation (KOSPI/SPX 분리, --quick1/--quick2)
        Phase 9: Artifact export

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
        enable_paper_auto (bool): Enable auto paper execution from trade plan.
        paper_account (str): Paper account name.
        paper_capital (float): Initial paper capital for account bootstrap.
        paper_poll_only (bool): Poll pending LIMIT orders only.
        paper_backtest (bool): Run allocation backtest after auto execution.
        paper_enforce_approval (bool): Block auto execution when human approval is required.
        debate_full_lookback (int): FULL mode lookback window for phase3 debate.
        debate_ref_lookback (int): REFERENCE mode lookback window for phase3 debate.
        debate_skip_reference (bool): Skip REFERENCE mode and mirror FULL mode result.
    
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
    output_path = resolve_output_path(output_dir, __file__)
    should_generate_report = generate_report and not cron_mode
    phase_timings: Dict[str, Dict[str, Any]] = {}
    result.pipeline_phase_timings = phase_timings
    runtime = PhaseRuntimeTracker(phase_timings)
    pipeline_profile = resolve_pipeline_profile(pipeline_profile_name)
    result.audit_metadata["pipeline_profile"] = pipeline_profile.name
    result.audit_metadata["pipeline_profile_policy"] = pipeline_profile.to_dict()

    runtime.print_pipeline_banner(output_path, cron_mode=cron_mode)
    print(f"  Profile: {pipeline_profile.name}")

    if quick_mode:
        # --short: 경량 수집 → 실시간 → 운용 → 모의주문 → DB 적재
        print("  Mode: SHORT (realtime response + DB ingest)")
        output_file, _ = await run_short_pipeline_phases(
            runtime=runtime,
            result=result,
            output_path=output_path,
            enable_realtime=enable_realtime,
            realtime_duration=realtime_duration,
            enable_paper_auto=enable_paper_auto,
            paper_account=paper_account,
            paper_capital=paper_capital,
            paper_poll_only=paper_poll_only,
            paper_backtest=paper_backtest,
            paper_enforce_approval=paper_enforce_approval,
            output_dir=output_dir,
            use_parallel=use_parallel,
        )
    else:
        # --full: 전체 시장환경 분석 파이프라인
        print("  Mode: FULL (market environment analysis)")
        output_file, _ = await run_pipeline_phases(
            runtime=runtime,
            result=result,
            output_path=output_path,
            quick_mode=quick_mode,
            enable_realtime=enable_realtime,
            realtime_duration=realtime_duration,
            enable_backtest=enable_backtest,
            enable_attribution=enable_attribution,
            enable_stress_test=enable_stress_test,
            should_generate_report=should_generate_report,
            full_mode=full_mode,
            quick_validation_mode=quick_validation_mode,
            enable_paper_auto=enable_paper_auto,
            paper_account=paper_account,
            paper_capital=paper_capital,
            paper_poll_only=paper_poll_only,
            paper_backtest=paper_backtest,
            paper_enforce_approval=paper_enforce_approval,
            debate_full_lookback=debate_full_lookback,
            debate_ref_lookback=debate_ref_lookback,
            debate_skip_reference=debate_skip_reference,
            pipeline_profile=pipeline_profile,
            use_parallel=use_parallel,
        )

    # Summary
    elapsed = perf_counter() - start_perf
    runtime.record_total(elapsed)
    runtime.apply_result_timing_metadata(
        result,
        elapsed,
        datetime.now().isoformat(),
    )
    runtime.persist_final_snapshot(output_file, result.to_dict())

    runtime.print_timing_summary(top_n=8)
    runtime.print_pipeline_completion(elapsed, output_file)

    return result


# ============================================================================
# CLI Entry Point
# ============================================================================

def _build_main_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the canonical pipeline entrypoint."""
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
    parser.add_argument('--paper-auto', action='store_true', help='Enable auto paper LIMIT orders from trade_plan')
    parser.add_argument('--paper-account', default='ra_auto', help='Paper trading account name')
    parser.add_argument('--paper-capital', type=float, default=100000.0, help='Initial capital for paper account')
    parser.add_argument('--paper-poll-only', action='store_true', help='Poll pending paper LIMIT orders only')
    parser.add_argument('--paper-backtest', action='store_true', help='Run allocation backtest after auto paper execution')
    parser.add_argument('--paper-enforce-approval', action='store_true', help='Require human approval gate for auto paper execution')
    parser.add_argument('--debate-full-lookback', type=int, default=365, help='Phase3 FULL mode lookback window (days)')
    parser.add_argument('--debate-ref-lookback', type=int, default=90, help='Phase3 REFERENCE mode lookback window (days)')
    parser.add_argument('--debate-skip-reference', action='store_true', help='Skip REFERENCE debate mode and mirror FULL mode')
    parser.add_argument(
        '--profile',
        default='legacy',
        choices=pipeline_profile_choices(),
        help='Pipeline profile (legacy|us-trader-v1)',
    )
    parser.add_argument('--output-dir', default='outputs', help='Output directory for artifacts')
    parser.add_argument('--cron-mode', action='store_true', help='Scheduled mode (skip AI report generation)')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel data collection in Phase 1 (experimental)')
    return parser


def _resolve_market_focus(args: argparse.Namespace) -> str | None:
    """Resolve quick validation market focus from CLI flags."""
    if args.quick1:
        return 'KOSPI'
    if args.quick2:
        return 'SPX'
    return None


def _build_pipeline_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Map CLI args to run_integrated_pipeline kwargs."""
    is_short = args.short or args.quick
    return {
        "enable_realtime": args.realtime,
        "realtime_duration": args.duration,
        "quick_mode": is_short,
        "generate_report": not is_short,
        "full_mode": args.full,
        "enable_backtest": args.backtest,
        "enable_attribution": args.attribution,
        "enable_stress_test": args.stress_test,
        "quick_validation_mode": _resolve_market_focus(args),
        "output_dir": args.output_dir,
        "cron_mode": args.cron_mode,
        "enable_paper_auto": args.paper_auto,
        "paper_account": args.paper_account,
        "paper_capital": args.paper_capital,
        "paper_poll_only": args.paper_poll_only,
        "paper_backtest": args.paper_backtest,
        "paper_enforce_approval": args.paper_enforce_approval,
        "debate_full_lookback": args.debate_full_lookback,
        "debate_ref_lookback": args.debate_ref_lookback,
        "debate_skip_reference": args.debate_skip_reference,
        "pipeline_profile_name": args.profile,
        "use_parallel": args.parallel,
    }


def main(argv: list[str] | None = None):
    """Canonical CLI entrypoint for the integrated pipeline."""
    parser = _build_main_parser()
    args = parser.parse_args(argv)
    asyncio.run(run_integrated_pipeline(**_build_pipeline_kwargs(args)))


if __name__ == "__main__":
    main()
