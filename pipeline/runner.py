import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.schemas import EIMASResult
from pipeline.collection.runner import run_data_collection
from pipeline.analysis.runner import run_analysis
from pipeline.signal.runner import run_debate
from pipeline.realtime.runner import run_realtime_monitor_pipeline
from pipeline.storage.runner import run_storage
from pipeline.reporting.runner import run_report_and_qa
from pipeline.standalone.runner import run_standalone_scripts

async def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    output_dir: str = 'outputs',
    cron_mode: bool = False,
    full_mode: bool = False
) -> EIMASResult:
    """
    EIMAS 통합 파이프라인 실행
    """
    start_time = datetime.now()
    result = EIMASResult(timestamp=start_time.isoformat())

    print("=" * 70)
    print("  EIMAS - Integrated Analysis Pipeline")
    print("=" * 70)
    mode_str = 'Full' if full_mode else ('Quick' if quick_mode else 'Standard')
    print(f"  Mode: {mode_str}")
    print(f"  Realtime: {'Enabled' if enable_realtime else 'Disabled'}")
    if full_mode:
        print(f"  Standalone Scripts: Enabled")
    print("=" * 70)

    # Phase 1: Data Collection
    market_data, indicators_summary = run_data_collection(result, quick_mode)

    # Phase 2: Analysis
    result = await run_analysis(result, market_data, result.fred_summary, quick_mode)

    # Phase 3: Multi-Agent Debate
    result = await run_debate(result, market_data, quick_mode)

    # Phase 4: Real-time Monitoring
    result = await run_realtime_monitor_pipeline(result, enable_realtime, realtime_duration)

    # Phase 5: Storage
    output_json, output_md = run_storage(result, market_data, output_dir, cron_mode)

    # Phase 6 & 7: Report & QA
    if not cron_mode:
        await run_report_and_qa(result, market_data, output_md, quick_mode)

    # Phase 12: Standalone Scripts (if full_mode)
    if full_mode:
        await run_standalone_scripts(result)

    print("\n" + "=" * 70)
    print(f"  EIMAS Pipeline Completed in {(datetime.now() - start_time).total_seconds():.1f}s")
    print("=" * 70)

    return result