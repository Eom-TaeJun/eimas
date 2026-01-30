#!/usr/bin/env python3
"""
EIMAS - Economic Intelligence Multi-Agent System (Refactored Orchestrator)
==========================================================================
Modularized entry point for the EIMAS pipeline.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure lib can be imported
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.schemas import EIMASResult
from pipeline.pipeline_data import run_data_collection
from pipeline.pipeline_analysis import run_analysis
from pipeline.pipeline_debate import run_debate
from pipeline.pipeline_realtime import run_realtime_monitor_pipeline
from pipeline.pipeline_storage import run_storage
from pipeline.pipeline_report import run_report_and_qa
from pipeline.pipeline_standalone import run_standalone_scripts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eimas.orchestrator')

async def run_pipeline(args):
    start_time = datetime.now()
    
    # Initialize Result Object
    result = EIMASResult(timestamp=start_time.isoformat())
    
    print("=" * 70)
    print("  EIMAS - Integrated Analysis Pipeline (Modular)")
    print("=" * 70)
    
    # Determine flags
    quick_mode = args.quick or args.mode == 'quick'
    full_mode = args.full or args.mode == 'full'
    enable_realtime = args.realtime and not args.cron
    generate_report = args.report or args.mode == 'report'
    
    # Phase 1: Data
    market_data, indicators_summary = run_data_collection(result, quick_mode)
    
    # Phase 2: Analysis
    await run_analysis(result, market_data, result.fred_summary, quick_mode)
    
    # Phase 3: Debate
    await run_debate(result, market_data, quick_mode)
    
    # Phase 4: Realtime
    if enable_realtime:
        await run_realtime_monitor_pipeline(result, enable_realtime, args.duration)
    
    # Phase 5: Storage
    json_path, md_path = run_storage(result, market_data, args.output, args.cron)
    
    # Phase 6 & 7: Report & QA
    if generate_report:
        await run_report_and_qa(result, market_data, json_path, quick_mode)
        
    # Phase 8: Standalone
    run_standalone_scripts(result, full_mode)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print(f"EIMAS PIPELINE COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print(f"Action: {result.final_recommendation} ({result.confidence:.0%})")
    print(f"Risk: {result.risk_score:.1f}/100")
    print(f"Regime: {result.regime.get('regime', 'Unknown')}")
    if json_path: print(f"JSON: {json_path}")
    if md_path: print(f"MD:   {md_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='EIMAS Refactored Orchestrator')
    parser.add_argument('--mode', choices=['full', 'quick', 'report'], default='full', help='Analysis mode')
    parser.add_argument('--quick', action='store_true', help='Quick mode alias')
    parser.add_argument('--full', action='store_true', help='Full mode alias')
    parser.add_argument('--report', action='store_true', help='Generate AI report')
    parser.add_argument('--realtime', action='store_true', help='Enable realtime VPIN')
    parser.add_argument('--duration', type=int, default=30, help='Realtime duration')
    parser.add_argument('--cron', action='store_true', help='Cron mode')
    parser.add_argument('--output', default='outputs', help='Output directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_pipeline(args))
