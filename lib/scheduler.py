import time
import logging
import argparse
import sys
import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import specific components
from lib.realtime_pipeline import RealtimePipeline, PipelineConfig, get_current_signals

from lib.final_report_agent import FinalReportAgent
from lib.regime_detector import RegimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('eimas.scheduler')

import yfinance as yf
import pandas as pd
import numpy as np

def calculate_correlations(tickers=['SPY', 'QQQ', 'TLT', 'GLD', 'HYG', 'XLF', 'XLE', 'XLV']):
    """Calculate correlation matrix for given tickers"""
    try:
        logger.info(f"Calculating correlations for {len(tickers)} assets...")
        data = yf.download(tickers, period="3mo", progress=False)['Close']
        if data.empty:
            return None, None
            
        # Calculate daily returns
        returns = data.pct_change().dropna()
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Format for JSON (list of lists)
        matrix_list = []
        ordered_tickers = [t for t in tickers if t in corr_matrix.columns]
        
        for t1 in ordered_tickers:
            row = []
            for t2 in ordered_tickers:
                val = corr_matrix.loc[t1, t2]
                row.append(round(float(val), 2))
            matrix_list.append(row)
            
        return ordered_tickers, matrix_list
    except Exception as e:
        logger.error(f"Correlation calculation failed: {e}")
        return None, None

from main import run_integrated_pipeline

async def run_pipeline_async():
    """Execute the async integrated pipeline"""
    logger.info("Starting Full Integrated Pipeline (Unified)...")
    
    try:
        # Run the full integrated pipeline
        # Using main.py's unified pipeline
        result = await run_integrated_pipeline(
            enable_realtime=True,
            realtime_duration=30,
            quick_mode=False,     # Full analysis
            generate_report=True, # AI report
            full_mode=True        # Activate AI validation and all features
        )
        
        # Determine output filename from result timestamp
        try:
            ts = datetime.fromisoformat(result.timestamp)
            filename = f"outputs/eimas_{ts.strftime('%Y%m%d_%H%M%S')}.json"
            logger.info(f"Integrated pipeline finished. Output: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Could not determine filename from result: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Integrated Pipeline failed: {e}", exc_info=True)
        return None

def run_pipeline():
    """Run the complete synchronous wrapper"""
    print(f"\n{'='*60}")
    print(f"EIMAS Scheduled Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        json_path = asyncio.run(run_pipeline_async())
        
        if json_path:
            print(f"\n[SUCCESS] Pipeline finished. Data saved to: {json_path}")
        else:
            print("\n[ERROR] Pipeline failed or produced no output.")
            
    except Exception as e:
        logger.error(f"Error during pipeline run: {str(e)}", exc_info=True)
        print(f"\n[ERROR] Pipeline failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='EIMAS Scheduler')
    parser.add_argument('--run-now', action='store_true', help='Run pipeline immediately and exit')
    parser.add_argument('--interval', type=int, default=60, help='Schedule interval in minutes')
    args = parser.parse_args()
    
    if args.run_now:
        run_pipeline()
        return

    # Simple scheduler loop
    logger.info(f"Scheduler started. Running every {args.interval} minutes.")
    print(f"Scheduler running. Jobs scheduled every {args.interval} minutes. Press Ctrl+C to exit.")
    
    last_run = datetime.now()
    # Initial run check? No, manual only if requested or explicit.
    # But usually scheduler runs immediately or waits. Let's wait.
    
    while True:
        try:
            now = datetime.now()
            next_run = last_run + timedelta(minutes=args.interval)
            
            if now >= next_run:
                run_pipeline()
                last_run = now
            
            time.sleep(10) # Check every 10 seconds
            
        except KeyboardInterrupt:
            print("\nScheduler stopped.")
            break
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
