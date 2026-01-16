
import sys
import logging
import subprocess
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.intraday_collector import IntradayCollector
from lib.crypto_collector import CryptoCollector
from lib.event_predictor import EventPredictor
from lib.event_attribution import EventAttributor
from lib.news_correlator import NewsCorrelator

logger = logging.getLogger('eimas.pipeline.standalone')

def run_standalone_scripts(result: Any, full_mode: bool) -> Any:
    """
    Phase 8: Standalone Scripts Execution
    """
    if not full_mode:
        return result

    print("\n" + "=" * 50)
    print("PHASE 8: STANDALONE SCRIPTS EXECUTION")
    print("=" * 50)

    # 8.1 Intraday
    print("\n[8.1] Intraday data collection...")
    try:
        intraday = IntradayCollector()
        intraday_result = intraday.collect_missing_days(days_back=5)
        result.intraday_summary = {
            'dates_collected': intraday_result.get('dates_collected', []),
            'tickers_collected': intraday_result.get('total_tickers', 0),
            'status': intraday_result.get('status', 'UNKNOWN')
        }
        print(f"      ✓ Status: {result.intraday_summary['status']}")
    except Exception as e:
        print(f"      ✗ Intraday error: {e}")

    # 8.2 Crypto
    print("\n[8.2] 24/7 Cryptocurrency monitoring...")
    try:
        crypto_monitor = CryptoCollector()
        prices = crypto_monitor.collect_current_prices()
        anomalies = crypto_monitor.detect_anomalies()
        result.crypto_monitoring = {
            'symbols_monitored': len(prices),
            'anomalies_detected': len(anomalies),
            'risk_level': 'HIGH' if len(anomalies) > 2 else 'LOW'
        }
        print(f"      ✓ Risk: {result.crypto_monitoring['risk_level']}")
    except Exception as e:
        print(f"      ✗ Crypto error: {e}")

    # 8.3 Pipeline Script
    print("\n[8.3] Multi-source data pipeline...")
    try:
        # Assuming we are running from root or adjusting path
        cwd = str(Path(__file__).parent.parent)
        pipeline_cmd = ['python', 'lib/market_data_pipeline.py', '--all']
        pipeline_proc = subprocess.run(pipeline_cmd, capture_output=True, text=True, timeout=60, cwd=cwd)
        if pipeline_proc.returncode == 0:
             print(f"      ✓ Pipeline executed successfully")
        else:
             print(f"      ✗ Pipeline failed")
    except Exception as e:
        print(f"      ✗ Pipeline execution error: {e}")

    # 8.4 Event Predictor
    print("\n[8.4] Economic event predictions...")
    try:
        predictor = EventPredictor()
        predictions = predictor.predict_upcoming_events()
        result.event_predictions = [{'event_type': p.event_type, 'predicted_date': str(p.predicted_date)} for p in predictions[:5]]
        print(f"      ✓ Events predicted: {len(predictions)}")
    except Exception as e:
        print(f"      ✗ Event predictor error: {e}")

    # 8.5 Event Attribution
    print("\n[8.5] Event cause analysis...")
    try:
        attributor = EventAttributor()
        report = attributor.analyze_recent_events(days_back=7)
        if report:
            result.event_attributions = [{'summary': a.summary[:100]} for a in report.attributions[:5]]
        print(f"      ✓ Attributions analyzed")
    except Exception as e:
        print(f"      ✗ Event attributor error: {e}")
    
    # 8.6 Backtester skipped

    # 8.7 News Correlator
    print("\n[8.7] Anomaly-news correlation analysis...")
    try:
        correlator = NewsCorrelator()
        attributions = correlator.process_recent_anomalies(hours_back=24)
        result.news_correlations = [{'ticker': a.ticker, 'summary': a.summary[:80]} for a in attributions[:5]]
        print(f"      ✓ Correlations found: {len(attributions)}")
    except Exception as e:
        print(f"      ✗ News correlator error: {e}")

    return result
