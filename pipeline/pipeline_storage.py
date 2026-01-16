import sys
import logging
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.event_db import EventDatabase
from lib.trading_db import TradingDB, Signal, SignalSource, SignalAction, PortfolioCandidate, InvestorProfile
from lib.predictions_db import save_eimas_result
from lib.realtime_pipeline import IntegratedSignal, SignalDatabase
from lib.dashboard_generator import generate_dashboard

logger = logging.getLogger('eimas.pipeline.storage')

def run_storage(result: Any, market_data: Dict, output_dir: str = 'outputs', cron_mode: bool = False) -> tuple:
    """
    Phase 5: Database Storage
    """
    print("\n" + "=" * 50)
    print("PHASE 5: DATABASE STORAGE")
    print("=" * 50)

    # 5.1 이벤트 DB
    print("\n[5.1] Saving to Event Database...")
    try:
        event_db = EventDatabase('data/events.db')
        for event in result.events_detected:
            event_db.save_detected_event({
                'event_type': event['type'],
                'importance': event['importance'],
                'description': event['description'],
                'timestamp': result.timestamp,
            })
        
        snapshot_id = str(uuid.uuid4())[:8]
        event_db.save_market_snapshot({
            'snapshot_id': snapshot_id,
            'timestamp': result.timestamp,
            'spy_price': float(market_data['SPY']['Close'].iloc[-1]) if 'SPY' in market_data else 0.0,
            'spy_change_1d': 0.0, # Simplified
            'spy_change_5d': 0.0,
            'spy_vs_ma20': 0.0,
            'qqq_price': float(market_data['QQQ']['Close'].iloc[-1]) if 'QQQ' in market_data else 0.0,
            'iwm_price': float(market_data['IWM']['Close'].iloc[-1]) if 'IWM' in market_data else 0.0,
            'tlt_price': float(market_data['TLT']['Close'].iloc[-1]) if 'TLT' in market_data else 0.0,
            'gld_price': float(market_data['GLD']['Close'].iloc[-1]) if 'GLD' in market_data else 0.0,
            'vix_level': float(market_data['^VIX']['Close'].iloc[-1]) if '^VIX' in market_data else 0.0,
            'vix_percentile': 50.0,
            'rsi_14': 50.0,
            'macd_signal': 'neutral',
            'trend': result.regime.get('trend', 'unknown'),
            'volatility_regime': result.regime.get('volatility', 'normal'),
            'put_call_ratio': 1.0,
            'fear_greed_index': 50.0,
            'days_to_fomc': 0,
            'days_to_cpi': 0,
            'days_to_nfp': 0,
        })
        print(f"      ✓ Saved events and snapshot (ID: {snapshot_id})")
    except Exception as e:
        print(f"      ✗ Event DB error: {e}")

    # 5.2 시그널 DB
    print("\n[5.2] Saving to Signal Database...")
    try:
        signal_db = SignalDatabase('outputs/realtime_signals.db')
        integrated_signal = IntegratedSignal(
            timestamp=datetime.now(),
            symbol='INTEGRATED',
            combined_signal=result.final_recommendation.lower(),
            confidence=result.confidence,
            action=result.final_recommendation.lower(),
            alerts=result.warnings,
            liquidity_regime=result.fred_summary.get('liquidity_regime', 'Unknown'),
            rrp_delta=result.fred_summary.get('rrp_delta', 0),
            tga_delta=result.fred_summary.get('tga_delta', 0),
            net_liquidity=result.fred_summary.get('net_liquidity', 0),
            macro_signal=result.liquidity_signal.lower(),
            ofi=0, vpin=0, depth_ratio=1.0, micro_signal='neutral'
        )
        signal_db.save_signal(integrated_signal)
        print(f"      ✓ Saved integrated signal")
    except Exception as e:
        print(f"      ✗ Signal DB error: {e}")

    # 5.2.1 예측 DB
    try:
        save_eimas_result(result)
        print(f"      ✓ Saved predictions")
    except Exception as e:
        print(f"      ✗ Predictions DB error: {e}")

    # 5.2.2 Trading DB
    print("\n[5.2.2] Saving to Trading Database...")
    try:
        trading_db = TradingDB('data/trading.db')
        if result.portfolio_weights:
            for ticker, weight in result.portfolio_weights.items():
                candidate = PortfolioCandidate(
                    ticker=ticker, weight=weight, expected_return=0.0, expected_risk=0.0,
                    sharpe_ratio=0.0, profile=InvestorProfile.BALANCED, reason=f"GC-HRP allocation: {weight:.1%}"
                )
                trading_db.save_portfolio_candidate(candidate)
        result.trading_db_status = "SUCCESS"
        print(f"      ✓ Saved portfolio candidates")
    except Exception as e:
        print(f"      ✗ Trading DB error: {e}")

    # 5.3 JSON/MD Saving
    if not cron_mode:
        print("\n[5.3] Saving results...")
        if isinstance(output_dir, str):
            output_dir = Path(output_dir) if os.path.isabs(output_dir) else Path(__file__).parent.parent / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"integrated_{timestamp_str}.json"
        
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"      - JSON: {output_file}")
        
        md_file = output_dir / f"integrated_{timestamp_str}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(result.to_markdown())
        print(f"      - MD: {md_file}")
        
        # HTML Dashboard
        try:
            dashboard_html = generate_dashboard(
                signals=result.integrated_signals,
                summary=f"Risk: {result.risk_score:.1f}/100, Regime: {result.regime.get('regime', 'Unknown')}",
                regime_data=result.regime,
                crypto_panel={
                    'stress_test': result.crypto_stress_test,
                    'monitoring': result.crypto_monitoring if hasattr(result, 'crypto_monitoring') else {}
                },
                risk_data={
                    'risk_score': result.risk_score,
                    'risk_level': result.risk_level,
                    'base_risk': result.base_risk_score,
                    'market_quality': result.market_quality.__dict__ if result.market_quality else {},
                    'bubble_risk': result.bubble_risk.__dict__ if result.bubble_risk else {}
                },
                critical_path_data=result.critical_path_monitoring if hasattr(result, 'critical_path_monitoring') else {},
                macro_indicators=result.fred_summary,
                llm_summary=f"{result.full_mode_position} (FULL) vs {result.reference_mode_position} (REF)"
            )
            html_file = output_dir / f"dashboard_{timestamp_str}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            print(f"      - HTML: {html_file}")
        except Exception as e:
            print(f"      ✗ HTML dashboard error: {e}")

        return str(output_file), str(md_file)
    return "", ""