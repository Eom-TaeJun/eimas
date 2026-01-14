#!/usr/bin/env python3
"""
EIMAS - Economic Intelligence Multi-Agent System
=================================================
통합 실행 파이프라인

모든 기능을 순차적으로 실행:
1. 데이터 수집 (FRED, Market, Crypto)
2. 레짐 탐지
3. 이벤트 탐지
4. 유동성 분석 (Granger Causality)
5. 멀티에이전트 토론 (소수의견 보호)
6. 실시간 스트리밍 (선택적)
7. DB 저장
8. 알림 발송

제외 항목:
- CME FedWatch 히스토리컬 분석 (2024-2025 확정 데이터)
- LASSO 예측 (히스토리컬 패턴 기반)

Usage:
    python main_integrated.py              # 전체 파이프라인
    python main_integrated.py --realtime   # 실시간 스트리밍 포함
    python main_integrated.py --quick      # 빠른 분석만
"""

import argparse
import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

# 프로젝트 루트
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# Imports
# ============================================================================

# Core
from core.schemas import AnalysisMode, HistoricalDataConfig
from core.debate import DebateProtocol

# Agents
from agents.orchestrator import MetaOrchestrator

# Data Collection
from lib.fred_collector import FREDCollector
from lib.data_collector import DataManager
from lib.unified_data_store import UnifiedDataStore
from lib.market_indicators import MarketIndicatorsCollector

# Analysis
from lib.regime_detector import RegimeDetector
from lib.event_framework import QuantitativeEventDetector, EventType
from lib.liquidity_analysis import LiquidityMarketAnalyzer
from lib.causal_network import CausalNetworkAnalyzer
from lib.critical_path import CriticalPathAggregator
from lib.correlation_monitor import CorrelationMonitor
from lib.etf_flow_analyzer import ETFFlowAnalyzer

# Real-time
from lib.binance_stream import BinanceStreamer, StreamConfig
from lib.microstructure import MicrostructureAnalyzer
from lib.realtime_pipeline import RealtimePipeline, PipelineConfig, SignalDatabase

# Database
from lib.trading_db import TradingDB, Signal
from lib.event_db import EventDatabase

# Dual Mode
from lib.dual_mode_analyzer import DualModeAnalyzer, ModeResult

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('eimas.integrated')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EIMASResult:
    """통합 실행 결과"""
    timestamp: str

    # 데이터 수집
    fred_summary: Dict = field(default_factory=dict)
    market_data_count: int = 0
    crypto_data_count: int = 0

    # 분석 결과
    regime: Dict = field(default_factory=dict)
    events_detected: List[Dict] = field(default_factory=list)
    liquidity_signal: str = "NEUTRAL"
    risk_score: float = 0.0

    # 에이전트 토론
    debate_consensus: Dict = field(default_factory=dict)
    dissent_records: List[Dict] = field(default_factory=list)
    has_strong_dissent: bool = False

    # Dual Mode
    full_mode_position: str = "NEUTRAL"
    reference_mode_position: str = "NEUTRAL"
    modes_agree: bool = True

    # 최종 권고
    final_recommendation: str = "HOLD"
    confidence: float = 0.5
    risk_level: str = "MEDIUM"
    warnings: List[str] = field(default_factory=list)

    # 실시간 (선택)
    realtime_signals: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# Main Pipeline
# ============================================================================

async def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False
) -> EIMASResult:
    """
    EIMAS 통합 파이프라인 실행

    Args:
        enable_realtime: 실시간 Binance 스트리밍 활성화
        realtime_duration: 실시간 스트리밍 시간 (초)
        quick_mode: 빠른 분석 모드 (일부 생략)

    Returns:
        EIMASResult: 통합 분석 결과
    """
    start_time = datetime.now()
    result = EIMASResult(timestamp=start_time.isoformat())

    print("=" * 70)
    print("  EIMAS - Integrated Analysis Pipeline")
    print("=" * 70)
    print(f"  Mode: {'Quick' if quick_mode else 'Full'}")
    print(f"  Realtime: {'Enabled' if enable_realtime else 'Disabled'}")
    print("=" * 70)

    # ========================================================================
    # Phase 1: Data Collection
    # ========================================================================
    print("\n" + "=" * 50)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 50)

    # 1.1 FRED 데이터 수집
    print("\n[1.1] Collecting FRED data...")
    try:
        fred = FREDCollector()
        fred_summary = fred.collect_all()
        result.fred_summary = {
            'rrp': fred_summary.rrp,
            'rrp_delta': fred_summary.rrp_delta,
            'tga': fred_summary.tga,
            'tga_delta': fred_summary.tga_delta,
            'fed_assets': fred_summary.fed_assets,
            'net_liquidity': fred_summary.net_liquidity,
            'liquidity_regime': fred_summary.liquidity_regime,
            'fed_funds': fred_summary.fed_funds,
            'treasury_10y': fred_summary.treasury_10y,
            'spread_10y2y': fred_summary.spread_10y2y,
            'curve_status': fred_summary.curve_status,
        }
        print(f"      ✓ RRP: ${fred_summary.rrp:.0f}B (Δ{fred_summary.rrp_delta:+.0f}B)")
        print(f"      ✓ TGA: ${fred_summary.tga:.0f}B (Δ{fred_summary.tga_delta:+.0f}B)")
        print(f"      ✓ Net Liquidity: ${fred_summary.net_liquidity:.0f}B ({fred_summary.liquidity_regime})")
        print(f"      ✓ Curve: {fred_summary.curve_status} (10Y-2Y: {fred_summary.spread_10y2y:.2f}%)")
    except Exception as e:
        print(f"      ✗ FRED error: {e}")
        fred_summary = None

    # 1.2 시장 데이터 수집
    print("\n[1.2] Collecting market data...")
    try:
        dm = DataManager(lookback_days=365 if not quick_mode else 90)
        tickers_config = {
            'market': [
                {'ticker': 'SPY'}, {'ticker': 'QQQ'}, {'ticker': 'IWM'},
                {'ticker': 'DIA'}, {'ticker': 'TLT'}, {'ticker': 'GLD'},
                {'ticker': 'USO'}, {'ticker': 'UUP'}, {'ticker': '^VIX'}
            ],
            'crypto': [
                {'ticker': 'BTC-USD'}, {'ticker': 'ETH-USD'}
            ]
        }
        market_data, macro_data = dm.collect_all(tickers_config)
        result.market_data_count = len(market_data)
        print(f"      ✓ Collected {len(market_data)} tickers")
    except Exception as e:
        print(f"      ✗ Market data error: {e}")
        market_data = {}
        macro_data = None

    # 1.3 암호화폐 데이터 (DataManager가 이미 수집)
    print("\n[1.3] Crypto data collected with market data...")
    crypto_tickers = ['BTC-USD', 'ETH-USD']
    result.crypto_data_count = sum(1 for t in crypto_tickers if t in market_data)
    print(f"      ✓ Crypto in market data: {result.crypto_data_count} tickers")

    # 1.4 시장 지표
    if not quick_mode:
        print("\n[1.4] Collecting market indicators...")
        try:
            indicators = MarketIndicatorsCollector()
            indicators_summary = indicators.collect_all()
            print(f"      ✓ VIX: {indicators_summary.vix.current:.2f}")
            print(f"      ✓ Fear & Greed: {indicators_summary.vix.fear_greed_level}")
        except Exception as e:
            print(f"      ✗ Indicators error: {e}")

    # ========================================================================
    # Phase 2: Analysis
    # ========================================================================
    print("\n" + "=" * 50)
    print("PHASE 2: ANALYSIS")
    print("=" * 50)

    # 2.1 레짐 탐지
    print("\n[2.1] Detecting market regime...")
    try:
        regime_detector = RegimeDetector(ticker='SPY')  # ticker는 생성자에서
        regime_result = regime_detector.detect()  # 인자 없이 호출
        result.regime = {
            'regime': regime_result.regime.value if hasattr(regime_result.regime, 'value') else str(regime_result.regime),
            'trend': regime_result.trend_state.value if hasattr(regime_result.trend_state, 'value') else str(regime_result.trend_state),
            'volatility': regime_result.volatility_state.value if hasattr(regime_result.volatility_state, 'value') else str(regime_result.volatility_state),
            'confidence': regime_result.confidence / 100 if regime_result.confidence > 1 else regime_result.confidence,
            'description': regime_result.description,
            'strategy': regime_result.strategy,
        }
        print(f"      ✓ Regime: {result.regime['regime']}")
        print(f"      ✓ Trend: {result.regime['trend']}, Volatility: {result.regime['volatility']}")
        print(f"      ✓ Confidence: {result.regime['confidence']:.0%}")
    except Exception as e:
        print(f"      ✗ Regime error: {e}")

    # 2.2 이벤트 탐지
    print("\n[2.2] Detecting events...")
    try:
        event_detector = QuantitativeEventDetector()

        # 유동성 이벤트
        if fred_summary:
            liquidity_data = {
                'rrp': fred_summary.rrp,
                'rrp_delta': fred_summary.rrp_delta,
                'tga': fred_summary.tga,
                'tga_delta': fred_summary.tga_delta,
                'net_liquidity': fred_summary.net_liquidity,
            }
            liquidity_events = event_detector.detect_liquidity_events(liquidity_data)
            result.events_detected.extend([{
                'type': e.event_type.value,
                'importance': e.importance.value,
                'description': e.description
            } for e in liquidity_events])

            if liquidity_events:
                for e in liquidity_events:
                    print(f"      ⚠ {e.event_type.value}: {e.description}")
            else:
                print("      ✓ No liquidity events detected")

        # 시장 이벤트
        if fred_summary and market_data:
            market_events_data = {
                'vix': market_data.get('VIX', {}).get('Close', [0])[-1] if 'VIX' in market_data else 0,
                'spread_10y2y': fred_summary.spread_10y2y,
                'hy_oas': fred_summary.hy_oas,
            }
            # Additional market event detection can go here

    except Exception as e:
        print(f"      ✗ Event detection error: {e}")

    # 2.3 유동성 분석 (Granger Causality)
    if not quick_mode:
        print("\n[2.3] Liquidity-Market causality analysis...")
        try:
            liquidity_analyzer = LiquidityMarketAnalyzer()
            liquidity_signal = liquidity_analyzer.generate_signals()
            result.liquidity_signal = liquidity_signal.get('signal', 'NEUTRAL')
            print(f"      ✓ Liquidity Signal: {result.liquidity_signal}")
            if liquidity_signal.get('causality_results'):
                for var, strength in liquidity_signal['causality_results'].items():
                    print(f"        - {var}: {strength:.2f}")
        except Exception as e:
            print(f"      ✗ Liquidity analysis error: {e}")

    # 2.4 Critical Path 분석
    print("\n[2.4] Critical path analysis...")
    try:
        critical_path = CriticalPathAggregator()
        if market_data:
            cp_result = critical_path.analyze(market_data)
            # CriticalPathResult는 dataclass이므로 getattr 사용
            result.risk_score = getattr(cp_result, 'total_risk_score', 0)
            risk_level = getattr(cp_result, 'risk_level', 'Unknown')
            print(f"      ✓ Risk Score: {result.risk_score:.1f}/100")
            print(f"      ✓ Risk Level: {risk_level}")
            print(f"      ✓ Primary Risk Path: {getattr(cp_result, 'primary_risk_path', 'N/A')}")

            # 경고 추가
            if result.risk_score > 50:
                result.warnings.append(f"High risk score: {result.risk_score:.1f}")
    except Exception as e:
        print(f"      ✗ Critical path error: {e}")

    # 2.5 ETF Flow 분석
    if not quick_mode:
        print("\n[2.5] ETF flow analysis...")
        try:
            etf_analyzer = ETFFlowAnalyzer()
            etf_result = etf_analyzer.analyze()
            print(f"      ✓ Sector Rotation: {etf_result.get('rotation_signal', 'N/A')}")
            print(f"      ✓ Style: {etf_result.get('style_signal', 'N/A')}")
        except Exception as e:
            print(f"      ✗ ETF flow error: {e}")

    # ========================================================================
    # Phase 3: Multi-Agent Debate
    # ========================================================================
    print("\n" + "=" * 50)
    print("PHASE 3: MULTI-AGENT DEBATE")
    print("=" * 50)

    # 3.1 FULL Mode (Historical 365일)
    print("\n[3.1] Running FULL mode analysis...")
    try:
        orchestrator_full = MetaOrchestrator(verbose=False)
        query = "Analyze current market conditions, risks, and generate trading signals"
        result_full = await orchestrator_full.run_with_debate(query, market_data)

        # consensus는 토픽별 딕셔너리 → 종합 계산 필요
        consensus_by_topic = result_full.get('debate', {}).get('consensus', {})
        analysis_data = result_full.get('analysis', {})

        if consensus_by_topic:
            # 모든 토픽의 confidence 평균 계산
            confidences = [c.get('confidence', 0.5) for c in consensus_by_topic.values()]
            full_confidence = sum(confidences) / len(confidences) if confidences else 0.5

            # 포지션 결정: analysis의 regime + risk 기반
            regime = analysis_data.get('current_regime', 'NEUTRAL')
            risk_score = analysis_data.get('total_risk_score', 50)
            regime_conf = analysis_data.get('regime_confidence', 50) / 100

            if regime == 'BULL' and risk_score < 30:
                result.full_mode_position = 'BULLISH'
                full_confidence = max(full_confidence, regime_conf)
            elif regime == 'BEAR' or risk_score > 70:
                result.full_mode_position = 'BEARISH'
                full_confidence = max(full_confidence, regime_conf)
            else:
                result.full_mode_position = 'NEUTRAL'

            full_dissent = []
            full_strong_dissent = False
        else:
            result.full_mode_position = 'NEUTRAL'
            full_confidence = 0.5
            full_dissent = []
            full_strong_dissent = False

        print(f"      ✓ FULL Position: {result.full_mode_position}")
        print(f"      ✓ Confidence: {full_confidence:.0%}")
        print(f"      ✓ Regime: {analysis_data.get('current_regime', 'N/A')}, Risk: {analysis_data.get('total_risk_score', 0):.1f}")
        if full_dissent:
            print(f"      ⚠ Dissent Records: {len(full_dissent)}")
    except Exception as e:
        print(f"      ✗ FULL mode error: {e}")
        full_confidence = 0.5
        full_dissent = []
        full_strong_dissent = False

    # 3.2 REFERENCE Mode (Recent 90일)
    print("\n[3.2] Running REFERENCE mode analysis...")
    try:
        dm_ref = DataManager(lookback_days=90)
        market_data_ref, _ = dm_ref.collect_all(tickers_config)

        orchestrator_ref = MetaOrchestrator(verbose=False)
        result_ref = await orchestrator_ref.run_with_debate(query, market_data_ref)

        # consensus는 토픽별 딕셔너리 → 종합 계산 필요
        consensus_by_topic_ref = result_ref.get('debate', {}).get('consensus', {})
        analysis_data_ref = result_ref.get('analysis', {})

        if consensus_by_topic_ref:
            # 모든 토픽의 confidence 평균 계산
            confidences_ref = [c.get('confidence', 0.5) for c in consensus_by_topic_ref.values()]
            ref_confidence = sum(confidences_ref) / len(confidences_ref) if confidences_ref else 0.5

            # 포지션 결정: analysis의 regime + risk 기반
            regime_ref = analysis_data_ref.get('current_regime', 'NEUTRAL')
            risk_score_ref = analysis_data_ref.get('total_risk_score', 50)
            regime_conf_ref = analysis_data_ref.get('regime_confidence', 50) / 100

            if regime_ref == 'BULL' and risk_score_ref < 30:
                result.reference_mode_position = 'BULLISH'
                ref_confidence = max(ref_confidence, regime_conf_ref)
            elif regime_ref == 'BEAR' or risk_score_ref > 70:
                result.reference_mode_position = 'BEARISH'
                ref_confidence = max(ref_confidence, regime_conf_ref)
            else:
                result.reference_mode_position = 'NEUTRAL'

            ref_dissent = []
            ref_strong_dissent = False
        else:
            result.reference_mode_position = 'NEUTRAL'
            ref_confidence = 0.5
            ref_dissent = []
            ref_strong_dissent = False

        print(f"      ✓ REFERENCE Position: {result.reference_mode_position}")
        print(f"      ✓ Confidence: {ref_confidence:.0%}")
        print(f"      ✓ Regime: {analysis_data_ref.get('current_regime', 'N/A')}, Risk: {analysis_data_ref.get('total_risk_score', 0):.1f}")
        if ref_dissent:
            print(f"      ⚠ Dissent Records: {len(ref_dissent)}")
    except Exception as e:
        print(f"      ✗ REFERENCE mode error: {e}")
        ref_confidence = 0.5
        ref_dissent = []
        ref_strong_dissent = False

    # 3.3 모드 비교
    print("\n[3.3] Comparing modes...")
    result.modes_agree = (result.full_mode_position == result.reference_mode_position)
    result.dissent_records = full_dissent + ref_dissent
    result.has_strong_dissent = full_strong_dissent or ref_strong_dissent

    # Dual Mode 분석기로 최종 권고
    analyzer = DualModeAnalyzer()
    full_mode = ModeResult(
        mode=AnalysisMode.FULL,
        consensus=None,
        confidence=full_confidence,
        position=result.full_mode_position,
        dissent_count=len(full_dissent),
        has_strong_dissent=full_strong_dissent
    )
    ref_mode = ModeResult(
        mode=AnalysisMode.REFERENCE,
        consensus=None,
        confidence=ref_confidence,
        position=result.reference_mode_position,
        dissent_count=len(ref_dissent),
        has_strong_dissent=ref_strong_dissent
    )
    comparison = analyzer.compare_modes(full_mode, ref_mode)

    result.final_recommendation = comparison.recommended_action
    result.confidence = (full_confidence + ref_confidence) / 2
    result.risk_level = comparison.risk_level

    if not result.modes_agree:
        result.warnings.append(f"Mode divergence: FULL={result.full_mode_position}, REF={result.reference_mode_position}")
    if result.has_strong_dissent:
        result.warnings.append("Strong dissent exists - review carefully")

    print(f"      ✓ Modes Agree: {'Yes' if result.modes_agree else 'NO'}")
    print(f"      ✓ Final Recommendation: {result.final_recommendation}")
    print(f"      ✓ Risk Level: {result.risk_level}")

    # ========================================================================
    # Phase 4: Real-time Streaming (Optional)
    # ========================================================================
    if enable_realtime:
        print("\n" + "=" * 50)
        print("PHASE 4: REAL-TIME STREAMING")
        print("=" * 50)

        print(f"\n[4.1] Starting Binance WebSocket ({realtime_duration}s)...")
        try:
            config = PipelineConfig(symbols=['BTCUSDT'])
            pipeline = RealtimePipeline(config)

            # 간단한 테스트 실행
            from lib.binance_stream import BinanceStreamer

            signals_collected = []

            def on_metrics(metrics):
                signals_collected.append({
                    'timestamp': datetime.now().isoformat(),
                    'ofi': metrics.ofi,
                    'vpin': metrics.vpin,
                    'signal': metrics.signal
                })

            stream_config = StreamConfig(symbols=['BTCUSDT'])
            streamer = BinanceStreamer(config=stream_config, on_metrics=on_metrics, verbose=False)

            task = asyncio.create_task(streamer.start())
            await asyncio.sleep(realtime_duration)
            streamer.stop()

            try:
                await asyncio.wait_for(task, timeout=2)
            except:
                pass

            result.realtime_signals = signals_collected[-10:]  # 마지막 10개
            print(f"      ✓ Collected {len(signals_collected)} real-time signals")
            print(f"      ✓ Stats: {streamer.stats.depth_updates} depth, {streamer.stats.trades_received} trades")

            if signals_collected:
                avg_ofi = sum(s['ofi'] for s in signals_collected) / len(signals_collected)
                avg_vpin = sum(s['vpin'] for s in signals_collected) / len(signals_collected)
                print(f"      ✓ Avg OFI: {avg_ofi:.3f}, Avg VPIN: {avg_vpin:.3f}")

        except Exception as e:
            print(f"      ✗ Real-time error: {e}")

    # ========================================================================
    # Phase 5: Database Storage
    # ========================================================================
    print("\n" + "=" * 50)
    print("PHASE 5: DATABASE STORAGE")
    print("=" * 50)

    # 5.1 이벤트 DB 저장
    print("\n[5.1] Saving to Event Database...")
    try:
        event_db = EventDatabase('data/events.db')

        # 이벤트 저장
        for event in result.events_detected:
            event_db.save_detected_event({
                'event_type': event['type'],
                'importance': event['importance'],
                'description': event['description'],
                'timestamp': result.timestamp,
            })

        # 마켓 스냅샷 저장 (snapshot_id 필수)
        import uuid
        snapshot_id = str(uuid.uuid4())[:8]

        def get_latest_price(ticker: str) -> float:
            """DataFrame에서 최신 가격 추출"""
            if ticker not in market_data:
                return 0.0
            df = market_data[ticker]
            if hasattr(df, 'iloc') and len(df) > 0:
                return float(df['Close'].iloc[-1]) if 'Close' in df.columns else 0.0
            return 0.0

        event_db.save_market_snapshot({
            'snapshot_id': snapshot_id,
            'timestamp': result.timestamp,
            'spy_price': get_latest_price('SPY'),
            'spy_change_1d': 0.0,
            'spy_change_5d': 0.0,
            'spy_vs_ma20': 0.0,
            'qqq_price': get_latest_price('QQQ'),
            'iwm_price': get_latest_price('IWM'),
            'tlt_price': get_latest_price('TLT'),
            'gld_price': get_latest_price('GLD'),
            'vix_level': get_latest_price('^VIX'),
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

        print(f"      ✓ Saved {len(result.events_detected)} events")
        print(f"      ✓ Saved market snapshot (ID: {snapshot_id})")
    except Exception as e:
        print(f"      ✗ Event DB error: {e}")

    # 5.2 시그널 DB 저장
    print("\n[5.2] Saving to Signal Database...")
    try:
        signal_db = SignalDatabase('outputs/realtime_signals.db')

        from lib.realtime_pipeline import IntegratedSignal
        integrated_signal = IntegratedSignal(
            timestamp=datetime.now(),
            symbol='INTEGRATED',
            liquidity_regime=result.fred_summary.get('liquidity_regime', 'Unknown'),
            rrp_delta=result.fred_summary.get('rrp_delta', 0),
            tga_delta=result.fred_summary.get('tga_delta', 0),
            net_liquidity=result.fred_summary.get('net_liquidity', 0),
            macro_signal=result.liquidity_signal.lower(),
            ofi=0,
            vpin=0,
            depth_ratio=1.0,
            micro_signal='neutral',
            combined_signal=result.final_recommendation.lower(),
            confidence=result.confidence,
            action=result.final_recommendation.lower(),
            alerts=result.warnings
        )
        signal_db.save_signal(integrated_signal)
        print(f"      ✓ Saved integrated signal")
    except Exception as e:
        print(f"      ✗ Signal DB error: {e}")

    # 5.3 결과 JSON 저장
    print("\n[5.3] Saving JSON result...")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"integrated_{timestamp_str}.json"

    with open(output_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print(f"      ✓ Saved: {output_file}")

    return result, market_data, output_file


async def run_ai_report(
    analysis_result,
    market_data: Dict,
    output_file: str
) -> Optional[str]:
    """
    Phase 6: AI Report Generation
    """
    print("\n" + "=" * 50)
    print("PHASE 6: AI REPORT GENERATION")
    print("=" * 50)

    try:
        from lib.ai_report_generator import AIReportGenerator

        generator = AIReportGenerator(verbose=True)

        print("\n[6.1] Generating AI-powered investment report...")
        report = await generator.generate(analysis_result.to_dict(), market_data)

        print("\n[6.2] Saving report...")
        report_path = await generator.save_report(report)

        print(f"\n      ✓ AI Report saved: {report_path}")

        # 요약 출력
        print("\n" + "-" * 50)
        print("📝 AI REPORT HIGHLIGHTS")
        print("-" * 50)

        if report.notable_stocks:
            print("\n주목할 종목:")
            for stock in report.notable_stocks[:3]:
                print(f"  • {stock.ticker}: {stock.notable_reason}")

        print(f"\n최종 제안: {report.final_recommendation[:200]}...")

        if report.action_items:
            print("\n액션 아이템:")
            for item in report.action_items[:3]:
                print(f"  • {item}")

        return report_path

    except Exception as e:
        print(f"      ✗ AI Report error: {e}")
        return None


async def run_full_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    generate_report: bool = False
):
    """전체 파이프라인 실행 (분석 + 리포트)"""
    start_time = datetime.now()

    # Phase 1-5: 분석 실행
    result, market_data, output_file = await run_integrated_pipeline(
        enable_realtime=enable_realtime,
        realtime_duration=realtime_duration,
        quick_mode=quick_mode
    )

    # Phase 6: AI 리포트 생성 (옵션)
    report_path = None
    if generate_report:
        report_path = await run_ai_report(result, market_data, output_file)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("EIMAS INTEGRATED ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print()
    print("📊 DATA SUMMARY")
    print(f"   FRED: RRP=${result.fred_summary.get('rrp', 0):.0f}B, Net Liq=${result.fred_summary.get('net_liquidity', 0):.0f}B")
    print(f"   Market: {result.market_data_count} tickers, Crypto: {result.crypto_data_count} tickers")
    print()
    print("📈 ANALYSIS SUMMARY")
    print(f"   Regime: {result.regime.get('regime', 'Unknown')}")
    print(f"   Risk Score: {result.risk_score:.1f}/100")
    print(f"   Events: {len(result.events_detected)} detected")
    print()
    print("🤖 AGENT DEBATE")
    print(f"   FULL Mode: {result.full_mode_position}")
    print(f"   REFERENCE Mode: {result.reference_mode_position}")
    print(f"   Modes Agree: {'✓' if result.modes_agree else '✗'}")
    print(f"   Dissent Records: {len(result.dissent_records)}")
    print()
    print("🎯 FINAL RECOMMENDATION")
    print(f"   Action: {result.final_recommendation}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Risk Level: {result.risk_level}")

    if result.warnings:
        print()
        print("⚠️  WARNINGS")
        for w in result.warnings:
            print(f"   - {w}")

    print()
    print(f"📁 Results saved: {output_file}")
    if report_path:
        print(f"📝 AI Report: {report_path}")
    print("=" * 70)

    return result


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='EIMAS - Integrated Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python main_integrated.py              # 전체 파이프라인
    python main_integrated.py --report     # AI 제안서 생성 포함
    python main_integrated.py --quick --report  # 빠른 분석 + AI 제안서
    python main_integrated.py --realtime   # 실시간 스트리밍 포함 (30초)
    python main_integrated.py --realtime --duration 60  # 60초 스트리밍
        '''
    )

    parser.add_argument(
        '--realtime', '-r',
        action='store_true',
        help='Enable real-time Binance streaming'
    )

    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=30,
        help='Real-time streaming duration in seconds (default: 30)'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode (skip some analysis)'
    )

    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate AI-powered investment report (requires API keys)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = await run_full_pipeline(
        enable_realtime=args.realtime,
        realtime_duration=args.duration,
        quick_mode=args.quick,
        generate_report=args.report
    )

    return result


if __name__ == "__main__":
    asyncio.run(main())
