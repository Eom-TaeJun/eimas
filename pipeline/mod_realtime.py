#!/usr/bin/env python3
"""
EIMAS Pipeline Realtime
========================
Phase 4: 실시간 데이터 스트리밍 모듈
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any

# EIMAS 라이브러리
from lib.binance_stream import BinanceStreamer, StreamConfig
from lib.realtime_pipeline import PipelineConfig, RealtimePipeline
from pipeline.schemas import RealtimeSignal

async def run_realtime_stream(duration: int = 30, symbols: List[str] = None) -> List[RealtimeSignal]:
    """
    실시간 Binance WebSocket 스트리밍 실행
    
    Args:
        duration: 스트리밍 지속 시간 (초)
        symbols: 모니터링할 심볼 리스트 (예: ['BTCUSDT'])
        
    Returns:
        List[RealtimeSignal]: 수집된 실시간 시그널 리스트
    """
    if symbols is None:
        symbols = ['BTCUSDT']
        
    print("\n" + "=" * 50)
    print("PHASE 4: REAL-TIME STREAMING")
    print("=" * 50)
    print(f"\n[4.1] Starting Binance WebSocket ({duration}s for {symbols})...")
    
    signals_collected = []

    def on_metrics(metrics):
        """메트릭 수신 콜백"""
        # metrics 객체에서 필요한 정보 추출 (lib.binance_stream의 Metrics 객체 가정)
        signal = RealtimeSignal(
            timestamp=datetime.now().isoformat(),
            symbol=symbols[0], # 단순화를 위해 첫 번째 심볼 사용
            ofi=getattr(metrics, 'ofi', 0.0),
            vpin=getattr(metrics, 'vpin', 0.0),
            signal=getattr(metrics, 'signal', 'neutral')
        )
        signals_collected.append(signal)

    try:
        # 스트리머 설정
        stream_config = StreamConfig(symbols=symbols)
        streamer = BinanceStreamer(config=stream_config, on_metrics=on_metrics, verbose=False)

        # 스트리밍 시작 (백그라운드 태스크)
        task = asyncio.create_task(streamer.start())
        
        # 지정된 시간 동안 대기
        await asyncio.sleep(duration)
        
        # 스트리밍 중지
        streamer.stop()

        # 태스크 종료 대기
        try:
            await asyncio.wait_for(task, timeout=2)
        except asyncio.TimeoutError:
            pass

        print(f"      ✓ Collected {len(signals_collected)} real-time signals")
        if streamer.stats:
            print(f"      ✓ Stats: {streamer.stats.depth_updates} depth, {streamer.stats.trades_received} trades")

        if signals_collected:
            avg_ofi = sum(s.ofi for s in signals_collected) / len(signals_collected)
            avg_vpin = sum(s.vpin for s in signals_collected) / len(signals_collected)
            print(f"      ✓ Avg OFI: {avg_ofi:.3f}, Avg VPIN: {avg_vpin:.3f}")
            
        return signals_collected

    except Exception as e:
        print(f"      ✗ Real-time error: {e}")
        return []
