"""
Binance WebSocket Streaming Module
==================================
실시간 호가창/체결 데이터 스트리밍 및 마이크로스트럭처 분석

주요 기능:
1. 실시간 호가창 (Depth) 스트리밍
2. 실시간 체결 (Trade) 스트리밍
3. OFI/VPIN 실시간 계산
4. 신호 생성 및 알림
5. 데이터베이스 저장

사용법:
    streamer = BinanceStreamer(symbols=['BTCUSDT', 'ETHUSDT'])
    await streamer.start()

Author: EIMAS Team
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from lib.microstructure import (
    OrderBook, OrderBookLevel, Trade,
    MicrostructureAnalyzer, MicrostructureMetrics
)


# ============================================================================
# Constants
# ============================================================================

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws"
BINANCE_WS_STREAM_URL = "wss://stream.binance.com:9443/stream"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class StreamConfig:
    """스트림 설정"""
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT'])
    depth_levels: int = 10       # 호가 레벨 수
    update_speed: str = '100ms'  # 업데이트 속도 (100ms or 1000ms)
    include_trades: bool = True  # 체결 포함 여부

    # 분석 설정
    ofi_levels: int = 5
    vpin_bucket_size: float = 1.0  # BTC 기준
    vpin_n_buckets: int = 50

    # 알림 설정
    alert_ofi_threshold: float = 0.5   # OFI 임계값
    alert_vpin_threshold: float = 0.7  # VPIN 임계값


@dataclass
class StreamStats:
    """스트림 통계"""
    start_time: datetime = field(default_factory=datetime.now)
    messages_received: int = 0
    depth_updates: int = 0
    trades_received: int = 0
    errors: int = 0
    last_update: Optional[datetime] = None

    def to_dict(self) -> Dict:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'elapsed_seconds': elapsed,
            'messages_received': self.messages_received,
            'depth_updates': self.depth_updates,
            'trades_received': self.trades_received,
            'errors': self.errors,
            'messages_per_second': self.messages_received / max(elapsed, 1)
        }


# ============================================================================
# Binance WebSocket Streamer
# ============================================================================

class BinanceStreamer:
    """
    Binance WebSocket 스트리머

    실시간 호가창/체결 데이터 수신 및 분석
    """

    def __init__(
        self,
        config: StreamConfig = None,
        on_metrics: Callable[[MicrostructureMetrics], None] = None,
        on_alert: Callable[[str, Dict], None] = None,
        verbose: bool = True
    ):
        """
        Parameters:
        -----------
        config : StreamConfig
            스트림 설정
        on_metrics : Callable
            지표 업데이트 콜백
        on_alert : Callable
            알림 콜백
        verbose : bool
            상세 출력 여부
        """
        self.config = config or StreamConfig()
        self.on_metrics = on_metrics
        self.on_alert = on_alert
        self.verbose = verbose

        # 심볼별 분석기
        self.analyzers: Dict[str, MicrostructureAnalyzer] = {}
        for symbol in self.config.symbols:
            self.analyzers[symbol] = MicrostructureAnalyzer(
                ofi_levels=self.config.ofi_levels,
                vpin_bucket_size=self.config.vpin_bucket_size,
                vpin_n_buckets=self.config.vpin_n_buckets
            )

        # 상태
        self.running = False
        self.stats = StreamStats()
        self.latest_metrics: Dict[str, MicrostructureMetrics] = {}

        # WebSocket
        self.ws = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _build_stream_url(self) -> str:
        """스트림 URL 생성"""
        streams = []

        for symbol in self.config.symbols:
            symbol_lower = symbol.lower()

            # 호가창 스트림
            depth_stream = f"{symbol_lower}@depth{self.config.depth_levels}@{self.config.update_speed}"
            streams.append(depth_stream)

            # 체결 스트림
            if self.config.include_trades:
                trade_stream = f"{symbol_lower}@trade"
                streams.append(trade_stream)

        # Combined stream URL
        stream_param = "/".join(streams)
        return f"{BINANCE_WS_STREAM_URL}?streams={stream_param}"

    async def start(self, duration_seconds: int = None):
        """
        스트리밍 시작

        Parameters:
        -----------
        duration_seconds : int
            실행 시간 (None = 무한)
        """
        self.running = True
        self.stats = StreamStats()

        url = self._build_stream_url()
        self._log(f"Connecting to Binance WebSocket...")
        self._log(f"Symbols: {self.config.symbols}")

        try:
            async with websockets.connect(url) as ws:
                self.ws = ws
                self._log("Connected!")

                start_time = datetime.now()

                while self.running:
                    # 시간 체크
                    if duration_seconds:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed >= duration_seconds:
                            break

                    try:
                        # 메시지 수신 (타임아웃 5초)
                        msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        await self._process_message(msg)

                    except asyncio.TimeoutError:
                        # 타임아웃은 정상
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        self._log("Connection closed, reconnecting...")
                        break
                    except Exception as e:
                        self.stats.errors += 1
                        self._log(f"Error: {e}")

        except Exception as e:
            self._log(f"Connection error: {e}")
            self.stats.errors += 1

        self.running = False
        self._log("Stream stopped")
        self._print_summary()

    async def _process_message(self, raw_msg: str):
        """메시지 처리"""
        self.stats.messages_received += 1
        self.stats.last_update = datetime.now()

        try:
            data = json.loads(raw_msg)

            # Combined stream format
            if 'stream' in data:
                stream_name = data['stream']
                payload = data['data']

                if '@depth' in stream_name:
                    await self._process_depth(stream_name, payload)
                elif '@trade' in stream_name:
                    await self._process_trade(stream_name, payload)

        except json.JSONDecodeError as e:
            self.stats.errors += 1

    async def _process_depth(self, stream_name: str, data: Dict):
        """호가창 업데이트 처리"""
        self.stats.depth_updates += 1

        # 심볼 추출 (btcusdt@depth10@100ms -> BTCUSDT)
        symbol = stream_name.split('@')[0].upper()

        if symbol not in self.analyzers:
            return

        # OrderBook 생성
        bids = [
            OrderBookLevel(price=float(b[0]), quantity=float(b[1]), side='bid')
            for b in data.get('bids', [])
        ]
        asks = [
            OrderBookLevel(price=float(a[0]), quantity=float(a[1]), side='ask')
            for a in data.get('asks', [])
        ]

        orderbook = OrderBook(
            symbol=symbol,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

        # 분석
        metrics = self.analyzers[symbol].process_orderbook(orderbook)
        self.latest_metrics[symbol] = metrics

        # 콜백
        if self.on_metrics:
            self.on_metrics(metrics)

        # 알림 체크
        await self._check_alerts(metrics)

        # 주기적 출력 (100회마다)
        if self.verbose and self.stats.depth_updates % 100 == 0:
            self._print_metrics(metrics)

    async def _process_trade(self, stream_name: str, data: Dict):
        """체결 처리"""
        self.stats.trades_received += 1

        symbol = stream_name.split('@')[0].upper()

        if symbol not in self.analyzers:
            return

        # Trade 생성
        trade = Trade(
            symbol=symbol,
            timestamp=datetime.fromtimestamp(data['T'] / 1000),
            price=float(data['p']),
            quantity=float(data['q']),
            side='buy' if data['m'] is False else 'sell'  # m=True means seller is maker
        )

        # VPIN에 추가
        self.analyzers[symbol].process_trade(trade)

    async def _check_alerts(self, metrics: MicrostructureMetrics):
        """알림 체크"""
        alerts = []

        # OFI 알림
        if abs(metrics.ofi_normalized) > self.config.alert_ofi_threshold:
            direction = "BULLISH" if metrics.ofi_normalized > 0 else "BEARISH"
            alerts.append({
                'type': 'ofi_extreme',
                'symbol': metrics.symbol,
                'value': metrics.ofi_normalized,
                'direction': direction,
                'message': f"{metrics.symbol} OFI {direction}: {metrics.ofi_normalized:+.2f}"
            })

        # VPIN 알림
        if metrics.vpin > self.config.alert_vpin_threshold:
            alerts.append({
                'type': 'vpin_high',
                'symbol': metrics.symbol,
                'value': metrics.vpin,
                'message': f"{metrics.symbol} High VPIN: {metrics.vpin:.2f} - 변동성 주의"
            })

        # Depth 불균형 알림
        if metrics.depth_ratio > 2.0:
            alerts.append({
                'type': 'depth_imbalance',
                'symbol': metrics.symbol,
                'value': metrics.depth_ratio,
                'direction': 'BID_WALL',
                'message': f"{metrics.symbol} Strong Bid Wall: {metrics.depth_ratio:.2f}x"
            })
        elif metrics.depth_ratio < 0.5:
            alerts.append({
                'type': 'depth_imbalance',
                'symbol': metrics.symbol,
                'value': metrics.depth_ratio,
                'direction': 'ASK_WALL',
                'message': f"{metrics.symbol} Strong Ask Wall: {metrics.depth_ratio:.2f}x"
            })

        # 알림 전송
        for alert in alerts:
            if self.on_alert:
                self.on_alert(alert['type'], alert)
            if self.verbose:
                self._log(f"⚠️ ALERT: {alert['message']}")

    def _print_metrics(self, metrics: MicrostructureMetrics):
        """지표 출력"""
        signal_icon = {
            'bullish': '🟢',
            'bearish': '🔴',
            'neutral': '⚪'
        }.get(metrics.signal, '⚪')

        print(
            f"[{metrics.timestamp.strftime('%H:%M:%S')}] {metrics.symbol} | "
            f"${metrics.mid_price:,.2f} | "
            f"OFI: {metrics.ofi_normalized:+.2f} | "
            f"VPIN: {metrics.vpin:.2f} | "
            f"Depth: {metrics.depth_ratio:.2f} | "
            f"{signal_icon} {metrics.signal.upper()}"
        )

    def _print_summary(self):
        """요약 출력"""
        stats = self.stats.to_dict()
        print("\n" + "=" * 60)
        print("Stream Summary")
        print("=" * 60)
        print(f"  Duration: {stats['elapsed_seconds']:.1f}s")
        print(f"  Messages: {stats['messages_received']:,}")
        print(f"  Depth Updates: {stats['depth_updates']:,}")
        print(f"  Trades: {stats['trades_received']:,}")
        print(f"  Errors: {stats['errors']}")
        print(f"  Rate: {stats['messages_per_second']:.1f} msg/s")

        for symbol, metrics in self.latest_metrics.items():
            print(f"\n  [{symbol}]")
            print(f"    Last Price: ${metrics.mid_price:,.2f}")
            print(f"    OFI: {metrics.ofi_normalized:+.3f}")
            print(f"    VPIN: {metrics.vpin:.3f}")
            print(f"    Depth Ratio: {metrics.depth_ratio:.3f}")
            print(f"    Signal: {metrics.signal.upper()}")

        print("=" * 60)

    def stop(self):
        """스트림 중지"""
        self.running = False

    def get_latest_metrics(self, symbol: str = None) -> Optional[MicrostructureMetrics]:
        """최신 지표 조회"""
        if symbol:
            return self.latest_metrics.get(symbol)
        return self.latest_metrics

    def get_analyzer(self, symbol: str) -> Optional[MicrostructureAnalyzer]:
        """분석기 조회"""
        return self.analyzers.get(symbol)


# ============================================================================
# Simple Streamer (단일 심볼)
# ============================================================================

async def stream_symbol(
    symbol: str = 'BTCUSDT',
    duration: int = 30,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    단일 심볼 스트리밍

    Parameters:
    -----------
    symbol : str
        심볼 (예: BTCUSDT)
    duration : int
        실행 시간 (초)
    verbose : bool
        출력 여부

    Returns:
    --------
    Dict : 분석 결과
    """
    config = StreamConfig(
        symbols=[symbol],
        depth_levels=10,
        update_speed='100ms',
        include_trades=True
    )

    results = {'metrics': [], 'alerts': []}

    def on_metrics(m):
        results['metrics'].append(m.to_dict())

    def on_alert(alert_type, data):
        results['alerts'].append(data)

    streamer = BinanceStreamer(
        config=config,
        on_metrics=on_metrics if not verbose else None,
        on_alert=on_alert,
        verbose=verbose
    )

    await streamer.start(duration_seconds=duration)

    # 최종 결과
    final_metrics = streamer.get_latest_metrics(symbol)
    if final_metrics:
        results['final'] = final_metrics.to_dict()
        results['summary'] = {
            'symbol': symbol,
            'duration': duration,
            'ofi': final_metrics.ofi_normalized,
            'vpin': final_metrics.vpin,
            'depth_ratio': final_metrics.depth_ratio,
            'signal': final_metrics.signal,
            'alert_count': len(results['alerts'])
        }

    return results


# ============================================================================
# Multi-Symbol Streamer
# ============================================================================

async def stream_multi(
    symbols: List[str] = None,
    duration: int = 60,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    다중 심볼 스트리밍

    Parameters:
    -----------
    symbols : List[str]
        심볼 목록
    duration : int
        실행 시간 (초)
    verbose : bool
        출력 여부

    Returns:
    --------
    Dict : 심볼별 분석 결과
    """
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT']

    config = StreamConfig(
        symbols=symbols,
        depth_levels=10,
        update_speed='100ms',
        include_trades=True
    )

    streamer = BinanceStreamer(config=config, verbose=verbose)
    await streamer.start(duration_seconds=duration)

    results = {}
    for symbol in symbols:
        metrics = streamer.get_latest_metrics(symbol)
        if metrics:
            analyzer = streamer.get_analyzer(symbol)
            results[symbol] = {
                'final': metrics.to_dict(),
                'vpin': metrics.vpin,
                'ofi': metrics.ofi_normalized,
                'signal': metrics.signal
            }

    return results


# ============================================================================
# Test
# ============================================================================

async def test_stream():
    """테스트 스트림"""
    print("=" * 60)
    print("Binance WebSocket Streaming Test")
    print("=" * 60)

    # 짧은 테스트 (10초)
    print("\n[1] Single Symbol Test (BTCUSDT, 10s)")
    print("-" * 40)

    result = await stream_symbol('BTCUSDT', duration=10, verbose=True)

    if 'summary' in result:
        print("\n[Summary]")
        for k, v in result['summary'].items():
            print(f"  {k}: {v}")

    print("\n[2] Multi-Symbol Test (BTC, ETH, 10s)")
    print("-" * 40)

    results = await stream_multi(['BTCUSDT', 'ETHUSDT'], duration=10, verbose=True)

    print("\n[Final Metrics]")
    for symbol, data in results.items():
        print(f"\n  {symbol}:")
        print(f"    OFI: {data['ofi']:+.3f}")
        print(f"    VPIN: {data['vpin']:.3f}")
        print(f"    Signal: {data['signal'].upper()}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_stream())
