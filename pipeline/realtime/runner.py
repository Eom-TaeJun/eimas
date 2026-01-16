
import sys
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.binance_stream import BinanceStreamer, StreamConfig

logger = logging.getLogger('eimas.pipeline.realtime')

class RealtimeVPINMonitor:
    """
    실시간 VPIN 모니터링 시스템
    """
    VPIN_THRESHOLDS = {
        'normal': 0.4,
        'elevated': 0.5,
        'high': 0.6,
        'extreme': 0.7
    }

    def __init__(self, symbols: List[str] = None, verbose: bool = True):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.verbose = verbose
        self.minute_vpin: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.minute_start: Dict[str, datetime] = {}
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_cooldown = 30
        self.alerts_fired = 0
        self.vpin_history: Dict[str, List[Dict]] = {s: [] for s in self.symbols}

    def _log(self, msg: str, level: str = 'info'):
        if not self.verbose: return
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def _on_metrics(self, metrics):
        symbol = metrics.symbol
        vpin = metrics.vpin
        now = datetime.now()
        
        if symbol not in self.minute_start or self.minute_start[symbol] is None:
            self.minute_start[symbol] = now
        
        self.minute_vpin[symbol].append(vpin)
        
        elapsed = (now - self.minute_start[symbol]).total_seconds()
        if elapsed >= 60:
            if self.minute_vpin[symbol]:
                avg = sum(self.minute_vpin[symbol]) / len(self.minute_vpin[symbol])
                self.vpin_history[symbol].append({
                    'timestamp': now.isoformat(),
                    'avg_vpin': avg,
                    'max_vpin': max(self.minute_vpin[symbol]),
                    'samples': len(self.minute_vpin[symbol])
                })
                self._log(f"{symbol} 1-min VPIN: {avg:.3f}")
            self.minute_vpin[symbol] = []
            self.minute_start[symbol] = now

    def _on_alert(self, alert_type: str, alert_data: Dict):
        if alert_type == 'vpin_high':
            self._log(f"Stream Alert: {alert_data.get('message', '')}", level='warning')

    async def start(self, duration: int = 60):
        print("\n" + "=" * 70)
        print("  EIMAS Real-time VPIN Monitor")
        print("=" * 70)
        
        config = StreamConfig(symbols=self.symbols, depth_levels=10, update_speed='100ms', include_trades=True)
        streamer = BinanceStreamer(config=config, on_metrics=self._on_metrics, on_alert=self._on_alert, verbose=False)
        
        try:
            await streamer.start(duration_seconds=duration)
        except Exception as e:
            self._log(f"Error: {e}", level='warning')
        finally:
            streamer.stop()
            
        return {
            'alerts_fired': self.alerts_fired,
            'vpin_history': self.vpin_history,
            'stream_stats': streamer.stats.to_dict()
        }

async def run_realtime_monitor_pipeline(result: Any, enable_realtime: bool, duration: int) -> Any:
    """
    Phase 4: Real-time VPIN Monitoring
    """
    if not enable_realtime:
        return result

    print("\n" + "=" * 50)
    print("PHASE 4: REAL-TIME VPIN MONITORING")
    print("=" * 50)
    
    try:
        monitor = RealtimeVPINMonitor(symbols=['BTCUSDT', 'ETHUSDT'], verbose=True)
        monitor_result = await monitor.start(duration=duration)
        
        vpin_history = monitor_result.get('vpin_history', {})
        all_signals = []
        for symbol, history in vpin_history.items():
            for h in history:
                all_signals.append({'timestamp': h['timestamp'], 'symbol': symbol, 'avg_vpin': h['avg_vpin'], 'max_vpin': h['max_vpin']})
        
        result.realtime_signals = all_signals[-20:]
        print(f"\n[4.2] Real-time Summary: {len(all_signals)} samples collected")
    except Exception as e:
        print(f"      ✗ Real-time error: {e}")
        
    return result
