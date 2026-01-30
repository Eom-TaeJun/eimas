import sys
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lib.binance_stream import BinanceStreamer, StreamConfig
# Import updated Microstructure tools
from lib.microstructure import MicrostructureMetrics

logger = logging.getLogger('eimas.pipeline.realtime')

class LiquidityRiskMonitor:
    """
    실시간 유동성 리스크 모니터링 시스템
    
    통합 지표:
    1. VPIN (Volume-Synchronized Probability of Informed Trading)
    2. OFI (Order Flow Imbalance)
    3. Volume Anomaly (3-sigma)
    """
    
    RISK_THRESHOLDS = {
        'low': 20,
        'moderate': 40,
        'high': 60,
        'extreme': 80
    }

    def __init__(self, symbols: List[str] = None, verbose: bool = True):
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.verbose = verbose
        
        # 상태 저장소
        self.minute_metrics: Dict[str, List[MicrostructureMetrics]] = {s: [] for s in self.symbols}
        self.minute_start: Dict[str, datetime] = {}
        self.risk_history: Dict[str, List[Dict]] = {s: [] for s in self.symbols}
        
        # 이상 거래량 카운터
        self.volume_anomalies: Dict[str, int] = {s: 0 for s in self.symbols}

    def _log(self, msg: str, level: str = 'info'):
        if not self.verbose: return
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    def calculate_liquidity_risk_score(self, metrics: MicrostructureMetrics, anomalies: int) -> float:
        """
        유동성 리스크 점수 계산 (0-100)
        
        Components:
        - VPIN (40%): 정보 비대칭성
        - OFI (30%): 주문 흐름 불균형
        - Depth Ratio (20%): 호가창 불균형
        - Volume Anomaly (10%): 이상 거래량
        """
        score = 0.0
        
        # 1. VPIN (0~1 -> 0~100)
        score += metrics.vpin * 40
        
        # 2. OFI (Normalized -1~1 -> 절대값 * 30)
        score += abs(metrics.ofi_normalized) * 30
        
        # 3. Depth Ratio (1.0 기준 이탈 정도)
        # 0.5 ~ 1.5 범위를 벗어나면 리스크 증가
        depth_deviation = abs(metrics.depth_ratio - 1.0)
        score += min(depth_deviation * 20, 20)
        
        # 4. Volume Anomaly (건당 5점, 최대 10점)
        score += min(anomalies * 5, 10)
        
        return min(score, 100.0)

    def _on_metrics(self, metrics: MicrostructureMetrics):
        """MicrostructureAnalyzer에서 지표 수신"""
        symbol = metrics.symbol
        now = datetime.now()
        
        if symbol not in self.minute_start or self.minute_start[symbol] is None:
            self.minute_start[symbol] = now
            
        self.minute_metrics[symbol].append(metrics)
        
        # 1분 집계
        elapsed = (now - self.minute_start[symbol]).total_seconds()
        if elapsed >= 60:
            if self.minute_metrics[symbol]:
                # 평균 지표 계산
                avg_vpin = sum(m.vpin for m in self.minute_metrics[symbol]) / len(self.minute_metrics[symbol])
                avg_ofi = sum(m.ofi_normalized for m in self.minute_metrics[symbol]) / len(self.minute_metrics[symbol])
                last_metric = self.minute_metrics[symbol][-1]
                
                # 리스크 점수 계산
                anomalies = self.volume_anomalies[symbol]
                risk_score = self.calculate_liquidity_risk_score(last_metric, anomalies)
                
                # 기록
                record = {
                    'timestamp': now.isoformat(),
                    'risk_score': round(risk_score, 2),
                    'vpin': round(avg_vpin, 3),
                    'ofi': round(avg_ofi, 3),
                    'volume_anomalies': anomalies
                }
                self.risk_history[symbol].append(record)
                
                # 로그 출력
                risk_level = "LOW"
                if risk_score > self.RISK_THRESHOLDS['extreme']: risk_level = "EXTREME 🔴"
                elif risk_score > self.RISK_THRESHOLDS['high']: risk_level = "HIGH 🟠"
                elif risk_score > self.RISK_THRESHOLDS['moderate']: risk_level = "MODERATE 🟡"
                elif risk_score > self.RISK_THRESHOLDS['low']: risk_level = "LOW 🟢"
                
                self._log(f"{symbol} Risk Score: {risk_score:.1f} [{risk_level}] | VPIN: {avg_vpin:.2f} | OFI: {avg_ofi:.2f}")
                
            # 초기화
            self.minute_metrics[symbol] = []
            self.volume_anomalies[symbol] = 0
            self.minute_start[symbol] = now

    def _on_alert(self, alert_type: str, alert_data: Dict):
        """스트림 알림 처리"""
        if alert_type == 'volume_anomaly':
            symbol = alert_data.get('symbol')
            if symbol in self.volume_anomalies:
                self.volume_anomalies[symbol] += 1
                self._log(f"🚨 Volume Anomaly Detected: {symbol} (Z-Score: {alert_data.get('z_score', 0):.2f})", level='warning')

    async def start(self, duration: int = 60):
        print("\n" + "=" * 70)
        print("  EIMAS Real-time Liquidity Risk Monitor")
        print("=" * 70)
        print(f"  Duration: {duration}s")
        print(f"  Assets: {', '.join(self.symbols)}")
        print("-" * 70)
        
        # BinanceStreamer 설정
        # include_trades=True -> Volume Anomaly 감지
        config = StreamConfig(
            symbols=self.symbols, 
            depth_levels=10, 
            update_speed='100ms', 
            include_trades=True
        )
        
        streamer = BinanceStreamer(
            config=config, 
            on_metrics=self._on_metrics, 
            on_alert=self._on_alert, 
            verbose=False
        )
        
        try:
            await streamer.start(duration_seconds=duration)
        except Exception as e:
            self._log(f"Error: {e}", level='warning')
        finally:
            streamer.stop()
            
        return {
            'risk_history': self.risk_history,
            'stream_stats': streamer.stats.to_dict()
        }

async def run_realtime_monitor_pipeline(result: Any, enable_realtime: bool, duration: int) -> Any:
    """
    Phase 4: Real-time Liquidity Risk Monitoring
    """
    if not enable_realtime:
        return result

    print("\n" + "=" * 50)
    print("PHASE 4: REAL-TIME LIQUIDITY RISK MONITORING")
    print("=" * 50)
    
    try:
        # 주요 자산 모니터링
        monitor = LiquidityRiskMonitor(symbols=['BTCUSDT', 'ETHUSDT'], verbose=True)
        monitor_result = await monitor.start(duration=duration)
        
        # 결과 저장
        risk_history = monitor_result.get('risk_history', {})
        all_signals = []
        for symbol, history in risk_history.items():
            for h in history:
                all_signals.append({
                    'timestamp': h['timestamp'], 
                    'symbol': symbol, 
                    'risk_score': h['risk_score'],
                    'vpin': h['vpin']
                })
        
        # 최근 신호 저장
        if hasattr(result, 'realtime_signals'):
            result.realtime_signals = all_signals[-20:]
        
        print(f"\n[4.2] Real-time Summary: {len(all_signals)} risk scores calculated")
        
    except Exception as e:
        print(f"      ✗ Real-time error: {e}")
        import traceback
        traceback.print_exc()
        
    return result