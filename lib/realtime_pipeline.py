"""
Real-time Data Pipeline
=======================
FRED 유동성 + 시장 마이크로스트럭처 통합 파이프라인

핵심 기능:
1. FRED 유동성 데이터 (RRP/TGA/Fed Assets) 주기적 수집
2. Binance 실시간 호가/체결 스트리밍
3. OFI/VPIN 실시간 계산
4. Macro-Micro 신호 통합
5. 데이터베이스 저장 및 알림

사용법:
    pipeline = RealtimePipeline(symbols=['BTCUSDT', 'ETHUSDT'])
    await pipeline.start()

Author: EIMAS Team
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import sqlite3

# 내부 모듈
from lib.fred_collector import FREDCollector, FREDSummary
from lib.microstructure import MicrostructureMetrics, MicrostructureAnalyzer
from lib.binance_stream import BinanceStreamer, StreamConfig
from lib.event_framework import EventType, EventImportance, Event, QuantitativeEventDetector


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class IntegratedSignal:
    """통합 신호"""
    timestamp: datetime
    symbol: str

    # Macro (FRED)
    liquidity_regime: str = "Normal"
    rrp_delta: float = 0.0
    tga_delta: float = 0.0
    net_liquidity: float = 0.0
    macro_signal: str = "neutral"

    # Micro (OFI/VPIN)
    ofi: float = 0.0
    vpin: float = 0.0
    depth_ratio: float = 1.0
    micro_signal: str = "neutral"

    # 통합 신호
    combined_signal: str = "neutral"  # strong_bullish, bullish, neutral, bearish, strong_bearish
    confidence: float = 0.0
    action: str = "hold"  # buy, sell, hold

    # 알림
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'liquidity_regime': self.liquidity_regime,
            'rrp_delta': self.rrp_delta,
            'net_liquidity': self.net_liquidity,
            'macro_signal': self.macro_signal,
            'ofi': self.ofi,
            'vpin': self.vpin,
            'depth_ratio': self.depth_ratio,
            'micro_signal': self.micro_signal,
            'combined_signal': self.combined_signal,
            'confidence': self.confidence,
            'action': self.action,
            'alerts': self.alerts
        }


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 심볼
    symbols: List[str] = field(default_factory=lambda: ['BTCUSDT'])

    # FRED 업데이트 간격 (초)
    fred_interval: int = 3600  # 1시간

    # 신호 임계값
    ofi_threshold: float = 0.3
    vpin_threshold: float = 0.7
    rrp_threshold: float = 50.0  # Billions

    # 데이터베이스
    db_path: str = "outputs/realtime_signals.db"

    # 알림
    enable_alerts: bool = True


# ============================================================================
# Database Manager
# ============================================================================

class SignalDatabase:
    """신호 데이터베이스"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 마이크로스트럭처 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS microstructure (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                mid_price REAL,
                ofi REAL,
                ofi_normalized REAL,
                vpin REAL,
                depth_ratio REAL,
                spread_bps REAL,
                signal TEXT,
                signal_strength REAL
            )
        ''')

        # 유동성 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS liquidity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rrp REAL,
                rrp_delta REAL,
                tga REAL,
                tga_delta REAL,
                fed_assets REAL,
                net_liquidity REAL,
                regime TEXT
            )
        ''')

        # 통합 신호 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrated_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                macro_signal TEXT,
                micro_signal TEXT,
                combined_signal TEXT,
                confidence REAL,
                action TEXT,
                alerts TEXT
            )
        ''')

        # 기본 인덱스
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_micro_ts ON microstructure(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_micro_symbol ON microstructure(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_liq_ts ON liquidity(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_ts ON integrated_signals(timestamp)')

        # 복합 인덱스 (성능 최적화)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_micro_ts_symbol ON microstructure(timestamp, symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_ts_symbol ON integrated_signals(timestamp, symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_action ON integrated_signals(action)')

        conn.commit()
        conn.close()

    def save_microstructure(self, metrics: MicrostructureMetrics):
        """마이크로스트럭처 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO microstructure
            (timestamp, symbol, mid_price, ofi, ofi_normalized, vpin, depth_ratio, spread_bps, signal, signal_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.timestamp.isoformat(),
            metrics.symbol,
            metrics.mid_price,
            metrics.ofi,
            metrics.ofi_normalized,
            metrics.vpin,
            metrics.depth_ratio,
            metrics.spread_bps,
            metrics.signal,
            metrics.signal_strength
        ))

        conn.commit()
        conn.close()

    def save_liquidity(self, summary: FREDSummary):
        """유동성 저장 (일별 1회만, UPSERT)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 오늘 날짜 추출
        today = datetime.now().date().isoformat()

        # 오늘 데이터가 이미 있는지 확인
        cursor.execute('''
            SELECT id FROM liquidity WHERE DATE(timestamp) = ?
        ''', (today,))
        existing = cursor.fetchone()

        if existing:
            # UPDATE 기존 레코드
            cursor.execute('''
                UPDATE liquidity
                SET timestamp=?, rrp=?, rrp_delta=?, tga=?, tga_delta=?,
                    fed_assets=?, net_liquidity=?, regime=?
                WHERE id=?
            ''', (
                summary.timestamp,
                summary.rrp,
                summary.rrp_delta,
                summary.tga,
                summary.tga_delta,
                summary.fed_assets,
                summary.net_liquidity,
                summary.liquidity_regime,
                existing[0]
            ))
        else:
            # INSERT 새 레코드
            cursor.execute('''
                INSERT INTO liquidity
                (timestamp, rrp, rrp_delta, tga, tga_delta, fed_assets, net_liquidity, regime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                summary.timestamp,
                summary.rrp,
                summary.rrp_delta,
                summary.tga,
                summary.tga_delta,
                summary.fed_assets,
                summary.net_liquidity,
                summary.liquidity_regime
            ))

        conn.commit()
        conn.close()

    def save_signal(self, signal: IntegratedSignal):
        """통합 신호 저장"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO integrated_signals
            (timestamp, symbol, macro_signal, micro_signal, combined_signal, confidence, action, alerts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp.isoformat(),
            signal.symbol,
            signal.macro_signal,
            signal.micro_signal,
            signal.combined_signal,
            signal.confidence,
            signal.action,
            json.dumps(signal.alerts)
        ))

        conn.commit()
        conn.close()

    def get_recent_signals(self, hours: int = 24) -> List[Dict]:
        """최근 신호 조회"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since = (datetime.now() - timedelta(hours=hours)).isoformat()

        cursor.execute('''
            SELECT * FROM integrated_signals
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 100
        ''', (since,))

        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]

        conn.close()
        return results


# ============================================================================
# Signal Generator
# ============================================================================

class SignalGenerator:
    """신호 생성기"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.event_detector = QuantitativeEventDetector()

    def generate_macro_signal(self, summary: FREDSummary) -> str:
        """매크로 신호 생성"""
        score = 0

        # RRP 변화 (-: 유동성 유입 = 불리시)
        if summary.rrp_delta < -self.config.rrp_threshold:
            score += 2
        elif summary.rrp_delta < -20:
            score += 1
        elif summary.rrp_delta > self.config.rrp_threshold:
            score -= 2
        elif summary.rrp_delta > 20:
            score -= 1

        # TGA 변화 (+: 유동성 흡수 = 베어리시)
        if summary.tga_delta > 50:
            score -= 1
        elif summary.tga_delta < -50:
            score += 1

        # 유동성 레짐
        if summary.liquidity_regime == "Abundant":
            score += 1
        elif summary.liquidity_regime == "Stressed":
            score -= 2
        elif summary.liquidity_regime == "Tight":
            score -= 1

        if score >= 2:
            return "bullish"
        elif score <= -2:
            return "bearish"
        return "neutral"

    def generate_micro_signal(self, metrics: MicrostructureMetrics) -> str:
        """마이크로 신호 생성"""
        return metrics.signal

    def combine_signals(
        self,
        macro_signal: str,
        micro_signal: str,
        fred_summary: FREDSummary,
        metrics: MicrostructureMetrics
    ) -> IntegratedSignal:
        """신호 통합"""
        now = datetime.now()

        # 점수 계산
        macro_score = {"bullish": 1, "neutral": 0, "bearish": -1}.get(macro_signal, 0)
        micro_score = {"bullish": 1, "neutral": 0, "bearish": -1}.get(micro_signal, 0)

        # 가중 평균 (매크로 40%, 마이크로 60%)
        total_score = macro_score * 0.4 + micro_score * 0.6

        # 통합 신호 결정
        if total_score >= 0.6:
            combined = "strong_bullish"
            action = "buy"
        elif total_score >= 0.3:
            combined = "bullish"
            action = "buy"
        elif total_score <= -0.6:
            combined = "strong_bearish"
            action = "sell"
        elif total_score <= -0.3:
            combined = "bearish"
            action = "sell"
        else:
            combined = "neutral"
            action = "hold"

        # 신뢰도 계산
        confidence = abs(total_score)

        # VPIN 높으면 신뢰도 낮춤 (변동성 높음)
        if metrics.vpin > 0.5:
            confidence *= 0.7

        # 알림 생성
        alerts = []

        # 유동성 이벤트
        liq_events = self.event_detector.detect_liquidity_events(
            rrp=fred_summary.rrp,
            rrp_delta=fred_summary.rrp_delta,
            rrp_delta_pct=fred_summary.rrp_delta_pct,
            tga=fred_summary.tga,
            tga_delta=fred_summary.tga_delta,
            net_liquidity=fred_summary.net_liquidity,
            liquidity_regime=fred_summary.liquidity_regime
        )
        for event in liq_events:
            alerts.append(f"[{event.importance.name}] {event.name}")

        # OFI 극단값
        if abs(metrics.ofi_normalized) > 0.7:
            direction = "매수" if metrics.ofi_normalized > 0 else "매도"
            alerts.append(f"OFI 극단: {direction} 압력 강함 ({metrics.ofi_normalized:+.2f})")

        # VPIN 높음
        if metrics.vpin > 0.6:
            alerts.append(f"VPIN 높음: 변동성 주의 ({metrics.vpin:.2f})")

        return IntegratedSignal(
            timestamp=now,
            symbol=metrics.symbol,
            liquidity_regime=fred_summary.liquidity_regime,
            rrp_delta=fred_summary.rrp_delta,
            tga_delta=fred_summary.tga_delta,
            net_liquidity=fred_summary.net_liquidity,
            macro_signal=macro_signal,
            ofi=metrics.ofi_normalized,
            vpin=metrics.vpin,
            depth_ratio=metrics.depth_ratio,
            micro_signal=micro_signal,
            combined_signal=combined,
            confidence=confidence,
            action=action,
            alerts=alerts
        )


# ============================================================================
# Real-time Pipeline
# ============================================================================

class RealtimePipeline:
    """
    실시간 파이프라인

    FRED + Binance WebSocket 통합
    """

    def __init__(
        self,
        config: PipelineConfig = None,
        on_signal: Callable[[IntegratedSignal], None] = None,
        verbose: bool = True
    ):
        self.config = config or PipelineConfig()
        self.on_signal = on_signal
        self.verbose = verbose

        # 컴포넌트
        self.db = SignalDatabase(self.config.db_path)
        self.signal_gen = SignalGenerator(self.config)
        self.fred_collector = None
        self.binance_streamer = None

        # 상태
        self.running = False
        self.fred_summary: Optional[FREDSummary] = None
        self.latest_signals: Dict[str, IntegratedSignal] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    async def start(self, duration_seconds: int = None):
        """
        파이프라인 시작

        Parameters:
        -----------
        duration_seconds : int
            실행 시간 (None = 무한)
        """
        self.running = True
        self._log("Starting Real-time Pipeline...")
        self._log(f"Symbols: {self.config.symbols}")

        # FRED 초기 수집
        await self._fetch_fred()

        # Binance 스트리머 설정
        stream_config = StreamConfig(
            symbols=self.config.symbols,
            depth_levels=10,
            update_speed='100ms',
            include_trades=True
        )

        self.binance_streamer = BinanceStreamer(
            config=stream_config,
            on_metrics=self._on_metrics,
            verbose=False
        )

        # 병렬 실행
        tasks = [
            asyncio.create_task(self.binance_streamer.start(duration_seconds)),
            asyncio.create_task(self._fred_update_loop(duration_seconds)),
            asyncio.create_task(self._signal_output_loop(duration_seconds))
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass

        self.running = False
        self._print_summary()

    async def _fetch_fred(self):
        """FRED 데이터 수집"""
        try:
            self._log("Fetching FRED liquidity data...")
            self.fred_collector = FREDCollector()
            self.fred_summary = self.fred_collector.collect_all()
            self.db.save_liquidity(self.fred_summary)
            self._log(f"  RRP: ${self.fred_summary.rrp:.0f}B, Regime: {self.fred_summary.liquidity_regime}")
        except Exception as e:
            self._log(f"FRED fetch error: {e}")

    async def _fred_update_loop(self, duration_seconds: int = None):
        """FRED 주기적 업데이트"""
        start_time = datetime.now()

        while self.running:
            await asyncio.sleep(self.config.fred_interval)

            if duration_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration_seconds:
                    break

            await self._fetch_fred()

    def _on_metrics(self, metrics: MicrostructureMetrics):
        """마이크로스트럭처 지표 수신 콜백"""
        try:
            # 초기화
            if not hasattr(self, '_metrics_count'):
                self._metrics_count = 0
                self._last_micro_save = datetime.now()
                self._micro_save_interval = 60  # 60초마다 저장

            self._metrics_count += 1

            # 시간 기반 저장 (60초마다)
            now = datetime.now()
            elapsed = (now - self._last_micro_save).total_seconds()

            if elapsed >= self._micro_save_interval:
                try:
                    self.db.save_microstructure(metrics)
                    self._last_micro_save = now
                    self._log(f"  [DB] Microstructure saved: {metrics.symbol} (OFI={metrics.ofi_normalized:.2f}, VPIN={metrics.vpin:.2f})")
                except Exception as e:
                    self._log(f"  [DB ERROR] Microstructure save failed: {e}")

            # 신호 생성 (FRED 데이터 있을 때만)
            if self.fred_summary:
                macro_signal = self.signal_gen.generate_macro_signal(self.fred_summary)
                micro_signal = self.signal_gen.generate_micro_signal(metrics)

                integrated = self.signal_gen.combine_signals(
                    macro_signal=macro_signal,
                    micro_signal=micro_signal,
                    fred_summary=self.fred_summary,
                    metrics=metrics
                )

                self.latest_signals[metrics.symbol] = integrated

                # 콜백
                if self.on_signal:
                    self.on_signal(integrated)

                # 중요 신호면 저장
                if integrated.combined_signal in ['strong_bullish', 'strong_bearish'] or integrated.alerts:
                    self.db.save_signal(integrated)

        except Exception as e:
            self._log(f"  [ERROR] _on_metrics failed: {e}")

    async def _signal_output_loop(self, duration_seconds: int = None):
        """주기적 신호 출력"""
        start_time = datetime.now()
        interval = 10  # 10초마다 출력

        while self.running:
            await asyncio.sleep(interval)

            if duration_seconds:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= duration_seconds:
                    break

            self._print_signals()

    def _print_signals(self):
        """현재 신호 출력"""
        if not self.latest_signals:
            return

        print("\n" + "=" * 70)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INTEGRATED SIGNALS")
        print("=" * 70)

        # Macro 상태
        if self.fred_summary:
            print(f"\n[MACRO] Regime: {self.fred_summary.liquidity_regime}")
            print(f"  RRP: ${self.fred_summary.rrp:.0f}B ({self.fred_summary.rrp_delta:+.0f}B)")
            print(f"  Net Liquidity: ${self.fred_summary.net_liquidity/1000:.2f}T")

        # 심볼별 신호
        for symbol, signal in self.latest_signals.items():
            icon = {
                'strong_bullish': '🟢🟢',
                'bullish': '🟢',
                'neutral': '⚪',
                'bearish': '🔴',
                'strong_bearish': '🔴🔴'
            }.get(signal.combined_signal, '⚪')

            print(f"\n[{symbol}] {icon} {signal.combined_signal.upper()}")
            print(f"  Macro: {signal.macro_signal} | Micro: {signal.micro_signal}")
            print(f"  OFI: {signal.ofi:+.2f} | VPIN: {signal.vpin:.2f} | Depth: {signal.depth_ratio:.2f}")
            print(f"  Confidence: {signal.confidence:.2f} | Action: {signal.action.upper()}")

            if signal.alerts:
                print(f"  Alerts:")
                for alert in signal.alerts[:3]:
                    print(f"    - {alert}")

        print("=" * 70)

    def _print_summary(self):
        """최종 요약"""
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)

        for symbol, signal in self.latest_signals.items():
            print(f"\n[{symbol}]")
            print(f"  Final Signal: {signal.combined_signal.upper()}")
            print(f"  Action: {signal.action.upper()}")
            print(f"  Confidence: {signal.confidence:.2f}")

        print("\n" + "=" * 70)

    def stop(self):
        """파이프라인 중지"""
        self.running = False
        if self.binance_streamer:
            self.binance_streamer.stop()


# ============================================================================
# Convenience Functions
# ============================================================================

async def run_pipeline(
    symbols: List[str] = None,
    duration: int = 60,
    verbose: bool = True
) -> Dict[str, IntegratedSignal]:
    """
    파이프라인 실행

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
    Dict : 심볼별 최종 신호
    """
    if symbols is None:
        symbols = ['BTCUSDT']

    config = PipelineConfig(symbols=symbols)
    pipeline = RealtimePipeline(config=config, verbose=verbose)

    await pipeline.start(duration_seconds=duration)

    return pipeline.latest_signals


def get_current_signals(db_path: str = "outputs/realtime_signals.db") -> List[Dict]:
    """저장된 신호 조회"""
    db = SignalDatabase(db_path)
    return db.get_recent_signals(hours=24)


# ============================================================================
# Test
# ============================================================================

async def test_pipeline():
    """파이프라인 테스트"""
    print("=" * 70)
    print("Real-time Pipeline Test")
    print("=" * 70)

    print("\nRunning pipeline for 30 seconds...")
    print("(FRED + Binance WebSocket integration)\n")

    signals = await run_pipeline(
        symbols=['BTCUSDT'],
        duration=30,
        verbose=True
    )

    print("\n[Final Results]")
    for symbol, signal in signals.items():
        print(f"\n{symbol}:")
        print(f"  Combined Signal: {signal.combined_signal}")
        print(f"  Macro: {signal.macro_signal}, Micro: {signal.micro_signal}")
        print(f"  Action: {signal.action} (confidence: {signal.confidence:.2f})")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
