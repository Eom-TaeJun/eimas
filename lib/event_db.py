#!/usr/bin/env python3
"""
EIMAS Event Database
====================
이벤트, 예측, 시장 상태 저장 및 관리

저장 대상:
✅ DB 저장 (영구 보존)
  - detected_events: 감지된 시장 이벤트
  - event_predictions: 이벤트 예측 및 결과
  - market_snapshots: 시장 상태 스냅샷
  - prediction_outcomes: 예측 정확도 추적

❌ DB 저장 안함 (임시/휘발성)
  - 실시간 가격 캐시 (재조회 가능)
  - 중간 계산 결과 (재계산 가능)
  - 설정/파라미터 (파일로 관리)

사용법:
    from lib.event_db import EventDatabase, AutoSaveSession

    # 일반 사용
    db = EventDatabase()
    db.save_detected_event(event_dict)
    db.save_prediction(prediction_dict)

    # 자동 저장 세션 (종료 시 자동 저장)
    with AutoSaveSession() as session:
        session.collect_market_data()
        session.run_event_detection()
        # 종료 시 자동 저장
"""

import sqlite3
import json
import atexit
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import threading

# DB 경로
EVENT_DB_PATH = Path(__file__).parent.parent / "data" / "events.db"


# ============================================================================
# Database Schema
# ============================================================================

SCHEMA = """
-- ============================================================================
-- 감지된 이벤트 테이블
-- ============================================================================
CREATE TABLE IF NOT EXISTS detected_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    event_date TEXT NOT NULL,
    event_type TEXT NOT NULL,           -- volume_spike, price_shock, etc.
    ticker TEXT NOT NULL,
    asset_class TEXT,                    -- equity, bond, commodity, etc.
    importance TEXT,                     -- LOW, MEDIUM, HIGH, CRITICAL

    -- 정량 데이터
    value REAL,                          -- 주요 수치 (z-score, % 등)
    price_change REAL,
    volume_ratio REAL,

    -- 설명
    name TEXT,
    description TEXT,

    -- 원인 분석
    attributed_cause TEXT,
    confidence REAL,

    -- 메타데이터
    metadata_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_date ON detected_events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_ticker ON detected_events(ticker);
CREATE INDEX IF NOT EXISTS idx_events_type ON detected_events(event_type);

-- ============================================================================
-- 이벤트 예측 테이블
-- ============================================================================
CREATE TABLE IF NOT EXISTS event_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,            -- fomc, cpi, nfp, etc.
    event_date TEXT NOT NULL,

    -- 예측 시점 시장 상태
    spy_price REAL,
    vix_level REAL,
    rsi_14 REAL,
    market_trend TEXT,

    -- 예측 결과
    predicted_t1_return REAL,
    predicted_t5_return REAL,
    predicted_direction TEXT,            -- bullish, bearish, neutral
    confidence REAL,

    -- 시나리오별 예측 (JSON)
    scenarios_json TEXT,

    -- 가격 목표
    target_t1 REAL,
    target_t5 REAL,

    -- 권고
    recommendation TEXT
);

CREATE INDEX IF NOT EXISTS idx_predictions_event_date ON event_predictions(event_date);
CREATE INDEX IF NOT EXISTS idx_predictions_event_type ON event_predictions(event_type);

-- ============================================================================
-- 예측 결과 추적 테이블
-- ============================================================================
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    event_date TEXT NOT NULL,

    -- 실제 결과
    actual_t1_return REAL,
    actual_t5_return REAL,
    actual_direction TEXT,

    -- 정확도
    t1_error REAL,                       -- predicted - actual
    t5_error REAL,
    direction_correct INTEGER,           -- 0 or 1

    -- 서프라이즈 정보
    surprise_type TEXT,                  -- hawkish, dovish, hot, cool, etc.
    surprise_magnitude REAL,

    recorded_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (prediction_id) REFERENCES event_predictions(prediction_id)
);

CREATE INDEX IF NOT EXISTS idx_outcomes_prediction ON prediction_outcomes(prediction_id);

-- ============================================================================
-- 시장 스냅샷 테이블
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,

    -- 가격 데이터
    spy_price REAL,
    spy_change_1d REAL,
    spy_change_5d REAL,
    spy_vs_ma20 REAL,

    qqq_price REAL,
    iwm_price REAL,
    tlt_price REAL,
    gld_price REAL,

    -- 변동성
    vix_level REAL,
    vix_percentile REAL,

    -- 기술적 지표
    rsi_14 REAL,
    macd_signal TEXT,                    -- bullish, bearish, neutral

    -- 시장 상태
    trend TEXT,                          -- strong_up, up, neutral, down, strong_down
    volatility_regime TEXT,              -- low, normal, elevated, high

    -- 센티먼트
    put_call_ratio REAL,
    fear_greed_index REAL,

    -- 다음 이벤트
    days_to_fomc INTEGER,
    days_to_cpi INTEGER,
    days_to_nfp INTEGER,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON market_snapshots(timestamp);
"""


# ============================================================================
# Event Database Class
# ============================================================================

class EventDatabase:
    """이벤트 데이터베이스 관리"""

    def __init__(self, db_path: str = None, verbose: bool = True):
        self.db_path = Path(db_path) if db_path else EVENT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self._init_tables()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[EventDB] {msg}")

    @contextmanager
    def _get_connection(self):
        """연결 컨텍스트 매니저"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_tables(self):
        """테이블 초기화"""
        with self._get_connection() as conn:
            conn.executescript(SCHEMA)
        self._log(f"Database initialized: {self.db_path}")

    # ========================================================================
    # 이벤트 저장/조회
    # ========================================================================

    def save_detected_event(self, event: Dict[str, Any]) -> bool:
        """감지된 이벤트 저장"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO detected_events (
                        event_id, event_date, event_type, ticker, asset_class,
                        importance, value, price_change, volume_ratio,
                        name, description, attributed_cause, confidence, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.get('event_id'),
                    event.get('event_date'),
                    event.get('event_type'),
                    event.get('ticker'),
                    event.get('asset_class'),
                    event.get('importance'),
                    event.get('value'),
                    event.get('price_change'),
                    event.get('volume_ratio'),
                    event.get('name'),
                    event.get('description'),
                    event.get('attributed_cause'),
                    event.get('confidence'),
                    json.dumps(event.get('metadata', {}))
                ))
                return True
            except Exception as e:
                self._log(f"Error saving event: {e}")
                return False

    def save_detected_events(self, events: List[Dict]) -> int:
        """여러 이벤트 일괄 저장"""
        saved = 0
        for event in events:
            if self.save_detected_event(event):
                saved += 1
        self._log(f"Saved {saved}/{len(events)} events")
        return saved

    def get_events(
        self,
        start_date: str = None,
        end_date: str = None,
        ticker: str = None,
        event_type: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """이벤트 조회"""
        query = "SELECT * FROM detected_events WHERE 1=1"
        params = []

        if start_date:
            query += " AND event_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND event_date <= ?"
            params.append(end_date)
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += f" ORDER BY event_date DESC LIMIT {limit}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # 예측 저장/조회
    # ========================================================================

    def save_prediction(self, prediction: Dict[str, Any]) -> bool:
        """예측 저장"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO event_predictions (
                        prediction_id, created_at, event_type, event_date,
                        spy_price, vix_level, rsi_14, market_trend,
                        predicted_t1_return, predicted_t5_return, predicted_direction,
                        confidence, scenarios_json, target_t1, target_t5, recommendation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.get('prediction_id'),
                    prediction.get('created_at'),
                    prediction.get('event_type'),
                    prediction.get('event_date'),
                    prediction.get('spy_price'),
                    prediction.get('vix_level'),
                    prediction.get('rsi_14'),
                    prediction.get('market_trend'),
                    prediction.get('predicted_t1_return'),
                    prediction.get('predicted_t5_return'),
                    prediction.get('predicted_direction'),
                    prediction.get('confidence'),
                    json.dumps(prediction.get('scenarios', [])),
                    prediction.get('target_t1'),
                    prediction.get('target_t5'),
                    prediction.get('recommendation')
                ))
                return True
            except Exception as e:
                self._log(f"Error saving prediction: {e}")
                return False

    def save_prediction_outcome(self, outcome: Dict[str, Any]) -> bool:
        """예측 결과 저장"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO prediction_outcomes (
                        prediction_id, event_date, actual_t1_return, actual_t5_return,
                        actual_direction, t1_error, t5_error, direction_correct,
                        surprise_type, surprise_magnitude
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    outcome.get('prediction_id'),
                    outcome.get('event_date'),
                    outcome.get('actual_t1_return'),
                    outcome.get('actual_t5_return'),
                    outcome.get('actual_direction'),
                    outcome.get('t1_error'),
                    outcome.get('t5_error'),
                    outcome.get('direction_correct'),
                    outcome.get('surprise_type'),
                    outcome.get('surprise_magnitude')
                ))
                return True
            except Exception as e:
                self._log(f"Error saving outcome: {e}")
                return False

    def get_predictions(
        self,
        event_type: str = None,
        start_date: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """예측 조회"""
        query = "SELECT * FROM event_predictions WHERE 1=1"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if start_date:
            query += " AND event_date >= ?"
            params.append(start_date)

        query += f" ORDER BY created_at DESC LIMIT {limit}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # 시장 스냅샷 저장/조회
    # ========================================================================

    def save_market_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """시장 스냅샷 저장"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO market_snapshots (
                        snapshot_id, timestamp, spy_price, spy_change_1d, spy_change_5d,
                        spy_vs_ma20, qqq_price, iwm_price, tlt_price, gld_price,
                        vix_level, vix_percentile, rsi_14, macd_signal,
                        trend, volatility_regime, put_call_ratio, fear_greed_index,
                        days_to_fomc, days_to_cpi, days_to_nfp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.get('snapshot_id'),
                    snapshot.get('timestamp'),
                    snapshot.get('spy_price'),
                    snapshot.get('spy_change_1d'),
                    snapshot.get('spy_change_5d'),
                    snapshot.get('spy_vs_ma20'),
                    snapshot.get('qqq_price'),
                    snapshot.get('iwm_price'),
                    snapshot.get('tlt_price'),
                    snapshot.get('gld_price'),
                    snapshot.get('vix_level'),
                    snapshot.get('vix_percentile'),
                    snapshot.get('rsi_14'),
                    snapshot.get('macd_signal'),
                    snapshot.get('trend'),
                    snapshot.get('volatility_regime'),
                    snapshot.get('put_call_ratio'),
                    snapshot.get('fear_greed_index'),
                    snapshot.get('days_to_fomc'),
                    snapshot.get('days_to_cpi'),
                    snapshot.get('days_to_nfp')
                ))
                return True
            except Exception as e:
                self._log(f"Error saving snapshot: {e}")
                return False

    def get_latest_snapshot(self) -> Optional[Dict]:
        """최신 스냅샷 조회"""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM market_snapshots
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            return dict(row) if row else None

    def get_snapshots(self, days_back: int = 7) -> List[Dict]:
        """스냅샷 히스토리 조회"""
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM market_snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff,)).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # 통계 및 정확도
    # ========================================================================

    def get_prediction_accuracy(self, event_type: str = None) -> Dict:
        """예측 정확도 통계"""
        query = """
            SELECT
                event_type,
                COUNT(*) as total,
                AVG(ABS(t1_error)) as avg_t1_error,
                AVG(ABS(t5_error)) as avg_t5_error,
                SUM(direction_correct) * 100.0 / COUNT(*) as direction_accuracy
            FROM prediction_outcomes po
            JOIN event_predictions ep ON po.prediction_id = ep.prediction_id
        """

        if event_type:
            query += f" WHERE ep.event_type = '{event_type}'"

        query += " GROUP BY event_type"

        with self._get_connection() as conn:
            rows = conn.execute(query).fetchall()
            return {row['event_type']: dict(row) for row in rows}

    def get_event_stats(self) -> Dict:
        """이벤트 통계"""
        with self._get_connection() as conn:
            # 총 이벤트 수
            total = conn.execute("SELECT COUNT(*) FROM detected_events").fetchone()[0]

            # 유형별
            by_type = conn.execute("""
                SELECT event_type, COUNT(*) as count
                FROM detected_events
                GROUP BY event_type
                ORDER BY count DESC
            """).fetchall()

            # 티커별
            by_ticker = conn.execute("""
                SELECT ticker, COUNT(*) as count
                FROM detected_events
                GROUP BY ticker
                ORDER BY count DESC
                LIMIT 10
            """).fetchall()

            return {
                "total_events": total,
                "by_type": {row['event_type']: row['count'] for row in by_type},
                "top_tickers": {row['ticker']: row['count'] for row in by_ticker}
            }


# ============================================================================
# Auto-Save Session
# ============================================================================

class AutoSaveSession:
    """종료 시 자동 저장 세션"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.db = EventDatabase(verbose=verbose)

        # 수집된 데이터
        self._events: List[Dict] = []
        self._predictions: List[Dict] = []
        self._snapshot: Optional[Dict] = None

        # 자동 저장 등록
        self._registered = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"[AutoSaveSession] {msg}")

    def __enter__(self):
        self._log("Session started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save_all()
        self._log("Session ended")
        return False

    def _save_all(self):
        """모든 데이터 저장"""
        saved_count = 0

        # 스냅샷 저장
        if self._snapshot:
            if self.db.save_market_snapshot(self._snapshot):
                saved_count += 1
                self._log("Market snapshot saved")

        # 이벤트 저장
        if self._events:
            count = self.db.save_detected_events(self._events)
            saved_count += count
            self._log(f"Saved {count} events")

        # 예측 저장
        for pred in self._predictions:
            if self.db.save_prediction(pred):
                saved_count += 1
        if self._predictions:
            self._log(f"Saved {len(self._predictions)} predictions")

        self._log(f"Total saved: {saved_count} records")

    def collect_market_data(self) -> Dict:
        """실시간 시장 데이터 수집"""
        import yfinance as yf
        import pandas as pd
        import numpy as np

        self._log("Collecting market data...")

        now = datetime.now()
        snapshot_id = f"snap_{now.strftime('%Y%m%d_%H%M%S')}"

        # 가격 데이터 수집
        tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "^VIX"]
        data = {}

        for ticker in tickers:
            try:
                df = yf.download(ticker, period="1mo", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(1, axis=1)
                data[ticker] = df
            except:
                pass

        # 스냅샷 생성
        spy = data.get("SPY", pd.DataFrame())
        vix = data.get("^VIX", pd.DataFrame())

        if spy.empty:
            self._log("No SPY data available")
            return {}

        spy_price = spy['Close'].iloc[-1]
        spy_change_1d = (spy['Close'].iloc[-1] / spy['Close'].iloc[-2] - 1) * 100
        spy_change_5d = (spy['Close'].iloc[-1] / spy['Close'].iloc[-6] - 1) * 100 if len(spy) > 5 else 0

        ma_20 = spy['Close'].rolling(20).mean().iloc[-1]
        spy_vs_ma20 = (spy_price / ma_20 - 1) * 100

        vix_level = vix['Close'].iloc[-1] if not vix.empty else 15
        vix_1y = vix['Close'].iloc[-252:] if len(vix) > 252 else vix['Close']
        vix_percentile = (vix_level <= vix_1y).sum() / len(vix_1y) * 100 if len(vix_1y) > 0 else 50

        # RSI
        delta = spy['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi_14 = (100 - (100 / (1 + rs))).iloc[-1]

        # 추세 판단
        if spy_vs_ma20 > 2.0:
            trend = "strong_up"
        elif spy_vs_ma20 > 0.5:
            trend = "up"
        elif spy_vs_ma20 > -0.5:
            trend = "neutral"
        elif spy_vs_ma20 > -2.0:
            trend = "down"
        else:
            trend = "strong_down"

        # 변동성 레짐
        if vix_level < 12:
            vol_regime = "low"
        elif vix_level < 18:
            vol_regime = "normal"
        elif vix_level < 25:
            vol_regime = "elevated"
        else:
            vol_regime = "high"

        # 이벤트까지 일수
        from lib.event_framework import CalendarEventManager, EventType
        calendar = CalendarEventManager()

        self._snapshot = {
            "snapshot_id": snapshot_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "spy_price": round(spy_price, 2),
            "spy_change_1d": round(spy_change_1d, 2),
            "spy_change_5d": round(spy_change_5d, 2),
            "spy_vs_ma20": round(spy_vs_ma20, 2),
            "qqq_price": round(data.get("QQQ", pd.DataFrame())['Close'].iloc[-1], 2) if "QQQ" in data and not data["QQQ"].empty else None,
            "iwm_price": round(data.get("IWM", pd.DataFrame())['Close'].iloc[-1], 2) if "IWM" in data and not data["IWM"].empty else None,
            "tlt_price": round(data.get("TLT", pd.DataFrame())['Close'].iloc[-1], 2) if "TLT" in data and not data["TLT"].empty else None,
            "gld_price": round(data.get("GLD", pd.DataFrame())['Close'].iloc[-1], 2) if "GLD" in data and not data["GLD"].empty else None,
            "vix_level": round(vix_level, 2),
            "vix_percentile": round(vix_percentile, 1),
            "rsi_14": round(rsi_14, 1),
            "macd_signal": None,  # TODO: 계산 추가
            "trend": trend,
            "volatility_regime": vol_regime,
            "put_call_ratio": None,  # TODO: 옵션 데이터
            "fear_greed_index": None,  # TODO: 외부 API
            "days_to_fomc": calendar.days_to_next_event(EventType.FOMC),
            "days_to_cpi": calendar.days_to_next_event(EventType.CPI),
            "days_to_nfp": calendar.days_to_next_event(EventType.NFP)
        }

        self._log(f"Snapshot collected: SPY ${spy_price:.2f}, VIX {vix_level:.1f}")
        return self._snapshot

    def run_event_detection(self, tickers: List[str] = None) -> List[Dict]:
        """이벤트 감지 실행"""
        import yfinance as yf
        import pandas as pd
        from lib.event_framework import QuantitativeEventDetector

        if tickers is None:
            tickers = ["SPY", "QQQ", "GLD", "TLT", "IWM"]

        self._log(f"Running event detection for {tickers}...")

        detector = QuantitativeEventDetector()

        for ticker in tickers:
            try:
                df = yf.download(ticker, period="3mo", progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(1, axis=1)

                if df.empty:
                    continue

                events = detector.detect_all(df, ticker=ticker)

                # 최근 7일 이벤트만
                cutoff = datetime.now() - timedelta(days=7)
                recent = [e for e in events if e.timestamp >= cutoff]

                for event in recent:
                    self._events.append({
                        "event_id": event.event_id,
                        "event_date": event.timestamp.strftime("%Y-%m-%d"),
                        "event_type": event.event_type.value,
                        "ticker": event.ticker,
                        "asset_class": event.asset_class.value,
                        "importance": event.importance.name,
                        "value": event.metadata.get("z_score") or event.metadata.get("return_pct"),
                        "price_change": event.metadata.get("return_pct"),
                        "volume_ratio": event.metadata.get("volume"),
                        "name": event.name,
                        "description": event.description,
                        "confidence": event.confidence,
                        "metadata": event.metadata
                    })
            except Exception as e:
                self._log(f"Error detecting events for {ticker}: {e}")

        self._log(f"Detected {len(self._events)} events")
        return self._events

    def run_predictions(self) -> List[Dict]:
        """예측 실행"""
        from lib.event_predictor import EventPredictor

        self._log("Running predictions...")

        predictor = EventPredictor(verbose=False)
        predictions = predictor.predict_upcoming_events()

        for pred in predictions:
            pred_id = f"pred_{pred.event_type}_{pred.event_date}_{datetime.now().strftime('%H%M%S')}"

            scenarios = [
                {
                    "name": s.scenario_name,
                    "probability": s.probability,
                    "t1_return": s.t_plus_1_return,
                    "t5_return": s.t_plus_5_return
                }
                for s in pred.scenarios
            ]

            targets = predictor.generate_price_targets(pred)

            self._predictions.append({
                "prediction_id": pred_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": pred.event_type,
                "event_date": pred.event_date,
                "spy_price": pred.current_price,
                "vix_level": pred.market_state.vix_level,
                "rsi_14": pred.market_state.rsi_14,
                "market_trend": pred.market_state.trend,
                "predicted_t1_return": pred.weighted_t1_return,
                "predicted_t5_return": pred.weighted_t5_return,
                "predicted_direction": "bullish" if pred.weighted_t5_return > 0.2 else "bearish" if pred.weighted_t5_return < -0.2 else "neutral",
                "confidence": pred.confidence,
                "scenarios": scenarios,
                "target_t1": targets["weighted"]["t_plus_1"],
                "target_t5": targets["weighted"]["t_plus_5"],
                "recommendation": pred.recommendation
            })

        self._log(f"Generated {len(self._predictions)} predictions")
        return self._predictions


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EIMAS Event Database - Auto Save Session")
    print("=" * 70)

    with AutoSaveSession(verbose=True) as session:
        # 1. 시장 데이터 수집
        snapshot = session.collect_market_data()

        # 2. 이벤트 감지
        events = session.run_event_detection()

        # 3. 예측 실행
        predictions = session.run_predictions()

        print("\n" + "-" * 50)
        print(f"Collected: 1 snapshot, {len(events)} events, {len(predictions)} predictions")
        print("-" * 50)

    # 세션 종료 시 자동 저장됨
    print("\n✅ All data saved to database!")

    # DB 통계 확인
    db = EventDatabase(verbose=False)
    stats = db.get_event_stats()
    print(f"\nDB Statistics:")
    print(f"  Total Events: {stats['total_events']}")
    print(f"  By Type: {stats['by_type']}")
