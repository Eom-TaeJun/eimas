#!/usr/bin/env python3
"""
Volatile Data Store
===================
휘발성/실시간 데이터 저장소

저장 대상 (시점 의존, 이벤트 기반):
- detected_events: 감지된 이상 이벤트
- intraday_alerts: 장중 알림 (VIX 스파이크, 급락 등)
- active_predictions: 진행 중인 예측
- market_snapshots: 시장 스냅샷 (수집 시점)

특징:
- 실시간 수집 데이터
- 시점에 따라 값이 달라질 수 있음
- 일정 기간 후 정리/아카이브 가능
"""

import sqlite3
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import contextmanager
import uuid

# DB 경로
VOLATILE_DB_PATH = Path(__file__).parent / "volatile" / "realtime.db"


class VolatileStore:
    """휘발성/실시간 데이터 저장소"""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or VOLATILE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """DB 초기화"""
        with self._get_connection() as conn:
            conn.executescript(VOLATILE_SCHEMA)
        print(f"[VolatileStore] Initialized: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """DB 연결"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ========================================================================
    # Detected Events (감지된 이벤트)
    # ========================================================================

    def save_detected_event(self, event: Dict) -> bool:
        """감지된 이벤트 저장"""
        event_id = event.get('event_id') or f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO detected_events (
                        event_id, timestamp, ticker, event_type,
                        value, threshold, deviation,
                        price_at_event, vix_at_event, volume_ratio,
                        importance, description, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    event.get('timestamp') or datetime.now().isoformat(),
                    event.get('ticker'),
                    event.get('event_type'),
                    event.get('value'),
                    event.get('threshold'),
                    event.get('deviation'),
                    event.get('price_at_event'),
                    event.get('vix_at_event'),
                    event.get('volume_ratio'),
                    event.get('importance', 'MEDIUM'),
                    event.get('description'),
                    json.dumps(event.get('metadata', {}))
                ))
                return True
            except Exception as e:
                print(f"[VolatileStore] Error saving event: {e}")
                return False

    def get_recent_events(
        self,
        hours_back: int = 24,
        ticker: str = None,
        event_type: str = None
    ) -> List[Dict]:
        """최근 이벤트 조회"""
        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()

        query = "SELECT * FROM detected_events WHERE timestamp >= ?"
        params = [cutoff]

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        query += " ORDER BY timestamp DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # Intraday Alerts (장중 알림)
    # ========================================================================

    def save_intraday_alert(self, alert: Dict) -> bool:
        """장중 알림 저장"""
        alert_id = alert.get('alert_id') or f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO intraday_alerts (
                        alert_id, timestamp, ticker, alert_type,
                        value, threshold, deviation,
                        price_at_alert, vix_at_alert, volume_ratio,
                        description, triggered_action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert_id,
                    alert.get('timestamp') or datetime.now().isoformat(),
                    alert.get('ticker'),
                    alert.get('alert_type'),
                    alert.get('value'),
                    alert.get('threshold'),
                    alert.get('deviation'),
                    alert.get('price_at_alert'),
                    alert.get('vix_at_alert'),
                    alert.get('volume_ratio'),
                    alert.get('description'),
                    alert.get('triggered_action')
                ))
                return True
            except Exception as e:
                print(f"[VolatileStore] Error saving alert: {e}")
                return False

    def get_today_alerts(self, ticker: str = None) -> List[Dict]:
        """오늘 알림 조회"""
        today = date.today().isoformat()

        query = "SELECT * FROM intraday_alerts WHERE DATE(timestamp) = ?"
        params = [today]

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        query += " ORDER BY timestamp DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # Active Predictions (진행 중인 예측)
    # ========================================================================

    def save_active_prediction(self, prediction: Dict) -> bool:
        """진행 중인 예측 저장"""
        pred_id = prediction.get('prediction_id') or f"pred_{prediction.get('event_type')}_{prediction.get('event_date')}"

        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO active_predictions (
                        prediction_id, created_at, event_type, event_date,
                        days_to_event,
                        spy_price_at_creation, vix_at_creation, rsi_at_creation,
                        predicted_t1, predicted_t5, predicted_direction,
                        confidence, best_scenario, worst_scenario,
                        target_t1, target_t5, stop_loss,
                        recommendation, scenarios_json, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pred_id,
                    prediction.get('created_at') or datetime.now().isoformat(),
                    prediction.get('event_type'),
                    prediction.get('event_date'),
                    prediction.get('days_to_event'),
                    prediction.get('spy_price'),
                    prediction.get('vix_level'),
                    prediction.get('rsi'),
                    prediction.get('predicted_t1'),
                    prediction.get('predicted_t5'),
                    prediction.get('predicted_direction'),
                    prediction.get('confidence'),
                    prediction.get('best_scenario'),
                    prediction.get('worst_scenario'),
                    prediction.get('target_t1'),
                    prediction.get('target_t5'),
                    prediction.get('stop_loss'),
                    prediction.get('recommendation'),
                    json.dumps(prediction.get('scenarios', [])),
                    prediction.get('status', 'ACTIVE')
                ))
                return True
            except Exception as e:
                print(f"[VolatileStore] Error saving prediction: {e}")
                return False

    def get_active_predictions(self) -> List[Dict]:
        """활성 예측 조회"""
        today = date.today().isoformat()

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM active_predictions
                WHERE event_date >= ? AND status = 'ACTIVE'
                ORDER BY event_date
            """, (today,)).fetchall()
            return [dict(row) for row in rows]

    def update_prediction_status(self, prediction_id: str, status: str) -> bool:
        """예측 상태 업데이트"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    UPDATE active_predictions
                    SET status = ?, updated_at = ?
                    WHERE prediction_id = ?
                """, (status, datetime.now().isoformat(), prediction_id))
                return True
            except Exception as e:
                print(f"[VolatileStore] Error updating prediction: {e}")
                return False

    # ========================================================================
    # Market Snapshots (시장 스냅샷)
    # ========================================================================

    def save_market_snapshot(self, snapshot: Dict) -> bool:
        """시장 스냅샷 저장"""
        snap_id = snapshot.get('snapshot_id') or f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO market_snapshots (
                        snapshot_id, timestamp, collection_type,
                        spy_price, spy_change_pct, qqq_price, iwm_price,
                        vix_level, vix_change_pct,
                        tlt_price, gld_price,
                        rsi_14, trend, volatility_regime,
                        days_to_fomc, days_to_cpi, days_to_nfp,
                        notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snap_id,
                    snapshot.get('timestamp') or datetime.now().isoformat(),
                    snapshot.get('collection_type', 'manual'),  # manual, scheduled, alert
                    snapshot.get('spy_price'),
                    snapshot.get('spy_change_pct'),
                    snapshot.get('qqq_price'),
                    snapshot.get('iwm_price'),
                    snapshot.get('vix_level'),
                    snapshot.get('vix_change_pct'),
                    snapshot.get('tlt_price'),
                    snapshot.get('gld_price'),
                    snapshot.get('rsi_14'),
                    snapshot.get('trend'),
                    snapshot.get('volatility_regime'),
                    snapshot.get('days_to_fomc'),
                    snapshot.get('days_to_cpi'),
                    snapshot.get('days_to_nfp'),
                    snapshot.get('notes')
                ))
                return True
            except Exception as e:
                print(f"[VolatileStore] Error saving snapshot: {e}")
                return False

    def get_latest_snapshot(self) -> Optional[Dict]:
        """최신 스냅샷 조회"""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM market_snapshots
                ORDER BY timestamp DESC LIMIT 1
            """).fetchone()
            return dict(row) if row else None

    def get_snapshots(self, hours_back: int = 24) -> List[Dict]:
        """스냅샷 히스토리 조회"""
        cutoff = (datetime.now() - timedelta(hours=hours_back)).isoformat()

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM market_snapshots
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (cutoff,)).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # Cleanup (정리)
    # ========================================================================

    def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """오래된 데이터 정리"""
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        deleted = {}

        with self._get_connection() as conn:
            # 이벤트 정리
            cursor = conn.execute(
                "DELETE FROM detected_events WHERE timestamp < ?", (cutoff,)
            )
            deleted['detected_events'] = cursor.rowcount

            # 알림 정리
            cursor = conn.execute(
                "DELETE FROM intraday_alerts WHERE timestamp < ?", (cutoff,)
            )
            deleted['intraday_alerts'] = cursor.rowcount

            # 완료된 예측 정리
            cursor = conn.execute(
                "DELETE FROM active_predictions WHERE event_date < ? AND status != 'ACTIVE'", (cutoff,)
            )
            deleted['active_predictions'] = cursor.rowcount

            # 스냅샷 정리
            cursor = conn.execute(
                "DELETE FROM market_snapshots WHERE timestamp < ?", (cutoff,)
            )
            deleted['market_snapshots'] = cursor.rowcount

        print(f"[VolatileStore] Cleaned up: {deleted}")
        return deleted


# ============================================================================
# Schema
# ============================================================================

VOLATILE_SCHEMA = """
-- ============================================================================
-- 감지된 이벤트
-- ============================================================================
CREATE TABLE IF NOT EXISTS detected_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    event_type TEXT NOT NULL,   -- price_shock, volume_spike, vix_spike, etc.

    -- 수치
    value REAL,                 -- 주요 수치
    threshold REAL,             -- 기준치
    deviation REAL,             -- 이탈 정도 (z-score 등)

    -- 컨텍스트
    price_at_event REAL,
    vix_at_event REAL,
    volume_ratio REAL,

    importance TEXT,            -- LOW, MEDIUM, HIGH, CRITICAL
    description TEXT,
    metadata_json TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON detected_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_ticker ON detected_events(ticker);
CREATE INDEX IF NOT EXISTS idx_events_type ON detected_events(event_type);

-- ============================================================================
-- 장중 알림
-- ============================================================================
CREATE TABLE IF NOT EXISTS intraday_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    alert_type TEXT NOT NULL,   -- vix_spike, price_crash, gap_reversal, etc.

    -- 수치
    value REAL,
    threshold REAL,
    deviation REAL,

    -- 컨텍스트
    price_at_alert REAL,
    vix_at_alert REAL,
    volume_ratio REAL,

    description TEXT,
    triggered_action TEXT,      -- 트리거된 액션 (알림, 주문 등)

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_ts ON intraday_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON intraday_alerts(alert_type);

-- ============================================================================
-- 진행 중인 예측
-- ============================================================================
CREATE TABLE IF NOT EXISTS active_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL,   -- fomc, cpi, nfp
    event_date TEXT NOT NULL,
    days_to_event INTEGER,

    -- 생성 시점 시장 상태
    spy_price_at_creation REAL,
    vix_at_creation REAL,
    rsi_at_creation REAL,

    -- 예측
    predicted_t1 REAL,
    predicted_t5 REAL,
    predicted_direction TEXT,   -- UP, DOWN, NEUTRAL
    confidence REAL,

    -- 시나리오
    best_scenario TEXT,
    worst_scenario TEXT,

    -- 트레이딩 레벨
    target_t1 REAL,
    target_t5 REAL,
    stop_loss REAL,

    recommendation TEXT,
    scenarios_json TEXT,

    status TEXT DEFAULT 'ACTIVE',  -- ACTIVE, COMPLETED, EXPIRED
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_pred_date ON active_predictions(event_date);
CREATE INDEX IF NOT EXISTS idx_pred_status ON active_predictions(status);

-- ============================================================================
-- 시장 스냅샷
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT UNIQUE NOT NULL,
    timestamp TEXT NOT NULL,
    collection_type TEXT,       -- manual, scheduled, alert

    -- 가격
    spy_price REAL,
    spy_change_pct REAL,
    qqq_price REAL,
    iwm_price REAL,

    -- VIX
    vix_level REAL,
    vix_change_pct REAL,

    -- 기타 자산
    tlt_price REAL,
    gld_price REAL,

    -- 지표
    rsi_14 REAL,
    trend TEXT,
    volatility_regime TEXT,

    -- 이벤트 정보
    days_to_fomc INTEGER,
    days_to_cpi INTEGER,
    days_to_nfp INTEGER,

    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_snap_ts ON market_snapshots(timestamp);
"""


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    store = VolatileStore()

    # 이벤트 저장 테스트
    print("\n테스트: 감지 이벤트 저장")
    store.save_detected_event({
        "ticker": "SPY",
        "event_type": "price_shock",
        "value": -1.5,
        "threshold": -1.0,
        "deviation": 2.5,
        "price_at_event": 680.50,
        "vix_at_event": 16.5,
        "importance": "HIGH",
        "description": "SPY 1.5% 급락 감지"
    })

    # 알림 저장 테스트
    print("\n테스트: 장중 알림 저장")
    store.save_intraday_alert({
        "ticker": "VIX",
        "alert_type": "vix_spike",
        "value": 18.5,
        "threshold": 16.0,
        "deviation": 15.6,  # 15.6% 상승
        "price_at_alert": 680.50,
        "vix_at_alert": 18.5,
        "description": "VIX 15% 이상 급등"
    })

    # 예측 저장 테스트
    print("\n테스트: 활성 예측 저장")
    store.save_active_prediction({
        "event_type": "fomc",
        "event_date": "2026-01-28",
        "days_to_event": 25,
        "spy_price": 683.17,
        "vix_level": 14.5,
        "rsi": 46,
        "predicted_t1": 0.26,
        "predicted_t5": 0.76,
        "predicted_direction": "UP",
        "confidence": 0.81,
        "best_scenario": "dovish_surprise",
        "worst_scenario": "hawkish_surprise",
        "target_t1": 685.0,
        "target_t5": 688.4,
        "stop_loss": 675.0,
        "recommendation": "CAUTIOUS"
    })

    # 스냅샷 저장 테스트
    print("\n테스트: 시장 스냅샷 저장")
    store.save_market_snapshot({
        "collection_type": "manual",
        "spy_price": 683.17,
        "spy_change_pct": 0.18,
        "vix_level": 14.51,
        "vix_change_pct": 2.5,
        "rsi_14": 46,
        "trend": "NEUTRAL",
        "days_to_fomc": 25,
        "days_to_cpi": 11,
        "days_to_nfp": 6
    })

    # 조회 테스트
    print("\n최근 이벤트:")
    events = store.get_recent_events(hours_back=24)
    for e in events:
        print(f"  [{e['timestamp'][:16]}] {e['ticker']} - {e['event_type']}")

    print("\n활성 예측:")
    preds = store.get_active_predictions()
    for p in preds:
        print(f"  [{p['event_date']}] {p['event_type'].upper()} - T+5: {p['predicted_t5']:.2f}%")

    print("\n최신 스냅샷:")
    snap = store.get_latest_snapshot()
    if snap:
        print(f"  SPY: ${snap['spy_price']}, VIX: {snap['vix_level']}")

    print("\n✅ VolatileStore 테스트 완료")
