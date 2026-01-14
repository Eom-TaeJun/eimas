#!/usr/bin/env python3
"""
Stable Data Store
=================
안정적인 확정 데이터 저장소

저장 대상 (장 마감 후 확정):
- daily_prices: 일별 OHLCV
- intraday_summary: 장중 집계 (시가갭, 고저시간, VWAP 등)
- economic_calendar: 경제 이벤트 일정
- prediction_outcomes: 예측 결과 (확정 후)

특징:
- 전일 종가 기준 지표만 저장
- 한번 저장되면 변경 없음
- 영구 보존
"""

import sqlite3
import json
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import contextmanager

# DB 경로
STABLE_DB_PATH = Path(__file__).parent / "stable" / "market.db"


class StableStore:
    """안정 데이터 저장소"""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or STABLE_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """DB 초기화"""
        with self._get_connection() as conn:
            conn.executescript(STABLE_SCHEMA)
        print(f"[StableStore] Initialized: {self.db_path}")

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
    # Daily Prices (일별 가격)
    # ========================================================================

    def save_daily_price(self, ticker: str, data: Dict) -> bool:
        """일별 가격 저장"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO daily_prices (
                        date, ticker, open, high, low, close, adj_close, volume,
                        prev_close, change_pct, gap_pct,
                        ma5, ma20, ma50, vs_ma5, vs_ma20, vs_ma50,
                        rsi_14, high_52w, low_52w
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get('date'),
                    ticker,
                    data.get('open'),
                    data.get('high'),
                    data.get('low'),
                    data.get('close'),
                    data.get('adj_close'),
                    data.get('volume'),
                    data.get('prev_close'),
                    data.get('change_pct'),
                    data.get('gap_pct'),
                    data.get('ma5'),
                    data.get('ma20'),
                    data.get('ma50'),
                    data.get('vs_ma5'),
                    data.get('vs_ma20'),
                    data.get('vs_ma50'),
                    data.get('rsi_14'),
                    data.get('high_52w'),
                    data.get('low_52w')
                ))
                return True
            except Exception as e:
                print(f"[StableStore] Error saving daily price: {e}")
                return False

    def get_daily_prices(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """일별 가격 조회"""
        query = "SELECT * FROM daily_prices WHERE ticker = ?"
        params = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += f" ORDER BY date DESC LIMIT {limit}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # Intraday Summary (장중 집계)
    # ========================================================================

    def save_intraday_summary(self, ticker: str, data: Dict) -> bool:
        """장중 집계 저장 (다음날 아침 저장)"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO intraday_summary (
                        date, ticker,
                        prev_close, open_price, opening_gap_pct,
                        first_30min_high, first_30min_low, first_30min_range_pct,
                        intraday_high, intraday_high_time,
                        intraday_low, intraday_low_time,
                        intraday_range_pct,
                        vwap, close_vs_vwap_pct,
                        volume_total, volume_morning_pct, volume_afternoon_pct, volume_power_hour_pct,
                        vix_open, vix_high, vix_low, vix_close
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get('date'),
                    ticker,
                    data.get('prev_close'),
                    data.get('open_price'),
                    data.get('opening_gap_pct'),
                    data.get('first_30min_high'),
                    data.get('first_30min_low'),
                    data.get('first_30min_range_pct'),
                    data.get('intraday_high'),
                    data.get('intraday_high_time'),
                    data.get('intraday_low'),
                    data.get('intraday_low_time'),
                    data.get('intraday_range_pct'),
                    data.get('vwap'),
                    data.get('close_vs_vwap_pct'),
                    data.get('volume_total'),
                    data.get('volume_morning_pct'),
                    data.get('volume_afternoon_pct'),
                    data.get('volume_power_hour_pct'),
                    data.get('vix_open'),
                    data.get('vix_high'),
                    data.get('vix_low'),
                    data.get('vix_close')
                ))
                return True
            except Exception as e:
                print(f"[StableStore] Error saving intraday summary: {e}")
                return False

    def get_intraday_summary(
        self,
        ticker: str,
        start_date: str = None,
        limit: int = 30
    ) -> List[Dict]:
        """장중 집계 조회"""
        query = "SELECT * FROM intraday_summary WHERE ticker = ?"
        params = [ticker]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        query += f" ORDER BY date DESC LIMIT {limit}"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # Economic Calendar (경제 이벤트 일정)
    # ========================================================================

    def save_economic_event(self, event: Dict) -> bool:
        """경제 이벤트 저장"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO economic_calendar (
                        event_id, event_date, event_type, event_name,
                        expected_value, actual_value, prior_value,
                        surprise, surprise_pct, importance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.get('event_id'),
                    event.get('event_date'),
                    event.get('event_type'),
                    event.get('event_name'),
                    event.get('expected_value'),
                    event.get('actual_value'),
                    event.get('prior_value'),
                    event.get('surprise'),
                    event.get('surprise_pct'),
                    event.get('importance')
                ))
                return True
            except Exception as e:
                print(f"[StableStore] Error saving economic event: {e}")
                return False

    def get_upcoming_events(self, days_ahead: int = 30) -> List[Dict]:
        """다가오는 이벤트 조회"""
        today = date.today().isoformat()
        future = (date.today() + timedelta(days=days_ahead)).isoformat()

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM economic_calendar
                WHERE event_date BETWEEN ? AND ?
                ORDER BY event_date
            """, (today, future)).fetchall()
            return [dict(row) for row in rows]

    # ========================================================================
    # Prediction Outcomes (예측 결과)
    # ========================================================================

    def save_prediction_outcome(self, outcome: Dict) -> bool:
        """예측 결과 저장 (이벤트 확정 후)"""
        with self._get_connection() as conn:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO prediction_outcomes (
                        prediction_id, event_type, event_date,
                        predicted_t1, predicted_t5, predicted_direction,
                        actual_t1, actual_t5, actual_direction,
                        t1_error, t5_error, direction_correct,
                        scenario_matched, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    outcome.get('prediction_id'),
                    outcome.get('event_type'),
                    outcome.get('event_date'),
                    outcome.get('predicted_t1'),
                    outcome.get('predicted_t5'),
                    outcome.get('predicted_direction'),
                    outcome.get('actual_t1'),
                    outcome.get('actual_t5'),
                    outcome.get('actual_direction'),
                    outcome.get('t1_error'),
                    outcome.get('t5_error'),
                    outcome.get('direction_correct'),
                    outcome.get('scenario_matched'),
                    outcome.get('notes')
                ))
                return True
            except Exception as e:
                print(f"[StableStore] Error saving prediction outcome: {e}")
                return False

    def get_prediction_accuracy(self, event_type: str = None) -> Dict:
        """예측 정확도 통계"""
        query = "SELECT * FROM prediction_outcomes WHERE actual_t1 IS NOT NULL"
        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()

            if not rows:
                return {"total": 0, "accuracy": None}

            total = len(rows)
            correct = sum(1 for r in rows if r['direction_correct'])
            avg_t1_error = sum(abs(r['t1_error'] or 0) for r in rows) / total
            avg_t5_error = sum(abs(r['t5_error'] or 0) for r in rows) / total

            return {
                "total": total,
                "direction_accuracy": correct / total if total > 0 else 0,
                "avg_t1_error": avg_t1_error,
                "avg_t5_error": avg_t5_error
            }


# ============================================================================
# Schema
# ============================================================================

STABLE_SCHEMA = """
-- ============================================================================
-- 일별 가격 (전일 종가 기준 지표 포함)
-- ============================================================================
CREATE TABLE IF NOT EXISTS daily_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,

    -- OHLCV
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,

    -- 전일 대비
    prev_close REAL,
    change_pct REAL,
    gap_pct REAL,               -- 시가 갭 (open vs prev_close)

    -- 이동평균 (전일 종가 기준)
    ma5 REAL,
    ma20 REAL,
    ma50 REAL,
    vs_ma5 REAL,                -- close vs MA5 (%)
    vs_ma20 REAL,
    vs_ma50 REAL,

    -- 기술적 지표 (전일 종가 기준)
    rsi_14 REAL,

    -- 52주 고저
    high_52w REAL,
    low_52w REAL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_prices(date);
CREATE INDEX IF NOT EXISTS idx_daily_ticker ON daily_prices(ticker);

-- ============================================================================
-- 장중 집계 (다음날 아침 저장)
-- ============================================================================
CREATE TABLE IF NOT EXISTS intraday_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,

    -- 시가 갭
    prev_close REAL,
    open_price REAL,
    opening_gap_pct REAL,

    -- 오프닝 레인지 (첫 30분)
    first_30min_high REAL,
    first_30min_low REAL,
    first_30min_range_pct REAL,

    -- 장중 고저점
    intraday_high REAL,
    intraday_high_time TEXT,    -- HH:MM
    intraday_low REAL,
    intraday_low_time TEXT,
    intraday_range_pct REAL,

    -- VWAP
    vwap REAL,
    close_vs_vwap_pct REAL,

    -- 거래량 분포
    volume_total INTEGER,
    volume_morning_pct REAL,    -- 09:30-12:00
    volume_afternoon_pct REAL,  -- 12:00-15:00
    volume_power_hour_pct REAL, -- 15:00-16:00

    -- VIX
    vix_open REAL,
    vix_high REAL,
    vix_low REAL,
    vix_close REAL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_intraday_date ON intraday_summary(date);

-- ============================================================================
-- 경제 이벤트 일정
-- ============================================================================
CREATE TABLE IF NOT EXISTS economic_calendar (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE,
    event_date TEXT NOT NULL,
    event_type TEXT NOT NULL,   -- fomc, cpi, nfp, pce, gdp
    event_name TEXT,

    -- 수치
    expected_value REAL,
    actual_value REAL,
    prior_value REAL,
    surprise REAL,              -- actual - expected
    surprise_pct REAL,

    importance TEXT,            -- LOW, MEDIUM, HIGH

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_calendar_date ON economic_calendar(event_date);
CREATE INDEX IF NOT EXISTS idx_calendar_type ON economic_calendar(event_type);

-- ============================================================================
-- 예측 결과 (이벤트 후 확정)
-- ============================================================================
CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE,
    event_type TEXT NOT NULL,
    event_date TEXT NOT NULL,

    -- 예측값
    predicted_t1 REAL,
    predicted_t5 REAL,
    predicted_direction TEXT,   -- UP, DOWN, NEUTRAL

    -- 실제값
    actual_t1 REAL,
    actual_t5 REAL,
    actual_direction TEXT,

    -- 평가
    t1_error REAL,
    t5_error REAL,
    direction_correct INTEGER,  -- 1 or 0
    scenario_matched TEXT,      -- 어떤 시나리오가 맞았는지

    notes TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_outcomes_type ON prediction_outcomes(event_type);
"""


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    store = StableStore()

    # 테스트 데이터 저장
    print("\n테스트: 일별 가격 저장")
    store.save_daily_price("SPY", {
        "date": "2025-12-31",
        "open": 687.14,
        "high": 687.36,
        "low": 681.71,
        "close": 681.84,
        "volume": 61244086,
        "prev_close": 687.64,
        "change_pct": -0.84,
        "gap_pct": -0.07,
        "ma5": 687.50,
        "ma20": 682.50,
        "vs_ma5": -0.82,
        "vs_ma20": -0.10,
        "rsi_14": 46.3
    })

    print("\n테스트: 장중 집계 저장")
    store.save_intraday_summary("SPY", {
        "date": "2025-12-31",
        "prev_close": 687.64,
        "open_price": 687.14,
        "opening_gap_pct": -0.07,
        "first_30min_high": 687.36,
        "first_30min_low": 685.20,
        "first_30min_range_pct": 0.32,
        "intraday_high": 687.36,
        "intraday_high_time": "09:30",
        "intraday_low": 681.71,
        "intraday_low_time": "15:59",
        "vwap": 684.44,
        "close_vs_vwap_pct": -0.38,
        "volume_total": 61244086,
        "volume_morning_pct": 39,
        "volume_afternoon_pct": 26,
        "volume_power_hour_pct": 35
    })

    # 조회 테스트
    print("\n저장된 일별 가격:")
    prices = store.get_daily_prices("SPY", limit=3)
    for p in prices:
        print(f"  {p['date']}: ${p['close']:.2f} ({p['change_pct']:+.2f}%)")

    print("\n저장된 장중 집계:")
    summaries = store.get_intraday_summary("SPY", limit=3)
    for s in summaries:
        print(f"  {s['date']}: 갭 {s['opening_gap_pct']:+.2f}%, VWAP ${s['vwap']:.2f}")

    print("\n✅ StableStore 테스트 완료")
