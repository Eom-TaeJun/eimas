#!/usr/bin/env python3
"""
EIMAS Database Manager
======================
SQLite 기반 분석 결과 저장소

Tables:
- ark_holdings: ARK ETF 일별 보유종목
- ark_weight_changes: 비중 변화 이력
- etf_analysis: ETF 분석 결과
- market_regime: 시장 레짐 이력
- signals: 생성된 신호
- actions: 권고 액션
"""

import sqlite3
import json
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# 기본 DB 경로
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "eimas.db"


class DatabaseManager:
    """
    EIMAS 통합 데이터베이스 관리자

    사용법:
        db = DatabaseManager()
        db.save_ark_holdings(holdings_list)
        db.save_signal(signal_dict)

        # 조회
        holdings = db.get_ark_holdings(date="2025-01-01", etf="ARKK")
        signals = db.get_signals(start_date="2025-01-01", ticker="TSLA")
    """

    def __init__(self, db_path: str = None):
        """
        Args:
            db_path: DB 파일 경로 (기본: data/eimas.db)
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    @contextmanager
    def _get_connection(self):
        """컨텍스트 매니저로 연결 관리"""
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
            cursor = conn.cursor()

            # ================================================================
            # ARK Holdings 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ark_holdings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    etf TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    company TEXT,
                    cusip TEXT,
                    shares REAL,
                    market_value REAL,
                    weight REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, etf, ticker)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ark_holdings_date ON ark_holdings(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ark_holdings_etf ON ark_holdings(etf)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ark_holdings_ticker ON ark_holdings(ticker)")

            # ================================================================
            # ARK 비중 변화 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ark_weight_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    etf TEXT NOT NULL,
                    prev_weight REAL,
                    curr_weight REAL,
                    weight_change REAL,
                    change_type TEXT,  -- INCREASE, DECREASE, NEW, EXIT
                    prev_shares REAL,
                    curr_shares REAL,
                    share_change REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, etf, ticker)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_weight_changes_date ON ark_weight_changes(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_weight_changes_ticker ON ark_weight_changes(ticker)")

            # ================================================================
            # ETF 분석 결과 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS etf_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,  -- comparison, sector_rotation, market_regime
                    data JSON NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, analysis_type)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_etf_analysis_date ON etf_analysis(date)")

            # ================================================================
            # 시장 레짐 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_regime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    sentiment TEXT,  -- RISK_ON, RISK_OFF, NEUTRAL
                    cycle_phase TEXT,  -- EARLY, MID, LATE, RECESSION
                    style_rotation TEXT,  -- GROWTH, VALUE, BALANCED
                    risk_appetite_score REAL,
                    breadth_score REAL,
                    growth_value_spread REAL,
                    large_small_spread REAL,
                    equity_bond_spread REAL,
                    hy_treasury_spread REAL,
                    vix_estimate REAL,
                    signals_json TEXT,
                    warnings_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_regime_date ON market_regime(date)")

            # ================================================================
            # Signal 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE,
                    date TEXT NOT NULL,
                    type TEXT NOT NULL,  -- etf_flow, sector_rotation, market_regime, ark_holdings
                    ticker TEXT NOT NULL,
                    name TEXT,
                    indicator TEXT,
                    value REAL,
                    threshold REAL,
                    z_score REAL,
                    level TEXT,  -- WARNING, ALERT, CRITICAL
                    description TEXT,
                    confidence REAL,
                    direction TEXT,  -- long, short, neutral
                    horizon TEXT,  -- short, medium, long
                    source TEXT,
                    regime_aligned INTEGER,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(type)")

            # ================================================================
            # Action 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id TEXT UNIQUE,
                    date TEXT NOT NULL,
                    signal_id TEXT,
                    ticker TEXT NOT NULL,
                    action_type TEXT NOT NULL,  -- BUY_SIGNAL, SELL_SIGNAL, etc.
                    direction TEXT,  -- long, short
                    position_size REAL,
                    entry_strategy TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    time_horizon TEXT,
                    rationale TEXT,
                    risk_reward REAL,
                    priority INTEGER,
                    metadata_json TEXT,
                    executed INTEGER DEFAULT 0,
                    executed_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals(signal_id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_date ON actions(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_ticker ON actions(ticker)")

            # ================================================================
            # 분석 실행 로그 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    analysis_type TEXT NOT NULL,
                    status TEXT NOT NULL,  -- SUCCESS, FAILED, PARTIAL
                    duration_seconds REAL,
                    records_processed INTEGER,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

    # ========================================================================
    # ARK Holdings 메서드
    # ========================================================================

    def save_ark_holdings(self, holdings: List[Dict[str, Any]], date_str: str = None):
        """
        ARK 보유종목 저장

        Args:
            holdings: HoldingData 딕셔너리 리스트
            date_str: 날짜 (기본: 오늘)
        """
        if date_str is None:
            date_str = date.today().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            for h in holdings:
                cursor.execute("""
                    INSERT OR REPLACE INTO ark_holdings
                    (date, etf, ticker, company, cusip, shares, market_value, weight)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_str,
                    h.get('etf', ''),
                    h.get('ticker', ''),
                    h.get('company', ''),
                    h.get('cusip', ''),
                    h.get('shares', 0),
                    h.get('market_value', 0),
                    h.get('weight', 0)
                ))

        return len(holdings)

    def get_ark_holdings(self, date_str: str = None, etf: str = None,
                         ticker: str = None) -> List[Dict[str, Any]]:
        """
        ARK 보유종목 조회

        Args:
            date_str: 특정 날짜 (None이면 최신)
            etf: ETF 필터 (예: "ARKK")
            ticker: 종목 필터 (예: "TSLA")
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM ark_holdings WHERE 1=1"
            params = []

            if date_str:
                query += " AND date = ?"
                params.append(date_str)
            else:
                # 최신 날짜
                query += " AND date = (SELECT MAX(date) FROM ark_holdings)"

            if etf:
                query += " AND etf = ?"
                params.append(etf)

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)

            query += " ORDER BY weight DESC"

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_ark_holdings_history(self, ticker: str, days: int = 30) -> List[Dict[str, Any]]:
        """특정 종목의 보유 이력 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT date, etf, weight, shares, market_value
                FROM ark_holdings
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT ?
            """, (ticker, days * 6))  # 6개 ETF
            return [dict(row) for row in cursor.fetchall()]

    def save_ark_weight_changes(self, changes: List[Dict[str, Any]], date_str: str = None):
        """비중 변화 저장"""
        if date_str is None:
            date_str = date.today().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            for c in changes:
                cursor.execute("""
                    INSERT OR REPLACE INTO ark_weight_changes
                    (date, ticker, etf, prev_weight, curr_weight, weight_change,
                     change_type, prev_shares, curr_shares, share_change)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_str,
                    c.get('ticker', ''),
                    c.get('etf', ''),
                    c.get('prev_weight', 0),
                    c.get('curr_weight', 0),
                    c.get('weight_change', 0),
                    c.get('change_type', ''),
                    c.get('prev_shares', 0),
                    c.get('curr_shares', 0),
                    c.get('share_change', 0)
                ))

        return len(changes)

    # ========================================================================
    # Market Regime 메서드
    # ========================================================================

    def save_market_regime(self, regime: Dict[str, Any], date_str: str = None):
        """시장 레짐 저장"""
        if date_str is None:
            date_str = date.today().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO market_regime
                (date, sentiment, cycle_phase, style_rotation, risk_appetite_score,
                 breadth_score, growth_value_spread, large_small_spread,
                 equity_bond_spread, hy_treasury_spread, vix_estimate,
                 signals_json, warnings_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                regime.get('sentiment', ''),
                regime.get('cycle_phase', ''),
                regime.get('style_rotation', ''),
                regime.get('risk_appetite_score', 0),
                regime.get('breadth_score', 0),
                regime.get('growth_value_spread', 0),
                regime.get('large_small_spread', 0),
                regime.get('equity_bond_spread', 0),
                regime.get('hy_treasury_spread', 0),
                regime.get('vix_estimate', 0),
                json.dumps(regime.get('signals', [])),
                json.dumps(regime.get('warnings', []))
            ))

    def get_market_regime(self, date_str: str = None) -> Optional[Dict[str, Any]]:
        """시장 레짐 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if date_str:
                cursor.execute("SELECT * FROM market_regime WHERE date = ?", (date_str,))
            else:
                cursor.execute("SELECT * FROM market_regime ORDER BY date DESC LIMIT 1")

            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['signals'] = json.loads(result.get('signals_json', '[]'))
                result['warnings'] = json.loads(result.get('warnings_json', '[]'))
                return result
            return None

    def get_market_regime_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """시장 레짐 이력 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM market_regime
                ORDER BY date DESC
                LIMIT ?
            """, (days,))

            results = []
            for row in cursor.fetchall():
                r = dict(row)
                r['signals'] = json.loads(r.get('signals_json', '[]'))
                r['warnings'] = json.loads(r.get('warnings_json', '[]'))
                results.append(r)
            return results

    # ========================================================================
    # Signal 메서드
    # ========================================================================

    def save_signal(self, signal: Dict[str, Any], date_str: str = None):
        """신호 저장"""
        if date_str is None:
            date_str = date.today().isoformat()

        signal_id = signal.get('signal_id') or f"{signal.get('type', 'unknown')}_{signal.get('ticker', 'NA')}_{date_str}_{datetime.now().strftime('%H%M%S')}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO signals
                (signal_id, date, type, ticker, name, indicator, value, threshold,
                 z_score, level, description, confidence, direction, horizon,
                 source, regime_aligned, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id,
                date_str,
                signal.get('type', ''),
                signal.get('ticker', ''),
                signal.get('name', ''),
                signal.get('indicator', ''),
                signal.get('value', 0),
                signal.get('threshold', 0),
                signal.get('z_score', 0),
                signal.get('level', ''),
                signal.get('description', ''),
                signal.get('confidence', 0),
                signal.get('direction', ''),
                signal.get('horizon', ''),
                signal.get('source', ''),
                1 if signal.get('regime_aligned') else 0,
                json.dumps(signal.get('metadata', {}))
            ))

        return signal_id

    def save_signals(self, signals: List[Dict[str, Any]], date_str: str = None):
        """여러 신호 저장"""
        signal_ids = []
        for sig in signals:
            signal_id = self.save_signal(sig, date_str)
            signal_ids.append(signal_id)
        return signal_ids

    def get_signals(self, date_str: str = None, start_date: str = None,
                    end_date: str = None, ticker: str = None,
                    signal_type: str = None, min_confidence: float = None) -> List[Dict[str, Any]]:
        """신호 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM signals WHERE 1=1"
            params = []

            if date_str:
                query += " AND date = ?"
                params.append(date_str)
            elif start_date and end_date:
                query += " AND date BETWEEN ? AND ?"
                params.extend([start_date, end_date])
            elif start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)

            if signal_type:
                query += " AND type = ?"
                params.append(signal_type)

            if min_confidence:
                query += " AND confidence >= ?"
                params.append(min_confidence)

            query += " ORDER BY date DESC, confidence DESC"

            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                r = dict(row)
                r['metadata'] = json.loads(r.get('metadata_json', '{}'))
                r['regime_aligned'] = bool(r.get('regime_aligned'))
                results.append(r)
            return results

    # ========================================================================
    # Action 메서드
    # ========================================================================

    def save_action(self, action: Dict[str, Any], date_str: str = None):
        """액션 저장"""
        if date_str is None:
            date_str = date.today().isoformat()

        action_id = action.get('action_id') or f"action_{action.get('ticker', 'NA')}_{date_str}_{datetime.now().strftime('%H%M%S')}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO actions
                (action_id, date, signal_id, ticker, action_type, direction,
                 position_size, entry_strategy, stop_loss, take_profit,
                 time_horizon, rationale, risk_reward, priority, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                action_id,
                date_str,
                action.get('signal_id'),
                action.get('ticker', ''),
                action.get('action_type', ''),
                action.get('direction', ''),
                action.get('position_size', 0),
                action.get('entry_strategy', ''),
                action.get('stop_loss'),
                action.get('take_profit'),
                action.get('time_horizon', ''),
                action.get('rationale', ''),
                action.get('risk_reward'),
                action.get('priority', 0),
                json.dumps(action.get('metadata', {}))
            ))

        return action_id

    def save_actions(self, actions: List[Dict[str, Any]], date_str: str = None):
        """여러 액션 저장"""
        action_ids = []
        for act in actions:
            action_id = self.save_action(act, date_str)
            action_ids.append(action_id)
        return action_ids

    def get_actions(self, date_str: str = None, ticker: str = None,
                    executed: bool = None) -> List[Dict[str, Any]]:
        """액션 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM actions WHERE 1=1"
            params = []

            if date_str:
                query += " AND date = ?"
                params.append(date_str)

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)

            if executed is not None:
                query += " AND executed = ?"
                params.append(1 if executed else 0)

            query += " ORDER BY date DESC, priority DESC"

            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                r = dict(row)
                r['metadata'] = json.loads(r.get('metadata_json', '{}'))
                r['executed'] = bool(r.get('executed'))
                results.append(r)
            return results

    def mark_action_executed(self, action_id: str):
        """액션 실행 완료 표시"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE actions
                SET executed = 1, executed_at = ?
                WHERE action_id = ?
            """, (datetime.now().isoformat(), action_id))

    # ========================================================================
    # ETF 분석 결과 메서드
    # ========================================================================

    def save_etf_analysis(self, analysis_type: str, data: Dict[str, Any],
                          date_str: str = None):
        """ETF 분석 결과 저장"""
        if date_str is None:
            date_str = date.today().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO etf_analysis
                (date, analysis_type, data)
                VALUES (?, ?, ?)
            """, (date_str, analysis_type, json.dumps(data)))

    def get_etf_analysis(self, analysis_type: str, date_str: str = None) -> Optional[Dict[str, Any]]:
        """ETF 분석 결과 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if date_str:
                cursor.execute("""
                    SELECT * FROM etf_analysis
                    WHERE analysis_type = ? AND date = ?
                """, (analysis_type, date_str))
            else:
                cursor.execute("""
                    SELECT * FROM etf_analysis
                    WHERE analysis_type = ?
                    ORDER BY date DESC LIMIT 1
                """, (analysis_type,))

            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['data'] = json.loads(result.get('data', '{}'))
                return result
            return None

    # ========================================================================
    # 분석 로그 메서드
    # ========================================================================

    def log_analysis(self, analysis_type: str, status: str,
                     duration: float = None, records: int = None,
                     error: str = None, date_str: str = None):
        """분석 실행 로그 기록"""
        if date_str is None:
            date_str = date.today().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_log
                (date, analysis_type, status, duration_seconds,
                 records_processed, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (date_str, analysis_type, status, duration, records, error))

    # ========================================================================
    # 유틸리티 메서드
    # ========================================================================

    def get_latest_dates(self) -> Dict[str, str]:
        """각 테이블의 최신 날짜 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            tables = ['ark_holdings', 'market_regime', 'signals', 'actions']
            dates = {}

            for table in tables:
                cursor.execute(f"SELECT MAX(date) FROM {table}")
                row = cursor.fetchone()
                dates[table] = row[0] if row and row[0] else None

            return dates

    def get_stats(self) -> Dict[str, Any]:
        """DB 통계 조회"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            stats = {
                'db_path': str(self.db_path),
                'tables': {}
            }

            tables = ['ark_holdings', 'ark_weight_changes', 'market_regime',
                      'signals', 'actions', 'etf_analysis', 'analysis_log']

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]

                cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
                row = cursor.fetchone()

                stats['tables'][table] = {
                    'count': count,
                    'min_date': row[0],
                    'max_date': row[1]
                }

            return stats

    def vacuum(self):
        """DB 최적화"""
        with self._get_connection() as conn:
            conn.execute("VACUUM")


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EIMAS Database Manager Test")
    print("=" * 70)

    # DB 초기화
    db = DatabaseManager()
    print(f"\nDB Path: {db.db_path}")

    # 테스트 데이터 저장
    print("\n[1] Testing ARK Holdings Save...")
    test_holdings = [
        {'etf': 'ARKK', 'ticker': 'TSLA', 'company': 'Tesla Inc',
         'shares': 1000000, 'market_value': 250000000, 'weight': 10.5},
        {'etf': 'ARKK', 'ticker': 'COIN', 'company': 'Coinbase',
         'shares': 500000, 'market_value': 100000000, 'weight': 4.2},
    ]
    count = db.save_ark_holdings(test_holdings)
    print(f"    Saved {count} holdings")

    # 조회 테스트
    print("\n[2] Testing ARK Holdings Query...")
    holdings = db.get_ark_holdings(etf='ARKK')
    for h in holdings[:3]:
        print(f"    {h['ticker']:6s} {h['weight']:5.2f}% {h['company']}")

    # Signal 저장 테스트
    print("\n[3] Testing Signal Save...")
    test_signal = {
        'type': 'ark_holdings',
        'ticker': 'TSLA',
        'name': 'ARK Weight Increase',
        'indicator': 'weight_change',
        'value': 1.5,
        'confidence': 0.75,
        'direction': 'long',
        'description': 'ARKK increased TSLA weight by 1.5%',
        'metadata': {'etf': 'ARKK', 'prev_weight': 9.0}
    }
    signal_id = db.save_signal(test_signal)
    print(f"    Signal ID: {signal_id}")

    # Action 저장 테스트
    print("\n[4] Testing Action Save...")
    test_action = {
        'signal_id': signal_id,
        'ticker': 'TSLA',
        'action_type': 'BUY_SIGNAL',
        'direction': 'long',
        'position_size': 0.05,
        'entry_strategy': 'limit_order',
        'rationale': 'Following ARK weight increase'
    }
    action_id = db.save_action(test_action)
    print(f"    Action ID: {action_id}")

    # Market Regime 저장 테스트
    print("\n[5] Testing Market Regime Save...")
    test_regime = {
        'sentiment': 'RISK_ON',
        'cycle_phase': 'MID',
        'style_rotation': 'GROWTH',
        'risk_appetite_score': 65.0,
        'breadth_score': 72.0,
        'growth_value_spread': 2.5,
        'signals': ['Strong momentum', 'Positive breadth'],
        'warnings': ['VIX elevated']
    }
    db.save_market_regime(test_regime)
    print("    Saved market regime")

    # 통계 출력
    print("\n[6] Database Stats:")
    stats = db.get_stats()
    for table, info in stats['tables'].items():
        print(f"    {table:20s}: {info['count']:5d} records")

    # 최신 날짜
    print("\n[7] Latest Dates:")
    dates = db.get_latest_dates()
    for table, d in dates.items():
        print(f"    {table:20s}: {d}")

    print("\n" + "=" * 70)
    print("Database Test Complete!")
    print("=" * 70)
