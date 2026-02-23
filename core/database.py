#!/usr/bin/env python3
"""
EIMAS Database Manager
======================
SQLite 기반 분석 결과 저장소

Tables (Core):
- ark_holdings: ARK ETF 일별 보유종목
- ark_weight_changes: 비중 변화 이력
- etf_analysis: ETF 분석 결과
- market_regime: 시장 레짐 이력
- signals: 생성된 신호
- actions: 권고 액션

Tables (fi_ra Staging — Track E):
- stg_ra_macro_regime: 레짐 분석 원시 적재 (staging)
- stg_ra_etf_signal: ETF 신호 원시 적재 (staging)

Tables (fi_ra Mart — Track E):
- mart_ra_macro_regime: 레짐 집계 결과 (mart)
- mart_ra_etf_signal: ETF 신호 + 레짐 컨텍스트 집계 (mart)

Views (fi_ra — Track E):
- v_ra_macro_regime: 최신 레짐 분석 통합 뷰 (mart + staging fallback)
- v_ra_etf_signal: ETF 신호 + 레짐 컨텍스트 조인 뷰
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

        # Track E — fi_ra 뷰 기반 조회
        regime = db.get_latest_regime()
        signals_by_regime = db.get_signals_by_regime("RISK_ON")
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

            # ================================================================
            # Track E — fi_ra Staging 테이블
            # ================================================================
            # stg_ra_macro_regime: 레짐 분석 결과 원시 적재
            # ETL 파이프라인에서 외부 소스 데이터를 그대로 적재.
            # mart로 집계되기 전 중간 저장소 역할.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stg_ra_macro_regime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_run_id TEXT,           -- 분석 실행 ID (배치 추적용)
                    regime_date TEXT NOT NULL,    -- 레짐 기준 날짜
                    regime_type TEXT,             -- RISK_ON, RISK_OFF, NEUTRAL
                    cycle_phase TEXT,             -- EARLY, MID, LATE, RECESSION
                    style_bias TEXT,              -- GROWTH, VALUE, BALANCED
                    confidence REAL,              -- 레짐 신뢰도 (0.0~1.0)
                    risk_appetite_score REAL,     -- 위험 선호 점수
                    breadth_score REAL,           -- 시장 폭 점수
                    vix_estimate REAL,            -- VIX 추정치
                    growth_value_spread REAL,     -- 성장/가치 스프레드
                    equity_bond_spread REAL,      -- 주식/채권 스프레드
                    hy_treasury_spread REAL,      -- HY/국채 스프레드
                    key_indicators_json TEXT,     -- 핵심 지표 JSON (dict)
                    raw_signals_json TEXT,        -- 원시 신호 JSON (list)
                    loaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_run_id, regime_date)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stg_macro_regime_date
                ON stg_ra_macro_regime(regime_date)
            """)

            # stg_ra_etf_signal: ETF 신호 원시 적재
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stg_ra_etf_signal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_run_id TEXT,           -- 분석 실행 ID
                    signal_date TEXT NOT NULL,    -- 신호 발생 날짜
                    etf TEXT NOT NULL,            -- ETF 심볼 (ARKK, ARKG, ...)
                    ticker TEXT NOT NULL,         -- 종목 심볼
                    signal_type TEXT,             -- etf_flow, weight_change, sector_rotation
                    direction TEXT,               -- long, short, neutral
                    confidence REAL,              -- 신뢰도 (0.0~1.0)
                    indicator TEXT,               -- 신호 지표명
                    indicator_value REAL,         -- 지표 값
                    z_score REAL,                 -- Z-점수
                    level TEXT,                   -- WARNING, ALERT, CRITICAL
                    description TEXT,             -- 신호 설명
                    horizon TEXT,                 -- short, medium, long
                    metadata_json TEXT,           -- 추가 메타데이터 JSON
                    loaded_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source_run_id, signal_date, etf, ticker, signal_type)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stg_etf_signal_date
                ON stg_ra_etf_signal(signal_date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_stg_etf_signal_etf_ticker
                ON stg_ra_etf_signal(etf, ticker)
            """)

            # ================================================================
            # Track E — fi_ra Mart 테이블
            # ================================================================
            # mart_ra_macro_regime: 레짐 분석 집계 결과
            # staging에서 ETL 처리 후 분석에 직접 사용하는 mart 테이블.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mart_ra_macro_regime (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    regime_date TEXT NOT NULL UNIQUE, -- 집계 기준 날짜 (일별 1행)
                    regime_type TEXT NOT NULL,        -- 최종 확정 레짐 유형
                    cycle_phase TEXT,                 -- 사이클 국면
                    style_bias TEXT,                  -- 스타일 편향
                    avg_confidence REAL,              -- 평균 신뢰도
                    max_confidence REAL,              -- 최대 신뢰도
                    avg_risk_appetite REAL,           -- 평균 위험 선호 점수
                    avg_breadth_score REAL,           -- 평균 시장 폭 점수
                    avg_vix_estimate REAL,            -- 평균 VIX 추정치
                    avg_growth_value_spread REAL,     -- 평균 성장/가치 스프레드
                    avg_equity_bond_spread REAL,      -- 평균 주식/채권 스프레드
                    avg_hy_treasury_spread REAL,      -- 평균 HY/국채 스프레드
                    source_run_count INTEGER,         -- 집계에 사용된 소스 실행 수
                    key_indicators_json TEXT,         -- 대표 핵심 지표 JSON
                    etl_processed_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mart_macro_regime_date
                ON mart_ra_macro_regime(regime_date)
            """)

            # mart_ra_etf_signal: ETF 신호 + 레짐 컨텍스트 집계
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mart_ra_etf_signal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_date TEXT NOT NULL,        -- 신호 날짜
                    etf TEXT NOT NULL,                -- ETF 심볼
                    ticker TEXT NOT NULL,             -- 종목 심볼
                    signal_type TEXT NOT NULL,        -- 신호 유형
                    direction TEXT,                   -- 방향성
                    avg_confidence REAL,              -- 평균 신뢰도
                    max_confidence REAL,              -- 최대 신뢰도
                    avg_z_score REAL,                 -- 평균 Z-점수
                    dominant_level TEXT,              -- 지배적 경보 수준
                    horizon TEXT,                     -- 투자 시계
                    regime_type TEXT,                 -- 조인된 레짐 유형 (당일)
                    regime_confidence REAL,           -- 레짐 신뢰도
                    regime_aligned INTEGER,           -- 레짐 정렬 여부 (1/0)
                    signal_count INTEGER,             -- 집계 신호 수
                    description TEXT,                 -- 대표 설명
                    etl_processed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(signal_date, etf, ticker, signal_type)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mart_etf_signal_date
                ON mart_ra_etf_signal(signal_date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_mart_etf_signal_regime
                ON mart_ra_etf_signal(regime_type)
            """)

            # ================================================================
            # Track E — fi_ra SQL Views
            # ================================================================

            # v_ra_macro_regime: 레짐 분석 통합 조회 뷰
            #
            # 우선순위: mart 테이블 (ETL 완료) → market_regime 코어 테이블 (fallback)
            # mart에 데이터가 있으면 mart 기준, 없으면 market_regime 원본 사용.
            # 컬럼: regime_date, regime_type, cycle_phase, style_bias,
            #        confidence, risk_appetite_score, breadth_score,
            #        vix_estimate, key_indicators_json, data_source
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS v_ra_macro_regime AS
                SELECT
                    m.regime_date,
                    m.regime_type,
                    m.cycle_phase,
                    m.style_bias,
                    m.avg_confidence        AS confidence,
                    m.avg_risk_appetite     AS risk_appetite_score,
                    m.avg_breadth_score     AS breadth_score,
                    m.avg_vix_estimate      AS vix_estimate,
                    m.avg_growth_value_spread AS growth_value_spread,
                    m.avg_equity_bond_spread  AS equity_bond_spread,
                    m.avg_hy_treasury_spread  AS hy_treasury_spread,
                    m.source_run_count,
                    m.key_indicators_json,
                    m.etl_processed_at,
                    'mart' AS data_source
                FROM mart_ra_macro_regime m

                UNION ALL

                SELECT
                    r.date                  AS regime_date,
                    r.sentiment             AS regime_type,
                    r.cycle_phase,
                    r.style_rotation        AS style_bias,
                    NULL                    AS confidence,
                    r.risk_appetite_score,
                    r.breadth_score,
                    r.vix_estimate,
                    r.growth_value_spread,
                    r.equity_bond_spread,
                    r.hy_treasury_spread,
                    NULL                    AS source_run_count,
                    NULL                    AS key_indicators_json,
                    r.created_at            AS etl_processed_at,
                    'core' AS data_source
                FROM market_regime r
                WHERE r.date NOT IN (
                    SELECT regime_date FROM mart_ra_macro_regime
                )
            """)

            # ================================================================
            # Korea Savings Bank Indicators 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS korea_savings_bank (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    npl_ratio REAL,           -- 고정이하여신비율 (%)
                    bis_capital_ratio REAL,   -- BIS 자기자본비율 (%)
                    roa REAL,                 -- 총자산순이익률 (%)
                    data_source TEXT,         -- "fred+fss_mock", "fss_mock", etc.
                    signals_json TEXT,        -- 경보 신호 목록 JSON
                    note TEXT,
                    is_valid INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_ksb_date ON korea_savings_bank(date)"
            )

            # v_ra_etf_signal: ETF 신호 + 레짐 컨텍스트 조인 뷰
            #
            # mart_ra_etf_signal을 기반으로 당일 레짐 정보를 함께 노출.
            # 레짐 정보는 v_ra_macro_regime 뷰에서 조인하여 최신 데이터 보장.
            # 컬럼: signal_date, etf, ticker, signal_type, direction,
            #        confidence, z_score, level, regime_type, regime_confidence,
            #        regime_aligned, horizon, description
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS v_ra_etf_signal AS
                SELECT
                    s.signal_date,
                    s.etf,
                    s.ticker,
                    s.signal_type,
                    s.direction,
                    s.avg_confidence        AS confidence,
                    s.avg_z_score           AS z_score,
                    s.dominant_level        AS level,
                    s.horizon,
                    s.signal_count,
                    s.description,
                    COALESCE(s.regime_type, r.regime_type) AS regime_type,
                    COALESCE(s.regime_confidence, r.confidence) AS regime_confidence,
                    s.regime_aligned,
                    r.cycle_phase           AS regime_cycle_phase,
                    r.style_bias            AS regime_style_bias,
                    r.risk_appetite_score   AS regime_risk_appetite,
                    s.etl_processed_at
                FROM mart_ra_etf_signal s
                LEFT JOIN v_ra_macro_regime r
                    ON s.signal_date = r.regime_date
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
    # Korea Savings Bank 메서드
    # ========================================================================

    def save_korea_savings_bank(self, data: Dict[str, Any], date_str: str = None):
        """한국 저축은행 건전성 지표 저장"""
        if date_str is None:
            date_str = date.today().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO korea_savings_bank
                (date, npl_ratio, bis_capital_ratio, roa,
                 data_source, signals_json, note, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                date_str,
                data.get("npl_ratio", 0.0),
                data.get("bis_capital_ratio", 0.0),
                data.get("roa", 0.0),
                data.get("data_source", "fss_mock"),
                json.dumps(data.get("signals", [])),
                data.get("note", ""),
                1 if data.get("is_valid", True) else 0,
            ))

    def get_korea_savings_bank(self, date_str: str = None) -> Optional[Dict[str, Any]]:
        """한국 저축은행 건전성 지표 조회 (기본: 최신)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if date_str:
                cursor.execute(
                    "SELECT * FROM korea_savings_bank WHERE date = ?", (date_str,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM korea_savings_bank ORDER BY date DESC LIMIT 1"
                )
            row = cursor.fetchone()
            if row:
                result = dict(row)
                raw = result.get("signals_json")
                result["signals"] = json.loads(raw) if raw else []
                result["is_valid"] = bool(result.get("is_valid"))
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
    # Track E — fi_ra Staging 적재 메서드
    # ========================================================================

    def stage_macro_regime(self, regime: Dict[str, Any],
                           run_id: str = None,
                           date_str: str = None) -> int:
        """
        레짐 분석 결과를 staging 테이블에 적재

        Args:
            regime: 레짐 분석 결과 딕셔너리
            run_id: 배치 실행 ID (추적용, 기본: 타임스탬프)
            date_str: 레짐 기준 날짜 (기본: 오늘)

        Returns:
            삽입된 행의 ID
        """
        if date_str is None:
            date_str = date.today().isoformat()
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO stg_ra_macro_regime
                (source_run_id, regime_date, regime_type, cycle_phase, style_bias,
                 confidence, risk_appetite_score, breadth_score, vix_estimate,
                 growth_value_spread, equity_bond_spread, hy_treasury_spread,
                 key_indicators_json, raw_signals_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                date_str,
                regime.get('sentiment') or regime.get('regime_type', ''),
                regime.get('cycle_phase', ''),
                regime.get('style_rotation') or regime.get('style_bias', ''),
                regime.get('confidence'),
                regime.get('risk_appetite_score'),
                regime.get('breadth_score'),
                regime.get('vix_estimate'),
                regime.get('growth_value_spread'),
                regime.get('equity_bond_spread'),
                regime.get('hy_treasury_spread'),
                json.dumps(regime.get('key_indicators', {})),
                json.dumps(regime.get('signals', []))
            ))
            return cursor.lastrowid

    def stage_etf_signals(self, signals: List[Dict[str, Any]],
                          run_id: str = None,
                          date_str: str = None) -> int:
        """
        ETF 신호 목록을 staging 테이블에 적재

        Args:
            signals: 신호 딕셔너리 리스트
            run_id: 배치 실행 ID
            date_str: 신호 날짜 (기본: 오늘)

        Returns:
            적재된 신호 수
        """
        if date_str is None:
            date_str = date.today().isoformat()
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        count = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for sig in signals:
                cursor.execute("""
                    INSERT OR REPLACE INTO stg_ra_etf_signal
                    (source_run_id, signal_date, etf, ticker, signal_type,
                     direction, confidence, indicator, indicator_value, z_score,
                     level, description, horizon, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id,
                    date_str,
                    sig.get('etf', ''),
                    sig.get('ticker', ''),
                    sig.get('type') or sig.get('signal_type', ''),
                    sig.get('direction', ''),
                    sig.get('confidence'),
                    sig.get('indicator', ''),
                    sig.get('value') or sig.get('indicator_value'),
                    sig.get('z_score'),
                    sig.get('level', ''),
                    sig.get('description', ''),
                    sig.get('horizon', ''),
                    json.dumps(sig.get('metadata', {}))
                ))
                count += 1
        return count

    # ========================================================================
    # Track E — ETL: Staging → Mart
    # ========================================================================

    def etl_regime_to_mart(self, regime_date: str = None) -> int:
        """
        stg_ra_macro_regime → mart_ra_macro_regime ETL 실행

        staging 테이블의 레짐 데이터를 집계하여 mart 테이블에 upsert.
        같은 날짜에 여러 실행(run)이 있을 경우 평균값 사용.

        Args:
            regime_date: 특정 날짜만 처리 (None이면 미처리 전체)

        Returns:
            처리된 날짜 수
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            date_filter = "WHERE regime_date = ?" if regime_date else ""
            params = (regime_date,) if regime_date else ()

            cursor.execute(f"""
                INSERT OR REPLACE INTO mart_ra_macro_regime
                (regime_date, regime_type, cycle_phase, style_bias,
                 avg_confidence, max_confidence,
                 avg_risk_appetite, avg_breadth_score, avg_vix_estimate,
                 avg_growth_value_spread, avg_equity_bond_spread,
                 avg_hy_treasury_spread, source_run_count, key_indicators_json)
                SELECT
                    regime_date,
                    -- 최다 출현 레짐 유형 선택 (GROUP_CONCAT + 서브쿼리 대신 단순화)
                    regime_type,
                    cycle_phase,
                    style_bias,
                    AVG(COALESCE(confidence, 0.5))          AS avg_confidence,
                    MAX(COALESCE(confidence, 0.0))          AS max_confidence,
                    AVG(COALESCE(risk_appetite_score, 0))   AS avg_risk_appetite,
                    AVG(COALESCE(breadth_score, 0))         AS avg_breadth_score,
                    AVG(COALESCE(vix_estimate, 0))          AS avg_vix_estimate,
                    AVG(COALESCE(growth_value_spread, 0))   AS avg_growth_value_spread,
                    AVG(COALESCE(equity_bond_spread, 0))    AS avg_equity_bond_spread,
                    AVG(COALESCE(hy_treasury_spread, 0))    AS avg_hy_treasury_spread,
                    COUNT(DISTINCT source_run_id)           AS source_run_count,
                    -- 가장 최근 실행의 key_indicators 사용
                    MAX(key_indicators_json)                AS key_indicators_json
                FROM stg_ra_macro_regime
                {date_filter}
                GROUP BY regime_date, regime_type, cycle_phase, style_bias
            """, params)

            return cursor.rowcount

    def etl_etf_signals_to_mart(self, signal_date: str = None) -> int:
        """
        stg_ra_etf_signal → mart_ra_etf_signal ETL 실행

        staging 신호를 집계하고 당일 레짐 정보를 조인하여 mart에 upsert.

        Args:
            signal_date: 특정 날짜만 처리 (None이면 미처리 전체)

        Returns:
            처리된 행 수
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            date_filter = "WHERE s.signal_date = ?" if signal_date else ""
            params = (signal_date,) if signal_date else ()

            cursor.execute(f"""
                INSERT OR REPLACE INTO mart_ra_etf_signal
                (signal_date, etf, ticker, signal_type, direction,
                 avg_confidence, max_confidence, avg_z_score, dominant_level,
                 horizon, regime_type, regime_confidence, regime_aligned,
                 signal_count, description)
                SELECT
                    s.signal_date,
                    s.etf,
                    s.ticker,
                    s.signal_type,
                    s.direction,
                    AVG(COALESCE(s.confidence, 0))          AS avg_confidence,
                    MAX(COALESCE(s.confidence, 0))          AS max_confidence,
                    AVG(COALESCE(s.z_score, 0))             AS avg_z_score,
                    -- 가장 높은 경보 수준 선택 (CRITICAL > ALERT > WARNING)
                    CASE
                        WHEN MAX(CASE s.level
                            WHEN 'CRITICAL' THEN 3
                            WHEN 'ALERT'    THEN 2
                            WHEN 'WARNING'  THEN 1
                            ELSE 0 END) = 3 THEN 'CRITICAL'
                        WHEN MAX(CASE s.level
                            WHEN 'CRITICAL' THEN 3
                            WHEN 'ALERT'    THEN 2
                            WHEN 'WARNING'  THEN 1
                            ELSE 0 END) = 2 THEN 'ALERT'
                        WHEN MAX(CASE s.level
                            WHEN 'CRITICAL' THEN 3
                            WHEN 'ALERT'    THEN 2
                            WHEN 'WARNING'  THEN 1
                            ELSE 0 END) = 1 THEN 'WARNING'
                        ELSE NULL
                    END                                     AS dominant_level,
                    s.horizon,
                    r.sentiment                             AS regime_type,
                    r.risk_appetite_score                   AS regime_confidence,
                    -- regime_aligned: 신호 방향과 레짐 일치 여부
                    CASE
                        WHEN r.sentiment = 'RISK_ON'  AND s.direction = 'long'  THEN 1
                        WHEN r.sentiment = 'RISK_OFF' AND s.direction = 'short' THEN 1
                        WHEN r.sentiment = 'NEUTRAL'                             THEN 1
                        ELSE 0
                    END                                     AS regime_aligned,
                    COUNT(*)                                AS signal_count,
                    MAX(s.description)                      AS description
                FROM stg_ra_etf_signal s
                LEFT JOIN market_regime r ON s.signal_date = r.date
                {date_filter}
                GROUP BY s.signal_date, s.etf, s.ticker, s.signal_type,
                         s.direction, s.horizon, r.sentiment, r.risk_appetite_score
            """, params)

            return cursor.rowcount

    # ========================================================================
    # Track E — fi_ra View 기반 조회 함수
    # ========================================================================

    def get_latest_regime(self) -> Optional[Dict[str, Any]]:
        """
        v_ra_macro_regime 뷰에서 최신 레짐 분석 결과 조회

        mart 데이터를 우선 사용하고, 없으면 core market_regime 테이블 사용.

        Returns:
            최신 레짐 딕셔너리 또는 None
            {
                'regime_date': str,
                'regime_type': str,   # RISK_ON / RISK_OFF / NEUTRAL
                'cycle_phase': str,   # EARLY / MID / LATE / RECESSION
                'style_bias': str,    # GROWTH / VALUE / BALANCED
                'confidence': float,
                'risk_appetite_score': float,
                'breadth_score': float,
                'vix_estimate': float,
                'growth_value_spread': float,
                'equity_bond_spread': float,
                'hy_treasury_spread': float,
                'key_indicators': dict,
                'data_source': str,   # 'mart' or 'core'
            }
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT *
                FROM v_ra_macro_regime
                ORDER BY regime_date DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                result = dict(row)
                # key_indicators_json 파싱
                raw_ki = result.get('key_indicators_json')
                result['key_indicators'] = json.loads(raw_ki) if raw_ki else {}
                return result
            return None

    def get_regime_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        v_ra_macro_regime 뷰에서 레짐 이력 조회

        Args:
            days: 조회할 최근 일수

        Returns:
            레짐 이력 리스트 (최신순)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT *
                FROM v_ra_macro_regime
                ORDER BY regime_date DESC
                LIMIT ?
            """, (days,))
            results = []
            for row in cursor.fetchall():
                r = dict(row)
                raw_ki = r.get('key_indicators_json')
                r['key_indicators'] = json.loads(raw_ki) if raw_ki else {}
                results.append(r)
            return results

    def get_signals_by_regime(self, regime_type: str,
                               min_confidence: float = 0.0,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """
        v_ra_etf_signal 뷰에서 특정 레짐에 속한 ETF 신호 조회

        mart 테이블과 레짐 뷰를 조인한 결과를 반환.

        Args:
            regime_type: 레짐 유형 필터 (예: "RISK_ON", "RISK_OFF", "NEUTRAL")
            min_confidence: 최소 신뢰도 필터 (0.0~1.0)
            limit: 최대 반환 행 수

        Returns:
            신호 딕셔너리 리스트
            각 딕셔너리 포함 컬럼:
            signal_date, etf, ticker, signal_type, direction,
            confidence, z_score, level, horizon, signal_count,
            regime_type, regime_confidence, regime_aligned,
            regime_cycle_phase, regime_style_bias, regime_risk_appetite
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT *
                FROM v_ra_etf_signal
                WHERE regime_type = ?
                  AND COALESCE(confidence, 0) >= ?
                ORDER BY signal_date DESC, confidence DESC
                LIMIT ?
            """, (regime_type, min_confidence, limit))
            return [dict(row) for row in cursor.fetchall()]

    def get_latest_etf_signals(self, etf: str = None,
                                ticker: str = None,
                                limit: int = 20) -> List[Dict[str, Any]]:
        """
        v_ra_etf_signal 뷰에서 최신 ETF 신호 조회

        Args:
            etf: ETF 심볼 필터 (예: "ARKK")
            ticker: 종목 심볼 필터 (예: "TSLA")
            limit: 최대 반환 행 수

        Returns:
            ETF 신호 딕셔너리 리스트 (레짐 컨텍스트 포함)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM v_ra_etf_signal WHERE 1=1"
            params: List[Any] = []

            if etf:
                query += " AND etf = ?"
                params.append(etf)
            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)

            query += " ORDER BY signal_date DESC, confidence DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

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
                      'signals', 'actions', 'etf_analysis', 'analysis_log',
                      'stg_ra_macro_regime', 'stg_ra_etf_signal',
                      'mart_ra_macro_regime', 'mart_ra_etf_signal',
                      'korea_savings_bank']

            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]

                # date 컬럼이 없는 테이블은 NULL 처리
                try:
                    cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
                    row = cursor.fetchone()
                    min_date, max_date = row[0], row[1]
                except Exception:
                    min_date, max_date = None, None

                stats['tables'][table] = {
                    'count': count,
                    'min_date': min_date,
                    'max_date': max_date
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

    # Track E — Staging 적재 테스트
    print("\n[6] Testing Track E — Staging Load...")
    run_id = "test_run_001"
    stg_id = db.stage_macro_regime(test_regime, run_id=run_id)
    print(f"    Staged macro regime, stg id: {stg_id}")

    etf_signals = [
        {
            'etf': 'ARKK', 'ticker': 'TSLA', 'type': 'etf_flow',
            'direction': 'long', 'confidence': 0.80,
            'indicator': 'weight_change', 'value': 1.5,
            'z_score': 2.1, 'level': 'ALERT',
            'description': 'ARKK TSLA weight increase alert',
            'horizon': 'short'
        },
        {
            'etf': 'ARKG', 'ticker': 'RXRX', 'type': 'etf_flow',
            'direction': 'long', 'confidence': 0.65,
            'indicator': 'weight_change', 'value': 0.8,
            'z_score': 1.5, 'level': 'WARNING',
            'description': 'ARKG RXRX weight increase',
            'horizon': 'medium'
        },
    ]
    stg_count = db.stage_etf_signals(etf_signals, run_id=run_id)
    print(f"    Staged {stg_count} ETF signals")

    # Track E — ETL 실행 테스트
    print("\n[7] Testing Track E — ETL to Mart...")
    etl_regime_rows = db.etl_regime_to_mart()
    print(f"    ETL regime → mart: {etl_regime_rows} rows processed")
    etl_signal_rows = db.etl_etf_signals_to_mart()
    print(f"    ETL etf_signal → mart: {etl_signal_rows} rows processed")

    # Track E — View 기반 조회 테스트
    print("\n[8] Testing Track E — View Queries...")
    latest_regime = db.get_latest_regime()
    if latest_regime:
        print(f"    get_latest_regime(): {latest_regime.get('regime_type')} "
              f"({latest_regime.get('data_source')}) "
              f"date={latest_regime.get('regime_date')}")
    else:
        print("    get_latest_regime(): No data")

    signals_by_regime = db.get_signals_by_regime("RISK_ON")
    print(f"    get_signals_by_regime('RISK_ON'): {len(signals_by_regime)} signals")

    latest_etf = db.get_latest_etf_signals(etf='ARKK')
    print(f"    get_latest_etf_signals(etf='ARKK'): {len(latest_etf)} signals")

    # 통계 출력
    print("\n[9] Database Stats:")
    stats = db.get_stats()
    for table, info in stats['tables'].items():
        print(f"    {table:30s}: {info['count']:5d} records")

    # 최신 날짜
    print("\n[10] Latest Dates:")
    dates = db.get_latest_dates()
    for table, d in dates.items():
        print(f"    {table:20s}: {d}")

    print("\n" + "=" * 70)
    print("Database Test Complete!")
    print("=" * 70)
