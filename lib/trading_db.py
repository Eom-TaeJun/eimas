#!/usr/bin/env python3
"""
EIMAS Trading Database
======================
트레이딩 실행, 시그널, 포트폴리오, 성과 추적 DB

Usage:
    from lib.trading_db import TradingDB, Signal, PortfolioCandidate

    db = TradingDB()
    db.save_signal(signal)
    db.save_portfolio(portfolio)
    db.update_performance(portfolio_id, actual_returns)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import sqlite3
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path


# ============================================================================
# Constants & Enums
# ============================================================================

class SignalSource(str, Enum):
    """시그널 소스"""
    REGIME_DETECTOR = "regime_detector"
    CRITICAL_PATH = "critical_path"
    ETF_FLOW = "etf_flow"
    FEAR_GREED = "fear_greed"
    FRED_INDICATOR = "fred_indicator"
    VIX_STRUCTURE = "vix_structure"
    BACKTESTER = "backtester"
    MANUAL = "manual"


class SignalAction(str, Enum):
    """시그널 액션"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    REDUCE = "reduce"
    HEDGE = "hedge"


class InvestorProfile(str, Enum):
    """투자자 프로파일"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    TACTICAL = "tactical"


class SessionType(str, Enum):
    """거래 세션"""
    PRE_MARKET = "pre_market"
    OPENING = "opening"
    MID_DAY = "mid_day"
    POWER_HOUR = "power_hour"
    AFTER_HOURS = "after_hours"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Signal:
    """시그널 데이터"""
    source: SignalSource
    action: SignalAction
    ticker: str = "SPY"
    conviction: float = 0.5  # 0.0 - 1.0
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None

    @property
    def confidence(self) -> float:
        """Alias for conviction"""
        return self.conviction

    def to_dict(self) -> Dict:
        return {
            'source': self.source.value,
            'action': self.action.value,
            'ticker': self.ticker,
            'conviction': self.conviction,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class PortfolioCandidate:
    """포트폴리오 후보"""
    profile: InvestorProfile
    allocations: Dict[str, float]  # {"SPY": 0.4, "TLT": 0.3, "GLD": 0.2, "CASH": 0.1}
    expected_return: float  # 연환산 예상 수익률
    expected_risk: float    # 연환산 예상 변동성
    expected_sharpe: float
    signals_used: List[int]  # 사용된 시그널 ID
    reasoning: str = ""
    rank: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            'profile': self.profile.value,
            'allocations': self.allocations,
            'expected_return': round(self.expected_return, 4),
            'expected_risk': round(self.expected_risk, 4),
            'expected_sharpe': round(self.expected_sharpe, 2),
            'signals_used': self.signals_used,
            'reasoning': self.reasoning,
            'rank': self.rank,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class Execution:
    """실행 기록"""
    portfolio_id: int
    ticker: str
    action: SignalAction
    session: SessionType
    target_price: float
    executed_price: float
    shares: float
    commission: float = 0.0
    slippage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    id: Optional[int] = None


@dataclass
class PerformanceRecord:
    """성과 기록"""
    portfolio_id: int
    date: date
    # 예측값
    predicted_return_1d: float = 0.0
    predicted_return_1w: float = 0.0
    predicted_return_1m: float = 0.0
    # 실제값 (나중에 업데이트)
    actual_return_1d: Optional[float] = None
    actual_return_1w: Optional[float] = None
    actual_return_1m: Optional[float] = None
    # 평가
    mape: Optional[float] = None
    id: Optional[int] = None


@dataclass
class SignalPerformance:
    """시그널 성과"""
    signal_id: int
    evaluation_date: date
    return_1d: float
    return_5d: float
    return_20d: float
    max_gain: float
    max_loss: float
    signal_accuracy: bool
    id: Optional[int] = None


@dataclass
class SessionAnalysis:
    """세션별 분석"""
    date: date
    ticker: str
    pre_market_return: float = 0.0
    opening_hour_return: float = 0.0
    mid_day_return: float = 0.0
    power_hour_return: float = 0.0
    after_hours_return: float = 0.0
    overnight_return: float = 0.0
    best_buy_time: Optional[str] = None
    best_sell_time: Optional[str] = None
    volume_distribution: Dict[str, float] = field(default_factory=dict)
    id: Optional[int] = None


# ============================================================================
# Database Class
# ============================================================================

class TradingDB:
    """트레이딩 데이터베이스"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path(__file__).parent.parent / "data" / "trading.db")

        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """테이블 초기화"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 1. Signals 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                signal_source VARCHAR(50) NOT NULL,
                signal_action VARCHAR(20) NOT NULL,
                ticker VARCHAR(10),
                conviction FLOAT,
                reasoning TEXT,
                metadata JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Portfolio Candidates 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                profile_type VARCHAR(20) NOT NULL,
                candidate_rank INTEGER,
                allocations JSON,
                expected_return FLOAT,
                expected_risk FLOAT,
                expected_sharpe FLOAT,
                signals_used JSON,
                reasoning TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. Executions 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                execution_time DATETIME NOT NULL,
                session_type VARCHAR(20),
                ticker VARCHAR(10) NOT NULL,
                action VARCHAR(10) NOT NULL,
                target_price FLOAT,
                executed_price FLOAT,
                slippage FLOAT,
                shares FLOAT,
                commission FLOAT,
                status VARCHAR(20) DEFAULT 'filled',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES portfolio_candidates(id)
            )
        """)

        # 4. Performance Tracking 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                date DATE NOT NULL,
                predicted_return_1d FLOAT,
                predicted_return_1w FLOAT,
                predicted_return_1m FLOAT,
                predicted_volatility FLOAT,
                actual_return_1d FLOAT,
                actual_return_1w FLOAT,
                actual_return_1m FLOAT,
                actual_volatility FLOAT,
                prediction_error_1d FLOAT,
                prediction_error_1w FLOAT,
                prediction_error_1m FLOAT,
                mape FLOAT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME,
                FOREIGN KEY (portfolio_id) REFERENCES portfolio_candidates(id)
            )
        """)

        # 5. Signal Performance 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                evaluation_date DATE,
                return_1d FLOAT,
                return_5d FLOAT,
                return_20d FLOAT,
                return_60d FLOAT,
                max_gain FLOAT,
                max_loss FLOAT,
                signal_accuracy BOOLEAN,
                profit_factor FLOAT,
                information_ratio FLOAT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (signal_id) REFERENCES signals(id)
            )
        """)

        # 6. Session Analysis 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                pre_market_return FLOAT,
                opening_hour_return FLOAT,
                mid_day_return FLOAT,
                power_hour_return FLOAT,
                after_hours_return FLOAT,
                overnight_return FLOAT,
                best_buy_time TIME,
                best_sell_time TIME,
                volume_distribution JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 7. Backtest Runs 테이블 (v2.0 추가)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name VARCHAR(100) NOT NULL,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                initial_capital FLOAT NOT NULL,
                final_capital FLOAT,
                total_return FLOAT,
                annual_return FLOAT,
                benchmark_return FLOAT,
                alpha FLOAT,
                volatility FLOAT,
                max_drawdown FLOAT,
                max_drawdown_duration INTEGER,
                sharpe_ratio FLOAT,
                sortino_ratio FLOAT,
                calmar_ratio FLOAT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate FLOAT,
                avg_win FLOAT,
                avg_loss FLOAT,
                profit_factor FLOAT,
                avg_holding_days FLOAT,
                total_commission FLOAT,
                total_slippage FLOAT,
                total_short_cost FLOAT,
                parameters JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 8. Backtest Trades 테이블 (v2.0 추가)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                entry_date DATE NOT NULL,
                exit_date DATE NOT NULL,
                ticker VARCHAR(10) NOT NULL,
                direction VARCHAR(10) NOT NULL,
                entry_price FLOAT,
                exit_price FLOAT,
                shares FLOAT,
                pnl FLOAT,
                pnl_pct FLOAT,
                holding_days INTEGER,
                commission FLOAT,
                slippage_cost FLOAT,
                short_cost FLOAT,
                signal_reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
            )
        """)

        # 9. Walk-Forward Results 테이블 (v2.0 추가)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS walk_forward_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                fold_number INTEGER NOT NULL,
                train_start DATE NOT NULL,
                train_end DATE NOT NULL,
                test_start DATE NOT NULL,
                test_end DATE NOT NULL,
                in_sample_return FLOAT,
                in_sample_sharpe FLOAT,
                out_sample_return FLOAT,
                out_sample_sharpe FLOAT,
                degradation_pct FLOAT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES backtest_runs(id)
            )
        """)

        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(signal_source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_profile ON portfolio_candidates(profile_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_tracking(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_perf_date ON signal_performance(evaluation_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_date_ticker ON session_analysis(date, ticker)")

        # v2.0 백테스트 인덱스
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_runs(strategy_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_date ON backtest_runs(start_date, end_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(run_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_walk_forward_run ON walk_forward_results(run_id)")

        conn.commit()
        conn.close()

    # ========================================================================
    # Signal Methods
    # ========================================================================

    def save_signal(self, signal: Signal) -> int:
        """시그널 저장"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (timestamp, signal_source, signal_action, ticker,
                                conviction, reasoning, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.timestamp.isoformat(),
            signal.source.value,
            signal.action.value,
            signal.ticker,
            signal.conviction,
            signal.reasoning,
            json.dumps(signal.metadata),
        ))

        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id

    def get_signals(
        self,
        source: SignalSource = None,
        start_date: date = None,
        end_date: date = None,
        limit: int = 100
    ) -> List[Dict]:
        """시그널 조회"""
        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT * FROM signals WHERE 1=1"
        params = []

        if source:
            query += " AND signal_source = ?"
            params.append(source.value)

        if start_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND DATE(timestamp) <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_recent_signals(self, hours: int = 24) -> List[Dict]:
        """최근 시그널 조회"""
        cutoff = datetime.now() - timedelta(hours=hours)
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM signals
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (cutoff.isoformat(),))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    # ========================================================================
    # Portfolio Methods
    # ========================================================================

    def save_portfolio(self, portfolio: PortfolioCandidate) -> int:
        """포트폴리오 후보 저장"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO portfolio_candidates
            (timestamp, profile_type, candidate_rank, allocations,
             expected_return, expected_risk, expected_sharpe, signals_used, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            portfolio.timestamp.isoformat(),
            portfolio.profile.value,
            portfolio.rank,
            json.dumps(portfolio.allocations),
            portfolio.expected_return,
            portfolio.expected_risk,
            portfolio.expected_sharpe,
            json.dumps(portfolio.signals_used),
            portfolio.reasoning,
        ))

        portfolio_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return portfolio_id

    def get_portfolios(
        self,
        profile: InvestorProfile = None,
        start_date: date = None,
        limit: int = 50
    ) -> List[Dict]:
        """포트폴리오 조회"""
        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT * FROM portfolio_candidates WHERE 1=1"
        params = []

        if profile:
            query += " AND profile_type = ?"
            params.append(profile.value)

        if start_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(start_date.isoformat())

        query += " ORDER BY timestamp DESC, candidate_rank ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            d = dict(row)
            d['allocations'] = json.loads(d['allocations']) if d['allocations'] else {}
            d['signals_used'] = json.loads(d['signals_used']) if d['signals_used'] else []
            results.append(d)

        return results

    # ========================================================================
    # Execution Methods
    # ========================================================================

    def save_execution(self, execution: Execution) -> int:
        """실행 기록 저장"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO executions
            (portfolio_id, execution_time, session_type, ticker, action,
             target_price, executed_price, slippage, shares, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.portfolio_id,
            execution.timestamp.isoformat(),
            execution.session.value,
            execution.ticker,
            execution.action.value,
            execution.target_price,
            execution.executed_price,
            execution.slippage,
            execution.shares,
            execution.commission,
        ))

        exec_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return exec_id

    # ========================================================================
    # Performance Methods
    # ========================================================================

    def save_prediction(
        self,
        portfolio_id: int,
        pred_1d: float,
        pred_1w: float,
        pred_1m: float
    ) -> int:
        """예측값 저장"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO performance_tracking
            (portfolio_id, date, predicted_return_1d, predicted_return_1w, predicted_return_1m)
            VALUES (?, ?, ?, ?, ?)
        """, (
            portfolio_id,
            date.today().isoformat(),
            pred_1d,
            pred_1w,
            pred_1m,
        ))

        perf_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return perf_id

    def update_actual_returns(
        self,
        portfolio_id: int,
        record_date: date,
        actual_1d: float = None,
        actual_1w: float = None,
        actual_1m: float = None
    ):
        """실제 수익률 업데이트"""
        conn = self._get_conn()
        cursor = conn.cursor()

        updates = []
        params = []

        if actual_1d is not None:
            updates.append("actual_return_1d = ?")
            params.append(actual_1d)

        if actual_1w is not None:
            updates.append("actual_return_1w = ?")
            params.append(actual_1w)

        if actual_1m is not None:
            updates.append("actual_return_1m = ?")
            params.append(actual_1m)

        if not updates:
            return

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.extend([portfolio_id, record_date.isoformat()])

        query = f"""
            UPDATE performance_tracking
            SET {', '.join(updates)}
            WHERE portfolio_id = ? AND date = ?
        """

        cursor.execute(query, params)

        # MAPE 계산
        cursor.execute("""
            UPDATE performance_tracking
            SET
                prediction_error_1d = actual_return_1d - predicted_return_1d,
                prediction_error_1w = actual_return_1w - predicted_return_1w,
                prediction_error_1m = actual_return_1m - predicted_return_1m,
                mape = (
                    ABS(actual_return_1d - predicted_return_1d) / NULLIF(ABS(actual_return_1d), 0) +
                    ABS(actual_return_1w - predicted_return_1w) / NULLIF(ABS(actual_return_1w), 0) +
                    ABS(actual_return_1m - predicted_return_1m) / NULLIF(ABS(actual_return_1m), 0)
                ) / 3
            WHERE portfolio_id = ? AND date = ?
        """, (portfolio_id, record_date.isoformat()))

        conn.commit()
        conn.close()

    def get_performance_history(
        self,
        portfolio_id: int = None,
        days: int = 30
    ) -> List[Dict]:
        """성과 이력 조회"""
        conn = self._get_conn()
        cursor = conn.cursor()

        query = """
            SELECT * FROM performance_tracking
            WHERE date >= ?
        """
        params = [(date.today() - timedelta(days=days)).isoformat()]

        if portfolio_id:
            query += " AND portfolio_id = ?"
            params.append(portfolio_id)

        query += " ORDER BY date DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    # ========================================================================
    # Signal Performance Methods
    # ========================================================================

    def save_signal_performance(self, perf: SignalPerformance) -> int:
        """시그널 성과 저장"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signal_performance
            (signal_id, evaluation_date, return_1d, return_5d, return_20d,
             max_gain, max_loss, signal_accuracy)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            perf.signal_id,
            perf.evaluation_date.isoformat(),
            perf.return_1d,
            perf.return_5d,
            perf.return_20d,
            perf.max_gain,
            perf.max_loss,
            perf.signal_accuracy,
        ))

        perf_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return perf_id

    def get_signal_accuracy_by_source(self) -> Dict[str, float]:
        """소스별 시그널 정확도"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                s.signal_source,
                COUNT(*) as total,
                SUM(CASE WHEN sp.signal_accuracy = 1 THEN 1 ELSE 0 END) as correct,
                AVG(CASE WHEN sp.signal_accuracy = 1 THEN 1.0 ELSE 0.0 END) as accuracy
            FROM signals s
            LEFT JOIN signal_performance sp ON s.id = sp.signal_id
            WHERE sp.id IS NOT NULL
            GROUP BY s.signal_source
        """)

        rows = cursor.fetchall()
        conn.close()

        return {row['signal_source']: round(row['accuracy'] * 100, 1) for row in rows}

    # ========================================================================
    # Session Analysis Methods
    # ========================================================================

    def save_session_analysis(self, analysis: SessionAnalysis) -> int:
        """세션 분석 저장"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO session_analysis
            (date, ticker, pre_market_return, opening_hour_return, mid_day_return,
             power_hour_return, after_hours_return, overnight_return,
             best_buy_time, best_sell_time, volume_distribution)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis.date.isoformat(),
            analysis.ticker,
            analysis.pre_market_return,
            analysis.opening_hour_return,
            analysis.mid_day_return,
            analysis.power_hour_return,
            analysis.after_hours_return,
            analysis.overnight_return,
            analysis.best_buy_time,
            analysis.best_sell_time,
            json.dumps(analysis.volume_distribution),
        ))

        analysis_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return analysis_id

    def get_session_stats(self, ticker: str = "SPY", days: int = 30) -> Dict:
        """세션별 통계"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                AVG(pre_market_return) as avg_pre_market,
                AVG(opening_hour_return) as avg_opening,
                AVG(mid_day_return) as avg_mid_day,
                AVG(power_hour_return) as avg_power_hour,
                AVG(after_hours_return) as avg_after_hours,
                AVG(overnight_return) as avg_overnight
            FROM session_analysis
            WHERE ticker = ? AND date >= ?
        """, (ticker, (date.today() - timedelta(days=days)).isoformat()))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'pre_market': round(row['avg_pre_market'] or 0, 4),
                'opening_hour': round(row['avg_opening'] or 0, 4),
                'mid_day': round(row['avg_mid_day'] or 0, 4),
                'power_hour': round(row['avg_power_hour'] or 0, 4),
                'after_hours': round(row['avg_after_hours'] or 0, 4),
                'overnight': round(row['avg_overnight'] or 0, 4),
            }
        return {}

    # ========================================================================
    # Backtest Methods (v2.0)
    # ========================================================================

    def save_backtest_run(self, result: Dict) -> int:
        """백테스트 실행 결과 저장

        Args:
            result: BacktestResult.to_dict() 결과

        Returns:
            run_id: 저장된 백테스트 ID
        """
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO backtest_runs
            (strategy_name, start_date, end_date, initial_capital, final_capital,
             total_return, annual_return, benchmark_return, alpha,
             volatility, max_drawdown, max_drawdown_duration,
             sharpe_ratio, sortino_ratio, calmar_ratio,
             total_trades, winning_trades, losing_trades, win_rate,
             avg_win, avg_loss, profit_factor, avg_holding_days,
             total_commission, total_slippage, total_short_cost, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.get('strategy_name'),
            result.get('start_date'),
            result.get('end_date'),
            result.get('initial_capital'),
            result.get('final_capital'),
            result.get('total_return'),
            result.get('annual_return'),
            result.get('benchmark_return'),
            result.get('alpha'),
            result.get('volatility'),
            result.get('max_drawdown'),
            result.get('max_drawdown_duration'),
            result.get('sharpe_ratio'),
            result.get('sortino_ratio'),
            result.get('calmar_ratio'),
            result.get('total_trades'),
            result.get('winning_trades'),
            result.get('losing_trades'),
            result.get('win_rate'),
            result.get('avg_win'),
            result.get('avg_loss'),
            result.get('profit_factor'),
            result.get('avg_holding_days'),
            result.get('total_commission', 0),
            result.get('total_slippage', 0),
            result.get('total_short_cost', 0),
            json.dumps(result.get('parameters', {})),
        ))

        run_id = cursor.lastrowid

        # 개별 거래 저장
        trades = result.get('trades', [])
        for trade in trades:
            cursor.execute("""
                INSERT INTO backtest_trades
                (run_id, entry_date, exit_date, ticker, direction,
                 entry_price, exit_price, shares, pnl, pnl_pct,
                 holding_days, commission, slippage_cost, short_cost, signal_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                trade.get('entry_date'),
                trade.get('exit_date'),
                trade.get('ticker'),
                trade.get('direction'),
                trade.get('entry_price'),
                trade.get('exit_price'),
                trade.get('shares'),
                trade.get('pnl'),
                trade.get('pnl_pct'),
                trade.get('holding_days'),
                trade.get('commission', 0),
                trade.get('slippage_cost', 0),
                trade.get('short_cost', 0),
                trade.get('signal_reason'),
            ))

        conn.commit()
        conn.close()
        return run_id

    def get_backtest_runs(
        self,
        strategy_name: str = None,
        start_date: str = None,
        limit: int = 50
    ) -> List[Dict]:
        """백테스트 실행 이력 조회"""
        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT * FROM backtest_runs WHERE 1=1"
        params = []

        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)

        if start_date:
            query += " AND start_date >= ?"
            params.append(start_date)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            d = dict(row)
            d['parameters'] = json.loads(d['parameters']) if d['parameters'] else {}
            results.append(d)

        return results

    def get_backtest_trades(self, run_id: int) -> List[Dict]:
        """특정 백테스트의 거래 내역 조회"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM backtest_trades
            WHERE run_id = ?
            ORDER BY entry_date
        """, (run_id,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_strategy_performance_history(self, strategy_name: str) -> List[Dict]:
        """전략별 성과 이력 조회"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                start_date, end_date,
                total_return, sharpe_ratio, max_drawdown,
                total_trades, win_rate, profit_factor,
                created_at
            FROM backtest_runs
            WHERE strategy_name = ?
            ORDER BY created_at DESC
        """, (strategy_name,))

        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def save_walk_forward_result(
        self,
        run_id: int,
        fold_number: int,
        train_start: str,
        train_end: str,
        test_start: str,
        test_end: str,
        in_sample_return: float,
        in_sample_sharpe: float,
        out_sample_return: float,
        out_sample_sharpe: float
    ) -> int:
        """Walk-Forward 결과 저장"""
        conn = self._get_conn()
        cursor = conn.cursor()

        # 성과 저하율 계산
        degradation = 0.0
        if in_sample_sharpe != 0:
            degradation = ((in_sample_sharpe - out_sample_sharpe) / abs(in_sample_sharpe)) * 100

        cursor.execute("""
            INSERT INTO walk_forward_results
            (run_id, fold_number, train_start, train_end, test_start, test_end,
             in_sample_return, in_sample_sharpe, out_sample_return, out_sample_sharpe,
             degradation_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, fold_number, train_start, train_end, test_start, test_end,
            in_sample_return, in_sample_sharpe, out_sample_return, out_sample_sharpe,
            degradation
        ))

        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id

    def get_walk_forward_summary(self, run_id: int) -> Dict:
        """Walk-Forward 결과 요약"""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) as total_folds,
                AVG(in_sample_sharpe) as avg_is_sharpe,
                AVG(out_sample_sharpe) as avg_oos_sharpe,
                AVG(degradation_pct) as avg_degradation,
                MIN(out_sample_sharpe) as min_oos_sharpe,
                MAX(out_sample_sharpe) as max_oos_sharpe
            FROM walk_forward_results
            WHERE run_id = ?
        """, (run_id,))

        row = cursor.fetchone()
        conn.close()

        if row and row['total_folds'] > 0:
            return {
                'total_folds': row['total_folds'],
                'avg_in_sample_sharpe': round(row['avg_is_sharpe'] or 0, 2),
                'avg_out_sample_sharpe': round(row['avg_oos_sharpe'] or 0, 2),
                'avg_degradation_pct': round(row['avg_degradation'] or 0, 1),
                'min_oos_sharpe': round(row['min_oos_sharpe'] or 0, 2),
                'max_oos_sharpe': round(row['max_oos_sharpe'] or 0, 2),
            }
        return {}

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_summary(self) -> Dict:
        """DB 요약 정보"""
        conn = self._get_conn()
        cursor = conn.cursor()

        summary = {}

        cursor.execute("SELECT COUNT(*) as cnt FROM signals")
        summary['total_signals'] = cursor.fetchone()['cnt']

        cursor.execute("SELECT COUNT(*) as cnt FROM portfolio_candidates")
        summary['total_portfolios'] = cursor.fetchone()['cnt']

        cursor.execute("SELECT COUNT(*) as cnt FROM executions")
        summary['total_executions'] = cursor.fetchone()['cnt']

        cursor.execute("SELECT COUNT(*) as cnt FROM signal_performance")
        summary['total_signal_evaluations'] = cursor.fetchone()['cnt']

        cursor.execute("""
            SELECT signal_source, COUNT(*) as cnt
            FROM signals GROUP BY signal_source
        """)
        summary['signals_by_source'] = {
            row['signal_source']: row['cnt'] for row in cursor.fetchall()
        }

        # v2.0: 백테스트 통계 추가
        cursor.execute("SELECT COUNT(*) as cnt FROM backtest_runs")
        summary['total_backtest_runs'] = cursor.fetchone()['cnt']

        cursor.execute("SELECT COUNT(*) as cnt FROM backtest_trades")
        summary['total_backtest_trades'] = cursor.fetchone()['cnt']

        cursor.execute("""
            SELECT strategy_name, COUNT(*) as cnt, AVG(sharpe_ratio) as avg_sharpe
            FROM backtest_runs GROUP BY strategy_name
        """)
        summary['backtests_by_strategy'] = {
            row['strategy_name']: {
                'count': row['cnt'],
                'avg_sharpe': round(row['avg_sharpe'] or 0, 2)
            } for row in cursor.fetchall()
        }

        conn.close()
        return summary

    def print_summary(self):
        """요약 출력"""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("EIMAS Trading DB Summary")
        print("=" * 50)
        print(f"Total Signals:      {summary['total_signals']}")
        print(f"Total Portfolios:   {summary['total_portfolios']}")
        print(f"Total Executions:   {summary['total_executions']}")
        print(f"Signal Evaluations: {summary['total_signal_evaluations']}")

        if summary['signals_by_source']:
            print("\nSignals by Source:")
            for source, cnt in summary['signals_by_source'].items():
                print(f"  {source}: {cnt}")

        print("=" * 50)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Trading DB Test")
    print("=" * 60)

    # DB 초기화
    db = TradingDB()
    print(f"✅ DB initialized: {db.db_path}")

    # 테스트 시그널 저장
    test_signal = Signal(
        source=SignalSource.REGIME_DETECTOR,
        action=SignalAction.BUY,
        ticker="SPY",
        conviction=0.75,
        reasoning="Bull + Low Vol regime detected",
        metadata={"regime": "bull_low_vol", "confidence": 0.8}
    )
    signal_id = db.save_signal(test_signal)
    print(f"✅ Signal saved: ID={signal_id}")

    # 테스트 포트폴리오 저장
    test_portfolio = PortfolioCandidate(
        profile=InvestorProfile.MODERATE,
        allocations={"SPY": 0.5, "TLT": 0.3, "GLD": 0.1, "CASH": 0.1},
        expected_return=0.12,
        expected_risk=0.15,
        expected_sharpe=0.80,
        signals_used=[signal_id],
        reasoning="Balanced allocation for moderate risk profile",
        rank=1
    )
    portfolio_id = db.save_portfolio(test_portfolio)
    print(f"✅ Portfolio saved: ID={portfolio_id}")

    # 예측 저장
    db.save_prediction(portfolio_id, pred_1d=0.003, pred_1w=0.015, pred_1m=0.04)
    print(f"✅ Prediction saved")

    # 시그널 조회
    signals = db.get_recent_signals(hours=24)
    print(f"✅ Recent signals: {len(signals)}")

    # 포트폴리오 조회
    portfolios = db.get_portfolios(limit=10)
    print(f"✅ Recent portfolios: {len(portfolios)}")

    # 요약 출력
    db.print_summary()

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
