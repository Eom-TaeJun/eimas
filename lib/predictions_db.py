"""
EIMAS Predictions Database
예측 데이터 저장 및 검증을 위한 모듈

DB.md 스키마 기반 구현
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import os

DB_PATH = Path(__file__).parent.parent / "data" / "predictions.db"


def get_connection() -> sqlite3.Connection:
    """DB 연결 반환"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """데이터베이스 초기화 - 테이블 생성"""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. regime_predictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS regime_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        predicted_regime TEXT,
        confidence REAL,
        gmm_bull_prob REAL,
        gmm_neutral_prob REAL,
        gmm_bear_prob REAL,
        shannon_entropy REAL,
        trend TEXT,
        volatility TEXT,
        spy_price_at_prediction REAL,
        validated INTEGER DEFAULT 0,
        actual_return_1d REAL,
        actual_return_5d REAL,
        actual_return_20d REAL,
        validated_at DATETIME
    )
    """)

    # 2. risk_predictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS risk_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        final_risk_score REAL,
        base_score REAL,
        microstructure_adj REAL,
        bubble_adj REAL,
        risk_level TEXT,
        avg_liquidity_score REAL,
        bubble_status TEXT,
        vix_at_prediction REAL,
        validated INTEGER DEFAULT 0,
        actual_max_drawdown_5d REAL,
        actual_max_drawdown_20d REAL,
        validated_at DATETIME
    )
    """)

    # 3. debate_predictions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS debate_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        full_mode_position TEXT,
        ref_mode_position TEXT,
        modes_agree INTEGER,
        final_recommendation TEXT,
        confidence REAL,
        dissent_count INTEGER,
        devils_advocate_1 TEXT,
        devils_advocate_2 TEXT,
        devils_advocate_3 TEXT,
        spy_price_at_prediction REAL,
        validated INTEGER DEFAULT 0,
        actual_direction_1d TEXT,
        actual_direction_5d TEXT,
        actual_return_5d REAL,
        validated_at DATETIME
    )
    """)

    # 4. vpin_snapshots
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vpin_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        symbol TEXT NOT NULL,
        vpin_1m REAL,
        vpin_5m REAL,
        vpin_15m REAL,
        alert_level TEXT,
        buy_volume REAL,
        sell_volume REAL,
        imbalance_ratio REAL,
        price_at_capture REAL,
        price_1h_later REAL,
        price_change_1h REAL,
        validated INTEGER DEFAULT 0
    )
    """)

    # 5. stablecoin_snapshots
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stablecoin_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        usdt_price REAL,
        usdc_price REAL,
        dai_price REAL,
        usdt_deviation REAL,
        usdc_deviation REAL,
        dai_deviation REAL,
        max_deviation REAL,
        depeg_alert INTEGER DEFAULT 0,
        stress_test_depeg_prob REAL,
        stress_test_expected_loss REAL,
        actual_depeg_24h INTEGER,
        validated INTEGER DEFAULT 0
    )
    """)

    # 6. bubble_alerts
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bubble_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        ticker TEXT NOT NULL,
        alert_level TEXT,
        runup_2y_pct REAL,
        volatility_zscore REAL,
        issuance_growth REAL,
        bubble_score REAL,
        price_at_alert REAL,
        validated INTEGER DEFAULT 0,
        max_drawdown_30d REAL,
        max_drawdown_90d REAL,
        crash_occurred INTEGER,
        validated_at DATETIME
    )
    """)

    # 7. portfolio_snapshots
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        portfolio_json TEXT,
        top5_tickers TEXT,
        top5_weights TEXT,
        cluster_count INTEGER,
        allocation_rationale TEXT,
        spy_weight REAL,
        tlt_weight REAL,
        validated INTEGER DEFAULT 0,
        portfolio_return_5d REAL,
        portfolio_return_20d REAL,
        spy_return_5d REAL,
        spy_return_20d REAL,
        outperformed_5d INTEGER,
        outperformed_20d INTEGER,
        validated_at DATETIME
    )
    """)

    # 8. validation_log
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS validation_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        validated_at DATETIME NOT NULL,
        prediction_type TEXT NOT NULL,
        prediction_id INTEGER NOT NULL,
        prediction_timestamp DATETIME,
        predicted_value TEXT,
        actual_value TEXT,
        is_correct INTEGER,
        accuracy_score REAL,
        horizon_days INTEGER,
        notes TEXT
    )
    """)

    # 인덱스 생성
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_ts ON regime_predictions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_ts ON risk_predictions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_debate_ts ON debate_predictions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vpin_ts ON vpin_snapshots(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stablecoin_ts ON stablecoin_snapshots(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_bubble_ts ON bubble_alerts(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_ts ON portfolio_snapshots(timestamp)")

    conn.commit()
    conn.close()
    print(f"[predictions_db] Initialized: {DB_PATH}")


class PredictionsDB:
    """예측 데이터 저장 클래스"""

    def __init__(self):
        init_db()
        self.conn = get_connection()

    def close(self):
        self.conn.close()

    def save_regime_prediction(
        self,
        timestamp: str,
        regime: Dict,
        gmm_analysis: Optional[Dict] = None,
        spy_price: Optional[float] = None
    ) -> int:
        """레짐 예측 저장"""
        cursor = self.conn.cursor()

        gmm = gmm_analysis or {}
        probs = gmm.get('probabilities', {})

        cursor.execute("""
        INSERT INTO regime_predictions (
            timestamp, predicted_regime, confidence,
            gmm_bull_prob, gmm_neutral_prob, gmm_bear_prob,
            shannon_entropy, trend, volatility, spy_price_at_prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            regime.get('regime', 'Unknown'),
            regime.get('confidence', 0) / 100 if regime.get('confidence', 0) > 1 else regime.get('confidence', 0),
            probs.get('bull', 0),
            probs.get('neutral', 0),
            probs.get('bear', 0),
            gmm.get('entropy', 0),
            regime.get('trend', 'Unknown'),
            regime.get('volatility', 'Unknown'),
            spy_price
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_risk_prediction(
        self,
        timestamp: str,
        final_risk: float,
        base_score: float,
        micro_adj: float,
        bubble_adj: float,
        risk_level: str,
        liquidity_score: float,
        bubble_status: str,
        vix: Optional[float] = None
    ) -> int:
        """리스크 예측 저장"""
        cursor = self.conn.cursor()

        cursor.execute("""
        INSERT INTO risk_predictions (
            timestamp, final_risk_score, base_score,
            microstructure_adj, bubble_adj, risk_level,
            avg_liquidity_score, bubble_status, vix_at_prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, final_risk, base_score,
            micro_adj, bubble_adj, risk_level,
            liquidity_score, bubble_status, vix
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_debate_prediction(
        self,
        timestamp: str,
        full_position: str,
        ref_position: str,
        modes_agree: bool,
        final_rec: str,
        confidence: float,
        devils_advocate: List[str],
        spy_price: Optional[float] = None
    ) -> int:
        """토론 결과 저장"""
        cursor = self.conn.cursor()

        da = devils_advocate or []

        cursor.execute("""
        INSERT INTO debate_predictions (
            timestamp, full_mode_position, ref_mode_position,
            modes_agree, final_recommendation, confidence,
            dissent_count, devils_advocate_1, devils_advocate_2, devils_advocate_3,
            spy_price_at_prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, full_position, ref_position,
            1 if modes_agree else 0, final_rec, confidence,
            len(da),
            da[0] if len(da) > 0 else None,
            da[1] if len(da) > 1 else None,
            da[2] if len(da) > 2 else None,
            spy_price
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_vpin_snapshot(
        self,
        timestamp: str,
        symbol: str,
        vpin_1m: float,
        alert_level: str,
        price: float,
        vpin_5m: Optional[float] = None,
        buy_volume: Optional[float] = None,
        sell_volume: Optional[float] = None
    ) -> int:
        """VPIN 스냅샷 저장"""
        cursor = self.conn.cursor()

        imbalance = None
        if buy_volume and sell_volume and (buy_volume + sell_volume) > 0:
            imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)

        cursor.execute("""
        INSERT INTO vpin_snapshots (
            timestamp, symbol, vpin_1m, vpin_5m, alert_level,
            buy_volume, sell_volume, imbalance_ratio, price_at_capture
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, symbol, vpin_1m, vpin_5m, alert_level,
            buy_volume, sell_volume, imbalance, price
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_stablecoin_snapshot(
        self,
        timestamp: str,
        prices: Dict[str, float],
        stress_test: Optional[Dict] = None
    ) -> int:
        """스테이블코인 스냅샷 저장"""
        cursor = self.conn.cursor()

        usdt = prices.get('USDT', 1.0)
        usdc = prices.get('USDC', 1.0)
        dai = prices.get('DAI', 1.0)

        usdt_dev = abs(usdt - 1.0) * 100
        usdc_dev = abs(usdc - 1.0) * 100
        dai_dev = abs(dai - 1.0) * 100
        max_dev = max(usdt_dev, usdc_dev, dai_dev)

        st = stress_test or {}

        cursor.execute("""
        INSERT INTO stablecoin_snapshots (
            timestamp, usdt_price, usdc_price, dai_price,
            usdt_deviation, usdc_deviation, dai_deviation, max_deviation,
            depeg_alert, stress_test_depeg_prob, stress_test_expected_loss
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, usdt, usdc, dai,
            usdt_dev, usdc_dev, dai_dev, max_dev,
            1 if max_dev > 0.5 else 0,
            st.get('depeg_probability', 0),
            st.get('expected_loss', 0)
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_bubble_alert(
        self,
        timestamp: str,
        ticker: str,
        alert_level: str,
        runup_pct: float,
        vol_zscore: float,
        bubble_score: float,
        price: float,
        issuance_growth: Optional[float] = None
    ) -> int:
        """버블 경고 저장"""
        cursor = self.conn.cursor()

        cursor.execute("""
        INSERT INTO bubble_alerts (
            timestamp, ticker, alert_level, runup_2y_pct,
            volatility_zscore, issuance_growth, bubble_score, price_at_alert
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, ticker, alert_level, runup_pct,
            vol_zscore, issuance_growth, bubble_score, price
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_portfolio_snapshot(
        self,
        timestamp: str,
        weights: Dict[str, float],
        rationale: str,
        cluster_count: Optional[int] = None
    ) -> int:
        """포트폴리오 스냅샷 저장"""
        cursor = self.conn.cursor()

        # Top 5 추출
        sorted_weights = sorted(weights.items(), key=lambda x: -x[1])[:5]
        top5_tickers = ','.join([t for t, _ in sorted_weights])
        top5_weights = ','.join([f"{w:.1%}" for _, w in sorted_weights])

        cursor.execute("""
        INSERT INTO portfolio_snapshots (
            timestamp, portfolio_json, top5_tickers, top5_weights,
            cluster_count, allocation_rationale, spy_weight, tlt_weight
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            json.dumps(weights),
            top5_tickers,
            top5_weights,
            cluster_count,
            rationale,
            weights.get('SPY', 0),
            weights.get('TLT', 0)
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_unvalidated(self, table: str, days_ago: int = 5) -> List[Dict]:
        """검증되지 않은 예측 조회"""
        cursor = self.conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days_ago)).isoformat()

        cursor.execute(f"""
        SELECT * FROM {table}
        WHERE validated = 0 AND timestamp < ?
        ORDER BY timestamp ASC
        """, (cutoff,))

        return [dict(row) for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, int]:
        """테이블별 레코드 수 조회"""
        cursor = self.conn.cursor()
        stats = {}

        tables = [
            'regime_predictions', 'risk_predictions', 'debate_predictions',
            'vpin_snapshots', 'stablecoin_snapshots', 'bubble_alerts',
            'portfolio_snapshots', 'validation_log'
        ]

        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]

        return stats


def save_eimas_result(result) -> Dict[str, int]:
    """
    EIMASResult 객체에서 예측 데이터 추출하여 저장

    Args:
        result: EIMASResult 객체

    Returns:
        Dict[str, int]: 각 테이블에 저장된 레코드 ID
    """
    db = PredictionsDB()
    saved_ids = {}

    timestamp = result.timestamp

    try:
        # 1. 레짐 예측 저장
        if result.regime:
            gmm = result.gmm_analysis if hasattr(result, 'gmm_analysis') else None
            saved_ids['regime'] = db.save_regime_prediction(
                timestamp=timestamp,
                regime=result.regime,
                gmm_analysis=gmm,
                spy_price=None  # TODO: 실시간 가격 추가
            )

        # 2. 리스크 예측 저장
        saved_ids['risk'] = db.save_risk_prediction(
            timestamp=timestamp,
            final_risk=result.risk_score,
            base_score=getattr(result, 'base_risk_score', result.risk_score),
            micro_adj=getattr(result, 'microstructure_adjustment', 0),
            bubble_adj=getattr(result, 'bubble_risk_adjustment', 0),
            risk_level=result.risk_level,
            liquidity_score=getattr(result, 'market_quality', {}).get('avg_liquidity_score', 0) if isinstance(getattr(result, 'market_quality', None), dict) else (result.market_quality.avg_liquidity_score if hasattr(result, 'market_quality') and result.market_quality else 0),
            bubble_status=getattr(result, 'bubble_risk', {}).get('overall_status', 'NONE') if isinstance(getattr(result, 'bubble_risk', None), dict) else (result.bubble_risk.overall_status if hasattr(result, 'bubble_risk') and result.bubble_risk else 'NONE'),
            vix=None  # TODO: VIX 추가
        )

        # 3. 토론 결과 저장
        saved_ids['debate'] = db.save_debate_prediction(
            timestamp=timestamp,
            full_position=result.full_mode_position,
            ref_position=result.reference_mode_position,
            modes_agree=result.modes_agree,
            final_rec=result.final_recommendation,
            confidence=result.confidence,
            devils_advocate=getattr(result, 'devils_advocate_arguments', []),
            spy_price=None
        )

        # 4. 포트폴리오 저장
        if hasattr(result, 'portfolio_weights') and result.portfolio_weights:
            saved_ids['portfolio'] = db.save_portfolio_snapshot(
                timestamp=timestamp,
                weights=result.portfolio_weights,
                rationale=getattr(result, 'hrp_allocation_rationale', ''),
                cluster_count=None
            )

        # 5. 버블 경고 저장
        if hasattr(result, 'bubble_risk'):
            bubble = result.bubble_risk
            if isinstance(bubble, dict):
                risk_tickers = bubble.get('risk_tickers', [])
            elif hasattr(bubble, 'risk_tickers'):
                risk_tickers = bubble.risk_tickers
            else:
                risk_tickers = []

            for ticker_info in risk_tickers:
                if isinstance(ticker_info, dict) and ticker_info.get('level') in ['WATCH', 'WARNING', 'DANGER']:
                    saved_ids[f'bubble_{ticker_info.get("ticker")}'] = db.save_bubble_alert(
                        timestamp=timestamp,
                        ticker=ticker_info.get('ticker', ''),
                        alert_level=ticker_info.get('level', ''),
                        runup_pct=ticker_info.get('runup_pct', 0),
                        vol_zscore=ticker_info.get('vol_zscore', 0),
                        bubble_score=ticker_info.get('risk_score', 0),
                        price=0  # TODO: 가격 추가
                    )

        # 6. 스테이블코인 저장 (Crypto Stress Test 결과가 있는 경우)
        if hasattr(result, 'crypto_stress_test') and result.crypto_stress_test:
            st = result.crypto_stress_test
            saved_ids['stablecoin'] = db.save_stablecoin_snapshot(
                timestamp=timestamp,
                prices={'USDT': 1.0, 'USDC': 1.0, 'DAI': 1.0},  # TODO: 실제 가격
                stress_test={
                    'depeg_probability': st.get('depeg_probability', 0),
                    'expected_loss': st.get('expected_loss', 0)
                }
            )

        # 7. VPIN 스냅샷 저장 (realtime_signals가 있는 경우)
        if hasattr(result, 'realtime_signals') and result.realtime_signals:
            vpin_count = 0
            for sig in result.realtime_signals:
                symbol = sig.get('symbol', '')
                avg_vpin = sig.get('avg_vpin', 0)
                max_vpin = sig.get('max_vpin', 0)

                if symbol and (avg_vpin > 0 or max_vpin > 0):
                    # alert_level 결정
                    if max_vpin >= 0.7:
                        alert_level = 'EXTREME'
                    elif max_vpin >= 0.6:
                        alert_level = 'HIGH'
                    elif max_vpin >= 0.5:
                        alert_level = 'ELEVATED'
                    else:
                        alert_level = 'NORMAL'

                    db.save_vpin_snapshot(
                        timestamp=sig.get('timestamp', timestamp),
                        symbol=symbol,
                        vpin_1m=avg_vpin,
                        alert_level=alert_level,
                        price=sig.get('price', 0),
                        vpin_5m=max_vpin,
                        buy_volume=sig.get('buy_volume'),
                        sell_volume=sig.get('sell_volume')
                    )
                    vpin_count += 1

            if vpin_count > 0:
                saved_ids['vpin_snapshots'] = vpin_count

    finally:
        db.close()

    return saved_ids


if __name__ == "__main__":
    # 테스트
    init_db()

    db = PredictionsDB()
    stats = db.get_stats()

    print("=" * 50)
    print("Predictions DB 상태")
    print("=" * 50)
    for table, count in stats.items():
        print(f"  {table}: {count}개")
    print("=" * 50)

    db.close()
