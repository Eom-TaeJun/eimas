#!/usr/bin/env python3
"""
Critical Path Monitor
=====================
17개 시장 경로 실시간 모니터링

경로 분류:
1-10: Normal Market Paths (일반 시장 신호)
11-17: Crisis Paths (위기 신호)

각 경로는 선행 지표로서 향후 시장 방향을 예측하는 데 사용됩니다.

참고: Market Intelligence v3.0 DESIGN_WORKFLOW.md
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

from core.database import DatabaseManager


# ============================================================================
# Constants & Enums
# ============================================================================

class PathType(str, Enum):
    """경로 유형"""
    NORMAL = "normal"        # 일반 시장 신호
    CRISIS = "crisis"        # 위기 신호


class SignalLevel(str, Enum):
    """신호 레벨"""
    NORMAL = "normal"        # 정상
    WATCH = "watch"          # 관찰
    WARNING = "warning"      # 경고
    CRITICAL = "critical"    # 위험


class PathStatus(str, Enum):
    """경로 상태"""
    INACTIVE = "inactive"    # 비활성
    ACTIVE = "active"        # 활성 (신호 감지)
    TRIGGERED = "triggered"  # 트리거됨 (임계값 돌파)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PathDefinition:
    """경로 정의"""
    id: int
    name: str
    path_type: PathType
    description: str
    lead_indicator: str          # 선행 지표
    target: str                  # 예측 대상
    lead_time: str               # 리드 타임
    thresholds: Dict[str, float] # 임계값


@dataclass
class PathSignal:
    """경로 신호"""
    path_id: int
    path_name: str
    status: PathStatus
    level: SignalLevel
    value: float                 # 현재 값
    threshold: float             # 임계값
    deviation: float             # 이탈 정도 (%)
    message: str
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            'path_id': self.path_id,
            'path_name': self.path_name,
            'status': self.status.value,
            'level': self.level.value,
            'value': round(self.value, 4),
            'threshold': round(self.threshold, 4),
            'deviation': round(self.deviation, 2),
            'message': self.message,
            'timestamp': self.timestamp,
        }


@dataclass
class CriticalPathSummary:
    """경로 모니터링 요약"""
    timestamp: str
    total_paths: int = 17
    active_paths: int = 0
    triggered_paths: int = 0
    critical_count: int = 0
    warning_count: int = 0
    watch_count: int = 0

    # 경로별 신호
    signals: List[PathSignal] = field(default_factory=list)

    # 종합 판단
    market_regime: str = "NORMAL"    # NORMAL, CAUTION, RISK_OFF, CRISIS
    risk_score: float = 0.0          # 0-100

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'total_paths': self.total_paths,
            'active_paths': self.active_paths,
            'triggered_paths': self.triggered_paths,
            'critical_count': self.critical_count,
            'warning_count': self.warning_count,
            'watch_count': self.watch_count,
            'signals': [s.to_dict() for s in self.signals],
            'market_regime': self.market_regime,
            'risk_score': self.risk_score,
        }


# ============================================================================
# Path Definitions (17 Critical Paths)
# ============================================================================

CRITICAL_PATHS: List[PathDefinition] = [
    # === Normal Market Paths (1-10) ===

    PathDefinition(
        id=1, name="Yield Curve", path_type=PathType.NORMAL,
        description="10Y-2Y Treasury Spread - 경기침체 선행지표",
        lead_indicator="T10Y2Y", target="Recession", lead_time="12-18 months",
        thresholds={'warning': -0.25, 'critical': -0.5}
    ),

    PathDefinition(
        id=2, name="Copper/Gold Ratio", path_type=PathType.NORMAL,
        description="Cu/Au Ratio - 글로벌 산업 활동 지표",
        lead_indicator="Cu/Au", target="Industrial Activity", lead_time="1-3 months",
        thresholds={'warning': 0.20, 'critical': 0.15}  # 비율 하락 시 경고
    ),

    PathDefinition(
        id=3, name="HY Spreads", path_type=PathType.NORMAL,
        description="High Yield OAS - 크레딧 사이클 지표",
        lead_indicator="HY_OAS", target="Credit Cycle", lead_time="2-3 quarters",
        thresholds={'warning': 400, 'critical': 550}  # bp
    ),

    PathDefinition(
        id=4, name="Dollar Smile", path_type=PathType.NORMAL,
        description="DXY + VIX 조합 - 글로벌 리스크",
        lead_indicator="DXY+VIX", target="Global Risk", lead_time="simultaneous",
        thresholds={'warning': 115, 'critical': 120}  # DXY + VIX/4
    ),

    PathDefinition(
        id=5, name="Sector Rotation", path_type=PathType.NORMAL,
        description="XLY/XLP Ratio - 경기 사이클 위치",
        lead_indicator="XLY/XLP", target="Business Cycle", lead_time="1-3 months",
        thresholds={'warning': 0.95, 'critical': 0.85}  # 비율 하락 시 경고
    ),

    PathDefinition(
        id=6, name="Breakevens", path_type=PathType.NORMAL,
        description="5Y Breakeven Inflation - Fed 정책 방향",
        lead_indicator="T5YIE", target="Fed Policy", lead_time="1-6 months",
        thresholds={'warning': 2.8, 'critical': 3.2}  # %
    ),

    PathDefinition(
        id=7, name="VIX Structure", path_type=PathType.NORMAL,
        description="VIX/VIX3M Ratio - 변동성 체제",
        lead_indicator="VIX/VIX3M", target="Volatility Regime", lead_time="days-weeks",
        thresholds={'warning': 1.05, 'critical': 1.15}  # Backwardation
    ),

    PathDefinition(
        id=8, name="EM Flows", path_type=PathType.NORMAL,
        description="EEM Performance - 글로벌 유동성",
        lead_indicator="EEM_momentum", target="Global Liquidity", lead_time="2-4 weeks",
        thresholds={'warning': -5, 'critical': -10}  # 20일 수익률 %
    ),

    PathDefinition(
        id=9, name="Gold/Silver Ratio", path_type=PathType.NORMAL,
        description="Au/Ag Ratio - 인플레이션 유형",
        lead_indicator="Au/Ag", target="Inflation Type", lead_time="simultaneous",
        thresholds={'warning': 85, 'critical': 95}  # 비율 상승 시 디플레 우려
    ),

    PathDefinition(
        id=10, name="Bank Stocks", path_type=PathType.NORMAL,
        description="XLF vs SPY - 신용 가용성",
        lead_indicator="XLF/SPY", target="Credit Availability", lead_time="2 quarters",
        thresholds={'warning': -8, 'critical': -15}  # 상대 성과 %
    ),

    # === Crisis Paths (11-17) ===

    PathDefinition(
        id=11, name="Crack Sequence", path_type=PathType.CRISIS,
        description="ARKK → IWM → HYG 순차 붕괴",
        lead_indicator="Speculative_cascade", target="Market Crash", lead_time="weeks-months",
        thresholds={'warning': 2, 'critical': 3}  # 동시 하락 자산 수
    ),

    PathDefinition(
        id=12, name="Liquidity Cascade", path_type=PathType.CRISIS,
        description="펀딩 스프레드, ETF 할인",
        lead_indicator="Liquidity_stress", target="Funding Crisis", lead_time="days-weeks",
        thresholds={'warning': 1.5, 'critical': 2.5}  # 스트레스 점수
    ),

    PathDefinition(
        id=13, name="Melt-Up Detection", path_type=PathType.CRISIS,
        description="RSI 과매수, 변동성 압축",
        lead_indicator="Melt_up_score", target="Bubble", lead_time="days-weeks",
        thresholds={'warning': 70, 'critical': 80}  # RSI or composite
    ),

    PathDefinition(
        id=14, name="Correlation Breakdown", path_type=PathType.CRISIS,
        description="주식-채권 상관관계 전환",
        lead_indicator="Stock_Bond_Corr", target="Regime Change", lead_time="days",
        thresholds={'warning': 0.3, 'critical': 0.5}  # 양의 상관
    ),

    PathDefinition(
        id=15, name="Capitulation", path_type=PathType.CRISIS,
        description="VIX 급등, 거래량 폭증",
        lead_indicator="Capitulation_score", target="Panic Selling", lead_time="hours-days",
        thresholds={'warning': 35, 'critical': 45}  # VIX level
    ),

    PathDefinition(
        id=16, name="Contagion Mapping", path_type=PathType.CRISIS,
        description="섹터 간 전파 속도",
        lead_indicator="Contagion_speed", target="Systemic Risk", lead_time="days-weeks",
        thresholds={'warning': 0.7, 'critical': 0.85}  # 상관 급증
    ),

    PathDefinition(
        id=17, name="Divergence Warnings", path_type=PathType.CRISIS,
        description="시장 간 불일치",
        lead_indicator="Divergence_count", target="Hidden Risk", lead_time="weeks",
        thresholds={'warning': 3, 'critical': 5}  # 불일치 개수
    ),
]


# ============================================================================
# Critical Path Monitor
# ============================================================================

class CriticalPathMonitor:
    """
    17개 Critical Path 모니터링

    사용법:
        monitor = CriticalPathMonitor()
        summary = monitor.analyze()
        monitor.print_report(summary)
        monitor.save_to_db(summary)
    """

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self._cache: Dict[str, pd.DataFrame] = {}

    def _fetch_data(self, tickers: List[str], period: str = "3mo") -> pd.DataFrame:
        """가격 데이터 수집"""
        cache_key = "_".join(sorted(tickers))
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            df = yf.download(tickers, period=period, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            self._cache[cache_key] = df
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def _check_path_1_yield_curve(self) -> PathSignal:
        """Path 1: Yield Curve (10Y-2Y Spread)"""
        path = CRITICAL_PATHS[0]

        try:
            # FRED에서 직접 가져오거나 ETF 기반 추정
            # 여기서는 IEF/SHY 비율로 추정
            df = self._fetch_data(['IEF', 'SHY'])
            if df.empty:
                return self._create_inactive_signal(path)

            # 간단히 가격 비율 변화로 스프레드 추정
            ief = df['IEF'].iloc[-1]
            shy = df['SHY'].iloc[-1]
            ief_20d = df['IEF'].iloc[-20] if len(df) > 20 else ief
            shy_20d = df['SHY'].iloc[-20] if len(df) > 20 else shy

            # 상대 변화 (10Y 가격 하락 = 금리 상승)
            spread_proxy = (ief / ief_20d - 1) - (shy / shy_20d - 1)
            spread_estimate = spread_proxy * 100  # 대략적 bp 변환

            # 실제 스프레드는 FRED에서 가져와야 정확
            # 임시로 추정값 사용
            value = spread_estimate - 0.2  # 현재 약 -0.2 ~ 0.3 range

            return self._evaluate_path(path, value, lower_is_worse=True)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_2_copper_gold(self) -> PathSignal:
        """Path 2: Copper/Gold Ratio"""
        path = CRITICAL_PATHS[1]

        try:
            # COPX (구리 ETF) / GLD
            df = self._fetch_data(['COPX', 'GLD'])
            if df.empty or 'COPX' not in df.columns:
                # 대안: 직접 구리 선물 사용
                df = self._fetch_data(['GLD'])
                if df.empty:
                    return self._create_inactive_signal(path)
                value = 0.22  # 기본값
            else:
                value = df['COPX'].iloc[-1] / df['GLD'].iloc[-1]

            return self._evaluate_path(path, value, lower_is_worse=True)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_3_hy_spreads(self) -> PathSignal:
        """Path 3: HY Spreads (HYG/LQD 기반)"""
        path = CRITICAL_PATHS[2]

        try:
            df = self._fetch_data(['HYG', 'LQD', 'TLT'])
            if df.empty:
                return self._create_inactive_signal(path)

            # HY-IG 스프레드 추정 (가격 비율 기반)
            hyg = df['HYG'].iloc[-1]
            lqd = df['LQD'].iloc[-1]

            # 낮은 비율 = 높은 스프레드
            ratio = hyg / lqd
            # 대략적 스프레드 추정 (bp)
            spread_estimate = (0.77 - ratio) * 1000  # 대략적 변환

            return self._evaluate_path(path, max(0, spread_estimate), lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_4_dollar_smile(self) -> PathSignal:
        """Path 4: Dollar Smile (DXY + VIX)"""
        path = CRITICAL_PATHS[3]

        try:
            df = self._fetch_data(['^VIX', 'UUP'])
            if df.empty:
                return self._create_inactive_signal(path)

            vix = df['^VIX'].iloc[-1] if '^VIX' in df.columns else 20
            # UUP를 DXY 프록시로 사용
            uup = df['UUP'].iloc[-1] if 'UUP' in df.columns else 28

            # DXY 추정 (UUP 기준)
            dxy_estimate = uup * 3.7  # 대략적 변환

            # Dollar Smile 점수
            value = dxy_estimate + vix / 4

            return self._evaluate_path(path, value, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_5_sector_rotation(self) -> PathSignal:
        """Path 5: Sector Rotation (XLY/XLP)"""
        path = CRITICAL_PATHS[4]

        try:
            df = self._fetch_data(['XLY', 'XLP'])
            if df.empty:
                return self._create_inactive_signal(path)

            ratio = df['XLY'].iloc[-1] / df['XLP'].iloc[-1]

            return self._evaluate_path(path, ratio, lower_is_worse=True)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_6_breakevens(self) -> PathSignal:
        """Path 6: Breakevens (TIP/IEF 기반)"""
        path = CRITICAL_PATHS[5]

        try:
            df = self._fetch_data(['TIP', 'IEF'])
            if df.empty:
                return self._create_inactive_signal(path)

            # TIP/IEF 비율로 인플레 기대 추정
            ratio = df['TIP'].iloc[-1] / df['IEF'].iloc[-1]
            # 대략적 breakeven 추정
            breakeven_estimate = (ratio - 0.95) * 20 + 2.2

            return self._evaluate_path(path, breakeven_estimate, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_7_vix_structure(self) -> PathSignal:
        """Path 7: VIX Term Structure"""
        path = CRITICAL_PATHS[6]

        try:
            df = self._fetch_data(['^VIX', '^VIX3M'])
            if df.empty or '^VIX' not in df.columns:
                return self._create_inactive_signal(path)

            vix = df['^VIX'].iloc[-1]
            vix3m = df['^VIX3M'].iloc[-1] if '^VIX3M' in df.columns else vix * 1.1

            ratio = vix / vix3m

            return self._evaluate_path(path, ratio, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_8_em_flows(self) -> PathSignal:
        """Path 8: EM Flows (EEM momentum)"""
        path = CRITICAL_PATHS[7]

        try:
            df = self._fetch_data(['EEM'])
            if df.empty or len(df) < 20:
                return self._create_inactive_signal(path)

            # 20일 수익률
            returns_20d = (df['EEM'].iloc[-1] / df['EEM'].iloc[-20] - 1) * 100

            return self._evaluate_path(path, returns_20d, lower_is_worse=True)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_9_gold_silver(self) -> PathSignal:
        """Path 9: Gold/Silver Ratio"""
        path = CRITICAL_PATHS[8]

        try:
            df = self._fetch_data(['GLD', 'SLV'])
            if df.empty:
                return self._create_inactive_signal(path)

            ratio = df['GLD'].iloc[-1] / df['SLV'].iloc[-1]
            # 실제 Au/Ag 비율로 변환 (대략)
            au_ag_ratio = ratio * 4.5  # GLD/SLV → Au/Ag

            return self._evaluate_path(path, au_ag_ratio, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_10_bank_stocks(self) -> PathSignal:
        """Path 10: Bank Stocks (XLF/SPY)"""
        path = CRITICAL_PATHS[9]

        try:
            df = self._fetch_data(['XLF', 'SPY'])
            if df.empty or len(df) < 60:
                return self._create_inactive_signal(path)

            # 60일 상대 성과
            xlf_ret = (df['XLF'].iloc[-1] / df['XLF'].iloc[-60] - 1) * 100
            spy_ret = (df['SPY'].iloc[-1] / df['SPY'].iloc[-60] - 1) * 100
            relative_perf = xlf_ret - spy_ret

            return self._evaluate_path(path, relative_perf, lower_is_worse=True)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_11_crack_sequence(self) -> PathSignal:
        """Path 11: Crack Sequence (ARKK → IWM → HYG)"""
        path = CRITICAL_PATHS[10]

        try:
            df = self._fetch_data(['ARKK', 'IWM', 'HYG'])
            if df.empty or len(df) < 20:
                return self._create_inactive_signal(path)

            # 20일 수익률 확인
            crash_count = 0
            for ticker in ['ARKK', 'IWM', 'HYG']:
                if ticker in df.columns:
                    ret = (df[ticker].iloc[-1] / df[ticker].iloc[-20] - 1) * 100
                    if ret < -10:  # 10% 이상 하락
                        crash_count += 1

            return self._evaluate_path(path, crash_count, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_12_liquidity(self) -> PathSignal:
        """Path 12: Liquidity Cascade"""
        path = CRITICAL_PATHS[11]

        try:
            # 유동성 스트레스 프록시: 거래량 급증, 가격 급락
            df = self._fetch_data(['SPY', 'TLT', 'HYG'])
            if df.empty:
                return self._create_inactive_signal(path)

            # 간단한 스트레스 점수
            stress_score = 0.0

            for ticker in ['SPY', 'TLT', 'HYG']:
                if ticker in df.columns and len(df) >= 5:
                    ret_5d = (df[ticker].iloc[-1] / df[ticker].iloc[-5] - 1) * 100
                    if ret_5d < -3:
                        stress_score += abs(ret_5d) / 10

            return self._evaluate_path(path, stress_score, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_13_meltup(self) -> PathSignal:
        """Path 13: Melt-Up Detection"""
        path = CRITICAL_PATHS[12]

        try:
            df = self._fetch_data(['SPY', '^VIX'])
            if df.empty or len(df) < 14:
                return self._create_inactive_signal(path)

            # RSI 계산
            spy = df['SPY']
            delta = spy.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])

            return self._evaluate_path(path, current_rsi, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_14_correlation(self) -> PathSignal:
        """Path 14: Correlation Breakdown (Stock-Bond)"""
        path = CRITICAL_PATHS[13]

        try:
            df = self._fetch_data(['SPY', 'TLT'])
            if df.empty or len(df) < 20:
                return self._create_inactive_signal(path)

            # 20일 상관계수
            spy_ret = df['SPY'].pct_change()
            tlt_ret = df['TLT'].pct_change()
            corr = spy_ret.tail(20).corr(tlt_ret.tail(20))

            return self._evaluate_path(path, corr, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_15_capitulation(self) -> PathSignal:
        """Path 15: Capitulation"""
        path = CRITICAL_PATHS[14]

        try:
            df = self._fetch_data(['^VIX'])
            if df.empty:
                return self._create_inactive_signal(path)

            vix = float(df['^VIX'].iloc[-1])

            return self._evaluate_path(path, vix, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_16_contagion(self) -> PathSignal:
        """Path 16: Contagion Mapping"""
        path = CRITICAL_PATHS[15]

        try:
            # 섹터 간 상관 급증 확인
            sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY']
            df = self._fetch_data(sectors)
            if df.empty or len(df) < 20:
                return self._create_inactive_signal(path)

            # 평균 상관계수
            returns = df.pct_change().tail(20)
            corr_matrix = returns.corr()
            avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, 1)].mean()

            return self._evaluate_path(path, avg_corr, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _check_path_17_divergence(self) -> PathSignal:
        """Path 17: Divergence Warnings"""
        path = CRITICAL_PATHS[16]

        try:
            # 시장 간 불일치 개수
            pairs = [
                ('SPY', 'IWM'),    # 대형 vs 소형
                ('SPY', 'EEM'),    # 미국 vs EM
                ('TLT', 'HYG'),    # 국채 vs HY
                ('GLD', 'SPY'),    # 금 vs 주식
                ('XLY', 'XLP'),    # 경기민감 vs 방어
            ]

            divergence_count = 0
            for t1, t2 in pairs:
                df = self._fetch_data([t1, t2])
                if not df.empty and len(df) >= 20:
                    r1 = (df[t1].iloc[-1] / df[t1].iloc[-20] - 1) * 100
                    r2 = (df[t2].iloc[-1] / df[t2].iloc[-20] - 1) * 100
                    # 방향 불일치
                    if (r1 > 2 and r2 < -2) or (r1 < -2 and r2 > 2):
                        divergence_count += 1

            return self._evaluate_path(path, divergence_count, lower_is_worse=False)

        except Exception as e:
            return self._create_error_signal(path, str(e))

    def _evaluate_path(self, path: PathDefinition, value: float,
                       lower_is_worse: bool = False) -> PathSignal:
        """경로 평가"""
        timestamp = datetime.now().isoformat()
        warn_th = path.thresholds['warning']
        crit_th = path.thresholds['critical']

        if lower_is_worse:
            # 낮을수록 나쁨 (예: XLY/XLP)
            if value <= crit_th:
                status = PathStatus.TRIGGERED
                level = SignalLevel.CRITICAL
            elif value <= warn_th:
                status = PathStatus.ACTIVE
                level = SignalLevel.WARNING
            else:
                status = PathStatus.INACTIVE
                level = SignalLevel.NORMAL
            deviation = ((warn_th - value) / abs(warn_th) * 100) if warn_th != 0 else 0
        else:
            # 높을수록 나쁨 (예: VIX)
            if value >= crit_th:
                status = PathStatus.TRIGGERED
                level = SignalLevel.CRITICAL
            elif value >= warn_th:
                status = PathStatus.ACTIVE
                level = SignalLevel.WARNING
            else:
                status = PathStatus.INACTIVE
                level = SignalLevel.NORMAL
            deviation = ((value - warn_th) / abs(warn_th) * 100) if warn_th != 0 else 0

        # 메시지 생성
        if status == PathStatus.TRIGGERED:
            message = f"CRITICAL: {path.name} at {value:.2f} (threshold: {crit_th})"
        elif status == PathStatus.ACTIVE:
            message = f"WARNING: {path.name} at {value:.2f} (threshold: {warn_th})"
        else:
            message = f"{path.name}: {value:.2f} (normal)"

        return PathSignal(
            path_id=path.id,
            path_name=path.name,
            status=status,
            level=level,
            value=value,
            threshold=warn_th,
            deviation=deviation,
            message=message,
            timestamp=timestamp,
        )

    def _create_inactive_signal(self, path: PathDefinition) -> PathSignal:
        """비활성 신호 생성"""
        return PathSignal(
            path_id=path.id,
            path_name=path.name,
            status=PathStatus.INACTIVE,
            level=SignalLevel.NORMAL,
            value=0.0,
            threshold=path.thresholds['warning'],
            deviation=0.0,
            message=f"{path.name}: Data unavailable",
            timestamp=datetime.now().isoformat(),
        )

    def _create_error_signal(self, path: PathDefinition, error: str) -> PathSignal:
        """에러 신호 생성"""
        return PathSignal(
            path_id=path.id,
            path_name=path.name,
            status=PathStatus.INACTIVE,
            level=SignalLevel.NORMAL,
            value=0.0,
            threshold=path.thresholds['warning'],
            deviation=0.0,
            message=f"{path.name}: Error - {error}",
            timestamp=datetime.now().isoformat(),
        )

    def analyze(self) -> CriticalPathSummary:
        """전체 경로 분석"""
        print("Analyzing 17 Critical Paths...")

        # 각 경로 체크
        check_methods = [
            self._check_path_1_yield_curve,
            self._check_path_2_copper_gold,
            self._check_path_3_hy_spreads,
            self._check_path_4_dollar_smile,
            self._check_path_5_sector_rotation,
            self._check_path_6_breakevens,
            self._check_path_7_vix_structure,
            self._check_path_8_em_flows,
            self._check_path_9_gold_silver,
            self._check_path_10_bank_stocks,
            self._check_path_11_crack_sequence,
            self._check_path_12_liquidity,
            self._check_path_13_meltup,
            self._check_path_14_correlation,
            self._check_path_15_capitulation,
            self._check_path_16_contagion,
            self._check_path_17_divergence,
        ]

        signals = []
        for i, method in enumerate(check_methods, 1):
            print(f"  [{i}/17] Checking Path {i}...")
            signal = method()
            signals.append(signal)

        # 집계
        active_paths = sum(1 for s in signals if s.status != PathStatus.INACTIVE)
        triggered_paths = sum(1 for s in signals if s.status == PathStatus.TRIGGERED)
        critical_count = sum(1 for s in signals if s.level == SignalLevel.CRITICAL)
        warning_count = sum(1 for s in signals if s.level == SignalLevel.WARNING)
        watch_count = sum(1 for s in signals if s.level == SignalLevel.WATCH)

        # 리스크 점수 계산
        risk_score = critical_count * 20 + warning_count * 10 + watch_count * 5
        risk_score = min(100, risk_score)

        # 시장 레짐 판단
        if critical_count >= 3 or risk_score >= 60:
            market_regime = "CRISIS"
        elif critical_count >= 1 or warning_count >= 3:
            market_regime = "RISK_OFF"
        elif warning_count >= 1:
            market_regime = "CAUTION"
        else:
            market_regime = "NORMAL"

        return CriticalPathSummary(
            timestamp=datetime.now().isoformat(),
            active_paths=active_paths,
            triggered_paths=triggered_paths,
            critical_count=critical_count,
            warning_count=warning_count,
            watch_count=watch_count,
            signals=signals,
            market_regime=market_regime,
            risk_score=risk_score,
        )

    def save_to_db(self, summary: CriticalPathSummary,
                   db: DatabaseManager = None) -> bool:
        """DB에 저장"""
        if db is None:
            db = DatabaseManager()

        today = datetime.now().strftime("%Y-%m-%d")

        try:
            db.save_etf_analysis('critical_paths', summary.to_dict(), today)
            db.log_analysis('critical_paths', 'SUCCESS', len(summary.signals), today)
            return True
        except Exception as e:
            print(f"Error saving to DB: {e}")
            return False

    def print_report(self, summary: CriticalPathSummary):
        """리포트 출력"""
        print("\n" + "=" * 70)
        print("CRITICAL PATH MONITOR")
        print(f"Generated: {summary.timestamp[:19]}")
        print("=" * 70)

        # 요약
        print(f"\n[Summary]")
        print(f"  Market Regime:   {summary.market_regime}")
        print(f"  Risk Score:      {summary.risk_score:.0f}/100")
        print(f"  Active Paths:    {summary.active_paths}/17")
        print(f"  Triggered:       {summary.triggered_paths}")
        print(f"  Critical:        {summary.critical_count}")
        print(f"  Warning:         {summary.warning_count}")

        # 활성 신호 (중요도순)
        active_signals = [s for s in summary.signals if s.status != PathStatus.INACTIVE]
        active_signals.sort(key=lambda x: (
            x.level != SignalLevel.CRITICAL,
            x.level != SignalLevel.WARNING,
        ))

        if active_signals:
            print(f"\n[Active Signals]")
            for sig in active_signals:
                icon = "🚨" if sig.level == SignalLevel.CRITICAL else "⚠️" if sig.level == SignalLevel.WARNING else "👁"
                print(f"  {icon} [{sig.path_id:2d}] {sig.message}")

        # 정상 경로
        normal_signals = [s for s in summary.signals if s.status == PathStatus.INACTIVE and s.value != 0]
        if normal_signals:
            print(f"\n[Normal Paths]")
            for sig in normal_signals[:5]:  # 상위 5개만
                print(f"  ✅ [{sig.path_id:2d}] {sig.path_name}: {sig.value:.2f}")

        print("\n" + "=" * 70)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Critical Path Monitor Test")
    print("=" * 70)

    monitor = CriticalPathMonitor()
    summary = monitor.analyze()
    monitor.print_report(summary)

    # DB 저장
    print("\n[Saving to Database]")
    db = DatabaseManager()
    if monitor.save_to_db(summary, db):
        print("  Saved successfully!")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
