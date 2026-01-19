#!/usr/bin/env python3
"""
EIMAS Correlation Monitor
==========================
자산 간 상관관계 모니터링 및 이상 감지

주요 기능:
1. 롤링 상관관계 계산
2. 상관관계 체제 변화 감지
3. 상관관계 붕괴 경고 (Crisis Correlation)
4. 분산효과 모니터링

Usage:
    from lib.correlation_monitor import CorrelationMonitor

    cm = CorrelationMonitor()
    result = cm.analyze()
    cm.print_summary(result)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

# 주요 자산 유니버스
ASSET_UNIVERSE = {
    # Equity
    'SPY': 'S&P 500',
    'QQQ': 'NASDAQ 100',
    'IWM': 'Russell 2000',
    'EFA': 'EAFE (Developed)',
    'EEM': 'Emerging Markets',

    # Fixed Income
    'TLT': 'Long Treasury (20Y+)',
    'IEF': 'Med Treasury (7-10Y)',
    'LQD': 'Investment Grade Corp',
    'HYG': 'High Yield Corp',

    # Alternatives
    'GLD': 'Gold',
    'SLV': 'Silver',
    'USO': 'Oil',
    'UNG': 'Natural Gas',

    # Volatility
    'VXX': 'VIX Short-term',
}

# 핵심 상관관계 쌍 (정상 상태에서의 기대)
CORRELATION_PAIRS = {
    ('SPY', 'TLT'): {'normal': -0.3, 'range': 0.25, 'name': 'Stock-Bond'},
    ('SPY', 'GLD'): {'normal': 0.1, 'range': 0.3, 'name': 'Stock-Gold'},
    ('TLT', 'GLD'): {'normal': 0.2, 'range': 0.25, 'name': 'Bond-Gold'},
    ('SPY', 'QQQ'): {'normal': 0.95, 'range': 0.05, 'name': 'Large-Tech'},
    ('SPY', 'IWM'): {'normal': 0.85, 'range': 0.10, 'name': 'Large-Small'},
    ('SPY', 'EEM'): {'normal': 0.70, 'range': 0.15, 'name': 'US-EM'},
    ('SPY', 'VXX'): {'normal': -0.80, 'range': 0.10, 'name': 'Stock-VIX'},
    ('LQD', 'HYG'): {'normal': 0.75, 'range': 0.15, 'name': 'IG-HY'},
    ('GLD', 'SLV'): {'normal': 0.85, 'range': 0.10, 'name': 'Gold-Silver'},
}

# 설정
DEFAULT_LOOKBACK = 252  # 1년
ROLLING_WINDOWS = [21, 63, 126]  # 1M, 3M, 6M
CRISIS_THRESHOLD = 0.8  # 위기 시 상관관계 수렴 임계값
BREAKDOWN_THRESHOLD = 0.5  # 상관관계 붕괴 임계값 (정상 대비)


# ============================================================================
# Data Classes
# ============================================================================

class CorrelationState(str, Enum):
    """상관관계 상태"""
    NORMAL = "normal"          # 정상 범위
    ELEVATED = "elevated"      # 상승 (위기 가능)
    BREAKDOWN = "breakdown"    # 붕괴 (구조 변화)
    CRISIS = "crisis"          # 위기 (전체 상관관계 수렴)


@dataclass
class PairCorrelation:
    """상관관계 쌍 분석"""
    asset1: str
    asset2: str
    name: str
    current: float
    rolling_21d: float
    rolling_63d: float
    rolling_126d: float
    normal: float
    deviation: float  # 정상 대비 편차
    state: CorrelationState
    percentile: float  # 역사적 백분위


@dataclass
class CorrelationMatrix:
    """상관관계 행렬"""
    assets: List[str]
    matrix: pd.DataFrame
    timestamp: datetime


@dataclass
class DiversificationMetrics:
    """분산효과 지표"""
    average_correlation: float
    max_correlation: float
    min_correlation: float
    effective_assets: float  # 유효 자산 수 (1/sum(w^2))
    diversification_ratio: float
    crisis_indicator: float  # 위기 시 상관관계 수렴 정도


@dataclass
class CorrelationAlert:
    """상관관계 경고"""
    pair: Tuple[str, str]
    name: str
    alert_type: str  # 'breakdown', 'crisis', 'regime_change'
    message: str
    current_corr: float
    normal_corr: float
    severity: str  # 'warning', 'critical'


@dataclass
class CorrelationAnalysis:
    """전체 분석 결과"""
    timestamp: datetime
    pair_correlations: List[PairCorrelation]
    current_matrix: CorrelationMatrix
    diversification: DiversificationMetrics
    alerts: List[CorrelationAlert]
    regime: CorrelationState
    summary: str


# ============================================================================
# Correlation Monitor
# ============================================================================

class CorrelationMonitor:
    """상관관계 모니터"""

    def __init__(
        self,
        assets: List[str] = None,
        lookback: int = DEFAULT_LOOKBACK,
    ):
        self.assets = assets or list(ASSET_UNIVERSE.keys())
        self.lookback = lookback
        self.data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def fetch_data(self, period: str = "2y") -> pd.DataFrame:
        """가격 데이터 수집"""
        print(f"Fetching data for {len(self.assets)} assets...")

        try:
            # yfinance download
            data = yf.download(
                self.assets,
                period=period,
                progress=False,
                auto_adjust=True
            )
            
            # Extract Close prices safely
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker: columns are (Price, Ticker)
                if 'Close' in data.columns.get_level_values(0):
                     df = data['Close']
                else:
                     # If only one level or different structure
                     df = data
            else:
                # Single ticker or flat structure
                if 'Close' in data.columns:
                    df = pd.DataFrame({self.assets[0]: data['Close']})
                else:
                    df = data

            # Final check to flatten columns if still MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)

            # Drop assets with too much missing data
            if not df.empty:
                # Filter columns that have enough data
                valid_cols = []
                for col in df.columns:
                    # Check non-NaN ratio
                    if df[col].notna().sum() > len(df) * 0.6: # Relaxed to 60%
                        valid_cols.append(col)
                
                if valid_cols:
                    # Handle different trading calendars (e.g. Crypto vs Stocks)
                    # Forward fill missing values (e.g. weekends for stocks when mixed with crypto)
                    df = df[valid_cols].ffill().dropna()
                else:
                    df = pd.DataFrame()

            self.data = df
            self.returns = df.pct_change().dropna()

            print(f"  Loaded {len(df.columns)} assets, {len(df)} days")
            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def calculate_rolling_correlations(
        self,
        asset1: str,
        asset2: str,
        windows: List[int] = None,
    ) -> Dict[int, pd.Series]:
        """롤링 상관관계 계산"""
        if self.returns is None:
            self.fetch_data()

        windows = windows or ROLLING_WINDOWS
        results = {}

        if asset1 not in self.returns.columns or asset2 not in self.returns.columns:
            return results

        for window in windows:
            corr = self.returns[asset1].rolling(window).corr(self.returns[asset2])
            results[window] = corr

        return results

    def analyze_pair(self, asset1: str, asset2: str) -> Optional[PairCorrelation]:
        """쌍별 상관관계 분석"""
        if self.returns is None or asset1 not in self.returns.columns or asset2 not in self.returns.columns:
            return None

        rolling = self.calculate_rolling_correlations(asset1, asset2)
        if not rolling:
            return None

        current = rolling[21].iloc[-1] if 21 in rolling else np.nan
        r21 = rolling[21].iloc[-1] if 21 in rolling else np.nan
        r63 = rolling[63].iloc[-1] if 63 in rolling else np.nan
        r126 = rolling[126].iloc[-1] if 126 in rolling else np.nan

        # 정상 상관관계 조회
        pair_key = (asset1, asset2)
        reverse_key = (asset2, asset1)

        if pair_key in CORRELATION_PAIRS:
            info = CORRELATION_PAIRS[pair_key]
        elif reverse_key in CORRELATION_PAIRS:
            info = CORRELATION_PAIRS[reverse_key]
        else:
            # 정의되지 않은 쌍: 126일 평균을 정상으로
            full_corr = self.returns[asset1].corr(self.returns[asset2])
            info = {'normal': full_corr, 'range': 0.2, 'name': f'{asset1}-{asset2}'}

        normal = info['normal']
        range_val = info['range']
        name = info['name']

        deviation = current - normal

        # 상태 판단
        if abs(deviation) <= range_val:
            state = CorrelationState.NORMAL
        elif deviation > range_val:
            if current > CRISIS_THRESHOLD:
                state = CorrelationState.CRISIS
            else:
                state = CorrelationState.ELEVATED
        else:  # deviation < -range_val
            state = CorrelationState.BREAKDOWN

        # 역사적 백분위
        full_series = rolling[63].dropna() if 63 in rolling else pd.Series()
        if len(full_series) > 0:
            percentile = (full_series < current).mean() * 100
        else:
            percentile = 50.0

        return PairCorrelation(
            asset1=asset1,
            asset2=asset2,
            name=name,
            current=float(current) if not np.isnan(current) else 0.0,
            rolling_21d=float(r21) if not np.isnan(r21) else 0.0,
            rolling_63d=float(r63) if not np.isnan(r63) else 0.0,
            rolling_126d=float(r126) if not np.isnan(r126) else 0.0,
            normal=normal,
            deviation=float(deviation) if not np.isnan(deviation) else 0.0,
            state=state,
            percentile=float(percentile),
        )

    def calculate_correlation_matrix(self, window: int = 63) -> CorrelationMatrix:
        """상관관계 행렬 계산"""
        if self.returns is None:
            self.fetch_data()

        recent = self.returns.tail(window)
        matrix = recent.corr()

        return CorrelationMatrix(
            assets=list(matrix.columns),
            matrix=matrix,
            timestamp=datetime.now(),
        )

    def calculate_diversification_metrics(
        self,
        weights: Dict[str, float] = None,
    ) -> DiversificationMetrics:
        """분산효과 지표 계산"""
        matrix = self.calculate_correlation_matrix()
        corr = matrix.matrix

        # 대각선 제외
        np.fill_diagonal(corr.values, np.nan)

        avg_corr = np.nanmean(corr.values)
        max_corr = np.nanmax(corr.values)
        min_corr = np.nanmin(corr.values)

        # 동일 가중 기준 유효 자산 수
        n = len(corr)
        if weights:
            w = np.array([weights.get(a, 0) for a in corr.columns])
            w = w / w.sum()
            effective = 1 / np.sum(w ** 2)
        else:
            effective = n  # 동일 가중 시 n과 같음

        # 분산효과 비율 (개별 변동성 가중합 / 포트폴리오 변동성)
        # 여기서는 간단히 1 - 평균상관관계로 근사
        div_ratio = 1 - avg_corr

        # 위기 지표: 상관관계가 높을수록 위기
        crisis_ind = max(0, (avg_corr - 0.3) / 0.5)  # 0.3 초과 시 증가

        return DiversificationMetrics(
            average_correlation=float(avg_corr),
            max_correlation=float(max_corr),
            min_correlation=float(min_corr),
            effective_assets=float(effective),
            diversification_ratio=float(div_ratio),
            crisis_indicator=float(crisis_ind),
        )

    def detect_regime(self, pairs: List[PairCorrelation]) -> CorrelationState:
        """전체 상관관계 체제 감지"""
        if not pairs:
            return CorrelationState.NORMAL

        crisis_count = sum(1 for p in pairs if p.state == CorrelationState.CRISIS)
        breakdown_count = sum(1 for p in pairs if p.state == CorrelationState.BREAKDOWN)
        elevated_count = sum(1 for p in pairs if p.state == CorrelationState.ELEVATED)

        total = len(pairs)

        if crisis_count >= total * 0.3:
            return CorrelationState.CRISIS
        elif breakdown_count >= total * 0.2:
            return CorrelationState.BREAKDOWN
        elif elevated_count >= total * 0.4:
            return CorrelationState.ELEVATED
        else:
            return CorrelationState.NORMAL

    def generate_alerts(self, pairs: List[PairCorrelation]) -> List[CorrelationAlert]:
        """경고 생성"""
        alerts = []

        for pair in pairs:
            if pair.state == CorrelationState.CRISIS:
                alerts.append(CorrelationAlert(
                    pair=(pair.asset1, pair.asset2),
                    name=pair.name,
                    alert_type='crisis',
                    message=f"Crisis correlation detected! {pair.name} correlation at {pair.current:.2f}",
                    current_corr=pair.current,
                    normal_corr=pair.normal,
                    severity='critical',
                ))
            elif pair.state == CorrelationState.BREAKDOWN:
                alerts.append(CorrelationAlert(
                    pair=(pair.asset1, pair.asset2),
                    name=pair.name,
                    alert_type='breakdown',
                    message=f"Correlation breakdown: {pair.name} at {pair.current:.2f} (normal: {pair.normal:.2f})",
                    current_corr=pair.current,
                    normal_corr=pair.normal,
                    severity='warning',
                ))

        # Stock-Bond 특별 경고
        stock_bond = next((p for p in pairs if p.name == 'Stock-Bond'), None)
        if stock_bond and stock_bond.current > 0.3:
            alerts.append(CorrelationAlert(
                pair=('SPY', 'TLT'),
                name='Stock-Bond',
                alert_type='regime_change',
                message=f"Stock-Bond correlation turned POSITIVE ({stock_bond.current:.2f}). 60/40 may not diversify!",
                current_corr=stock_bond.current,
                normal_corr=stock_bond.normal,
                severity='critical',
            ))

        return alerts

    def analyze(self) -> CorrelationAnalysis:
        """전체 분석 실행"""
        print("\n" + "=" * 60)
        print("EIMAS Correlation Monitor")
        print("=" * 60)

        # 데이터 수집
        if self.returns is None:
            self.fetch_data()

        # 쌍별 분석
        pair_results = []
        for (a1, a2), info in CORRELATION_PAIRS.items():
            result = self.analyze_pair(a1, a2)
            if result:
                pair_results.append(result)

        # 행렬
        matrix = self.calculate_correlation_matrix()

        # 분산효과
        div_metrics = self.calculate_diversification_metrics()

        # 체제 감지
        regime = self.detect_regime(pair_results)

        # 경고
        alerts = self.generate_alerts(pair_results)

        # 요약
        summary = self._generate_summary(pair_results, div_metrics, regime, alerts)

        return CorrelationAnalysis(
            timestamp=datetime.now(),
            pair_correlations=pair_results,
            current_matrix=matrix,
            diversification=div_metrics,
            alerts=alerts,
            regime=regime,
            summary=summary,
        )

    def _generate_summary(
        self,
        pairs: List[PairCorrelation],
        div: DiversificationMetrics,
        regime: CorrelationState,
        alerts: List[CorrelationAlert],
    ) -> str:
        """요약 생성"""
        lines = []

        # 체제
        regime_emoji = {
            CorrelationState.NORMAL: "🟢",
            CorrelationState.ELEVATED: "🟡",
            CorrelationState.BREAKDOWN: "🟠",
            CorrelationState.CRISIS: "🔴",
        }
        lines.append(f"{regime_emoji[regime]} Correlation Regime: {regime.value.upper()}")

        # 분산효과
        lines.append(f"Average Correlation: {div.average_correlation:.2f}")
        lines.append(f"Diversification Ratio: {div.diversification_ratio:.2f}")

        # 주요 쌍
        if pairs:
            lines.append("\nKey Pairs:")
            for p in pairs[:5]:
                state_emoji = regime_emoji.get(p.state, "⚪")
                lines.append(f"  {state_emoji} {p.name}: {p.current:.2f} (normal: {p.normal:.2f})")

        # 경고
        if alerts:
            lines.append(f"\n⚠️ {len(alerts)} Alert(s)")

        return "\n".join(lines)

    def print_summary(self, result: CorrelationAnalysis):
        """결과 출력"""
        print("\n" + result.summary)

        if result.alerts:
            print("\n" + "-" * 40)
            print("ALERTS:")
            for alert in result.alerts:
                icon = "🚨" if alert.severity == 'critical' else "⚠️"
                print(f"  {icon} {alert.message}")

        print("\n" + "=" * 60)

    def get_correlation_heatmap_data(self) -> Dict[str, Any]:
        """히트맵용 데이터 반환"""
        matrix = self.calculate_correlation_matrix()
        return {
            'assets': matrix.assets,
            'values': matrix.matrix.values.tolist(),
            'timestamp': matrix.timestamp.isoformat(),
        }


# ============================================================================
# Utility Functions
# ============================================================================

def quick_correlation_check(
    assets: List[str] = ['SPY', 'TLT', 'GLD', 'QQQ'],
    window: int = 21,
) -> pd.DataFrame:
    """빠른 상관관계 확인"""
    df = yf.download(assets, period="6mo", progress=False)['Close']
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    returns = df.pct_change().dropna()
    return returns.tail(window).corr()


def detect_correlation_spike(
    asset1: str,
    asset2: str,
    threshold: float = 0.9,
    window: int = 21,
) -> Dict[str, Any]:
    """상관관계 급등 감지"""
    cm = CorrelationMonitor([asset1, asset2])
    cm.fetch_data()

    rolling = cm.calculate_rolling_correlations(asset1, asset2)

    if 21 not in rolling:
        return {'spike': False}

    current = rolling[21].iloc[-1]
    historical_max = rolling[21].max()
    spike = current > threshold or current > historical_max * 0.95

    return {
        'spike': spike,
        'current': float(current),
        'historical_max': float(historical_max),
        'percentile': float((rolling[21] < current).mean() * 100),
    }


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # 테스트 실행
    cm = CorrelationMonitor()
    result = cm.analyze()
    cm.print_summary(result)

    # 히트맵 데이터
    print("\nHeatmap data preview:")
    heatmap = cm.get_correlation_heatmap_data()
    print(f"  Assets: {heatmap['assets'][:5]}...")
    print(f"  Matrix shape: {len(heatmap['values'])}x{len(heatmap['values'][0])}")

    # 빠른 체크
    print("\nQuick correlation check (SPY, TLT, GLD, QQQ):")
    quick = quick_correlation_check()
    print(quick.round(2))
