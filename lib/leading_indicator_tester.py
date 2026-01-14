#!/usr/bin/env python3
"""
EIMAS Leading Indicator Tester
==============================
선행지표 유효성 자동 검증 (Granger Causality)

주요 기능:
1. 지표 간 Granger Causality 테스트
2. 최적 Lag 탐색
3. 선행 관계 시각화

Usage:
    from lib.leading_indicator_tester import LeadingIndicatorTester

    tester = LeadingIndicatorTester()
    results = tester.test_all_indicators()
    tester.print_report(results)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Granger tests disabled.")


# ============================================================================
# Constants
# ============================================================================

# 테스트할 선행지표 쌍
INDICATOR_PAIRS = [
    # (Leading Indicator, Target, Ticker for Leading, Ticker for Target)
    ("VIX", "SPY_Returns", "^VIX", "SPY"),
    ("TLT_Returns", "SPY_Returns", "TLT", "SPY"),  # 채권 → 주식
    ("GLD_Returns", "SPY_Returns", "GLD", "SPY"),  # 금 → 주식
    ("HYG_Returns", "SPY_Returns", "HYG", "SPY"),  # 하이일드 → 주식
    ("IWM_Returns", "SPY_Returns", "IWM", "SPY"),  # 소형주 → 대형주
    ("EEM_Returns", "SPY_Returns", "EEM", "SPY"),  # 이머징 → 미국
    ("XLF_Returns", "SPY_Returns", "XLF", "SPY"),  # 금융 → 전체
    ("COPPER_Returns", "SPY_Returns", "COPX", "SPY"),  # 구리 → 주식
]

# Lag 범위
MAX_LAG = 20  # 최대 20일


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GrangerResult:
    """Granger Causality 결과"""
    leading_indicator: str
    target: str
    optimal_lag: int
    p_value: float
    is_significant: bool  # p < 0.05
    f_statistic: float
    direction: str  # "leading", "lagging", "bidirectional", "none"


@dataclass
class LeadingIndicatorReport:
    """선행지표 리포트"""
    test_date: date
    total_pairs: int
    significant_pairs: int
    results: List[GrangerResult]
    rankings: List[Tuple[str, float]]  # (indicator, avg_lag)


# ============================================================================
# Leading Indicator Tester
# ============================================================================

class LeadingIndicatorTester:
    """선행지표 테스터"""

    def __init__(self):
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def _fetch_data(self, ticker: str, years: int = 3) -> pd.DataFrame:
        """데이터 로드"""
        if ticker not in self._data_cache:
            end = datetime.now()
            start = end - timedelta(days=365 * years)

            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            self._data_cache[ticker] = df

        return self._data_cache[ticker]

    def _prepare_series(
        self,
        ticker: str,
        is_return: bool = True
    ) -> pd.Series:
        """시계열 준비"""
        df = self._fetch_data(ticker)

        if df.empty:
            return pd.Series()

        if is_return:
            # 로그 수익률
            series = np.log(df['Close']).diff().dropna()
        else:
            # 레벨
            series = df['Close']

        return series

    def _check_stationarity(self, series: pd.Series) -> bool:
        """정상성 검정 (ADF 테스트)"""
        if not HAS_STATSMODELS:
            return True  # 기본적으로 통과

        try:
            result = adfuller(series.dropna())
            p_value = result[1]
            return p_value < 0.05  # 정상성 있음
        except:
            return False

    def test_granger_causality(
        self,
        leading_ticker: str,
        target_ticker: str,
        max_lag: int = MAX_LAG
    ) -> Optional[GrangerResult]:
        """Granger Causality 테스트"""
        if not HAS_STATSMODELS:
            print("  statsmodels required for Granger test")
            return None

        # 데이터 준비
        leading_is_return = "_Returns" in leading_ticker or leading_ticker != "^VIX"

        if leading_ticker == "^VIX":
            leading = self._prepare_series("^VIX", is_return=False)
            leading_name = "VIX"
        else:
            ticker = leading_ticker.replace("_Returns", "")
            leading = self._prepare_series(ticker, is_return=True)
            leading_name = leading_ticker

        target = self._prepare_series(target_ticker.replace("_Returns", ""), is_return=True)
        target_name = target_ticker

        if leading.empty or target.empty:
            return None

        # 인덱스 정렬
        combined = pd.DataFrame({
            'leading': leading,
            'target': target
        }).dropna()

        if len(combined) < max_lag * 2:
            return None

        # 정상성 확인
        if not self._check_stationarity(combined['leading']):
            print(f"  Warning: {leading_name} not stationary")

        if not self._check_stationarity(combined['target']):
            print(f"  Warning: {target_name} not stationary")

        # Granger 테스트 (leading → target)
        try:
            result = grangercausalitytests(
                combined[['target', 'leading']],
                maxlag=max_lag,
                verbose=False
            )

            # 최적 lag 찾기 (가장 낮은 p-value)
            best_lag = 1
            best_p = 1.0
            best_f = 0.0

            for lag in range(1, max_lag + 1):
                if lag in result:
                    # F-test p-value
                    p_value = result[lag][0]['ssr_ftest'][1]
                    f_stat = result[lag][0]['ssr_ftest'][0]

                    if p_value < best_p:
                        best_p = p_value
                        best_lag = lag
                        best_f = f_stat

            # 역방향도 테스트 (target → leading)
            result_reverse = grangercausalitytests(
                combined[['leading', 'target']],
                maxlag=max_lag,
                verbose=False
            )

            reverse_best_p = min(
                result_reverse[lag][0]['ssr_ftest'][1]
                for lag in range(1, max_lag + 1)
                if lag in result_reverse
            )

            # 방향 결정
            if best_p < 0.05 and reverse_best_p < 0.05:
                direction = "bidirectional"
            elif best_p < 0.05:
                direction = "leading"
            elif reverse_best_p < 0.05:
                direction = "lagging"
            else:
                direction = "none"

            return GrangerResult(
                leading_indicator=leading_name,
                target=target_name,
                optimal_lag=best_lag,
                p_value=best_p,
                is_significant=best_p < 0.05,
                f_statistic=best_f,
                direction=direction,
            )

        except Exception as e:
            print(f"  Error in Granger test: {e}")
            return None

    def test_all_indicators(self) -> LeadingIndicatorReport:
        """모든 지표 쌍 테스트"""
        print("=" * 70)
        print("EIMAS Leading Indicator Test")
        print("=" * 70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Testing {len(INDICATOR_PAIRS)} indicator pairs...")

        results = []

        for leading, target, leading_ticker, target_ticker in INDICATOR_PAIRS:
            print(f"\n[{leading} → {target}]")

            result = self.test_granger_causality(leading_ticker, target_ticker)

            if result:
                results.append(result)
                status = "✓ SIGNIFICANT" if result.is_significant else "✗ Not significant"
                print(f"  Optimal Lag: {result.optimal_lag} days")
                print(f"  P-value: {result.p_value:.4f}")
                print(f"  Direction: {result.direction}")
                print(f"  {status}")
            else:
                print("  ✗ Test failed")

        # 랭킹 (유의한 지표만)
        significant = [r for r in results if r.is_significant and r.direction == "leading"]
        rankings = [(r.leading_indicator, r.optimal_lag) for r in significant]
        rankings.sort(key=lambda x: x[1])  # lag 짧은 순

        return LeadingIndicatorReport(
            test_date=date.today(),
            total_pairs=len(INDICATOR_PAIRS),
            significant_pairs=len(significant),
            results=results,
            rankings=rankings,
        )

    def print_report(self, report: LeadingIndicatorReport):
        """리포트 출력"""
        print("\n" + "=" * 70)
        print("Leading Indicator Summary")
        print("=" * 70)

        print(f"\nTotal Pairs Tested: {report.total_pairs}")
        print(f"Significant Leading Indicators: {report.significant_pairs}")

        print(f"\n{'Indicator':<20} {'Target':<15} {'Lag':>6} {'P-value':>10} {'Direction':>15}")
        print("-" * 70)

        for r in sorted(report.results, key=lambda x: x.p_value):
            sig = "***" if r.p_value < 0.01 else "**" if r.p_value < 0.05 else "*" if r.p_value < 0.1 else ""
            print(f"{r.leading_indicator:<20} {r.target:<15} {r.optimal_lag:>5}d "
                  f"{r.p_value:>9.4f}{sig} {r.direction:>15}")

        if report.rankings:
            print("\n📊 Confirmed Leading Indicators (by lead time):")
            for indicator, lag in report.rankings:
                print(f"  {indicator}: {lag} days ahead")

        print("=" * 70)

    def get_cross_correlation(
        self,
        ticker1: str,
        ticker2: str,
        max_lag: int = 30
    ) -> Dict[int, float]:
        """교차 상관관계 분석"""
        s1 = self._prepare_series(ticker1)
        s2 = self._prepare_series(ticker2)

        if s1.empty or s2.empty:
            return {}

        combined = pd.DataFrame({'s1': s1, 's2': s2}).dropna()

        correlations = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # s1이 s2보다 앞서는 경우
                corr = combined['s1'].iloc[:lag].corr(combined['s2'].iloc[-lag:])
            elif lag > 0:
                # s2가 s1보다 앞서는 경우
                corr = combined['s1'].iloc[lag:].corr(combined['s2'].iloc[:-lag])
            else:
                corr = combined['s1'].corr(combined['s2'])

            correlations[lag] = corr

        return correlations

    def find_optimal_lead(
        self,
        ticker1: str,
        ticker2: str
    ) -> Tuple[int, float]:
        """최적 선행 기간 찾기"""
        correlations = self.get_cross_correlation(ticker1, ticker2)

        if not correlations:
            return 0, 0.0

        # 가장 높은 상관관계의 lag
        optimal_lag = max(correlations.keys(), key=lambda x: abs(correlations[x]))
        optimal_corr = correlations[optimal_lag]

        return optimal_lag, optimal_corr


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EIMAS Leading Indicator Tester")
    print("=" * 70)

    tester = LeadingIndicatorTester()

    # 전체 테스트
    report = tester.test_all_indicators()
    tester.print_report(report)

    # 교차 상관관계 예시
    print("\n" + "-" * 70)
    print("Cross-Correlation Example: VIX vs SPY")
    print("-" * 70)

    correlations = tester.get_cross_correlation("^VIX", "SPY")
    if correlations:
        # 상위 5개 lag
        sorted_lags = sorted(correlations.items(), key=lambda x: -abs(x[1]))[:5]
        for lag, corr in sorted_lags:
            direction = "VIX leads" if lag < 0 else "SPY leads" if lag > 0 else "Contemporaneous"
            print(f"  Lag {lag:>3}: {corr:>+.3f} ({direction})")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
