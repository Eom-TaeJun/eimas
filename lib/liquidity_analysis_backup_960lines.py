"""
Liquidity Analysis Module
==========================
RRP/TGA/Fed Assets와 시장 변수의 인과관계 분석

핵심 가설:
- RRP 감소 → 유동성 유입 → 위험자산 상승 (30분~1시간 선행)
- TGA 증가 → 유동성 흡수 → 위험자산 하락
- Net Liquidity 변화 → SPY/BTC 호가 구조 변화

기존 causal_network.py의 Granger Causality를 활용
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# 내부 모듈
from lib.causal_network import (
    CausalNetworkAnalyzer,
    GrangerTestResult,
    CausalPath,
    NetworkAnalysisResult
)
from lib.fred_collector import FREDCollector, FRED_SERIES


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class LiquidityImpactResult:
    """유동성 → 시장 영향 분석 결과"""
    timestamp: datetime

    # RRP 영향
    rrp_to_spy_lag: int = 0          # RRP가 SPY에 선행하는 일수
    rrp_to_spy_pvalue: float = 1.0
    rrp_to_spy_significant: bool = False

    rrp_to_vix_lag: int = 0
    rrp_to_vix_pvalue: float = 1.0
    rrp_to_vix_significant: bool = False

    # TGA 영향
    tga_to_spy_lag: int = 0
    tga_to_spy_pvalue: float = 1.0
    tga_to_spy_significant: bool = False

    # Net Liquidity 영향
    net_liq_to_spy_lag: int = 0
    net_liq_to_spy_pvalue: float = 1.0
    net_liq_to_spy_significant: bool = False

    # 핵심 드라이버
    key_drivers: List[str] = field(default_factory=list)
    critical_path: Optional[str] = None

    # 전체 결과
    granger_results: List[GrangerTestResult] = field(default_factory=list)
    network_stats: Dict[str, Any] = field(default_factory=dict)

    # 해석
    interpretation: str = ""
    trading_signal: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    signal_confidence: float = 0.0


@dataclass
class LiquidityCorrelation:
    """유동성-시장 상관관계"""
    variable1: str
    variable2: str
    correlation: float
    rolling_corr_mean: float
    rolling_corr_std: float
    is_significant: bool
    correlation_regime: str  # "high_positive", "low", "negative"


# ============================================================================
# Liquidity Market Analyzer
# ============================================================================

class LiquidityMarketAnalyzer:
    """
    유동성 → 시장 영향 분석기

    FRED 유동성 데이터와 시장 데이터를 결합하여
    Granger Causality 분석 수행
    """

    # 유동성 변수들
    LIQUIDITY_VARIABLES = [
        'rrp',           # Overnight RRP
        'tga',           # Treasury General Account
        'fed_assets',    # Fed Total Assets
        'net_liquidity', # Fed - RRP - TGA
        'rrp_delta',     # RRP 일간 변화
        'tga_delta',     # TGA 주간 변화
    ]

    # 시장 변수들
    MARKET_VARIABLES = [
        'spy',           # S&P 500 ETF
        'vix',           # VIX
        'spy_return',    # SPY 일간 수익률
        'spy_volatility', # SPY 변동성
        'hy_spread',     # HY 스프레드
    ]

    def __init__(
        self,
        max_lag: int = 10,
        significance_level: float = 0.05
    ):
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.analyzer = CausalNetworkAnalyzer(
            max_lag=max_lag,
            significance_level=significance_level
        )

    def prepare_data(
        self,
        liquidity_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        유동성 + 시장 데이터 결합

        Parameters:
        -----------
        liquidity_df : DataFrame
            날짜 인덱스, 컬럼: rrp, tga, fed_assets, net_liquidity 등
        market_df : DataFrame (optional)
            날짜 인덱스, 컬럼: spy, vix, hy_spread 등
            None이면 liquidity_df에 시장 데이터가 이미 포함된 것으로 간주

        Returns:
        --------
        DataFrame : 결합된 데이터
        """
        # 인덱스를 datetime으로 변환
        if not isinstance(liquidity_df.index, pd.DatetimeIndex):
            liquidity_df.index = pd.to_datetime(liquidity_df.index)

        # market_df가 None이면 liquidity_df를 그대로 사용
        if market_df is None:
            combined = liquidity_df.copy()
        else:
            if not isinstance(market_df.index, pd.DatetimeIndex):
                market_df.index = pd.to_datetime(market_df.index)

            # 겹치는 컬럼 제거 후 결합
            overlap = set(liquidity_df.columns) & set(market_df.columns)
            market_df_clean = market_df.drop(columns=list(overlap), errors='ignore')

            combined = pd.merge(
                liquidity_df,
                market_df_clean,
                left_index=True,
                right_index=True,
                how='inner'
            )

        # 파생 변수 계산
        if 'spy' in combined.columns:
            combined['spy_return'] = combined['spy'].pct_change() * 100
            combined['spy_volatility'] = combined['spy_return'].rolling(5).std()

        if 'rrp' in combined.columns:
            combined['rrp_delta'] = combined['rrp'].diff()
            combined['rrp_pct_change'] = combined['rrp'].pct_change() * 100

        if 'tga' in combined.columns:
            combined['tga_delta'] = combined['tga'].diff()

        # Net Liquidity 계산 (없으면)
        if 'net_liquidity' not in combined.columns:
            if all(c in combined.columns for c in ['fed_assets', 'rrp', 'tga']):
                combined['net_liquidity'] = (
                    combined['fed_assets'] * 1000 -  # T → B
                    combined['rrp'] -
                    combined['tga']
                )

        return combined.dropna()

    def analyze_liquidity_impact(
        self,
        data: pd.DataFrame,
        target: str = 'spy'
    ) -> LiquidityImpactResult:
        """
        유동성 → 시장 영향 분석

        Parameters:
        -----------
        data : DataFrame
            prepare_data()로 준비된 데이터
        target : str
            분석 타겟 (기본: spy)

        Returns:
        --------
        LiquidityImpactResult
        """
        result = LiquidityImpactResult(timestamp=datetime.now())

        # 분석할 변수 선택
        available_liq = [v for v in self.LIQUIDITY_VARIABLES if v in data.columns]
        available_mkt = [v for v in self.MARKET_VARIABLES if v in data.columns]

        all_vars = available_liq + available_mkt

        if len(all_vars) < 2:
            result.interpretation = "분석 가능한 변수가 부족합니다."
            return result

        # Granger Causality 분석
        network_result = self.analyzer.analyze(
            data=data[all_vars],
            target_variable=target,
            make_stationary=True
        )

        result.granger_results = network_result.granger_results
        result.network_stats = network_result.network_stats
        result.key_drivers = network_result.key_drivers

        if network_result.critical_path:
            result.critical_path = network_result.critical_path.description

        # 개별 유동성 변수 영향 추출
        for gr in network_result.granger_results:
            # RRP → SPY
            if gr.cause == 'rrp' and gr.effect == target:
                result.rrp_to_spy_lag = gr.optimal_lag
                result.rrp_to_spy_pvalue = gr.p_value
                result.rrp_to_spy_significant = gr.is_significant

            # RRP → VIX
            if gr.cause == 'rrp' and gr.effect == 'vix':
                result.rrp_to_vix_lag = gr.optimal_lag
                result.rrp_to_vix_pvalue = gr.p_value
                result.rrp_to_vix_significant = gr.is_significant

            # TGA → SPY
            if gr.cause == 'tga' and gr.effect == target:
                result.tga_to_spy_lag = gr.optimal_lag
                result.tga_to_spy_pvalue = gr.p_value
                result.tga_to_spy_significant = gr.is_significant

            # Net Liquidity → SPY
            if gr.cause == 'net_liquidity' and gr.effect == target:
                result.net_liq_to_spy_lag = gr.optimal_lag
                result.net_liq_to_spy_pvalue = gr.p_value
                result.net_liq_to_spy_significant = gr.is_significant

        # 해석 생성
        result.interpretation = self._generate_interpretation(result, data)
        result.trading_signal, result.signal_confidence = self._generate_signal(result, data)

        return result

    def calculate_correlations(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> List[LiquidityCorrelation]:
        """
        유동성-시장 상관관계 계산

        Parameters:
        -----------
        data : DataFrame
            결합된 데이터
        window : int
            롤링 윈도우 크기

        Returns:
        --------
        List[LiquidityCorrelation]
        """
        correlations = []

        liq_vars = [v for v in self.LIQUIDITY_VARIABLES if v in data.columns]
        mkt_vars = [v for v in self.MARKET_VARIABLES if v in data.columns]

        for liq in liq_vars:
            for mkt in mkt_vars:
                if liq == mkt:
                    continue

                # 정적 상관관계
                corr = data[liq].corr(data[mkt])

                # 롤링 상관관계
                rolling_corr = data[liq].rolling(window).corr(data[mkt])
                rolling_mean = rolling_corr.mean()
                rolling_std = rolling_corr.std()

                # 레짐 판단
                if corr > 0.5:
                    regime = "high_positive"
                elif corr < -0.3:
                    regime = "negative"
                else:
                    regime = "low"

                correlations.append(LiquidityCorrelation(
                    variable1=liq,
                    variable2=mkt,
                    correlation=corr,
                    rolling_corr_mean=rolling_mean,
                    rolling_corr_std=rolling_std,
                    is_significant=abs(corr) > 0.3,
                    correlation_regime=regime
                ))

        # 절대 상관관계 순 정렬
        correlations.sort(key=lambda x: abs(x.correlation), reverse=True)

        return correlations

    def _generate_interpretation(
        self,
        result: LiquidityImpactResult,
        data: pd.DataFrame
    ) -> str:
        """분석 결과 해석 생성"""
        lines = []

        # RRP 영향
        if result.rrp_to_spy_significant:
            lines.append(
                f"RRP가 SPY에 {result.rrp_to_spy_lag}일 선행 "
                f"(p={result.rrp_to_spy_pvalue:.4f})"
            )

        # TGA 영향
        if result.tga_to_spy_significant:
            lines.append(
                f"TGA가 SPY에 {result.tga_to_spy_lag}일 선행 "
                f"(p={result.tga_to_spy_pvalue:.4f})"
            )

        # Net Liquidity 영향
        if result.net_liq_to_spy_significant:
            lines.append(
                f"Net Liquidity가 SPY에 {result.net_liq_to_spy_lag}일 선행 "
                f"(p={result.net_liq_to_spy_pvalue:.4f})"
            )

        # 핵심 드라이버
        if result.key_drivers:
            lines.append(f"핵심 선행 지표: {', '.join(result.key_drivers)}")

        # Critical Path
        if result.critical_path:
            lines.append(f"주요 경로: {result.critical_path}")

        if not lines:
            lines.append("유의미한 유동성 → 시장 인과관계가 발견되지 않았습니다.")

        return "\n".join(lines)

    def _generate_signal(
        self,
        result: LiquidityImpactResult,
        data: pd.DataFrame
    ) -> Tuple[str, float]:
        """매매 신호 생성"""
        bullish_score = 0
        bearish_score = 0

        # 최근 유동성 변화 확인
        if 'rrp_delta' in data.columns:
            recent_rrp_delta = data['rrp_delta'].iloc[-5:].mean()
            if recent_rrp_delta < -10:  # RRP 감소 = 유동성 유입
                bullish_score += 2 if result.rrp_to_spy_significant else 1
            elif recent_rrp_delta > 10:
                bearish_score += 2 if result.rrp_to_spy_significant else 1

        if 'tga_delta' in data.columns:
            recent_tga_delta = data['tga_delta'].iloc[-5:].mean()
            if recent_tga_delta > 20:  # TGA 증가 = 유동성 흡수
                bearish_score += 1
            elif recent_tga_delta < -20:
                bullish_score += 1

        if 'net_liquidity' in data.columns:
            recent_net_liq = data['net_liquidity'].iloc[-1]
            if recent_net_liq > 4000:  # Abundant
                bullish_score += 1
            elif recent_net_liq < 2500:  # Stressed
                bearish_score += 2

        # 신호 결정
        total = bullish_score + bearish_score
        if total == 0:
            return "NEUTRAL", 0.0

        if bullish_score > bearish_score:
            confidence = bullish_score / (total + 1)
            return "BULLISH", min(confidence, 1.0)
        elif bearish_score > bullish_score:
            confidence = bearish_score / (total + 1)
            return "BEARISH", min(confidence, 1.0)
        else:
            return "NEUTRAL", 0.3

    def generate_signals(self) -> Dict[str, Any]:
        """
        유동성 기반 매매 신호 생성 (단축 메서드)

        FRED 데이터를 자동으로 수집하여 신호 생성

        Returns:
        --------
        Dict with signal information:
            - signal: BULLISH/BEARISH/NEUTRAL
            - confidence: 신뢰도 (0-1)
            - interpretation: 해석
        """
        try:
            # FRED 데이터 수집
            fred = FREDCollector()
            liquidity_data = fred.collect_all()

            if liquidity_data is None:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': 0.0,
                    'interpretation': 'FRED 데이터 수집 실패'
                }

            # DataFrame 준비
            data_dict = {
                'rrp': liquidity_data.get('rrp', 0),
                'tga': liquidity_data.get('tga', 0),
                'fed_assets': liquidity_data.get('fed_assets', 0),
                'net_liquidity': liquidity_data.get('net_liquidity', 0),
            }

            # 단일 행이라도 신호 생성
            if data_dict['net_liquidity'] > 5000:
                signal = 'BULLISH'
                confidence = 0.6
                interpretation = f"Net Liquidity ${data_dict['net_liquidity']:.0f}B - 풍부한 유동성"
            elif data_dict['net_liquidity'] > 4000:
                signal = 'NEUTRAL'
                confidence = 0.5
                interpretation = f"Net Liquidity ${data_dict['net_liquidity']:.0f}B - 정상 유동성"
            else:
                signal = 'BEARISH'
                confidence = 0.6
                interpretation = f"Net Liquidity ${data_dict['net_liquidity']:.0f}B - 유동성 부족"

            return {
                'signal': signal,
                'confidence': confidence,
                'interpretation': interpretation,
                'rrp': data_dict['rrp'],
                'tga': data_dict['tga'],
                'net_liquidity': data_dict['net_liquidity']
            }

        except Exception as e:
            return {
                'signal': 'NEUTRAL',
                'confidence': 0.0,
                'interpretation': f'분석 오류: {str(e)}'
            }


# ============================================================================
# Dynamic Lag Analyzer (Elicit Report Enhancement)
# ============================================================================

@dataclass
class AssetClassLag:
    """자산 클래스별 동적 시차 결과"""
    asset_class: str                    # equity, fixed_income, real_estate, commodity, crypto
    optimal_lag_days: int               # 최적 시차 (일)
    optimal_lag_weeks: float            # 최적 시차 (주)
    p_value: float
    r_squared: float
    is_significant: bool
    lag_confidence: str                 # HIGH, MEDIUM, LOW
    economic_interpretation: str


@dataclass
class RegimeConditionalLag:
    """레짐별 시차 분석 결과"""
    regime: str                         # BULL, BEAR, NEUTRAL, CRISIS
    avg_lag_days: float
    lag_volatility: float               # 시차 변동성
    sample_size: int
    interpretation: str


@dataclass
class DynamicLagResult:
    """동적 시차 분석 종합 결과"""
    timestamp: datetime

    # 자산 클래스별 시차
    asset_class_lags: List[AssetClassLag] = field(default_factory=list)

    # 레짐별 시차
    regime_lags: List[RegimeConditionalLag] = field(default_factory=list)

    # Cross-asset 시차 구조
    lag_matrix: Optional[pd.DataFrame] = None

    # 핵심 인사이트
    fastest_response_asset: str = ""
    slowest_response_asset: str = ""
    current_regime_lag: int = 0

    # 투자 시사점
    actionable_insight: str = ""


class DynamicLagAnalyzer:
    """
    동적 시차 분석기 (Elicit Report Enhancement)

    학술적 근거:
    - Bernanke & Kuttner (2005): 통화정책 → 주식시장 전이
    - Lastrapes (1998): 자산 클래스별 차별화된 반응 시차
    - Longin & Solnik (2001): 위기 시 상관관계 구조 변화

    Perplexity 검증 결과:
    - 주식시장: 정책 발표 당일~수일 내 반응 (글로벌 연구)
    - 부동산: 6-12개월 지연 (전통적 연구)
    - 위기 시: 상관관계 61.4% 증가 (Forbes-Rigobon 확인)
    """

    # 자산 클래스 정의
    ASSET_CLASSES = {
        'equity': ['SPY', 'QQQ', 'IWM', 'DIA'],
        'fixed_income': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG'],
        'real_estate': ['VNQ', 'IYR', 'XLRE'],
        'commodity': ['GLD', 'SLV', 'USO', 'DBA'],
        'crypto': ['BTC-USD', 'ETH-USD']
    }

    # 선험적 시차 가이드라인 (Elicit + Perplexity 종합)
    PRIOR_LAGS = {
        'equity': {'min': 0, 'max': 10, 'expected': 3},        # 즉각~2주
        'fixed_income': {'min': 0, 'max': 5, 'expected': 1},   # 즉각~1주
        'real_estate': {'min': 20, 'max': 60, 'expected': 40}, # 1~3개월
        'commodity': {'min': 0, 'max': 15, 'expected': 5},     # 즉각~3주
        'crypto': {'min': 0, 'max': 7, 'expected': 2}          # 즉각~1주
    }

    def __init__(
        self,
        max_lag: int = 60,
        significance_level: float = 0.05,
        min_observations: int = 100
    ):
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.min_observations = min_observations

    def analyze_asset_class_lags(
        self,
        policy_variable: pd.Series,
        asset_returns: Dict[str, pd.Series],
        regime_series: Optional[pd.Series] = None
    ) -> DynamicLagResult:
        """
        자산 클래스별 동적 시차 분석

        Parameters:
        -----------
        policy_variable : Series
            통화정책 변수 (예: Fed Funds Rate 변화, Net Liquidity 변화)
        asset_returns : Dict[str, Series]
            자산별 수익률 시계열
        regime_series : Series (optional)
            시장 레짐 시계열 (BULL/BEAR/NEUTRAL)

        Returns:
        --------
        DynamicLagResult
        """
        result = DynamicLagResult(timestamp=datetime.now())

        # 자산 클래스별 분석
        for asset_class, tickers in self.ASSET_CLASSES.items():
            available_tickers = [t for t in tickers if t in asset_returns]
            if not available_tickers:
                continue

            # 자산 클래스 대표 수익률 (평균)
            class_returns = pd.concat(
                [asset_returns[t] for t in available_tickers],
                axis=1
            ).mean(axis=1)

            # 최적 시차 탐색
            lag_result = self._find_optimal_lag(
                policy_variable,
                class_returns,
                self.PRIOR_LAGS.get(asset_class, {'min': 0, 'max': 30, 'expected': 10})
            )

            if lag_result:
                result.asset_class_lags.append(lag_result)

        # 레짐별 시차 분석
        if regime_series is not None:
            result.regime_lags = self._analyze_regime_conditional_lags(
                policy_variable, asset_returns, regime_series
            )

        # 핵심 인사이트 도출
        if result.asset_class_lags:
            sorted_lags = sorted(result.asset_class_lags, key=lambda x: x.optimal_lag_days)
            result.fastest_response_asset = sorted_lags[0].asset_class
            result.slowest_response_asset = sorted_lags[-1].asset_class

            # 투자 시사점
            result.actionable_insight = self._generate_actionable_insight(result)

        return result

    def _find_optimal_lag(
        self,
        x: pd.Series,
        y: pd.Series,
        prior: Dict[str, int]
    ) -> Optional[AssetClassLag]:
        """최적 시차 탐색 (Granger Causality 기반)"""
        from statsmodels.tsa.stattools import grangercausalitytests

        # 데이터 정렬
        data = pd.DataFrame({'x': x, 'y': y}).dropna()
        if len(data) < self.min_observations:
            return None

        # 정상성 확보 (차분)
        data_diff = data.diff().dropna()

        # 시차별 테스트
        best_lag = prior['expected']
        best_pvalue = 1.0

        search_range = range(
            max(1, prior['min']),
            min(prior['max'] + 1, self.max_lag + 1)
        )

        for lag in search_range:
            try:
                test_result = grangercausalitytests(
                    data_diff[['y', 'x']],
                    maxlag=lag,
                    verbose=False
                )
                # F-test p-value
                p_value = test_result[lag][0]['ssr_ftest'][1]
                if p_value < best_pvalue:
                    best_pvalue = p_value
                    best_lag = lag
            except Exception:
                continue

        # R-squared 계산 (shifted regression)
        from scipy import stats
        y_shifted = data['y'].shift(-best_lag)
        valid = ~(data['x'].isna() | y_shifted.isna())
        if valid.sum() > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                data['x'][valid], y_shifted[valid]
            )
            r_squared = r_value ** 2
        else:
            r_squared = 0.0

        # 신뢰도 결정
        if best_pvalue < 0.01 and r_squared > 0.1:
            confidence = "HIGH"
        elif best_pvalue < 0.05:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        # 경제학적 해석
        interpretation = self._interpret_lag(best_lag, prior['expected'])

        return AssetClassLag(
            asset_class=prior.get('name', 'unknown'),
            optimal_lag_days=best_lag,
            optimal_lag_weeks=best_lag / 5,  # 거래일 기준
            p_value=best_pvalue,
            r_squared=r_squared,
            is_significant=best_pvalue < self.significance_level,
            lag_confidence=confidence,
            economic_interpretation=interpretation
        )

    def _analyze_regime_conditional_lags(
        self,
        policy_variable: pd.Series,
        asset_returns: Dict[str, pd.Series],
        regime_series: pd.Series
    ) -> List[RegimeConditionalLag]:
        """레짐별 조건부 시차 분석"""
        results = []

        # 대표 자산 (SPY) 사용
        if 'SPY' not in asset_returns:
            return results

        spy_returns = asset_returns['SPY']
        data = pd.DataFrame({
            'policy': policy_variable,
            'spy': spy_returns,
            'regime': regime_series
        }).dropna()

        for regime in data['regime'].unique():
            regime_data = data[data['regime'] == regime]
            if len(regime_data) < 50:  # 최소 샘플
                continue

            # 레짐별 최적 시차
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                diff_data = regime_data[['spy', 'policy']].diff().dropna()

                lags_found = []
                for lag in range(1, min(15, len(diff_data) // 10)):
                    try:
                        test = grangercausalitytests(diff_data, maxlag=lag, verbose=False)
                        p_val = test[lag][0]['ssr_ftest'][1]
                        if p_val < 0.1:
                            lags_found.append(lag)
                    except:
                        continue

                avg_lag = np.mean(lags_found) if lags_found else 5
                lag_vol = np.std(lags_found) if len(lags_found) > 1 else 0

                # 해석
                if str(regime).upper() in ['BEAR', 'CRISIS']:
                    interp = f"{regime} 레짐: 시장 반응 가속화 경향 (위기 시 상관관계 증가)"
                else:
                    interp = f"{regime} 레짐: 정상적 시차 구조"

                results.append(RegimeConditionalLag(
                    regime=str(regime),
                    avg_lag_days=avg_lag,
                    lag_volatility=lag_vol,
                    sample_size=len(regime_data),
                    interpretation=interp
                ))
            except Exception:
                continue

        return results

    def _interpret_lag(self, actual_lag: int, expected_lag: int) -> str:
        """시차 해석"""
        if actual_lag < expected_lag * 0.5:
            return "시장이 예상보다 빠르게 반응 - 효율적 정보 반영"
        elif actual_lag > expected_lag * 1.5:
            return "시장이 예상보다 느리게 반응 - 정보 마찰 또는 비유동성"
        else:
            return "시장이 예상 범위 내에서 반응 - 정상적 전이 메커니즘"

    def _generate_actionable_insight(self, result: DynamicLagResult) -> str:
        """투자 시사점 생성"""
        insights = []

        if result.fastest_response_asset:
            insights.append(
                f"가장 빠른 반응: {result.fastest_response_asset} "
                f"(정책 변화 시 즉각적 포지션 조정 필요)"
            )

        if result.slowest_response_asset:
            insights.append(
                f"가장 느린 반응: {result.slowest_response_asset} "
                f"(정책 변화 후 지연된 기회 포착 가능)"
            )

        # 레짐별 시사점
        crisis_lags = [r for r in result.regime_lags if 'BEAR' in r.regime.upper() or 'CRISIS' in r.regime.upper()]
        if crisis_lags:
            avg_crisis_lag = np.mean([r.avg_lag_days for r in crisis_lags])
            insights.append(
                f"위기 시 평균 반응 시차: {avg_crisis_lag:.1f}일 "
                f"(Longin-Solnik 효과: 상관관계 증가로 분산 효과 감소)"
            )

        return " | ".join(insights) if insights else "분석 데이터 부족"


# ============================================================================
# Convenience Functions
# ============================================================================

def analyze_rrp_spy_relationship(
    rrp_data: pd.Series,
    spy_data: pd.Series,
    max_lag: int = 10
) -> Dict[str, Any]:
    """
    RRP와 SPY의 관계 빠른 분석

    Parameters:
    -----------
    rrp_data : Series
        RRP 시계열 (일간)
    spy_data : Series
        SPY 가격 시계열
    max_lag : int
        최대 래그

    Returns:
    --------
    Dict with analysis results
    """
    data = pd.DataFrame({
        'rrp': rrp_data,
        'spy': spy_data
    }).dropna()

    data['rrp_delta'] = data['rrp'].diff()
    data['spy_return'] = data['spy'].pct_change() * 100

    analyzer = LiquidityMarketAnalyzer(max_lag=max_lag)
    result = analyzer.analyze_liquidity_impact(data.dropna(), target='spy')

    return {
        'rrp_to_spy_lag': result.rrp_to_spy_lag,
        'rrp_to_spy_pvalue': result.rrp_to_spy_pvalue,
        'rrp_to_spy_significant': result.rrp_to_spy_significant,
        'key_drivers': result.key_drivers,
        'interpretation': result.interpretation,
        'trading_signal': result.trading_signal,
        'signal_confidence': result.signal_confidence
    }


def get_liquidity_regime_from_fred() -> Dict[str, Any]:
    """
    FRED에서 현재 유동성 레짐 가져오기

    Returns:
    --------
    Dict with liquidity regime information
    """
    try:
        collector = FREDCollector()
        summary = collector.collect_all()

        return {
            'rrp': summary.rrp,
            'rrp_delta': summary.rrp_delta,
            'tga': summary.tga,
            'fed_assets': summary.fed_assets,
            'net_liquidity': summary.net_liquidity,
            'liquidity_regime': summary.liquidity_regime,
            'timestamp': summary.timestamp
        }
    except Exception as e:
        return {'error': str(e)}


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Liquidity Analysis Module Test")
    print("=" * 60)

    # 1. 시뮬레이션 데이터로 테스트
    np.random.seed(42)
    n = 200

    # RRP → SPY 관계 시뮬레이션
    # RRP 감소 → 3일 후 SPY 상승
    rrp = 500 + np.random.randn(n).cumsum() * 10  # RRP in Billions
    spy = np.zeros(n)
    spy[0] = 450

    for i in range(3, n):
        # RRP 변화가 3일 후 SPY에 영향
        rrp_change = rrp[i-3] - rrp[i-4]
        spy[i] = spy[i-1] - 0.05 * rrp_change + np.random.randn() * 2

    data = pd.DataFrame({
        'rrp': rrp,
        'spy': spy,
        'tga': 800 + np.random.randn(n).cumsum() * 5,
        'fed_assets': 7.0 + np.random.randn(n).cumsum() * 0.01,
        'vix': 20 + np.abs(np.random.randn(n).cumsum() * 2)
    }, index=pd.date_range('2024-01-01', periods=n, freq='D'))

    print(f"\n[Simulated Data]")
    print(f"  Shape: {data.shape}")
    print(f"  Simulated: RRP change → 3 days → SPY change")

    # 2. 분석 실행
    analyzer = LiquidityMarketAnalyzer(max_lag=7, significance_level=0.10)
    prepared = analyzer.prepare_data(data)  # 이미 모든 데이터가 하나의 DataFrame에

    print(f"\n[Prepared Data]")
    print(f"  Columns: {list(prepared.columns)}")

    result = analyzer.analyze_liquidity_impact(prepared, target='spy')

    print(f"\n[Analysis Results]")
    print(f"  RRP → SPY: lag={result.rrp_to_spy_lag}, p={result.rrp_to_spy_pvalue:.4f}, sig={result.rrp_to_spy_significant}")
    print(f"  TGA → SPY: lag={result.tga_to_spy_lag}, p={result.tga_to_spy_pvalue:.4f}, sig={result.tga_to_spy_significant}")
    print(f"  Key Drivers: {result.key_drivers}")
    print(f"  Critical Path: {result.critical_path}")
    print(f"  Trading Signal: {result.trading_signal} (conf: {result.signal_confidence:.2f})")

    print(f"\n[Interpretation]")
    print(result.interpretation)

    # 3. 상관관계
    corrs = analyzer.calculate_correlations(prepared)
    print(f"\n[Top Correlations]")
    for c in corrs[:5]:
        print(f"  {c.variable1} ↔ {c.variable2}: {c.correlation:.3f} ({c.correlation_regime})")

    # 4. FRED 연동 테스트
    print(f"\n[FRED Liquidity Regime]")
    regime = get_liquidity_regime_from_fred()
    if 'error' not in regime:
        print(f"  RRP: ${regime['rrp']:.0f}B (delta: {regime['rrp_delta']:+.0f}B)")
        print(f"  TGA: ${regime['tga']:.0f}B")
        print(f"  Fed Assets: ${regime['fed_assets']:.2f}T")
        print(f"  Net Liquidity: ${regime['net_liquidity']/1000:.2f}T")
        print(f"  Regime: {regime['liquidity_regime']}")
    else:
        print(f"  Error: {regime['error']}")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
