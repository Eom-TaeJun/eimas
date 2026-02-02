#!/usr/bin/env python3
"""
Asset Universe Manager
======================

무한 자산(Infinite Assets) 개념을 반영한 자산 유니버스 관리

경제학적 근거:
1. 토큰화(Tokenization)로 자산 개수 N -> infinity 확장
   - 주식, 채권, 원자재, 크립토, 부동산 등 모든 자산이 토큰화 가능
   - 전통적 자산군 경계가 무너지고 있음

2. 자산 간 상관관계 분석의 중요성:
   - 디지털 금(BTC) vs 실물 금(Gold) 상관관계
   - 크립토와 나스닥 상관관계 증가 추세
   - 원자재(Copper)와 산업생산 연결

3. 분산투자의 재정의:
   - 단순히 종목 수가 아닌 상관관계 기반 분산
   - 낮은 상관관계 자산군 포함이 핵심

Usage:
    universe = AssetUniverseManager()
    data = universe.fetch_all_assets()
    corr_matrix = universe.calculate_correlation_matrix(data)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger('eimas.asset_universe')


# =============================================================================
# Enums & Data Classes
# =============================================================================

class AssetClass(Enum):
    """자산군 분류"""
    EQUITY_US = "us_equity"           # 미국 주식
    EQUITY_INTL = "intl_equity"       # 해외 주식
    FIXED_INCOME = "fixed_income"     # 채권
    COMMODITY = "commodity"           # 원자재
    CRYPTO = "crypto"                 # 암호화폐
    CURRENCY = "currency"             # 통화
    REAL_ESTATE = "real_estate"       # 부동산
    ALTERNATIVE = "alternative"       # 대체투자


class AssetSubClass(Enum):
    """자산 세부 분류"""
    # 주식
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"
    GROWTH = "growth"
    VALUE = "value"
    TECH = "tech"
    FINANCIALS = "financials"
    ENERGY = "energy"
    HEALTHCARE = "healthcare"

    # 채권
    TREASURY = "treasury"
    CORPORATE_IG = "corporate_ig"
    CORPORATE_HY = "high_yield"
    TIPS = "tips"
    MUNI = "municipal"

    # 원자재
    PRECIOUS_METALS = "precious_metals"
    INDUSTRIAL_METALS = "industrial_metals"
    ENERGY_COMMODITY = "energy_commodity"
    AGRICULTURE = "agriculture"

    # 크립토
    CRYPTO_MAJOR = "crypto_major"      # BTC, ETH
    CRYPTO_ALT = "crypto_alt"          # 알트코인
    CRYPTO_DEFI = "crypto_defi"        # DeFi 토큰
    STABLECOIN = "stablecoin"


@dataclass
class Asset:
    """
    자산 정의

    경제학적 의미:
    - ticker: 시장 식별자
    - asset_class: 전통적 자산군 분류
    - sub_class: 세부 분류 (분산 투자 분석용)
    - correlation_group: 상관관계 기반 그룹 (토큰화 시대의 새로운 분류)
    """
    ticker: str
    name: str
    asset_class: AssetClass
    sub_class: Optional[AssetSubClass] = None
    correlation_group: Optional[str] = None
    weight_in_benchmark: float = 0.0
    description: str = ""


@dataclass
class CorrelationResult:
    """
    상관관계 분석 결과

    경제학적 의미:
    - correlation_matrix: 자산 간 상관계수 행렬
    - rolling_correlations: 시간에 따른 상관관계 변화 (regime 변화 탐지)
    - cluster_assignments: 상관관계 기반 군집화 결과
    """
    correlation_matrix: pd.DataFrame
    rolling_correlations: Optional[Dict[str, pd.DataFrame]] = None
    cluster_assignments: Optional[Dict[str, int]] = None
    analysis_period: str = ""
    key_findings: List[str] = field(default_factory=list)


# =============================================================================
# Asset Universe Definition
# =============================================================================

# 확장된 자산 유니버스 (Infinite Assets 개념)
EXPANDED_ASSET_UNIVERSE: List[Asset] = [
    # === US Equity - Major Indices ===
    Asset("SPY", "S&P 500 ETF", AssetClass.EQUITY_US, AssetSubClass.LARGE_CAP,
          "risk_on", 0.30, "S&P 500 추종, 미국 대형주 대표"),
    Asset("QQQ", "Nasdaq 100 ETF", AssetClass.EQUITY_US, AssetSubClass.TECH,
          "tech_growth", 0.15, "나스닥 100, 기술주 집중"),
    Asset("IWM", "Russell 2000 ETF", AssetClass.EQUITY_US, AssetSubClass.SMALL_CAP,
          "risk_on", 0.05, "소형주 지수, 경기 민감"),
    Asset("DIA", "Dow Jones ETF", AssetClass.EQUITY_US, AssetSubClass.LARGE_CAP,
          "risk_on", 0.05, "다우존스 30, 우량주"),

    # === US Equity - Sectors ===
    Asset("XLF", "Financial Select Sector", AssetClass.EQUITY_US, AssetSubClass.FINANCIALS,
          "rate_sensitive", 0.03, "금융 섹터, 금리 민감"),
    Asset("XLE", "Energy Select Sector", AssetClass.EQUITY_US, AssetSubClass.ENERGY,
          "commodity_linked", 0.03, "에너지 섹터, 원유 연동"),
    Asset("XLK", "Technology Select Sector", AssetClass.EQUITY_US, AssetSubClass.TECH,
          "tech_growth", 0.05, "기술 섹터"),
    Asset("XLV", "Healthcare Select Sector", AssetClass.EQUITY_US, AssetSubClass.HEALTHCARE,
          "defensive", 0.03, "헬스케어 섹터, 방어적"),

    # === International Equity ===
    Asset("EFA", "MSCI EAFE ETF", AssetClass.EQUITY_INTL, None,
          "global_risk", 0.05, "선진국 해외 주식"),
    Asset("EEM", "Emerging Markets ETF", AssetClass.EQUITY_INTL, None,
          "global_risk", 0.03, "신흥국 주식"),
    Asset("FXI", "China Large-Cap ETF", AssetClass.EQUITY_INTL, None,
          "china", 0.02, "중국 대형주"),

    # === Fixed Income ===
    Asset("TLT", "20+ Year Treasury ETF", AssetClass.FIXED_INCOME, AssetSubClass.TREASURY,
          "safe_haven", 0.10, "장기 국채, 금리 역상관"),
    Asset("IEF", "7-10 Year Treasury ETF", AssetClass.FIXED_INCOME, AssetSubClass.TREASURY,
          "safe_haven", 0.05, "중기 국채"),
    Asset("SHY", "1-3 Year Treasury ETF", AssetClass.FIXED_INCOME, AssetSubClass.TREASURY,
          "cash_proxy", 0.03, "단기 국채, 현금 대용"),
    Asset("LQD", "Investment Grade Corporate", AssetClass.FIXED_INCOME, AssetSubClass.CORPORATE_IG,
          "credit", 0.03, "투자등급 회사채"),
    Asset("HYG", "High Yield Corporate", AssetClass.FIXED_INCOME, AssetSubClass.CORPORATE_HY,
          "credit", 0.02, "하이일드 채권, 신용 위험"),
    Asset("TIP", "TIPS ETF", AssetClass.FIXED_INCOME, AssetSubClass.TIPS,
          "inflation_hedge", 0.02, "물가연동채, 인플레이션 헤지"),

    # === Commodities - Precious Metals ===
    Asset("GLD", "Gold ETF", AssetClass.COMMODITY, AssetSubClass.PRECIOUS_METALS,
          "safe_haven", 0.05, "금, 안전자산/인플레 헤지"),
    Asset("SLV", "Silver ETF", AssetClass.COMMODITY, AssetSubClass.PRECIOUS_METALS,
          "precious_metals", 0.01, "은, 산업/투자 수요"),
    Asset("GC=F", "Gold Futures", AssetClass.COMMODITY, AssetSubClass.PRECIOUS_METALS,
          "safe_haven", 0.0, "금 선물"),

    # === Commodities - Industrial ===
    Asset("HG=F", "Copper Futures", AssetClass.COMMODITY, AssetSubClass.INDUSTRIAL_METALS,
          "economic_growth", 0.0, "구리 선물, 경기 선행지표 'Dr. Copper'"),
    Asset("COPX", "Copper Miners ETF", AssetClass.COMMODITY, AssetSubClass.INDUSTRIAL_METALS,
          "economic_growth", 0.01, "구리 채굴주"),

    # === Commodities - Energy ===
    Asset("USO", "US Oil Fund", AssetClass.COMMODITY, AssetSubClass.ENERGY_COMMODITY,
          "commodity_linked", 0.02, "원유 ETF"),
    Asset("CL=F", "WTI Crude Futures", AssetClass.COMMODITY, AssetSubClass.ENERGY_COMMODITY,
          "commodity_linked", 0.0, "WTI 원유 선물"),
    Asset("UNG", "Natural Gas ETF", AssetClass.COMMODITY, AssetSubClass.ENERGY_COMMODITY,
          "commodity_linked", 0.01, "천연가스"),

    # === Cryptocurrency - Major ===
    Asset("BTC-USD", "Bitcoin", AssetClass.CRYPTO, AssetSubClass.CRYPTO_MAJOR,
          "digital_gold", 0.03, "비트코인, 디지털 금"),
    Asset("ETH-USD", "Ethereum", AssetClass.CRYPTO, AssetSubClass.CRYPTO_MAJOR,
          "tech_growth", 0.02, "이더리움, 스마트 컨트랙트 플랫폼"),

    # === Cryptocurrency - Alt ===
    Asset("SOL-USD", "Solana", AssetClass.CRYPTO, AssetSubClass.CRYPTO_ALT,
          "tech_growth", 0.005, "솔라나, 고속 블록체인"),
    Asset("BNB-USD", "Binance Coin", AssetClass.CRYPTO, AssetSubClass.CRYPTO_ALT,
          "crypto_ecosystem", 0.005, "바이낸스 코인"),

    # === Currency ===
    Asset("UUP", "US Dollar Index ETF", AssetClass.CURRENCY, None,
          "dollar", 0.02, "달러 인덱스, 달러 강세시 상승"),
    Asset("FXE", "Euro ETF", AssetClass.CURRENCY, None,
          "dollar_inverse", 0.01, "유로/달러, 달러 역상관"),

    # === Real Estate ===
    Asset("VNQ", "Vanguard Real Estate ETF", AssetClass.REAL_ESTATE, None,
          "real_assets", 0.02, "미국 리츠, 부동산"),

    # === Volatility ===
    Asset("^VIX", "VIX Index", AssetClass.ALTERNATIVE, None,
          "volatility", 0.0, "변동성 지수, 공포 지표"),
]


# =============================================================================
# Asset Universe Manager
# =============================================================================

class AssetUniverseManager:
    """
    자산 유니버스 관리자

    기능:
    1. 확장된 자산 유니버스 정의 (Infinite Assets)
    2. 모든 자산 데이터 수집
    3. 상관관계 매트릭스 계산
    4. 상관관계 기반 자산 군집화
    """

    def __init__(
        self,
        custom_assets: List[Asset] = None,
        lookback_days: int = 365,
        verbose: bool = False
    ):
        """
        Args:
            custom_assets: 추가 자산 리스트 (기본 유니버스에 추가)
            lookback_days: 데이터 수집 기간 (일)
            verbose: 상세 로깅
        """
        self.assets = EXPANDED_ASSET_UNIVERSE.copy()
        if custom_assets:
            self.assets.extend(custom_assets)

        self.lookback_days = lookback_days
        self.verbose = verbose
        self._price_data: Optional[pd.DataFrame] = None
        self._returns_data: Optional[pd.DataFrame] = None

    def _log(self, msg: str):
        """로깅"""
        if self.verbose:
            logger.info(msg)
            print(f"[AssetUniverse] {msg}")

    def get_tickers(self, asset_class: AssetClass = None) -> List[str]:
        """
        티커 리스트 조회

        Args:
            asset_class: 특정 자산군만 조회 (None이면 전체)

        Returns:
            티커 리스트
        """
        if asset_class:
            return [a.ticker for a in self.assets if a.asset_class == asset_class]
        return [a.ticker for a in self.assets]

    def get_asset_info(self, ticker: str) -> Optional[Asset]:
        """자산 정보 조회"""
        for asset in self.assets:
            if asset.ticker == ticker:
                return asset
        return None

    def fetch_all_assets(
        self,
        tickers: List[str] = None,
        start_date: datetime = None,
        end_date: datetime = None
    ) -> pd.DataFrame:
        """
        모든 자산 가격 데이터 수집

        경제학적 의미:
        - 일별 종가 기준 수집
        - 자산 간 비교를 위해 동일 기간 데이터 사용

        Args:
            tickers: 수집할 티커 (None이면 전체)
            start_date: 시작일
            end_date: 종료일

        Returns:
            DataFrame with price data (columns = tickers)
        """
        if tickers is None:
            tickers = self.get_tickers()

        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.lookback_days)

        self._log(f"Fetching {len(tickers)} assets from {start_date.date()} to {end_date.date()}")

        price_data = {}
        failed_tickers = []

        for ticker in tickers:
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                if len(data) > 0:
                    price_data[ticker] = data['Close']
                    self._log(f"  {ticker}: {len(data)} rows")
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                self._log(f"  {ticker}: Failed - {e}")
                failed_tickers.append(ticker)

        if failed_tickers:
            self._log(f"Failed to fetch: {failed_tickers}")

        self._price_data = pd.DataFrame(price_data)
        return self._price_data

    def calculate_returns(
        self,
        price_data: pd.DataFrame = None,
        method: str = "log"
    ) -> pd.DataFrame:
        """
        수익률 계산

        경제학적 의미:
        - 로그 수익률: 연속 복리, 정규성 가정에 적합
        - 단순 수익률: 직관적, 포트폴리오 합산에 적합

        Args:
            price_data: 가격 데이터 (None이면 저장된 데이터 사용)
            method: "log" or "simple"

        Returns:
            수익률 DataFrame
        """
        if price_data is None:
            price_data = self._price_data

        if price_data is None:
            raise ValueError("No price data available. Call fetch_all_assets first.")

        if method == "log":
            returns = np.log(price_data / price_data.shift(1))
        else:
            returns = price_data.pct_change()

        self._returns_data = returns.dropna()
        return self._returns_data

    def calculate_correlation_matrix(
        self,
        returns_data: pd.DataFrame = None,
        method: str = "pearson",
        min_periods: int = 30
    ) -> CorrelationResult:
        """
        상관관계 매트릭스 계산

        경제학적 의미:
        - 자산 간 상관관계는 분산투자 효과의 핵심
        - 상관관계가 낮을수록 포트폴리오 위험 감소
        - Pearson: 선형 관계, Spearman: 순위 기반 (이상치에 강건)

        Args:
            returns_data: 수익률 데이터
            method: "pearson", "spearman", "kendall"
            min_periods: 상관계수 계산에 필요한 최소 기간

        Returns:
            CorrelationResult
        """
        if returns_data is None:
            returns_data = self._returns_data

        if returns_data is None:
            raise ValueError("No returns data. Call calculate_returns first.")

        self._log(f"Calculating {method} correlation matrix...")

        # 상관관계 매트릭스
        corr_matrix = returns_data.corr(method=method, min_periods=min_periods)

        # 주요 발견 사항 추출
        key_findings = self._extract_key_findings(corr_matrix)

        return CorrelationResult(
            correlation_matrix=corr_matrix,
            analysis_period=f"{returns_data.index[0].date()} to {returns_data.index[-1].date()}",
            key_findings=key_findings
        )

    def calculate_rolling_correlations(
        self,
        returns_data: pd.DataFrame = None,
        window: int = 60,
        pairs: List[Tuple[str, str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        롤링 상관관계 계산

        경제학적 의미:
        - 상관관계는 시간에 따라 변화 (regime 변화)
        - 위기 시 상관관계 급등 (correlation breakdown)
        - 롤링 분석으로 regime 변화 탐지

        Args:
            returns_data: 수익률 데이터
            window: 롤링 윈도우 (기본 60일)
            pairs: 분석할 자산 쌍 리스트 [(ticker1, ticker2), ...]

        Returns:
            {pair_name: rolling_correlation_series}
        """
        if returns_data is None:
            returns_data = self._returns_data

        if returns_data is None:
            raise ValueError("No returns data available.")

        # 기본 분석 쌍 정의
        if pairs is None:
            pairs = [
                ("SPY", "TLT"),           # 주식-채권
                ("SPY", "GLD"),           # 주식-금
                ("BTC-USD", "SPY"),       # 비트코인-주식
                ("BTC-USD", "GLD"),       # 비트코인-금 (디지털 금 가설)
                ("HG=F", "SPY"),          # 구리-주식 (경기 연동)
                ("QQQ", "BTC-USD"),       # 나스닥-비트코인
                ("TLT", "GLD"),           # 채권-금
            ]

        rolling_correlations = {}

        for ticker1, ticker2 in pairs:
            if ticker1 in returns_data.columns and ticker2 in returns_data.columns:
                pair_name = f"{ticker1}_vs_{ticker2}"
                rolling_corr = returns_data[ticker1].rolling(window).corr(returns_data[ticker2])
                rolling_correlations[pair_name] = rolling_corr

                self._log(f"  {pair_name}: latest corr = {rolling_corr.iloc[-1]:.3f}")

        return rolling_correlations

    def _extract_key_findings(self, corr_matrix: pd.DataFrame) -> List[str]:
        """상관관계 매트릭스에서 주요 발견 추출"""
        findings = []

        # 가장 높은 상관관계 쌍
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        max_corr = upper_tri.max().max()
        max_pair = upper_tri.stack().idxmax()
        findings.append(f"Highest correlation: {max_pair[0]} & {max_pair[1]} ({max_corr:.3f})")

        # 가장 낮은 상관관계 쌍 (분산투자 기회)
        min_corr = upper_tri.min().min()
        min_pair = upper_tri.stack().idxmin()
        findings.append(f"Lowest correlation: {min_pair[0]} & {min_pair[1]} ({min_corr:.3f})")

        # BTC-Gold 상관관계 (디지털 금 가설)
        if "BTC-USD" in corr_matrix.columns and "GLD" in corr_matrix.columns:
            btc_gold_corr = corr_matrix.loc["BTC-USD", "GLD"]
            interpretation = "supports" if btc_gold_corr > 0.3 else "challenges"
            findings.append(f"BTC-Gold correlation: {btc_gold_corr:.3f} ({interpretation} digital gold thesis)")

        # BTC-SPY 상관관계 (Risk asset 가설)
        if "BTC-USD" in corr_matrix.columns and "SPY" in corr_matrix.columns:
            btc_spy_corr = corr_matrix.loc["BTC-USD", "SPY"]
            findings.append(f"BTC-SPY correlation: {btc_spy_corr:.3f} (crypto-equity linkage)")

        # Copper-SPY 상관관계 (경기 연동)
        if "HG=F" in corr_matrix.columns and "SPY" in corr_matrix.columns:
            copper_spy_corr = corr_matrix.loc["HG=F", "SPY"]
            findings.append(f"Copper-SPY correlation: {copper_spy_corr:.3f} (economic cycle proxy)")

        return findings

    def get_correlation_summary(self, corr_result: CorrelationResult) -> str:
        """
        상관관계 분석 요약 리포트 생성

        Returns:
            요약 문자열
        """
        corr_matrix = corr_result.correlation_matrix

        lines = []
        lines.append("=" * 60)
        lines.append("CORRELATION ANALYSIS SUMMARY")
        lines.append("(Infinite Assets Cross-Correlation)")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Analysis Period: {corr_result.analysis_period}")
        lines.append(f"Assets Analyzed: {len(corr_matrix.columns)}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("KEY FINDINGS")
        lines.append("-" * 60)
        for finding in corr_result.key_findings:
            lines.append(f"  * {finding}")
        lines.append("")

        # 자산군별 평균 상관관계
        lines.append("-" * 60)
        lines.append("CROSS-ASSET CLASS CORRELATIONS")
        lines.append("-" * 60)

        # 주요 자산군 대표 티커
        representatives = {
            "US Equity": "SPY",
            "Tech": "QQQ",
            "Treasury": "TLT",
            "Gold": "GLD",
            "Bitcoin": "BTC-USD",
            "Copper": "HG=F",
        }

        lines.append("")
        lines.append("| Asset 1     | Asset 2     | Correlation |")
        lines.append("|-------------|-------------|-------------|")

        for name1, ticker1 in representatives.items():
            for name2, ticker2 in representatives.items():
                if ticker1 != ticker2 and ticker1 in corr_matrix.columns and ticker2 in corr_matrix.columns:
                    if list(representatives.keys()).index(name1) < list(representatives.keys()).index(name2):
                        corr = corr_matrix.loc[ticker1, ticker2]
                        lines.append(f"| {name1:<11} | {name2:<11} | {corr:+.3f}       |")

        return "\n".join(lines)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing AssetUniverseManager...")
    print()

    # 관리자 생성
    manager = AssetUniverseManager(lookback_days=180, verbose=True)

    # 주요 자산만 테스트
    test_tickers = [
        "SPY", "QQQ", "TLT", "GLD", "BTC-USD", "ETH-USD", "HG=F", "GC=F"
    ]

    # 데이터 수집
    print("\n1. Fetching asset data...")
    price_data = manager.fetch_all_assets(tickers=test_tickers)
    print(f"   Fetched {len(price_data.columns)} assets, {len(price_data)} days")

    # 수익률 계산
    print("\n2. Calculating returns...")
    returns = manager.calculate_returns(method="log")
    print(f"   Returns shape: {returns.shape}")

    # 상관관계 계산
    print("\n3. Calculating correlation matrix...")
    corr_result = manager.calculate_correlation_matrix()
    print(corr_result.correlation_matrix.round(3))

    # 롤링 상관관계
    print("\n4. Calculating rolling correlations...")
    rolling = manager.calculate_rolling_correlations(window=30)
    for pair, data in rolling.items():
        print(f"   {pair}: latest = {data.iloc[-1]:.3f}")

    # 요약 리포트
    print("\n5. Correlation Summary:")
    print(manager.get_correlation_summary(corr_result))
