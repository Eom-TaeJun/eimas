#!/usr/bin/env python3
"""
Macro Liquidity Analyzer
========================

현대 화폐 이론(Genius Act)과 확장 유동성 분석

경제학적 근거:
1. 현대 통화량 공식: M = B + S * B*
   - B: 기본 유동성 (Fed B/S - TGA - RRP)
   - S: 스테이블코인 승수
   - B*: 민간/프로토콜 파생 유동성 (USDT, USDC 등)

2. Genius Act (스테이블코인 규제법):
   - 스테이블코인 발행자는 단기 국채를 담보로 보유해야 함
   - USDT/USDC 발행량 증가 -> 단기 국채 수요 증가 -> 금리 하락 압력
   - 결과: 위험자산 선호(Risk-On) 환경 조성

3. 유동성 사이클:
   - Fed QE + 스테이블코인 확장 -> 강력한 Risk-On
   - Fed QT + 스테이블코인 수축 -> 강력한 Risk-Off

Usage:
    analyzer = MacroLiquidityAnalyzer()
    result = analyzer.analyze()
    print(result.commentary)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import requests
import os

logger = logging.getLogger('eimas.macro_analyzer')


# =============================================================================
# Enums & Data Classes
# =============================================================================

class LiquidityRegime(Enum):
    """유동성 레짐 분류"""
    STRONG_EXPANSION = "strong_expansion"     # 강한 확장 (M > 10% YoY)
    EXPANSION = "expansion"                   # 확장 (M > 5% YoY)
    NEUTRAL = "neutral"                       # 중립 (-5% < M < 5%)
    CONTRACTION = "contraction"               # 수축 (M < -5% YoY)
    STRONG_CONTRACTION = "strong_contraction" # 강한 수축 (M < -10% YoY)


class GeniusActSignal(Enum):
    """Genius Act 신호"""
    ACTIVE_EXPANSION = "genius_act_active"      # 스테이블코인 급증 -> Risk-On
    MODERATE_SUPPORT = "moderate_support"       # 완만한 지원
    NEUTRAL = "neutral"                         # 중립
    WEAKENING = "weakening"                     # 약화
    REVERSAL = "reversal"                       # 역전 (Risk-Off)


@dataclass
class StablecoinMetrics:
    """
    스테이블코인 지표

    경제학적 의미:
    - total_supply: 민간 파생 유동성의 총량 (B* in M = B + S*B*)
    - usdt_supply: 가장 큰 스테이블코인, 오프쇼어 달러 수요 대리변수
    - usdc_supply: 규제 준수 스테이블코인, 미국 내 기관 수요
    - supply_change_7d: 단기 유동성 추세
    - supply_change_30d: 중기 유동성 추세
    """
    usdt_supply: float = 0.0          # USDT 시가총액 (십억 달러)
    usdc_supply: float = 0.0          # USDC 시가총액 (십억 달러)
    dai_supply: float = 0.0           # DAI 시가총액 (십억 달러)
    usds_supply: float = 0.0          # USDS(Sky) 시가총액
    total_supply: float = 0.0         # 총 스테이블코인 공급
    supply_change_7d: float = 0.0     # 7일 변화율 (%)
    supply_change_30d: float = 0.0    # 30일 변화율 (%)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """총 공급량 자동 계산"""
        if self.total_supply == 0:
            self.total_supply = self.usdt_supply + self.usdc_supply + self.dai_supply + self.usds_supply


@dataclass
class FedLiquidityMetrics:
    """
    연준 유동성 지표

    경제학적 의미:
    - fed_balance_sheet: 연준 총 자산 (기본 통화 공급)
    - rrp_balance: 역레포 잔액 (시스템에서 제거된 유동성)
    - tga_balance: 재무부 일반계정 (정부가 보유한 현금)
    - net_liquidity: 순 유동성 = Fed B/S - RRP - TGA
    """
    fed_balance_sheet: float = 0.0    # Fed 총 자산 (조 달러)
    rrp_balance: float = 0.0          # 역레포 잔액 (조 달러)
    tga_balance: float = 0.0          # TGA 잔액 (조 달러)
    net_liquidity: float = 0.0        # 순 유동성 (조 달러)
    rrp_delta_1w: float = 0.0         # RRP 1주 변화
    tga_delta_1w: float = 0.0         # TGA 1주 변화
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """순 유동성 자동 계산"""
        if self.net_liquidity == 0:
            self.net_liquidity = self.fed_balance_sheet - self.rrp_balance - self.tga_balance


@dataclass
class ExtendedLiquidityResult:
    """
    확장 유동성 분석 결과

    M = B + S * B* (현대 화폐 이론 공식)

    B: base_liquidity (전통적 순유동성)
    S: stablecoin_multiplier (스테이블코인 승수, 담보비율 역수)
    B*: stablecoin_derived_liquidity (스테이블코인 기반 파생 유동성)
    M: total_effective_liquidity (총 유효 유동성)
    """
    # 기본 유동성 (B)
    base_liquidity: float = 0.0

    # 스테이블코인 승수 (S)
    stablecoin_multiplier: float = 1.0

    # 스테이블코인 파생 유동성 (B*)
    stablecoin_derived_liquidity: float = 0.0

    # 총 유효 유동성 (M = B + S*B*)
    total_effective_liquidity: float = 0.0

    # 유동성 레짐
    regime: LiquidityRegime = LiquidityRegime.NEUTRAL

    # Genius Act 신호
    genius_act_signal: GeniusActSignal = GeniusActSignal.NEUTRAL

    # 변화율
    liquidity_change_1m: float = 0.0   # 1개월 변화율 (%)
    liquidity_change_3m: float = 0.0   # 3개월 변화율 (%)

    # 시장 해석
    market_implication: str = ""
    risk_appetite: str = "NEUTRAL"     # RISK_ON, NEUTRAL, RISK_OFF

    # 상세 코멘터리
    commentary: str = ""

    # 개별 지표
    fed_metrics: Optional[FedLiquidityMetrics] = None
    stablecoin_metrics: Optional[StablecoinMetrics] = None

    # 타임스탬프
    timestamp: datetime = field(default_factory=datetime.now)


# =============================================================================
# Stablecoin Data Fetcher
# =============================================================================

class StablecoinDataFetcher:
    """
    스테이블코인 데이터 수집기

    데이터 소스:
    - CoinGecko API (무료)
    - DefiLlama API (대안)
    """

    COINGECKO_API = "https://api.coingecko.com/api/v3"
    DEFILLAMA_API = "https://stablecoins.llama.fi"

    # 주요 스테이블코인 ID (CoinGecko)
    STABLECOIN_IDS = {
        'usdt': 'tether',
        'usdc': 'usd-coin',
        'dai': 'dai',
        'usds': 'usds',
        'busd': 'binance-usd',
        'tusd': 'true-usd',
        'frax': 'frax',
    }

    def __init__(self, use_cache: bool = True, cache_ttl_minutes: int = 60):
        """
        Args:
            use_cache: 캐시 사용 여부
            cache_ttl_minutes: 캐시 유효 시간 (분)
        """
        self.use_cache = use_cache
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

    def _get_cached(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        if not self.use_cache or key not in self._cache:
            return None
        data, cached_at = self._cache[key]
        if datetime.now() - cached_at > self.cache_ttl:
            del self._cache[key]
            return None
        return data

    def _set_cache(self, key: str, data: Any):
        """캐시에 데이터 저장"""
        if self.use_cache:
            self._cache[key] = (data, datetime.now())

    def fetch_from_coingecko(self) -> StablecoinMetrics:
        """
        CoinGecko API로 스테이블코인 시가총액 조회

        Returns:
            StablecoinMetrics: 스테이블코인 지표
        """
        cache_key = "coingecko_stablecoins"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            # 주요 스테이블코인 시가총액 조회
            ids = ','.join(self.STABLECOIN_IDS.values())
            url = f"{self.COINGECKO_API}/simple/price"
            params = {
                'ids': ids,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_change': 'true'
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # 시가총액 추출 (십억 달러 단위)
            usdt_mcap = data.get('tether', {}).get('usd_market_cap', 0) / 1e9
            usdc_mcap = data.get('usd-coin', {}).get('usd_market_cap', 0) / 1e9
            dai_mcap = data.get('dai', {}).get('usd_market_cap', 0) / 1e9
            usds_mcap = data.get('usds', {}).get('usd_market_cap', 0) / 1e9

            metrics = StablecoinMetrics(
                usdt_supply=usdt_mcap,
                usdc_supply=usdc_mcap,
                dai_supply=dai_mcap,
                usds_supply=usds_mcap,
            )

            self._set_cache(cache_key, metrics)
            return metrics

        except Exception as e:
            logger.warning(f"CoinGecko API error: {e}, using defaults")
            # 2024년 기준 대략적인 기본값
            return StablecoinMetrics(
                usdt_supply=120.0,   # ~$120B
                usdc_supply=35.0,    # ~$35B
                dai_supply=5.0,      # ~$5B
                usds_supply=2.0,     # ~$2B
            )

    def fetch_from_defillama(self) -> StablecoinMetrics:
        """
        DefiLlama API로 스테이블코인 데이터 조회 (대안)

        Returns:
            StablecoinMetrics: 스테이블코인 지표
        """
        cache_key = "defillama_stablecoins"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            url = f"{self.DEFILLAMA_API}/stablecoins?includePrices=true"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            stablecoins = data.get('peggedAssets', [])

            usdt_supply = 0
            usdc_supply = 0
            dai_supply = 0
            usds_supply = 0

            for coin in stablecoins:
                symbol = coin.get('symbol', '').upper()
                # circulating은 dict 형태: {'peggedUSD': value}
                circulating = coin.get('circulating', {})
                supply = circulating.get('peggedUSD', 0) / 1e9  # 십억 달러

                if symbol == 'USDT':
                    usdt_supply = supply
                elif symbol == 'USDC':
                    usdc_supply = supply
                elif symbol == 'DAI':
                    dai_supply = supply
                elif symbol == 'USDS':
                    usds_supply = supply

            metrics = StablecoinMetrics(
                usdt_supply=usdt_supply,
                usdc_supply=usdc_supply,
                dai_supply=dai_supply,
                usds_supply=usds_supply,
            )

            self._set_cache(cache_key, metrics)
            return metrics

        except Exception as e:
            logger.warning(f"DefiLlama API error: {e}, falling back to CoinGecko")
            return self.fetch_from_coingecko()

    def fetch(self) -> StablecoinMetrics:
        """
        스테이블코인 데이터 조회 (자동 폴백)

        Returns:
            StablecoinMetrics: 스테이블코인 지표
        """
        try:
            return self.fetch_from_defillama()
        except Exception:
            return self.fetch_from_coingecko()


# =============================================================================
# Macro Liquidity Analyzer
# =============================================================================

class MacroLiquidityAnalyzer:
    """
    매크로 유동성 분석기

    핵심 개념:
    1. 확장 유동성 공식: M = B + S * B*
       - B: 기본 유동성 (Fed B/S - TGA - RRP)
       - S: 스테이블코인 승수 (평균 담보비율 역수, ~1.1-1.3)
       - B*: 스테이블코인 시가총액 (USDT + USDC + DAI + ...)

    2. Genius Act 해석:
       - 스테이블코인 발행량 증가 -> 단기 국채 담보 수요 -> 금리 하락 압력
       - 이는 위험자산에 유리한 환경 조성

    3. 유동성 레짐 분류:
       - M 성장률 > 10%: 강한 확장 (강력한 Risk-On)
       - 5% < M < 10%: 확장 (Risk-On)
       - -5% < M < 5%: 중립
       - M < -5%: 수축 (Risk-Off)
    """

    # 스테이블코인 승수 (담보비율 역수 추정)
    # 대부분의 스테이블코인은 100-110% 담보 -> 승수 ~0.9-1.0
    # 부분지급준비 효과 고려 -> 실질 승수 ~1.1-1.3
    DEFAULT_STABLECOIN_MULTIPLIER = 1.15

    def __init__(
        self,
        stablecoin_multiplier: float = None,
        include_stablecoins: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            stablecoin_multiplier: 스테이블코인 승수 (기본값: 1.15)
            include_stablecoins: 확장 유동성 공식에 스테이블코인 포함 여부
            verbose: 상세 로깅
        """
        self.stablecoin_multiplier = stablecoin_multiplier or self.DEFAULT_STABLECOIN_MULTIPLIER
        self.include_stablecoins = include_stablecoins
        self.verbose = verbose

        self.stablecoin_fetcher = StablecoinDataFetcher()
        self._historical_liquidity: List[float] = []

    def _log(self, msg: str):
        """로깅"""
        if self.verbose:
            logger.info(msg)
            print(f"[MacroLiquidity] {msg}")

    def analyze(
        self,
        fed_metrics: Optional[FedLiquidityMetrics] = None,
        stablecoin_metrics: Optional[StablecoinMetrics] = None,
        previous_liquidity: Optional[float] = None
    ) -> ExtendedLiquidityResult:
        """
        확장 유동성 분석 수행

        경제학적 근거:
        M = B + S * B* 공식을 적용하여 총 유효 유동성 계산

        Args:
            fed_metrics: 연준 유동성 지표 (없으면 기본값 사용)
            stablecoin_metrics: 스테이블코인 지표 (없으면 API로 조회)
            previous_liquidity: 이전 기간 유동성 (변화율 계산용)

        Returns:
            ExtendedLiquidityResult: 분석 결과
        """
        self._log("Starting extended liquidity analysis...")

        # 1. 연준 유동성 지표 (기본값 또는 입력값)
        if fed_metrics is None:
            # 2024년 말 기준 대략적인 기본값 (조 달러)
            fed_metrics = FedLiquidityMetrics(
                fed_balance_sheet=7.0,  # ~$7T
                rrp_balance=0.5,        # ~$500B (크게 감소한 상태)
                tga_balance=0.7,        # ~$700B
            )

        # 2. 스테이블코인 지표
        if stablecoin_metrics is None:
            stablecoin_metrics = self.stablecoin_fetcher.fetch()

        # 3. 기본 유동성 계산 (B)
        base_liquidity = fed_metrics.net_liquidity
        if base_liquidity == 0:
            base_liquidity = (
                fed_metrics.fed_balance_sheet -
                fed_metrics.rrp_balance -
                fed_metrics.tga_balance
            )

        self._log(f"Base Liquidity (B): ${base_liquidity:.2f}T")

        # 4. 스테이블코인 파생 유동성 계산 (B*)
        stablecoin_derived = stablecoin_metrics.total_supply / 1000  # 십억 -> 조 달러

        self._log(f"Stablecoin Derived (B*): ${stablecoin_derived:.3f}T")
        self._log(f"  - USDT: ${stablecoin_metrics.usdt_supply:.1f}B")
        self._log(f"  - USDC: ${stablecoin_metrics.usdc_supply:.1f}B")

        # 5. 총 유효 유동성 계산 (M = B + S * B*)
        if self.include_stablecoins:
            total_liquidity = base_liquidity + (self.stablecoin_multiplier * stablecoin_derived)
        else:
            total_liquidity = base_liquidity

        self._log(f"Total Effective Liquidity (M): ${total_liquidity:.2f}T")

        # 6. 변화율 계산
        liquidity_change = 0.0
        if previous_liquidity and previous_liquidity > 0:
            liquidity_change = ((total_liquidity - previous_liquidity) / previous_liquidity) * 100

        # 7. 유동성 레짐 분류
        regime = self._classify_regime(liquidity_change, stablecoin_metrics)

        # 8. Genius Act 신호 분석
        genius_act_signal = self._analyze_genius_act(stablecoin_metrics, fed_metrics)

        # 9. 시장 함의 생성
        market_implication, risk_appetite = self._generate_market_implication(
            regime, genius_act_signal, stablecoin_metrics
        )

        # 10. 상세 코멘터리 생성
        commentary = self._generate_commentary(
            base_liquidity,
            stablecoin_derived,
            total_liquidity,
            regime,
            genius_act_signal,
            stablecoin_metrics,
            fed_metrics
        )

        return ExtendedLiquidityResult(
            base_liquidity=base_liquidity,
            stablecoin_multiplier=self.stablecoin_multiplier,
            stablecoin_derived_liquidity=stablecoin_derived,
            total_effective_liquidity=total_liquidity,
            regime=regime,
            genius_act_signal=genius_act_signal,
            liquidity_change_1m=liquidity_change,
            market_implication=market_implication,
            risk_appetite=risk_appetite,
            commentary=commentary,
            fed_metrics=fed_metrics,
            stablecoin_metrics=stablecoin_metrics,
        )

    def _classify_regime(
        self,
        liquidity_change: float,
        stablecoin_metrics: StablecoinMetrics
    ) -> LiquidityRegime:
        """
        유동성 레짐 분류

        경제학적 근거:
        - 유동성 성장률이 위험자산 수익률의 선행지표
        - 5% 이상 성장 시 Risk-On 환경
        """
        # 스테이블코인 성장률도 고려
        sc_growth = stablecoin_metrics.supply_change_30d

        # 복합 점수 계산 (유동성 변화 60%, 스테이블코인 성장 40%)
        composite_score = liquidity_change * 0.6 + sc_growth * 0.4

        if composite_score > 8:
            return LiquidityRegime.STRONG_EXPANSION
        elif composite_score > 3:
            return LiquidityRegime.EXPANSION
        elif composite_score > -3:
            return LiquidityRegime.NEUTRAL
        elif composite_score > -8:
            return LiquidityRegime.CONTRACTION
        else:
            return LiquidityRegime.STRONG_CONTRACTION

    def _analyze_genius_act(
        self,
        stablecoin_metrics: StablecoinMetrics,
        fed_metrics: FedLiquidityMetrics
    ) -> GeniusActSignal:
        """
        Genius Act 신호 분석

        경제학적 근거:
        - Genius Act: 스테이블코인 발행자는 단기 국채를 담보로 보유
        - 스테이블코인 증가 -> 국채 수요 증가 -> 금리 하락 -> Risk-On
        - RRP 감소와 동시에 스테이블코인 증가 = 최적의 유동성 환경
        """
        # 스테이블코인 성장률
        sc_growth_7d = stablecoin_metrics.supply_change_7d
        sc_growth_30d = stablecoin_metrics.supply_change_30d

        # RRP 감소 여부 (감소 = 유동성 공급)
        rrp_draining = fed_metrics.rrp_delta_1w < 0

        # Genius Act 활성화 조건
        # 1. 스테이블코인 7일 성장률 > 1%
        # 2. 또는 30일 성장률 > 3%
        # 3. RRP 감소와 동시에 발생하면 더 강한 신호

        if sc_growth_7d > 2 or (sc_growth_30d > 5 and rrp_draining):
            return GeniusActSignal.ACTIVE_EXPANSION
        elif sc_growth_7d > 0.5 or sc_growth_30d > 2:
            return GeniusActSignal.MODERATE_SUPPORT
        elif sc_growth_7d > -0.5 and sc_growth_30d > -2:
            return GeniusActSignal.NEUTRAL
        elif sc_growth_7d > -2 or sc_growth_30d > -5:
            return GeniusActSignal.WEAKENING
        else:
            return GeniusActSignal.REVERSAL

    def _generate_market_implication(
        self,
        regime: LiquidityRegime,
        genius_act_signal: GeniusActSignal,
        stablecoin_metrics: StablecoinMetrics
    ) -> Tuple[str, str]:
        """
        시장 함의 생성

        Returns:
            Tuple[market_implication, risk_appetite]
        """
        # Risk appetite 결정
        if regime in [LiquidityRegime.STRONG_EXPANSION, LiquidityRegime.EXPANSION]:
            if genius_act_signal == GeniusActSignal.ACTIVE_EXPANSION:
                risk_appetite = "STRONG_RISK_ON"
            else:
                risk_appetite = "RISK_ON"
        elif regime == LiquidityRegime.NEUTRAL:
            risk_appetite = "NEUTRAL"
        else:
            if genius_act_signal == GeniusActSignal.REVERSAL:
                risk_appetite = "STRONG_RISK_OFF"
            else:
                risk_appetite = "RISK_OFF"

        # 시장 함의 텍스트
        implications = {
            "STRONG_RISK_ON": "강력한 유동성 확장. 위험자산 적극 선호. 크립토/성장주 강세 예상.",
            "RISK_ON": "유동성 확장 국면. 위험자산 선호. 주식/크립토 긍정적.",
            "NEUTRAL": "유동성 중립. 섹터 로테이션 및 종목 선별 필요.",
            "RISK_OFF": "유동성 수축. 안전자산 선호. 채권/금 강세 가능.",
            "STRONG_RISK_OFF": "강력한 유동성 수축. 현금 보유 권장. 위험자산 회피.",
        }

        return implications.get(risk_appetite, ""), risk_appetite

    def _generate_commentary(
        self,
        base_liquidity: float,
        stablecoin_derived: float,
        total_liquidity: float,
        regime: LiquidityRegime,
        genius_act_signal: GeniusActSignal,
        stablecoin_metrics: StablecoinMetrics,
        fed_metrics: FedLiquidityMetrics
    ) -> str:
        """
        상세 코멘터리 생성

        경제학적 해석을 포함한 시장 분석 코멘터리
        """
        lines = []

        # 1. 유동성 요약
        lines.append("=" * 50)
        lines.append("EXTENDED LIQUIDITY ANALYSIS (M = B + S*B*)")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"Base Liquidity (B): ${base_liquidity:.2f}T")
        lines.append(f"  - Fed B/S: ${fed_metrics.fed_balance_sheet:.2f}T")
        lines.append(f"  - RRP: ${fed_metrics.rrp_balance:.2f}T")
        lines.append(f"  - TGA: ${fed_metrics.tga_balance:.2f}T")
        lines.append("")
        lines.append(f"Stablecoin Derived (B*): ${stablecoin_derived:.3f}T")
        lines.append(f"  - USDT: ${stablecoin_metrics.usdt_supply:.1f}B")
        lines.append(f"  - USDC: ${stablecoin_metrics.usdc_supply:.1f}B")
        lines.append(f"  - DAI: ${stablecoin_metrics.dai_supply:.1f}B")
        lines.append("")
        lines.append(f"Stablecoin Multiplier (S): {self.stablecoin_multiplier:.2f}")
        lines.append(f"Total Effective Liquidity (M): ${total_liquidity:.2f}T")
        lines.append("")

        # 2. Genius Act 해석
        lines.append("-" * 50)
        lines.append("GENIUS ACT INTERPRETATION")
        lines.append("-" * 50)

        if genius_act_signal == GeniusActSignal.ACTIVE_EXPANSION:
            lines.append("")
            lines.append("[GENIUS ACT ACTIVE]")
            lines.append("Genius Act 작동 중: 스테이블코인 발행 급증")
            lines.append("- 달러 유동성 확장 및 국채 수요 증가")
            lines.append("- 단기 금리 하락 압력으로 위험자산 선호(Risk-On) 환경")
            lines.append("- 크립토, 성장주, 기술주에 긍정적")
        elif genius_act_signal == GeniusActSignal.MODERATE_SUPPORT:
            lines.append("")
            lines.append("[MODERATE SUPPORT]")
            lines.append("스테이블코인 완만한 증가세")
            lines.append("- 유동성 여건 점진적 개선")
            lines.append("- 위험자산에 완만한 지원")
        elif genius_act_signal == GeniusActSignal.NEUTRAL:
            lines.append("")
            lines.append("[NEUTRAL]")
            lines.append("스테이블코인 공급 안정")
            lines.append("- 유동성 방향성 중립")
        elif genius_act_signal == GeniusActSignal.WEAKENING:
            lines.append("")
            lines.append("[WEAKENING]")
            lines.append("스테이블코인 공급 감소 추세")
            lines.append("- 달러 유동성 수축 신호")
            lines.append("- 위험자산 주의 필요")
        else:
            lines.append("")
            lines.append("[REVERSAL - RISK OFF]")
            lines.append("스테이블코인 급격한 감소")
            lines.append("- 심각한 유동성 수축")
            lines.append("- 디레버리징 위험, 안전자산 선호")

        lines.append("")

        # 3. 레짐 해석
        lines.append("-" * 50)
        lines.append(f"LIQUIDITY REGIME: {regime.value.upper()}")
        lines.append("-" * 50)

        regime_interpretations = {
            LiquidityRegime.STRONG_EXPANSION:
                "강력한 유동성 확장 국면. 위험자산 적극 매수 고려.",
            LiquidityRegime.EXPANSION:
                "유동성 확장 국면. 위험자산 비중 확대 권장.",
            LiquidityRegime.NEUTRAL:
                "유동성 중립. 선별적 투자 및 분산 권장.",
            LiquidityRegime.CONTRACTION:
                "유동성 수축 국면. 방어적 포지션 권장.",
            LiquidityRegime.STRONG_CONTRACTION:
                "강력한 유동성 수축. 현금 비중 확대 및 위험 회피.",
        }

        lines.append(regime_interpretations.get(regime, ""))

        return "\n".join(lines)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import asyncio

    print("Testing MacroLiquidityAnalyzer...")
    print()

    # 분석기 생성
    analyzer = MacroLiquidityAnalyzer(verbose=True)

    # 테스트용 Fed 지표 (2024년 말 기준 가정)
    fed_metrics = FedLiquidityMetrics(
        fed_balance_sheet=7.0,   # $7T
        rrp_balance=0.4,         # $400B (크게 감소)
        tga_balance=0.7,         # $700B
        rrp_delta_1w=-0.05,      # -$50B (감소 중)
    )

    # 테스트용 스테이블코인 지표
    stablecoin_metrics = StablecoinMetrics(
        usdt_supply=120.0,       # $120B
        usdc_supply=35.0,        # $35B
        dai_supply=5.0,          # $5B
        supply_change_7d=1.5,    # +1.5%
        supply_change_30d=4.0,   # +4.0%
    )

    # 분석 실행
    result = analyzer.analyze(
        fed_metrics=fed_metrics,
        stablecoin_metrics=stablecoin_metrics,
        previous_liquidity=5.8  # 이전 분기 유동성
    )

    print()
    print(result.commentary)
    print()
    print(f"Risk Appetite: {result.risk_appetite}")
    print(f"Market Implication: {result.market_implication}")
