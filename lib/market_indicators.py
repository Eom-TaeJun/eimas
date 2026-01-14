#!/usr/bin/env python3
"""
Market Indicators Collector
============================
시장 밸류에이션, 24/7 자산, 크레딧 스프레드 등 추가 지표 수집

지표 카테고리:
1. 밸류에이션 (Shiller CAPE, Buffett Indicator, ERP)
2. 24/7 자산 (BTC, ETH, 주요 FX)
3. 크레딧 스프레드 (HY-IG, HY-Treasury)
4. VIX Term Structure (Contango/Backwardation)
5. 금리/수익률 곡선 (2Y-10Y Spread)

데이터 소스:
- yfinance: ETF 가격 기반 계산
- FRED API: 거시 지표 (CAPE, GDP 등)
- 계산: 스프레드, 비율 등
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import warnings
import requests

warnings.filterwarnings('ignore')

from core.database import DatabaseManager


# ============================================================================
# Constants
# ============================================================================

# VIX 관련 ETF/ETN
VIX_TICKERS = {
    'VIX': '^VIX',           # VIX Index
    'VIX3M': '^VIX3M',       # 3-Month VIX
    'VVIX': '^VVIX',         # VIX of VIX
    'VXX': 'VXX',            # Short-term VIX futures ETN
    'UVXY': 'UVXY',          # 1.5x VIX Short-term
    'SVXY': 'SVXY',          # -0.5x VIX Short-term
}

# 24/7 자산
CRYPTO_TICKERS = {
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD',
    'SOL': 'SOL-USD',
}

# FX
FX_TICKERS = {
    'USDJPY': 'JPY=X',       # USD/JPY
    'EURUSD': 'EURUSD=X',    # EUR/USD
    'AUDUSD': 'AUDUSD=X',    # AUD/USD
    'DXY': 'DX-Y.NYB',       # Dollar Index
}

# 크레딧 스프레드 관련
CREDIT_TICKERS = {
    'HYG': 'HYG',            # High Yield Corporate Bond
    'LQD': 'LQD',            # Investment Grade Corporate Bond
    'TLT': 'TLT',            # 20+ Year Treasury
    'IEF': 'IEF',            # 7-10 Year Treasury
    'SHY': 'SHY',            # 1-3 Year Treasury
    'JNK': 'JNK',            # High Yield Bond
}

# 밸류에이션 관련
VALUATION_TICKERS = {
    'SPY': 'SPY',            # S&P 500
    'VTI': 'VTI',            # Total Stock Market
}

# FRED 시리즈 ID
FRED_SERIES = {
    'GDP': 'GDP',                    # Nominal GDP
    'CAPE': 'MULTPL/SHILLER_PE_RATIO_MONTH',  # (별도 소스 필요)
    'T10Y2Y': 'T10Y2Y',             # 10Y-2Y Treasury Spread
    'T10YIE': 'T10YIE',             # 10Y Breakeven Inflation
    'DGS10': 'DGS10',               # 10Y Treasury Rate
    'DGS2': 'DGS2',                 # 2Y Treasury Rate
    'BAMLH0A0HYM2': 'BAMLH0A0HYM2', # ICE BofA US High Yield OAS
    'VIXCLS': 'VIXCLS',             # VIX Close
}


# ============================================================================
# Data Classes
# ============================================================================

class ValuationLevel(str, Enum):
    """밸류에이션 수준"""
    CHEAP = "cheap"           # 저평가
    FAIR = "fair"             # 적정
    EXPENSIVE = "expensive"   # 고평가
    EXTREME = "extreme"       # 극단적 고평가


class CreditCondition(str, Enum):
    """크레딧 상태"""
    TIGHT = "tight"           # 스프레드 좁음 (낙관)
    NORMAL = "normal"         # 정상
    WIDE = "wide"             # 스프레드 확대 (우려)
    STRESS = "stress"         # 스트레스 (위기)


class VIXStructure(str, Enum):
    """VIX Term Structure"""
    CONTANGO = "contango"           # 정상 (선물 > 현물)
    FLAT = "flat"                   # 평탄
    BACKWARDATION = "backwardation" # 역전 (현물 > 선물, 공포)


@dataclass
class ValuationMetrics:
    """밸류에이션 지표"""
    timestamp: str

    # PE 관련
    pe_ratio: float = 0.0           # 현재 P/E
    pe_percentile: float = 0.0      # 역사적 백분위

    # Shiller CAPE
    cape_ratio: float = 0.0         # Shiller CAPE
    cape_percentile: float = 0.0    # 역사적 백분위

    # Buffett Indicator (추정)
    buffett_indicator: float = 0.0  # 시총/GDP 비율
    buffett_percentile: float = 0.0

    # Equity Risk Premium
    earnings_yield: float = 0.0     # E/P (PE 역수)
    treasury_yield: float = 0.0     # 10Y Treasury
    equity_risk_premium: float = 0.0 # EY - Treasury

    # 종합 판단
    level: ValuationLevel = ValuationLevel.FAIR
    score: float = 50.0             # 0-100 (50=fair)

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['level'] = self.level.value
        return data


class FearGreedLevel(str, Enum):
    """Fear & Greed 레벨"""
    EXTREME_FEAR = "Extreme Fear"     # 0-25
    FEAR = "Fear"                      # 26-45
    NEUTRAL = "Neutral"                # 46-55
    GREED = "Greed"                    # 56-75
    EXTREME_GREED = "Extreme Greed"    # 76-100


@dataclass
class CryptoMetrics:
    """24/7 암호화폐 지표"""
    timestamp: str

    # 가격
    btc_price: float = 0.0
    eth_price: float = 0.0
    sol_price: float = 0.0

    # 변화율
    btc_change_24h: float = 0.0
    btc_change_7d: float = 0.0
    eth_change_24h: float = 0.0

    # BTC Dominance (추정)
    btc_dominance: float = 0.0

    # ETH/BTC Ratio
    eth_btc_ratio: float = 0.0

    # 주말 변동 (금요일 마감 → 일요일 기준)
    btc_weekend_change: float = 0.0

    # Fear & Greed Index (Alternative.me)
    fear_greed_value: int = 50          # 0-100
    fear_greed_label: str = "Neutral"   # Extreme Fear ~ Extreme Greed
    fear_greed_yesterday: int = 50      # 어제 값
    fear_greed_week_ago: int = 50       # 1주일 전 값

    @property
    def fear_greed_level(self) -> str:
        """Alias for fear_greed_label"""
        return self.fear_greed_label

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CreditMetrics:
    """크레딧/채권 지표"""
    timestamp: str

    # 스프레드
    hy_ig_spread: float = 0.0       # HY-IG 스프레드
    hy_treasury_spread: float = 0.0 # HY-Treasury 스프레드

    # 수익률 곡선
    yield_2y: float = 0.0
    yield_10y: float = 0.0
    yield_curve_spread: float = 0.0 # 10Y-2Y
    curve_inverted: bool = False

    # 상태
    condition: CreditCondition = CreditCondition.NORMAL
    stress_score: float = 50.0      # 0-100 (높을수록 스트레스)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'hy_ig_spread': self.hy_ig_spread,
            'hy_treasury_spread': self.hy_treasury_spread,
            'yield_2y': self.yield_2y,
            'yield_10y': self.yield_10y,
            'yield_curve_spread': self.yield_curve_spread,
            'curve_inverted': int(self.curve_inverted),  # bool -> int
            'condition': self.condition.value,
            'stress_score': self.stress_score,
        }


@dataclass
class VIXMetrics:
    """VIX 지표"""
    timestamp: str

    # VIX 레벨
    vix: float = 0.0
    vix_3m: float = 0.0
    vvix: float = 0.0

    # Term Structure
    vix_term_spread: float = 0.0    # VIX3M - VIX
    structure: VIXStructure = VIXStructure.CONTANGO

    # 백분위
    vix_percentile: float = 0.0     # 1년 기준

    # ETF 신호
    vxx_change_1d: float = 0.0

    # 레벨 분류
    regime: str = "NORMAL"          # CALM, NORMAL, ELEVATED, CRISIS

    @property
    def spot(self) -> float:
        """Alias for vix"""
        return self.vix

    @property
    def current(self) -> float:
        """Alias for vix (현재 VIX 값)"""
        return self.vix

    @property
    def fear_greed_level(self) -> str:
        """VIX 기반 Fear & Greed 레벨"""
        if self.vix < 12:
            return "Extreme Greed"
        elif self.vix < 17:
            return "Greed"
        elif self.vix < 22:
            return "Neutral"
        elif self.vix < 30:
            return "Fear"
        else:
            return "Extreme Fear"

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['structure'] = self.structure.value
        return data


@dataclass
class FXMetrics:
    """FX 지표"""
    timestamp: str

    # 주요 환율
    usdjpy: float = 0.0
    eurusd: float = 0.0
    audusd: float = 0.0
    dxy: float = 0.0

    # 변화율
    usdjpy_change_1d: float = 0.0
    dxy_change_1d: float = 0.0

    # 달러 강세/약세
    dollar_strength: str = "NEUTRAL"  # STRONG, NEUTRAL, WEAK

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MarketIndicatorsSummary:
    """종합 지표 요약"""
    timestamp: str
    valuation: ValuationMetrics
    crypto: CryptoMetrics
    credit: CreditMetrics
    vix: VIXMetrics
    fx: FXMetrics

    # 종합 점수
    risk_score: float = 50.0        # 0-100 (높을수록 리스크)
    opportunity_score: float = 50.0  # 0-100 (높을수록 기회)

    # 주요 신호
    signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'valuation': self.valuation.to_dict(),
            'crypto': self.crypto.to_dict(),
            'credit': self.credit.to_dict(),
            'vix': self.vix.to_dict(),
            'fx': self.fx.to_dict(),
            'risk_score': self.risk_score,
            'opportunity_score': self.opportunity_score,
            'signals': self.signals,
            'warnings': self.warnings,
        }


# ============================================================================
# Market Indicators Collector
# ============================================================================

class MarketIndicatorsCollector:
    """
    시장 지표 수집기

    사용법:
        collector = MarketIndicatorsCollector()
        summary = collector.collect_all()
        collector.save_to_db(summary)
    """

    def __init__(self, lookback_days: int = 252):
        """
        Args:
            lookback_days: 백분위 계산을 위한 과거 데이터 일수
        """
        self.lookback_days = lookback_days
        self._cache = {}

    def _fetch_prices(self, tickers: Dict[str, str], period: str = "1y") -> Dict[str, pd.DataFrame]:
        """가격 데이터 수집"""
        data = {}
        ticker_list = list(tickers.values())

        try:
            df = yf.download(ticker_list, period=period, progress=False)

            for name, ticker in tickers.items():
                if ticker in df['Close'].columns:
                    data[name] = df['Close'][ticker].dropna()
                elif len(ticker_list) == 1:
                    data[name] = df['Close'].dropna()
        except Exception as e:
            print(f"Error fetching prices: {e}")

        return data

    def collect_valuation(self) -> ValuationMetrics:
        """밸류에이션 지표 수집"""
        timestamp = datetime.now().isoformat()

        try:
            # SPY 데이터
            spy = yf.Ticker("SPY")
            spy_info = spy.info

            # P/E Ratio
            pe_ratio = spy_info.get('trailingPE', 0) or 0

            # Earnings Yield (E/P)
            earnings_yield = (1 / pe_ratio * 100) if pe_ratio > 0 else 0

            # 10Y Treasury (TLT 기반 추정 또는 FRED)
            tlt = yf.Ticker("TLT")
            tlt_info = tlt.info
            treasury_yield = tlt_info.get('yield', 0.04) * 100 if tlt_info.get('yield') else 4.0

            # Equity Risk Premium
            erp = earnings_yield - treasury_yield

            # CAPE 추정 (역사적 평균 사용, 실제는 FRED에서)
            # 현재 CAPE는 대략 30-35 수준
            cape_ratio = pe_ratio * 1.1  # 단순 추정

            # Buffett Indicator 추정
            # 시총/GDP 비율 (현재 약 180-200% 수준)
            # VTI 시총 사용
            vti = yf.Ticker("VTI")
            vti_info = vti.info
            # 간단히 P/E 기반 추정
            buffett_indicator = pe_ratio * 6  # 대략적 추정

            # 백분위 계산 (1년 기준)
            prices = self._fetch_prices({'SPY': 'SPY'}, period='5y')
            if 'SPY' in prices and len(prices['SPY']) > 0:
                spy_prices = prices['SPY']
                current_price = spy_prices.iloc[-1]
                pe_percentile = (spy_prices < current_price).mean() * 100
            else:
                pe_percentile = 50.0

            # 밸류에이션 레벨 판단
            if cape_ratio < 15:
                level = ValuationLevel.CHEAP
                score = 25
            elif cape_ratio < 20:
                level = ValuationLevel.FAIR
                score = 45
            elif cape_ratio < 30:
                level = ValuationLevel.EXPENSIVE
                score = 70
            else:
                level = ValuationLevel.EXTREME
                score = 90

            return ValuationMetrics(
                timestamp=timestamp,
                pe_ratio=round(pe_ratio, 2),
                pe_percentile=round(pe_percentile, 1),
                cape_ratio=round(cape_ratio, 2),
                cape_percentile=round(pe_percentile, 1),  # 간단히 동일 사용
                buffett_indicator=round(buffett_indicator, 1),
                buffett_percentile=round(pe_percentile, 1),
                earnings_yield=round(earnings_yield, 2),
                treasury_yield=round(treasury_yield, 2),
                equity_risk_premium=round(erp, 2),
                level=level,
                score=round(score, 1),
            )

        except Exception as e:
            print(f"Error collecting valuation: {e}")
            return ValuationMetrics(timestamp=timestamp)

    def _fetch_fear_greed(self) -> Dict[str, Any]:
        """
        Crypto Fear & Greed Index 수집 (Alternative.me API)
        https://alternative.me/crypto/fear-and-greed-index/
        """
        try:
            # 최근 7일 데이터 요청
            url = "https://api.alternative.me/fng/?limit=7"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('data'):
                items = data['data']
                today = items[0] if len(items) > 0 else {}
                yesterday = items[1] if len(items) > 1 else {}
                week_ago = items[6] if len(items) > 6 else items[-1] if items else {}

                return {
                    'value': int(today.get('value', 50)),
                    'label': today.get('value_classification', 'Neutral'),
                    'yesterday': int(yesterday.get('value', 50)),
                    'week_ago': int(week_ago.get('value', 50)),
                }
        except Exception as e:
            print(f"  Warning: Fear & Greed API error: {e}")

        return {'value': 50, 'label': 'Neutral', 'yesterday': 50, 'week_ago': 50}

    def collect_crypto(self) -> CryptoMetrics:
        """24/7 암호화폐 지표 수집"""
        timestamp = datetime.now().isoformat()

        try:
            prices = self._fetch_prices(CRYPTO_TICKERS, period='1mo')

            btc_price = float(prices.get('BTC', pd.Series([0])).iloc[-1])
            eth_price = float(prices.get('ETH', pd.Series([0])).iloc[-1])
            sol_price = float(prices.get('SOL', pd.Series([0])).iloc[-1])

            # 변화율 계산
            btc_series = prices.get('BTC', pd.Series())
            eth_series = prices.get('ETH', pd.Series())

            btc_change_24h = 0.0
            btc_change_7d = 0.0
            eth_change_24h = 0.0

            if len(btc_series) >= 2:
                btc_change_24h = ((btc_series.iloc[-1] / btc_series.iloc[-2]) - 1) * 100
            if len(btc_series) >= 8:
                btc_change_7d = ((btc_series.iloc[-1] / btc_series.iloc[-8]) - 1) * 100
            if len(eth_series) >= 2:
                eth_change_24h = ((eth_series.iloc[-1] / eth_series.iloc[-2]) - 1) * 100

            # ETH/BTC ratio
            eth_btc_ratio = eth_price / btc_price if btc_price > 0 else 0

            # 주말 변동 (금요일 → 현재, 간단히 3일 전 비교)
            btc_weekend_change = 0.0
            if len(btc_series) >= 4:
                btc_weekend_change = ((btc_series.iloc[-1] / btc_series.iloc[-4]) - 1) * 100

            # Fear & Greed Index 수집
            fng = self._fetch_fear_greed()

            return CryptoMetrics(
                timestamp=timestamp,
                btc_price=round(btc_price, 2),
                eth_price=round(eth_price, 2),
                sol_price=round(sol_price, 2),
                btc_change_24h=round(btc_change_24h, 2),
                btc_change_7d=round(btc_change_7d, 2),
                eth_change_24h=round(eth_change_24h, 2),
                btc_dominance=0.0,  # 별도 API 필요
                eth_btc_ratio=round(eth_btc_ratio, 4),
                btc_weekend_change=round(btc_weekend_change, 2),
                fear_greed_value=fng['value'],
                fear_greed_label=fng['label'],
                fear_greed_yesterday=fng['yesterday'],
                fear_greed_week_ago=fng['week_ago'],
            )

        except Exception as e:
            print(f"Error collecting crypto: {e}")
            return CryptoMetrics(timestamp=timestamp)

    def collect_credit(self) -> CreditMetrics:
        """크레딧/채권 지표 수집"""
        timestamp = datetime.now().isoformat()

        try:
            prices = self._fetch_prices(CREDIT_TICKERS, period='1mo')

            # 현재 가격 및 수익률 추정
            hyg = prices.get('HYG', pd.Series([0]))
            lqd = prices.get('LQD', pd.Series([0]))
            tlt = prices.get('TLT', pd.Series([0]))

            hyg_price = float(hyg.iloc[-1]) if len(hyg) > 0 else 0
            lqd_price = float(lqd.iloc[-1]) if len(lqd) > 0 else 0
            tlt_price = float(tlt.iloc[-1]) if len(tlt) > 0 else 0

            # HY-IG 스프레드 (가격 비율 기반 추정)
            hy_ig_spread = 0.0
            if lqd_price > 0:
                hy_ig_ratio = hyg_price / lqd_price
                # 정상: ~0.75, 스트레스: ~0.70 이하
                hy_ig_spread = (0.77 - hy_ig_ratio) * 100  # 대략적 추정

            # HY-Treasury 스프레드
            hy_treasury_spread = 0.0
            if tlt_price > 0:
                hy_tlt_ratio = hyg_price / tlt_price
                hy_treasury_spread = (0.90 - hy_tlt_ratio) * 100  # 대략적 추정

            # Yield Curve (SHY, IEF 가격 변화 기반 추정)
            shy = prices.get('SHY', pd.Series([0]))
            ief = prices.get('IEF', pd.Series([0]))

            # 채권 가격과 수익률은 역관계
            yield_2y = 4.5  # 기본값
            yield_10y = 4.3  # 기본값

            if len(shy) > 20 and len(ief) > 20:
                # 가격 변화로 수익률 변화 추정
                shy_change = (shy.iloc[-1] / shy.iloc[-20] - 1) * 100
                ief_change = (ief.iloc[-1] / ief.iloc[-20] - 1) * 100
                # 가격 하락 = 수익률 상승
                yield_2y = 4.5 - shy_change * 0.5
                yield_10y = 4.3 - ief_change * 0.3

            yield_curve_spread = yield_10y - yield_2y
            curve_inverted = yield_curve_spread < 0

            # 크레딧 상태 판단
            if hy_ig_spread < 1.5:
                condition = CreditCondition.TIGHT
                stress_score = 20
            elif hy_ig_spread < 3.0:
                condition = CreditCondition.NORMAL
                stress_score = 40
            elif hy_ig_spread < 5.0:
                condition = CreditCondition.WIDE
                stress_score = 70
            else:
                condition = CreditCondition.STRESS
                stress_score = 90

            return CreditMetrics(
                timestamp=timestamp,
                hy_ig_spread=round(hy_ig_spread, 2),
                hy_treasury_spread=round(hy_treasury_spread, 2),
                yield_2y=round(yield_2y, 2),
                yield_10y=round(yield_10y, 2),
                yield_curve_spread=round(yield_curve_spread, 2),
                curve_inverted=curve_inverted,
                condition=condition,
                stress_score=round(stress_score, 1),
            )

        except Exception as e:
            print(f"Error collecting credit: {e}")
            return CreditMetrics(timestamp=timestamp)

    def collect_vix(self) -> VIXMetrics:
        """VIX 지표 수집"""
        timestamp = datetime.now().isoformat()

        try:
            prices = self._fetch_prices(VIX_TICKERS, period='1y')

            # VIX 현재값
            vix_series = prices.get('VIX', pd.Series([20]))
            vix = float(vix_series.iloc[-1]) if len(vix_series) > 0 else 20.0

            # VIX3M
            vix3m_series = prices.get('VIX3M', pd.Series([22]))
            vix_3m = float(vix3m_series.iloc[-1]) if len(vix3m_series) > 0 else vix * 1.1

            # VVIX
            vvix_series = prices.get('VVIX', pd.Series([100]))
            vvix = float(vvix_series.iloc[-1]) if len(vvix_series) > 0 else 100.0

            # Term Structure
            vix_term_spread = vix_3m - vix

            if vix_term_spread > 2:
                structure = VIXStructure.CONTANGO
            elif vix_term_spread < -2:
                structure = VIXStructure.BACKWARDATION
            else:
                structure = VIXStructure.FLAT

            # 백분위 (1년 기준)
            if len(vix_series) > 20:
                vix_percentile = (vix_series < vix).mean() * 100
            else:
                vix_percentile = 50.0

            # VXX 1일 변화
            vxx_series = prices.get('VXX', pd.Series())
            vxx_change_1d = 0.0
            if len(vxx_series) >= 2:
                vxx_change_1d = ((vxx_series.iloc[-1] / vxx_series.iloc[-2]) - 1) * 100

            # VIX 레짐
            if vix < 15:
                regime = "CALM"
            elif vix < 25:
                regime = "NORMAL"
            elif vix < 35:
                regime = "ELEVATED"
            else:
                regime = "CRISIS"

            return VIXMetrics(
                timestamp=timestamp,
                vix=round(vix, 2),
                vix_3m=round(vix_3m, 2),
                vvix=round(vvix, 2),
                vix_term_spread=round(vix_term_spread, 2),
                structure=structure,
                vix_percentile=round(vix_percentile, 1),
                vxx_change_1d=round(vxx_change_1d, 2),
                regime=regime,
            )

        except Exception as e:
            print(f"Error collecting VIX: {e}")
            return VIXMetrics(timestamp=timestamp)

    def collect_fx(self) -> FXMetrics:
        """FX 지표 수집"""
        timestamp = datetime.now().isoformat()

        try:
            prices = self._fetch_prices(FX_TICKERS, period='1mo')

            # 현재 환율
            usdjpy = float(prices.get('USDJPY', pd.Series([150])).iloc[-1])
            eurusd = float(prices.get('EURUSD', pd.Series([1.05])).iloc[-1])
            audusd = float(prices.get('AUDUSD', pd.Series([0.65])).iloc[-1])
            dxy = float(prices.get('DXY', pd.Series([105])).iloc[-1])

            # 1일 변화
            usdjpy_series = prices.get('USDJPY', pd.Series())
            dxy_series = prices.get('DXY', pd.Series())

            usdjpy_change_1d = 0.0
            dxy_change_1d = 0.0

            if len(usdjpy_series) >= 2:
                usdjpy_change_1d = ((usdjpy_series.iloc[-1] / usdjpy_series.iloc[-2]) - 1) * 100
            if len(dxy_series) >= 2:
                dxy_change_1d = ((dxy_series.iloc[-1] / dxy_series.iloc[-2]) - 1) * 100

            # 달러 강세/약세 판단
            if dxy > 106:
                dollar_strength = "STRONG"
            elif dxy < 100:
                dollar_strength = "WEAK"
            else:
                dollar_strength = "NEUTRAL"

            return FXMetrics(
                timestamp=timestamp,
                usdjpy=round(usdjpy, 2),
                eurusd=round(eurusd, 4),
                audusd=round(audusd, 4),
                dxy=round(dxy, 2),
                usdjpy_change_1d=round(usdjpy_change_1d, 2),
                dxy_change_1d=round(dxy_change_1d, 2),
                dollar_strength=dollar_strength,
            )

        except Exception as e:
            print(f"Error collecting FX: {e}")
            return FXMetrics(timestamp=timestamp)

    def collect_all(self) -> MarketIndicatorsSummary:
        """모든 지표 수집"""
        print("Collecting market indicators...")

        print("  [1/5] Valuation metrics...")
        valuation = self.collect_valuation()

        print("  [2/5] Crypto metrics...")
        crypto = self.collect_crypto()

        print("  [3/5] Credit metrics...")
        credit = self.collect_credit()

        print("  [4/5] VIX metrics...")
        vix = self.collect_vix()

        print("  [5/5] FX metrics...")
        fx = self.collect_fx()

        # 종합 점수 계산
        risk_score = self._calculate_risk_score(valuation, credit, vix)
        opportunity_score = self._calculate_opportunity_score(valuation, crypto, vix)

        # 신호/경고 생성
        signals, warnings = self._generate_signals(valuation, crypto, credit, vix, fx)

        return MarketIndicatorsSummary(
            timestamp=datetime.now().isoformat(),
            valuation=valuation,
            crypto=crypto,
            credit=credit,
            vix=vix,
            fx=fx,
            risk_score=round(risk_score, 1),
            opportunity_score=round(opportunity_score, 1),
            signals=signals,
            warnings=warnings,
        )

    def _calculate_risk_score(self, val: ValuationMetrics, credit: CreditMetrics,
                              vix: VIXMetrics) -> float:
        """리스크 점수 계산 (0-100)"""
        score = 50.0

        # 밸류에이션 기여
        score += (val.score - 50) * 0.3

        # 크레딧 기여
        score += (credit.stress_score - 50) * 0.3

        # VIX 기여
        if vix.regime == "CALM":
            score -= 10
        elif vix.regime == "ELEVATED":
            score += 15
        elif vix.regime == "CRISIS":
            score += 30

        # Backwardation 추가 리스크
        if vix.structure == VIXStructure.BACKWARDATION:
            score += 10

        return max(0, min(100, score))

    def _calculate_opportunity_score(self, val: ValuationMetrics, crypto: CryptoMetrics,
                                     vix: VIXMetrics) -> float:
        """기회 점수 계산 (0-100)"""
        score = 50.0

        # 저평가일 때 기회
        if val.level == ValuationLevel.CHEAP:
            score += 20
        elif val.level == ValuationLevel.EXTREME:
            score -= 20

        # VIX 급등 후 반등 기회
        if vix.regime == "CRISIS" and vix.structure == VIXStructure.BACKWARDATION:
            score += 15  # 공포 극대화 시 역발상

        # 크립토 급락 시 기회
        if crypto.btc_change_7d < -10:
            score += 10

        return max(0, min(100, score))

    def _generate_signals(self, val: ValuationMetrics, crypto: CryptoMetrics,
                          credit: CreditMetrics, vix: VIXMetrics,
                          fx: FXMetrics) -> Tuple[List[str], List[str]]:
        """신호 및 경고 생성"""
        signals = []
        warnings = []

        # 밸류에이션 신호
        if val.level == ValuationLevel.CHEAP:
            signals.append(f"밸류에이션 저평가 (CAPE: {val.cape_ratio:.1f})")
        elif val.level == ValuationLevel.EXTREME:
            warnings.append(f"밸류에이션 극단적 고평가 (CAPE: {val.cape_ratio:.1f})")

        if val.equity_risk_premium < 0:
            warnings.append(f"ERP 음수 ({val.equity_risk_premium:.1f}%) - 주식 매력도 낮음")

        # 크립토 신호
        if abs(crypto.btc_change_24h) > 5:
            if crypto.btc_change_24h > 0:
                signals.append(f"BTC 24시간 급등 (+{crypto.btc_change_24h:.1f}%)")
            else:
                warnings.append(f"BTC 24시간 급락 ({crypto.btc_change_24h:.1f}%)")

        if abs(crypto.btc_weekend_change) > 5:
            warnings.append(f"BTC 주말 변동 큼 ({crypto.btc_weekend_change:+.1f}%)")

        # Fear & Greed 신호
        if crypto.fear_greed_value <= 25:
            signals.append(f"Crypto Extreme Fear ({crypto.fear_greed_value}) - 매수 기회?")
        elif crypto.fear_greed_value >= 75:
            warnings.append(f"Crypto Extreme Greed ({crypto.fear_greed_value}) - 과열 주의")

        # Fear & Greed 급변
        fng_change = crypto.fear_greed_value - crypto.fear_greed_yesterday
        if abs(fng_change) >= 10:
            if fng_change > 0:
                signals.append(f"Fear & Greed 급등 ({crypto.fear_greed_yesterday} → {crypto.fear_greed_value})")
            else:
                warnings.append(f"Fear & Greed 급락 ({crypto.fear_greed_yesterday} → {crypto.fear_greed_value})")

        # 크레딧 신호
        if credit.condition == CreditCondition.STRESS:
            warnings.append(f"크레딧 스트레스 (HY-IG: {credit.hy_ig_spread:.1f}%)")

        if credit.curve_inverted:
            warnings.append(f"수익률 곡선 역전 ({credit.yield_curve_spread:.2f}%)")

        # VIX 신호
        if vix.regime == "CRISIS":
            warnings.append(f"VIX 위기 수준 ({vix.vix:.1f})")
        elif vix.regime == "ELEVATED":
            warnings.append(f"VIX 상승 ({vix.vix:.1f})")

        if vix.structure == VIXStructure.BACKWARDATION:
            warnings.append(f"VIX Backwardation - 공포 신호")

        # FX 신호
        if fx.dollar_strength == "STRONG":
            signals.append(f"달러 강세 (DXY: {fx.dxy:.1f})")
        elif fx.dollar_strength == "WEAK":
            signals.append(f"달러 약세 (DXY: {fx.dxy:.1f})")

        return signals, warnings

    def save_to_db(self, summary: MarketIndicatorsSummary,
                   db: DatabaseManager = None) -> bool:
        """DB에 저장"""
        if db is None:
            db = DatabaseManager()

        today = datetime.now().strftime("%Y-%m-%d")

        try:
            # ETF Analysis 테이블에 저장
            db.save_etf_analysis('market_indicators', summary.to_dict(), today)

            # Market Regime에 VIX 정보 추가
            regime_data = {
                'vix_estimate': summary.vix.vix,
                'risk_appetite_score': 100 - summary.risk_score,
            }

            # 로그
            db.log_analysis(
                analysis_type='market_indicators',
                status='SUCCESS',
                records=5,  # 5개 카테고리
                date_str=today
            )

            return True

        except Exception as e:
            print(f"Error saving to DB: {e}")
            return False

    def print_report(self, summary: MarketIndicatorsSummary):
        """리포트 출력"""
        print("\n" + "=" * 70)
        print("MARKET INDICATORS REPORT")
        print(f"Generated: {summary.timestamp[:19]}")
        print("=" * 70)

        # 종합 점수
        print(f"\n[Overall Scores]")
        print(f"  Risk Score:        {summary.risk_score:.0f}/100")
        print(f"  Opportunity Score: {summary.opportunity_score:.0f}/100")

        # 밸류에이션
        val = summary.valuation
        print(f"\n[Valuation] - {val.level.value.upper()}")
        print(f"  P/E Ratio:         {val.pe_ratio:.1f} ({val.pe_percentile:.0f}%ile)")
        print(f"  CAPE (est):        {val.cape_ratio:.1f}")
        print(f"  Earnings Yield:    {val.earnings_yield:.2f}%")
        print(f"  10Y Treasury:      {val.treasury_yield:.2f}%")
        print(f"  Equity Risk Prem:  {val.equity_risk_premium:.2f}%")

        # 암호화폐
        crypto = summary.crypto
        print(f"\n[Crypto - 24/7 Monitor]")
        print(f"  BTC: ${crypto.btc_price:,.0f} ({crypto.btc_change_24h:+.1f}% 24h, {crypto.btc_change_7d:+.1f}% 7d)")
        print(f"  ETH: ${crypto.eth_price:,.0f} ({crypto.eth_change_24h:+.1f}% 24h)")
        print(f"  ETH/BTC: {crypto.eth_btc_ratio:.4f}")
        print(f"  Weekend Change: {crypto.btc_weekend_change:+.1f}%")
        print(f"  Fear & Greed: {crypto.fear_greed_value} ({crypto.fear_greed_label})")
        print(f"    Yesterday: {crypto.fear_greed_yesterday}, Week Ago: {crypto.fear_greed_week_ago}")

        # 크레딧
        credit = summary.credit
        print(f"\n[Credit] - {credit.condition.value.upper()}")
        print(f"  HY-IG Spread:      {credit.hy_ig_spread:.2f}%")
        print(f"  HY-Treasury:       {credit.hy_treasury_spread:.2f}%")
        print(f"  2Y Yield:          {credit.yield_2y:.2f}%")
        print(f"  10Y Yield:         {credit.yield_10y:.2f}%")
        print(f"  Curve (10Y-2Y):    {credit.yield_curve_spread:.2f}% {'[INVERTED]' if credit.curve_inverted else ''}")

        # VIX
        vix = summary.vix
        print(f"\n[VIX] - {vix.regime}")
        print(f"  VIX:               {vix.vix:.2f} ({vix.vix_percentile:.0f}%ile)")
        print(f"  VIX3M:             {vix.vix_3m:.2f}")
        print(f"  Term Spread:       {vix.vix_term_spread:.2f} ({vix.structure.value})")
        print(f"  VVIX:              {vix.vvix:.1f}")

        # FX
        fx = summary.fx
        print(f"\n[FX] - Dollar {fx.dollar_strength}")
        print(f"  DXY:               {fx.dxy:.2f} ({fx.dxy_change_1d:+.2f}%)")
        print(f"  USD/JPY:           {fx.usdjpy:.2f}")
        print(f"  EUR/USD:           {fx.eurusd:.4f}")
        print(f"  AUD/USD:           {fx.audusd:.4f}")

        # 신호
        if summary.signals:
            print(f"\n[Signals]")
            for sig in summary.signals:
                print(f"  + {sig}")

        # 경고
        if summary.warnings:
            print(f"\n[Warnings]")
            for warn in summary.warnings:
                print(f"  ! {warn}")

        print("\n" + "=" * 70)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Market Indicators Collector Test")
    print("=" * 70)

    collector = MarketIndicatorsCollector()
    summary = collector.collect_all()

    # 리포트 출력
    collector.print_report(summary)

    # DB 저장
    print("\n[Saving to Database]")
    db = DatabaseManager()
    if collector.save_to_db(summary, db):
        print("  Saved successfully!")

    # DB 확인
    result = db.get_etf_analysis('market_indicators')
    if result:
        print(f"  Retrieved from DB: {result['date']}")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
