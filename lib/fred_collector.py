#!/usr/bin/env python3
"""
FRED API Collector
==================
Federal Reserve Economic Data (FRED) API를 통한 거시경제 지표 수집

주요 지표:
- 금리: Fed Funds Rate, 2Y/10Y/30Y Treasury
- 스프레드: 10Y-2Y, HY OAS
- 인플레이션: CPI, PCE, Breakeven
- 고용: 실업률, 비농업 고용
- 기타: GDP, 산업생산

사용법:
    collector = FREDCollector()
    data = collector.collect_all()
    collector.save_to_db(data)
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from enum import Enum
import json

from core.database import DatabaseManager


# ============================================================================
# Constants
# ============================================================================

FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# 주요 FRED 시리즈
FRED_SERIES = {
    # 금리
    'fed_funds': 'DFF',                   # Effective Federal Funds Rate (Daily)
    'fed_target_upper': 'DFEDTARU',       # Fed Target Upper
    'fed_target_lower': 'DFEDTARL',       # Fed Target Lower
    'treasury_3m': 'DGS3MO',              # 3-Month Treasury
    'treasury_2y': 'DGS2',                # 2-Year Treasury
    'treasury_5y': 'DGS5',                # 5-Year Treasury
    'treasury_10y': 'DGS10',              # 10-Year Treasury
    'treasury_30y': 'DGS30',              # 30-Year Treasury

    # 스프레드
    'spread_10y2y': 'T10Y2Y',             # 10Y-2Y Spread
    'spread_10y3m': 'T10Y3M',             # 10Y-3M Spread
    'hy_oas': 'BAMLH0A0HYM2',             # ICE BofA US High Yield OAS
    'ig_oas': 'BAMLC0A4CBBB',             # ICE BofA BBB Corporate OAS

    # 인플레이션
    'cpi': 'CPIAUCSL',                    # CPI All Urban Consumers (Monthly)
    'core_cpi': 'CPILFESL',               # Core CPI (Monthly)
    'pce': 'PCEPI',                       # PCE Price Index (Monthly)
    'core_pce': 'PCEPILFE',               # Core PCE (Monthly)
    'breakeven_5y': 'T5YIE',              # 5-Year Breakeven Inflation
    'breakeven_10y': 'T10YIE',            # 10-Year Breakeven Inflation

    # 고용
    'unemployment': 'UNRATE',              # Unemployment Rate (Monthly)
    'payrolls': 'PAYEMS',                  # Total Nonfarm Payrolls (Monthly)
    'initial_claims': 'ICSA',              # Initial Jobless Claims (Weekly)

    # 경제활동
    'gdp': 'GDP',                          # GDP (Quarterly)
    'industrial_prod': 'INDPRO',           # Industrial Production (Monthly)
    'retail_sales': 'RSAFS',               # Retail Sales (Monthly)

    # 유동성 지표 (Liquidity) - 핵심 Alpha 신호
    'rrp': 'RRPONTSYD',                    # Overnight Reverse Repo (Daily, Billions)
    'tga': 'WTREGEN',                      # Treasury General Account (Weekly, Billions)
    'fed_assets': 'WALCL',                 # Fed Total Assets (Weekly, Millions) - QT 추적
    'reserves': 'TOTRESNS',                # Total Reserves (Monthly, Billions)
    'excess_reserves': 'EXCSRESNS',        # Excess Reserves (Monthly, Billions) - 2020년 이후 중단
    'iorb': 'IORB',                        # Interest on Reserve Balances (Daily, %)

    # 기타
    'vix': 'VIXCLS',                       # VIX Close
    'dxy': 'DTWEXBGS',                     # Trade Weighted Dollar Index
    'sp500': 'SP500',                      # S&P 500
}

# 카테고리별 분류
SERIES_CATEGORIES = {
    'rates': ['fed_funds', 'fed_target_upper', 'fed_target_lower',
              'treasury_3m', 'treasury_2y', 'treasury_5y', 'treasury_10y', 'treasury_30y'],
    'spreads': ['spread_10y2y', 'spread_10y3m', 'hy_oas', 'ig_oas'],
    'inflation': ['cpi', 'core_cpi', 'pce', 'core_pce', 'breakeven_5y', 'breakeven_10y'],
    'employment': ['unemployment', 'payrolls', 'initial_claims'],
    'activity': ['gdp', 'industrial_prod', 'retail_sales'],
    'liquidity': ['rrp', 'tga', 'fed_assets', 'reserves', 'iorb'],  # 핵심 유동성 지표
    'markets': ['vix', 'dxy', 'sp500'],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class FREDDataPoint:
    """FRED 데이터 포인트"""
    series_id: str
    name: str
    date: str
    value: float
    unit: str = ""
    frequency: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FREDSummary:
    """FRED 데이터 요약"""
    timestamp: str

    # 금리
    fed_funds: float = 0.0
    treasury_2y: float = 0.0
    treasury_10y: float = 0.0
    treasury_30y: float = 0.0

    # 스프레드
    spread_10y2y: float = 0.0
    spread_10y3m: float = 0.0
    hy_oas: float = 0.0

    # 인플레이션
    cpi_yoy: float = 0.0
    core_pce_yoy: float = 0.0
    breakeven_5y: float = 0.0
    breakeven_10y: float = 0.0

    # 고용
    unemployment: float = 0.0
    initial_claims: int = 0

    # 유동성 지표 (Liquidity) - Alpha 핵심
    rrp: float = 0.0               # Overnight RRP (Billions)
    rrp_delta: float = 0.0         # 전일 대비 변화 (Billions)
    rrp_delta_pct: float = 0.0     # 전일 대비 변화율 (%)
    tga: float = 0.0               # Treasury General Account (Billions)
    tga_delta: float = 0.0         # 전주 대비 변화 (Billions)
    fed_assets: float = 0.0        # Fed Total Assets (Trillions)
    fed_assets_delta: float = 0.0  # 전주 대비 변화 (Billions) - QT 추적
    net_liquidity: float = 0.0     # Fed Assets - RRP - TGA (핵심 지표)
    liquidity_regime: str = "Normal"  # Abundant, Normal, Tight, Stressed

    # 수익률 곡선 상태
    curve_inverted: bool = False
    curve_status: str = "Normal"  # Normal, Flat, Inverted

    # 신호
    signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'fed_funds': self.fed_funds,
            'treasury_2y': self.treasury_2y,
            'treasury_10y': self.treasury_10y,
            'treasury_30y': self.treasury_30y,
            'spread_10y2y': self.spread_10y2y,
            'spread_10y3m': self.spread_10y3m,
            'hy_oas': self.hy_oas,
            'cpi_yoy': self.cpi_yoy,
            'core_pce_yoy': self.core_pce_yoy,
            'breakeven_5y': self.breakeven_5y,
            'breakeven_10y': self.breakeven_10y,
            'unemployment': self.unemployment,
            'initial_claims': self.initial_claims,
            # 유동성 지표
            'rrp': self.rrp,
            'rrp_delta': self.rrp_delta,
            'rrp_delta_pct': self.rrp_delta_pct,
            'tga': self.tga,
            'tga_delta': self.tga_delta,
            'fed_assets': self.fed_assets,
            'fed_assets_delta': self.fed_assets_delta,
            'net_liquidity': self.net_liquidity,
            'liquidity_regime': self.liquidity_regime,
            # 상태
            'curve_inverted': int(self.curve_inverted),
            'curve_status': self.curve_status,
            'signals': self.signals,
            'warnings': self.warnings,
        }


# ============================================================================
# FRED Collector
# ============================================================================

class FREDCollector:
    """
    FRED API 데이터 수집기

    사용법:
        collector = FREDCollector()
        summary = collector.collect_all()
        collector.print_report(summary)
        collector.save_to_db(summary)
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or FRED_API_KEY
        if not self.api_key:
            raise ValueError("FRED_API_KEY not set. Set environment variable or pass api_key.")
        self._cache: Dict[str, pd.Series] = {}
        self.request_timeout_sec = max(
            1.0,
            float(os.getenv("EIMAS_FRED_TIMEOUT_SEC", "15")),
        )
        self.fail_fast_network = os.getenv(
            "EIMAS_FRED_FAIL_FAST_NETWORK",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._network_unavailable = False
        self._network_fail_reason = ""
        self._failfast_notice_printed = False

    @staticmethod
    def _is_network_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "failed to resolve",
                "name or service not known",
                "name resolution",
                "temporary failure in name resolution",
                "network is unreachable",
                "connection reset",
                "connection aborted",
                "max retries exceeded",
            )
        )

    def _fetch_series(self, series_id: str, start_date: str = None,
                      end_date: str = None) -> Optional[pd.Series]:
        """FRED 시리즈 데이터 수집"""
        if series_id in self._cache:
            return self._cache[series_id]
        if self._network_unavailable:
            if not self._failfast_notice_printed:
                reason = self._network_fail_reason or "network unavailable"
                print(f"  FRED fail-fast active: skipping remaining requests ({reason})")
                self._failfast_notice_printed = True
            return None

        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'sort_order': 'desc',
            'limit': 100,
        }

        try:
            response = requests.get(
                FRED_BASE_URL,
                params=params,
                timeout=self.request_timeout_sec,
            )
            response.raise_for_status()
            data = response.json()

            observations = data.get('observations', [])
            if not observations:
                return None

            # DataFrame으로 변환
            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])
            df = df.set_index('date')['value']

            self._cache[series_id] = df
            return df

        except Exception as e:
            if self.fail_fast_network and self._is_network_error(e):
                self._network_unavailable = True
                self._network_fail_reason = str(e).strip()[:200]
            print(f"  Error fetching {series_id}: {e}")
            return None

    def get_latest(self, series_name: str) -> Optional[float]:
        """최신 값 조회"""
        series_id = FRED_SERIES.get(series_name)
        if not series_id:
            return None

        data = self._fetch_series(series_id)
        if data is not None and len(data) > 0:
            return float(data.iloc[0])
        return None

    def get_series(self, series_name: str, days: int = 365) -> Optional[pd.Series]:
        """시계열 데이터 조회"""
        series_id = FRED_SERIES.get(series_name)
        if not series_id:
            return None

        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return self._fetch_series(series_id, start_date=start_date)

    def calculate_yoy_change(self, series_name: str) -> Optional[float]:
        """YoY 변화율 계산 (인플레이션 등)"""
        data = self.get_series(series_name, days=400)
        if data is None or len(data) < 12:
            return None

        # 최근 값과 1년 전 값 비교
        latest = data.iloc[0]
        year_ago_idx = min(12, len(data) - 1)  # 월간 데이터 기준
        year_ago = data.iloc[year_ago_idx]

        if year_ago > 0:
            return ((latest / year_ago) - 1) * 100
        return None

    def collect_rates(self) -> Dict[str, float]:
        """금리 데이터 수집"""
        print("  [1/5] Collecting rates...")
        rates = {}

        for name in SERIES_CATEGORIES['rates']:
            value = self.get_latest(name)
            if value is not None:
                rates[name] = value

        return rates

    def collect_spreads(self) -> Dict[str, float]:
        """스프레드 데이터 수집"""
        print("  [2/5] Collecting spreads...")
        spreads = {}

        for name in SERIES_CATEGORIES['spreads']:
            value = self.get_latest(name)
            if value is not None:
                spreads[name] = value

        return spreads

    def collect_inflation(self) -> Dict[str, float]:
        """인플레이션 데이터 수집"""
        print("  [3/5] Collecting inflation...")
        inflation = {}

        # CPI YoY
        cpi_yoy = self.calculate_yoy_change('cpi')
        if cpi_yoy:
            inflation['cpi_yoy'] = cpi_yoy

        # Core PCE YoY
        pce_yoy = self.calculate_yoy_change('core_pce')
        if pce_yoy:
            inflation['core_pce_yoy'] = pce_yoy

        # Breakeven
        for name in ['breakeven_5y', 'breakeven_10y']:
            value = self.get_latest(name)
            if value is not None:
                inflation[name] = value

        return inflation

    def collect_employment(self) -> Dict[str, float]:
        """고용 데이터 수집"""
        print("  [4/5] Collecting employment...")
        employment = {}

        unemployment = self.get_latest('unemployment')
        if unemployment:
            employment['unemployment'] = unemployment

        claims = self.get_latest('initial_claims')
        if claims:
            employment['initial_claims'] = int(claims)  # 이미 실제 값

        return employment

    def collect_liquidity(self) -> Dict[str, float]:
        """
        유동성 데이터 수집 - Alpha 핵심 지표

        핵심 공식:
        Net Liquidity = Fed Assets - RRP - TGA

        해석:
        - RRP 감소 → 유동성 시장 유입 → 위험자산 상승
        - TGA 증가 → 유동성 흡수 → 위험자산 하락
        - Fed Assets 감소 (QT) → 유동성 축소 → 변동성 증가
        """
        print("  [5/5] Collecting liquidity (RRP/TGA/Fed)...")
        liquidity = {}

        # RRP - Overnight Reverse Repo (일간, Billions)
        rrp_series = self.get_series('rrp', days=30)
        if rrp_series is not None and len(rrp_series) >= 2:
            rrp_latest = float(rrp_series.iloc[0])
            rrp_prev = float(rrp_series.iloc[1])
            liquidity['rrp'] = rrp_latest
            liquidity['rrp_delta'] = rrp_latest - rrp_prev
            if rrp_prev > 0:
                liquidity['rrp_delta_pct'] = ((rrp_latest - rrp_prev) / rrp_prev) * 100

        # TGA - Treasury General Account (주간, FRED는 Millions → Billions 변환)
        tga_series = self.get_series('tga', days=60)
        if tga_series is not None and len(tga_series) >= 2:
            tga_latest = float(tga_series.iloc[0]) / 1000  # Millions → Billions
            tga_prev = float(tga_series.iloc[1]) / 1000
            liquidity['tga'] = tga_latest
            liquidity['tga_delta'] = tga_latest - tga_prev

        # Fed Assets - 연준 총자산 (주간, Millions → Trillions 변환)
        fed_series = self.get_series('fed_assets', days=60)
        if fed_series is not None and len(fed_series) >= 2:
            fed_latest = float(fed_series.iloc[0]) / 1_000_000  # Millions → Trillions
            fed_prev = float(fed_series.iloc[1]) / 1_000_000
            liquidity['fed_assets'] = fed_latest
            liquidity['fed_assets_delta'] = (fed_latest - fed_prev) * 1000  # Billions

        # Net Liquidity 계산 (모든 값을 Billions로 통일)
        rrp = liquidity.get('rrp', 0)
        tga = liquidity.get('tga', 0)
        fed = liquidity.get('fed_assets', 0) * 1000  # Trillions → Billions

        if fed > 0:
            net_liq = fed - rrp - tga
            liquidity['net_liquidity'] = net_liq

            # 유동성 레짐 판단 (기준: 2020-2024 평균 약 3.5T)
            if net_liq > 4000:  # > $4T
                liquidity['liquidity_regime'] = "Abundant"
            elif net_liq > 3000:  # $3T - $4T
                liquidity['liquidity_regime'] = "Normal"
            elif net_liq > 2500:  # $2.5T - $3T
                liquidity['liquidity_regime'] = "Tight"
            else:  # < $2.5T
                liquidity['liquidity_regime'] = "Stressed"

        return liquidity

    def collect_all(self) -> FREDSummary:
        """모든 FRED 데이터 수집"""
        print("Collecting FRED data...")

        rates = self.collect_rates()
        spreads = self.collect_spreads()
        inflation = self.collect_inflation()
        employment = self.collect_employment()
        liquidity = self.collect_liquidity()

        # 수익률 곡선 상태 판단
        spread_10y2y = spreads.get('spread_10y2y', 0)
        if spread_10y2y < -0.25:
            curve_status = "Inverted"
            curve_inverted = True
        elif spread_10y2y < 0.25:
            curve_status = "Flat"
            curve_inverted = False
        else:
            curve_status = "Normal"
            curve_inverted = False

        # 신호 생성 (유동성 포함)
        signals, warnings = self._generate_signals(rates, spreads, inflation, employment, liquidity)

        return FREDSummary(
            timestamp=datetime.now().isoformat(),
            fed_funds=round(rates.get('fed_funds', 0), 2),
            treasury_2y=round(rates.get('treasury_2y', 0), 2),
            treasury_10y=round(rates.get('treasury_10y', 0), 2),
            treasury_30y=round(rates.get('treasury_30y', 0), 2),
            spread_10y2y=round(spread_10y2y, 2),
            spread_10y3m=round(spreads.get('spread_10y3m', 0), 2),
            hy_oas=round(spreads.get('hy_oas', 0), 2),
            cpi_yoy=round(inflation.get('cpi_yoy', 0), 2),
            core_pce_yoy=round(inflation.get('core_pce_yoy', 0), 2),
            breakeven_5y=round(inflation.get('breakeven_5y', 0), 2),
            breakeven_10y=round(inflation.get('breakeven_10y', 0), 2),
            unemployment=round(employment.get('unemployment', 0), 1),
            initial_claims=employment.get('initial_claims', 0),
            # 유동성 지표
            rrp=round(liquidity.get('rrp', 0), 1),
            rrp_delta=round(liquidity.get('rrp_delta', 0), 1),
            rrp_delta_pct=round(liquidity.get('rrp_delta_pct', 0), 2),
            tga=round(liquidity.get('tga', 0), 1),
            tga_delta=round(liquidity.get('tga_delta', 0), 1),
            fed_assets=round(liquidity.get('fed_assets', 0), 3),
            fed_assets_delta=round(liquidity.get('fed_assets_delta', 0), 1),
            net_liquidity=round(liquidity.get('net_liquidity', 0), 1),
            liquidity_regime=liquidity.get('liquidity_regime', 'Normal'),
            # 상태
            curve_inverted=curve_inverted,
            curve_status=curve_status,
            signals=signals,
            warnings=warnings,
        )

    def _generate_signals(self, rates: Dict, spreads: Dict,
                          inflation: Dict, employment: Dict,
                          liquidity: Dict = None) -> tuple:
        """신호 생성"""
        signals = []
        warnings = []
        liquidity = liquidity or {}

        # 수익률 곡선
        spread = spreads.get('spread_10y2y', 0)
        if spread < -0.25:
            warnings.append(f"수익률 곡선 역전 ({spread:.2f}%) - 경기침체 신호")
        elif spread < 0:
            warnings.append(f"수익률 곡선 거의 역전 ({spread:.2f}%)")

        # HY 스프레드
        hy_oas = spreads.get('hy_oas', 0)
        if hy_oas > 5.0:
            warnings.append(f"HY 스프레드 확대 ({hy_oas:.0f}bp) - 크레딧 스트레스")
        elif hy_oas > 4.0:
            warnings.append(f"HY 스프레드 상승 ({hy_oas:.0f}bp)")
        elif hy_oas < 3.0:
            signals.append(f"HY 스프레드 안정 ({hy_oas:.0f}bp)")

        # 인플레이션
        cpi = inflation.get('cpi_yoy', 0)
        if cpi > 4.0:
            warnings.append(f"CPI 높음 ({cpi:.1f}%) - 긴축 지속 우려")
        elif cpi < 2.0:
            signals.append(f"CPI 안정 ({cpi:.1f}%)")

        # 실업률
        unemp = employment.get('unemployment', 0)
        if unemp > 5.0:
            warnings.append(f"실업률 상승 ({unemp:.1f}%)")
        elif unemp < 4.0:
            signals.append(f"고용시장 견고 (실업률 {unemp:.1f}%)")

        # 실업수당 청구
        claims = employment.get('initial_claims', 0)
        if claims > 300000:
            warnings.append(f"실업수당 청구 증가 ({claims:,}건)")

        # ============================================================
        # 유동성 신호 (Alpha 핵심) - RRP/TGA/Fed Assets
        # ============================================================

        # RRP 급변 감지 (일간 변화)
        rrp_delta = liquidity.get('rrp_delta', 0)
        rrp_delta_pct = liquidity.get('rrp_delta_pct', 0)
        if rrp_delta < -50:  # RRP $50B 이상 감소
            signals.append(f"🔥 RRP 급감 ({rrp_delta:+.0f}B, {rrp_delta_pct:+.1f}%) - 유동성 시장 유입")
        elif rrp_delta > 50:  # RRP $50B 이상 증가
            warnings.append(f"RRP 급증 ({rrp_delta:+.0f}B) - 유동성 흡수")

        # TGA 변화 감지 (주간)
        tga_delta = liquidity.get('tga_delta', 0)
        if tga_delta > 50:  # TGA $50B 이상 증가
            warnings.append(f"TGA 증가 ({tga_delta:+.0f}B) - 유동성 흡수 중")
        elif tga_delta < -50:  # TGA $50B 이상 감소
            signals.append(f"TGA 감소 ({tga_delta:+.0f}B) - 유동성 방출 중")

        # Fed Assets (QT 추적)
        fed_delta = liquidity.get('fed_assets_delta', 0)
        if fed_delta < -20:  # 주간 $20B 이상 감소
            warnings.append(f"QT 진행 중 (Fed -{abs(fed_delta):.0f}B/주)")

        # Net Liquidity 레짐
        regime = liquidity.get('liquidity_regime', 'Normal')
        net_liq = liquidity.get('net_liquidity', 0)
        if regime == "Abundant":
            signals.append(f"유동성 풍부 (Net ${net_liq/1000:.2f}T) - Risk-On 우호적")
        elif regime == "Tight":
            warnings.append(f"유동성 긴축 (Net ${net_liq/1000:.2f}T) - 변동성 주의")
        elif regime == "Stressed":
            warnings.append(f"⚠️ 유동성 스트레스 (Net ${net_liq/1000:.2f}T) - 고위험")

        return signals, warnings

    def save_to_db(self, summary: FREDSummary, db: DatabaseManager = None) -> bool:
        """DB에 저장"""
        if db is None:
            db = DatabaseManager()

        today = datetime.now().strftime("%Y-%m-%d")

        try:
            db.save_etf_analysis('fred_indicators', summary.to_dict(), today)
            db.log_analysis('fred_indicators', 'SUCCESS', len(FRED_SERIES), today)
            return True
        except Exception as e:
            print(f"Error saving to DB: {e}")
            return False

    def print_report(self, summary: FREDSummary):
        """리포트 출력"""
        print("\n" + "=" * 60)
        print("FRED ECONOMIC INDICATORS REPORT")
        print(f"Generated: {summary.timestamp[:19]}")
        print("=" * 60)

        print(f"\n[Interest Rates]")
        print(f"  Fed Funds:     {summary.fed_funds:.2f}%")
        print(f"  2Y Treasury:   {summary.treasury_2y:.2f}%")
        print(f"  10Y Treasury:  {summary.treasury_10y:.2f}%")
        print(f"  30Y Treasury:  {summary.treasury_30y:.2f}%")

        print(f"\n[Yield Curve] - {summary.curve_status}")
        print(f"  10Y-2Y Spread: {summary.spread_10y2y:.2f}%")
        print(f"  10Y-3M Spread: {summary.spread_10y3m:.2f}%")
        if summary.curve_inverted:
            print(f"  *** INVERTED ***")

        print(f"\n[Credit Spreads]")
        print(f"  HY OAS:        {summary.hy_oas:.0f} bp")

        print(f"\n[Inflation]")
        print(f"  CPI YoY:       {summary.cpi_yoy:.1f}%")
        print(f"  Core PCE YoY:  {summary.core_pce_yoy:.1f}%")
        print(f"  5Y Breakeven:  {summary.breakeven_5y:.2f}%")
        print(f"  10Y Breakeven: {summary.breakeven_10y:.2f}%")

        print(f"\n[Employment]")
        print(f"  Unemployment:  {summary.unemployment:.1f}%")
        print(f"  Initial Claims:{summary.initial_claims:,}")

        # 유동성 섹션 (핵심 Alpha 지표)
        print(f"\n[Liquidity] - {summary.liquidity_regime}")
        print(f"  RRP:           ${summary.rrp:.0f}B ({summary.rrp_delta:+.0f}B, {summary.rrp_delta_pct:+.1f}%)")
        print(f"  TGA:           ${summary.tga:.0f}B ({summary.tga_delta:+.0f}B)")
        print(f"  Fed Assets:    ${summary.fed_assets:.2f}T ({summary.fed_assets_delta:+.0f}B/wk)")
        print(f"  Net Liquidity: ${summary.net_liquidity/1000:.2f}T")
        if summary.liquidity_regime in ["Tight", "Stressed"]:
            print(f"  *** {summary.liquidity_regime.upper()} LIQUIDITY ***")

        if summary.signals:
            print(f"\n[Signals]")
            for sig in summary.signals:
                print(f"  + {sig}")

        if summary.warnings:
            print(f"\n[Warnings]")
            for warn in summary.warnings:
                print(f"  ! {warn}")

        print("\n" + "=" * 60)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("FRED Collector Test")
    print("=" * 60)

    try:
        collector = FREDCollector()

        # 전체 수집
        summary = collector.collect_all()

        # 리포트 출력
        collector.print_report(summary)

        # DB 저장
        print("\n[Saving to Database]")
        db = DatabaseManager()
        if collector.save_to_db(summary, db):
            print("  Saved successfully!")

        print("\n" + "=" * 60)
        print("Test Complete!")
        print("=" * 60)

    except ValueError as e:
        print(f"Error: {e}")
        print("Please set FRED_API_KEY environment variable")
