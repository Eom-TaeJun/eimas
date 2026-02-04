#!/usr/bin/env python3
"""
Korea Market Data Collector
============================
한국 시장 데이터 수집 (KOSPI, KRX ETF, 대형주)

Purpose:
- Fair KOSPI 산출을 위한 한국 시장 데이터
- 한미 비교 분석 (KOSPI vs SPX)
- 글로벌 자산배분을 위한 한국 자산군 포함

Data Sources:
- Yahoo Finance: ^KS11 (KOSPI), ETF (KODEX, TIGER), 대형주
- FinanceDataReader (optional): KRX 직접 데이터

Economic Foundation:
- Fama-French (1992): International Asset Pricing
- Solnik (1974): International Diversification
- Harvey (1995): Predictable Risk and Returns in Emerging Markets
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# KOSPI Asset Configuration
# ============================================================================

KOSPI_CONFIG = {
    # 지수
    'indices': {
        'KOSPI': '^KS11',           # KOSPI 지수
        'KOSPI200': '^KS200',       # KOSPI 200
        'KOSDAQ': '^KQ11',          # KOSDAQ
    },

    # 섹터 ETF (KODEX)
    'sector_etfs': {
        'KODEX_Bank': '091170.KS',         # 은행
        'KODEX_Semi': '091180.KS',         # 반도체
        'KODEX_Bio': '228790.KS',          # 바이오
        'KODEX_2ndBattery': '305720.KS',   # 2차전지
        'KODEX_Ship': '000270.KS',         # 조선
        'KODEX_Steel': '117460.KS',        # 철강
        'KODEX_Auto': '091160.KS',         # 자동차
    },

    # 대형주 (시가총액 상위)
    'large_caps': {
        'Samsung_Electronics': '005930.KS',  # 삼성전자
        'SK_Hynix': '000660.KS',            # SK하이닉스
        'LG_Energy': '373220.KS',           # LG에너지솔루션
        'Samsung_Bio': '207940.KS',         # 삼성바이오로직스
        'Hyundai_Motor': '005380.KS',       # 현대차
        'NAVER': '035420.KS',               # 네이버
        'Kakao': '035720.KS',               # 카카오
        'POSCO': '005490.KS',               # 포스코홀딩스
    },

    # 채권 ETF
    'bond_etfs': {
        'KODEX_KTB3Y': '153130.KS',        # 국고채 3년
        'KODEX_KTB10Y': '148070.KS',       # 국고채 10년
        'KODEX_AAA': '219480.KS',          # AAA- 회사채
    },

    # 통화
    'currency': {
        'USDKRW': 'USDKRW=X',              # 달러-원 환율
    }
}


class KoreaDataCollector:
    """한국 시장 데이터 수집기"""

    def __init__(self, lookback_days: int = 365):
        """
        Args:
            lookback_days: 과거 데이터 기간 (기본 1년)
        """
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
        self.collection_status = {}

    def _fetch_ticker(self, ticker: str, name: str) -> Optional[pd.DataFrame]:
        """단일 티커 다운로드"""
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                logger.warning(f"  ✗ {name} ({ticker}): No data")
                return None

            # MultiIndex 처리
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            logger.info(f"  ✓ {name} ({ticker}): {len(data)} days")
            return data

        except Exception as e:
            logger.error(f"  ✗ {name} ({ticker}): {e}")
            return None

    def collect_indices(self) -> Dict[str, pd.DataFrame]:
        """지수 데이터 수집"""
        logger.info("[1/5] Collecting Korean Indices...")

        results = {}
        for name, ticker in KOSPI_CONFIG['indices'].items():
            data = self._fetch_ticker(ticker, name)
            if data is not None:
                results[name] = data

        return results

    def collect_sector_etfs(self) -> Dict[str, pd.DataFrame]:
        """섹터 ETF 데이터 수집"""
        logger.info("[2/5] Collecting Korean Sector ETFs...")

        results = {}
        for name, ticker in KOSPI_CONFIG['sector_etfs'].items():
            data = self._fetch_ticker(ticker, name)
            if data is not None:
                results[name] = data

        return results

    def collect_large_caps(self) -> Dict[str, pd.DataFrame]:
        """대형주 데이터 수집"""
        logger.info("[3/5] Collecting Korean Large Caps...")

        results = {}
        for name, ticker in KOSPI_CONFIG['large_caps'].items():
            data = self._fetch_ticker(ticker, name)
            if data is not None:
                results[name] = data

        return results

    def collect_bond_etfs(self) -> Dict[str, pd.DataFrame]:
        """채권 ETF 데이터 수집"""
        logger.info("[4/5] Collecting Korean Bond ETFs...")

        results = {}
        for name, ticker in KOSPI_CONFIG['bond_etfs'].items():
            data = self._fetch_ticker(ticker, name)
            if data is not None:
                results[name] = data

        return results

    def collect_currency(self) -> Dict[str, pd.DataFrame]:
        """환율 데이터 수집"""
        logger.info("[5/5] Collecting Currency (USDKRW)...")

        results = {}
        for name, ticker in KOSPI_CONFIG['currency'].items():
            data = self._fetch_ticker(ticker, name)
            if data is not None:
                results[name] = data

        return results

    def collect_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """전체 한국 자산 데이터 수집"""
        logger.info("\n" + "="*80)
        logger.info("KOREA MARKET DATA COLLECTION")
        logger.info("="*80)

        all_data = {
            'indices': self.collect_indices(),
            'sector_etfs': self.collect_sector_etfs(),
            'large_caps': self.collect_large_caps(),
            'bond_etfs': self.collect_bond_etfs(),
            'currency': self.collect_currency()
        }

        # Summary
        total_collected = sum(len(v) for v in all_data.values())
        total_expected = (
            len(KOSPI_CONFIG['indices']) +
            len(KOSPI_CONFIG['sector_etfs']) +
            len(KOSPI_CONFIG['large_caps']) +
            len(KOSPI_CONFIG['bond_etfs']) +
            len(KOSPI_CONFIG['currency'])
        )

        logger.info("\n" + "="*80)
        logger.info(f"Collection Summary: {total_collected}/{total_expected} successful")
        logger.info("="*80)

        return all_data

    def get_latest_prices(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, float]:
        """최신 가격 추출"""
        prices = {}

        for category, assets in data.items():
            for name, df in assets.items():
                if not df.empty and 'Close' in df.columns:
                    prices[f"{category}_{name}"] = float(df['Close'].iloc[-1])

        return prices

    def calculate_kospi_returns(
        self,
        kospi_data: pd.DataFrame,
        period: int = 252
    ) -> Dict[str, float]:
        """
        KOSPI 수익률 계산

        Args:
            kospi_data: KOSPI 지수 데이터
            period: 계산 기간 (기본 252일 = 1년)

        Returns:
            수익률 통계
        """
        if 'Close' not in kospi_data.columns or len(kospi_data) < period:
            return {}

        prices = kospi_data['Close']
        returns = prices.pct_change().dropna()

        # 최근 period 기간
        recent_returns = returns.tail(period)

        stats = {
            'total_return': (prices.iloc[-1] / prices.iloc[-period] - 1) * 100,
            'annualized_return': recent_returns.mean() * 252 * 100,
            'annualized_volatility': recent_returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': (recent_returns.mean() / recent_returns.std()) * np.sqrt(252) if recent_returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(prices.tail(period)) * 100,
            'current_price': float(prices.iloc[-1]),
            'period_days': period
        }

        return stats

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """최대 낙폭 계산"""
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        return drawdown.min()


# ============================================================================
# Parallel Korea Collector (Optional)
# ============================================================================

from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelKoreaCollector:
    """병렬 한국 데이터 수집기"""

    def __init__(self, lookback_days: int = 365, max_workers: int = 8):
        self.lookback_days = lookback_days
        self.max_workers = max_workers
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)

    def _fetch_single(self, ticker: str, name: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """단일 티커 다운로드 (thread-safe)"""
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,
                threads=False
            )

            if data.empty:
                return name, None

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return name, data

        except Exception:
            return name, None

    def collect_all(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """병렬 전체 수집"""
        logger.info("\n" + "="*80)
        logger.info("KOREA MARKET DATA COLLECTION (Parallel)")
        logger.info("="*80)

        # Flatten all tickers
        all_tasks = []
        for category, tickers in KOSPI_CONFIG.items():
            for name, ticker in tickers.items():
                all_tasks.append((category, name, ticker))

        results = {k: {} for k in KOSPI_CONFIG.keys()}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_single, ticker, name): (category, name)
                for category, name, ticker in all_tasks
            }

            for future in as_completed(futures):
                category, name = futures[future]
                try:
                    asset_name, data = future.result()
                    if data is not None:
                        results[category][asset_name] = data
                        logger.info(f"  ✓ {category}/{asset_name}")
                except Exception as e:
                    logger.error(f"  ✗ {category}/{name}: {e}")

        total_collected = sum(len(v) for v in results.values())
        logger.info(f"\n✓ Parallel collection: {total_collected}/{len(all_tasks)} successful")

        return results


# ============================================================================
# KOSPI vs SPX Comparison
# ============================================================================

def compare_kospi_vs_spx(
    kospi_data: pd.DataFrame,
    spx_data: pd.DataFrame,
    period: int = 252
) -> Dict:
    """
    KOSPI vs S&P 500 비교 분석

    Args:
        kospi_data: KOSPI 지수 데이터
        spx_data: SPX 지수 데이터
        period: 비교 기간 (기본 1년)

    Returns:
        비교 통계
    """
    if len(kospi_data) < period or len(spx_data) < period:
        return {}

    kospi_returns = kospi_data['Close'].pct_change().dropna().tail(period)
    spx_returns = spx_data['Close'].pct_change().dropna().tail(period)

    # Align dates
    common_dates = kospi_returns.index.intersection(spx_returns.index)
    kospi_returns = kospi_returns.loc[common_dates]
    spx_returns = spx_returns.loc[common_dates]

    comparison = {
        'kospi': {
            'return': kospi_returns.mean() * 252 * 100,
            'volatility': kospi_returns.std() * np.sqrt(252) * 100,
            'sharpe': (kospi_returns.mean() / kospi_returns.std()) * np.sqrt(252) if kospi_returns.std() > 0 else 0
        },
        'spx': {
            'return': spx_returns.mean() * 252 * 100,
            'volatility': spx_returns.std() * np.sqrt(252) * 100,
            'sharpe': (spx_returns.mean() / spx_returns.std()) * np.sqrt(252) if spx_returns.std() > 0 else 0
        },
        'correlation': kospi_returns.corr(spx_returns),
        'relative_performance': ((kospi_returns.mean() / spx_returns.mean()) - 1) * 100 if spx_returns.mean() != 0 else 0
    }

    return comparison


if __name__ == "__main__":
    # Test collection
    collector = KoreaDataCollector(lookback_days=365)
    data = collector.collect_all()

    # Print summary
    if 'indices' in data and 'KOSPI' in data['indices']:
        kospi_stats = collector.calculate_kospi_returns(data['indices']['KOSPI'])
        print("\n" + "="*80)
        print("KOSPI Statistics (1Y)")
        print("="*80)
        for key, value in kospi_stats.items():
            print(f"{key:>25}: {value:>10.2f}")
