#!/usr/bin/env python3
"""
EIMAS Data Loader - Infinite Assets
====================================
RWA(Real World Asset) 및 토큰화 자산 지원

경제학적 근거:
- "Asset이 infinite하다면... 모든 거래 가능한 걸 토큰화하기(Tokenization)"
- "블록체인 ETF, 희토류 채굴권 등 RWA(Real World Asset)가 중요해짐"

지원 자산 카테고리:
1. Market (주식/ETF)
2. Crypto (암호화폐)
3. RWA (토큰화 실물자산)
4. Commodity (원자재/귀금속)

Usage:
    from lib.data_loader import AssetLoader, AssetCategory, get_default_tickers

    loader = AssetLoader()
    tickers = get_default_tickers()
    data = loader.load_all(tickers)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger('eimas.data_loader')


# =============================================================================
# Asset Categories
# =============================================================================

class AssetCategory(str, Enum):
    """자산 카테고리"""
    MARKET = "market"           # 주식/ETF
    CRYPTO = "crypto"           # 암호화폐
    RWA = "rwa"                 # 토큰화 실물자산 (Real World Asset)
    COMMODITY = "commodity"     # 원자재/귀금속
    BOND = "bond"               # 채권
    MACRO = "macro"             # 매크로 지표


# =============================================================================
# Asset Definitions
# =============================================================================

@dataclass
class AssetInfo:
    """자산 정보"""
    ticker: str
    name: str
    category: AssetCategory
    subcategory: str = ""
    description: str = ""
    is_tokenized: bool = False
    underlying: str = ""        # 토큰화 자산의 기초자산

    def to_dict(self) -> Dict:
        return {
            'ticker': self.ticker,
            'name': self.name,
            'category': self.category.value,
            'subcategory': self.subcategory,
            'is_tokenized': self.is_tokenized,
            'underlying': self.underlying
        }


# =============================================================================
# Default Asset Registry
# =============================================================================

# 주요 지수 ETF
MARKET_ASSETS = {
    'SPY': AssetInfo('SPY', 'S&P 500 ETF', AssetCategory.MARKET, 'index_etf', 'US Large Cap'),
    'QQQ': AssetInfo('QQQ', 'Nasdaq 100 ETF', AssetCategory.MARKET, 'index_etf', 'Tech Heavy'),
    'IWM': AssetInfo('IWM', 'Russell 2000 ETF', AssetCategory.MARKET, 'index_etf', 'US Small Cap'),
    'DIA': AssetInfo('DIA', 'Dow Jones ETF', AssetCategory.MARKET, 'index_etf', 'US Blue Chip'),
}

# 섹터 ETF
SECTOR_ASSETS = {
    'XLK': AssetInfo('XLK', 'Technology Select', AssetCategory.MARKET, 'sector_etf', 'Technology'),
    'XLF': AssetInfo('XLF', 'Financial Select', AssetCategory.MARKET, 'sector_etf', 'Financials'),
    'XLE': AssetInfo('XLE', 'Energy Select', AssetCategory.MARKET, 'sector_etf', 'Energy'),
    'XLV': AssetInfo('XLV', 'Healthcare Select', AssetCategory.MARKET, 'sector_etf', 'Healthcare'),
    'XLI': AssetInfo('XLI', 'Industrial Select', AssetCategory.MARKET, 'sector_etf', 'Industrials'),
    'SMH': AssetInfo('SMH', 'Semiconductor ETF', AssetCategory.MARKET, 'sector_etf', 'Semiconductors'),
    'SOXX': AssetInfo('SOXX', 'iShares Semiconductor', AssetCategory.MARKET, 'sector_etf', 'Semiconductors'),
}

# 채권 ETF
BOND_ASSETS = {
    'TLT': AssetInfo('TLT', '20+ Year Treasury', AssetCategory.BOND, 'treasury', 'Long Duration'),
    'TIP': AssetInfo('TIP', 'TIPS Bond ETF', AssetCategory.BOND, 'tips', 'Inflation Protected'),
    'LQD': AssetInfo('LQD', 'Investment Grade Corp', AssetCategory.BOND, 'corporate', 'IG Corporate'),
    'HYG': AssetInfo('HYG', 'High Yield Corp', AssetCategory.BOND, 'high_yield', 'Junk Bonds'),
}

# 원자재
COMMODITY_ASSETS = {
    'GLD': AssetInfo('GLD', 'SPDR Gold Trust', AssetCategory.COMMODITY, 'precious_metal', 'Physical Gold Backed'),
    'USO': AssetInfo('USO', 'US Oil Fund', AssetCategory.COMMODITY, 'energy', 'Crude Oil Futures'),
    'UUP': AssetInfo('UUP', 'US Dollar Index', AssetCategory.COMMODITY, 'currency', 'USD Strength'),
}

# 암호화폐
CRYPTO_ASSETS = {
    'BTC-USD': AssetInfo('BTC-USD', 'Bitcoin', AssetCategory.CRYPTO, 'layer1', 'Digital Gold'),
    'ETH-USD': AssetInfo('ETH-USD', 'Ethereum', AssetCategory.CRYPTO, 'layer1', 'Smart Contract Platform'),
    'SOL-USD': AssetInfo('SOL-USD', 'Solana', AssetCategory.CRYPTO, 'layer1', 'High Performance Chain'),
}

# RWA (Real World Asset) - 토큰화 자산
# Note: ONDO-USD, PAXG-USD는 yfinance 호환 형식 (crypto tokens)
RWA_ASSETS = {
    'ONDO-USD': AssetInfo(
        'ONDO-USD', 'Ondo Finance', AssetCategory.RWA, 'tokenized_treasury',
        'US Treasuries Tokenized Protocol',
        is_tokenized=True, underlying='US Treasury Bonds'
    ),
    'PAXG-USD': AssetInfo(
        'PAXG-USD', 'Pax Gold', AssetCategory.RWA, 'tokenized_gold',
        'Gold Tokenized (1 token = 1 oz Gold)',
        is_tokenized=True, underlying='Physical Gold'
    ),
    'COIN': AssetInfo(
        'COIN', 'Coinbase Global', AssetCategory.RWA, 'crypto_infra',
        'Crypto Infrastructure Proxy (Exchange)',
        is_tokenized=False, underlying='Crypto Market'
    ),
}

# 매크로 지표
MACRO_INDICATORS = {
    '^VIX': AssetInfo('^VIX', 'VIX Volatility Index', AssetCategory.MACRO, 'volatility', 'Fear Gauge'),
}


# =============================================================================
# Combined Asset Registry
# =============================================================================

ALL_ASSETS: Dict[str, AssetInfo] = {
    **MARKET_ASSETS,
    **SECTOR_ASSETS,
    **BOND_ASSETS,
    **COMMODITY_ASSETS,
    **CRYPTO_ASSETS,
    **RWA_ASSETS,
    **MACRO_INDICATORS,
}


def get_default_tickers() -> Dict[str, List[Dict]]:
    """
    기본 티커 설정 반환 (main.py 호환 형식)

    Returns:
        {'market': [...], 'crypto': [...], 'rwa': [...]} 형태
    """
    return {
        'market': [
            # 주요 지수 ETF
            {'ticker': 'SPY'}, {'ticker': 'QQQ'}, {'ticker': 'IWM'},
            {'ticker': 'DIA'}, {'ticker': 'TLT'}, {'ticker': 'GLD'},
            {'ticker': 'USO'}, {'ticker': 'UUP'}, {'ticker': '^VIX'},
            # 섹터 ETF
            {'ticker': 'XLK'}, {'ticker': 'XLF'}, {'ticker': 'XLE'},
            {'ticker': 'XLV'}, {'ticker': 'XLI'},
            # 반도체 ETF
            {'ticker': 'SMH'}, {'ticker': 'SOXX'},
            # 채권 ETF
            {'ticker': 'TIP'}, {'ticker': 'LQD'}, {'ticker': 'HYG'},
        ],
        'crypto': [
            {'ticker': 'BTC-USD'}, {'ticker': 'ETH-USD'},
        ],
        'rwa': [
            # RWA 및 토큰화 자산 (NEW) - yfinance 호환 형식
            {'ticker': 'ONDO-USD'},   # US Treasuries Tokenized
            {'ticker': 'PAXG-USD'},   # Gold Tokenized
            {'ticker': 'COIN'},       # Crypto Infrastructure Proxy (주식)
        ]
    }


def get_asset_info(ticker: str) -> Optional[AssetInfo]:
    """티커의 자산 정보 반환"""
    return ALL_ASSETS.get(ticker)


def get_category(ticker: str) -> AssetCategory:
    """티커의 카테고리 반환"""
    info = ALL_ASSETS.get(ticker)
    if info:
        return info.category

    # 알려지지 않은 티커 추론
    if ticker.endswith('-USD'):
        return AssetCategory.CRYPTO
    elif ticker.startswith('^'):
        return AssetCategory.MACRO
    else:
        return AssetCategory.MARKET


# =============================================================================
# Asset Loader
# =============================================================================

class AssetLoader:
    """
    통합 자산 데이터 로더

    RWA, Crypto, 전통 자산을 통합 관리
    """

    def __init__(self, lookback_days: int = 365):
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
        self.collection_log: List[Dict] = []

    def load_single(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        단일 자산 데이터 로드

        Returns:
            (DataFrame, status_dict)
        """
        asset_info = get_asset_info(ticker)
        category = asset_info.category if asset_info else get_category(ticker)

        status = {
            'ticker': ticker,
            'category': category.value,
            'is_tokenized': asset_info.is_tokenized if asset_info else False,
            'success': False,
            'error': None,
            'data_points': 0,
            'timestamp': datetime.now().isoformat()
        }

        try:
            # RWA 자산 특별 처리
            if category == AssetCategory.RWA:
                data = self._load_rwa_asset(ticker)
            else:
                data = self._load_via_yfinance(ticker)

            if data is not None and not data.empty:
                status['success'] = True
                status['data_points'] = len(data)
                self.collection_log.append(status)
                return data, status
            else:
                status['error'] = 'Empty data returned'

        except Exception as e:
            status['error'] = str(e)
            logger.warning(f"Failed to load {ticker}: {e}")

        self.collection_log.append(status)
        return None, status

    def _load_via_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """yfinance를 통한 데이터 로드"""
        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                return None

            # MultiIndex 처리
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return data

        except Exception as e:
            logger.warning(f"yfinance error for {ticker}: {e}")
            return None

    def _load_rwa_asset(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        RWA 자산 로드 (특별 처리)

        일부 RWA 토큰은 yfinance에서 직접 지원되지 않을 수 있음
        """
        # ONDO, PAXG는 yfinance에서 지원됨
        if ticker in ['ONDO', 'PAXG', 'COIN']:
            return self._load_via_yfinance(ticker)

        # 추후 다른 RWA 소스 추가 가능 (CoinGecko, Chainlink 등)
        logger.warning(f"RWA ticker {ticker} not supported yet")
        return None

    def load_all(
        self,
        tickers_config: Dict[str, List[Dict]]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        모든 자산 데이터 로드

        Args:
            tickers_config: {'market': [...], 'crypto': [...], 'rwa': [...]}

        Returns:
            (data_dict, summary)
        """
        all_data = {}
        summary = {
            'total_requested': 0,
            'total_loaded': 0,
            'by_category': {},
            'rwa_assets': [],
            'tokenized_assets': [],
            'failed': []
        }

        for category, tickers in tickers_config.items():
            summary['by_category'][category] = {'requested': 0, 'loaded': 0}

            for ticker_info in tickers:
                ticker = ticker_info.get('ticker') if isinstance(ticker_info, dict) else ticker_info
                summary['total_requested'] += 1
                summary['by_category'][category]['requested'] += 1

                data, status = self.load_single(ticker)

                if data is not None:
                    all_data[ticker] = data
                    summary['total_loaded'] += 1
                    summary['by_category'][category]['loaded'] += 1

                    # RWA 및 토큰화 자산 추적
                    asset_info = get_asset_info(ticker)
                    if asset_info:
                        if asset_info.category == AssetCategory.RWA:
                            summary['rwa_assets'].append(ticker)
                        if asset_info.is_tokenized:
                            summary['tokenized_assets'].append(ticker)
                else:
                    summary['failed'].append(ticker)

        return all_data, summary

    def get_collection_summary(self) -> Dict:
        """수집 요약 반환"""
        success_count = sum(1 for log in self.collection_log if log['success'])

        by_category = {}
        for log in self.collection_log:
            cat = log['category']
            if cat not in by_category:
                by_category[cat] = {'total': 0, 'success': 0}
            by_category[cat]['total'] += 1
            if log['success']:
                by_category[cat]['success'] += 1

        return {
            'total': len(self.collection_log),
            'success': success_count,
            'failed': len(self.collection_log) - success_count,
            'by_category': by_category,
            'tokenized_count': sum(1 for log in self.collection_log
                                   if log.get('is_tokenized') and log['success'])
        }


# =============================================================================
# Utility Functions
# =============================================================================

def categorize_ticker(ticker: str) -> str:
    """
    티커를 카테고리로 분류 (main.py 호환)

    Returns:
        'market', 'crypto', 'rwa', 'commodity', 'bond', 'macro'
    """
    info = get_asset_info(ticker)
    if info:
        return info.category.value

    # 휴리스틱 분류
    if ticker.endswith('-USD'):
        return 'crypto'
    elif ticker.startswith('^'):
        return 'macro'
    elif ticker in ['GLD', 'USO', 'UUP', 'PAXG']:
        return 'commodity'
    elif ticker in ['TLT', 'TIP', 'LQD', 'HYG']:
        return 'bond'
    elif ticker in ['ONDO', 'COIN']:
        return 'rwa'
    else:
        return 'market'


def is_tokenized_asset(ticker: str) -> bool:
    """토큰화 자산 여부 확인"""
    info = get_asset_info(ticker)
    return info.is_tokenized if info else False


def get_rwa_tickers() -> List[str]:
    """RWA 티커 목록 반환"""
    return list(RWA_ASSETS.keys())


def get_all_tickers_flat() -> List[str]:
    """모든 티커를 flat 리스트로 반환"""
    config = get_default_tickers()
    all_tickers = []
    for category_tickers in config.values():
        for t in category_tickers:
            ticker = t.get('ticker') if isinstance(t, dict) else t
            all_tickers.append(ticker)
    return all_tickers


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EIMAS Data Loader - RWA & Tokenized Assets")
    print("=" * 60)

    # 1. 기본 티커 설정 확인
    print("\n[1] Default Tickers Configuration:")
    tickers = get_default_tickers()
    for category, ticker_list in tickers.items():
        print(f"    {category}: {len(ticker_list)} tickers")
        for t in ticker_list:
            ticker = t.get('ticker')
            info = get_asset_info(ticker)
            if info and info.is_tokenized:
                print(f"      → {ticker} [TOKENIZED] - {info.description}")
            elif info and info.category == AssetCategory.RWA:
                print(f"      → {ticker} [RWA] - {info.description}")

    # 2. RWA 자산 정보
    print("\n[2] RWA Assets Detail:")
    for ticker, info in RWA_ASSETS.items():
        print(f"    {ticker}:")
        print(f"      Name: {info.name}")
        print(f"      Category: {info.category.value}/{info.subcategory}")
        print(f"      Tokenized: {info.is_tokenized}")
        print(f"      Underlying: {info.underlying}")

    # 3. 데이터 로드 테스트
    print("\n[3] Loading RWA Data (30 days)...")
    loader = AssetLoader(lookback_days=30)

    rwa_tickers = {'rwa': [{'ticker': t} for t in get_rwa_tickers()]}
    data, summary = loader.load_all(rwa_tickers)

    print(f"    Loaded: {summary['total_loaded']}/{summary['total_requested']}")
    for ticker, df in data.items():
        if df is not None:
            latest_price = df['Close'].iloc[-1] if 'Close' in df.columns else 'N/A'
            print(f"    → {ticker}: {len(df)} days, Latest: ${latest_price:.2f}")

    if summary['failed']:
        print(f"    Failed: {', '.join(summary['failed'])}")

    print("\n" + "=" * 60)
    print("Data Loader Test Complete!")
    print("=" * 60)
