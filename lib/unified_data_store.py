#!/usr/bin/env python3
"""
Unified Data Store
==================
모든 데이터 수집기를 DB와 통합하는 저장소

수집 대상:
1. 주식/ETF 일별 가격 (yfinance)
2. ETF 구성 및 비중 (섹터별, 상위 종목)
3. FRED 거시지표 (금리, 인플레이션, 고용 등)
4. 암호화폐 가격 (BTC, ETH 등)
5. 시장 레짐 및 신호

DB 테이블:
- daily_prices: 일별 가격 (OHLCV)
- etf_composition: ETF 구성 종목 및 비중
- fred_indicators: FRED 거시지표
- crypto_prices: 암호화폐 가격
- market_snapshots: 일별 시장 스냅샷 (종합)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import sqlite3
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from core.database import DatabaseManager

# ETF Universe
ETF_UNIVERSE = {
    "market": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
    "style": ["VUG", "VTV", "IWF", "IWD", "MTUM", "QUAL"],
    "size": ["SPY", "IJH", "IWM", "IJR"],
    "sector": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"],
    "bond": ["AGG", "TLT", "IEF", "SHY", "HYG", "LQD", "TIP", "BND"],
    "alternative": ["GLD", "SLV", "USO", "DBC", "PDBC"],
    "global": ["EFA", "EEM", "VEU", "FXI", "EWJ", "EWY", "EWZ"],
    "thematic": ["ARKK", "ARKW", "ARKG", "SOXX", "XBI", "ICLN", "TAN", "BOTZ"],
    "volatility": ["VXX", "UVXY", "SVXY"],
}

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD"]

# 주요 개별 주식
MAJOR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "UNH", "HD", "PG", "MA", "XOM", "CVX",
]

# FRED 시리즈
FRED_SERIES = {
    # 금리
    "DFF": "Fed Funds Rate",
    "DGS2": "2Y Treasury",
    "DGS10": "10Y Treasury",
    "DGS30": "30Y Treasury",
    "DFII10": "10Y TIPS",
    # 스프레드
    "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread",
    "BAMLH0A0HYM2": "HY Spread",
    # 인플레이션
    "CPIAUCSL": "CPI",
    "PCEPILFE": "Core PCE",
    "T10YIE": "10Y Breakeven",
    # 고용
    "UNRATE": "Unemployment",
    "ICSA": "Initial Claims",
    # 경제활동
    "INDPRO": "Industrial Production",
    "UMCSENT": "Consumer Sentiment",
    # VIX
    "VIXCLS": "VIX",
}


class UnifiedDataStore:
    """
    통합 데이터 저장소

    사용법:
        store = UnifiedDataStore()
        store.collect_and_store_all()  # 전체 수집 및 저장

        # 조회
        prices = store.get_daily_prices(ticker="SPY", days=30)
        etf_comp = store.get_etf_composition(etf="SPY")
    """

    def __init__(self, db_path: str = None):
        """
        Args:
            db_path: DB 경로 (기본: data/eimas.db)
        """
        self.db = DatabaseManager(db_path) if db_path else DatabaseManager()
        self._init_extended_tables()

    def _init_extended_tables(self):
        """확장 테이블 생성"""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            # ================================================================
            # 일별 가격 테이블 (OHLCV)
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    asset_type TEXT,  -- stock, etf, crypto, index
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    adj_close REAL,
                    change_pct REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker ON daily_prices(ticker)")

            # ================================================================
            # ETF 구성 테이블 (보유종목, 섹터 비중)
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS etf_composition (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    etf TEXT NOT NULL,
                    composition_type TEXT,  -- holdings, sector, country
                    item_name TEXT,         -- 종목명 또는 섹터명
                    item_ticker TEXT,       -- 종목 티커 (있을 경우)
                    weight REAL,            -- 비중 (%)
                    rank INTEGER,           -- 순위
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, etf, composition_type, item_name)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_etf_comp_date ON etf_composition(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_etf_comp_etf ON etf_composition(etf)")

            # ================================================================
            # FRED 거시지표 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fred_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    series_id TEXT NOT NULL,
                    series_name TEXT,
                    value REAL,
                    prev_value REAL,
                    change REAL,
                    change_pct REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, series_id)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fred_date ON fred_indicators(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fred_series ON fred_indicators(series_id)")

            # ================================================================
            # 암호화폐 가격 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS crypto_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    name TEXT,
                    price REAL,
                    market_cap REAL,
                    volume_24h REAL,
                    change_24h REAL,
                    change_7d REAL,
                    high_24h REAL,
                    low_24h REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_crypto_date ON crypto_prices(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_crypto_ticker ON crypto_prices(ticker)")

            # ================================================================
            # 일별 시장 스냅샷 (종합)
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    -- 주요 지수
                    spy_close REAL,
                    spy_change REAL,
                    qqq_close REAL,
                    qqq_change REAL,
                    iwm_close REAL,
                    iwm_change REAL,
                    -- 변동성
                    vix REAL,
                    vix_change REAL,
                    -- 채권
                    tlt_close REAL,
                    yield_10y REAL,
                    yield_2y REAL,
                    yield_curve REAL,
                    -- 크립토
                    btc_price REAL,
                    btc_change REAL,
                    eth_price REAL,
                    -- 섹터 리더/래거드
                    sector_leader TEXT,
                    sector_laggard TEXT,
                    -- 스타일
                    growth_value_spread REAL,
                    large_small_spread REAL,
                    -- 달러
                    dxy REAL,
                    -- 메타데이터
                    data_quality TEXT,  -- complete, partial, missing
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ================================================================
            # ETF 성과 비교 테이블
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS etf_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    category TEXT,        -- market, sector, style, etc.
                    return_1d REAL,
                    return_5d REAL,
                    return_20d REAL,
                    return_60d REAL,
                    volatility_20d REAL,
                    volume_ratio REAL,    -- vs 20d avg
                    relative_strength REAL, -- vs SPY
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker)
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_etf_perf_date ON etf_performance(date)")

    # ========================================================================
    # 가격 데이터 수집 및 저장
    # ========================================================================

    def collect_prices(self, tickers: List[str], period: str = "3mo") -> Dict[str, pd.DataFrame]:
        """가격 데이터 수집"""
        print(f"  Collecting prices for {len(tickers)} tickers...")

        try:
            data = yf.download(tickers, period=period, progress=False, group_by='ticker')

            result = {}
            if len(tickers) == 1:
                result[tickers[0]] = data
            else:
                for ticker in tickers:
                    if ticker in data.columns.get_level_values(0):
                        ticker_data = data[ticker].dropna(how='all')
                        if not ticker_data.empty:
                            result[ticker] = ticker_data

            return result
        except Exception as e:
            print(f"    Error: {e}")
            return {}

    def save_daily_prices(self, prices: Dict[str, pd.DataFrame],
                          asset_type: str = "etf") -> int:
        """일별 가격 저장"""
        count = 0

        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            for ticker, df in prices.items():
                if df.empty:
                    continue

                for date_idx, row in df.iterrows():
                    date_str = date_idx.strftime("%Y-%m-%d")

                    close = float(row.get('Close', 0)) if pd.notna(row.get('Close')) else 0
                    prev_close = float(df['Close'].shift(1).loc[date_idx]) if date_idx != df.index[0] else close
                    change_pct = ((close / prev_close) - 1) * 100 if prev_close > 0 else 0

                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO daily_prices
                            (date, ticker, asset_type, open, high, low, close, volume, adj_close, change_pct)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            date_str,
                            ticker,
                            asset_type,
                            float(row.get('Open', 0)) if pd.notna(row.get('Open')) else None,
                            float(row.get('High', 0)) if pd.notna(row.get('High')) else None,
                            float(row.get('Low', 0)) if pd.notna(row.get('Low')) else None,
                            close,
                            float(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None,
                            float(row.get('Adj Close', close)) if pd.notna(row.get('Adj Close', close)) else close,
                            round(change_pct, 4),
                        ))
                        count += 1
                    except Exception as e:
                        continue

        return count

    # ========================================================================
    # ETF 구성 수집 및 저장
    # ========================================================================

    def collect_etf_info(self, etf_ticker: str) -> Dict[str, Any]:
        """ETF 상세 정보 수집"""
        try:
            etf = yf.Ticker(etf_ticker)
            info = etf.info

            result = {
                'ticker': etf_ticker,
                'name': info.get('longName', ''),
                'category': info.get('category', ''),
                'total_assets': info.get('totalAssets', 0),
                'nav': info.get('navPrice', 0),
                'yield': info.get('yield', 0),
                'expense_ratio': info.get('annualReportExpenseRatio', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'holdings': [],
                'sector_weights': {},
            }

            # 상위 보유종목
            try:
                holdings = etf.get_holdings()
                if holdings is not None and not holdings.empty:
                    for idx, row in holdings.head(15).iterrows():
                        result['holdings'].append({
                            'ticker': row.get('Symbol', ''),
                            'name': row.get('Name', idx),
                            'weight': float(row.get('% Assets', 0)),
                        })
            except:
                pass

            # 섹터 비중
            try:
                sector_weights = info.get('sectorWeightings', {})
                if sector_weights:
                    for sector in sector_weights:
                        for key, value in sector.items():
                            result['sector_weights'][key] = round(value * 100, 2)
            except:
                pass

            return result

        except Exception as e:
            print(f"    Error fetching {etf_ticker}: {e}")
            return {'ticker': etf_ticker, 'holdings': [], 'sector_weights': {}}

    def save_etf_composition(self, etf_info: Dict[str, Any], date_str: str = None) -> int:
        """ETF 구성 저장"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        count = 0

        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            etf_ticker = etf_info.get('ticker', '')

            # 상위 보유종목 저장
            for i, holding in enumerate(etf_info.get('holdings', [])):
                cursor.execute("""
                    INSERT OR REPLACE INTO etf_composition
                    (date, etf, composition_type, item_name, item_ticker, weight, rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_str,
                    etf_ticker,
                    'holdings',
                    holding.get('name', ''),
                    holding.get('ticker', ''),
                    holding.get('weight', 0),
                    i + 1,
                ))
                count += 1

            # 섹터 비중 저장
            sector_weights = etf_info.get('sector_weights', {})
            for i, (sector, weight) in enumerate(sorted(sector_weights.items(),
                                                         key=lambda x: x[1], reverse=True)):
                cursor.execute("""
                    INSERT OR REPLACE INTO etf_composition
                    (date, etf, composition_type, item_name, item_ticker, weight, rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_str,
                    etf_ticker,
                    'sector',
                    sector,
                    '',
                    weight,
                    i + 1,
                ))
                count += 1

        return count

    # ========================================================================
    # ETF 성과 계산 및 저장
    # ========================================================================

    def calculate_etf_performance(self, prices: Dict[str, pd.DataFrame]) -> List[Dict]:
        """ETF 성과 지표 계산"""
        results = []

        # SPY를 기준으로
        spy_data = prices.get('SPY')
        if spy_data is None or spy_data.empty:
            return results

        for ticker, df in prices.items():
            if df.empty or 'Close' not in df.columns:
                continue

            try:
                close = df['Close']
                latest_date = close.index[-1].strftime("%Y-%m-%d")

                # 수익률 계산
                return_1d = ((close.iloc[-1] / close.iloc[-2]) - 1) * 100 if len(close) > 1 else 0
                return_5d = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) > 5 else 0
                return_20d = ((close.iloc[-1] / close.iloc[-21]) - 1) * 100 if len(close) > 20 else 0
                return_60d = ((close.iloc[-1] / close.iloc[-61]) - 1) * 100 if len(close) > 60 else 0

                # 변동성
                if len(close) > 20:
                    returns = close.pct_change().dropna()
                    volatility_20d = returns.tail(20).std() * np.sqrt(252) * 100
                else:
                    volatility_20d = 0

                # 거래량 비율
                if 'Volume' in df.columns:
                    vol = df['Volume']
                    vol_avg = vol.tail(20).mean()
                    volume_ratio = vol.iloc[-1] / vol_avg if vol_avg > 0 else 1
                else:
                    volume_ratio = 1

                # SPY 대비 상대 강도
                relative_strength = 0
                if spy_data is not None and len(spy_data) > 20:
                    spy_return = ((spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-21]) - 1) * 100
                    relative_strength = return_20d - spy_return

                # 카테고리 찾기
                category = "other"
                for cat, tickers in ETF_UNIVERSE.items():
                    if ticker in tickers:
                        category = cat
                        break

                results.append({
                    'date': latest_date,
                    'ticker': ticker,
                    'category': category,
                    'return_1d': round(return_1d, 4),
                    'return_5d': round(return_5d, 4),
                    'return_20d': round(return_20d, 4),
                    'return_60d': round(return_60d, 4),
                    'volatility_20d': round(volatility_20d, 2),
                    'volume_ratio': round(volume_ratio, 2),
                    'relative_strength': round(relative_strength, 4),
                })

            except Exception as e:
                continue

        return results

    def save_etf_performance(self, performances: List[Dict]) -> int:
        """ETF 성과 저장"""
        count = 0

        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            for perf in performances:
                cursor.execute("""
                    INSERT OR REPLACE INTO etf_performance
                    (date, ticker, category, return_1d, return_5d, return_20d, return_60d,
                     volatility_20d, volume_ratio, relative_strength)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    perf['date'],
                    perf['ticker'],
                    perf['category'],
                    perf['return_1d'],
                    perf['return_5d'],
                    perf['return_20d'],
                    perf['return_60d'],
                    perf['volatility_20d'],
                    perf['volume_ratio'],
                    perf['relative_strength'],
                ))
                count += 1

        return count

    # ========================================================================
    # 암호화폐 수집 및 저장
    # ========================================================================

    def collect_and_save_crypto(self) -> int:
        """암호화폐 데이터 수집 및 저장"""
        today = datetime.now().strftime("%Y-%m-%d")
        count = 0

        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            for ticker in CRYPTO_TICKERS:
                try:
                    crypto = yf.Ticker(ticker)
                    hist = crypto.history(period="7d")

                    if hist.empty:
                        continue

                    close = float(hist['Close'].iloc[-1])
                    prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else close
                    week_ago = float(hist['Close'].iloc[0]) if len(hist) >= 7 else close

                    change_24h = ((close / prev_close) - 1) * 100 if prev_close > 0 else 0
                    change_7d = ((close / week_ago) - 1) * 100 if week_ago > 0 else 0

                    cursor.execute("""
                        INSERT OR REPLACE INTO crypto_prices
                        (date, ticker, name, price, change_24h, change_7d, high_24h, low_24h, volume_24h)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        today,
                        ticker.replace('-USD', ''),
                        ticker.split('-')[0],
                        round(close, 2),
                        round(change_24h, 2),
                        round(change_7d, 2),
                        float(hist['High'].iloc[-1]) if 'High' in hist else None,
                        float(hist['Low'].iloc[-1]) if 'Low' in hist else None,
                        float(hist['Volume'].iloc[-1]) if 'Volume' in hist else None,
                    ))
                    count += 1

                except Exception as e:
                    continue

        return count

    # ========================================================================
    # 시장 스냅샷
    # ========================================================================

    def create_market_snapshot(self, prices: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """일별 시장 스냅샷 생성"""
        today = datetime.now().strftime("%Y-%m-%d")

        def get_latest(ticker: str, field: str = 'Close') -> Tuple[float, float]:
            if ticker not in prices or prices[ticker].empty:
                return 0.0, 0.0
            df = prices[ticker]
            if field not in df.columns:
                return 0.0, 0.0
            close = float(df[field].iloc[-1])
            prev = float(df[field].iloc[-2]) if len(df) > 1 else close
            change = ((close / prev) - 1) * 100 if prev > 0 else 0
            return close, round(change, 2)

        spy_close, spy_change = get_latest('SPY')
        qqq_close, qqq_change = get_latest('QQQ')
        iwm_close, iwm_change = get_latest('IWM')

        # VIX
        vix, vix_change = 0, 0
        try:
            vix_data = yf.download('^VIX', period='5d', progress=False)
            if not vix_data.empty:
                vix = float(vix_data['Close'].iloc[-1])
                vix_prev = float(vix_data['Close'].iloc[-2]) if len(vix_data) > 1 else vix
                vix_change = ((vix / vix_prev) - 1) * 100
        except:
            pass

        # 채권
        tlt_close, _ = get_latest('TLT')

        # 수익률 곡선 (가격 기반 추정)
        shy_close, _ = get_latest('SHY')
        ief_close, _ = get_latest('IEF')
        yield_curve = 0
        if shy_close > 0 and ief_close > 0:
            # 가격이 낮을수록 수익률이 높음
            yield_curve = (shy_close / ief_close - 1) * 10  # 대략적 추정

        # 성장 vs 가치
        vug_close, _ = get_latest('VUG')
        vtv_close, _ = get_latest('VTV')
        growth_value_spread = 0
        if vug_close > 0 and vtv_close > 0:
            # 20일 수익률 비교
            if 'VUG' in prices and 'VTV' in prices:
                vug_df = prices['VUG']
                vtv_df = prices['VTV']
                if len(vug_df) > 20 and len(vtv_df) > 20:
                    vug_ret = (vug_df['Close'].iloc[-1] / vug_df['Close'].iloc[-21] - 1) * 100
                    vtv_ret = (vtv_df['Close'].iloc[-1] / vtv_df['Close'].iloc[-21] - 1) * 100
                    growth_value_spread = vug_ret - vtv_ret

        # 대형 vs 소형
        large_small_spread = 0
        if spy_close > 0 and iwm_close > 0:
            if 'SPY' in prices and 'IWM' in prices:
                spy_df = prices['SPY']
                iwm_df = prices['IWM']
                if len(spy_df) > 20 and len(iwm_df) > 20:
                    spy_ret = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-21] - 1) * 100
                    iwm_ret = (iwm_df['Close'].iloc[-1] / iwm_df['Close'].iloc[-21] - 1) * 100
                    large_small_spread = spy_ret - iwm_ret

        # 섹터 리더/래거드
        sector_returns = {}
        for sector_etf in ETF_UNIVERSE['sector']:
            if sector_etf in prices and not prices[sector_etf].empty:
                df = prices[sector_etf]
                if len(df) > 5:
                    ret = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1) * 100
                    sector_returns[sector_etf] = ret

        sector_leader = max(sector_returns, key=sector_returns.get) if sector_returns else ""
        sector_laggard = min(sector_returns, key=sector_returns.get) if sector_returns else ""

        # 크립토
        btc_price, btc_change = 0, 0
        eth_price = 0
        try:
            btc = yf.download('BTC-USD', period='5d', progress=False)
            if not btc.empty:
                btc_price = float(btc['Close'].iloc[-1])
                btc_prev = float(btc['Close'].iloc[-2]) if len(btc) > 1 else btc_price
                btc_change = ((btc_price / btc_prev) - 1) * 100
            eth = yf.download('ETH-USD', period='2d', progress=False)
            if not eth.empty:
                eth_price = float(eth['Close'].iloc[-1])
        except:
            pass

        # DXY
        dxy = 0
        try:
            dxy_data = yf.download('DX-Y.NYB', period='2d', progress=False)
            if not dxy_data.empty:
                dxy = float(dxy_data['Close'].iloc[-1])
        except:
            pass

        # 데이터 품질
        data_quality = "complete"
        if spy_close == 0 or vix == 0:
            data_quality = "partial"
        if spy_close == 0 and qqq_close == 0:
            data_quality = "missing"

        return {
            'date': today,
            'spy_close': round(spy_close, 2),
            'spy_change': round(spy_change, 2),
            'qqq_close': round(qqq_close, 2),
            'qqq_change': round(qqq_change, 2),
            'iwm_close': round(iwm_close, 2),
            'iwm_change': round(iwm_change, 2),
            'vix': round(vix, 2),
            'vix_change': round(vix_change, 2),
            'tlt_close': round(tlt_close, 2),
            'yield_10y': 0,  # FRED에서 가져와야 함
            'yield_2y': 0,
            'yield_curve': round(yield_curve, 3),
            'btc_price': round(btc_price, 2),
            'btc_change': round(btc_change, 2),
            'eth_price': round(eth_price, 2),
            'sector_leader': sector_leader,
            'sector_laggard': sector_laggard,
            'growth_value_spread': round(growth_value_spread, 2),
            'large_small_spread': round(large_small_spread, 2),
            'dxy': round(dxy, 2),
            'data_quality': data_quality,
        }

    def save_market_snapshot(self, snapshot: Dict[str, Any]) -> bool:
        """시장 스냅샷 저장"""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO market_snapshots
                (date, spy_close, spy_change, qqq_close, qqq_change, iwm_close, iwm_change,
                 vix, vix_change, tlt_close, yield_10y, yield_2y, yield_curve,
                 btc_price, btc_change, eth_price, sector_leader, sector_laggard,
                 growth_value_spread, large_small_spread, dxy, data_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot['date'],
                snapshot['spy_close'],
                snapshot['spy_change'],
                snapshot['qqq_close'],
                snapshot['qqq_change'],
                snapshot['iwm_close'],
                snapshot['iwm_change'],
                snapshot['vix'],
                snapshot['vix_change'],
                snapshot['tlt_close'],
                snapshot['yield_10y'],
                snapshot['yield_2y'],
                snapshot['yield_curve'],
                snapshot['btc_price'],
                snapshot['btc_change'],
                snapshot['eth_price'],
                snapshot['sector_leader'],
                snapshot['sector_laggard'],
                snapshot['growth_value_spread'],
                snapshot['large_small_spread'],
                snapshot['dxy'],
                snapshot['data_quality'],
            ))
        return True

    # ========================================================================
    # 전체 수집 실행
    # ========================================================================

    def collect_and_store_all(self, include_composition: bool = True) -> Dict[str, int]:
        """전체 데이터 수집 및 저장"""
        print("=" * 60)
        print("Unified Data Store - Full Collection")
        print("=" * 60)

        stats = {
            'daily_prices': 0,
            'etf_performance': 0,
            'etf_composition': 0,
            'crypto': 0,
            'snapshot': 0,
        }

        # 1. ETF 가격 수집
        print("\n[1/5] Collecting ETF prices...")
        all_etf_tickers = []
        for category, tickers in ETF_UNIVERSE.items():
            all_etf_tickers.extend(tickers)
        all_etf_tickers = list(set(all_etf_tickers))

        etf_prices = self.collect_prices(all_etf_tickers, period="3mo")
        stats['daily_prices'] += self.save_daily_prices(etf_prices, asset_type="etf")
        print(f"    Saved {stats['daily_prices']} ETF price records")

        # 2. 주요 주식 가격
        print("\n[2/5] Collecting stock prices...")
        stock_prices = self.collect_prices(MAJOR_STOCKS, period="3mo")
        stock_count = self.save_daily_prices(stock_prices, asset_type="stock")
        stats['daily_prices'] += stock_count
        print(f"    Saved {stock_count} stock price records")

        # 3. ETF 성과 계산 및 저장
        print("\n[3/5] Calculating ETF performance...")
        all_prices = {**etf_prices, **stock_prices}
        performances = self.calculate_etf_performance(etf_prices)
        stats['etf_performance'] = self.save_etf_performance(performances)
        print(f"    Saved {stats['etf_performance']} performance records")

        # 4. ETF 구성 (선택적, API 제한 주의)
        if include_composition:
            print("\n[4/5] Collecting ETF composition (top 10 ETFs)...")
            important_etfs = ["SPY", "QQQ", "IWM", "VUG", "VTV", "XLK", "XLF", "XLE", "TLT", "HYG"]
            for etf in important_etfs:
                info = self.collect_etf_info(etf)
                count = self.save_etf_composition(info)
                stats['etf_composition'] += count
                print(f"    {etf}: {count} records")
        else:
            print("\n[4/5] Skipping ETF composition...")

        # 5. 암호화폐
        print("\n[5/5] Collecting crypto prices...")
        stats['crypto'] = self.collect_and_save_crypto()
        print(f"    Saved {stats['crypto']} crypto records")

        # 6. 시장 스냅샷
        print("\n[Bonus] Creating market snapshot...")
        snapshot = self.create_market_snapshot(all_prices)
        self.save_market_snapshot(snapshot)
        stats['snapshot'] = 1
        print(f"    Snapshot saved for {snapshot['date']}")

        # 로그
        self.db.log_analysis('unified_data_store', 'SUCCESS',
                            records=sum(stats.values()))

        print("\n" + "=" * 60)
        print("Collection Complete!")
        print("=" * 60)
        print(f"\nTotal records saved: {sum(stats.values())}")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        return stats

    # ========================================================================
    # 조회 메서드
    # ========================================================================

    def get_daily_prices(self, ticker: str = None, days: int = 30,
                         asset_type: str = None) -> pd.DataFrame:
        """일별 가격 조회"""
        with self.db._get_connection() as conn:
            query = "SELECT * FROM daily_prices WHERE 1=1"
            params = []

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker)

            if asset_type:
                query += " AND asset_type = ?"
                params.append(asset_type)

            query += f" AND date >= date('now', '-{days} days')"
            query += " ORDER BY date DESC, ticker"

            df = pd.read_sql_query(query, conn, params=params)
            return df

    def get_etf_composition(self, etf: str, date_str: str = None) -> pd.DataFrame:
        """ETF 구성 조회"""
        with self.db._get_connection() as conn:
            query = "SELECT * FROM etf_composition WHERE etf = ?"
            params = [etf]

            if date_str:
                query += " AND date = ?"
                params.append(date_str)
            else:
                query += " AND date = (SELECT MAX(date) FROM etf_composition WHERE etf = ?)"
                params.append(etf)

            query += " ORDER BY composition_type, rank"

            df = pd.read_sql_query(query, conn, params=params)
            return df

    def get_etf_performance(self, category: str = None, days: int = 1) -> pd.DataFrame:
        """ETF 성과 조회"""
        with self.db._get_connection() as conn:
            query = "SELECT * FROM etf_performance WHERE date >= date('now', '-{} days')".format(days)
            params = []

            if category:
                query += " AND category = ?"
                params.append(category)

            query += " ORDER BY date DESC, return_1d DESC"

            df = pd.read_sql_query(query, conn, params=params)
            return df

    def get_market_snapshot(self, date_str: str = None) -> Dict[str, Any]:
        """시장 스냅샷 조회"""
        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            if date_str:
                cursor.execute("SELECT * FROM market_snapshots WHERE date = ?", (date_str,))
            else:
                cursor.execute("SELECT * FROM market_snapshots ORDER BY date DESC LIMIT 1")

            row = cursor.fetchone()
            if row:
                return dict(row)
            return {}

    def get_crypto_prices(self, days: int = 7) -> pd.DataFrame:
        """암호화폐 가격 조회"""
        with self.db._get_connection() as conn:
            query = f"""
                SELECT * FROM crypto_prices
                WHERE date >= date('now', '-{days} days')
                ORDER BY date DESC, ticker
            """
            df = pd.read_sql_query(query, conn)
            return df

    def print_summary(self):
        """저장된 데이터 요약 출력"""
        print("\n" + "=" * 60)
        print("Unified Data Store Summary")
        print("=" * 60)

        with self.db._get_connection() as conn:
            cursor = conn.cursor()

            tables = ['daily_prices', 'etf_composition', 'etf_performance',
                      'crypto_prices', 'market_snapshots']

            for table in tables:
                cursor.execute(f"SELECT COUNT(*), MIN(date), MAX(date) FROM {table}")
                count, min_date, max_date = cursor.fetchone()
                print(f"  {table:20s}: {count:6d} records ({min_date or 'N/A'} ~ {max_date or 'N/A'})")

        # 최신 스냅샷
        snapshot = self.get_market_snapshot()
        if snapshot:
            print(f"\n[Latest Snapshot: {snapshot.get('date', 'N/A')}]")
            print(f"  SPY: ${snapshot.get('spy_close', 0):.2f} ({snapshot.get('spy_change', 0):+.2f}%)")
            print(f"  VIX: {snapshot.get('vix', 0):.2f}")
            print(f"  BTC: ${snapshot.get('btc_price', 0):,.0f}")
            print(f"  Leader: {snapshot.get('sector_leader', 'N/A')}")
            print(f"  Laggard: {snapshot.get('sector_laggard', 'N/A')}")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Unified Data Store Test")
    print("=" * 60)

    store = UnifiedDataStore()

    # 전체 수집 (ETF 구성은 시간이 걸리므로 일부만)
    stats = store.collect_and_store_all(include_composition=True)

    # 요약 출력
    store.print_summary()

    # 샘플 조회
    print("\n[Sample Queries]")

    # SPY 최근 5일
    spy_prices = store.get_daily_prices(ticker="SPY", days=5)
    print(f"\nSPY prices (last 5 days): {len(spy_prices)} records")

    # 섹터 성과
    sector_perf = store.get_etf_performance(category="sector", days=1)
    if not sector_perf.empty:
        print(f"\nSector Performance (today):")
        for _, row in sector_perf.head(5).iterrows():
            print(f"  {row['ticker']:5s}: {row['return_1d']:+.2f}% (1D)")

    print("\n" + "=" * 60)
    print("Test Complete!")
