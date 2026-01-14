#!/usr/bin/env python3
"""
Market Anomaly Detector - Data Collectors
==========================================
yfinance와 FRED API를 사용한 시장 데이터 수집
+ Crypto 데이터 수집 (with fallback)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import os

warnings.filterwarnings('ignore')


# ============================================================
# Crypto 티커 설정
# ============================================================
CRYPTO_TICKERS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'SOL-USD': 'Solana',
}


class CryptoDataCollector:
    """암호화폐 데이터 수집기 (with fallback to yfinance)"""
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
        self.collection_status: Dict[str, Dict] = {}
    
    def _fetch_via_yfinance(self, ticker: str) -> Optional[pd.DataFrame]:
        """yfinance를 통한 데이터 수집 (기본 및 fallback)"""
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
            return None
    
    def _fetch_via_primary(self, ticker: str) -> Optional[pd.DataFrame]:
        """Primary 데이터 소스 (현재는 yfinance, 추후 확장 가능)
        
        Note: 실제 프로덕션에서는 Binance, CoinGecko 등의 API를 
        primary로 사용하고 yfinance를 fallback으로 사용할 수 있음
        """
        # 현재는 yfinance가 primary
        return self._fetch_via_yfinance(ticker)
    
    def collect_single(self, ticker: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """단일 암호화폐 데이터 수집 (with status tracking)"""
        status = {
            'ticker': ticker,
            'name': CRYPTO_TICKERS.get(ticker, ticker),
            'success': False,
            'source': None,
            'fallback_used': False,
            'error': None,
            'timestamp': datetime.now().isoformat(),
            'data_points': 0,
        }
        
        # 1. Primary source 시도
        try:
            data = self._fetch_via_primary(ticker)
            if data is not None and not data.empty:
                status['success'] = True
                status['source'] = 'primary'
                status['data_points'] = len(data)
                self.collection_status[ticker] = status
                return data, status
        except Exception as e:
            status['error'] = f"Primary failed: {str(e)}"
        
        # 2. Fallback to yfinance (if primary was different)
        try:
            data = self._fetch_via_yfinance(ticker)
            if data is not None and not data.empty:
                status['success'] = True
                status['source'] = 'yfinance_fallback'
                status['fallback_used'] = True
                status['data_points'] = len(data)
                status['error'] = None  # Clear error on success
                self.collection_status[ticker] = status
                return data, status
        except Exception as e:
            status['error'] = f"Fallback failed: {str(e)}"
        
        status['success'] = False
        self.collection_status[ticker] = status
        return None, status
    
    def collect_all(self, tickers: List[str] = None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        """모든 암호화폐 데이터 수집"""
        if tickers is None:
            tickers = list(CRYPTO_TICKERS.keys())
        
        results = {}
        overall_status = {
            'timestamp': datetime.now().isoformat(),
            'total_tickers': len(tickers),
            'successful': 0,
            'failed': 0,
            'fallback_used_count': 0,
            'tickers': {}
        }
        
        print(f"   🪙 Collecting {len(tickers)} crypto tickers...")
        
        for ticker in tickers:
            data, status = self.collect_single(ticker)
            overall_status['tickers'][ticker] = status
            
            if data is not None:
                results[ticker] = data
                overall_status['successful'] += 1
                if status.get('fallback_used'):
                    overall_status['fallback_used_count'] += 1
                print(f"   ✅ {ticker} ({status['name']})")
            else:
                overall_status['failed'] += 1
                print(f"   ❌ {ticker}: {status.get('error', 'Unknown error')}")
        
        return results, overall_status
    
    def get_collection_status(self) -> Dict:
        """수집 상태 반환"""
        successful = sum(1 for s in self.collection_status.values() if s.get('success'))
        failed = sum(1 for s in self.collection_status.values() if not s.get('success'))
        fallback_used = sum(1 for s in self.collection_status.values() if s.get('fallback_used'))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total': len(self.collection_status),
            'successful': successful,
            'failed': failed,
            'fallback_used': fallback_used,
            'success_rate': round(successful / len(self.collection_status) * 100, 1) if self.collection_status else 0,
            'details': self.collection_status
        }


class MarketDataCollector:
    """시장 데이터 수집기 (yfinance)"""
    
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
    
    def collect_single(self, ticker: str) -> Optional[pd.DataFrame]:
        """단일 티커 데이터 수집"""
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
            
            # 컬럼명 정리 (MultiIndex 처리)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            return data
        except Exception as e:
            print(f"   ⚠️ {ticker}: {e}")
            return None
    
    def collect_batch(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """배치 티커 데이터 수집"""
        results = {}
        
        # yfinance 배치 다운로드
        try:
            data = yf.download(
                tickers, 
                start=self.start_date, 
                end=self.end_date,
                progress=False,
                auto_adjust=True,
                group_by='ticker'
            )
            
            if data.empty:
                return results
            
            # 개별 티커 분리
            if len(tickers) == 1:
                results[tickers[0]] = data
            else:
                for ticker in tickers:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            ticker_data = data[ticker].dropna(how='all')
                            if not ticker_data.empty:
                                results[ticker] = ticker_data
                    except Exception:
                        continue
        except Exception as e:
            print(f"   ⚠️ Batch download error: {e}")
            # 개별 수집 fallback
            for ticker in tickers:
                data = self.collect_single(ticker)
                if data is not None:
                    results[ticker] = data
        
        return results
    
    def collect_all(self, tickers_config: Dict) -> Dict[str, pd.DataFrame]:
        """모든 자산군 데이터 수집"""
        all_tickers = []
        
        def extract_tickers_recursive(data):
            if isinstance(data, dict):
                # 'ticker' 키가 직접 있는 경우
                if 'ticker' in data:
                    all_tickers.append(data['ticker'])
                # 하위 값들을 순회
                for value in data.values():
                    extract_tickers_recursive(value)
            elif isinstance(data, list):
                for item in data:
                    extract_tickers_recursive(item)
        
        # macro 섹션은 제외하고 나머지 섹션에서 티커 추출 (FREDCollector가 처리)
        config_to_scan = {k: v for k, v in tickers_config.items() if k != 'macro'}
        extract_tickers_recursive(config_to_scan)
        
        # 중복 제거
        all_tickers = list(set(all_tickers))
        
        print(f"   📊 Collecting {len(all_tickers)} tickers...")
        
        return self.collect_batch(all_tickers)
    
    def get_latest_prices(self, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """최신 가격 추출"""
        prices = {}
        for ticker, df in data.items():
            if not df.empty and 'Close' in df.columns:
                prices[ticker] = float(df['Close'].iloc[-1])
        return prices
    
    def get_daily_returns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """일일 수익률 계산"""
        closes = {}
        for ticker, df in data.items():
            if not df.empty and 'Close' in df.columns:
                closes[ticker] = df['Close']
        
        if not closes:
            return pd.DataFrame()
        
        close_df = pd.DataFrame(closes)
        returns = close_df.pct_change()
        
        return returns


class FREDDataCollector:
    """FRED 거시경제 데이터 수집기"""
    
    def __init__(self, api_key: Optional[str] = None, lookback_days: int = 365):
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        self.lookback_days = lookback_days
        self.fred = None
        
        if self.api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.api_key)
            except ImportError:
                print("   ⚠️ fredapi not installed")
    
    def is_available(self) -> bool:
        """FRED API 사용 가능 여부"""
        return self.fred is not None
    
    def collect_single(self, series_id: str) -> Optional[pd.Series]:
        """단일 시리즈 수집"""
        if not self.is_available():
            return None
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date, 
                observation_end=end_date
            )
            return data
        except Exception as e:
            print(f"   ⚠️ {series_id}: {e}")
            return None
    
    def collect_batch(self, series_ids: List[str]) -> pd.DataFrame:
        """배치 시리즈 수집"""
        results = {}
        
        for series_id in series_ids:
            data = self.collect_single(series_id)
            if data is not None:
                results[series_id] = data
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df = df.ffill()  # Forward fill for missing values
        
        return df
    
    def collect_all(self, macro_config: Dict) -> pd.DataFrame:
        """모든 거시경제 지표 수집"""
        if not self.is_available():
            print("   ⚠️ FRED API not available")
            return pd.DataFrame()
        
        all_series = []
        
        for category in ['rates', 'spreads', 'inflation', 'employment', 'credit', 'housing']:
            if category in macro_config:
                for item in macro_config[category]:
                    all_series.append(item['id'])
        
        print(f"   📈 Collecting {len(all_series)} FRED series...")
        
        return self.collect_batch(all_series)


class DataManager:
    """통합 데이터 관리자"""
    
    def __init__(self, lookback_days: int = 60):
        self.market_collector = MarketDataCollector(lookback_days)
        self.fred_collector = FREDDataCollector(lookback_days=365)
        self.crypto_collector = CryptoDataCollector(lookback_days)
        
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.crypto_data: Dict[str, pd.DataFrame] = {}
        self.macro_data: pd.DataFrame = pd.DataFrame()
        self.daily_returns: pd.DataFrame = pd.DataFrame()
        self.crypto_collection_status: Dict = {}
    
    def collect_all(self, tickers_config: Dict) -> Tuple[Dict, pd.DataFrame]:
        """모든 데이터 수집"""
        # 시장 데이터 수집
        self.market_data = self.market_collector.collect_all(tickers_config)
        print(f"   ✅ Collected {len(self.market_data)} market tickers")
        
        # 거시경제 데이터 수집
        macro_config = tickers_config.get('macro', {})
        if macro_config and self.fred_collector.is_available():
            self.macro_data = self.fred_collector.collect_all(macro_config)
            print(f"   ✅ Collected {len(self.macro_data.columns)} FRED series")
            
            # FRED 데이터를 market_data에 병합 (ForecastAgent 등에서 통합 사용)
            if not self.macro_data.empty:
                for col in self.macro_data.columns:
                    # FRED 데이터는 Series 형태이므로 DataFrame으로 변환
                    df = self.macro_data[[col]].copy()
                    # 컬럼 이름을 'Close'로 변경 (선택사항, 일관성 유지)
                    df.columns = ['Close']
                    self.market_data[col] = df
        
        # 암호화폐 데이터 수집 (with status tracking)
        crypto_tickers = [item['ticker'] for item in tickers_config.get('crypto', [])]
        if crypto_tickers:
            self.crypto_data, self.crypto_collection_status = self.crypto_collector.collect_all(crypto_tickers)
            print(f"   ✅ Collected {len(self.crypto_data)} crypto tickers")
            
            # 암호화폐 데이터를 market_data에 병합 (통합 분석용)
            self.market_data.update(self.crypto_data)
        
        # 일일 수익률 계산
        self.daily_returns = self.market_collector.get_daily_returns(self.market_data)
        
        return self.market_data, self.macro_data
    
    def get_crypto_collection_status(self) -> Dict:
        """암호화폐 수집 상태 반환"""
        return self.crypto_collection_status
    
    def get_crypto_data(self) -> Dict[str, pd.DataFrame]:
        """암호화폐 데이터만 반환"""
        return self.crypto_data
    
    def get_latest_snapshot(self) -> Dict:
        """최신 스냅샷 반환"""
        snapshot = {
            'prices': self.market_collector.get_latest_prices(self.market_data),
            'returns_1d': {},
            'returns_5d': {},
            'returns_20d': {},
        }
        
        if not self.daily_returns.empty:
            for ticker in self.daily_returns.columns:
                returns = self.daily_returns[ticker].dropna()
                if len(returns) >= 1:
                    snapshot['returns_1d'][ticker] = float(returns.iloc[-1])
                if len(returns) >= 5:
                    snapshot['returns_5d'][ticker] = float(returns.iloc[-5:].sum())
                if len(returns) >= 20:
                    snapshot['returns_20d'][ticker] = float(returns.iloc[-20:].sum())
        
        return snapshot
    
    def get_close_prices_df(self) -> pd.DataFrame:
        """종가 DataFrame 반환"""
        closes = {}
        for ticker, df in self.market_data.items():
            if not df.empty and 'Close' in df.columns:
                closes[ticker] = df['Close']
        
        return pd.DataFrame(closes) if closes else pd.DataFrame()
    
    def collect_macro_indicators(self) -> Dict:
        """
        거시경제 선행지표 수집
        
        수집 지표:
        1. Yield Curve Slope (10Y - 2Y): 역전 시 경기침체 선행 신호
        2. Credit Spread 변화율 (HYG/LQD): 신용위험 확대 신호
        3. TED Spread (선택적): 은행간 신용위험 지표
        
        Returns:
            {
                'yield_curve_slope': float,      # 10Y - 2Y (bp)
                'yield_curve_zscore': float,     # 20일 Z-score
                'yield_curve_status': str,       # 'NORMAL', 'FLAT', 'INVERTED'
                'credit_spread_change': float,   # HYG/LQD 20일 변화율 (%)
                'credit_spread_zscore': float,   # 20일 Z-score
                'ted_spread': float,             # TED spread (bp, 선택적)
                'interpretation': str            # 해석 텍스트
            }
        """
        indicators = {
            'yield_curve_slope': None,
            'yield_curve_zscore': None,
            'yield_curve_status': 'UNKNOWN',
            'credit_spread_change': None,
            'credit_spread_zscore': None,
            'ted_spread': None,
            'interpretation': ''
        }
        
        # ============================================================
        # 1. Yield Curve Slope (10Y - 2Y)
        # ============================================================
        if self.fred_collector.is_available():
            try:
                # FRED에서 금리 데이터 수집
                dgs10 = self.fred_collector.collect_single('DGS10')  # 10년물 국채 금리
                dgs2 = self.fred_collector.collect_single('DGS2')    # 2년물 국채 금리
                
                if dgs10 is not None and dgs2 is not None and len(dgs10) > 0 and len(dgs2) > 0:
                    # 최신값 사용
                    rate_10y = float(dgs10.iloc[-1])
                    rate_2y = float(dgs2.iloc[-1])
                    
                    if not np.isnan(rate_10y) and not np.isnan(rate_2y):
                        # Slope 계산 (bp 단위)
                        slope = (rate_10y - rate_2y) * 100  # 퍼센트를 bp로 변환
                        indicators['yield_curve_slope'] = float(slope)
                        
                        # 20일 Z-score 계산
                        if len(dgs10) >= 20 and len(dgs2) >= 20:
                            recent_10y = dgs10.iloc[-20:].dropna()
                            recent_2y = dgs2.iloc[-20:].dropna()
                            
                            if len(recent_10y) >= 10 and len(recent_2y) >= 10:
                                recent_slopes = (recent_10y - recent_2y) * 100
                                mean_slope = float(recent_slopes.mean())
                                std_slope = float(recent_slopes.std())
                                
                                if std_slope > 0:
                                    zscore = (slope - mean_slope) / std_slope
                                    indicators['yield_curve_zscore'] = float(zscore)
                        
                        # Yield Curve 상태 판단
                        if slope < -50:  # -50bp 이하
                            indicators['yield_curve_status'] = 'INVERTED'
                        elif slope < 50:  # 50bp 미만
                            indicators['yield_curve_status'] = 'FLAT'
                        else:
                            indicators['yield_curve_status'] = 'NORMAL'
            except Exception as e:
                print(f"   ⚠️ Yield Curve 계산 실패: {e}")
        
        # ============================================================
        # 2. Credit Spread 변화율 (HYG/LQD)
        # ============================================================
        try:
            hyg_data = self.market_data.get('HYG')
            lqd_data = self.market_data.get('LQD')
            
            if hyg_data is not None and lqd_data is not None:
                if not hyg_data.empty and not lqd_data.empty and 'Close' in hyg_data.columns and 'Close' in lqd_data.columns:
                    # HYG/LQD 비율 계산
                    hyg_close = hyg_data['Close']
                    lqd_close = lqd_data['Close']
                    
                    # 공통 인덱스로 정렬
                    common_index = hyg_close.index.intersection(lqd_close.index)
                    if len(common_index) >= 20:
                        hyg_aligned = hyg_close.loc[common_index]
                        lqd_aligned = lqd_close.loc[common_index]
                        
                        spread_ratio = hyg_aligned / lqd_aligned
                        
                        # 20일 변화율 계산
                        if len(spread_ratio) >= 20:
                            current_ratio = float(spread_ratio.iloc[-1])
                            past_ratio = float(spread_ratio.iloc[-20])
                            
                            if past_ratio > 0:
                                change_pct = ((current_ratio - past_ratio) / past_ratio) * 100
                                indicators['credit_spread_change'] = float(change_pct)
                                
                                # 20일 Z-score 계산
                                mean_ratio = float(spread_ratio.iloc[-20:].mean())
                                std_ratio = float(spread_ratio.iloc[-20:].std())
                                
                                if std_ratio > 0:
                                    zscore = (current_ratio - mean_ratio) / std_ratio
                                    indicators['credit_spread_zscore'] = float(zscore)
        except Exception as e:
            print(f"   ⚠️ Credit Spread 계산 실패: {e}")
        
        # ============================================================
        # 3. TED Spread (선택적)
        # ============================================================
        if self.fred_collector.is_available():
            try:
                # 3개월 LIBOR (USD3MTD156N) 또는 대체 시리즈
                # 3개월 T-Bill (DGS3MO)
                libor_3m = self.fred_collector.collect_single('USD3MTD156N')  # 3M LIBOR
                tbill_3m = self.fred_collector.collect_single('DGS3MO')       # 3M T-Bill
                
                if libor_3m is not None and tbill_3m is not None:
                    if len(libor_3m) > 0 and len(tbill_3m) > 0:
                        libor_latest = float(libor_3m.iloc[-1])
                        tbill_latest = float(tbill_3m.iloc[-1])
                        
                        if not np.isnan(libor_latest) and not np.isnan(tbill_latest):
                            ted_spread = (libor_latest - tbill_latest) * 100  # bp 단위
                            indicators['ted_spread'] = float(ted_spread)
            except Exception as e:
                # TED Spread는 선택적이므로 실패해도 계속 진행
                pass
        
        # ============================================================
        # 해석 텍스트 생성
        # ============================================================
        interpretation_parts = []
        
        # Yield Curve 해석
        if indicators['yield_curve_slope'] is not None:
            slope = indicators['yield_curve_slope']
            status = indicators['yield_curve_status']
            
            if status == 'INVERTED':
                interpretation_parts.append(f"⚠️ 수익률 곡선 역전 ({slope:.1f}bp): 경기침체 선행 신호. 역사적으로 역전 후 6-18개월 내 경기침체 발생.")
            elif status == 'FLAT':
                interpretation_parts.append(f"📊 수익률 곡선 평탄화 ({slope:.1f}bp): 경기 둔화 신호. 주시 필요.")
            else:
                interpretation_parts.append(f"✅ 수익률 곡선 정상 ({slope:.1f}bp): 경기 확장 국면.")
        
        # Credit Spread 해석
        if indicators['credit_spread_change'] is not None:
            change = indicators['credit_spread_change']
            if change < -5:
                interpretation_parts.append(f"⚠️ 신용 스프레드 급격한 확대 ({change:.1f}%): 신용위험 상승, 리스크오프 신호.")
            elif change < -2:
                interpretation_parts.append(f"📊 신용 스프레드 확대 ({change:.1f}%): 신용 환경 악화 주시.")
            elif change > 5:
                interpretation_parts.append(f"✅ 신용 스프레드 축소 ({change:.1f}%): 신용 환경 개선.")
            else:
                interpretation_parts.append(f"📊 신용 스프레드 안정 ({change:.1f}%): 신용 환경 정상.")
        
        # TED Spread 해석
        if indicators['ted_spread'] is not None:
            ted = indicators['ted_spread']
            if ted > 100:
                interpretation_parts.append(f"⚠️ TED Spread 확대 ({ted:.1f}bp): 은행간 신용위험 상승, 유동성 스트레스 신호.")
            elif ted > 50:
                interpretation_parts.append(f"📊 TED Spread 상승 ({ted:.1f}bp): 은행간 신용위험 주시.")
            else:
                interpretation_parts.append(f"✅ TED Spread 정상 ({ted:.1f}bp): 은행간 신용 환경 안정.")
        
        if not interpretation_parts:
            indicators['interpretation'] = "거시경제 지표 데이터 부족"
        else:
            indicators['interpretation'] = " | ".join(interpretation_parts)
        
        return indicators


# ============================================================
# UnifiedDataCollector - LASSO 분석용 통합 데이터 수집기
# ============================================================

# LASSO 분석용 티커 설정
LASSO_YAHOO_TICKERS = {
    # 지수
    'SP500': '^GSPC', 'Nasdaq': '^IXIC', 'Russell2000': '^RUT',
    'Dow': '^DJI', 'VIX': '^VIX',

    # 섹터 ETF
    'Sector_Materials': 'XLB', 'Sector_Comm': 'XLC', 'Sector_Energy': 'XLE',
    'Sector_Financials': 'XLF', 'Sector_Industrials': 'XLI', 'Sector_Tech': 'XLK',
    'Sector_Staples': 'XLP', 'Sector_RealEstate': 'XLRE', 'Sector_Utilities': 'XLU',
    'Sector_Health': 'XLV', 'Sector_Discretionary': 'XLY',

    # 원자재
    'Gold': 'GC=F', 'Silver': 'SI=F', 'Copper': 'HG=F',
    'Oil_WTI': 'CL=F', 'NatGas': 'NG=F', 'Commodity_Idx': 'DBC',

    # 채권 ETF
    'Treasury_20Y': 'TLT', 'Treasury_7_10Y': 'IEF', 'Treasury_1_3Y': 'SHY',
    'Corp_InvGrade': 'LQD', 'HighYield_ETF': 'HYG',

    # 암호화폐
    'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD', 'Solana': 'SOL-USD',

    # 환율
    'Dollar_Idx': 'DX=F', 'EURUSD': 'EURUSD=X', 'USDJPY': 'USDJPY=X',
    'GBPUSD': 'GBPUSD=X', 'USDKRW': 'USDKRW=X', 'USDCNY': 'USDCNY=X'
}

# FRED 금리 티커
LASSO_FRED_RATES = {
    'US10Y': 'DGS10', 'US2Y': 'DGS2',
    'Baa_Yield': 'DBAA', 'HighYield_Rate': 'BAMLH0A0HYM2',
    'Breakeven5Y': 'T5YIE', 'RealYield10Y': 'DFII10',
}


class UnifiedDataCollector:
    """
    LASSO 분석용 통합 데이터 수집기

    forecasting 프로젝트의 collect_macro_finance_v2.py 로직을 통합.
    Yahoo Finance + FRED 데이터를 수집하고 Ret_*, d_* 변수로 변환.

    Example:
        >>> collector = UnifiedDataCollector(start_date='2024-09-01')
        >>> df = collector.collect_all()
        >>> print(df.columns)  # Ret_SP500, d_US10Y, d_Spread_Baa, ...
    """

    def __init__(self, start_date: str = '2024-09-01', fred_api_key: Optional[str] = None):
        """
        Args:
            start_date: 데이터 수집 시작일 (YYYY-MM-DD)
            fred_api_key: FRED API 키 (None이면 환경변수에서 로드)
        """
        self.start_date = start_date
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred = None

        if self.fred_api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.fred_api_key)
                print("   ✓ FRED API 연결 완료")
            except Exception as e:
                print(f"   ⚠ FRED API 초기화 실패: {e}")

    def fetch_yahoo(self) -> pd.DataFrame:
        """Yahoo Finance에서 자산 가격 수집"""
        print(f"   📈 Yahoo Finance 데이터 수집 중... ({self.start_date} ~ )")

        try:
            df = yf.download(
                list(LASSO_YAHOO_TICKERS.values()),
                start=self.start_date,
                progress=False,
                auto_adjust=True
            )

            if df.empty:
                print("   ⚠ Yahoo 데이터 없음")
                return pd.DataFrame()

            # MultiIndex 처리
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df['Close']
                except KeyError:
                    df = df.iloc[:, 0]

            # 티커 이름 매핑
            inv_map = {v: k for k, v in LASSO_YAHOO_TICKERS.items()}
            df.rename(columns=inv_map, inplace=True)

            print(f"   ✓ Yahoo: {len(df)}행, {len(df.columns)}개 자산")
            return df

        except Exception as e:
            print(f"   ⚠ Yahoo 오류: {e}")
            return pd.DataFrame()

    def fetch_fred_rates(self) -> pd.DataFrame:
        """FRED에서 금리 데이터 수집"""
        if self.fred is None:
            print("   ⚠ FRED API 사용 불가")
            return pd.DataFrame()

        print("   📊 FRED 금리 데이터 수집 중...")
        dfs = []

        for alias, code in LASSO_FRED_RATES.items():
            try:
                s = self.fred.get_series(code, observation_start=self.start_date)
                s.name = alias
                dfs.append(s)
            except Exception as e:
                print(f"   ⚠ FRED {alias} 실패: {e}")

        if dfs:
            result = pd.concat(dfs, axis=1)
            print(f"   ✓ FRED: {len(result)}행, {len(result.columns)}개 금리")
            return result
        return pd.DataFrame()

    def collect_all(self) -> pd.DataFrame:
        """
        모든 데이터 수집 및 변환

        Returns:
            LASSO 분석용 DataFrame (Ret_*, d_* 변수 포함)
        """
        print("\n🔄 LASSO용 통합 데이터 수집 시작...")

        # 1. 데이터 수집
        df_yahoo = self.fetch_yahoo()
        df_rates = self.fetch_fred_rates()

        if df_yahoo.empty:
            print("   ⚠ Yahoo 데이터 없음, 수집 실패")
            return pd.DataFrame()

        # 2. 병합
        print("   🔗 데이터 병합 중...")
        if not df_rates.empty:
            df = df_yahoo.join(df_rates, how='outer')
        else:
            df = df_yahoo

        df.sort_index(inplace=True)
        df.index.name = 'Date'

        # 주말 제거 (US10Y 기준)
        if 'US10Y' in df.columns:
            df = df.dropna(subset=['US10Y'], how='all')

        # Forward fill 결측치
        df = df.ffill()

        # 3. 파생변수 생성
        print("   🧮 파생변수 계산 중...")

        # 스프레드 계산
        if 'Baa_Yield' in df.columns and 'US10Y' in df.columns:
            df['Spread_Baa'] = df['Baa_Yield'] - df['US10Y']
        if 'HighYield_Rate' in df.columns and 'US10Y' in df.columns:
            df['Spread_HighYield'] = df['HighYield_Rate'] - df['US10Y']
        if 'US10Y' in df.columns and 'US2Y' in df.columns:
            df['Term_Spread'] = df['US10Y'] - df['US2Y']

        # Copper/Gold Ratio
        if 'Copper' in df.columns and 'Gold' in df.columns:
            df['Copper_Gold_Ratio'] = df['Copper'] / df['Gold']

        # 4. 변환 변수 생성
        # 수익률 (Ret_*): 자산 가격 → 퍼센트 변화
        yahoo_cols = [c for c in df.columns if c in LASSO_YAHOO_TICKERS.keys()]
        for c in yahoo_cols:
            df[f'Ret_{c}'] = df[c].pct_change() * 100

        # Copper/Gold 수익률
        if 'Copper_Gold_Ratio' in df.columns:
            df['Ret_Copper_Gold'] = df['Copper_Gold_Ratio'].pct_change() * 100

        # 차분 (d_*): 금리/지수 → 일별 변화
        rate_cols = list(LASSO_FRED_RATES.keys()) + ['Spread_Baa', 'Spread_HighYield', 'Term_Spread']
        for c in rate_cols:
            if c in df.columns:
                df[f'd_{c}'] = df[c].diff()

        # VIX, Dollar_Idx도 차분
        for c in ['VIX', 'Dollar_Idx']:
            if c in df.columns:
                df[f'd_{c}'] = df[c].diff()

        # 첫 행 제거 (NaN)
        df = df.iloc[1:]

        # 5. 결과 출력
        ret_cols = [c for c in df.columns if c.startswith('Ret_')]
        d_cols = [c for c in df.columns if c.startswith('d_')]

        print(f"\n✅ 데이터 수집 완료!")
        print(f"   - 기간: {df.index.min().date()} ~ {df.index.max().date()}")
        print(f"   - 관측치: {len(df)}행")
        print(f"   - 원시 변수: {len(df.columns) - len(ret_cols) - len(d_cols)}개")
        print(f"   - 수익률 (Ret_*): {len(ret_cols)}개")
        print(f"   - 차분 (d_*): {len(d_cols)}개")

        return df

    def save_to_csv(self, df: pd.DataFrame, output_file: str = 'expanded_market_data.csv'):
        """DataFrame을 CSV로 저장"""
        df.to_csv(output_file)
        print(f"   💾 저장 완료: {output_file}")


if __name__ == "__main__":
    # 테스트
    import yaml
    
    with open('config/tickers.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    manager = DataManager(lookback_days=60)
    market_data, macro_data = manager.collect_all(config)
    
    print(f"\nMarket data: {len(market_data)} tickers")
    print(f"Macro data: {macro_data.shape if not macro_data.empty else 'N/A'}")
    
    snapshot = manager.get_latest_snapshot()
    print(f"Latest prices: {len(snapshot['prices'])} tickers")