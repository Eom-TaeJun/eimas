"""
Crypto Macro Analysis Module
============================
스테이블코인(Digital M2)과 거시경제(국채 금리) 간의 상호작용 분석

기능:
1. Digital M2 지표 생성 (USDT + USDC 시가총액)
2. 국채 수요 및 금리 영향 분석 (Correlation Analysis)

Theory:
- Stablecoin issuance -> Increased demand for collateral (T-Bills) -> Yield compression
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger('eimas.crypto_macro')

class CryptoMacroAnalyzer:
    """
    크립토-매크로 연관성 분석기
    """
    
    # Tickers
    STABLECOINS = {
        'USDT': 'USDT-USD',
        'USDC': 'USDC-USD'
    }
    
    # Treasury Yield Tickers (Yahoo Finance)
    # ^IRX: 13 Week Treasury Bill Yield
    # ^FVX: 5 Year Treasury Yield
    # ^TNX: 10 Year Treasury Yield
    TREASURIES = {
        '3M': '^IRX',
        '5Y': '^FVX',
        '10Y': '^TNX'
    }

    def __init__(self):
        self.data_cache = {}

    def get_digital_m2(self, lookback_days: int = 365) -> Dict[str, float]:
        """
        Digital M2 (주요 스테이블코인 시가총액 합계) 및 변화율 계산
        
        Note: Yahoo Finance는 암호화폐의 'Volume'과 'Price'는 제공하지만
        'Market Cap' 시계열을 직접 제공하지 않을 수 있음.
        여기서는 Price * Supply 추정 대신, 
        Volume과 Price 변동성을 통해 유동성 흐름을 대리(Proxy)하거나
        가능하면 시가총액 데이터를 가져옴.
        
        대안: 간단히 최근 가격 데이터와 Volume을 가져오고, 
        Supply는 외부 주입되거나 고정된 가정하에 가격 변동을 이용하지 않음(스테이블코인이므로).
        
        *실제 구현에서는 CoinGecko API가 시가총액에 더 적합하나,
        여기서는 yfinance Volume을 유동성 Proxy로 사용하거나
        genuis_act_macro.py의 로직(가격 프리미엄기반 공급 추정)을 차용.*
        """
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        total_m2_series = pd.Series(dtype=float)
        
        # 간단한 시가총액 추정 (2025-2026 기준 대략적 공급량 베이스라인)
        # 실제로는 API가 필요하지만, 데모를 위해 Volume + Price 변동으로 추세 생성
        # 혹은 yfinance의 'Volume'을 활용 (거래량이 시총과 비례한다고 가정하긴 어렵지만 활성도 측정 가능)
        
        # 여기서는 yfinance 데이터를 받아오되, 
        # 데이터가 불충분하면 시뮬레이션 데이터를 생성하여 로직을 검증하도록 함.
        
        dfs = {}
        try:
            tickers = list(self.STABLECOINS.values())
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)
            
            if not data.empty and 'Close' in data:
                # 데이터가 MultiIndex일 수 있음
                close_df = data['Close']
                volume_df = data['Volume']
                
                # Digital M2 Proxy: 
                # 스테이블코인은 가격이 $1 고정이므로, 시가총액 변화를 정확히 알기 어렵다(yfinance 기준).
                # 따라서 여기서는 '거래량(Volume)'을 활성 유동성 지표로 사용하거나,
                # 만약 Market Cap 데이터가 있다면 그것을 쓴다.
                # yfinance에서 암호화폐 Market Cap을 시계열로 주는지 확인 필요 -> 보통 안 줌.
                
                # 대안: 거래량(Volume)의 7일 이동평균을 'Active Digital Liquidity'로 정의
                for symbol, ticker in self.STABLECOINS.items():
                    if ticker in close_df.columns:
                        vol = volume_df[ticker]
                        # 거래량 평활화
                        dfs[symbol] = vol.rolling(window=7).mean()
                
                if dfs:
                    total_m2_series = pd.concat(dfs.values(), axis=1).sum(axis=1)
            
        except Exception as e:
            logger.warning(f"Failed to download crypto data: {e}")

        # 데이터가 없으면 None 반환
        if total_m2_series.empty:
            return {
                'current_m2_proxy': 0.0,
                'wow_change_pct': 0.0,
                'trend': 'neutral'
            }

        current = total_m2_series.iloc[-1]
        week_ago = total_m2_series.iloc[-8] if len(total_m2_series) > 7 else current
        
        wow_change = (current - week_ago) / week_ago if week_ago > 0 else 0.0
        
        return {
            'current_m2_proxy': float(current),
            'wow_change_pct': float(wow_change * 100),
            'trend': 'expansion' if wow_change > 0.01 else 'contraction' if wow_change < -0.01 else 'neutral',
            'series': total_m2_series  # 상관분석용
        }

    def analyze_treasury_correlation(self, m2_series: pd.Series) -> Dict[str, float]:
        """
        국채 금리와의 상관관계 분석
        
        Hypothesis: Digital M2 (Liquidity) Increase -> Treasury Demand Up -> Yield Down
        Expected Correlation: Negative
        """
        if m2_series.empty:
            return {'correlation_3m': 0.0, 'correlation_10y': 0.0}
            
        try:
            # 국채 금리 데이터 다운로드
            tickers = list(self.TREASURIES.values())
            # m2_series의 기간에 맞춤
            start_date = m2_series.index[0]
            end_date = m2_series.index[-1] + timedelta(days=1)
            
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
            
            correlations = {}
            
            # 인덱스 정렬 및 결측치 처리
            common_idx = m2_series.index.intersection(data.index)
            
            if len(common_idx) < 30:
                logger.warning("Insufficient overlapping data for correlation analysis")
                return {'correlation_3m': 0.0, 'correlation_10y': 0.0}

            m2_aligned = m2_series.loc[common_idx]
            
            # 상관계수 계산
            # 3M Yield
            if self.TREASURIES['3M'] in data.columns:
                yield_3m = data.loc[common_idx, self.TREASURIES['3M']]
                # 변화율끼리 상관관계 (Level 변수는 Spurious Correlation 가능성)
                # 하지만 요청은 '수요 예측'이므로 Level 간 관계나, 
                # M2 변화율 vs 금리 변화율을 보는 것이 맞음.
                # 여기서는 'M2 Proxy(Volume) Trend' vs 'Yield Trend'
                
                corr_3m = m2_aligned.corr(yield_3m)
                correlations['correlation_3m'] = float(corr_3m)
                
            # 10Y Yield
            if self.TREASURIES['10Y'] in data.columns:
                yield_10y = data.loc[common_idx, self.TREASURIES['10Y']]
                corr_10y = m2_aligned.corr(yield_10y)
                correlations['correlation_10y'] = float(corr_10y)
                
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {e}")
            return {'correlation_3m': 0.0, 'correlation_10y': 0.0}

    def run_analysis(self) -> Dict:
        """전체 분석 실행"""
        m2_data = self.get_digital_m2()
        
        if 'series' in m2_data:
            m2_series = m2_data.pop('series') # 결과 dict에서 시리즈는 제거 (너무 큼)
            correlations = self.analyze_treasury_correlation(m2_series)
        else:
            correlations = {'correlation_3m': 0.0, 'correlation_10y': 0.0}
            
        # 해석 생성
        corr_3m = correlations.get('correlation_3m', 0)
        interpretation = "데이터 부족 또는 중립적"
        
        if corr_3m < -0.3:
            interpretation = "Digital M2 증가가 단기 국채 금리 하락(수요 증가)과 연동됨 (가설 지지)"
        elif corr_3m > 0.3:
            interpretation = "Digital M2 증가 시 국채 금리 상승 (가설 기각/Risk-On 선호)"
        else:
            interpretation = "뚜렷한 상관관계 관찰되지 않음"

        return {
            'digital_m2': m2_data,
            'treasury_correlation': correlations,
            'impact_analysis': interpretation,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Test Code
    analyzer = CryptoMacroAnalyzer()
    result = analyzer.run_analysis()
    print("Crypto Macro Analysis Result:")
    print(result)
