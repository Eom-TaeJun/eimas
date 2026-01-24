#!/usr/bin/env python3
"""
Extended Data Sources Collector
================================
무료 API를 활용한 확장 데이터 수집 모듈

1. Yahoo Finance Options: Put/Call Ratio 계산
2. Yahoo Finance Fundamentals: PE Ratio, Earnings Yield
3. DefiLlama: Stablecoin Market Cap (Digital Liquidity)
4. FRED: High Yield Spreads (Risk Appetite)

Usage:
    collector = ExtendedDataCollector()
    data = await collector.collect_all()
"""

import yfinance as yf
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class ExtendedDataCollector:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}

    async def collect_all(self) -> Dict[str, Any]:
        """모든 확장 데이터 수집"""
        print("\n[Extended Data] Collecting additional metrics...")
        
        results = {}
        
        # 병렬 실행
        task_pcr = self.calculate_put_call_ratio('SPY')
        task_fund = self.get_sp500_fundamentals()
        task_stable = self.get_stablecoin_mcap()
        task_spread = self.get_credit_spreads()
        
        pcr, fund, stable, spread = await asyncio.gather(
            task_pcr, task_fund, task_stable, task_spread, 
            return_exceptions=True
        )
        
        # 결과 처리
        if not isinstance(pcr, Exception):
            results['put_call_ratio'] = pcr
            print(f"      ✓ SPY Put/Call Ratio: {pcr.get('ratio', 0.0):.2f}")
            
        if not isinstance(fund, Exception):
            results['fundamentals'] = fund
            print(f"      ✓ SP500 Earnings Yield: {fund.get('earnings_yield', 0.0):.2f}%")
            
        if not isinstance(stable, Exception):
            results['digital_liquidity'] = stable
            print(f"      ✓ Stablecoin Market Cap: ${stable.get('total_mcap', 0)/1e9:.1f}B")
            
        if not isinstance(spread, Exception):
            results['credit_spreads'] = spread
            print(f"      ✓ High Yield Spread: {spread.get('value', 0.0):.2f}%")
            
        return results

    async def calculate_put_call_ratio(self, ticker: str) -> Dict[str, float]:
        """
        옵션 체인을 통한 Put/Call Ratio 계산 (Volume 기준)
        """
        try:
            # 동기 함수인 yfinance를 비동기로 실행
            def _get_options():
                obj = yf.Ticker(ticker)
                # 가장 가까운 만기일 2개 선택
                expirations = obj.options[:2]
                total_puts = 0
                total_calls = 0
                
                for exp in expirations:
                    opt = obj.option_chain(exp)
                    total_puts += opt.puts['volume'].sum()
                    total_calls += opt.calls['volume'].sum()
                
                return total_puts, total_calls

            loop = asyncio.get_event_loop()
            puts, calls = await loop.run_in_executor(None, _get_options)
            
            ratio = puts / calls if calls > 0 else 0.0
            
            # 해석
            sentiment = "NEUTRAL"
            if ratio > 1.0: sentiment = "BEARISH/HEDGING" # 풋이 더 많음 (공포)
            elif ratio < 0.7: sentiment = "BULLISH/GREED" # 콜이 훨씬 많음 (탐욕)
            
            return {
                "ratio": float(ratio),
                "total_puts": int(puts),
                "total_calls": int(calls),
                "sentiment": sentiment
            }
        except Exception as e:
            # print(f"Error calculating PCR: {e}")
            return {"ratio": 0.0, "sentiment": "ERROR"}

    async def get_sp500_fundamentals(self) -> Dict[str, float]:
        """
        S&P 500 (SPY) 펀더멘털 데이터
        - PE Ratio -> Earnings Yield (1/PE)
        """
        try:
            def _get_info():
                return yf.Ticker('SPY').info

            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, _get_info)
            
            # Trailing PE가 없으면 기본값 사용 (안전장치)
            pe_ratio = info.get('trailingPE', 25.0) 
            if pe_ratio is None: pe_ratio = 25.0
            
            earnings_yield = (1.0 / pe_ratio) * 100
            
            return {
                "pe_ratio": float(pe_ratio),
                "earnings_yield": float(earnings_yield)
            }
        except Exception as e:
            return {"pe_ratio": 0.0, "earnings_yield": 0.0}

    async def get_stablecoin_mcap(self) -> Dict[str, float]:
        """
        DefiLlama API를 통한 스테이블코인 시총 (디지털 유동성)
        """
        url = "https://stablecoins.llama.fi/stablecoins?includePrices=true"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        # 상위 5개 스테이블코인 합산 (USDT, USDC, DAI 등)
                        pegged_coins = data.get('peggedAssets', [])
                        total_mcap = 0.0
                        
                        top_coins = sorted(pegged_coins, key=lambda x: x.get('circulating', {}).get('peggedUSD', 0), reverse=True)[:5]
                        
                        for coin in top_coins:
                            mcap = coin.get('circulating', {}).get('peggedUSD', 0)
                            total_mcap += mcap
                            
                        return {
                            "total_mcap": float(total_mcap),
                            "top_coin": top_coins[0]['symbol'] if top_coins else "None"
                        }
        except Exception:
            pass
        return {"total_mcap": 0.0}

    async def get_credit_spreads(self) -> Dict[str, float]:
        """
        FRED 데이터 대신 yfinance로 근사치 계산 (BAMLC0A0CM 대용)
        High Yield (HYG) vs Treasury (IEF) 비율로 추세 확인
        * 실제 FRED API 키가 있으면 FRED 사용 권장
        """
        try:
            def _get_spread():
                # HYG (High Yield)와 IEF (7-10Y Treasury)의 최근 가격
                data = yf.download(['HYG', 'IEF'], period='5d', progress=False)['Close']
                if len(data) > 0:
                    # 간단한 Risk On/Off 비율
                    # HYG/IEF가 오르면 Risk On (스프레드 축소와 유사 효과)
                    ratio = data['HYG'].iloc[-1] / data['IEF'].iloc[-1]
                    
                    # 5일 변화율
                    change = (ratio - (data['HYG'].iloc[0] / data['IEF'].iloc[0])) * 100
                    return float(ratio), float(change)
                return 0.0, 0.0

            loop = asyncio.get_event_loop()
            ratio, change = await loop.run_in_executor(None, _get_spread)
            
            return {
                "risk_ratio_hyg_ief": ratio,
                "change_5d": change,
                "interpretation": "Risk ON" if change > 0 else "Risk OFF"
            }
        except Exception:
            return {"value": 0.0}

if __name__ == "__main__":
    collector = ExtendedDataCollector()
    data = asyncio.run(collector.collect_all())
    print(data)