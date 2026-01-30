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
        task_fng = self.get_crypto_fear_greed()
        task_tvl = self.get_defi_tvl()
        task_depth = self.get_market_depth()
        task_news = self.get_news_sentiment()
        task_krw = self.get_korea_risk_index()
        
        pcr, fund, stable, spread, fng, tvl, depth, news, krw = await asyncio.gather(
            task_pcr, task_fund, task_stable, task_spread, 
            task_fng, task_tvl, task_depth, task_news, task_krw,
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

        if not isinstance(fng, Exception):
            results['crypto_fng'] = fng
            print(f"      ✓ Crypto Fear & Greed: {fng.get('value', 'N/A')} ({fng.get('classification', 'N/A')})")

        if not isinstance(tvl, Exception):
            results['defi_tvl'] = tvl
            print(f"      ✓ DeFi Total TVL: ${tvl.get('total_tvl', 0)/1e9:.1f}B")

        if not isinstance(depth, Exception):
            results['market_depth'] = depth
            print(f"      ✓ Market Depth (Short Interest): SPY {depth.get('SPY_short_float', 0)*100:.2f}%, TSLA {depth.get('TSLA_short_float', 0)*100:.2f}%")

        if not isinstance(news, Exception):
            results['news_sentiment'] = news
            print(f"      ✓ News Sentiment Score: {news.get('score', 0):.1f} ({news.get('label', 'Neutral')})")

        if not isinstance(krw, Exception):
            results['korea_risk'] = krw
            print(f"      ✓ KRW Risk Index: {krw.get('status', 'Normal')} (Vol: {krw.get('volatility', 0):.2f})")
            
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
    
    async def get_crypto_fear_greed(self) -> Dict[str, Any]:
        """
        Alternative.me API를 통한 Crypto Fear & Greed Index
        """
        url = "https://api.alternative.me/fng/"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        item = data['data'][0]
                        return {
                            "value": int(item['value']),
                            "classification": item['value_classification'],
                            "timestamp": int(item['timestamp'])
                        }
        except Exception:
            pass
        return {"value": 0, "classification": "Unknown"}

    async def get_defi_tvl(self) -> Dict[str, float]:
        """
        DefiLlama API를 통한 전체 DeFi TVL
        """
        url = "https://api.llama.fi/v2/chains"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        total_tvl = sum(chain.get('tvl', 0) for chain in data)
                        return {
                            "total_tvl": float(total_tvl),
                            "top_chain_tvl": data[0].get('tvl', 0) if data else 0
                        }
        except Exception:
            pass
        return {"total_tvl": 0.0}

    async def get_market_depth(self) -> Dict[str, float]:
        """
        시장 심층 데이터: 공매도 비율(Short Interest) 및 기관 보유율
        """
        tickers = ['SPY', 'IWM', 'NVDA', 'TSLA']
        results = {}
        
        try:
            def _get_depth(t):
                info = yf.Ticker(t).info
                return {
                    f"{t}_short_float": info.get('shortPercentOfFloat', 0),
                    f"{t}_inst_held": info.get('heldPercentInstitutions', 0)
                }

            loop = asyncio.get_event_loop()
            
            # 병렬 처리를 위해 각 티커별 태스크 생성은 하지 않고 순차적으로 하되 executor 안에서 처리
            # (yfinance 내부 락 이슈 방지 겸 단순화)
            for t in tickers:
                data = await loop.run_in_executor(None, _get_depth, t)
                results.update(data)
                
            return results
        except Exception:
            return {"SPY_short_float": 0.0}

    async def get_news_sentiment(self) -> Dict[str, Any]:
        """
        RSS 피드 기반 뉴스 감성 분석 (CNBC)
        """
        import xml.etree.ElementTree as ET
        
        # CNBC Top News RSS
        url = "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"
        
        # 키워드 사전
        bullish_words = {'rise', 'jump', 'surge', 'soar', 'gain', 'bull', 'rally', 'record', 'high', 'optimism', 'beat', 'growth', 'up'}
        bearish_words = {'fall', 'drop', 'plunge', 'sink', 'bear', 'crash', 'down', 'recession', 'inflation', 'fear', 'crisis', 'loss', 'miss', 'risk'}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        root = ET.fromstring(xml_data)
                        
                        score = 0
                        count = 0
                        titles = []
                        
                        # 채널 내 아이템 순회
                        for item in root.findall('./channel/item')[:10]: # 최근 10개만
                            title = item.find('title').text.lower() if item.find('title') is not None else ""
                            desc = item.find('description').text.lower() if item.find('description') is not None else ""
                            text = f"{title} {desc}"
                            
                            item_score = 0
                            for word in bullish_words:
                                if f" {word} " in f" {text} ": # 단순 매칭
                                    item_score += 1
                            for word in bearish_words:
                                if f" {word} " in f" {text} ":
                                    item_score -= 1
                                    
                            score += item_score
                            count += 1
                            titles.append(title)
                            
                        # 결과 정규화
                        final_score = score / count if count > 0 else 0
                        
                        label = "Neutral"
                        if final_score > 0.5: label = "Bullish"
                        elif final_score < -0.5: label = "Bearish"
                        
                        return {
                            "score": round(final_score, 2),
                            "label": label,
                            "analyzed_count": count,
                            "top_headline": titles[0] if titles else ""
                        }
        except Exception as e:
            # print(f"News sentiment error: {e}")
            pass
        return {"score": 0, "label": "Neutral"}

    async def get_korea_risk_index(self) -> Dict[str, Any]:
        """
        원달러 환율(KRW=X) 기반 한국 시장 리스크 지표
        """
        try:
            def _get_krw():
                # 최근 1개월 데이터
                df = yf.download('KRW=X', period='1mo', progress=False)
                if len(df) < 20:
                    return None
                
                # Close 컬럼 추출 (MultiIndex 대응)
                if isinstance(df.columns, pd.MultiIndex):
                    prices = df['Close']['KRW=X']
                else:
                    prices = df['Close']
                
                current_price = float(prices.iloc[-1])
                ma20 = float(prices.rolling(window=20).mean().iloc[-1])
                std20 = float(prices.rolling(window=20).std().iloc[-1])
                
                # 볼린저 밴드
                upper = ma20 + (2 * std20)
                
                # 변동성 (전일 대비 등락률의 표준편차 * sqrt(252) -> 연율화 X, 그냥 단기 변동성)
                daily_ret = prices.pct_change().dropna()
                volatility = float(daily_ret.std() * 100) # 퍼센트 단위
                
                status = "Normal"
                if current_price > upper:
                    status = "Overheated (Depreciating Fast)" # 환율 급등 = 원화 가치 급락
                elif volatility > 1.0: # 일일 변동성이 큼
                    status = "Volatile"
                    
                return {
                    "price": current_price,
                    "ma20": ma20,
                    "upper_band": upper,
                    "volatility": volatility,
                    "status": status
                }

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _get_krw)
            
            if result:
                return result
        except Exception:
            pass
        return {"status": "Unknown", "volatility": 0.0}

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