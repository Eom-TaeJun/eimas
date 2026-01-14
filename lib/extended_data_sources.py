"""
Extended Data Sources - 확장 데이터 소스

1. On-Chain Data (DeFiLlama, etc.)
2. Middle East Market Data (Saudi, UAE, Qatar, etc.)
3. Alternative Data Sources
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import time


# ============================================================================
# On-Chain Data (DeFiLlama)
# ============================================================================

@dataclass
class TVLData:
    """Total Value Locked 데이터"""
    protocol: str
    chain: str
    tvl: float
    tvl_change_24h: float
    tvl_change_7d: float
    timestamp: str


@dataclass
class StablecoinData:
    """스테이블코인 시장 데이터"""
    symbol: str
    name: str
    circulating: float
    circulating_change_7d: float
    price: float
    peg_deviation: float


class DeFiLlamaCollector:
    """
    DeFiLlama API를 통한 온체인 데이터 수집

    무료 API, 인증 불필요
    https://defillama.com/docs/api
    """

    BASE_URL = "https://api.llama.fi"
    STABLECOINS_URL = "https://stablecoins.llama.fi"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json'
        })

    def get_protocols_tvl(self) -> List[Dict]:
        """전체 프로토콜 TVL 조회"""
        try:
            resp = self.session.get(f"{self.BASE_URL}/protocols")
            resp.raise_for_status()
            data = resp.json()

            # 상위 20개만
            protocols = []
            for p in data[:20]:
                protocols.append({
                    'name': p.get('name'),
                    'chain': p.get('chain', 'Multi'),
                    'tvl': p.get('tvl', 0),
                    'change_1d': p.get('change_1d', 0),
                    'change_7d': p.get('change_7d', 0),
                    'category': p.get('category', 'Unknown')
                })

            return protocols
        except Exception as e:
            print(f"[DeFiLlama] Error fetching protocols: {e}")
            return []

    def get_chain_tvl(self, chain: str = "Ethereum") -> Dict:
        """특정 체인 TVL 조회"""
        try:
            resp = self.session.get(f"{self.BASE_URL}/v2/historicalChainTvl/{chain}")
            resp.raise_for_status()
            data = resp.json()

            if data:
                latest = data[-1]
                prev_day = data[-2] if len(data) > 1 else latest
                prev_week = data[-7] if len(data) > 7 else latest

                return {
                    'chain': chain,
                    'tvl': latest.get('tvl', 0),
                    'tvl_change_1d': (latest['tvl'] / prev_day['tvl'] - 1) * 100 if prev_day['tvl'] else 0,
                    'tvl_change_7d': (latest['tvl'] / prev_week['tvl'] - 1) * 100 if prev_week['tvl'] else 0,
                    'timestamp': datetime.fromtimestamp(latest.get('date', 0)).isoformat()
                }
            return {}
        except Exception as e:
            print(f"[DeFiLlama] Error fetching chain TVL: {e}")
            return {}

    def get_stablecoins(self) -> List[StablecoinData]:
        """스테이블코인 시장 데이터 조회"""
        try:
            resp = self.session.get(f"{self.STABLECOINS_URL}/stablecoins?includePrices=true")
            resp.raise_for_status()
            data = resp.json()

            stablecoins = []
            for s in data.get('peggedAssets', [])[:10]:
                circulating = s.get('circulating', {}).get('peggedUSD', 0)
                price = s.get('price', 1.0)

                stablecoins.append(StablecoinData(
                    symbol=s.get('symbol', ''),
                    name=s.get('name', ''),
                    circulating=circulating,
                    circulating_change_7d=0,  # API에서 직접 제공 안 함
                    price=price if price else 1.0,
                    peg_deviation=abs(price - 1.0) * 100 if price else 0
                ))

            return stablecoins
        except Exception as e:
            print(f"[DeFiLlama] Error fetching stablecoins: {e}")
            return []

    def get_yields(self, pool_count: int = 10) -> List[Dict]:
        """DeFi Yield 데이터 조회"""
        try:
            resp = self.session.get(f"{self.BASE_URL}/pools")
            resp.raise_for_status()
            data = resp.json()

            # TVL 기준 상위 풀
            pools = sorted(data.get('data', []), key=lambda x: x.get('tvlUsd', 0), reverse=True)

            result = []
            for p in pools[:pool_count]:
                result.append({
                    'pool': p.get('pool', ''),
                    'project': p.get('project', ''),
                    'chain': p.get('chain', ''),
                    'symbol': p.get('symbol', ''),
                    'tvl': p.get('tvlUsd', 0),
                    'apy': p.get('apy', 0),
                    'apy_base': p.get('apyBase', 0),
                    'apy_reward': p.get('apyReward', 0)
                })

            return result
        except Exception as e:
            print(f"[DeFiLlama] Error fetching yields: {e}")
            return []

    def get_summary(self) -> Dict:
        """온체인 데이터 요약"""
        chains = ['Ethereum', 'Solana', 'Arbitrum', 'Base', 'Polygon']

        chain_tvls = {}
        total_tvl = 0
        for chain in chains:
            data = self.get_chain_tvl(chain)
            if data:
                chain_tvls[chain] = data
                total_tvl += data.get('tvl', 0)

        stablecoins = self.get_stablecoins()
        total_stablecoin = sum(s.circulating for s in stablecoins)

        return {
            'timestamp': datetime.now().isoformat(),
            'total_tvl': total_tvl,
            'chain_tvls': chain_tvls,
            'stablecoin_market_cap': total_stablecoin,
            'top_stablecoins': [
                {'symbol': s.symbol, 'circulating': s.circulating, 'peg_deviation': s.peg_deviation}
                for s in stablecoins[:5]
            ]
        }


# ============================================================================
# Middle East Market Data
# ============================================================================

class MiddleEastMarketCollector:
    """
    중동 시장 데이터 수집

    지원 시장:
    - Saudi Arabia (Tadawul)
    - UAE (DFM, ADX)
    - Qatar (QSE)
    - Kuwait (Boursa Kuwait)
    """

    # 중동 시장 ETF 및 지수
    MENA_TICKERS = {
        # ETF
        'KSA': 'iShares MSCI Saudi Arabia ETF',
        'UAE': 'iShares MSCI UAE ETF',
        # 'GULF': 상장폐지됨 (2026-01-10 확인)
        'TUR': 'iShares MSCI Turkey ETF',
        'EGPT': 'VanEck Egypt Index ETF',
        'QAT': 'iShares MSCI Qatar ETF',

        # 주요 기업 ADR
        'ARAMCO': 'Saudi Aramco (참고용)',
        'EIBOR': 'Emirates NBD (참고용)',
    }

    # 중동 주요 지수 (Yahoo Finance 심볼)
    MENA_INDICES = {
        '^TASI': 'Tadawul All Share Index (Saudi)',
        '^DFMGI': 'Dubai Financial Market General Index',
        '^ADI': 'Abu Dhabi Securities Exchange Index',
        '^QSI': 'Qatar Stock Exchange Index',
    }

    def __init__(self):
        import yfinance as yf
        self.yf = yf

    def get_etf_data(self, period: str = '1mo') -> Dict[str, pd.DataFrame]:
        """중동 ETF 데이터 수집"""
        data = {}
        for ticker in ['KSA', 'UAE', 'TUR', 'QAT']:
            try:
                df = self.yf.download(ticker, period=period, progress=False)
                if len(df) > 0:
                    data[ticker] = df
            except Exception as e:
                print(f"[MENA] Error fetching {ticker}: {e}")

        return data

    def get_performance_summary(self) -> Dict:
        """중동 시장 성과 요약"""
        etf_data = self.get_etf_data('3mo')

        summary = {
            'timestamp': datetime.now().isoformat(),
            'etfs': {}
        }

        for ticker, df in etf_data.items():
            if len(df) < 2:
                continue

            close = df['Close']
            current = float(close.iloc[-1])
            prev_day = float(close.iloc[-2]) if len(close) > 1 else current
            prev_week = float(close.iloc[-5]) if len(close) > 5 else current
            prev_month = float(close.iloc[-22]) if len(close) > 22 else current

            summary['etfs'][ticker] = {
                'name': self.MENA_TICKERS.get(ticker, ticker),
                'price': current,
                'return_1d': (current / prev_day - 1) * 100,
                'return_1w': (current / prev_week - 1) * 100,
                'return_1m': (current / prev_month - 1) * 100,
                'volume': float(df['Volume'].iloc[-1])
            }

        # 지역별 평균 수익률
        if summary['etfs']:
            returns_1m = [e['return_1m'] for e in summary['etfs'].values()]
            summary['avg_return_1m'] = sum(returns_1m) / len(returns_1m)

        return summary

    def get_oil_correlation(self) -> Dict:
        """중동 시장과 유가 상관관계"""
        try:
            # 데이터 수집
            tickers = ['KSA', 'UAE', 'USO']  # 사우디, UAE, 유가
            data = {}
            for ticker in tickers:
                df = self.yf.download(ticker, period='1y', progress=False)
                if len(df) > 0:
                    data[ticker] = df['Close']

            if len(data) < 3:
                return {}

            # 상관관계 계산
            combined = pd.DataFrame(data)
            returns = combined.pct_change().dropna()
            corr = returns.corr()

            return {
                'timestamp': datetime.now().isoformat(),
                'ksa_oil_corr': float(corr.loc['KSA', 'USO']),
                'uae_oil_corr': float(corr.loc['UAE', 'USO']),
                'interpretation': self._interpret_oil_corr(float(corr.loc['KSA', 'USO']))
            }
        except Exception as e:
            print(f"[MENA] Error calculating oil correlation: {e}")
            return {}

    def _interpret_oil_corr(self, corr: float) -> str:
        """유가 상관관계 해석"""
        if corr > 0.7:
            return "강한 양의 상관관계 - 유가 상승 시 중동 시장 상승 기대"
        elif corr > 0.4:
            return "중간 양의 상관관계 - 유가가 중동 시장에 부분적 영향"
        elif corr > 0:
            return "약한 양의 상관관계 - 유가 외 요인이 더 중요"
        else:
            return "음의 상관관계 - 비정상적 상황, 추가 분석 필요"


# ============================================================================
# 통합 데이터 수집기
# ============================================================================

class ExtendedDataCollector:
    """확장 데이터 통합 수집"""

    def __init__(self):
        self.defi = DeFiLlamaCollector()
        self.mena = MiddleEastMarketCollector()

    def collect_all(self) -> Dict:
        """모든 확장 데이터 수집"""
        print("[ExtendedData] Collecting on-chain data...")
        onchain = self.defi.get_summary()

        print("[ExtendedData] Collecting MENA market data...")
        mena = self.mena.get_performance_summary()
        oil_corr = self.mena.get_oil_correlation()

        return {
            'timestamp': datetime.now().isoformat(),
            'onchain': onchain,
            'mena_markets': mena,
            'mena_oil_correlation': oil_corr
        }

    def get_risk_signals(self) -> List[Dict]:
        """확장 데이터 기반 리스크 시그널"""
        signals = []

        # 1. 스테이블코인 De-peg 체크
        stablecoins = self.defi.get_stablecoins()
        for s in stablecoins:
            if s.peg_deviation > 0.5:  # 0.5% 이상 괴리
                signals.append({
                    'type': 'stablecoin_depeg',
                    'severity': 'HIGH' if s.peg_deviation > 1.0 else 'MEDIUM',
                    'symbol': s.symbol,
                    'deviation': s.peg_deviation,
                    'message': f"{s.symbol} 페깅 이탈 {s.peg_deviation:.2f}%"
                })

        # 2. TVL 급락 체크
        eth_tvl = self.defi.get_chain_tvl('Ethereum')
        if eth_tvl.get('tvl_change_7d', 0) < -10:
            signals.append({
                'type': 'tvl_drop',
                'severity': 'HIGH',
                'chain': 'Ethereum',
                'change': eth_tvl['tvl_change_7d'],
                'message': f"Ethereum TVL 7일간 {eth_tvl['tvl_change_7d']:.1f}% 감소"
            })

        # 3. 중동 시장 급변 체크
        mena = self.mena.get_performance_summary()
        for ticker, data in mena.get('etfs', {}).items():
            if abs(data.get('return_1d', 0)) > 3:  # 일일 3% 이상 변동
                signals.append({
                    'type': 'mena_volatility',
                    'severity': 'MEDIUM',
                    'ticker': ticker,
                    'return_1d': data['return_1d'],
                    'message': f"{ticker} 일일 {data['return_1d']:+.1f}% 변동"
                })

        return signals


# ============================================================================
# 테스트
# ============================================================================

if __name__ == "__main__":
    print("Extended Data Sources Test")
    print("=" * 60)

    collector = ExtendedDataCollector()

    # 온체인 데이터
    print("\n[1] On-Chain Data (DeFiLlama)")
    print("-" * 40)
    onchain = collector.defi.get_summary()
    print(f"  Total TVL: ${onchain.get('total_tvl', 0)/1e9:.2f}B")
    print(f"  Stablecoin Market Cap: ${onchain.get('stablecoin_market_cap', 0)/1e9:.2f}B")
    print("  Top Stablecoins:")
    for s in onchain.get('top_stablecoins', [])[:3]:
        print(f"    - {s['symbol']}: ${s['circulating']/1e9:.2f}B, peg: {s['peg_deviation']:.3f}%")

    # 중동 시장
    print("\n[2] Middle East Markets")
    print("-" * 40)
    mena = collector.mena.get_performance_summary()
    for ticker, data in mena.get('etfs', {}).items():
        print(f"  {ticker}: ${data['price']:.2f}, 1M: {data['return_1m']:+.1f}%")

    # 유가 상관관계
    oil_corr = collector.mena.get_oil_correlation()
    if oil_corr:
        print(f"\n  Oil Correlation:")
        print(f"    KSA-Oil: {oil_corr.get('ksa_oil_corr', 0):.2f}")
        print(f"    Interpretation: {oil_corr.get('interpretation', 'N/A')}")

    # 리스크 시그널
    print("\n[3] Risk Signals")
    print("-" * 40)
    signals = collector.get_risk_signals()
    if signals:
        for s in signals:
            print(f"  [{s['severity']}] {s['message']}")
    else:
        print("  No risk signals detected")

    print("\n" + "=" * 60)
