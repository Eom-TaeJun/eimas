#!/usr/bin/env python3
"""
Crypto Collector (24/7)
=======================
암호화폐 24시간 수집 및 이상 탐지 시스템

주말/휴일에도 작동하는 암호화폐 모니터링:
- 실시간 가격 수집 (BTC, ETH, SOL 등)
- 이상 탐지 (급등/급락, 거래량 폭발)
- 뉴스 검색으로 원인 분석

사용법:
    # 현재 상태 스냅샷
    python lib/crypto_collector.py

    # 이상 탐지 실행
    python lib/crypto_collector.py --detect

    # 원인 분석 포함
    python lib/crypto_collector.py --detect --analyze

    # 특정 코인만
    python lib/crypto_collector.py --coins BTC,ETH,SOL
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from data.volatile_store import VolatileStore

# Perplexity API for news search
try:
    from openai import OpenAI
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False

import os


class CryptoCollector:
    """24/7 암호화폐 수집기"""

    # 기본 수집 대상 (yfinance 심볼)
    DEFAULT_CRYPTOS = {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'SOL-USD': 'Solana',
        'XRP-USD': 'Ripple',
        'ADA-USD': 'Cardano',
        'DOGE-USD': 'Dogecoin',
        'AVAX-USD': 'Avalanche',
        'DOT-USD': 'Polkadot',
        'MATIC-USD': 'Polygon',
        'LINK-USD': 'Chainlink',
    }

    # 이상 감지 임계값
    THRESHOLDS = {
        'price_spike_1h_pct': 5.0,      # 1시간 내 5% 이상 변동
        'price_spike_15m_pct': 3.0,     # 15분 내 3% 이상 변동
        'volume_spike_ratio': 3.0,       # 평균 대비 3배 이상 거래량
        'volatility_spike_std': 2.5,     # 2.5 표준편차 이상
    }

    def __init__(self, cryptos: Dict[str, str] = None):
        self.cryptos = cryptos or self.DEFAULT_CRYPTOS
        self.volatile_store = VolatileStore()

        # Perplexity client
        self.perplexity_client = None
        if PERPLEXITY_AVAILABLE:
            api_key = os.environ.get('PERPLEXITY_API_KEY')
            if api_key:
                self.perplexity_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.perplexity.ai"
                )

        print(f"[CryptoCollector] Initialized with {len(self.cryptos)} cryptos")
        if self.perplexity_client:
            print(f"[CryptoCollector] Perplexity API: ✅ Available")
        else:
            print(f"[CryptoCollector] Perplexity API: ❌ Not configured")

    # ========================================================================
    # 데이터 수집
    # ========================================================================

    def collect_current_prices(self) -> Dict[str, Dict]:
        """현재 가격 수집"""
        print(f"\n📥 암호화폐 현재가 수집 중...")

        results = {}
        tickers = list(self.cryptos.keys())

        # 최근 2일 데이터로 변화율 계산
        end = datetime.now()
        start = end - timedelta(days=2)

        for ticker in tickers:
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(start=start, end=end, interval="1h")

                if hist.empty:
                    continue

                # MultiIndex 처리
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)

                current = float(hist['Close'].iloc[-1])

                # 변화율 계산
                if len(hist) >= 2:
                    prev_1h = float(hist['Close'].iloc[-2])
                    change_1h = (current / prev_1h - 1) * 100
                else:
                    change_1h = 0

                if len(hist) >= 24:
                    prev_24h = float(hist['Close'].iloc[-24])
                    change_24h = (current / prev_24h - 1) * 100
                else:
                    change_24h = 0

                # 거래량
                volume_24h = float(hist['Volume'].tail(24).sum()) if len(hist) >= 24 else float(hist['Volume'].sum())
                avg_volume = float(hist['Volume'].mean())
                volume_ratio = volume_24h / (avg_volume * 24) if avg_volume > 0 else 1

                results[ticker] = {
                    'name': self.cryptos[ticker],
                    'price': current,
                    'change_1h': change_1h,
                    'change_24h': change_24h,
                    'volume_24h': volume_24h,
                    'volume_ratio': volume_ratio,
                    'high_24h': float(hist['High'].tail(24).max()),
                    'low_24h': float(hist['Low'].tail(24).min()),
                    'timestamp': datetime.now().isoformat()
                }

                symbol = ticker.replace('-USD', '')
                direction = '🟢' if change_1h >= 0 else '🔴'
                print(f"  {direction} {symbol}: ${current:,.2f} ({change_1h:+.2f}% 1H, {change_24h:+.2f}% 24H)")

            except Exception as e:
                print(f"  ❌ {ticker}: {e}")

        return results

    def collect_intraday_data(self, ticker: str, period: str = "1d", interval: str = "5m") -> Optional[pd.DataFrame]:
        """장중 데이터 수집"""
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period=period, interval=interval)

            if df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            return df
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            return None

    # ========================================================================
    # 이상 탐지
    # ========================================================================

    def detect_anomalies(self) -> List[Dict]:
        """이상 탐지 실행"""
        print(f"\n🔍 이상 탐지 실행 중...")

        anomalies = []

        for ticker, name in self.cryptos.items():
            # 최근 24시간 5분봉 데이터
            df = self.collect_intraday_data(ticker, period="1d", interval="5m")

            if df is None or len(df) < 10:
                continue

            symbol = ticker.replace('-USD', '')
            current_price = float(df['Close'].iloc[-1])

            # 1. 15분 내 급등/급락 감지
            df['pct_change_3'] = df['Close'].pct_change(3) * 100  # 15분 (3 x 5분)

            spikes_15m = df[abs(df['pct_change_3']) >= self.THRESHOLDS['price_spike_15m_pct']]
            for idx, row in spikes_15m.tail(5).iterrows():  # 최근 5개만
                direction = 'surge' if row['pct_change_3'] > 0 else 'crash'
                anomalies.append({
                    'timestamp': idx.isoformat(),
                    'ticker': ticker,
                    'symbol': symbol,
                    'name': name,
                    'alert_type': f'price_{direction}_15m',
                    'value': float(row['pct_change_3']),
                    'threshold': self.THRESHOLDS['price_spike_15m_pct'],
                    'price_at_alert': float(row['Close']),
                    'description': f"{symbol} {row['pct_change_3']:+.2f}% 변동 (15분)"
                })

            # 2. 1시간 내 급등/급락 감지
            df['pct_change_12'] = df['Close'].pct_change(12) * 100  # 1시간 (12 x 5분)

            spikes_1h = df[abs(df['pct_change_12']) >= self.THRESHOLDS['price_spike_1h_pct']]
            for idx, row in spikes_1h.tail(3).iterrows():  # 최근 3개만
                direction = 'surge' if row['pct_change_12'] > 0 else 'crash'
                anomalies.append({
                    'timestamp': idx.isoformat(),
                    'ticker': ticker,
                    'symbol': symbol,
                    'name': name,
                    'alert_type': f'price_{direction}_1h',
                    'value': float(row['pct_change_12']),
                    'threshold': self.THRESHOLDS['price_spike_1h_pct'],
                    'price_at_alert': float(row['Close']),
                    'description': f"{symbol} {row['pct_change_12']:+.2f}% 변동 (1시간)"
                })

            # 3. 거래량 폭발 감지
            avg_volume = df['Volume'].mean()
            if avg_volume > 0:
                df['volume_ratio'] = df['Volume'] / avg_volume
                volume_spikes = df[df['volume_ratio'] >= self.THRESHOLDS['volume_spike_ratio']]

                for idx, row in volume_spikes.tail(3).iterrows():
                    anomalies.append({
                        'timestamp': idx.isoformat(),
                        'ticker': ticker,
                        'symbol': symbol,
                        'name': name,
                        'alert_type': 'volume_explosion',
                        'value': float(row['volume_ratio']),
                        'threshold': self.THRESHOLDS['volume_spike_ratio'],
                        'price_at_alert': float(row['Close']),
                        'volume_ratio': float(row['volume_ratio']),
                        'description': f"{symbol} 거래량 {row['volume_ratio']:.1f}배 폭발"
                    })

            # 4. 변동성 급등 감지
            df['returns'] = df['Close'].pct_change()
            rolling_std = df['returns'].rolling(12).std()  # 1시간 롤링

            if len(rolling_std.dropna()) > 0:
                mean_std = rolling_std.mean()
                std_std = rolling_std.std()

                if std_std > 0:
                    df['volatility_z'] = (rolling_std - mean_std) / std_std
                    vol_spikes = df[df['volatility_z'] >= self.THRESHOLDS['volatility_spike_std']]

                    for idx, row in vol_spikes.tail(2).iterrows():
                        anomalies.append({
                            'timestamp': idx.isoformat(),
                            'ticker': ticker,
                            'symbol': symbol,
                            'name': name,
                            'alert_type': 'volatility_spike',
                            'value': float(row['volatility_z']),
                            'threshold': self.THRESHOLDS['volatility_spike_std'],
                            'price_at_alert': float(row['Close']),
                            'description': f"{symbol} 변동성 급등 ({row['volatility_z']:.1f}σ)"
                        })

        # 중복 제거 (같은 시간대 같은 티커)
        seen = set()
        unique_anomalies = []
        for a in anomalies:
            key = (a['timestamp'][:16], a['ticker'], a['alert_type'])
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(a)

        print(f"\n⚠️ 총 {len(unique_anomalies)}개 이상 감지됨")

        return unique_anomalies

    # ========================================================================
    # 뉴스 검색 (원인 분석)
    # ========================================================================

    def search_news_for_anomaly(self, anomaly: Dict) -> Optional[str]:
        """Perplexity API로 이상 원인 검색"""
        if not self.perplexity_client:
            return None

        symbol = anomaly.get('symbol', '')
        name = anomaly.get('name', '')
        alert_type = anomaly.get('alert_type', '')
        value = anomaly.get('value', 0)

        # 검색 쿼리 생성
        if 'surge' in alert_type or 'crash' in alert_type:
            direction = "급등" if 'surge' in alert_type else "급락"
            query = f"{name} ({symbol}) {direction} 이유 원인 뉴스 {datetime.now().strftime('%Y-%m-%d')}"
        elif 'volume' in alert_type:
            query = f"{name} ({symbol}) 거래량 급증 이유 뉴스 {datetime.now().strftime('%Y-%m-%d')}"
        else:
            query = f"{name} ({symbol}) 암호화폐 뉴스 {datetime.now().strftime('%Y-%m-%d')}"

        try:
            response = self.perplexity_client.chat.completions.create(
                model="sonar",  # 2025년 현재 모델명
                messages=[
                    {
                        "role": "system",
                        "content": "You are a crypto market analyst. Provide a brief, factual summary of recent news that might explain the price movement. Answer in Korean. Keep it under 3 sentences."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=300
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"  뉴스 검색 실패: {e}")
            return None

    def analyze_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """이상 원인 분석"""
        if not anomalies:
            return anomalies

        print(f"\n📰 원인 분석 중 (Perplexity API)...")

        # 중요한 이상만 분석 (상위 5개)
        important = sorted(anomalies, key=lambda x: abs(x.get('value', 0)), reverse=True)[:5]

        for anomaly in important:
            symbol = anomaly.get('symbol', '')
            print(f"  🔍 {symbol} 분석 중...")

            news = self.search_news_for_anomaly(anomaly)
            if news:
                anomaly['news_analysis'] = news
                print(f"     → {news[:80]}...")

        return anomalies

    # ========================================================================
    # 저장 및 리포트
    # ========================================================================

    def save_anomalies(self, anomalies: List[Dict]) -> int:
        """이상 감지 결과 저장"""
        saved = 0
        for anomaly in anomalies:
            if self.volatile_store.save_detected_event({
                'ticker': anomaly.get('ticker'),
                'event_type': anomaly.get('alert_type'),
                'value': anomaly.get('value'),
                'threshold': anomaly.get('threshold'),
                'price_at_event': anomaly.get('price_at_alert'),
                'volume_ratio': anomaly.get('volume_ratio'),
                'importance': 'HIGH' if abs(anomaly.get('value', 0)) > 5 else 'MEDIUM',
                'description': anomaly.get('description'),
                'metadata': {
                    'symbol': anomaly.get('symbol'),
                    'name': anomaly.get('name'),
                    'news_analysis': anomaly.get('news_analysis')
                }
            }):
                saved += 1

        print(f"\n💾 {saved}개 이상 이벤트 저장됨")
        return saved

    def save_snapshot(self, prices: Dict[str, Dict]):
        """시장 스냅샷 저장"""
        # BTC 기준 스냅샷
        btc = prices.get('BTC-USD', {})
        eth = prices.get('ETH-USD', {})

        snapshot = {
            'collection_type': 'crypto_24_7',
            'spy_price': btc.get('price'),  # BTC를 spy_price에 저장
            'spy_change_pct': btc.get('change_24h'),
            'qqq_price': eth.get('price'),  # ETH를 qqq_price에 저장
            'notes': json.dumps({
                'type': 'crypto',
                'btc': btc,
                'eth': eth,
                'total_cryptos': len(prices)
            })
        }

        self.volatile_store.save_market_snapshot(snapshot)

    def generate_report(self, prices: Dict, anomalies: List[Dict]) -> str:
        """리포트 생성"""
        report = []
        report.append("")
        report.append("=" * 70)
        report.append("🪙 EIMAS 암호화폐 24/7 모니터링 리포트")
        report.append("=" * 70)
        report.append(f"생성시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 현재 시장 현황
        report.append("-" * 70)
        report.append("📊 현재 시장 현황")
        report.append("-" * 70)

        for ticker, data in sorted(prices.items(), key=lambda x: x[1].get('price', 0) * -1):
            symbol = ticker.replace('-USD', '')
            price = data.get('price', 0)
            change_1h = data.get('change_1h', 0)
            change_24h = data.get('change_24h', 0)

            icon = '🟢' if change_24h >= 0 else '🔴'
            report.append(f"  {icon} {symbol:<6} ${price:>12,.2f}   1H: {change_1h:>+6.2f}%   24H: {change_24h:>+6.2f}%")

        # 이상 감지
        if anomalies:
            report.append("")
            report.append("-" * 70)
            report.append(f"⚠️ 이상 감지: {len(anomalies)}건")
            report.append("-" * 70)

            for a in anomalies[:10]:  # 상위 10개
                ts = a.get('timestamp', '')
                if len(ts) > 16:
                    ts = ts[11:16]  # HH:MM만

                symbol = a.get('symbol', '')
                alert_type = a.get('alert_type', '')
                value = a.get('value', 0)
                desc = a.get('description', '')

                report.append(f"  [{ts}] {desc}")

                # 뉴스 분석 결과
                if a.get('news_analysis'):
                    report.append(f"         → {a['news_analysis'][:60]}...")
        else:
            report.append("")
            report.append("✅ 이상 감지 없음")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)

    # ========================================================================
    # 메인 실행
    # ========================================================================

    def run(self, detect: bool = True, analyze: bool = False) -> Dict:
        """전체 실행"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'prices': {},
            'anomalies': [],
            'saved': 0
        }

        # 1. 현재 가격 수집
        prices = self.collect_current_prices()
        results['prices'] = prices

        # 2. 스냅샷 저장
        self.save_snapshot(prices)

        # 3. 이상 탐지
        if detect:
            anomalies = self.detect_anomalies()

            # 4. 원인 분석 (선택)
            if analyze and anomalies:
                anomalies = self.analyze_anomalies(anomalies)

            results['anomalies'] = anomalies

            # 5. 저장
            if anomalies:
                results['saved'] = self.save_anomalies(anomalies)

        # 6. 리포트 출력
        report = self.generate_report(prices, results.get('anomalies', []))
        print(report)

        return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Crypto 24/7 Collector')
    parser.add_argument('--detect', action='store_true', help='Run anomaly detection')
    parser.add_argument('--analyze', action='store_true', help='Analyze anomalies with news')
    parser.add_argument('--coins', type=str, help='Comma-separated coins (e.g., BTC,ETH,SOL)')

    args = parser.parse_args()

    # 커스텀 코인 설정
    cryptos = None
    if args.coins:
        coins = [c.strip().upper() for c in args.coins.split(',')]
        cryptos = {f"{c}-USD": c for c in coins}

    collector = CryptoCollector(cryptos=cryptos)
    collector.run(detect=args.detect or True, analyze=args.analyze)
