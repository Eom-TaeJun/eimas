"""
Market Data Pipeline - 무료 API 기반 다중 자산 수집

지원 Provider:
- Twelve Data: 미국 주식, 원자재, FX (무료: 800 calls/day, 8 calls/min)
- CryptoCompare: 암호화폐 (무료: 100,000 calls/month)

사용법:
    python lib/market_data_pipeline.py
    python lib/market_data_pipeline.py --provider twelvedata --symbol AAPL
    python lib/market_data_pipeline.py --all
"""

import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional
from dotenv import load_dotenv

import pandas as pd
import requests


# .env 파일 로드
load_dotenv()


class DataProvider(ABC):
    """데이터 제공자 추상 클래스"""

    @abstractmethod
    def fetch(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """데이터 조회 - 반환: DataFrame(datetime index, OHLCV columns)"""
        pass

    @abstractmethod
    def get_rate_limit_delay(self) -> float:
        """API 호출 간 대기 시간 (초)"""
        pass


class TwelveDataProvider(DataProvider):
    """
    Twelve Data API Provider

    무료 플랜 제한:
    - 800 API calls/day
    - 8 API calls/minute
    - End-of-day data only (실시간은 유료)

    지원 자산:
    - 주식: AAPL, MSFT, GOOGL 등
    - FX: EUR/USD, USD/JPY 등
    - 원자재: XAU/USD (금), XAG/USD (은), 원유는 ETF로 대체
    """

    BASE_URL = "https://api.twelvedata.com"

    # 간격 매핑
    INTERVAL_MAP = {
        '1min': '1min',
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '1h': '1h',
        '4h': '4h',
        '1d': '1day',
        '1day': '1day',
        '1week': '1week',
        '1month': '1month',
    }

    def __init__(self):
        self.api_key = os.getenv('TWELVEDATA_API_KEY')
        if not self.api_key:
            raise ValueError("TWELVEDATA_API_KEY not found in environment variables")

    def fetch(self, symbol: str, interval: str = '1d', limit: int = 100) -> pd.DataFrame:
        """Twelve Data에서 OHLCV 데이터 조회"""

        # 간격 변환
        td_interval = self.INTERVAL_MAP.get(interval, interval)

        # 심볼 정리 (슬래시 유지)
        clean_symbol = symbol.replace('-', '/')

        url = f"{self.BASE_URL}/time_series"
        params = {
            'symbol': clean_symbol,
            'interval': td_interval,
            'outputsize': limit,
            'apikey': self.api_key,
            'format': 'JSON',
        }

        print(f"  📡 Twelve Data: {clean_symbol} ({td_interval})")

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # 에러 체크
            if 'status' in data and data['status'] == 'error':
                print(f"  ❌ API 에러: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()

            if 'values' not in data:
                print(f"  ⚠️ 데이터 없음: {data}")
                return pd.DataFrame()

            # DataFrame 변환
            df = pd.DataFrame(data['values'])

            # 컬럼명 표준화
            df = df.rename(columns={
                'datetime': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })

            # 타입 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            if 'volume' in df.columns:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
            else:
                df['volume'] = 0  # 원자재/FX는 거래량 없음

            # 시간순 정렬
            df = df.sort_index()

            print(f"  ✅ {len(df)}개 레코드 조회")
            return df[['open', 'high', 'low', 'close', 'volume']]

        except requests.exceptions.RequestException as e:
            print(f"  ❌ 요청 실패: {e}")
            return pd.DataFrame()

    def get_rate_limit_delay(self) -> float:
        """8 calls/min = 7.5초 간격 (안전 마진 포함)"""
        return 8.0


class CryptoCompareProvider(DataProvider):
    """
    CryptoCompare API Provider

    무료 플랜 제한:
    - 100,000 calls/month
    - Rate limit: 없음 (합리적인 사용 시)

    지원 자산:
    - 암호화폐: BTC, ETH, SOL 등 수천 개
    """

    BASE_URL = "https://min-api.cryptocompare.com/data/v2"

    # 간격별 엔드포인트
    INTERVAL_ENDPOINTS = {
        '1min': 'histominute',
        '5min': 'histominute',  # aggregate로 처리
        '15min': 'histominute',
        '30min': 'histominute',
        '1h': 'histohour',
        '4h': 'histohour',
        '1d': 'histoday',
        '1day': 'histoday',
    }

    # 간격별 aggregate 값
    AGGREGATE_MAP = {
        '1min': 1,
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '1h': 1,
        '4h': 4,
        '1d': 1,
        '1day': 1,
    }

    def __init__(self):
        self.api_key = os.getenv('CRYPTOCOMPARE_API_KEY', '')
        # CryptoCompare는 API 키 없이도 기본 호출 가능 (제한적)

    def fetch(self, symbol: str, interval: str = '1d', limit: int = 100) -> pd.DataFrame:
        """CryptoCompare에서 OHLCV 데이터 조회"""

        # 심볼 파싱 (BTC-USD → fsym=BTC, tsym=USD)
        if '-' in symbol:
            fsym, tsym = symbol.split('-')
        elif '/' in symbol:
            fsym, tsym = symbol.split('/')
        else:
            fsym, tsym = symbol, 'USD'

        # 엔드포인트 결정
        endpoint = self.INTERVAL_ENDPOINTS.get(interval, 'histoday')
        aggregate = self.AGGREGATE_MAP.get(interval, 1)

        url = f"{self.BASE_URL}/{endpoint}"
        params = {
            'fsym': fsym.upper(),
            'tsym': tsym.upper(),
            'limit': limit,
            'aggregate': aggregate,
        }

        headers = {}
        if self.api_key:
            headers['authorization'] = f'Apikey {self.api_key}'

        print(f"  📡 CryptoCompare: {fsym}/{tsym} ({interval})")

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # 에러 체크
            if data.get('Response') == 'Error':
                print(f"  ❌ API 에러: {data.get('Message', 'Unknown error')}")
                return pd.DataFrame()

            if 'Data' not in data or 'Data' not in data['Data']:
                print(f"  ⚠️ 데이터 없음")
                return pd.DataFrame()

            # DataFrame 변환
            df = pd.DataFrame(data['Data']['Data'])

            # 타임스탬프 변환
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('timestamp', inplace=True)

            # 컬럼명 표준화
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume'  # 기준 통화 거래량
            })

            # 시간순 정렬
            df = df.sort_index()

            # 0 값 행 제거 (데이터 없는 기간)
            df = df[df['close'] > 0]

            print(f"  ✅ {len(df)}개 레코드 조회")
            return df[['open', 'high', 'low', 'close', 'volume']]

        except requests.exceptions.RequestException as e:
            print(f"  ❌ 요청 실패: {e}")
            return pd.DataFrame()

    def get_rate_limit_delay(self) -> float:
        """CryptoCompare는 관대함 - 1초면 충분"""
        return 1.0


class YFinanceProvider(DataProvider):
    """
    yfinance Provider (백업용)

    제한: 없음 (비공식 API)
    주의: 과도한 호출 시 차단될 수 있음
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

    def fetch(self, symbol: str, interval: str = '1d', limit: int = 100) -> pd.DataFrame:
        """yfinance에서 OHLCV 데이터 조회"""

        # 간격 매핑
        yf_interval_map = {
            '1min': '1m',
            '5min': '5m',
            '15min': '15m',
            '30min': '30m',
            '1h': '1h',
            '4h': '4h',  # 지원 안함, 1h로 대체
            '1d': '1d',
            '1day': '1d',
        }

        yf_interval = yf_interval_map.get(interval, '1d')

        # 기간 계산 (limit에 따라)
        period_map = {
            '1m': '7d',   # 1분봉은 최대 7일
            '5m': '60d',
            '15m': '60d',
            '30m': '60d',
            '1h': '730d',
            '1d': 'max',
        }
        period = period_map.get(yf_interval, '1y')

        # 심볼 정리
        clean_symbol = symbol.replace('/', '-').replace('_', '-')

        print(f"  📡 yfinance: {clean_symbol} ({yf_interval})")

        try:
            ticker = self.yf.Ticker(clean_symbol)
            df = ticker.history(period=period, interval=yf_interval)

            if df.empty:
                print(f"  ⚠️ 데이터 없음")
                return pd.DataFrame()

            # 컬럼명 소문자로
            df.columns = df.columns.str.lower()

            # 필요한 컬럼만
            df = df[['open', 'high', 'low', 'close', 'volume']]

            # limit 적용
            if len(df) > limit:
                df = df.tail(limit)

            print(f"  ✅ {len(df)}개 레코드 조회")
            return df

        except Exception as e:
            print(f"  ❌ 조회 실패: {e}")
            return pd.DataFrame()

    def get_rate_limit_delay(self) -> float:
        """yfinance는 2초 권장"""
        return 2.0


# Provider 레지스트리
PROVIDERS = {
    'twelvedata': TwelveDataProvider,
    'cryptocompare': CryptoCompareProvider,
    'yfinance': YFinanceProvider,
}


def fetch_data(
    provider: str,
    symbol: str,
    interval: str = '1d',
    limit: int = 100
) -> pd.DataFrame:
    """
    공통 데이터 조회 인터페이스

    Args:
        provider: 'twelvedata', 'cryptocompare', 'yfinance'
        symbol: 자산 심볼 (예: 'AAPL', 'BTC-USD', 'XAU/USD')
        interval: 간격 (예: '1min', '5min', '1h', '1d')
        limit: 조회할 데이터 개수

    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(PROVIDERS.keys())}")

    provider_instance = PROVIDERS[provider]()
    return provider_instance.fetch(symbol, interval, limit)


def save_data(
    df: pd.DataFrame,
    provider: str,
    symbol: str,
    interval: str,
    data_dir: str = None
) -> str:
    """
    데이터를 CSV로 저장

    저장 경로: data/{provider}_{symbol}_{interval}.csv
    """
    if df.empty:
        return ""

    # 데이터 디렉토리
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'market'
        )

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    # 파일명 (심볼의 특수문자 치환)
    clean_symbol = symbol.replace('/', '_').replace('-', '_')
    filename = f"{provider}_{clean_symbol}_{interval}.csv"
    filepath = os.path.join(data_dir, filename)

    # 저장
    df.to_csv(filepath)
    print(f"  💾 저장: {filepath}")

    return filepath


def run_pipeline(
    assets: dict[str, list[tuple[str, str]]],  # {provider: [(symbol, interval), ...]}
    limit: int = 100,
    data_dir: str = None
) -> dict[str, pd.DataFrame]:
    """
    전체 파이프라인 실행

    Args:
        assets: 수집할 자산 딕셔너리
            예: {
                'twelvedata': [('AAPL', '1d'), ('MSFT', '1d')],
                'cryptocompare': [('BTC-USD', '1h'), ('ETH-USD', '1h')]
            }
        limit: 각 자산당 데이터 개수
        data_dir: 저장 디렉토리

    Returns:
        {symbol: DataFrame} 딕셔너리
    """
    results = {}

    for provider, symbols in assets.items():
        print(f"\n{'='*50}")
        print(f"Provider: {provider.upper()}")
        print('='*50)

        try:
            provider_instance = PROVIDERS[provider]()
            delay = provider_instance.get_rate_limit_delay()
        except Exception as e:
            print(f"❌ Provider 초기화 실패: {e}")
            continue

        for i, (symbol, interval) in enumerate(symbols):
            print(f"\n[{i+1}/{len(symbols)}] {symbol}")

            try:
                df = fetch_data(provider, symbol, interval, limit)

                if not df.empty:
                    save_data(df, provider, symbol, interval, data_dir)
                    results[f"{provider}_{symbol}"] = df

                # Rate limit 대기 (마지막 요청 제외)
                if i < len(symbols) - 1:
                    print(f"  ⏳ {delay}초 대기 (rate limit)")
                    time.sleep(delay)

            except Exception as e:
                print(f"  ❌ 실패: {e}")

    return results


# 기본 자산 설정
DEFAULT_ASSETS = {
    'twelvedata': [
        # 미국 주식
        ('AAPL', '1d'),
        ('MSFT', '1d'),
        # 원자재 (Twelve Data 지원 심볼)
        ('XAU/USD', '1d'),  # 금
        ('XAG/USD', '1d'),  # 은
    ],
    'cryptocompare': [
        # 암호화폐
        ('BTC-USD', '1d'),
        ('ETH-USD', '1d'),
    ],
}

# 원유는 yfinance 백업 사용 (Twelve Data 무료에서 제한적)
BACKUP_ASSETS = {
    'yfinance': [
        ('CL=F', '1d'),   # WTI 원유 선물
        ('BZ=F', '1d'),   # 브렌트 원유 선물
    ],
}


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description='Market Data Pipeline')
    parser.add_argument('--provider', choices=list(PROVIDERS.keys()),
                        help='Data provider')
    parser.add_argument('--symbol', help='Asset symbol (e.g., AAPL, BTC-USD)')
    parser.add_argument('--interval', default='1d',
                        help='Time interval (default: 1d)')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of records (default: 100)')
    parser.add_argument('--all', action='store_true',
                        help='Fetch all default assets')
    parser.add_argument('--with-oil', action='store_true',
                        help='Include oil futures (via yfinance)')

    args = parser.parse_args()

    print("=" * 60)
    print("Market Data Pipeline - 무료 API 기반 다중 자산 수집")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if args.all:
        # 전체 기본 자산 수집
        assets = DEFAULT_ASSETS.copy()

        if args.with_oil:
            assets.update(BACKUP_ASSETS)

        results = run_pipeline(assets, limit=args.limit)

        print(f"\n{'='*60}")
        print(f"완료: {len(results)}개 자산 수집")
        print("=" * 60)

    elif args.provider and args.symbol:
        # 단일 자산 수집
        df = fetch_data(args.provider, args.symbol, args.interval, args.limit)

        if not df.empty:
            save_data(df, args.provider, args.symbol, args.interval)
            print(f"\n최근 5개 레코드:")
            print(df.tail())

    else:
        # 기본: 도움말 출력
        print("\n사용 예:")
        print("  # 전체 기본 자산 수집")
        print("  python lib/market_data_pipeline.py --all")
        print()
        print("  # 원유 포함 수집")
        print("  python lib/market_data_pipeline.py --all --with-oil")
        print()
        print("  # 단일 자산 수집")
        print("  python lib/market_data_pipeline.py --provider twelvedata --symbol AAPL")
        print("  python lib/market_data_pipeline.py --provider cryptocompare --symbol BTC-USD --interval 1h")
        print()
        print("기본 자산:")
        for provider, symbols in DEFAULT_ASSETS.items():
            print(f"  {provider}:")
            for sym, interval in symbols:
                print(f"    - {sym} ({interval})")


if __name__ == '__main__':
    main()
