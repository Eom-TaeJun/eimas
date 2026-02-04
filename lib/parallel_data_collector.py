#!/usr/bin/env python3
"""
Parallel Data Collector
=======================
병렬 데이터 수집으로 성능 최적화

Performance Target (TODO.md Priority 2):
- 기존: ~75초 (순차 수집)
- 목표: ~30초 (-60% 개선)
- 방법: ThreadPoolExecutor로 티커 병렬 다운로드

Economic Foundation:
- 시장 데이터는 독립적 → I/O bound → 병렬화 적합
- Network latency 감소가 주요 개선 포인트

Usage:
    from lib.parallel_data_collector import ParallelMarketCollector

    collector = ParallelMarketCollector()
    data = collector.collect_all(tickers=['SPY', 'QQQ', ...], max_workers=10)
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import time
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ParallelMarketCollector:
    """병렬 시장 데이터 수집기"""

    def __init__(self, lookback_days: int = 365, max_workers: int = 10):
        """
        Args:
            lookback_days: 과거 데이터 기간
            max_workers: 병렬 워커 수 (기본 10)
        """
        self.lookback_days = lookback_days
        self.max_workers = max_workers
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)

    def _fetch_single_ticker(self, ticker: str) -> Dict[str, any]:
        """
        단일 티커 다운로드 (thread-safe)

        Returns:
            {'ticker': str, 'data': DataFrame, 'success': bool, 'error': str}
        """
        result = {
            'ticker': ticker,
            'data': None,
            'success': False,
            'error': None,
            'fetch_time_ms': 0
        }

        start_time = time.time()

        try:
            data = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,
                threads=False  # Disable internal threading (we handle it)
            )

            if data.empty:
                result['error'] = "Empty data"
                return result

            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            result['data'] = data
            result['success'] = True
            result['fetch_time_ms'] = int((time.time() - start_time) * 1000)

        except Exception as e:
            result['error'] = str(e)
            result['fetch_time_ms'] = int((time.time() - start_time) * 1000)

        return result

    def collect_all(
        self,
        tickers: List[str],
        progress_callback: Optional[Callable[[str, bool], None]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        병렬 데이터 수집

        Args:
            tickers: 티커 리스트
            progress_callback: 진행 콜백 함수 (ticker, success)

        Returns:
            {ticker: DataFrame}
        """
        logger.info(f"Parallel collection: {len(tickers)} tickers, {self.max_workers} workers")

        results = {}
        failed = []
        total_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self._fetch_single_ticker, ticker): ticker
                for ticker in tickers
            }

            # Collect results as they complete
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()

                    if result['success']:
                        results[ticker] = result['data']
                        logger.debug(f"✓ {ticker} ({result['fetch_time_ms']}ms)")

                        if progress_callback:
                            progress_callback(ticker, True)
                    else:
                        failed.append((ticker, result['error']))
                        logger.warning(f"✗ {ticker}: {result['error']}")

                        if progress_callback:
                            progress_callback(ticker, False)

                except Exception as e:
                    failed.append((ticker, str(e)))
                    logger.error(f"✗ {ticker}: {e}")

                    if progress_callback:
                        progress_callback(ticker, False)

        total_time = time.time() - total_time

        # Summary
        logger.info(
            f"Collection complete: {len(results)}/{len(tickers)} successful "
            f"({len(failed)} failed) in {total_time:.1f}s"
        )

        if failed:
            logger.warning(f"Failed tickers: {[t for t, _ in failed]}")

        return results


class ParallelCryptoCollector:
    """병렬 크립토 데이터 수집기"""

    def __init__(self, lookback_days: int = 90, max_workers: int = 5):
        """
        Args:
            lookback_days: 과거 데이터 기간
            max_workers: 병렬 워커 수 (크립토는 5개 정도면 충분)
        """
        self.lookback_days = lookback_days
        self.max_workers = max_workers
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)

    def _fetch_single_crypto(self, ticker: str) -> Dict[str, any]:
        """단일 크립토 다운로드"""
        result = {
            'ticker': ticker,
            'data': None,
            'success': False,
            'error': None
        }

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
                result['error'] = "Empty data"
                return result

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            result['data'] = data
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)

        return result

    def collect_all(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """병렬 크립토 수집"""
        logger.info(f"Parallel crypto collection: {len(tickers)} assets")

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._fetch_single_crypto, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()

                    if result['success']:
                        results[ticker] = result['data']
                        logger.info(f"✓ {ticker}")
                    else:
                        logger.warning(f"✗ {ticker}: {result['error']}")

                except Exception as e:
                    logger.error(f"✗ {ticker}: {e}")

        logger.info(f"Crypto collection: {len(results)}/{len(tickers)} successful")

        return results


class ParallelFREDCollector:
    """병렬 FRED 데이터 수집기"""

    def __init__(self, api_key: str, lookback_days: int = 365, max_workers: int = 5):
        """
        Args:
            api_key: FRED API 키
            lookback_days: 과거 데이터 기간
            max_workers: 병렬 워커 수 (FRED API rate limit 고려)
        """
        self.api_key = api_key
        self.lookback_days = lookback_days
        self.max_workers = max_workers

        try:
            from fredapi import Fred
            self.fred = Fred(api_key=api_key)
        except ImportError:
            raise ImportError("fredapi not installed. Run: pip install fredapi")

    def _fetch_single_series(self, series_id: str) -> Dict[str, any]:
        """단일 FRED 시리즈 다운로드"""
        result = {
            'series_id': series_id,
            'data': None,
            'success': False,
            'error': None
        }

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)

            data = self.fred.get_series(
                series_id,
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )

            if data is None or len(data) == 0:
                result['error'] = "No data"
                return result

            result['data'] = pd.DataFrame({'value': data})
            result['success'] = True

        except Exception as e:
            result['error'] = str(e)

        return result

    def collect_all(self, series_map: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """
        병렬 FRED 시리즈 수집

        Args:
            series_map: {name: series_id} 매핑

        Returns:
            {name: DataFrame}
        """
        logger.info(f"Parallel FRED collection: {len(series_map)} series")

        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(self._fetch_single_series, series_id): name
                for name, series_id in series_map.items()
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result = future.result()

                    if result['success']:
                        results[name] = result['data']
                        logger.info(f"✓ {name}")
                    else:
                        logger.warning(f"✗ {name}: {result['error']}")

                except Exception as e:
                    logger.error(f"✗ {name}: {e}")

        logger.info(f"FRED collection: {len(results)}/{len(series_map)} successful")

        return results


# ============================================================================
# Benchmark Utility
# ============================================================================

def benchmark_collection(tickers: List[str], lookback_days: int = 365):
    """
    순차 vs 병렬 수집 성능 비교

    Args:
        tickers: 테스트할 티커 리스트
        lookback_days: 과거 데이터 기간
    """
    import time

    print("="*80)
    print("DATA COLLECTION BENCHMARK")
    print("="*80)
    print(f"Tickers: {len(tickers)}")
    print(f"Lookback: {lookback_days} days\n")

    # Sequential collection
    print("[1/2] Sequential Collection...")
    sequential_start = time.time()

    from lib.data_collector import MarketDataCollector
    seq_collector = MarketDataCollector(lookback_days=lookback_days)
    seq_results = seq_collector.collect_batch(tickers)

    sequential_time = time.time() - sequential_start
    print(f"  Time: {sequential_time:.1f}s")
    print(f"  Success: {len(seq_results)}/{len(tickers)}\n")

    # Parallel collection
    print("[2/2] Parallel Collection...")
    parallel_start = time.time()

    parallel_collector = ParallelMarketCollector(lookback_days=lookback_days, max_workers=10)
    parallel_results = parallel_collector.collect_all(tickers)

    parallel_time = time.time() - parallel_start
    print(f"  Time: {parallel_time:.1f}s")
    print(f"  Success: {len(parallel_results)}/{len(tickers)}\n")

    # Summary
    improvement = (sequential_time - parallel_time) / sequential_time * 100

    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Sequential:   {sequential_time:.1f}s")
    print(f"Parallel:     {parallel_time:.1f}s")
    print(f"Improvement:  {improvement:+.1f}%")
    print(f"Speedup:      {sequential_time/parallel_time:.2f}x\n")

    if improvement >= 50:
        print("✓ Target achieved: 50% improvement (TODO.md Priority 2)")
    else:
        print(f"✗ Target missed: {improvement:.1f}% < 50%")

    print("="*80)


if __name__ == "__main__":
    # Test with EIMAS standard ticker set
    test_tickers = [
        # US Indices
        'SPY', 'QQQ', 'DIA', 'IWM',
        # Sectors
        'XLF', 'XLE', 'XLV', 'XLK', 'XLI', 'XLB', 'XLP', 'XLY', 'XLU',
        # Bonds
        'TLT', 'IEF', 'SHY', 'LQD', 'HYG',
        # Commodities
        'GLD', 'SLV', 'USO', 'UNG',
        # International
        'EEM', 'EFA',
        # Volatility
        'UVXY'
    ]

    benchmark_collection(test_tickers, lookback_days=365)
