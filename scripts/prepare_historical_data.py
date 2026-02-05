#!/usr/bin/env python3
"""
Historical Data Collection for Backtesting
===========================================
2025-02-04 ~ 2026-02-04 (12개월) 데이터 수집 및 저장

Purpose:
- Collect historical data for backtesting EIMAS strategies
- Save daily snapshots to enable walk-forward analysis
- Target: 12 months of FRED, market, crypto data

Economic Foundation:
- Prado (2018): Chapter 7 - Cross-Validation for Financial Data
- Bailey et al. (2014): Backtest overfitting prevention

Output:
- data/backtest_historical.parquet (or CSV)
- Daily snapshots: FRED + 24 market tickers + crypto
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import argparse
import logging

from lib.fred_collector import FREDCollector
from lib.data_collector import DataManager
from lib.data_loader import collect_crypto_rwa_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """12개월 과거 데이터 수집기"""

    def __init__(self, start_date: str, end_date: str):
        """
        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self.fred_collector = FREDCollector()
        self.data_manager = DataManager()

        # Market tickers (EIMAS standard set)
        self.market_tickers = [
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

        # Crypto & RWA
        self.crypto_tickers = [
            'BTC-USD', 'ETH-USD',
            'ONDO-USD', 'PAXG-USD', 'COIN'
        ]

    def collect_all(self) -> pd.DataFrame:
        """
        전체 데이터 수집

        Returns:
            DataFrame with columns:
            - date (index)
            - RRP, TGA, WRESBAL, EFFR (FRED)
            - SPY, QQQ, ... (market prices)
            - BTC-USD, ETH-USD, ... (crypto prices)
        """
        logger.info(f"Collecting data from {self.start_date.date()} to {self.end_date.date()}")

        # 1. FRED Data
        logger.info("[1/3] Collecting FRED data...")
        fred_data = self._collect_fred_data()

        # 2. Market Data
        logger.info("[2/3] Collecting market data (24 tickers)...")
        market_data = self._collect_market_data()

        # 3. Crypto & RWA
        logger.info("[3/3] Collecting crypto & RWA data...")
        crypto_data = self._collect_crypto_data()

        # Merge all data
        logger.info("Merging datasets...")
        combined = fred_data.join(market_data, how='outer')
        combined = combined.join(crypto_data, how='outer')

        # Filter date range
        combined = combined.loc[self.start_date:self.end_date]

        # Forward fill missing values (weekends/holidays)
        combined = combined.ffill()

        # Drop rows with all NaN
        combined = combined.dropna(how='all')

        logger.info(f"✓ Collection complete: {len(combined)} days, {len(combined.columns)} columns")

        return combined

    def _collect_fred_data(self) -> pd.DataFrame:
        """FRED 데이터 수집"""
        fred_series = {
            'RRP': 'RRPONTSYD',
            'TGA': 'WTREGEN',
            'WRESBAL': 'WALCL',
            'EFFR': 'EFFR'
        }

        data = {}

        for name, series_id in fred_series.items():
            try:
                df = self.fred_collector.fetch_series(
                    series_id,
                    start_date=self.start_date.strftime('%Y-%m-%d'),
                    end_date=self.end_date.strftime('%Y-%m-%d')
                )
                if df is not None and len(df) > 0:
                    data[name] = df['value']
                    logger.info(f"  ✓ {name}: {len(df)} points")
                else:
                    logger.warning(f"  ✗ {name}: No data")
            except Exception as e:
                logger.error(f"  ✗ {name}: {e}")

        if not data:
            logger.warning("No FRED data collected, returning empty DataFrame")
            return pd.DataFrame()

        fred_df = pd.DataFrame(data)
        fred_df.index = pd.to_datetime(fred_df.index)

        return fred_df

    def _collect_market_data(self) -> pd.DataFrame:
        """시장 데이터 수집 (yfinance)"""
        import yfinance as yf

        try:
            data = yf.download(
                self.market_tickers,
                start=self.start_date.strftime('%Y-%m-%d'),
                end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )['Close']

            if isinstance(data, pd.Series):
                # Single ticker case
                data = data.to_frame(name=self.market_tickers[0])

            logger.info(f"  ✓ Market: {len(data)} days, {len(data.columns)} tickers")

            return data

        except Exception as e:
            logger.error(f"  ✗ Market data error: {e}")
            return pd.DataFrame()

    def _collect_crypto_data(self) -> pd.DataFrame:
        """크립토 & RWA 데이터 수집"""
        import yfinance as yf

        try:
            data = yf.download(
                self.crypto_tickers,
                start=self.start_date.strftime('%Y-%m-%d'),
                end=(self.end_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )['Close']

            if isinstance(data, pd.Series):
                data = data.to_frame(name=self.crypto_tickers[0])

            logger.info(f"  ✓ Crypto: {len(data)} days, {len(data.columns)} assets")

            return data

        except Exception as e:
            logger.error(f"  ✗ Crypto data error: {e}")
            return pd.DataFrame()

    def save(self, data: pd.DataFrame, output_path: str):
        """데이터 저장"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save as Parquet (efficient)
        if output_path.endswith('.parquet'):
            data.to_parquet(output_path)
            logger.info(f"✓ Saved to {output_path} (Parquet format)")

        # Save as CSV (human-readable)
        elif output_path.endswith('.csv'):
            data.to_csv(output_path)
            logger.info(f"✓ Saved to {output_path} (CSV format)")

        else:
            raise ValueError("Output path must end with .parquet or .csv")

        # Print summary
        logger.info(f"\nData Summary:")
        logger.info(f"  Date range: {data.index.min().date()} to {data.index.max().date()}")
        logger.info(f"  Total days: {len(data)}")
        logger.info(f"  Columns: {len(data.columns)}")
        logger.info(f"  Missing %: {data.isna().sum().sum() / (len(data) * len(data.columns)) * 100:.2f}%")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Collect historical data for backtesting')
    parser.add_argument(
        '--start',
        type=str,
        default='2025-02-04',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2026-02-04',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/backtest_historical.parquet',
        help='Output file path (.parquet or .csv)'
    )

    args = parser.parse_args()

    # Collect data
    collector = HistoricalDataCollector(args.start, args.end)
    data = collector.collect_all()

    # Save
    if len(data) > 0:
        collector.save(data, args.output)
        print(f"\n{'='*80}")
        print(f"✓ Historical data collection complete!")
        print(f"{'='*80}")
        print(f"File: {args.output}")
        print(f"Dates: {data.index.min().date()} to {data.index.max().date()}")
        print(f"Days: {len(data)}")
        print(f"Features: {len(data.columns)}")
        print(f"{'='*80}\n")
    else:
        logger.error("No data collected. Check API keys and internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()
