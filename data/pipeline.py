#!/usr/bin/env python3
"""
EIMAS Data Pipeline
====================
ETL 파이프라인 및 데이터 품질 관리

주요 기능:
1. Extract: 다중 소스 데이터 추출
2. Transform: 데이터 변환 및 정규화
3. Load: 데이터 저장 및 캐싱
4. Validate: 데이터 품질 검증

Usage:
    from data.pipeline import DataPipeline

    pipeline = DataPipeline()
    data = pipeline.run()
"""

import numpy as np
import pandas as pd
import yfinance as yf
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / 'cache'
DB_PATH = DATA_DIR / 'eimas_data.db'

# 기본 티커 유니버스
DEFAULT_TICKERS = {
    'equity': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI'],
    'sector': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB', 'XLRE', 'XLC'],
    'bond': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP'],
    'commodity': ['GLD', 'SLV', 'USO', 'UNG', 'DBA'],
    'international': ['EFA', 'EEM', 'VEA', 'VWO'],
    'volatility': ['VXX', 'UVXY'],
}

# 데이터 품질 임계값
QUALITY_THRESHOLDS = {
    'max_missing_pct': 0.05,      # 최대 결측치 비율
    'max_zero_pct': 0.10,         # 최대 0값 비율
    'max_outlier_std': 5.0,       # 이상치 표준편차
    'min_data_points': 20,        # 최소 데이터 포인트
}


# ============================================================================
# Data Classes
# ============================================================================

class DataSource(str, Enum):
    """데이터 소스"""
    YFINANCE = 'yfinance'
    FRED = 'fred'
    CACHE = 'cache'
    DATABASE = 'database'


class DataQuality(str, Enum):
    """데이터 품질 등급"""
    EXCELLENT = 'excellent'    # 결측치 0%, 이상치 없음
    GOOD = 'good'              # 결측치 < 1%, 이상치 < 1%
    ACCEPTABLE = 'acceptable'  # 결측치 < 5%, 이상치 < 5%
    POOR = 'poor'              # 결측치 >= 5% 또는 이상치 >= 5%
    INVALID = 'invalid'        # 사용 불가


@dataclass
class DataValidation:
    """데이터 검증 결과"""
    ticker: str
    total_rows: int
    missing_count: int
    missing_pct: float
    zero_count: int
    zero_pct: float
    outlier_count: int
    outlier_pct: float
    quality: DataQuality
    issues: List[str]


@dataclass
class PipelineResult:
    """파이프라인 결과"""
    timestamp: datetime
    tickers_requested: int
    tickers_loaded: int
    tickers_failed: List[str]
    data: pd.DataFrame
    validations: List[DataValidation]
    cache_hits: int
    cache_misses: int
    execution_time: float
    summary: str


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    key: str
    data: Any
    created_at: datetime
    expires_at: datetime
    source: DataSource
    checksum: str


# ============================================================================
# Data Pipeline
# ============================================================================

class DataPipeline:
    """데이터 파이프라인"""

    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        period: str = '1y',
        cache_ttl: int = 3600,  # 캐시 TTL (초)
        use_cache: bool = True,
        parallel: bool = True,
        max_workers: int = 5,
    ):
        self.tickers = tickers or self._get_default_tickers()
        self.period = period
        self.cache_ttl = cache_ttl
        self.use_cache = use_cache
        self.parallel = parallel
        self.max_workers = max_workers

        # 캐시 디렉토리 생성
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 통계
        self.cache_hits = 0
        self.cache_misses = 0

    def _get_default_tickers(self) -> List[str]:
        """기본 티커 목록"""
        tickers = []
        for category in DEFAULT_TICKERS.values():
            tickers.extend(category)
        return list(set(tickers))

    # -------------------------------------------------------------------------
    # Extract
    # -------------------------------------------------------------------------

    def extract(self) -> pd.DataFrame:
        """데이터 추출"""
        logger.info(f"Extracting data for {len(self.tickers)} tickers...")

        if self.parallel:
            return self._extract_parallel()
        else:
            return self._extract_sequential()

    def _extract_parallel(self) -> pd.DataFrame:
        """병렬 데이터 추출"""
        results = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_ticker, ticker): ticker
                for ticker in self.tickers
            }

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        results[ticker] = data
                except Exception as e:
                    logger.warning(f"Failed to fetch {ticker}: {e}")

        if not results:
            return pd.DataFrame()

        # 데이터 병합
        df = pd.DataFrame(results)
        return df

    def _extract_sequential(self) -> pd.DataFrame:
        """순차 데이터 추출"""
        results = {}

        for ticker in self.tickers:
            try:
                data = self._fetch_ticker(ticker)
                if data is not None and not data.empty:
                    results[ticker] = data
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        return df

    def _fetch_ticker(self, ticker: str) -> Optional[pd.Series]:
        """개별 티커 데이터 추출"""
        # 캐시 확인
        if self.use_cache:
            cached = self._get_from_cache(ticker)
            if cached is not None:
                self.cache_hits += 1
                return cached

        self.cache_misses += 1

        # yfinance에서 추출
        try:
            df = yf.download(ticker, period=self.period, progress=False)
            if df.empty:
                return None

            # Close 컬럼 추출 (MultiIndex 처리)
            if isinstance(df.columns, pd.MultiIndex):
                # ('Close', 'SPY') 형태의 MultiIndex
                close_col = ('Close', ticker)
                if close_col in df.columns:
                    data = df[close_col].copy()
                else:
                    # Close 레벨에서 첫 번째 컬럼
                    close_df = df.xs('Close', axis=1, level=0)
                    data = close_df.iloc[:, 0].copy() if not close_df.empty else pd.Series()
            elif 'Close' in df.columns:
                data = df['Close'].copy()
            else:
                data = df.iloc[:, 0].copy()

            # Series인지 확인하고 이름 설정 (tuple 이름을 문자열로 변환)
            if isinstance(data, pd.Series) and not data.empty and len(data) > 0:
                data = data.rename(ticker)
                # 캐시 저장
                if self.use_cache:
                    self._save_to_cache(ticker, data)
                return data
        except Exception as e:
            logger.warning(f"yfinance error for {ticker}: {e}")

        return None

    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 변환"""
        if df.empty:
            return df

        logger.info("Transforming data...")

        # 1. 결측치 처리
        df = self._handle_missing(df)

        # 2. 이상치 처리
        df = self._handle_outliers(df)

        # 3. 수익률 계산
        returns = df.pct_change().dropna()

        return returns

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측치 처리"""
        # Forward fill -> Backward fill -> Drop remaining
        df = df.ffill().bfill()

        # 여전히 NaN이 많은 컬럼 제거
        missing_pct = df.isnull().sum() / len(df)
        valid_cols = missing_pct[missing_pct < QUALITY_THRESHOLDS['max_missing_pct']].index
        df = df[valid_cols]

        return df

    def _handle_outliers(self, df: pd.DataFrame, std_threshold: float = 5.0) -> pd.DataFrame:
        """이상치 처리 (Winsorization)"""
        for col in df.columns:
            mean = df[col].mean()
            std = df[col].std()

            lower = mean - std_threshold * std
            upper = mean + std_threshold * std

            df[col] = df[col].clip(lower, upper)

        return df

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    def load(self, df: pd.DataFrame, table_name: str = 'price_data') -> bool:
        """데이터 저장"""
        if df.empty:
            return False

        logger.info(f"Loading data to {table_name}...")

        try:
            conn = sqlite3.connect(DB_PATH)

            # 메타데이터 저장
            meta = {
                'tickers': list(df.columns),
                'start_date': str(df.index.min()),
                'end_date': str(df.index.max()),
                'rows': len(df),
                'updated_at': datetime.now().isoformat(),
            }

            df.to_sql(table_name, conn, if_exists='replace', index=True)

            # 메타데이터 테이블
            meta_df = pd.DataFrame([meta])
            meta_df.to_sql(f'{table_name}_meta', conn, if_exists='replace', index=False)

            conn.close()
            logger.info(f"Saved {len(df)} rows to {table_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save data: {e}")
            return False

    def load_from_db(self, table_name: str = 'price_data') -> pd.DataFrame:
        """DB에서 데이터 로드"""
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql(f'SELECT * FROM {table_name}', conn, index_col='Date', parse_dates=['Date'])
            conn.close()
            return df
        except Exception as e:
            logger.warning(f"Failed to load from DB: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> List[DataValidation]:
        """데이터 검증"""
        validations = []

        for col in df.columns:
            series = df[col]
            total = len(series)

            # 결측치
            missing = series.isnull().sum()
            missing_pct = missing / total if total > 0 else 0

            # 0값
            zeros = (series == 0).sum()
            zero_pct = zeros / total if total > 0 else 0

            # 이상치 (3 std 이상)
            if series.std() > 0:
                z_scores = np.abs((series - series.mean()) / series.std())
                outliers = (z_scores > 3).sum()
                outlier_pct = outliers / total if total > 0 else 0
            else:
                outliers = 0
                outlier_pct = 0

            # 품질 등급 결정
            issues = []
            if missing_pct >= QUALITY_THRESHOLDS['max_missing_pct']:
                issues.append(f"High missing: {missing_pct:.1%}")
            if zero_pct >= QUALITY_THRESHOLDS['max_zero_pct']:
                issues.append(f"High zeros: {zero_pct:.1%}")
            if outlier_pct >= 0.05:
                issues.append(f"High outliers: {outlier_pct:.1%}")
            if total < QUALITY_THRESHOLDS['min_data_points']:
                issues.append(f"Low data points: {total}")

            if not issues:
                if missing_pct == 0 and outlier_pct == 0:
                    quality = DataQuality.EXCELLENT
                elif missing_pct < 0.01 and outlier_pct < 0.01:
                    quality = DataQuality.GOOD
                else:
                    quality = DataQuality.ACCEPTABLE
            elif len(issues) <= 1:
                quality = DataQuality.POOR
            else:
                quality = DataQuality.INVALID

            validations.append(DataValidation(
                ticker=col,
                total_rows=total,
                missing_count=int(missing),
                missing_pct=float(missing_pct),
                zero_count=int(zeros),
                zero_pct=float(zero_pct),
                outlier_count=int(outliers),
                outlier_pct=float(outlier_pct),
                quality=quality,
                issues=issues,
            ))

        return validations

    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------

    def _get_cache_key(self, ticker: str) -> str:
        """캐시 키 생성"""
        key = f"{ticker}_{self.period}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_from_cache(self, ticker: str) -> Optional[pd.Series]:
        """캐시에서 데이터 조회"""
        cache_key = self._get_cache_key(ticker)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"

        if not cache_file.exists():
            return None

        try:
            # 캐시 만료 확인
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime > timedelta(seconds=self.cache_ttl):
                cache_file.unlink()  # 만료된 캐시 삭제
                return None

            data = pd.read_pickle(cache_file)
            return data

        except Exception as e:
            logger.warning(f"Cache read error for {ticker}: {e}")
            return None

    def _save_to_cache(self, ticker: str, data: pd.Series) -> bool:
        """캐시에 데이터 저장"""
        cache_key = self._get_cache_key(ticker)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"

        try:
            data.to_pickle(cache_file)
            return True
        except Exception as e:
            logger.warning(f"Cache write error for {ticker}: {e}")
            return False

    def clear_cache(self):
        """캐시 초기화"""
        for f in CACHE_DIR.glob('*.pkl'):
            f.unlink()
        logger.info("Cache cleared")

    # -------------------------------------------------------------------------
    # Run Pipeline
    # -------------------------------------------------------------------------

    def run(self, save_to_db: bool = False) -> PipelineResult:
        """전체 파이프라인 실행"""
        start_time = datetime.now()

        print(f"\n{'='*50}")
        print("EIMAS Data Pipeline")
        print('='*50)

        # 1. Extract
        print("\n[1/4] Extracting data...")
        raw_data = self.extract()
        print(f"  Loaded {len(raw_data.columns)} tickers")

        # 2. Transform
        print("\n[2/4] Transforming data...")
        transformed = self.transform(raw_data)
        print(f"  {len(transformed)} rows after transform")

        # 3. Validate
        print("\n[3/4] Validating data...")
        validations = self.validate(transformed)
        quality_counts = {}
        for v in validations:
            quality_counts[v.quality.value] = quality_counts.get(v.quality.value, 0) + 1
        print(f"  Quality: {quality_counts}")

        # 4. Load (optional)
        if save_to_db:
            print("\n[4/4] Loading to database...")
            self.load(transformed)
        else:
            print("\n[4/4] Skipping database load")

        # 결과 집계
        execution_time = (datetime.now() - start_time).total_seconds()
        failed = [t for t in self.tickers if t not in transformed.columns]

        summary = self._generate_summary(transformed, validations, execution_time)

        result = PipelineResult(
            timestamp=datetime.now(),
            tickers_requested=len(self.tickers),
            tickers_loaded=len(transformed.columns),
            tickers_failed=failed,
            data=transformed,
            validations=validations,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            execution_time=execution_time,
            summary=summary,
        )

        print("\n" + summary)
        print('='*50)

        return result

    def _generate_summary(
        self,
        df: pd.DataFrame,
        validations: List[DataValidation],
        execution_time: float,
    ) -> str:
        """요약 생성"""
        quality_counts = {}
        for v in validations:
            quality_counts[v.quality.value] = quality_counts.get(v.quality.value, 0) + 1

        lines = [
            "=== Pipeline Summary ===",
            f"Tickers: {len(df.columns)} loaded",
            f"Date Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}",
            f"Data Points: {len(df)} rows",
            "",
            "=== Data Quality ===",
        ]

        for quality, count in sorted(quality_counts.items()):
            lines.append(f"  {quality}: {count}")

        lines.extend([
            "",
            "=== Cache Stats ===",
            f"  Hits: {self.cache_hits}",
            f"  Misses: {self.cache_misses}",
            f"  Hit Rate: {self.cache_hits / (self.cache_hits + self.cache_misses) * 100:.1f}%" if (self.cache_hits + self.cache_misses) > 0 else "  Hit Rate: N/A",
            "",
            f"Execution Time: {execution_time:.2f}s",
        ])

        return "\n".join(lines)


# ============================================================================
# Incremental Update
# ============================================================================

class IncrementalUpdater:
    """증분 업데이트"""

    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

    def update_since_last(self) -> pd.DataFrame:
        """마지막 업데이트 이후 데이터만 추가"""
        # 기존 데이터 로드
        existing = self.pipeline.load_from_db()

        if existing.empty:
            # 전체 로드
            return self.pipeline.run(save_to_db=True).data

        last_date = existing.index.max()
        days_since = (datetime.now() - last_date).days

        if days_since <= 0:
            logger.info("Data is up to date")
            return existing

        # 새 데이터만 추출
        self.pipeline.period = f'{days_since + 5}d'  # 여유분 추가
        new_data = self.pipeline.extract()

        if new_data.empty:
            return existing

        # 변환
        new_transformed = self.pipeline.transform(new_data)

        # 병합
        combined = pd.concat([existing, new_transformed])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()

        # 저장
        self.pipeline.load(combined)

        return combined


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_pipeline(
    tickers: Optional[List[str]] = None,
    period: str = '6mo',
) -> pd.DataFrame:
    """빠른 파이프라인 실행"""
    pipeline = DataPipeline(tickers=tickers, period=period)
    result = pipeline.run()
    return result.data


def get_quality_report(
    tickers: Optional[List[str]] = None,
) -> List[DataValidation]:
    """데이터 품질 리포트"""
    pipeline = DataPipeline(tickers=tickers, period='1mo')
    raw = pipeline.extract()
    transformed = pipeline.transform(raw)
    return pipeline.validate(transformed)


def refresh_cache(tickers: Optional[List[str]] = None):
    """캐시 갱신"""
    pipeline = DataPipeline(tickers=tickers, use_cache=False)
    pipeline.clear_cache()
    pipeline.run()
    print("Cache refreshed")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    # 기본 테스트
    tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']

    pipeline = DataPipeline(tickers=tickers, period='3mo')
    result = pipeline.run(save_to_db=True)

    print("\n\nValidation Details:")
    for v in result.validations:
        print(f"  {v.ticker}: {v.quality.value} - {v.issues if v.issues else 'OK'}")

    # 증분 업데이트 테스트
    print("\n\nIncremental Update Test:")
    updater = IncrementalUpdater(pipeline)
    updated = updater.update_since_last()
    print(f"Updated data shape: {updated.shape}")
