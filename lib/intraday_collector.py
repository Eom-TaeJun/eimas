#!/usr/bin/env python3
"""
Intraday Collector
==================
장중 데이터 수집 및 분석 모듈

기능:
1. yfinance에서 과거 1분봉 데이터 조회 (최대 7일)
2. 장중 집계 계산 (시가갭, 고저시간, VWAP, 거래량분포)
3. 이상 감지 (VIX 스파이크, 급락 등)
4. 안정/휘발성 저장소에 분리 저장

사용법:
    # 매일 아침 실행
    collector = IntradayCollector()
    collector.collect_and_save()  # 어제 데이터 수집 및 저장
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from data.stable_store import StableStore
from data.volatile_store import VolatileStore


class IntradayCollector:
    """장중 데이터 수집기"""

    # 기본 수집 대상
    DEFAULT_TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']
    VIX_TICKER = '^VIX'

    # 이상 감지 임계값
    THRESHOLDS = {
        'vix_spike_pct': 15.0,       # VIX 15% 이상 급등
        'price_crash_pct': -1.0,     # 1% 이상 급락 (5분 내)
        'price_surge_pct': 1.0,      # 1% 이상 급등 (5분 내)
        'volume_spike_ratio': 3.0,   # 평균 대비 3배 이상 거래량
        'gap_significant_pct': 0.5,  # 시가갭 0.5% 이상
    }

    def __init__(self):
        self.stable_store = StableStore()
        self.volatile_store = VolatileStore()
        print("[IntradayCollector] Initialized")

    # ========================================================================
    # 메인 수집 함수
    # ========================================================================

    def collect_and_save(
        self,
        target_date: date = None,
        tickers: List[str] = None
    ) -> Dict[str, Any]:
        """
        장중 데이터 수집 및 저장

        Args:
            target_date: 수집 대상 날짜 (기본: 어제)
            tickers: 수집 대상 티커 (기본: DEFAULT_TICKERS)

        Returns:
            수집 결과 요약
        """
        target_date = target_date or (date.today() - timedelta(days=1))
        tickers = tickers or self.DEFAULT_TICKERS

        print(f"\n{'='*60}")
        print(f"📊 장중 데이터 수집: {target_date}")
        print(f"{'='*60}")

        results = {
            'date': target_date.isoformat(),
            'tickers_processed': 0,
            'summaries_saved': 0,
            'alerts_detected': 0,
            'errors': []
        }

        # 1. 1분봉 데이터 조회
        intraday_data = self._fetch_intraday_data(tickers, target_date)

        # 2. VIX 데이터 조회
        vix_data = self._fetch_vix_data(target_date)

        # 3. 각 티커별 처리
        for ticker in tickers:
            if ticker not in intraday_data:
                results['errors'].append(f"{ticker}: 데이터 없음")
                continue

            try:
                df = intraday_data[ticker]

                # 장중 집계 계산
                summary = self._calculate_intraday_summary(ticker, df, vix_data)

                # 안정 저장소에 저장
                if self.stable_store.save_intraday_summary(ticker, summary):
                    results['summaries_saved'] += 1
                    print(f"  ✅ {ticker}: 집계 저장 완료")

                # 이상 감지 및 휘발성 저장소에 저장
                alerts = self._detect_intraday_anomalies(ticker, df, vix_data)
                for alert in alerts:
                    if self.volatile_store.save_intraday_alert(alert):
                        results['alerts_detected'] += 1

                results['tickers_processed'] += 1

            except Exception as e:
                results['errors'].append(f"{ticker}: {str(e)}")
                print(f"  ❌ {ticker}: {e}")

        # 4. 시장 스냅샷 저장 (종가 기준)
        self._save_daily_snapshot(intraday_data, vix_data, target_date)

        print(f"\n{'='*60}")
        print(f"✅ 수집 완료: {results['summaries_saved']}/{len(tickers)} 저장, {results['alerts_detected']} 알림")
        print(f"{'='*60}")

        return results

    # ========================================================================
    # 데이터 조회
    # ========================================================================

    def _fetch_intraday_data(
        self,
        tickers: List[str],
        target_date: date
    ) -> Dict[str, pd.DataFrame]:
        """1분봉 데이터 조회"""
        print(f"\n📥 1분봉 데이터 조회 중...")

        data = {}
        # 7일치 조회 후 해당 날짜 필터링
        end = datetime.now()
        start = end - timedelta(days=7)

        for ticker in tickers:
            try:
                tk = yf.Ticker(ticker)
                df = tk.history(start=start, end=end, interval="1m")

                if df.empty:
                    continue

                # MultiIndex 컬럼 처리
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 해당 날짜만 필터링
                df['trade_date'] = df.index.date
                day_df = df[df['trade_date'] == target_date].copy()

                if len(day_df) > 0:
                    data[ticker] = day_df
                    print(f"  {ticker}: {len(day_df)}개 1분봉")

            except Exception as e:
                print(f"  {ticker}: 조회 실패 - {e}")

        return data

    def _fetch_vix_data(self, target_date: date) -> Optional[pd.DataFrame]:
        """VIX 1분봉 데이터 조회"""
        try:
            end = datetime.now()
            start = end - timedelta(days=7)

            tk = yf.Ticker(self.VIX_TICKER)
            df = tk.history(start=start, end=end, interval="1m")

            if df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['trade_date'] = df.index.date
            day_df = df[df['trade_date'] == target_date].copy()

            return day_df if len(day_df) > 0 else None

        except Exception as e:
            print(f"  VIX 조회 실패: {e}")
            return None

    # ========================================================================
    # 집계 계산
    # ========================================================================

    def _calculate_intraday_summary(
        self,
        ticker: str,
        df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """장중 집계 계산"""

        summary = {
            'date': df['trade_date'].iloc[0].isoformat(),
        }

        # 시가/종가
        open_price = float(df['Open'].iloc[0])
        close_price = float(df['Close'].iloc[-1])
        high_price = float(df['High'].max())
        low_price = float(df['Low'].min())

        # 전일 종가 (간단히 시가로 대체, 실제로는 별도 조회 필요)
        # TODO: 전일 종가 정확히 조회
        prev_close = open_price  # 임시

        summary['prev_close'] = prev_close
        summary['open_price'] = open_price
        summary['opening_gap_pct'] = (open_price / prev_close - 1) * 100 if prev_close else 0

        # 첫 30분 레인지
        first_30 = df.head(30)
        if len(first_30) > 0:
            f30_high = float(first_30['High'].max())
            f30_low = float(first_30['Low'].min())
            summary['first_30min_high'] = f30_high
            summary['first_30min_low'] = f30_low
            summary['first_30min_range_pct'] = (f30_high / f30_low - 1) * 100 if f30_low else 0

        # 장중 고저점
        summary['intraday_high'] = high_price
        summary['intraday_high_time'] = df['High'].idxmax().strftime('%H:%M')
        summary['intraday_low'] = low_price
        summary['intraday_low_time'] = df['Low'].idxmin().strftime('%H:%M')
        summary['intraday_range_pct'] = (high_price / low_price - 1) * 100 if low_price else 0

        # VWAP 계산
        df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
        df['TPxVol'] = df['TP'] * df['Volume']
        total_vol = df['Volume'].sum()

        if total_vol > 0:
            vwap = df['TPxVol'].sum() / total_vol
            summary['vwap'] = float(vwap)
            summary['close_vs_vwap_pct'] = (close_price / vwap - 1) * 100
        else:
            summary['vwap'] = close_price
            summary['close_vs_vwap_pct'] = 0

        # 거래량 분포
        summary['volume_total'] = int(total_vol)

        morning = df[df.index.hour < 12]['Volume'].sum()
        afternoon = df[(df.index.hour >= 12) & (df.index.hour < 15)]['Volume'].sum()
        power_hour = df[df.index.hour >= 15]['Volume'].sum()

        if total_vol > 0:
            summary['volume_morning_pct'] = float(morning / total_vol * 100)
            summary['volume_afternoon_pct'] = float(afternoon / total_vol * 100)
            summary['volume_power_hour_pct'] = float(power_hour / total_vol * 100)
        else:
            summary['volume_morning_pct'] = 0
            summary['volume_afternoon_pct'] = 0
            summary['volume_power_hour_pct'] = 0

        # VIX 정보
        if vix_df is not None and len(vix_df) > 0:
            summary['vix_open'] = float(vix_df['Open'].iloc[0])
            summary['vix_high'] = float(vix_df['High'].max())
            summary['vix_low'] = float(vix_df['Low'].min())
            summary['vix_close'] = float(vix_df['Close'].iloc[-1])

        return summary

    # ========================================================================
    # 이상 감지
    # ========================================================================

    def _detect_intraday_anomalies(
        self,
        ticker: str,
        df: pd.DataFrame,
        vix_df: Optional[pd.DataFrame]
    ) -> List[Dict]:
        """장중 이상 감지"""
        alerts = []
        target_date = df['trade_date'].iloc[0].isoformat()

        # 1. 급격한 가격 변동 감지 (5분 윈도우)
        df['pct_change_5m'] = df['Close'].pct_change(5) * 100

        # 급락 감지
        crashes = df[df['pct_change_5m'] <= self.THRESHOLDS['price_crash_pct']]
        for idx, row in crashes.iterrows():
            alerts.append({
                'timestamp': idx.isoformat(),
                'ticker': ticker,
                'alert_type': 'price_crash',
                'value': float(row['pct_change_5m']),
                'threshold': self.THRESHOLDS['price_crash_pct'],
                'deviation': abs(row['pct_change_5m']),
                'price_at_alert': float(row['Close']),
                'description': f"{ticker} {row['pct_change_5m']:.2f}% 급락 (5분)"
            })

        # 급등 감지
        surges = df[df['pct_change_5m'] >= self.THRESHOLDS['price_surge_pct']]
        for idx, row in surges.iterrows():
            alerts.append({
                'timestamp': idx.isoformat(),
                'ticker': ticker,
                'alert_type': 'price_surge',
                'value': float(row['pct_change_5m']),
                'threshold': self.THRESHOLDS['price_surge_pct'],
                'deviation': row['pct_change_5m'],
                'price_at_alert': float(row['Close']),
                'description': f"{ticker} +{row['pct_change_5m']:.2f}% 급등 (5분)"
            })

        # 2. 거래량 급증 감지
        avg_volume = df['Volume'].mean()
        if avg_volume > 0:
            df['volume_ratio'] = df['Volume'] / avg_volume
            volume_spikes = df[df['volume_ratio'] >= self.THRESHOLDS['volume_spike_ratio']]

            for idx, row in volume_spikes.iterrows():
                alerts.append({
                    'timestamp': idx.isoformat(),
                    'ticker': ticker,
                    'alert_type': 'volume_spike',
                    'value': float(row['volume_ratio']),
                    'threshold': self.THRESHOLDS['volume_spike_ratio'],
                    'volume_ratio': float(row['volume_ratio']),
                    'price_at_alert': float(row['Close']),
                    'description': f"{ticker} 거래량 {row['volume_ratio']:.1f}배 급증"
                })

        # 3. VIX 스파이크 감지
        if vix_df is not None and len(vix_df) > 0:
            vix_df['pct_change_5m'] = vix_df['Close'].pct_change(5) * 100

            vix_spikes = vix_df[vix_df['pct_change_5m'] >= self.THRESHOLDS['vix_spike_pct']]
            for idx, row in vix_spikes.iterrows():
                # SPY와 연결해서 저장
                spy_price = df.loc[df.index <= idx, 'Close'].iloc[-1] if len(df.loc[df.index <= idx]) > 0 else None

                alerts.append({
                    'timestamp': idx.isoformat(),
                    'ticker': 'VIX',
                    'alert_type': 'vix_spike',
                    'value': float(row['pct_change_5m']),
                    'threshold': self.THRESHOLDS['vix_spike_pct'],
                    'vix_at_alert': float(row['Close']),
                    'price_at_alert': float(spy_price) if spy_price else None,
                    'description': f"VIX +{row['pct_change_5m']:.1f}% 급등 (5분)"
                })

        if alerts:
            print(f"    ⚠️ {ticker}: {len(alerts)}개 이상 감지")

        return alerts

    # ========================================================================
    # 스냅샷 저장
    # ========================================================================

    def _save_daily_snapshot(
        self,
        intraday_data: Dict[str, pd.DataFrame],
        vix_df: Optional[pd.DataFrame],
        target_date: date
    ):
        """일별 종가 스냅샷 저장"""

        snapshot = {
            'timestamp': datetime.combine(target_date, datetime.max.time()).isoformat(),
            'collection_type': 'daily_close'
        }

        # SPY
        if 'SPY' in intraday_data:
            spy = intraday_data['SPY']
            snapshot['spy_price'] = float(spy['Close'].iloc[-1])
            snapshot['spy_change_pct'] = float((spy['Close'].iloc[-1] / spy['Open'].iloc[0] - 1) * 100)

        # QQQ
        if 'QQQ' in intraday_data:
            snapshot['qqq_price'] = float(intraday_data['QQQ']['Close'].iloc[-1])

        # IWM
        if 'IWM' in intraday_data:
            snapshot['iwm_price'] = float(intraday_data['IWM']['Close'].iloc[-1])

        # TLT
        if 'TLT' in intraday_data:
            snapshot['tlt_price'] = float(intraday_data['TLT']['Close'].iloc[-1])

        # GLD
        if 'GLD' in intraday_data:
            snapshot['gld_price'] = float(intraday_data['GLD']['Close'].iloc[-1])

        # VIX
        if vix_df is not None and len(vix_df) > 0:
            snapshot['vix_level'] = float(vix_df['Close'].iloc[-1])
            snapshot['vix_change_pct'] = float((vix_df['Close'].iloc[-1] / vix_df['Open'].iloc[0] - 1) * 100)

        self.volatile_store.save_market_snapshot(snapshot)

    # ========================================================================
    # 유틸리티
    # ========================================================================

    def get_available_dates(self, ticker: str = 'SPY') -> List[date]:
        """조회 가능한 날짜 목록"""
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(period="7d", interval="1m")

            if df.empty:
                return []

            df['trade_date'] = df.index.date
            return sorted(df['trade_date'].unique().tolist())

        except Exception as e:
            print(f"Error: {e}")
            return []

    def collect_missing_days(self, days_back: int = 5) -> Dict[str, Any]:
        """누락된 일자 수집"""
        available = self.get_available_dates()
        results = {'collected': [], 'skipped': [], 'errors': []}

        for d in available:
            # 이미 저장되어 있는지 확인
            existing = self.stable_store.get_intraday_summary('SPY', start_date=d.isoformat())
            if any(s['date'] == d.isoformat() for s in existing):
                results['skipped'].append(d.isoformat())
                continue

            try:
                self.collect_and_save(target_date=d)
                results['collected'].append(d.isoformat())
            except Exception as e:
                results['errors'].append(f"{d}: {e}")

        return results


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Intraday Data Collector')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--backfill', action='store_true', help='Backfill missing days')
    parser.add_argument('--tickers', type=str, help='Comma-separated tickers')

    args = parser.parse_args()

    collector = IntradayCollector()

    if args.backfill:
        print("\n누락된 일자 수집 중...")
        results = collector.collect_missing_days()
        print(f"\n수집: {results['collected']}")
        print(f"스킵: {results['skipped']}")
        if results['errors']:
            print(f"에러: {results['errors']}")

    else:
        target_date = None
        if args.date:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()

        tickers = None
        if args.tickers:
            tickers = [t.strip() for t in args.tickers.split(',')]

        collector.collect_and_save(target_date=target_date, tickers=tickers)
