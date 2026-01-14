#!/usr/bin/env python3
"""
EIMAS Session Analyzer
======================
시장 세션별 가격 데이터 수집 및 분석

세션 구분 (ET):
- Pre-Market: 04:00 - 09:30
- Opening Hour: 09:30 - 10:30
- Mid-Day: 10:30 - 14:00
- Power Hour: 15:00 - 16:00
- After-Hours: 16:00 - 20:00

Usage:
    from lib.session_analyzer import SessionAnalyzer

    analyzer = SessionAnalyzer()
    analysis = analyzer.analyze_today("SPY")
    analyzer.save_analysis(analysis)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import pytz

from lib.trading_db import TradingDB, SessionAnalysis, SessionType


# ============================================================================
# Constants
# ============================================================================

ET = pytz.timezone('US/Eastern')
KST = pytz.timezone('Asia/Seoul')

# 세션 시간 정의 (ET 기준)
SESSION_TIMES = {
    SessionType.PRE_MARKET: (time(4, 0), time(9, 30)),
    SessionType.OPENING: (time(9, 30), time(10, 30)),
    SessionType.MID_DAY: (time(10, 30), time(15, 0)),
    SessionType.POWER_HOUR: (time(15, 0), time(16, 0)),
    SessionType.AFTER_HOURS: (time(16, 0), time(20, 0)),
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SessionReturn:
    """세션별 수익률"""
    session: SessionType
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    return_pct: float
    volume: int
    start_time: datetime
    end_time: datetime


@dataclass
class DailySessionAnalysis:
    """일일 세션 분석"""
    date: date
    ticker: str
    sessions: Dict[SessionType, SessionReturn]
    overnight_return: float = 0.0  # 전일 종가 → 금일 시가
    total_return: float = 0.0
    best_buy_session: Optional[SessionType] = None
    best_sell_session: Optional[SessionType] = None
    volume_distribution: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Session Analyzer
# ============================================================================

class SessionAnalyzer:
    """세션별 분석기"""

    def __init__(self, db: TradingDB = None):
        self.db = db or TradingDB()

    def fetch_intraday_data(
        self,
        ticker: str,
        target_date: date = None,
        interval: str = "5m"
    ) -> pd.DataFrame:
        """
        장중 데이터 수집

        Note: yfinance 무료 버전은 최근 60일 분봉만 제공
        """
        if target_date is None:
            target_date = date.today()

        # yfinance는 start/end를 사용
        start = datetime.combine(target_date, time(0, 0))
        end = datetime.combine(target_date + timedelta(days=1), time(0, 0))

        print(f"Fetching {interval} data for {ticker} on {target_date}...")

        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                prepost=True  # Pre/After market 포함
            )

            if df.empty:
                print(f"  No data available for {target_date}")
                return pd.DataFrame()

            # MultiIndex 처리
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 시간대 변환 (UTC → ET)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(ET)

            print(f"  Loaded {len(df)} bars")
            return df

        except Exception as e:
            print(f"  Error: {e}")
            return pd.DataFrame()

    def _get_session_data(
        self,
        df: pd.DataFrame,
        session: SessionType,
        target_date: date
    ) -> Optional[SessionReturn]:
        """특정 세션의 데이터 추출"""
        start_time, end_time = SESSION_TIMES[session]

        # 시간 필터링
        session_start = ET.localize(datetime.combine(target_date, start_time))
        session_end = ET.localize(datetime.combine(target_date, end_time))

        mask = (df.index >= session_start) & (df.index < session_end)
        session_df = df[mask]

        if session_df.empty:
            return None

        open_price = float(session_df['Open'].iloc[0])
        close_price = float(session_df['Close'].iloc[-1])
        high_price = float(session_df['High'].max())
        low_price = float(session_df['Low'].min())
        volume = int(session_df['Volume'].sum())
        return_pct = (close_price / open_price - 1) * 100 if open_price > 0 else 0

        return SessionReturn(
            session=session,
            open_price=open_price,
            close_price=close_price,
            high_price=high_price,
            low_price=low_price,
            return_pct=return_pct,
            volume=volume,
            start_time=session_df.index[0].to_pydatetime(),
            end_time=session_df.index[-1].to_pydatetime(),
        )

    def analyze_day(
        self,
        ticker: str,
        target_date: date = None,
        prev_close: float = None
    ) -> Optional[DailySessionAnalysis]:
        """일일 세션 분석"""
        if target_date is None:
            target_date = date.today()

        df = self.fetch_intraday_data(ticker, target_date)
        if df.empty:
            return None

        sessions = {}
        total_volume = 0

        # 각 세션 분석
        for session_type in SessionType:
            session_data = self._get_session_data(df, session_type, target_date)
            if session_data:
                sessions[session_type] = session_data
                total_volume += session_data.volume

        if not sessions:
            print(f"  No session data for {target_date}")
            return None

        # 거래량 분포
        volume_dist = {}
        for session_type, data in sessions.items():
            volume_dist[session_type.value] = round(data.volume / total_volume, 3) if total_volume > 0 else 0

        # Overnight 수익률 (전일 종가 → 금일 Pre-market 또는 시가)
        overnight_return = 0.0
        if prev_close and SessionType.PRE_MARKET in sessions:
            overnight_return = (sessions[SessionType.PRE_MARKET].open_price / prev_close - 1) * 100
        elif prev_close and SessionType.OPENING in sessions:
            overnight_return = (sessions[SessionType.OPENING].open_price / prev_close - 1) * 100

        # 총 수익률
        first_session = min(sessions.keys(), key=lambda x: SESSION_TIMES[x][0])
        last_session = max(sessions.keys(), key=lambda x: SESSION_TIMES[x][1])
        total_return = 0.0
        if first_session in sessions and last_session in sessions:
            first_open = sessions[first_session].open_price
            last_close = sessions[last_session].close_price
            total_return = (last_close / first_open - 1) * 100 if first_open > 0 else 0

        # 최적 매수/매도 세션 (당일 기준)
        # 최저가 세션 = 매수 최적, 최고가 세션 = 매도 최적
        best_buy = min(sessions.keys(), key=lambda x: sessions[x].low_price)
        best_sell = max(sessions.keys(), key=lambda x: sessions[x].high_price)

        return DailySessionAnalysis(
            date=target_date,
            ticker=ticker,
            sessions=sessions,
            overnight_return=overnight_return,
            total_return=total_return,
            best_buy_session=best_buy,
            best_sell_session=best_sell,
            volume_distribution=volume_dist,
        )

    def analyze_period(
        self,
        ticker: str,
        days: int = 30
    ) -> List[DailySessionAnalysis]:
        """기간 분석"""
        results = []
        prev_close = None

        # 최근 60일까지만 (yfinance 제한)
        days = min(days, 60)

        for i in range(days):
            target_date = date.today() - timedelta(days=i)

            # 주말 스킵
            if target_date.weekday() >= 5:
                continue

            analysis = self.analyze_day(ticker, target_date, prev_close)
            if analysis:
                results.append(analysis)

                # 다음 날을 위해 종가 저장
                if SessionType.POWER_HOUR in analysis.sessions:
                    prev_close = analysis.sessions[SessionType.POWER_HOUR].close_price
                elif SessionType.MID_DAY in analysis.sessions:
                    prev_close = analysis.sessions[SessionType.MID_DAY].close_price

        return results

    def get_session_statistics(
        self,
        analyses: List[DailySessionAnalysis]
    ) -> Dict[str, Any]:
        """세션별 통계"""
        if not analyses:
            return {}

        stats = {session.value: {
            'avg_return': [],
            'win_rate': 0,
            'avg_volume_pct': [],
            'best_buy_count': 0,
            'best_sell_count': 0,
        } for session in SessionType}

        for analysis in analyses:
            for session_type, data in analysis.sessions.items():
                key = session_type.value
                stats[key]['avg_return'].append(data.return_pct)
                if session_type == analysis.best_buy_session:
                    stats[key]['best_buy_count'] += 1
                if session_type == analysis.best_sell_session:
                    stats[key]['best_sell_count'] += 1

            for session_name, vol_pct in analysis.volume_distribution.items():
                if session_name in stats:
                    stats[session_name]['avg_volume_pct'].append(vol_pct)

        # 집계
        result = {}
        total_days = len(analyses)

        for session_name, data in stats.items():
            if data['avg_return']:
                returns = data['avg_return']
                result[session_name] = {
                    'avg_return': round(np.mean(returns), 3),
                    'std_return': round(np.std(returns), 3),
                    'win_rate': round(sum(1 for r in returns if r > 0) / len(returns) * 100, 1),
                    'avg_volume_pct': round(np.mean(data['avg_volume_pct']) * 100, 1) if data['avg_volume_pct'] else 0,
                    'best_buy_pct': round(data['best_buy_count'] / total_days * 100, 1),
                    'best_sell_pct': round(data['best_sell_count'] / total_days * 100, 1),
                }

        return result

    def save_analysis(self, analysis: DailySessionAnalysis) -> int:
        """분석 결과 DB 저장"""
        session_returns = {}
        for session_type, data in analysis.sessions.items():
            session_returns[session_type.value] = data.return_pct

        db_record = SessionAnalysis(
            date=analysis.date,
            ticker=analysis.ticker,
            pre_market_return=session_returns.get('pre_market', 0),
            opening_hour_return=session_returns.get('opening', 0),
            mid_day_return=session_returns.get('mid_day', 0),
            power_hour_return=session_returns.get('power_hour', 0),
            after_hours_return=session_returns.get('after_hours', 0),
            overnight_return=analysis.overnight_return,
            best_buy_time=analysis.best_buy_session.value if analysis.best_buy_session else None,
            best_sell_time=analysis.best_sell_session.value if analysis.best_sell_session else None,
            volume_distribution=analysis.volume_distribution,
        )

        return self.db.save_session_analysis(db_record)

    def print_analysis(self, analysis: DailySessionAnalysis):
        """분석 결과 출력"""
        print("\n" + "=" * 70)
        print(f"Session Analysis: {analysis.ticker} - {analysis.date}")
        print("=" * 70)

        print(f"\n{'Session':<15} {'Open':>10} {'Close':>10} {'Return':>10} {'Volume':>15}")
        print("-" * 70)

        for session_type in SessionType:
            if session_type in analysis.sessions:
                data = analysis.sessions[session_type]
                marker = ""
                if session_type == analysis.best_buy_session:
                    marker = " 📉 BUY"
                if session_type == analysis.best_sell_session:
                    marker += " 📈 SELL"

                print(f"{session_type.value:<15} ${data.open_price:>9.2f} ${data.close_price:>9.2f} "
                      f"{data.return_pct:>+9.2f}% {data.volume:>15,}{marker}")

        print("-" * 70)
        print(f"{'Overnight':<15} {'':<10} {'':<10} {analysis.overnight_return:>+9.2f}%")
        print(f"{'Total':<15} {'':<10} {'':<10} {analysis.total_return:>+9.2f}%")

        print(f"\nVolume Distribution:")
        for session, pct in sorted(analysis.volume_distribution.items()):
            bar = "█" * int(pct * 50)
            print(f"  {session:<15} {pct*100:>5.1f}% {bar}")

        print("=" * 70)

    def print_statistics(self, stats: Dict[str, Any]):
        """통계 출력"""
        print("\n" + "=" * 70)
        print("Session Statistics Summary")
        print("=" * 70)

        print(f"\n{'Session':<15} {'Avg Ret':>10} {'Std':>8} {'WinRate':>10} {'BestBuy':>10} {'BestSell':>10}")
        print("-" * 70)

        for session in ['pre_market', 'opening', 'mid_day', 'power_hour', 'after_hours']:
            if session in stats:
                s = stats[session]
                print(f"{session:<15} {s['avg_return']:>+9.2f}% {s['std_return']:>7.2f}% "
                      f"{s['win_rate']:>9.1f}% {s['best_buy_pct']:>9.1f}% {s['best_sell_pct']:>9.1f}%")

        print("=" * 70)

        # 권고사항
        print("\n📊 Recommendations:")

        # 최적 매수 세션
        best_buy = min(stats.items(), key=lambda x: x[1]['avg_return'])
        print(f"  Best Buy Session:  {best_buy[0]} (avg {best_buy[1]['avg_return']:+.2f}%)")

        # 최적 매도 세션
        best_sell = max(stats.items(), key=lambda x: x[1]['avg_return'])
        print(f"  Best Sell Session: {best_sell[0]} (avg {best_sell[1]['avg_return']:+.2f}%)")

        # 가장 안정적인 세션
        most_stable = min(stats.items(), key=lambda x: x[1]['std_return'])
        print(f"  Most Stable:       {most_stable[0]} (std {most_stable[1]['std_return']:.2f}%)")


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EIMAS Session Analyzer Test")
    print("=" * 70)

    analyzer = SessionAnalyzer()

    # 최근 5일 분석
    print("\nAnalyzing last 5 trading days...")
    analyses = analyzer.analyze_period("SPY", days=7)

    if analyses:
        # 가장 최근 날짜 상세 출력
        analyzer.print_analysis(analyses[0])

        # 통계
        stats = analyzer.get_session_statistics(analyses)
        analyzer.print_statistics(stats)

        # DB 저장
        for analysis in analyses:
            analyzer.save_analysis(analysis)
        print(f"\n✅ Saved {len(analyses)} days to DB")
    else:
        print("No data available")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
