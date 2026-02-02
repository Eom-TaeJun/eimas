#!/usr/bin/env python3
"""
EIMAS Event Backtester
======================
이벤트 전후 예측 성과 검증

주요 기능:
1. 경제 이벤트 전후 가격 움직임 분석
2. 이벤트 유형별 승률/평균 수익 계산
3. 서프라이즈 여부에 따른 성과 분석
4. 이벤트 기반 전략 백테스트

사용법:
    from lib.event_backtester import EventBacktester

    backtester = EventBacktester()
    result = backtester.analyze_event_impact("FOMC", ticker="SPY", lookback_years=2)
    backtester.print_report(result)
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# Historical Event Data
# ============================================================================

# 과거 주요 경제 이벤트 (2023-2025)
HISTORICAL_EVENTS = {
    "fomc": [
        # 2023
        {"date": "2023-02-01", "action": "raise", "rate_change": 0.25},
        {"date": "2023-03-22", "action": "raise", "rate_change": 0.25},
        {"date": "2023-05-03", "action": "raise", "rate_change": 0.25},
        {"date": "2023-06-14", "action": "hold", "rate_change": 0.0},
        {"date": "2023-07-26", "action": "raise", "rate_change": 0.25},
        {"date": "2023-09-20", "action": "hold", "rate_change": 0.0},
        {"date": "2023-11-01", "action": "hold", "rate_change": 0.0},
        {"date": "2023-12-13", "action": "hold", "rate_change": 0.0},
        # 2024
        {"date": "2024-01-31", "action": "hold", "rate_change": 0.0},
        {"date": "2024-03-20", "action": "hold", "rate_change": 0.0},
        {"date": "2024-05-01", "action": "hold", "rate_change": 0.0},
        {"date": "2024-06-12", "action": "hold", "rate_change": 0.0},
        {"date": "2024-07-31", "action": "hold", "rate_change": 0.0},
        {"date": "2024-09-18", "action": "cut", "rate_change": -0.50},
        {"date": "2024-11-07", "action": "cut", "rate_change": -0.25},
        {"date": "2024-12-18", "action": "cut", "rate_change": -0.25},
        # 2025
        {"date": "2025-01-29", "action": "hold", "rate_change": 0.0},
        {"date": "2025-03-19", "action": "hold", "rate_change": 0.0},
        {"date": "2025-05-07", "action": "hold", "rate_change": 0.0},
        {"date": "2025-06-18", "action": "hold", "rate_change": 0.0},
        {"date": "2025-07-30", "action": "hold", "rate_change": 0.0},
        {"date": "2025-09-17", "action": "hold", "rate_change": 0.0},
        {"date": "2025-11-05", "action": "hold", "rate_change": 0.0},
        {"date": "2025-12-17", "action": "hold", "rate_change": 0.0},
    ],
    "cpi": [
        # 2024 (YoY 값)
        {"date": "2024-01-11", "actual": 3.4, "expected": 3.2},
        {"date": "2024-02-13", "actual": 3.1, "expected": 2.9},
        {"date": "2024-03-12", "actual": 3.2, "expected": 3.1},
        {"date": "2024-04-10", "actual": 3.5, "expected": 3.4},
        {"date": "2024-05-15", "actual": 3.4, "expected": 3.4},
        {"date": "2024-06-12", "actual": 3.3, "expected": 3.4},
        {"date": "2024-07-11", "actual": 3.0, "expected": 3.1},
        {"date": "2024-08-14", "actual": 2.9, "expected": 3.0},
        {"date": "2024-09-11", "actual": 2.5, "expected": 2.6},
        {"date": "2024-10-10", "actual": 2.4, "expected": 2.3},
        {"date": "2024-11-13", "actual": 2.6, "expected": 2.6},
        {"date": "2024-12-11", "actual": 2.7, "expected": 2.7},
        # 2025
        {"date": "2025-01-15", "actual": 2.9, "expected": 2.8},
        {"date": "2025-02-12", "actual": 3.0, "expected": 2.9},
        {"date": "2025-03-12", "actual": 2.8, "expected": 2.9},
        {"date": "2025-04-10", "actual": 2.4, "expected": 2.5},
        {"date": "2025-05-13", "actual": 2.3, "expected": 2.3},
        {"date": "2025-06-11", "actual": 2.2, "expected": 2.4},
        {"date": "2025-07-11", "actual": 2.4, "expected": 2.3},
        {"date": "2025-08-12", "actual": 2.5, "expected": 2.5},
        {"date": "2025-09-11", "actual": 2.3, "expected": 2.4},
        {"date": "2025-10-10", "actual": 2.2, "expected": 2.3},
        {"date": "2025-11-13", "actual": 2.4, "expected": 2.3},
        {"date": "2025-12-10", "actual": 2.5, "expected": 2.5},
    ],
    "nfp": [
        # 2024 (in thousands)
        {"date": "2024-01-05", "actual": 216, "expected": 175},
        {"date": "2024-02-02", "actual": 353, "expected": 180},
        {"date": "2024-03-08", "actual": 275, "expected": 200},
        {"date": "2024-04-05", "actual": 303, "expected": 214},
        {"date": "2024-05-03", "actual": 175, "expected": 243},
        {"date": "2024-06-07", "actual": 272, "expected": 180},
        {"date": "2024-07-05", "actual": 206, "expected": 190},
        {"date": "2024-08-02", "actual": 114, "expected": 175},
        {"date": "2024-09-06", "actual": 142, "expected": 161},
        {"date": "2024-10-04", "actual": 254, "expected": 140},
        {"date": "2024-11-01", "actual": 12, "expected": 100},
        {"date": "2024-12-06", "actual": 227, "expected": 200},
        # 2025
        {"date": "2025-01-10", "actual": 256, "expected": 160},
        {"date": "2025-02-07", "actual": 143, "expected": 170},
        {"date": "2025-03-07", "actual": 151, "expected": 160},
        {"date": "2025-04-04", "actual": 228, "expected": 135},
        {"date": "2025-05-02", "actual": 177, "expected": 130},
        {"date": "2025-06-06", "actual": 272, "expected": 180},
        {"date": "2025-07-03", "actual": 206, "expected": 185},
        {"date": "2025-08-01", "actual": 179, "expected": 175},
        {"date": "2025-09-05", "actual": 142, "expected": 163},
        {"date": "2025-10-03", "actual": 110, "expected": 150},
        {"date": "2025-11-07", "actual": 190, "expected": 120},
        {"date": "2025-12-05", "actual": 155, "expected": 177},
    ],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EventAnalysis:
    """개별 이벤트 분석 결과"""
    date: str
    event_type: str
    ticker: str

    # 이벤트 전 가격 움직임
    t_minus_5: float = 0.0
    t_minus_3: float = 0.0
    t_minus_1: float = 0.0

    # 이벤트 당일 및 이후
    t_0: float = 0.0
    t_plus_1: float = 0.0
    t_plus_3: float = 0.0
    t_plus_5: float = 0.0

    # 추가 정보
    surprise: Optional[float] = None
    action: Optional[str] = None
    volatility_before: float = 0.0
    volatility_after: float = 0.0
    volume_ratio: float = 1.0


@dataclass
class EventBacktestResult:
    """이벤트 백테스트 종합 결과"""
    event_type: str
    ticker: str
    start_date: str
    end_date: str
    total_events: int

    # 수익률 통계
    avg_t0_return: float = 0.0
    avg_t1_return: float = 0.0
    avg_t5_return: float = 0.0

    # 승률
    t0_win_rate: float = 0.0
    t1_win_rate: float = 0.0
    t5_win_rate: float = 0.0

    # 서프라이즈 분석
    positive_surprise_return: float = 0.0
    negative_surprise_return: float = 0.0

    # 변동성
    avg_vol_increase: float = 0.0
    avg_volume_ratio: float = 1.0

    # 개별 이벤트 분석
    events: List[EventAnalysis] = field(default_factory=list)

    # 전략 수익
    strategy_return: float = 0.0
    buy_and_hold_return: float = 0.0


# ============================================================================
# Event Backtester
# ============================================================================

class EventBacktester:
    """이벤트 기반 백테스터"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(f"[EventBacktester] {msg}")

    def _get_price_data(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """가격 데이터 조회 (캐시 활용)"""
        cache_key = f"{ticker}_{start}_{end}"

        if cache_key not in self._price_cache:
            self._log(f"Downloading {ticker} data...")
            df = yf.download(ticker, start=start, end=end, progress=False)

            # MultiIndex 처리
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)

            self._price_cache[cache_key] = df

        return self._price_cache[cache_key]

    def _get_return_at_offset(
        self,
        prices: pd.DataFrame,
        event_date: datetime,
        offset: int
    ) -> float:
        """이벤트 날짜 기준 offset일 후 수익률"""
        try:
            # 이벤트 날짜 또는 가장 가까운 거래일 찾기
            event_idx = prices.index.get_indexer([event_date], method='ffill')[0]

            if event_idx < 0 or event_idx + offset < 0:
                return 0.0

            target_idx = min(event_idx + offset, len(prices) - 1)

            if event_idx >= len(prices):
                return 0.0

            base_price = prices['Close'].iloc[event_idx]
            target_price = prices['Close'].iloc[target_idx]

            return (target_price / base_price - 1) * 100

        except (IndexError, KeyError):
            return 0.0

    def _calculate_volatility(
        self,
        prices: pd.DataFrame,
        event_date: datetime,
        window: int = 5,
        before: bool = True
    ) -> float:
        """이벤트 전/후 변동성 계산"""
        try:
            event_idx = prices.index.get_indexer([event_date], method='ffill')[0]

            if before:
                start_idx = max(0, event_idx - window)
                end_idx = event_idx
            else:
                start_idx = event_idx
                end_idx = min(len(prices), event_idx + window)

            returns = prices['Close'].iloc[start_idx:end_idx].pct_change()
            return returns.std() * np.sqrt(252) * 100  # 연환산

        except (IndexError, KeyError):
            return 0.0

    def _calculate_volume_ratio(
        self,
        prices: pd.DataFrame,
        event_date: datetime,
        lookback: int = 20
    ) -> float:
        """이벤트 당일 거래량 비율"""
        try:
            if 'Volume' not in prices.columns:
                return 1.0

            event_idx = prices.index.get_indexer([event_date], method='ffill')[0]

            if event_idx < lookback:
                return 1.0

            avg_volume = prices['Volume'].iloc[event_idx-lookback:event_idx].mean()
            event_volume = prices['Volume'].iloc[event_idx]

            return event_volume / avg_volume if avg_volume > 0 else 1.0

        except (IndexError, KeyError):
            return 1.0

    def analyze_single_event(
        self,
        event_type: str,
        event_data: Dict,
        ticker: str,
        prices: pd.DataFrame
    ) -> EventAnalysis:
        """단일 이벤트 분석"""
        event_date = datetime.strptime(event_data['date'], "%Y-%m-%d")

        analysis = EventAnalysis(
            date=event_data['date'],
            event_type=event_type,
            ticker=ticker
        )

        # 이벤트 전 수익률
        analysis.t_minus_5 = self._get_return_at_offset(prices, event_date, -5)
        analysis.t_minus_3 = self._get_return_at_offset(prices, event_date, -3)
        analysis.t_minus_1 = self._get_return_at_offset(prices, event_date, -1)

        # 이벤트 당일 및 이후
        analysis.t_0 = self._get_return_at_offset(prices, event_date, 0)
        analysis.t_plus_1 = self._get_return_at_offset(prices, event_date, 1)
        analysis.t_plus_3 = self._get_return_at_offset(prices, event_date, 3)
        analysis.t_plus_5 = self._get_return_at_offset(prices, event_date, 5)

        # 서프라이즈 (CPI, NFP)
        if 'actual' in event_data and 'expected' in event_data:
            actual = event_data['actual']
            expected = event_data['expected']
            if event_type == 'cpi':
                # CPI: 낮을수록 좋음
                analysis.surprise = expected - actual
            else:
                # NFP: 높을수록 좋음
                analysis.surprise = actual - expected

        # FOMC 액션
        if 'action' in event_data:
            analysis.action = event_data['action']

        # 변동성
        analysis.volatility_before = self._calculate_volatility(prices, event_date, before=True)
        analysis.volatility_after = self._calculate_volatility(prices, event_date, before=False)

        # 거래량
        analysis.volume_ratio = self._calculate_volume_ratio(prices, event_date)

        return analysis

    def analyze_event_impact(
        self,
        event_type: str,
        ticker: str = "SPY",
        start_date: str = "2024-01-01",
        end_date: str = None
    ) -> EventBacktestResult:
        """이벤트 유형별 임팩트 분석"""

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        self._log(f"Analyzing {event_type.upper()} impact on {ticker}...")

        # 이벤트 데이터 가져오기
        event_key = event_type.lower()
        if event_key not in HISTORICAL_EVENTS:
            self._log(f"Unknown event type: {event_type}")
            return EventBacktestResult(
                event_type=event_type,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                total_events=0
            )

        # 기간 내 이벤트 필터링
        events = []
        for event_data in HISTORICAL_EVENTS[event_key]:
            event_date = event_data['date']
            if start_date <= event_date <= end_date:
                events.append(event_data)

        self._log(f"  Found {len(events)} events in period")

        if not events:
            return EventBacktestResult(
                event_type=event_type,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                total_events=0
            )

        # 가격 데이터 (이벤트 전후 버퍼 포함)
        buffer_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        buffer_end = (datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
        prices = self._get_price_data(ticker, buffer_start, buffer_end)

        if prices.empty:
            self._log("  No price data available")
            return EventBacktestResult(
                event_type=event_type,
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                total_events=0
            )

        # 개별 이벤트 분석
        analyses = []
        for event_data in events:
            analysis = self.analyze_single_event(event_type, event_data, ticker, prices)
            analyses.append(analysis)

        # 통계 계산
        result = EventBacktestResult(
            event_type=event_type,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            total_events=len(analyses),
            events=analyses
        )

        if analyses:
            # 평균 수익률
            result.avg_t0_return = np.mean([a.t_0 for a in analyses])
            result.avg_t1_return = np.mean([a.t_plus_1 for a in analyses])
            result.avg_t5_return = np.mean([a.t_plus_5 for a in analyses])

            # 승률 (T+1 기준)
            result.t0_win_rate = sum(1 for a in analyses if a.t_0 > 0) / len(analyses)
            result.t1_win_rate = sum(1 for a in analyses if a.t_plus_1 > 0) / len(analyses)
            result.t5_win_rate = sum(1 for a in analyses if a.t_plus_5 > 0) / len(analyses)

            # 서프라이즈 분석
            positive_surprises = [a for a in analyses if a.surprise and a.surprise > 0]
            negative_surprises = [a for a in analyses if a.surprise and a.surprise < 0]

            if positive_surprises:
                result.positive_surprise_return = np.mean([a.t_plus_1 for a in positive_surprises])
            if negative_surprises:
                result.negative_surprise_return = np.mean([a.t_plus_1 for a in negative_surprises])

            # 변동성 증가
            result.avg_vol_increase = np.mean([
                (a.volatility_after - a.volatility_before) / a.volatility_before * 100
                if a.volatility_before > 0 else 0
                for a in analyses
            ])

            # 거래량 배수
            result.avg_volume_ratio = np.mean([a.volume_ratio for a in analyses])

        return result

    def backtest_event_strategy(
        self,
        event_type: str,
        ticker: str = "SPY",
        strategy: str = "long_after",  # long_after, fade_move, follow_surprise
        start_date: str = "2024-01-01",
        end_date: str = None,
        hold_days: int = 5
    ) -> Dict[str, Any]:
        """이벤트 기반 전략 백테스트"""

        result = self.analyze_event_impact(event_type, ticker, start_date, end_date)

        if result.total_events == 0:
            return {"error": "No events found"}

        trades = []

        for event in result.events:
            # 전략에 따른 방향 결정
            if strategy == "long_after":
                # 이벤트 다음날 매수
                direction = 1
            elif strategy == "fade_move":
                # 이벤트 당일 반대 방향
                direction = -1 if event.t_0 > 0 else 1
            elif strategy == "follow_surprise":
                # 서프라이즈 방향 추종
                if event.surprise is None:
                    direction = 1
                else:
                    direction = 1 if event.surprise > 0 else -1
            else:
                direction = 1

            # 수익률 계산 (T+1 ~ T+hold_days)
            if hold_days == 1:
                pnl = event.t_plus_1
            elif hold_days == 3:
                pnl = event.t_plus_3
            else:
                pnl = event.t_plus_5

            trades.append({
                "date": event.date,
                "direction": "LONG" if direction > 0 else "SHORT",
                "pnl": pnl * direction
            })

        total_return = sum(t['pnl'] for t in trades)
        win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades) if trades else 0

        return {
            "strategy": strategy,
            "event_type": event_type,
            "ticker": ticker,
            "total_trades": len(trades),
            "total_return": total_return,
            "avg_return": total_return / len(trades) if trades else 0,
            "win_rate": win_rate,
            "trades": trades
        }

    def print_report(self, result: EventBacktestResult):
        """분석 결과 리포트 출력"""
        print("\n" + "=" * 70)
        print(f"Event Impact Analysis: {result.event_type.upper()}")
        print("=" * 70)

        print(f"\nTicker: {result.ticker}")
        print(f"Period: {result.start_date} ~ {result.end_date}")
        print(f"Total Events: {result.total_events}")

        if result.total_events == 0:
            print("\nNo events found in this period.")
            return

        print("\n" + "-" * 40)
        print("Average Returns Around Event")
        print("-" * 40)
        print(f"  T+0 (Event Day): {result.avg_t0_return:+.2f}%")
        print(f"  T+1 (Next Day):  {result.avg_t1_return:+.2f}%")
        print(f"  T+5 (5 Days):    {result.avg_t5_return:+.2f}%")

        print("\n" + "-" * 40)
        print("Win Rates (Positive Returns)")
        print("-" * 40)
        print(f"  T+0: {result.t0_win_rate*100:.0f}%")
        print(f"  T+1: {result.t1_win_rate*100:.0f}%")
        print(f"  T+5: {result.t5_win_rate*100:.0f}%")

        if result.positive_surprise_return != 0 or result.negative_surprise_return != 0:
            print("\n" + "-" * 40)
            print("Surprise Analysis (T+1 Returns)")
            print("-" * 40)
            print(f"  Positive Surprise: {result.positive_surprise_return:+.2f}%")
            print(f"  Negative Surprise: {result.negative_surprise_return:+.2f}%")

        print("\n" + "-" * 40)
        print("Volatility & Volume")
        print("-" * 40)
        print(f"  Avg Volatility Change: {result.avg_vol_increase:+.1f}%")
        print(f"  Avg Volume Ratio: {result.avg_volume_ratio:.1f}x")

        # 개별 이벤트 테이블
        print("\n" + "-" * 40)
        print("Individual Events")
        print("-" * 40)
        print(f"{'Date':<12} {'T+0':>8} {'T+1':>8} {'T+5':>8} {'Vol Ratio':>10}")
        print("-" * 50)

        for event in result.events[-10:]:  # 최근 10개만
            print(f"{event.date:<12} {event.t_0:>+7.2f}% {event.t_plus_1:>+7.2f}% {event.t_plus_5:>+7.2f}% {event.volume_ratio:>9.1f}x")

        print("=" * 70)

    def compare_events(
        self,
        ticker: str = "SPY",
        start_date: str = "2024-01-01",
        end_date: str = None
    ) -> pd.DataFrame:
        """여러 이벤트 유형 비교"""
        results = []

        for event_type in ["fomc", "cpi", "nfp"]:
            result = self.analyze_event_impact(event_type, ticker, start_date, end_date)

            if result.total_events > 0:
                results.append({
                    "Event": event_type.upper(),
                    "Count": result.total_events,
                    "Avg T+1": f"{result.avg_t1_return:+.2f}%",
                    "Avg T+5": f"{result.avg_t5_return:+.2f}%",
                    "Win Rate T+1": f"{result.t1_win_rate*100:.0f}%",
                    "Vol Ratio": f"{result.avg_volume_ratio:.1f}x"
                })

        return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    backtester = EventBacktester(verbose=True)

    # 1. FOMC 분석
    fomc_result = backtester.analyze_event_impact("fomc", "SPY", "2024-01-01")
    backtester.print_report(fomc_result)

    # 2. CPI 분석
    cpi_result = backtester.analyze_event_impact("cpi", "SPY", "2024-01-01")
    backtester.print_report(cpi_result)

    # 3. NFP 분석
    nfp_result = backtester.analyze_event_impact("nfp", "SPY", "2024-01-01")
    backtester.print_report(nfp_result)

    # 4. 이벤트 비교
    print("\n" + "=" * 70)
    print("Event Comparison Summary")
    print("=" * 70)
    comparison = backtester.compare_events("SPY", "2024-01-01")
    print(comparison.to_string(index=False))

    # 5. 전략 백테스트
    print("\n" + "=" * 70)
    print("Strategy Backtest: Long After FOMC")
    print("=" * 70)
    strategy_result = backtester.backtest_event_strategy(
        "fomc", "SPY", strategy="long_after", hold_days=5
    )
    print(f"Total Trades: {strategy_result['total_trades']}")
    print(f"Total Return: {strategy_result['total_return']:.2f}%")
    print(f"Avg Return per Trade: {strategy_result['avg_return']:.2f}%")
    print(f"Win Rate: {strategy_result['win_rate']*100:.0f}%")
