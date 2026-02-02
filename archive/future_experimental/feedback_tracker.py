#!/usr/bin/env python3
"""
EIMAS Feedback Tracker
======================
예측 vs 실제 비교 및 시그널 성과 추적

핵심 기능:
1. 포트폴리오 예측값 vs 실제값 비교
2. 시그널별 정확도 추적
3. 자동 가중치 조정 제안

Usage:
    from lib.feedback_tracker import FeedbackTracker

    tracker = FeedbackTracker()
    tracker.update_all_performance()
    tracker.print_signal_accuracy()
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

from lib.trading_db import (
    TradingDB, Signal, SignalSource, SignalAction,
    PortfolioCandidate, SignalPerformance
)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SignalEvaluation:
    """시그널 평가 결과"""
    signal_id: int
    signal_source: str
    signal_action: str
    ticker: str
    signal_date: date
    conviction: float
    # 성과
    return_1d: float
    return_5d: float
    return_20d: float
    max_gain: float
    max_loss: float
    # 평가
    is_correct: bool
    score: float  # -1 ~ +1


@dataclass
class SourceAccuracy:
    """소스별 정확도"""
    source: str
    total_signals: int
    correct_signals: int
    accuracy: float
    avg_return_when_correct: float
    avg_return_when_wrong: float
    profit_factor: float
    suggested_weight: float


# ============================================================================
# Feedback Tracker
# ============================================================================

class FeedbackTracker:
    """피드백 추적 시스템"""

    def __init__(self, db: TradingDB = None):
        self.db = db or TradingDB()
        self._price_cache: Dict[str, pd.DataFrame] = {}

    def _get_price_data(self, ticker: str, days: int = 90) -> pd.DataFrame:
        """가격 데이터 캐시 로드"""
        if ticker not in self._price_cache:
            end = datetime.now()
            start = end - timedelta(days=days)

            df = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            self._price_cache[ticker] = df

        return self._price_cache[ticker]

    def _calculate_returns(
        self,
        ticker: str,
        signal_date: date,
        lookforward_days: List[int] = [1, 5, 20]
    ) -> Dict[str, float]:
        """시그널 날짜 기준 수익률 계산"""
        df = self._get_price_data(ticker)
        if df.empty:
            return {}

        # 시그널 날짜의 종가 찾기
        signal_dt = pd.Timestamp(signal_date)

        # 시그널 날짜 이후 데이터
        future_df = df[df.index >= signal_dt]
        if len(future_df) < 2:
            return {}

        base_price = float(future_df['Close'].iloc[0])
        returns = {}

        for days in lookforward_days:
            if len(future_df) > days:
                future_price = float(future_df['Close'].iloc[days])
                returns[f'return_{days}d'] = (future_price / base_price - 1) * 100
            else:
                returns[f'return_{days}d'] = None

        # 최대 상승/하락 (20일 기준)
        if len(future_df) > 20:
            period_df = future_df.iloc[:21]
            max_price = float(period_df['High'].max())
            min_price = float(period_df['Low'].min())
            returns['max_gain'] = (max_price / base_price - 1) * 100
            returns['max_loss'] = (min_price / base_price - 1) * 100
        else:
            returns['max_gain'] = None
            returns['max_loss'] = None

        return returns

    def _evaluate_signal(
        self,
        signal: Dict,
        returns: Dict[str, float]
    ) -> SignalEvaluation:
        """개별 시그널 평가"""
        action = signal['signal_action']
        conviction = signal.get('conviction', 0.5)

        # 5일 수익률 기준 평가
        return_5d = returns.get('return_5d', 0) or 0

        # 시그널 방향 맞았는지
        if action == 'buy':
            is_correct = return_5d > 0
            score = return_5d / 10  # 정규화
        elif action == 'sell':
            is_correct = return_5d < 0
            score = -return_5d / 10
        elif action == 'reduce':
            is_correct = return_5d < 0
            score = -return_5d / 10
        elif action == 'hedge':
            # 헤지는 변동성이 높으면 정답
            max_move = max(abs(returns.get('max_gain', 0) or 0), abs(returns.get('max_loss', 0) or 0))
            is_correct = max_move > 3  # 3% 이상 움직임
            score = max_move / 10 if is_correct else -0.1
        else:  # hold
            # 보합 시그널은 ±2% 이내면 정답
            is_correct = abs(return_5d) < 2
            score = 0.1 if is_correct else -abs(return_5d) / 10

        # 날짜 파싱
        signal_date = signal['timestamp']
        if isinstance(signal_date, str):
            signal_date = datetime.fromisoformat(signal_date).date()
        elif isinstance(signal_date, datetime):
            signal_date = signal_date.date()

        return SignalEvaluation(
            signal_id=signal['id'],
            signal_source=signal['signal_source'],
            signal_action=action,
            ticker=signal.get('ticker', 'SPY'),
            signal_date=signal_date,
            conviction=conviction,
            return_1d=returns.get('return_1d', 0) or 0,
            return_5d=return_5d,
            return_20d=returns.get('return_20d', 0) or 0,
            max_gain=returns.get('max_gain', 0) or 0,
            max_loss=returns.get('max_loss', 0) or 0,
            is_correct=is_correct,
            score=min(max(score, -1), 1),  # -1 ~ +1
        )

    def evaluate_signals(self, days: int = 30) -> List[SignalEvaluation]:
        """모든 시그널 평가"""
        # 평가 가능한 시그널 조회 (최소 5일 전)
        cutoff = date.today() - timedelta(days=5)
        start_date = cutoff - timedelta(days=days)

        signals = self.db.get_signals(start_date=start_date, end_date=cutoff, limit=500)

        print(f"Evaluating {len(signals)} signals from {start_date} to {cutoff}...")

        evaluations = []
        for signal in signals:
            ticker = signal.get('ticker', 'SPY')

            # 날짜 파싱
            signal_date = signal['timestamp']
            if isinstance(signal_date, str):
                signal_date = datetime.fromisoformat(signal_date).date()
            elif isinstance(signal_date, datetime):
                signal_date = signal_date.date()

            returns = self._calculate_returns(ticker, signal_date)

            if returns:
                evaluation = self._evaluate_signal(signal, returns)
                evaluations.append(evaluation)

                # DB에 저장
                self._save_signal_performance(evaluation)

        return evaluations

    def _save_signal_performance(self, eval: SignalEvaluation):
        """시그널 성과 DB 저장"""
        perf = SignalPerformance(
            signal_id=eval.signal_id,
            evaluation_date=date.today(),
            return_1d=eval.return_1d,
            return_5d=eval.return_5d,
            return_20d=eval.return_20d,
            max_gain=eval.max_gain,
            max_loss=eval.max_loss,
            signal_accuracy=eval.is_correct,
        )
        self.db.save_signal_performance(perf)

    def get_source_accuracy(self, evaluations: List[SignalEvaluation]) -> List[SourceAccuracy]:
        """소스별 정확도 계산"""
        by_source: Dict[str, List[SignalEvaluation]] = {}

        for e in evaluations:
            if e.signal_source not in by_source:
                by_source[e.signal_source] = []
            by_source[e.signal_source].append(e)

        results = []
        for source, evals in by_source.items():
            total = len(evals)
            correct = sum(1 for e in evals if e.is_correct)
            accuracy = correct / total if total > 0 else 0

            correct_returns = [e.return_5d for e in evals if e.is_correct]
            wrong_returns = [e.return_5d for e in evals if not e.is_correct]

            avg_correct = np.mean(correct_returns) if correct_returns else 0
            avg_wrong = np.mean(wrong_returns) if wrong_returns else 0

            # Profit Factor
            gains = sum(e.return_5d for e in evals if e.return_5d > 0)
            losses = abs(sum(e.return_5d for e in evals if e.return_5d < 0))
            profit_factor = gains / losses if losses > 0 else float('inf')

            # 가중치 제안 (정확도 + Profit Factor 기반)
            suggested_weight = min(accuracy * profit_factor / 2, 1.0)

            results.append(SourceAccuracy(
                source=source,
                total_signals=total,
                correct_signals=correct,
                accuracy=round(accuracy * 100, 1),
                avg_return_when_correct=round(avg_correct, 2),
                avg_return_when_wrong=round(avg_wrong, 2),
                profit_factor=round(profit_factor, 2),
                suggested_weight=round(suggested_weight, 2),
            ))

        return sorted(results, key=lambda x: -x.accuracy)

    def update_portfolio_performance(self, portfolio_id: int) -> bool:
        """포트폴리오 실제 성과 업데이트"""
        portfolios = self.db.get_portfolios(limit=100)
        portfolio = next((p for p in portfolios if p['id'] == portfolio_id), None)

        if not portfolio:
            return False

        # 포트폴리오 생성 날짜
        created = portfolio['timestamp']
        if isinstance(created, str):
            created = datetime.fromisoformat(created)

        created_date = created.date() if isinstance(created, datetime) else created

        # 배분에 따른 실제 수익률 계산
        allocations = portfolio['allocations']
        if not allocations:
            return False

        total_return_1d = 0
        total_return_1w = 0
        total_return_1m = 0

        for ticker, weight in allocations.items():
            if ticker == 'CASH':
                continue

            returns = self._calculate_returns(ticker, created_date, [1, 5, 20])

            if returns:
                total_return_1d += weight * (returns.get('return_1d', 0) or 0)
                total_return_1w += weight * (returns.get('return_5d', 0) or 0)
                total_return_1m += weight * (returns.get('return_20d', 0) or 0)

        # DB 업데이트
        self.db.update_actual_returns(
            portfolio_id=portfolio_id,
            record_date=created_date,
            actual_1d=total_return_1d,
            actual_1w=total_return_1w,
            actual_1m=total_return_1m,
        )

        return True

    def generate_weight_recommendations(
        self,
        accuracies: List[SourceAccuracy]
    ) -> Dict[str, float]:
        """가중치 조정 권고"""
        recommendations = {}

        # 기본 가중치
        base_weights = {
            'regime_detector': 0.25,
            'critical_path': 0.25,
            'fear_greed': 0.20,
            'vix_structure': 0.15,
            'etf_flow': 0.15,
        }

        for acc in accuracies:
            source = acc.source
            if source in base_weights:
                # 정확도에 따라 가중치 조정
                if acc.accuracy >= 60:
                    multiplier = 1.2
                elif acc.accuracy >= 50:
                    multiplier = 1.0
                elif acc.accuracy >= 40:
                    multiplier = 0.8
                else:
                    multiplier = 0.5

                recommendations[source] = round(base_weights[source] * multiplier, 2)
            else:
                recommendations[source] = acc.suggested_weight

        # 정규화
        total = sum(recommendations.values())
        if total > 0:
            recommendations = {k: round(v / total, 2) for k, v in recommendations.items()}

        return recommendations

    def print_signal_accuracy(self, accuracies: List[SourceAccuracy]):
        """정확도 리포트 출력"""
        print("\n" + "=" * 70)
        print("Signal Source Accuracy Report")
        print("=" * 70)

        print(f"\n{'Source':<20} {'Total':>8} {'Correct':>8} {'Accuracy':>10} {'PF':>8} {'Weight':>8}")
        print("-" * 70)

        for acc in accuracies:
            print(f"{acc.source:<20} {acc.total_signals:>8} {acc.correct_signals:>8} "
                  f"{acc.accuracy:>9.1f}% {acc.profit_factor:>8.2f} {acc.suggested_weight:>7.2f}")

        print("=" * 70)

        # 권고사항
        recommendations = self.generate_weight_recommendations(accuracies)
        print("\n📊 Recommended Weights:")
        for source, weight in sorted(recommendations.items(), key=lambda x: -x[1]):
            bar = "█" * int(weight * 40)
            print(f"  {source:<20} {weight:>5.0%} {bar}")

    def run_daily_update(self) -> Dict[str, Any]:
        """일일 업데이트 실행"""
        print("=" * 70)
        print("EIMAS Daily Feedback Update")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)

        results = {
            'date': date.today().isoformat(),
            'signals_evaluated': 0,
            'portfolios_updated': 0,
            'source_accuracies': {},
        }

        # 1. 시그널 평가
        print("\n[1/3] Evaluating signals...")
        evaluations = self.evaluate_signals(days=30)
        results['signals_evaluated'] = len(evaluations)
        print(f"  Evaluated: {len(evaluations)} signals")

        # 2. 소스별 정확도
        print("\n[2/3] Calculating source accuracy...")
        accuracies = self.get_source_accuracy(evaluations)
        self.print_signal_accuracy(accuracies)
        results['source_accuracies'] = {a.source: a.accuracy for a in accuracies}

        # 3. 포트폴리오 성과 업데이트
        print("\n[3/3] Updating portfolio performance...")
        portfolios = self.db.get_portfolios(limit=50)
        updated = 0
        for p in portfolios:
            if self.update_portfolio_performance(p['id']):
                updated += 1
        results['portfolios_updated'] = updated
        print(f"  Updated: {updated} portfolios")

        print("\n" + "=" * 70)
        print("Daily Update Complete!")
        print("=" * 70)

        return results


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EIMAS Feedback Tracker Test")
    print("=" * 70)

    tracker = FeedbackTracker()

    # 일일 업데이트 실행
    results = tracker.run_daily_update()

    # DB 요약
    tracker.db.print_summary()

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
