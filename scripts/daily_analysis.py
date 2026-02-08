#!/usr/bin/env python3
"""
EIMAS Daily Analysis Runner
============================
매일 실행되는 통합 분석 파이프라인

실행 순서:
1. 시그널 수집 → DB 저장
2. 포트폴리오 후보 생성
3. 세션 분석 (전일)
4. 피드백 업데이트
5. 일일 리포트 생성

Usage:
    python scripts/daily_analysis.py
    python scripts/daily_analysis.py --report-only
"""

from pathlib import Path

try:
    from _project_bootstrap import ensure_project_root
except ImportError:
    from scripts._project_bootstrap import ensure_project_root

ensure_project_root(__file__)

import argparse
from datetime import datetime, date, timedelta
from typing import Dict, Any
import json

from lib.trading_db import TradingDB
from lib.signal_pipeline import SignalPipeline, PortfolioGenerator
from lib.session_analyzer import SessionAnalyzer
from lib.feedback_tracker import FeedbackTracker


# ============================================================================
# Daily Runner
# ============================================================================

class DailyAnalysisRunner:
    """일일 분석 러너"""

    def __init__(self):
        self.db = TradingDB()
        self.results: Dict[str, Any] = {
            'run_date': datetime.now().isoformat(),
            'steps': {},
        }

    def run_signal_collection(self) -> Dict:
        """Step 1: 시그널 수집"""
        print("\n" + "=" * 70)
        print("[Step 1/5] Signal Collection")
        print("=" * 70)

        pipeline = SignalPipeline(self.db)
        signals = pipeline.run()
        consensus = pipeline.get_consensus()

        result = {
            'signals_count': len(signals),
            'consensus_action': consensus['action'],
            'consensus_conviction': consensus['conviction'],
        }

        self.results['steps']['signal_collection'] = result
        self.results['signals'] = [s.to_dict() for s in signals]

        pipeline.print_summary()
        return result

    def run_portfolio_generation(self) -> Dict:
        """Step 2: 포트폴리오 생성"""
        print("\n" + "=" * 70)
        print("[Step 2/5] Portfolio Generation")
        print("=" * 70)

        # 최근 시그널 사용
        signals = self.db.get_recent_signals(hours=24)

        # Signal 객체로 변환
        from lib.trading_db import Signal, SignalSource, SignalAction
        signal_objects = []
        for s in signals:
            try:
                signal = Signal(
                    source=SignalSource(s['signal_source']),
                    action=SignalAction(s['signal_action']),
                    ticker=s.get('ticker', 'SPY'),
                    conviction=s.get('conviction', 0.5),
                    reasoning=s.get('reasoning', ''),
                    metadata=json.loads(s.get('metadata', '{}')) if s.get('metadata') else {},
                )
                signal.id = s['id']
                signal_objects.append(signal)
            except:
                pass

        generator = PortfolioGenerator(self.db)
        portfolios = generator.generate_all_profiles(signal_objects)

        result = {
            'portfolios_generated': len(portfolios),
            'profiles': [p.profile.value for p in portfolios],
        }

        self.results['steps']['portfolio_generation'] = result
        self.results['portfolios'] = [p.to_dict() for p in portfolios]

        generator.print_portfolios(portfolios)
        return result

    def run_session_analysis(self) -> Dict:
        """Step 3: 세션 분석"""
        print("\n" + "=" * 70)
        print("[Step 3/5] Session Analysis")
        print("=" * 70)

        analyzer = SessionAnalyzer(self.db)

        # 어제 분석 (오늘은 아직 끝나지 않았으므로)
        yesterday = date.today() - timedelta(days=1)

        # 주말 스킵
        while yesterday.weekday() >= 5:
            yesterday -= timedelta(days=1)

        analysis = analyzer.analyze_day("SPY", yesterday)

        result = {
            'date_analyzed': yesterday.isoformat(),
            'sessions_available': 0,
        }

        if analysis:
            result['sessions_available'] = len(analysis.sessions)
            result['total_return'] = round(analysis.total_return, 3)
            result['best_buy_session'] = analysis.best_buy_session.value if analysis.best_buy_session else None
            result['best_sell_session'] = analysis.best_sell_session.value if analysis.best_sell_session else None

            analyzer.save_analysis(analysis)
            analyzer.print_analysis(analysis)
        else:
            print(f"  No session data for {yesterday}")

        self.results['steps']['session_analysis'] = result
        return result

    def run_feedback_update(self) -> Dict:
        """Step 4: 피드백 업데이트"""
        print("\n" + "=" * 70)
        print("[Step 4/5] Feedback Update")
        print("=" * 70)

        tracker = FeedbackTracker(self.db)

        # 시그널 평가
        evaluations = tracker.evaluate_signals(days=30)

        # 소스별 정확도
        accuracies = tracker.get_source_accuracy(evaluations)

        result = {
            'signals_evaluated': len(evaluations),
            'source_accuracies': {a.source: a.accuracy for a in accuracies},
        }

        # 포트폴리오 성과 업데이트
        portfolios = self.db.get_portfolios(limit=50)
        updated = 0
        for p in portfolios:
            if tracker.update_portfolio_performance(p['id']):
                updated += 1

        result['portfolios_updated'] = updated

        self.results['steps']['feedback_update'] = result

        if accuracies:
            tracker.print_signal_accuracy(accuracies)

        return result

    def generate_daily_report(self) -> str:
        """Step 5: 일일 리포트 생성"""
        print("\n" + "=" * 70)
        print("[Step 5/5] Daily Report Generation")
        print("=" * 70)

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("EIMAS Daily Analysis Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_lines.append("=" * 70)

        # 시그널 요약
        if 'signal_collection' in self.results['steps']:
            sc = self.results['steps']['signal_collection']
            report_lines.append(f"\n📊 Signals Collected: {sc['signals_count']}")
            report_lines.append(f"   Consensus: {sc['consensus_action'].upper()} ({sc['consensus_conviction']:.0%})")

        # 포트폴리오 요약
        if 'portfolios' in self.results:
            report_lines.append(f"\n💼 Portfolios Generated: {len(self.results['portfolios'])}")
            for p in self.results['portfolios']:
                report_lines.append(f"   {p['profile'].upper()}: Sharpe {p['expected_sharpe']:.2f}")

        # 세션 분석
        if 'session_analysis' in self.results['steps']:
            sa = self.results['steps']['session_analysis']
            if sa['sessions_available'] > 0:
                report_lines.append(f"\n⏰ Session Analysis ({sa['date_analyzed']}):")
                report_lines.append(f"   Total Return: {sa.get('total_return', 0):+.2f}%")
                report_lines.append(f"   Best Buy:  {sa.get('best_buy_session', 'N/A')}")
                report_lines.append(f"   Best Sell: {sa.get('best_sell_session', 'N/A')}")

        # 피드백
        if 'feedback_update' in self.results['steps']:
            fu = self.results['steps']['feedback_update']
            report_lines.append(f"\n📈 Feedback Update:")
            report_lines.append(f"   Signals Evaluated: {fu['signals_evaluated']}")
            report_lines.append(f"   Portfolios Updated: {fu['portfolios_updated']}")
            if fu['source_accuracies']:
                report_lines.append("   Source Accuracy:")
                for source, acc in fu['source_accuracies'].items():
                    report_lines.append(f"     {source}: {acc:.1f}%")

        # DB 요약
        summary = self.db.get_summary()
        report_lines.append(f"\n📁 Database Status:")
        report_lines.append(f"   Total Signals: {summary['total_signals']}")
        report_lines.append(f"   Total Portfolios: {summary['total_portfolios']}")
        report_lines.append(f"   Total Executions: {summary['total_executions']}")

        report_lines.append("\n" + "=" * 70)

        report = "\n".join(report_lines)
        print(report)

        # 파일 저장
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)

        report_file = output_dir / f"daily_report_{date.today().isoformat()}.txt"
        with open(report_file, 'w') as f:
            f.write(report)

        # JSON 저장
        json_file = output_dir / f"daily_analysis_{date.today().isoformat()}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n✅ Report saved: {report_file}")
        print(f"✅ Data saved: {json_file}")

        return report

    def run(self, skip_steps: list = None):
        """전체 파이프라인 실행"""
        skip_steps = skip_steps or []

        print("=" * 70)
        print("EIMAS Daily Analysis Pipeline")
        print("=" * 70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if 'signals' not in skip_steps:
            self.run_signal_collection()

        if 'portfolios' not in skip_steps:
            self.run_portfolio_generation()

        if 'sessions' not in skip_steps:
            self.run_session_analysis()

        if 'feedback' not in skip_steps:
            self.run_feedback_update()

        report = self.generate_daily_report()

        print("\n" + "=" * 70)
        print("Pipeline Complete!")
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        return self.results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIMAS Daily Analysis")
    parser.add_argument('--report-only', action='store_true', help='Generate report only')
    parser.add_argument('--skip', nargs='+', choices=['signals', 'portfolios', 'sessions', 'feedback'],
                       help='Steps to skip')

    args = parser.parse_args()

    runner = DailyAnalysisRunner()

    if args.report_only:
        runner.generate_daily_report()
    else:
        runner.run(skip_steps=args.skip or [])
