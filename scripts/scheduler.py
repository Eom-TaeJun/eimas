#!/usr/bin/env python3
"""
EIMAS Scheduler
===============
자동화 스케줄러 - 정기 작업 실행

Usage:
    python scripts/scheduler.py

Schedule:
    - 06:00 AM: Pre-market analysis
    - 09:30 AM: Market open analysis
    - 12:00 PM: Mid-day check
    - 04:00 PM: Market close analysis
    - 09:00 PM: Daily report generation

Requires: schedule (pip install schedule)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import time
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Any
import threading

# Schedule library (optional)
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    print("Warning: 'schedule' library not installed. Run: pip install schedule")


# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(OUTPUT_DIR / "scheduler.log")),
    ]
)
logger = logging.getLogger('EIMAS-Scheduler')


# ============================================================================
# Task Definitions
# ============================================================================

class SchedulerTask:
    """스케줄러 작업"""

    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description
        self.last_run = None
        self.run_count = 0
        self.last_error = None

    def run(self):
        """작업 실행"""
        logger.info(f"Starting task: {self.name}")
        start = datetime.now()

        try:
            result = self.func()
            elapsed = (datetime.now() - start).total_seconds()
            self.last_run = datetime.now()
            self.run_count += 1
            self.last_error = None
            logger.info(f"Completed: {self.name} ({elapsed:.1f}s)")
            return result

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Failed: {self.name} - {e}")
            return None


# ============================================================================
# Task Functions
# ============================================================================

def task_premarket_analysis():
    """프리마켓 분석 (06:00 AM)"""
    from lib.signal_pipeline import SignalPipeline
    from lib.regime_detector import RegimeDetector
    from lib.alert_manager import AlertManager

    logger.info("=== Pre-Market Analysis ===")

    # 레짐 감지
    detector = RegimeDetector(ticker='SPY')
    regime = detector.detect()
    logger.info(f"Regime: {regime.regime.value}")

    # 시그널 수집
    pipeline = SignalPipeline()
    signals = pipeline.run()
    consensus = pipeline.get_consensus()
    logger.info(f"Signals: {len(signals)}, Consensus: {consensus['action']}")

    # 알림 전송 (중요한 시그널만)
    am = AlertManager()
    events = am.process_signals(signals)

    return {
        'regime': regime.regime.value,
        'signals': len(signals),
        'consensus': consensus,
    }


def task_market_open():
    """장 오픈 분석 (09:30 AM)"""
    from lib.sector_rotation import SectorRotationModel
    from lib.correlation_monitor import CorrelationMonitor

    logger.info("=== Market Open Analysis ===")

    # 섹터 분석
    sector_model = SectorRotationModel()
    sector_result = sector_model.analyze()
    logger.info(f"Cycle: {sector_result.cycle.current_cycle.value}")
    logger.info(f"Top sectors: {sector_result.top_sectors}")

    # 상관관계 체크
    cm = CorrelationMonitor(['SPY', 'TLT', 'GLD', 'QQQ'])
    corr_result = cm.analyze()
    logger.info(f"Correlation regime: {corr_result.regime.value}")

    return {
        'cycle': sector_result.cycle.current_cycle.value,
        'top_sectors': sector_result.top_sectors,
        'correlation_regime': corr_result.regime.value,
    }


def task_midday_check():
    """정오 체크 (12:00 PM)"""
    from lib.paper_trader import PaperTrader
    from lib.risk_manager import RiskManager

    logger.info("=== Mid-Day Check ===")

    # 포트폴리오 상태
    trader = PaperTrader(account_name='default')
    summary = trader.get_portfolio_summary()
    logger.info(f"Portfolio: ${summary.total_value:,.2f} ({summary.total_pnl_pct:+.2f}%)")

    # 리스크 체크
    if summary.positions:
        holdings = {t: p.market_value / summary.total_value
                   for t, p in summary.positions.items()}
        rm = RiskManager()
        risk = rm.calculate_portfolio_risk(holdings, summary.total_value)
        logger.info(f"Risk level: {risk.risk_level.value}")

        return {
            'total_value': summary.total_value,
            'pnl_pct': summary.total_pnl_pct,
            'risk_level': risk.risk_level.value,
        }

    return {'total_value': summary.total_value, 'pnl_pct': 0}


def task_market_close():
    """장 마감 분석 (04:00 PM)"""
    from lib.session_analyzer import SessionAnalyzer
    from lib.feedback_tracker import FeedbackTracker
    from lib.paper_trader import PaperTrader

    logger.info("=== Market Close Analysis ===")

    # 일일 스냅샷 저장
    trader = PaperTrader(account_name='default')
    trader.save_daily_snapshot()
    summary = trader.get_portfolio_summary()
    logger.info(f"Daily close: ${summary.total_value:,.2f}")

    # 피드백 업데이트
    tracker = FeedbackTracker()
    update_result = tracker.run_daily_update()
    logger.info(f"Signals evaluated: {update_result.get('signals_evaluated', 0)}")

    return {
        'total_value': summary.total_value,
        'signals_evaluated': update_result.get('signals_evaluated', 0),
    }


def task_daily_report():
    """일일 리포트 (09:00 PM)"""
    from lib.report_generator import ReportGenerator
    from lib.alert_manager import AlertManager
    from lib.signal_pipeline import SignalPipeline, PortfolioGenerator
    from lib.paper_trader import PaperTrader

    logger.info("=== Daily Report Generation ===")

    # 리포트 생성
    generator = ReportGenerator()
    report_path = generator.generate_daily_report()
    logger.info(f"Report saved: {report_path}")

    # 일일 요약 알림
    pipeline = SignalPipeline()
    signals = pipeline.run()
    consensus = pipeline.get_consensus()

    pg = PortfolioGenerator(pipeline.db)
    portfolios = pg.generate_all_profiles(signals)

    am = AlertManager()
    am.send_daily_summary(
        signals_count=len(signals),
        consensus_action=consensus['action'],
        consensus_conviction=consensus['conviction'],
        portfolios=[p.to_dict() for p in portfolios],
    )

    return {
        'report_path': report_path,
        'signals': len(signals),
        'portfolios': len(portfolios),
    }


def task_hourly_signal_check():
    """시간별 시그널 체크"""
    from lib.signal_pipeline import SignalPipeline
    from lib.alert_manager import AlertManager

    logger.info("=== Hourly Signal Check ===")

    pipeline = SignalPipeline()
    signals = pipeline.run()

    # 높은 conviction 시그널만 알림
    high_conviction = [s for s in signals if s.conviction >= 0.8]

    if high_conviction:
        am = AlertManager()
        events = am.process_signals(high_conviction)
        logger.info(f"High conviction signals: {len(high_conviction)}")

    return {'checked': len(signals), 'high_conviction': len(high_conviction)}


def task_risk_monitor():
    """리스크 모니터링 (15분마다)"""
    from lib.paper_trader import PaperTrader
    from lib.risk_manager import RiskManager
    from lib.alert_manager import AlertManager

    logger.debug("Risk monitoring check...")

    trader = PaperTrader(account_name='default')
    summary = trader.get_portfolio_summary()

    if not summary.positions:
        return {'status': 'no_positions'}

    holdings = {t: p.market_value / summary.total_value
               for t, p in summary.positions.items()}

    rm = RiskManager()
    risk = rm.calculate_portfolio_risk(holdings, summary.total_value)

    # 높은 리스크 시 알림
    if risk.risk_level.value in ['high', 'extreme']:
        am = AlertManager()
        am.alert_risk_warning(
            risk_type='Portfolio Risk',
            current_value=risk.var_95,
            threshold=0.05,
            message=f"Risk level: {risk.risk_level.value}",
        )
        logger.warning(f"High risk detected: {risk.risk_level.value}")

    return {'risk_level': risk.risk_level.value}


# ============================================================================
# Scheduler
# ============================================================================

class EIMASScheduler:
    """EIMAS 스케줄러"""

    def __init__(self):
        self.tasks: Dict[str, SchedulerTask] = {}
        self.running = False

        # 작업 등록
        self._register_tasks()

    def _register_tasks(self):
        """작업 등록"""
        self.tasks['premarket'] = SchedulerTask(
            'premarket_analysis',
            task_premarket_analysis,
            'Pre-market analysis at 06:00 AM'
        )
        self.tasks['market_open'] = SchedulerTask(
            'market_open',
            task_market_open,
            'Market open analysis at 09:30 AM'
        )
        self.tasks['midday'] = SchedulerTask(
            'midday_check',
            task_midday_check,
            'Mid-day check at 12:00 PM'
        )
        self.tasks['market_close'] = SchedulerTask(
            'market_close',
            task_market_close,
            'Market close analysis at 04:00 PM'
        )
        self.tasks['daily_report'] = SchedulerTask(
            'daily_report',
            task_daily_report,
            'Daily report at 09:00 PM'
        )
        self.tasks['hourly_signal'] = SchedulerTask(
            'hourly_signal',
            task_hourly_signal_check,
            'Hourly signal check'
        )
        self.tasks['risk_monitor'] = SchedulerTask(
            'risk_monitor',
            task_risk_monitor,
            'Risk monitoring every 15 minutes'
        )

    def setup_schedule(self):
        """스케줄 설정"""
        if not SCHEDULE_AVAILABLE:
            logger.error("Schedule library not available")
            return

        # 주요 일정 (동부시간 기준)
        schedule.every().day.at("06:00").do(self.tasks['premarket'].run)
        schedule.every().day.at("09:30").do(self.tasks['market_open'].run)
        schedule.every().day.at("12:00").do(self.tasks['midday'].run)
        schedule.every().day.at("16:00").do(self.tasks['market_close'].run)
        schedule.every().day.at("21:00").do(self.tasks['daily_report'].run)

        # 주기적 작업
        schedule.every().hour.do(self.tasks['hourly_signal'].run)
        schedule.every(15).minutes.do(self.tasks['risk_monitor'].run)

        logger.info("Schedule configured")

    def run_task(self, task_name: str):
        """특정 작업 실행"""
        if task_name in self.tasks:
            return self.tasks[task_name].run()
        else:
            logger.error(f"Unknown task: {task_name}")
            return None

    def run_all_now(self):
        """모든 작업 즉시 실행"""
        results = {}
        for name, task in self.tasks.items():
            results[name] = task.run()
        return results

    def start(self):
        """스케줄러 시작"""
        if not SCHEDULE_AVAILABLE:
            logger.error("Schedule library not available")
            return

        self.setup_schedule()
        self.running = True

        logger.info("EIMAS Scheduler started")
        logger.info("Press Ctrl+C to stop")

        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 1분마다 체크
            except KeyboardInterrupt:
                self.stop()
                break

    def stop(self):
        """스케줄러 중지"""
        self.running = False
        logger.info("EIMAS Scheduler stopped")

    def status(self):
        """상태 조회"""
        print("\n" + "=" * 50)
        print("EIMAS Scheduler Status")
        print("=" * 50)

        for name, task in self.tasks.items():
            status = "✅" if task.last_error is None else "❌"
            last_run = task.last_run.strftime("%Y-%m-%d %H:%M") if task.last_run else "Never"
            print(f"{status} {name}: {task.description}")
            print(f"   Last run: {last_run} | Count: {task.run_count}")
            if task.last_error:
                print(f"   Error: {task.last_error}")

        if SCHEDULE_AVAILABLE:
            print("\nPending jobs:")
            for job in schedule.get_jobs():
                print(f"  - {job}")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="EIMAS Scheduler")
    parser.add_argument("--run", help="Run specific task", choices=[
        'premarket', 'market_open', 'midday', 'market_close',
        'daily_report', 'hourly_signal', 'risk_monitor', 'all'
    ])
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    scheduler = EIMASScheduler()

    if args.status:
        scheduler.status()
    elif args.run:
        if args.run == 'all':
            scheduler.run_all_now()
        else:
            scheduler.run_task(args.run)
    elif args.daemon:
        scheduler.start()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scheduler.py --run premarket   # Run pre-market analysis")
        print("  python scheduler.py --run all         # Run all tasks")
        print("  python scheduler.py --daemon          # Start scheduler daemon")
        print("  python scheduler.py --status          # Show status")


if __name__ == "__main__":
    main()
