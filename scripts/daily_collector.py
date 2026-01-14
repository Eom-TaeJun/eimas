#!/usr/bin/env python3
"""
EIMAS Daily Data Collector
==========================
매일 장 마감 후 실행되는 자동 데이터 수집 스크립트

수집 대상:
1. ETF/주식 가격 (UnifiedDataStore)
2. ARK Holdings (ARKHoldingsAnalyzer)
3. 시장 지표 (MarketIndicatorsCollector)
4. FRED 거시지표 (FREDCollector)

사용법:
    # 직접 실행
    python scripts/daily_collector.py

    # cron 설정 (매일 오후 5시, 미국 동부시간)
    0 17 * * 1-5 cd /home/tj/projects/autoai/eimas && python scripts/daily_collector.py >> logs/daily.log 2>&1

    # systemd timer 설정
    systemctl --user enable eimas-daily.timer
"""

import sys
import os

# 프로젝트 루트 추가
sys.path.insert(0, '/home/tj/projects/autoai/eimas')
os.chdir('/home/tj/projects/autoai/eimas')

import argparse
from datetime import datetime
import traceback
from typing import Dict, Any, List

from core.database import DatabaseManager
from lib.unified_data_store import UnifiedDataStore
from lib.market_indicators import MarketIndicatorsCollector
from lib.fred_collector import FREDCollector
from lib.ark_holdings_analyzer import ARKHoldingsCollector, ARKHoldingsAnalyzer
from lib.notifier import EIMASNotifier, AlertLevel


class DailyCollector:
    """
    일일 데이터 수집 관리자

    수집 순서:
    1. ETF/주식 가격 데이터
    2. 시장 지표 (VIX, Credit, FX)
    3. FRED 거시지표
    4. ARK Holdings
    5. 알림 발송
    """

    def __init__(self, notify: bool = True, verbose: bool = True):
        self.db = DatabaseManager()
        self.notifier = EIMASNotifier() if notify else None
        self.verbose = verbose
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def log(self, message: str, level: str = "INFO"):
        """로그 출력"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")

    def collect_prices(self) -> bool:
        """ETF/주식 가격 수집"""
        self.log("Collecting prices (ETF, stocks, crypto)...")

        try:
            store = UnifiedDataStore()
            stats = store.collect_and_store_all(include_composition=False)

            self.results['prices'] = stats
            self.log(f"  Prices: {stats.get('daily_prices', 0)} records")
            self.log(f"  Performance: {stats.get('etf_performance', 0)} records")
            self.log(f"  Crypto: {stats.get('crypto', 0)} records")
            return True

        except Exception as e:
            self.errors.append(f"Price collection failed: {e}")
            self.log(f"Error: {e}", "ERROR")
            return False

    def collect_indicators(self) -> bool:
        """시장 지표 수집"""
        self.log("Collecting market indicators...")

        try:
            collector = MarketIndicatorsCollector()
            summary = collector.collect_all()
            collector.save_to_db(summary, self.db)

            # 결과 저장
            self.results['indicators'] = {
                'vix': summary.vix.vix,
                'vix_regime': summary.vix.regime,
                'fear_greed': summary.crypto.fear_greed_value,
                'fear_greed_label': summary.crypto.fear_greed_label,
                'risk_score': summary.risk_score,
                'opportunity_score': summary.opportunity_score,
            }

            # 경고 수집
            self.warnings.extend(summary.warnings)

            self.log(f"  VIX: {summary.vix.vix:.1f} ({summary.vix.regime})")
            self.log(f"  Fear & Greed: {summary.crypto.fear_greed_value} ({summary.crypto.fear_greed_label})")
            return True

        except Exception as e:
            self.errors.append(f"Indicator collection failed: {e}")
            self.log(f"Error: {e}", "ERROR")
            return False

    def collect_fred(self) -> bool:
        """FRED 거시지표 수집"""
        self.log("Collecting FRED data...")

        try:
            collector = FREDCollector()
            summary = collector.collect_all()
            collector.save_to_db(summary, self.db)

            self.results['fred'] = {
                'fed_funds': summary.fed_funds,
                'treasury_10y': summary.treasury_10y,
                'spread_10y2y': summary.spread_10y2y,
                'curve_status': summary.curve_status,
            }

            # 경고 수집
            self.warnings.extend(summary.warnings)

            self.log(f"  Fed Funds: {summary.fed_funds:.2f}%")
            self.log(f"  10Y-2Y: {summary.spread_10y2y:.2f}% ({summary.curve_status})")
            return True

        except Exception as e:
            self.errors.append(f"FRED collection failed: {e}")
            self.log(f"Error: {e}", "ERROR")
            return False

    def collect_ark(self) -> bool:
        """ARK Holdings 수집"""
        self.log("Collecting ARK Holdings...")

        try:
            collector = ARKHoldingsCollector()
            analyzer = ARKHoldingsAnalyzer(collector)
            result = analyzer.run_analysis()
            stats = analyzer.save_to_db(result, db=self.db)

            self.results['ark'] = {
                'holdings': stats.get('holdings', 0),
                'signals': stats.get('signals', 0),
            }

            self.log(f"  Holdings: {stats.get('holdings', 0)} records")
            return True

        except Exception as e:
            self.errors.append(f"ARK collection failed: {e}")
            self.log(f"Error: {e}", "ERROR")
            return False

    def run(self) -> Dict[str, Any]:
        """전체 수집 실행"""
        start_time = datetime.now()
        self.log("=" * 60)
        self.log("EIMAS Daily Collector Started")
        self.log(f"Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 60)

        # 수집 실행
        success = {
            'prices': self.collect_prices(),
            'indicators': self.collect_indicators(),
            'fred': self.collect_fred(),
            'ark': self.collect_ark(),
        }

        # 완료 시간
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 요약
        self.log("=" * 60)
        self.log("Collection Summary")
        self.log("=" * 60)

        for task, ok in success.items():
            status = "OK" if ok else "FAILED"
            self.log(f"  {task}: {status}")

        self.log(f"\nDuration: {duration:.1f} seconds")

        if self.errors:
            self.log(f"\nErrors ({len(self.errors)}):", "ERROR")
            for err in self.errors:
                self.log(f"  - {err}", "ERROR")

        if self.warnings:
            self.log(f"\nWarnings ({len(self.warnings)}):", "WARNING")
            for warn in self.warnings[:10]:  # 최대 10개
                self.log(f"  - {warn}", "WARNING")

        # 알림 발송
        if self.notifier and self.notifier.telegram.is_configured():
            self._send_notification(success, duration)

        # 결과 반환
        return {
            'success': all(success.values()),
            'tasks': success,
            'results': self.results,
            'errors': self.errors,
            'warnings': self.warnings,
            'duration': duration,
        }

    def _send_notification(self, success: Dict[str, bool], duration: float):
        """알림 발송"""
        # 모두 성공
        if all(success.values()):
            # 요약 메시지
            indicators = self.results.get('indicators', {})
            fred = self.results.get('fred', {})

            data = {
                'spy_close': 0,  # TODO: prices에서 가져오기
                'spy_change': 0,
                'vix': indicators.get('vix', 0),
                'btc_price': 0,  # TODO: prices에서 가져오기
                'fear_greed_value': indicators.get('fear_greed', 50),
                'fear_greed_label': indicators.get('fear_greed_label', 'Neutral'),
                'signals': [],
                'warnings': self.warnings[:5],
            }

            self.notifier.telegram.send_market_summary(data)

        else:
            # 에러 알림
            failed = [k for k, v in success.items() if not v]
            self.notifier.telegram.send_alert(
                title="EIMAS Collection Failed",
                message=f"Failed tasks: {', '.join(failed)}\n\nErrors:\n" +
                        "\n".join(self.errors[:3]),
                level=AlertLevel.WARNING
            )


def main():
    parser = argparse.ArgumentParser(description='EIMAS Daily Data Collector')
    parser.add_argument('--no-notify', action='store_true',
                        help='Disable notifications')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode (less output)')
    parser.add_argument('--task', choices=['prices', 'indicators', 'fred', 'ark', 'all'],
                        default='all', help='Specific task to run')

    args = parser.parse_args()

    collector = DailyCollector(
        notify=not args.no_notify,
        verbose=not args.quiet
    )

    if args.task == 'all':
        result = collector.run()
    else:
        # 개별 태스크 실행
        task_methods = {
            'prices': collector.collect_prices,
            'indicators': collector.collect_indicators,
            'fred': collector.collect_fred,
            'ark': collector.collect_ark,
        }
        success = task_methods[args.task]()
        result = {'success': success, 'task': args.task}

    # 종료 코드
    sys.exit(0 if result.get('success', False) else 1)


if __name__ == "__main__":
    main()
