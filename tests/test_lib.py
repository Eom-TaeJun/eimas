#!/usr/bin/env python3
"""
EIMAS Library Tests
====================
주요 모듈 단위 테스트

Usage:
    python -m pytest tests/test_lib.py -v
    python tests/test_lib.py  # 직접 실행
"""

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


# ============================================================================
# Test Cases
# ============================================================================

class TestPortfolioOptimizer(unittest.TestCase):
    """포트폴리오 최적화 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.portfolio = {
            'SPY': 0.4,
            'QQQ': 0.3,
            'TLT': 0.2,
            'GLD': 0.1,
        }

    def test_import(self):
        """임포트 테스트"""
        from lib.portfolio_optimizer import PortfolioOptimizer
        optimizer = PortfolioOptimizer(list(self.portfolio.keys()))
        self.assertIsNotNone(optimizer)

    def test_optimization_types(self):
        """최적화 타입 테스트"""
        from lib.portfolio_optimizer import OptimizationType
        # 실제 구현된 enum 값 확인
        types = [t.value for t in OptimizationType]
        self.assertTrue(len(types) > 0)


class TestSectorRotation(unittest.TestCase):
    """섹터 로테이션 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from lib.sector_rotation import SectorRotationModel, SECTOR_ETFS
        self.assertIn('XLK', SECTOR_ETFS)
        self.assertIn('XLF', SECTOR_ETFS)

    def test_economic_cycles(self):
        """경기 사이클 테스트"""
        from lib.sector_rotation import EconomicCycle
        cycles = [c.value for c in EconomicCycle]
        self.assertIn('early_recovery', cycles)
        self.assertIn('recession', cycles)


class TestOptionsFlow(unittest.TestCase):
    """옵션 플로우 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from lib.options_flow import OptionsFlowAnalyzer, OptionsSentiment
        analyzer = OptionsFlowAnalyzer('AAPL')
        self.assertEqual(analyzer.ticker, 'AAPL')

    def test_sentiment_levels(self):
        """센티먼트 레벨 테스트"""
        from lib.options_flow import OptionsSentiment
        self.assertEqual(OptionsSentiment.BULLISH.value, 'bullish')
        self.assertEqual(OptionsSentiment.BEARISH.value, 'bearish')


class TestSentimentAnalyzer(unittest.TestCase):
    """센티먼트 분석 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from lib.sentiment_analyzer import SentimentAnalyzer, SentimentLevel
        analyzer = SentimentAnalyzer(['SPY'])
        self.assertEqual(analyzer.tickers, ['SPY'])

    def test_sentiment_levels(self):
        """센티먼트 레벨 테스트"""
        from lib.sentiment_analyzer import SentimentLevel
        levels = [l.value for l in SentimentLevel]
        self.assertIn('extreme_fear', levels)
        self.assertIn('extreme_greed', levels)


class TestPerformanceAttribution(unittest.TestCase):
    """성과 분해 테스트"""

    def setUp(self):
        self.portfolio = {
            'SPY': 0.5,
            'QQQ': 0.3,
            'TLT': 0.2,
        }

    def test_import(self):
        """임포트 테스트"""
        from lib.performance_attribution import PerformanceAttribution
        attr = PerformanceAttribution(self.portfolio)
        self.assertEqual(attr.benchmark, 'SPY')

    def test_brinson_structure(self):
        """Brinson 분해 구조 테스트"""
        from lib.performance_attribution import BrinsonAttribution, AllocationEffect
        # 데이터 클래스 필드 확인
        brinson_fields = [f.name for f in BrinsonAttribution.__dataclass_fields__.values()]
        alloc_fields = [f.name for f in AllocationEffect.__dataclass_fields__.values()]
        self.assertIn('allocation_effect', brinson_fields)
        self.assertIn('effect', alloc_fields)


class TestFactorAnalyzer(unittest.TestCase):
    """팩터 분석 테스트"""

    def setUp(self):
        self.portfolio = {
            'SPY': 0.4,
            'QQQ': 0.3,
            'IWM': 0.3,
        }

    def test_import(self):
        """임포트 테스트"""
        from lib.factor_analyzer import FactorAnalyzer, FACTOR_PROXIES
        analyzer = FactorAnalyzer(self.portfolio)
        self.assertEqual(analyzer.period, '1y')

    def test_factor_proxies(self):
        """팩터 프록시 테스트"""
        from lib.factor_analyzer import FACTOR_PROXIES
        self.assertEqual(FACTOR_PROXIES['market'], 'SPY')
        self.assertEqual(FACTOR_PROXIES['momentum'], 'MTUM')


class TestPaperTrader(unittest.TestCase):
    """페이퍼 트레이더 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from lib.paper_trader import PaperTrader, OrderType, OrderSide
        trader = PaperTrader(initial_capital=100000)
        self.assertEqual(trader.initial_capital, 100000)

    def test_order_types(self):
        """주문 타입 테스트"""
        from lib.paper_trader import OrderType, OrderSide
        self.assertEqual(OrderType.MARKET.value, 'market')
        self.assertEqual(OrderSide.BUY.value, 'buy')


class TestRiskManager(unittest.TestCase):
    """리스크 관리 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from lib.risk_manager import RiskManager, RiskLevel
        manager = RiskManager()
        self.assertIsNotNone(manager)

    def test_risk_levels(self):
        """리스크 레벨 테스트"""
        from lib.risk_manager import RiskLevel
        levels = [l.value for l in RiskLevel]
        self.assertIn('low', levels)
        self.assertIn('high', levels)


class TestCorrelationMonitor(unittest.TestCase):
    """상관관계 모니터 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from lib.correlation_monitor import CorrelationMonitor, ASSET_UNIVERSE
        monitor = CorrelationMonitor()
        self.assertIn('SPY', ASSET_UNIVERSE)

    def test_correlation_states(self):
        """상관관계 상태 테스트"""
        from lib.correlation_monitor import CorrelationState
        states = [s.value for s in CorrelationState]
        # 실제 구현된 상태값 확인
        self.assertTrue(len(states) > 0)


class TestRegimeDetector(unittest.TestCase):
    """레짐 감지 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from lib.regime_detector import RegimeDetector, MarketRegime
        detector = RegimeDetector(ticker='SPY')
        self.assertEqual(detector.ticker, 'SPY')

    def test_market_regimes(self):
        """시장 레짐 테스트"""
        from lib.regime_detector import MarketRegime
        regimes = [r.value for r in MarketRegime]
        # 실제 구현된 레짐 확인
        self.assertTrue(len(regimes) > 0)


class TestBacktester(unittest.TestCase):
    """백테스터 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from lib.backtester import Backtester, Strategy
        backtester = Backtester('SPY')
        self.assertIsNotNone(backtester)

    def test_signal_types(self):
        """시그널 타입 테스트"""
        from lib.backtester import SignalType
        types = [t.value for t in SignalType]
        self.assertIn('buy', types)
        self.assertIn('sell', types)


class TestDataPipeline(unittest.TestCase):
    """데이터 파이프라인 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from data.pipeline import DataPipeline, DataQuality
        pipeline = DataPipeline(tickers=['SPY'], period='1mo')
        self.assertEqual(pipeline.period, '1mo')

    def test_data_quality_levels(self):
        """데이터 품질 레벨 테스트"""
        from data.pipeline import DataQuality
        qualities = [q.value for q in DataQuality]
        self.assertIn('excellent', qualities)
        self.assertIn('poor', qualities)


class TestCacheSystem(unittest.TestCase):
    """캐시 시스템 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from data.cache import CacheManager, LRUCache
        cache = LRUCache(max_size=10)
        self.assertEqual(cache.max_size, 10)

    def test_lru_cache_operations(self):
        """LRU 캐시 연산 테스트"""
        from data.cache import LRUCache
        cache = LRUCache(max_size=3)

        cache.set('a', 1)
        cache.set('b', 2)
        cache.set('c', 3)

        self.assertEqual(cache.get('a'), 1)
        self.assertEqual(cache.size(), 3)

        # 용량 초과 테스트
        cache.set('d', 4)
        self.assertEqual(cache.size(), 3)

    def test_cache_ttl(self):
        """캐시 TTL 테스트"""
        from data.cache import LRUCache
        import time

        cache = LRUCache()
        cache.set('temp', 'value', ttl=1)

        self.assertEqual(cache.get('temp'), 'value')
        time.sleep(1.5)
        self.assertIsNone(cache.get('temp'))


class TestLoggingConfig(unittest.TestCase):
    """로깅 설정 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from core.logging_config import setup_logging, get_logger
        logger = get_logger(__name__)
        self.assertIsNotNone(logger)

    def test_log_levels(self):
        """로그 레벨 테스트"""
        from core.logging_config import LogLevel
        levels = [l.value for l in LogLevel]
        self.assertIn('DEBUG', levels)
        self.assertIn('ERROR', levels)


class TestHealthCheck(unittest.TestCase):
    """헬스체크 테스트"""

    def test_import(self):
        """임포트 테스트"""
        from core.health_check import HealthChecker, HealthStatus
        checker = HealthChecker()
        self.assertIsNotNone(checker)

    def test_health_statuses(self):
        """상태 테스트"""
        from core.health_check import HealthStatus
        statuses = [s.value for s in HealthStatus]
        self.assertIn('healthy', statuses)
        self.assertIn('unhealthy', statuses)

    def test_system_metrics(self):
        """시스템 메트릭 테스트"""
        from core.health_check import get_metrics
        metrics = get_metrics()
        self.assertGreaterEqual(metrics.cpu_percent, 0)
        self.assertGreaterEqual(metrics.memory_percent, 0)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration(unittest.TestCase):
    """통합 테스트"""

    def test_lib_imports(self):
        """lib 전체 임포트 테스트"""
        from lib import (
            # Portfolio
            PortfolioOptimizer,
            SectorRotationModel,
            PaperTrader,
            PerformanceAttribution,

            # Analysis
            OptionsFlowAnalyzer,
            SentimentAnalyzer,
            FactorAnalyzer,
            CorrelationMonitor,

            # Risk
            RiskManager,
            RegimeDetector,
            Backtester,
        )

        self.assertTrue(True)  # 임포트 성공

    def test_data_imports(self):
        """data 모듈 임포트 테스트"""
        from data.pipeline import DataPipeline
        from data.cache import CacheManager

        self.assertTrue(True)

    def test_core_imports(self):
        """core 모듈 임포트 테스트"""
        from core.logging_config import get_logger
        from core.health_check import check_system_health

        self.assertTrue(True)


# ============================================================================
# Run Tests
# ============================================================================

def run_tests():
    """테스트 실행"""
    print("\n" + "="*60)
    print("EIMAS Library Tests")
    print("="*60 + "\n")

    # 테스트 로더
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 테스트 케이스 추가
    test_classes = [
        TestPortfolioOptimizer,
        TestSectorRotation,
        TestOptionsFlow,
        TestSentimentAnalyzer,
        TestPerformanceAttribution,
        TestFactorAnalyzer,
        TestPaperTrader,
        TestRiskManager,
        TestCorrelationMonitor,
        TestRegimeDetector,
        TestBacktester,
        TestDataPipeline,
        TestCacheSystem,
        TestLoggingConfig,
        TestHealthCheck,
        TestIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 결과 요약
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    success = result.wasSuccessful()
    print(f"\nStatus: {'PASSED' if success else 'FAILED'}")

    return success


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
