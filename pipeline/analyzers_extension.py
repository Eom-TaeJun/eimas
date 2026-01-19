# ============================================================================
# Missing Advanced Analyzers (Restored)
# ============================================================================

from lib.volume_analyzer import VolumeAnalyzer
from lib.event_tracker import EventTracker
from lib.adaptive_agents import AdaptivePortfolioManager, MarketCondition

def analyze_volume_anomalies(market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """거래량 이상 징후 탐지"""
    print("\n[2.11] Volume anomaly detection...")
    try:
        analyzer = VolumeAnalyzer()
        anomalies = analyzer.detect_anomalies(market_data)
        print(f"      ✓ Anomalies: {len(anomalies)} detected")
        return anomalies
    except Exception as e:
        log_error(logger, "Volume anomaly detection failed", e)
        return []

async def track_events_with_news(market_data: Dict[str, pd.DataFrame]) -> Dict:
    """이상 징후 발생 시 뉴스 검색 (Event Tracking)"""
    print("\n[2.12] Event tracking (anomaly -> news)...")
    try:
        tracker = EventTracker()
        # 최근 데이터 기준 이상 감지 및 뉴스 검색
        # 주의: API 호출 비용 발생 가능
        results = await tracker.detect_and_track(market_data)
        print(f"      ✓ Tracked Events: {len(results.get('events', []))}")
        return results
    except Exception as e:
        log_error(logger, "Event tracking failed", e)
        return {}

def run_adaptive_portfolio(regime_result: RegimeResult) -> Dict:
    """적응형 포트폴리오 전략 수립"""
    print("\n[2.13] Adaptive portfolio agents...")
    try:
        manager = AdaptivePortfolioManager()
        
        # RegimeResult -> MarketCondition 변환
        condition = MarketCondition(
            timestamp=regime_result.timestamp,
            regime_confidence=regime_result.confidence * 100, # 0-100 scale
            trend=regime_result.trend,
            volatility=regime_result.volatility
        )
        
        allocation = manager.propose_allocation(condition)
        print(f"      ✓ Adaptive Allocation: {allocation}")
        return allocation
    except Exception as e:
        log_error(logger, "Adaptive portfolio analysis failed", e)
        return {}

