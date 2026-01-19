"""
EIMAS Pipeline Package
======================
모듈화된 분석 파이프라인 구성 요소
"""

from .collectors import (
    collect_fred_data,
    collect_market_data,
    collect_crypto_data,
    collect_market_indicators
)
from .analyzers import (
    detect_regime,
    detect_events,
    analyze_liquidity,
    analyze_critical_path,
    analyze_etf_flow,
    generate_explanation
)
from .debate import (
    run_dual_mode_debate,
    extract_consensus
)
from .realtime import (
    run_realtime_stream
)
from .storage import (
    save_result_json,
    save_to_trading_db,
    save_to_event_db
)
from .report import (
    generate_ai_report
)
from .schemas import (
    EIMASResult,
    FREDSummary,
    IndicatorsSummary,
    RegimeResult,
    Event,
    LiquiditySignal,
    CriticalPathResult,
    ETFFlowResult,
    DebateResult,
    RealtimeSignal,
    AIReport,
    MarketQualityMetrics,
    BubbleRiskMetrics
)

__all__ = [
    # Collectors
    'collect_fred_data',
    'collect_market_data',
    'collect_crypto_data',
    'collect_market_indicators',
    
    # Analyzers
    'detect_regime',
    'detect_events',
    'analyze_liquidity',
    'analyze_critical_path',
    'analyze_etf_flow',
    'generate_explanation',
    
    # Debate
    'run_dual_mode_debate',
    'extract_consensus',
    
    # Realtime
    'run_realtime_stream',
    
    # Storage
    'save_result_json',
    'save_to_trading_db',
    'save_to_event_db',
    
    # Report
    'generate_ai_report',
    
    # Schemas
    'EIMASResult',
    'FREDSummary',
    'IndicatorsSummary',
    'RegimeResult',
    'Event',
    'LiquiditySignal',
    'CriticalPathResult',
    'ETFFlowResult',
    'DebateResult',
    'RealtimeSignal',
    'AIReport',
    'MarketQualityMetrics',
    'BubbleRiskMetrics'
]