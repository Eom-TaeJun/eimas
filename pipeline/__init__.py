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
    generate_explanation,
    analyze_genius_act,
    analyze_theme_etf,
    analyze_shock_propagation,
    optimize_portfolio_mst,
    analyze_volume_anomalies,
    track_events_with_news,
    run_adaptive_portfolio,
    # NEW: 2026-01-29 통합
    analyze_bubble_risk,
    analyze_sentiment,
    run_ai_validation
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
    save_result_md,
    save_to_trading_db,
    save_to_event_db
)
from .report import (
    generate_ai_report,
    run_whitening_check,
    run_fact_check
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
    BubbleRiskMetrics,
    GeniusActResult,
    ThemeETFResult,
    ShockAnalysisResult,
    PortfolioResult
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
    'analyze_genius_act',
    'analyze_theme_etf',
    'analyze_shock_propagation',
    'optimize_portfolio_mst',
    'analyze_volume_anomalies',
    'track_events_with_news',
    'run_adaptive_portfolio',
    # NEW: 2026-01-29
    'analyze_bubble_risk',
    'analyze_sentiment',
    'run_ai_validation',
    
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
    'run_whitening_check',
    'run_fact_check',
    
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
    'BubbleRiskMetrics',
    'GeniusActResult',
    'ThemeETFResult',
    'ShockAnalysisResult',
    'PortfolioResult'
]