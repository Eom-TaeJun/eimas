"""
EIMAS Pipeline Package
======================
Lazy export surface for pipeline modules.

This keeps compatibility for `from pipeline import ...` while avoiding
heavy eager imports at package import time.
"""

from importlib import import_module

_EXPORT_MAP = {
    # Collectors
    "collect_fred_data": ("pipeline.collectors", "collect_fred_data"),
    "collect_market_data": ("pipeline.collectors", "collect_market_data"),
    "collect_crypto_data": ("pipeline.collectors", "collect_crypto_data"),
    "collect_market_indicators": ("pipeline.collectors", "collect_market_indicators"),
    # Analyzers
    "detect_regime": ("pipeline.analyzers", "detect_regime"),
    "detect_events": ("pipeline.analyzers", "detect_events"),
    "analyze_liquidity": ("pipeline.analyzers", "analyze_liquidity"),
    "analyze_critical_path": ("pipeline.analyzers", "analyze_critical_path"),
    "analyze_etf_flow": ("pipeline.analyzers", "analyze_etf_flow"),
    "generate_explanation": ("pipeline.analyzers", "generate_explanation"),
    "analyze_genius_act": ("pipeline.analyzers", "analyze_genius_act"),
    "analyze_theme_etf": ("pipeline.analyzers", "analyze_theme_etf"),
    "analyze_shock_propagation": ("pipeline.analyzers", "analyze_shock_propagation"),
    "optimize_portfolio_mst": ("pipeline.analyzers", "optimize_portfolio_mst"),
    "analyze_volume_anomalies": ("pipeline.analyzers", "analyze_volume_anomalies"),
    "track_events_with_news": ("pipeline.analyzers", "track_events_with_news"),
    "run_adaptive_portfolio": ("pipeline.analyzers", "run_adaptive_portfolio"),
    "analyze_bubble_risk": ("pipeline.analyzers", "analyze_bubble_risk"),
    "analyze_sentiment": ("pipeline.analyzers", "analyze_sentiment"),
    "run_ai_validation": ("pipeline.analyzers", "run_ai_validation"),
    "run_allocation_engine": ("pipeline.analyzers", "run_allocation_engine"),
    "run_rebalancing_policy": ("pipeline.analyzers", "run_rebalancing_policy"),
    # Debate
    "run_dual_mode_debate": ("pipeline.debate", "run_dual_mode_debate"),
    "extract_consensus": ("pipeline.debate", "extract_consensus"),
    # Realtime
    "run_realtime_stream": ("pipeline.realtime", "run_realtime_stream"),
    # Storage
    "save_result_json": ("pipeline.storage", "save_result_json"),
    "save_result_md": ("pipeline.storage", "save_result_md"),
    "save_to_trading_db": ("pipeline.storage", "save_to_trading_db"),
    "save_to_event_db": ("pipeline.storage", "save_to_event_db"),
    # Report
    "generate_ai_report": ("pipeline.report", "generate_ai_report"),
    "run_whitening_check": ("pipeline.report", "run_whitening_check"),
    "run_fact_check": ("pipeline.report", "run_fact_check"),
    # Schemas
    "EIMASResult": ("pipeline.schemas", "EIMASResult"),
    "FREDSummary": ("pipeline.schemas", "FREDSummary"),
    "IndicatorsSummary": ("pipeline.schemas", "IndicatorsSummary"),
    "RegimeResult": ("pipeline.schemas", "RegimeResult"),
    "Event": ("pipeline.schemas", "Event"),
    "LiquiditySignal": ("pipeline.schemas", "LiquiditySignal"),
    "CriticalPathResult": ("pipeline.schemas", "CriticalPathResult"),
    "ETFFlowResult": ("pipeline.schemas", "ETFFlowResult"),
    "DebateResult": ("pipeline.schemas", "DebateResult"),
    "RealtimeSignal": ("pipeline.schemas", "RealtimeSignal"),
    "AIReport": ("pipeline.schemas", "AIReport"),
    "MarketQualityMetrics": ("pipeline.schemas", "MarketQualityMetrics"),
    "BubbleRiskMetrics": ("pipeline.schemas", "BubbleRiskMetrics"),
    "GeniusActResult": ("pipeline.schemas", "GeniusActResult"),
    "ThemeETFResult": ("pipeline.schemas", "ThemeETFResult"),
    "ShockAnalysisResult": ("pipeline.schemas", "ShockAnalysisResult"),
    "PortfolioResult": ("pipeline.schemas", "PortfolioResult"),
}

__all__ = list(_EXPORT_MAP.keys())


def __getattr__(name: str):
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'pipeline' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
