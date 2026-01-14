"""
EIMAS Library Module
====================
경제 분석을 위한 핵심 라이브러리

Modules:
- critical_path: 리스크/불확실성 분석 (기존)
- causal_network: Granger Causality 기반 인과관계 네트워크
- lasso_model: LASSO 회귀 분석
- data_collector: 데이터 수집
- enhanced_data_sources: CME FedWatch, Enhanced FRED, 경제 캘린더
- dashboard_generator: 대시보드 생성
- etf_flow_analyzer: ETF 비중/자금흐름 기반 시장 분석
- etf_signal_generator: ETF 분석 → Signal 변환
- ark_holdings_analyzer: ARK ETF 보유종목 추적 및 분석
- market_indicators: 밸류에이션, 24/7 자산, 크레딧, VIX 지표
- unified_data_store: 통합 데이터 수집 및 DB 저장
- risk_manager: 포지션 사이징 및 리스크 관리
- alert_manager: 시그널 → 알림 통합
- report_generator: HTML 리포트 생성
- correlation_monitor: 자산 간 상관관계 모니터링
- portfolio_optimizer: Mean-Variance, Black-Litterman 최적화
- sector_rotation: 경기 사이클 기반 섹터 로테이션
- paper_trader: 페이퍼 트레이딩 시스템
- options_flow: 옵션 플로우 분석 (Put/Call, Gamma Exposure)
- sentiment_analyzer: 센티먼트 분석 (Fear & Greed, News)
- performance_attribution: Brinson 성과 분해
- factor_analyzer: Fama-French 팩터 분석
"""

from .causal_network import (
    GrangerCausalityAnalyzer,
    CausalNetworkBuilder,
    CausalNetworkAnalyzer,
    GrangerTestResult,
    CausalEdge,
    CausalPath,
    NetworkAnalysisResult,
    CausalDirection
)

from .enhanced_data_sources import (
    # Data Classes
    DataFrequency,
    FedWatchData,
    EconomicEvent,
    SentimentData,
    # Constants
    FRED_INDICATORS,
    FOMC_DATES_2025,
    # Collectors
    CMEFedWatchCollector,
    EnhancedFREDCollector,
    EconomicCalendar,
    SentimentCollector
)

from .etf_flow_analyzer import (
    ETFFlowAnalyzer,
    ETFData,
    FlowComparison,
    SectorRotationResult,
    MarketRegimeResult,
    MarketSentiment,
    StyleRotation,
    CyclePhase,
    ETF_UNIVERSE,
)

from .etf_signal_generator import (
    ETFSignalGenerator,
    run_integrated_analysis,
)

from .ark_holdings_analyzer import (
    ARKHoldingsCollector,
    ARKHoldingsAnalyzer,
    HoldingData,
    WeightChange,
    SectorSummary,
    ARKAnalysisResult,
    ARK_ETFS,
)

from .market_indicators import (
    MarketIndicatorsCollector,
    ValuationMetrics,
    CryptoMetrics,
    CreditMetrics,
    VIXMetrics,
    FXMetrics,
    MarketIndicatorsSummary,
    ValuationLevel,
    CreditCondition,
    VIXStructure,
    FearGreedLevel,
)

from .unified_data_store import (
    UnifiedDataStore,
    ETF_UNIVERSE as UNIFIED_ETF_UNIVERSE,
    CRYPTO_TICKERS as UNIFIED_CRYPTO_TICKERS,
    MAJOR_STOCKS,
)

from .fred_collector import (
    FREDCollector,
    FREDSummary,
    FREDDataPoint,
    FRED_SERIES,
)

from .notifier import (
    TelegramNotifier,
    SlackNotifier,
    EIMASNotifier,
    AlertLevel,
)

from .critical_path_monitor import (
    CriticalPathMonitor,
    CriticalPathSummary,
    PathSignal,
    PathStatus,
    SignalLevel,
    CRITICAL_PATHS,
)

from .regime_detector import (
    RegimeDetector,
    RegimeResult,
    MarketRegime,
    TrendState,
    VolatilityState,
)

from .backtester import (
    Backtester,
    BacktestResult,
    Strategy,
    Trade,
    SignalType,
    PositionType,
    create_ma_crossover_strategy,
    create_rsi_strategy,
    create_vix_regime_strategy,
    create_fear_greed_contrarian_strategy,
    # EIMAS 전략
    create_yield_curve_strategy,
    create_copper_gold_strategy,
    create_regime_based_strategy,
    create_vix_mean_reversion_strategy,
    create_multi_factor_strategy,
)

from .trading_db import (
    TradingDB,
    Signal,
    PortfolioCandidate,
    Execution,
    PerformanceRecord,
    SignalPerformance,
    SessionAnalysis,
    SignalSource,
    SignalAction,
    InvestorProfile,
    SessionType,
)

from .signal_pipeline import (
    SignalPipeline,
    PortfolioGenerator,
    INVESTOR_PROFILES,
)

from .session_analyzer import (
    SessionAnalyzer,
    DailySessionAnalysis,
    SessionReturn,
)

from .feedback_tracker import (
    FeedbackTracker,
    SignalEvaluation,
    SourceAccuracy,
)

from .leading_indicator_tester import (
    LeadingIndicatorTester,
    GrangerResult,
    LeadingIndicatorReport,
    INDICATOR_PAIRS,
)

from .risk_manager import (
    RiskManager,
    PositionSizer,
    PortfolioRisk,
    PositionLimit,
    RebalanceRecommendation,
    RiskLevel,
    RebalanceTrigger,
)

from .alert_manager import (
    AlertManager,
    AlertEvent,
    ALERT_THRESHOLDS,
)

from .report_generator import (
    ReportGenerator,
)

from .correlation_monitor import (
    CorrelationMonitor,
    PairCorrelation,
    CorrelationMatrix,
    DiversificationMetrics,
    CorrelationAlert,
    CorrelationAnalysis,
    CorrelationState,
    ASSET_UNIVERSE,
    CORRELATION_PAIRS,
    quick_correlation_check,
    detect_correlation_spike,
)

from .portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationResult,
    OptimizationConstraints,
    EfficientFrontierPoint,
    BlackLittermanResult,
    OptimizationType,
    quick_optimize,
    compare_optimizations,
)

from .sector_rotation import (
    SectorRotationModel,
    SectorStats,
    CycleDetection,
    RotationSignal,
    SectorRotationResult as SectorRotationAnalysis,
    EconomicCycle,
    SectorSignal,
    SECTOR_ETFS,
    CYCLE_SECTOR_MAP,
    quick_sector_analysis,
    get_top_sectors,
    sector_momentum_ranking,
)

from .paper_trader import (
    PaperTrader,
    Order,
    Position,
    Trade,
    PortfolioSummary,
    OrderType,
    OrderSide,
    OrderStatus,
    quick_paper_trade,
    get_paper_portfolio,
)

from .options_flow import (
    OptionsFlowMonitor,
    OptionsFlowSummary,
    OptionFlow,
    OptionType,
    FlowSignal,
)

from .sentiment_analyzer import (
    SentimentAnalyzer,
    SentimentAnalysisResult,
    FearGreedData,
    NewsSentiment,
    SocialSentiment,
    CompositeSentiment,
    SentimentLevel,
    quick_sentiment_check,
    get_contrarian_signal,
)

from .performance_attribution import (
    PerformanceAttribution,
    AttributionResult,
    BrinsonAttribution,
    AllocationEffect,
    SelectionEffect,
    InteractionEffect,
    RiskAttribution,
    PerformanceMetrics,
    quick_attribution,
    compare_to_benchmark,
)

from .factor_analyzer import (
    FactorAnalyzer,
    FactorAnalysisResult,
    FactorExposure,
    FactorAttribution,
    StyleAnalysis,
    FactorRisk,
    FACTOR_PROXIES,
    STYLE_BOX,
    quick_factor_analysis,
    get_factor_exposures,
    detect_style_drift,
)

__all__ = [
    # Causal Network Analysis
    'GrangerCausalityAnalyzer',
    'CausalNetworkBuilder',
    'CausalNetworkAnalyzer',
    'GrangerTestResult',
    'CausalEdge',
    'CausalPath',
    'NetworkAnalysisResult',
    'CausalDirection',

    # Enhanced Data Sources
    'DataFrequency',
    'FedWatchData',
    'EconomicEvent',
    'SentimentData',
    'FRED_INDICATORS',
    'FOMC_DATES_2025',
    'CMEFedWatchCollector',
    'EnhancedFREDCollector',
    'EconomicCalendar',
    'SentimentCollector',

    # ETF Flow Analysis
    'ETFFlowAnalyzer',
    'ETFData',
    'FlowComparison',
    'SectorRotationResult',
    'MarketRegimeResult',
    'MarketSentiment',
    'StyleRotation',
    'CyclePhase',
    'ETF_UNIVERSE',
    'ETFSignalGenerator',
    'run_integrated_analysis',

    # ARK Holdings Analysis
    'ARKHoldingsCollector',
    'ARKHoldingsAnalyzer',
    'HoldingData',
    'WeightChange',
    'SectorSummary',
    'ARKAnalysisResult',
    'ARK_ETFS',

    # Market Indicators
    'MarketIndicatorsCollector',
    'ValuationMetrics',
    'CryptoMetrics',
    'CreditMetrics',
    'VIXMetrics',
    'FXMetrics',
    'MarketIndicatorsSummary',
    'ValuationLevel',
    'CreditCondition',
    'VIXStructure',
    'FearGreedLevel',

    # Unified Data Store
    'UnifiedDataStore',
    'UNIFIED_ETF_UNIVERSE',
    'UNIFIED_CRYPTO_TICKERS',
    'MAJOR_STOCKS',

    # FRED Collector
    'FREDCollector',
    'FREDSummary',
    'FREDDataPoint',
    'FRED_SERIES',

    # Notifier
    'TelegramNotifier',
    'SlackNotifier',
    'EIMASNotifier',
    'AlertLevel',

    # Critical Path Monitor
    'CriticalPathMonitor',
    'CriticalPathSummary',
    'PathSignal',
    'PathStatus',
    'SignalLevel',
    'CRITICAL_PATHS',

    # Regime Detector
    'RegimeDetector',
    'RegimeResult',
    'MarketRegime',
    'TrendState',
    'VolatilityState',

    # Backtester
    'Backtester',
    'BacktestResult',
    'Strategy',
    'Trade',
    'SignalType',
    'PositionType',
    'create_ma_crossover_strategy',
    'create_rsi_strategy',
    'create_vix_regime_strategy',
    'create_fear_greed_contrarian_strategy',
    # EIMAS Strategies
    'create_yield_curve_strategy',
    'create_copper_gold_strategy',
    'create_regime_based_strategy',
    'create_vix_mean_reversion_strategy',
    'create_multi_factor_strategy',

    # Trading DB
    'TradingDB',
    'Signal',
    'PortfolioCandidate',
    'Execution',
    'PerformanceRecord',
    'SignalPerformance',
    'SessionAnalysis',
    'SignalSource',
    'SignalAction',
    'InvestorProfile',
    'SessionType',

    # Signal Pipeline
    'SignalPipeline',
    'PortfolioGenerator',
    'INVESTOR_PROFILES',

    # Session Analyzer
    'SessionAnalyzer',
    'DailySessionAnalysis',
    'SessionReturn',

    # Feedback Tracker
    'FeedbackTracker',
    'SignalEvaluation',
    'SourceAccuracy',

    # Leading Indicator Tester
    'LeadingIndicatorTester',
    'GrangerResult',
    'LeadingIndicatorReport',
    'INDICATOR_PAIRS',

    # Risk Manager
    'RiskManager',
    'PositionSizer',
    'PortfolioRisk',
    'PositionLimit',
    'RebalanceRecommendation',
    'RiskLevel',
    'RebalanceTrigger',

    # Alert Manager
    'AlertManager',
    'AlertEvent',
    'ALERT_THRESHOLDS',

    # Report Generator
    'ReportGenerator',

    # Correlation Monitor
    'CorrelationMonitor',
    'PairCorrelation',
    'CorrelationMatrix',
    'DiversificationMetrics',
    'CorrelationAlert',
    'CorrelationAnalysis',
    'CorrelationState',
    'ASSET_UNIVERSE',
    'CORRELATION_PAIRS',
    'quick_correlation_check',
    'detect_correlation_spike',

    # Portfolio Optimizer
    'PortfolioOptimizer',
    'OptimizationResult',
    'OptimizationConstraints',
    'EfficientFrontierPoint',
    'BlackLittermanResult',
    'OptimizationType',
    'quick_optimize',
    'compare_optimizations',

    # Sector Rotation
    'SectorRotationModel',
    'SectorStats',
    'CycleDetection',
    'RotationSignal',
    'SectorRotationAnalysis',
    'EconomicCycle',
    'SectorSignal',
    'SECTOR_ETFS',
    'CYCLE_SECTOR_MAP',
    'quick_sector_analysis',
    'get_top_sectors',
    'sector_momentum_ranking',

    # Paper Trader
    'PaperTrader',
    'Order',
    'Position',
    'Trade',
    'PortfolioSummary',
    'OrderType',
    'OrderSide',
    'OrderStatus',
    'quick_paper_trade',
    'get_paper_portfolio',

    # Options Flow
    'OptionsFlowMonitor',
    'OptionsFlowSummary',
    'OptionFlow',
    'OptionType',
    'FlowSignal',

    # Sentiment Analyzer
    'SentimentAnalyzer',
    'SentimentAnalysisResult',
    'FearGreedData',
    'NewsSentiment',
    'SocialSentiment',
    'CompositeSentiment',
    'SentimentLevel',
    'quick_sentiment_check',
    'get_contrarian_signal',

    # Performance Attribution
    'PerformanceAttribution',
    'AttributionResult',
    'BrinsonAttribution',
    'AllocationEffect',
    'SelectionEffect',
    'InteractionEffect',
    'RiskAttribution',
    'PerformanceMetrics',
    'quick_attribution',
    'compare_to_benchmark',

    # Factor Analyzer
    'FactorAnalyzer',
    'FactorAnalysisResult',
    'FactorExposure',
    'FactorAttribution',
    'StyleAnalysis',
    'FactorRisk',
    'FACTOR_PROXIES',
    'STYLE_BOX',
    'quick_factor_analysis',
    'get_factor_exposures',
    'detect_style_drift',
]
