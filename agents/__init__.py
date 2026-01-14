"""
Multi-Agent System - Agents Module
===================================
Economic Intelligence Multi-Agent System (EIMAS) 에이전트 모듈

Agents:
- BaseAgent: 모든 에이전트의 기본 클래스
- AnalysisAgent: 데이터 분석 에이전트
- ForecastAgent: 예측 에이전트
- ResearchAgent: Perplexity 기반 연구 자료 수집
- StrategyAgent: Critical Path 기반 매매 전략 제안
- MethodologyDebateAgent: Multi-AI 방법론 토론
- MetaOrchestrator: 에이전트 조율

Modules:
- regime_change: 구조 변화 탐지 파이프라인
- methodology_debate: 방법론 토론 모듈
"""

from .base_agent import BaseAgent, AgentConfig
from .analysis_agent import AnalysisAgent
from .forecast_agent import ForecastAgent
from .orchestrator import MetaOrchestrator
from .research_agent import ResearchAgent, ResearchReport, ResearchCategory
from .verification_agent import (
    VerificationAgent,
    VerificationResult,
    HallucinationCheck,
    SycophancyCheck
)
from .strategy_agent import (
    StrategyAgent,
    PortfolioStrategy,
    TradeRecommendation,
    MarketState,
    CriticalPathState,
    create_market_state_from_data,
    create_critical_path_state
)
from .regime_change import (
    RegimeChangeDetectionPipeline,
    VolumeBreakoutDetector,
    NewsSearchAgent,
    NewsClassificationAgent,
    ImpactAssessmentDebate,
    RegimeChangeDecision,
    VolumeEvent,
    NewsClassification,
    RegimeChangeResult
)
from .methodology_debate import (
    MethodologyDebateAgent,
    MethodologyDebate,
    MethodologyProposal,
    MethodologyDecision,
    MethodologyType,
    ResearchGoal,
    DataSummary,
    METHODOLOGY_KNOWLEDGE,
    GOAL_METHOD_MAPPING
)
from .interpretation_debate import (
    InterpretationDebateAgent,
    InterpretationDebate,
    InterpretationConsensus,
    SchoolInterpretation,
    AnalysisResult,
    EconomicSchool,
    SCHOOL_SYSTEM_PROMPTS,
    create_school_participants
)
from .top_down_orchestrator import (
    TopDownOrchestrator,
    TopDownResult,
    AnalysisLevel,
    RiskLevel,
    Stance,
    PolicyStance,
    LiquidityRegime,
    AllocationWeight,
    CyclePosition,
    GeopoliticalResult,
    MonetaryResult,
    AssetClassResult,
    SectorResult,
    IndividualResult,
    LevelAnalysisResult,
    run_quick_top_down,
    LEVEL_DEBATE_PROMPTS
)

__all__ = [
    # Base
    'BaseAgent',
    'AgentConfig',

    # Core Agents
    'AnalysisAgent',
    'ForecastAgent',
    'ResearchAgent',
    'StrategyAgent',
    'VerificationAgent',
    'MethodologyDebateAgent',
    'MetaOrchestrator',

    # Research
    'ResearchReport',
    'ResearchCategory',

    # Verification
    'VerificationResult',
    'HallucinationCheck',
    'SycophancyCheck',

    # Strategy
    'PortfolioStrategy',
    'TradeRecommendation',
    'MarketState',
    'CriticalPathState',
    'create_market_state_from_data',
    'create_critical_path_state',

    # Regime Change Detection
    'RegimeChangeDetectionPipeline',
    'VolumeBreakoutDetector',
    'NewsSearchAgent',
    'NewsClassificationAgent',
    'ImpactAssessmentDebate',
    'RegimeChangeDecision',
    'VolumeEvent',
    'NewsClassification',
    'RegimeChangeResult',

    # Methodology Debate
    'MethodologyDebate',
    'MethodologyProposal',
    'MethodologyDecision',
    'MethodologyType',
    'ResearchGoal',
    'DataSummary',
    'METHODOLOGY_KNOWLEDGE',
    'GOAL_METHOD_MAPPING',

    # Interpretation Debate (경제학파별 해석)
    'InterpretationDebateAgent',
    'InterpretationDebate',
    'InterpretationConsensus',
    'SchoolInterpretation',
    'AnalysisResult',
    'EconomicSchool',
    'SCHOOL_SYSTEM_PROMPTS',
    'create_school_participants',

    # Top-Down Orchestrator (하향식 분석)
    'TopDownOrchestrator',
    'TopDownResult',
    'AnalysisLevel',
    'RiskLevel',
    'Stance',
    'PolicyStance',
    'LiquidityRegime',
    'AllocationWeight',
    'CyclePosition',
    'GeopoliticalResult',
    'MonetaryResult',
    'AssetClassResult',
    'SectorResult',
    'IndividualResult',
    'LevelAnalysisResult',
    'run_quick_top_down',
    'LEVEL_DEBATE_PROMPTS',
]