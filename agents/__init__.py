"""
Multi-Agent System - Agents Module
===================================
Economic Intelligence Multi-Agent System (EIMAS) 에이전트 모듈

Active Agents (7개):
- BaseAgent: 모든 에이전트의 기본 클래스
- AnalysisAgent: CriticalPath 기반 데이터 분석
- ForecastAgent: LASSO 기반 예측
- ResearchAgent: Perplexity/Claude 기반 리서치
- StrategyAgent: 포트폴리오 전략 생성
- VerificationAgent: AI 출력 검증 (Hallucination/Sycophancy)
- InterpretationDebateAgent: 경제학파별 해석 토론 (Phase 2)
- MethodologyDebateAgent: 분석 방법론 토론 (Phase 2)
- MetaOrchestrator: 에이전트 조율 및 토론

Archived Agents:
- agents/archive/ 디렉토리 참조
"""

from .base_agent import BaseAgent, AgentConfig
from .analysis_agent import AnalysisAgent
from .forecast_agent import ForecastAgent
from .research_agent import ResearchAgent
from .strategy_agent import StrategyAgent
from .verification_agent import VerificationAgent
from .interpretation_debate import InterpretationDebateAgent
from .methodology_debate import MethodologyDebateAgent
from .orchestrator import MetaOrchestrator

__all__ = [
    # Base
    'BaseAgent',
    'AgentConfig',

    # Active Agents (7개)
    'AnalysisAgent',
    'ForecastAgent',
    'ResearchAgent',
    'StrategyAgent',
    'VerificationAgent',
    'InterpretationDebateAgent',
    'MethodologyDebateAgent',

    # Orchestrator
    'MetaOrchestrator',
]
