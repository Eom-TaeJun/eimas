"""
Multi-Agent System - Agents Module
===================================
Economic Intelligence Multi-Agent System (EIMAS) 에이전트 모듈

Active Agents:
- BaseAgent: 모든 에이전트의 기본 클래스
- AnalysisAgent: CriticalPath 기반 데이터 분석
- ForecastAgent: LASSO 기반 예측
- MetaOrchestrator: 에이전트 조율 및 토론

Archived Agents:
- agents/archive/ 디렉토리 참조
- 필요시 직접 import: from agents.archive.xxx import XxxAgent
"""

from .base_agent import BaseAgent, AgentConfig
from .analysis_agent import AnalysisAgent
from .forecast_agent import ForecastAgent
from .orchestrator import MetaOrchestrator

__all__ = [
    # Base
    'BaseAgent',
    'AgentConfig',

    # Active Agents
    'AnalysisAgent',
    'ForecastAgent',
    'MetaOrchestrator',
]
