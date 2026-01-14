"""
Multi-Agent System - Core Module
=================================
핵심 프레임워크 컴포넌트

Modules:
- schemas: 데이터 스키마 정의
- debate: 기본 토론 프로토콜
- debate_framework: Multi-AI 토론 프레임워크 (확장)
- signal_action: 신호-액션 매핑 프레임워크
- database: SQLite 기반 분석 결과 저장소
"""

from .schemas import (
    AgentRequest,
    AgentResponse,
    AgentOpinion,
    Consensus,
    Conflict,
    WorkflowPlan,
    AgentRole,
    TaskPriority,
    OpinionStrength,
)
from .debate import DebateProtocol
from .debate_framework import (
    DebateFramework,
    DebateParticipant,
    DebateConfig,
    DebateResult,
    Opinion,
    Critique,
    ConsensusType,
    DebatePhase,
    AIProvider,
    AIClient,
    get_default_participants,
)
from .signal_action import (
    SignalStrength,
    ActionType,
    PositionDirection,
    RiskProfileType,
    MarketRegime,
    ConflictType,
    RiskProfile,
    EnhancedSignal,
    Action,
    ActionLog,
    SignalConflict,
    SignalActionMapper,
)
from .database import DatabaseManager

__all__ = [
    # Schemas
    'AgentRequest',
    'AgentResponse',
    'AgentOpinion',
    'Consensus',
    'Conflict',
    'WorkflowPlan',
    'AgentRole',
    'TaskPriority',
    'OpinionStrength',

    # Debate (basic)
    'DebateProtocol',

    # Debate Framework (extended)
    'DebateFramework',
    'DebateParticipant',
    'DebateConfig',
    'DebateResult',
    'Opinion',
    'Critique',
    'ConsensusType',
    'DebatePhase',
    'AIProvider',
    'AIClient',
    'get_default_participants',

    # Signal-Action Framework
    'SignalStrength',
    'ActionType',
    'PositionDirection',
    'RiskProfileType',
    'MarketRegime',
    'ConflictType',
    'RiskProfile',
    'EnhancedSignal',
    'Action',
    'ActionLog',
    'SignalConflict',
    'SignalActionMapper',

    # Database
    'DatabaseManager',
]
