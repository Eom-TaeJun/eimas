#!/usr/bin/env python3
"""
Validation Package
============================================================

Multi-agent validation framework with consensus and feedback loops

Public API:
    - ValidationAgentManager: Main validation coordinator
    - ValidationLoopManager: Feedback loop manager
    - BaseValidationAgent: Base class for custom agents
    - ConsensusEngine: Multi-agent consensus
    - ValidationResult: Validation result enum

AI Providers:
    - ClaudeValidationAgent
    - PerplexityValidationAgent
    - GeminiValidationAgent
    - GPTValidationAgent

Usage:
    from lib.validation import ValidationAgentManager

    manager = ValidationAgentManager()
    result = manager.validate_all(content, criteria)
"""

from .manager import ValidationAgentManager
from .loop_manager import ValidationLoopManager
from .base import BaseValidationAgent
from .consensus import ConsensusEngine
from .enums import ValidationResult
from .schemas import AIValidation, ConsensusResult, FeedbackResult
from .claude import ClaudeValidationAgent
from .perplexity import PerplexityValidationAgent
from .gemini import GeminiValidationAgent
from .gpt import GPTValidationAgent
from .feedback import FeedbackValidationAgent

__all__ = [
    "ValidationAgentManager",
    "ValidationLoopManager",
    "BaseValidationAgent",
    "ConsensusEngine",
    "ValidationResult",
    "AIValidation",
    "ConsensusResult",
    "FeedbackResult",
    "ClaudeValidationAgent",
    "PerplexityValidationAgent",
    "GeminiValidationAgent",
    "GPTValidationAgent",
    "FeedbackValidationAgent",
]
