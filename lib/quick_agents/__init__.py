"""
Quick Mode AI Agents
====================
Quick 모드용 특화 AI 에이전트 시스템

Purpose:
- Full 모드 결과 검증
- KOSPI/SPX 전용 분석
- 경제학 이론 적합성 검증
- 최종 의사결정 지원

Agents:
- PortfolioValidator: 포트폴리오 이론 검증 (Claude API)
- AllocationReasoner: 자산배분 논리 분석 (Perplexity API)
- MarketSentimentAgent: 시장 정서 (KOSPI/SPX 분리, Claude API)
- AlternativeAssetAgent: 대체자산 판단 (Perplexity API)
- FinalValidator: 최종 검증 종합 (Claude API)
- QuickOrchestrator: 전체 조율자

Economic Foundation:
- Markowitz (1952): Mean-Variance Optimization
- Black-Litterman (1992): Bayesian Portfolio
- Qian (2005): Risk Parity
- Baker & Wurgler (2006): Market Sentiment
- Gorton & Rouwenhorst (2006): Commodity Futures
- Baur & Lucey (2010): Gold as Safe Haven
- Kahneman & Tversky: Behavioral Finance

API Requirements:
- ANTHROPIC_API_KEY: Claude API (PortfolioValidator, MarketSentimentAgent, FinalValidator)
- PERPLEXITY_API_KEY: Perplexity API (AllocationReasoner, AlternativeAssetAgent)

Usage:
    from lib.quick_agents import QuickOrchestrator

    orchestrator = QuickOrchestrator()
    result = orchestrator.run_quick_validation(full_json_path)

    # Result structure:
    # {
    #     'portfolio_validation': {...},
    #     'allocation_reasoning': {...},
    #     'market_sentiment': {...},
    #     'alternative_assets': {...},
    #     'final_validation': {...},
    #     'execution_time_seconds': float
    # }
"""

from .portfolio_validator import PortfolioValidator
from .allocation_reasoner import AllocationReasoner
from .market_sentiment_agent import MarketSentimentAgent
from .alternative_asset_agent import AlternativeAssetAgent
from .final_validator import FinalValidator
from .quick_orchestrator import QuickOrchestrator

__all__ = [
    'PortfolioValidator',
    'AllocationReasoner',
    'MarketSentimentAgent',
    'AlternativeAssetAgent',
    'FinalValidator',
    'QuickOrchestrator',
]
