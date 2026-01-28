import pytest
import asyncio
import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.orchestrator import MetaOrchestrator
from core.multi_llm_debate import MultiLLMDebate
from core.schemas import AgentRequest, AgentRole, TaskPriority

async def test_full_agent_pipeline():
    """전체 에이전트 파이프라인 테스트"""
    print("\n[Test] Running Full Agent Pipeline...")
    orchestrator = MetaOrchestrator(verbose=True)

    # Mock market data
    market_data = {
        'SPY': pd.DataFrame({'Close': [448, 449, 450]}),
        'VIX': pd.DataFrame({'Close': [15, 14.5, 14.2]}),
    }

    # Running orchestrator workflow
    # Note: run_with_debate is the high-level method
    result = await orchestrator.run_with_debate(
        query="Current market outlook",
        market_data=market_data
    )

    # Assertions
    print(f"  Result keys: {result.keys()}")
    
    assert 'analysis' in result
    assert 'forecast' in result
    assert 'debate' in result
    assert 'recommendations' in result
    
    # Check debate consensus
    consensus = result.get('debate', {}).get('consensus', {})
    assert consensus is not None
    
    # Check if new agents (Research/Strategy) contributed (indirectly via opinions)
    # The opinions are inside 'debate' -> 'opinions'
    opinions = result.get('debate', {}).get('opinions', [])
    roles = [op['agent_role'] for op in opinions]
    print(f"  Participating Agents: {set(roles)}")
    
    # We expect Research and Strategy to be there if topics aligned
    # But run_with_debate auto-detects topics. 
    # With 'total_risk_score': 45 (uncertain range), 'market_outlook' should be added.
    # Research and Strategy agents respond to 'market_outlook'.
    assert 'research' in roles or 'strategy' in roles or 'analysis' in roles

async def test_multi_llm_debate():
    """Multi-LLM 토론 테스트"""
    print("\n[Test] Running Multi-LLM Debate Engine...")
    debate = MultiLLMDebate()

    result = await debate.run_debate(
        topic="Fed Policy Direction Q1 2025",
        context={
            'regime': 'Bull',
            'risk_score': 45,
            'key_metrics': {'VIX': 14.2, 'Credit_Spread': 285}
        }
    )

    # 합의 도출 확인
    print(f"  Consensus: {result.consensus_position}")
    assert result.consensus_position in ['BULLISH', 'BEARISH', 'NEUTRAL']
    assert result.consensus_confidence[0] <= result.consensus_confidence[1]

    # 모든 모델 참여 확인 (Mock mode falls back to generic response, 
    # but transcript should have entries)
    # Note: If API keys are missing, it uses fallback mock which sets model_name
    print(f"  Model Contributions: {result.model_contributions.keys()}")
    # In mock mode (if keys missing), model_names are 'claude', 'gpt4', 'gemini'
    # even if they are mocked.
    assert 'claude' in result.model_contributions
    assert 'gpt4' in result.model_contributions
    assert 'gemini' in result.model_contributions

if __name__ == "__main__":
    # Allow running without pytest
    asyncio.run(test_full_agent_pipeline())
    asyncio.run(test_multi_llm_debate())
