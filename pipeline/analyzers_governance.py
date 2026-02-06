#!/usr/bin/env python3
"""
EIMAS Pipeline - Analyzers Governance Module

Purpose: AI validation, allocation, and rebalancing functions (Phase 2.24, 2.11-2.12 rebalancing)
Functions: run_ai_validation, run_allocation_engine, run_rebalancing_policy
"""

from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

# NEW: Allocation & Rebalancing (2026-02-02)
from lib.adapters import (
    AllocationConstraints,
    AllocationEngine,
    AllocationResult,
    AllocationStrategy,
    AssetClassBounds,
    RebalanceConfig,
    RebalanceDecision,
    RebalanceFrequency,
    RebalancePolicy,
    RebalancingPolicy,
    TradingCostModel,
)

# Schemas
from pipeline.exceptions import get_logger, log_error

logger = get_logger("analyzers.governance")

def run_ai_validation(result_data: Dict, use_cache: bool = True) -> Dict[str, Any]:
    """
    AI 기반 투자 전략 검증 (Multi-LLM)

    기능:
    - Claude, Gemini, Perplexity, OpenAI 4개 AI의 독립 검증
    - 합의 도출 및 신뢰도 계산
    - 24시간 캐싱으로 API 비용 절감

    References:
    - validation_agents.py
    """
    print("\n[2.24] AI Validation (Multi-LLM)...")
    try:
        import os
        import json
        from datetime import datetime, timedelta
        from pathlib import Path

        # 캐시 확인
        cache_path = Path("outputs/.validation_cache.json")
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                cache_time = datetime.fromisoformat(cache['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=24):
                    print("      ✓ Using cached validation result (< 24h old)")
                    return cache['result']
            except Exception:
                pass

        # API 키 확인
        has_apis = all([
            os.getenv('ANTHROPIC_API_KEY'),
            os.getenv('OPENAI_API_KEY'),
        ])

        if not has_apis:
            print("      ⚠ AI Validation skipped: Missing API keys")
            return {'status': 'SKIPPED', 'reason': 'Missing API keys'}

        from lib.validation_agents import ValidationAgentManager

        manager = ValidationAgentManager()

        # 검증 실행
        agent_decision = {
            'recommendation': result_data.get('final_recommendation', 'HOLD'),
            'confidence': result_data.get('confidence', 0.5),
            'risk_level': result_data.get('risk_level', 'MEDIUM'),
        }
        market_condition = {
            'regime': result_data.get('regime', {}),
            'risk_score': result_data.get('risk_score', 50),
        }

        consensus = manager.validate_all(agent_decision, market_condition)

        validation_result = {
            'final_result': consensus.final_result.value,
            'consensus_confidence': consensus.consensus_confidence,
            'agreement_ratio': consensus.agreement_ratio,
            'key_concerns': consensus.key_concerns,
            'action_items': consensus.action_items,
            'summary': consensus.summary
        }

        # 캐시 저장
        if use_cache:
            try:
                cache_path.parent.mkdir(exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'result': validation_result
                    }, f)
            except Exception:
                pass

        print(f"      ✓ AI Consensus: {consensus.final_result.value}")
        print(f"      ✓ Agreement: {consensus.agreement_ratio:.0%}")

        return validation_result
    except Exception as e:
        log_error(logger, "AI validation failed", e)
        return {'status': 'ERROR', 'error': str(e)}


def run_allocation_engine(
    market_data: Dict[str, pd.DataFrame],
    strategy: str = "risk_parity",
    constraints: Dict = None,
    current_weights: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    비중 산출 엔진 실행

    Args:
        market_data: 시장 데이터 {ticker: DataFrame}
        strategy: 배분 전략 (risk_parity, mvo_max_sharpe, hrp, equal_weight, inverse_vol)
        constraints: 제약 조건 (min_weight, max_weight, asset_limits)
        current_weights: 현재 비중 (리밸런싱용)

    Returns:
        Dict with allocation_result and rebalance_decision
    """
    print("\n[2.11] Running Allocation Engine...")

    result = {
        'allocation_result': {},
        'rebalance_decision': {},
        'allocation_strategy': strategy,
        'allocation_config': {},
        'status': 'SUCCESS',
        'warnings': []
    }

    try:
        # 수익률 DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) < 3:
            result['status'] = 'HOLD'
            result['warnings'].append("Insufficient assets for optimization (< 3)")
            logger.warning("Allocation engine: Insufficient assets, defaulting to HOLD")
            return result

        # 데이터 품질 검사
        if returns_df.isnull().sum().sum() > len(returns_df) * 0.1:
            result['status'] = 'HOLD'
            result['warnings'].append("Data quality issue: >10% missing values")
            logger.warning("Allocation engine: Data quality issue, defaulting to HOLD")
            return result

        # 제약 조건 설정
        if constraints:
            alloc_constraints = AllocationConstraints(**constraints)
        else:
            alloc_constraints = AllocationConstraints(
                min_weight=0.0,
                max_weight=0.4,  # 단일 자산 최대 40%
                long_only=True
            )

        # Allocation Engine 실행
        engine = AllocationEngine(risk_free_rate=0.045)

        try:
            strategy_enum = AllocationStrategy(strategy)
        except ValueError:
            strategy_enum = AllocationStrategy.RISK_PARITY
            result['warnings'].append(f"Unknown strategy '{strategy}', using risk_parity")

        allocation = engine.allocate(
            returns=returns_df,
            strategy=strategy_enum,
            constraints=alloc_constraints
        )

        result['allocation_result'] = allocation.to_dict()
        result['allocation_strategy'] = strategy_enum.value
        result['allocation_config'] = {
            'min_weight': alloc_constraints.min_weight,
            'max_weight': alloc_constraints.max_weight,
            'long_only': alloc_constraints.long_only,
            'risk_free_rate': engine.risk_free_rate
        }

        # Fallback transparency: 최적화 실패 시 경고 추가
        if allocation.is_fallback:
            result['warnings'].append(f"Optimization fallback: {allocation.fallback_reason}")
            logger.warning(f"Allocation engine used fallback: {allocation.optimization_status}")

        # 상위 비중 출력
        top_weights = sorted(allocation.weights.items(), key=lambda x: x[1], reverse=True)[:5]
        top_str = ", ".join([f"{t}:{w:.1%}" for t, w in top_weights])
        print(f"      ✓ Strategy: {strategy_enum.value} ({allocation.optimization_status})")
        print(f"      ✓ Top Allocation: {top_str}")
        print(f"      ✓ Expected Return: {allocation.expected_return:.2%}")
        print(f"      ✓ Expected Vol: {allocation.expected_volatility:.2%}")
        print(f"      ✓ Sharpe: {allocation.sharpe_ratio:.2f}")
        if allocation.is_fallback:
            print(f"      ⚠️  Fallback used: {allocation.fallback_reason}")

        # Rebalancing Policy 평가 (현재 비중이 제공된 경우)
        if current_weights:
            result['rebalance_decision'] = run_rebalancing_policy(
                current_weights=current_weights,
                target_weights=allocation.weights
            )

    except Exception as e:
        log_error(logger, "Allocation engine failed", e)
        result['status'] = 'HOLD'
        result['warnings'].append(f"Allocation failed: {str(e)}")

    return result


def run_rebalancing_policy(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    last_rebalance_date: datetime = None,
    market_data_quality: str = "COMPLETE",
    config: Dict = None
) -> Dict[str, Any]:
    """
    리밸런싱 정책 평가

    Args:
        current_weights: 현재 포트폴리오 비중
        target_weights: 목표 비중
        last_rebalance_date: 마지막 리밸런싱 일자
        market_data_quality: 데이터 품질 상태
        config: 리밸런싱 설정

    Returns:
        RebalanceDecision dict
    """
    print("\n[2.12] Evaluating Rebalancing Policy...")

    try:
        # 설정 생성
        if config:
            rebal_config = RebalanceConfig.from_dict(config)
        else:
            rebal_config = RebalanceConfig(
                policy=RebalancePolicy.HYBRID,
                frequency=RebalanceFrequency.MONTHLY,
                drift_threshold=0.05,       # 5%
                turnover_cap=0.30,          # 30%
                min_trade_size=0.01,        # 1%
                cost_model=TradingCostModel(
                    commission_rate=0.001,  # 0.1%
                    spread_cost=0.0005,     # 0.05%
                    market_impact=0.001
                ),
                asset_bounds=AssetClassBounds.moderate()
            )

        # 정책 매니저 생성
        policy = RebalancingPolicy(rebal_config)

        # 시장 데이터 품질 정보
        market_data = {'data_quality': market_data_quality}

        # 평가 실행
        decision = policy.evaluate(
            current_weights=current_weights,
            target_weights=target_weights,
            last_rebalance_date=last_rebalance_date,
            market_data=market_data
        )

        print(f"      ✓ Action: {decision.action}")
        print(f"      ✓ Reason: {decision.reason}")
        if decision.should_rebalance:
            print(f"      ✓ Turnover: {decision.turnover:.2%}")
            print(f"      ✓ Estimated Cost: {decision.estimated_cost:.4f}")

        return decision.to_dict()

    except Exception as e:
        log_error(logger, "Rebalancing policy failed", e)
        return {
            'should_rebalance': False,
            'action': 'HOLD',
            'reason': f'Policy evaluation failed: {str(e)}',
            'current_weights': current_weights,
            'target_weights': target_weights,
            'warnings': [str(e)]
        }
