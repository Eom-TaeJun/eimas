#!/usr/bin/env python3
"""
Operational - Utilities
============================================================

유틸리티 함수들
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging

# Import from same package
from .config import OperationalConfig
from .enums import FinalStance, ReasonCode, TriggerType, SignalType

logger = logging.getLogger(__name__)


def get_indicator_classification(result_data: Dict) -> Dict:
    """
    CORE vs AUX 지표 분류 반환 (JSON 직렬화 가능)

    Core indicators: 의사결정 규칙에 직접 사용
    Auxiliary indicators: 투명성/감사 목적, 의사결정에 미사용

    Args:
        result_data: EIMAS 결과 딕셔너리

    Returns:
        indicator_classification 딕셔너리
    """
    regime_data = result_data.get('regime', {})
    regime_signal = 'NEUTRAL'
    regime_confidence = 0.5
    if isinstance(regime_data, dict):
        regime_raw = regime_data.get('regime', 'NEUTRAL')
        if 'Bull' in str(regime_raw) or 'BULL' in str(regime_raw):
            regime_signal = 'BULL'
        elif 'Bear' in str(regime_raw) or 'BEAR' in str(regime_raw):
            regime_signal = 'BEAR'
        regime_confidence = regime_data.get('confidence', 0.5)

    # Calculate canonical risk score
    base = result_data.get('base_risk_score', result_data.get('risk_score', 50.0))
    micro_adj = result_data.get('microstructure_adjustment', 0.0)
    bubble_adj = result_data.get('bubble_risk_adjustment', 0.0)
    ext_adj = result_data.get('extended_data_adjustment', 0.0)
    canonical_risk = max(0.0, min(100.0, base + micro_adj + bubble_adj + ext_adj))

    core_indicators = [
        {
            'name': 'regime_signal',
            'value': regime_signal,
            'source': 'RegimeDetector',
            'used_in_rules': ['RULE_5', 'RULE_7']
        },
        {
            'name': 'regime_confidence',
            'value': round(regime_confidence, 3),
            'source': 'RegimeDetector',
            'used_in_rules': ['RULE_1', 'RULE_2']
        },
        {
            'name': 'canonical_risk_score',
            'value': round(canonical_risk, 1),
            'source': 'ScoreDefinitions',
            'used_in_rules': ['RULE_4']
        },
        {
            'name': 'agent_consensus_stance',
            'value': result_data.get('full_mode_position', 'NEUTRAL'),
            'source': 'MetaOrchestrator',
            'used_in_rules': ['RULE_5', 'RULE_6', 'RULE_7']
        },
        {
            'name': 'agent_consensus_confidence',
            'value': round(result_data.get('confidence', 0.5), 3),
            'source': 'MetaOrchestrator',
            'used_in_rules': ['RULE_3', 'RULE_6']
        },
        {
            'name': 'constraint_status',
            'value': 'OK',  # Will be updated by operational engine
            'source': 'ConstraintRepair',
            'used_in_rules': ['RULE_2']
        },
        {
            'name': 'client_profile_status',
            'value': result_data.get('client_profile_status', 'COMPLETE'),
            'source': 'ClientConfig',
            'used_in_rules': ['RULE_1']
        }
    ]

    auxiliary_indicators = [
        {
            'name': 'base_risk_score',
            'value': round(base, 1),
            'source': 'CriticalPathAggregator',
            'description': 'Base risk from critical path analysis'
        },
        {
            'name': 'microstructure_adjustment',
            'value': round(micro_adj, 1),
            'source': 'DailyMicrostructureAnalyzer',
            'description': 'Market quality adjustment'
        },
        {
            'name': 'bubble_risk_adjustment',
            'value': round(bubble_adj, 1),
            'source': 'BubbleDetector',
            'description': 'Bubble risk overlay'
        },
        {
            'name': 'extended_data_adjustment',
            'value': round(ext_adj, 1),
            'source': 'ExtendedDataCollector',
            'description': 'PCR/Sentiment/Credit adjustment'
        },
        {
            'name': 'full_mode_position',
            'value': result_data.get('full_mode_position', 'NEUTRAL'),
            'source': 'MetaOrchestrator (full mode)',
            'description': 'Position from full analysis mode'
        },
        {
            'name': 'reference_mode_position',
            'value': result_data.get('reference_mode_position', 'NEUTRAL'),
            'source': 'MetaOrchestrator (reference mode)',
            'description': 'Position from reference mode'
        },
        {
            'name': 'modes_agree',
            'value': result_data.get('modes_agree', True),
            'source': 'DualModeAnalyzer',
            'description': 'Whether full and reference modes agree'
        }
    ]

    # Add market quality if available
    mq = result_data.get('market_quality', {})
    if isinstance(mq, dict) and 'avg_liquidity_score' in mq:
        auxiliary_indicators.append({
            'name': 'liquidity_score',
            'value': round(mq['avg_liquidity_score'], 1),
            'source': 'MicrostructureAnalyzer',
            'description': 'Market liquidity score'
        })

    return {
        'core_indicators': core_indicators,
        'auxiliary_indicators': auxiliary_indicators,
        'note': 'Only core_indicators are used in decision rules. auxiliary_indicators are for transparency only.'
    }


def get_input_validation(result_data: Dict) -> Dict:
    """
    입력 데이터 검증 결과 반환 (JSON 직렬화 가능)

    핵심 지표 검증 및 데이터 품질 상태 확인

    Args:
        result_data: EIMAS 결과 딕셔너리

    Returns:
        input_validation 딕셔너리
    """
    # Define required core indicators
    required_indicators = [
        'regime', 'risk_score', 'full_mode_position', 'confidence', 'modes_agree', 'portfolio_weights'
    ]

    missing_indicators = []
    anomalies_detected = []
    valid_count = 0

    for indicator in required_indicators:
        if indicator not in result_data or result_data.get(indicator) is None:
            missing_indicators.append(indicator)
        else:
            valid_count += 1

    # Check for data anomalies
    risk_score = result_data.get('risk_score', 50.0)
    if risk_score < 0 or risk_score > 100:
        anomalies_detected.append(f"risk_score out of range: {risk_score}")

    confidence = result_data.get('confidence', 0.5)
    if confidence < 0 or confidence > 1:
        anomalies_detected.append(f"confidence out of range: {confidence}")

    # Check market quality
    mq = result_data.get('market_quality', {})
    data_quality = mq.get('data_quality', 'COMPLETE') if isinstance(mq, dict) else 'COMPLETE'

    # Determine overall status
    if missing_indicators or data_quality == 'DEGRADED':
        status = 'DEGRADED'
    elif anomalies_detected:
        status = 'PARTIAL'
    else:
        status = 'COMPLETE'

    return {
        'data_quality_status': status,
        'core_indicators_checked': len(required_indicators),
        'core_indicators_valid': valid_count,
        'missing_indicators': missing_indicators,
        'anomalies_detected': anomalies_detected
    }


def get_operational_controls(
    result_data: Dict,
    rebalance_decision: Dict = None,
    constraint_result: Dict = None
) -> Dict:
    """
    운영 통제 상태 반환 (JSON 직렬화 가능)

    Args:
        result_data: EIMAS 결과 딕셔너리
        rebalance_decision: 리밸런싱 결정 딕셔너리
        constraint_result: 제약 수정 결과 딕셔너리

    Returns:
        operational_controls 딕셔너리
    """
    rebal = rebalance_decision or result_data.get('rebalance_decision', {})

    # Turnover control
    turnover = rebal.get('turnover', 0.0)
    turnover_cap = 0.30  # Default from OperationalConfig

    # Determine if turnover cap was triggered
    turnover_cap_triggered = turnover >= turnover_cap * 0.95  # Within 5% of cap

    # Constraint status
    constraints_satisfied = True
    constraint_violations = []

    if constraint_result:
        constraints_satisfied = constraint_result.get('constraints_satisfied', True)
        violations = constraint_result.get('violations_found', [])
        for v in violations:
            if isinstance(v, dict):
                constraint_violations.append(
                    f"{v.get('asset_class', 'unknown')} weight {v.get('current_value', 0):.1%} "
                    f"{'< min' if v.get('violation_type') == 'BELOW_MIN' else '> max'} "
                    f"{v.get('limit_value', 0):.1%}"
                )
    elif result_data.get('operational_report', {}).get('constraint_repair'):
        cr = result_data['operational_report']['constraint_repair']
        constraints_satisfied = cr.get('constraints_satisfied', True)
        for v in cr.get('violations_found', []):
            if isinstance(v, dict):
                constraint_violations.append(
                    f"{v.get('asset_class', 'unknown')} weight {v.get('current_value', 0):.1%} "
                    f"{'< min' if v.get('violation_type') == 'BELOW_MIN' else '> max'} "
                    f"{v.get('limit_value', 0):.1%}"
                )

    return {
        'turnover_cap': turnover_cap,
        'turnover_applied': round(turnover, 4),
        'turnover_cap_triggered': turnover_cap_triggered,
        'constraints_satisfied': constraints_satisfied,
        'constraint_violations': constraint_violations
    }


def get_audit_metadata(result_data: Dict) -> Dict:
    """
    감사 메타데이터 반환 (JSON 직렬화 가능)

    Args:
        result_data: EIMAS 결과 딕셔너리

    Returns:
        audit_metadata 딕셔너리
    """
    return {
        'timestamp': result_data.get('timestamp', datetime.now().isoformat()),
        'schema_version': result_data.get('schema_version', '2.2.3'),
        'pipeline_version': 'EIMAS v2.2.3',
        'deterministic': True
    }


def get_approval_status(
    result_data: Dict,
    rebalance_plan: Dict = None
) -> Dict:
    """
    승인 상태 반환 (JSON 직렬화 가능)

    Args:
        result_data: EIMAS 결과 딕셔너리
        rebalance_plan: 리밸런싱 계획 딕셔너리

    Returns:
        approval_status 딕셔너리
    """
    plan = rebalance_plan or result_data.get('operational_report', {}).get('rebalance_plan', {})
    approval = plan.get('approval', {})

    requires_approval = approval.get('requires_human_approval', False)
    checklist = approval.get('approval_checklist', [])

    # If no plan available, use default
    if not plan:
        requires_approval = False
        checklist = []

    return {
        'requires_human_approval': requires_approval,
        'approval_checklist': checklist,
        'approval_granted': False,  # Always false by default
        'approval_reason': approval.get('approval_reason', '') if requires_approval else ''
    }


# =============================================================================
# Test Code
# =============================================================================
