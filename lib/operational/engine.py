#!/usr/bin/env python3
"""
Operational - Main Engine
==========================

운영 엔진 메인 클래스

Economic Foundation:
    - Decision Governance: 체계적 의사결정 프로세스
    - Risk-Adjusted Allocation: 리스크 조정 자산배분
    - Constraint-Based Optimization: 제약조건 기반 최적화

Main Class:
    - OperationalEngine: 전체 운영 파이프라인 조율

Design Philosophy:
    - No new signals generated (새 시그널 생성 금지)
    - Deterministic output (결정론적 출력)
    - Fully auditable (완전 감사 가능)

Pipeline:
    1. Input Validation → 2. Decision Resolution → 3. Constraint Repair
    → 4. Rebalance Planning → 5. Report Generation
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# Import from same package
from .config import OperationalConfig
from .enums import ReasonCode, TriggerType, SignalType
from .signal import SignalHierarchyReport, SignalClassification
from .hold_policy import HoldPolicyReport, evaluate_hold_conditions
from .risk_metrics import ScoreDefinitions, AuxiliaryRiskMetric
from .decision import DecisionPolicy, DecisionInputs, resolve_decision
from .constraints import repair_constraints, ConstraintRepairResult
from .rebalance import generate_rebalance_plan, RebalancePlan
from .reports import OperationalReport, AuditMetadata

logger = logging.getLogger(__name__)

# Core signal names allowed in decision layer.
CORE_SIGNALS = frozenset([
    'regime_signal',
    'regime_confidence',
    'canonical_risk_score',
    'agent_consensus_stance',
    'agent_consensus_confidence',
    'constraint_status',
    'client_profile_status',
])


class OperationalEngine:
    """
    운영 엔진

    EIMAS 결과를 운영 가능한 형태로 변환

    Example:
        >>> engine = OperationalEngine()
        >>> report = engine.process(eimas_result, current_weights)
        >>> print(report.to_markdown())
    """

    def __init__(self, config: Optional[OperationalConfig] = None):
        self.config = config or OperationalConfig()

    def process(
        self,
        eimas_result: Dict,
        current_weights: Optional[Dict[str, float]] = None
    ) -> OperationalReport:
        """
        EIMAS 결과를 운영 리포트로 변환

        Args:
            eimas_result: EIMAS JSON 결과
            current_weights: 현재 포트폴리오 비중 (리밸런싱용)

        Returns:
            OperationalReport
        """
        # 1. Score Definitions
        score_defs = self._extract_scores(eimas_result)

        # 2. Extract allocation
        allocation = self._extract_allocation(eimas_result)

        # 3. Constraint Repair
        constraint_result = repair_constraints(allocation, self.config)

        # 4. Decision Inputs
        decision_inputs = self._extract_decision_inputs(
            eimas_result, score_defs, constraint_result
        )

        # 5. Resolve Decision
        decision = resolve_decision(decision_inputs, self.config)

        # 6. Rebalance Plan
        rebalance = self._generate_rebalance_plan(
            eimas_result, current_weights, constraint_result, decision
        )

        # 7. Signal Hierarchy Report
        signal_hierarchy = self._build_signal_hierarchy(
            eimas_result, score_defs, decision_inputs
        )

        # 8. HOLD Policy Evaluation
        hold_policy = evaluate_hold_conditions(
            regime_signal=decision_inputs.regime,
            regime_confidence=decision_inputs.regime_confidence,
            canonical_risk_score=score_defs.canonical_risk_score,
            agent_consensus_stance=decision_inputs.agent_consensus,
            agent_consensus_confidence=decision_inputs.confidence,
            constraint_status=decision_inputs.constraint_status,
            client_profile_status=eimas_result.get('client_profile_status', 'COMPLETE'),
            data_quality=decision_inputs.data_quality,
            modes_agree=decision_inputs.modes_agree,
            config=self.config,
        )

        # 9. Audit Metadata
        audit = AuditMetadata(
            timestamp=datetime.now().isoformat(),
            system_version="EIMAS v2.2.2",
        )

        return OperationalReport(
            decision_policy=decision,
            score_definitions=score_defs,
            allocation=constraint_result.repaired_weights if constraint_result.repair_successful else allocation,
            constraint_repair=constraint_result,
            rebalance_plan=rebalance,
            audit_metadata=audit,
            signal_hierarchy=signal_hierarchy,
            hold_policy=hold_policy,
            raw_inputs={'eimas_result': eimas_result},
        )

    def _extract_scores(self, data: Dict) -> ScoreDefinitions:
        """
        리스크 점수 추출 (Unified Risk Scoring)

        Uses ScoreDefinitions.from_components() to ensure:
        - canonical_risk_score is computed consistently
        - risk_level is derived from canonical score
        - sub-scores are labeled as auxiliary
        """
        base = data.get('base_risk_score', data.get('risk_score', 50.0))
        micro_adj = data.get('microstructure_adjustment', 0.0)
        bubble_adj = data.get('bubble_risk_adjustment', 0.0)
        ext_adj = data.get('extended_data_adjustment', 0.0)

        # Extract any additional auxiliary metrics
        auxiliary_metrics = []

        # Add volatility if available
        if 'volatility_score' in data:
            auxiliary_metrics.append(AuxiliaryRiskMetric(
                name='volatility_score',
                value=data['volatility_score'],
                unit='points',
                source='RegimeDetector',
                description='시장 변동성 지표 (auxiliary)'
            ))

        # Add liquidity score if available
        market_quality = data.get('market_quality', {})
        if isinstance(market_quality, dict) and 'avg_liquidity_score' in market_quality:
            auxiliary_metrics.append(AuxiliaryRiskMetric(
                name='liquidity_score',
                value=market_quality['avg_liquidity_score'],
                unit='points',
                source='DailyMicrostructureAnalyzer',
                description='시장 유동성 지표 (auxiliary)'
            ))

        # Add credit spread if available
        if 'credit_spread' in data:
            auxiliary_metrics.append(AuxiliaryRiskMetric(
                name='credit_spread',
                value=data['credit_spread'],
                unit='bps',
                source='CreditAnalyzer',
                description='신용 스프레드 (auxiliary)'
            ))

        return ScoreDefinitions.from_components(
            base_risk_score=base,
            microstructure_adjustment=micro_adj,
            bubble_risk_adjustment=bubble_adj,
            extended_data_adjustment=ext_adj,
            auxiliary_metrics=auxiliary_metrics,
        )

    def _extract_allocation(self, data: Dict) -> Dict[str, float]:
        """배분 비중 추출"""
        # allocation_result가 있으면 사용
        alloc_result = data.get('allocation_result', {})
        if alloc_result and 'weights' in alloc_result:
            return alloc_result['weights']

        # portfolio_weights 사용
        return data.get('portfolio_weights', {})

    def _extract_decision_inputs(
        self,
        data: Dict,
        scores: ScoreDefinitions,
        constraint_result: ConstraintRepairResult
    ) -> DecisionInputs:
        """결정 입력 추출"""
        regime_data = data.get('regime', {})
        if isinstance(regime_data, dict):
            regime = regime_data.get('regime', 'NEUTRAL')
            # Extract regime part (e.g., "Bull (Low Vol)" -> "BULL")
            if 'Bull' in regime or 'BULL' in regime:
                regime = 'BULL'
            elif 'Bear' in regime or 'BEAR' in regime:
                regime = 'BEAR'
            else:
                regime = 'NEUTRAL'
            regime_confidence = regime_data.get('confidence', 0.5)
        else:
            regime = 'NEUTRAL'
            regime_confidence = 0.5

        # Constraint status
        if not constraint_result.violations_found:
            constraint_status = "OK"
        elif constraint_result.force_hold:
            # Even if constraints_satisfied=True, force_hold means repair was inadequate
            constraint_status = "UNREPAIRED"
        elif constraint_result.constraints_satisfied:
            constraint_status = "REPAIRED"
        else:
            constraint_status = "UNREPAIRED"

        # Data quality
        mq = data.get('market_quality', {})
        data_quality = mq.get('data_quality', 'COMPLETE') if isinstance(mq, dict) else 'COMPLETE'

        # Client profile status
        client_profile_status = data.get('client_profile_status', 'COMPLETE')

        return DecisionInputs(
            regime=regime,
            regime_confidence=regime_confidence,
            risk_score=scores.canonical_risk_score,
            agent_consensus=data.get('full_mode_position', 'NEUTRAL'),
            full_mode_position=data.get('full_mode_position', 'NEUTRAL'),
            reference_mode_position=data.get('reference_mode_position', 'NEUTRAL'),
            modes_agree=data.get('modes_agree', True),
            confidence=data.get('confidence', 0.5),
            constraint_status=constraint_status,
            client_profile_status=client_profile_status,
            data_quality=data_quality,
        )

    def _generate_rebalance_plan(
        self,
        data: Dict,
        current_weights: Optional[Dict[str, float]],
        constraint_result: ConstraintRepairResult,
        decision: DecisionPolicy
    ) -> RebalancePlan:
        """
        리밸런싱 계획 생성 (Enhanced Execution Plan)

        When rebalancing is NOT executed, explicitly states why.
        """
        target = constraint_result.repaired_weights if constraint_result.repair_successful else {}

        if not current_weights:
            current_weights = data.get('portfolio_weights', {})

        # Case 1: No weights available
        if not target or not current_weights:
            return RebalancePlan(
                should_execute=False,
                not_executed_reason="No portfolio weights available (current or target)",
                trigger_type=TriggerType.MANUAL.value,
                trigger_reason="No weights available",
            )

        # Determine trigger type
        rebal_decision = data.get('rebalance_decision', {})
        reason = rebal_decision.get('reason', 'Manual request')

        if 'Threshold exceeded' in reason or 'drift' in reason.lower():
            trigger_type = TriggerType.DRIFT
        elif constraint_result.violations_found:
            trigger_type = TriggerType.CONSTRAINT_REPAIR
        elif 'regime' in reason.lower():
            trigger_type = TriggerType.REGIME_CHANGE
        elif 'Periodic' in reason or 'scheduled' in reason.lower():
            trigger_type = TriggerType.SCHEDULED
        else:
            trigger_type = TriggerType.MANUAL

        # Case 2: Decision is HOLD due to issues - block rebalancing
        hold_reason_codes = [
            ReasonCode.DATA_QUALITY_ISSUE.value,
            ReasonCode.CONSTRAINT_VIOLATION_UNREPAIRED.value,
            ReasonCode.LOW_CONFIDENCE.value,
            ReasonCode.CLIENT_PROFILE_MISSING.value,
            ReasonCode.AGENT_CONFLICT.value,
            ReasonCode.REGIME_STANCE_MISMATCH.value,
        ]

        if decision.final_stance == "HOLD":
            blocking_codes = [c for c in decision.reason_codes if c in hold_reason_codes]
            if blocking_codes:
                # Map codes to human-readable reasons
                code_explanations = {
                    ReasonCode.DATA_QUALITY_ISSUE.value: "Data quality degraded",
                    ReasonCode.CONSTRAINT_VIOLATION_UNREPAIRED.value: "Constraints could not be repaired",
                    ReasonCode.LOW_CONFIDENCE.value: "Confidence below threshold",
                    ReasonCode.CLIENT_PROFILE_MISSING.value: "Client profile missing",
                    ReasonCode.AGENT_CONFLICT.value: "Agent conflict unresolved",
                    ReasonCode.REGIME_STANCE_MISMATCH.value: "Regime contradicts consensus",
                }
                explanations = [code_explanations.get(c, c) for c in blocking_codes]

                return RebalancePlan(
                    should_execute=False,
                    not_executed_reason=f"Decision policy HOLD: {'; '.join(explanations)}",
                    trigger_type=trigger_type.value,
                    trigger_reason=f"Blocked by decision policy (stance=HOLD, codes={blocking_codes})",
                )

            # HOLD for other reasons (mixed signals, default) - still block
            return RebalancePlan(
                should_execute=False,
                not_executed_reason=f"Decision policy HOLD: {', '.join(decision.reason_codes)}",
                trigger_type=trigger_type.value,
                trigger_reason=f"Blocked by HOLD stance (reasons: {decision.reason_codes})",
            )

        # Case 3: Normal rebalancing - generate full execution plan
        return generate_rebalance_plan(
            current_weights=current_weights,
            target_weights=target,
            trigger_type=trigger_type,
            trigger_reason=reason,
            config=self.config,
        )

    def _build_signal_hierarchy(
        self,
        data: Dict,
        scores: ScoreDefinitions,
        inputs: DecisionInputs
    ) -> SignalHierarchyReport:
        """
        Build Signal Hierarchy Report classifying signals as CORE or AUXILIARY.

        Core signals are directly used in decision rules.
        Auxiliary signals are for context and transparency only.
        """
        core_signals = []
        auxiliary_signals = []

        # === CORE SIGNALS ===
        # These are used in resolve_decision() rules

        # 1. Regime signal
        core_signals.append(SignalClassification(
            name='regime_signal',
            signal_type=SignalType.CORE.value,
            value=inputs.regime,
            source='RegimeDetector',
            used_in_rules=['R1', 'R2', 'R5'],
            description='Current market regime (BULL/BEAR/NEUTRAL)'
        ))

        # 2. Regime confidence
        core_signals.append(SignalClassification(
            name='regime_confidence',
            signal_type=SignalType.CORE.value,
            value=inputs.regime_confidence,
            source='RegimeDetector',
            used_in_rules=['R1', 'R2'],
            description='Confidence in regime classification'
        ))

        # 3. Canonical risk score
        core_signals.append(SignalClassification(
            name='canonical_risk_score',
            signal_type=SignalType.CORE.value,
            value=scores.canonical_risk_score,
            source='ScoreDefinitions (unified)',
            used_in_rules=['R3', 'R6'],
            description='Unified risk score (0-100)'
        ))

        # 4. Agent consensus stance
        core_signals.append(SignalClassification(
            name='agent_consensus_stance',
            signal_type=SignalType.CORE.value,
            value=inputs.agent_consensus,
            source='MetaOrchestrator',
            used_in_rules=['R4', 'R5', 'R7'],
            description='Final consensus from agent debate'
        ))

        # 5. Agent consensus confidence
        core_signals.append(SignalClassification(
            name='agent_consensus_confidence',
            signal_type=SignalType.CORE.value,
            value=inputs.confidence,
            source='MetaOrchestrator',
            used_in_rules=['R3', 'R6'],
            description='Confidence in agent consensus'
        ))

        # 6. Constraint status
        core_signals.append(SignalClassification(
            name='constraint_status',
            signal_type=SignalType.CORE.value,
            value=inputs.constraint_status,
            source='ConstraintRepair',
            used_in_rules=['R7'],
            description='Status of asset class constraints'
        ))

        # 7. Client profile status (assumed from data)
        client_status = data.get('client_profile_status', 'COMPLETE')
        core_signals.append(SignalClassification(
            name='client_profile_status',
            signal_type=SignalType.CORE.value,
            value=client_status,
            source='ClientProfile',
            used_in_rules=['R7'],
            description='Client profile availability'
        ))

        # === AUXILIARY SIGNALS ===
        # These are for context only, never used in decision rules

        # 1. Base risk score (component of canonical)
        auxiliary_signals.append(SignalClassification(
            name='aux_base_risk_score',
            signal_type=SignalType.AUXILIARY.value,
            value=scores.base_risk_score,
            source='CriticalPathAggregator',
            description='Base risk from critical path analysis (auxiliary)'
        ))

        # 2. Microstructure adjustment
        auxiliary_signals.append(SignalClassification(
            name='aux_microstructure_adjustment',
            signal_type=SignalType.AUXILIARY.value,
            value=scores.microstructure_adjustment,
            source='MicrostructureAnalyzer',
            description='Market quality adjustment (auxiliary)'
        ))

        # 3. Bubble risk adjustment
        auxiliary_signals.append(SignalClassification(
            name='aux_bubble_risk_adjustment',
            signal_type=SignalType.AUXILIARY.value,
            value=scores.bubble_risk_adjustment,
            source='BubbleDetector',
            description='Bubble risk overlay (auxiliary)'
        ))

        # 4. Full mode position (component of consensus)
        auxiliary_signals.append(SignalClassification(
            name='full_mode_position',
            signal_type=SignalType.AUXILIARY.value,
            value=inputs.full_mode_position,
            source='MetaOrchestrator (full mode)',
            description='Position from full analysis mode (auxiliary)'
        ))

        # 5. Reference mode position (component of consensus)
        auxiliary_signals.append(SignalClassification(
            name='reference_mode_position',
            signal_type=SignalType.AUXILIARY.value,
            value=inputs.reference_mode_position,
            source='MetaOrchestrator (reference mode)',
            description='Position from reference mode (auxiliary)'
        ))

        # 6. Modes agree
        auxiliary_signals.append(SignalClassification(
            name='modes_agree',
            signal_type=SignalType.AUXILIARY.value,
            value=inputs.modes_agree,
            source='DualModeAnalyzer',
            description='Whether full and reference modes agree (auxiliary)'
        ))

        # Add auxiliary metrics from ScoreDefinitions
        for metric in scores.auxiliary_metrics:
            auxiliary_signals.append(SignalClassification(
                name=metric.name,
                signal_type=SignalType.AUXILIARY.value,
                value=metric.value,
                source=metric.source,
                description=f'{metric.description} (auxiliary)'
            ))

        # Validation: ensure only core signals are in CORE_SIGNALS set
        validation_passed = True
        validation_errors = []
        for sig in core_signals:
            if sig.name not in CORE_SIGNALS:
                validation_passed = False
                validation_errors.append(f"Signal '{sig.name}' classified as CORE but not in CORE_SIGNALS set")

        return SignalHierarchyReport(
            core_signals=core_signals,
            auxiliary_signals=auxiliary_signals,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
        )
