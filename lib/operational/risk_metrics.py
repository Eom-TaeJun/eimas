#!/usr/bin/env python3
"""
Operational - Risk Metrics
============================================================

리스크 점수 정의 및 보조 메트릭
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

# Risk threshold constants
RISK_THRESHOLD_LOW = 30.0      # 0-30: LOW risk
RISK_THRESHOLD_HIGH = 70.0     # 70-100: HIGH risk


@dataclass
class AuxiliaryRiskMetric:
    """
    보조 리스크 지표 (auxiliary, not used for decisions)

    이 지표들은 정보 제공 목적으로만 사용되며,
    의사결정에는 canonical_risk_score만 사용됨.
    """
    name: str                  # 지표 이름
    value: float               # 지표 값
    unit: str                  # 단위 (points, %, bps 등)
    source: str                # 출처 모듈
    description: str           # 설명

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'source': self.source,
            'description': self.description,
        }


@dataclass
class ScoreDefinitions:
    """
    통합 리스크 점수 정의 (Unified Risk Scoring)

    ============================================================================
    CANONICAL RISK SCORE (의사결정용 공식 점수)
    ============================================================================

    Scale: 0 - 100 (고정)
    Formula: canonical = base + microstructure_adj + bubble_adj + extended_adj
    Clamping: min(100, max(0, canonical))

    Interpretation Bands:
    ---------------------
    | Range   | Level  | Action Guidance                              |
    |---------|--------|----------------------------------------------|
    | 0-30    | LOW    | 공격적 포지션 허용                           |
    | 30-70   | MEDIUM | 표준 리스크 관리 적용                        |
    | 70-100  | HIGH   | 방어적 스탠스, 포지션 축소 고려              |

    Decision Rule Usage:
    --------------------
    - RULE_4 (High Risk Override): canonical_risk_score >= 70 → 방어적 스탠스
    - resolve_final_decision(): canonical_risk_score만 참조

    ============================================================================
    AUXILIARY SUB-SCORES (정보 제공용, 의사결정에 미사용)
    ============================================================================

    | Sub-Score                | Range   | Source                    |
    |--------------------------|---------|---------------------------|
    | base_risk_score          | 0-100   | CriticalPathAggregator    |
    | microstructure_adjustment| ±10     | DailyMicrostructureAnalyzer|
    | bubble_risk_adjustment   | +0~15   | BubbleDetector            |
    | extended_data_adjustment | ±15     | PCR/Sentiment/Credit      |

    Note: Sub-scores are ONLY for transparency and audit.
          All decision logic uses canonical_risk_score exclusively.
    """

    # =========================================================================
    # CANONICAL RISK SCORE (의사결정용 - 이 값만 decision rules에서 사용)
    # =========================================================================
    canonical_risk_score: float = 50.0     # Scale: 0-100, clamped

    # =========================================================================
    # INTERPRETATION (canonical_risk_score 기반)
    # =========================================================================
    risk_level: str = "MEDIUM"             # LOW (0-30) / MEDIUM (30-70) / HIGH (70-100)
    risk_level_threshold_low: float = field(default=RISK_THRESHOLD_LOW, repr=False)
    risk_level_threshold_high: float = field(default=RISK_THRESHOLD_HIGH, repr=False)

    # =========================================================================
    # AUXILIARY SUB-SCORES (정보 제공용 - 의사결정에 사용 금지)
    # =========================================================================
    # Note: These are labeled "auxiliary" to emphasize they are not for decisions

    # Base score from CriticalPathAggregator
    aux_base_risk_score: float = 50.0
    aux_base_source: str = "CriticalPathAggregator"

    # Microstructure adjustment (market quality)
    aux_microstructure_adjustment: float = 0.0
    aux_microstructure_source: str = "DailyMicrostructureAnalyzer"

    # Bubble risk overlay
    aux_bubble_risk_adjustment: float = 0.0
    aux_bubble_source: str = "BubbleDetector"

    # Extended data adjustment (PCR, sentiment, credit)
    aux_extended_data_adjustment: float = 0.0
    aux_extended_source: str = "ExtendedDataCollector"

    # =========================================================================
    # OPTIONAL: Additional auxiliary metrics (for information only)
    # =========================================================================
    auxiliary_metrics: List[AuxiliaryRiskMetric] = field(default_factory=list)

    def __post_init__(self):
        """Calculate risk_level from canonical_risk_score"""
        self.risk_level = self._interpret_risk_level(self.canonical_risk_score)

    def _interpret_risk_level(self, score: float) -> str:
        """
        Interpret canonical risk score into risk level.

        Fixed bands (not configurable to ensure consistency):
        - LOW: 0-30
        - MEDIUM: 30-70
        - HIGH: 70-100
        """
        if score < self.risk_level_threshold_low:
            return "LOW"
        elif score >= self.risk_level_threshold_high:
            return "HIGH"
        else:
            return "MEDIUM"

    @classmethod
    def from_components(
        cls,
        base_risk_score: float,
        microstructure_adjustment: float = 0.0,
        bubble_risk_adjustment: float = 0.0,
        extended_data_adjustment: float = 0.0,
        auxiliary_metrics: List[AuxiliaryRiskMetric] = None
    ) -> 'ScoreDefinitions':
        """
        Create ScoreDefinitions from component scores.

        Computes canonical_risk_score = base + micro + bubble + extended,
        clamped to [0, 100].
        """
        canonical = base_risk_score + microstructure_adjustment + bubble_risk_adjustment + extended_data_adjustment
        canonical = max(0.0, min(100.0, canonical))

        return cls(
            canonical_risk_score=canonical,
            aux_base_risk_score=base_risk_score,
            aux_microstructure_adjustment=microstructure_adjustment,
            aux_bubble_risk_adjustment=bubble_risk_adjustment,
            aux_extended_data_adjustment=extended_data_adjustment,
            auxiliary_metrics=auxiliary_metrics or [],
        )

    def to_dict(self) -> Dict:
        """Export as dictionary for JSON serialization"""
        return {
            # Canonical score (for decisions)
            'canonical_risk_score': self.canonical_risk_score,
            'risk_level': self.risk_level,
            'scale': {
                'min': 0,
                'max': 100,
                'interpretation': {
                    'LOW': f'0 - {self.risk_level_threshold_low}',
                    'MEDIUM': f'{self.risk_level_threshold_low} - {self.risk_level_threshold_high}',
                    'HIGH': f'{self.risk_level_threshold_high} - 100',
                }
            },
            # Auxiliary sub-scores (for transparency only)
            'auxiliary_sub_scores': {
                'base_risk_score': {
                    'value': self.aux_base_risk_score,
                    'source': self.aux_base_source,
                    'note': 'AUXILIARY - not used for decisions'
                },
                'microstructure_adjustment': {
                    'value': self.aux_microstructure_adjustment,
                    'source': self.aux_microstructure_source,
                    'note': 'AUXILIARY - not used for decisions'
                },
                'bubble_risk_adjustment': {
                    'value': self.aux_bubble_risk_adjustment,
                    'source': self.aux_bubble_source,
                    'note': 'AUXILIARY - not used for decisions'
                },
                'extended_data_adjustment': {
                    'value': self.aux_extended_data_adjustment,
                    'source': self.aux_extended_source,
                    'note': 'AUXILIARY - not used for decisions'
                },
            },
            'formula': 'canonical = base + microstructure_adj + bubble_adj + extended_adj',
            'additional_auxiliary_metrics': [m.to_dict() for m in self.auxiliary_metrics],
        }

    def to_markdown(self) -> str:
        """
        Generate score_definitions section for report.

        This section documents the unified risk scoring system.
        """
        lines = []
        lines.append("## score_definitions")
        lines.append("")

        # Canonical Score
        lines.append("### Canonical Risk Score")
        lines.append("")
        lines.append(f"**Score: {self.canonical_risk_score:.1f} / 100**")
        lines.append(f"**Level: {self.risk_level}**")
        lines.append("")

        # Scale interpretation
        lines.append("### Scale Interpretation")
        lines.append("")
        lines.append("| Range | Level | Description |")
        lines.append("|-------|-------|-------------|")
        lines.append(f"| 0 - {self.risk_level_threshold_low:.0f} | LOW | 공격적 포지션 허용 |")
        lines.append(f"| {self.risk_level_threshold_low:.0f} - {self.risk_level_threshold_high:.0f} | MEDIUM | 표준 리스크 관리 |")
        lines.append(f"| {self.risk_level_threshold_high:.0f} - 100 | HIGH | 방어적 스탠스 권고 |")
        lines.append("")

        # Formula
        lines.append("### Formula")
        lines.append("")
        lines.append("```")
        lines.append("canonical_risk_score = base + microstructure_adj + bubble_adj + extended_adj")
        lines.append(f"                     = {self.aux_base_risk_score:.1f} + ({self.aux_microstructure_adjustment:+.1f}) + ({self.aux_bubble_risk_adjustment:+.1f}) + ({self.aux_extended_data_adjustment:+.1f})")
        lines.append(f"                     = {self.canonical_risk_score:.1f}")
        lines.append("```")
        lines.append("")

        # Auxiliary sub-scores
        lines.append("### Auxiliary Sub-Scores")
        lines.append("")
        lines.append("**Note:** These sub-scores are for transparency only. Decision rules use ONLY `canonical_risk_score`.")
        lines.append("")
        lines.append("| Sub-Score | Value | Source | Purpose |")
        lines.append("|-----------|-------|--------|---------|")
        lines.append(f"| base_risk_score | {self.aux_base_risk_score:.1f} | {self.aux_base_source} | 기본 리스크 |")
        lines.append(f"| microstructure_adj | {self.aux_microstructure_adjustment:+.1f} | {self.aux_microstructure_source} | 시장 미세구조 |")
        lines.append(f"| bubble_risk_adj | {self.aux_bubble_risk_adjustment:+.1f} | {self.aux_bubble_source} | 버블 리스크 |")
        lines.append(f"| extended_data_adj | {self.aux_extended_data_adjustment:+.1f} | {self.aux_extended_source} | PCR/감성/신용 |")
        lines.append("")

        # Additional auxiliary metrics
        if self.auxiliary_metrics:
            lines.append("### Additional Auxiliary Metrics")
            lines.append("")
            lines.append("| Metric | Value | Unit | Source |")
            lines.append("|--------|-------|------|--------|")
            for m in self.auxiliary_metrics:
                lines.append(f"| {m.name} | {m.value:.2f} | {m.unit} | {m.source} |")
            lines.append("")

        # Decision usage note
        lines.append("### Usage in Decision Rules")
        lines.append("")
        lines.append("```")
        lines.append("RULE_4 (High Risk Override):")
        lines.append(f"  IF canonical_risk_score >= {self.risk_level_threshold_high:.0f}")
        lines.append("  THEN apply defensive stance")
        lines.append("")
        lines.append("All other rules reference canonical_risk_score exclusively.")
        lines.append("Sub-scores are NEVER used in decision logic.")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    # Backward compatibility aliases
    @property
    def base_risk_score(self) -> float:
        """Alias for aux_base_risk_score (backward compatibility)"""
        return self.aux_base_risk_score

    @property
    def microstructure_adjustment(self) -> float:
        """Alias for aux_microstructure_adjustment (backward compatibility)"""
        return self.aux_microstructure_adjustment

    @property
    def bubble_risk_adjustment(self) -> float:
        """Alias for aux_bubble_risk_adjustment (backward compatibility)"""
        return self.aux_bubble_risk_adjustment

    @property
    def extended_data_adjustment(self) -> float:
        """Alias for aux_extended_data_adjustment (backward compatibility)"""
        return self.aux_extended_data_adjustment

