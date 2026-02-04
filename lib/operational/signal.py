#!/usr/bin/env python3
"""
Operational - Signal Classification
============================================================

시그널 분류 및 계층 구조
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


@dataclass
class SignalClassification:
    """개별 시그널 분류"""
    name: str
    signal_type: str              # CORE / AUXILIARY
    value: Any
    source: str
    used_in_rules: List[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'signal_type': self.signal_type,
            'value': str(self.value) if not isinstance(self.value, (int, float, bool, str)) else self.value,
            'source': self.source,
            'used_in_rules': self.used_in_rules,
            'description': self.description,
        }


@dataclass
class SignalHierarchyReport:
    """
    시그널 계층 구조 리포트

    Documents which signals are Core vs Auxiliary and ensures
    only Core signals influence decisions.
    """
    core_signals: List[SignalClassification] = field(default_factory=list)
    auxiliary_signals: List[SignalClassification] = field(default_factory=list)
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'core_signals': [s.to_dict() for s in self.core_signals],
            'auxiliary_signals': [s.to_dict() for s in self.auxiliary_signals],
            'validation': {
                'passed': self.validation_passed,
                'errors': self.validation_errors,
            }
        }

    def to_markdown(self) -> str:
        """Generate signal_hierarchy section for report"""
        lines = []
        lines.append("## signal_hierarchy")
        lines.append("")

        # Validation status
        if self.validation_passed:
            lines.append("**Validation: PASSED** ✓")
            lines.append("")
            lines.append("Only Core signals were used in decision rules.")
        else:
            lines.append("**Validation: FAILED** ✗")
            lines.append("")
            for error in self.validation_errors:
                lines.append(f"- ⚠️ {error}")
        lines.append("")

        # Core Signals
        lines.append("### Core Signals (Used in Decision Rules)")
        lines.append("")
        lines.append("| Signal | Value | Source | Used In |")
        lines.append("|--------|-------|--------|---------|")
        for s in self.core_signals:
            rules = ", ".join(s.used_in_rules) if s.used_in_rules else "—"
            value_str = f"{s.value:.2f}" if isinstance(s.value, float) else str(s.value)
            lines.append(f"| {s.name} | {value_str} | {s.source} | {rules} |")
        lines.append("")

        # Auxiliary Signals
        lines.append("### Auxiliary Signals (Context Only)")
        lines.append("")
        lines.append("**Note:** These signals are for transparency only. They do NOT influence decisions.")
        lines.append("")
        lines.append("| Signal | Value | Source | Description |")
        lines.append("|--------|-------|--------|-------------|")
        for s in self.auxiliary_signals:
            value_str = f"{s.value:.2f}" if isinstance(s.value, float) else str(s.value)
            lines.append(f"| {s.name} | {value_str} | {s.source} | {s.description} |")
        lines.append("")

        return "\n".join(lines)

