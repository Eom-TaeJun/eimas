#!/usr/bin/env python3
"""
Causality Analysis - Enums
============================================================

Causality types and edge classifications

Economic Foundation:
    - Granger (1969) causality testing
    - Causal inference in time series
"""

from enum import Enum


class EdgeType(Enum):
    """엣지 유형"""
    SUPPLY_DEPENDENCY = "supply_dependency"      # 공급망 의존성
    PRICE_CORRELATION = "price_correlation"      # 가격 상관관계
    GRANGER_CAUSALITY = "granger_causality"      # Granger 인과관계
    SECTOR_LINKAGE = "sector_linkage"            # 섹터 연결
    EXTERNAL_SHOCK = "external_shock"            # 외부 충격


class NodeType(Enum):
    """노드 유형"""
    COMPANY = "company"
    SECTOR = "sector"
    COMMODITY = "commodity"
    MACRO_INDICATOR = "macro_indicator"
    EXTERNAL_EVENT = "external_event"


class CausalDirection(Enum):
    """인과 방향"""
    X_TO_Y = "x_causes_y"       # X → Y
    Y_TO_X = "y_causes_x"       # Y → X
    BIDIRECTIONAL = "bidirectional"  # X ↔ Y
    NO_CAUSALITY = "no_causality"

