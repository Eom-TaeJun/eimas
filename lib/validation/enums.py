#!/usr/bin/env python3
"""
Validation - Enumerations
============================================================

Validation result types
"""

from enum import Enum


class ValidationResult(Enum):
    """검증 결과 타입"""
    APPROVE = "APPROVE"           # 승인
    REJECT = "REJECT"             # 거부
    MODIFY = "MODIFY"             # 수정 필요
    NEEDS_INFO = "NEEDS_INFO"     # 추가 정보 필요
