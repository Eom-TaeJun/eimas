#!/usr/bin/env python3
"""
lib/report_generator.py — Shim (패키지 호환 레이어)
=====================================================
실제 구현은 lib/reports/ 패키지로 통합됨.
이 파일은 기존 import 경로 호환성 유지용입니다.

    from lib.report_generator import ReportGenerator  # 기존 경로 (유지)
    from lib.reports import ReportGenerator           # 신규 경로 (권장)
"""
# 실제 구현체를 직접 import (순환 방지를 위해 reports/__init__ 경유 안 함)
from lib.ai_report_generator import AIReportGenerator


class ReportGenerator:
    """
    레거시 호환 래퍼.
    신규 코드에서는 lib.reports.AIReportGenerator 또는
    lib.ai_report_generator.AIReportGenerator를 직접 사용하세요.
    """
    def __init__(self, *args, **kwargs):
        self._agent = AIReportGenerator(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._agent, name)


__all__ = ["ReportGenerator"]
