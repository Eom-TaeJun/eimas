"""
EIMAS Reports — 리포트 생성 패키지
====================================
모든 리포트 관련 모듈의 단일 진입점.

기존 경로(from lib.ai_report_generator import ...)는 그대로 유지됨.
신규 권장 경로: from lib.reports import AIReportGenerator
"""

from .ai_report_generator import AIReportGenerator
from .final_report_agent import FinalReportAgent
from .allocation_report_agent import AllocationReportAgent
from .report_generator import ReportGenerator
from .whitening_engine import WhiteningEngine
from .json_to_html_converter import convert_json_to_html
from .json_to_md_converter import convert_json_to_md
from .business_summary import generate_business_summary, BusinessSummary

__all__ = [
    'AIReportGenerator',
    'FinalReportAgent',
    'AllocationReportAgent',
    'ReportGenerator',
    'WhiteningEngine',
    'convert_json_to_html',
    'convert_json_to_md',
    'generate_business_summary',
    'BusinessSummary',
]
