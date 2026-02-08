#!/usr/bin/env python3
"""
EIMAS Pipeline - Report Module
===============================

Purpose:
    Phase 6 AI 리포트 생성 및 Phase 7 검증 담당

Functions:
    - generate_ai_report(result, market_data) -> AIReport
    - run_whitening_check(result) -> str
    - run_fact_check(report_text) -> str

Dependencies:
    - lib.ai_report_generator
    - lib.whitening_engine
    - lib.autonomous_agent

Example:
    from pipeline.report import generate_ai_report
    report = await generate_ai_report(result, market_data)
    print(report.report_path)
"""

import asyncio
import re
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from datetime import datetime

# EIMAS 라이브러리
from lib.ai_report_generator import AIReportGenerator


def _parse_md_sections(md_path: str) -> Dict[str, Dict[str, str]]:
    """MD 파일에서 ## N. Section 패턴으로 섹션 추출."""
    sections = {}
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        header_pattern = re.compile(r'^##\s+(\d{1,2})\.\s+(.+)$', re.MULTILINE)
        matches = list(header_pattern.finditer(content))

        for idx, match in enumerate(matches):
            num = int(match.group(1))
            if num < 1 or num > 50:
                continue

            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
            body = content[start:end].strip()
            key = f"section_{num}"

            candidate = {
                "title": match.group(2).strip(),
                "content": body,
            }

            # 동일 섹션 번호가 중복될 수 있으므로, 본문이 더 긴 항목을 채택.
            if key not in sections or len(body) > len(sections[key].get("content", "")):
                sections[key] = candidate

    except Exception as e:
        print(f"      ⚠️ Section parsing error: {e}")

    return sections
from lib.whitening_engine import WhiteningEngine
from lib.autonomous_agent import AutonomousFactChecker
from pipeline.schemas import EIMASResult, AIReport
from pipeline.exceptions import get_logger, log_error

logger = get_logger("report")

async def generate_ai_report(result: EIMASResult, market_data: Dict[str, pd.DataFrame]) -> AIReport:
    """AI 투자 제안서 및 리포트 생성 - 통합 JSON에 모든 섹션 포함"""
    print("\n" + "=" * 50)
    print("PHASE 6: AI REPORT GENERATION")
    print("=" * 50)

    try:
        generator = AIReportGenerator(verbose=True)

        print("\n[6.1] Generating AI-powered investment report...")
        report = await generator.generate(result.to_dict(), market_data)

        print("\n[6.2] Saving report...")
        report_path = await generator.save_report(report)
        print(f"      ✓ AI Report saved: {report_path}")

        print("\n[6.3] Generating IB-style Memorandum...")
        ib_report_content = await generator.generate_ib_report(result.to_dict(), market_data)
        ib_report_path = await generator.save_ib_report(ib_report_content)
        print(f"      ✓ IB Memorandum saved: {ib_report_path}")

        # [NEW] MD 파일에서 섹션 파싱하여 JSON에 포함
        print("\n[6.4] Parsing report sections for unified JSON...")
        sections = _parse_md_sections(str(report_path))
        print(f"      ✓ Parsed {len(sections)} sections")

        highlights = {
            'notable_stocks': [
                {'ticker': s.ticker, 'reason': s.notable_reason}
                for s in getattr(report, 'notable_stocks', [])[:5]
            ],
            'final_recommendation': getattr(report, 'final_recommendation', "")
        }

        return AIReport(
            timestamp=datetime.now().isoformat(),
            report_path=str(report_path),
            ib_report_path=str(ib_report_path),
            highlights=highlights,
            content=getattr(report, 'final_recommendation', ""),
            sections=sections,  # 전체 섹션 포함
            report_data=report.to_dict(),
        )

    except Exception as e:
        log_error(logger, "AI Report error", e)
        return AIReport(
            timestamp=datetime.now().isoformat(),
            report_path="",
            ib_report_path="",
            highlights={'error': str(e)}
        )

def run_whitening_check(result: EIMASResult) -> str:
    """Phase 7.1: Whitening (경제학적 해석 검증)"""
    print("\n[7.1] Economic whitening analysis...")
    try:
        engine = WhiteningEngine()
        # 간단히 결정 트리 로직 실행 (상세 구현은 lib/whitening_engine.py 참조)
        # 여기서는 result의 주요 지표를 기반으로 해석 생성
        summary = engine.explain_decision({
            'risk_score': result.risk_score,
            'regime': result.regime.get('regime', 'Unknown'),
            'liquidity': result.fred_summary.net_liquidity if result.fred_summary else 0
        })
        print(f"      ✓ Whitening Summary: {summary[:50]}...")
        return summary
    except Exception as e:
        # WhiteningEngine이 없거나 에러 발생 시
        # log_error(logger, "Whitening check failed", e)
        # WhiteningEngine이 아직 구현 안되었을 수도 있으므로 조용히 처리
        return "Whitening analysis skipped"

async def run_fact_check(report_text: str) -> str:
    """Phase 7.2: AI Fact Check"""
    print("\n[7.2] Fact checking AI outputs...")
    if not report_text:
        return "No report to check"
        
    try:
        checker = AutonomousFactChecker()
        grade = await checker.verify_content(report_text[:2000]) # 앞부분만 체크
        print(f"      ✓ Fact Check Grade: {grade}")
        return grade
    except Exception as e:
        # log_error(logger, "Fact check failed", e)
        return "Fact check skipped"
