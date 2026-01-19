#!/usr/bin/env python3
"""
EIMAS Pipeline Reporting
=========================
Phase 6: AI 리포트 생성 모듈
"""

import asyncio
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime

# EIMAS 라이브러리
from lib.ai_report_generator import AIReportGenerator
from pipeline.schemas import EIMASResult, AIReport

async def generate_ai_report(result: EIMASResult, market_data: Dict[str, pd.DataFrame]) -> AIReport:
    """
    AI 투자 제안서 및 리포트 생성
    
    Args:
        result: 통합 분석 결과
        market_data: 분석에 사용된 시장 데이터
        
    Returns:
        AIReport: 생성된 리포트 정보
    """
    print("\n" + "=" * 50)
    print("PHASE 6: AI REPORT GENERATION")
    print("=" * 50)

    try:
        generator = AIReportGenerator(verbose=True)

        print("\n[6.1] Generating AI-powered investment report...")
        # AIReportGenerator는 dict 형태의 결과를 받으므로 변환하여 전달
        report = await generator.generate(result.to_dict(), market_data)

        print("\n[6.2] Saving report...")
        report_path = await generator.save_report(report)
        print(f"      ✓ AI Report saved: {report_path}")

        print("\n[6.3] Generating IB-style Memorandum...")
        ib_report_content = await generator.generate_ib_report(result.to_dict(), market_data)
        ib_report_path = await generator.save_ib_report(ib_report_content)
        print(f"      ✓ IB Memorandum saved: {ib_report_path}")

        # 하이라이트 정보 추출
        highlights = {
            'notable_stocks': [
                {'ticker': s.ticker, 'reason': s.notable_reason} 
                for s in getattr(report, 'notable_stocks', [])[:3]
            ],
            'final_recommendation': getattr(report, 'final_recommendation', "")[:200] + "..."
        }

        return AIReport(
            timestamp=datetime.now().isoformat(),
            report_path=str(report_path),
            ib_report_path=str(ib_report_path),
            highlights=highlights,
            content=getattr(report, 'final_recommendation', "")
        )

    except Exception as e:
        print(f"      ✗ AI Report error: {e}")
        return AIReport(
            timestamp=datetime.now().isoformat(),
            report_path="",
            ib_report_path="",
            highlights={'error': str(e)}
        )
