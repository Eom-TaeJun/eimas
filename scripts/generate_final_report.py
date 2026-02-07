#!/usr/bin/env python3
"""
EIMAS Final Report Generator
=============================
실행된 분석 결과(JSON)를 바탕으로 AI 기반 심층 리포트를 생성합니다.

기능:
1. 최신 eimas_*.json 로드 (legacy integrated_*.json fallback)
2. AIReportGenerator를 통해 IB 스타일 Memorandum 생성
3. Proof-of-Index, DTW, HFT 등 신규 지표 반영 확인
4. 결과 저장
"""

import json
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ai_report_generator import AIReportGenerator

async def main():
    print("=" * 60)
    print("EIMAS AI Report Generator")
    print("=" * 60)
    
    # 1. 최신 분석 결과 로드
    output_dir = PROJECT_ROOT / "outputs"
    json_files = sorted(output_dir.glob("eimas_*.json"), reverse=True)
    if not json_files:
        json_files = sorted(output_dir.glob("integrated_*.json"), reverse=True)

    if not json_files:
        print("❌ 분석 결과 파일(eimas_*.json)을 찾을 수 없습니다.")
        print("먼저 'python main.py --full'를 실행해주세요.")
        return
        
    latest_file = json_files[0]
    print(f"📂 Loading latest analysis: {latest_file}")
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            analysis_result = json.load(f)
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return

    # 2. 리포트 생성기 초기화
    generator = AIReportGenerator(verbose=True)
    
    # 3. IB 리포트 생성
    print("\n🚀 Generating Investment Banking Memorandum...")
    try:
        # 시장 데이터는 이미 analysis_result에 요약되어 있다고 가정하거나, 
        # 필요시 yfinance로 다시 가져올 수 있으나 여기서는 분석 결과만 활용
        
        # IB 리포트 생성 (내부적으로 _build_ib_prompt -> _format_new_metrics 호출)
        report_content = await generator.generate_ib_report(analysis_result)
        
        if report_content:
            # 4. 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"EIMAS_IB_Memorandum_{timestamp}.md"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            print(f"\n✅ Report generated successfully!")
            print(f"📄 Saved to: {output_file}")
            
            # 내용 미리보기
            print("\n" + "="*60)
            print("REPORT PREVIEW (First 500 chars)")
            print("="*60)
            print(report_content[:500] + "...")
            print("="*60)
            
        else:
            print("❌ 리포트 내용이 비어있습니다.")
            
    except Exception as e:
        print(f"❌ 리포트 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
