#!/bin/bash
echo "========================================================"
echo "   EIMAS: Full Analysis & AI Report Generation Pipeline"
echo "========================================================"

# 1. 가상환경 활성화 (필요시)
# source venv/bin/activate

# 2. 전체 데이터 분석 실행
echo "Step 1: Running Quantitative Analysis..."
python run_full_analysis.py
if [ $? -ne 0 ]; then
    echo "❌ Analysis failed. Aborting."
    exit 1
fi

# 3. AI 리포트 생성
echo "Step 2: Generating AI Investment Memorandum..."
python generate_final_report.py

echo "========================================================"
echo "✅ All tasks completed successfully!"
echo "   Check 'outputs/' directory for results."
echo "========================================================"
