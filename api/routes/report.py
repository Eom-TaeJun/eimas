"""
EIMAS Report Generator Router
==============================

Generate professional reports using Perplexity research + Claude writing.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')

# API clients
import anthropic
import httpx

router = APIRouter(tags=["Report"])

# Global state reference
_state: Dict[str, Any] = {"results": {}, "last_analysis_id": None}


def set_state(state: dict):
    """Set state reference from server"""
    global _state
    _state = state


class ReportRequest(BaseModel):
    """Report generation request"""
    analysis_id: Optional[str] = Field(None, description="Analysis ID (uses last if not provided)")
    report_type: str = Field("executive", description="Report type: executive, detailed, or investment")
    language: str = Field("ko", description="Language: ko (Korean) or en (English)")


class ReportResponse(BaseModel):
    """Report generation response"""
    analysis_id: str
    report_type: str
    title: str
    generated_at: str
    research_context: str
    executive_summary: str
    full_report: str
    recommendations: list


async def search_perplexity(query: str) -> str:
    """Search for additional context using Perplexity API"""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "Perplexity API key not found"

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a financial research assistant. Provide concise, factual information with recent data."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": 1000
            }
        )

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Perplexity search failed: {response.status_code}"


async def generate_with_claude(prompt: str, system: str = "") -> str:
    """Generate content using Claude API"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Anthropic API key not found"

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=system if system else "You are an expert financial analyst and report writer.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return message.content[0].text


def extract_analysis_summary(result: Dict) -> Dict[str, Any]:
    """Extract key information from analysis result"""
    stages = result.get('stages', {})

    # Data collection
    dc = stages.get('data_collection', {}).get('result', {})
    fred_data = dc.get('fred_data', {})

    # Top-down
    td = stages.get('top_down_analysis', {}).get('result', {})

    # Core analysis
    ca = stages.get('core_analysis', {}).get('result', {})

    # Interpretation
    interp = stages.get('interpretation', {}).get('result', {})

    # Strategy
    strat = stages.get('strategy_generation', {}).get('result', {})

    return {
        "question": interp.get('topic', 'Economic Analysis'),
        "fed_rate": fred_data.get('DFF'),
        "ten_year": fred_data.get('DGS10'),
        "vix": dc.get('volatility', {}).get('vix'),
        "inflation": dc.get('indicators', {}).get('inflation'),
        "stance": td.get('final_stance'),
        "confidence": td.get('total_confidence'),
        "recommendation": td.get('final_recommendation'),
        "methodology": ca.get('methodology'),
        "key_findings": ca.get('key_findings', []),
        "statistics": ca.get('statistics', {}),
        "divergence_points": interp.get('divergence_points', []),
        "strategy_stance": strat.get('stance'),
        "strategy_recommendation": strat.get('recommendation')
    }


@router.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """
    Generate a professional report from analysis results.

    Uses:
    1. Perplexity for additional market research and context
    2. Claude for professional report writing

    Args:
        request: Report generation request

    Returns:
        ReportResponse with full report content
    """
    # Get analysis result
    target_id = request.analysis_id or _state.get("last_analysis_id")

    if not target_id:
        raise HTTPException(status_code=404, detail="No analysis found")

    if target_id not in _state.get("results", {}):
        raise HTTPException(status_code=404, detail=f"Analysis {target_id} not found")

    result = _state["results"][target_id]

    # Extract summary
    summary = extract_analysis_summary(result.__dict__ if hasattr(result, '__dict__') else result)

    # Step 1: Research with Perplexity
    research_query = f"""
    2025년 현재 Fed 금리 정책과 주식시장 전망에 대해 알려줘:
    1. 현재 Fed 기준금리와 향후 전망
    2. 최근 주식시장 동향
    3. 주요 투자은행들의 2025년 전망
    4. 주요 리스크 요인
    """

    research_context = await search_perplexity(research_query)

    # Step 2: Generate report with Claude
    report_prompt = f"""
다음 분석 결과를 바탕으로 {"한국어로" if request.language == "ko" else "in English"} 전문적인 투자 보고서를 작성해주세요.

## 분석 데이터
- 질문: {summary['question']}
- Fed 기준금리: {summary['fed_rate']}%
- 10년물 금리: {summary['ten_year']}%
- VIX: {summary['vix']}
- 인플레이션: {summary['inflation']}%
- 분석 스탠스: {summary['stance']}
- 신뢰도: {summary['confidence']}
- 분석 방법론: {summary['methodology']}

## 주요 발견
{json.dumps(summary['key_findings'], ensure_ascii=False, indent=2)}

## 통계
{json.dumps(summary['statistics'], ensure_ascii=False, indent=2)}

## 경제학파별 해석
{json.dumps(summary['divergence_points'], ensure_ascii=False, indent=2)}

## 추천
{summary['recommendation']}

## 최신 시장 리서치 (Perplexity)
{research_context}

---

보고서 형식:
1. Executive Summary (3-4 문장)
2. 시장 환경 분석
3. 핵심 분석 결과
4. 투자 전략 권고
5. 리스크 요인
6. 결론

전문적이고 명확하게 작성해주세요.
"""

    system_prompt = """당신은 Goldman Sachs, Morgan Stanley 수준의 전문 투자 애널리스트입니다.
데이터 기반의 객관적인 분석을 제공하고, 명확한 투자 권고를 합니다.
보고서는 기관투자자가 읽을 수 있는 수준으로 작성합니다."""

    full_report = await generate_with_claude(report_prompt, system_prompt)

    # Extract executive summary (first paragraph)
    exec_summary = full_report.split('\n\n')[0] if full_report else ""

    # Extract recommendations
    recommendations = []
    if summary['key_findings']:
        recommendations = summary['key_findings']

    return ReportResponse(
        analysis_id=target_id,
        report_type=request.report_type,
        title=f"투자 분석 보고서: {summary['question']}",
        generated_at=datetime.now().isoformat(),
        research_context=research_context,
        executive_summary=exec_summary,
        full_report=full_report,
        recommendations=recommendations
    )


@router.get("/report/{analysis_id}")
async def get_report(
    analysis_id: str,
    language: str = Query("ko", description="Language: ko or en")
):
    """
    Generate report for specific analysis ID.
    """
    request = ReportRequest(analysis_id=analysis_id, language=language)
    return await generate_report(request)
