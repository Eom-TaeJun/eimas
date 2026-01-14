"""
Multi-LLM Insight Discussion System
====================================
여러 LLM 모델을 활용한 인사이트 추출 및 토론 시스템

지원 모델:
- Perplexity (sonar-pro): 실시간 검색 기반 인사이트
- Claude (claude-opus-4-5-20251101): 심층 분석
- Gemini (gemini-2.0-flash-exp): 빠른 패턴 인식
- OpenAI (o1-mini / gpt-4o): 구조적 추론
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


class LLMProvider(Enum):
    """LLM 제공자"""
    PERPLEXITY = "perplexity"
    CLAUDE = "claude"
    GEMINI = "gemini"
    OPENAI = "openai"


@dataclass
class LLMInsight:
    """개별 LLM의 인사이트"""
    provider: LLMProvider
    model: str
    topic: str
    insight: str
    confidence: float  # 0.0 ~ 1.0
    key_points: List[str] = field(default_factory=list)
    risks_identified: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    disagreements: List[str] = field(default_factory=list)  # 다른 모델과의 의견 차이
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class DiscussionResult:
    """토론 결과"""
    topic: str
    insights: List[LLMInsight]
    consensus_points: List[str]  # 모든 모델이 동의하는 포인트
    divergence_points: List[str]  # 의견 차이가 있는 포인트
    final_synthesis: str  # 최종 종합
    actionable_items: List[str]  # 실행 가능한 항목
    confidence_score: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MultiLLMDiscussion:
    """멀티 LLM 토론 시스템"""

    # 최신 모델 설정
    MODELS = {
        LLMProvider.PERPLEXITY: "sonar-pro",
        LLMProvider.CLAUDE: "claude-opus-4-5-20251101",
        LLMProvider.GEMINI: "gemini-2.0-flash-exp",
        LLMProvider.OPENAI: "gpt-4o"
    }

    def __init__(self):
        self._clients = {}
        self._validate_api_keys()

    def _validate_api_keys(self) -> Dict[str, bool]:
        """API 키 검증"""
        keys = {
            LLMProvider.PERPLEXITY: os.getenv('PERPLEXITY_API_KEY'),
            LLMProvider.CLAUDE: os.getenv('ANTHROPIC_API_KEY'),
            LLMProvider.GEMINI: os.getenv('GEMINI_API_KEY'),
            LLMProvider.OPENAI: os.getenv('OPENAI_API_KEY')
        }
        self.available_providers = {k: bool(v) for k, v in keys.items()}
        return self.available_providers

    def _get_client(self, provider: LLMProvider):
        """API 클라이언트 반환"""
        if provider in self._clients:
            return self._clients[provider]

        if provider == LLMProvider.PERPLEXITY:
            from openai import OpenAI
            self._clients[provider] = OpenAI(
                api_key=os.getenv('PERPLEXITY_API_KEY'),
                base_url="https://api.perplexity.ai"
            )

        elif provider == LLMProvider.CLAUDE:
            import anthropic
            self._clients[provider] = anthropic.Anthropic(
                api_key=os.getenv('ANTHROPIC_API_KEY')
            )

        elif provider == LLMProvider.GEMINI:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            self._clients[provider] = genai.GenerativeModel(self.MODELS[provider])

        elif provider == LLMProvider.OPENAI:
            from openai import OpenAI
            self._clients[provider] = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        return self._clients[provider]

    async def get_insight(
        self,
        provider: LLMProvider,
        topic: str,
        context: Dict[str, Any],
        other_insights: Optional[List[LLMInsight]] = None
    ) -> Optional[LLMInsight]:
        """개별 LLM에서 인사이트 추출"""

        if not self.available_providers.get(provider):
            print(f"[WARN] {provider.value} API key not available")
            return None

        # 컨텍스트 요약
        context_summary = self._summarize_context(context)

        # 다른 인사이트가 있으면 참조
        other_views = ""
        if other_insights:
            other_views = "\n\n### 다른 모델들의 의견:\n"
            for insight in other_insights:
                other_views += f"- {insight.provider.value}: {insight.insight[:200]}...\n"

        prompt = f"""당신은 금융 시장 분석 전문가입니다.
다음 데이터를 분석하고 인사이트를 제공해주세요.

### 토픽: {topic}

### 현재 시장 데이터:
{context_summary}
{other_views}

### 요청사항:
1. 핵심 인사이트 (3-5문장)
2. 주요 포인트 (3-5개 bullet points)
3. 식별된 리스크 (있다면)
4. 기회 요인 (있다면)
5. 신뢰도 (0.0-1.0)
6. 다른 의견과 다른 점 (있다면)

JSON 형식으로 응답해주세요:
{{
    "insight": "핵심 인사이트...",
    "key_points": ["포인트1", "포인트2", ...],
    "risks": ["리스크1", ...],
    "opportunities": ["기회1", ...],
    "confidence": 0.8,
    "disagreements": ["다른 점1", ...]
}}"""

        try:
            response = await self._call_llm(provider, prompt)
            return self._parse_insight(provider, topic, response)
        except Exception as e:
            print(f"[ERROR] {provider.value}: {e}")
            return None

    async def _call_llm(self, provider: LLMProvider, prompt: str) -> str:
        """LLM API 호출"""
        client = self._get_client(provider)

        if provider == LLMProvider.PERPLEXITY:
            response = client.chat.completions.create(
                model=self.MODELS[provider],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return response.choices[0].message.content

        elif provider == LLMProvider.CLAUDE:
            response = client.messages.create(
                model=self.MODELS[provider],
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif provider == LLMProvider.GEMINI:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: client.generate_content(prompt)
            )
            return response.text

        elif provider == LLMProvider.OPENAI:
            response = client.chat.completions.create(
                model=self.MODELS[provider],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return response.choices[0].message.content

        return ""

    def _summarize_context(self, context: Dict[str, Any]) -> str:
        """컨텍스트 요약"""
        summary_parts = []

        if 'market_summary' in context:
            summary_parts.append(f"시장 요약: {context['market_summary']}")

        if 'regime_analysis' in context:
            summary_parts.append(f"레짐 분석: {context['regime_analysis']}")

        if 'technical_indicators' in context:
            ti = context['technical_indicators']
            summary_parts.append(f"""기술 지표:
- VIX: {ti.get('vix', 'N/A')} (변동: {ti.get('vix_change', 'N/A')})
- RSI: {ti.get('rsi_14', 'N/A')}
- MACD: {ti.get('macd', 'N/A')}
- 현재가: {ti.get('current_price', 'N/A')}
- 지지선: {ti.get('support_level', 'N/A')}
- 저항선: {ti.get('resistance_level', 'N/A')}""")

        if 'global_market' in context:
            gm = context['global_market']
            summary_parts.append(f"""글로벌 시장:
- DXY: {gm.get('dxy', 'N/A')}
- Nikkei: {gm.get('nikkei', 'N/A')} ({gm.get('nikkei_change', 'N/A'):.1f}%)
- DAX: {gm.get('dax', 'N/A')} ({gm.get('dax_change', 'N/A'):.1f}%)
- Gold: {gm.get('gold', 'N/A')}
- WTI: {gm.get('wti', 'N/A')}
- 글로벌 센티먼트: {gm.get('global_sentiment', 'N/A')}""")

        if 'scenarios' in context:
            scenarios = context['scenarios']
            scenario_str = "시나리오:\n"
            for s in scenarios:
                scenario_str += f"- {s.get('name', 'N/A')}: {s.get('probability', 'N/A')}% ({s.get('expected_return', 'N/A')})\n"
            summary_parts.append(scenario_str)

        if 'sector_recommendations' in context:
            sr = context['sector_recommendations']
            bullish = [s['name'] for s in sr.get('bullish_sectors', [])]
            bearish = [s['name'] for s in sr.get('bearish_sectors', [])]
            summary_parts.append(f"""섹터 권고:
- 유망: {', '.join(bullish)}
- 주의: {', '.join(bearish)}""")

        return "\n\n".join(summary_parts)

    def _parse_insight(self, provider: LLMProvider, topic: str, response: str) -> Optional[LLMInsight]:
        """응답 파싱"""
        try:
            # JSON 추출 시도
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # JSON 없으면 텍스트 그대로 사용
                return LLMInsight(
                    provider=provider,
                    model=self.MODELS[provider],
                    topic=topic,
                    insight=response[:500],
                    confidence=0.5,
                    key_points=[],
                    risks_identified=[],
                    opportunities=[],
                    disagreements=[]
                )

            return LLMInsight(
                provider=provider,
                model=self.MODELS[provider],
                topic=topic,
                insight=data.get('insight', ''),
                confidence=data.get('confidence', 0.7),
                key_points=data.get('key_points', []),
                risks_identified=data.get('risks', []),
                opportunities=data.get('opportunities', []),
                disagreements=data.get('disagreements', [])
            )
        except Exception as e:
            print(f"[WARN] Parse error for {provider.value}: {e}")
            return LLMInsight(
                provider=provider,
                model=self.MODELS[provider],
                topic=topic,
                insight=response[:500] if response else "Failed to parse",
                confidence=0.3,
                key_points=[],
                risks_identified=[],
                opportunities=[],
                disagreements=[]
            )

    async def run_discussion(
        self,
        topic: str,
        context: Dict[str, Any],
        rounds: int = 2
    ) -> DiscussionResult:
        """멀티 라운드 토론 실행"""

        print(f"\n{'='*60}")
        print(f"🤖 Multi-LLM Discussion: {topic}")
        print(f"{'='*60}")

        all_insights: List[LLMInsight] = []

        # Round 1: 개별 인사이트 수집 (병렬)
        print(f"\n[Round 1] 개별 인사이트 수집...")
        providers = [p for p in LLMProvider if self.available_providers.get(p)]

        tasks = [
            self.get_insight(provider, topic, context)
            for provider in providers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, LLMInsight):
                all_insights.append(result)
                print(f"  ✅ {result.provider.value}: 신뢰도 {result.confidence:.2f}")
            elif isinstance(result, Exception):
                print(f"  ❌ Error: {result}")

        # Round 2+: 다른 의견 참조하여 재분석
        if rounds > 1 and len(all_insights) > 1:
            print(f"\n[Round 2] 상호 참조 분석...")

            updated_insights = []
            for insight in all_insights:
                other_insights = [i for i in all_insights if i.provider != insight.provider]
                updated = await self.get_insight(
                    insight.provider,
                    topic,
                    context,
                    other_insights
                )
                if updated:
                    updated_insights.append(updated)
                    print(f"  🔄 {updated.provider.value}: 업데이트됨")

            if updated_insights:
                all_insights = updated_insights

        # 합의점 & 차이점 분석
        consensus, divergence = self._analyze_consensus(all_insights)

        # 최종 종합
        synthesis = self._synthesize(topic, all_insights, consensus, divergence)

        # 실행 가능 항목 추출
        actionables = self._extract_actionables(all_insights)

        # 평균 신뢰도
        avg_confidence = sum(i.confidence for i in all_insights) / len(all_insights) if all_insights else 0.0

        return DiscussionResult(
            topic=topic,
            insights=all_insights,
            consensus_points=consensus,
            divergence_points=divergence,
            final_synthesis=synthesis,
            actionable_items=actionables,
            confidence_score=avg_confidence
        )

    def _analyze_consensus(self, insights: List[LLMInsight]) -> tuple:
        """합의점과 차이점 분석"""
        if not insights:
            return [], []

        # 모든 key_points 수집
        all_points = []
        for insight in insights:
            all_points.extend(insight.key_points)

        # 중복도 기반 합의점 추출 (단순화된 버전)
        consensus = []
        divergence = []

        # 리스크 관련 합의
        all_risks = []
        for insight in insights:
            all_risks.extend(insight.risks_identified)
        if all_risks:
            unique_risks = list(set(all_risks))[:3]
            consensus.append(f"식별된 주요 리스크: {', '.join(unique_risks)}")

        # 기회 관련 합의
        all_opps = []
        for insight in insights:
            all_opps.extend(insight.opportunities)
        if all_opps:
            unique_opps = list(set(all_opps))[:3]
            consensus.append(f"식별된 기회: {', '.join(unique_opps)}")

        # 의견 차이
        for insight in insights:
            if insight.disagreements:
                divergence.extend([
                    f"[{insight.provider.value}] {d}" for d in insight.disagreements
                ])

        return consensus, divergence

    def _synthesize(
        self,
        topic: str,
        insights: List[LLMInsight],
        consensus: List[str],
        divergence: List[str]
    ) -> str:
        """최종 종합"""

        parts = [f"## {topic} - 멀티 LLM 종합 분석\n"]

        # 참여 모델
        models = [f"{i.provider.value} ({i.model})" for i in insights]
        parts.append(f"**참여 모델**: {', '.join(models)}\n")

        # 개별 인사이트 요약
        parts.append("\n### 개별 분석 요약\n")
        for insight in insights:
            parts.append(f"**{insight.provider.value.upper()}** (신뢰도: {insight.confidence:.0%})")
            parts.append(f"> {insight.insight}\n")

        # 합의점
        if consensus:
            parts.append("\n### ✅ 합의된 포인트\n")
            for point in consensus:
                parts.append(f"- {point}")

        # 차이점
        if divergence:
            parts.append("\n### ⚠️ 의견 차이\n")
            for point in divergence:
                parts.append(f"- {point}")

        return "\n".join(parts)

    def _extract_actionables(self, insights: List[LLMInsight]) -> List[str]:
        """실행 가능 항목 추출"""
        actionables = []

        # 높은 신뢰도 인사이트에서 기회 추출
        high_conf = [i for i in insights if i.confidence >= 0.7]
        for insight in high_conf:
            for opp in insight.opportunities[:2]:
                actionables.append(f"[{insight.provider.value}] {opp}")

        return actionables[:5]  # 최대 5개

    def to_markdown(self, result: DiscussionResult) -> str:
        """마크다운 출력"""
        lines = [
            f"# Multi-LLM Insight Discussion",
            f"**토픽**: {result.topic}",
            f"**시간**: {result.timestamp}",
            f"**종합 신뢰도**: {result.confidence_score:.0%}",
            "",
            result.final_synthesis,
            "",
            "### 📋 실행 가능 항목",
        ]

        for item in result.actionable_items:
            lines.append(f"- {item}")

        return "\n".join(lines)


async def discuss_report_insights(report_path: str) -> DiscussionResult:
    """리포트 파일 기반 인사이트 토론"""

    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    discussion = MultiLLMDiscussion()

    # 사용 가능한 API 출력
    print("\n🔑 Available APIs:")
    for provider, available in discussion.available_providers.items():
        status = "✅" if available else "❌"
        print(f"  {status} {provider.value}")

    # 주요 토픽별 토론
    topics = [
        "현재 레짐에서의 최적 투자 전략",
        "리스크 대비 기회 요인 분석",
        "향후 1개월 시장 방향성"
    ]

    results = []
    for topic in topics:
        result = await discussion.run_discussion(topic, report, rounds=2)
        results.append(result)
        print(f"\n{discussion.to_markdown(result)}")

    return results[0] if results else None


# CLI 테스트
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        report_path = sys.argv[1]
    else:
        # 기본 경로
        report_path = "/home/tj/projects/autoai/eimas/outputs/ai_report_20260107_015128.json"

    if os.path.exists(report_path):
        asyncio.run(discuss_report_insights(report_path))
    else:
        print(f"Report not found: {report_path}")
