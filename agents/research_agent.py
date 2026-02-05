"""
Research Agent - Perplexity API Integration

실시간 웹 검색을 통한 연구 자료 수집
ECON_AI_AGENT_SYSTEM.md Phase 1 구현

Author: EIMAS Team
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from agents.base_agent import BaseAgent, AgentConfig
from core.schemas import AgentRequest, AgentResponse, AgentRole


# ============================================================================
# Data Classes
# ============================================================================

class ResearchCategory(Enum):
    """연구 카테고리"""
    FED_POLICY = "fed_policy"           # Fed 정책
    MARKET_NEWS = "market_news"         # 시장 뉴스
    ACADEMIC = "academic"               # 논문/연구
    INDUSTRY_REPORT = "industry_report" # 산업 리포트
    COMPANY = "company"                 # 기업 뉴스
    MACRO = "macro"                     # 거시경제


@dataclass
class ResearchItem:
    """연구 자료 아이템"""
    title: str
    source: str
    date: Optional[datetime]
    category: ResearchCategory
    summary: str
    url: Optional[str] = None
    relevance_score: float = 0.0
    raw_content: str = ""


@dataclass
class ResearchReport:
    """연구 보고서"""
    query: str
    timestamp: datetime
    items: List[ResearchItem] = field(default_factory=list)
    summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def item_count(self) -> int:
        return len(self.items)

    def get_by_category(self, category: ResearchCategory) -> List[ResearchItem]:
        return [item for item in self.items if item.category == category]


# ============================================================================
# Research Agent
# ============================================================================

class ResearchAgent(BaseAgent):
    """
    Perplexity API를 활용한 연구 자료 수집 에이전트

    역할:
    - 실시간 웹 검색으로 최신 정보 수집
    - Fed 발언, 시장 뉴스, 논문, 리포트 등 다양한 소스
    - Top-Down 분석에 필요한 거시/산업/기업 정보 제공
    """

    # Perplexity 모델
    PERPLEXITY_MODELS = {
        "online": "sonar",
        "large_online": "sonar-pro",
        "chat": "sonar"
    }

    # 카테고리별 검색 템플릿
    SEARCH_TEMPLATES = {
        ResearchCategory.FED_POLICY: """
            Federal Reserve policy news and FOMC statements.
            Focus on:
            - Recent Fed speeches and communications
            - Interest rate expectations
            - Quantitative tightening/easing policies
            - Inflation outlook from Fed perspective
            Time range: last 2 weeks
        """,
        ResearchCategory.MARKET_NEWS: """
            Financial market news and analysis.
            Focus on:
            - Major market movements
            - Sector performance
            - Trading volumes and flows
            - Market sentiment indicators
            Time range: last week
        """,
        ResearchCategory.ACADEMIC: """
            Academic research and working papers on economics and finance.
            Focus on:
            - NBER working papers
            - Federal Reserve research
            - Macroeconomic studies
            - Monetary policy research
        """,
        ResearchCategory.INDUSTRY_REPORT: """
            Investment bank and research firm reports.
            Focus on:
            - Economic outlook reports
            - Sector analysis
            - Strategy recommendations
            - Risk assessments
        """,
        ResearchCategory.COMPANY: """
            Company-specific news and analysis.
            Focus on:
            - Earnings reports
            - Major announcements
            - Management changes
            - Product launches
        """,
        ResearchCategory.MACRO: """
            Macroeconomic data and analysis.
            Focus on:
            - GDP growth
            - Employment data
            - Inflation metrics
            - Trade balances
            - Global economic conditions
        """
    }

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        perplexity_api_key: Optional[str] = None,
        model: str = "online"
    ):
        """
        Parameters:
        -----------
        config : AgentConfig
            에이전트 설정
        perplexity_api_key : str
            Perplexity API 키. None이면 환경변수에서 로드
        model : str
            사용할 모델 ("online", "large_online", "chat")
        """
        if config is None:
            config = AgentConfig(
                name="ResearchAgent",
                role=AgentRole.RESEARCH,
                model="perplexity-online"
            )
        super().__init__(config)

        self.api_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model_name = self.PERPLEXITY_MODELS.get(model, self.PERPLEXITY_MODELS["online"])
        self._client = None
        self._claude_client = None

    @property
    def client(self):
        """Lazy initialization of Perplexity client"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.perplexity.ai"
            )
        return self._client

    @property
    def claude_client(self):
        """Lazy initialization of Claude client"""
        if self._claude_client is None:
            from anthropic import Anthropic
            self._claude_client = Anthropic(api_key=self.anthropic_api_key)
        return self._claude_client

    async def _execute(self, request: AgentRequest) -> Any:
        """
        연구 자료 수집 실행

        Args:
            request: AgentRequest containing 'query' in instruction or context

        Returns:
            AgentResponse containing ResearchReport
        """
        # 요청 파싱
        query = request.instruction
        context = request.context.get("context", "")
        categories = request.context.get("categories", list(ResearchCategory))

        # 병렬로 각 카테고리 검색
        search_tasks = []
        for category in categories:
            task = self._search_category(query, category, context)
            search_tasks.append(task)

        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 결과 집계
        all_items = []
        for result in results:
            if isinstance(result, list):
                all_items.extend(result)

        # 관련성 순 정렬
        all_items.sort(key=lambda x: x.relevance_score, reverse=True)

        # 보고서 생성
        # Claude로 경제학적 맥락 해석 (NEW)
        try:
            summary = await self._interpret_with_claude(query, all_items)
        except Exception as e:
            self.logger.warning(f"Claude interpretation failed: {e}, falling back to Perplexity")
            summary = await self._generate_summary(query, all_items)

        key_insights = await self._extract_insights(all_items)
        
        report = ResearchReport(
            query=query,
            timestamp=datetime.now(),
            items=all_items,
            summary=summary,
            key_insights=key_insights
        )

        return {
            "content": report,
            "sources": [item.source for item in all_items],
            "confidence": self._calculate_confidence(report)
        }

    async def _interpret_with_claude(self, query: str, items: List[ResearchItem]) -> str:
        """Claude를 사용하여 연구 결과의 경제학적 맥락 해석"""
        if not items:
            return "검색 결과 없음"

        # 상위 항목들을 요약
        top_items = items[:15]
        content_summary = '\n'.join([
            f"- {item.title}: {item.summary[:200]}"
            for item in top_items
        ])

        prompt = f"""
        You are an expert economic analyst. Analyze these research findings about "{query}":

        {content_summary}

        Provide a sophisticated economic interpretation highlighting:
        1. **Core Economic Narrative**: What is the dominant story?
        2. **Divergent Signals**: Are there conflicting data points?
        3. **Implications**: What does this mean for market regime and risk?

        Focus on causality and interconnections between variables.
        """

        response = await asyncio.to_thread(
            self.claude_client.messages.create,
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def _search_category(
        self,
        query: str,
        category: ResearchCategory,
        context: str = ""
    ) -> List[ResearchItem]:
        """특정 카테고리 검색"""
        template = self.SEARCH_TEMPLATES.get(category, "")

        prompt = f"""
        Search for information about: {query}

        {template}

        Additional context: {context}

        Provide structured results with:
        1. Source name
        2. Publication date (if available)
        3. Brief summary (2-3 sentences)
        4. Key findings or data points

        Format each result clearly.
        """

        try:
            response = await asyncio.to_thread(
                self._call_perplexity,
                prompt
            )
            return self._parse_search_results(response, category)
        except Exception as e:
            self.logger.error(f"Search failed for {category.value}: {e}")
            return []

    def _call_perplexity(self, prompt: str) -> str:
        """Perplexity API 호출"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert financial researcher.
                    Provide accurate, well-sourced information.
                    Always cite sources when available.
                    Focus on recent, relevant data."""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content

    def _parse_search_results(
        self,
        response: str,
        category: ResearchCategory
    ) -> List[ResearchItem]:
        """검색 결과 파싱"""
        items = []

        # 줄 단위 파싱
        lines = response.split('\n')
        current_item = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_item:
                    current_item['content'] = ' '.join(current_content)
                    items.append(self._create_item(current_item, category))
                    current_item = None
                    current_content = []
                continue

            # 새 항목 시작 감지 (번호, 제목 등)
            if (line[0].isdigit() and '.' in line[:5]) or line.startswith('-') or line.startswith('•'):
                if current_item:
                    current_item['content'] = ' '.join(current_content)
                    items.append(self._create_item(current_item, category))

                current_item = {'title': line.lstrip('0123456789.-•) ')}
                current_content = []
            elif current_item:
                current_content.append(line)

        # 마지막 항목
        if current_item:
            current_item['content'] = ' '.join(current_content)
            items.append(self._create_item(current_item, category))

        return items[:5]  # 카테고리당 최대 5개

    def _create_item(
        self,
        parsed: Dict[str, str],
        category: ResearchCategory
    ) -> ResearchItem:
        """ResearchItem 생성"""
        title = parsed.get('title', 'Untitled')
        content = parsed.get('content', '')

        return ResearchItem(
            title=title[:200],  # 제목 길이 제한
            source="Perplexity Search",
            date=datetime.now(),
            category=category,
            summary=content[:500],
            relevance_score=self._calculate_relevance(title + ' ' + content),
            raw_content=content
        )

    def _calculate_relevance(self, text: str) -> float:
        """관련성 점수 계산"""
        # 핵심 키워드 기반 점수
        keywords = {
            'fed': 0.1, 'federal reserve': 0.15, 'fomc': 0.15,
            'interest rate': 0.1, 'inflation': 0.1, 'gdp': 0.08,
            'market': 0.05, 'economic': 0.05, 'growth': 0.05,
            'policy': 0.08, 'monetary': 0.1, 'fiscal': 0.08
        }

        text_lower = text.lower()
        score = 0.3  # 기본 점수

        for keyword, weight in keywords.items():
            if keyword in text_lower:
                score += weight

        return min(score, 1.0)

    async def _generate_summary(
        self,
        query: str,
        items: List[ResearchItem]
    ) -> str:
        """전체 연구 결과 요약 생성"""
        if not items:
            return "검색 결과 없음"

        # 상위 항목들을 요약
        top_items = items[:10]
        content_summary = '\n'.join([
            f"- {item.title}: {item.summary[:200]}"
            for item in top_items
        ])

        prompt = f"""
        Summarize these research findings about "{query}":

        {content_summary}

        Provide a 2-3 paragraph executive summary highlighting:
        1. Key themes and findings
        2. Notable data points
        3. Implications for analysis
        """

        try:
            summary = await asyncio.to_thread(
                self._call_perplexity,
                prompt
            )
            return summary
        except Exception:
            return f"Found {len(items)} items related to {query}"

    async def _extract_insights(self, items: List[ResearchItem]) -> List[str]:
        """핵심 인사이트 추출"""
        insights = []

        # 카테고리별 상위 인사이트
        for category in ResearchCategory:
            category_items = [i for i in items if i.category == category]
            if category_items:
                top_item = category_items[0]
                insights.append(f"[{category.value}] {top_item.title[:100]}")

        return insights[:5]  # 최대 5개

    async def form_opinion(self, topic: str, context: Dict[str, Any]) -> "AgentOpinion":
        """
        특정 주제에 대한 의견 형성 (BaseAgent 인터페이스)

        Parameters:
        -----------
        topic : str
            의견을 형성할 주제
        context : Dict
            추가 컨텍스트 정보

        Returns:
        --------
        AgentOpinion
            표준화된 의견 객체
        """
        from core.schemas import AgentOpinion, OpinionStrength

        # Mock 연구 수행 (Perplexity API 없이도 작동)
        try:
            report = await self._mock_research(topic, context)
        except Exception as e:
            self.logger.warning(f"Research failed: {e}, using fallback")
            report = ResearchReport(
                query=topic,
                timestamp=datetime.now(),
                summary="Research unavailable - using context-based analysis",
                items=[]
            )

        # 의견 형성
        stance = self._determine_stance_from_context(topic, context)
        confidence = self._calculate_confidence_from_context(context)

        # OpinionStrength 결정
        if confidence > 0.7:
            strength = OpinionStrength.STRONG_AGREE if stance == "BULLISH" else OpinionStrength.STRONG_DISAGREE if stance == "BEARISH" else OpinionStrength.NEUTRAL
        elif confidence > 0.5:
            strength = OpinionStrength.AGREE if stance == "BULLISH" else OpinionStrength.DISAGREE if stance == "BEARISH" else OpinionStrength.NEUTRAL
        else:
            strength = OpinionStrength.NEUTRAL

        return AgentOpinion(
            agent_role=self.config.role,
            topic=topic,
            position=self._generate_position(stance, topic, context),
            strength=strength,
            confidence=confidence,
            evidence=self._extract_evidence(context),
            caveats=["Based on available context data", "Real-time research not performed"],
            key_metrics=self._extract_key_metrics(context)
        )

    async def _mock_research(self, topic: str, context: Dict[str, Any]) -> ResearchReport:
        """Mock 연구 수행 (API 없이 컨텍스트 기반)"""
        return ResearchReport(
            query=topic,
            timestamp=datetime.now(),
            summary=f"Context-based analysis for {topic}",
            items=[],
            key_insights=self._generate_insights_from_context(context)
        )

    def _determine_stance_from_context(self, topic: str, context: Dict[str, Any]) -> str:
        """컨텍스트 기반 stance 결정"""
        risk_score = context.get('total_risk_score', 50)
        regime = context.get('current_regime', 'UNKNOWN').upper()

        if topic == "market_outlook":
            if risk_score < 40 and regime in ['BULL', 'EXPANSION']:
                return "BULLISH"
            elif risk_score > 60 or regime in ['BEAR', 'CONTRACTION']:
                return "BEARISH"
            return "NEUTRAL"
        elif topic == "primary_risk":
            if risk_score > 60:
                return "HIGH_RISK"
            elif risk_score < 40:
                return "LOW_RISK"
            return "MODERATE_RISK"
        return "NEUTRAL"

    def _calculate_confidence_from_context(self, context: Dict[str, Any]) -> float:
        """컨텍스트 기반 신뢰도 계산"""
        # 데이터 완전성 기반 신뢰도
        data_fields = ['total_risk_score', 'current_regime', 'regime_confidence', 'transition_probability']
        available = sum(1 for f in data_fields if f in context)
        return min(0.3 + (available / len(data_fields)) * 0.5, 0.8)

    def _generate_position(self, stance: str, topic: str, context: Dict[str, Any]) -> str:
        """입장 문장 생성"""
        risk_score = context.get('total_risk_score', 50)
        regime = context.get('current_regime', 'UNKNOWN')

        if topic == "market_outlook":
            return f"Market outlook is {stance} based on {regime} regime and risk score {risk_score:.1f}"
        elif topic == "primary_risk":
            return f"Primary risk assessment: {stance} (score: {risk_score:.1f}/100)"
        return f"Research assessment for {topic}: {stance}"

    def _extract_evidence(self, context: Dict[str, Any]) -> List[str]:
        """컨텍스트에서 증거 추출"""
        evidence = []
        if 'total_risk_score' in context:
            evidence.append(f"Risk Score: {context['total_risk_score']:.1f}/100")
        if 'current_regime' in context:
            evidence.append(f"Current Regime: {context['current_regime']}")
        if 'transition_probability' in context:
            prob = context['transition_probability']
            if prob > 1:
                prob /= 100
            evidence.append(f"Regime Transition Probability: {prob:.1%}")
        if 'active_warnings' in context:
            warnings = context['active_warnings'][:2]
            evidence.extend(warnings)
        return evidence if evidence else ["Context data limited"]

    def _extract_key_metrics(self, context: Dict[str, Any]) -> Dict[str, float]:
        """주요 메트릭 추출"""
        metrics = {}
        if 'total_risk_score' in context:
            metrics['risk_score'] = context['total_risk_score']
        if 'regime_confidence' in context:
            metrics['regime_confidence'] = context['regime_confidence']
        if 'transition_probability' in context:
            prob = context['transition_probability']
            metrics['transition_prob'] = prob / 100 if prob > 1 else prob
        return metrics

    def _generate_insights_from_context(self, context: Dict[str, Any]) -> List[str]:
        """컨텍스트에서 인사이트 생성"""
        insights = []
        risk = context.get('total_risk_score', 50)
        if risk > 60:
            insights.append("Elevated risk conditions warrant caution")
        elif risk < 40:
            insights.append("Low risk environment supports growth positioning")
        return insights

    def _determine_stance(self, report: ResearchReport) -> str:
        """연구 결과 기반 stance 결정"""
        # 간단한 감성 분석
        positive_keywords = ['growth', 'increase', 'positive', 'strong', 'bullish', 'recovery']
        negative_keywords = ['decline', 'decrease', 'negative', 'weak', 'bearish', 'recession']

        text = report.summary.lower()

        pos_count = sum(1 for kw in positive_keywords if kw in text)
        neg_count = sum(1 for kw in negative_keywords if kw in text)

        if pos_count > neg_count + 2:
            return "BULLISH"
        elif neg_count > pos_count + 2:
            return "BEARISH"
        else:
            return "NEUTRAL"

    def _calculate_confidence(self, report: ResearchReport) -> float:
        """신뢰도 계산"""
        # 소스 수와 관련성 기반
        source_score = min(report.item_count / 10, 0.5)  # 최대 0.5

        if report.items:
            avg_relevance = sum(i.relevance_score for i in report.items) / len(report.items)
        else:
            avg_relevance = 0

        return min(source_score + avg_relevance * 0.5, 1.0)

    # ========================================================================
    # Specialized Search Methods
    # ========================================================================

    async def search_fed_communications(
        self,
        days_back: int = 14
    ) -> List[ResearchItem]:
        """Fed 발언 및 통신 검색"""
        prompt = f"""
        Find Federal Reserve communications from the last {days_back} days:

        1. FOMC meeting minutes and statements
        2. Fed Chair and Governor speeches
        3. Fed Bank President speeches
        4. Economic projections and dot plots

        Provide specific quotes and dates.
        """

        response = await asyncio.to_thread(self._call_perplexity, prompt)
        return self._parse_search_results(response, ResearchCategory.FED_POLICY)

    async def search_market_reports(
        self,
        topic: str
    ) -> List[ResearchItem]:
        """투자은행 리포트 검색"""
        prompt = f"""
        Find recent investment bank and research firm reports about: {topic}

        Focus on:
        - Goldman Sachs, JPMorgan, Morgan Stanley research
        - Economic outlook reports
        - Strategy recommendations

        Provide specific report titles and key findings.
        """

        response = await asyncio.to_thread(self._call_perplexity, prompt)
        return self._parse_search_results(response, ResearchCategory.INDUSTRY_REPORT)

    async def search_academic_papers(
        self,
        topic: str
    ) -> List[ResearchItem]:
        """학술 논문 검색"""
        prompt = f"""
        Find academic research papers related to: {topic}

        Focus on:
        - NBER Working Papers
        - Federal Reserve research papers
        - IMF, World Bank publications
        - Top economics journals

        Provide paper titles, authors, and key findings.
        """

        response = await asyncio.to_thread(self._call_perplexity, prompt)
        return self._parse_search_results(response, ResearchCategory.ACADEMIC)

    async def search_company_news(
        self,
        company_name: str,
        ticker: str
    ) -> List[ResearchItem]:
        """기업 뉴스 검색"""
        prompt = f"""
        Find recent news about {company_name} ({ticker}):

        1. Latest earnings reports and guidance
        2. Major announcements (contracts, partnerships, products)
        3. Management or strategic changes
        4. Analyst ratings and price targets

        Provide specific news headlines and dates.
        """

        response = await asyncio.to_thread(self._call_perplexity, prompt)
        return self._parse_search_results(response, ResearchCategory.COMPANY)

    async def search_macro_data(
        self,
        indicators: List[str]
    ) -> List[ResearchItem]:
        """거시경제 데이터 검색"""
        indicators_str = ', '.join(indicators)

        prompt = f"""
        Find latest macroeconomic data for: {indicators_str}

        Include:
        1. Most recent releases
        2. Comparison to expectations
        3. Trend analysis
        4. Expert commentary

        Provide specific numbers and dates.
        """

        response = await asyncio.to_thread(self._call_perplexity, prompt)
        return self._parse_search_results(response, ResearchCategory.MACRO)


# ============================================================================
# Test / Demo
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("Research Agent Demo")
        print("=" * 60)

        # API 키 확인
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            print("Warning: PERPLEXITY_API_KEY not set")
            print("Set environment variable to run full demo")
            return

        agent = ResearchAgent()

        # 테스트 쿼리
        query = "Federal Reserve interest rate policy outlook 2025"
        print(f"\n[Query] {query}")

        # 연구 실행
        report = await agent._execute(query)

        print(f"\n[Results]")
        print(f"  Total items: {report.item_count}")
        print(f"\n[Summary]")
        print(report.summary[:500])

        print(f"\n[Top Items]")
        for i, item in enumerate(report.items[:5], 1):
            print(f"  {i}. [{item.category.value}] {item.title[:60]}...")

        print(f"\n[Key Insights]")
        for insight in report.key_insights:
            print(f"  - {insight}")

        print("\n" + "=" * 60)

    asyncio.run(demo())
