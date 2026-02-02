#!/usr/bin/env python3
"""
Causality Narrative Generator
==============================
Supply Chain 구조에서 인과관계 내러티브를 LLM으로 생성

소스 이론: "경제학은 인과관계(Causality)다. 그래프 이론으로 설명해야 한다."

Usage:
    from lib.causality_narrative import CausalityNarrativeGenerator

    generator = CausalityNarrativeGenerator()
    narrative = await generator.generate(
        bottlenecks=['ASML', 'AMAT'],
        hub_nodes=['TSM', 'NVDA'],
        supply_chain_layers={...},
        market_data={...}
    )
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger('eimas.causality_narrative')


@dataclass
class CausalityInsight:
    """인과관계 인사이트"""
    path: str                    # 경로: "AI Demand → ASML → TSM → NVDA"
    insight: str                 # 인사이트 설명
    causality_type: str          # positive, negative, neutral
    confidence: float            # 신뢰도
    affected_assets: List[str]   # 영향받는 자산

    def to_dict(self) -> Dict:
        return {
            'path': self.path,
            'insight': self.insight,
            'causality_type': self.causality_type,
            'confidence': self.confidence,
            'affected_assets': self.affected_assets
        }


class CausalityNarrativeGenerator:
    """
    인과관계 내러티브 생성기

    두 가지 모드:
    1. Rule-based: 템플릿 기반 빠른 생성 (LLM 없이)
    2. LLM-enhanced: Claude API를 사용한 상세 분석
    """

    # 인과관계 템플릿
    CAUSALITY_TEMPLATES = {
        'supply_disruption': (
            "**Path:** {external_shock} (External Shock) → {bottleneck} (Bottleneck) → "
            "{midstream} (Production) → {downstream} (Revenue).\n"
            "**Insight:** {bottleneck}의 {disruption_type}은 {midstream}의 가동률 하락을 "
            "유발하여 {downstream}의 출하량에 병목을 일으킬 수 있음 (Negative Causality)."
        ),
        'demand_surge': (
            "**Path:** {external_shock} (Demand Shock) → {downstream} (Revenue Up) → "
            "{midstream} (Utilization Up) → {bottleneck} (Capacity Strain).\n"
            "**Insight:** {external_shock}로 인해 {downstream} 수요가 급증하면, "
            "{midstream} 가동률이 상승하고 {bottleneck}에 용량 압박이 발생함 (Positive Causality)."
        ),
        'hub_influence': (
            "**Path:** {hub} (Hub Node) ← {upstream} / → {downstream}.\n"
            "**Insight:** {hub}는 공급망 네트워크의 핵심 허브로, "
            "{upstream}의 공급과 {downstream}의 수요를 모두 연결. "
            "이 종목의 변동은 양방향으로 전파됨 (Bidirectional Causality)."
        ),
        'bottleneck_risk': (
            "**Path:** Risk Event → {bottleneck} (Bottleneck) → Supply Chain Disruption.\n"
            "**Insight:** {bottleneck}는 대체 불가능한 병목 지점으로, "
            "이 종목에 문제 발생 시 전체 공급망이 {impact_duration} 이상 지연될 수 있음 (Critical Dependency)."
        ),
    }

    # 외부 충격 유형
    EXTERNAL_SHOCKS = {
        'AI Demand Surge': {'type': 'demand_surge', 'disruption_type': 'capacity expansion'},
        'Export Restriction': {'type': 'supply_disruption', 'disruption_type': 'equipment export ban'},
        'Chip Shortage': {'type': 'supply_disruption', 'disruption_type': 'wafer shortage'},
        'Geopolitical Tension': {'type': 'supply_disruption', 'disruption_type': 'trade conflict'},
        'Energy Crisis': {'type': 'supply_disruption', 'disruption_type': 'production halt'},
        'EV Demand Surge': {'type': 'demand_surge', 'disruption_type': 'battery demand'},
        'Data Center Expansion': {'type': 'demand_surge', 'disruption_type': 'server demand'},
    }

    def __init__(self, use_llm: bool = False):
        """
        Args:
            use_llm: LLM API 사용 여부 (False면 템플릿 기반)
        """
        self.use_llm = use_llm
        self._api_config = None

    def _get_api_config(self):
        """API 설정 lazy loading"""
        if self._api_config is None:
            try:
                from core.config import APIConfig
                self._api_config = APIConfig()
            except:
                self._api_config = None
        return self._api_config

    def generate_rule_based(
        self,
        bottlenecks: List[str],
        hub_nodes: List[str],
        supply_chain_layers: Dict[str, List[str]],
        external_shock: str = 'AI Demand Surge',
        market_data: Dict[str, float] = None
    ) -> List[CausalityInsight]:
        """
        Rule-based 인과관계 내러티브 생성 (LLM 없이)

        Args:
            bottlenecks: 병목 지점 종목들 ['ASML', 'AMAT']
            hub_nodes: 허브 노드 종목들 ['TSM', 'NVDA']
            supply_chain_layers: 레이어별 종목 {'equipment': [...], 'manufacturer': [...]}
            external_shock: 외부 충격 유형
            market_data: 시장 데이터 (가격 변동률)

        Returns:
            List[CausalityInsight]: 인과관계 인사이트 목록
        """
        insights = []

        # 레이어별 대표 종목 추출
        equipment = supply_chain_layers.get('equipment', [])
        manufacturer = supply_chain_layers.get('manufacturer', [])
        integrator = supply_chain_layers.get('integrator', [])
        end_user = supply_chain_layers.get('end_user', [])

        shock_info = self.EXTERNAL_SHOCKS.get(external_shock, {
            'type': 'supply_disruption',
            'disruption_type': 'disruption'
        })

        # 1. 주요 병목 → 공급망 전파 인과관계
        if bottlenecks and (manufacturer or integrator):
            main_bottleneck = bottlenecks[0]
            midstream = manufacturer[0] if manufacturer else (integrator[0] if integrator else 'N/A')
            downstream = integrator[0] if integrator else (end_user[0] if end_user else 'N/A')

            path = f"{external_shock} (External Shock) → {main_bottleneck} (Bottleneck) → {midstream} (Production) → {downstream} (Revenue)"

            if shock_info['type'] == 'demand_surge':
                insight_text = (
                    f"{external_shock}로 인해 {downstream} 수요가 급증하면, "
                    f"{midstream} 가동률이 상승하고 {main_bottleneck}에 용량 압박이 발생함 (Positive Causality)."
                )
                causality_type = 'positive'
            else:
                insight_text = (
                    f"{main_bottleneck}의 {shock_info['disruption_type']}은 {midstream}의 가동률 하락을 "
                    f"유발하여 {downstream}의 출하량에 병목을 일으킬 수 있음 (Negative Causality)."
                )
                causality_type = 'negative'

            insights.append(CausalityInsight(
                path=path,
                insight=insight_text,
                causality_type=causality_type,
                confidence=0.85,
                affected_assets=[main_bottleneck, midstream, downstream]
            ))

        # 2. 허브 노드 양방향 영향
        if hub_nodes and len(hub_nodes) >= 1:
            hub = hub_nodes[0]
            upstream = equipment[0] if equipment else (bottlenecks[0] if bottlenecks else 'Suppliers')
            downstream = end_user[0] if end_user else (integrator[-1] if integrator else 'Customers')

            path = f"{upstream} → {hub} (Hub Node) → {downstream}"
            insight_text = (
                f"{hub}는 공급망 네트워크의 핵심 허브로, "
                f"{upstream}의 공급과 {downstream}의 수요를 모두 연결. "
                f"이 종목의 변동은 양방향으로 전파됨 (Bidirectional Causality)."
            )

            insights.append(CausalityInsight(
                path=path,
                insight=insight_text,
                causality_type='neutral',
                confidence=0.80,
                affected_assets=[upstream, hub, downstream]
            ))

        # 3. 병목 리스크 경고 (대체 불가능 지점)
        critical_bottlenecks = [b for b in bottlenecks if b in ['ASML', 'TSM', 'NVDA']]
        for bottleneck in critical_bottlenecks[:2]:
            path = f"Risk Event → {bottleneck} (Critical Bottleneck) → Supply Chain Disruption"

            if bottleneck == 'ASML':
                insight_text = (
                    f"{bottleneck}는 EUV 리소그래피 장비의 유일한 공급자로, "
                    f"대체 불가능한 병목 지점. 이 종목에 문제 발생 시 "
                    f"최첨단 반도체 생산이 12-18개월 이상 지연될 수 있음 (Critical Dependency)."
                )
            elif bottleneck == 'TSM':
                insight_text = (
                    f"{bottleneck}는 세계 파운드리 시장의 ~55%를 점유하는 핵심 생산자로, "
                    f"대만 지정학적 리스크가 현실화되면 글로벌 칩 공급에 "
                    f"심각한 차질이 발생함 (Geopolitical Critical Dependency)."
                )
            elif bottleneck == 'NVDA':
                insight_text = (
                    f"{bottleneck}는 AI 가속기 시장의 ~80%를 점유하며, "
                    f"AI 인프라 확장의 핵심 병목. 공급 제약 시 "
                    f"하이퍼스케일러 AI 투자가 지연됨 (AI Infrastructure Dependency)."
                )
            else:
                insight_text = f"{bottleneck}는 공급망의 중요 병목 지점으로 모니터링 필요."

            insights.append(CausalityInsight(
                path=path,
                insight=insight_text,
                causality_type='negative',
                confidence=0.90,
                affected_assets=[bottleneck]
            ))

        # 4. 시장 데이터 기반 동적 인과관계 (가격 변동 반영)
        if market_data:
            significant_moves = [(k, v) for k, v in market_data.items() if abs(v) > 2.0]
            significant_moves.sort(key=lambda x: abs(x[1]), reverse=True)

            for ticker, change in significant_moves[:2]:
                direction = "Up" if change > 0 else "Down"
                causality = "positive" if change > 0 else "negative"

                # 해당 종목이 어느 레이어에 있는지 확인
                layer = None
                for layer_name, tickers in supply_chain_layers.items():
                    if ticker in tickers:
                        layer = layer_name
                        break

                if layer:
                    path = f"Market Move: {ticker} ({change:+.1f}%) → {layer.title()} Layer Impact"
                    insight_text = (
                        f"{ticker}가 {abs(change):.1f}% {'상승' if change > 0 else '하락'}하여 "
                        f"{layer} 레이어 전체에 {'긍정적' if change > 0 else '부정적'} 신호 전파. "
                        f"관련 종목 동조화 가능성 높음."
                    )

                    insights.append(CausalityInsight(
                        path=path,
                        insight=insight_text,
                        causality_type=causality,
                        confidence=0.75,
                        affected_assets=[ticker]
                    ))

        return insights if insights else [CausalityInsight(
            path="No significant causality detected",
            insight="공급망 구조가 안정적이며 현재 주요 인과관계 이벤트가 없음.",
            causality_type='neutral',
            confidence=0.70,
            affected_assets=[]
        )]

    async def generate_with_llm(
        self,
        bottlenecks: List[str],
        hub_nodes: List[str],
        supply_chain_layers: Dict[str, List[str]],
        external_shock: str = 'AI Demand Surge',
        market_data: Dict[str, float] = None
    ) -> List[CausalityInsight]:
        """
        LLM을 사용한 상세 인과관계 내러티브 생성

        Claude API를 사용하여 더 정교한 분석 생성
        """
        api_config = self._get_api_config()
        if not api_config:
            logger.warning("API config not available, falling back to rule-based")
            return self.generate_rule_based(
                bottlenecks, hub_nodes, supply_chain_layers, external_shock, market_data
            )

        try:
            client = api_config.get_client('claude')

            # 프롬프트 구성
            prompt = f"""You are an expert economist analyzing supply chain causality in the semiconductor industry.

Given the following supply chain structure:
- Bottlenecks (critical nodes): {', '.join(bottlenecks)}
- Hub Nodes (central connectors): {', '.join(hub_nodes)}
- Supply Chain Layers:
  - Equipment: {', '.join(supply_chain_layers.get('equipment', []))}
  - Manufacturer: {', '.join(supply_chain_layers.get('manufacturer', []))}
  - Integrator: {', '.join(supply_chain_layers.get('integrator', []))}
  - End User: {', '.join(supply_chain_layers.get('end_user', []))}

External Shock Context: {external_shock}
Market Data (price changes): {market_data if market_data else 'No significant moves'}

Generate 2-3 causality insights in the following JSON format:
[
  {{
    "path": "Event → Node1 → Node2 → Impact",
    "insight": "Detailed economic explanation of the causality chain...",
    "causality_type": "positive|negative|neutral",
    "confidence": 0.85,
    "affected_assets": ["TICKER1", "TICKER2"]
  }}
]

Focus on:
1. Supply disruption propagation paths
2. Demand shock transmission
3. Critical dependency risks
4. Bidirectional hub influences

Respond ONLY with valid JSON array, no markdown or explanation."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            # JSON 파싱
            import json
            response_text = response.content[0].text.strip()

            # JSON 추출 (혹시 마크다운이 포함되어 있을 경우)
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1])

            insights_data = json.loads(response_text)

            insights = []
            for item in insights_data:
                insights.append(CausalityInsight(
                    path=item.get('path', ''),
                    insight=item.get('insight', ''),
                    causality_type=item.get('causality_type', 'neutral'),
                    confidence=item.get('confidence', 0.7),
                    affected_assets=item.get('affected_assets', [])
                ))

            return insights

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self.generate_rule_based(
                bottlenecks, hub_nodes, supply_chain_layers, external_shock, market_data
            )

    async def generate(
        self,
        bottlenecks: List[str],
        hub_nodes: List[str],
        supply_chain_layers: Dict[str, List[str]],
        external_shock: str = 'AI Demand Surge',
        market_data: Dict[str, float] = None
    ) -> List[CausalityInsight]:
        """
        인과관계 내러티브 생성 (메인 인터페이스)

        use_llm 설정에 따라 LLM 또는 Rule-based 방식 선택
        """
        if self.use_llm:
            return await self.generate_with_llm(
                bottlenecks, hub_nodes, supply_chain_layers, external_shock, market_data
            )
        else:
            return self.generate_rule_based(
                bottlenecks, hub_nodes, supply_chain_layers, external_shock, market_data
            )

    def format_for_markdown(self, insights: List[CausalityInsight]) -> str:
        """
        마크다운 형식으로 포맷팅

        integrated_*.md 리포트에 삽입할 형식
        """
        if not insights:
            return "No significant causality insights detected."

        lines = []
        for i, insight in enumerate(insights, 1):
            lines.append(f"**[{i}] {insight.path}**")
            lines.append(f"- {insight.insight}")
            lines.append(f"- Causality: `{insight.causality_type.upper()}` (Confidence: {insight.confidence:.0%})")
            if insight.affected_assets:
                lines.append(f"- Affected: {', '.join(insight.affected_assets)}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Causality Narrative Generator Test")
    print("=" * 60)

    generator = CausalityNarrativeGenerator(use_llm=False)

    # 테스트 데이터
    bottlenecks = ['ASML', 'AMAT', 'LRCX']
    hub_nodes = ['TSM', 'NVDA', 'INTC']
    supply_chain_layers = {
        'equipment': ['ASML', 'AMAT', 'LRCX', 'KLAC'],
        'manufacturer': ['TSM', 'INTC'],
        'integrator': ['NVDA', 'AMD', 'AVGO'],
        'end_user': ['MSFT', 'GOOGL', 'AMZN']
    }
    market_data = {'NVDA': 3.5, 'ASML': -2.1, 'TSM': 0.5}

    # 생성
    insights = generator.generate_rule_based(
        bottlenecks=bottlenecks,
        hub_nodes=hub_nodes,
        supply_chain_layers=supply_chain_layers,
        external_shock='AI Demand Surge',
        market_data=market_data
    )

    print("\n[Generated Insights]")
    print("-" * 40)

    for i, insight in enumerate(insights, 1):
        print(f"\n[{i}] {insight.path}")
        print(f"    Insight: {insight.insight}")
        print(f"    Type: {insight.causality_type}, Confidence: {insight.confidence:.0%}")

    print("\n[Markdown Format]")
    print("-" * 40)
    print(generator.format_for_markdown(insights))

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)
