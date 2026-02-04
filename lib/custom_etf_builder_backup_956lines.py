"""
Custom ETF Builder - Theme/Sector Basket with Supply Chain Graph
================================================================

사용자 정의 테마 ETF 생성기:
1. 테마 정의 (AI, EV, Clean Energy 등)
2. 공급망 그래프 구축 (upstream → midstream → downstream)
3. 동적 비중 조절 (모멘텀, 밸류, 공급망 중심성 기반)
4. 리밸런싱 시그널 생성
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import networkx as nx
from collections import defaultdict


class ThemeCategory(Enum):
    """투자 테마 카테고리"""
    AI_SEMICONDUCTOR = "ai_semiconductor"       # AI/반도체
    ELECTRIC_VEHICLE = "electric_vehicle"       # 전기차
    CLEAN_ENERGY = "clean_energy"               # 클린에너지
    DEFENSE = "defense"                          # 방산
    BIOTECH = "biotech"                          # 바이오텍
    CYBERSECURITY = "cybersecurity"             # 사이버보안
    FINTECH = "fintech"                          # 핀테크
    SPACE = "space"                              # 우주산업
    BLOCKCHAIN = "blockchain"                    # 블록체인
    CLOUD_SAAS = "cloud_saas"                   # 클라우드/SaaS
    CUSTOM = "custom"                            # 사용자 정의


class SupplyChainLayer(Enum):
    """공급망 레이어"""
    RAW_MATERIAL = "raw_material"   # 원자재
    COMPONENT = "component"          # 부품
    EQUIPMENT = "equipment"          # 장비
    MANUFACTURER = "manufacturer"    # 제조
    INTEGRATOR = "integrator"        # 통합
    DISTRIBUTION = "distribution"    # 유통
    END_USER = "end_user"            # 최종 사용자


@dataclass
class ThemeStock:
    """테마 내 개별 종목"""
    ticker: str
    name: str
    layer: SupplyChainLayer
    weight_base: float = 0.0
    suppliers: List[str] = field(default_factory=list)
    customers: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class ThemeETF:
    """테마 ETF 정의"""
    name: str
    category: ThemeCategory
    description: str
    stocks: List[ThemeStock]
    target_weight: Dict[str, float] = field(default_factory=dict)
    supply_chain_graph: Optional[nx.DiGraph] = None


# =============================================================================
# 사전 정의된 테마 ETF 템플릿
# =============================================================================

THEME_TEMPLATES: Dict[ThemeCategory, Dict] = {
    ThemeCategory.AI_SEMICONDUCTOR: {
        "name": "AI & Semiconductor Revolution",
        "description": "AI 인프라와 반도체 밸류체인",
        "stocks": [
            # Raw Material / Component
            ThemeStock("AMAT", "Applied Materials", SupplyChainLayer.EQUIPMENT, 0.08,
                      customers=["TSM", "INTC", "NVDA"]),
            ThemeStock("ASML", "ASML Holding", SupplyChainLayer.EQUIPMENT, 0.10,
                      customers=["TSM", "INTC", "SSNLF"]),
            ThemeStock("LRCX", "Lam Research", SupplyChainLayer.EQUIPMENT, 0.06,
                      customers=["TSM", "INTC"]),
            ThemeStock("KLAC", "KLA Corp", SupplyChainLayer.EQUIPMENT, 0.05,
                      customers=["TSM", "INTC"]),
            # Manufacturer
            ThemeStock("TSM", "Taiwan Semiconductor", SupplyChainLayer.MANUFACTURER, 0.12,
                      suppliers=["ASML", "AMAT"], customers=["NVDA", "AMD", "AAPL"]),
            ThemeStock("INTC", "Intel", SupplyChainLayer.MANUFACTURER, 0.06,
                      suppliers=["ASML", "AMAT"]),
            # Integrator (Chip Designer)
            ThemeStock("NVDA", "NVIDIA", SupplyChainLayer.INTEGRATOR, 0.15,
                      suppliers=["TSM"], customers=["MSFT", "GOOGL", "AMZN"]),
            ThemeStock("AMD", "AMD", SupplyChainLayer.INTEGRATOR, 0.08,
                      suppliers=["TSM"], customers=["MSFT", "GOOGL"]),
            ThemeStock("AVGO", "Broadcom", SupplyChainLayer.INTEGRATOR, 0.08,
                      suppliers=["TSM"]),
            ThemeStock("MRVL", "Marvell", SupplyChainLayer.INTEGRATOR, 0.04,
                      suppliers=["TSM"]),
            # End User (AI Hyperscalers)
            ThemeStock("MSFT", "Microsoft", SupplyChainLayer.END_USER, 0.08,
                      suppliers=["NVDA", "AMD"]),
            ThemeStock("GOOGL", "Alphabet", SupplyChainLayer.END_USER, 0.06,
                      suppliers=["NVDA", "AMD"]),
            ThemeStock("AMZN", "Amazon", SupplyChainLayer.END_USER, 0.04,
                      suppliers=["NVDA"]),
        ]
    },

    ThemeCategory.ELECTRIC_VEHICLE: {
        "name": "Electric Vehicle Ecosystem",
        "description": "전기차 밸류체인 (배터리 → 차량 → 충전)",
        "stocks": [
            # Raw Material
            ThemeStock("ALB", "Albemarle", SupplyChainLayer.RAW_MATERIAL, 0.06,
                      customers=["PCRFY", "LG에너지"]),
            ThemeStock("SQM", "SQM", SupplyChainLayer.RAW_MATERIAL, 0.05,
                      customers=["PCRFY"]),
            ThemeStock("LAC", "Lithium Americas", SupplyChainLayer.RAW_MATERIAL, 0.03),
            # Component (Battery)
            ThemeStock("PCRFY", "Panasonic", SupplyChainLayer.COMPONENT, 0.08,
                      suppliers=["ALB"], customers=["TSLA"]),
            ThemeStock("QS", "QuantumScape", SupplyChainLayer.COMPONENT, 0.03),
            # Manufacturer (EV Makers)
            ThemeStock("TSLA", "Tesla", SupplyChainLayer.MANUFACTURER, 0.20,
                      suppliers=["PCRFY"], customers=["CHPT"]),
            ThemeStock("RIVN", "Rivian", SupplyChainLayer.MANUFACTURER, 0.06),
            ThemeStock("LCID", "Lucid", SupplyChainLayer.MANUFACTURER, 0.04),
            ThemeStock("NIO", "NIO", SupplyChainLayer.MANUFACTURER, 0.05,
                      customers=["XPEV"]),
            ThemeStock("XPEV", "XPeng", SupplyChainLayer.MANUFACTURER, 0.04),
            ThemeStock("LI", "Li Auto", SupplyChainLayer.MANUFACTURER, 0.04),
            ThemeStock("F", "Ford", SupplyChainLayer.MANUFACTURER, 0.06),
            ThemeStock("GM", "General Motors", SupplyChainLayer.MANUFACTURER, 0.06),
            # Infrastructure (Charging)
            ThemeStock("CHPT", "ChargePoint", SupplyChainLayer.DISTRIBUTION, 0.05,
                      suppliers=["TSLA"]),
            ThemeStock("BLNK", "Blink Charging", SupplyChainLayer.DISTRIBUTION, 0.03),
            ThemeStock("EVGO", "EVgo", SupplyChainLayer.DISTRIBUTION, 0.02),
        ]
    },

    ThemeCategory.CLEAN_ENERGY: {
        "name": "Clean Energy Transition",
        "description": "재생에너지 + ESS + 그리드",
        "stocks": [
            # Equipment
            ThemeStock("FSLR", "First Solar", SupplyChainLayer.EQUIPMENT, 0.12,
                      customers=["NEE"]),
            ThemeStock("ENPH", "Enphase", SupplyChainLayer.EQUIPMENT, 0.10,
                      customers=["RUN"]),
            ThemeStock("SEDG", "SolarEdge", SupplyChainLayer.EQUIPMENT, 0.06),
            ThemeStock("CSIQ", "Canadian Solar", SupplyChainLayer.EQUIPMENT, 0.05),
            # Storage
            ThemeStock("FLUENCE", "Fluence Energy", SupplyChainLayer.COMPONENT, 0.05),
            # Wind
            ThemeStock("VWDRY", "Vestas Wind", SupplyChainLayer.EQUIPMENT, 0.06),
            # Utility
            ThemeStock("NEE", "NextEra Energy", SupplyChainLayer.END_USER, 0.15,
                      suppliers=["FSLR"]),
            ThemeStock("AES", "AES Corp", SupplyChainLayer.END_USER, 0.06),
            ThemeStock("BEP", "Brookfield Renewable", SupplyChainLayer.END_USER, 0.08),
            # Installer/Service
            ThemeStock("RUN", "Sunrun", SupplyChainLayer.DISTRIBUTION, 0.08,
                      suppliers=["ENPH"]),
            ThemeStock("NOVA", "Sunnova", SupplyChainLayer.DISTRIBUTION, 0.04),
            # Hydrogen
            ThemeStock("PLUG", "Plug Power", SupplyChainLayer.COMPONENT, 0.05),
            ThemeStock("BE", "Bloom Energy", SupplyChainLayer.COMPONENT, 0.05),
        ]
    },

    ThemeCategory.DEFENSE: {
        "name": "Defense & Aerospace",
        "description": "방산/항공우주 밸류체인",
        "stocks": [
            ThemeStock("LMT", "Lockheed Martin", SupplyChainLayer.INTEGRATOR, 0.18),
            ThemeStock("RTX", "RTX Corp", SupplyChainLayer.INTEGRATOR, 0.15),
            ThemeStock("NOC", "Northrop Grumman", SupplyChainLayer.INTEGRATOR, 0.12),
            ThemeStock("GD", "General Dynamics", SupplyChainLayer.INTEGRATOR, 0.12),
            ThemeStock("BA", "Boeing", SupplyChainLayer.MANUFACTURER, 0.10),
            ThemeStock("LHX", "L3Harris", SupplyChainLayer.COMPONENT, 0.08),
            ThemeStock("HII", "Huntington Ingalls", SupplyChainLayer.MANUFACTURER, 0.06),
            ThemeStock("KTOS", "Kratos Defense", SupplyChainLayer.COMPONENT, 0.05),
            ThemeStock("PLTR", "Palantir", SupplyChainLayer.END_USER, 0.08),
            ThemeStock("LDOS", "Leidos", SupplyChainLayer.END_USER, 0.06),
        ]
    },

    ThemeCategory.CYBERSECURITY: {
        "name": "Cybersecurity Shield",
        "description": "사이버보안 생태계",
        "stocks": [
            ThemeStock("CRWD", "CrowdStrike", SupplyChainLayer.END_USER, 0.15),
            ThemeStock("PANW", "Palo Alto Networks", SupplyChainLayer.END_USER, 0.15),
            ThemeStock("FTNT", "Fortinet", SupplyChainLayer.INTEGRATOR, 0.12),
            ThemeStock("ZS", "Zscaler", SupplyChainLayer.END_USER, 0.10),
            ThemeStock("OKTA", "Okta", SupplyChainLayer.COMPONENT, 0.08),
            ThemeStock("NET", "Cloudflare", SupplyChainLayer.COMPONENT, 0.10),
            ThemeStock("S", "SentinelOne", SupplyChainLayer.END_USER, 0.08),
            ThemeStock("CYBR", "CyberArk", SupplyChainLayer.COMPONENT, 0.07),
            ThemeStock("TENB", "Tenable", SupplyChainLayer.COMPONENT, 0.05),
            ThemeStock("QLYS", "Qualys", SupplyChainLayer.COMPONENT, 0.05),
            ThemeStock("RPD", "Rapid7", SupplyChainLayer.COMPONENT, 0.05),
        ]
    },

    ThemeCategory.BLOCKCHAIN: {
        "name": "Blockchain & Digital Assets",
        "description": "블록체인 인프라 및 관련 기업",
        "stocks": [
            ThemeStock("COIN", "Coinbase", SupplyChainLayer.DISTRIBUTION, 0.15),
            ThemeStock("MSTR", "MicroStrategy", SupplyChainLayer.END_USER, 0.12),
            ThemeStock("SQ", "Block Inc", SupplyChainLayer.INTEGRATOR, 0.12),
            ThemeStock("MARA", "Marathon Digital", SupplyChainLayer.MANUFACTURER, 0.08),
            ThemeStock("RIOT", "Riot Platforms", SupplyChainLayer.MANUFACTURER, 0.08),
            ThemeStock("CLSK", "CleanSpark", SupplyChainLayer.MANUFACTURER, 0.06),
            ThemeStock("HUT", "Hut 8 Mining", SupplyChainLayer.MANUFACTURER, 0.05),
            ThemeStock("GLXY", "Galaxy Digital", SupplyChainLayer.INTEGRATOR, 0.08),
            ThemeStock("SI", "Silvergate", SupplyChainLayer.DISTRIBUTION, 0.05),
            ThemeStock("HOOD", "Robinhood", SupplyChainLayer.DISTRIBUTION, 0.08),
            # Add crypto exposure
            ThemeStock("BTC-USD", "Bitcoin", SupplyChainLayer.RAW_MATERIAL, 0.08),
            ThemeStock("ETH-USD", "Ethereum", SupplyChainLayer.RAW_MATERIAL, 0.05),
        ]
    },
}


class SupplyChainGraph:
    """공급망 그래프 분석"""

    def __init__(self, stocks: List[ThemeStock]):
        self.stocks = stocks
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.DiGraph:
        """공급망 방향 그래프 구축"""
        G = nx.DiGraph()

        # 노드 추가
        for stock in self.stocks:
            G.add_node(
                stock.ticker,
                name=stock.name,
                layer=stock.layer.value,
                weight_base=stock.weight_base
            )

        # 엣지 추가 (supplier → customer)
        for stock in self.stocks:
            for customer in stock.customers:
                if any(s.ticker == customer for s in self.stocks):
                    G.add_edge(stock.ticker, customer, relationship="supplies_to")
            for supplier in stock.suppliers:
                if any(s.ticker == supplier for s in self.stocks):
                    G.add_edge(supplier, stock.ticker, relationship="supplies_to")

        return G

    def get_centrality_scores(self) -> Dict[str, float]:
        """중심성 점수 계산 (공급망에서 중요한 위치)"""
        if len(self.graph) == 0:
            return {}

        # PageRank for importance
        try:
            pagerank = nx.pagerank(self.graph, alpha=0.85)
        except:
            pagerank = {node: 1.0/len(self.graph) for node in self.graph.nodes()}

        # Betweenness for bottleneck detection
        try:
            betweenness = nx.betweenness_centrality(self.graph)
        except:
            betweenness = {node: 0.0 for node in self.graph.nodes()}

        # Combined score
        combined = {}
        for node in self.graph.nodes():
            combined[node] = 0.6 * pagerank.get(node, 0) + 0.4 * betweenness.get(node, 0)

        return combined

    def find_bottlenecks(self) -> List[str]:
        """병목 지점 (단일 공급원) 식별"""
        bottlenecks = []
        for node in self.graph.nodes():
            # 많은 downstream을 가지면서 대체재가 없는 경우
            out_degree = self.graph.out_degree(node)
            in_degree = self.graph.in_degree(node)

            if out_degree >= 2 and in_degree <= 1:
                bottlenecks.append(node)

        return bottlenecks

    def get_layer_distribution(self) -> Dict[str, List[str]]:
        """레이어별 종목 분포"""
        distribution = defaultdict(list)
        for node, data in self.graph.nodes(data=True):
            layer = data.get('layer', 'unknown')
            distribution[layer].append(node)
        return dict(distribution)

    def get_shock_propagation_path(self, source: str) -> List[Tuple[str, int]]:
        """특정 노드에서 시작하는 충격 전파 경로"""
        if source not in self.graph:
            return []

        paths = []
        visited = set()
        queue = [(source, 0)]

        while queue:
            node, depth = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            paths.append((node, depth))

            for successor in self.graph.successors(node):
                if successor not in visited:
                    queue.append((successor, depth + 1))

        return paths

    def visualize_structure(self) -> str:
        """텍스트 기반 그래프 시각화"""
        lines = ["Supply Chain Structure:"]
        lines.append("=" * 50)

        # Layer별 정렬
        layer_order = [
            SupplyChainLayer.RAW_MATERIAL,
            SupplyChainLayer.COMPONENT,
            SupplyChainLayer.EQUIPMENT,
            SupplyChainLayer.MANUFACTURER,
            SupplyChainLayer.INTEGRATOR,
            SupplyChainLayer.DISTRIBUTION,
            SupplyChainLayer.END_USER,
        ]

        for layer in layer_order:
            nodes = [n for n, d in self.graph.nodes(data=True)
                    if d.get('layer') == layer.value]
            if nodes:
                lines.append(f"\n[{layer.value.upper()}]")
                for node in nodes:
                    successors = list(self.graph.successors(node))
                    if successors:
                        lines.append(f"  {node} → {', '.join(successors)}")
                    else:
                        lines.append(f"  {node}")

        return "\n".join(lines)

    def generate_causality_chain(
        self,
        event: str = None,
        source_node: str = None,
        market_data: Dict[str, float] = None
    ) -> List[str]:
        """
        동적 인과관계 체인 생성

        소스 이론: "경제학은 인과관계(Causality)다. 엔비디아 수출 -> 삼성 영향처럼
        그래프로 설명해야 한다."

        Returns:
            List[str]: "Event -> Node(기업) -> Impact" 형태의 인과관계 체인들
        """
        chains = []

        # 1. 레이어별 전파 경로 (구조적 인과관계)
        layer_order = [
            SupplyChainLayer.RAW_MATERIAL,
            SupplyChainLayer.COMPONENT,
            SupplyChainLayer.EQUIPMENT,
            SupplyChainLayer.MANUFACTURER,
            SupplyChainLayer.INTEGRATOR,
            SupplyChainLayer.DISTRIBUTION,
            SupplyChainLayer.END_USER,
        ]

        # 레이어별 노드 수집
        layer_nodes = {}
        for layer in layer_order:
            nodes = [n for n, d in self.graph.nodes(data=True)
                    if d.get('layer') == layer.value]
            if nodes:
                layer_nodes[layer.value] = nodes

        # 2. 특정 이벤트 기반 인과관계 체인 생성
        event_templates = {
            'AI Demand Surge': {
                'upstream': ['ASML', 'AMAT', 'LRCX'],  # 장비
                'midstream': ['TSM', 'INTC'],          # 제조
                'downstream': ['NVDA', 'AMD', 'AVGO'], # 칩
                'impact': ['MSFT', 'GOOGL', 'AMZN'],   # 최종 사용자
                'effect': 'Utilization Rate Increase'
            },
            'Export Restriction': {
                'upstream': ['ASML'],
                'midstream': ['TSM', 'SSNLF'],
                'downstream': ['NVDA', 'AMD'],
                'impact': ['MSFT', 'GOOGL'],
                'effect': 'Supply Shortage'
            },
            'Chip Shortage': {
                'upstream': ['TSM'],
                'midstream': ['NVDA', 'AMD', 'AVGO'],
                'downstream': ['MSFT', 'GOOGL', 'AAPL'],
                'impact': [],
                'effect': 'Production Delay'
            },
            'EV Demand Surge': {
                'upstream': ['ALB', 'SQM'],         # 리튬
                'midstream': ['PCRFY'],             # 배터리
                'downstream': ['TSLA', 'RIVN'],    # 완성차
                'impact': ['CHPT'],                 # 충전
                'effect': 'Revenue Increase'
            },
            'Solar Demand Surge': {
                'upstream': ['FSLR', 'ENPH'],
                'midstream': ['RUN', 'NOVA'],
                'downstream': ['NEE', 'AES'],
                'impact': [],
                'effect': 'Capacity Expansion'
            }
        }

        # 3. 이벤트가 주어진 경우 해당 템플릿 사용
        if event and event in event_templates:
            template = event_templates[event]
            for upstream in template['upstream']:
                if upstream in self.graph.nodes():
                    for midstream in template['midstream']:
                        if midstream in self.graph.nodes():
                            for downstream in template['downstream']:
                                if downstream in self.graph.nodes():
                                    chain = f"{event} → {upstream} Revenue Up → {midstream} {template['effect']}"
                                    if downstream:
                                        chain += f" → {downstream} Impact"
                                    chains.append(chain)
                                    break  # 하나만 생성
                            break
                    break

        # 4. 소스 노드 기반 충격 전파 경로
        if source_node and source_node in self.graph.nodes():
            propagation = self.get_shock_propagation_path(source_node)
            if len(propagation) > 1:
                path_nodes = [p[0] for p in propagation[:4]]
                node_data = self.graph.nodes[source_node]
                layer = node_data.get('layer', 'unknown')

                # 레이어에 따른 이벤트 유형 추론
                if layer == 'equipment':
                    event_type = "Supply Disruption"
                elif layer == 'manufacturer':
                    event_type = "Production Halt"
                elif layer == 'integrator':
                    event_type = "Chip Shortage"
                else:
                    event_type = "Market Shock"

                chain = f"{event_type} @ {source_node} → {' → '.join(path_nodes[1:])}"
                chains.append(chain)

        # 5. 병목 기반 인과관계 (그래프에서 자동 추출)
        bottlenecks = self.find_bottlenecks()
        for bottleneck in bottlenecks[:2]:
            successors = list(self.graph.successors(bottleneck))
            if successors:
                node_data = self.graph.nodes[bottleneck]
                name = node_data.get('name', bottleneck)
                chain = f"Disruption @ {bottleneck}({name}) → {', '.join(successors[:3])} Affected"
                chains.append(chain)

        # 6. 중심성 기반 허브 노드 영향
        centrality = self.get_centrality_scores()
        top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:2]
        for node, score in top_central:
            successors = list(self.graph.successors(node))
            predecessors = list(self.graph.predecessors(node))
            if successors and predecessors:
                chain = f"Hub Node: {', '.join(predecessors[:2])} → {node} → {', '.join(successors[:2])}"
                chains.append(chain)

        # 7. 시장 데이터 기반 동적 인과관계 (가격 변동 반영)
        if market_data:
            significant_moves = {k: v for k, v in market_data.items() if abs(v) > 2.0}
            for ticker, change in sorted(significant_moves.items(), key=lambda x: abs(x[1]), reverse=True)[:2]:
                if ticker in self.graph.nodes():
                    direction = "Up" if change > 0 else "Down"
                    successors = list(self.graph.successors(ticker))
                    if successors:
                        chain = f"Market Move: {ticker} {change:+.1f}% {direction} → {', '.join(successors[:2])} Price Impact"
                        chains.append(chain)

        # 8. 기본 구조적 인과관계 체인 (이벤트 없을 때)
        if not chains and layer_nodes:
            # 레이어 순서대로 연결
            layers_with_nodes = [l for l in layer_order if l.value in layer_nodes]
            if len(layers_with_nodes) >= 3:
                first_layer = layers_with_nodes[0].value
                mid_layer = layers_with_nodes[len(layers_with_nodes)//2].value
                last_layer = layers_with_nodes[-1].value

                first_node = layer_nodes[first_layer][0]
                mid_node = layer_nodes[mid_layer][0]
                last_node = layer_nodes[last_layer][0]

                chains.append(f"Structural: {first_node} ({first_layer}) → {mid_node} ({mid_layer}) → {last_node} ({last_layer})")

        return chains if chains else ["No significant causality chain detected (stable supply chain)"]


class CustomETFBuilder:
    """Custom ETF 빌더"""

    def __init__(self):
        self.theme_templates = THEME_TEMPLATES

    def create_etf(
        self,
        category: ThemeCategory,
        custom_stocks: Optional[List[ThemeStock]] = None,
        custom_name: Optional[str] = None,
        custom_description: Optional[str] = None
    ) -> ThemeETF:
        """테마 ETF 생성"""

        if category == ThemeCategory.CUSTOM:
            if not custom_stocks:
                raise ValueError("Custom ETF requires custom_stocks")
            stocks = custom_stocks
            name = custom_name or "Custom Theme ETF"
            description = custom_description or "User-defined theme basket"
        else:
            template = self.theme_templates.get(category)
            if not template:
                raise ValueError(f"Unknown theme category: {category}")
            stocks = template["stocks"]
            name = template["name"]
            description = template["description"]

        # 공급망 그래프 구축
        supply_chain = SupplyChainGraph(stocks)

        # 기본 비중 계산
        target_weight = self._calculate_target_weights(stocks, supply_chain)

        return ThemeETF(
            name=name,
            category=category,
            description=description,
            stocks=stocks,
            target_weight=target_weight,
            supply_chain_graph=supply_chain.graph
        )

    def _calculate_target_weights(
        self,
        stocks: List[ThemeStock],
        supply_chain: SupplyChainGraph
    ) -> Dict[str, float]:
        """목표 비중 계산"""

        # 1. 기본 비중
        base_weights = {s.ticker: s.weight_base for s in stocks}

        # 2. 공급망 중심성 조정
        centrality = supply_chain.get_centrality_scores()

        # 3. 병목 지점 프리미엄
        bottlenecks = set(supply_chain.find_bottlenecks())

        # 최종 비중 계산
        final_weights = {}
        total = 0

        for stock in stocks:
            ticker = stock.ticker
            w = base_weights.get(ticker, 0.05)

            # 중심성 가중
            c = centrality.get(ticker, 0)
            w *= (1 + c * 0.2)  # 최대 20% 가중

            # 병목 프리미엄
            if ticker in bottlenecks:
                w *= 1.1  # 10% 프리미엄

            final_weights[ticker] = w
            total += w

        # 정규화
        for ticker in final_weights:
            final_weights[ticker] /= total

        return final_weights

    def adjust_weights_with_signals(
        self,
        etf: ThemeETF,
        momentum_scores: Optional[Dict[str, float]] = None,
        value_scores: Optional[Dict[str, float]] = None,
        quality_scores: Optional[Dict[str, float]] = None,
        momentum_tilt: float = 0.15,
        value_tilt: float = 0.10,
        quality_tilt: float = 0.05
    ) -> Dict[str, float]:
        """시그널 기반 비중 조정"""

        adjusted = etf.target_weight.copy()

        # Momentum Tilt
        if momentum_scores:
            mom_total = sum(momentum_scores.values())
            if mom_total > 0:
                for ticker in adjusted:
                    mom = momentum_scores.get(ticker, 0) / mom_total
                    adjusted[ticker] += momentum_tilt * (mom - 1/len(adjusted))

        # Value Tilt
        if value_scores:
            val_total = sum(value_scores.values())
            if val_total > 0:
                for ticker in adjusted:
                    val = value_scores.get(ticker, 0) / val_total
                    adjusted[ticker] += value_tilt * (val - 1/len(adjusted))

        # Quality Tilt
        if quality_scores:
            qual_total = sum(quality_scores.values())
            if qual_total > 0:
                for ticker in adjusted:
                    qual = quality_scores.get(ticker, 0) / qual_total
                    adjusted[ticker] += quality_tilt * (qual - 1/len(adjusted))

        # 정규화 (음수 방지)
        for ticker in adjusted:
            adjusted[ticker] = max(0.01, adjusted[ticker])

        total = sum(adjusted.values())
        for ticker in adjusted:
            adjusted[ticker] /= total

        return adjusted

    def generate_rebalance_signals(
        self,
        etf: ThemeETF,
        current_weights: Dict[str, float],
        threshold: float = 0.05
    ) -> List[Dict]:
        """리밸런싱 시그널 생성"""

        signals = []

        for ticker in etf.target_weight:
            target = etf.target_weight.get(ticker, 0)
            current = current_weights.get(ticker, 0)
            diff = target - current

            if abs(diff) > threshold:
                signals.append({
                    "ticker": ticker,
                    "action": "BUY" if diff > 0 else "SELL",
                    "current_weight": f"{current*100:.1f}%",
                    "target_weight": f"{target*100:.1f}%",
                    "adjustment": f"{diff*100:+.1f}%",
                    "priority": "HIGH" if abs(diff) > threshold * 2 else "MEDIUM"
                })

        # 우선순위 정렬
        signals.sort(key=lambda x: abs(float(x["adjustment"].rstrip('%'))), reverse=True)

        return signals

    def analyze_risk_concentration(
        self,
        etf: ThemeETF
    ) -> Dict:
        """리스크 집중도 분석"""

        supply_chain = SupplyChainGraph(etf.stocks)

        # 1. 레이어 집중도
        layer_dist = supply_chain.get_layer_distribution()
        layer_weights = {}
        for layer, tickers in layer_dist.items():
            layer_weights[layer] = sum(
                etf.target_weight.get(t, 0) for t in tickers
            )

        # 2. 병목 리스크
        bottlenecks = supply_chain.find_bottlenecks()
        bottleneck_weight = sum(
            etf.target_weight.get(t, 0) for t in bottlenecks
        )

        # 3. Top 5 집중도
        sorted_weights = sorted(
            etf.target_weight.items(),
            key=lambda x: x[1],
            reverse=True
        )
        top5_weight = sum(w for _, w in sorted_weights[:5])

        # 4. HHI (Herfindahl-Hirschman Index)
        hhi = sum(w**2 for w in etf.target_weight.values())

        return {
            "layer_concentration": layer_weights,
            "bottleneck_exposure": {
                "tickers": bottlenecks,
                "total_weight": f"{bottleneck_weight*100:.1f}%"
            },
            "top5_concentration": f"{top5_weight*100:.1f}%",
            "hhi": f"{hhi:.4f}",
            "diversification_score": f"{(1-hhi)*100:.1f}%",
            "risk_warnings": self._generate_risk_warnings(
                layer_weights, bottleneck_weight, top5_weight, hhi
            )
        }

    def _generate_risk_warnings(
        self,
        layer_weights: Dict[str, float],
        bottleneck_weight: float,
        top5_weight: float,
        hhi: float
    ) -> List[str]:
        """리스크 경고 생성"""
        warnings = []

        # 단일 레이어 과집중
        for layer, weight in layer_weights.items():
            if weight > 0.4:
                warnings.append(f"⚠️ {layer} 레이어에 {weight*100:.0f}% 집중")

        # 병목 노출
        if bottleneck_weight > 0.2:
            warnings.append(f"⚠️ 병목 지점에 {bottleneck_weight*100:.0f}% 노출 - 공급망 리스크")

        # Top5 집중
        if top5_weight > 0.6:
            warnings.append(f"⚠️ 상위 5개 종목에 {top5_weight*100:.0f}% 집중")

        # HHI
        if hhi > 0.15:
            warnings.append(f"⚠️ HHI {hhi:.3f} - 고집중 포트폴리오")

        return warnings

    def get_shock_impact_analysis(
        self,
        etf: ThemeETF,
        shock_ticker: str
    ) -> Dict:
        """특정 종목 충격 시 영향 분석"""

        supply_chain = SupplyChainGraph(etf.stocks)
        propagation = supply_chain.get_shock_propagation_path(shock_ticker)

        if not propagation:
            return {"error": f"{shock_ticker} not found in ETF"}

        # 영향받는 종목들
        affected = []
        total_affected_weight = 0

        for ticker, depth in propagation:
            weight = etf.target_weight.get(ticker, 0)
            total_affected_weight += weight
            affected.append({
                "ticker": ticker,
                "depth": depth,
                "weight": f"{weight*100:.1f}%",
                "impact": "DIRECT" if depth == 0 else f"INDIRECT (depth {depth})"
            })

        return {
            "shock_source": shock_ticker,
            "propagation_path": [t for t, _ in propagation],
            "affected_stocks": affected,
            "total_affected_weight": f"{total_affected_weight*100:.1f}%",
            "recommendation": self._generate_hedge_recommendation(
                shock_ticker, total_affected_weight
            )
        }

    def _generate_hedge_recommendation(
        self,
        shock_ticker: str,
        affected_weight: float
    ) -> str:
        """헤지 권고 생성"""
        if affected_weight > 0.5:
            return f"HIGH RISK: {shock_ticker} 충격 시 포트폴리오 {affected_weight*100:.0f}% 영향. 대체 공급원 확보 또는 비중 축소 권고"
        elif affected_weight > 0.3:
            return f"MEDIUM RISK: {shock_ticker} 충격 시 포트폴리오 {affected_weight*100:.0f}% 영향. 모니터링 강화 권고"
        else:
            return f"LOW RISK: {shock_ticker} 충격 시 제한적 영향 ({affected_weight*100:.0f}%)"


# =============================================================================
# ETF 비교 분석
# =============================================================================

class ETFComparator:
    """테마 ETF 간 비교 분석"""

    def __init__(self, etfs: List[ThemeETF]):
        self.etfs = etfs

    def find_overlapping_stocks(self) -> Dict[str, List[str]]:
        """중복 종목 찾기"""
        stock_etfs = defaultdict(list)

        for etf in self.etfs:
            for stock in etf.stocks:
                stock_etfs[stock.ticker].append(etf.name)

        return {
            ticker: etf_names
            for ticker, etf_names in stock_etfs.items()
            if len(etf_names) > 1
        }

    def calculate_correlation_estimate(self) -> pd.DataFrame:
        """테마 간 상관관계 추정 (종목 기반)"""
        n = len(self.etfs)
        corr_matrix = np.zeros((n, n))

        for i, etf1 in enumerate(self.etfs):
            tickers1 = set(s.ticker for s in etf1.stocks)
            for j, etf2 in enumerate(self.etfs):
                tickers2 = set(s.ticker for s in etf2.stocks)

                # Jaccard similarity
                intersection = len(tickers1 & tickers2)
                union = len(tickers1 | tickers2)

                corr_matrix[i, j] = intersection / union if union > 0 else 0

        names = [etf.name for etf in self.etfs]
        return pd.DataFrame(corr_matrix, index=names, columns=names)

    def get_combined_supply_chain_exposure(self) -> Dict[str, float]:
        """결합 포트폴리오의 공급망 레이어 노출도"""
        combined = defaultdict(float)
        total_weight = 0

        for etf in self.etfs:
            etf_weight = 1.0 / len(self.etfs)  # 균등 배분 가정

            for stock in etf.stocks:
                layer = stock.layer.value
                weight = etf.target_weight.get(stock.ticker, 0) * etf_weight
                combined[layer] += weight
                total_weight += weight

        # 정규화
        if total_weight > 0:
            for layer in combined:
                combined[layer] /= total_weight

        return dict(combined)


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Custom ETF Builder Test")
    print("=" * 60)

    builder = CustomETFBuilder()

    # 1. AI/Semiconductor ETF 생성
    print("\n1. Creating AI Semiconductor ETF...")
    ai_etf = builder.create_etf(ThemeCategory.AI_SEMICONDUCTOR)
    print(f"   Name: {ai_etf.name}")
    print(f"   Stocks: {len(ai_etf.stocks)}")
    print(f"   Description: {ai_etf.description}")

    # 2. 공급망 분석
    print("\n2. Supply Chain Analysis:")
    supply_chain = SupplyChainGraph(ai_etf.stocks)
    print(supply_chain.visualize_structure())

    # 3. 목표 비중
    print("\n3. Target Weights:")
    sorted_weights = sorted(ai_etf.target_weight.items(), key=lambda x: x[1], reverse=True)
    for ticker, weight in sorted_weights[:10]:
        print(f"   {ticker}: {weight*100:.1f}%")

    # 4. 리스크 집중도 분석
    print("\n4. Risk Concentration Analysis:")
    risk = builder.analyze_risk_concentration(ai_etf)
    print(f"   Top 5 Concentration: {risk['top5_concentration']}")
    print(f"   HHI: {risk['hhi']}")
    print(f"   Diversification Score: {risk['diversification_score']}")
    print(f"   Bottleneck Exposure: {risk['bottleneck_exposure']}")

    if risk['risk_warnings']:
        print("\n   Risk Warnings:")
        for warning in risk['risk_warnings']:
            print(f"   {warning}")

    # 5. 충격 영향 분석 (ASML 공급 중단 시나리오)
    print("\n5. Shock Impact Analysis (ASML disruption):")
    impact = builder.get_shock_impact_analysis(ai_etf, "ASML")
    print(f"   Propagation Path: {' → '.join(impact['propagation_path'][:5])}")
    print(f"   Total Affected Weight: {impact['total_affected_weight']}")
    print(f"   Recommendation: {impact['recommendation']}")

    # 6. 리밸런싱 시그널 (현재 비중 vs 목표)
    print("\n6. Rebalancing Signals:")
    # 임의의 현재 비중
    current_weights = {t: w * (0.8 + 0.4 * np.random.random())
                       for t, w in ai_etf.target_weight.items()}
    total = sum(current_weights.values())
    current_weights = {t: w/total for t, w in current_weights.items()}

    signals = builder.generate_rebalance_signals(ai_etf, current_weights, threshold=0.03)
    for sig in signals[:5]:
        print(f"   {sig['action']} {sig['ticker']}: {sig['current_weight']} → {sig['target_weight']} ({sig['adjustment']})")

    # 7. 다중 테마 비교
    print("\n7. Multi-Theme Comparison:")
    ev_etf = builder.create_etf(ThemeCategory.ELECTRIC_VEHICLE)
    clean_etf = builder.create_etf(ThemeCategory.CLEAN_ENERGY)

    comparator = ETFComparator([ai_etf, ev_etf, clean_etf])

    overlaps = comparator.find_overlapping_stocks()
    print(f"   Overlapping stocks: {len(overlaps)}")
    for ticker, etfs in list(overlaps.items())[:5]:
        print(f"     {ticker}: {', '.join(etfs)}")

    corr = comparator.calculate_correlation_estimate()
    print(f"\n   Theme Correlation (Jaccard):")
    print(corr.round(2).to_string())

    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
