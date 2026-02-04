#!/usr/bin/env python3
"""
Shock Propagation - Data Schemas
============================================================

Causality analysis result schemas

Economic Foundation:
    - Granger Causality: Granger (1969)
    - Lead-lag relationships: Cross-correlation analysis
    - Shock transmission: Network propagation

Contains:
    - LeadLagResult, GrangerResult
    - EconomicEdge, ShockPath
    - NodeAnalysis, PropagationAnalysis
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .enums import NodeLayer, CausalityStrength


class LeadLagResult:
    """Lead-Lag 분석 결과"""
    source: str
    target: str
    optimal_lag: int           # 양수: source가 lead, 음수: target이 lead
    max_correlation: float     # 최적 lag에서의 상관관계
    correlation_at_zero: float # lag=0에서의 상관관계
    is_leading: bool           # source가 target을 선행하는가?
    confidence: float          # 신뢰도 (상관관계 기반)


@dataclass
class GrangerResult:
    """Granger Causality 검정 결과"""
    source: str
    target: str
    optimal_lag: int
    f_statistic: float
    p_value: float
    strength: CausalityStrength
    is_significant: bool
    bidirectional: bool = False  # 양방향 인과관계


@dataclass
class EconomicEdge:
    """
    경제학적 인과 엣지 (Palantir Ontology Style)
    
    모든 엣지는 Impulse Response Path로 취급:
    - 충격이 Node A에서 발생하면 어떻게 Node B, C로 전파되는지 경제 이론에 기반하여 설명
    
    Attributes:
        source: 원인 노드
        target: 결과 노드
        sign: 인과 방향 (+: 양의 관계, -: 음의 관계)
        lag: 전파 시차 (거래일)
        time_horizon: 효과 지속 기간 (short/medium/long)
        mechanism: 전달 메커니즘 (monetary_transmission, risk_premium, etc.)
        theory_reference: 경제 이론 출처 (IS-LM, QTM, Taylor Rule, etc.)
        narrative: 자연어 설명
    """
    source: str
    target: str
    sign: str  # "+" or "-"
    lag: int = 0
    time_horizon: str = "short"  # "short" (1-5d), "medium" (1-4w), "long" (1m+)
    mechanism: str = ""
    theory_reference: str = ""
    narrative: str = ""
    p_value: float = 1.0
    strength: CausalityStrength = CausalityStrength.NONE
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source,
            'target': self.target,
            'sign': self.sign,
            'lag': self.lag,
            'time_horizon': self.time_horizon,
            'mechanism': self.mechanism,
            'theory_reference': self.theory_reference,
            'narrative': self.narrative,
            'p_value': self.p_value,
            'strength': self.strength.value
        }
    
    def to_arrow(self) -> str:
        """화살표 형태로 표현"""
        sign_symbol = "↑" if self.sign == "+" else "↓"
        return f"{self.source} --[{self.sign}, lag={self.lag}d]--> {self.target}{sign_symbol}"


@dataclass
class ShockPath:
    """충격 전파 경로"""
    source: str
    path: List[str]
    total_lag: int             # 전체 전파 시간 (일)
    cumulative_impact: float   # 누적 충격 강도
    bottleneck: Optional[str]  # 병목 노드


@dataclass
class NodeAnalysis:
    """노드 분석 결과"""
    node: str
    layer: NodeLayer
    in_degree: int             # 영향 받는 관계 수
    out_degree: int            # 영향 주는 관계 수
    leading_score: float       # 선행 점수 (out - in)
    betweenness: float         # 전파 중개 점수
    avg_lead_time: float       # 평균 선행 시간
    role: str                  # "LEADING", "LAGGING", "BRIDGE", "ISOLATED"


@dataclass
class PropagationAnalysis:
    """전체 전파 분석 결과"""
    timestamp: str
    nodes: List[NodeAnalysis]
    edges: List[Dict]
    critical_paths: List[ShockPath]
    leading_indicators: List[str]
    lagging_indicators: List[str]
    bridge_nodes: List[str]

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'nodes': [asdict(n) for n in self.nodes],
            'edges': self.edges,
            'critical_paths': [asdict(p) for p in self.critical_paths],
            'leading_indicators': self.leading_indicators,
            'lagging_indicators': self.lagging_indicators,
            'bridge_nodes': self.bridge_nodes
        }


# ============================================================================
# Layer Classification (경제학적 도메인 지식)
# ============================================================================

LAYER_MAPPING = {
    # Policy Layer
    'DFF': NodeLayer.POLICY,         # Fed Funds Rate
    'FEDFUNDS': NodeLayer.POLICY,
    'ECB_RATE': NodeLayer.POLICY,

    # Liquidity Layer
    'RRP': NodeLayer.LIQUIDITY,      # Reverse Repo
    'TGA': NodeLayer.LIQUIDITY,      # Treasury General Account
    'M2': NodeLayer.LIQUIDITY,
    'USDT_SUPPLY': NodeLayer.LIQUIDITY,
    'USDC_SUPPLY': NodeLayer.LIQUIDITY,
    'NET_LIQUIDITY': NodeLayer.LIQUIDITY,

    # Risk Premium Layer
    'VIX': NodeLayer.RISK_PREMIUM,
    '^VIX': NodeLayer.RISK_PREMIUM,
    'VIXCLS': NodeLayer.RISK_PREMIUM,
    'HY_SPREAD': NodeLayer.RISK_PREMIUM,
    'BAMLH0A0HYM2': NodeLayer.RISK_PREMIUM,
    'CREDIT_SPREAD': NodeLayer.RISK_PREMIUM,
    'T10Y2Y': NodeLayer.RISK_PREMIUM,

    # Asset Price Layer
    'SPY': NodeLayer.ASSET_PRICE,
    'QQQ': NodeLayer.ASSET_PRICE,
    'TLT': NodeLayer.ASSET_PRICE,
    'GLD': NodeLayer.ASSET_PRICE,
    'BTC': NodeLayer.ASSET_PRICE,
    'BTC-USD': NodeLayer.ASSET_PRICE,
    'ETH-USD': NodeLayer.ASSET_PRICE,
    'DXY': NodeLayer.ASSET_PRICE,
    'DX-Y.NYB': NodeLayer.ASSET_PRICE,
}


# ============================================================================
# Economic Transmission Templates (Domain Knowledge)
# ============================================================================

TRANSMISSION_TEMPLATES: Dict[Tuple[str, str], Dict] = {
    # -------------------------------------------------------------------------
    # IS-LM Model: M↑ → r↓ → I↑ → Y↑
    # "통화 공급 증가 → 이자율 하락 → 투자 증가 → 소득 증가"
    # -------------------------------------------------------------------------
    ("M2", "DFF"): {
        "sign": "-",
        "mechanism": "liquidity_preference",
        "theory_reference": "IS-LM",
        "time_horizon": "short",
        "narrative": "통화 공급 증가는 유동성 선호 이론에 따라 이자율 하락 압력을 가함"
    },
    ("DFF", "SPY"): {
        "sign": "-",
        "mechanism": "discount_rate",
        "theory_reference": "IS-LM",
        "time_horizon": "medium",
        "narrative": "금리 하락 시 할인율 감소로 주식 현재가치 상승"
    },
    ("DFF", "TLT"): {
        "sign": "-",
        "mechanism": "bond_pricing",
        "theory_reference": "Fixed Income",
        "time_horizon": "short",
        "narrative": "금리 하락 시 채권 가격 상승 (역의 관계)"
    },
    
    # -------------------------------------------------------------------------
    # Quantity Theory of Money (QTM): MV = PY
    # 장기적으로 M↑ → P↑ (통화 중립성)
    # -------------------------------------------------------------------------
    ("M2", "GLD"): {
        "sign": "+",
        "mechanism": "inflation_hedge",
        "theory_reference": "QTM",
        "time_horizon": "long",
        "narrative": "장기적 통화 증가는 인플레이션 기대를 통해 금 가격 상승"
    },
    ("M2", "BTC-USD"): {
        "sign": "+",
        "mechanism": "inflation_hedge",
        "theory_reference": "QTM",
        "time_horizon": "long",
        "narrative": "비트코인은 디지털 금으로서 통화 팽창에 대한 인플레이션 헤지"
    },
    
    # -------------------------------------------------------------------------
    # Risk Premium Channel
    # VIX↑ → 위험자산↓, 안전자산↑
    # -------------------------------------------------------------------------
    ("VIX", "SPY"): {
        "sign": "-",
        "mechanism": "risk_premium",
        "theory_reference": "CAPM",
        "time_horizon": "short",
        "narrative": "변동성 상승 시 주식 위험 프리미엄 증가로 가격 하락"
    },
    ("VIX", "TLT"): {
        "sign": "+",
        "mechanism": "flight_to_safety",
        "theory_reference": "Risk Parity",
        "time_horizon": "short",
        "narrative": "변동성 급등 시 안전자산 선호로 국채 가격 상승"
    },
    ("VIX", "GLD"): {
        "sign": "+",
        "mechanism": "safe_haven",
        "theory_reference": "Portfolio Theory",
        "time_horizon": "short",
        "narrative": "공포 심리 상승 시 금으로 자금 이동"
    },
    
    # -------------------------------------------------------------------------
    # Crypto-Treasury Feedback Loop (Stablecoin ↔ Treasury)
    # USDT 발행 → Treasury 매입 → 수익률 하락 → Risk-On → Crypto 상승 → USDT 추가 발행
    # -------------------------------------------------------------------------
    ("USDT_SUPPLY", "TLT"): {
        "sign": "+",
        "mechanism": "stablecoin_treasury_demand",
        "theory_reference": "Crypto Liquidity",
        "time_horizon": "short",
        "narrative": "스테이블코인 발행 증가 → 담보로 국채 매입 → 채권 가격 상승"
    },
    ("TLT", "BTC-USD"): {
        "sign": "+",
        "mechanism": "liquidity_spillover",
        "theory_reference": "Crypto Liquidity",
        "time_horizon": "short",
        "narrative": "채권 수익률 하락 → 수익 추구 자금이 크립토로 이동"
    },
    ("BTC-USD", "USDT_SUPPLY"): {
        "sign": "+",
        "mechanism": "crypto_demand",
        "theory_reference": "Crypto Liquidity",
        "time_horizon": "short",
        "narrative": "크립토 상승 → 거래 수요 증가 → 스테이블코인 추가 발행 (피드백 루프)"
    },
    
    # -------------------------------------------------------------------------
    # Taylor Rule: π↑ → FFR↑ → Asset↓
    # -------------------------------------------------------------------------
    ("DFF", "QQQ"): {
        "sign": "-",
        "mechanism": "growth_stock_sensitivity",
        "theory_reference": "Taylor Rule",
        "time_horizon": "short",
        "narrative": "성장주는 금리 민감도 높음 - 금리 상승 시 QQQ 하락"
    },
    
    # -------------------------------------------------------------------------
    # Credit Spread Channel
    # -------------------------------------------------------------------------
    ("HY_SPREAD", "SPY"): {
        "sign": "-",
        "mechanism": "credit_risk_premium",
        "theory_reference": "Credit Cycle",
        "time_horizon": "medium",
        "narrative": "하이일드 스프레드 확대는 신용 환경 악화 → 주식 하락"
    },
    ("HY_SPREAD", "BTC-USD"): {
        "sign": "-",
        "mechanism": "risk_off_flow",
        "theory_reference": "Credit Cycle",
        "time_horizon": "short",
        "narrative": "신용 스프레드 확대 시 위험자산 회피로 크립토 하락"
    },
    
    # -------------------------------------------------------------------------
    # Net Liquidity (Fed Balance Sheet - TGA - RRP)
    # -------------------------------------------------------------------------
    ("NET_LIQUIDITY", "SPY"): {
        "sign": "+",
        "mechanism": "liquidity_injection",
        "theory_reference": "Fed Put",
        "time_horizon": "short",
        "narrative": "순유동성 증가는 위험자산 가격 상승 지지"
    },
    ("NET_LIQUIDITY", "BTC-USD"): {
        "sign": "+",
        "mechanism": "liquidity_injection",
        "theory_reference": "Fed Put",
        "time_horizon": "short",
        "narrative": "유동성 확장 시 크립토 자산도 동반 상승"
    },
    ("RRP", "NET_LIQUIDITY"): {
        "sign": "-",
        "mechanism": "liquidity_drain",
        "theory_reference": "Fed Operations",
        "time_horizon": "short",
        "narrative": "역레포 증가는 시스템에서 유동성 흡수"
    },
    ("TGA", "NET_LIQUIDITY"): {
        "sign": "-",
        "mechanism": "treasury_cash_buildup",
        "theory_reference": "Treasury Operations",
        "time_horizon": "short",
        "narrative": "재무부 현금 축적은 시스템 유동성 감소"
    },
}


def get_node_layer(node_name: str) -> NodeLayer:
    """노드의 경제학적 레이어 결정"""
    # 직접 매핑 확인
    if node_name in LAYER_MAPPING:
        return LAYER_MAPPING[node_name]

    # 패턴 매칭
    name_upper = node_name.upper()
    if any(x in name_upper for x in ['FED', 'RATE', 'POLICY']):
        return NodeLayer.POLICY
    if any(x in name_upper for x in ['LIQUID', 'M2', 'RRP', 'TGA', 'STABLE']):
        return NodeLayer.LIQUIDITY
    if any(x in name_upper for x in ['VIX', 'SPREAD', 'CREDIT', 'YIELD']):
        return NodeLayer.RISK_PREMIUM

    return NodeLayer.ASSET_PRICE  # 기본값


def get_economic_edge(
    source: str, 
    target: str, 
    lag: int = 0,
    p_value: float = 0.05,
    correlation: float = 0.0
) -> EconomicEdge:
    """
    소스-타겟 쌍에 대한 경제학적 엣지 생성
    
    TRANSMISSION_TEMPLATES에서 도메인 지식을 가져오고,
    없으면 상관관계 부호로 기본 엣지 생성
    """
    template = TRANSMISSION_TEMPLATES.get((source, target), None)
    
    if template:
        # 템플릿에서 경제 이론 메타데이터 가져오기
        strength = CausalityStrength.STRONG if p_value < 0.01 else \
                   CausalityStrength.MODERATE if p_value < 0.05 else \
                   CausalityStrength.WEAK if p_value < 0.10 else \
                   CausalityStrength.NONE
        
        return EconomicEdge(
            source=source,
            target=target,
            sign=template["sign"],
            lag=lag,
            time_horizon=template.get("time_horizon", "short"),
            mechanism=template.get("mechanism", ""),
            theory_reference=template.get("theory_reference", ""),
            narrative=template.get("narrative", ""),
            p_value=p_value,
            strength=strength
        )
    else:
        # 템플릿 없으면 상관관계 부호로 기본 생성
        sign = "+" if correlation >= 0 else "-"
        source_layer = get_node_layer(source)
        target_layer = get_node_layer(target)
        
        # 레이어 기반 메커니즘 추론
        if source_layer == NodeLayer.POLICY:
            mechanism = "monetary_policy"
        elif source_layer == NodeLayer.LIQUIDITY:
            mechanism = "liquidity_channel"
        elif source_layer == NodeLayer.RISK_PREMIUM:
            mechanism = "risk_premium_channel"
        else:
            mechanism = "market_correlation"
        
        strength = CausalityStrength.STRONG if p_value < 0.01 else \
                   CausalityStrength.MODERATE if p_value < 0.05 else \
                   CausalityStrength.WEAK if p_value < 0.10 else \
                   CausalityStrength.NONE
        
        return EconomicEdge(
            source=source,
            target=target,
            sign=sign,
            lag=lag,
            time_horizon="short",
            mechanism=mechanism,
            theory_reference="Statistical",
            narrative=f"{source} {'positively' if sign=='+' else 'negatively'} affects {target}",
            p_value=p_value,
            strength=strength
        )


def generate_shock_narrative(path: List[str], edges: List[EconomicEdge]) -> str:
    """
    충격 전파 경로에 대한 경제학적 서사 생성
    
    Example output:
    "Fed 금리 인하 (DFF↓) → 유동성 증가 (M2↑) → 위험 프리미엄 감소 (VIX↓) → 주식 상승 (SPY↑)"
    
    Args:
        path: 노드 경로 리스트 ['DFF', 'M2', 'VIX', 'SPY']
        edges: 경로의 EconomicEdge 리스트
    
    Returns:
        자연어 서사 문자열
    """
    if len(path) < 2:
        return f"단일 노드: {path[0]}"
    
    narratives = []
    
    for i, (source, target) in enumerate(zip(path[:-1], path[1:])):
        if i < len(edges):
            edge = edges[i]
            sign_symbol = "↑" if edge.sign == "+" else "↓"
            
            if edge.narrative:
                narratives.append(f"[{edge.theory_reference}] {edge.narrative}")
            else:
                narratives.append(f"{source} → {target}{sign_symbol}")
    
    # 요약 생성
    first_node = path[0]
    last_node = path[-1]
    total_lag = sum(e.lag for e in edges)
    
    summary = f"\n📊 충격 전파 요약: {first_node} → ... → {last_node}\n"
    summary += f"   전파 시간: {total_lag}일\n"
    summary += f"   경로 길이: {len(path)}개 노드\n"
    
    return "\n→ ".join(narratives) + summary


def generate_impulse_response_text(
    shock_source: str,
    shock_magnitude: float,
    affected_nodes: Dict[str, float]
) -> str:
    """
    임펄스 반응 분석 텍스트 생성
    
    Args:
        shock_source: 충격 발생 노드
        shock_magnitude: 충격 크기 (예: -0.10 = -10%)
        affected_nodes: {노드명: 영향도}
    
    Returns:
        자연어 분석 텍스트
    """
    direction = "하락" if shock_magnitude < 0 else "상승"
    pct = abs(shock_magnitude) * 100
    
    lines = [
        f"# 임펄스 반응 분석 (Impulse Response)",
        f"",
        f"## 충격 정의",
        f"- 충격 노드: **{shock_source}**",
        f"- 충격 크기: **{pct:.1f}% {direction}**",
        f"",
        f"## 전파 효과",
    ]
    
    # 영향도 순 정렬
    sorted_effects = sorted(affected_nodes.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for node, impact in sorted_effects[:10]:
        impact_pct = impact * 100
        impact_dir = "+" if impact > 0 else ""
        
        # 템플릿에서 메커니즘 가져오기
        template = TRANSMISSION_TEMPLATES.get((shock_source, node), {})
        mechanism = template.get("mechanism", "indirect_effect")
        theory = template.get("theory_reference", "")
        
        theory_str = f" [{theory}]" if theory else ""
        lines.append(f"| {node} | {impact_dir}{impact_pct:.2f}% | {mechanism}{theory_str} |")
    
    lines.append("")
    lines.append("## 경제학적 해석")
    
    # 주요 해석 추가
    template = TRANSMISSION_TEMPLATES.get((shock_source, list(affected_nodes.keys())[0] if affected_nodes else ""), {})
    if template.get("narrative"):
        lines.append(f"> {template['narrative']}")
    
    return "\n".join(lines)


# ============================================================================
# Lead-Lag Analysis
# ============================================================================
