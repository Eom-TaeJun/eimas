"""
Shock Propagation Graph (SPG)
==============================
거시지표와 자산 간의 인과관계를 규명하는 충격 전파 네트워크

경제학적 배경:
- 단순 상관관계(Correlation)가 아닌 인과관계(Causality) 규명
- Granger Causality: "X가 Y를 예측하는 데 도움이 되는가?"
- Impulse Response: 충격이 어떻게 전파되는가?

핵심 기능:
1. Lead-Lag Analysis: 최적 시차 탐색
2. Granger Causality: 통계적 인과관계 검정
3. DAG Construction: 방향성 그래프 구축
4. Critical Path: 최장 전파 경로 탐색
5. Node Centrality: 선행/후행 지표 구분

References:
- Granger (1969): "Investigating Causal Relations by Econometric Models"
- Palantir Ontology: Event-based relationship modeling
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# Optional imports
try:
    from statsmodels.tsa.stattools import grangercausalitytests, adfuller
    from statsmodels.tsa.api import VAR
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("[WARN] statsmodels not available. Granger tests will use simplified method.")

try:
    from scipy import stats
    from scipy.signal import correlate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class NodeLayer(Enum):
    """
    노드 레이어 (경제학적 계층)

    충격 전파 순서:
    POLICY → LIQUIDITY → RISK_PREMIUM → ASSET_PRICE
    """
    POLICY = 1        # Fed Funds, ECB Rate
    LIQUIDITY = 2     # RRP, TGA, M2, Stablecoin Supply
    RISK_PREMIUM = 3  # VIX, Credit Spread, HY Spread
    ASSET_PRICE = 4   # SPY, QQQ, BTC, Gold
    UNKNOWN = 5


class CausalityStrength(Enum):
    """인과관계 강도"""
    STRONG = "strong"      # p < 0.01
    MODERATE = "moderate"  # p < 0.05
    WEAK = "weak"          # p < 0.10
    NONE = "none"          # p >= 0.10


@dataclass
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

class LeadLagAnalyzer:
    """
    Lead-Lag 관계 분석

    Cross-correlation at multiple lags를 통해
    어떤 시계열이 다른 시계열을 선행하는지 탐색
    """

    def __init__(self, max_lag: int = 20):
        self.max_lag = max_lag

    def analyze(
        self,
        source: pd.Series,
        target: pd.Series,
        source_name: str = "source",
        target_name: str = "target"
    ) -> LeadLagResult:
        """
        Lead-Lag 분석 수행

        Args:
            source: 소스 시계열
            target: 타겟 시계열

        Returns:
            LeadLagResult
        """
        # 데이터 정렬 및 결측치 처리
        df = pd.DataFrame({'source': source, 'target': target}).dropna()

        if len(df) < self.max_lag * 2:
            return LeadLagResult(
                source=source_name,
                target=target_name,
                optimal_lag=0,
                max_correlation=0.0,
                correlation_at_zero=df['source'].corr(df['target']),
                is_leading=False,
                confidence=0.0
            )

        correlations = {}

        for lag in range(-self.max_lag, self.max_lag + 1):
            if lag > 0:
                # source가 target을 lead (source[t] → target[t+lag])
                corr = df['source'].iloc[:-lag].corr(df['target'].iloc[lag:])
            elif lag < 0:
                # target이 source를 lead
                corr = df['source'].iloc[-lag:].corr(df['target'].iloc[:lag])
            else:
                corr = df['source'].corr(df['target'])

            correlations[lag] = corr if not np.isnan(corr) else 0.0

        # 최대 상관관계 lag 찾기
        optimal_lag = max(correlations, key=lambda k: abs(correlations[k]))
        max_corr = correlations[optimal_lag]
        zero_corr = correlations[0]

        # 신뢰도: 최적 lag의 상관관계가 0보다 얼마나 높은지
        confidence = abs(max_corr) - abs(zero_corr)
        confidence = max(0, min(1, confidence * 5))  # 0-1 정규화

        return LeadLagResult(
            source=source_name,
            target=target_name,
            optimal_lag=optimal_lag,
            max_correlation=max_corr,
            correlation_at_zero=zero_corr,
            is_leading=(optimal_lag > 0 and max_corr > 0) or (optimal_lag < 0 and max_corr < 0),
            confidence=confidence
        )

    def analyze_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        모든 변수 쌍에 대한 Lead-Lag 매트릭스 생성

        Returns:
            DataFrame with optimal lags (양수 = row가 column을 lead)
        """
        columns = data.columns.tolist()
        n = len(columns)

        lag_matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=columns,
            columns=columns
        )

        for i, source in enumerate(columns):
            for j, target in enumerate(columns):
                if i != j:
                    result = self.analyze(
                        data[source], data[target],
                        source, target
                    )
                    lag_matrix.loc[source, target] = result.optimal_lag

        return lag_matrix


# ============================================================================
# Granger Causality
# ============================================================================

class GrangerCausalityAnalyzer:
    """
    Granger Causality 검정

    "X가 Y를 Granger-cause 한다" =
    "X의 과거 정보가 Y의 예측에 통계적으로 유의한 기여를 한다"

    주의: Granger Causality ≠ True Causality
    하지만 예측적 관계의 좋은 프록시
    """

    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        self.max_lag = max_lag
        self.significance_level = significance_level

    def test(
        self,
        source: pd.Series,
        target: pd.Series,
        source_name: str = "source",
        target_name: str = "target"
    ) -> GrangerResult:
        """
        Granger Causality 검정

        H0: source does not Granger-cause target
        H1: source Granger-causes target
        """
        # 데이터 준비
        df = pd.DataFrame({'target': target, 'source': source}).dropna()

        if len(df) < self.max_lag * 3:
            return GrangerResult(
                source=source_name,
                target=target_name,
                optimal_lag=0,
                f_statistic=0.0,
                p_value=1.0,
                strength=CausalityStrength.NONE,
                is_significant=False
            )

        if not STATSMODELS_AVAILABLE:
            # Simplified fallback using correlation
            return self._simplified_test(df, source_name, target_name)

        try:
            # Granger Causality 검정
            result = grangercausalitytests(
                df[['target', 'source']],
                maxlag=self.max_lag,
                verbose=False
            )

            # 최적 lag 및 p-value 추출
            best_lag = 1
            best_pvalue = 1.0
            best_fstat = 0.0

            for lag in range(1, self.max_lag + 1):
                if lag in result:
                    # F-test 결과 사용
                    ftest = result[lag][0]['ssr_ftest']
                    pvalue = ftest[1]
                    fstat = ftest[0]

                    if pvalue < best_pvalue:
                        best_pvalue = pvalue
                        best_lag = lag
                        best_fstat = fstat

            # 강도 판정
            if best_pvalue < 0.01:
                strength = CausalityStrength.STRONG
            elif best_pvalue < 0.05:
                strength = CausalityStrength.MODERATE
            elif best_pvalue < 0.10:
                strength = CausalityStrength.WEAK
            else:
                strength = CausalityStrength.NONE

            return GrangerResult(
                source=source_name,
                target=target_name,
                optimal_lag=best_lag,
                f_statistic=best_fstat,
                p_value=best_pvalue,
                strength=strength,
                is_significant=(best_pvalue < self.significance_level)
            )

        except Exception as e:
            # 에러 시 fallback
            return self._simplified_test(df, source_name, target_name)

    def _simplified_test(
        self,
        df: pd.DataFrame,
        source_name: str,
        target_name: str
    ) -> GrangerResult:
        """Simplified causality test using lagged correlation"""
        lead_lag = LeadLagAnalyzer(max_lag=self.max_lag)
        result = lead_lag.analyze(
            df['source'], df['target'],
            source_name, target_name
        )

        # 상관관계 기반 유사 p-value 계산
        n = len(df)
        r = result.max_correlation
        if abs(r) > 0:
            t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
            # 근사 p-value
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if SCIPY_AVAILABLE else 0.05
        else:
            t_stat = 0
            p_value = 1.0

        if p_value < 0.01:
            strength = CausalityStrength.STRONG
        elif p_value < 0.05:
            strength = CausalityStrength.MODERATE
        elif p_value < 0.10:
            strength = CausalityStrength.WEAK
        else:
            strength = CausalityStrength.NONE

        return GrangerResult(
            source=source_name,
            target=target_name,
            optimal_lag=abs(result.optimal_lag),
            f_statistic=t_stat**2,
            p_value=p_value,
            strength=strength,
            is_significant=(p_value < self.significance_level)
        )

    def test_bidirectional(
        self,
        series1: pd.Series,
        series2: pd.Series,
        name1: str,
        name2: str
    ) -> Tuple[GrangerResult, GrangerResult]:
        """양방향 Granger Causality 검정"""
        result_1to2 = self.test(series1, series2, name1, name2)
        result_2to1 = self.test(series2, series1, name2, name1)

        # 양방향 여부 업데이트
        if result_1to2.is_significant and result_2to1.is_significant:
            result_1to2.bidirectional = True
            result_2to1.bidirectional = True

        return result_1to2, result_2to1


# ============================================================================
# Shock Propagation Graph
# ============================================================================

class ShockPropagationGraph:
    """
    충격 전파 그래프

    경제 시스템에서 충격이 어떻게 전파되는지를
    방향성 그래프(DAG)로 모델링
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_lag: int = 20,
        enforce_layer_order: bool = True
    ):
        self.significance_level = significance_level
        self.max_lag = max_lag
        self.enforce_layer_order = enforce_layer_order

        self.graph = nx.DiGraph()
        self.lead_lag_analyzer = LeadLagAnalyzer(max_lag)
        self.granger_analyzer = GrangerCausalityAnalyzer(max_lag, significance_level)

        # 결과 저장
        self.lead_lag_results: Dict[Tuple[str, str], LeadLagResult] = {}
        self.granger_results: Dict[Tuple[str, str], GrangerResult] = {}

    def build_from_data(
        self,
        data: pd.DataFrame,
        min_observations: int = 60
    ) -> nx.DiGraph:
        """
        시계열 데이터에서 충격 전파 그래프 구축

        Args:
            data: 시계열 DataFrame (columns = variables)
            min_observations: 최소 관측치 수

        Returns:
            NetworkX DiGraph
        """
        # 충분한 데이터가 있는 변수만 선택
        valid_columns = [
            col for col in data.columns
            if data[col].dropna().shape[0] >= min_observations
        ]

        if len(valid_columns) < 2:
            print("[SPG] Not enough valid columns")
            return self.graph

        print(f"[SPG] Building graph with {len(valid_columns)} variables")

        # 노드 추가
        for col in valid_columns:
            layer = get_node_layer(col)
            self.graph.add_node(
                col,
                layer=layer.value,
                layer_name=layer.name
            )

        # 엣지 추가 (Granger Causality 기반)
        n_edges = 0
        for i, source in enumerate(valid_columns):
            for j, target in enumerate(valid_columns):
                if i == j:
                    continue

                # Lead-Lag 분석
                ll_result = self.lead_lag_analyzer.analyze(
                    data[source], data[target],
                    source, target
                )
                self.lead_lag_results[(source, target)] = ll_result

                # Granger Causality 검정
                gc_result = self.granger_analyzer.test(
                    data[source], data[target],
                    source, target
                )
                self.granger_results[(source, target)] = gc_result

                # 유의미한 인과관계만 엣지로 추가
                if gc_result.is_significant:
                    # Layer 순서 강제 (선택적)
                    source_layer = get_node_layer(source)
                    target_layer = get_node_layer(target)

                    if self.enforce_layer_order and source_layer.value > target_layer.value:
                        # 역방향 (하위 레이어 → 상위 레이어)
                        # 피드백 루프로 표시
                        edge_type = "feedback"
                    else:
                        edge_type = "propagation"

                    self.graph.add_edge(
                        source, target,
                        lag=gc_result.optimal_lag,
                        strength=gc_result.strength.value,
                        p_value=gc_result.p_value,
                        correlation=ll_result.max_correlation,
                        edge_type=edge_type
                    )
                    n_edges += 1

        print(f"[SPG] Added {n_edges} significant edges")
        return self.graph

    def find_critical_path(
        self,
        source: str,
        max_depth: int = 10
    ) -> Optional[ShockPath]:
        """
        특정 소스에서 시작하는 최장 충격 전파 경로 탐색

        Critical Path = 가장 긴 전파 체인
        """
        if source not in self.graph:
            return None

        def dfs(node: str, path: List[str], total_lag: int, visited: Set[str]) -> Tuple[List[str], int]:
            """DFS로 최장 경로 탐색"""
            best_path = path
            best_lag = total_lag

            if len(path) >= max_depth:
                return best_path, best_lag

            for neighbor in self.graph.successors(node):
                if neighbor not in visited:
                    edge_data = self.graph.edges[node, neighbor]
                    new_lag = total_lag + edge_data.get('lag', 1)
                    new_path = path + [neighbor]

                    result_path, result_lag = dfs(
                        neighbor,
                        new_path,
                        new_lag,
                        visited | {neighbor}
                    )

                    if len(result_path) > len(best_path):
                        best_path = result_path
                        best_lag = result_lag

            return best_path, best_lag

        path, total_lag = dfs(source, [source], 0, {source})

        if len(path) <= 1:
            return None

        # 누적 충격 강도 계산
        cumulative_impact = 1.0
        for i in range(len(path) - 1):
            edge_data = self.graph.edges[path[i], path[i+1]]
            corr = abs(edge_data.get('correlation', 0.5))
            cumulative_impact *= corr

        # 병목 노드 탐색 (betweenness 가장 높은 노드)
        betweenness = nx.betweenness_centrality(self.graph)
        path_nodes = path[1:-1]  # 시작/끝 제외
        bottleneck = max(path_nodes, key=lambda n: betweenness.get(n, 0)) if path_nodes else None

        return ShockPath(
            source=source,
            path=path,
            total_lag=total_lag,
            cumulative_impact=cumulative_impact,
            bottleneck=bottleneck
        )

    def find_all_critical_paths(self, top_n: int = 5) -> List[ShockPath]:
        """모든 소스에서 시작하는 Critical Path 탐색"""
        paths = []

        for node in self.graph.nodes:
            layer = get_node_layer(node)
            # Policy/Liquidity 레이어만 소스로
            if layer in [NodeLayer.POLICY, NodeLayer.LIQUIDITY]:
                path = self.find_critical_path(node)
                if path and len(path.path) > 2:
                    paths.append(path)

        # 경로 길이 기준 정렬
        paths.sort(key=lambda p: len(p.path), reverse=True)
        return paths[:top_n]

    def analyze_nodes(self) -> List[NodeAnalysis]:
        """모든 노드 분석"""
        analyses = []

        betweenness = nx.betweenness_centrality(self.graph)

        for node in self.graph.nodes:
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)

            # 평균 선행 시간 계산
            out_edges = self.graph.out_edges(node, data=True)
            avg_lead_time = np.mean([
                e[2].get('lag', 0) for e in out_edges
            ]) if out_edges else 0

            # 역할 판정
            leading_score = out_deg - in_deg
            if leading_score > 2:
                role = "LEADING"
            elif leading_score < -2:
                role = "LAGGING"
            elif betweenness.get(node, 0) > 0.1:
                role = "BRIDGE"
            elif in_deg == 0 and out_deg == 0:
                role = "ISOLATED"
            else:
                role = "NEUTRAL"

            analyses.append(NodeAnalysis(
                node=node,
                layer=get_node_layer(node),
                in_degree=in_deg,
                out_degree=out_deg,
                leading_score=leading_score,
                betweenness=betweenness.get(node, 0),
                avg_lead_time=avg_lead_time,
                role=role
            ))

        return analyses

    def get_leading_indicators(self, top_n: int = 5) -> List[str]:
        """선행 지표 추출"""
        analyses = self.analyze_nodes()
        leading = [a for a in analyses if a.role == "LEADING"]
        leading.sort(key=lambda a: a.leading_score, reverse=True)
        return [a.node for a in leading[:top_n]]

    def get_lagging_indicators(self, top_n: int = 5) -> List[str]:
        """후행 지표 추출"""
        analyses = self.analyze_nodes()
        lagging = [a for a in analyses if a.role == "LAGGING"]
        lagging.sort(key=lambda a: a.leading_score)
        return [a.node for a in lagging[:top_n]]

    def get_bridge_nodes(self) -> List[str]:
        """브릿지 노드 (전파 중개자) 추출"""
        analyses = self.analyze_nodes()
        return [a.node for a in analyses if a.role == "BRIDGE"]

    def run_full_analysis(self, data: pd.DataFrame) -> PropagationAnalysis:
        """전체 분석 실행"""
        print("[SPG] Running full analysis...")

        # 그래프 구축
        self.build_from_data(data)

        # 노드 분석
        node_analyses = self.analyze_nodes()

        # 엣지 정보
        edges = []
        for u, v, d in self.graph.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'lag': d.get('lag', 0),
                'strength': d.get('strength', 'none'),
                'p_value': d.get('p_value', 1.0),
                'correlation': d.get('correlation', 0),
                'edge_type': d.get('edge_type', 'unknown')
            })

        # Critical Paths
        critical_paths = self.find_all_critical_paths()

        return PropagationAnalysis(
            timestamp=datetime.now().isoformat(),
            nodes=node_analyses,
            edges=edges,
            critical_paths=critical_paths,
            leading_indicators=self.get_leading_indicators(),
            lagging_indicators=self.get_lagging_indicators(),
            bridge_nodes=self.get_bridge_nodes()
        )

    def to_adjacency_matrix(self) -> pd.DataFrame:
        """인접 행렬 반환 (lag 값)"""
        nodes = list(self.graph.nodes)
        n = len(nodes)

        matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=nodes,
            columns=nodes
        )

        for u, v, d in self.graph.edges(data=True):
            matrix.loc[u, v] = d.get('lag', 1)

        return matrix

    def visualize_text(self) -> str:
        """텍스트 기반 시각화"""
        lines = ["=" * 60, "Shock Propagation Graph", "=" * 60, ""]

        # 레이어별 노드
        layers = {layer: [] for layer in NodeLayer}
        for node in self.graph.nodes:
            layer = get_node_layer(node)
            layers[layer].append(node)

        for layer in NodeLayer:
            if layers[layer]:
                lines.append(f"[{layer.name}]")
                lines.append(f"  {', '.join(layers[layer])}")
                lines.append("")

        # 주요 엣지
        lines.append("Key Edges (p < 0.05):")
        for u, v, d in self.graph.edges(data=True):
            if d.get('p_value', 1) < 0.05:
                lag = d.get('lag', 0)
                corr = d.get('correlation', 0)
                lines.append(f"  {u} --[lag={lag}, r={corr:.2f}]--> {v}")

        # Critical Paths
        lines.append("")
        lines.append("Critical Paths:")
        for path in self.find_all_critical_paths(3):
            path_str = " → ".join(path.path)
            lines.append(f"  {path_str} (lag={path.total_lag}d)")

        return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_macro_data(n_days: int = 500) -> pd.DataFrame:
    """테스트용 거시경제 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # 기본 노이즈
    noise = np.random.randn(n_days) * 0.01

    # Fed Funds (기본 드라이버)
    fed_funds = np.cumsum(np.random.randn(n_days) * 0.001) + 4.5

    # DXY (Fed Funds에 2일 후행)
    dxy = pd.Series(fed_funds).shift(2).fillna(method='bfill').values * 20 + 100 + noise * 2

    # VIX (DXY에 3일 후행)
    vix = pd.Series(dxy).shift(3).fillna(method='bfill').values * 0.2 + 15 + np.abs(noise) * 10

    # SPY (VIX에 1일 후행, 역상관)
    spy = 500 - pd.Series(vix).shift(1).fillna(method='bfill').values * 2 + noise * 50

    # TLT (Fed Funds에 5일 후행)
    tlt = 100 - pd.Series(fed_funds).shift(5).fillna(method='bfill').values * 2 + noise * 20

    # GLD (VIX와 양상관, 2일 후행)
    gld = pd.Series(vix).shift(2).fillna(method='bfill').values * 5 + 1800 + noise * 50

    return pd.DataFrame({
        'FED_FUNDS': fed_funds,
        'DXY': dxy,
        'VIX': vix,
        'SPY': spy,
        'TLT': tlt,
        'GLD': gld
    }, index=dates)


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Shock Propagation Graph Test")
    print("=" * 60)

    # 샘플 데이터 생성
    print("\n1. Generating sample macro data (500 days)...")
    data = create_sample_macro_data(500)
    print(f"   Variables: {data.columns.tolist()}")

    # SPG 구축
    print("\n2. Building Shock Propagation Graph...")
    spg = ShockPropagationGraph(
        significance_level=0.05,
        max_lag=10,
        enforce_layer_order=True
    )

    analysis = spg.run_full_analysis(data)

    # 결과 출력
    print("\n3. Results:")
    print(f"   Nodes: {len(analysis.nodes)}")
    print(f"   Edges: {len(analysis.edges)}")
    print(f"   Leading Indicators: {analysis.leading_indicators}")
    print(f"   Lagging Indicators: {analysis.lagging_indicators}")
    print(f"   Bridge Nodes: {analysis.bridge_nodes}")

    print("\n4. Critical Paths:")
    for path in analysis.critical_paths:
        path_str = " → ".join(path.path)
        print(f"   {path_str}")
        print(f"      Total Lag: {path.total_lag} days")
        print(f"      Cumulative Impact: {path.cumulative_impact:.2%}")

    print("\n5. Node Roles:")
    for node in analysis.nodes:
        print(f"   {node.node}: {node.role} (in={node.in_degree}, out={node.out_degree})")

    print("\n" + spg.visualize_text())

    print("\n" + "=" * 60)
    print("Test completed successfully!")
