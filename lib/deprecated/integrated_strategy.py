"""
Integrated Strategy Engine
===========================
Graph-Clustered Portfolio + Shock Propagation을 통합한 투자 전략 엔진

경제학적 철학:
- Whitebox AI: 설명 가능한 인과관계 기반 전략
- Volume > Price: 거래량 급증 = 정보 비대칭 신호
- M = B + S·B*: 확장된 유동성 공식 고려
- Impulse Response: 충격 전파 경로 기반 헤지

핵심 기능:
1. Leading Indicator Tilt: 선행지표에 가중치 부여
2. Shock Early Warning: 상위 레이어 충격 감지 → 하위 레이어 경고
3. Causal Risk Budget: 인과관계 기반 리스크 배분
4. Volume Anomaly Detection: 정보 비대칭 신호 탐지
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# Local imports
from lib.graph_clustered_portfolio import (
    GraphClusteredPortfolio,
    PortfolioAllocation,
    ClusteringMethod,
    RepresentativeMethod
)
from lib.shock_propagation_graph import (
    ShockPropagationGraph,
    PropagationAnalysis,
    NodeLayer,
    get_node_layer,
    ShockPath
)


class SignalType(Enum):
    """시그널 유형"""
    LEADING_TILT = "leading_tilt"       # 선행지표 기반 틸팅
    SHOCK_WARNING = "shock_warning"     # 충격 전파 경고
    VOLUME_SPIKE = "volume_spike"       # 거래량 급증
    REGIME_SHIFT = "regime_shift"       # 레짐 변화
    REBALANCE = "rebalance"             # 리밸런싱 필요


class ActionType(Enum):
    """행동 유형"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    HEDGE = "hedge"
    REDUCE = "reduce"
    INCREASE = "increase"


@dataclass
class Signal:
    """투자 시그널"""
    timestamp: str
    signal_type: SignalType
    source: str                    # 시그널 발생 소스
    affected_assets: List[str]     # 영향 받는 자산
    confidence: float              # 신뢰도 (0-1)
    urgency: str                   # "HIGH", "MEDIUM", "LOW"
    description: str
    action_suggested: ActionType
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyRecommendation:
    """전략 권고"""
    timestamp: str
    portfolio_weights: Dict[str, float]
    tilted_weights: Dict[str, float]     # 틸팅 적용 후 가중치
    tilt_factors: Dict[str, float]       # 자산별 틸팅 팩터
    signals: List[Signal]
    risk_metrics: Dict[str, float]

    # 경제학적 해석
    leading_exposure: float              # 선행지표 노출도
    lagging_exposure: float              # 후행지표 노출도
    shock_vulnerability: float           # 충격 취약도

    # 실행 가이드
    actions: List[Dict]
    warnings: List[str]

    def to_dict(self) -> Dict:
        result = asdict(self)
        result['signals'] = [asdict(s) for s in self.signals]
        return result


@dataclass
class VolumeAnomaly:
    """거래량 이상치"""
    asset: str
    timestamp: str
    volume: float
    volume_ma20: float
    surge_ratio: float             # volume / MA20
    interpretation: str            # "NEW_INFORMATION", "EXHAUSTION", "MANIPULATION"
    confidence: float


# ============================================================================
# Integrated Strategy Engine
# ============================================================================

class IntegratedStrategy:
    """
    통합 투자 전략 엔진

    Foundation (GC-HRP) + Intelligence (SPG) = Application
    """

    def __init__(
        self,
        # Portfolio params
        correlation_threshold: float = 0.3,
        clustering_method: ClusteringMethod = ClusteringMethod.KMEANS,

        # Causality params
        significance_level: float = 0.05,
        max_lag: int = 20,

        # Strategy params
        leading_tilt_factor: float = 0.15,      # 선행지표 틸팅 강도
        volume_surge_threshold: float = 3.0,    # 거래량 급증 임계값 (MA20 대비)
        shock_window: int = 5                   # 충격 감지 윈도우 (일)
    ):
        # Portfolio engine
        self.portfolio_engine = GraphClusteredPortfolio(
            correlation_threshold=correlation_threshold,
            clustering_method=clustering_method,
            representative_method=RepresentativeMethod.CENTRALITY
        )

        # Causality engine
        self.causality_engine = ShockPropagationGraph(
            significance_level=significance_level,
            max_lag=max_lag,
            enforce_layer_order=True
        )

        # Strategy params
        self.leading_tilt_factor = leading_tilt_factor
        self.volume_surge_threshold = volume_surge_threshold
        self.shock_window = shock_window

        # Results
        self.portfolio_allocation: Optional[PortfolioAllocation] = None
        self.causality_analysis: Optional[PropagationAnalysis] = None
        self.signals: List[Signal] = []

    def fit(
        self,
        returns: pd.DataFrame,
        macro_data: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None
    ) -> StrategyRecommendation:
        """
        전략 수립

        Args:
            returns: 자산 수익률
            macro_data: 거시지표 데이터 (Fed Funds, VIX, etc.)
            volumes: 거래량 데이터

        Returns:
            StrategyRecommendation
        """
        print("[Strategy] Starting integrated strategy...")
        self.signals = []

        # Step 1: 포트폴리오 구축 (Foundation)
        print("[Strategy] Step 1: Building portfolio allocation...")
        self.portfolio_allocation = self.portfolio_engine.fit(returns, volumes)
        base_weights = self.portfolio_allocation.weights

        # Step 2: 인과관계 분석 (Intelligence)
        print("[Strategy] Step 2: Analyzing causal relationships...")
        self.causality_analysis = self.causality_engine.run_full_analysis(macro_data)

        # Step 3: 선행지표 틸팅
        print("[Strategy] Step 3: Calculating leading indicator tilt...")
        tilt_factors = self._calculate_leading_tilt(returns.columns.tolist())
        tilted_weights = self._apply_tilt(base_weights, tilt_factors)

        # Step 4: 충격 경고 생성
        print("[Strategy] Step 4: Generating shock warnings...")
        shock_signals = self._detect_shock_warnings(macro_data)
        self.signals.extend(shock_signals)

        # Step 5: 거래량 이상치 탐지
        print("[Strategy] Step 5: Detecting volume anomalies...")
        if volumes is not None:
            volume_signals = self._detect_volume_anomalies(volumes, returns)
            self.signals.extend(volume_signals)

        # Step 6: 리스크 메트릭 계산
        risk_metrics = self._calculate_risk_metrics(returns, tilted_weights)

        # Step 7: 실행 액션 생성
        actions = self._generate_actions(base_weights, tilted_weights, self.signals)
        warnings = self._generate_warnings(self.signals)

        # 노출도 계산
        leading_exp = self._calculate_leading_exposure(tilted_weights)
        lagging_exp = self._calculate_lagging_exposure(tilted_weights)
        shock_vuln = self._calculate_shock_vulnerability(tilted_weights)

        return StrategyRecommendation(
            timestamp=datetime.now().isoformat(),
            portfolio_weights=base_weights,
            tilted_weights=tilted_weights,
            tilt_factors=tilt_factors,
            signals=self.signals,
            risk_metrics=risk_metrics,
            leading_exposure=leading_exp,
            lagging_exposure=lagging_exp,
            shock_vulnerability=shock_vuln,
            actions=actions,
            warnings=warnings
        )

    def _calculate_leading_tilt(self, assets: List[str]) -> Dict[str, float]:
        """
        선행지표 기반 틸팅 팩터 계산

        경제학적 근거:
        - Out-degree 높은 자산: 시장을 리드 → 오버웨이트
        - In-degree 높은 자산: 후행 → 리밸런싱 신호로 활용
        """
        tilt_factors = {}

        if self.causality_analysis is None:
            return {a: 1.0 for a in assets}

        # 노드 분석 결과에서 선행 점수 추출
        node_scores = {}
        for node in self.causality_analysis.nodes:
            node_scores[node.node] = {
                'leading_score': node.leading_score,
                'role': node.role,
                'layer': node.layer.value
            }

        for asset in assets:
            if asset in node_scores:
                score = node_scores[asset]

                # 선행 지표: 틸팅 증가
                if score['role'] == 'LEADING':
                    tilt_factors[asset] = 1.0 + self.leading_tilt_factor

                    self.signals.append(Signal(
                        timestamp=datetime.now().isoformat(),
                        signal_type=SignalType.LEADING_TILT,
                        source=asset,
                        affected_assets=[asset],
                        confidence=0.7,
                        urgency="LOW",
                        description=f"{asset}는 선행지표로 식별됨. 비중 확대 권고.",
                        action_suggested=ActionType.INCREASE
                    ))

                # 후행 지표: 틸팅 감소
                elif score['role'] == 'LAGGING':
                    tilt_factors[asset] = 1.0 - self.leading_tilt_factor * 0.5

                # 브릿지: 헤지 목적 유지
                elif score['role'] == 'BRIDGE':
                    tilt_factors[asset] = 1.0  # 유지

                else:
                    tilt_factors[asset] = 1.0
            else:
                tilt_factors[asset] = 1.0

        return tilt_factors

    def _apply_tilt(
        self,
        weights: Dict[str, float],
        tilt_factors: Dict[str, float]
    ) -> Dict[str, float]:
        """틸팅 적용 및 정규화"""
        tilted = {}

        for asset, weight in weights.items():
            tilt = tilt_factors.get(asset, 1.0)
            tilted[asset] = weight * tilt

        # 정규화
        total = sum(tilted.values())
        if total > 0:
            tilted = {k: v / total for k, v in tilted.items()}

        return tilted

    def _detect_shock_warnings(self, macro_data: pd.DataFrame) -> List[Signal]:
        """
        충격 전파 경고 생성

        상위 레이어(POLICY, LIQUIDITY)에서 급격한 변화 감지 시
        하위 레이어(ASSET_PRICE) 경고 발생
        """
        signals = []

        if self.causality_analysis is None:
            return signals

        # 최근 N일 변화율 계산
        recent = macro_data.tail(self.shock_window)

        for col in macro_data.columns:
            layer = get_node_layer(col)

            # 상위 레이어만 모니터링
            if layer not in [NodeLayer.POLICY, NodeLayer.LIQUIDITY, NodeLayer.RISK_PREMIUM]:
                continue

            # 변화율 계산
            if len(recent[col].dropna()) < 2:
                continue

            first_val = recent[col].dropna().iloc[0]
            last_val = recent[col].dropna().iloc[-1]

            if first_val == 0:
                continue

            change_pct = (last_val - first_val) / abs(first_val) * 100

            # 급격한 변화 감지 (2% 이상)
            if abs(change_pct) > 2:
                # Critical Path에서 영향받는 자산 탐색
                affected = self._find_affected_assets(col)

                urgency = "HIGH" if abs(change_pct) > 5 else "MEDIUM"

                direction = "상승" if change_pct > 0 else "하락"

                signals.append(Signal(
                    timestamp=datetime.now().isoformat(),
                    signal_type=SignalType.SHOCK_WARNING,
                    source=col,
                    affected_assets=affected,
                    confidence=min(0.9, abs(change_pct) / 10),
                    urgency=urgency,
                    description=f"{col} {self.shock_window}일간 {change_pct:.1f}% {direction}. "
                               f"영향 예상 자산: {', '.join(affected[:3])}",
                    action_suggested=ActionType.HEDGE if change_pct > 0 else ActionType.HOLD,
                    metadata={'change_pct': change_pct, 'layer': layer.name}
                ))

        return signals

    def _find_affected_assets(self, source: str) -> List[str]:
        """충격 소스에서 영향받는 자산 탐색"""
        affected = []

        path = self.causality_engine.find_critical_path(source)
        if path:
            # Critical Path의 마지막 노드들 (자산)
            for node in path.path:
                if get_node_layer(node) == NodeLayer.ASSET_PRICE:
                    affected.append(node)

        # Direct successors 추가
        if source in self.causality_engine.graph:
            for successor in self.causality_engine.graph.successors(source):
                if successor not in affected:
                    affected.append(successor)

        return affected

    def _detect_volume_anomalies(
        self,
        volumes: pd.DataFrame,
        returns: pd.DataFrame
    ) -> List[Signal]:
        """
        거래량 이상치 탐지

        경제학적 의미:
        - 거래량 급증 = 참여자 간 기대 불일치 또는 새로운 정보 유입
        - 가격 상승 + 거래량 급증 = 강한 매수 신호
        - 가격 하락 + 거래량 급증 = 패닉 또는 매도 신호
        """
        signals = []

        for asset in volumes.columns:
            if asset not in returns.columns:
                continue

            vol_series = volumes[asset].dropna()
            if len(vol_series) < 20:
                continue

            # 20일 이동평균
            vol_ma20 = vol_series.rolling(20).mean()

            # 최근 거래량
            recent_vol = vol_series.iloc[-1]
            recent_ma = vol_ma20.iloc[-1]

            if recent_ma == 0:
                continue

            surge_ratio = recent_vol / recent_ma

            # 거래량 급증 탐지
            if surge_ratio >= self.volume_surge_threshold:
                # 가격 방향 확인
                recent_return = returns[asset].iloc[-1] if len(returns[asset]) > 0 else 0

                if recent_return > 0.01:  # 1% 이상 상승
                    interpretation = "강한 매수세 유입 (NEW_INFORMATION)"
                    action = ActionType.HOLD  # 추세 추종
                elif recent_return < -0.01:  # 1% 이상 하락
                    interpretation = "패닉 매도 또는 고점 신호 (EXHAUSTION)"
                    action = ActionType.REDUCE
                else:
                    interpretation = "방향성 불명확 (ACCUMULATION)"
                    action = ActionType.HOLD

                signals.append(Signal(
                    timestamp=datetime.now().isoformat(),
                    signal_type=SignalType.VOLUME_SPIKE,
                    source=asset,
                    affected_assets=[asset],
                    confidence=min(0.9, surge_ratio / 10),
                    urgency="MEDIUM" if surge_ratio < 5 else "HIGH",
                    description=f"{asset} 거래량 급증 (MA20 대비 {surge_ratio:.1f}배). {interpretation}",
                    action_suggested=action,
                    metadata={
                        'surge_ratio': surge_ratio,
                        'recent_return': recent_return,
                        'interpretation': interpretation
                    }
                ))

        return signals

    def _calculate_risk_metrics(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """리스크 메트릭 계산"""
        assets = [a for a in weights.keys() if a in returns.columns]
        w = np.array([weights[a] for a in assets])

        # 공분산
        cov = returns[assets].cov().values

        # 포트폴리오 변동성
        port_var = np.dot(w, np.dot(cov, w))
        port_vol = np.sqrt(port_var) * np.sqrt(252)

        # VaR (95%)
        port_returns = returns[assets].dot(pd.Series(weights)[assets])
        var_95 = np.percentile(port_returns.dropna(), 5) * np.sqrt(252)

        # CVaR (Expected Shortfall)
        es_95 = port_returns[port_returns <= np.percentile(port_returns, 5)].mean() * np.sqrt(252)

        # 최대 드로우다운
        cumulative = (1 + port_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min()

        return {
            'volatility': port_vol,
            'var_95': var_95,
            'cvar_95': es_95 if not np.isnan(es_95) else var_95,
            'max_drawdown': max_dd,
            'sharpe_estimate': port_returns.mean() * 252 / (port_vol + 1e-10)
        }

    def _calculate_leading_exposure(self, weights: Dict[str, float]) -> float:
        """선행지표 노출도 계산"""
        if self.causality_analysis is None:
            return 0.0

        leading_weight = 0.0
        for node in self.causality_analysis.nodes:
            if node.role == 'LEADING' and node.node in weights:
                leading_weight += weights[node.node]

        return leading_weight

    def _calculate_lagging_exposure(self, weights: Dict[str, float]) -> float:
        """후행지표 노출도 계산"""
        if self.causality_analysis is None:
            return 0.0

        lagging_weight = 0.0
        for node in self.causality_analysis.nodes:
            if node.role == 'LAGGING' and node.node in weights:
                lagging_weight += weights[node.node]

        return lagging_weight

    def _calculate_shock_vulnerability(self, weights: Dict[str, float]) -> float:
        """
        충격 취약도 계산

        Critical Path 상에서 영향받는 자산의 가중치 합
        """
        if self.causality_analysis is None:
            return 0.0

        vulnerable_assets = set()

        for path in self.causality_analysis.critical_paths:
            # 경로 끝에 있는 자산들 (최종 영향받는 자산)
            for node in path.path[-3:]:  # 마지막 3개
                if get_node_layer(node) == NodeLayer.ASSET_PRICE:
                    vulnerable_assets.add(node)

        vulnerability = sum(weights.get(a, 0) for a in vulnerable_assets)
        return vulnerability

    def _generate_actions(
        self,
        base_weights: Dict[str, float],
        tilted_weights: Dict[str, float],
        signals: List[Signal]
    ) -> List[Dict]:
        """실행 액션 생성"""
        actions = []

        # 1. 틸팅으로 인한 리밸런싱
        for asset in tilted_weights:
            base = base_weights.get(asset, 0)
            tilted = tilted_weights.get(asset, 0)
            diff = tilted - base

            if abs(diff) > 0.01:  # 1% 이상 변화
                action_type = "INCREASE" if diff > 0 else "DECREASE"
                actions.append({
                    'asset': asset,
                    'action': action_type,
                    'from_weight': f"{base:.2%}",
                    'to_weight': f"{tilted:.2%}",
                    'change': f"{diff:+.2%}",
                    'reason': 'Leading indicator tilt'
                })

        # 2. 시그널 기반 액션
        for signal in signals:
            if signal.urgency in ["HIGH", "MEDIUM"]:
                for asset in signal.affected_assets[:3]:
                    actions.append({
                        'asset': asset,
                        'action': signal.action_suggested.value.upper(),
                        'reason': signal.description[:100],
                        'urgency': signal.urgency,
                        'confidence': f"{signal.confidence:.0%}"
                    })

        return actions

    def _generate_warnings(self, signals: List[Signal]) -> List[str]:
        """경고 메시지 생성"""
        warnings = []

        high_urgency = [s for s in signals if s.urgency == "HIGH"]

        for signal in high_urgency:
            warnings.append(f"⚠️ [{signal.signal_type.value}] {signal.description}")

        # 충격 취약도 경고
        if self.causality_analysis:
            critical_paths = self.causality_analysis.critical_paths
            if critical_paths:
                longest = max(critical_paths, key=lambda p: len(p.path))
                if len(longest.path) > 4:
                    warnings.append(
                        f"🔗 긴 충격 전파 경로 감지: {' → '.join(longest.path[:5])}... "
                        f"(총 {longest.total_lag}일 소요)"
                    )

        return warnings

    def get_summary(self) -> str:
        """전략 요약 텍스트"""
        if self.portfolio_allocation is None:
            return "Strategy not yet fitted."

        lines = [
            "=" * 60,
            "INTEGRATED STRATEGY SUMMARY",
            "=" * 60,
            "",
            f"Portfolio: {self.portfolio_allocation.methodology}",
            f"  - Clusters: {len(self.portfolio_allocation.clusters)}",
            f"  - Diversification Ratio: {self.portfolio_allocation.diversification_ratio:.2f}",
            f"  - Effective N: {self.portfolio_allocation.effective_n:.1f}",
            ""
        ]

        if self.causality_analysis:
            lines.extend([
                "Causality Analysis:",
                f"  - Leading Indicators: {', '.join(self.causality_analysis.leading_indicators[:3]) or 'None'}",
                f"  - Bridge Nodes: {', '.join(self.causality_analysis.bridge_nodes[:3]) or 'None'}",
                ""
            ])

            if self.causality_analysis.critical_paths:
                path = self.causality_analysis.critical_paths[0]
                lines.append(f"  - Critical Path: {' → '.join(path.path)}")
                lines.append(f"    (Total lag: {path.total_lag} days)")
                lines.append("")

        lines.append(f"Signals Generated: {len(self.signals)}")
        high_signals = [s for s in self.signals if s.urgency == "HIGH"]
        if high_signals:
            lines.append(f"  ⚠️ HIGH urgency signals: {len(high_signals)}")

        return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def create_integrated_sample_data(
    n_assets: int = 50,
    n_days: int = 500
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """통합 테스트용 샘플 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    # 1. 거시 데이터 (인과관계 포함)
    fed_funds = np.cumsum(np.random.randn(n_days) * 0.001) + 4.5
    dxy = pd.Series(fed_funds).shift(2).fillna(method='bfill').values * 20 + 100
    vix = pd.Series(dxy).shift(3).fillna(method='bfill').values * 0.2 + 15

    macro_data = pd.DataFrame({
        'FED_FUNDS': fed_funds,
        'DXY': dxy + np.random.randn(n_days) * 0.5,
        'VIX': vix + np.abs(np.random.randn(n_days)) * 2
    }, index=dates)

    # 2. 자산 수익률 (팩터 기반)
    n_factors = 3
    factor_returns = np.random.randn(n_days, n_factors) * 0.01
    loadings = np.random.randn(n_assets, n_factors)
    idiosyncratic = np.random.randn(n_days, n_assets) * 0.02

    asset_returns = np.dot(factor_returns, loadings.T) + idiosyncratic

    assets = [f'ASSET_{i:02d}' for i in range(n_assets)]
    returns = pd.DataFrame(asset_returns, index=dates, columns=assets)

    # 3. 거래량 (일부 급증 포함)
    volumes = np.exp(np.random.randn(n_days, n_assets) + 10)

    # 특정 시점 거래량 급증
    spike_idx = np.random.choice(range(50, n_days), size=10, replace=False)
    spike_assets = np.random.choice(range(n_assets), size=10, replace=False)
    for idx, asset_idx in zip(spike_idx, spike_assets):
        volumes[idx, asset_idx] *= 5  # 5배 급증

    volumes_df = pd.DataFrame(volumes, index=dates, columns=assets)

    return returns, macro_data, volumes_df


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Integrated Strategy Test")
    print("=" * 60)

    # 샘플 데이터 생성
    print("\n1. Generating sample data...")
    returns, macro_data, volumes = create_integrated_sample_data(
        n_assets=50, n_days=500
    )
    print(f"   Returns: {returns.shape}")
    print(f"   Macro: {macro_data.shape}")
    print(f"   Volumes: {volumes.shape}")

    # 통합 전략 실행
    print("\n2. Running integrated strategy...")
    strategy = IntegratedStrategy(
        correlation_threshold=0.3,
        clustering_method=ClusteringMethod.KMEANS,
        leading_tilt_factor=0.15,
        volume_surge_threshold=3.0
    )

    recommendation = strategy.fit(returns, macro_data, volumes)

    # 결과 출력
    print("\n3. Strategy Summary:")
    print(strategy.get_summary())

    print("\n4. Risk Metrics:")
    for metric, value in recommendation.risk_metrics.items():
        print(f"   {metric}: {value:.4f}")

    print("\n5. Exposure Analysis:")
    print(f"   Leading Exposure: {recommendation.leading_exposure:.2%}")
    print(f"   Lagging Exposure: {recommendation.lagging_exposure:.2%}")
    print(f"   Shock Vulnerability: {recommendation.shock_vulnerability:.2%}")

    print("\n6. Signals Generated:")
    for signal in recommendation.signals[:5]:
        print(f"   [{signal.urgency}] {signal.signal_type.value}: {signal.description[:60]}...")

    print("\n7. Top Actions:")
    for action in recommendation.actions[:5]:
        print(f"   {action.get('action', 'N/A')}: {action.get('asset', 'N/A')} - {action.get('reason', '')[:50]}")

    print("\n8. Warnings:")
    for warning in recommendation.warnings:
        print(f"   {warning}")

    print("\n9. Top 10 Tilted Weights:")
    sorted_weights = sorted(
        recommendation.tilted_weights.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for asset, weight in sorted_weights[:10]:
        base = recommendation.portfolio_weights.get(asset, 0)
        tilt = recommendation.tilt_factors.get(asset, 1.0)
        print(f"   {asset}: {weight:.2%} (base: {base:.2%}, tilt: {tilt:.2f})")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
