from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

@dataclass
class MarketQualityMetrics:
    """시장 미세구조 품질 메트릭"""
    avg_liquidity_score: float = 50.0
    liquidity_scores: Dict[str, float] = field(default_factory=dict)
    high_toxicity_tickers: List[str] = field(default_factory=list)
    illiquid_tickers: List[str] = field(default_factory=list)
    data_quality: str = "COMPLETE"

    def to_dict(self) -> Dict:
        return {
            'avg_liquidity_score': round(self.avg_liquidity_score, 2),
            'liquidity_scores': self.liquidity_scores,
            'high_toxicity_tickers': self.high_toxicity_tickers,
            'illiquid_tickers': self.illiquid_tickers,
            'data_quality': self.data_quality
        }

@dataclass
class BubbleRiskMetrics:
    """버블 리스크 메트릭"""
    overall_status: str = "NONE"
    risk_tickers: List[Dict] = field(default_factory=list)
    highest_risk_ticker: str = ""
    highest_risk_score: float = 0.0
    methodology_notes: str = "Bubbles for Fama (2019)"

    def to_dict(self) -> Dict:
        return {
            'overall_status': self.overall_status,
            'risk_tickers': self.risk_tickers,
            'highest_risk_ticker': self.highest_risk_ticker,
            'highest_risk_score': self.highest_risk_score,
            'methodology_notes': self.methodology_notes
        }

@dataclass
class EIMASResult:
    """통합 실행 결과"""
    timestamp: str

    # 데이터 수집
    fred_summary: Dict = field(default_factory=dict)
    market_data_count: int = 0
    crypto_data_count: int = 0

    # 분석 결과
    regime: Dict = field(default_factory=dict)
    events_detected: List[Dict] = field(default_factory=list)
    liquidity_signal: str = "NEUTRAL"
    risk_score: float = 0.0

    # 에이전트 토론
    debate_consensus: Dict = field(default_factory=dict)
    dissent_records: List[Dict] = field(default_factory=list)
    has_strong_dissent: bool = False

    # Dual Mode
    full_mode_position: str = "NEUTRAL"
    reference_mode_position: str = "NEUTRAL"
    modes_agree: bool = True

    # 최종 권고
    final_recommendation: str = "HOLD"
    confidence: float = 0.5
    risk_level: str = "MEDIUM"
    warnings: List[str] = field(default_factory=list)

    # 실시간 (선택)
    realtime_signals: List[Dict] = field(default_factory=list)

    # Advanced Strategy
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    shock_propagation: Dict = field(default_factory=dict)
    integrated_signals: List[Dict] = field(default_factory=list)
    genius_act_regime: str = "NEUTRAL"
    genius_act_signals: List[Dict] = field(default_factory=list)
    whitening_summary: str = ""
    fact_check_grade: str = "N/A"
    theme_etf_analysis: Dict = field(default_factory=dict)

    # Volume Anomaly
    volume_anomalies: List[Dict] = field(default_factory=list)
    volume_analysis_summary: str = ""

    # Market Quality & Bubble Risk
    market_quality: Optional[MarketQualityMetrics] = None
    bubble_risk: Optional[BubbleRiskMetrics] = None

    # Risk Score Transparency
    base_risk_score: float = 0.0
    microstructure_adjustment: float = 0.0
    bubble_risk_adjustment: float = 0.0

    # Crypto Stress Test
    crypto_stress_test: Dict = field(default_factory=dict)

    # Devil's Advocate
    devils_advocate_arguments: List[str] = field(default_factory=list)

    # HRP Allocation Rationale
    hrp_allocation_rationale: str = ""

    # Extended Data Sources
    defi_tvl: Dict = field(default_factory=dict)
    mena_markets: Dict = field(default_factory=dict)
    onchain_risk_signals: List[Dict] = field(default_factory=list)

    # Event Tracking
    event_tracking: Dict = field(default_factory=dict)
    tracked_events: List[Dict] = field(default_factory=list)

    # Adaptive Portfolio
    adaptive_portfolios: Dict = field(default_factory=dict)
    validation_loop_result: Dict = field(default_factory=dict)

    # Correlation Analysis
    correlation_matrix: List[List[float]] = field(default_factory=list)
    correlation_tickers: List[str] = field(default_factory=list)

    # Extended Standalone Scripts
    intraday_summary: Dict = field(default_factory=dict)
    crypto_monitoring: Dict = field(default_factory=dict)
    event_predictions: List[Dict] = field(default_factory=list)
    event_attributions: List[Dict] = field(default_factory=list)
    event_backtest_results: Dict = field(default_factory=dict)
    news_correlations: List[Dict] = field(default_factory=list)

    # Additional Integrated Modules
    ark_analysis: Dict = field(default_factory=dict)
    critical_path_monitoring: Dict = field(default_factory=dict)
    trading_db_status: str = "N/A"

    # Missing Phase 2 Analyses
    liquidity_analysis: Dict = field(default_factory=dict)
    etf_flow_result: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def _generate_potential_concerns(self) -> List[str]:
        concerns = []
        regime = self.regime.get('regime', 'Unknown')
        if regime == 'BULL':
            concerns.append("현재 Bull 레짐이나, 과열 신호 전환 가능성 모니터링 필요")
        elif regime == 'BEAR':
            concerns.append("Bear 레짐에서 추가 하락 리스크 존재. 방어적 포지션 유지 권고")
        
        if self.risk_score > 60:
            concerns.append(f"리스크 점수 {self.risk_score:.1f}로 상승. 포지션 축소 고려")
            
        if not concerns:
            concerns = ["현재 분석 기준 주요 리스크 미탐지, 블랙스완 주의"]
        return concerns[:3]

    def to_markdown(self) -> str:
        md = []
        md.append("# EIMAS Analysis Report")
        md.append(f"**Generated**: {self.timestamp}")
        md.append("")

        # 1. Data Summary
        md.append("## 1. Data Summary")
        if self.fred_summary:
            md.append(f"- **RRP**: ${self.fred_summary.get('rrp', 0):.0f}B")
            md.append(f"- **Net Liquidity**: ${self.fred_summary.get('net_liquidity', 0):.0f}B")
        md.append(f"- **Tickers**: {self.market_data_count} (Crypto: {self.crypto_data_count})")
        md.append("")

        # 2. Regime
        md.append("## 2. Regime Analysis")
        if self.regime:
            md.append(f"- **Regime**: {self.regime.get('regime', 'Unknown')}")
            md.append(f"- **Volatility**: {self.regime.get('volatility', 'Unknown')}")
            if self.regime.get('gmm_regime'):
                md.append(f"- **GMM**: {self.regime.get('gmm_regime')} (Entropy: {self.regime.get('entropy', 0):.3f})")
        md.append("")

        # 3. Risk
        md.append("## 3. Risk Assessment")
        md.append(f"- **Risk Score**: {self.risk_score:.1f}/100")
        
        if self.base_risk_score > 0:
            md.append("### Breakdown")
            md.append(f"- Base: {self.base_risk_score:.1f}")
            md.append(f"- Microstructure Adj: {self.microstructure_adjustment:+.1f}")
            md.append(f"- Bubble Adj: +{self.bubble_risk_adjustment:.0f}")

        if self.bubble_risk:
            md.append(f"- **Bubble Status**: {self.bubble_risk.overall_status}")
            if self.bubble_risk.risk_tickers:
                md.append("  - Risk Tickers: " + ", ".join([t['ticker'] for t in self.bubble_risk.risk_tickers]))
        md.append("")

        # 4. Events
        md.append("## 4. Events Detected")
        if self.events_detected:
            for e in self.events_detected:
                md.append(f"- **{e.get('type')}**: {e.get('description')}")
        else:
            md.append("- None")
        md.append("")

        # 5. Debate
        md.append("## 5. Multi-Agent Debate")
        md.append(f"- **Full Mode**: {self.full_mode_position}")
        md.append(f"- **Ref Mode**: {self.reference_mode_position}")
        md.append(f"- **Agreement**: {'Yes' if self.modes_agree else 'No'}")
        
        if self.devils_advocate_arguments:
            md.append("### Devil's Advocate")
            for arg in self.devils_advocate_arguments:
                md.append(f"- {arg}")
        md.append("")

        # 6. Advanced
        md.append("## 6. Advanced Analysis")
        if self.genius_act_signals:
            md.append(f"### Genius Act: {len(self.genius_act_signals)} signals")
        
        if self.portfolio_weights:
            md.append("### GC-HRP Portfolio")
            sorted_w = sorted(self.portfolio_weights.items(), key=lambda x: x[1], reverse=True)[:10]
            for t, w in sorted_w:
                md.append(f"- {t}: {w:.1%}")
            if self.hrp_allocation_rationale:
                md.append(f"  - Rationale: {self.hrp_allocation_rationale}")
        
        if self.theme_etf_analysis:
            md.append(f"### Theme ETF: {self.theme_etf_analysis.get('theme')}")
            
        if self.liquidity_analysis:
            md.append("### Liquidity Causality")
            if self.liquidity_analysis.get('rrp_to_spy_significant'):
                md.append("- RRP -> SPY Significant")
        
        if self.etf_flow_result:
            md.append(f"### Sector Rotation: {self.etf_flow_result.get('rotation_signal', 'N/A')}")

        if self.volume_anomalies:
            md.append(f"### Volume Anomalies: {len(self.volume_anomalies)} detected")

        md.append("")

        # 7. Recommendation
        md.append("## 7. Final Recommendation")
        md.append(f"- **Action**: {self.final_recommendation}")
        md.append(f"- **Confidence**: {self.confidence:.0%}")
        md.append(f"- **Risk Level**: {self.risk_level}")
        md.append("")

        # 8. Standalone
        if self.intraday_summary or self.crypto_monitoring:
            md.append("## 8. Standalone Scripts")
            if self.intraday_summary:
                md.append(f"- Intraday: {self.intraday_summary.get('status')}")
            if self.crypto_monitoring:
                md.append(f"- Crypto Risk: {self.crypto_monitoring.get('risk_level')}")
            if self.event_predictions:
                md.append(f"- Events Predicted: {len(self.event_predictions)}")
            if self.news_correlations:
                md.append(f"- News Correlations: {len(self.news_correlations)}")

        return "\n".join(md)