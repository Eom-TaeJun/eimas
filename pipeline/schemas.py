from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json

# Import new schemas from core
try:
    from core.schemas import AgentOutputs, DebateResults, VerificationResults
except ImportError:
    # Fallback if core path not set
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.schemas import AgentOutputs, DebateResults, VerificationResults

@dataclass
class FREDSummary:
    """FRED 데이터 요약"""
    timestamp: str
    fed_funds: float = 0.0
    treasury_2y: float = 0.0
    treasury_10y: float = 0.0
    treasury_30y: float = 0.0
    spread_10y2y: float = 0.0
    spread_10y3m: float = 0.0
    hy_oas: float = 0.0
    cpi_yoy: float = 0.0
    core_pce_yoy: float = 0.0
    breakeven_5y: float = 0.0
    breakeven_10y: float = 0.0
    unemployment: float = 0.0
    initial_claims: int = 0
    rrp: float = 0.0
    rrp_delta: float = 0.0
    rrp_delta_pct: float = 0.0
    tga: float = 0.0
    tga_delta: float = 0.0
    fed_assets: float = 0.0
    fed_assets_delta: float = 0.0
    net_liquidity: float = 0.0
    liquidity_regime: str = "Normal"
    curve_inverted: bool = False
    curve_status: str = "Normal"
    signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class IndicatorsSummary:
    """시장 지표 요약"""
    timestamp: str
    vix_current: float = 0.0
    fear_greed_level: str = "Neutral"
    risk_score: float = 50.0
    opportunity_score: float = 50.0
    signals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

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
class RegimeResult:
    timestamp: str
    regime: str
    trend: str
    volatility: str
    confidence: float
    description: str
    strategy: str

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class Event:
    type: str
    importance: str
    description: str
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class LiquiditySignal:
    signal: str
    causality_results: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class CriticalPathResult:
    risk_score: float
    risk_level: str
    primary_risk_path: str
    details: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ETFFlowResult:
    rotation_signal: str
    style_signal: str
    details: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DebateResult:
    full_mode_position: str
    reference_mode_position: str
    modes_agree: bool
    final_recommendation: str
    confidence: float
    risk_level: str
    dissent_records: List[Dict]
    warnings: List[str]
    # Phase 2-3 Enhanced Fields (NEW)
    enhanced_debate: Dict = field(default_factory=dict)  # interpretation + methodology
    reasoning_chain: List[Dict] = field(default_factory=list)  # 추론 과정 추적
    agent_outputs: Dict = field(default_factory=dict)  # 에이전트별 출력
    verification: Dict = field(default_factory=dict)  # 검증 결과
    metadata: Dict = field(default_factory=dict)  # 통계 정보

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class RealtimeSignal:
    timestamp: str
    symbol: str
    ofi: float
    vpin: float
    signal: str
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class AIReport:
    timestamp: str
    report_path: str
    ib_report_path: str
    highlights: Dict[str, Any]
    content: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class GeniusActResult:
    regime: str
    signals: List[Dict]
    digital_m2: float
    details: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ThemeETFResult:
    theme: str
    score: float
    constituents: List[str]
    details: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class ShockAnalysisResult:
    impact_score: float
    contagion_path: List[str]
    vulnerable_assets: List[str]
    details: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class PortfolioResult:
    weights: Dict[str, float]
    risk_contribution: Dict[str, float]
    diversification_ratio: float
    mst_hubs: List[str]
    details: Dict

    def to_dict(self) -> Dict:
        return asdict(self)

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

    # NEW: Enhanced Analysis (2026-01-25)
    hft_microstructure: Dict = field(default_factory=dict)
    garch_volatility: Dict = field(default_factory=dict)
    information_flow: Dict = field(default_factory=dict)
    proof_of_index: Dict = field(default_factory=dict)
    dtw_similarity: Dict = field(default_factory=dict)
    dbscan_outliers: Dict = field(default_factory=dict)
    
    # Extended Data Sources (PCR, Valuation, Crypto)
    extended_data: Dict = field(default_factory=dict)
    
    # NEW: Sentiment Analysis (2026-01-29)
    sentiment_analysis: Dict = field(default_factory=dict)

    # Phase 3.Enhanced: New Agent Outputs
    agent_outputs: Optional[AgentOutputs] = None
    debate_results: Optional[DebateResults] = None
    verification: Optional[VerificationResults] = None
    reasoning_chain: List[Dict] = field(default_factory=list)
    
    # NEW: AI Report 통합 (2026-01-29)
    ai_report: Optional[Dict] = None  # FinalReport.to_dict()

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

        # Helper to get value from dict or object
        def get_val(obj, key, default=None):
            if obj is None: return default
            if isinstance(obj, dict): return obj.get(key, default)
            return getattr(obj, key, default)

        # 1. Data Summary
        md.append("## 1. Data Summary")
        if self.fred_summary:
            md.append(f"- **RRP**: ${get_val(self.fred_summary, 'rrp', 0):.0f}B")
            md.append(f"- **Net Liquidity**: ${get_val(self.fred_summary, 'net_liquidity', 0):.0f}B")
        md.append(f"- **Tickers**: {self.market_data_count} (Crypto: {self.crypto_data_count})")
        md.append("")

        # 2. Regime
        md.append("## 2. Regime Analysis")
        if self.regime:
            md.append(f"- **Regime**: {get_val(self.regime, 'regime', 'Unknown')}")
            md.append(f"- **Volatility**: {get_val(self.regime, 'volatility', 'Unknown')}")
            if get_val(self.regime, 'gmm_regime'):
                md.append(f"- **GMM**: {get_val(self.regime, 'gmm_regime')} (Entropy: {get_val(self.regime, 'entropy', 0):.3f})")
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

        # 5.1 Enhanced Debate (Multi-LLM) - from debate_consensus['enhanced'] or debate_results
        enhanced = self.debate_consensus.get('enhanced', {}) if self.debate_consensus else {}
        if enhanced or self.debate_results:
            md.append("### Enhanced Debate (Multi-LLM)")

            # Interpretation results (경제학파 토론)
            interp = enhanced.get('interpretation', {})
            if interp:
                md.append(f"- **Interpretation**: {interp.get('recommended_action', 'N/A')}")
                md.append(f"  - Schools: Monetarist / Keynesian / Austrian")
                if interp.get('consensus_points'):
                    md.append("  - Consensus: " + "; ".join(interp['consensus_points'][:2]))

            # Methodology results (방법론 토론)
            method = enhanced.get('methodology', {})
            if method:
                md.append(f"- **Methodology**: {method.get('selected_methodology', 'N/A')}")
                if method.get('rationale'):
                    md.append(f"  - Rationale: {method['rationale'][:100]}...")

            # Fallback to debate_results dataclass
            if self.debate_results and not enhanced:
                md.append(f"- **Consensus**: {get_val(self.debate_results, 'consensus_position', 'N/A')}")
                consensus_conf = get_val(self.debate_results, 'consensus_confidence', (0, 0))
                if isinstance(consensus_conf, tuple):
                    md.append(f"- **Confidence**: {consensus_conf[0]:.0f}-{consensus_conf[1]:.0f}%")
                consensus_pts = get_val(self.debate_results, 'consensus_points', [])
                if consensus_pts:
                    md.append("#### Consensus Points")
                    for p in consensus_pts[:3]:
                        md.append(f"- {p}")

        # 5.2 Agent Contributions
        agent_out = self.debate_consensus.get('agent_outputs', {}) if self.debate_consensus else {}
        if agent_out or self.agent_outputs:
            md.append("### Agent Contributions")
            ao = agent_out if agent_out else self.agent_outputs
            if isinstance(ao, dict):
                if ao.get('analysis'):
                    md.append(f"- **Analysis**: Risk={ao['analysis'].get('total_risk_score', 'N/A')}, Regime={ao['analysis'].get('current_regime', 'N/A')}")
                if ao.get('forecast'):
                    fc = ao['forecast'].get('forecast_results', [{}])[0] if ao['forecast'].get('forecast_results') else {}
                    md.append(f"- **Forecast**: Point={fc.get('point_forecast', 'N/A')}")
            elif ao:
                research = get_val(ao, 'research', {})
                strategy = get_val(ao, 'strategy', {})
                if research:
                    md.append(f"- **Research**: {str(research.get('summary', 'N/A'))[:100]}...")
                if strategy:
                    md.append(f"- **Strategy**: {str(strategy.get('reasoning', 'N/A'))[:100]}...")

        # 5.3 Verification Report
        verif = self.debate_consensus.get('verification', {}) if self.debate_consensus else {}
        if verif or self.verification:
            md.append("### Verification Report")
            v = verif if verif else self.verification
            if isinstance(v, dict):
                md.append(f"- **Reliability**: {v.get('overall_score', 'N/A')}/100")
                md.append(f"- **Passed**: {'✅' if v.get('passed') else '❌'}")
                md.append(f"- **Hallucination Risk**: {v.get('hallucination_risk', 'N/A')}")
                if v.get('warnings'):
                    md.append("#### Warnings")
                    for w in v['warnings'][:3]:
                        md.append(f"- ⚠️ {w}")
            else:
                md.append(f"- **Reliability**: {get_val(v, 'overall_reliability', 0):.1f}/100")
                md.append(f"- **Consistency**: {get_val(v, 'consistency_score', 0):.1f}%")
                warnings = get_val(v, 'warnings', [])
                if warnings:
                    md.append("#### Warnings")
                    for w in warnings[:3]:
                        md.append(f"- ⚠️ {w}")

        # 5.4 Reasoning Chain (Traceability)
        if self.reasoning_chain:
            md.append("### Reasoning Chain (Audit Trail)")
            md.append(f"*{len(self.reasoning_chain)} steps tracked*")
            for step in self.reasoning_chain:
                agent = step.get('agent', 'Unknown')
                output = step.get('output', step.get('output_summary', ''))[:80]
                conf = step.get('confidence', 0)
                md.append(f"- **{agent}** ({conf:.0f}%): {output}...")
                factors = step.get('key_factors', [])
                if factors:
                    md.append(f"  - Factors: {', '.join(str(f) for f in factors[:2])}")
        
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

        # NEW: Enhanced Analysis Sections
        md.append("## 6.5 Enhanced Analysis (New)")
        
        if self.hft_microstructure:
            md.append("### HFT Microstructure")
            tr = self.hft_microstructure.get('tick_rule', {})
            if tr: md.append(f"- Tick Rule Buy Ratio: {tr.get('buy_ratio', 0):.1%}")
            kl = self.hft_microstructure.get('kyles_lambda', {})
            if kl: md.append(f"- Kyle's Lambda: {kl.get('lambda', 0):.6f}")

        if self.garch_volatility:
            md.append("### GARCH Volatility")
            md.append(f"- Current: {self.garch_volatility.get('current_volatility', 0):.1%}")
            md.append(f"- Forecast: {self.garch_volatility.get('forecast_avg_volatility', 0):.1%}")

        if self.information_flow:
            md.append("### Information Flow")
            av = self.information_flow.get('abnormal_volume', {})
            if av: md.append(f"- Abnormal Days: {av.get('total_abnormal_days', 0)}")
            capm = self.information_flow.get('capm_QQQ', {})
            if capm: md.append(f"- QQQ Alpha: {capm.get('alpha', 0)*252:+.1%}/yr")

        if self.proof_of_index:
            md.append("### Proof-of-Index")
            md.append(f"- Index Value: {self.proof_of_index.get('index_value', 0):.2f}")
            verify = self.proof_of_index.get('verification', {})
            md.append(f"- On-chain Verification: {'✅ PASS' if verify.get('is_valid') else '❌ FAIL'}")

        if self.dtw_similarity:
            md.append("### DTW Time Series Similarity")
            ll = self.dtw_similarity.get('lead_lag_spy_qqq', {})
            if ll: md.append(f"- Lead-Lag: {ll.get('interpretation')}")
            pair = self.dtw_similarity.get('most_similar_pair', {})
            if pair: md.append(f"- Most Similar: {pair.get('asset1')} <-> {pair.get('asset2')}")

        if self.dbscan_outliers:
            md.append("### DBSCAN Outliers")
            md.append(f"- Outliers: {self.dbscan_outliers.get('n_outliers', 0)} ({self.dbscan_outliers.get('outlier_ratio', 0):.1%})")

        if self.ark_analysis:
            md.append("### ARK Invest Analysis (Cathie Wood)")
            ark = self.ark_analysis
            if ark.get('consensus_buys'):
                md.append(f"- **Consensus BUY**: {', '.join(ark['consensus_buys'])}")
            if ark.get('consensus_sells'):
                md.append(f"- **Consensus SELL**: {', '.join(ark['consensus_sells'])}")
            if ark.get('new_positions'):
                md.append(f"- **New Positions**: {', '.join(ark['new_positions'])}")
            if not (ark.get('consensus_buys') or ark.get('consensus_sells')):
                md.append("- No major consensus trades detected")

        # NEW: Extended Metrics (PCR, Valuation)
        if self.extended_data:
            md.append("### Extended Market Metrics")
            ext = self.extended_data
            pcr = ext.get('put_call_ratio', {})
            if pcr: md.append(f"- **Put/Call Ratio**: {pcr.get('ratio', 0):.2f} ({pcr.get('sentiment')})")
            
            fund = ext.get('fundamentals', {})
            if fund: md.append(f"- **SP500 Earnings Yield**: {fund.get('earnings_yield', 0):.2f}%")
            
            stable = ext.get('digital_liquidity', {})
            if stable: md.append(f"- **Stablecoin Mcap**: ${stable.get('total_mcap', 0)/1e9:.1f}B")

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