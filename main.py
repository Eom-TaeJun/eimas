#!/usr/bin/env python3
"""
EIMAS - Economic Intelligence Multi-Agent System
=================================================
통합 실행 파이프라인

모든 기능을 순차적으로 실행:
1. 데이터 수집 (FRED, Market, Crypto)
2. 레짐 탐지
3. 이벤트 탐지
4. 유동성 분석 (Granger Causality)
5. 멀티에이전트 토론 (소수의견 보호)
6. 실시간 스트리밍 (선택적)
7. DB 저장
8. 알림 발송

제외 항목:
- CME FedWatch 히스토리컬 분석 (2024-2025 확정 데이터)
- LASSO 예측 (히스토리컬 패턴 기반)

Usage:
    python main_integrated.py              # 전체 파이프라인
    python main_integrated.py --realtime   # 실시간 스트리밍 포함
    python main_integrated.py --quick      # 빠른 분석만
"""

import argparse
import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

# 프로젝트 루트
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# Imports
# ============================================================================

# Core
from core.schemas import AnalysisMode, HistoricalDataConfig
from core.debate import DebateProtocol

# Agents
from agents.orchestrator import MetaOrchestrator

# Data Collection
from lib.fred_collector import FREDCollector
from lib.data_collector import DataManager
from lib.unified_data_store import UnifiedDataStore
from lib.market_indicators import MarketIndicatorsCollector

# Analysis
from lib.regime_detector import RegimeDetector
from lib.regime_analyzer import GMMRegimeAnalyzer, get_gmm_regime_summary
from lib.event_framework import QuantitativeEventDetector, EventType
from lib.liquidity_analysis import LiquidityMarketAnalyzer
from lib.causal_network import CausalNetworkAnalyzer
from lib.critical_path import CriticalPathAggregator
from lib.correlation_monitor import CorrelationMonitor
from lib.etf_flow_analyzer import ETFFlowAnalyzer

# Real-time
from lib.binance_stream import BinanceStreamer, StreamConfig
from lib.microstructure import MicrostructureAnalyzer, DailyMicrostructureAnalyzer
from lib.realtime_pipeline import RealtimePipeline, PipelineConfig, SignalDatabase

# Bubble Detection & Market Quality (v2.1.1)
from lib.bubble_detector import BubbleDetector, BubbleWarningLevel

# Database
from lib.trading_db import TradingDB, Signal
from lib.event_db import EventDatabase

# Dual Mode
from lib.dual_mode_analyzer import DualModeAnalyzer, ModeResult

# Advanced Strategy Modules (Part 2 & 3)
from lib.graph_clustered_portfolio import GraphClusteredPortfolio, ClusteringMethod
from lib.shock_propagation_graph import ShockPropagationGraph
from lib.integrated_strategy import IntegratedStrategy, SignalType
from lib.whitening_engine import WhiteningEngine
from lib.custom_etf_builder import CustomETFBuilder, ThemeCategory
from lib.genius_act_macro import GeniusActMacroStrategy, LiquidityIndicators, CryptoRiskEvaluator
from lib.autonomous_agent import AutonomousFactChecker, AIOutputVerifier
from lib.volume_analyzer import VolumeAnalyzer
from lib.causality_graph import CausalityGraphEngine, NodeType, EdgeType
from lib.predictions_db import PredictionsDB, save_eimas_result

# 2026-01-10 추가 모듈
from lib.extended_data_sources import ExtendedDataCollector, DeFiLlamaCollector
from lib.event_tracker import EventTracker, EventTrackingResult
from lib.adaptive_agents import (
    AdaptiveAgentManager, MarketCondition,
    AggressiveAdaptiveAgent, BalancedAdaptiveAgent, ConservativeAdaptiveAgent
)
from lib.validation_agents import ValidationLoopManager, FeedbackLoopResult

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('eimas.integrated')


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MarketQualityMetrics:
    """시장 미세구조 품질 메트릭 (API 검증 결과 반영)"""
    avg_liquidity_score: float = 50.0
    liquidity_scores: Dict[str, float] = field(default_factory=dict)  # {ticker: score}
    high_toxicity_tickers: List[str] = field(default_factory=list)    # VPIN 높은 종목
    illiquid_tickers: List[str] = field(default_factory=list)         # 유동성 낮은 종목
    data_quality: str = "COMPLETE"  # COMPLETE, PARTIAL, DEGRADED

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
    """버블 리스크 메트릭 (Greenwood-Shleifer 기반)"""
    overall_status: str = "NONE"  # NONE, WATCH, WARNING, DANGER
    risk_tickers: List[Dict] = field(default_factory=list)  # [{ticker, level, runup_pct, vol_zscore, risk_score}]
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

    # Advanced Strategy (Part 2 & 3)
    portfolio_weights: Dict[str, float] = field(default_factory=dict)
    shock_propagation: Dict = field(default_factory=dict)
    integrated_signals: List[Dict] = field(default_factory=list)
    genius_act_regime: str = "NEUTRAL"
    genius_act_signals: List[Dict] = field(default_factory=list)
    whitening_summary: str = ""
    fact_check_grade: str = "N/A"
    theme_etf_analysis: Dict = field(default_factory=dict)

    # Volume Anomaly (Task 4)
    volume_anomalies: List[Dict] = field(default_factory=list)
    volume_analysis_summary: str = ""

    # Market Quality & Bubble Risk (v2.1.1 - API 검증 결과 반영)
    market_quality: Optional[MarketQualityMetrics] = None
    bubble_risk: Optional[BubbleRiskMetrics] = None

    # Risk Score Transparency (리스크 점수 분해)
    base_risk_score: float = 0.0
    microstructure_adjustment: float = 0.0  # ±10 범위
    bubble_risk_adjustment: float = 0.0     # multiplier 효과

    # Crypto Stress Test (v2.1.2 - Elicit Enhancement)
    crypto_stress_test: Dict = field(default_factory=dict)

    # Devil's Advocate Summary (v2.1.2 - 반대 논거)
    devils_advocate_arguments: List[str] = field(default_factory=list)

    # HRP Allocation Rationale (v2.1.2 - 배분 근거)
    hrp_allocation_rationale: str = ""

    # Extended Data Sources (v2.1.3 - DeFiLlama, MENA)
    defi_tvl: Dict = field(default_factory=dict)
    mena_markets: Dict = field(default_factory=dict)
    onchain_risk_signals: List[Dict] = field(default_factory=list)

    # Event Tracking (v2.1.3 - 이상→뉴스 역추적)
    event_tracking: Dict = field(default_factory=dict)
    tracked_events: List[Dict] = field(default_factory=list)

    # Adaptive Portfolio (v2.1.3 - 동적 포트폴리오)
    adaptive_portfolios: Dict = field(default_factory=dict)  # {agent_type: portfolio}
    validation_loop_result: Dict = field(default_factory=dict)

    # Correlation Analysis (v2.1.4 - 상관관계 히트맵)
    correlation_matrix: List[List[float]] = field(default_factory=list)  # NxN 상관관계 매트릭스
    correlation_tickers: List[str] = field(default_factory=list)  # 티커 목록

    def to_dict(self) -> Dict:
        return asdict(self)

    def _generate_potential_concerns(self) -> List[str]:
        """
        만장일치 상황에서도 투자자에게 제공할 잠재적 우려사항 생성

        현재 분석 결과를 기반으로 AI가 검토한 리스크 요소들을 반환
        """
        concerns = []

        # 1. 레짐 기반 우려사항
        regime_info = self.regime or {}
        regime = regime_info.get('regime', 'Unknown')
        volatility = regime_info.get('volatility', 'Unknown')

        if regime == 'BULL':
            concerns.append(
                f"현재 Bull 레짐이나, 과열 신호(과매수) 전환 가능성 상시 모니터링 필요"
            )
        elif regime == 'BEAR':
            concerns.append(
                f"Bear 레짐에서 추가 하락 리스크 존재. 방어적 포지션 유지 권고"
            )

        # 2. 리스크 스코어 기반 우려사항
        if self.risk_score < 30:
            concerns.append(
                f"리스크 점수 {self.risk_score:.1f}/100으로 낮지만, 급격한 외부 충격(지정학적 이벤트 등)에 취약할 수 있음"
            )
        elif self.risk_score > 60:
            concerns.append(
                f"리스크 점수 {self.risk_score:.1f}/100으로 상승. 포지션 축소 또는 헤지 고려"
            )

        # 3. 유동성 기반 우려사항
        fred_info = self.fred_summary or {}
        rrp = fred_info.get('rrp', 0)
        if rrp and rrp < 100:  # RRP 100B 미만
            concerns.append(
                f"역레포(RRP) 잔액 ${rrp:.0f}B로 감소. 유동성 완충 여력 축소 가능성"
            )

        # 4. 버블 리스크 기반 우려사항
        if hasattr(self, 'bubble_risk') and self.bubble_risk:
            bubble_status = getattr(self.bubble_risk, 'overall_status', 'NONE')
            if bubble_status != 'NONE':
                concerns.append(
                    f"버블 리스크 상태: {bubble_status}. 고평가 자산 비중 점검 필요"
                )

        # 5. 모드 일치 시에도 신뢰도 경고
        if self.confidence < 0.7:
            concerns.append(
                f"분석 신뢰도 {self.confidence*100:.0f}%로 보통 수준. 추가 검증 권장"
            )

        # 기본 우려사항 (항상 포함)
        if not concerns:
            concerns = [
                "현재 분석 기준으로는 주요 리스크 요소 미탐지. 그러나 예측 불가 이벤트(블랙스완)는 상시 존재",
                "과거 데이터 기반 분석의 한계 인식 필요. 시장 구조 변화 시 모델 재검토 권장",
                "단기 변동성보다 중장기 펀더멘털 변화에 주목할 것"
            ]

        return concerns[:3]

    def to_markdown(self) -> str:
        """마크다운 형식 리포트 생성"""
        md = []
        md.append("# EIMAS Analysis Report")
        md.append(f"**Generated**: {self.timestamp}")
        md.append("")

        # 1. Data Summary
        md.append("## 1. Data Summary")
        md.append("")
        md.append("### FRED Data")
        if self.fred_summary:
            md.append(f"- **RRP**: ${self.fred_summary.get('rrp', 0):.0f}B (Delta: {self.fred_summary.get('rrp_delta', 0):+.0f}B)")
            md.append(f"- **TGA**: ${self.fred_summary.get('tga', 0):.0f}B (Delta: {self.fred_summary.get('tga_delta', 0):+.0f}B)")
            md.append(f"- **Net Liquidity**: ${self.fred_summary.get('net_liquidity', 0):.0f}B")
            md.append(f"- **Liquidity Regime**: {self.fred_summary.get('liquidity_regime', 'N/A')}")
            md.append(f"- **Fed Funds**: {self.fred_summary.get('fed_funds', 0):.2f}%")
            md.append(f"- **10Y-2Y Spread**: {self.fred_summary.get('spread_10y2y', 0):.2f}% ({self.fred_summary.get('curve_status', 'N/A')})")
        else:
            md.append("- No FRED data available")
        md.append("")
        md.append(f"### Market Data")
        md.append(f"- **Tickers collected**: {self.market_data_count}")
        md.append(f"- **Crypto tickers**: {self.crypto_data_count}")
        md.append("")

        # 2. Regime Analysis
        md.append("## 2. Regime Analysis")
        md.append("")
        if self.regime:
            md.append(f"- **Current Regime**: {self.regime.get('regime', 'Unknown')}")
            md.append(f"- **Trend**: {self.regime.get('trend', 'Unknown')}")
            md.append(f"- **Volatility**: {self.regime.get('volatility', 'Unknown')}")
            md.append(f"- **Confidence**: {self.regime.get('confidence', 0):.0%}")
            if self.regime.get('description'):
                md.append(f"- **Description**: {self.regime.get('description')}")
            if self.regime.get('strategy'):
                md.append(f"- **Strategy**: {self.regime.get('strategy')}")

            # GMM & Entropy 분석 결과 (통계적 고도화)
            if self.regime.get('gmm_regime'):
                md.append("")
                md.append("**GMM Statistical Analysis (통계적 레짐 분석):**")
                gmm_probs = self.regime.get('gmm_probabilities', {})
                md.append(f"- **GMM Regime**: {self.regime.get('gmm_regime')}")
                md.append(f"- **Probabilities**: Bull:{gmm_probs.get('Bull', 0):.0%} / Neutral:{gmm_probs.get('Neutral', 0):.0%} / Bear:{gmm_probs.get('Bear', 0):.0%}")
                md.append(f"- **Shannon Entropy**: {self.regime.get('entropy', 0):.3f} ({self.regime.get('entropy_level', 'N/A')})")
                md.append(f"- **Signal Interpretation**: {self.regime.get('entropy_interpretation', 'N/A')}")
        else:
            md.append("- No regime data available")
        md.append("")

        # 3. Risk Assessment
        md.append("## 3. Risk Assessment")
        md.append("")
        md.append(f"- **Risk Score**: {self.risk_score:.1f}/100")
        md.append(f"- **Risk Level**: {self.risk_level}")
        md.append(f"- **Liquidity Signal**: {self.liquidity_signal}")
        md.append("")

        # Risk Score Transparency (리스크 점수 분해)
        if self.base_risk_score > 0 or self.microstructure_adjustment != 0 or self.bubble_risk_adjustment != 0:
            md.append("### Risk Score Breakdown")
            md.append("")
            md.append("| Component | Value | Description |")
            md.append("|-----------|-------|-------------|")
            md.append(f"| Base Score | {self.base_risk_score:.1f} | CriticalPath 기본 점수 |")
            micro_adj_desc = "유동성 우수" if self.microstructure_adjustment < 0 else "유동성 부족" if self.microstructure_adjustment > 0 else "중립"
            md.append(f"| Microstructure Adj. | {self.microstructure_adjustment:+.1f} | {micro_adj_desc} |")
            bubble_adj_desc = {0: "버블 징후 없음", 5: "관찰 필요", 10: "경고", 15: "위험"}.get(int(self.bubble_risk_adjustment), "N/A")
            md.append(f"| Bubble Risk Adj. | +{self.bubble_risk_adjustment:.0f} | {bubble_adj_desc} |")
            md.append(f"| **Final Score** | **{self.risk_score:.1f}** | |")
            md.append("")

        # 3.1 Market Quality & Bubble Risk (v2.1.1)
        if self.market_quality or self.bubble_risk:
            md.append("### Market Quality & Bubble Risk")
            md.append("")

            # Market Quality Metrics
            if self.market_quality:
                md.append("**Market Microstructure Quality:**")
                md.append(f"- Avg Liquidity Score: {self.market_quality.avg_liquidity_score:.1f}/100")
                if self.market_quality.high_toxicity_tickers:
                    md.append(f"- High Toxicity (VPIN>50%): {', '.join(self.market_quality.high_toxicity_tickers[:5])}")
                if self.market_quality.illiquid_tickers:
                    md.append(f"- Illiquid Tickers: {', '.join(self.market_quality.illiquid_tickers[:5])}")
                md.append(f"- Data Quality: {self.market_quality.data_quality}")
                md.append("")

            # Bubble Risk Metrics
            if self.bubble_risk:
                status_emoji = {"NONE": "🟢", "WATCH": "🟡", "WARNING": "🟠", "DANGER": "🔴", "ERROR": "⚫"}.get(self.bubble_risk.overall_status, "⚪")
                md.append(f"**Bubble Risk Assessment:** {status_emoji} **{self.bubble_risk.overall_status}**")
                md.append("")

                if self.bubble_risk.risk_tickers:
                    md.append("| Ticker | Level | 2Y Run-up | Vol Z-Score | Risk Score |")
                    md.append("|--------|-------|-----------|-------------|------------|")
                    for rt in self.bubble_risk.risk_tickers[:5]:
                        level_emoji = {"WATCH": "🟡", "WARNING": "🟠", "DANGER": "🔴"}.get(rt['level'], "")
                        md.append(f"| {rt['ticker']} | {level_emoji} {rt['level']} | {rt['runup_pct']:+.0f}% | {rt['vol_zscore']:.1f}σ | {rt['risk_score']:.0f} |")
                    md.append("")

                    if self.bubble_risk.highest_risk_ticker:
                        md.append(f"> **Alert**: {self.bubble_risk.highest_risk_ticker} shows elevated bubble characteristics. Consider hedging or position reduction.")
                        md.append("")

                md.append(f"_Methodology: {self.bubble_risk.methodology_notes}_")
                md.append("")

        # 4. Events Detected
        md.append("## 4. Events Detected")
        md.append("")
        if self.events_detected:
            for event in self.events_detected:
                md.append(f"- **{event.get('type', 'Unknown')}** [{event.get('importance', 'N/A')}]: {event.get('description', '')}")
        else:
            md.append("- No events detected")
        md.append("")

        # 5. Multi-Agent Debate
        md.append("## 5. Multi-Agent Debate")
        md.append("")
        md.append(f"- **FULL Mode Position**: {self.full_mode_position}")
        md.append(f"- **REFERENCE Mode Position**: {self.reference_mode_position}")
        agree_status = "YES" if self.modes_agree else "NO"
        md.append(f"- **Modes Agree**: {agree_status}")
        if self.dissent_records:
            md.append(f"- **Dissent Records**: {len(self.dissent_records)}")
        if self.has_strong_dissent:
            md.append("- **[!] Strong dissent exists - review carefully**")
        md.append("")

        # Devil's Advocate Summary (v2.1.2) - 항상 출력
        md.append("### Devil's Advocate (반대 논거)")
        md.append("")
        if self.devils_advocate_arguments:
            md.append("_토론 과정에서 제기된 반대 의견:_")
            md.append("")
            for i, arg in enumerate(self.devils_advocate_arguments[:3], 1):
                md.append(f"- **{i}.** {arg}")
        else:
            # 만장일치 시에도 잠재적 우려사항 표시
            md.append("_토론 결과 만장일치. 다음은 AI가 검토한 잠재적 우려사항:_")
            md.append("")
            potential_concerns = self._generate_potential_concerns()
            for i, concern in enumerate(potential_concerns[:3], 1):
                md.append(f"- **{i}.** {concern}")
        md.append("")

        # 6. Advanced Analysis
        md.append("## 6. Advanced Analysis")
        md.append("")

        # Genius Act
        md.append("### Genius Act Macro")
        md.append(f"- **Regime**: {self.genius_act_regime}")
        if self.genius_act_signals:
            md.append(f"- **Signals**: {len(self.genius_act_signals)} detected")
            md.append("")
            md.append("**Signal Details (Why 설명 포함):**")
            for sig in self.genius_act_signals[:5]:
                if isinstance(sig, dict):
                    strength_val = sig.get('strength', 0)
                    try:
                        strength_fmt = f"{float(strength_val):.2f}"
                    except (ValueError, TypeError):
                        strength_fmt = str(strength_val)
                    md.append(f"- **{sig.get('type', 'N/A')}** (strength: {strength_fmt})")
                    md.append(f"  - Description: {sig.get('description', 'N/A')}")
                    md.append(f"  - Why: {sig.get('why', 'N/A')}")
                    if sig.get('affected_assets'):
                        md.append(f"  - Affected: {', '.join(sig['affected_assets'][:5])}")
        md.append("")

        # Crypto Stress Test (v2.1.2 - Elicit Enhancement) - 항상 출력
        md.append("### Crypto Stress Test")
        md.append("")
        if self.crypto_stress_test and not self.crypto_stress_test.get('error'):
            md.append(f"**Scenario**: {self.crypto_stress_test.get('scenario', 'N/A')}")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            md.append(f"| De-peg Probability | **{self.crypto_stress_test.get('depeg_probability_pct', '0.0%')}** |")
            md.append(f"| Estimated Loss under Stress | **${self.crypto_stress_test.get('estimated_loss_under_stress', 0):,.0f}** ({self.crypto_stress_test.get('estimated_loss_pct', '0.0%')}) |")
            md.append(f"| Total Value at Risk | ${self.crypto_stress_test.get('total_value', 0):,.0f} |")
            md.append(f"| Risk Rating | {self.crypto_stress_test.get('risk_rating', 'N/A')} |")
            md.append("")

            # Breakdown by coin
            breakdown = self.crypto_stress_test.get('breakdown_by_coin', [])
            if breakdown:
                md.append("**Breakdown by Stablecoin:**")
                md.append("")
                md.append("| Coin | Amount | De-peg Prob | Expected Loss |")
                md.append("|------|--------|-------------|---------------|")
                for coin in breakdown[:5]:
                    md.append(f"| {coin['ticker']} | ${coin['amount']:,.0f} | {coin['depeg_probability']*100:.1f}% | ${coin['expected_loss']:,.0f} |")
                md.append("")

            methodology = self.crypto_stress_test.get('methodology_note',
                '스트레스 테스트: 담보 유형별 리스크 가중치 적용. De-peg 확률 및 예상 손실 산출.')
            md.append(f"_Methodology: {methodology}_")
        else:
            # 데이터 없을 때도 표 구조 표시 (검증 증거)
            md.append("**Scenario**: Moderate (신용위기 수준)")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            md.append("| De-peg Probability | **0.0%** (데이터 미수집) |")
            md.append("| Estimated Loss under Stress | **$0** |")
            md.append("| Total Value at Risk | $0 |")
            md.append("| Risk Rating | N/A |")
            md.append("")
            md.append("_Note: 스테이블코인 데이터가 수집되지 않았습니다. 전체 분석 모드로 실행하세요._")
        md.append("")

        # Theme ETF
        if self.theme_etf_analysis:
            md.append("### Theme ETF Analysis")
            md.append(f"- **Theme**: {self.theme_etf_analysis.get('theme', 'N/A')}")
            if self.theme_etf_analysis.get('description'):
                md.append(f"- **Description**: {self.theme_etf_analysis.get('description')}")
            md.append(f"- **Stocks Count**: {self.theme_etf_analysis.get('stocks_count', 0)}")
            top5 = self.theme_etf_analysis.get('top5_concentration', 0)
            try:
                md.append(f"- **Top 5 Concentration**: {float(top5):.1%}")
            except (ValueError, TypeError):
                md.append(f"- **Top 5 Concentration**: {top5}")
            div_score = self.theme_etf_analysis.get('diversification_score', 0)
            try:
                md.append(f"- **Diversification Score**: {float(div_score):.2f}")
            except (ValueError, TypeError):
                md.append(f"- **Diversification Score**: {div_score}")

            # Supply Chain 인과관계 설명
            supply_chain = self.theme_etf_analysis.get('supply_chain', {})
            if supply_chain:
                md.append("")
                md.append("**Supply Chain Structure:**")
                if supply_chain.get('bottlenecks'):
                    md.append(f"- Bottlenecks: {', '.join(supply_chain['bottlenecks'])}")
                if supply_chain.get('top_central'):
                    md.append(f"- Hub Nodes: {', '.join(supply_chain['top_central'])}")

            # Graph-based Causality Narrative (자연어 인과관계)
            graph_narrative = self.theme_etf_analysis.get('graph_narrative', '')
            if graph_narrative and graph_narrative != "Not enough correlation data to build causality chain yet.":
                md.append("")
                md.append("**Causality Network Analysis (인과관계 네트워크):**")
                md.append("")
                md.append(graph_narrative)
                md.append("")
            elif graph_narrative:
                # Fallback message
                md.append("")
                md.append("**Causality Chain (인과관계):**")
                md.append(f"> {graph_narrative}")
                md.append("")
            else:
                # Legacy causality explanation (하위 호환)
                causality = self.theme_etf_analysis.get('causality_explanation', '')
                if causality:
                    md.append("")
                    md.append("**Causality Chain (인과관계):**")
                    md.append("```")
                    md.append(causality)
                    md.append("```")
            md.append("")

        # Shock Propagation
        if self.shock_propagation:
            md.append("### Shock Propagation")
            md.append(f"- **Nodes**: {self.shock_propagation.get('nodes', 0)}")
            md.append(f"- **Edges**: {self.shock_propagation.get('edges', 0)}")
            if self.shock_propagation.get('critical_path'):
                path_str = ' -> '.join(self.shock_propagation['critical_path'][:5])
                md.append(f"- **Critical Path**: {path_str}")
            md.append("")

        # Portfolio
        if self.portfolio_weights:
            md.append("### GC-HRP Portfolio")
            sorted_weights = sorted(self.portfolio_weights.items(), key=lambda x: x[1], reverse=True)
            md.append("| Ticker | Weight |")
            md.append("|--------|--------|")
            for ticker, weight in sorted_weights[:10]:
                md.append(f"| {ticker} | {weight:.1%} |")
            md.append("")

            # HRP Allocation Rationale (v2.1.2)
            if self.hrp_allocation_rationale:
                md.append(f"**Allocation Rationale**: {self.hrp_allocation_rationale}")
                md.append("")

        # Integrated Signals
        if self.integrated_signals:
            md.append("### Integrated Signals")
            for sig in self.integrated_signals[:5]:
                md.append(f"- **{sig.get('type', 'Unknown')}** [{sig.get('urgency', 'N/A')}]: {sig.get('description', '')[:80]}")
            md.append("")

        # Volume Anomalies
        if self.volume_anomalies:
            md.append("### Volume Anomaly Detection")
            md.append(f"_{self.volume_analysis_summary}_")
            md.append("")
            md.append("| Ticker | Volume Ratio | Severity | Alert |")
            md.append("|--------|--------------|----------|-------|")
            for anomaly in self.volume_anomalies[:10]:
                md.append(f"| {anomaly.get('ticker', 'N/A')} | {anomaly.get('volume_ratio', 0):.1f}x | {anomaly.get('severity', 'N/A')} | {anomaly.get('alert_message', '')[:50]}... |")
            md.append("")

        # 7. Final Recommendation
        md.append("## 7. Final Recommendation")
        md.append("")
        md.append(f"| Item | Value |")
        md.append("|------|-------|")
        md.append(f"| Action | **{self.final_recommendation}** |")
        md.append(f"| Confidence | {self.confidence:.0%} |")
        md.append(f"| Risk Level | {self.risk_level} |")
        md.append("")

        # 8. Warnings
        if self.warnings:
            md.append("## 8. Warnings")
            md.append("")
            for warning in self.warnings:
                md.append(f"- [!] {warning}")
            md.append("")

        # 9. Real-time VPIN Monitoring (if available)
        if self.realtime_signals:
            md.append("## 9. Real-time VPIN Monitoring")
            md.append("")
            md.append("_Binance WebSocket을 통한 실시간 시장 미세구조 분석 결과_")
            md.append("")

            # 심볼별 요약
            symbol_data = {}
            for sig in self.realtime_signals:
                symbol = sig.get('symbol', 'UNKNOWN')
                if symbol not in symbol_data:
                    symbol_data[symbol] = {'vpins': [], 'max_vpin': 0}
                avg_vpin = sig.get('avg_vpin', 0)
                max_vpin = sig.get('max_vpin', 0)
                symbol_data[symbol]['vpins'].append(avg_vpin)
                symbol_data[symbol]['max_vpin'] = max(symbol_data[symbol]['max_vpin'], max_vpin)

            md.append("### Summary by Symbol")
            md.append("")
            md.append("| Symbol | Avg VPIN | Max VPIN | Samples | Status |")
            md.append("|--------|----------|----------|---------|--------|")

            for symbol, data in symbol_data.items():
                avg = sum(data['vpins']) / len(data['vpins']) if data['vpins'] else 0
                max_v = data['max_vpin']
                samples = len(data['vpins'])

                # 상태 판정
                if max_v >= 0.7:
                    status = "🚨 EXTREME"
                elif max_v >= 0.6:
                    status = "🔶 HIGH"
                elif max_v >= 0.5:
                    status = "⚠️ ELEVATED"
                else:
                    status = "✅ NORMAL"

                md.append(f"| {symbol} | {avg:.3f} | {max_v:.3f} | {samples} | {status} |")

            md.append("")

            # 최근 기록 (최대 10개)
            md.append("### Recent 1-min VPIN Records")
            md.append("")
            md.append("| Timestamp | Symbol | Avg VPIN | Max VPIN |")
            md.append("|-----------|--------|----------|----------|")

            for sig in self.realtime_signals[-10:]:
                ts = sig.get('timestamp', 'N/A')
                if isinstance(ts, str) and 'T' in ts:
                    ts = ts.split('T')[1][:8]  # HH:MM:SS만 추출
                symbol = sig.get('symbol', 'N/A')
                avg_vpin = sig.get('avg_vpin', 0)
                max_vpin = sig.get('max_vpin', 0)
                md.append(f"| {ts} | {symbol} | {avg_vpin:.3f} | {max_vpin:.3f} |")

            md.append("")
            md.append("_VPIN Thresholds: Normal(<0.4), Elevated(0.5), High(0.6), Extreme(0.7)_")
            md.append("")

        # 10. Whitening & Fact Check (if available)
        if self.whitening_summary or self.fact_check_grade != "N/A":
            md.append("## 10. Quality Assurance")
            md.append("")
            if self.whitening_summary:
                md.append(f"### Whitening Summary")
                md.append(f"{self.whitening_summary}")
                md.append("")
            if self.fact_check_grade != "N/A":
                md.append(f"### Fact Check Grade: {self.fact_check_grade}")
            md.append("")

        md.append("---")
        md.append("*Generated by EIMAS (Economic Intelligence Multi-Agent System)*")

        return "\n".join(md)


# ============================================================================
# Helper Functions
# ============================================================================

def _get_genius_act_why(signal_type: str, metadata: Dict) -> str:
    """
    Genius Act Signal의 경제학적 이유(Why) 설명 생성

    경제학적 근거:
    - Genius Act(스테이블코인 규제법): 스테이블코인 담보로 미국 국채 요구
    - M = B + S·B* 확장 유동성 공식 (순유동성 + 스테이블코인 기여도)
    """
    why_map = {
        'stablecoin_surge': (
            "스테이블코인(USDT/USDC) 발행량 급증 → "
            "Genius Act 담보 요건으로 미국 국채 수요 상승 → "
            "국채 가격 강세(금리 하락) 및 크립토 매수 대기 자금 증가"
        ),
        'stablecoin_drain': (
            "스테이블코인 공급 감소 → "
            "크립토 시장에서 자금 이탈 신호 → "
            "리스크오프 전환, 현금화 압력 증가"
        ),
        'rrp_drain': (
            "역레포(RRP) 잔액 감소 → "
            "시중 유동성 공급 (B = Fed BS - RRP - TGA 공식) → "
            "위험자산(주식, 크립토) 강세 환경 조성"
        ),
        'tga_drain': (
            "재무부 일반계정(TGA) 감소 → "
            "정부 지출로 시중 유동성 주입 → "
            "소비 및 투자 확대 기대, 주식 강세"
        ),
        'liquidity_injection': (
            "순 유동성(Net Liquidity) 증가 → "
            "Fed BS - RRP - TGA 확대 → "
            "모든 위험자산에 우호적 환경"
        ),
        'liquidity_drain': (
            "순 유동성 감소 → "
            "긴축 환경, 자산 가격 하락 압력 → "
            "포트폴리오 방어적 전환 필요"
        ),
        'crypto_risk_on': (
            "크립토 리스크온 환경 → "
            "스테이블코인 유입 + 유동성 확대 → "
            "비트코인/이더리움 상승 모멘텀"
        ),
        'crypto_risk_off': (
            "크립토 리스크오프 환경 → "
            "스테이블코인 이탈 + 유동성 축소 → "
            "비트코인/이더리움 하락 압력"
        ),
        'treasury_demand': (
            "국채 수요 증가 → "
            "안전자산 선호 또는 스테이블코인 담보 수요 → "
            "금리 하락, 성장주 상대적 강세"
        ),
        'treasury_supply': (
            "국채 공급 증가 (재정적자 확대) → "
            "금리 상승 압력 → "
            "밸류주/금융주 상대적 강세, 성장주 약세"
        ),
    }

    base_why = why_map.get(signal_type, "경제학적 분석 결과에 기반한 시그널")

    # 메타데이터에서 추가 정보
    if metadata:
        if 'rrp_drain' in metadata:
            base_why += f" (RRP 감소: {metadata['rrp_drain']})"
        if 'total_supply' in metadata:
            base_why += f" (스테이블코인 총 공급: {metadata['total_supply']})"

    return base_why


def _generate_hrp_rationale(
    weights: Dict[str, float],
    returns_df: 'pd.DataFrame',
    clusters: List[Dict]
) -> str:
    """
    HRP 포트폴리오 배분 근거 자동 생성 (v2.1.2 - Elicit Enhancement)

    경제학적 근거:
    - HRP는 낮은 변동성 자산에 높은 비중 부여 (역변동성 가중)
    - 상관관계가 낮은 자산일수록 분산 효과 기여도 높음
    - 달러(UUP) 등 방어 자산은 포트폴리오 변동성 방어 역할
    """
    if not weights:
        return "No allocation data available"

    import pandas as pd

    # Top 3 자산 분석
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    rationale_parts = []

    # 자산별 특성 분석
    ASSET_CHARACTERISTICS = {
        'UUP': ('US Dollar', 'volatility hedge, negative equity correlation'),
        'TLT': ('Long Treasury', 'flight-to-quality, duration exposure'),
        'GLD': ('Gold', 'inflation hedge, crisis alpha'),
        'SHY': ('Short Treasury', 'cash proxy, capital preservation'),
        'SPY': ('S&P 500', 'core equity exposure, market beta'),
        'QQQ': ('Nasdaq 100', 'tech/growth exposure, high beta'),
        'IWM': ('Small Cap', 'domestic growth, higher volatility'),
        'EFA': ('Intl Developed', 'geographic diversification'),
        'EEM': ('Emerging Markets', 'growth potential, currency risk'),
        'VNQ': ('REITs', 'real asset exposure, income'),
        'XLE': ('Energy', 'commodity exposure, inflation hedge'),
        'XLF': ('Financials', 'rate sensitivity, economic cycle'),
        'XLK': ('Technology', 'secular growth, momentum'),
        'XLV': ('Healthcare', 'defensive growth, demographics'),
        'BTC-USD': ('Bitcoin', 'digital gold, high volatility'),
        'ETH-USD': ('Ethereum', 'smart contract platform, tech beta'),
    }

    # 각 Top 자산의 비중 이유 분석
    for ticker, weight in sorted_weights:
        pct = weight * 100
        asset_name, characteristic = ASSET_CHARACTERISTICS.get(
            ticker, (ticker, 'portfolio diversification')
        )

        # 변동성 계산 (가능한 경우)
        vol_comment = ""
        if ticker in returns_df.columns:
            vol = returns_df[ticker].std() * (252 ** 0.5) * 100  # Annualized %
            if vol < 15:
                vol_comment = "low volatility"
            elif vol > 30:
                vol_comment = "high volatility, diversification benefit"
            else:
                vol_comment = "moderate volatility"

        if pct >= 15:
            reason = f"{ticker} ({pct:.0f}%): {characteristic}"
            if vol_comment:
                reason += f" [{vol_comment}]"
            rationale_parts.append(reason)

    # 클러스터 정보 활용
    cluster_comment = ""
    if clusters:
        n_clusters = len(clusters)
        cluster_comment = f" | {n_clusters} clusters identified for risk parity"

    if rationale_parts:
        return "; ".join(rationale_parts) + cluster_comment
    else:
        return f"Diversified allocation across {len(weights)} assets{cluster_comment}"


def _extract_devils_advocate_arguments(dissent_records: List[Dict]) -> List[str]:
    """
    토론 기록에서 Devil's Advocate 논거 추출 (v2.1.2 - Elicit Enhancement)

    다수 의견에 대한 주요 반대 논거를 요약하여 반환
    """
    if not dissent_records:
        return []

    arguments = []

    # 반대 의견 중 가장 중요한 3가지 추출
    for record in dissent_records[:5]:
        dissenter = record.get('dissenter', 'Unknown')
        reason = record.get('reason', record.get('dissent_reason', ''))
        confidence = record.get('confidence', 0)

        if reason:
            # 이유 요약 (첫 150자)
            reason_summary = reason[:150].strip()
            if len(reason) > 150:
                reason_summary += "..."

            argument = f"[{dissenter}] {reason_summary}"
            arguments.append(argument)

    # 최대 3개 반환
    return arguments[:3]


# ============================================================================
# Real-time Monitor (Phase 4)
# ============================================================================

class RealtimeVPINMonitor:
    """
    실시간 VPIN 모니터링 시스템

    Binance WebSocket으로 BTC/ETH 가격을 실시간 수신하고,
    1분 단위로 VPIN을 계산하여 임계치 초과 시 터미널 경고

    사용법:
        monitor = RealtimeVPINMonitor()
        await monitor.start(duration=60)  # 60초 실행
    """

    # VPIN 임계치 (경제학적 근거: Easley et al. 2012)
    VPIN_THRESHOLDS = {
        'normal': 0.4,      # 정상 범위
        'elevated': 0.5,    # 주의
        'high': 0.6,        # 높음
        'extreme': 0.7      # 극단 - 변동성 급증 가능
    }

    def __init__(self, symbols: List[str] = None, verbose: bool = True):
        """
        Parameters:
        -----------
        symbols : List[str]
            모니터링할 심볼 리스트 (기본: ['BTCUSDT', 'ETHUSDT'])
        verbose : bool
            상세 출력 여부
        """
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.verbose = verbose

        # 1분 단위 VPIN 집계
        self.minute_vpin: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.minute_start: Dict[str, datetime] = {}

        # 알림 기록 (중복 방지)
        self.last_alert_time: Dict[str, datetime] = {}
        self.alert_cooldown = 30  # 30초 쿨다운

        # 통계
        self.alerts_fired = 0
        self.vpin_history: Dict[str, List[Dict]] = {s: [] for s in self.symbols}

    def _log(self, msg: str, level: str = 'info'):
        """로깅"""
        if not self.verbose:
            return

        timestamp = datetime.now().strftime('%H:%M:%S')

        if level == 'alert':
            print(f"\n🚨 [{timestamp}] {msg}")
        elif level == 'warning':
            print(f"⚠️  [{timestamp}] {msg}")
        elif level == 'success':
            print(f"✅ [{timestamp}] {msg}")
        else:
            print(f"   [{timestamp}] {msg}")

    def _check_vpin_threshold(self, symbol: str, vpin: float) -> Optional[str]:
        """VPIN 임계치 확인 및 경고 레벨 반환"""
        if vpin >= self.VPIN_THRESHOLDS['extreme']:
            return 'extreme'
        elif vpin >= self.VPIN_THRESHOLDS['high']:
            return 'high'
        elif vpin >= self.VPIN_THRESHOLDS['elevated']:
            return 'elevated'
        return None

    def _should_alert(self, symbol: str) -> bool:
        """쿨다운 확인 - 중복 알림 방지"""
        now = datetime.now()
        last = self.last_alert_time.get(symbol)

        if last is None:
            return True

        elapsed = (now - last).total_seconds()
        return elapsed >= self.alert_cooldown

    def _fire_alert(self, symbol: str, vpin: float, level: str):
        """터미널 경고 출력"""
        if not self._should_alert(symbol):
            return

        self.last_alert_time[symbol] = datetime.now()
        self.alerts_fired += 1

        # 레벨별 메시지
        level_info = {
            'elevated': ('⚠️ ELEVATED', 'Yellow', '주의 필요'),
            'high': ('🔶 HIGH', 'Orange', '변동성 상승 예상'),
            'extreme': ('🚨 EXTREME', 'Red', '급변동 임박!')
        }

        icon, color, desc = level_info.get(level, ('⚠️', 'Unknown', ''))

        alert_msg = f"""
╔══════════════════════════════════════════════════════════════╗
║  {icon} VPIN ALERT - {symbol}
║--------------------------------------------------------------║
║  VPIN Value: {vpin:.3f} ({level.upper()})
║  Threshold:  {self.VPIN_THRESHOLDS.get(level, 0.5):.2f}
║  Message:    {desc}
║--------------------------------------------------------------║
║  Action: 포지션 점검, 손절 라인 확인, 변동성 대비 필요
╚══════════════════════════════════════════════════════════════╝
"""
        print(alert_msg)

    def _on_metrics(self, metrics):
        """메트릭 수신 콜백 - 1분 단위 VPIN 집계"""
        symbol = metrics.symbol
        vpin = metrics.vpin
        now = datetime.now()

        # 1분 윈도우 시작
        if symbol not in self.minute_start or self.minute_start[symbol] is None:
            self.minute_start[symbol] = now

        # VPIN 값 수집
        self.minute_vpin[symbol].append(vpin)

        # 1분 경과 시 집계
        elapsed = (now - self.minute_start[symbol]).total_seconds()
        if elapsed >= 60:
            # 1분 평균 VPIN 계산
            if self.minute_vpin[symbol]:
                avg_vpin = sum(self.minute_vpin[symbol]) / len(self.minute_vpin[symbol])
                max_vpin = max(self.minute_vpin[symbol])
                min_vpin = min(self.minute_vpin[symbol])

                # 히스토리 저장
                self.vpin_history[symbol].append({
                    'timestamp': now.isoformat(),
                    'avg_vpin': avg_vpin,
                    'max_vpin': max_vpin,
                    'min_vpin': min_vpin,
                    'samples': len(self.minute_vpin[symbol])
                })

                # 1분 요약 출력
                self._log(
                    f"{symbol} 1-min VPIN: avg={avg_vpin:.3f}, "
                    f"max={max_vpin:.3f}, min={min_vpin:.3f} "
                    f"({len(self.minute_vpin[symbol])} samples)",
                    level='info'
                )

                # 임계치 확인
                alert_level = self._check_vpin_threshold(symbol, max_vpin)
                if alert_level:
                    self._fire_alert(symbol, max_vpin, alert_level)

            # 리셋
            self.minute_vpin[symbol] = []
            self.minute_start[symbol] = now

        # 실시간 임계치 체크 (극단적 경우 즉시 알림)
        if vpin >= self.VPIN_THRESHOLDS['extreme']:
            self._fire_alert(symbol, vpin, 'extreme')

    def _on_alert(self, alert_type: str, alert_data: Dict):
        """BinanceStreamer 알림 콜백"""
        if alert_type == 'vpin_high':
            self._log(f"Stream Alert: {alert_data.get('message', '')}", level='warning')

    async def start(self, duration: int = 60):
        """
        실시간 모니터링 시작

        Parameters:
        -----------
        duration : int
            모니터링 시간 (초, 기본 60초)
        """
        print("\n" + "=" * 70)
        print("  EIMAS Real-time VPIN Monitor")
        print("=" * 70)
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(f"  Duration: {duration}s")
        print(f"  VPIN Thresholds: {self.VPIN_THRESHOLDS}")
        print("=" * 70 + "\n")

        # StreamConfig 설정
        config = StreamConfig(
            symbols=self.symbols,
            depth_levels=10,
            update_speed='100ms',
            include_trades=True,
            alert_vpin_threshold=self.VPIN_THRESHOLDS['elevated']
        )

        # BinanceStreamer 생성
        streamer = BinanceStreamer(
            config=config,
            on_metrics=self._on_metrics,
            on_alert=self._on_alert,
            verbose=False  # 자체 로깅 사용
        )

        self._log(f"Connecting to Binance WebSocket...", level='info')

        try:
            # 스트리밍 시작
            await streamer.start(duration_seconds=duration)
        except KeyboardInterrupt:
            self._log("Interrupted by user", level='warning')
        except Exception as e:
            self._log(f"Error: {e}", level='warning')
        finally:
            streamer.stop()

        # 요약 출력
        self._print_summary(streamer)

        return {
            'alerts_fired': self.alerts_fired,
            'vpin_history': self.vpin_history,
            'stream_stats': streamer.stats.to_dict()
        }

    def _print_summary(self, streamer):
        """실행 요약 출력"""
        print("\n" + "=" * 70)
        print("  Real-time Monitor Summary")
        print("=" * 70)

        stats = streamer.stats.to_dict()
        print(f"  Duration: {stats['elapsed_seconds']:.1f}s")
        print(f"  Messages: {stats['messages_received']:,}")
        print(f"  Alerts Fired: {self.alerts_fired}")

        for symbol in self.symbols:
            history = self.vpin_history.get(symbol, [])
            if history:
                avg_vpins = [h['avg_vpin'] for h in history]
                overall_avg = sum(avg_vpins) / len(avg_vpins)
                overall_max = max(h['max_vpin'] for h in history)

                print(f"\n  [{symbol}]")
                print(f"    1-min Samples: {len(history)}")
                print(f"    Overall Avg VPIN: {overall_avg:.4f}")
                print(f"    Overall Max VPIN: {overall_max:.4f}")

                # 최종 상태 판정
                if overall_max >= self.VPIN_THRESHOLDS['extreme']:
                    print(f"    Status: 🚨 EXTREME - 고위험")
                elif overall_max >= self.VPIN_THRESHOLDS['high']:
                    print(f"    Status: 🔶 HIGH - 주의")
                elif overall_max >= self.VPIN_THRESHOLDS['elevated']:
                    print(f"    Status: ⚠️ ELEVATED - 관찰")
                else:
                    print(f"    Status: ✅ NORMAL - 안정")

        print("=" * 70)


async def run_realtime_monitor(
    symbols: List[str] = None,
    duration: int = 60,
    verbose: bool = True
) -> Dict:
    """
    실시간 VPIN 모니터링 실행 (독립 함수)

    Parameters:
    -----------
    symbols : List[str]
        모니터링할 심볼 (기본: BTC, ETH)
    duration : int
        실행 시간 (초)
    verbose : bool
        상세 출력

    Returns:
    --------
    Dict with monitoring results

    사용 예:
        # 기본 실행 (BTC, ETH 60초)
        result = await run_realtime_monitor()

        # 커스텀 설정
        result = await run_realtime_monitor(
            symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
            duration=120
        )
    """
    symbols = symbols or ['BTCUSDT', 'ETHUSDT']
    monitor = RealtimeVPINMonitor(symbols=symbols, verbose=verbose)
    return await monitor.start(duration=duration)


# ============================================================================
# Main Pipeline
# ============================================================================

async def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    output_dir: str = 'outputs',
    cron_mode: bool = False
) -> EIMASResult:
    """
    EIMAS 통합 파이프라인 실행

    Args:
        enable_realtime: 실시간 Binance 스트리밍 활성화
        realtime_duration: 실시간 스트리밍 시간 (초)
        output_dir: 출력 디렉토리
        cron_mode: 서버 자동화 모드
        quick_mode: 빠른 분석 모드 (일부 생략)

    Returns:
        EIMASResult: 통합 분석 결과
    """
    start_time = datetime.now()
    result = EIMASResult(timestamp=start_time.isoformat())

    print("=" * 70)
    print("  EIMAS - Integrated Analysis Pipeline")
    print("=" * 70)
    print(f"  Mode: {'Quick' if quick_mode else 'Full'}")
    print(f"  Realtime: {'Enabled' if enable_realtime else 'Disabled'}")
    print("=" * 70)

    # ========================================================================
    # Phase 1: Data Collection
    # ========================================================================
    print("\n" + "=" * 50)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 50)

    # 1.1 FRED 데이터 수집
    print("\n[1.1] Collecting FRED data...")
    try:
        fred = FREDCollector()
        fred_summary = fred.collect_all()
        result.fred_summary = {
            'rrp': fred_summary.rrp,
            'rrp_delta': fred_summary.rrp_delta,
            'tga': fred_summary.tga,
            'tga_delta': fred_summary.tga_delta,
            'fed_assets': fred_summary.fed_assets,
            'net_liquidity': fred_summary.net_liquidity,
            'liquidity_regime': fred_summary.liquidity_regime,
            'fed_funds': fred_summary.fed_funds,
            'treasury_10y': fred_summary.treasury_10y,
            'spread_10y2y': fred_summary.spread_10y2y,
            'curve_status': fred_summary.curve_status,
        }
        print(f"      ✓ RRP: ${fred_summary.rrp:.0f}B (Δ{fred_summary.rrp_delta:+.0f}B)")
        print(f"      ✓ TGA: ${fred_summary.tga:.0f}B (Δ{fred_summary.tga_delta:+.0f}B)")
        print(f"      ✓ Net Liquidity: ${fred_summary.net_liquidity:.0f}B ({fred_summary.liquidity_regime})")
        print(f"      ✓ Curve: {fred_summary.curve_status} (10Y-2Y: {fred_summary.spread_10y2y:.2f}%)")
    except Exception as e:
        print(f"      ✗ FRED error: {e}")
        fred_summary = None

    # 1.2 시장 데이터 수집
    print("\n[1.2] Collecting market data...")
    try:
        dm = DataManager(lookback_days=365 if not quick_mode else 90)
        # 확장된 티커 목록: 더 많은 종목으로 거래량 이상 탐지 개선
        tickers_config = {
            'market': [
                # 주요 지수 ETF
                {'ticker': 'SPY'}, {'ticker': 'QQQ'}, {'ticker': 'IWM'},
                {'ticker': 'DIA'}, {'ticker': 'TLT'}, {'ticker': 'GLD'},
                {'ticker': 'USO'}, {'ticker': 'UUP'}, {'ticker': '^VIX'},
                # 섹터 ETF (거래량 이상 탐지 확대)
                {'ticker': 'XLK'},   # Technology
                {'ticker': 'XLF'},   # Financials
                {'ticker': 'XLE'},   # Energy
                {'ticker': 'XLV'},   # Healthcare
                {'ticker': 'XLI'},   # Industrials
                # 반도체 및 AI 관련 (Theme ETF 연동)
                {'ticker': 'SMH'},   # VanEck Semiconductor
                {'ticker': 'SOXX'},  # iShares Semiconductor
                # 채권 ETF
                {'ticker': 'HYG'},   # High Yield
                {'ticker': 'LQD'},   # Investment Grade
                {'ticker': 'TIP'},   # TIPS
            ],
            'crypto': [
                {'ticker': 'BTC-USD'}, {'ticker': 'ETH-USD'}
            ],
            # RWA (Real World Asset) - 토큰화 자산 [NEW]
            # 경제학적 근거: "Asset이 infinite... 모든 거래 가능한 걸 토큰화"
            # Note: ONDO-USD, PAXG-USD는 yfinance 호환 형식 (crypto tokens)
            'rwa': [
                {'ticker': 'ONDO-USD'},   # US Treasuries Tokenized Protocol
                {'ticker': 'PAXG-USD'},   # Gold Tokenized (1 token = 1 oz Gold)
                {'ticker': 'COIN'},       # Crypto Infrastructure Proxy (주식)
            ]
        }
        market_data, macro_data = dm.collect_all(tickers_config)
        result.market_data_count = len(market_data)
        print(f"      ✓ Collected {len(market_data)} tickers")
    except Exception as e:
        print(f"      ✗ Market data error: {e}")
        market_data = {}
        macro_data = None

    # 1.3 암호화폐 및 RWA 데이터 (DataManager가 이미 수집)
    print("\n[1.3] Crypto & RWA data collected with market data...")
    crypto_tickers = ['BTC-USD', 'ETH-USD']
    rwa_tickers = ['ONDO-USD', 'PAXG-USD', 'COIN']
    result.crypto_data_count = sum(1 for t in crypto_tickers if t in market_data)
    rwa_count = sum(1 for t in rwa_tickers if t in market_data)
    print(f"      ✓ Crypto: {result.crypto_data_count} tickers")
    print(f"      ✓ RWA (Tokenized Assets): {rwa_count} tickers")

    # 1.4 시장 지표
    if not quick_mode:
        print("\n[1.4] Collecting market indicators...")
        try:
            indicators = MarketIndicatorsCollector()
            indicators_summary = indicators.collect_all()
            print(f"      ✓ VIX: {indicators_summary.vix.current:.2f}")
            print(f"      ✓ Fear & Greed: {indicators_summary.vix.fear_greed_level}")
        except Exception as e:
            print(f"      ✗ Indicators error: {e}")

    # 1.5 확장 데이터 소스 (DeFiLlama, MENA)
    if not quick_mode:
        print("\n[1.5] Extended data sources (DeFi, MENA)...")
        try:
            ext_collector = ExtendedDataCollector()

            # DeFi TVL
            defi_summary = ext_collector.defi.get_summary()
            result.defi_tvl = {
                'total_tvl': defi_summary.get('total_tvl', 0),
                'stablecoin_mcap': defi_summary.get('stablecoin_market_cap', 0),
                'top_stablecoins': defi_summary.get('top_stablecoins', [])
            }
            print(f"      ✓ DeFi TVL: ${defi_summary.get('total_tvl', 0)/1e9:.2f}B")
            print(f"      ✓ Stablecoin MCap: ${defi_summary.get('stablecoin_market_cap', 0)/1e9:.2f}B")

            # MENA Markets
            mena_summary = ext_collector.mena.get_performance_summary()
            result.mena_markets = mena_summary
            if mena_summary.get('etfs'):
                avg_return = mena_summary.get('avg_return_1m', 0)
                print(f"      ✓ MENA ETFs: {len(mena_summary['etfs'])} tracked")
                print(f"      ✓ MENA Avg 1M Return: {avg_return:+.1f}%")

            # On-Chain 리스크 시그널
            onchain_signals = ext_collector.get_risk_signals()
            result.onchain_risk_signals = onchain_signals
            if onchain_signals:
                print(f"      ✓ On-Chain Risk Signals: {len(onchain_signals)}")
                for sig in onchain_signals[:2]:
                    print(f"        - [{sig.get('severity')}] {sig.get('message', '')[:50]}")

        except Exception as e:
            print(f"      ✗ Extended data error: {e}")

    # 1.6 상관관계 매트릭스 계산
    print("\n[1.6] Calculating correlation matrix...")
    correlation_matrix = []
    correlation_tickers = []
    try:
        if market_data:
            # 각 티커의 Close 가격을 하나의 DataFrame으로 합치기
            price_data = {}
            for ticker, df in market_data.items():
                if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
                    price_data[ticker] = df['Close']

            if price_data:
                # DataFrame으로 변환
                prices_df = pd.DataFrame(price_data)

                # 결측치 처리 (forward fill)
                prices_df = prices_df.fillna(method='ffill').dropna()

                # 상관관계 매트릭스 계산
                corr_df = prices_df.corr()
                correlation_matrix = corr_df.values.tolist()
                correlation_tickers = corr_df.columns.tolist()

                print(f"      ✓ Correlation matrix: {len(correlation_tickers)}x{len(correlation_tickers)}")

                # 가장 높은 상관관계 쌍 출력
                corr_values = []
                for i in range(len(correlation_tickers)):
                    for j in range(i+1, len(correlation_tickers)):
                        corr_values.append((correlation_tickers[i], correlation_tickers[j], corr_df.iloc[i, j]))

                if corr_values:
                    # 절대값 기준으로 정렬
                    corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
                    top_corr = corr_values[0]
                    print(f"      ✓ Strongest correlation: {top_corr[0]} ↔ {top_corr[1]}: {top_corr[2]:.3f}")
    except Exception as e:
        print(f"      ✗ Correlation calculation error: {e}")
        correlation_matrix = []
        correlation_tickers = []

    # ========================================================================
    # Phase 2: Analysis
    # ========================================================================
    print("\n" + "=" * 50)
    print("PHASE 2: ANALYSIS")
    print("=" * 50)

    # 2.1 레짐 탐지
    print("\n[2.1] Detecting market regime...")
    try:
        regime_detector = RegimeDetector(ticker='SPY')  # ticker는 생성자에서
        regime_result = regime_detector.detect()  # 인자 없이 호출
        result.regime = {
            'regime': regime_result.regime.value if hasattr(regime_result.regime, 'value') else str(regime_result.regime),
            'trend': regime_result.trend_state.value if hasattr(regime_result.trend_state, 'value') else str(regime_result.trend_state),
            'volatility': regime_result.volatility_state.value if hasattr(regime_result.volatility_state, 'value') else str(regime_result.volatility_state),
            'confidence': regime_result.confidence / 100 if regime_result.confidence > 1 else regime_result.confidence,
            'description': regime_result.description,
            'strategy': regime_result.strategy,
        }
        print(f"      ✓ Regime: {result.regime['regime']}")
        print(f"      ✓ Trend: {result.regime['trend']}, Volatility: {result.regime['volatility']}")
        print(f"      ✓ Confidence: {result.regime['confidence']:.0%}")
    except Exception as e:
        print(f"      ✗ Regime error: {e}")

    # 2.1.1 GMM & Entropy 기반 레짐 분석 (통계적 고도화)
    # 경제학적 근거: "GMM(Gaussian Mixture Model)을 써야 함", "엔트로피로 불확실성 측정"
    gmm_result = None
    if market_data and not quick_mode:
        print("\n[2.1.1] GMM & Entropy regime analysis...")
        try:
            gmm_summary = get_gmm_regime_summary(market_data)
            result.regime['gmm_regime'] = gmm_summary['regime']
            result.regime['gmm_probabilities'] = gmm_summary['probabilities']
            result.regime['entropy'] = gmm_summary['entropy']
            result.regime['entropy_level'] = gmm_summary['entropy_level']
            result.regime['entropy_interpretation'] = gmm_summary['interpretation']
            result.regime['gmm_report_line'] = gmm_summary['report_line']

            print(f"      ✓ GMM Regime: {gmm_summary['regime']}")
            probs = gmm_summary['probabilities']
            print(f"      ✓ Probabilities: Bull:{probs.get('Bull', 0):.0%} / Neutral:{probs.get('Neutral', 0):.0%} / Bear:{probs.get('Bear', 0):.0%}")
            print(f"      ✓ Shannon Entropy: {gmm_summary['entropy']:.3f} ({gmm_summary['entropy_level']})")
            print(f"      ✓ Interpretation: {gmm_summary['interpretation']}")
        except Exception as e:
            print(f"      △ GMM analysis (optional): {e}")

    # 2.2 이벤트 탐지
    print("\n[2.2] Detecting events...")
    try:
        event_detector = QuantitativeEventDetector()

        # 유동성 이벤트
        if fred_summary:
            liquidity_data = {
                'rrp': fred_summary.rrp,
                'rrp_delta': fred_summary.rrp_delta,
                'tga': fred_summary.tga,
                'tga_delta': fred_summary.tga_delta,
                'net_liquidity': fred_summary.net_liquidity,
            }
            liquidity_events = event_detector.detect_liquidity_events(liquidity_data)
            result.events_detected.extend([{
                'type': e.event_type.value,
                'importance': e.importance.value,
                'description': e.description
            } for e in liquidity_events])

            if liquidity_events:
                for e in liquidity_events:
                    print(f"      ⚠ {e.event_type.value}: {e.description}")
            else:
                print("      ✓ No liquidity events detected")

        # 시장 이벤트
        if fred_summary and market_data:
            market_events_data = {
                'vix': market_data.get('VIX', {}).get('Close', [0])[-1] if 'VIX' in market_data else 0,
                'spread_10y2y': fred_summary.spread_10y2y,
                'hy_oas': fred_summary.hy_oas,
            }
            # Additional market event detection can go here

    except Exception as e:
        print(f"      ✗ Event detection error: {e}")

    # 2.3 유동성 분석 (Granger Causality)
    if not quick_mode:
        print("\n[2.3] Liquidity-Market causality analysis...")
        try:
            liquidity_analyzer = LiquidityMarketAnalyzer()
            liquidity_signal = liquidity_analyzer.generate_signals()
            result.liquidity_signal = liquidity_signal.get('signal', 'NEUTRAL')
            print(f"      ✓ Liquidity Signal: {result.liquidity_signal}")
            if liquidity_signal.get('causality_results'):
                for var, strength in liquidity_signal['causality_results'].items():
                    print(f"        - {var}: {strength:.2f}")
        except Exception as e:
            print(f"      ✗ Liquidity analysis error: {e}")

    # 2.4 Critical Path 분석
    print("\n[2.4] Critical path analysis...")
    try:
        critical_path = CriticalPathAggregator()
        if market_data:
            cp_result = critical_path.analyze(market_data)
            # CriticalPathResult는 dataclass이므로 getattr 사용
            result.risk_score = getattr(cp_result, 'total_risk_score', 0)
            risk_level = getattr(cp_result, 'risk_level', 'Unknown')
            print(f"      ✓ Risk Score: {result.risk_score:.1f}/100")
            print(f"      ✓ Risk Level: {risk_level}")
            print(f"      ✓ Primary Risk Path: {getattr(cp_result, 'primary_risk_path', 'N/A')}")

            # 경고 추가
            if result.risk_score > 50:
                result.warnings.append(f"High risk score: {result.risk_score:.1f}")
    except Exception as e:
        print(f"      ✗ Critical path error: {e}")

    # 2.4.1 Market Microstructure Risk Enhancement (API 검증: Option C)
    if not quick_mode and market_data:
        print("\n[2.4.1] Microstructure risk enhancement...")
        try:
            micro_analyzer = DailyMicrostructureAnalyzer()
            micro_results = micro_analyzer.analyze_multiple(market_data)

            # MarketQualityMetrics 구성
            liquidity_scores = {}
            high_toxicity = []
            illiquid_tickers = []

            for ticker, micro_result in micro_results.items():
                # 유동성 점수 (0-100 스케일로 변환)
                liq_score = getattr(micro_result, 'overall_liquidity_score', 50)
                liquidity_scores[ticker] = liq_score

                # VPIN 기반 독성 체크
                vpin_result = getattr(micro_result, 'vpin', None)
                if vpin_result and hasattr(vpin_result, 'vpin') and vpin_result.vpin > 0.5:
                    high_toxicity.append(ticker)

                # 유동성 낮은 종목 (점수 30 이하)
                if liq_score < 30:
                    illiquid_tickers.append(ticker)

            avg_liq = sum(liquidity_scores.values()) / len(liquidity_scores) if liquidity_scores else 50

            result.market_quality = MarketQualityMetrics(
                avg_liquidity_score=avg_liq,
                liquidity_scores=liquidity_scores,
                high_toxicity_tickers=high_toxicity,
                illiquid_tickers=illiquid_tickers,
                data_quality="COMPLETE" if len(micro_results) == len(market_data) else "PARTIAL"
            )

            # 마이크로스트럭처 조정 계산 (±10 범위)
            # 평균 유동성이 낮으면 리스크 증가, 높으면 감소
            micro_adjustment = (50 - avg_liq) / 5  # 50점 기준, 10점 차이당 ±2
            micro_adjustment = max(-10, min(10, micro_adjustment))  # ±10 클램핑
            result.microstructure_adjustment = micro_adjustment

            print(f"      ✓ Avg Liquidity Score: {avg_liq:.1f}/100")
            print(f"      ✓ High Toxicity Tickers: {len(high_toxicity)} ({', '.join(high_toxicity[:3]) if high_toxicity else 'None'})")
            print(f"      ✓ Risk Adjustment: {micro_adjustment:+.1f}")

        except Exception as e:
            print(f"      ✗ Microstructure analysis error: {e}")
            result.market_quality = MarketQualityMetrics(data_quality="DEGRADED")

    # 2.4.2 Bubble Risk Overlay (Greenwood-Shleifer 기반)
    if not quick_mode:
        print("\n[2.4.2] Bubble risk overlay...")
        try:
            bubble_detector = BubbleDetector()
            tickers_to_check = list(market_data.keys()) if market_data else []

            # 버블 분석 실행
            bubble_results = {}
            for ticker in tickers_to_check:
                try:
                    df = market_data.get(ticker)
                    if df is not None and not df.empty:
                        bubble_results[ticker] = bubble_detector.analyze(ticker, df)
                except Exception as e:
                    logger.debug(f"Bubble analysis skipped for {ticker}: {e}")

            # BubbleRiskMetrics 구성
            risk_tickers = []
            highest_risk_ticker = ""
            highest_risk_score = 0.0
            overall_status = "NONE"

            level_priority = {
                BubbleWarningLevel.NONE: 0,
                BubbleWarningLevel.WATCH: 1,
                BubbleWarningLevel.WARNING: 2,
                BubbleWarningLevel.DANGER: 3
            }

            for ticker, bubble_result in bubble_results.items():
                level = bubble_result.bubble_warning_level
                score = bubble_result.risk_score

                if level != BubbleWarningLevel.NONE:
                    risk_tickers.append({
                        'ticker': ticker,
                        'level': level.value,
                        'runup_pct': round(bubble_result.runup.cumulative_return * 100, 1),
                        'vol_zscore': round(bubble_result.volatility.zscore, 2) if bubble_result.volatility else 0,
                        'risk_score': round(score, 1)
                    })

                if score > highest_risk_score:
                    highest_risk_score = score
                    highest_risk_ticker = ticker

                # 전체 상태 업데이트 (가장 높은 수준으로)
                if level_priority.get(level, 0) > level_priority.get(BubbleWarningLevel[overall_status], 0):
                    overall_status = level.value

            # 위험도순 정렬
            risk_tickers.sort(key=lambda x: x['risk_score'], reverse=True)

            result.bubble_risk = BubbleRiskMetrics(
                overall_status=overall_status,
                risk_tickers=risk_tickers[:5],  # Top 5만 저장
                highest_risk_ticker=highest_risk_ticker,
                highest_risk_score=highest_risk_score,
                methodology_notes="Greenwood-Shleifer 2019: Run-up + Volatility + Issuance"
            )

            # 버블 리스크 조정 계산 (multiplier 효과)
            # DANGER: +15, WARNING: +10, WATCH: +5
            bubble_adjustment = 0
            if overall_status == "DANGER":
                bubble_adjustment = 15
            elif overall_status == "WARNING":
                bubble_adjustment = 10
            elif overall_status == "WATCH":
                bubble_adjustment = 5
            result.bubble_risk_adjustment = bubble_adjustment

            # 최종 리스크 점수 업데이트 (Base + Micro + Bubble)
            result.base_risk_score = result.risk_score
            adjusted_risk = result.risk_score + result.microstructure_adjustment + bubble_adjustment
            result.risk_score = max(0, min(100, adjusted_risk))

            print(f"      ✓ Overall Bubble Status: {overall_status}")
            print(f"      ✓ Risk Tickers: {len(risk_tickers)} detected")
            if risk_tickers:
                top_risk = risk_tickers[0]
                print(f"      ✓ Highest Risk: {top_risk['ticker']} ({top_risk['level']}, {top_risk['runup_pct']:+.0f}% run-up)")
            print(f"      ✓ Bubble Adjustment: +{bubble_adjustment}")
            print(f"      ✓ Final Risk Score: {result.base_risk_score:.1f} → {result.risk_score:.1f}")

        except Exception as e:
            print(f"      ✗ Bubble detection error: {e}")
            result.bubble_risk = BubbleRiskMetrics(overall_status="ERROR")

    # 2.5 ETF Flow 분석
    if not quick_mode:
        print("\n[2.5] ETF flow analysis...")
        try:
            etf_analyzer = ETFFlowAnalyzer()
            etf_result = etf_analyzer.analyze()
            print(f"      ✓ Sector Rotation: {etf_result.get('rotation_signal', 'N/A')}")
            print(f"      ✓ Style: {etf_result.get('style_signal', 'N/A')}")
        except Exception as e:
            print(f"      ✗ ETF flow error: {e}")

    # 2.6 Genius Act Macro (스테이블코인-유동성 분석)
    if not quick_mode and fred_summary:
        print("\n[2.6] Genius Act Macro analysis...")
        try:
            from lib.genius_act_macro import StablecoinDataCollector

            # 스테이블코인 데이터 수집 (7일 델타 계산)
            stablecoin_collector = StablecoinDataCollector()
            stablecoin_data = stablecoin_collector.fetch_stablecoin_supply(lookback_days=14)
            stablecoin_comment = stablecoin_collector.generate_detailed_comment(stablecoin_data)

            # 스테이블코인 시가총액에서 공급량 추출 (십억 달러)
            usdt_current = stablecoin_data.get('USDT', {}).get('current', 140)
            usdc_current = stablecoin_data.get('USDC', {}).get('current', 45)
            dai_current = stablecoin_data.get('DAI', {}).get('current', 5)

            usdt_week_ago = stablecoin_data.get('USDT', {}).get('week_ago', 140)
            usdc_week_ago = stablecoin_data.get('USDC', {}).get('week_ago', 45)
            dai_week_ago = stablecoin_data.get('DAI', {}).get('week_ago', 5)

            # 현재 지표 구성 (실제 스테이블코인 데이터 사용)
            current_liq = LiquidityIndicators(
                fed_balance_sheet=fred_summary.fed_assets / 1000 if fred_summary.fed_assets else 7.5,
                rrp_balance=fred_summary.rrp / 1000 if fred_summary.rrp else 0.5,
                tga_balance=fred_summary.tga / 1000 if fred_summary.tga else 0.5,
                usdt_supply=usdt_current,
                usdc_supply=usdc_current,
                dai_supply=dai_current,
            )
            # 이전 지표 (7일 전 데이터 사용)
            previous_liq = LiquidityIndicators(
                fed_balance_sheet=current_liq.fed_balance_sheet,
                rrp_balance=current_liq.rrp_balance - (fred_summary.rrp_delta / 1000 if fred_summary.rrp_delta else 0),
                tga_balance=current_liq.tga_balance - (fred_summary.tga_delta / 1000 if fred_summary.tga_delta else 0),
                usdt_supply=usdt_week_ago,
                usdc_supply=usdc_week_ago,
                dai_supply=dai_week_ago,
            )

            genius_strategy = GeniusActMacroStrategy()
            genius_result = genius_strategy.analyze(current_liq, previous_liq)

            result.genius_act_regime = genius_result['regime']

            # 시그널을 딕셔너리로 변환하면서 Why(이유) 설명 포함
            signals_with_why = []
            for sig in genius_result['signals']:
                # 시그널이 객체인지 dict인지 확인 후 처리
                if hasattr(sig, 'signal_type'):
                    # MacroSignal 객체인 경우
                    sig_type = sig.signal_type.value
                    sig_desc = sig.description
                    sig_strength = sig.strength
                    sig_confidence = sig.confidence
                    sig_assets = sig.affected_assets
                    sig_metadata = sig.metadata if hasattr(sig, 'metadata') else {}
                elif isinstance(sig, dict):
                    # 이미 dict인 경우
                    sig_type = sig.get('type', sig.get('signal_type', 'unknown'))
                    sig_desc = sig.get('description', 'N/A')
                    sig_strength = sig.get('strength', 0)
                    sig_confidence = sig.get('confidence', 0)
                    sig_assets = sig.get('affected_assets', [])
                    sig_metadata = sig.get('metadata', {})
                else:
                    continue  # 알 수 없는 형식 스킵

                why_explanation = _get_genius_act_why(sig_type, sig_metadata)

                signals_with_why.append({
                    'type': sig_type,
                    'description': sig_desc,
                    'why': why_explanation,
                    'strength': sig_strength,
                    'confidence': sig_confidence,
                    'affected_assets': sig_assets,
                    'metadata': sig_metadata
                })

            # 스테이블코인 상세 코멘트를 시그널에 추가
            signals_with_why.append({
                'type': 'stablecoin_analysis',
                'description': stablecoin_comment['detailed_comment'],
                'why': stablecoin_comment['economic_interpretation'],
                'strength': abs(stablecoin_comment['total_delta_pct']) / 10,
                'confidence': 0.8,
                'affected_assets': ['BTC-USD', 'ETH-USD', 'TLT', 'SHY'],
                'metadata': {
                    'total_market_cap': stablecoin_comment['total_market_cap'],
                    'total_delta_7d': stablecoin_comment['total_delta_7d'],
                    'total_delta_pct': stablecoin_comment['total_delta_pct'],
                    'genius_act_status': stablecoin_comment['genius_act_status'],
                    'components': stablecoin_comment['components']
                }
            })
            result.genius_act_signals = signals_with_why

            print(f"      ✓ Regime: {genius_result['regime']}")
            print(f"      ✓ Signals: {len(genius_result['signals'])} detected")
            print(f"      ✓ Stablecoin: {stablecoin_comment['detailed_comment']}")
            print(f"        → Total Market Cap: ${stablecoin_comment['total_market_cap']:.1f}B")
            print(f"        → 7-Day Delta: ${stablecoin_comment['total_delta_7d']:+.1f}B ({stablecoin_comment['total_delta_pct']:+.1f}%)")
            print(f"        → Genius Act Status: {stablecoin_comment['genius_act_status'].upper()}")

            # 시그널별 Why 출력
            for sig in signals_with_why[:3]:
                print(f"        → {sig['type']}: {sig['description'][:80]}")
                print(f"          Why: {sig['why'][:100]}...")

            if genius_result['positions']:
                print(f"      ✓ Positions: {len(genius_result['positions'])} recommended")

        except Exception as e:
            print(f"      ✗ Genius Act error: {e}")

    # 2.6.1 Crypto Stress Test (v2.1.2 - Elicit Enhancement) - 독립 실행
    if not quick_mode and fred_summary:
        print("\n[2.6.1] Crypto Stress Test...")
        try:
            crypto_evaluator = CryptoRiskEvaluator()
            # 스테이블코인 시가총액 데이터 직접 수집
            from lib.genius_act_macro import StablecoinDataCollector
            stablecoin_collector = StablecoinDataCollector()
            stablecoin_data = stablecoin_collector.fetch_stablecoin_supply(lookback_days=7)

            # 스테이블코인 홀딩 추정 (시가총액 기준, 십억 달러 -> 달러)
            stablecoin_holdings = {
                'USDT': stablecoin_data.get('USDT', {}).get('current', 140) * 1e9,
                'USDC': stablecoin_data.get('USDC', {}).get('current', 45) * 1e9,
                'DAI': stablecoin_data.get('DAI', {}).get('current', 5) * 1e9,
            }
            stress_result = crypto_evaluator.run_stress_test(
                stablecoin_holdings=stablecoin_holdings,
                stress_scenario='moderate'
            )
            result.crypto_stress_test = stress_result
            print(f"      ✓ Scenario: {stress_result.get('scenario', 'N/A')}")
            print(f"      ✓ De-peg Probability: {stress_result.get('depeg_probability_pct', 'N/A')}")
            print(f"      ✓ Estimated Loss: ${stress_result.get('estimated_loss_under_stress', 0):,.0f}")
            print(f"      ✓ Risk Rating: {stress_result.get('risk_rating', 'N/A')}")
        except Exception as e:
            print(f"      ✗ Crypto Stress Test error: {e}")
            result.crypto_stress_test = {'error': str(e)}

    # 2.7 Theme ETF Analysis (Supply Chain Causality 포함)
    if not quick_mode:
        print("\n[2.7] Theme ETF analysis...")
        try:
            from lib.custom_etf_builder import SupplyChainGraph

            etf_builder = CustomETFBuilder()
            ai_etf = etf_builder.create_etf(ThemeCategory.AI_SEMICONDUCTOR)
            risk_analysis = etf_builder.analyze_risk_concentration(ai_etf)

            # Supply Chain 인과관계 분석
            supply_chain = SupplyChainGraph(ai_etf.stocks)
            layer_dist = supply_chain.get_layer_distribution()
            bottlenecks = supply_chain.find_bottlenecks()
            centrality = supply_chain.get_centrality_scores()

            # 시장 데이터에서 가격 변동률 추출 (인과관계 생성용)
            price_changes = {}
            for ticker, df in market_data.items() if market_data else []:
                if hasattr(df, 'pct_change') and len(df) > 1:
                    try:
                        price_changes[ticker] = float(df['Close'].pct_change().iloc[-1] * 100)
                    except:
                        pass

            # 동적 인과관계 체인 생성 (그래프 기반)
            causality_chains = supply_chain.generate_causality_chain(
                event='AI Demand Surge',  # 기본 이벤트
                source_node='NVDA',        # 주요 노드
                market_data=price_changes  # 실제 시장 데이터 반영
            )

            # LLM 기반 Narrative 생성 (Rule-based 모드)
            from lib.causality_narrative import CausalityNarrativeGenerator
            narrative_gen = CausalityNarrativeGenerator(use_llm=False)  # Rule-based 사용
            causality_insights = narrative_gen.generate_rule_based(
                bottlenecks=bottlenecks,
                hub_nodes=[t[0] for t in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]],
                supply_chain_layers=layer_dist,
                external_shock='AI Demand Surge',
                market_data=price_changes
            )

            # 인과관계 설명 포맷팅 (Narrative 형식)
            causality_explanation = []
            causality_explanation.append("=== Supply Chain Causality (인과관계) ===")

            # Narrative Insights 추가 (LLM/Rule-based 생성)
            causality_explanation.append("\n[Causality Insights (인과관계 분석)]")
            for insight in causality_insights[:3]:
                causality_explanation.append(f"\n**Path:** {insight.path}")
                causality_explanation.append(f"**Insight:** {insight.insight}")
                causality_explanation.append(f"(Causality: {insight.causality_type.upper()}, Confidence: {insight.confidence:.0%})")

            # 동적 생성된 인과관계 체인 추가
            causality_explanation.append("\n\n[인과관계 체인 (Event → Node → Impact)]")
            for chain in causality_chains[:5]:  # 최대 5개
                causality_explanation.append(f"• {chain}")

            # 레이어별 전파 경로 설명
            causality_explanation.append("\n[전파 경로] EQUIPMENT → MANUFACTURER → INTEGRATOR → END_USER")

            # 병목 지점 설명
            if bottlenecks:
                causality_explanation.append(f"\n[병목 지점] {', '.join(bottlenecks)}")
                causality_explanation.append("• 이 종목들이 타격 받으면 전체 공급망에 충격 전파")

            # 핵심 종목 (중심성 기준)
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_central:
                causality_explanation.append(f"\n[핵심 허브] {', '.join([t[0] for t in top_central])}")
                causality_explanation.append("• 공급망 네트워크에서 가장 중요한 위치 (PageRank + Betweenness)")

            # 충격 전파 시뮬레이션 (NVDA 예시)
            if 'NVDA' in [s.ticker for s in ai_etf.stocks]:
                propagation = supply_chain.get_shock_propagation_path('NVDA')
                if propagation:
                    prop_path = ' → '.join([p[0] for p in propagation[:4]])
                    causality_explanation.append(f"\n[충격 전파 예시] NVDA 하락 시: {prop_path}")

            # === CausalityGraphEngine 통합 (고급 그래프 분석) ===
            graph_engine_insights = []
            shock_simulation_results = {}
            try:
                # 그래프 엔진 초기화 (LLM 비활성화 - Rule-based 먼저)
                graph_engine = CausalityGraphEngine(use_llm=False)

                # 공급망 데이터로 그래프 구축
                graph_engine.build_from_supply_chain(
                    supply_chain_layers=layer_dist,
                    stock_info={s.ticker: {'name': s.name, 'sector': getattr(s, 'sector', '')}
                               for s in ai_etf.stocks}
                )

                # 시장 데이터로 상관관계/Granger 엣지 추가
                if market_data:
                    graph_engine.build_from_market_data(
                        market_data=market_data,
                        correlation_threshold=0.5,
                        granger_pvalue_threshold=0.05
                    )

                # 병목점 식별 (고급)
                graph_bottlenecks = graph_engine.identify_bottlenecks(top_n=5)

                # 충격 전파 시뮬레이션 (각 주요 노드에서)
                for bn in graph_bottlenecks[:3]:
                    impacts = graph_engine.simulate_shock_propagation(bn.id, shock_magnitude=-0.10)
                    shock_simulation_results[bn.id] = impacts

                # 인사이트 생성 (비동기 - 이미 실행중인 루프 처리)
                try:
                    loop = asyncio.get_running_loop()
                    # 이미 루프가 실행중이면 새 태스크로 처리
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        graph_engine_insights = pool.submit(
                            asyncio.run, graph_engine.generate_insights(max_insights=5)
                        ).result(timeout=30)
                except RuntimeError:
                    # 루프가 없으면 새로 생성
                    graph_engine_insights = asyncio.run(graph_engine.generate_insights(max_insights=5))

                # 자연어 Narrative 생성 (리포트용)
                graph_narrative = graph_engine.generate_report_narrative(
                    external_shock='AI Demand Surge',
                    include_shock_sim=True
                )

                print(f"      ✓ Graph Engine: {len(graph_engine.nodes)} nodes, {len(graph_engine.edges)} edges")
                print(f"      ✓ Advanced Bottlenecks: {', '.join([bn.id for bn in graph_bottlenecks])}")
                print(f"      ✓ Narrative Generated: {len(graph_narrative)} chars")

            except Exception as graph_err:
                print(f"      △ Graph Engine (optional): {graph_err}")
                graph_narrative = "Not enough correlation data to build causality chain yet."

            result.theme_etf_analysis = {
                'theme': 'AI_SEMICONDUCTOR',
                'description': ai_etf.description,
                'stocks_count': len(ai_etf.stocks),
                'top5_concentration': risk_analysis['top5_concentration'],
                'diversification_score': risk_analysis['diversification_score'],
                'warnings': risk_analysis['risk_warnings'],
                'supply_chain': {
                    'layers': {k: v for k, v in layer_dist.items()},
                    'bottlenecks': bottlenecks,
                    'top_central': [t[0] for t in top_central] if top_central else [],
                },
                'causality_explanation': '\n'.join(causality_explanation),
                'causality_insights': [i.to_dict() for i in causality_insights],  # LLM/Rule-based 인사이트
                # CausalityGraphEngine 결과 (고급 분석)
                'graph_engine_insights': [i.to_dict() for i in graph_engine_insights] if graph_engine_insights else [],
                'shock_simulation': shock_simulation_results,
                # 리포트용 자연어 Narrative
                'graph_narrative': graph_narrative
            }

            print(f"      ✓ Theme: {ai_etf.name}")
            print(f"      ✓ Stocks: {len(ai_etf.stocks)}, Diversification: {risk_analysis['diversification_score']}")
            print(f"      ✓ Bottlenecks: {', '.join(bottlenecks) if bottlenecks else 'None'}")
            print(f"      ✓ Hub Nodes: {', '.join([t[0] for t in top_central]) if top_central else 'N/A'}")
            print(f"      ✓ Causality Insights: {len(causality_insights)} generated")
            # 첫 번째 인사이트 출력
            if causality_insights:
                first = causality_insights[0]
                print(f"        → Path: {first.path[:60]}...")
                print(f"        → Insight: {first.insight[:80]}...")
        except Exception as e:
            print(f"      ✗ Theme ETF error: {e}")

    # 2.8 Shock Propagation Graph (인과관계 분석)
    if not quick_mode and market_data:
        print("\n[2.8] Shock propagation analysis...")
        try:
            # 수익률 데이터 준비
            import pandas as pd
            returns_dict = {}
            for ticker, df in market_data.items():
                if hasattr(df, 'pct_change') and len(df) > 20:
                    returns_dict[ticker] = df['Close'].pct_change().dropna()

            if len(returns_dict) >= 3:
                returns_df = pd.DataFrame(returns_dict).dropna()

                if len(returns_df) > 60:
                    shock_graph = ShockPropagationGraph()
                    shock_graph.build_from_returns(returns_df)

                    # find_critical_path() 는 source를 필요로 하므로, 주요 노드에서 경로 탐색
                    # 노드가 많으면 첫 번째 노드를 source로 사용
                    source_node = list(shock_graph.graph.nodes())[0] if shock_graph.graph.nodes() else None
                    if source_node:
                        critical_path = shock_graph.find_critical_path(source=source_node)

                        if critical_path:
                            # ShockPath 객체는 데이터클래스이므로 속성 직접 접근
                            result.shock_propagation = {
                                'nodes': len(shock_graph.graph.nodes()),
                                'edges': len(shock_graph.graph.edges()),
                                'critical_path': critical_path.path,
                                'total_lag': critical_path.total_lag
                            }

                            print(f"      ✓ Nodes: {result.shock_propagation['nodes']}, Edges: {result.shock_propagation['edges']}")
                            if critical_path.path:
                                path_str = ' → '.join(critical_path.path[:5])
                                print(f"      ✓ Critical Path: {path_str}")
                    else:
                        print(f"      △ No nodes in shock graph")
        except Exception as e:
            print(f"      ✗ Shock propagation error: {e}")

    # 2.9 Graph-Clustered Portfolio (GC-HRP)
    if not quick_mode and market_data:
        print("\n[2.9] Graph-Clustered Portfolio optimization...")
        try:
            import pandas as pd
            # 수익률 데이터 준비
            returns_dict = {}
            for ticker, df in market_data.items():
                if hasattr(df, 'pct_change') and len(df) > 20:
                    returns_dict[ticker] = df['Close'].pct_change().dropna()

            if len(returns_dict) >= 3:
                returns_df = pd.DataFrame(returns_dict).dropna()

                if len(returns_df) > 60:
                    gc_portfolio = GraphClusteredPortfolio(
                        correlation_threshold=0.3,
                        clustering_method=ClusteringMethod.KMEANS,
                        max_representatives_per_cluster=2
                    )
                    allocation = gc_portfolio.fit(returns_df)

                    result.portfolio_weights = allocation.weights
                    print(f"      ✓ Clusters: {len(allocation.clusters)}")
                    print(f"      ✓ Diversification Ratio: {allocation.diversification_ratio:.2f}")
                    print(f"      ✓ Effective N: {allocation.effective_n:.1f}")

                    # Top 5 weights 출력
                    sorted_weights = sorted(allocation.weights.items(), key=lambda x: x[1], reverse=True)[:5]
                    weights_str = ', '.join([f"{t}:{w:.1%}" for t, w in sorted_weights])
                    print(f"      ✓ Top Weights: {weights_str}")

                    # HRP Allocation Rationale (v2.1.2 - Elicit Enhancement)
                    result.hrp_allocation_rationale = _generate_hrp_rationale(
                        allocation.weights, returns_df, allocation.clusters
                    )
                    print(f"      ✓ Rationale: {result.hrp_allocation_rationale[:80]}...")
        except Exception as e:
            print(f"      ✗ GC-HRP error: {e}")

    # 2.10 Integrated Strategy (Portfolio + Causality)
    if not quick_mode and market_data and fred_summary:
        print("\n[2.10] Integrated Strategy analysis...")
        try:
            import pandas as pd
            # 수익률 데이터 준비
            returns_dict = {}
            for ticker, df in market_data.items():
                if hasattr(df, 'pct_change') and len(df) > 20:
                    returns_dict[ticker] = df['Close'].pct_change().dropna()

            if len(returns_dict) >= 3:
                returns_df = pd.DataFrame(returns_dict).dropna()

                # 매크로 데이터 구성
                macro_df = pd.DataFrame({
                    'FED_FUNDS': [fred_summary.fed_funds] * len(returns_df),
                    'VIX': [indicators_summary.vix.current if 'indicators_summary' in dir() else 15.0] * len(returns_df),
                }, index=returns_df.index)

                if len(returns_df) > 60:
                    integrated = IntegratedStrategy(
                        correlation_threshold=0.3,
                        clustering_method=ClusteringMethod.KMEANS,
                        leading_tilt_factor=0.15,
                        volume_surge_threshold=3.0
                    )

                    recommendation = integrated.fit(returns_df, macro_df)

                    # 시그널 저장
                    result.integrated_signals = [
                        {
                            'type': s.signal_type.value,
                            'source': s.source,
                            'urgency': s.urgency,
                            'description': s.description[:100],
                            'action': s.action_suggested.value
                        }
                        for s in recommendation.signals[:10]
                    ]

                    print(f"      ✓ Signals: {len(recommendation.signals)} generated")
                    print(f"      ✓ Leading Exposure: {recommendation.leading_exposure:.1%}")
                    print(f"      ✓ Shock Vulnerability: {recommendation.shock_vulnerability:.1%}")

                    if recommendation.warnings:
                        for w in recommendation.warnings[:2]:
                            print(f"      ⚠ {w[:80]}")
        except Exception as e:
            print(f"      ✗ Integrated Strategy error: {e}")

    # 2.11 Volume Anomaly Detection (정보 비대칭 탐지)
    if not quick_mode and market_data:
        print("\n[2.11] Volume anomaly detection...")
        try:
            # 민감도 조정: 2.5배(MEDIUM), 4.0배(HIGH) - 더 많은 이상 탐지
            # 소스 이론: "20일 이평선 대비 3~5배 거래량은 사적 정보 유입" (Kyle, 1985)
            volume_analyzer = VolumeAnalyzer(
                lookback_period=20,
                surge_threshold_medium=2.5,  # 2.5배 = MEDIUM (민감도 상향)
                surge_threshold_high=4.0,    # 4.0배 = HIGH (민감도 상향)
                verbose=False
            )
            vol_result = volume_analyzer.detect_anomalies(market_data)

            # 결과 저장
            result.volume_anomalies = [a.to_dict() for a in vol_result.anomalies[:10]]
            result.volume_analysis_summary = vol_result.summary

            print(f"      ✓ Analyzed: {vol_result.total_tickers_analyzed} tickers")
            print(f"      ✓ Anomalies: {vol_result.anomalies_detected} detected")
            print(f"      ✓ High severity: {vol_result.high_severity_count}")

            # 명시적 메시지: 이상이 없을 때도 상태 출력
            if vol_result.anomalies_detected == 0:
                print(f"      ✓ Volume profile is normal (No asymmetric info detected)")
                print(f"        → All tickers within normal range (<2.5x MA20)")
                print(f"        → Kyle(1985): No evidence of Private Information Inflow")

            # 고심각도 경고를 Events로 추가 (Private Information Inflow)
            high_severity = volume_analyzer.filter_by_severity(vol_result, "HIGH")
            for anomaly in high_severity[:5]:
                print(f"      ⚠ {anomaly.alert_message}")
                result.warnings.append(f"Volume Alert: {anomaly.ticker} - {anomaly.volume_ratio:.1f}x surge detected")

                # Events Detected에 추가
                event_desc = (
                    f"Private Information Inflow Detected: {anomaly.ticker} "
                    f"({anomaly.volume_ratio:.1f}x avg volume, price {anomaly.price_change_1d:+.1f}%). "
                    f"Kyle(1985): 거래량 급증은 사적 정보 유입 신호."
                )
                result.events_detected.append({
                    'type': 'VOLUME_ANOMALY',
                    'importance': anomaly.severity,
                    'description': event_desc,
                    'ticker': anomaly.ticker,
                    'volume_ratio': anomaly.volume_ratio,
                    'info_type': anomaly.information_type.value if hasattr(anomaly.information_type, 'value') else str(anomaly.information_type)
                })

            # MEDIUM severity도 이벤트로 추가 (정보용)
            medium_severity = [a for a in vol_result.anomalies if a.severity == "MEDIUM"]
            for anomaly in medium_severity[:3]:
                event_desc = (
                    f"Abnormal Volume: {anomaly.ticker} "
                    f"({anomaly.volume_ratio:.1f}x avg volume). "
                    f"잠재적 정보 비대칭 가능성."
                )
                result.events_detected.append({
                    'type': 'VOLUME_ANOMALY',
                    'importance': 'MEDIUM',
                    'description': event_desc,
                    'ticker': anomaly.ticker,
                    'volume_ratio': anomaly.volume_ratio
                })

            # Top Movers 표시 (강제 감지 - 이상이 없어도 상위 종목 표시)
            if vol_result.top_movers:
                print(f"      ✓ Top Movers: {vol_result.top_movers_summary}")
                for mover in vol_result.top_movers[:3]:
                    print(f"        → {mover.ticker}: {mover.volume_ratio:.2f}x (price {mover.price_change_1d:+.1f}%)")

                # 이상이 없을 때 Top Movers를 Events에 추가 (Debug용)
                if vol_result.anomalies_detected == 0:
                    result.events_detected.append({
                        'type': 'TOP_MOVERS_DEBUG',
                        'importance': 'INFO',
                        'description': vol_result.top_movers_summary,
                        'top_movers': [m.to_dict() for m in vol_result.top_movers]
                    })

            if vol_result.warnings:
                for w in vol_result.warnings[:2]:
                    print(f"      ⚠ {w[:80]}")

        except Exception as e:
            print(f"      ✗ Volume analysis error: {e}")

    # 2.12 Event Tracking (거래량 이상 → 뉴스 역추적)
    if not quick_mode:
        print("\n[2.12] Event tracking (anomaly → news)...")
        try:
            event_tracker = EventTracker(use_perplexity=True)

            # 주요 티커에 대해 이상 탐지 및 뉴스 검색
            tracking_tickers = list(market_data.keys())[:10]  # 상위 10개
            tracking_result = await event_tracker.track_anomaly_events(
                tickers=tracking_tickers,
                days=14,
                max_events=5
            )

            result.event_tracking = {
                'anomalies_found': tracking_result.anomalies_found,
                'events_matched': tracking_result.events_matched,
                'event_types': tracking_result.event_type_distribution
            }
            result.tracked_events = [e.to_dict() for e in tracking_result.tracked_events]

            print(f"      ✓ Anomalies: {tracking_result.anomalies_found}")
            print(f"      ✓ Events Matched: {tracking_result.events_matched}")

            if tracking_result.tracked_events:
                for e in tracking_result.tracked_events[:3]:
                    if e.news_found:
                        print(f"        → [{e.ticker}] {e.timestamp}: {e.event_type} ({e.sentiment})")
                        print(f"          {e.news_summary[:60]}...")

        except Exception as e:
            print(f"      ✗ Event tracking error: {e}")

    # ========================================================================
    # Phase 3: Multi-Agent Debate
    # ========================================================================
    print("\n" + "=" * 50)
    print("PHASE 3: MULTI-AGENT DEBATE")
    print("=" * 50)

    # 3.1 FULL Mode (Historical 365일)
    print("\n[3.1] Running FULL mode analysis...")
    try:
        orchestrator_full = MetaOrchestrator(verbose=False)
        query = "Analyze current market conditions, risks, and generate trading signals"
        result_full = await orchestrator_full.run_with_debate(query, market_data)

        # consensus는 토픽별 딕셔너리 → 종합 계산 필요
        consensus_by_topic = result_full.get('debate', {}).get('consensus', {})
        analysis_data = result_full.get('analysis', {})

        if consensus_by_topic:
            # 모든 토픽의 confidence 평균 계산
            confidences = [c.get('confidence', 0.5) for c in consensus_by_topic.values()]
            full_confidence = sum(confidences) / len(confidences) if confidences else 0.5

            # 포지션 결정: analysis의 regime + risk 기반
            regime = analysis_data.get('current_regime', 'NEUTRAL')
            risk_score = analysis_data.get('total_risk_score', 50)
            regime_conf = analysis_data.get('regime_confidence', 50) / 100

            if regime == 'BULL' and risk_score < 30:
                result.full_mode_position = 'BULLISH'
                full_confidence = max(full_confidence, regime_conf)
            elif regime == 'BEAR' or risk_score > 70:
                result.full_mode_position = 'BEARISH'
                full_confidence = max(full_confidence, regime_conf)
            else:
                result.full_mode_position = 'NEUTRAL'

            full_dissent = []
            full_strong_dissent = False
        else:
            result.full_mode_position = 'NEUTRAL'
            full_confidence = 0.5
            full_dissent = []
            full_strong_dissent = False

        print(f"      ✓ FULL Position: {result.full_mode_position}")
        print(f"      ✓ Confidence: {full_confidence:.0%}")
        print(f"      ✓ Regime: {analysis_data.get('current_regime', 'N/A')}, Risk: {analysis_data.get('total_risk_score', 0):.1f}")
        if full_dissent:
            print(f"      ⚠ Dissent Records: {len(full_dissent)}")
    except Exception as e:
        print(f"      ✗ FULL mode error: {e}")
        full_confidence = 0.5
        full_dissent = []
        full_strong_dissent = False

    # 3.2 REFERENCE Mode (Recent 90일)
    print("\n[3.2] Running REFERENCE mode analysis...")
    try:
        dm_ref = DataManager(lookback_days=90)
        market_data_ref, _ = dm_ref.collect_all(tickers_config)

        orchestrator_ref = MetaOrchestrator(verbose=False)
        result_ref = await orchestrator_ref.run_with_debate(query, market_data_ref)

        # consensus는 토픽별 딕셔너리 → 종합 계산 필요
        consensus_by_topic_ref = result_ref.get('debate', {}).get('consensus', {})
        analysis_data_ref = result_ref.get('analysis', {})

        if consensus_by_topic_ref:
            # 모든 토픽의 confidence 평균 계산
            confidences_ref = [c.get('confidence', 0.5) for c in consensus_by_topic_ref.values()]
            ref_confidence = sum(confidences_ref) / len(confidences_ref) if confidences_ref else 0.5

            # 포지션 결정: analysis의 regime + risk 기반
            regime_ref = analysis_data_ref.get('current_regime', 'NEUTRAL')
            risk_score_ref = analysis_data_ref.get('total_risk_score', 50)
            regime_conf_ref = analysis_data_ref.get('regime_confidence', 50) / 100

            if regime_ref == 'BULL' and risk_score_ref < 30:
                result.reference_mode_position = 'BULLISH'
                ref_confidence = max(ref_confidence, regime_conf_ref)
            elif regime_ref == 'BEAR' or risk_score_ref > 70:
                result.reference_mode_position = 'BEARISH'
                ref_confidence = max(ref_confidence, regime_conf_ref)
            else:
                result.reference_mode_position = 'NEUTRAL'

            ref_dissent = []
            ref_strong_dissent = False
        else:
            result.reference_mode_position = 'NEUTRAL'
            ref_confidence = 0.5
            ref_dissent = []
            ref_strong_dissent = False

        print(f"      ✓ REFERENCE Position: {result.reference_mode_position}")
        print(f"      ✓ Confidence: {ref_confidence:.0%}")
        print(f"      ✓ Regime: {analysis_data_ref.get('current_regime', 'N/A')}, Risk: {analysis_data_ref.get('total_risk_score', 0):.1f}")
        if ref_dissent:
            print(f"      ⚠ Dissent Records: {len(ref_dissent)}")
    except Exception as e:
        print(f"      ✗ REFERENCE mode error: {e}")
        ref_confidence = 0.5
        ref_dissent = []
        ref_strong_dissent = False

    # 3.3 모드 비교
    print("\n[3.3] Comparing modes...")
    result.modes_agree = (result.full_mode_position == result.reference_mode_position)
    result.dissent_records = full_dissent + ref_dissent
    result.has_strong_dissent = full_strong_dissent or ref_strong_dissent

    # Devil's Advocate 논거 추출 (v2.1.2 - Elicit Enhancement)
    result.devils_advocate_arguments = _extract_devils_advocate_arguments(result.dissent_records)
    if result.devils_advocate_arguments:
        print(f"      ✓ Devil's Advocate: {len(result.devils_advocate_arguments)} counter-arguments extracted")

    # Dual Mode 분석기로 최종 권고
    analyzer = DualModeAnalyzer()
    full_mode = ModeResult(
        mode=AnalysisMode.FULL,
        consensus=None,
        confidence=full_confidence,
        position=result.full_mode_position,
        dissent_count=len(full_dissent),
        has_strong_dissent=full_strong_dissent
    )
    ref_mode = ModeResult(
        mode=AnalysisMode.REFERENCE,
        consensus=None,
        confidence=ref_confidence,
        position=result.reference_mode_position,
        dissent_count=len(ref_dissent),
        has_strong_dissent=ref_strong_dissent
    )
    comparison = analyzer.compare_modes(full_mode, ref_mode)

    result.final_recommendation = comparison.recommended_action
    result.confidence = (full_confidence + ref_confidence) / 2
    result.risk_level = comparison.risk_level

    if not result.modes_agree:
        result.warnings.append(f"Mode divergence: FULL={result.full_mode_position}, REF={result.reference_mode_position}")
    if result.has_strong_dissent:
        result.warnings.append("Strong dissent exists - review carefully")

    print(f"      ✓ Modes Agree: {'Yes' if result.modes_agree else 'NO'}")
    print(f"      ✓ Final Recommendation: {result.final_recommendation}")
    print(f"      ✓ Risk Level: {result.risk_level}")

    # 3.4 Adaptive Portfolio Agents + Validation Loop
    if not quick_mode:
        print("\n[3.4] Adaptive portfolio agents...")
        try:
            # 시장 상황 구성
            vix_level = indicators_summary.vix.current if 'indicators_summary' in dir() else 20.0
            market_condition = MarketCondition(
                regime=result.regime.get('regime', 'NEUTRAL'),
                risk_score=result.risk_score,
                vix_level=vix_level,
                liquidity_signal=result.liquidity_signal,
                vpin_alert=any(sig.get('alert_level') == 'HIGH' for sig in result.realtime_signals) if result.realtime_signals else False,
                bubble_status=result.bubble_risk.overall_status if result.bubble_risk else 'NONE'
            )

            print(f"      Market: {market_condition.regime}, Risk: {market_condition.risk_score:.1f}")
            print(f"      Urgency: {market_condition.urgency_score():.1f}, Opportunity: {market_condition.opportunity_score():.1f}")

            # 3개 에이전트 실행
            agent_manager = AdaptiveAgentManager()
            portfolios = await agent_manager.run_all_agents(market_condition, list(market_data.keys())[:15])

            result.adaptive_portfolios = {
                name: {
                    'risk_level': p.adjusted_risk_level,
                    'action': p.action,
                    'allocations': p.allocations,
                    'rationale': p.rationale
                }
                for name, p in portfolios.items()
            }

            for name, p in portfolios.items():
                print(f"      → {name}: {p.action} (risk={p.adjusted_risk_level})")

            # Validation Loop (Aggressive 에이전트 결정 검증)
            print("\n[3.4.1] Validation loop (Claude + Perplexity)...")
            aggressive_decision = {
                'agent_type': 'aggressive',
                'action': portfolios['aggressive'].action,
                'risk_level': portfolios['aggressive'].adjusted_risk_level,
                'rationale': portfolios['aggressive'].rationale,
                'allocations': portfolios['aggressive'].allocations
            }

            loop_manager = ValidationLoopManager(max_rounds=2)
            loop_result = await loop_manager.run_validation_loop(
                original_decision=aggressive_decision,
                market_condition={
                    'regime': market_condition.regime,
                    'risk_score': market_condition.risk_score,
                    'vix_level': market_condition.vix_level,
                    'volatility': result.regime.get('volatility', 'Medium'),
                    'liquidity_signal': market_condition.liquidity_signal,
                    'vpin_alert': market_condition.vpin_alert
                }
            )

            result.validation_loop_result = {
                'rounds_completed': loop_result.rounds_completed,
                'original_risk': aggressive_decision['risk_level'],
                'final_risk': loop_result.final_decision.get('risk_level'),
                'modifications': len(loop_result.modification_history),
                'consensus_reached': loop_result.consensus_reached,
                'summary': loop_result.summary
            }

            print(f"      ✓ Validation: {loop_result.rounds_completed} rounds")
            print(f"      ✓ Risk: {aggressive_decision['risk_level']} → {loop_result.final_decision.get('risk_level')}")
            print(f"      ✓ Consensus: {'Yes' if loop_result.consensus_reached else 'No'}")

        except Exception as e:
            print(f"      ✗ Adaptive agents error: {e}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # Phase 4: Real-time VPIN Monitoring (Optional)
    # ========================================================================
    if enable_realtime:
        print("\n" + "=" * 50)
        print("PHASE 4: REAL-TIME VPIN MONITORING")
        print("=" * 50)

        print(f"\n[4.1] Starting Real-time VPIN Monitor ({realtime_duration}s)...")
        print("      Symbols: BTCUSDT, ETHUSDT")
        print("      VPIN Thresholds: 0.5 (elevated), 0.6 (high), 0.7 (extreme)")

        try:
            # 새로운 RealtimeVPINMonitor 사용
            monitor_result = await run_realtime_monitor(
                symbols=['BTCUSDT', 'ETHUSDT'],
                duration=realtime_duration,
                verbose=True
            )

            # 결과 저장
            vpin_history = monitor_result.get('vpin_history', {})
            all_signals = []

            for symbol, history in vpin_history.items():
                for h in history:
                    all_signals.append({
                        'timestamp': h['timestamp'],
                        'symbol': symbol,
                        'avg_vpin': h['avg_vpin'],
                        'max_vpin': h['max_vpin'],
                        'samples': h['samples']
                    })

            result.realtime_signals = all_signals[-20:]  # 마지막 20개

            # 요약 출력
            stream_stats = monitor_result.get('stream_stats', {})
            alerts = monitor_result.get('alerts_fired', 0)

            print(f"\n[4.2] Real-time Summary:")
            print(f"      ✓ Alerts Fired: {alerts}")
            print(f"      ✓ 1-min VPIN Samples: {len(all_signals)}")
            print(f"      ✓ Messages Processed: {stream_stats.get('messages_received', 0):,}")

        except Exception as e:
            print(f"      ✗ Real-time error: {e}")
            import traceback
            traceback.print_exc()

    # Correlation 데이터 저장
    result.correlation_matrix = correlation_matrix
    result.correlation_tickers = correlation_tickers

    # ========================================================================
    # Phase 5: Database Storage
    # ========================================================================
    print("\n" + "=" * 50)
    print("PHASE 5: DATABASE STORAGE")
    print("=" * 50)

    # 5.1 이벤트 DB 저장
    print("\n[5.1] Saving to Event Database...")
    try:
        event_db = EventDatabase('data/events.db')

        # 이벤트 저장
        for event in result.events_detected:
            event_db.save_detected_event({
                'event_type': event['type'],
                'importance': event['importance'],
                'description': event['description'],
                'timestamp': result.timestamp,
            })

        # 마켓 스냅샷 저장 (snapshot_id 필수)
        import uuid
        snapshot_id = str(uuid.uuid4())[:8]

        def get_latest_price(ticker: str) -> float:
            """DataFrame에서 최신 가격 추출"""
            if ticker not in market_data:
                return 0.0
            df = market_data[ticker]
            if hasattr(df, 'iloc') and len(df) > 0:
                return float(df['Close'].iloc[-1]) if 'Close' in df.columns else 0.0
            return 0.0

        event_db.save_market_snapshot({
            'snapshot_id': snapshot_id,
            'timestamp': result.timestamp,
            'spy_price': get_latest_price('SPY'),
            'spy_change_1d': 0.0,
            'spy_change_5d': 0.0,
            'spy_vs_ma20': 0.0,
            'qqq_price': get_latest_price('QQQ'),
            'iwm_price': get_latest_price('IWM'),
            'tlt_price': get_latest_price('TLT'),
            'gld_price': get_latest_price('GLD'),
            'vix_level': get_latest_price('^VIX'),
            'vix_percentile': 50.0,
            'rsi_14': 50.0,
            'macd_signal': 'neutral',
            'trend': result.regime.get('trend', 'unknown'),
            'volatility_regime': result.regime.get('volatility', 'normal'),
            'put_call_ratio': 1.0,
            'fear_greed_index': 50.0,
            'days_to_fomc': 0,
            'days_to_cpi': 0,
            'days_to_nfp': 0,
        })

        print(f"      ✓ Saved {len(result.events_detected)} events")
        print(f"      ✓ Saved market snapshot (ID: {snapshot_id})")
    except Exception as e:
        print(f"      ✗ Event DB error: {e}")

    # 5.2 시그널 DB 저장
    print("\n[5.2] Saving to Signal Database...")
    try:
        signal_db = SignalDatabase('outputs/realtime_signals.db')

        from lib.realtime_pipeline import IntegratedSignal
        integrated_signal = IntegratedSignal(
            timestamp=datetime.now(),
            symbol='INTEGRATED',
            liquidity_regime=result.fred_summary.get('liquidity_regime', 'Unknown'),
            rrp_delta=result.fred_summary.get('rrp_delta', 0),
            tga_delta=result.fred_summary.get('tga_delta', 0),
            net_liquidity=result.fred_summary.get('net_liquidity', 0),
            macro_signal=result.liquidity_signal.lower(),
            ofi=0,
            vpin=0,
            depth_ratio=1.0,
            micro_signal='neutral',
            combined_signal=result.final_recommendation.lower(),
            confidence=result.confidence,
            action=result.final_recommendation.lower(),
            alerts=result.warnings
        )
        signal_db.save_signal(integrated_signal)
        print(f"      ✓ Saved integrated signal")
    except Exception as e:
        print(f"      ✗ Signal DB error: {e}")

    # 5.2.1 예측 DB 저장 (검증용)
    print("\n[5.2.1] Saving to Predictions Database...")
    try:
        saved_ids = save_eimas_result(result)
        print(f"      ✓ Saved predictions: {list(saved_ids.keys())}")
    except Exception as e:
        print(f"      ✗ Predictions DB error: {e}")

    # 5.3 결과 JSON 저장
    if not cron_mode:
        print("\n[5.3] Saving results...")

    # 출력 디렉토리 설정 (파라미터 또는 기본값)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir) if os.path.isabs(output_dir) else Path(__file__).parent / output_dir
    output_dir.mkdir(exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON 저장
    output_file = output_dir / f"integrated_{timestamp_str}.json"
    with open(output_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    print(f"      - JSON: {output_file}")

    # Markdown 저장
    md_file = output_dir / f"integrated_{timestamp_str}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(result.to_markdown())
    print(f"      - MD: {md_file}")

    return result, market_data, output_file, md_file


async def run_ai_report(
    analysis_result,
    market_data: Dict,
    output_file: str
) -> Optional[str]:
    """
    Phase 6: AI Report Generation
    """
    print("\n" + "=" * 50)
    print("PHASE 6: AI REPORT GENERATION")
    print("=" * 50)

    try:
        from lib.ai_report_generator import AIReportGenerator

        generator = AIReportGenerator(verbose=True)

        print("\n[6.1] Generating AI-powered investment report...")
        report = await generator.generate(analysis_result.to_dict(), market_data)

        print("\n[6.2] Saving report...")
        report_path = await generator.save_report(report)

        print(f"\n      ✓ AI Report saved: {report_path}")

        # 요약 출력
        print("\n" + "-" * 50)
        print("📝 AI REPORT HIGHLIGHTS")
        print("-" * 50)

        if report.notable_stocks:
            print("\n주목할 종목:")
            for stock in report.notable_stocks[:3]:
                print(f"  • {stock.ticker}: {stock.notable_reason}")

        print(f"\n최종 제안: {report.final_recommendation[:200]}...")

        if report.action_items:
            print("\n액션 아이템:")
            for item in report.action_items[:3]:
                print(f"  • {item}")

        return report_path

    except Exception as e:
        print(f"      ✗ AI Report error: {e}")
        return None


async def run_full_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    generate_report: bool = False,
    target_ticker: str = None,
    output_dir: str = 'outputs',
    cron_mode: bool = False
):
    """
    전체 파이프라인 실행 (분석 + 리포트)

    Args:
        enable_realtime: 실시간 스트리밍 활성화
        realtime_duration: 스트리밍 지속 시간 (초)
        quick_mode: 빠른 분석 모드
        generate_report: AI 리포트 생성
        target_ticker: 특정 티커 중심 분석
        output_dir: 출력 디렉토리
        cron_mode: 서버 자동화 모드 (시각화 없음)
    """
    start_time = datetime.now()

    # Phase 1-5: 분석 실행
    result, market_data, output_file, md_file = await run_integrated_pipeline(
        enable_realtime=enable_realtime,
        realtime_duration=realtime_duration,
        quick_mode=quick_mode,
        output_dir=output_dir,
        cron_mode=cron_mode
    )

    # Phase 6: AI 리포트 생성 (옵션)
    report_path = None
    if generate_report:
        report_path = await run_ai_report(result, market_data, output_file)

    # Phase 7: Whitening & Fact Check (옵션)
    if generate_report and not quick_mode:
        print("\n" + "=" * 50)
        print("PHASE 7: WHITENING & FACT CHECK")
        print("=" * 50)

        # 7.1 Whitening Engine - 결과 경제학적 해석
        print("\n[7.1] Economic whitening analysis...")
        try:
            whitening = WhiteningEngine()

            # 포트폴리오 결과 구성
            portfolio_result = {
                'allocation': result.portfolio_weights if result.portfolio_weights else {'SPY': 0.3, 'QQQ': 0.2, 'TLT': 0.15},
                'changes': {},
                'returns': {}
            }

            explanation = whitening.explain_allocation(portfolio_result)
            result.whitening_summary = explanation.summary

            print(f"      ✓ Summary: {explanation.summary[:100]}...")
            print(f"      ✓ Key Drivers: {len(explanation.key_drivers)}")
            print(f"      ✓ Confidence: {explanation.overall_confidence:.0%}")
        except Exception as e:
            print(f"      ✗ Whitening error: {e}")

        # 7.2 Autonomous Fact Checker
        print("\n[7.2] Fact checking AI outputs...")
        try:
            fact_checker = AutonomousFactChecker(use_perplexity=False, verbose=False)

            # 검증할 텍스트 구성
            check_text = f"""
            Current regime is {result.regime.get('regime', 'Unknown')}.
            Risk score is {result.risk_score:.1f} out of 100.
            Net liquidity is {result.fred_summary.get('net_liquidity', 0):.0f} billion dollars.
            The recommendation is {result.final_recommendation} with {result.confidence:.0%} confidence.
            """

            check_result = await fact_checker.verify_document(check_text, max_claims=5)
            result.fact_check_grade = check_result['summary']['grade']

            print(f"      ✓ Claims checked: {check_result['summary']['total_claims']}")
            print(f"      ✓ Verified: {check_result['summary']['verified']}")
            print(f"      ✓ Grade: {check_result['summary']['grade']} ({check_result['summary']['grade_description']})")
        except Exception as e:
            print(f"      ✗ Fact check error: {e}")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("EIMAS INTEGRATED ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print()
    print("📊 DATA SUMMARY")
    print(f"   FRED: RRP=${result.fred_summary.get('rrp', 0):.0f}B, Net Liq=${result.fred_summary.get('net_liquidity', 0):.0f}B")
    print(f"   Market: {result.market_data_count} tickers, Crypto: {result.crypto_data_count} tickers")
    print()
    print("📈 ANALYSIS SUMMARY")
    print(f"   Regime: {result.regime.get('regime', 'Unknown')}")
    print(f"   Risk Score: {result.risk_score:.1f}/100")
    print(f"   Events: {len(result.events_detected)} detected")
    print()
    print("🤖 AGENT DEBATE")
    print(f"   FULL Mode: {result.full_mode_position}")
    print(f"   REFERENCE Mode: {result.reference_mode_position}")
    print(f"   Modes Agree: {'✓' if result.modes_agree else '✗'}")
    print(f"   Dissent Records: {len(result.dissent_records)}")
    print()
    print("🔬 ADVANCED ANALYSIS")
    print(f"   Genius Act Regime: {result.genius_act_regime}")
    print(f"   Genius Act Signals: {len(result.genius_act_signals)}")
    if result.shock_propagation:
        print(f"   Shock Graph: {result.shock_propagation.get('nodes', 0)} nodes, {result.shock_propagation.get('edges', 0)} edges")
    if result.theme_etf_analysis:
        print(f"   Theme ETF: {result.theme_etf_analysis.get('theme', 'N/A')}")
    if result.portfolio_weights:
        top_3 = sorted(result.portfolio_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        weights_str = ', '.join([f"{t}:{w:.1%}" for t, w in top_3])
        print(f"   GC-HRP Portfolio: {weights_str}")
    if result.integrated_signals:
        print(f"   Integrated Signals: {len(result.integrated_signals)}")
    if result.volume_anomalies:
        high_sev = len([a for a in result.volume_anomalies if a.get('severity') in ['HIGH', 'CRITICAL']])
        print(f"   Volume Anomalies: {len(result.volume_anomalies)} detected ({high_sev} high severity)")
    if result.whitening_summary:
        print(f"   Whitening: {result.whitening_summary[:60]}...")
    if result.fact_check_grade != "N/A":
        print(f"   Fact Check Grade: {result.fact_check_grade}")
    print()
    print("🎯 FINAL RECOMMENDATION")
    print(f"   Action: {result.final_recommendation}")
    print(f"   Confidence: {result.confidence:.0%}")
    print(f"   Risk Level: {result.risk_level}")

    if result.warnings:
        print()
        print("⚠️  WARNINGS")
        for w in result.warnings:
            print(f"   - {w}")

    print()
    print(f"Results saved:")
    print(f"   JSON: {output_file}")
    print(f"   MD:   {md_file}")
    if report_path:
        print(f"   AI Report: {report_path}")
    print("=" * 70)

    return result


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='EIMAS - Economic Intelligence Multi-Agent System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python main.py                         # 전체 파이프라인 (기본)
    python main.py --mode full             # 전체 분석
    python main.py --mode quick            # 빠른 분석 (Phase 2.3-2.10 스킵)
    python main.py --mode full --target NVDA   # NVDA 중심 분석
    python main.py --report                # AI 제안서 생성 포함
    python main.py --realtime --duration 60    # 60초 실시간 스트리밍
    python main.py --cron                  # 서버 자동화 모드 (백그라운드)
    python main.py --cron --output /data/reports  # 지정 디렉토리에 저장

Terminal Automation:
    # 매일 오전 9시 자동 실행 (crontab)
    0 9 * * * cd /path/to/eimas && python main.py --cron >> /var/log/eimas.log 2>&1
        '''
    )

    # 모드 선택
    parser.add_argument(
        '--mode', '-m',
        choices=['full', 'quick', 'report'],
        default='full',
        help='Analysis mode: full (default), quick (fast), report (includes AI report)'
    )

    # 타겟 티커 (선택)
    parser.add_argument(
        '--target', '-t',
        type=str,
        default=None,
        help='Target ticker for focused analysis (e.g., NVDA, AAPL)'
    )

    # 서버 자동화 모드 (cron)
    parser.add_argument(
        '--cron',
        action='store_true',
        help='Cron mode: no visualization, background execution, markdown report only'
    )

    # 출력 디렉토리
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs',
        help='Output directory for reports (default: outputs)'
    )

    # 실시간 스트리밍
    parser.add_argument(
        '--realtime', '-r',
        action='store_true',
        help='Enable real-time Binance streaming'
    )

    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=30,
        help='Real-time streaming duration in seconds (default: 30)'
    )

    # 빠른 모드 (하위 호환)
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick mode (alias for --mode quick)'
    )

    # AI 리포트 생성 (하위 호환)
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate AI-powered investment report (alias for --mode report)'
    )

    # 상세 로깅
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    # 버전 정보
    parser.add_argument(
        '--version',
        action='version',
        version='EIMAS v2.1.0 (Real-World Agent Edition)'
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    # 로깅 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Cron 모드: 최소 출력
    if args.cron:
        logging.getLogger().setLevel(logging.WARNING)
        print(f"[CRON] EIMAS starting at {datetime.now().isoformat()}")

    # 모드 결정 (하위 호환성)
    quick_mode = args.quick or args.mode == 'quick'
    generate_report = args.report or args.mode == 'report'

    # Cron 모드에서는 realtime 비활성화
    enable_realtime = args.realtime and not args.cron

    # 타겟 티커 처리
    target_ticker = args.target
    if target_ticker:
        print(f"[INFO] Focused analysis on: {target_ticker}")

    # 출력 디렉토리 설정
    output_dir = args.output
    if output_dir != 'outputs':
        print(f"[INFO] Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 파이프라인 실행
    result = await run_full_pipeline(
        enable_realtime=enable_realtime,
        realtime_duration=args.duration,
        quick_mode=quick_mode,
        generate_report=generate_report,
        target_ticker=target_ticker,
        output_dir=output_dir,
        cron_mode=args.cron
    )

    # Cron 모드: 완료 메시지
    if args.cron:
        print(f"[CRON] EIMAS completed at {datetime.now().isoformat()}")
        if result:
            print(f"[CRON] Recommendation: {result.final_recommendation}")
            print(f"[CRON] Confidence: {result.confidence:.0%}")
            print(f"[CRON] Risk Level: {result.risk_level}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
