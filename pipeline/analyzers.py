#!/usr/bin/env python3
"""
EIMAS Pipeline - Analyzers Module
==================================

Purpose:
    Phase 2 시장 분석 담당 (Market Analysis)

Functions:
    - detect_regime(ticker) -> RegimeResult
    - detect_events(fred_summary, market_data) -> List[Event]
    - analyze_liquidity() -> LiquiditySignal
    - analyze_critical_path(market_data) -> CriticalPathResult
    - analyze_etf_flow() -> ETFFlowResult
    - generate_explanation(market_data) -> Dict
    - analyze_genius_act() -> GeniusActResult (Advanced)
    - analyze_theme_etf() -> ThemeETFResult (Advanced)
    - analyze_shock_propagation(market_data) -> ShockAnalysisResult (Advanced)
    - optimize_portfolio_mst(market_data) -> PortfolioResult (Advanced)

Dependencies:
    - lib.regime_detector
    - lib.event_framework
    - lib.liquidity_analysis
    - lib.critical_path
    - lib.genius_act_macro
    - lib.custom_etf_builder
    - lib.shock_propagation_graph
    - lib.graph_clustered_portfolio

Example:
    from pipeline.analyzers import detect_regime
    regime = detect_regime('SPY')
    print(regime.confidence)
"""

from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

# EIMAS 라이브러리
from lib.regime_detector import RegimeDetector
from lib.event_framework import QuantitativeEventDetector
from lib.liquidity_analysis import LiquidityMarketAnalyzer
from lib.critical_path import CriticalPathAggregator
from lib.etf_flow_analyzer import ETFFlowAnalyzer
from lib.explanation_generator import MarketExplanationGenerator

# Advanced Libraries
from lib.genius_act_macro import GeniusActMacroStrategy
from lib.custom_etf_builder import CustomETFBuilder
from lib.shock_propagation_graph import ShockPropagationGraph
from lib.graph_clustered_portfolio import GraphClusteredPortfolio

# Missing Microstructure & Adaptive Modules
from lib.volume_analyzer import VolumeAnalyzer
from lib.event_tracker import EventTracker
from lib.adaptive_agents import AdaptiveAgentManager, MarketCondition

# Schemas
from pipeline.schemas import (
    RegimeResult, Event, LiquiditySignal, 
    CriticalPathResult, ETFFlowResult, FREDSummary,
    GeniusActResult, ThemeETFResult, ShockAnalysisResult, PortfolioResult
)
from pipeline.exceptions import get_logger, log_error

logger = get_logger("analyzers")

def detect_regime(ticker: str = 'SPY') -> RegimeResult:
    """시장 레짐 탐지"""
    print("\n[2.1] Detecting market regime...")
    try:
        detector = RegimeDetector(ticker=ticker)
        result = detector.detect()
        
        return RegimeResult(
            timestamp=datetime.now().isoformat(),
            regime=result.regime.value if hasattr(result.regime, 'value') else str(result.regime),
            trend=result.trend_state.value if hasattr(result.trend_state, 'value') else str(result.trend_state),
            volatility=result.volatility_state.value if hasattr(result.volatility_state, 'value') else str(result.volatility_state),
            confidence=result.confidence / 100 if result.confidence > 1 else result.confidence,
            description=result.description,
            strategy=result.strategy
        )
    except Exception as e:
        log_error(logger, "Regime detection failed", e)
        return RegimeResult(
            timestamp=datetime.now().isoformat(),
            regime="Unknown", trend="Unknown", volatility="Unknown",
            confidence=0.0, description="", strategy=""
        )

def detect_events(fred_summary: FREDSummary, market_data: Dict[str, pd.DataFrame]) -> List[Event]:
    """이벤트 탐지"""
    print("\n[2.2] Detecting events...")
    events = []
    try:
        detector = QuantitativeEventDetector()
        
        # 유동성 이벤트
        if fred_summary:
            liquidity_data = {
                'rrp': fred_summary.rrp,
                'rrp_delta': fred_summary.rrp_delta,
                'tga': fred_summary.tga,
                'tga_delta': fred_summary.tga_delta,
                'net_liquidity': fred_summary.net_liquidity,
            }
            liquidity_events = detector.detect_liquidity_events(liquidity_data)
            
            for e in liquidity_events:
                events.append(Event(
                    type=e.event_type.value,
                    importance=e.importance.value,
                    description=e.description,
                    timestamp=datetime.now().isoformat()
                ))
                print(f"      ⚠ {e.event_type.value}: {e.description}")
        
        if not events:
            print("      ✓ No events detected")
            
        return events
        
    except Exception as e:
        log_error(logger, "Event detection failed", e)
        return []

def analyze_liquidity() -> LiquiditySignal:
    """유동성 인과관계 분석"""
    print("\n[2.3] Liquidity-Market causality analysis...")
    try:
        analyzer = LiquidityMarketAnalyzer()
        result = analyzer.generate_signals()
        
        signal = result.get('signal', 'NEUTRAL')
        print(f"      ✓ Liquidity Signal: {signal}")
        
        return LiquiditySignal(
            signal=signal,
            causality_results=result.get('causality_results', {})
        )
    except Exception as e:
        log_error(logger, "Liquidity analysis failed", e)
        return LiquiditySignal(signal="NEUTRAL", causality_results={})

def analyze_critical_path(market_data: Dict[str, pd.DataFrame]) -> CriticalPathResult:
    """Critical Path 리스크 분석"""
    print("\n[2.4] Critical path analysis...")
    try:
        aggregator = CriticalPathAggregator()
        if not market_data:
            raise ValueError("No market data")
            
        result = aggregator.analyze(market_data)
        risk_score = getattr(result, 'total_risk_score', 0)
        
        print(f"      ✓ Risk Score: {risk_score:.1f}/100")
        
        return CriticalPathResult(
            risk_score=risk_score,
            risk_level=getattr(result, 'risk_level', 'Unknown'),
            primary_risk_path=getattr(result, 'primary_risk_path', 'N/A'),
            details=getattr(result, 'path_scores', {})
        )
    except Exception as e:
        log_error(logger, "Critical path analysis failed", e)
        return CriticalPathResult(
            risk_score=0.0, risk_level="Unknown", 
            primary_risk_path="N/A", details={}
        )

def analyze_etf_flow() -> ETFFlowResult:
    """ETF 자금 흐름 및 섹터 분석"""
    print("\n[2.5] ETF flow analysis...")
    try:
        analyzer = ETFFlowAnalyzer()
        result = analyzer.analyze()
        
        rotation = result.get('rotation_signal', 'N/A')
        print(f"      ✓ Sector Rotation: {rotation}")
        
        return ETFFlowResult(
            rotation_signal=rotation,
            style_signal=result.get('style_signal', 'N/A'),
            details=result
        )
    except Exception as e:
        log_error(logger, "ETF flow analysis failed", e)
        return ETFFlowResult(rotation_signal="N/A", style_signal="N/A", details={})

def generate_explanation(market_data: Dict[str, pd.DataFrame]) -> Dict:
    """SHAP 기반 시장 설명"""
    print("\n[2.6] Generating market explanation (SHAP)...")
    try:
        generator = MarketExplanationGenerator()
        explanation = generator.generate_explanation(market_data)
        
        if "narrative" in explanation:
            print(f"      ✓ Narrative: {explanation['narrative'][:50]}...")
            
        return explanation
    except Exception as e:
        log_error(logger, "Explanation generation failed", e)
        return {}

# ============================================================================
# Advanced Analyzers (Restored)
# ============================================================================

def analyze_genius_act() -> GeniusActResult:
    """Genius Act (스테이블코인 유동성) 분석"""
    print("\n[2.7] Genius Act Macro analysis...")
    try:
        strategy = GeniusActMacroStrategy()
        signals = strategy.analyze_stablecoin_issuance()
        m2 = strategy.calculate_digital_m2()
        
        regime = "NEUTRAL"
        if signals:
            regime = signals[-1].get('regime', 'NEUTRAL')
            
        print(f"      ✓ Digital M2: ${m2:,.0f}")
        print(f"      ✓ Regime: {regime}")
        
        return GeniusActResult(
            regime=regime,
            signals=signals,
            digital_m2=m2,
            details={'signals_count': len(signals)}
        )
    except Exception as e:
        log_error(logger, "Genius Act analysis failed", e)
        return GeniusActResult(regime="N/A", signals=[], digital_m2=0.0, details={})

def analyze_theme_etf() -> ThemeETFResult:
    """테마 ETF 분석"""
    print("\n[2.8] Theme ETF analysis...")
    try:
        builder = CustomETFBuilder()
        result = builder.build_theme_etf("AI & Semiconductor")
        
        print(f"      ✓ Theme: {result.get('theme', 'Unknown')}")
        
        return ThemeETFResult(
            theme=result.get('theme', 'Unknown'),
            score=result.get('score', 0.0),
            constituents=result.get('constituents', []),
            details=result
        )
    except Exception as e:
        log_error(logger, "Theme ETF analysis failed", e)
        return ThemeETFResult(theme="N/A", score=0.0, constituents=[], details={})

def analyze_shock_propagation(market_data: Dict[str, pd.DataFrame]) -> ShockAnalysisResult:
    """충격 전파 시뮬레이션"""
    print("\n[2.9] Shock propagation analysis...")
    try:
        # returns DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()
        
        graph = ShockPropagationGraph()
        graph.build_from_returns(returns_df)
        
        # SPY 충격 시뮬레이션
        impact = graph.simulate_shock('SPY', shock_size=-0.05)
        
        print(f"      ✓ SPY Shock Impact: {len(impact.get('affected_nodes', []))} nodes affected")
        
        return ShockAnalysisResult(
            impact_score=impact.get('total_impact', 0.0),
            contagion_path=impact.get('propagation_path', []),
            vulnerable_assets=impact.get('vulnerable_assets', []),
            details=impact
        )
    except Exception as e:
        log_error(logger, "Shock propagation analysis failed", e)
        return ShockAnalysisResult(impact_score=0.0, contagion_path=[], vulnerable_assets=[], details={})

def optimize_portfolio_mst(market_data: Dict[str, pd.DataFrame]) -> PortfolioResult:
    """GC-HRP 및 MST 기반 포트폴리오 최적화"""
    print("\n[2.10] Graph-Clustered Portfolio optimization...")
    try:
        optimizer = GraphClusteredPortfolio()
        result = optimizer.optimize(market_data)
        
        weights = result.get('weights', {})
        mst_info = result.get('mst_info', {})
        
        top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join([f"{t}:{w:.0%}" for t, w in top_weights])
        
        print(f"      ✓ Top Allocation: {top_str}")
        print(f"      ✓ Diversification Ratio: {result.get('diversification_ratio', 0):.2f}")
        
        return PortfolioResult(
            weights=weights,
            risk_contribution=result.get('risk_contribution', {}),
            diversification_ratio=result.get('diversification_ratio', 0.0),
            mst_hubs=mst_info.get('hubs', []),
            details=result
        )
    except Exception as e:
        log_error(logger, "Portfolio optimization failed", e)
        return PortfolioResult(weights={}, risk_contribution={}, diversification_ratio=0.0, mst_hubs=[], details={})

# ============================================================================
# Missing Advanced Analyzers (Restored)
# ============================================================================

def analyze_volume_anomalies(market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """거래량 이상 징후 탐지"""
    print("\n[2.11] Volume anomaly detection...")
    try:
        analyzer = VolumeAnalyzer()
        anomalies = analyzer.detect_anomalies(market_data)
        print(f"      ✓ Anomalies: {len(anomalies)} detected")
        return anomalies
    except Exception as e:
        log_error(logger, "Volume anomaly detection failed", e)
        return []

async def track_events_with_news(market_data: Dict[str, pd.DataFrame]) -> Dict:
    """이상 징후 발생 시 뉴스 검색 (Event Tracking)"""
    print("\n[2.12] Event tracking (anomaly -> news)...")
    try:
        tracker = EventTracker()
        # 최근 데이터 기준 이상 감지 및 뉴스 검색
        # 주의: API 호출 비용 발생 가능
        results = await tracker.detect_and_track(market_data)
        print(f"      ✓ Tracked Events: {len(results.get('events', []))}")
        return results
    except Exception as e:
        log_error(logger, "Event tracking failed", e)
        return {}

def run_adaptive_portfolio(regime_result: RegimeResult) -> Dict:
    """적응형 포트폴리오 전략 수립"""
    print("\n[2.13] Adaptive portfolio agents...")
    try:
        manager = AdaptiveAgentManager()
        
        # RegimeResult -> MarketCondition 변환
        condition = MarketCondition(
            timestamp=regime_result.timestamp,
            regime=regime_result.regime,
            regime_confidence=regime_result.confidence * 100, # 0-100 scale
            trend=regime_result.trend,
            volatility=regime_result.volatility,
            risk_score=50, # Default if not passed
            vix_level=20, # Default
            liquidity_signal="NEUTRAL" # Default
        )
        
        # 가상의 가격 데이터 (실제 데이터 연동 필요하나 여기선 간소화)
        prices = {"SPY": 500.0, "QQQ": 400.0, "TLT": 95.0, "GLD": 200.0}
        
        results = manager.run_all(condition, prices)
        
        # 결과 요약
        summary = {name: res['action'] for name, res in results.items()}
        print(f"      ✓ Adaptive Allocation: {summary}")
        return summary
    except Exception as e:
        log_error(logger, "Adaptive portfolio analysis failed", e)
        return {}