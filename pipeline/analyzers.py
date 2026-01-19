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

Dependencies:
    - lib.regime_detector
    - lib.event_framework
    - lib.liquidity_analysis
    - lib.critical_path

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

# Schemas
from pipeline.schemas import (
    RegimeResult, Event, LiquiditySignal, 
    CriticalPathResult, ETFFlowResult, FREDSummary
)

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
        print(f"      ✗ Regime error: {e}")
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
        
        # 시장 이벤트 로직 추가 가능
        
        if not events:
            print("      ✓ No events detected")
            
        return events
        
    except Exception as e:
        print(f"      ✗ Event detection error: {e}")
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
        print(f"      ✗ Liquidity analysis error: {e}")
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
        print(f"      ✗ Critical path error: {e}")
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
        print(f"      ✗ ETF flow error: {e}")
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
        print(f"      ✗ Explanation error: {e}")
        return {}
