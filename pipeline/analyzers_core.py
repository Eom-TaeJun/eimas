#!/usr/bin/env python3
"""
EIMAS Pipeline - Analyzers Core Module

Purpose: Core market analysis functions (Phase 2.1-2.6)
Functions: detect_regime, detect_events, analyze_liquidity, analyze_critical_path, analyze_etf_flow, generate_explanation
"""

from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

# EIMAS libraries
from lib.regime_detector import RegimeDetector
from lib.event_framework import QuantitativeEventDetector
from lib.liquidity_analysis import LiquidityMarketAnalyzer
from lib.critical_path import CriticalPathAggregator
from lib.etf_flow_analyzer import ETFFlowAnalyzer

# Schemas
from pipeline.schemas import (
    RegimeResult, Event, LiquiditySignal,
    CriticalPathResult, ETFFlowResult, FREDSummary
)
from pipeline.exceptions import get_logger, log_error

logger = get_logger("analyzers.core")

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
            strategy=result.strategy,
            gmm_probabilities=result.gmm_probabilities if hasattr(result, 'gmm_probabilities') else None
        )
    except Exception as e:
        log_error(logger, "Regime detection failed", e)
        return RegimeResult(
            timestamp=datetime.now().isoformat(),
            regime="Unknown",
            trend="Unknown",
            volatility="Unknown",
            confidence=0.0,
            description="",
            strategy="",
            is_valid=False,
            error_code="REGIME_DETECTION_ERROR",
            error_msg=str(e)
        )

def detect_events(fred_summary: FREDSummary, market_data: Dict[str, pd.DataFrame]) -> List[Event]:
    """이벤트 탐지 (Liquidity + News + Market Signals)"""
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

        # 뉴스 기반 이벤트 (News + Price Signals)
        try:
            from lib.news_correlator import NewsCorrelator  # news_event_generator 대체
            macro_analysis = {
                'net_liquidity': fred_summary.net_liquidity if fred_summary else None,
                'treasury_10y': fred_summary.treasury_10y if fred_summary else None,
            } if fred_summary else None

            # news_event_generator 삭제됨 - 뉴스 이벤트 생성 스킵
            news_events = []

        except Exception as e:
            print(f"      ⚠ News event generation failed: {e}")

        if not events:
            print("      ✓ No events detected")
        else:
            print(f"      ✓ Detected {len(events)} events")

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
            risk_score=0.0,
            risk_level="Unknown",
            primary_risk_path="N/A",
            details={},
            is_valid=False,
            error_code="CRITICAL_PATH_ERROR",
            error_msg=str(e)
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
    """SHAP 기반 시장 설명 (explanation_generator 삭제로 비활성화)"""
    print("\n[2.6] Market explanation skipped (explanation_generator removed)")
    return {"narrative": "N/A", "skipped": True}
