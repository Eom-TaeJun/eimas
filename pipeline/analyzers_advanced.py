#!/usr/bin/env python3
"""
EIMAS Pipeline - Analyzers Advanced Module

Purpose: Advanced market analysis functions (Phase 2.7-2.13)
Functions: analyze_genius_act, analyze_theme_etf, analyze_shock_propagation, optimize_portfolio_mst, analyze_volume_anomalies, track_events_with_news (async), run_adaptive_portfolio
"""

import os
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

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
    RegimeResult, GeniusActResult, ThemeETFResult, ShockAnalysisResult, PortfolioResult
)
from pipeline.exceptions import get_logger, log_error

logger = get_logger("analyzers.advanced")

def analyze_genius_act() -> GeniusActResult:
    """Genius Act (스테이블코인 유동성) 분석 - 간소화 버전"""
    print("\n[2.7] Genius Act Macro analysis...")
    try:
        import asyncio
        from lib.extended_data_sources import ExtendedDataCollector

        collector = ExtendedDataCollector()

        # async 함수를 동기적으로 실행
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 이벤트 루프가 있으면 nest_asyncio 사용 또는 직접 호출
                import nest_asyncio
                nest_asyncio.apply()
                stablecoin_data = loop.run_until_complete(collector.get_stablecoin_mcap())
            else:
                stablecoin_data = asyncio.run(collector.get_stablecoin_mcap())
        except RuntimeError:
            stablecoin_data = asyncio.run(collector.get_stablecoin_mcap())

        mcap = stablecoin_data.get('total_mcap', 0) if stablecoin_data else 0

        # 간단한 레짐 판단 (스테이블코인 시총 기준)
        if mcap > 300e9:
            regime = "EXPANSION"
        elif mcap > 150e9:
            regime = "NEUTRAL"
        else:
            regime = "CONTRACTION"

        signals = [{'type': 'stablecoin_mcap', 'value': mcap, 'regime': regime}]

        print(f"      ✓ Stablecoin Mcap: ${mcap/1e9:,.1f}B")
        print(f"      ✓ Regime: {regime}")

        return GeniusActResult(
            regime=regime,
            signals=signals,
            digital_m2=mcap,
            details={'stablecoin_mcap': mcap}
        )
    except Exception as e:
        log_error(logger, "Genius Act analysis failed", e)
        return GeniusActResult(regime="N/A", signals=[], digital_m2=0.0, details={})

def analyze_theme_etf() -> ThemeETFResult:
    """테마 ETF 분석 - 간소화 버전"""
    print("\n[2.8] Theme ETF analysis...")
    try:
        # CustomETFBuilder.analyze_risk_concentration()은 ThemeETF 객체가 필요함
        # 간소화: 기본 AI/반도체 섹터 ETF 분석
        import yfinance as yf

        theme = "AI & Semiconductor"
        sector_etfs = ['SMH', 'SOXX', 'NVDA', 'AMD', 'AVGO']

        # 간단한 성과 분석
        performance = {}
        for ticker in sector_etfs[:3]:  # 상위 3개만
            try:
                data = yf.Ticker(ticker).history(period='1mo')
                if not data.empty:
                    ret = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
                    performance[ticker] = ret
            except:
                pass

        # 가장 강한 성과
        if performance:
            best_ticker = max(performance, key=performance.get)
            score = performance[best_ticker]
        else:
            best_ticker = 'N/A'
            score = 0.0

        constituents = list(performance.keys())

        print(f"      ✓ Theme: {theme}")
        print(f"      ✓ Best Performer: {best_ticker} ({score:+.1f}%)")

        return ThemeETFResult(
            theme=theme,
            score=score,
            constituents=constituents,
            details={'performance': performance}
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
            if isinstance(df, pd.DataFrame) and not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) < 3:
            raise ValueError("Need at least 3 assets for shock propagation analysis")

        graph = ShockPropagationGraph()
        graph.build_from_data(returns_df)

        # find_all_critical_paths() 사용 (simulate_shock 대신)
        critical_paths = graph.find_all_critical_paths(top_n=3)

        if critical_paths:
            top_path = critical_paths[0]
            impact_score = top_path.total_propagation_time if hasattr(top_path, 'total_propagation_time') else len(top_path.nodes)
            contagion_path = top_path.nodes if hasattr(top_path, 'nodes') else []
            vulnerable_assets = contagion_path[-3:] if len(contagion_path) >= 3 else contagion_path
        else:
            impact_score = 0.0
            contagion_path = []
            vulnerable_assets = []

        print(f"      ✓ Critical Paths Found: {len(critical_paths)}")
        if contagion_path:
            print(f"      ✓ Top Path: {' → '.join(contagion_path[:5])}")

        return ShockAnalysisResult(
            impact_score=impact_score,
            contagion_path=contagion_path,
            vulnerable_assets=vulnerable_assets,
            details={'paths_found': len(critical_paths), 'graph_nodes': len(graph.graph.nodes())}
        )
    except Exception as e:
        log_error(logger, "Shock propagation analysis failed", e)
        return ShockAnalysisResult(impact_score=0.0, contagion_path=[], vulnerable_assets=[], details={})

def optimize_portfolio_mst(market_data: Dict[str, pd.DataFrame]) -> PortfolioResult:
    """GC-HRP 및 MST 기반 포트폴리오 최적화"""
    print("\n[2.10] Graph-Clustered Portfolio optimization...")
    try:
        # returns DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) < 3:
            raise ValueError("Need at least 3 assets for portfolio optimization")

        optimizer = GraphClusteredPortfolio()
        allocation = optimizer.fit(returns_df)

        weights = allocation.weights
        mst_info = {}
        if allocation.mst_analysis:
            # MSTAnalysisResult 속성 안전하게 접근
            systemic_nodes = getattr(allocation.mst_analysis, 'systemic_risk_nodes', [])
            mst_info = {
                'hubs': [getattr(n, 'ticker', str(n)) for n in systemic_nodes] if systemic_nodes else [],
            }

        top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join([f"{t}:{w:.0%}" for t, w in top_weights])

        print(f"      ✓ Top Allocation: {top_str}")
        print(f"      ✓ Diversification Ratio: {allocation.diversification_ratio:.2f}")

        return PortfolioResult(
            weights=weights,
            risk_contribution=allocation.risk_contributions,
            diversification_ratio=allocation.diversification_ratio,
            mst_hubs=mst_info.get('hubs', []),
            details={'effective_n': allocation.effective_n, 'methodology': allocation.methodology}
        )
    except Exception as e:
        log_error(logger, "Portfolio optimization failed", e)
        return PortfolioResult(weights={}, risk_contribution={}, diversification_ratio=0.0, mst_hubs=[], details={})

def analyze_volume_anomalies(market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """거래량 이상 징후 탐지"""
    print("\n[2.11] Volume anomaly detection...")
    try:
        analyzer = VolumeAnalyzer()
        result = analyzer.detect_anomalies(market_data)

        # VolumeAnalysisResult 객체에서 정보 추출
        anomaly_count = getattr(result, 'anomalies_detected', 0)
        anomalies_list = getattr(result, 'anomalies', [])
        total_tickers = getattr(result, 'total_tickers_analyzed', 'N/A')

        print(f"      ✓ Anomalies: {anomaly_count} detected")
        print(f"      ✓ Tickers analyzed: {total_tickers}")

        # Dict 리스트로 변환 (severity 속성 안전하게 접근)
        output = []
        for a in anomalies_list:
            severity_val = getattr(a, 'severity', 'UNKNOWN')
            # Enum이면 .value, 아니면 str로 변환
            if hasattr(severity_val, 'value'):
                severity_str = severity_val.value
            else:
                severity_str = str(severity_val)
            output.append({
                'ticker': getattr(a, 'ticker', 'N/A'),
                'severity': severity_str,
                'description': getattr(a, 'description', '')
            })
        return output
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
        persist_db = os.getenv("EIMAS_ADAPTIVE_PERSIST_DB", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

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

        results = manager.run_all(condition, prices, persist=persist_db)
        if not persist_db:
            print("      i Adaptive DB persistence disabled (EIMAS_ADAPTIVE_PERSIST_DB=false)")

        # 결과 요약
        summary = {name: res['action'] for name, res in results.items()}
        print(f"      ✓ Adaptive Allocation: {summary}")
        return summary
    except Exception as e:
        log_error(logger, "Adaptive portfolio analysis failed", e)
        return {}
