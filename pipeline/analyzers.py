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
    - analyze_hft_microstructure(market_data) -> Dict (NEW 2026-01-24)
    - analyze_volatility_garch(market_data) -> Dict (NEW 2026-01-24)
    - analyze_information_flow(market_data) -> Dict (NEW 2026-01-24)
    - calculate_proof_of_index(market_data) -> Dict (NEW 2026-01-24)
    - enhance_portfolio_with_systemic_similarity(market_data) -> Dict (NEW 2026-01-24)

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
import numpy as np

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

# NEW: Enhanced Modules (2026-01-24 보완 작업)
from lib.microstructure import (
    tick_rule_classification,
    kyles_lambda,
    volume_clock_sampling,
    detect_quote_stuffing,
    DailyMicrostructureAnalyzer
)
from lib.regime_analyzer import GARCHModel
from lib.information_flow import InformationFlowAnalyzer
from lib.proof_of_index import ProofOfIndex
from lib.ark_holdings_analyzer import ARKHoldingsAnalyzer, ARKHoldingsCollector

# NEW: Advanced Clustering & Time Series (2026-01-25)
from lib.time_series_similarity import (
    compute_dtw_similarity_matrix,
    find_lead_lag_relationship,
    detect_regime_shift_dtw
)

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
            if not df.empty and 'Close' in df.columns:
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
            if not df.empty and 'Close' in df.columns:
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

# ============================================================================
# Missing Advanced Analyzers (Restored)
# ============================================================================

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

# ============================================================================
# NEW: Enhanced Analyzers (2026-01-24 보완 작업)
# ============================================================================

def analyze_hft_microstructure(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    HFT 미세구조 분석 (Enhanced)

    기능:
    - Tick Rule: 거래 방향 분류
    - Kyle's Lambda: Market Impact 측정
    - Volume Clock: VPIN 정확도 향상
    - Quote Stuffing: 시장 교란 탐지

    References:
    - Lee & Ready (1991)
    - Kyle (1985)
    - Easley et al. (2012)
    """
    print("\n[2.14] HFT Microstructure Analysis (Enhanced)...")
    try:
        results = {}

        # SPY 데이터로 테스트
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_data = market_data['SPY']

            # Tick Rule Classification
            if 'Close' in spy_data.columns:
                prices = spy_data['Close']
                directions = tick_rule_classification(prices)
                buy_ratio = (directions == 1).sum() / len(directions)

                results['tick_rule'] = {
                    'buy_ratio': buy_ratio,
                    'sell_ratio': 1 - buy_ratio,
                    'interpretation': 'BUY_PRESSURE' if buy_ratio > 0.55 else 'SELL_PRESSURE' if buy_ratio < 0.45 else 'NEUTRAL'
                }
                print(f"      ✓ Tick Rule: Buy Ratio {buy_ratio:.1%}")

            # Kyle's Lambda (Market Impact)
            if 'Close' in spy_data.columns and 'Volume' in spy_data.columns:
                try:
                    # 1. Series 추출 및 기본 정제
                    close_s = spy_data['Close'].squeeze()
                    vol_s = spy_data['Volume'].squeeze()
                    
                    if isinstance(close_s, pd.DataFrame): close_s = close_s.iloc[:, 0]
                    if isinstance(vol_s, pd.DataFrame): vol_s = vol_s.iloc[:, 0]

                    # 2. 계산
                    returns = close_s.pct_change().rename("returns")
                    directions = tick_rule_classification(close_s).rename("directions")
                    
                    # 3. 데이터프레임으로 통합 (자동 인덱스 정렬)
                    df_micro = pd.concat([returns, directions, vol_s], axis=1)
                    df_micro.columns = ['returns', 'directions', 'volume'] # 컬럼명 강제 지정
                    
                    # 4. 결측치 제거
                    df_micro = df_micro.dropna()
                    
                    # 5. Signed Volume 계산 및 Lambda 추정
                    if not df_micro.empty:
                        signed_vol = df_micro['directions'] * df_micro['volume']
                        lambda_result = kyles_lambda(df_micro['returns'], signed_vol)
                        results['kyles_lambda'] = lambda_result
                        print(f"      ✓ Kyle's Lambda: {lambda_result['lambda']:.6f} ({lambda_result['interpretation']})")
                    else:
                        print("      ⚠️ Kyle's Lambda skipped: No valid data")
                
                except Exception as ex:
                    print(f"      ⚠️ Kyle's Lambda skipped: {ex}")

            # Volume Clock Sampling (VPIN 향상용)
            if 'Volume' in spy_data.columns:
                volume_bucket = spy_data['Volume'].sum() / 20  # 20 buckets
                # reset_index()로 인해 컬럼명 확인 필요
                spy_df = spy_data.reset_index()
                # 컬럼명 통일 (volume_col 매개변수 사용)
                volume_col = 'Volume' if 'Volume' in spy_df.columns else 'volume'
                sampled = volume_clock_sampling(spy_df, volume_bucket, volume_col=volume_col)
                results['volume_clock'] = {
                    'original_samples': len(spy_data),
                    'volume_samples': len(sampled),
                    'compression_ratio': len(sampled) / len(spy_data)
                }
                print(f"      ✓ Volume Clock: {len(spy_data)} → {len(sampled)} samples")

        return results
    except Exception as e:
        log_error(logger, "HFT Microstructure analysis failed", e)
        return {}


def analyze_volatility_garch(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    GARCH 변동성 모델링

    기능:
    - GARCH(1,1) 모델 피팅
    - 조건부 변동성 추정
    - 다중 기간 변동성 예측

    References:
    - Engle (1982)
    - Bollerslev (1986)
    """
    print("\n[2.15] GARCH Volatility Modeling...")
    try:
        results = {}

        # SPY 수익률로 GARCH 모델링
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_data = market_data['SPY']

            if 'Close' in spy_data.columns:
                returns = spy_data['Close'].pct_change().dropna()

                # GARCH(1,1) 모델
                garch = GARCHModel(p=1, q=1)
                params = garch.fit(returns)

                # 10일 변동성 예측
                vol_forecast = garch.forecast(horizon=10)

                results['garch_params'] = params
                results['volatility_forecast_10d'] = vol_forecast.to_dict()

                # 스칼라로 변환 (Series 문제 방지)
                curr_vol = returns.std() * np.sqrt(252)
                curr_vol_scalar = float(curr_vol.item() if hasattr(curr_vol, 'item') else curr_vol)

                forecast_vol = vol_forecast.mean() * np.sqrt(252)
                forecast_vol_scalar = float(forecast_vol.item() if hasattr(forecast_vol, 'item') else forecast_vol)

                results['current_volatility'] = curr_vol_scalar
                results['forecast_avg_volatility'] = forecast_vol_scalar

                print(f"      ✓ GARCH(1,1) Persistence: {params['persistence']:.3f}")
                print(f"      ✓ Half-life: {params['half_life']:.1f} days")
                print(f"      ✓ Current Vol: {curr_vol_scalar:.1%}")
                print(f"      ✓ Forecast Vol (10d avg): {forecast_vol_scalar:.1%}")

        return results
    except Exception as e:
        log_error(logger, "GARCH volatility modeling failed", e)
        return {}


def analyze_information_flow(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    정보 플로우 분석

    기능:
    - 거래량 이상 탐지
    - Private Information Score
    - CAPM Alpha/Beta 추정

    References:
    - 금융경제정리.docx
    """
    print("\n[2.16] Information Flow Analysis...")
    try:
        analyzer = InformationFlowAnalyzer()
        results = {}

        # SPY 데이터 분석
        if 'SPY' in market_data and not market_data['SPY'].empty:
            spy_data = market_data['SPY']

            # 1. 거래량 이상 탐지
            if 'Volume' in spy_data.columns:
                abnormal_result = analyzer.detect_abnormal_volume(spy_data['Volume'])
                results['abnormal_volume'] = abnormal_result.to_dict()
                print(f"      ✓ Abnormal Volume: {abnormal_result.total_abnormal_days} days ({abnormal_result.abnormal_ratio:.1%})")

            # 2. CAPM Alpha/Beta (vs SPY)
            if 'Close' in spy_data.columns:
                spy_returns = spy_data['Close'].pct_change().dropna()

                # 다른 자산들과 비교
                for ticker in ['QQQ', 'TLT', 'GLD']:
                    if ticker in market_data and not market_data[ticker].empty:
                        asset_data = market_data[ticker]
                        if 'Close' in asset_data.columns:
                            asset_returns = asset_data['Close'].pct_change().dropna()

                            capm_result = analyzer.estimate_capm(asset_returns, spy_returns)
                            results[f'capm_{ticker}'] = capm_result.to_dict()

                            if ticker == 'QQQ':  # QQQ만 출력
                                print(f"      ✓ {ticker} CAPM: Alpha={capm_result.alpha*252:+.1%}/yr, Beta={capm_result.beta:.2f}")

        return results
    except Exception as e:
        log_error(logger, "Information flow analysis failed", e)
        return {}


def calculate_proof_of_index(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Proof-of-Index (블록체인 기반 투명 지수)

    기능:
    - 시가총액 가중 지수 계산
    - SHA-256 해시 검증
    - Mean Reversion 신호 생성

    References:
    - eco4.docx
    - Nakamoto (2008)
    """
    print("\n[2.17] Proof-of-Index Calculation...")
    try:
        poi = ProofOfIndex(divisor=100.0, name='EIMAS Portfolio Index')
        results = {}

        # 포트폴리오 구성 자산 선택
        portfolio_tickers = ['SPY', 'QQQ', 'TLT', 'GLD']
        available_tickers = [t for t in portfolio_tickers if t in market_data and not market_data[t].empty]

        if available_tickers:
            # 최신 가격 수집 (스칼라로 변환)
            prices = {}
            for ticker in available_tickers:
                if 'Close' in market_data[ticker].columns:
                    price_val = market_data[ticker]['Close'].iloc[-1]
                    # Series/array를 스칼라로 변환
                    prices[ticker] = float(price_val.item() if hasattr(price_val, 'item') else price_val)

            # 동일 가중 (실제로는 시가총액 가중)
            quantities = {ticker: 1.0 for ticker in prices.keys()}

            # 인덱스 계산
            snapshot = poi.calculate_index(prices, quantities)

            results['index_value'] = snapshot.index_value
            results['weights'] = snapshot.weights
            results['hash'] = snapshot.hash_value
            results['timestamp'] = snapshot.timestamp.isoformat()

            # SHA-256 검증
            reference_hash = poi.hash_index_weights(snapshot.weights, snapshot.timestamp)
            verification = poi.verify_on_chain(snapshot.hash_value, reference_hash)
            results['verification'] = verification

            print(f"      ✓ Index Value: {snapshot.index_value:.2f}")
            print(f"      ✓ Components: {', '.join([f'{t}:{w:.0%}' for t, w in sorted(snapshot.weights.items(), key=lambda x: -x[1])[:3]])}")
            print(f"      ✓ Hash Verification: {'✅ PASS' if verification['is_valid'] else '❌ FAIL'}")

            # Mean Reversion Signal (SPY 기준)
            if 'SPY' in market_data and 'Close' in market_data['SPY'].columns:
                spy_prices = market_data['SPY']['Close']
                signal = poi.mean_reversion_signal(spy_prices, window=20, threshold=2.0)
                results['mean_reversion_signal'] = signal.to_dict()
                print(f"      ✓ Mean Reversion: {signal.signal} (Z={signal.z_score:.2f})")

        return results
    except Exception as e:
        log_error(logger, "Proof-of-Index calculation failed", e)
        return {}


def enhance_portfolio_with_systemic_similarity(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    포트폴리오에 Systemic Similarity 추가 분석

    기능:
    - compute_systemic_similarity() 통합
    - 자산 간 상호작용 강도 정량화

    References:
    - De Prado (2016)
    - eco1.docx
    """
    print("\n[2.18] Systemic Similarity Enhancement...")
    try:
        from lib.graph_clustered_portfolio import CorrelationNetwork
        import numpy as np

        results = {}

        # returns DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) >= 3:
            # 네트워크 구축
            network = CorrelationNetwork()
            network.build_from_returns(returns_df)

            # Systemic Similarity 계산
            d_bar = network.compute_systemic_similarity()

            # 가장 유사한 자산 쌍 찾기
            d_bar_values = d_bar.values.copy()
            np.fill_diagonal(d_bar_values, np.inf)  # 대각선 제외
            min_idx = np.unravel_index(d_bar_values.argmin(), d_bar_values.shape)
            most_similar_pair = (d_bar.index[min_idx[0]], d_bar.columns[min_idx[1]])
            min_similarity = d_bar_values[min_idx]

            # 가장 상이한 자산 쌍
            max_idx = np.unravel_index(d_bar_values.argmax(), d_bar_values.shape)
            most_different_pair = (d_bar.index[max_idx[0]], d_bar.columns[max_idx[1]])
            max_similarity = d_bar_values[max_idx]

            results['systemic_similarity_matrix'] = d_bar.to_dict()
            results['most_similar_pair'] = {
                'assets': most_similar_pair,
                'similarity': min_similarity
            }
            results['most_different_pair'] = {
                'assets': most_different_pair,
                'dissimilarity': max_similarity
            }

            print(f"      ✓ Most Similar: {most_similar_pair[0]} ↔ {most_similar_pair[1]} (D̄={min_similarity:.3f})")
            print(f"      ✓ Most Different: {most_different_pair[0]} ↔ {most_different_pair[1]} (D̄={max_similarity:.3f})")

        return results
    except Exception as e:
        log_error(logger, "Systemic similarity enhancement failed", e)
        return {}


def detect_outliers_with_dbscan(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    DBSCAN 기반 이상치 탐지 (Phase 2.19)

    경제학적 배경:
    - 밀도 기반 클러스터링으로 노이즈 자산 제거
    - HRP 포트폴리오 품질 향상
    - 비정상 자산 자동 식별

    기능:
    - CorrelationNetwork.detect_outliers_dbscan() 통합
    - 이상치 자산 리스트 반환
    - 클러스터링 품질 메트릭

    References:
    - Ester et al. (1996)
    - 금융경제정리.docx
    """
    print("\n[2.19] DBSCAN Outlier Detection...")
    try:
        from lib.graph_clustered_portfolio import CorrelationNetwork

        results = {}

        # returns DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) >= 5:  # 최소 5개 자산 필요
            # 네트워크 구축
            network = CorrelationNetwork(correlation_threshold=0.2)
            network.build_from_returns(returns_df)

            # DBSCAN 실행
            outlier_result = network.detect_outliers_dbscan(
                eps=0.6,
                min_samples=3
            )

            # 결과 저장
            results['n_total_assets'] = outlier_result.n_total_assets
            results['n_outliers'] = outlier_result.n_outliers
            results['outlier_ratio'] = outlier_result.outlier_ratio
            results['outlier_tickers'] = outlier_result.outlier_tickers
            results['normal_tickers'] = outlier_result.normal_tickers
            results['n_clusters'] = outlier_result.n_clusters
            results['interpretation'] = outlier_result.interpretation
            results['eps'] = outlier_result.eps
            results['min_samples'] = outlier_result.min_samples

            print(f"      ✓ Total Assets: {outlier_result.n_total_assets}")
            print(f"      ✓ Outliers: {outlier_result.n_outliers} ({outlier_result.outlier_ratio:.1%})")
            print(f"      ✓ Clusters: {outlier_result.n_clusters}")
            print(f"      ✓ {outlier_result.interpretation}")

            if outlier_result.n_outliers > 0:
                print(f"      ✓ Outlier Assets (first 5): {', '.join(outlier_result.outlier_tickers[:5])}")

        return results
    except Exception as e:
        log_error(logger, "DBSCAN outlier detection failed", e)
        return {}


def analyze_dtw_similarity(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    DTW (Dynamic Time Warping) 시계열 유사도 분석 (Phase 2.20)

    경제학적 배경:
    - 시차를 무시하고 패턴 유사도만 측정
    - 리드-래그 관계 파악 (선행/후행 자산)
    - 레짐 전환 조기 감지

    기능:
    - compute_dtw_similarity_matrix() - 유사도 행렬
    - find_lead_lag_relationship() - 리드-래그 탐지
    - 상관관계와 비교 분석

    References:
    - Berndt & Clifford (1994)
    - todolist.md
    """
    print("\n[2.20] DTW Time Series Similarity Analysis...")
    try:
        results = {}

        # returns DataFrame 생성
        returns_df = pd.DataFrame()
        for ticker, df in market_data.items():
            if not df.empty and 'Close' in df.columns:
                returns_df[ticker] = df['Close'].pct_change()
        returns_df = returns_df.dropna()

        if len(returns_df.columns) >= 3:
            # DTW 유사도 행렬 계산
            dtw_result = compute_dtw_similarity_matrix(
                returns_df,
                window=20,
                normalize=True
            )

            results['n_series'] = dtw_result.n_series
            results['avg_distance'] = dtw_result.avg_distance
            results['most_similar_pair'] = {
                'asset1': dtw_result.most_similar_pair[0],
                'asset2': dtw_result.most_similar_pair[1],
                'distance': dtw_result.most_similar_pair[2]
            }
            results['most_dissimilar_pair'] = {
                'asset1': dtw_result.most_dissimilar_pair[0],
                'asset2': dtw_result.most_dissimilar_pair[1],
                'distance': dtw_result.most_dissimilar_pair[2]
            }

            print(f"      ✓ Assets Analyzed: {dtw_result.n_series}")
            print(f"      ✓ Avg DTW Distance: {dtw_result.avg_distance:.4f}")
            print(f"      ✓ Most Similar: {dtw_result.most_similar_pair[0]} ↔ "
                  f"{dtw_result.most_similar_pair[1]} (DTW={dtw_result.most_similar_pair[2]:.4f})")

            # 리드-래그 관계 탐지 (SPY vs QQQ 예시)
            if 'SPY' in returns_df.columns and 'QQQ' in returns_df.columns:
                lead_lag = find_lead_lag_relationship(
                    returns_df['SPY'],
                    returns_df['QQQ'],
                    max_lag=10,
                    series1_name='SPY',
                    series2_name='QQQ'
                )

                results['lead_lag_spy_qqq'] = {
                    'lead_asset': lead_lag.lead_asset,
                    'lag_asset': lead_lag.lag_asset,
                    'optimal_lag': lead_lag.optimal_lag,
                    'min_distance': lead_lag.min_distance,
                    'cross_correlation': lead_lag.cross_correlation,
                    'interpretation': lead_lag.interpretation
                }

                print(f"      ✓ Lead-Lag (SPY vs QQQ): {lead_lag.interpretation}")

        return results
    except Exception as e:
        log_error(logger, "DTW similarity analysis failed", e)
        return {}


def analyze_ark_trades() -> Dict[str, Any]:
    """
    ARK Invest (Cathie Wood) ETF 트레이딩 분석

    기능:
    - ARKK, ARKW 등 주요 ETF의 일일 보유량 변화 추적
    - 'Consensus Buy/Sell': 여러 ETF가 동시에 매수/매도한 종목 식별
    - 'New Position': 신규 편입 종목 식별

    경제학적 의미:
    - 스마트 머니(Smart Money)의 선행 지표 역할
    - 기술주/성장주 섹터의 센티먼트 파악
    """
    print("\n[2.21] ARK Invest Holdings Analysis...")
    try:
        # 데이터 수집 및 분석
        collector = ARKHoldingsCollector()
        analyzer = ARKHoldingsAnalyzer(collector)
        result = analyzer.run_analysis()
        
        # 신호 생성
        signals = analyzer.generate_signals(result)
        
        # 결과 요약
        summary = {
            'timestamp': result.timestamp,
            'consensus_buys': result.consensus_buys,
            'consensus_sells': result.consensus_sells,
            'new_positions': result.new_positions,
            'top_increases': [c.to_dict() for c in result.top_increases[:5]],
            'top_decreases': [c.to_dict() for c in result.top_decreases[:5]],
            'signals': [
                f"{s.direction.value} {s.ticker} ({s.description})" 
                for s in signals
            ]
        }
        
        # 주요 발견 출력
        if result.consensus_buys:
            print(f"      ✓ ARK Consensus BUY: {', '.join(result.consensus_buys)}")
        if result.consensus_sells:
            print(f"      ✓ ARK Consensus SELL: {', '.join(result.consensus_sells)}")
        if result.new_positions:
            print(f"      ✓ New Positions: {', '.join(result.new_positions)}")
        if not (result.consensus_buys or result.consensus_sells or result.new_positions):
            print("      ✓ No major ARK trades detected today")
            
        return summary

    except Exception as e:
        log_error(logger, "ARK holdings analysis failed", e)
        return {}


# ============================================================================
# NEW: Bubble, Sentiment, Validation (2026-01-29 통합 작업)
# ============================================================================

def analyze_bubble_risk(market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    버블 리스크 분석 (Bubbles for Fama 기반)

    기능:
    - 주요 자산의 Run-up 체크 (2년 누적 수익률 > 100%)
    - 변동성 급등 탐지
    - 주식 발행 증가 분석

    References:
    - Greenwood, Shleifer & You (2019)
    """
    print("\n[2.22] Bubble Risk Analysis (Bubbles for Fama)...")
    try:
        from lib.bubble_detector import BubbleDetector

        detector = BubbleDetector()
        results = {
            'overall_status': 'NONE',
            'risk_tickers': [],
            'highest_risk_ticker': '',
            'highest_risk_score': 0.0,
            'methodology_notes': 'Bubbles for Fama (2019)'
        }

        # 주요 자산 분석
        tickers_to_check = ['SPY', 'QQQ', 'IWM', 'ARKK']
        risk_tickers = []
        highest_score = 0.0
        highest_ticker = ''

        for ticker in tickers_to_check:
            if ticker in market_data and not market_data[ticker].empty:
                df = market_data[ticker]
                if 'Close' in df.columns:
                    try:
                        # analyze()는 DataFrame을 기대함 (Close 컬럼 포함)
                        detection = detector.analyze(ticker, price_data=df)
                        if detection.bubble_warning_level.value != "NONE":
                            runup_val = 0.0
                            if detection.runup and hasattr(detection.runup, 'cumulative_return'):
                                runup_val = detection.runup.cumulative_return
                            risk_tickers.append({
                                'ticker': ticker,
                                'warning_level': detection.bubble_warning_level.value,
                                'risk_score': detection.risk_score,
                                'runup': runup_val
                            })
                            if detection.risk_score > highest_score:
                                highest_score = detection.risk_score
                                highest_ticker = ticker
                    except Exception as ex:
                        logger.warning(f"Bubble detection failed for {ticker}: {ex}")

        # 전체 상태 결정
        if highest_score >= 70:
            results['overall_status'] = 'DANGER'
        elif highest_score >= 50:
            results['overall_status'] = 'WARNING'
        elif highest_score >= 30:
            results['overall_status'] = 'WATCH'

        results['risk_tickers'] = risk_tickers
        results['highest_risk_ticker'] = highest_ticker
        results['highest_risk_score'] = highest_score

        print(f"      ✓ Bubble Status: {results['overall_status']}")
        if risk_tickers:
            print(f"      ⚠ Risk Tickers: {', '.join([t['ticker'] for t in risk_tickers])}")

        return results
    except Exception as e:
        log_error(logger, "Bubble risk analysis failed", e)
        return {}


def analyze_sentiment() -> Dict[str, Any]:
    """
    센티먼트 분석 (Fear & Greed, VIX 구조, 뉴스)

    기능:
    - Fear & Greed Index 수집
    - VIX Term Structure 분석
    - 뉴스 센티먼트 분석

    References:
    - CNN Fear & Greed Index
    """
    print("\n[2.23] Sentiment Analysis...")
    try:
        from lib.sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()
        results = {}

        # Fear & Greed
        try:
            fg = analyzer.fetch_fear_greed_index()
            if fg:
                results['fear_greed'] = {
                    'value': fg.value,
                    'level': fg.level.value,
                    'previous_close': fg.previous_close,
                    'week_ago': fg.week_ago
                }
                print(f"      ✓ Fear & Greed: {fg.value} ({fg.level.value})")
        except Exception as ex:
            logger.warning(f"Fear & Greed fetch failed: {ex}")

        # VIX Term Structure
        try:
            vix = analyzer.analyze_vix_term_structure()
            if vix:
                results['vix_structure'] = {
                    'vix_spot': vix.vix_spot,
                    'structure': vix.structure.value,
                    'contango_ratio': vix.contango_ratio,
                    'signal': vix.signal
                }
                print(f"      ✓ VIX Structure: {vix.structure.value} (Signal: {vix.signal})")
        except Exception as ex:
            logger.warning(f"VIX structure analysis failed: {ex}")

        # News Sentiment
        try:
            news = analyzer.fetch_news_sentiment('SPY')
            if news:
                avg_score = sum(n.sentiment_score for n in news) / len(news)
                results['news_sentiment'] = {
                    'avg_score': avg_score,
                    'count': len(news),
                    'overall': 'BULLISH' if avg_score > 0.2 else 'BEARISH' if avg_score < -0.2 else 'NEUTRAL'
                }
                print(f"      ✓ News Sentiment: {results['news_sentiment']['overall']} (n={len(news)})")
        except Exception as ex:
            logger.warning(f"News sentiment fetch failed: {ex}")

        return results
    except Exception as e:
        log_error(logger, "Sentiment analysis failed", e)
        return {}


def run_ai_validation(result_data: Dict, use_cache: bool = True) -> Dict[str, Any]:
    """
    AI 기반 투자 전략 검증 (Multi-LLM)

    기능:
    - Claude, Gemini, Perplexity, OpenAI 4개 AI의 독립 검증
    - 합의 도출 및 신뢰도 계산
    - 24시간 캐싱으로 API 비용 절감

    References:
    - validation_agents.py
    """
    print("\n[2.24] AI Validation (Multi-LLM)...")
    try:
        import os
        import json
        from datetime import datetime, timedelta
        from pathlib import Path

        # 캐시 확인
        cache_path = Path("outputs/.validation_cache.json")
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
                cache_time = datetime.fromisoformat(cache['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=24):
                    print("      ✓ Using cached validation result (< 24h old)")
                    return cache['result']
            except Exception:
                pass

        # API 키 확인
        has_apis = all([
            os.getenv('ANTHROPIC_API_KEY'),
            os.getenv('OPENAI_API_KEY'),
        ])
        
        if not has_apis:
            print("      ⚠ AI Validation skipped: Missing API keys")
            return {'status': 'SKIPPED', 'reason': 'Missing API keys'}

        from lib.validation_agents import ValidationAgentManager

        manager = ValidationAgentManager()
        
        # 검증 실행
        agent_decision = {
            'recommendation': result_data.get('final_recommendation', 'HOLD'),
            'confidence': result_data.get('confidence', 0.5),
            'risk_level': result_data.get('risk_level', 'MEDIUM'),
        }
        market_condition = {
            'regime': result_data.get('regime', {}),
            'risk_score': result_data.get('risk_score', 50),
        }

        consensus = manager.validate_all(agent_decision, market_condition)
        
        validation_result = {
            'final_result': consensus.final_result.value,
            'consensus_confidence': consensus.consensus_confidence,
            'agreement_ratio': consensus.agreement_ratio,
            'key_concerns': consensus.key_concerns,
            'action_items': consensus.action_items,
            'summary': consensus.summary
        }

        # 캐시 저장
        if use_cache:
            try:
                cache_path.parent.mkdir(exist_ok=True)
                with open(cache_path, 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'result': validation_result
                    }, f)
            except Exception:
                pass

        print(f"      ✓ AI Consensus: {consensus.final_result.value}")
        print(f"      ✓ Agreement: {consensus.agreement_ratio:.0%}")

        return validation_result
    except Exception as e:
        log_error(logger, "AI validation failed", e)
        return {'status': 'ERROR', 'error': str(e)}