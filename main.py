#!/usr/bin/env python3
"""
EIMAS - Economic Intelligence Multi-Agent System
=================================================
통합 실행 파이프라인 (Unified Main Entry)

Modular Pipeline Architecture:
1. Collectors: 데이터 수집 (FRED, Market, Crypto)
2. Analyzers: 시장 분석 (Regime, Risk, Liquidity, HFT, PoI, DTW, DBSCAN)
3. Debate: AI 에이전트 토론 (Dual Mode)
4. Realtime: 실시간 스트리밍 (선택적)
5. Storage: 결과 저장 (JSON, DB)
6. Report: AI 리포트 생성 (IB Style & Standard)

Usage:
    python main.py              # 전체 파이프라인
    python main.py --realtime   # 실시간 스트리밍 포함
    python main.py --quick      # 빠른 분석만
    python main.py --report     # AI 리포트 생성 포함
"""

import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# Pipeline Modules
from pipeline import (
    EIMASResult,
    collect_fred_data, collect_market_data,
    collect_crypto_data, collect_market_indicators,
    detect_regime, detect_events, analyze_liquidity,
    analyze_critical_path, analyze_etf_flow, generate_explanation,
    analyze_genius_act, analyze_theme_etf, analyze_shock_propagation,
    optimize_portfolio_mst, analyze_volume_anomalies, track_events_with_news,
    run_adaptive_portfolio,
    run_dual_mode_debate, run_realtime_stream,
    save_result_json, save_result_md, save_to_trading_db, save_to_event_db,
    generate_ai_report, run_whitening_check, run_fact_check
)

# New Enhanced Analyzers (Phase 2.Enhanced)
from pipeline.analyzers import (
    analyze_hft_microstructure, analyze_volatility_garch,
    analyze_information_flow, calculate_proof_of_index,
    enhance_portfolio_with_systemic_similarity,
    detect_outliers_with_dbscan, analyze_dtw_similarity,
    analyze_ark_trades
)

async def run_integrated_pipeline(
    enable_realtime: bool = False,
    realtime_duration: int = 30,
    quick_mode: bool = False,
    generate_report: bool = False
) -> EIMASResult:
    """EIMAS 통합 파이프라인 실행"""
    start_time = datetime.now()
    result = EIMASResult(timestamp=start_time.isoformat())

    print("=" * 70)
    print("  EIMAS - Integrated Analysis Pipeline (Unified Main)")
    print("=" * 70)

    # Phase 1: Data Collection
    print("\n[Phase 1] Collecting Data...")
    result.fred_summary = collect_fred_data()
    market_data = collect_market_data(lookback_days=90 if quick_mode else 365)
    result.market_data_count = len(market_data)
    
    # Crypto & Indicators
    collect_crypto_data() # DataManager 캐시에 저장됨
    if not quick_mode:
        collect_market_indicators()

    # Phase 2: Analysis
    print("\n[Phase 2] Analyzing Market...")
    regime_res = detect_regime()
    result.regime = regime_res.to_dict()
    
    events = detect_events(result.fred_summary, market_data)
    result.events_detected = [e.to_dict() for e in events]
    
    # Phase 2.Enhanced: New Metrics (HFT, PoI, DTW, etc.)
    print("\n[Phase 2.Enhanced] Running Advanced Metrics...")
    
    try:
        result.hft_microstructure = analyze_hft_microstructure(market_data)
    except Exception as e:
        print(f"⚠️ HFT Analysis Error: {e}")

    try:
        result.garch_volatility = analyze_volatility_garch(market_data)
    except Exception as e:
        print(f"⚠️ GARCH Analysis Error: {e}")

    try:
        result.information_flow = analyze_information_flow(market_data)
    except Exception as e:
        print(f"⚠️ Information Flow Analysis Error: {e}")

    try:
        result.proof_of_index = calculate_proof_of_index(market_data)
    except Exception as e:
        print(f"⚠️ Proof-of-Index Error: {e}")
        
    try:
        result.ark_analysis = analyze_ark_trades()
    except Exception as e:
        print(f"⚠️ ARK Analysis Error: {e}")
    
    # Systemic Similarity
    try:
        sim_res = enhance_portfolio_with_systemic_similarity(market_data)
        # result 객체에 별도 필드가 없다면 딕셔너리로 저장하거나 무시
        # 현재는 enhance_portfolio_with_systemic_similarity가 반환하는 값 중 필요한 것만 사용
        pass 
    except Exception as e:
        print(f"⚠️ Systemic Similarity Error: {e}")

    if not quick_mode:
        try:
            result.dtw_similarity = analyze_dtw_similarity(market_data)
        except Exception as e:
            print(f"⚠️ DTW Analysis Error: {e}")

        try:
            result.dbscan_outliers = detect_outliers_with_dbscan(market_data)
        except Exception as e:
            print(f"⚠️ DBSCAN Analysis Error: {e}")
        
        try:
            liq_res = analyze_liquidity()
            result.liquidity_signal = liq_res.signal
            result.liquidity_analysis = liq_res.to_dict()
        except Exception as e:
            print(f"⚠️ Liquidity Analysis Error: {e}")
        
        try:
            etf_res = analyze_etf_flow()
            result.etf_flow_result = etf_res.to_dict()
        except Exception as e:
            print(f"⚠️ ETF Flow Analysis Error: {e}")
        
        # Advanced Analysis
        try:
            ga_res = analyze_genius_act()
            result.genius_act_signals = ga_res.signals
            result.genius_act_regime = ga_res.regime
        except Exception as e:
            print(f"⚠️ Genius Act Error: {e}")
        
        try:
            theme_res = analyze_theme_etf()
            result.theme_etf_analysis = theme_res.to_dict()
        except Exception as e:
            print(f"⚠️ Theme ETF Error: {e}")
        
        try:
            shock_res = analyze_shock_propagation(market_data)
            result.shock_propagation = shock_res.to_dict()
        except Exception as e:
            print(f"⚠️ Shock Propagation Error: {e}")
        
        try:
            port_res = optimize_portfolio_mst(market_data)
            result.portfolio_weights = port_res.weights
        except Exception as e:
            print(f"⚠️ Portfolio Optimization Error: {e}")
        
        # Microstructure & Event Tracking
        try:
            vol_anomalies = analyze_volume_anomalies(market_data)
            result.volume_anomalies = vol_anomalies
        except Exception as e:
            print(f"⚠️ Volume Analysis Error: {e}")
        
        try:
            tracked_events = await track_events_with_news(market_data)
            result.event_tracking = tracked_events
        except Exception as e:
            print(f"⚠️ Event Tracking Error: {e}")
        
        try:
            expl_res = generate_explanation(market_data)
            # result.market_explanation = expl_res 
        except Exception as e:
            print(f"⚠️ Explanation Gen Error: {e}")

    try:
        cp_res = analyze_critical_path(market_data)
        result.risk_score = cp_res.risk_score
        result.base_risk_score = cp_res.risk_score
    except Exception as e:
        print(f"⚠️ Critical Path Error: {e}")

    # Phase 3: Debate
    print("\n[Phase 3] Running AI Debate...")
    try:
        debate_res = await run_dual_mode_debate(market_data)
        result.full_mode_position = debate_res.full_mode_position
        result.reference_mode_position = debate_res.reference_mode_position
        result.modes_agree = debate_res.modes_agree
        result.final_recommendation = debate_res.final_recommendation
        result.confidence = debate_res.confidence
        result.risk_level = debate_res.risk_level
        result.warnings.extend(debate_res.warnings)
    except Exception as e:
        print(f"⚠️ Debate Error: {e}")
    
    # Adaptive Portfolio
    if not quick_mode:
        try:
            adaptive_alloc = run_adaptive_portfolio(regime_res)
            result.adaptive_portfolios = adaptive_alloc
        except Exception as e:
            print(f"⚠️ Adaptive Portfolio Error: {e}")

    # Phase 4: Realtime (Optional)
    if enable_realtime:
        print("\n[Phase 4] Realtime Streaming...")
        signals = await run_realtime_stream(duration=realtime_duration)
        result.realtime_signals = [s.to_dict() for s in signals]
        save_to_trading_db(signals) 

    # Phase 5: Storage
    print("\n[Phase 5] Saving Results...")
    save_to_event_db(events)
    output_file = save_result_json(result)
    save_result_md(result)

    # Phase 6: Report (Optional)
    report_content = ""
    if generate_report:
        print("\n[Phase 6] Generating Report...")
        try:
            ai_report = await generate_ai_report(result, market_data)
            report_content = ai_report.content
        except Exception as e:
            print(f"⚠️ Report Generation Error: {e}")

    # Phase 7: Validation (Whitening & Fact Check)
    if generate_report:
        print("\n" + "=" * 50)
        print("PHASE 7: VALIDATION")
        print("=" * 50)
        
        try:
            whitening = run_whitening_check(result)
            result.whitening_summary = whitening
            
            if report_content:
                fact_grade = await run_fact_check(report_content)
                result.fact_check_grade = fact_grade
                
            # Update JSON with validation results
            save_result_json(result)
        except Exception as e:
            print(f"⚠️ Validation Error: {e}")

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "=" * 70)
    print(f"EIMAS PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"Files: {output_file}")
    print("=" * 70)

    return result

# ============================================================================ 
# CLI Entry Point
# ============================================================================ 

def main():
    parser = argparse.ArgumentParser(description='EIMAS Modular Pipeline')
    parser.add_argument('--realtime', '-r', action='store_true', help='Enable realtime stream')
    parser.add_argument('--duration', '-d', type=int, default=30, help='Stream duration')
    # --short 옵션: 리포트 생성 안 함, 분석 기간 단축
    parser.add_argument('--short', '-s', action='store_true', help='Short analysis mode (no report, quick data)')
    # --quick은 하위 호환성을 위해 유지하되, --short와 동일하게 동작하도록 처리 가능
    parser.add_argument('--quick', '-q', action='store_true', help='Alias for --short')
    
    args = parser.parse_args()
    
    # short 또는 quick 모드이면 quick_mode=True, report=False
    is_short_mode = args.short or args.quick
    generate_report = not is_short_mode  # 기본적으로 리포트 생성 (short 모드가 아닐 때)
    
    asyncio.run(run_integrated_pipeline(
        enable_realtime=args.realtime,
        realtime_duration=args.duration,
        quick_mode=is_short_mode,
        generate_report=generate_report
    ))

if __name__ == "__main__":
    main()
