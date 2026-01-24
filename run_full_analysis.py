#!/usr/bin/env python3
"""
EIMAS 전체 분석 실행 (실제 시장 데이터)
======================================
2026-01-25 통합 버전

실행하는 분석:
- Phase 2.14: HFT 미세구조
- Phase 2.15: GARCH 변동성
- Phase 2.16: 정보 플로우
- Phase 2.17: Proof-of-Index
- Phase 2.18: Systemic Similarity
- Phase 2.19: DBSCAN Outlier Detection (NEW)
- Phase 2.20: DTW Time Series Similarity (NEW)

실제 데이터:
- yfinance로 SPY, QQQ, TLT, GLD, BTC-USD 다운로드
- 최근 1년 데이터 사용
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EIMAS 전체 분석 실행 (실제 시장 데이터)")
print("=" * 80)
print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


# ============================================================================
# 실제 시장 데이터 다운로드
# ============================================================================

def download_market_data():
    """yfinance로 실제 시장 데이터 다운로드"""
    print("\n[0] Downloading real market data...")

    try:
        import yfinance as yf
    except ImportError:
        print("   ❌ yfinance not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
        import yfinance as yf

    # 다운로드할 티커
    tickers = [
        'SPY',    # S&P 500
        'QQQ',    # Nasdaq 100
        'TLT',    # 20Y Treasury
        'GLD',    # Gold
        'BTC-USD', # Bitcoin
        'IWM',    # Russell 2000
        'EFA',    # EAFE
        'EEM',    # Emerging Markets
        'HYG',    # High Yield Corporate
        'LQD',    # Investment Grade Corporate
    ]

    # 기간: 최근 1년
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    market_data = {}

    for ticker in tickers:
        try:
            print(f"   Downloading {ticker}...", end=" ")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if not df.empty:
                market_data[ticker] = df
                print(f"✓ ({len(df)} days)")
            else:
                print(f"⚠️ No data")
        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n   ✅ Downloaded {len(market_data)} tickers")
    print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    return market_data


# ============================================================================
# Phase 2.14: HFT Microstructure
# ============================================================================

def run_hft_microstructure(market_data):
    """HFT 미세구조 분석"""
    print("\n" + "=" * 80)
    print("[Phase 2.14] HFT Microstructure Analysis")
    print("=" * 80)

    from pipeline.analyzers import analyze_hft_microstructure

    try:
        result = analyze_hft_microstructure(market_data)

        if result:
            print("\n✅ HFT Microstructure analysis completed")

            # 주요 결과 출력
            if 'kyle_lambda' in result:
                print(f"\nKyle's Lambda Results:")
                for ticker, data in list(result['kyle_lambda'].items())[:3]:
                    print(f"  {ticker}: {data.get('interpretation', 'N/A')}")

            return result
        else:
            print("⚠️ No results")
            return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Phase 2.15: GARCH Volatility
# ============================================================================

def run_garch_volatility(market_data):
    """GARCH 변동성 모델링"""
    print("\n" + "=" * 80)
    print("[Phase 2.15] GARCH Volatility Modeling")
    print("=" * 80)

    from pipeline.analyzers import analyze_volatility_garch

    try:
        result = analyze_volatility_garch(market_data)

        if result:
            print("\n✅ GARCH volatility analysis completed")

            # 주요 결과 출력
            if 'models' in result:
                print(f"\nGARCH Models Fitted:")
                for ticker, data in list(result['models'].items())[:3]:
                    params = data.get('parameters', {})
                    print(f"  {ticker}: α={params.get('alpha', 0):.4f}, β={params.get('beta', 0):.4f}")

            return result
        else:
            print("⚠️ No results")
            return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Phase 2.16: Information Flow
# ============================================================================

def run_information_flow(market_data):
    """정보 플로우 분석"""
    print("\n" + "=" * 80)
    print("[Phase 2.16] Information Flow Analysis")
    print("=" * 80)

    from pipeline.analyzers import analyze_information_flow

    try:
        result = analyze_information_flow(market_data)

        if result:
            print("\n✅ Information Flow analysis completed")

            # 주요 결과 출력
            if 'abnormal_volume' in result:
                ab_vol = result['abnormal_volume']
                print(f"\nAbnormal Volume Analysis:")
                print(f"  Total Abnormal Days: {ab_vol.get('total_abnormal_days', 0)}")
                print(f"  Abnormal Ratio: {ab_vol.get('abnormal_ratio', 0):.1%}")
                print(f"  Interpretation: {ab_vol.get('interpretation', 'N/A')}")

            return result
        else:
            print("⚠️ No results")
            return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Phase 2.17: Proof-of-Index
# ============================================================================

def run_proof_of_index(market_data):
    """Proof-of-Index 계산"""
    print("\n" + "=" * 80)
    print("[Phase 2.17] Proof-of-Index (Blockchain Transparency)")
    print("=" * 80)

    from pipeline.analyzers import calculate_proof_of_index

    try:
        result = calculate_proof_of_index(market_data)

        if result:
            print("\n✅ Proof-of-Index completed")

            # 주요 결과 출력
            if 'index_snapshot' in result:
                snapshot = result['index_snapshot']
                print(f"\nIndex Value: ${snapshot.get('index_value', 0):,.2f}")
                print(f"Hash Verification: {result.get('verification', {}).get('is_valid', False)}")

            return result
        else:
            print("⚠️ No results")
            return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Phase 2.18: Systemic Similarity
# ============================================================================

def run_systemic_similarity(market_data):
    """Systemic Similarity 분석"""
    print("\n" + "=" * 80)
    print("[Phase 2.18] Systemic Similarity Analysis")
    print("=" * 80)

    from pipeline.analyzers import enhance_portfolio_with_systemic_similarity

    try:
        result = enhance_portfolio_with_systemic_similarity(market_data)

        if result:
            print("\n✅ Systemic Similarity completed")

            # 주요 결과 출력
            if 'most_similar_pair' in result:
                pair = result['most_similar_pair']
                print(f"\nMost Similar: {pair['assets'][0]} ↔ {pair['assets'][1]}")
                print(f"Similarity: {pair['similarity']:.3f}")

            return result
        else:
            print("⚠️ No results")
            return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Phase 2.19: DBSCAN Outlier Detection (NEW)
# ============================================================================

def run_dbscan_outliers(market_data):
    """DBSCAN 이상치 탐지"""
    print("\n" + "=" * 80)
    print("[Phase 2.19] DBSCAN Outlier Detection (NEW)")
    print("=" * 80)

    from pipeline.analyzers import detect_outliers_with_dbscan

    try:
        result = detect_outliers_with_dbscan(market_data)

        if result:
            print("\n✅ DBSCAN Outlier Detection completed")

            # 주요 결과 출력
            print(f"\nOutliers: {result.get('n_outliers', 0)}/{result.get('n_total_assets', 0)}")
            print(f"Clusters: {result.get('n_clusters', 0)}")
            print(f"Interpretation: {result.get('interpretation', 'N/A')}")

            if result.get('outlier_tickers'):
                print(f"\nOutlier Tickers (first 5):")
                for ticker in result['outlier_tickers'][:5]:
                    print(f"  - {ticker}")

            return result
        else:
            print("⚠️ No results")
            return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Phase 2.20: DTW Time Series Similarity (NEW)
# ============================================================================

def run_dtw_similarity(market_data):
    """DTW 시계열 유사도 분석"""
    print("\n" + "=" * 80)
    print("[Phase 2.20] DTW Time Series Similarity (NEW)")
    print("=" * 80)

    from pipeline.analyzers import analyze_dtw_similarity

    try:
        result = analyze_dtw_similarity(market_data)

        if result:
            print("\n✅ DTW Similarity analysis completed")

            # 주요 결과 출력
            print(f"\nAssets Analyzed: {result.get('n_series', 0)}")
            print(f"Avg DTW Distance: {result.get('avg_distance', 0):.4f}")

            most_sim = result.get('most_similar_pair', {})
            if most_sim:
                print(f"\nMost Similar: {most_sim.get('asset1', 'N/A')} ↔ {most_sim.get('asset2', 'N/A')}")
                print(f"DTW Distance: {most_sim.get('distance', 0):.4f}")

            # Lead-Lag 결과
            lead_lag = result.get('lead_lag_spy_qqq', {})
            if lead_lag:
                print(f"\nLead-Lag (SPY vs QQQ):")
                print(f"  {lead_lag.get('interpretation', 'N/A')}")

            return result
        else:
            print("⚠️ No results")
            return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """메인 실행"""

    # 1. 시장 데이터 다운로드
    market_data = download_market_data()

    if not market_data:
        print("\n❌ No market data available")
        return 1

    # 결과 저장
    results = {
        'timestamp': datetime.now().isoformat(),
        'tickers': list(market_data.keys()),
        'analyses': {}
    }

    # 2. 분석 실행
    results['analyses']['hft_microstructure'] = run_hft_microstructure(market_data)
    results['analyses']['garch_volatility'] = run_garch_volatility(market_data)
    results['analyses']['information_flow'] = run_information_flow(market_data)
    results['analyses']['proof_of_index'] = run_proof_of_index(market_data)
    results['analyses']['systemic_similarity'] = run_systemic_similarity(market_data)
    results['analyses']['dbscan_outliers'] = run_dbscan_outliers(market_data)
    results['analyses']['dtw_similarity'] = run_dtw_similarity(market_data)

    # 3. 최종 요약
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    completed = sum(1 for v in results['analyses'].values() if v)
    total = len(results['analyses'])

    print(f"\nCompleted: {completed}/{total} analyses")

    for name, data in results['analyses'].items():
        status = "✅" if data else "❌"
        print(f"{status} {name}")

    # 4. 결과 저장
    import json
    from pathlib import Path

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"full_analysis_{timestamp}.json"

    # JSON 직렬화 가능하도록 변환
    serializable_results = {
        'timestamp': results['timestamp'],
        'tickers': results['tickers'],
        'analyses': {}
    }

    for key, value in results['analyses'].items():
        if value:
            serializable_results['analyses'][key] = {
                'completed': True,
                'summary': str(value)[:200] + '...' if len(str(value)) > 200 else str(value)
            }
        else:
            serializable_results['analyses'][key] = {'completed': False}

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")

    print("\n" + "=" * 80)
    if completed == total:
        print("🎉 모든 분석이 성공적으로 완료되었습니다!")
    else:
        print(f"⚠️ {total - completed}개 분석 실패")
    print("=" * 80)

    return 0 if completed == total else 1


if __name__ == "__main__":
    sys.exit(main())
