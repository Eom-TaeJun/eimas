#!/usr/bin/env python3
"""
EIMAS - Economic Intelligence Multi-Agent System
=================================================
Main entry point with full pipeline integration.

Features:
- DataManager로 시장 데이터 수집
- ForecastAgent로 LASSO 예측
- AnalysisAgent로 Critical Path 분석
- MetaOrchestrator로 멀티에이전트 토론 및 합의
- VisualizationAgent로 대시보드 생성

Usage:
    python main.py
    python main.py --config configs/default.yaml
    python main.py --output-dir outputs/dashboards
"""

import argparse
import asyncio
import json
import logging
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from core.schemas import (
    AgentRequest,
    AgentRole,
    DashboardConfig,
    ForecastResult,
    AnalysisMode,
    HistoricalDataConfig,
)
from agents.orchestrator import MetaOrchestrator
from agents.forecast_agent import ForecastAgent
from agents.visualization_agent import VisualizationAgent
from lib.data_collector import DataManager, UnifiedDataCollector
from lib.dual_mode_analyzer import DualModeAnalyzer, ModeResult, DualModeComparison
from lib.fred_collector import FREDCollector

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('eimas.main')


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    YAML 설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """기본 설정 반환"""
    return {
        'data': {
            'lookback_days': 365,
            'tickers_config': 'configs/tickers.yaml'
        },
        'agents': {
            'forecast': {
                'enabled': True,
                'n_splits': 5,
                'max_iter': 10000
            },
            'analysis': {
                'enabled': True
            },
            'visualization': {
                'enabled': True,
                'theme': 'dark',
                'language': 'ko'
            }
        },
        'debate': {
            'max_rounds': 3,
            'consensus_threshold': 0.85
        },
        'output': {
            'dir': 'outputs/dashboards',
            'save_json': True,
            'generate_dashboard': True
        }
    }


# ============================================================================
# Full Pipeline
# ============================================================================

async def run_full_pipeline(config_path: str = 'configs/default.yaml') -> str:
    """
    전체 EIMAS 파이프라인 실행
    
    1. DataManager로 데이터 수집
    2. AnalysisAgent로 Critical Path 분석
    3. ForecastAgent로 LASSO 예측
    4. MetaOrchestrator로 토론 및 합의
    5. VisualizationAgent로 대시보드 생성
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        생성된 대시보드 파일 경로
    """
    start_time = datetime.now()
    logger.info("="*70)
    logger.info("  EIMAS - Full Pipeline Execution")
    logger.info("="*70)
    
    # 1. 설정 로드
    logger.info("\n[1/5] Loading configuration...")
    config = load_config(config_path)
    
    # 2. 데이터 수집
    logger.info("\n[2/5] Collecting market data...")
    
    tickers_config_path = config.get('data', {}).get('tickers_config', 'configs/tickers.yaml')
    lookback_days = config.get('data', {}).get('lookback_days', 365)
    
    tickers_config = load_tickers_config(tickers_config_path)
    
    dm = DataManager(lookback_days=lookback_days)
    market_data, macro_data = dm.collect_all(tickers_config)
    
    logger.info(f"   → Collected {len(market_data)} tickers")
    logger.info(f"   → Collected {len(macro_data.columns)} macro series")
    
    # 3. ForecastAgent 실행 (LASSO 예측)
    forecast_results = []
    forecast_diagnostics = {}
    if config.get('agents', {}).get('forecast', {}).get('enabled', True):
        logger.info("\n[3/5] Running LASSO forecast...")
        forecast_results, forecast_diagnostics = await run_forecast_agent(market_data, macro_data, config)
        logger.info(f"   → Generated {len(forecast_results)} horizon forecasts")
        logger.info(f"   → Diagnostics: {forecast_diagnostics}")
    else:
        logger.info("\n[3/5] LASSO forecast skipped (disabled)")
    
    # 4. 멀티에이전트 토론 및 합의
    logger.info("\n[4/5] Running multi-agent debate...")
    orchestrator = MetaOrchestrator(verbose=True)
    
    query = "Analyze current market conditions and forecast Fed policy"
    debate_result = await orchestrator.run_with_debate(query, market_data)
    
    # 에이전트 의견 및 합의 추출
    agent_opinions = debate_result.get('debate', {}).get('opinions', [])
    consensus = debate_result.get('debate', {}).get('consensus')
    conflicts = debate_result.get('debate', {}).get('conflicts', [])
    
    logger.info(f"   → Collected {len(agent_opinions)} agent opinions")
    if consensus:
        logger.info(f"   → Consensus reached: {consensus.get('final_position', 'N/A')}")
    
    # 5. 대시보드 생성
    dashboard_path = ""
    if config.get('output', {}).get('generate_dashboard', True):
        logger.info("\n[5/5] Generating dashboard...")
        dashboard_path = await run_visualization_agent(
            forecast_results=forecast_results,
            debate_result=debate_result,
            agent_opinions=agent_opinions,
            consensus=consensus,
            conflicts=conflicts,
            config=config,
            forecast_diagnostics=forecast_diagnostics
        )
        logger.info(f"   → Dashboard saved: {dashboard_path}")
    else:
        logger.info("\n[5/5] Dashboard generation skipped (disabled)")
    
    # 6. JSON 결과 저장
    if config.get('output', {}).get('save_json', True):
        json_path = save_json_result(
            forecast_results=forecast_results,
            debate_result=debate_result,
            config=config
        )
        logger.info(f"   → JSON saved: {json_path}")
    
    # 완료 요약
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETED")
    logger.info("="*70)
    logger.info(f"Total time: {elapsed:.1f}s")
    
    if debate_result.get('analysis'):
        logger.info(f"Risk Score: {debate_result['analysis'].get('total_risk_score', 0):.1f}/100")
        logger.info(f"Regime: {debate_result['analysis'].get('current_regime', 'Unknown')}")
    
    if forecast_results:
        long_result = forecast_results[-1] if forecast_results else {}
        if isinstance(long_result, dict):
            r_squared = long_result.get('r_squared', 0)
        else:
            r_squared = getattr(long_result, 'r_squared', 0)
        logger.info(f"Long Horizon R²: {r_squared:.4f}")
    
    logger.info(f"\nDashboard: {dashboard_path}")

    return dashboard_path


# ============================================================================
# Dual Mode Pipeline (FULL vs REFERENCE)
# ============================================================================

async def run_dual_mode_pipeline(config_path: str = 'configs/default.yaml') -> Dict:
    """
    두 가지 분석 모드를 병렬로 실행하고 비교

    Mode 1 (FULL): 역사적 데이터 70% 가중치 - 최신 패턴 반영
    Mode 2 (REFERENCE): 역사적 데이터 20% 가중치 - Regime 변화에 강건

    Args:
        config_path: 설정 파일 경로

    Returns:
        비교 분석 결과
    """
    start_time = datetime.now()

    print("=" * 70)
    print("  EIMAS - Dual Mode Analysis")
    print("  Comparing FULL vs REFERENCE modes")
    print("=" * 70)

    # 1. 설정 로드
    print("\n[1/6] Loading configuration...")
    config = load_config(config_path)

    # 2. 데이터 수집
    print("\n[2/6] Collecting market data...")

    tickers_config_path = config.get('data', {}).get('tickers_config', 'configs/tickers.yaml')
    lookback_days = config.get('data', {}).get('lookback_days', 365)
    tickers_config = load_tickers_config(tickers_config_path)

    dm = DataManager(lookback_days=lookback_days)
    market_data, macro_data = dm.collect_all(tickers_config)
    print(f"   → Collected {len(market_data)} tickers, {len(macro_data.columns)} macro series")

    # 3. FRED 유동성 데이터 수집
    print("\n[3/6] Collecting FRED liquidity data...")
    try:
        fred = FREDCollector()
        fred_summary = fred.collect_all()
        print(f"   → RRP: ${fred_summary.rrp:.0f}B, TGA: ${fred_summary.tga:.0f}B")
        print(f"   → Net Liquidity: ${fred_summary.net_liquidity:.0f}B ({fred_summary.liquidity_regime})")
    except Exception as e:
        print(f"   → FRED collection failed: {e}")
        fred_summary = None

    # 4. MODE 1: FULL (preserve_dissent=True)
    print("\n[4/6] Running MODE 1: FULL (Historical 70%)...")
    print("-" * 50)

    from core.debate import DebateProtocol

    # Orchestrator with preserve_dissent
    orchestrator_full = MetaOrchestrator(verbose=True)
    # DebateProtocol은 내부에서 생성되므로, 설정을 전달할 방법이 필요
    # 일단 기본값이 preserve_dissent=True로 변경되었으므로 그대로 진행

    query = "Analyze current market conditions and forecast Fed policy"
    result_full = await orchestrator_full.run_with_debate(query, market_data)

    # 결과 추출
    consensus_full = result_full.get('debate', {}).get('consensus', {})
    full_mode_result = ModeResult(
        mode=AnalysisMode.FULL,
        consensus=None,
        confidence=consensus_full.get('confidence', 0.5) if isinstance(consensus_full, dict) else 0.5,
        position=consensus_full.get('final_position', 'NEUTRAL') if isinstance(consensus_full, dict) else 'NEUTRAL',
        dissent_count=len(consensus_full.get('dissenting_agents', [])) if isinstance(consensus_full, dict) else 0,
        has_strong_dissent=consensus_full.get('has_strong_dissent', False) if isinstance(consensus_full, dict) else False,
        warnings=[]
    )

    print(f"\n   FULL Mode Result: {full_mode_result.position} (conf: {full_mode_result.confidence:.0%})")

    # 5. MODE 2: REFERENCE (실시간 데이터 더 중시)
    print("\n[5/6] Running MODE 2: REFERENCE (Historical 20%)...")
    print("-" * 50)

    # Reference 모드: 더 짧은 lookback으로 최신 데이터 중시
    dm_ref = DataManager(lookback_days=90)  # 90일만 사용
    market_data_ref, macro_data_ref = dm_ref.collect_all(tickers_config)

    orchestrator_ref = MetaOrchestrator(verbose=True)
    result_ref = await orchestrator_ref.run_with_debate(query, market_data_ref)

    consensus_ref = result_ref.get('debate', {}).get('consensus', {})
    reference_mode_result = ModeResult(
        mode=AnalysisMode.REFERENCE,
        consensus=None,
        confidence=consensus_ref.get('confidence', 0.5) if isinstance(consensus_ref, dict) else 0.5,
        position=consensus_ref.get('final_position', 'NEUTRAL') if isinstance(consensus_ref, dict) else 'NEUTRAL',
        dissent_count=len(consensus_ref.get('dissenting_agents', [])) if isinstance(consensus_ref, dict) else 0,
        has_strong_dissent=consensus_ref.get('has_strong_dissent', False) if isinstance(consensus_ref, dict) else False,
        warnings=[]
    )

    print(f"\n   REFERENCE Mode Result: {reference_mode_result.position} (conf: {reference_mode_result.confidence:.0%})")

    # 6. 두 모드 비교
    print("\n[6/6] Comparing modes...")
    print("-" * 50)

    analyzer = DualModeAnalyzer()
    comparison = analyzer.compare_modes(full_mode_result, reference_mode_result)

    # 리포트 출력
    print(analyzer.generate_dual_report(comparison))

    # 완료 요약
    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 70)
    print("DUAL MODE ANALYSIS COMPLETED")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s")
    print(f"\nFinal Recommendation: {comparison.recommended_action}")
    print(f"Risk Level: {comparison.risk_level}")

    # 결과 저장
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"dual_mode_{timestamp}.json"

    result = {
        'timestamp': timestamp,
        'full_mode': {
            'position': full_mode_result.position,
            'confidence': full_mode_result.confidence,
            'dissent_count': full_mode_result.dissent_count,
            'has_strong_dissent': full_mode_result.has_strong_dissent,
            'analysis': result_full.get('analysis', {}),
        },
        'reference_mode': {
            'position': reference_mode_result.position,
            'confidence': reference_mode_result.confidence,
            'dissent_count': reference_mode_result.dissent_count,
            'has_strong_dissent': reference_mode_result.has_strong_dissent,
            'analysis': result_ref.get('analysis', {}),
        },
        'comparison': comparison.to_dict(),
        'fred_liquidity': {
            'rrp': fred_summary.rrp if fred_summary else None,
            'tga': fred_summary.tga if fred_summary else None,
            'net_liquidity': fred_summary.net_liquidity if fred_summary else None,
            'regime': fred_summary.liquidity_regime if fred_summary else None,
        }
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResults saved: {output_file}")

    return result


async def run_forecast_agent(
    market_data: Dict,
    macro_data,
    config: Dict
) -> Tuple[List[Dict], Dict]:
    """
    ForecastAgent 실행 (UnifiedDataCollector 사용)

    데이터 소스:
    1. UnifiedDataCollector: Yahoo + FRED → Ret_*, d_* 변수 생성
    2. CME 패널: d_Exp_Rate (종속변수), days_to_meeting

    Returns:
        Tuple[List[Dict], Dict]: (forecast_results, diagnostics)
    """
    # 진단 정보 수집
    diagnostics = {
        'cme_data_rows': 0,
        'market_data_rows': 0,
        'feature_count': 0,
        'common_dates': 0,
        'has_d_exp_rate': False,
        'days_to_meeting_missing': True,
        'market_date_range': None,
        'cme_date_range': None
    }

    # 1. UnifiedDataCollector로 시장 데이터 수집 (Ret_*, d_* 변수 포함)
    logger.info("Collecting market data with UnifiedDataCollector...")
    collector = UnifiedDataCollector(start_date='2024-09-01')
    market_features = collector.collect_all()

    if market_features.empty:
        logger.error("UnifiedDataCollector failed to collect data")
        return [], diagnostics

    diagnostics['market_data_rows'] = len(market_features)
    diagnostics['market_date_range'] = f"{market_features.index.min()} ~ {market_features.index.max()}"

    # 2. CME 패널 데이터 로드 (d_Exp_Rate, days_to_meeting)
    cme_data = load_cme_panel_data()

    if cme_data is not None and not cme_data.empty:
        logger.info(f"CME panel data loaded: {len(cme_data)} observations")
        diagnostics['cme_data_rows'] = len(cme_data)

        if 'asof_date' in cme_data.columns:
            cme_dates = pd.to_datetime(cme_data['asof_date'])
            diagnostics['cme_date_range'] = f"{cme_dates.min()} ~ {cme_dates.max()}"

        # CME 데이터 집계 (asof_date별)
        cme_agg = cme_data.groupby('asof_date').agg({
            'd_Exp_Rate': 'mean',
            'days_to_meeting': 'min',
            'exp_rate_bp': 'mean'
        })
        cme_agg.index = pd.to_datetime(cme_agg.index)

        # 시장 데이터와 CME 병합
        market_features.index = pd.to_datetime(market_features.index).normalize()
        cme_agg.index = cme_agg.index.normalize()

        # 겹치는 날짜 확인
        common_dates = market_features.index.intersection(cme_agg.index)
        logger.info(f"Common dates: {len(common_dates)}")

        if len(common_dates) > 0:
            prepared_data = market_features.join(cme_agg, how='inner')
            days_to_meeting = prepared_data['days_to_meeting']
        else:
            logger.warning("No overlapping dates between market and CME data")
            prepared_data = market_features
            days_to_meeting = calculate_days_to_fomc(prepared_data.index)
    else:
        # CME 데이터 없으면 시장 데이터만 사용
        logger.warning("CME panel data not available")
        prepared_data = market_features
        days_to_meeting = calculate_days_to_fomc(prepared_data.index)

    diagnostics['days_to_meeting_missing'] = days_to_meeting.isna().all() if isinstance(days_to_meeting, pd.Series) else True

    # 진단 정보 업데이트
    if not prepared_data.empty:
        diagnostics['feature_count'] = len(prepared_data.columns)
        diagnostics['common_dates'] = len(prepared_data)
        diagnostics['has_d_exp_rate'] = 'd_Exp_Rate' in prepared_data.columns
        
        logger.info(f"Prepared data: {len(prepared_data)} rows, {len(prepared_data.columns)} features")
        logger.info(f"Columns: {list(prepared_data.columns)[:10]}...")  # 처음 10개만
        logger.info(f"Has d_Exp_Rate: {diagnostics['has_d_exp_rate']}")
    
    # ForecastAgent 초기화
    forecast_agent = ForecastAgent()
    
    # 요청 생성 - prepared_data (DataFrame)을 직접 전달
    # ForecastAgent._execute()에서 DataFrame이면 _prepare_features() 건너뜀
    # → 이미 여기서 변환 완료했으므로 OK
    request = AgentRequest(
        task_id=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        role=AgentRole.FORECAST,
        instruction="Run LASSO forecast for Fed rate expectations",
        context={
            'market_data': prepared_data,
            'days_to_meeting': days_to_meeting
        }
    )
    
    # 실행
    try:
        result = await forecast_agent._execute(request)
        return result.get('forecasts', []), diagnostics
    except Exception as e:
        logger.error(f"Forecast agent failed: {e}")
        import traceback
        traceback.print_exc()
        return [], diagnostics


def load_cme_panel_data() -> Optional[pd.DataFrame]:
    """
    CME Fed Funds Futures 패널 데이터 로드
    
    파일 위치: plus/complete_cme_panel_history_*.csv
    
    컬럼:
        - meeting_date: FOMC 회의 날짜
        - asof_date: 관측 날짜
        - exp_rate_bp: 기대금리 (bp 단위)
        - days_to_meeting: FOMC까지 남은 일수
        - rate_uncertainty: 금리 불확실성
        
    Returns:
        CME 패널 DataFrame 또는 None
    """
    import glob
    
    # plus 폴더에서 CME 패널 파일 찾기
    cme_files = glob.glob('plus/complete_cme_panel_history_*.csv')
    
    if not cme_files:
        logger.warning("CME panel file not found in plus/ directory")
        return None
    
    # 가장 최신 파일 사용
    cme_file = sorted(cme_files)[-1]
    logger.info(f"Loading CME panel data from: {cme_file}")
    
    try:
        df = pd.read_csv(cme_file)
        
        # 날짜 컬럼 변환
        df['asof_date'] = pd.to_datetime(df['asof_date'])
        df['meeting_date'] = pd.to_datetime(df['meeting_date'])
        
        # 필수 컬럼 확인
        required_cols = ['meeting_date', 'asof_date', 'exp_rate_bp', 'days_to_meeting']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return None
        
        # 중복 제거 (같은 날짜-회의 조합은 최신 데이터 유지)
        df.drop_duplicates(subset=['meeting_date', 'asof_date'], keep='last', inplace=True)
        df.sort_values(['meeting_date', 'asof_date'], inplace=True)
        
        # 종속변수 계산: 일별 기대금리 변화
        df['d_Exp_Rate'] = df.groupby('meeting_date')['exp_rate_bp'].diff()
        
        # 통계 출력
        logger.info(f"CME panel: {len(df):,} observations, "
                   f"rate range: {df['exp_rate_bp'].min():.1f}-{df['exp_rate_bp'].max():.1f} bp")
        
        # Horizon별 관측치 수
        horizon_counts = {
            'VeryShort (≤30d)': len(df[df['days_to_meeting'] <= 30]),
            'Short (31-90d)': len(df[(df['days_to_meeting'] > 30) & (df['days_to_meeting'] <= 90)]),
            'Long (≥180d)': len(df[df['days_to_meeting'] >= 180])
        }
        for h, c in horizon_counts.items():
            logger.info(f"  {h}: {c:,} obs")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load CME panel: {e}")
        return None


def merge_cme_with_market(
    cme_data: pd.DataFrame,
    market_data: Dict,
    macro_data: pd.DataFrame
) -> pd.DataFrame:
    """
    CME 패널 데이터와 시장 데이터 병합
    
    forecasting_20251218.py의 방법론 적용:
    - CME 패널의 asof_date를 기준으로 시장 데이터 병합
    - d_Exp_Rate (종속변수)는 CME 데이터에서 가져옴
    - 시장 변수는 Ret_*, d_* 형식으로 변환
    
    Args:
        cme_data: CME 패널 DataFrame
        market_data: {ticker: DataFrame} 시장 데이터
        macro_data: FRED 거시경제 데이터
        
    Returns:
        병합된 LASSO 분석용 DataFrame
    """
    # 1. 시장 데이터 변환 (기존 함수 활용)
    market_features = prepare_lasso_features(market_data, macro_data)
    
    if market_features.empty:
        logger.warning("No market features prepared")
        # CME 데이터만 반환
        cme_pivot = cme_data.pivot_table(
            index='asof_date',
            values=['d_Exp_Rate', 'days_to_meeting'],
            aggfunc='mean'
        )
        return cme_pivot
    
    # 2. CME 데이터에서 d_Exp_Rate 추출 (asof_date별 평균)
    #    여러 meeting에 대한 관측치가 있으므로 평균 사용
    cme_agg = cme_data.groupby('asof_date').agg({
        'd_Exp_Rate': 'mean',
        'days_to_meeting': 'min',  # 가장 가까운 FOMC
        'exp_rate_bp': 'mean'
    }).rename(columns={'exp_rate_bp': 'Exp_Rate_Level'})
    
    # 디버그: 날짜 범위 확인
    logger.info(f"CME date range: {cme_agg.index.min()} ~ {cme_agg.index.max()}")
    logger.info(f"Market date range: {market_features.index.min()} ~ {market_features.index.max()}")
    
    # 3. 인덱스 타입 통일 (DatetimeIndex로)
    if not isinstance(cme_agg.index, pd.DatetimeIndex):
        cme_agg.index = pd.to_datetime(cme_agg.index)
    if not isinstance(market_features.index, pd.DatetimeIndex):
        market_features.index = pd.to_datetime(market_features.index)
    
    # 날짜만 비교 (시간 제거)
    cme_agg.index = cme_agg.index.normalize()
    market_features.index = market_features.index.normalize()
    
    # 중복 인덱스 제거 (평균)
    if market_features.index.duplicated().any():
        market_features = market_features.groupby(market_features.index).mean()
    
    # 4. 병합 (inner join - 겹치는 날짜만)
    logger.info(f"CME unique dates: {len(cme_agg)}, Market unique dates: {len(market_features)}")
    
    # 겹치는 날짜 확인
    common_dates = cme_agg.index.intersection(market_features.index)
    logger.info(f"Common dates: {len(common_dates)}")
    
    if len(common_dates) == 0:
        logger.error("No overlapping dates between CME and market data!")
        logger.error(f"CME sample dates: {list(cme_agg.index[:5])}")
        logger.error(f"Market sample dates: {list(market_features.index[:5])}")
        return pd.DataFrame()
    
    result = market_features.join(cme_agg, how='inner')
    
    # 5. d_Exp_Rate가 market_features에 이미 있으면 CME 값으로 덮어쓰기
    if 'd_Exp_Rate_y' in result.columns:
        result['d_Exp_Rate'] = result['d_Exp_Rate_y']
        result.drop(columns=['d_Exp_Rate_x', 'd_Exp_Rate_y'], errors='ignore', inplace=True)
    
    # 6. 결측치 처리
    result = result.ffill().bfill().dropna()
    
    logger.info(f"Merged data: {len(result)} observations, {len(result.columns)} features")
    logger.info(f"Date range: {result.index.min()} ~ {result.index.max()}")
    
    # 6. 주요 변수 확인
    key_vars = ['d_Exp_Rate', 'd_Spread_Baa', 'd_Breakeven5Y', 'Ret_SPY', 'd_VIX']
    found_vars = [v for v in key_vars if v in result.columns]
    logger.info(f"Key variables found: {found_vars}")
    
    return result


def prepare_lasso_features(
    market_data: Dict, 
    macro_data: pd.DataFrame
) -> pd.DataFrame:
    """
    LASSO 분석에 필요한 형식으로 데이터 변환
    
    변환 규칙:
    - 금리/변동성 → 차분 (d_* 접두사)
    - 주식/원자재/암호화폐 → 로그 수익률 (Ret_* 접두사)
    - Fed Funds Rate → d_Exp_Rate (종속변수)
    
    Args:
        market_data: {ticker: DataFrame} 형식의 시장 데이터
        macro_data: FRED 등에서 수집한 거시경제 데이터
        
    Returns:
        LASSO 분석용 DataFrame
    """
    # 모든 시리즈를 모아서 한 번에 concat (중복 컬럼 처리)
    all_series = []
    used_names = set()
    
    def add_series(series: pd.Series, name: str):
        """중복 체크 후 시리즈 추가"""
        if name in used_names:
            logger.debug(f"Skipping duplicate column: {name}")
            return
        series = series.copy()
        series.name = name
        all_series.append(series)
        used_names.add(name)
    
    # 1. 시장 데이터 변환
    for ticker, df in market_data.items():
        if df.empty:
            continue
        
        # 종가 추출
        if 'Close' in df.columns:
            close = df['Close']
        elif 'Adj Close' in df.columns:
            close = df['Adj Close']
        else:
            close = df.iloc[:, 0]
        
        # ticker를 기반으로 변수 타입 결정
        ticker_upper = ticker.upper()
        
        # 금리/변동성 관련 → 차분
        is_rate_or_vol = any(x in ticker_upper for x in [
            'VIX', 'YIELD', 'RATE', 'TREASURY', 'TLT', 'IEF', 'SHY', 
            'TIP', 'LQD', 'HYG', '^TNX', '^FVX', '^TYX', '^IRX'
        ])
        
        if is_rate_or_vol:
            diff = close.diff()
            add_series(diff, f'd_{ticker}')
        else:
            # 주식/원자재/암호화폐 → 로그 수익률
            ret = np.log(close / close.shift(1))
            add_series(ret, f'Ret_{ticker}')
    
    # 2. 거시경제 데이터 추가
    if macro_data is not None and not macro_data.empty:
        for col in macro_data.columns:
            col_upper = col.upper()
            
            # Fed Funds Rate → d_Exp_Rate (종속변수)
            if 'DFF' in col_upper or 'FEDFUNDS' in col_upper:
                d_exp_rate = macro_data[col].diff()
                add_series(d_exp_rate, 'd_Exp_Rate')
            
            # 금리 관련
            elif any(x in col_upper for x in ['DGS', 'DTB', 'RATE', 'YIELD']):
                diff = macro_data[col].diff()
                
                # 표준 변수명 매핑
                if 'DGS10' in col_upper or '10Y' in col_upper:
                    add_series(diff, 'd_US10Y')
                elif 'DGS2' in col_upper or '2Y' in col_upper:
                    add_series(diff, 'd_US2Y')
                elif 'T10Y2Y' in col_upper:
                    add_series(diff, 'd_Term_Spread')
                else:
                    add_series(diff, f'd_{col}')
            
            # 스프레드 (Baa, High Yield 등)
            elif any(x in col_upper for x in ['BAML', 'SPREAD', 'BAA']):
                diff = macro_data[col].diff()
                if 'HYM2' in col_upper or 'HIGHYIELD' in col_upper or 'HY' in col_upper:
                    add_series(diff, 'd_HighYield_Rate')
                elif 'BAA' in col_upper:
                    add_series(diff, 'd_Baa_Yield')
                elif 'BAML' in col_upper and 'C0A0' in col_upper:
                    add_series(diff, 'd_Spread_Baa')
                else:
                    add_series(diff, f'd_{col}')
            
            # 인플레이션 기대 (Breakeven)
            elif any(x in col_upper for x in ['T5YIE', 'T10YIE', 'BREAKEVEN']):
                diff = macro_data[col].diff()
                if '5Y' in col_upper:
                    add_series(diff, 'd_Breakeven5Y')
                elif '10Y' in col_upper:
                    add_series(diff, 'd_Breakeven10Y')
                else:
                    add_series(diff, f'd_{col}')
            
            # VIX
            elif 'VIX' in col_upper:
                diff = macro_data[col].diff()
                add_series(diff, 'd_VIX')
            
            # 달러 인덱스
            elif 'DTWEX' in col_upper or 'DOLLAR' in col_upper:
                diff = macro_data[col].diff()
                add_series(diff, 'd_Dollar_Idx')
    
    # 3. 모든 시리즈를 DataFrame으로 합치기
    if not all_series:
        logger.error("No data series collected")
        return pd.DataFrame()
    
    result = pd.concat(all_series, axis=1)
    
    # 4. 결측치 처리
    result = result.dropna(how='all')  # 모든 값이 NaN인 행 제거
    result = result.ffill().bfill()    # forward/backward fill (deprecated 경고 방지)
    result = result.dropna()           # 여전히 NaN이 있는 행 제거
    
    # 5. d_Exp_Rate가 없으면 proxy 생성 (Fed Funds Futures 대용)
    if 'd_Exp_Rate' not in result.columns:
        logger.warning("d_Exp_Rate not found in data, creating proxy from available rates")
        
        # 단기 금리 차분을 proxy로 사용
        rate_cols = [c for c in result.columns if c.startswith('d_') and 
                     any(x in c for x in ['US2Y', 'DTB', 'SHY', 'Rate'])]
        
        if rate_cols:
            result['d_Exp_Rate'] = result[rate_cols[0]]
            logger.info(f"Using {rate_cols[0]} as d_Exp_Rate proxy")
        else:
            # 마지막 수단: TLT 수익률의 반전 사용
            tlt_cols = [c for c in result.columns if 'TLT' in c.upper()]
            if tlt_cols:
                # 채권 가격 하락 = 금리 상승 기대
                result['d_Exp_Rate'] = -result[tlt_cols[0]] * 0.1
                logger.info(f"Using inverted {tlt_cols[0]} as d_Exp_Rate proxy")
            else:
                logger.error("Cannot create d_Exp_Rate - no suitable proxy found")
    
    logger.info(f"Prepared {len(result)} observations with {len(result.columns)} features")
    logger.info(f"Features: {list(result.columns)[:15]}...")
    
    return result


def calculate_days_to_fomc(index: pd.DatetimeIndex) -> pd.Series:
    """
    실제 FOMC 회의 일정을 기반으로 days_to_meeting 계산
    
    Args:
        index: 데이터의 DatetimeIndex
        
    Returns:
        각 날짜별 다음 FOMC까지 남은 거래일 수
    """
    # 2024-2025년 FOMC 회의 일정 (실제 데이터)
    # https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
    fomc_dates = [
        # 2024
        '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12',
        '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18',
        # 2025
        '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18',
        '2025-07-30', '2025-09-17', '2025-11-05', '2025-12-17',
        # 2026 (예상)
        '2026-01-28', '2026-03-18', '2026-05-06', '2026-06-17',
    ]
    fomc_dates = pd.to_datetime(fomc_dates)
    
    # 과거 FOMC 일정 추가 (2022-2023)
    past_fomc = [
        '2022-01-26', '2022-03-16', '2022-05-04', '2022-06-15',
        '2022-07-27', '2022-09-21', '2022-11-02', '2022-12-14',
        '2023-02-01', '2023-03-22', '2023-05-03', '2023-06-14',
        '2023-07-26', '2023-09-20', '2023-11-01', '2023-12-13',
    ]
    fomc_dates = pd.to_datetime(past_fomc + list(fomc_dates.strftime('%Y-%m-%d')))
    fomc_dates = fomc_dates.sort_values()
    
    # 각 날짜별 다음 FOMC까지 거래일 수 계산
    days_to_meeting = pd.Series(index=index, dtype=int)
    
    for date in index:
        # 다음 FOMC 찾기
        future_fomc = fomc_dates[fomc_dates > date]
        
        if len(future_fomc) > 0:
            next_fomc = future_fomc[0]
            # 거래일 수 계산 (주말 제외)
            business_days = np.busday_count(
                date.date(), 
                next_fomc.date()
            )
            days_to_meeting[date] = max(1, business_days)
        else:
            # FOMC 일정 이후 → 먼 미래로 설정
            days_to_meeting[date] = 365
    
    logger.info(f"Days to FOMC range: {days_to_meeting.min()} - {days_to_meeting.max()}")
    
    return days_to_meeting


async def run_visualization_agent(
    forecast_results: List,
    debate_result: Dict,
    agent_opinions: List,
    consensus: Any,
    conflicts: List,
    config: Dict,
    forecast_diagnostics: Dict = None
) -> str:
    """VisualizationAgent 실행"""
    
    # 대시보드 설정
    viz_config = config.get('agents', {}).get('visualization', {})
    dashboard_config = DashboardConfig(
        theme=viz_config.get('theme', 'dark'),
        language=viz_config.get('language', 'ko'),
        output_dir=config.get('output', {}).get('dir', 'outputs/dashboards'),
        include_lasso_results=True,
        include_agent_debate=True,
        include_regime=True
    )
    
    # VisualizationAgent 초기화
    viz_agent = VisualizationAgent(dashboard_config=dashboard_config)
    
    # 컨텍스트 준비
    context = {
        'signals': debate_result.get('analysis', {}).get('critical_paths', []),
        'regime_data': {
            'current_regime': debate_result.get('analysis', {}).get('current_regime', 'UNKNOWN'),
            'probability': debate_result.get('analysis', {}).get('regime_probability', 0.5)
        },
        'risk_metrics': debate_result.get('analysis', {}).get('risk_metrics', {}),
        'macro_indicators': debate_result.get('analysis', {}).get('macro_indicators', {}),
        'forecast_results': forecast_results,
        'forecast_diagnostics': forecast_diagnostics or {},  # 진단 정보 추가
        'agent_opinions': agent_opinions,
        'consensus': consensus,
        'conflicts': conflicts,
        'timestamp': datetime.now().isoformat(),
        'project_id': 'eimas_pipeline'
    }
    
    # 요청 생성
    request = AgentRequest(
        task_id=f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        role=AgentRole.STRATEGY,
        instruction="Generate comprehensive dashboard",
        context=context
    )
    
    # 실행
    try:
        result = await viz_agent._execute(request)
        return result.get('dashboard_path', '')
    except Exception as e:
        logger.error(f"Visualization agent failed: {e}")
        return ''


def save_json_result(
    forecast_results: List,
    debate_result: Dict,
    config: Dict
) -> str:
    """JSON 결과 저장"""
    output_dir = Path(config.get('output', {}).get('dir', 'outputs'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"report_{timestamp}.json"
    
    # 결과 정리
    result = {
        'timestamp': datetime.now().isoformat(),
        'forecast_results': forecast_results,
        'analysis': debate_result.get('analysis', {}),
        'debate': debate_result.get('debate', {}),
        'recommendations': debate_result.get('recommendations', [])
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    
    return str(output_file)


def load_tickers_config(config_path: str) -> Dict:
    """티커 설정 로드"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Tickers config not found: {config_path}")
        return {'market': ['SPY', 'QQQ', 'TLT', 'GLD']}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================================
# Legacy Main Function
# ============================================================================

async def main():
    """기존 main 함수 (호환성 유지)"""
    print("="*70)
    print("  EIMAS - Economic Intelligence Multi-Agent System")
    print("="*70)

    # 1. Data Collection
    print("\n[1/3] Collecting market data...")
    
    config_path = Path(__file__).parent / "configs" / "tickers.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            tickers_config = yaml.safe_load(f)
    else:
        tickers_config = {'market': ['SPY', 'QQQ', 'TLT', 'GLD']}
        
    dm = DataManager(lookback_days=365)
    market_data, macro_data = dm.collect_all(tickers_config)
    print(f"      → Collected {len(market_data)} tickers")
    print(f"      → Collected {len(macro_data.columns)} macro series")

    # 2. Run Workflow
    print("\n[2/3] Running multi-agent analysis...")
    orchestrator = MetaOrchestrator(verbose=True)
    
    query = "Analyze current market conditions and forecast Fed policy"
    result = await orchestrator.run_with_debate(query, market_data)

    # 3. Save Results
    print("\n[3/3] Saving results...")
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"report_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"      → Saved: {output_file}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if result.get('analysis'):
        print(f"Risk Score: {result['analysis'].get('total_risk_score', 0):.1f}/100")
        print(f"Regime: {result['analysis'].get('current_regime', 'Unknown')}")
    
    if result.get('recommendations'):
        print(f"\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"  {i}. {rec}")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='EIMAS - Economic Intelligence Multi-Agent System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    python main.py                              # 기본: Dual Mode 분석 (FULL vs REFERENCE)
    python main.py --dual-mode                  # Dual Mode 분석 (명시적)
    python main.py --full-pipeline              # 전체 파이프라인 (LASSO + Dashboard)
    python main.py --config configs/custom.yaml # 커스텀 설정
    python main.py --legacy                     # 기존 방식 실행
        '''
    )

    parser.add_argument(
        '--config', '-c',
        default='configs/default.yaml',
        help='Configuration file path (default: configs/default.yaml)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        default='outputs/dashboards',
        help='Output directory for dashboards (default: outputs/dashboards)'
    )

    parser.add_argument(
        '--dual-mode', '-d',
        action='store_true',
        help='Run dual mode analysis (FULL vs REFERENCE) - DEFAULT'
    )

    parser.add_argument(
        '--full-pipeline', '-f',
        action='store_true',
        help='Run full pipeline with LASSO and dashboard generation'
    )

    parser.add_argument(
        '--legacy', '-l',
        action='store_true',
        help='Run legacy main function'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


async def cli_main():
    """CLI 메인 함수"""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.legacy:
        # 기존 방식 (호환성)
        await main()
    elif args.full_pipeline:
        # 전체 파이프라인 (LASSO + Dashboard)
        dashboard_path = await run_full_pipeline(args.config)
        print(f"\nDashboard generated: {dashboard_path}")
    else:
        # 기본: Dual Mode 분석 (FULL vs REFERENCE)
        # --dual-mode 플래그 유무와 관계없이 기본값
        result = await run_dual_mode_pipeline(args.config)

        # 간단한 결과 출력
        print("\n" + "=" * 50)
        print("QUICK SUMMARY")
        print("=" * 50)
        comp = result.get('comparison', {}).get('comparison', {})
        print(f"FULL Mode:      {result['full_mode']['position']} ({result['full_mode']['confidence']:.0%})")
        print(f"REFERENCE Mode: {result['reference_mode']['position']} ({result['reference_mode']['confidence']:.0%})")
        print(f"Agreement:      {'✓' if comp.get('positions_agree') else '✗'}")
        print(f"Recommendation: {comp.get('recommended_action', 'N/A')}")
        print(f"Risk Level:     {comp.get('risk_level', 'N/A')}")


if __name__ == "__main__":
    asyncio.run(cli_main())
