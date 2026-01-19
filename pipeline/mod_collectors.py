#!/usr/bin/env python3
"""
EIMAS Pipeline Collectors
==========================
Phase 1: 데이터 수집 모듈
"""

import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

# EIMAS 라이브러리
from lib.fred_collector import FREDCollector
from lib.data_collector import DataManager
from lib.market_indicators import MarketIndicatorsCollector
from pipeline.schemas import FREDSummary, IndicatorsSummary

def collect_fred_data() -> FREDSummary:
    """FRED 데이터 수집"""
    print("\n[1.1] Collecting FRED data...")
    try:
        collector = FREDCollector()
        summary = collector.collect_all()
        
        # Schema 변환
        return FREDSummary(
            timestamp=summary.timestamp,
            fed_funds=summary.fed_funds,
            treasury_2y=summary.treasury_2y,
            treasury_10y=summary.treasury_10y,
            treasury_30y=summary.treasury_30y,
            spread_10y2y=summary.spread_10y2y,
            spread_10y3m=summary.spread_10y3m,
            hy_oas=summary.hy_oas,
            cpi_yoy=summary.cpi_yoy,
            core_pce_yoy=summary.core_pce_yoy,
            breakeven_5y=summary.breakeven_5y,
            breakeven_10y=summary.breakeven_10y,
            unemployment=summary.unemployment,
            initial_claims=summary.initial_claims,
            rrp=summary.rrp,
            rrp_delta=summary.rrp_delta,
            rrp_delta_pct=summary.rrp_delta_pct,
            tga=summary.tga,
            tga_delta=summary.tga_delta,
            fed_assets=summary.fed_assets,
            fed_assets_delta=summary.fed_assets_delta,
            net_liquidity=summary.net_liquidity,
            liquidity_regime=summary.liquidity_regime,
            curve_inverted=summary.curve_inverted,
            curve_status=summary.curve_status,
            signals=summary.signals,
            warnings=summary.warnings
        )
    except Exception as e:
        print(f"      ✗ FRED error: {e}")
        return FREDSummary(timestamp=datetime.now().isoformat())

def collect_market_data(lookback_days: int = 365) -> Dict[str, pd.DataFrame]:
    """시장 데이터 수집"""
    print("\n[1.2] Collecting market data...")
    try:
        dm = DataManager(lookback_days=lookback_days)
        tickers_config = {
            'market': [
                {'ticker': 'SPY'}, {'ticker': 'QQQ'}, {'ticker': 'IWM'},
                {'ticker': 'DIA'}, {'ticker': 'TLT'}, {'ticker': 'GLD'},
                {'ticker': 'USO'}, {'ticker': 'UUP'}, {'ticker': '^VIX'}
            ],
            'crypto': [
                {'ticker': 'BTC-USD'}, {'ticker': 'ETH-USD'}
            ]
        }
        market_data, _ = dm.collect_all(tickers_config)
        print(f"      ✓ Collected {len(market_data)} tickers")
        return market_data
    except Exception as e:
        print(f"      ✗ Market data error: {e}")
        return {}

def collect_crypto_data() -> Dict[str, pd.DataFrame]:
    """암호화폐 데이터 수집 (DataManager 활용)"""
    print("\n[1.3] Collecting crypto data...")
    try:
        dm = DataManager(lookback_days=90)
        tickers_config = {
            'crypto': [
                {'ticker': 'BTC-USD'}, {'ticker': 'ETH-USD'}, {'ticker': 'SOL-USD'}
            ]
        }
        crypto_data, _ = dm.collect_all(tickers_config)
        print(f"      ✓ Collected {len(crypto_data)} crypto tickers")
        return crypto_data
    except Exception as e:
        print(f"      ✗ Crypto data error: {e}")
        return {}

def collect_market_indicators() -> IndicatorsSummary:
    """시장 지표 수집"""
    print("\n[1.4] Collecting market indicators...")
    try:
        collector = MarketIndicatorsCollector()
        summary = collector.collect_all()
        
        return IndicatorsSummary(
            timestamp=summary.timestamp,
            vix_current=summary.vix.current,
            fear_greed_level=summary.vix.fear_greed_level,
            risk_score=summary.risk_score,
            opportunity_score=summary.opportunity_score,
            signals=summary.signals,
            warnings=summary.warnings,
            raw_data=summary.to_dict()
        )
    except Exception as e:
        print(f"      ✗ Indicators error: {e}")
        return IndicatorsSummary(timestamp=datetime.now().isoformat())

