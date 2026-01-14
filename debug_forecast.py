#!/usr/bin/env python3
"""
디버그 스크립트: LASSO 예측 데이터 흐름 확인
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import yaml

from main import load_cme_panel_data, merge_cme_with_market, prepare_lasso_features
from lib.data_collector import DataManager
from lib.lasso_model import get_horizon_mask

print("=" * 60)
print("LASSO Forecast Debug")
print("=" * 60)

# 1. CME Panel Data
print("\n=== 1. CME Panel Data ===")
cme = load_cme_panel_data()
if cme is not None:
    print(f"Shape: {cme.shape}")
    print(f"Date range: {cme['asof_date'].min()} ~ {cme['asof_date'].max()}")
    print(f"days_to_meeting range: {cme['days_to_meeting'].min()} ~ {cme['days_to_meeting'].max()}")
    print(f"d_Exp_Rate: NaN={cme['d_Exp_Rate'].isna().sum()}, Valid={cme['d_Exp_Rate'].notna().sum()}")
else:
    print("ERROR: CME data is None!")
    sys.exit(1)

# 2. Market Data
print("\n=== 2. Market Data ===")
try:
    with open('configs/tickers.yaml') as f:
        tickers_config = yaml.safe_load(f)
    dm = DataManager(lookback_days=365)
    market_data, macro_data = dm.collect_all(tickers_config)
    print(f"Market tickers: {len(market_data)}")
    print(f"Macro columns: {list(macro_data.columns)[:10]}...")
    
    # 날짜 범위 확인
    first_ticker = list(market_data.values())[0]
    print(f"Market date range: {first_ticker.index.min()} ~ {first_ticker.index.max()}")
    print(f"Macro date range: {macro_data.index.min()} ~ {macro_data.index.max()}")
except Exception as e:
    print(f"ERROR: {e}")
    market_data, macro_data = {}, pd.DataFrame()

# 3. Merge Test
print("\n=== 3. Merge Test ===")
if cme is not None:
    merged = merge_cme_with_market(cme, market_data, macro_data)
    print(f"Merged shape: {merged.shape}")
    
    if merged.empty:
        print("ERROR: Merged data is empty!")
        print("\nDebugging date overlap...")
        
        # CME 날짜
        cme_dates = set(cme['asof_date'].dt.date)
        print(f"CME unique dates: {len(cme_dates)}")
        print(f"CME date sample: {sorted(list(cme_dates))[:5]}")
        
        # Market 날짜
        if market_data:
            first_df = list(market_data.values())[0]
            market_dates = set(first_df.index.date)
            print(f"Market unique dates: {len(market_dates)}")
            print(f"Market date sample: {sorted(list(market_dates))[:5]}")
            
            # 겹치는 날짜
            overlap = cme_dates & market_dates
            print(f"Overlapping dates: {len(overlap)}")
            if overlap:
                print(f"Overlap sample: {sorted(list(overlap))[:5]}")
    else:
        print(f"Merged date range: {merged.index.min()} ~ {merged.index.max()}")
        print(f"Columns ({len(merged.columns)}): {list(merged.columns)[:15]}...")
        
        if 'd_Exp_Rate' in merged.columns:
            valid = merged['d_Exp_Rate'].notna()
            print(f"d_Exp_Rate: Valid={valid.sum()}, NaN={(~valid).sum()}")
            print(f"d_Exp_Rate stats: mean={merged['d_Exp_Rate'].mean():.4f}, std={merged['d_Exp_Rate'].std():.4f}")
        else:
            print("ERROR: d_Exp_Rate not in merged data!")
        
        # 4. Horizon Check
        print("\n=== 4. Horizon Check ===")
        if 'days_to_meeting' in merged.columns:
            days = merged['days_to_meeting']
        else:
            # CME에서 days_to_meeting 가져오기
            days_agg = cme.groupby('asof_date')['days_to_meeting'].min()
            days = days_agg.reindex(merged.index).ffill().bfill()
        
        print(f"days_to_meeting range: {days.min():.0f} ~ {days.max():.0f}")
        
        for horizon in ['VeryShort', 'Short', 'Long']:
            try:
                mask = get_horizon_mask(days, horizon)
                n_obs = mask.sum()
                print(f"  {horizon}: {n_obs} observations")
                
                if n_obs > 0 and 'd_Exp_Rate' in merged.columns:
                    y_valid = merged.loc[mask, 'd_Exp_Rate'].notna().sum()
                    print(f"    → d_Exp_Rate valid: {y_valid}")
            except Exception as e:
                print(f"  {horizon}: ERROR - {e}")

print("\n" + "=" * 60)
print("Debug completed!")
print("=" * 60)

