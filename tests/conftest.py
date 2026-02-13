"""
Pytest configuration and shared fixtures
"""
import sys
from pathlib import Path

# Add project root to sys.path for all tests (centralized)
project_root = Path(__file__).resolve().parent.parent
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

@pytest.fixture(scope="module")
def market_data():
    """종합 테스트용 시장 데이터 픽스처"""
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    market_data = {}
    
    # 공통 팩터 (시장 베타)
    market_factor = np.cumsum(np.random.randn(n_days)) * 0.01
    
    # SPY (시장 대표)
    spy_prices = 400 * np.exp(market_factor + np.random.randn(n_days) * 0.003)
    spy_volumes = np.exp(np.random.randn(n_days) * 0.2 + 18)
    
    market_data['SPY'] = pd.DataFrame({
        'Open': spy_prices * 0.995,
        'High': spy_prices * 1.005,
        'Low': spy_prices * 0.99,
        'Close': spy_prices,
        'Volume': spy_volumes,
        'Adj Close': spy_prices
    }, index=dates)
    
    # QQQ
    qqq_prices = 350 * np.exp(market_factor * 1.2 + np.random.randn(n_days) * 0.004)
    qqq_volumes = np.exp(np.random.randn(n_days) * 0.2 + 17.5)
    
    market_data['QQQ'] = pd.DataFrame({
        'Open': qqq_prices * 0.995,
        'High': qqq_prices * 1.005,
        'Low': qqq_prices * 0.99,
        'Close': qqq_prices,
        'Volume': qqq_volumes,
        'Adj Close': qqq_prices
    }, index=dates)
    
    # TLT (채권, 역상관)
    tlt_prices = 100 * np.exp(-market_factor * 0.3 + np.random.randn(n_days) * 0.002)
    tlt_volumes = np.exp(np.random.randn(n_days) * 0.3 + 15.5)
    
    market_data['TLT'] = pd.DataFrame({
        'Open': tlt_prices * 0.995,
        'High': tlt_prices * 1.005,
        'Low': tlt_prices * 0.99,
        'Close': tlt_prices,
        'Volume': tlt_volumes,
        'Adj Close': tlt_prices
    }, index=dates)
    
    return market_data
