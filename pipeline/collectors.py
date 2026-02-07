#!/usr/bin/env python3
"""
EIMAS Pipeline - Collectors Module
===================================

Purpose:
    Phase 1 데이터 수집 담당 (Data Collection)

Functions:
    - collect_fred_data() -> FREDSummary
    - collect_market_data(lookback_days) -> Dict[str, DataFrame]
    - collect_crypto_data() -> Dict[str, DataFrame]
    - collect_market_indicators() -> IndicatorsSummary

Dependencies:
    - lib.fred_collector
    - lib.data_collector
    - lib.market_indicators

Example:
    from pipeline.collectors import collect_fred_data
    fred = collect_fred_data()
    print(fred.net_liquidity)
"""

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import pandas as pd

# EIMAS 라이브러리
from lib.fred_collector import FREDCollector
from lib.data_collector import DataManager
from lib.market_indicators import MarketIndicatorsCollector
from pipeline.schemas import FREDSummary, IndicatorsSummary
from pipeline.exceptions import get_logger, log_error, CollectionError

logger = get_logger("collectors")

_FI_MARKET_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO", "UUP", "^VIX"]
_FI_MARKET_CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]
_FI_CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_alpha_probe_tickers() -> list[str]:
    configured = os.getenv("EIMAS_ALPHA_PROBE_TICKERS", "").strip()
    raw_items = configured.split(",") if configured else _FI_MARKET_TICKERS[:2]

    tickers: list[str] = []
    for raw in raw_items:
        ticker = raw.strip()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers or _FI_MARKET_TICKERS[:2]


def _resolve_market_tickers_for_collection(use_alpha_vantage: bool) -> list[str]:
    """
    Alpha Vantage free-tier safe mode:
    - default: small probe set (2 tickers)
    - full scan: set EIMAS_ALPHA_FULL_SCAN=true
    """
    if not use_alpha_vantage:
        return _FI_MARKET_TICKERS
    if _env_flag("EIMAS_ALPHA_FULL_SCAN", default=False):
        return _FI_MARKET_TICKERS
    probe_tickers = _resolve_alpha_probe_tickers()
    print(f"      i Alpha probe mode: {len(probe_tickers)} tickers ({', '.join(probe_tickers)})")
    return probe_tickers


@lru_cache(maxsize=1)
def _load_financial_indicators() -> Dict[str, Any]:
    """
    Load collectors from sibling project: ../financial_indicators.
    Returns empty dict on any failure so legacy collectors can continue.
    """
    configured = os.getenv("EIMAS_FINANCIAL_INDICATORS_PATH", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path(__file__).resolve().parents[2] / "financial_indicators")

    fi_root = next((path for path in candidates if path.exists()), None)
    if fi_root is None:
        return {}

    config_path = fi_root / "config.py"
    collectors_init = fi_root / "collectors" / "__init__.py"
    if not (config_path.exists() and collectors_init.exists()):
        logger.warning("financial_indicators layout invalid: %s", fi_root)
        return {}

    try:
        if str(fi_root) not in sys.path:
            sys.path.insert(0, str(fi_root))

        config_module_name = "_eimas_fi_config"
        if config_module_name in sys.modules:
            fi_config = sys.modules[config_module_name]
        else:
            config_spec = importlib.util.spec_from_file_location(config_module_name, config_path)
            if config_spec is None or config_spec.loader is None:
                raise ImportError(f"Cannot load {config_path}")
            fi_config = importlib.util.module_from_spec(config_spec)
            sys.modules[config_module_name] = fi_config
            config_spec.loader.exec_module(fi_config)

        package_name = "_eimas_financial_collectors"
        if package_name in sys.modules:
            fi_collectors = sys.modules[package_name]
        else:
            collectors_spec = importlib.util.spec_from_file_location(
                package_name,
                collectors_init,
                submodule_search_locations=[str(fi_root / "collectors")],
            )
            if collectors_spec is None or collectors_spec.loader is None:
                raise ImportError(f"Cannot load {collectors_init}")

            saved_config_module = sys.modules.get("config")
            sys.modules["config"] = fi_config
            try:
                fi_collectors = importlib.util.module_from_spec(collectors_spec)
                sys.modules[package_name] = fi_collectors
                collectors_spec.loader.exec_module(fi_collectors)
            finally:
                if saved_config_module is None:
                    sys.modules.pop("config", None)
                else:
                    sys.modules["config"] = saved_config_module

        classes = {
            "FREDCollector": getattr(fi_collectors, "FREDCollector", None),
            "MarketCollector": getattr(fi_collectors, "MarketCollector", None),
            "CryptoCollector": getattr(fi_collectors, "CryptoCollector", None),
        }

        if not any(classes.values()):
            logger.warning("financial_indicators collectors not found in %s", fi_root)
            return {}

        logger.info("financial_indicators linked from %s", fi_root)
        return classes

    except Exception as e:
        log_error(logger, "Failed to initialize financial_indicators bridge", e)
        return {}


def _collect_market_data_via_financial_indicators(lookback_days: int) -> Dict[str, pd.DataFrame]:
    """Collect core market + BTC/ETH via financial_indicators collectors."""
    classes = _load_financial_indicators()
    market_cls = classes.get("MarketCollector")
    crypto_cls = classes.get("CryptoCollector")
    if market_cls is None:
        return {}

    collected: Dict[str, pd.DataFrame] = {}
    use_alpha_vantage = _env_flag("EIMAS_USE_ALPHA_VANTAGE", default=False)
    market_tickers = _resolve_market_tickers_for_collection(use_alpha_vantage)
    alpha_probe_mode = use_alpha_vantage and not _env_flag("EIMAS_ALPHA_FULL_SCAN", default=False)
    market_collector = market_cls(
        lookback_days=lookback_days,
        use_alpha_vantage=use_alpha_vantage,
    )
    for ticker in market_tickers:
        data, _status = market_collector.fetch_ticker(ticker, ticker)
        if data is not None and not data.empty:
            collected[ticker] = data

    include_market_crypto = _env_flag("EIMAS_INCLUDE_MARKET_CRYPTO", default=not alpha_probe_mode)
    if include_market_crypto and crypto_cls is not None:
        crypto_collector = crypto_cls(lookback_days=lookback_days)
        for ticker in _FI_MARKET_CRYPTO_TICKERS:
            data, _status = crypto_collector.fetch_ticker(ticker, ticker)
            if data is not None and not data.empty:
                collected[ticker] = data

    return collected


def _collect_crypto_data_via_financial_indicators(lookback_days: int) -> Dict[str, pd.DataFrame]:
    """Collect BTC/ETH/SOL via financial_indicators multi-source crypto collector."""
    classes = _load_financial_indicators()
    crypto_cls = classes.get("CryptoCollector")
    if crypto_cls is None:
        return {}

    collected: Dict[str, pd.DataFrame] = {}
    crypto_collector = crypto_cls(lookback_days=lookback_days)

    for ticker in _FI_CRYPTO_TICKERS:
        data, _status = crypto_collector.fetch_ticker(ticker, ticker)
        if data is not None and not data.empty:
            collected[ticker] = data

    return collected

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
        log_error(logger, "FRED collection failed", e)
        return FREDSummary(timestamp=datetime.now().isoformat())

def collect_market_data(lookback_days: int = 365) -> Dict[str, pd.DataFrame]:
    """시장 데이터 수집"""
    print("\n[1.2] Collecting market data...")
    try:
        fi_data = _collect_market_data_via_financial_indicators(lookback_days)
        if fi_data:
            print(f"      ✓ Collected {len(fi_data)} tickers (financial_indicators)")
            return fi_data

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
        log_error(logger, "Market data collection failed", e)
        return {}

def collect_crypto_data(lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
    """암호화폐 데이터 수집 (DataManager 활용)"""
    print("\n[1.3] Collecting crypto data...")
    try:
        fi_data = _collect_crypto_data_via_financial_indicators(lookback_days)
        if fi_data:
            print(f"      ✓ Collected {len(fi_data)} crypto tickers (financial_indicators)")
            return fi_data

        dm = DataManager(lookback_days=lookback_days)
        tickers_config = {
            'crypto': [
                {'ticker': 'BTC-USD'}, {'ticker': 'ETH-USD'}, {'ticker': 'SOL-USD'}
            ]
        }
        crypto_data, _ = dm.collect_all(tickers_config)
        print(f"      ✓ Collected {len(crypto_data)} crypto tickers")
        return crypto_data
    except Exception as e:
        log_error(logger, "Crypto data collection failed", e)
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
        log_error(logger, "Indicator collection failed", e)
        return IndicatorsSummary(timestamp=datetime.now().isoformat())
