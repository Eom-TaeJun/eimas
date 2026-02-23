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

import importlib
import math
import os
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd

# EIMAS 라이브러리
from lib.fred_collector import FREDCollector
from lib.data_collector import DataManager
from lib.market_indicators import MarketIndicatorsCollector
from lib.ra_sql_store import ingest_company_ra_analysis_to_sql
from pipeline.schemas import FREDSummary, IndicatorsSummary
from pipeline.exceptions import get_logger, log_error, CollectionError

logger = get_logger("collectors")

_FI_MARKET_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "USO", "UUP", "^VIX"]
_FI_MARKET_CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]
_FI_CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD"]
_FI_RA_COMPANY_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "XOM"]
_FI_RA_ETF_TICKERS = ["SPY", "QQQ", "IWM", "XLF", "TLT", "GLD", "XLK", "XLE", "XLV", "XLI"]

_ETF_PROFILE_CATALOG: Dict[str, Dict[str, Any]] = {
    "SPY": {
        "asset_role": "Core",
        "factor_exposure": "Market Beta",
        "sector_or_theme": "미국 대형주 광범위",
        "duration_profile": "중립",
        "category": "market",
        "top_holdings": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
        "sector_weights": [
            {"sector": "Technology", "weight_pct": 31.0},
            {"sector": "Financials", "weight_pct": 13.0},
            {"sector": "Healthcare", "weight_pct": 12.0},
        ],
    },
    "QQQ": {
        "asset_role": "Growth Tilt",
        "factor_exposure": "Growth/Quality",
        "sector_or_theme": "나스닥100(기술 비중 높음)",
        "duration_profile": "상대적 고듀레이션",
        "category": "market",
        "top_holdings": ["AAPL", "MSFT", "NVDA", "AMZN", "META"],
        "sector_weights": [
            {"sector": "Technology", "weight_pct": 51.0},
            {"sector": "Communication", "weight_pct": 16.0},
            {"sector": "Consumer Discretionary", "weight_pct": 13.0},
        ],
    },
    "IWM": {
        "asset_role": "Size Tilt",
        "factor_exposure": "Small Cap",
        "sector_or_theme": "미국 소형주",
        "duration_profile": "경기 민감",
        "category": "market",
        "top_holdings": ["SMCI", "FTAI", "CRDO", "APPF", "TGTX"],
        "sector_weights": [
            {"sector": "Industrials", "weight_pct": 19.0},
            {"sector": "Financials", "weight_pct": 17.0},
            {"sector": "Healthcare", "weight_pct": 16.0},
        ],
    },
    "XLF": {
        "asset_role": "Sector Satellite",
        "factor_exposure": "Value/Cyclicals",
        "sector_or_theme": "금융 섹터",
        "duration_profile": "장단기금리차 민감",
        "category": "sector",
        "top_holdings": ["BRK-B", "JPM", "V", "MA", "BAC"],
        "sector_weights": [
            {"sector": "Banks", "weight_pct": 35.0},
            {"sector": "Capital Markets", "weight_pct": 24.0},
            {"sector": "Insurance", "weight_pct": 17.0},
        ],
    },
    "TLT": {
        "asset_role": "Rates Hedge",
        "factor_exposure": "Duration",
        "sector_or_theme": "미국 장기 국채",
        "duration_profile": "고듀레이션",
        "category": "bond",
        "top_holdings": ["UST 20Y+", "UST 25Y+", "UST 30Y+"],
        "sector_weights": [{"sector": "US Treasuries", "weight_pct": 100.0}],
    },
    "GLD": {
        "asset_role": "Real Asset Hedge",
        "factor_exposure": "Inflation/Real Rate",
        "sector_or_theme": "금(대체자산)",
        "duration_profile": "실질금리 역민감",
        "category": "alternative",
        "top_holdings": ["Physical Gold Bullion"],
        "sector_weights": [{"sector": "Precious Metals", "weight_pct": 100.0}],
    },
    "XLK": {
        "asset_role": "Sector Satellite",
        "factor_exposure": "Quality/Growth",
        "sector_or_theme": "기술 섹터",
        "duration_profile": "금리 민감",
        "category": "sector",
        "top_holdings": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE"],
        "sector_weights": [
            {"sector": "Software", "weight_pct": 39.0},
            {"sector": "Semiconductors", "weight_pct": 31.0},
            {"sector": "Tech Hardware", "weight_pct": 20.0},
        ],
    },
    "XLE": {
        "asset_role": "Sector Satellite",
        "factor_exposure": "Value/Cyclicals",
        "sector_or_theme": "에너지 섹터",
        "duration_profile": "유가 민감",
        "category": "sector",
        "top_holdings": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "sector_weights": [
            {"sector": "Integrated Oil & Gas", "weight_pct": 41.0},
            {"sector": "E&P", "weight_pct": 33.0},
            {"sector": "Oil Equipment/Services", "weight_pct": 15.0},
        ],
    },
    "XLV": {
        "asset_role": "Sector Satellite",
        "factor_exposure": "Defensive",
        "sector_or_theme": "헬스케어 섹터",
        "duration_profile": "방어적",
        "category": "sector",
        "top_holdings": ["LLY", "UNH", "JNJ", "ABBV", "MRK"],
        "sector_weights": [
            {"sector": "Pharmaceuticals", "weight_pct": 30.0},
            {"sector": "Healthcare Equipment", "weight_pct": 22.0},
            {"sector": "Biotech", "weight_pct": 16.0},
        ],
    },
    "XLI": {
        "asset_role": "Sector Satellite",
        "factor_exposure": "Cyclicals",
        "sector_or_theme": "산업재 섹터",
        "duration_profile": "경기 민감",
        "category": "sector",
        "top_holdings": ["GE", "RTX", "UNP", "CAT", "HON"],
        "sector_weights": [
            {"sector": "Aerospace & Defense", "weight_pct": 22.0},
            {"sector": "Industrial Conglomerates", "weight_pct": 18.0},
            {"sector": "Machinery", "weight_pct": 17.0},
        ],
    },
}


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


def _resolve_ra_company_tickers() -> list[str]:
    raw = os.getenv("EIMAS_RA_COMPANY_TICKERS", "").strip()
    base = raw.split(",") if raw else _FI_RA_COMPANY_TICKERS
    tickers: list[str] = []
    for item in base:
        ticker = item.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers or _FI_RA_COMPANY_TICKERS


def _resolve_ra_etf_tickers() -> list[str]:
    raw = os.getenv("EIMAS_RA_ETF_TICKERS", "").strip()
    base = raw.split(",") if raw else _FI_RA_ETF_TICKERS
    tickers: list[str] = []
    for item in base:
        ticker = item.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers or _FI_RA_ETF_TICKERS


def _safe_ratio_pct(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        val = float(value)
        if val <= 1.0:
            val *= 100.0
        return val
    except (TypeError, ValueError):
        return None


def _safe_count(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _calc_return_pct(close_series: Optional[pd.Series], days: int) -> Optional[float]:
    if close_series is None or close_series.empty or len(close_series) <= days:
        return None
    try:
        start = float(close_series.iloc[-days - 1])
        end = float(close_series.iloc[-1])
        if start <= 0:
            return None
        return (end / start - 1.0) * 100.0
    except Exception:
        return None


def _calc_rsi_14(close_series: Optional[pd.Series]) -> Optional[float]:
    if close_series is None or close_series.empty or len(close_series) < 15:
        return None
    try:
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        if gain.empty or loss.empty:
            return None
        avg_gain = float(gain.iloc[-1])
        avg_loss = float(loss.iloc[-1])
        if avg_loss <= 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception:
        return None


def _infer_momentum_label(ret_20d_pct: Optional[float]) -> str:
    if ret_20d_pct is None:
        return "NEUTRAL"
    if ret_20d_pct >= 5.0:
        return "UPTREND"
    if ret_20d_pct <= -5.0:
        return "DOWNTREND"
    return "NEUTRAL"


def _normalize_sector_weights(raw: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        iterator = raw.items()
    elif isinstance(raw, list):
        iterator = []
        for item in raw:
            if isinstance(item, dict):
                for k, v in item.items():
                    iterator.append((k, v))
    else:
        return rows

    for key, val in iterator:
        try:
            weight = float(val)
        except (TypeError, ValueError):
            continue
        if weight <= 1.0:
            weight *= 100.0
        rows.append({"sector": str(key), "weight_pct": round(weight, 2)})
    rows.sort(key=lambda x: x.get("weight_pct", 0.0), reverse=True)
    return rows[:5]


def _extract_top_holdings(info: Dict[str, Any], fallback: List[str]) -> List[str]:
    holdings: List[str] = []
    candidates = info.get("holdings")
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict):
                symbol = (
                    item.get("symbol")
                    or item.get("holdingSymbol")
                    or item.get("holdingName")
                    or item.get("name")
                )
                if symbol:
                    holdings.append(str(symbol))
            elif isinstance(item, str):
                holdings.append(item)
    alt = info.get("topHoldings")
    if not holdings and isinstance(alt, list):
        for item in alt:
            if isinstance(item, dict):
                symbol = item.get("symbol") or item.get("holdingSymbol") or item.get("holdingName") or item.get("name")
                if symbol:
                    holdings.append(str(symbol))
            elif isinstance(item, str):
                holdings.append(item)
    deduped: List[str] = []
    for name in holdings:
        token = str(name).strip()
        if token and token not in deduped:
            deduped.append(token)
    if deduped:
        return deduped[:5]
    return list(fallback[:5]) if isinstance(fallback, list) else []


def _configure_yfinance_cache_dir(yf_module: Any) -> None:
    cache_dir = os.getenv("EIMAS_YFINANCE_CACHE_DIR", "/tmp/eimas_yfinance_cache").strip()
    if not cache_dir or cache_dir.lower() in {"off", "none", "disable", "false", "0"}:
        return
    target = Path(cache_dir).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    try:
        if hasattr(yf_module, "set_tz_cache_location"):
            yf_module.set_tz_cache_location(str(target))
    except Exception:
        return


def _download_etf_price_frames(tickers: List[str], lookback_days: int) -> Dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except Exception:
        return {}
    _configure_yfinance_cache_dir(yf)

    if not tickers:
        return {}

    period_days = max(int(lookback_days), 120)
    frames: Dict[str, pd.DataFrame] = {}
    data = None
    try:
        data = yf.download(
            tickers,
            period=f"{period_days}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=False,
        )
    except Exception:
        data = None

    if isinstance(data, pd.DataFrame) and not data.empty:
        multi_level = isinstance(data.columns, pd.MultiIndex)
        for ticker in tickers:
            try:
                if multi_level:
                    if ticker not in data.columns.get_level_values(0):
                        continue
                    df = data[ticker].copy()
                else:
                    df = data.copy()
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                if "Close" not in df.columns:
                    continue
                frames[ticker] = df.dropna(subset=["Close"]).copy()
            except Exception:
                continue

    missing = [ticker for ticker in tickers if ticker not in frames]
    for ticker in missing:
        try:
            df = yf.download(
                ticker,
                period=f"{period_days}d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if "Close" not in df.columns:
                continue
            frames[ticker] = df.dropna(subset=["Close"]).copy()
        except Exception:
            continue
    return frames


def _download_etf_info_map(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    try:
        import yfinance as yf
    except Exception:
        return {}
    _configure_yfinance_cache_dir(yf)

    info_map: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            if isinstance(info, dict) and info:
                info_map[ticker] = info
        except Exception:
            continue
    return info_map


def _build_synthetic_etf_frame(
    ticker: str,
    lookback_days: int,
    anchor_price: Optional[float] = None,
) -> pd.DataFrame:
    points = max(120, min(int(lookback_days), 756))
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=points)
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(ticker))
    rng = random.Random(f"eimas-etf-{ticker}-{seed}")
    phase = (seed % 29) / 7.0
    drift = ((seed % 9) - 4) * 0.0002
    price = float(anchor_price) if isinstance(anchor_price, (int, float)) and anchor_price > 1.0 else (80.0 + float(seed % 70))
    closes: List[float] = []
    volumes: List[int] = []
    for i, _ in enumerate(dates):
        cycle = 0.0028 * math.sin((i / 13.0) + phase)
        shock = (rng.random() - 0.5) * 0.010
        ret = drift + cycle + shock
        price = max(5.0, price * (1.0 + ret))
        closes.append(round(price, 4))
        vol_base = 700_000 + (seed % 500_000)
        volume = int(vol_base * (1.0 + 0.35 * abs(math.sin((i / 9.0) + phase))) + rng.randint(0, 60_000))
        volumes.append(max(volume, 100_000))

    return pd.DataFrame({"Close": closes, "Volume": volumes}, index=dates)


def _enrich_ra_etf_snapshot(
    snapshot: Any,
    lookback_days: int,
) -> List[Dict[str, Any]]:
    base_rows = snapshot if isinstance(snapshot, list) else []
    base_map: Dict[str, Dict[str, Any]] = {}
    for item in base_rows:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).upper().strip()
        if ticker:
            base_map[ticker] = item

    tickers = _resolve_ra_etf_tickers()
    for ticker in base_map.keys():
        if ticker not in tickers:
            tickers.append(ticker)

    price_frames = _download_etf_price_frames(tickers=tickers, lookback_days=lookback_days)
    info_map = _download_etf_info_map(tickers=tickers)
    synthetic_tickers: set[str] = set()

    allow_synthetic = _env_flag("EIMAS_RA_ETF_ALLOW_SYNTHETIC_FALLBACK", default=True)
    if allow_synthetic:
        for ticker in tickers:
            if ticker in price_frames:
                continue
            base = base_map.get(ticker, {})
            anchor = _safe_numeric(base.get("last_close")) if isinstance(base, dict) else None
            synthetic_df = _build_synthetic_etf_frame(
                ticker=ticker,
                lookback_days=lookback_days,
                anchor_price=anchor,
            )
            if isinstance(synthetic_df, pd.DataFrame) and not synthetic_df.empty:
                price_frames[ticker] = synthetic_df
                synthetic_tickers.add(ticker)

    spy_ret_20 = None
    spy_df = price_frames.get("SPY")
    if isinstance(spy_df, pd.DataFrame) and not spy_df.empty:
        spy_ret_20 = _calc_return_pct(spy_df.get("Close"), 20)

    rows: List[Dict[str, Any]] = []
    for ticker in tickers:
        base = base_map.get(ticker, {})
        profile = _ETF_PROFILE_CATALOG.get(ticker, {})
        info = info_map.get(ticker, {})
        df = price_frames.get(ticker)
        close_series = df.get("Close") if isinstance(df, pd.DataFrame) and "Close" in df.columns else None
        vol_series = df.get("Volume") if isinstance(df, pd.DataFrame) and "Volume" in df.columns else None

        source_parts: List[str] = []
        if base:
            source_parts.append("financial_indicators")
        if close_series is not None and not close_series.empty and ticker not in synthetic_tickers:
            source_parts.append("yfinance_price")
        if ticker in synthetic_tickers:
            source_parts.append("synthetic_price")
        if info:
            source_parts.append("yfinance_info")
        if not source_parts:
            source_parts.append("catalog_fallback")

        base_ret_5 = _safe_numeric(base.get("ret_5d_pct")) if isinstance(base, dict) else None
        base_ret_20 = _safe_numeric(base.get("ret_20d_pct")) if isinstance(base, dict) else None
        ret_1 = _calc_return_pct(close_series, 1)
        ret_5 = base_ret_5 if base_ret_5 is not None else _calc_return_pct(close_series, 5)
        ret_20 = base_ret_20 if base_ret_20 is not None else _calc_return_pct(close_series, 20)
        ret_60 = _calc_return_pct(close_series, 60)

        last_close = _safe_numeric(base.get("last_close")) if isinstance(base, dict) else None
        if last_close is None and close_series is not None and not close_series.empty:
            last_close = _safe_numeric(close_series.iloc[-1])

        volume_ratio = None
        if vol_series is not None and len(vol_series) >= 20:
            try:
                latest_vol = float(vol_series.iloc[-1])
                avg20 = float(vol_series.iloc[-20:].mean())
                if avg20 > 0:
                    volume_ratio = latest_vol / avg20
            except Exception:
                volume_ratio = None

        rsi_14 = _calc_rsi_14(close_series)
        relative_strength = None
        if ret_20 is not None and spy_ret_20 is not None:
            relative_strength = ret_20 - spy_ret_20

        fallback_holdings = profile.get("top_holdings", []) if isinstance(profile.get("top_holdings"), list) else []
        top_holdings = _extract_top_holdings(info, fallback_holdings)
        sector_weights = _normalize_sector_weights(info.get("sectorWeightings"))
        if not sector_weights:
            catalog_weights = profile.get("sector_weights")
            if isinstance(catalog_weights, list):
                sector_weights = catalog_weights[:5]

        momentum = str(base.get("momentum_label", "")).strip() if isinstance(base, dict) else ""
        if not momentum:
            momentum = _infer_momentum_label(ret_20)

        quality = "FALLBACK"
        if "yfinance_price" in source_parts and "yfinance_info" in source_parts:
            quality = "COMPLETE"
        elif "synthetic_price" in source_parts:
            quality = "SYNTHETIC"
        elif "yfinance_price" in source_parts or "financial_indicators" in source_parts:
            quality = "PARTIAL"

        row: Dict[str, Any] = {
            "ticker": ticker,
            "etf_name": str(info.get("longName") or info.get("shortName") or profile.get("etf_name") or ticker),
            "category": str(profile.get("category") or info.get("category") or "unknown"),
            "asset_role": str(profile.get("asset_role") or "Satellite"),
            "factor_exposure": str(profile.get("factor_exposure") or "N/A"),
            "sector_or_theme": str(profile.get("sector_or_theme") or info.get("sector") or "N/A"),
            "duration_profile": str(profile.get("duration_profile") or "N/A"),
            "last_close": round(float(last_close), 4) if last_close is not None else None,
            "ret_1d_pct": round(float(ret_1), 2) if ret_1 is not None else None,
            "ret_5d_pct": round(float(ret_5), 2) if ret_5 is not None else None,
            "ret_20d_pct": round(float(ret_20), 2) if ret_20 is not None else None,
            "ret_60d_pct": round(float(ret_60), 2) if ret_60 is not None else None,
            "relative_strength_20d": round(float(relative_strength), 2) if relative_strength is not None else None,
            "volume_ratio_20d": round(float(volume_ratio), 3) if volume_ratio is not None else None,
            "rsi_14": round(float(rsi_14), 2) if rsi_14 is not None else None,
            "momentum_label": momentum,
            "expense_ratio_pct": round(float(_safe_ratio_pct(info.get("annualReportExpenseRatio") or info.get("expenseRatio"))), 3)
            if _safe_ratio_pct(info.get("annualReportExpenseRatio") or info.get("expenseRatio")) is not None
            else None,
            "dividend_yield_pct": round(float(_safe_ratio_pct(info.get("yield") or info.get("dividendYield") or info.get("trailingAnnualDividendYield"))), 3)
            if _safe_ratio_pct(info.get("yield") or info.get("dividendYield") or info.get("trailingAnnualDividendYield")) is not None
            else None,
            "total_assets_usd_bn": round(float(info.get("totalAssets")) / 1e9, 2)
            if _safe_numeric(info.get("totalAssets")) is not None
            else None,
            "holdings_count": _safe_count(info.get("holdingsCount"))
            or _safe_count(info.get("numberOfHoldings"))
            or len(top_holdings),
            "top_holdings": top_holdings,
            "sector_weights": sector_weights,
            "data_source": ",".join(source_parts),
            "synthetic_price_fallback": ticker in synthetic_tickers,
            "quality_flag": quality,
        }
        rows.append(row)

    rows.sort(key=lambda x: _safe_numeric(x.get("ret_20d_pct")) or -9999.0, reverse=True)
    return rows


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
        import sys
        if str(fi_root.parent) not in sys.path:
            sys.path.insert(0, str(fi_root.parent))
        package_name = fi_root.name
        if not package_name.isidentifier():
            logger.warning("financial_indicators package name invalid: %s", package_name)
            return {}

        fi_collectors = importlib.import_module(f"{package_name}.collectors")

        classes = {
            "FREDCollector": getattr(fi_collectors, "FREDCollector", None),
            "MarketCollector": getattr(fi_collectors, "MarketCollector", None),
            "CryptoCollector": getattr(fi_collectors, "CryptoCollector", None),
            "CompanyRACollector": getattr(fi_collectors, "CompanyRACollector", None),
        }

        if not any(classes.values()):
            logger.warning("financial_indicators collectors not found in %s", fi_root)
            return {}

        logger.info("financial_indicators linked from %s", fi_root)
        return classes

    except Exception as e:
        log_error(logger, "Failed to initialize financial_indicators bridge", e)
        return {}


def _collect_market_data_via_financial_indicators(
    lookback_days: int,
    include_crypto: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Collect market data via financial_indicators collectors."""
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

    include_market_crypto = include_crypto and _env_flag(
        "EIMAS_INCLUDE_MARKET_CRYPTO",
        default=not alpha_probe_mode,
    )
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


def _collect_company_ra_via_financial_indicators(lookback_days: int) -> Dict[str, Any]:
    """
    Collect company-level accounting/valuation outputs for RA workflow.
    This is best-effort and returns {} when bridge collector is unavailable.
    """
    classes = _load_financial_indicators()
    ra_cls = classes.get("CompanyRACollector")
    if ra_cls is None:
        return {}

    tickers = _resolve_ra_company_tickers()

    try:
        collector = ra_cls(lookback_days=lookback_days)
    except TypeError:
        collector = ra_cls()

    try:
        return collector.collect_all(tickers=tickers)
    except TypeError:
        return collector.collect_all(tickers)
    except Exception as e:
        log_error(logger, "Company RA collection failed via financial_indicators", e)
        return {}


def _safe_numeric(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _serialize_ohlcv_rows(df: pd.DataFrame) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    if df is None or df.empty:
        return rows

    normalized = df.sort_index()
    for idx, row in normalized.iterrows():
        date_value = idx.date().isoformat() if hasattr(idx, "date") else str(idx)
        item: Dict[str, Any] = {
            "date": date_value,
            "close": _safe_numeric(row.get("Close")),
            "open": _safe_numeric(row.get("Open")),
            "high": _safe_numeric(row.get("High")),
            "low": _safe_numeric(row.get("Low")),
            "volume": _safe_numeric(row.get("Volume")),
        }
        if item["close"] is None:
            continue
        rows.append(item)
    return rows


def build_financial_indicators_bridge_payload(
    kind: str,
    series: Dict[str, pd.DataFrame],
    lookback_days: int,
) -> Dict[str, Any]:
    """
    Build schema-compatible payload for financial_indicators bridge output.
    Schema reference:
      docs/references/financial_indicators_bridge_payload_v1.schema.json
    """
    if kind not in {"market", "crypto"}:
        raise ValueError(f"Unsupported bridge payload kind: {kind}")

    serialized_series: Dict[str, list[Dict[str, Any]]] = {}
    for ticker, df in series.items():
        rows = _serialize_ohlcv_rows(df)
        if rows:
            serialized_series[ticker] = rows

    return {
        "schema_version": "fi_bridge_v1",
        "source": "financial_indicators",
        "kind": kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": int(lookback_days),
        "series": serialized_series,
    }

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

def collect_market_data(lookback_days: int = 365, include_crypto: bool = True) -> Dict[str, pd.DataFrame]:
    """시장 데이터 수집"""
    print("\n[1.2] Collecting market data...")
    try:
        fi_data = _collect_market_data_via_financial_indicators(
            lookback_days,
            include_crypto=include_crypto,
        )
        if fi_data:
            print(f"      ✓ Collected {len(fi_data)} tickers (financial_indicators)")
            return fi_data

        dm = DataManager(lookback_days=lookback_days)
        tickers_config = {
            'market': [
                {'ticker': 'SPY'}, {'ticker': 'QQQ'}, {'ticker': 'IWM'},
                {'ticker': 'DIA'}, {'ticker': 'TLT'}, {'ticker': 'GLD'},
                {'ticker': 'USO'}, {'ticker': 'UUP'}, {'ticker': '^VIX'}
            ]
        }
        if include_crypto:
            tickers_config['crypto'] = [
                {'ticker': 'BTC-USD'}, {'ticker': 'ETH-USD'}
            ]
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


def collect_korea_savings_bank_indicators():
    """한국 저축은행 건전성 지표 수집 및 DB 저장 (NPL, BIS, ROA)."""
    print("\n[1.6] Collecting Korea Savings Bank Indicators...")
    try:
        from lib.korea_savings_bank import (
            KoreaSavingsBankIndicators,
            collect_korea_savings_bank_indicators as _collect,
        )
        result = _collect()

        # DB 저장 — DatabaseManager 기존 패턴 사용
        try:
            from core.database import DatabaseManager
            db = DatabaseManager()
            db.save_korea_savings_bank(result.to_dict())
        except Exception as db_exc:
            log_error(logger, "Korea savings bank DB save failed", db_exc)

        print(
            f"      ✓ NPL={result.npl_ratio:.1f}%  "
            f"BIS={result.bis_capital_ratio:.1f}%  "
            f"ROA={result.roa:.2f}%  [{result.data_source}]"
        )
        return result
    except Exception as e:
        log_error(logger, "Korea savings bank indicators collection failed", e)
        from lib.korea_savings_bank import KoreaSavingsBankIndicators
        return KoreaSavingsBankIndicators(
            timestamp=datetime.now().isoformat(),
            is_valid=False,
            error_msg=str(e),
        )


def collect_company_ra_analysis(lookback_days: int = 365) -> Dict[str, Any]:
    """RA-focused company accounting + valuation analysis (financial_indicators bridge)."""
    print("\n[1.5] Collecting RA Company Analysis...")
    try:
        data = _collect_company_ra_via_financial_indicators(lookback_days=lookback_days)
        companies = data.get("companies", []) if isinstance(data, dict) else []
        if isinstance(data, dict):
            raw_snapshot = data.get("etf_strategy_snapshot", [])
            enriched_snapshot = _enrich_ra_etf_snapshot(
                snapshot=raw_snapshot,
                lookback_days=lookback_days,
            )
            data["etf_strategy_snapshot"] = enriched_snapshot

            ra_support = data.get("ra_work_support", {})
            if not isinstance(ra_support, dict):
                ra_support = {}
            ra_support["etf_coverage_count"] = len(enriched_snapshot)
            ra_support.setdefault(
                "data_update_note",
                "ETF 가격/메타데이터는 시장 데이터(일간)와 펀드 메타데이터(저빈도)의 갱신 주기를 분리해 관리",
            )
            research_tasks = ra_support.get("research_tasks", [])
            if isinstance(research_tasks, list):
                extra_task = "ETF 보유종목/섹터 비중 스냅샷 업데이트 (top holdings + sector weights)"
                if extra_task not in research_tasks:
                    research_tasks.append(extra_task)
                ra_support["research_tasks"] = research_tasks
            data["ra_work_support"] = ra_support

            as_of_date = datetime.now().date().isoformat()
            data["internal_sql"] = ingest_company_ra_analysis_to_sql(
                company_ra_analysis=data,
                as_of_date=as_of_date,
            )
        etf_count = len(data.get("etf_strategy_snapshot", [])) if isinstance(data, dict) else 0
        print(f"      ✓ RA company analysis: {len(companies)} companies / {etf_count} ETFs")
        return data if isinstance(data, dict) else {}
    except Exception as e:
        log_error(logger, "Company RA analysis collection failed", e)
        return {}
