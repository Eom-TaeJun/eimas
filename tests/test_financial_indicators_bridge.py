#!/usr/bin/env python3
"""
Smoke tests for financial_indicators bridge wiring in pipeline.collectors.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

# Ensure project root is importable in pytest runs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline.collectors as bridge


def _fake_df(price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [price - 1.0, price],
            "High": [price, price + 1.0],
            "Low": [price - 2.0, price - 1.0],
            "Close": [price, price + 0.5],
            "Volume": [1000, 1200],
        },
        index=pd.to_datetime(["2026-02-06", "2026-02-07"]),
    )


class _FakeMarketCollector:
    def __init__(self, lookback_days: int, use_alpha_vantage: bool = False):
        self.lookback_days = lookback_days
        self.use_alpha_vantage = use_alpha_vantage

    def fetch_ticker(self, ticker: str, name: str):
        if ticker in {"SPY", "QQQ"}:
            return _fake_df(100.0), {"source": "fake_market"}
        return pd.DataFrame(), {"source": "fake_market"}


class _FakeCryptoCollector:
    def __init__(self, lookback_days: int):
        self.lookback_days = lookback_days

    def fetch_ticker(self, ticker: str, name: str):
        if ticker in {"BTC-USD", "ETH-USD", "SOL-USD"}:
            return _fake_df(200.0), {"source": "fake_crypto"}
        return pd.DataFrame(), {"source": "fake_crypto"}


class _FakeCompanyRACollector:
    def __init__(self, lookback_days: int = 365):
        self.lookback_days = lookback_days

    def collect_all(self, tickers):
        return {
            "timestamp": "2026-02-08T00:00:00",
            "companies": [{"ticker": ticker, "valuation_signal": "FAIR"} for ticker in tickers],
            "postgresql": {"enabled": False, "stored_rows": 0},
        }


def test_collect_market_data_via_bridge_market_only(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "_load_financial_indicators",
        lambda: {
            "MarketCollector": _FakeMarketCollector,
            "CryptoCollector": _FakeCryptoCollector,
        },
    )

    data = bridge._collect_market_data_via_financial_indicators(
        lookback_days=30,
        include_crypto=False,
    )

    assert set(data.keys()) == {"SPY", "QQQ"}
    assert all(not df.empty for df in data.values())


def test_collect_market_data_via_bridge_with_crypto(monkeypatch):
    monkeypatch.setenv("EIMAS_USE_ALPHA_VANTAGE", "false")
    monkeypatch.setenv("EIMAS_INCLUDE_MARKET_CRYPTO", "true")

    monkeypatch.setattr(
        bridge,
        "_load_financial_indicators",
        lambda: {
            "MarketCollector": _FakeMarketCollector,
            "CryptoCollector": _FakeCryptoCollector,
        },
    )

    data = bridge._collect_market_data_via_financial_indicators(
        lookback_days=30,
        include_crypto=True,
    )

    assert {"SPY", "QQQ", "BTC-USD", "ETH-USD"}.issubset(set(data.keys()))


def test_collect_crypto_data_via_bridge(monkeypatch):
    monkeypatch.setattr(
        bridge,
        "_load_financial_indicators",
        lambda: {
            "CryptoCollector": _FakeCryptoCollector,
        },
    )

    data = bridge._collect_crypto_data_via_financial_indicators(lookback_days=30)
    assert set(data.keys()) == {"BTC-USD", "ETH-USD", "SOL-USD"}


def test_collect_company_ra_analysis_via_bridge(monkeypatch):
    monkeypatch.setenv("EIMAS_RA_COMPANY_TICKERS", "AAPL,MSFT")
    monkeypatch.setattr(
        bridge,
        "_load_financial_indicators",
        lambda: {
            "CompanyRACollector": _FakeCompanyRACollector,
        },
    )

    data = bridge.collect_company_ra_analysis(lookback_days=120)
    tickers = [row["ticker"] for row in data["companies"]]
    assert tickers == ["AAPL", "MSFT"]


def test_load_financial_indicators_returns_empty_on_invalid_layout(tmp_path, monkeypatch):
    bridge._load_financial_indicators.cache_clear()
    monkeypatch.setenv("EIMAS_FINANCIAL_INDICATORS_PATH", str(tmp_path))

    # tmp_path exists, but required files (config.py, collectors/__init__.py) do not.
    classes = bridge._load_financial_indicators()
    assert classes == {}

    bridge._load_financial_indicators.cache_clear()
    monkeypatch.delenv("EIMAS_FINANCIAL_INDICATORS_PATH", raising=False)


def test_load_financial_indicators_keeps_global_config_module(tmp_path, monkeypatch):
    fi_root = tmp_path / "financial_indicators"
    (fi_root / "collectors").mkdir(parents=True)
    (fi_root / "config.py").write_text("FRED_API_KEY='x'\n", encoding="utf-8")
    (fi_root / "collectors" / "__init__.py").write_text("", encoding="utf-8")

    fake_collectors = SimpleNamespace(
        FREDCollector=object,
        MarketCollector=object,
        CryptoCollector=object,
    )

    sentinel_config = ModuleType("config")
    monkeypatch.setitem(sys.modules, "config", sentinel_config)
    monkeypatch.setenv("EIMAS_FINANCIAL_INDICATORS_PATH", str(fi_root))
    monkeypatch.setattr(bridge.importlib, "import_module", lambda _: fake_collectors)

    bridge._load_financial_indicators.cache_clear()
    classes = bridge._load_financial_indicators()

    assert classes["MarketCollector"] is object
    assert sys.modules.get("config") is sentinel_config

    bridge._load_financial_indicators.cache_clear()
    monkeypatch.delenv("EIMAS_FINANCIAL_INDICATORS_PATH", raising=False)


def test_build_bridge_payload_schema_shape():
    payload = bridge.build_financial_indicators_bridge_payload(
        kind="market",
        series={"SPY": _fake_df(123.0)},
        lookback_days=30,
    )

    assert payload["schema_version"] == "fi_bridge_v1"
    assert payload["source"] == "financial_indicators"
    assert payload["kind"] == "market"
    assert payload["lookback_days"] == 30
    assert "SPY" in payload["series"]
    assert payload["series"]["SPY"][0]["date"] == "2026-02-06"
    assert payload["series"]["SPY"][0]["close"] is not None


def test_build_bridge_payload_invalid_kind():
    with pytest.raises(ValueError):
        bridge.build_financial_indicators_bridge_payload(
            kind="unknown",
            series={},
            lookback_days=30,
        )


def _resolve_real_fi_root() -> Path | None:
    configured = os.getenv("EIMAS_FINANCIAL_INDICATORS_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if (path / "config.py").exists() and (path / "collectors" / "__init__.py").exists():
            return path

    candidate = Path(__file__).resolve().parents[2] / "financial_indicators"
    if (candidate / "config.py").exists() and (candidate / "collectors" / "__init__.py").exists():
        return candidate
    return None


def test_load_financial_indicators_real_import_smoke(monkeypatch):
    fi_root = _resolve_real_fi_root()
    if fi_root is None:
        pytest.skip("real financial_indicators project not found for smoke test")

    bridge._load_financial_indicators.cache_clear()
    monkeypatch.setenv("EIMAS_FINANCIAL_INDICATORS_PATH", str(fi_root))

    classes = bridge._load_financial_indicators()

    assert classes.get("MarketCollector") is not None
    assert classes.get("CryptoCollector") is not None
    assert callable(classes["MarketCollector"])
    assert callable(classes["CryptoCollector"])

    bridge._load_financial_indicators.cache_clear()
    monkeypatch.delenv("EIMAS_FINANCIAL_INDICATORS_PATH", raising=False)


def test_load_financial_indicators_real_import_keeps_config(monkeypatch):
    fi_root = _resolve_real_fi_root()
    if fi_root is None:
        pytest.skip("real financial_indicators project not found for smoke test")

    sentinel_config = ModuleType("config")
    monkeypatch.setitem(sys.modules, "config", sentinel_config)
    monkeypatch.setenv("EIMAS_FINANCIAL_INDICATORS_PATH", str(fi_root))

    bridge._load_financial_indicators.cache_clear()
    _ = bridge._load_financial_indicators()

    assert sys.modules.get("config") is sentinel_config

    bridge._load_financial_indicators.cache_clear()
    monkeypatch.delenv("EIMAS_FINANCIAL_INDICATORS_PATH", raising=False)
