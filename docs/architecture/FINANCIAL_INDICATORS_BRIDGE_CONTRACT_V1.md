# Financial Indicators Bridge Contract v1

## 1. Scope
- Producer: sibling project `financial_indicators`
- Consumer: `eimas` pipeline bridge in `pipeline/collectors.py`
- Version: `fi_bridge_v1`

This document defines the runtime contract for `eimas` loading and using
`financial_indicators` collectors, and the JSON payload contract used when the
bridge output is serialized.

## 2. Runtime Loader Contract
- Entry: `pipeline/collectors.py::_load_financial_indicators`
- Path resolution:
  - First: `EIMAS_FINANCIAL_INDICATORS_PATH` (if set)
  - Fallback: sibling `../financial_indicators`
- Load strategy:
  - Add `fi_root.parent` to `sys.path`
  - Import `{fi_root.name}.collectors` via `importlib.import_module`
- Required layout at resolved root:
  - `config.py`
  - `collectors/__init__.py`
- Failure contract:
  - Loader returns `{}` and caller falls back to legacy collectors
  - No exception should terminate the main EIMAS pipeline
  - Existing global `sys.modules["config"]` must not be overwritten by bridge loading

## 3. Required Collector Symbols
`financial_indicators.collectors` must export these classes:
- `FREDCollector` (optional for current bridge path)
- `MarketCollector` (required for market bridge)
- `CryptoCollector` (required for crypto bridge)

Required methods:
- `MarketCollector.fetch_ticker(ticker: str, name: str) -> (DataFrame | None, status_dict)`
- `CryptoCollector.fetch_ticker(ticker: str, name: str) -> (DataFrame | None, status_dict)`

## 4. DataFrame Output Contract
Each successful ticker fetch must return a non-empty pandas DataFrame.

Required columns:
- `Close`

Recommended columns:
- `Open`, `High`, `Low`, `Volume`

Index contract:
- Datetime-like index in ascending order

## 5. Serialized JSON Contract
When bridge output is converted to JSON, it must follow:
- Schema file:
  - `docs/references/financial_indicators_bridge_payload_v1.schema.json`
- Required top-level fields:
  - `schema_version = "fi_bridge_v1"`
  - `source = "financial_indicators"`
  - `kind in {"market","crypto"}`
  - `generated_at` (ISO-8601 datetime)
  - `series` (ticker -> OHLCV rows)

## 6. Environment Flags (Bridge Behavior)
- `EIMAS_FINANCIAL_INDICATORS_PATH`: explicit project root override
- `EIMAS_USE_ALPHA_VANTAGE`: use AV path in market collector
- `EIMAS_ALPHA_FULL_SCAN`: disable AV probe mode and collect full ticker set
- `EIMAS_ALPHA_PROBE_TICKERS`: comma-separated probe tickers in AV safe mode
- `EIMAS_INCLUDE_MARKET_CRYPTO`: include BTC/ETH in market collection path

## 7. Compatibility Rules
- Bridge must remain fail-safe:
  - If the sibling project is missing or layout is invalid, `eimas` must still run.
- Symbol names above are stable API and should not be renamed without bridge update.
- Any breaking contract change must bump schema version (`fi_bridge_v2`).
