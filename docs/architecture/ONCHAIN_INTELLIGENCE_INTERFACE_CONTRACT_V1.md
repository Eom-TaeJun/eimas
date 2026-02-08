# Onchain Intelligence Interface Contract v1

## 1. Scope
- Producer: `onchain_intelligence`
- Consumer: `eimas` integrated pipeline (`pipeline/schemas.py` fields)
- Version: `onchain_bridge_v1`

This contract defines the JSON payload expected when `onchain_intelligence`
results are injected into EIMAS.

## 2. EIMAS Target Fields
The payload maps into these existing EIMAS fields:
- `onchain_risk_signals` (`List[Dict]`)
- `defi_tvl` (`Dict`)

Reference:
- `pipeline/schemas.py` (`onchain_risk_signals`, `defi_tvl`)

## 3. Required Payload Schema
- Schema file:
  - `docs/references/onchain_intelligence_bridge_payload_v1.schema.json`
- Required top-level keys:
  - `schema_version = "onchain_bridge_v1"`
  - `source = "onchain_intelligence"`
  - `generated_at` (ISO-8601 datetime)
  - `onchain_risk_signals` (array)

Optional keys:
- `run_id`
- `elapsed_seconds`
- `defi_tvl`

## 4. Signal Contract
Each risk signal item must contain:
- `severity`: `LOW | MEDIUM | HIGH | CRITICAL`
- `category`: `stablecoin | defi | rwa | system`
- `message`: non-empty text

Optional enrichment fields:
- `ticker`, `metric`, `value`, `threshold`, `timestamp`

## 5. Compatibility Rules
- Producer may add extra fields (`additionalProperties: true` on each signal).
- Consumer must ignore unknown fields.
- Breaking changes require new version (`onchain_bridge_v2`).
