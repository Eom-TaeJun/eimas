# Execution Domain Contract v1 (2026-02-06)

## 1. Purpose
- `execution_intelligence` 분리 과정에서 EIMAS와 실행 도메인 간 I/O를 고정한다.
- 분리 중에도 `main.py --full` 결과 필드를 안정적으로 유지한다.
- General Lane 작업은 이 계약을 벗어나면 안 된다.

## 2. Scope
- Allocation
- Rebalancing
- Tactical Allocation (next split target)
- Stress Test (next split target)
- Operational Governance

## 3. Canonical Interfaces

### 3.1 Allocation
Function:
- `run_allocation_engine(market_data, strategy, constraints, current_weights) -> Dict`

Required top-level keys:
- `allocation_result` (dict)
- `rebalance_decision` (dict, optional if `current_weights` missing)
- `allocation_strategy` (str)
- `allocation_config` (dict)
- `status` (`SUCCESS|HOLD`)
- `warnings` (list[str])

`allocation_result` required keys:
- `weights` (dict[str,float])
- `strategy` (str)
- `optimization_status` (str)
- `is_fallback` (bool)

Invariants:
- `weights` sum should be `1.0 +/- 0.02`
- all weights `>= 0` (long-only default)

### 3.2 Rebalancing
Function:
- `run_rebalancing_policy(current_weights, target_weights, last_rebalance_date, market_data_quality, config) -> Dict`

Required keys:
- `should_rebalance` (bool)
- `action` (`REBALANCE|PARTIAL|HOLD`)
- `reason` (str)

Recommended keys:
- `turnover` (float)
- `estimated_cost` (float)
- `warnings` (list[str])

Failure fallback:
- `action=HOLD`, `should_rebalance=False`, `reason`에 오류 메시지 포함

### 3.3 Tactical Allocation (Wave 1-B target)
Current EIMAS field:
- `result.tactical_weights: Dict[str,float]`

Contract requirement:
- output payload key: `tactical_weights`
- weights keyset must match input strategic universe (or explicit subset + normalization)
- sum invariant: `1.0 +/- 0.02`

Failure fallback:
- `tactical_weights = strategic_weights` (no-op)
- warning append

### 3.4 Stress Test (Wave 1-B target)
Current EIMAS field:
- `result.stress_test_results: Dict`

Required structure:
- `historical`: list[StressScenarioResult]
- `hypothetical`: list[StressScenarioResult]
- `extreme`: StressScenarioResult

`StressScenarioResult` required keys:
- `scenario_name` (str)
- `initial_value` (float)
- `stressed_value` (float)
- `loss` (float)
- `loss_pct` (float)
- `var_breach` (bool)

Failure fallback:
- empty payload allowed:
  - `historical=[]`, `hypothetical=[]`, `extreme={}`
- warning append

### 3.5 Operational Governance
Function:
- `generate_operational_bundle(result_data, current_weights, rebalance_decision) -> OperationalBundle`

Required top-level keys:
- `op_report`
- `operational_report`
- `input_validation`
- `indicator_classification`
- `operational_controls`
- `audit_metadata`
- `approval_status`

Source of truth:
- `execution_intelligence/core/contracts.py` (`OperationalBundle`)

Failure fallback:
- adapter must return local backend result or deterministic safe payload

## 4. EIMAS Field Mapping (Must Preserve)
- `allocation_result` <- allocation contract
- `rebalance_decision` <- rebalancing contract
- `allocation_strategy` <- allocation contract
- `allocation_config` <- allocation contract
- `tactical_weights` <- tactical contract
- `stress_test_results` <- stress contract
- `operational_report` <- operational bundle
- `input_validation` <- operational bundle
- `indicator_classification` <- operational bundle
- `trade_plan` <- derived from operational/rebalance
- `operational_controls` <- operational bundle
- `audit_metadata` <- operational bundle
- `approval_status` <- operational bundle

## 5. Adapter Runtime Policy
- selector env:
  - `EIMAS_EXECUTION_BACKEND=local|external`
- external 실패 시 local fallback
- fallback 시 `warnings` 또는 `reason`을 통해 trace 남김

## 6. Validation Checklist (Per-Change)
- `py_compile` touched files
- import smoke:
  - allocation/rebalancing classes module path check (`local`/`external`)
- function smoke:
  - `run_allocation_engine` returns required keys
  - rebalancing action enum valid
- invariants:
  - weights sum tolerance
  - required keys non-null

## 7. Non-Goals (v1)
- tactical/stress 모델의 수학적 로직 개선
- operational decision policy 자체 변경
- full pipeline 성능 튜닝

## 8. Change Control
- 계약 변경은 Advanced Lane 승인(ADR 수준) 없이 금지
- General Lane은 계약 내 구현/이동/배선만 수행
