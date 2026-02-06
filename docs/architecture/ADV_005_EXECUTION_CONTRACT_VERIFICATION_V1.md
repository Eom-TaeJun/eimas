# ADV-005: Execution Contract Verification v1 (2026-02-06)

## 1. Purpose
- `EXECUTION_DOMAIN_CONTRACT_V1`을 코드 변경마다 일관되게 검증하기 위한 체크리스트.
- General Lane 작업의 완료 기준을 정량화한다.

## 2. Input Matrix

### Case A: local backend
- env: `EIMAS_EXECUTION_BACKEND=local`
- expected module source:
  - allocation: `lib.allocation_engine`
  - rebalancing: `lib.rebalancing_policy`

### Case B: external backend
- env: `EIMAS_EXECUTION_BACKEND=external`
- expected module source:
  - allocation: `execution_intelligence.models.allocation_engine`
  - rebalancing: `execution_intelligence.models.rebalancing_policy`

## 3. Required Assertions

### 3.1 Allocation Assertions
- returned dict keys must include:
  - `allocation_result`
  - `allocation_strategy`
  - `allocation_config`
  - `status`
  - `warnings`
- `allocation_result.weights` exists when `status=SUCCESS`
- weight sum tolerance:
  - `abs(sum(weights)-1.0) <= 0.02`

### 3.2 Rebalancing Assertions
- returned dict keys include:
  - `should_rebalance`
  - `action`
  - `reason`
- `action in {REBALANCE, PARTIAL, HOLD}`

### 3.3 Operational Bundle Assertions
- bundle keys include:
  - `op_report`
  - `operational_report`
  - `input_validation`
  - `indicator_classification`
  - `operational_controls`
  - `audit_metadata`
  - `approval_status`

## 4. Test Commands (Per-Change)

### 4.1 Compile
```bash
python3 -m py_compile \
  lib/adapters/execution_backend.py \
  lib/adapters/execution_models.py \
  pipeline/analyzers.py \
  main.py
```

### 4.2 Backend Source Check
```bash
python3 - <<'PY'
import os, sys
sys.path.insert(0, '/home/tj/projects/autoai/eimas')
for b in ('local', 'external'):
    os.environ['EIMAS_EXECUTION_BACKEND'] = b
    import importlib
    import pipeline.analyzers as a
    importlib.reload(a)
    print(b, a.AllocationEngine.__module__, a.RebalancingPolicy.__module__)
PY
```

### 4.3 Allocation/Rebalancing Smoke
```bash
python3 - <<'PY'
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/tj/projects/autoai/eimas')
os.environ['EIMAS_EXECUTION_BACKEND'] = 'external'
from pipeline.analyzers import run_allocation_engine
idx = pd.date_range('2025-01-01', periods=40, freq='D')
md = {}
for i,t in enumerate(['SPY','TLT','GLD','QQQ']):
    md[t] = pd.DataFrame({'Close': 100 + np.cumsum(np.random.normal(0,1,size=len(idx)))}, index=idx)
res = run_allocation_engine(md, strategy='risk_parity', current_weights={'SPY':0.25,'TLT':0.25,'GLD':0.25,'QQQ':0.25})
print('status', res.get('status'))
print('action', res.get('rebalance_decision', {}).get('action'))
w = res.get('allocation_result', {}).get('weights', {})
if w:
    print('weight_sum', sum(w.values()))
PY
```

## 5. Fail Conditions
- backend switch가 local/external 둘 중 하나에서 import 실패
- `status=SUCCESS`인데 `weights` 누락
- rebalance `action` enum 외 값
- operational bundle required key 누락

## 6. Acceptance
- Case A/B 모두 source check 통과
- smoke test 통과
- fail condition 0건

## 7. Work Order Mapping
- `GEN-203`: 본 체크리스트를 `scripts/check_execution_contract.sh`로 자동화
