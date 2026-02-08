# sys.path.insert Reduction Plan (2026-02-07)

## 1) Current Snapshot

- Active `sys.path.insert(...)` in `*.py`: `4` cases (down from `13`).
- Completed in this wave:
  - Added `scripts/_project_bootstrap.py`
  - Consolidated direct-script bootstrap in 8 script files.
  - Added `lib/path_bootstrap.py`
  - Consolidated dynamic external-path bootstrap in execution/collector adapters.

Current remaining files:
1. `api/main.py`
2. `cli/eimas.py`
3. `scripts/_project_bootstrap.py`
4. `lib/path_bootstrap.py`

## 2) Classification

1. Entrypoint compatibility (`api/main.py`, `cli/eimas.py`)
- Reason: keeps `python api/main.py` / `python cli/eimas.py` compatibility.
- Risk: medium (import path behavior differs by invocation method).

2. Script/bootstrap helpers (`scripts/_project_bootstrap.py`, `lib/path_bootstrap.py`)
- Reason: centralized path bootstrap points for script execution and dynamic external adapters.
- Risk: low (intended to remain until all direct script entrypoints are migrated).

## 3) Planned Steps

### Step A (Low risk): Entrypoint normalization
1. Standardize recommended run style:
   - API: `uvicorn api.main:app --reload --port 8000`
   - CLI: `python -m cli.eimas ...`
2. Add module-first import guard in `api/main.py`, `cli/eimas.py`. ✅ Applied
3. Keep temporary fallback for direct file execution until usage migration is confirmed.

### Step B (High risk): Dynamic path loader hardening
1. Keep adapter path insertion only via shared helper (`lib/path_bootstrap.py`).
2. Add explicit source/fallback assertions to `scripts/check_execution_contract.sh` (already present for execution adapters).
3. Add one focused smoke for `pipeline/collectors.py` external root resolution path.

### Step C (Final cleanup)
1. After Step A/B verification, remove fallback insertions where possible.
2. Keep only one intentional compatibility insertion if direct-script support is still required.

## 4.1 Policy Decision (Current)

- Current remaining 4 insertions are treated as **intentional compatibility paths**:
  1. `api/main.py` (direct file execution fallback)
  2. `cli/eimas.py` (direct file execution fallback)
  3. `scripts/_project_bootstrap.py` (script bootstrap single point)
  4. `lib/path_bootstrap.py` (dynamic external adapter bootstrap single point)
- Default recommendation remains module-first execution:
  - API: `uvicorn api.main:app --reload --port 8000`
  - CLI: `python -m cli.eimas ...`
- Additional removal of (1)(2) is gated on explicit deprecation of direct-file execution mode.

## 5) Validation Gate

Run after each step:

```bash
python3 -m py_compile \
  api/main.py \
  cli/eimas.py \
  pipeline/collectors.py \
  lib/adapters/execution_backend.py \
  lib/adapters/execution_models.py \
  lib/path_bootstrap.py \
  scripts/_project_bootstrap.py

bash scripts/check_execution_contract.sh
rg -n "sys\\.path\\.insert\\(" -g "*.py"
```

## 6) Definition of Done

1. No hardcoded absolute project path in active `*.py`.
2. `sys.path.insert(...)` is either removed or intentionally documented.
3. Execution contract checks stay PASS for local/external backend modes.
