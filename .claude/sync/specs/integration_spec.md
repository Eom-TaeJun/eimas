# Integration Spec: --parallel Flag

## Purpose
Define how the `--parallel` CLI flag threads through `main.py` ->
`run_integrated_pipeline()` -> `run_pipeline_phases()` -> Phase 1 parallel execution.

---

## 1. CLI Changes (main.py)

### New argument in `_build_main_parser()`

Add after the `--cron-mode` argument (around line 350):

```
parser.add_argument(
    '--parallel',
    action='store_true',
    help='Enable parallel data collection in Phase 1 (experimental)',
)
```

### New key in `_build_pipeline_kwargs()`

```python
def _build_pipeline_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    ...
    return {
        ...
        "use_parallel": args.parallel,   # <-- ADD THIS
    }
```

---

## 2. Pipeline Function Signature Changes

### run_integrated_pipeline() in main.py

Add parameter:

```
async def run_integrated_pipeline(
    ...
    use_parallel: bool = False,          # <-- ADD THIS
    pipeline_profile_name: str = "legacy",
) -> EIMASResult:
```

Thread it to `run_pipeline_phases`:

```python
output_file, _ = await run_pipeline_phases(
    ...
    use_parallel=use_parallel,           # <-- ADD THIS
)
```

### run_pipeline_phases() in orchestrator_steps.py

Add parameter:

```
async def run_pipeline_phases(
    *,
    ...
    debate_skip_reference: bool,
    pipeline_profile: PipelineProfile,
    use_parallel: bool = False,          # <-- ADD THIS
) -> Tuple[str | None, Dict[str, Any]]:
```

Replace the Phase 1 call:

```python
# BEFORE:
market_data = await runtime.run_async(
    "phase1_collect_data",
    phase1_collect_data,
    result,
    quick_mode,
)

# AFTER:
market_data = await runtime.run_async(
    "phase1_collect_data",
    _phase1_parallel_or_sequential,
    runtime, result, quick_mode, use_parallel,
)
```

---

## 3. New Helper in orchestrator_steps.py

```
async def _phase1_parallel_or_sequential(
    runtime: PhaseRuntimeTracker,
    result: EIMASResult,
    quick_mode: bool,
    use_parallel: bool,
) -> Dict[str, Any]:
    """
    Route Phase 1 to parallel or sequential execution.

    Args:
        runtime: Phase timing tracker (for sub-component recording).
        result: EIMASResult to mutate.
        quick_mode: Quick analysis flag.
        use_parallel: True when --parallel was passed.

    Returns:
        market_data dict.
    """
    if not use_parallel:
        return await phase1_collect_data(result, quick_mode)

    try:
        from pipeline.team_coordinator import TeamCoordinator
    except ImportError:
        logging.warning("team_coordinator not available; using sequential")
        return await phase1_collect_data(result, quick_mode)

    coordinator = TeamCoordinator(timeout_sec=120.0)
    try:
        return await coordinator.run_parallel_phase1(result, quick_mode)
    except Exception as exc:
        logging.warning(f"Parallel phase1 failed ({exc}); falling back to sequential")
        return await phase1_collect_data(result, quick_mode)
```

Add import at top of orchestrator_steps.py:

```python
import logging
```

---

## 4. Which Phases Run in Parallel

### Phase 1 Only (Initial Scope)

| Phase | Parallel? | Reason |
|-------|-----------|--------|
| Phase 1: Data Collection | YES | 5 independent API sources |
| Phase 2: Basic Analysis | No | Depends on Phase 1 output |
| Phase 2: Enhanced Analysis | No | Depends on Phase 2 basic |
| Phase 2: Sentiment/Bubble | No | Depends on market_data |
| Phase 3: AI Debate | No | Depends on Phase 2 results |
| Phase 4-9 | No | Sequential dependencies |

### Future Expansion (Not in scope)

- Phase 2 sub-analyses (sentiment + bubble + institutional) could be parallelized
  in a future iteration since they read from the same market_data but write to
  independent result fields.

---

## 5. Timing/Benchmarking Strategy

### Automatic Timing

The existing `PhaseRuntimeTracker` already wraps Phase 1 with timing.
The parallel path adds extra metadata:

```python
result.audit_metadata["phase1_parallel_mode"] = True
result.audit_metadata["phase1_parallel_wall_clock_sec"] = wall_clock
result.audit_metadata["phase1_parallel_sum_component_sec"] = sum_components
result.audit_metadata["phase1_parallel_speedup"] = sum_components / wall_clock
```

### Comparison Commands

```bash
# Sequential baseline
python main.py --quick
SEQ_TIME=$(jq '.audit_metadata.phase1_elapsed_sec' outputs/eimas_*.json | tail -1)

# Parallel run
python main.py --quick --parallel
PAR_TIME=$(jq '.audit_metadata.phase1_elapsed_sec' outputs/eimas_*.json | tail -1)
SPEEDUP=$(jq '.audit_metadata.phase1_parallel_speedup // "N/A"' outputs/eimas_*.json | tail -1)

echo "Sequential: ${SEQ_TIME}s, Parallel: ${PAR_TIME}s, Speedup: ${SPEEDUP}x"
```

### Success Criteria

- Phase 1 wall-clock reduction >= 40% compared to sequential
- All `phase1_component_timings` entries present (same keys as sequential)
- `phase1_parallel_speedup` >= 1.5x
- No regression in downstream phases (Phase 2+ outputs identical)
- `--parallel` flag has zero effect on non-Phase-1 execution

---

## 6. Validation Commands

```bash
# 1. Syntax check (no API keys)
python -m compileall pipeline/team_coordinator.py

# 2. Import check
python -c "from pipeline.team_coordinator import TeamCoordinator; print('OK')"

# 3. CLI help includes --parallel
python main.py --help | grep -q parallel && echo "OK: --parallel in help" || echo "FAIL"

# 4. risk_config still works (from user requirements)
python -c "from pipeline.risk_config import get_risk_config; print('OK')"

# 5. Dry-run with all data skipped (tests wiring, no API needed)
EIMAS_SKIP_MARKET_DATA=1 EIMAS_SKIP_CRYPTO_DATA=1 EIMAS_SKIP_EXTENDED_DATA=1 \
EIMAS_SKIP_KOREA_ASSETS=1 EIMAS_SKIP_MARKET_INDICATORS=1 \
  python -c "
import asyncio
from pipeline.team_coordinator import TeamCoordinator
from pipeline.schemas import EIMASResult
tc = TeamCoordinator(timeout_sec=10.0)
r = EIMASResult(timestamp='test')
d = asyncio.run(tc.run_parallel_phase1(r, quick_mode=True))
print(f'Result keys: {len(d)}, fred_summary type: {type(r.fred_summary).__name__}')
print('OK')
"

# 6. Full validation (needs API keys, --full for portfolio)
python main.py --full --parallel
jq '.portfolio_weights | keys | length' outputs/eimas_*.json | tail -1
# Expected: > 0
```

---

## 7. Files Changed Summary

| File | Change Type | Description |
|------|------------|-------------|
| `pipeline/team_coordinator.py` | NEW | TeamCoordinator class (~150-200 lines) |
| `main.py` | MODIFY | Add `--parallel` arg, `use_parallel` kwarg (~6 lines) |
| `pipeline/app/orchestrator_steps.py` | MODIFY | Add `_phase1_parallel_or_sequential`, thread `use_parallel` (~25 lines) |

### Files NOT Changed

| File | Reason |
|------|--------|
| `pipeline/phases/phase1_collect.py` | Sequential path preserved as-is (fallback) |
| `pipeline/schemas.py` | No new fields needed (audit_metadata is a free-form dict) |
| `pipeline/risk_config.py` | Unrelated to parallel execution |
| `configs/risk_adjustments.yaml` | Unrelated to parallel execution |

---

## 8. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Thread-safety in yfinance | Low | yfinance uses requests sessions; executor threads are isolated |
| Shared stdout interleaving | Medium | Print statements may interleave; cosmetic only |
| Memory spike from parallel DataFrames | Low | 5 collectors x ~2MB each = ~10MB total, negligible |
| API rate limiting from concurrent requests | Low | Different APIs; Yahoo handles 30-40 tickers fine |
| Fallback path tested | -- | Sequential path is the existing code, already battle-tested |
