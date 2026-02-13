# Module: pipeline/team_coordinator.py

## Purpose
Coordinate parallel execution of independent data collection tasks within Phase 1.
The TeamCoordinator wraps `asyncio.gather` with error isolation so that a failure in
one collector does not block the others. When parallel execution fails entirely,
it falls back to the existing sequential path automatically.

---

## Class: TeamCoordinator

```
class TeamCoordinator:
    """Thin orchestrator for parallel phase execution."""
```

### Constructor

```
def __init__(self, *, max_workers: int = 3, timeout_sec: float = 120.0) -> None:
    """
    Args:
        max_workers: Concurrency cap (unused in asyncio mode, reserved for
                     future thread-pool expansion).
        timeout_sec: Global timeout for the entire parallel batch.
    """
```

- Store `max_workers` and `timeout_sec` as instance attributes.
- No external dependencies beyond the standard library (`asyncio`, `time`, `logging`).

### Core Method: run_parallel_phase1

```
async def run_parallel_phase1(
    self,
    result: EIMASResult,
    quick_mode: bool,
) -> Dict[str, Any]:
    """
    Run Phase 1 data collectors in parallel using asyncio.gather.

    Collectors executed concurrently:
        1. FRED data           (collect_fred_data)          -- sync, run in executor
        2. Market data         (collect_market_data)        -- sync, run in executor
        3. Crypto data         (collect_crypto_data)        -- sync, run in executor
        4. Extended data       (ext_collector.collect_all)  -- already async
        5. Korea assets        (collect_korea_assets)       -- sync, run in executor

    Args:
        result: EIMASResult instance (mutated in place with fred_summary,
                extended_data, market_data_count, crypto_data_count, etc.)
        quick_mode: When True, lookback_days=90 and market_indicators skipped.

    Returns:
        Dict[str, Any] -- the merged market_data dict, identical in shape
        to what the sequential collect_data() returns.

    Raises:
        Never raises. On total failure, logs a warning and falls back
        to sequential collect_data().
    """
```

#### Internal Flow

1. Determine `lookback_days = 90 if quick_mode else 365`.
2. Build a list of async tasks by wrapping sync collectors with
   `asyncio.get_event_loop().run_in_executor(None, fn)`.
3. Call `asyncio.gather(*tasks, return_exceptions=True)`.
4. Wrap the entire gather in `asyncio.wait_for(..., timeout=self.timeout_sec)`.
5. Iterate results:
   - If a result is an `Exception`, log it as a warning and use the empty
     fallback value for that collector (`{}` for dicts, `None` for fred).
   - Otherwise, merge into `result` and `market_data`.
6. Run offline fallback injection (same logic as sequential path).
7. Run market_indicators and company_ra sequentially (they depend on
   market_data being populated).
8. Record `phase1_component_timings` in `result.audit_metadata`.
9. Return `market_data`.

#### Fallback Strategy

```
try:
    market_data = await self.run_parallel_phase1(result, quick_mode)
except Exception:
    logging.warning("Parallel phase1 failed, falling back to sequential")
    market_data = await collect_data(result, quick_mode)   # existing function
```

This fallback lives in the **caller** (orchestrator_steps.py), not inside
TeamCoordinator itself.

### Helper Method: _wrap_sync

```
@staticmethod
async def _wrap_sync(fn: Callable[..., T], *args: Any) -> T:
    """Run a synchronous function in the default executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args))
```

### Helper Method: _collect_timings

```
def _collect_timings(
    self,
    names: List[str],
    start_times: List[float],
    results: List[Any],
) -> Dict[str, Dict[str, Any]]:
    """Build phase1_component_timings dict from parallel results."""
```

---

## Integration with main.py / orchestrator_steps.py

The TeamCoordinator is **not** called from `main.py` directly. Instead,
`orchestrator_steps.py` gains a new top-level helper:

```
async def _phase1_parallel_or_sequential(
    runtime: PhaseRuntimeTracker,
    result: EIMASResult,
    quick_mode: bool,
    use_parallel: bool,
) -> Dict[str, Any]:
    """
    Choose parallel or sequential Phase 1 based on --parallel flag.
    """
    if use_parallel:
        from pipeline.team_coordinator import TeamCoordinator
        coordinator = TeamCoordinator(timeout_sec=120.0)
        try:
            return await coordinator.run_parallel_phase1(result, quick_mode)
        except Exception:
            logging.warning("Parallel phase1 failed entirely; sequential fallback")
            return await phase1_collect_data(result, quick_mode)
    return await phase1_collect_data(result, quick_mode)
```

In `run_pipeline_phases`, the existing call:
```python
market_data = await runtime.run_async("phase1_collect_data", phase1_collect_data, result, quick_mode)
```
becomes:
```python
market_data = await runtime.run_async(
    "phase1_collect_data",
    _phase1_parallel_or_sequential,
    runtime, result, quick_mode, use_parallel,
)
```

The `use_parallel` bool is threaded from `main.py --parallel` through
`run_integrated_pipeline(parallel=False)` -> `run_pipeline_phases(use_parallel=...)`.

---

## Error Handling Strategy

| Scenario | Behavior |
|----------|----------|
| Single collector raises | Log warning, use empty fallback, other collectors unaffected |
| Global timeout exceeded | `asyncio.TimeoutError` caught, fall back to sequential |
| All collectors fail | Return empty market_data (offline fallback still injects synthetics) |
| Import error (team_coordinator missing) | Caught at import, sequential path used |

---

## Environment Variable Respect

TeamCoordinator must honor the same `EIMAS_SKIP_*` and `EIMAS_*_FAIL_FAST_NETWORK`
env vars that the sequential `collect_data()` does. The parallel tasks should
read these flags **before** dispatching, and skip collectors that are disabled.

---

## Validation

```bash
# Import check (no API keys needed)
python -c "from pipeline.team_coordinator import TeamCoordinator; print('OK')"

# Integration check
python -c "
from pipeline.team_coordinator import TeamCoordinator
tc = TeamCoordinator(timeout_sec=60.0)
print(f'max_workers={tc.max_workers}, timeout={tc.timeout_sec}')
print('OK')
"
```

---

## File Dependencies

- `pipeline/schemas.py` (EIMASResult)
- `pipeline/collectors` (collect_fred_data, collect_market_data, collect_crypto_data, collect_market_indicators, collect_company_ra_analysis)
- `pipeline/korea_integration` (collect_korea_assets)
- `lib/extended_data_sources` (ExtendedDataCollector)
- `pipeline/phases/phase1_collect.py` (for fallback: collect_data)

## Estimated Lines: 150-200
