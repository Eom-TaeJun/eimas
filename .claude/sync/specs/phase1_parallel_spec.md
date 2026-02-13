# Phase 1 Parallel Execution Spec

## Purpose
Parallelize the five independent data collection tasks in Phase 1 to achieve
~40% wall-clock reduction. Currently these run sequentially (~60-90s total);
FRED, Market, Crypto, Extended, and Korea collectors have no data dependencies
on each other.

---

## Current Sequential Flow (phase1_collect.py, lines 266-602)

```
collect_data(result, quick_mode) -> Dict[str, Any]:
    1. collect_fred_data()                    # sync,  ~5-15s
    2. ExtendedDataCollector.collect_all()    # async, ~10-45s (with timeout)
    3. collect_market_data(lookback_days)     # sync,  ~15-30s
    4. collect_crypto_data(lookback_days)     # sync,  ~5-10s
    5. offline_fallback injection             # sync,  <1s
    6. collect_market_indicators()            # sync,  ~5-10s (skip in quick)
    7. collect_company_ra_analysis()          # sync,  ~3-5s
    8. collect_korea_assets(lookback_days)    # sync,  ~5-15s
```

Steps 1-4 and 8 are **independent** and can run concurrently.
Steps 5-7 depend on market_data being populated, so they run **after** the
parallel batch completes.

---

## Parallel Grouping

### Group A (concurrent via asyncio.gather)

| Task | Function | Type | Typical Duration |
|------|----------|------|-----------------|
| FRED | `collect_fred_data()` | sync -> executor | 5-15s |
| Market | `collect_market_data(lookback_days, include_crypto=False)` | sync -> executor | 15-30s |
| Crypto | `collect_crypto_data(lookback_days)` | sync -> executor | 5-10s |
| Extended | `ExtendedDataCollector().collect_all()` | async native | 10-45s |
| Korea | `collect_korea_assets(lookback_days, use_parallel=True)` | sync -> executor | 5-15s |

**Expected wall-clock**: max(15-30s) instead of sum(40-115s) = **50-70% reduction for Group A**.

### Group B (sequential, after Group A)

| Task | Function | Reason |
|------|----------|--------|
| Offline fallback | `_inject_offline_fallback_market_data()` | Needs market_data populated |
| Market indicators | `collect_market_indicators()` | Needs market_data context |
| Company RA | `collect_company_ra_analysis()` | Independent but cheap (~3s) |

Group B adds ~8-15s sequentially. Total Phase 1 with parallel: ~25-45s vs ~60-90s.

---

## Function Signatures for Parallel Wrappers

### In TeamCoordinator (see team_coordinator_spec.md)

Each sync collector is wrapped via `_wrap_sync`:

```
async def _collect_fred(self) -> Dict:
    """Wrap collect_fred_data() for async execution."""
    if self._skip_flag("EIMAS_SKIP_FRED_DATA"):
        return {}
    if self._fail_fast_check("EIMAS_FRED_FAIL_FAST_NETWORK", "EIMAS_FRED_PROBE_HOSTS", "api.stlouisfed.org"):
        return {}
    return await self._wrap_sync(collect_fred_data)

async def _collect_market(self, lookback_days: int) -> Dict[str, Any]:
    """Wrap collect_market_data() for async execution."""
    if self._skip_flag("EIMAS_SKIP_MARKET_DATA"):
        return {}
    if self._fail_fast_check("EIMAS_MARKET_DATA_FAIL_FAST_NETWORK", "EIMAS_MARKET_DATA_PROBE_HOSTS", "guce.yahoo.com,query1.finance.yahoo.com"):
        return {}
    return await self._wrap_sync(collect_market_data, lookback_days, False)

async def _collect_crypto(self, lookback_days: int) -> Dict[str, Any]:
    """Wrap collect_crypto_data() for async execution."""
    if self._skip_flag("EIMAS_SKIP_CRYPTO_DATA"):
        return {}
    if self._fail_fast_check("EIMAS_CRYPTO_DATA_FAIL_FAST_NETWORK", "EIMAS_CRYPTO_DATA_PROBE_HOSTS", "guce.yahoo.com,query1.finance.yahoo.com"):
        return {}
    return await self._wrap_sync(collect_crypto_data, lookback_days)

async def _collect_extended(self, timeout_sec: float) -> Dict:
    """Run ExtendedDataCollector.collect_all() with timeout."""
    if self._skip_flag("EIMAS_SKIP_EXTENDED_DATA"):
        return {}
    # Extended already has its own fail-fast logic
    ext = ExtendedDataCollector()
    try:
        return await asyncio.wait_for(ext.collect_all(), timeout=timeout_sec)
    except (asyncio.TimeoutError, Exception) as exc:
        logging.warning(f"Extended data parallel error: {exc}")
        return {}

async def _collect_korea(self, lookback_days: int) -> Dict:
    """Wrap collect_korea_assets() for async execution."""
    if self._skip_flag("EIMAS_SKIP_KOREA_ASSETS"):
        return {"data": {}, "summary": {"skipped": True}}
    return await self._wrap_sync(collect_korea_assets, lookback_days, True)
```

### Helper: _skip_flag and _fail_fast_check

```
def _skip_flag(self, env_name: str) -> bool:
    """Check if an EIMAS_SKIP_* env var is set to truthy."""
    return os.getenv(env_name, "false").strip().lower() in {"1", "true", "yes", "on"}

def _fail_fast_check(self, flag_env: str, hosts_env: str, default_hosts: str) -> bool:
    """Check fail-fast network flag + DNS probe. Return True to skip."""
    if not self._skip_flag(flag_env):
        return False
    hosts = [h.strip() for h in os.getenv(hosts_env, default_hosts).split(",") if h.strip()]
    for host in hosts:
        try:
            socket.getaddrinfo(host, 443)
            return False  # DNS works, don't skip
        except OSError:
            continue
    return True  # all DNS failed, skip
```

---

## Backward Compatibility

### No changes to phase1_collect.py

The existing `collect_data()` function in `phase1_collect.py` is **not modified**.
It remains the sequential fallback path. The parallel path lives entirely in
`pipeline/team_coordinator.py`.

### Return value contract

`TeamCoordinator.run_parallel_phase1()` returns the **same** `Dict[str, Any]`
(market_data dict) that `collect_data()` returns, and mutates the same
`EIMASResult` fields:
- `result.fred_summary`
- `result.extended_data`
- `result.market_data_count`
- `result.crypto_data_count`
- `result.market_indicators`
- `result.company_ra_analysis`
- `result.korea_data`
- `result.korea_summary`
- `result.audit_metadata["phase1_component_timings"]`

### Import guard

```python
# In orchestrator_steps.py
if use_parallel:
    try:
        from pipeline.team_coordinator import TeamCoordinator
    except ImportError:
        logging.warning("team_coordinator not available; using sequential")
        use_parallel = False
```

---

## API Rate Limiting Considerations

| Source | Rate Limit | Mitigation |
|--------|-----------|------------|
| FRED API | 120 req/min | Single call, no issue |
| yfinance (Yahoo) | Unofficial, ~2000/hr | market + crypto = 2 calls, no issue |
| Alternative.me (Crypto F&G) | Generous | Single call |
| Korea Exchange | Moderate | Single call per asset |
| Llama.fi (DeFi) | No key, generous | Single call |

Running these in parallel does **not** increase the total number of API calls.
It only changes the timing from sequential to concurrent. Since each collector
makes a small number of requests to different APIs, there is no rate-limit risk.

**Exception**: If `collect_market_data` and `collect_crypto_data` both hit
Yahoo Finance simultaneously, the combined request rate is still within
normal bounds (~30-40 ticker fetches total). yfinance already batches
internally.

---

## Timing/Benchmarking

The parallel path records individual collector timings using the same
`phase1_component_timings` dict structure. An additional field is added:

```python
result.audit_metadata["phase1_parallel_mode"] = True
result.audit_metadata["phase1_parallel_wall_clock_sec"] = <float>
result.audit_metadata["phase1_parallel_sum_component_sec"] = <float>  # sum of all
result.audit_metadata["phase1_parallel_speedup"] = <float>  # sum / wall_clock
```

This allows direct comparison between sequential and parallel runs.

---

## Validation Commands

```bash
# 1. Import check
python -c "from pipeline.team_coordinator import TeamCoordinator; print('OK')"

# 2. Dry-run parallel (no API, tests env flag handling)
EIMAS_SKIP_MARKET_DATA=1 EIMAS_SKIP_CRYPTO_DATA=1 EIMAS_SKIP_EXTENDED_DATA=1 EIMAS_SKIP_KOREA_ASSETS=1 \
  python -c "
import asyncio
from pipeline.team_coordinator import TeamCoordinator
from pipeline.schemas import EIMASResult
tc = TeamCoordinator(timeout_sec=30.0)
result = EIMASResult(timestamp='test')
data = asyncio.run(tc.run_parallel_phase1(result, quick_mode=True))
print(f'market_data keys: {len(data)}')
print('OK')
"

# 3. Full parallel run (needs API keys)
python main.py --parallel --quick
# Compare phase1 timing with:
python main.py --quick
# Check: jq '.audit_metadata.phase1_parallel_speedup' outputs/eimas_*.json | tail -1
```
