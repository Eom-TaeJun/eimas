# Phase 2 Cache Benchmark (2026-02-08)

## Scope

- Target: `pipeline/phases/phase2_enhanced.py`
- Cache backend: `pipeline/phases/phase_cache.py` (`outputs/.phase_cache/*`)
- Env:
  - `EIMAS_PHASE2_CACHE_ENABLED`
  - `EIMAS_PHASE2_CACHE_TTL=3600`

## Method

- Synthetic run with `market_data={}` to isolate phase orchestration overhead.
- Compared cache off / cache miss / cache hit.
- Measured from function entry to return (`time.perf_counter`).

## Results

### Quick mode (`quick_mode=True`)

1. `quick_cache_off`
- elapsed: `0.030s`
- `hits=0`, `misses=0`, `bypassed=1`

2. `quick_cache_on_miss`
- elapsed: `0.022s`
- `hits=0`, `misses=1`, `bypassed=0`

3. `quick_cache_on_hit`
- elapsed: `0.008s`
- `hits=1`, `misses=0`, `bypassed=0`

### Full-subset mode (`quick_mode=False`, `market_data={}`)

1. `full_cache_on_miss`
- elapsed: `0.365s`
- `hits=0`, `misses=5`, `bypassed=0`

2. `full_cache_on_hit`
- elapsed: `0.020s`
- `hits=5`, `misses=0`, `bypassed=0`

## Interpretation

- Cache hit path in Phase 2 heavy no-arg analyzers yields clear latency reduction in this environment.
- `phase2_cache_stats` provides direct evidence of hit/miss transition per key.

## Environment Notes

- Network/DNS is restricted in this workspace; external endpoints (`arkfunds.io`, `api.stlouisfed.org`) fail.
- Benchmark results are phase-local and should be treated as comparative (hit vs miss), not absolute production latency.
