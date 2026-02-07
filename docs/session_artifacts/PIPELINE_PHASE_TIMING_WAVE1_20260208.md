# Pipeline Phase Timing Wave 1 (2026-02-08)

## Scope
- Target: `main.py` integrated pipeline runtime observability
- Goal: persist per-phase duration telemetry for bottleneck profiling

## Implemented
- Added timed execution wrappers in `run_integrated_pipeline(...)`:
  - `_run_timed_sync(...)`
  - `_run_timed_async(...)`
- Added result fields:
  - `pipeline_phase_timings: Dict`
  - `pipeline_elapsed_sec: float`
- Added audit metadata:
  - `pipeline_elapsed_sec`
  - `pipeline_phase_count`
  - `pipeline_timing_recorded_at`
- Added final JSON snapshot update in `main.py` so timing fields remain in final artifact.

## Reliability Guards Added for Restricted Network
- `main.py`
  - `EIMAS_YFINANCE_CACHE_DIR` (default `/tmp/eimas_yfinance_cache`)
  - yfinance sqlite cache redirect to writable path (avoids `readonly database` noise)
- `lib/fred_collector.py`
  - `EIMAS_FRED_TIMEOUT_SEC` (default `15`)
  - `EIMAS_FRED_FAIL_FAST_NETWORK` (default `false`)
  - fail-fast behavior: after first DNS/network failure, skip remaining FRED requests.
- `pipeline/phases/phase1_collect.py`
  - `EIMAS_EXTENDED_DATA_TIMEOUT_SEC` (default `45`)
  - `EIMAS_SKIP_KOREA_ASSETS` (default `false`)
- `pipeline/analyzers_quant.py`, `pipeline/analyzers_advanced.py`, `pipeline/analyzers_governance.py`, `lib/volume_analyzer.py`
  - non-DataFrame payload guard (`market_data` 순회 시 `korea_data` dict 보호)
- `pipeline/debate.py`
  - `filtered_data` 비어 있을 때 fail-fast (`NEUTRAL`)로 즉시 반환

## Sample Run (restricted environment)
Command:
```bash
ANTHROPIC_API_KEY= OPENAI_API_KEY= PERPLEXITY_API_KEY= GOOGLE_API_KEY= GEMINI_API_KEY= \
EIMAS_FRED_TIMEOUT_SEC=2 EIMAS_FRED_FAIL_FAST_NETWORK=true \
EIMAS_EXTENDED_DATA_TIMEOUT_SEC=5 EIMAS_SKIP_KOREA_ASSETS=true \
python3 main.py --full --cron-mode --output-dir outputs/profile_runs_fastfail
```

Output artifact:
- `outputs/profile_runs_fastfail/eimas_20260208_021730.json`

Key timing values:
- `pipeline_elapsed_sec`: `6.966`
- `phase1_collect_data`: `6.206`
- `phase2_institutional_frameworks`: `0.211`
- `phase2_adaptive_portfolio`: `0.203`
- `phase2_enhanced_analyze`: `0.178`

Comparison note:
- Before debate fail-fast: `eimas_20260208_021512.json` -> `phase3_debate=4.013s`
- After debate fail-fast: `eimas_20260208_021730.json` -> `phase3_debate=0.000s`
- `readonly database` log noise removed in `run_20260208_wave2.log`

## Notes
- This sample is not a production baseline because DNS/API access is restricted.
- The telemetry plumbing is now in place; on unrestricted network the same JSON field can be used for full bottleneck budgeting (`249s -> 150s -> 120s` track).

---

## Wave 2 Update (Phase 1 bottleneck drill-down)

### Added
- `pipeline/phases/phase1_collect.py`
  - `audit_metadata.phase1_component_timings`
  - `audit_metadata.phase1_elapsed_sec`
  - `EIMAS_EXTENDED_FAIL_FAST_NETWORK` (DNS precheck + immediate skip)
  - `EIMAS_SKIP_EXTENDED_DATA` (manual skip)
  - `EIMAS_EXTENDED_NETWORK_PROBE_HOSTS` (host probe list override)
- `pipeline/collectors.py`
  - `collect_market_data(..., include_crypto=True)` and phase1 경로에서 `include_crypto=False` 적용
  - 목적: BTC/ETH 중복 다운로드 제거
- `pipeline/phases/phase45_operational.py`
  - `audit_metadata` overwrite -> merge (phase1 telemetry 유실 방지)

### Measurement (restricted environment)
- Baseline (`EIMAS_EXTENDED_FAIL_FAST_NETWORK=false`):
  - artifact: `outputs/profile_runs_fastfail/eimas_20260208_023521.json`
  - `phase1_collect_data`: `7.620s`
  - `audit_metadata.phase1_component_timings.extended_data`: `5.007s` (`timeout`)
  - `pipeline_elapsed_sec`: `9.693s`
- With fail-fast (`EIMAS_EXTENDED_FAIL_FAST_NETWORK=true`):
  - artifact: `outputs/profile_runs_fastfail/eimas_20260208_023713.json`
  - `phase1_collect_data`: `0.619s`
  - `audit_metadata.phase1_component_timings.extended_data`: `0.000s` (`skipped_network`)
  - `pipeline_elapsed_sec`: `1.093s`

### Repro Command (Wave 2)
```bash
ANTHROPIC_API_KEY= OPENAI_API_KEY= PERPLEXITY_API_KEY= GOOGLE_API_KEY= GEMINI_API_KEY= \
EIMAS_FRED_TIMEOUT_SEC=2 EIMAS_FRED_FAIL_FAST_NETWORK=true \
EIMAS_EXTENDED_DATA_TIMEOUT_SEC=5 EIMAS_EXTENDED_FAIL_FAST_NETWORK=true \
EIMAS_SKIP_KOREA_ASSETS=true \
python3 main.py --full --cron-mode --output-dir outputs/profile_runs_fastfail
```

---

## Wave 3 Update (Institutional frameworks fail-fast)

### Added
- `pipeline/phases/phase2_adjustment.py`
  - `EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK` (DNS precheck + skip)
  - `EIMAS_SKIP_INSTITUTIONAL_NETWORK_ANALYSIS` (manual skip)
  - `EIMAS_SKIP_INSTITUTIONAL_FRAMEWORKS` (full institutional skip)
  - `EIMAS_INSTITUTIONAL_NETWORK_PROBE_HOSTS` (host probe list override)
  - `audit_metadata.phase2_institutional_components` (bubble/gap/fomc component timing)
- Network unavailable 시 처리 정책:
  - `bubble_framework.stage = "SKIPPED_NETWORK"`
  - `gap_analysis.skipped = true`
  - `fomc_analysis`는 로컬 데이터 기반으로 계속 실행

### Measurement (restricted environment)
- Baseline (`EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK=false`):
  - artifact: `outputs/profile_runs_fastfail/eimas_20260208_023713.json`
  - `phase2_institutional_frameworks`: `0.187s`
  - `pipeline_elapsed_sec`: `1.093s`
- With institutional fail-fast (`EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK=true`):
  - artifact: `outputs/profile_runs_fastfail/eimas_20260208_024405.json`
  - `phase2_institutional_frameworks`: `0.001s`
  - `pipeline_elapsed_sec`: `0.889s`

### Repro Command (Wave 3)
```bash
ANTHROPIC_API_KEY= OPENAI_API_KEY= PERPLEXITY_API_KEY= GOOGLE_API_KEY= GEMINI_API_KEY= \
EIMAS_FRED_TIMEOUT_SEC=2 EIMAS_FRED_FAIL_FAST_NETWORK=true \
EIMAS_EXTENDED_DATA_TIMEOUT_SEC=5 EIMAS_EXTENDED_FAIL_FAST_NETWORK=true \
EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK=true \
EIMAS_SKIP_KOREA_ASSETS=true \
python3 main.py --full --cron-mode --output-dir outputs/profile_runs_fastfail
```

---

## Wave 4 Update (Adaptive portfolio DB batching)

### Added
- `lib/adaptive_agents.py`
  - `AdaptiveAgentManager.run_all(..., persist=True)` DB 저장을 single connection + single commit으로 배치 처리
  - `_save_order/_save_decision/_save_snapshot`가 외부 cursor 주입을 지원하도록 확장
- `pipeline/analyzers_advanced.py`
  - `EIMAS_ADAPTIVE_PERSIST_DB` (default `true`)
  - `false`일 때 adaptive DB persistence skip

### Measurement (restricted environment, fail-fast flags on)
- Baseline:
  - artifact: `outputs/profile_runs_fastfail/eimas_20260208_024405.json`
  - `phase2_adaptive_portfolio`: `0.174s`
- After DB batching:
  - artifact: `outputs/profile_runs_fastfail/eimas_20260208_024947.json`
  - `phase2_adaptive_portfolio`: `0.023s`

### Repro Command (Wave 4)
```bash
ANTHROPIC_API_KEY= OPENAI_API_KEY= PERPLEXITY_API_KEY= GOOGLE_API_KEY= GEMINI_API_KEY= \
EIMAS_FRED_TIMEOUT_SEC=2 EIMAS_FRED_FAIL_FAST_NETWORK=true \
EIMAS_EXTENDED_DATA_TIMEOUT_SEC=5 EIMAS_EXTENDED_FAIL_FAST_NETWORK=true \
EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK=true \
EIMAS_SKIP_KOREA_ASSETS=true \
python3 main.py --full --cron-mode --output-dir outputs/profile_runs_fastfail
```
