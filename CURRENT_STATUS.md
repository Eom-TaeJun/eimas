# EIMAS 현재 상태 (2026-02-07)

## 1) Snapshot

- 기준 브랜치: `main` (`49d89e7`, origin/main 동기화)
- 기준 실행 경로: `python main.py --full`
- Canonical entrypoints:
  - Pipeline: `main.py` (`run_integrated_pipeline`)
  - API: `api/main.py`
- 워킹트리: 변경 있음(문서/정리 작업 반영 중)

## 2) 최근 완료 사항 (2026-02-06 ~ 2026-02-07)

### 구조/실행 계약
- Analyzer monolith 분해 완료:
  - `pipeline/analyzers.py` (facade)
  - `pipeline/analyzers_core.py`
  - `pipeline/analyzers_advanced.py`
  - `pipeline/analyzers_quant.py`
  - `pipeline/analyzers_sentiment.py`
  - `pipeline/analyzers_governance.py`
- Phase 로직 `main.py`에서 `pipeline/phases/*`로 이관 완료:
  - `phase1_collect` ~ `phase8_validation`
- `pipeline/runner.py`는 canonical `main.run_integrated_pipeline(...)` 위임 래퍼로 고정.

### 아티팩트 경로 정책
- ADR 추가: `docs/architecture/ADV_007_ARTIFACT_PATH_POLICY_V1.md`
- 단일 실행 JSON 파일 갱신 정책 적용:
  - `pipeline/storage.py`의 `save_result_json(..., output_file=...)`
  - Phase 5 output path를 Phase 7/8로 전달하여 같은 JSON을 갱신

### 안정성 보강
- 컬럼 편차 내성 강화:
  - `pipeline/phases/phase6_portfolio.py`
  - `pipeline/phases/phase2_enhanced.py`
  - `Close` / `close` / `Adj Close` fallback 처리

### 레거시 제거
- 제거 완료:
  - `main_integrated.py`
  - `api/server.py`
  - `lib/deprecated/`
  - `archive/` (전체 제거)

### 문서 정합성 업데이트 (2026-02-07)
- `docs/architecture/EIMAS_OVERVIEW.md`의 엔트리포인트 표기를 `main_orchestrator.py` -> `main.py`로 정정
- `docs/architecture/EIMAS_TECHNICAL_DOCUMENTATION.md`의 오케스트레이터 표기를 `main.py` 기준으로 정정
- `pipeline/runner.py` docstring에서 archive runner 경로명 직접 참조 제거
- 실행 스크립트 inventory 문서 추가: `docs/manuals/RUN_SCRIPT_INVENTORY_20260207.md`
- 구형/위험 스크립트 제거: `scripts/merge_frontend.sh` (입력 원본 부재 + `rm -rf frontend` 위험)
- `README.md` 문서 가이드에 full-mode refactor 링크 추가 (`FULL_EXECUTION_PROCESS`, `CURRENT_STATUS`, `TODO`)
- artifact 보존 정책/정리 문서 추가: `docs/manuals/ARTIFACT_RETENTION_POLICY_20260207.md`

### Execution Intelligence Wave 1-B 반영 (2026-02-07)
- `lib/adapters/execution_models.py`가 아래 모델까지 backend 스위치 대상으로 확장:
  - `TacticalAssetAllocator`, `VolatilityTargeting`, `MomentumOverlay`
  - `StressTestEngine`, `generate_stress_test_report`
- `lib/adapters/__init__.py` export 확장으로 파이프라인 import 경로 통일
- 실행 경로 import 전환:
  - `pipeline/phases/phase6_portfolio.py`
  - `tests/test_portfolio_modules.py`
- `scripts/check_execution_contract.sh`의 backend source check를 tactical/stress 모델까지 확장

### Execution Intelligence Wave 1-C 초기 반영 (2026-02-07)
- `lib/operational/*` 핵심 파일을 EXIS 기준으로 동기화:
  - `config.py`, `enums.py`, `constraints.py`, `rebalance.py`, `engine.py`
- `lib/adapters/execution_backend.py` 로컬 경로를 package-first로 전환:
  - 기본: `lib.operational.*`
  - 실패 시: `lib.operational_engine` fallback (runtime continuity 유지)
  - 디버깅용 선택자 추가: `EIMAS_LOCAL_OPERATIONAL_BACKEND=package|monolith|package_strict`
- operational bundle provenance/trace 강화:
  - `backend_source` (`local_package` / `local_monolith` / `external`)
  - `backend_fallback_reason` (fallback 발생 시 원인 클래스)
- `phase45_operational.py`가 실행 provenance를 `audit_metadata`에 기록:
  - `execution_backend_source`
  - `execution_backend_fallback_reason`
- `scripts/check_execution_contract.sh` backend source check에 operational 경로 검증 추가:
  - Local expected: `lib.operational.*` (fallback 허용: `lib.operational_engine*`)
  - External expected: `execution_intelligence.operational.*` (fallback 허용)
- 계약 컴파일 게이트 확장:
  - `lib/operational/config.py`
  - `lib/operational/enums.py`
  - `lib/operational/constraints.py`
  - `lib/operational/rebalance.py`
  - `lib/operational/engine.py`

### Path bootstrap 정리 Wave (2026-02-07)
- `scripts/_project_bootstrap.py` 추가로 direct script 실행용 `sys.path` 주입 로직 공통화
- 아래 8개 스크립트에서 중복 `sys.path.insert(...)` 제거:
  - `daily_analysis.py`, `daily_collector.py`, `generate_final_report.py`, `prepare_historical_data.py`
  - `scheduler.py`, `run_backtest.py`, `validate_integration_design.py`, `validate_methodology.py`
- `lib/path_bootstrap.py` 추가로 동적 외부경로 주입 로직 공통화:
  - `pipeline/collectors.py`
  - `lib/adapters/execution_backend.py`
  - `lib/adapters/execution_models.py`
- `api/main.py`, `cli/eimas.py`를 module-first + direct-script fallback 가드로 전환
  - 모듈 실행(`python -m`, `uvicorn api.main:app`)에서는 경로 주입 스킵
  - 직접 파일 실행 시에만 경로 주입 유지
- 활성 `*.py` 기준 `sys.path.insert(...)` 사용 건수:
  - `13` -> `4`로 축소
- 잔여 4건은 entrypoint/script 호환 경로(`api/main.py`, `cli/eimas.py`, bootstrap helpers)로 분류

### Phase 2 성능 최적화 Wave (2026-02-07)
- `pipeline/phases/phase_cache.py` 추가:
  - `outputs/.phase_cache/<namespace>/*` 기반 파일 TTL 캐시 유틸
- `pipeline/phases/phase2_enhanced.py`에 `1h` 기본 TTL 캐시 적용:
  - `ark_analysis` (`analyze_ark_trades`)
  - `liquidity_signal` (`analyze_liquidity`)
  - `etf_flow_result` (`analyze_etf_flow().to_dict()`)
  - `genius_act` (`analyze_genius_act().to_dict()`)
  - `theme_etf_analysis` (`analyze_theme_etf().to_dict()`)
- `result.phase2_cache_stats` 텔레메트리 추가:
  - 실행별 `hits` / `misses` / `bypassed`
  - key별 cache 상태 집계 (`ark_analysis`, `liquidity_signal`, `etf_flow_result`, `genius_act`, `theme_etf_analysis`)
- 설정값:
  - `EIMAS_PHASE2_CACHE_ENABLED` (default: `true`)
  - `EIMAS_PHASE2_CACHE_TTL` (default: `3600`)
- 제한 환경 스모크(네트워크 제한) 기준:
  - 동일 조건 2회 실행 시 캐시 miss->hit 전환 확인
  - 샘플: `0.67s -> 0.03s` (Phase2 Enhanced subset)

### Strategic Allocation 안정화 (2026-02-07)
- `pipeline/korea_integration.py`의 valuation gap 추출 로직 공통화:
  - `_extract_valuation_gap(...)` 추가
- `calculate_strategic_allocation(...)`에서 `us_fair_gap`/`korea_fair_gap`을 선초기화하여
  evidence-based 경로에서도 `global_allocation`/`tactical_signals` 계산 시 미정의 변수 예외 제거

### AI Validation fan-out 최적화 (2026-02-07)
- `lib/validation_agents.py`:
  - `ValidationAgentManager.validate_all(...)`를 thread fan-out 기반으로 전환
  - blocking SDK 호출을 에이전트 단위 병렬 실행
  - `EIMAS_VALIDATION_AGENT_TIMEOUT_SEC`(default `90`) 기반 timeout fallback (`NEEDS_INFO`)
  - 재시도/백오프 기본 정책 추가:
    - `EIMAS_VALIDATION_RETRY_COUNT` (default `1`)
    - `EIMAS_VALIDATION_RETRY_BACKOFF_SEC` (default `1.0`)
- fake-agent 스모크 기준 병렬 실행 확인:
  - 2 agents x 250ms -> 총 약 `0.26s` 완료
  - timeout 스모크: timeout `1s`에서 `NEEDS_INFO` fallback 확인

### AI Validation selective retry 3차 (2026-02-08)
- `lib/validation_agents.py`:
  - 에이전트별 retry 정책 객체(`AgentRetryPolicy`) 도입
  - 실패 유형 분류기 추가 (`timeout`, `rate_limit`, `transient_network`, `server_overload`, `auth`, `bad_request`, `unknown`)
  - `validate_all(...)` fan-out 경로에서 `retry_on` 대상 실패만 선택 재시도
  - Perplexity 기본 정책을 별도 차등화(기본 대비 `+1` retry, backoff `x1.5`)
  - 선택 정책 오버라이드 환경변수 추가: `EIMAS_VALIDATION_RETRY_POLICY_OVERRIDES` (JSON)
  - 런타임 텔레메트리 추가: `consensus.validation_runtime_stats`
    - `total_retries`, `failure_type_counts`, `per_agent.{attempts,retries,last_failure_type}`
- `pipeline/analyzers_governance.py`:
  - `run_ai_validation(...)` 결과에 `validation_runtime_stats` 포함
  - 실행 로그에 `Retry Count` 출력
- fake-agent 스모크 기준 selective retry 확인:
  - `429 rate limit`은 재시도 후 복구 (`Perplexity calls: 3`)
  - `API key not configured`는 비재시도 즉시 종료 (`GPT calls: 1`)
- trace 스모크:
  - `failure_type_counts`: `{'auth': 1, 'rate_limit': 1}`
  - `retried_agents`: `['Perplexity']`
- 세션 아티팩트: `docs/session_artifacts/AI_VALIDATION_RETRY_WAVE3_20260208.md`

### Pipeline phase timing telemetry (2026-02-08)
- `main.py`:
  - phase 실행 공통 타이머 래퍼 추가 (`_run_timed_sync`, `_run_timed_async`)
  - 실행별 phase 타이밍 저장:
    - `result.pipeline_phase_timings`
    - `result.pipeline_elapsed_sec`
  - 타이밍 메타데이터를 `result.audit_metadata`에 기록
  - 최종 JSON 스냅샷 갱신으로 타이밍 필드 영속화
- `pipeline/schemas.py`:
  - `EIMASResult`에 `pipeline_phase_timings`, `pipeline_elapsed_sec` 필드 추가
- 제한 환경 실행 안정화:
  - `lib/fred_collector.py`: `EIMAS_FRED_TIMEOUT_SEC`, `EIMAS_FRED_FAIL_FAST_NETWORK` 추가
  - `main.py`: `EIMAS_YFINANCE_CACHE_DIR` 기반 yfinance cache dir 리다이렉트 (`/tmp/eimas_yfinance_cache` 기본)
  - `pipeline/phases/phase1_collect.py`:
    - `EIMAS_EXTENDED_DATA_TIMEOUT_SEC` 기반 extended_data timeout 추가
    - `EIMAS_SKIP_KOREA_ASSETS` 기반 Korea 수집 스킵 옵션 추가
  - `pipeline/analyzers_quant.py`, `pipeline/analyzers_advanced.py`, `pipeline/analyzers_governance.py`:
    - `market_data` 순회 시 non-DataFrame payload(`korea_data` 등) 타입 가드 추가
  - `lib/volume_analyzer.py`:
    - `_validate_data`/`_calculate_market_volume_percentile`에 non-DataFrame 가드 추가
- 샘플 프로파일 실행 결과:
  - 아티팩트: `outputs/profile_runs_fastfail/eimas_20260208_021730.json`
  - `pipeline_elapsed_sec`: `6.966s`
  - Top phases:
    - `phase1_collect_data`: `6.206s`
    - `phase2_institutional_frameworks`: `0.211s`
    - `phase2_adaptive_portfolio`: `0.203s`
    - `phase2_enhanced_analyze`: `0.178s`
  - 개선 확인:
    - `readonly database` 에러 미발생 (`run_20260208_wave2.log`)
    - debate fail-fast 적용 전/후: `phase3_debate 4.013s -> 0.000s` (`eimas_20260208_021512.json` vs `eimas_20260208_021730.json`)

### Phase 1 runtime stabilization Wave 2 (2026-02-08)
- `pipeline/collectors.py`
  - `collect_market_data(..., include_crypto=True)` 시그니처 확장
  - `phase1_collect` 경로에서 `include_crypto=False` 적용하여 BTC/ETH 중복 다운로드 제거
- `pipeline/phases/phase1_collect.py`
  - Phase 1 컴포넌트 타이밍 추가:
    - `audit_metadata.phase1_component_timings`
    - `audit_metadata.phase1_elapsed_sec`
  - 네트워크 제한 환경용 확장데이터 fail-fast 제어:
    - `EIMAS_EXTENDED_FAIL_FAST_NETWORK` (DNS precheck 기반)
    - `EIMAS_SKIP_EXTENDED_DATA` (강제 스킵)
    - `EIMAS_EXTENDED_NETWORK_PROBE_HOSTS` (comma-separated host list)
- `pipeline/phases/phase45_operational.py`
  - `audit_metadata` overwrite -> merge로 수정 (phase1 telemetry 유실 방지)
- 프로파일 비교:
  - before (`eimas_20260208_023521.json`, fail-fast off):
    - `phase1_collect_data`: `7.620s`
    - `extended_data`: `5.007s` (`timeout`)
    - `pipeline_elapsed_sec`: `9.693s`
  - after (`eimas_20260208_023713.json`, `EIMAS_EXTENDED_FAIL_FAST_NETWORK=true`):
    - `phase1_collect_data`: `0.619s`
    - `extended_data`: `0.000s` (`skipped_network`)
    - `pipeline_elapsed_sec`: `1.093s`

### Phase 2 institutional fail-fast Wave 3 (2026-02-08)
- `pipeline/phases/phase2_adjustment.py`
  - institutional 네트워크 가드/스킵 옵션 추가:
    - `EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK`
    - `EIMAS_SKIP_INSTITUTIONAL_NETWORK_ANALYSIS`
    - `EIMAS_SKIP_INSTITUTIONAL_FRAMEWORKS`
    - `EIMAS_INSTITUTIONAL_NETWORK_PROBE_HOSTS`
  - 네트워크 불가 시 Bubble/Gap 분석을 즉시 `SKIPPED_NETWORK`로 처리하고 FOMC(로컬 데이터)는 유지
  - 컴포넌트 타이밍 추가:
    - `audit_metadata.phase2_institutional_components`
- 프로파일 비교:
  - before (`eimas_20260208_023713.json`, institutional fail-fast off):
    - `phase2_institutional_frameworks`: `0.187s`
    - `pipeline_elapsed_sec`: `1.093s`
  - after (`eimas_20260208_024405.json`, `EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK=true`):
    - `phase2_institutional_frameworks`: `0.001s`
    - `pipeline_elapsed_sec`: `0.889s`
  - `bubble_framework.stage`: `SKIPPED_NETWORK`
  - `gap_analysis.skipped`: `true`

### Phase 2 adaptive portfolio optimization Wave 4 (2026-02-08)
- `lib/adaptive_agents.py`
  - `AdaptiveAgentManager.run_all(...)` DB 저장 경로를 단일 sqlite 트랜잭션으로 배치 처리
  - 기존 agent별/레코드별 connection open/commit/close 반복 제거
- `pipeline/analyzers_advanced.py`
  - `EIMAS_ADAPTIVE_PERSIST_DB` 옵션 추가 (`true` 기본)
  - `false`일 때 adaptive DB persistence 비활성화 (분석 결과만 유지)
- 프로파일 비교:
  - before (`eimas_20260208_024405.json`):
    - `phase2_adaptive_portfolio`: `0.174s`
    - `pipeline_elapsed_sec`: `0.889s`
  - after (`eimas_20260208_024947.json`):
    - `phase2_adaptive_portfolio`: `0.023s`
    - `pipeline_elapsed_sec`: `1.070s` (phase1 변동성 영향으로 총합은 샘플 편차 존재)

### Phase 2 캐시 벤치마크 (2026-02-08)
- 문서: `docs/session_artifacts/PHASE2_CACHE_BENCH_20260208.md`
- quick mode:
  - off `0.030s` / on-miss `0.022s` / on-hit `0.008s`
- full-subset (`market_data={}`):
  - on-miss `0.365s` / on-hit `0.020s`
- 제한 환경(DNS 차단) 특성상 절대값보다 miss->hit 상대 개선 지표로 해석

## 3) 검증 결과 (2026-02-07 ~ 2026-02-08 실행)

실행 명령:

```bash
python3 -m py_compile main.py api/main.py pipeline/runner.py lib/ai_report_generator.py
python3 -m py_compile lib/validation_agents.py
bash scripts/check_execution_contract.sh
```

결과:
- `py_compile`: PASS
- `scripts/check_execution_contract.sh`: PASS (3/3, operational source check 포함)
  - Compile check: PASS
  - Backend source check (local/external): PASS
  - Allocation/Rebalancing smoke test: PASS

참고:
- 스모크 테스트 중 OMP warning 1건 출력되었으나 계약 검증 상태는 PASS.
- `python main.py --quick` 실실행은 현재 환경의 DNS 제한으로 중단됨:
  - `api.stlouisfed.org` name resolution 실패
  - yfinance 일부 ticker download 실패

## 4) 남은 우선순위 작업

### A. 경로/정리
- [x] archive runner 참조 정리 (`pipeline.collection.runner` 계열)
- [x] 사용하지 않는 실행 스크립트 목록화 및 1차 정리 (`RUN_SCRIPT_INVENTORY_20260207`, `merge_frontend.sh` 제거)

### B. 도메인 분리
- [x] `execution_intelligence` 2차 이동 (adapter 연결 기준)
  - `lib/tactical_allocation.py` -> `execution_intelligence.models.tactical_allocation`
  - `lib/stress_test.py` -> `execution_intelligence.models.stress_test`
  - `phase6_portfolio.py` / `tests/test_portfolio_modules.py` import 전환 완료
- [ ] 운영결정 모듈 이동 마무리
  - `lib/operational_engine.py`
  - `lib/operational/` 정리
  - 현재 상태: package-first 연결 완료, monolith fallback 유지
- [ ] `reporting_intelligence` 폴더 신설 및 대상 확정
- [ ] `realtime_intelligence` 폴더 신설 및 대상 확정

### C. 신뢰성/성능
- [ ] `pytest` 실행 환경 정비
- [x] `sys.path.insert` 1차/2차 축소 + 제거 계획 수립 (활성 `.py` 기준 `13 -> 4`, `docs/session_artifacts/SYSPATH_INSERT_REDUCTION_PLAN_20260207.md`)
- [ ] 잔여 `sys.path.insert` 4건 정리 (`api/main.py`, `cli/eimas.py`, `scripts/_project_bootstrap.py`, `lib/path_bootstrap.py`)
- [x] 절대경로 하드코딩 제거 (실행 코드 기준)
- [x] Phase 2 분석 병목 캐시 1차 적용 (`pipeline/phases/phase2_enhanced.py`, `pipeline/phases/phase_cache.py`)
- [x] AI 검증 병렬화 3차 selective retry 적용 (`lib/validation_agents.py`)
- [ ] FULL 실행 성능 개선 (`249s -> 150s -> 120s` 목표)

### D. 문서/운영
- [x] `README.md`에 full-mode refactor 문서 링크 보강
- [x] 구버전 archive 제거 및 문서 링크 정리

## 5) 다음 세션 재시작 체크리스트

```bash
cd /home/tj/projects/autoai/eimas
git status --short
python3 -m py_compile main.py api/main.py pipeline/runner.py lib/ai_report_generator.py
bash scripts/check_execution_contract.sh
```

다음 착수 권장:
1. `사용하지 않는 실행 스크립트` inventory 작성 및 제거 후보 확정
2. `execution_intelligence` 운영결정 이동 마무리 (`operational_engine`, `operational/`)
3. `sys.path`/절대경로 하드코딩 제거 wave 지속
