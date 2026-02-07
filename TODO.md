# EIMAS TODO (Full Mode Refactor) - 2026-02-06

## Goal
- 기준 실행 경로를 `python main.py --full`로 고정
- 과도하게 결합된 기능을 도메인별로 분리
- `eimas`는 "full orchestration core"로 축소

## Refactor Rules
- 단일 진입점: `main.py` (`run_integrated_pipeline`)
- 호환성 유지: 기존 import 경로는 shim으로 유지 후 단계적 제거
- 분리 우선순위: "독립 실행 가능 + 외부 API 의존 + 파일 크기 큰 영역" 먼저
- 검증 정책: 작은 변경은 `py_compile + import/function smoke`, full은 milestone/merge에서만 실행
- 구조 기준 문서: `STRUCTURE_REDESIGN_MASTERPLAN.md`
- 비대화 해소 기준 문서: `BLOAT_RESOLUTION_ARCHITECTURE.md`

---

## Track A - Full Mode 안정화 (이번 세션)

### A1. 실행 경로 정리
- [x] `main_integrated.py` 제거 (단일 진입점 `main.py`로 통합)
- [x] `pipeline/runner.py`를 canonical 파이프라인 위임 방식으로 교체
- [x] `pipeline/runner.py`에 legacy `run_pipeline` alias 복구
- [x] `run_all_pipeline.sh`를 `python main.py --full` 기반으로 교체
- [x] `api/main.py`, `cli/eimas.py`의 legacy 주석/의존 제거
- [x] 단일 run JSON artifact 경로 정책 도입 (`ADV_007`, phase7/8 동일 파일 갱신)
- [x] `api/main.py`를 canonical API 엔트리로 고정, `api/server.py` 제거

### A2. 깨진/중복 경로 청소
- [x] `main_integrated` 직접 참조 제거 (활성 코드 경로 기준)
- [x] 구버전 archive/docs 및 docs/archive 제거
- [x] 활성 경로 backup 파일(`*_backup_*`, `.backup_*`) 제거
- [x] `lib/deprecated/` 제거 (레거시 import 의존 정리)
- [x] `archive/` 전체 제거 (구형 코드/문서 정리)
- [x] `pipeline.collection.runner` 등 archive 잔재 참조 제거
- [x] 사용하지 않는 실행 스크립트 목록화 및 1차 정리 (`RUN_SCRIPT_INVENTORY_20260207`, `merge_frontend.sh` 제거)

---

## Track B - 기능 분할 (온체인 방식으로 외부 폴더 분리)

### B0. 완료/진행 상태
- [x] `onchain_intelligence` 1차 분리 완료
- [ ] `eimas`와 `onchain_intelligence` 인터페이스 계약(JSON schema) 명시

### B1. 분리 후보 1: Execution Intelligence (우선)
대상: 운영결정/리밸런싱/제약복구/전술배분/스트레스테스트
- [x] 새 폴더 생성: `/home/tj/projects/autoai/execution_intelligence`
- [x] 1차 이동 완료:
  - `lib/allocation_engine.py` -> `execution_intelligence/models/allocation_engine.py`
  - `lib/rebalancing_policy.py` -> `execution_intelligence/models/rebalancing_policy.py`
  - `pipeline/analyzers.py`가 adapter 경유 import로 전환
- [x] 2차 이동 완료 (adapter 경유 전환):
  - `lib/tactical_allocation.py` -> `execution_intelligence.models.tactical_allocation`
  - `lib/stress_test.py` -> `execution_intelligence.models.stress_test`
  - `phase6_portfolio.py`, `tests/test_portfolio_modules.py` import를 `lib.adapters`로 전환
- [ ] 운영결정 이동 정리:
  - `lib/operational_engine.py`
  - `lib/operational/` (EXIS에 copy 완료, eimas 원본은 아직 유지)
  - 진행: `lib/operational/*`을 EXIS 기준으로 동기화, `execution_backend`는 package-first + monolith fallback으로 전환
  - 진행: `phase45_operational.py`가 `audit_metadata`에 `execution_backend_source`/`execution_backend_fallback_reason` 기록
  - 다음: fallback 관측(`backend_source`, `backend_fallback_reason`) 기준으로 monolith 제거 시점 결정
- [x] EIMAS adapter 작성: 실패 시 HOLD fallback 보장

### B2. 분리 후보 2: Reporting Intelligence
대상: AI 리포트/화이트닝/팩트체크/문서 변환
- [ ] 새 폴더 생성: `/home/tj/projects/autoai/reporting_intelligence`
- [ ] 이동 대상 확정:
  - `pipeline/report.py`
  - `lib/ai_report_generator.py`
  - `lib/whitening_engine.py`
  - `lib/autonomous_agent.py`
  - `lib/json_to_md_converter.py`
  - `lib/json_to_html_converter.py`

### B3. 분리 후보 3: Realtime Intelligence
대상: 바이낸스 스트림/실시간 파이프라인/알림
- [ ] 새 폴더 생성: `/home/tj/projects/autoai/realtime_intelligence`
- [ ] 이동 대상 확정:
  - `pipeline/realtime.py`
  - `lib/binance_stream.py`
  - `lib/realtime_pipeline.py`

---

## Track C - Full Mode 성능/신뢰성

### C1. 성능 예산
- [ ] FULL 총 실행 시간: `249s -> 150s -> 120s`
- [x] Phase 2 분석 병목 1차 캐시 도입 (`1h TTL`, 무인자 고비용 분석 결과 파일 캐시)
- [x] Phase 2 캐시 hit/miss 텔레메트리 (`result.phase2_cache_stats`) 추가
- [x] AI 검증 병렬화 1차: `ValidationAgentManager.validate_all` thread fan-out 적용 (`EIMAS_VALIDATION_AGENT_TIMEOUT_SEC`)
- [x] AI 검증 병렬화 2차: timeout/재시도/백오프 기본 정책 적용 (`EIMAS_VALIDATION_RETRY_COUNT`, `EIMAS_VALIDATION_RETRY_BACKOFF_SEC`)
- [x] AI 검증 병렬화 3차: agent별 정책 차등화 + 실패 유형별 selective retry (`EIMAS_VALIDATION_RETRY_POLICY_OVERRIDES`)
- [x] AI 검증 관측성 보강: `validation_runtime_stats`(agent별 attempts/retries/failure_type) 추가
- [x] 파이프라인 phase 타이밍 텔레메트리 추가 (`result.pipeline_phase_timings`, `result.pipeline_elapsed_sec`)
- [x] Phase 1 컴포넌트 타이밍 텔레메트리 추가 (`audit_metadata.phase1_component_timings`, `audit_metadata.phase1_elapsed_sec`)
- [x] Phase 1 market/crypto 중복 다운로드 제거 (`collect_market_data(..., include_crypto=False)` 적용)
- [x] Extended data 네트워크 fail-fast 스킵 옵션 추가 (`EIMAS_EXTENDED_FAIL_FAST_NETWORK`, `EIMAS_SKIP_EXTENDED_DATA`)
- [x] Institutional frameworks 네트워크 fail-fast/스킵 옵션 추가 (`EIMAS_INSTITUTIONAL_FAIL_FAST_NETWORK`, `EIMAS_SKIP_INSTITUTIONAL_NETWORK_ANALYSIS`)
- [x] Institutional frameworks 컴포넌트 타이밍 텔레메트리 추가 (`audit_metadata.phase2_institutional_components`)
- [x] Adaptive portfolio DB I/O 배치 최적화 (`AdaptiveAgentManager.run_all` 단일 트랜잭션)

### C2. 신뢰성
- [x] `sys.path.insert` 1차 축소 (`13 -> 6`, `scripts/_project_bootstrap.py`로 스크립트 경로부트스트랩 통합)
- [x] `sys.path.insert` 2차 축소 (`6 -> 4`, `lib/path_bootstrap.py`로 동적 외부경로 주입 통합)
- [x] `api/main.py`, `cli/eimas.py` module-first + direct-script fallback 가드 적용
- [x] `sys.path.insert` 제거 계획 문서화 (`docs/session_artifacts/SYSPATH_INSERT_REDUCTION_PLAN_20260207.md`)
- [ ] 잔여 `sys.path.insert` 4건 정리 (`api/main.py`, `cli/eimas.py`, `scripts/_project_bootstrap.py`, `lib/path_bootstrap.py`)
- [x] 절대경로 제거 (`/home/tj/projects/autoai/eimas` 하드코딩, 실행 코드 기준)
- [x] `Close`/`close` 컬럼 편차에 대한 backtest/전략배분 로직 내성 강화
- [x] `calculate_strategic_allocation`의 `us_fair_gap`/`korea_fair_gap` 미초기화 예외 수정
- [ ] `pytest` 실행 가능한 테스트 환경 정비

---

## Track D - 문서/운영 프로세스 재설계

- [x] `FULL_EXECUTION_PROCESS.md` 신설
- [x] `README.md`에 full-mode refactor 문서 링크 추가 (`FULL_EXECUTION_PROCESS`, `CURRENT_STATUS`, `TODO`)
- [x] `CURRENT_STATUS.md`를 refactor 진행 기준으로 업데이트
- [x] 구버전 archive 제거 + 문서 참조 정리

---

## 이번 주 실행 순서

1. `A` 완료: 실행 경로/깨진 import 정리 완료
2. `B1` 시작: Execution Intelligence 폴더 생성 + 모듈 이동 리스트 확정
3. `C` 착수: per-change/per-wave 검증 자동화
4. `B2`/`B3`로 확장: 보고서/실시간 분리

---

## 내일 바로 시작 (Restart)

1. 상태 확인:
   - `git status --short`
   - `python3 -m py_compile main.py api/main.py pipeline/runner.py lib/ai_report_generator.py`
2. 계약 확인:
   - `bash scripts/check_execution_contract.sh`
3. 다음 클리닝:
   - `docs/architecture/*`의 legacy 명령/엔트리 참조 정리
   - `sys.path.insert`/절대경로 하드코딩 2차 축소
4. 구조 리팩토링 재개:
   - `execution_intelligence` 운영결정 이동 정리 (`operational_engine`, `operational/`)

---

## Canonical Commands

```bash
# Full mode (기준 실행)
python main.py --full

# Full + realtime
python main.py --full --realtime -d 30

# Shell wrapper
./run_all_pipeline.sh
```
