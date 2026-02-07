# EIMAS Full Execution Process - 2026-02-07

## Scope
이 문서는 `python main.py --full` 기준의 실행/개선 프로세스를 정의합니다.
목표는 "실행 가능한 full 파이프라인 유지"와 "기능 분리"를 동시에 달성하는 것입니다.
상세 구조 설계는 `STRUCTURE_REDESIGN_MASTERPLAN.md`를 기준으로 합니다.

## Progress Update (2026-02-07)
- `main_integrated.py` shim removed (canonical entrypoint unified to `main.py`)
- `pipeline/runner.py` now delegates to canonical `main.run_integrated_pipeline`
- `pipeline/runner.py` legacy `run_pipeline` alias restored
- `api/main.py` and `cli/eimas.py` now import canonical `main`
- `api/server.py` removed (API entrypoint unified to `api.main`)
- `execution_intelligence` scaffold created
- `main.py` Phase 4.5 now uses adapter layer (`lib/adapters/execution_backend.py`)
- `execution_intelligence` now includes local `operational` package and runs in `EXIS_BACKEND=local` mode
- `allocation_engine`/`rebalancing_policy` moved to `execution_intelligence/models` (1차)
- `pipeline/analyzers.py` now resolves allocation/rebalancing classes via `lib/adapters/execution_models.py`
- `tactical_allocation`/`stress_test` backend switch connected (Wave 1-B)
  - `lib.adapters` exports extended for tactical/stress models
  - `phase6_portfolio.py` now imports tactical/stress via `lib.adapters`
  - `scripts/check_execution_contract.sh` verifies tactical/stress backend module sources
- `operational` path connected as package-first (Wave 1-C initial)
  - `lib/operational/*` synced with EXIS operational package baseline
  - `lib/adapters/execution_backend.py` local path now prefers `lib.operational.*`
  - local operational failure deterministically falls back to `lib.operational_engine`
  - fallback observability fields added: `backend_source`, `backend_fallback_reason`
  - `phase45_operational.py` writes backend provenance into `audit_metadata`
  - `scripts/check_execution_contract.sh` now verifies operational backend source in local/external modes
- backend switch verified:
  - `EIMAS_EXECUTION_BACKEND=local` -> `lib.*`
  - `EIMAS_EXECUTION_BACKEND=external` -> `execution_intelligence.models.*`

## 1) Canonical Runtime

### Entry
- Command: `python main.py --full`
- Entry function: `run_integrated_pipeline()` in `main.py`

### Full Mode Phase Order
1. Phase 1: Data Collection
2. Phase 2: Quant Analysis (basic + enhanced)
3. Phase 3: Debate (dual mode)
4. Phase 4: Realtime (optional)
5. Phase 4.5: Operational Report
6. Phase 5: Storage
7. Phase 6: Backtest/Attribution/Stress (optional flags)
8. Phase 7: AI Report Generation
9. Phase 8: Validation + Multi-LLM Validation
10. Phase 8.5: Quick Validation (optional)

## 2) Refactor Execution List

### Step 0. Freeze baseline
- 기준 브랜치에서 `main.py --full` 성공 로그 1회 저장
- 출력물(`outputs/eimas_*.json`) 샘플 1개를 회귀 기준으로 고정
- 이후에는 매 변경마다 full 실행하지 않고, milestone에서만 full 회귀 실행

### Step 1. Entry-path cleanup (완료/진행)
- `main_integrated.py` 제거 완료 (단일 진입점: `main.py`)
- `pipeline/runner.py`는 canonical main 파이프라인으로 위임
- `run_all_pipeline.sh`는 `python main.py --full`만 호출

### Step 2. Domain boundary tagging
각 기능을 아래 4개 도메인으로 태깅하고 owner를 지정:
- `core_full`: eimas 본체에 남는 orchestration + schema + contracts
- `onchain`: onchain_intelligence
- `execution`: execution_intelligence (신규)
- `reporting`: reporting_intelligence (신규)
- `realtime`: realtime_intelligence (신규)

### Step 3. Split-by-adapter pattern
- 직접 import 제거 대신 adapter 레이어를 먼저 둠
- adapter는 다음 규칙 준수:
  - 실패 시 deterministic fallback
  - fallback 기본값은 `HOLD` 또는 no-op
  - 입출력은 JSON serializable dict

### Step 4. Move modules wave-by-wave
Wave 1 (Execution) -> Wave 2 (Reporting) -> Wave 3 (Realtime)
- 각 wave 완료 조건:
  - `main.py --full` 회귀 성공
  - 1개 샘플 입력에 대해 핵심 필드 호환 유지
  - 실패 시 즉시 rollback 가능한 import switch 보유

## 3) Split Candidate Matrix

| Domain | Move First | Keep in EIMAS |
|---|---|---|
| onchain | (이미 분리) | onchain 결과 consume adapter |
| execution | operational/allocation/rebalancing/stress | 최종 의사결정 merge + failsafe |
| reporting | ai_report/whitening/fact_check/converters | 결과 저장 트리거 + 메타 관리 |
| realtime | stream/pipeline | full pipeline orchestration hook |

## 4) Safety Gates

### Gate A (per-change, always)
- touched file `py_compile`
- import smoke test
- domain function smoke test

### Gate B (per-wave, recommended)
- adapter 경유 external/local 스위치 확인
- fallback 동작 확인
- 회귀 필드 스냅샷 diff 점검 (domain 범위)

### Gate C (milestone/merge, required)
- `python main.py --full` 실행
- 필수 필드 무결성:
  - `risk_score`
  - `final_recommendation`
  - `full_mode_position`
  - `reference_mode_position`
  - `modes_agree`
- full runtime이 baseline 대비 +20% 이내
- critical import errors 0건

## 5) Weekly Operating Loop

1. 월요일: 분리 대상 선정 + 영향도 분석
2. 화요일: adapter 작성
3. 수요일: 모듈 이동 + import 스위치
4. 목요일: per-change/per-wave 검증 + 성능 측정
5. 금요일: milestone 후보만 full 회귀 테스트, 문서 업데이트 + 다음 wave 계획

## 6) Immediate Next Actions

1. `STRUCTURE_REDESIGN_MASTERPLAN.md` 기준으로 분리 단위를 고정
2. `execution_intelligence` Wave 1-C (`operational_engine`, `operational/`) 진행
3. `lib/operational_engine.py` 의존 지점을 adapter로 감싸고 local copy 정리 계획 수립
4. milestone 후보에서 full mode 회귀 (`python main.py --full`) 실행
