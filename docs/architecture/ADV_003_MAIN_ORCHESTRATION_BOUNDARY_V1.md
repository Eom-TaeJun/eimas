# ADV-003: Main Orchestration Boundary v1 (2026-02-06)

## 1. Objective
- `main.py`를 orchestration + CLI에만 집중시키고, phase 구현은 분리한다.
- `run_integrated_pipeline()` 시그니처/호출 계약은 유지한다.
- 분리 중에도 runtime 안정성을 유지한다.

## 2. Current State (Measured)
- `main.py`: `1,345 lines`
- helper 함수: `24` (phase helper + util)
- 대형 helper:
  - `_run_backtest` (`162`)
  - `_generate_operational_report` (`125`)
  - `_run_quick_validation` (`96`)
  - `_apply_extended_data_adjustment` (`90`)
- `run_integrated_pipeline`: `108 lines`

## 3. Boundary Decision (Advanced)

### AD-MAIN-01: Orchestration-Only Main
- `main.py`는 아래만 유지:
  - CLI arg parsing (`main()`)
  - pipeline entry (`run_integrated_pipeline()`)
  - phase 호출 순서/예외 경계
- 개별 phase 구현은 `pipeline/phases/*`로 이동

### AD-MAIN-02: Phase Module Ownership
- phase module은 side-effect와 result mutation 책임을 갖는다.
- `run_integrated_pipeline()`은 상태 전달(result, market_data, flags)만 담당.

### AD-MAIN-03: Compatibility First
- 외부 공개 함수명 유지:
  - `run_integrated_pipeline`
  - `main`
- 기존 shim/runner 경로와 충돌 없게 유지

## 4. Target Module Layout
- `pipeline/phases/phase1_collect.py`
- `pipeline/phases/phase2_basic.py`
- `pipeline/phases/phase2_enhanced.py`
- `pipeline/phases/phase2_adjustment.py`
- `pipeline/phases/phase3_debate.py`
- `pipeline/phases/phase4_realtime.py`
- `pipeline/phases/phase45_operational.py`
- `pipeline/phases/phase5_storage.py`
- `pipeline/phases/phase6_portfolio.py`
- `pipeline/phases/phase7_report.py`
- `pipeline/phases/phase8_validation.py`
- `pipeline/phases/common.py` (`safe_call` 등 공통 헬퍼)

## 5. Function Ownership Matrix

### 5.1 Data/Analysis
- `_collect_data` -> `phase1_collect.py`
- `_analyze_basic` -> `phase2_basic.py`
- `_analyze_enhanced` -> `phase2_enhanced.py`
- `_set_allocation_result` -> `phase2_enhanced.py`
- `_set_liquidity` -> `phase2_enhanced.py`
- `_set_genius_act` -> `phase2_enhanced.py`
- `_calculate_strategic_allocation` -> `phase2_enhanced.py`
- `_analyze_sentiment_bubble` -> `phase2_adjustment.py`
- `_apply_extended_data_adjustment` -> `phase2_adjustment.py`
- `_analyze_institutional_frameworks` -> `phase2_adjustment.py`
- `_run_adaptive_portfolio` -> `phase2_adjustment.py`

### 5.2 Debate/Realtime/Operational
- `_run_debate` -> `phase3_debate.py`
- `_run_realtime` -> `phase4_realtime.py`
- `_generate_operational_report` -> `phase45_operational.py`

### 5.3 Storage/Report/Validation
- `_save_results` -> `phase5_storage.py`
- `_generate_report` -> `phase7_report.py`
- `_validate_report` -> `phase8_validation.py`
- `_run_ai_validation_phase` -> `phase8_validation.py`
- `_run_quick_validation` -> `phase8_validation.py`

### 5.4 Portfolio Theory Optional
- `_run_backtest` -> `phase6_portfolio.py`
- `_run_performance_attribution` -> `phase6_portfolio.py`
- `_run_tactical_allocation` -> `phase6_portfolio.py`
- `_run_stress_test` -> `phase6_portfolio.py`

### 5.5 Shared
- `_safe_call` -> `pipeline/phases/common.py`

## 6. Migration Sequence

### Stage M1
- `pipeline/phases/` 생성
- 함수 이동 없이 wrapper skeleton 배치

### Stage M2
- helper를 phase 모듈로 순차 이동
- `main.py`는 phase 함수 import 후 호출만 유지

### Stage M3
- 중복 import 정리
- `main.py` 크기 목표 달성

## 7. Validation Contract

### Per-Change
- `py_compile`:
  - `main.py`
  - `pipeline/phases/*.py`
- import smoke:
  - `import main`
  - `from pipeline.phases.phase2_enhanced import ...`
- function smoke:
  - phase 함수 단위 입력/출력 핵심 필드 확인

### Milestone
- `python main.py --full` 1회
- 필수 필드 무결성:
  - `risk_score`
  - `final_recommendation`
  - `full_mode_position`
  - `reference_mode_position`
  - `modes_agree`

## 8. Success Criteria
- `main.py` < `700 lines` (v1 target)
- phase 로직이 `pipeline/phases/*`로 이동
- entrypoint 호환성 유지

## 9. Work Orders (Draft IDs)
- `GEN-301`: `pipeline/phases/*` skeleton 생성 + `common.py` 생성
- `GEN-302`: Phase1~2 helper 이동 + main import wiring
- `GEN-303`: Phase3~8 helper 이동 + main slim-down + validation 실행
