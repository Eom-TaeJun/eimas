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

## 3) 검증 결과 (2026-02-07 실행)

실행 명령:

```bash
python3 -m py_compile main.py api/main.py pipeline/runner.py lib/ai_report_generator.py
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
- [ ] `sys.path.insert` 제거 계획 수립 (활성 `.py` 기준 13건)
- [x] 절대경로 하드코딩 제거 (실행 코드 기준)
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
