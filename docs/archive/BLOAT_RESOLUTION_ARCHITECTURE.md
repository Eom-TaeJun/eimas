# EIMAS Bloat Resolution Architecture (2026-02-06)

## 1) Current Size Snapshot

### Codebase Metrics
- Python files (total): `397`
- Python files (active, excluding `archive/`, `deprecated/`, backup, tests): `276`

### Active Code Concentration
- `lib`: `197 files / 87,467 lines`
- `agents`: `10 files / 6,125 lines`
- `pipeline`: `13 files / 5,496 lines`
- `core`: `10 files / 4,668 lines`
- `main.py`: `1,345 lines`

### Largest Active Modules (selected)
- `lib/operational_engine.py` (`3,745`)
- `lib/final_report_agent.py` (`3,644`)
- `lib/ai_report_generator.py` (`2,399`)
- `lib/microstructure.py` (`2,136`)
- `lib/graph_clustered_portfolio.py` (`1,823`)
- `pipeline/analyzers.py` (`1,473`)
- `main.py` (`1,345`)

### Structural Duplication
- backup pair count in `lib/`: `16`  
  example: `operational_engine_backup_3745lines.py` + `operational_engine.py`

## 2) Root Causes (Advanced Diagnosis)

1. Runtime tree와 historical artifact가 혼재됨  
- `lib/`에 active + backup + deprecated 참조가 동시에 존재

2. Orchestration 계층의 과도한 책임  
- `main.py` out-degree가 높고(내부 의존 다수), phase orchestration 외 책임이 섞임
- `pipeline/analyzers.py`가 1개 파일에 과다 집적

3. Domain boundary 불완전  
- execution 분리는 시작했지만 아직 `tactical/stress/operational` 경계가 미완료
- 보고서/실시간 도메인은 아직 monolith 의존

4. Import policy 부재  
- 계층 간 import 규칙이 문서 수준에 머물고 자동 점검이 없음

## 3) Architecture Decisions (Advanced Lane)

### AD-01: Core Minimalization
- `eimas`는 orchestration + contracts + adapters만 유지
- 알고리즘/도메인 로직은 외부 도메인 패키지로 분리

### AD-02: Adapter-First Extraction
- 직접 import 제거보다 adapter 도입을 선행
- backend switch + deterministic fallback은 필수

### AD-03: Runtime-Clean Tree
- backup/deprecated 모듈은 active runtime 경로에서 제거
- 보존은 `archive/` snapshot으로만 유지

### AD-04: Single Canonical Module Rule
- 동일 기능의 구현은 하나만 canonical로 지정
- 나머지는 shim 또는 archive로 이동

### AD-05: Gate Policy
- per-change: `py_compile + import smoke + function smoke`
- milestone: `python main.py --full` + required output fields

## 4) Target Structure

### Core (`eimas`)
- `main.py` (entry/orchestration)
- `pipeline/*` (phase orchestration, thin)
- `pipeline/schemas.py`, `core/schemas.py` (contracts)
- `lib/adapters/*` (domain adapters)

### External Domains
- `onchain_intelligence`
- `execution_intelligence`
- `reporting_intelligence` (planned)
- `realtime_intelligence` (planned)

## 5) Decomposition Roadmap (Advanced Priority)

### P0: Execution Wave Completion
- finalize split:
  - `tactical_allocation`
  - `stress_test`
  - `operational_engine` + `operational/`
- lock adapter contract keys

### P1: Orchestration Slimming
- `pipeline/analyzers.py` 분할:
  - `pipeline/analyzers/core.py`
  - `pipeline/analyzers/advanced.py`
  - `pipeline/analyzers/allocation.py`
- `main.py` phase helper를 `pipeline/phases/*`로 이동

### P2: Runtime Tree Cleanup
- `lib/*_backup_*` runtime path 제거
- deprecated 참조를 explicit archive boundary로 격리

### P3: Reporting/Realtime Domain Split
- report/realtime를 adapter 경유 외부 도메인으로 이관

## 6) Advanced vs General Task Map

### Advanced Lane (Codex)
- 도메인 경계/계약 설계
- split sequence 결정
- 회귀 리스크 평가
- merge/milestone 게이트 판단

### General Lane (Claude-code)
- 파일 이동/생성/삭제
- import rewrite
- boilerplate adapter wiring
- per-change 검증 실행/로그 수집

## 7) Immediate Advanced Backlog

### ADV-001
- Execution contract 고정:
  - allocation/rebalancing/tactical/stress/operational payload schema
  - spec file: `docs/architecture/EXECUTION_DOMAIN_CONTRACT_V1.md`
  - status: DONE

### ADV-002
- `pipeline/analyzers.py` 분할 설계안 확정
- 함수 소유권 매트릭스 작성
- spec file: `docs/architecture/ADV_002_ANALYZERS_SPLIT_OWNERSHIP_V1.md`
  - status: DONE

### ADV-003
- `main.py` orchestration-only 목표 구조 설계
- phase module boundaries 정의
- spec file: `docs/architecture/ADV_003_MAIN_ORCHESTRATION_BOUNDARY_V1.md`
  - status: DONE

### ADV-004
- `pipeline/analyzers_extension.py` canonical policy 확정
- spec file: `docs/architecture/ADV_004_ANALYZERS_EXTENSION_POLICY_V1.md`
- status: DONE

### ADV-005
- execution contract per-change verification 체크리스트 확정
- spec file: `docs/architecture/ADV_005_EXECUTION_CONTRACT_VERIFICATION_V1.md`
- status: DONE

## 8) Done Criteria for Bloat Resolution
- active runtime에서 backup/deprecated 직접 의존 0
- `main.py` < 700 lines, `pipeline/analyzers.py` < 500 lines
- domain logic는 adapter 경유 외부 모듈에서 로드
- milestone full 회귀 통과
