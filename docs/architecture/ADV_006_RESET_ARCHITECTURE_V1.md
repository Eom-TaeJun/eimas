# ADV-006: EIMAS Reset Architecture v1 (2026-02-06)

## 1. Decision

기존 파일 중심 분할이 아니라, 책임 경계 중심으로 EIMAS를 재구성한다.

- `main.py`는 최종적으로 orchestration-only로 축소한다.
- 런타임 코드는 아래 4계층으로 분리한다:
  - `interface`: CLI/API/runner entrypoints
  - `application`: pipeline orchestration, phase sequencing, run config
  - `domain`: 분석 규칙, 결정 로직, 스키마/계약
  - `adapters`: DB, 외부 API, 파일 I/O, 외부 모델 연결

## 2. Why this reset

- 현재는 phase 분할, analyzer 분할, adapter 분할이 혼재되어 기준이 일관되지 않다.
- `main.py`에 여전히 phase 로직이 많이 남아 구조적 병목이 지속된다.
- 문서/워크오더/코드의 진실 소스가 분산되어 병렬 작업 충돌 위험이 높다.

## 3. Target Package Boundaries

### 3.1 Interface
- 후보: `cli/eimas.py`, `api/main.py`, `pipeline/runner.py`, `run_all_pipeline.sh`
- 원칙: 실행 인자 파싱/입력 변환만 담당, 비즈니스 로직 금지.

### 3.2 Application
- 후보: `pipeline/phases/*`, 향후 `pipeline/app/*`
- 원칙: phase 호출 순서, mode 분기(full/quick/cron), 오류 제어만 담당.

### 3.3 Domain
- 후보: `pipeline/schemas.py`, 분석 결과 계약, 정책 모델
- 원칙: 순수 로직/계약은 I/O 의존 금지.

### 3.4 Adapters
- 후보: `lib/adapters/*`, `pipeline/storage.py`, 외부 데이터 collector 연결
- 원칙: 외부 시스템 접근 책임 집중.

## 4. Migration Strategy (non-breaking)

### R0 (done)
- analyzer monolith 분할 + facade 유지.
- execution contract check script 도입.

### R1 (in progress)
- `pipeline/phases/*`에 main helper 이관 시작.
- 완료 상태: phase5/phase7/phase8 로직 main에서 이관.

### R2
- phase1~4.5 helper 이관 완료.
- `main.py`는 orchestrator 함수 + CLI만 유지.

### R3
- 공통 실행 설정을 application 계층으로 승격:
  - `RunConfig`
  - `RunArtifacts`
  - mode policy matrix

### R4
- 문서 체계 분리:
  - architecture ADR
  - execution contract
  - work orders
  - handoff
  각 문서의 source-of-truth 단일화.

## 5. Rules for parallel contributors

- 함수 이동 시 public 계약(이름/시그니처/핵심 반환 키) 변경 금지.
- 파이프라인 단계 이동은 1~2 phase 단위로 작게 수행.
- 각 PR/작업 종료 시 아래 최소 검증 수행:
  - `python3 -m py_compile ...`
  - `bash scripts/check_execution_contract.sh`
- full run(`python main.py --full`)은 구조 이관 묶음 완료 시점에만 실행.

## 6. Immediate Next Work Orders

- `GEN-302`: Phase1~2 helper 이동 + import wiring
- `GEN-303`: Phase3~8 helper 이동 + main slim-down + validation
- `GEN-304` (new): RunConfig/PathPolicy application layer 도입
- `GEN-305` (new): 문서 source-of-truth 재정렬 및 중복 문서 정리

