# EIMAS Structure Redesign Masterplan (2026-02-06)

## 1. Goal
- `eimas`를 "full orchestration core"로 축소한다.
- 기능 분리는 작은 단위로 진행한다.
- `python main.py --full`은 매 수정 검증이 아니라 병합/마일스톤 검증으로 사용한다.
- 비대화 해소의 상세 진단/결정은 `BLOAT_RESOLUTION_ARCHITECTURE.md`를 기준으로 한다.

## 2. Core Principle
- 작은 단위 분리 우선:
  - 한 번에 한 도메인, 한 기능군만 이동
  - import 전환은 adapter 레이어로 먼저 처리
- full 실행은 마일스톤에서만:
  - 매 변경마다 full 실행하지 않는다
  - wave 완료 또는 release 후보에서만 full 회귀 실행
- 문서 우선:
  - 코드 이동 전에 경계/계약/검증 기준을 md에 먼저 고정

## 3. Target Architecture

### 3.1 Domain Map
- `eimas` (core_full):
  - orchestration (`main.py`, `pipeline/*`)
  - 공통 schema/contract
  - adapter layer (`lib/adapters/*`)
- `onchain_intelligence`:
  - 온체인 데이터 수집/분석/신호
- `execution_intelligence`:
  - 비중산출, 리밸런싱, 운영결정
- `reporting_intelligence`:
  - 리포트 생성, whitening, fact-check, converter
- `realtime_intelligence`:
  - stream, realtime pipeline, alert

### 3.2 Dependency Rules
- `eimas` -> 외부 도메인 직접 의존 금지, adapter만 사용
- 외부 도메인 -> `eimas` 내부 모듈 의존 금지
- 공유 타입은 JSON contract 또는 최소 dataclass로 제한

## 4. Split Unit Definition
- 기본 단위:
  - 기능군 1개 (예: allocation/rebalancing)
  - 관련 import 전환 1회
  - 문서 업데이트 1회
- 변경 규모 가이드:
  - 신규/수정 파일 3~12개
  - 핵심 로직 이동 후 fallback 유지

## 5. Validation Policy (v2)

### 5.1 Per-Change (항상 실행)
- `py_compile` for touched files
- import smoke test
- domain-level function smoke test

### 5.2 Per-Wave (선택적, 권장)
- 해당 도메인 integration smoke
- adapter fallback 동작 확인

### 5.3 Milestone / Merge Gate (필수)
- `python main.py --full`
- 필수 결과 필드 확인:
  - `risk_score`
  - `final_recommendation`
  - `full_mode_position`
  - `reference_mode_position`
  - `modes_agree`

## 6. Execution Wave Plan

### Wave 1: Execution Intelligence
- W1-A (완료): `allocation_engine`, `rebalancing_policy` 분리 + adapter 연결
- W1-B (다음): `tactical_allocation`, `stress_test` 분리
- W1-C (다음): `operational_engine` + `operational/` 정리

### Wave 2: Reporting Intelligence
- `pipeline/report.py`
- `lib/ai_report_generator.py`
- `lib/whitening_engine.py`
- `lib/autonomous_agent.py`
- `lib/json_to_md_converter.py`
- `lib/json_to_html_converter.py`

### Wave 3: Realtime Intelligence
- `pipeline/realtime.py`
- `lib/binance_stream.py`
- `lib/realtime_pipeline.py`

## 7. Adapter Contract Checklist
- backend selector env var 존재
- external 실패 시 deterministic fallback
- 출력은 JSON-serializable payload
- 주요 키 누락 시 safe default

## 8. Working Agreement
- 긴 본문 설계는 md 파일에 직접 작성한다.
- 채팅에는 변경 파일/핵심 결론만 공유한다.
- 설계 문서 업데이트 없이 구조 변경을 진행하지 않는다.
- 작업 위임/병렬 실행 기준은 `PARALLEL_WORK_SPLIT_PROTOCOL.md`를 따른다.

## 9. Definition of Done
- 설계 문서 반영 완료
- 작은 단위 분리 완료
- per-change 검증 통과
- milestone 시 full 회귀 통과
