# ADV-007: Single Artifact Path Policy v1 (2026-02-06)

## 1. Decision

하나의 파이프라인 실행(run)에서 JSON 산출물 경로는 **단일 canonical 파일**로 유지한다.

- Phase 5에서 최초 JSON 파일을 생성한다.
- 이후 Phase 7/8 업데이트는 같은 파일을 덮어써 상태를 누적 반영한다.
- 최종 출력(`Output: ...`)은 해당 canonical 파일을 가리킨다.

## 2. Why

- 기존 구조에서는 `save_result_json(...)`가 호출될 때마다 timestamp 기반 새 파일을 생성했다.
- 그 결과:
  - 한 run에서 JSON 파일이 여러 개 생김
  - 콘솔 `Output` 경로와 최종 상태 파일이 불일치할 수 있음
  - 후속 단계/외부 자동화가 어떤 파일을 기준으로 삼아야 하는지 모호해짐

## 3. Contract

### 3.1 Storage API
- `pipeline/storage.py`
  - `save_result_json(result, output_dir=None, output_file=None) -> str`
  - `output_file`가 주어지면 해당 경로를 사용해 덮어쓴다.
  - `output_file`가 없으면 기존처럼 timestamp 파일을 신규 생성한다.

### 3.2 Phase usage
- Phase 5:
  - canonical `output_file`를 생성하고 반환
- Phase 7/8:
  - `output_file`를 인자로 받아 동일 경로에 업데이트 저장

### 3.3 Compatibility
- 기존 호출(`save_result_json(result, output_dir=...)`)은 그대로 동작한다.
- 신규 정책은 `output_file`를 전달한 경로에서만 활성화된다.

## 4. Implementation Notes

- 적용 파일:
  - `pipeline/storage.py`
  - `pipeline/phases/phase7_report.py`
  - `pipeline/phases/phase8_validation.py`
  - `main.py`
- 검증:
  - `python3 -m py_compile ...`
  - `bash scripts/check_execution_contract.sh`

## 5. Follow-up

- (선택) Markdown 산출물(`save_result_md`)도 run 단일 경로 정책으로 통일할지 결정한다.
- (선택) `RunArtifacts` dataclass를 도입해 JSON/MD/report 파일 경로를 명시적으로 관리한다.
