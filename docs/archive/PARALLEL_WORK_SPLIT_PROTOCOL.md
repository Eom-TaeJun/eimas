# EIMAS Parallel Work Split Protocol (2026-02-12)

## 1) Objective
- 상위 추론(리뷰/개선/설계)은 Opus 4.6이 담당한다.
- 실제 파일 수정/명령 실행은 Codex가 담당한다.
- 필요 시 Claude executor는 fallback으로만 사용한다.
- 최종 품질 게이트(버그/회귀/검증 누락 판단)는 Opus + Codex 합의로 통과시킨다.

## 2) Role Split

### A. Reasoning Lane (Opus 4.6)
- 구조 설계, 도메인 경계 정의, 의존성 정리
- 계약(JSON schema, adapter contract) 설계/변경 승인
- 리스크 분석, 회귀 영향도 분석, 우선순위 결정
- Work Order 작성 및 승인
- 최종 코드 리뷰(버그/회귀/테스트 누락 중심)

### B. Execution Lane (Codex Default)
- 파일 생성/이동/삭제
- import 경로 수정
- boilerplate adapter/wrapper 작성
- 반복 rename/sed/정리
- 문서 반영(Work Order 기반)
- per-change 검증 명령 실행 및 결과 수집

### C. Fallback Execution Lane (Claude Optional)
- Codex 실행 장애 시 동일 Work Order를 Claude로 재실행
- 결과는 동일하게 Opus 리뷰 게이트를 통과해야 함

## 3) Task Classification Rule
- 아래 중 하나라도 해당하면 Reasoning Lane:
  - 아키텍처 결정이 필요한 작업
  - 여러 도메인 계약을 바꾸는 작업
  - 실패 시 전략/의사결정이 바뀌는 작업
  - 요구사항 해석이 모호한 작업
- 위 조건이 없고, 명확한 변경 지시가 가능하면 Execution Lane

## 4) Work Order Template (Opus -> Codex)
아래 형식으로만 지시한다.

```md
[WORK_ORDER]
id: GEN-###
goal: (한 줄 목표)

context:
- (왜 필요한지 2~4줄)

scope_files:
- path/a.py
- path/b.md

tasks:
1. (정확한 수정/이동/삭제 지시)
2. (정확한 수정/이동/삭제 지시)

constraints:
- 구조 결정 금지 (설계 변경 금지)
- 지시 범위 외 파일 수정 금지
- fallback/예외 처리 패턴 유지

validation:
- python3 -m py_compile ...
- python3 -c "import ..."
- (필요한 smoke test)

deliverables:
- 변경 파일 목록
- 핵심 diff 요약
- validation 결과
[/WORK_ORDER]
```

## 5) Execution Loop (Opus + Codex)
1. Opus 4.6이 분석 후 Work Order 생성
2. Codex executor가 Work Order를 직접 실행
3. Codex가 변경/검증 로그 산출
4. Opus 4.6이 결과 리뷰 후 승인 또는 수정 지시
5. wave 완료 시 milestone gate로 full 회귀 실행

## 6) Validation Ownership
- Execution Lane(Codex): per-change 검증
  - `py_compile`
  - import smoke
  - domain function smoke
- Reasoning Lane(Opus): milestone gate
  - `python main.py --full`
  - 필수 필드 무결성 확인

## 7) Decision Boundary (중요)
- Codex가 설계 변경 필요를 감지하면 즉시 중단하고 BLOCKED로 반환한다.
- Opus 승인 없이 계약 필드/게이트 조건/핵심 흐름을 바꾸지 않는다.

## 8) Definition of Done
- Reasoning/Execution 경계가 Work Order로 명확히 기록됨
- Execution은 지시 범위 내에서만 수행됨
- Opus 리뷰에서 치명 이슈 없음
- milestone에서 full 회귀 통과

## 9) Runtime Setup (WSL)

### Recommended `.env`
```bash
# Lane roles
ADVANCED_REASONER_MODEL=opus4.6
GENERAL_LANE_EXECUTOR=codex

# Codex execution defaults
CODEX_GENERAL_MODEL=gpt-5-codex
CODEX_GENERAL_FULL_AUTO=true
CODEX_GENERAL_SANDBOX=workspace-write

# Claude fallback (optional)
CLAUDE_GENERAL_MODEL=claude-sonnet-4-5-20250929
CLAUDE_GENERAL_PERMISSION_MODE=acceptEdits
```

## 10) Run Commands

### Codex execution (default)
```bash
cd /home/tj/projects/autoai/eimas
./scripts/delegate_general_lane.sh --work-order work_orders/GEN-TEMPLATE.md --dry-run
./scripts/delegate_general_lane.sh --executor codex --work-order work_orders/GEN-101.md
```

### Claude fallback
```bash
./scripts/delegate_general_lane.sh --executor claude --work-order work_orders/GEN-101.md
```

## 11) Artifacts
- 출력 경로: `outputs/general_lane/<timestamp>_<WORK_ID>/`
- 포함 파일:
  - `request_prompt.md`
  - `response_stdout.log`
  - `response_stderr.log`
  - `meta.json`
  - `response_stdout.json` (Claude 실행 시 호환용)
