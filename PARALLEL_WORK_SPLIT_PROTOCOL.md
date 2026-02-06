# EIMAS Parallel Work Split Protocol (2026-02-06)

## 1) Objective
- 고급 추론 작업과 일반 실행 작업을 분리한다.
- 고급 작업은 Codex(여기)에서 수행한다.
- 일반 작업은 Claude-code에 병렬 위임한다.
- 최종 품질 판단과 병합 게이트는 Codex가 담당한다.

## 2) Role Split

### A. Advanced Lane (Codex Only)
- 구조 설계, 도메인 경계 정의, 의존성 정리
- 계약(JSON schema, adapter contract) 설계
- 리스크 분석, 회귀 영향도 분석, 우선순위 결정
- 작업 단위 분해와 지시서(Work Order) 작성
- 결과 코드 리뷰(버그/회귀/누락 테스트 중심)
- milestone merge 판단

### B. General Lane (Claude-code Delegated)
- 파일 생성/이동/삭제
- import 경로 수정
- boilerplate adapter/wrapper 작성
- 반복적 rename/sed/정리 작업
- 문서 반영(지시서 기반)
- per-change 검증 명령 실행 및 결과 수집

## 3) Task Classification Rule
- 아래 중 하나라도 해당하면 Advanced Lane:
  - 아키텍처 결정이 필요한 작업
  - 여러 도메인 계약을 바꾸는 작업
  - 실패 시 전략/의사결정이 바뀌는 작업
  - 요구사항 해석이 모호한 작업
- 위 조건이 없고, 명확한 변경 지시가 가능하면 General Lane

## 4) Work Order Template (Codex -> Claude-code)
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

## 5) Parallel Execution Loop
1. Codex가 Advanced 분석 후 Work Order 생성
2. 사용자(당신)가 Work Order를 Claude-code에 병렬 전달
3. Claude-code가 결과/검증 로그를 반환
4. Codex가 리뷰 후 승인/수정 지시
5. wave 완료 시 milestone gate로 full 회귀 실행

## 6) Validation Ownership
- General Lane: per-change 검증만 수행
  - `py_compile`
  - import smoke
  - domain function smoke
- Advanced Lane: milestone gate 수행
  - `python main.py --full`
  - 필수 필드 무결성 확인

## 7) Current Split for EIMAS

### Advanced Lane Backlog
- `execution_intelligence` 1차/2차 분리 경계 확정
- `tactical_allocation`/`stress_test` 계약 설계
- `operational_engine` 최종 이관 전략 설계
- reporting/realtime wave 순서와 게이트 설계

### General Lane Backlog
- 설계 문서 기준으로 파일 이동/adapter wiring
- import 전환 및 unused import 정리
- TODO/프로세스 문서 상태 업데이트
- per-change 검증 실행/결과 수집

## 8) Decision Boundary (중요)
- Claude-code가 "설계 변경이 필요하다"고 판단하면 즉시 중단하고 Codex에 되돌린다.
- Codex 승인 없이 계약 필드/게이트 조건/핵심 흐름을 바꾸지 않는다.

## 9) Definition of Done
- 고급/일반 작업 경계가 Work Order로 명확히 기록됨
- 일반 작업은 지시 범위 내에서만 수행됨
- Codex 리뷰에서 치명 이슈 없음
- milestone에서 full 회귀 통과

## 10) WSL Runtime (API Key + Sonnet 4.5)

### Setup
1. `.env`에 `ANTHROPIC_API_KEY`를 설정한다.
2. 필요 시 모델을 지정한다:
   - `CLAUDE_GENERAL_MODEL=claude-sonnet-4-5-20250929`

### Run (General Lane)
```bash
cd /home/tj/projects/autoai/eimas
./scripts/delegate_general_lane.sh --work-order work_orders/GEN-TEMPLATE.md --dry-run
./scripts/delegate_general_lane.sh --work-order work_orders/GEN-101.md
```

### Artifacts
- 출력 경로: `outputs/claude_general/<timestamp>_<WORK_ID>/`
- 포함 파일:
  - `request_prompt.md`
  - `response_stdout.json`
  - `response_stderr.log`
  - `meta.json`
