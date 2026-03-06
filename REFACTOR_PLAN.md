# EIMAS .claude/ 리팩토링 계획

Last Updated: 2026-03-06
Status: IN PROGRESS (세션 중단으로 작업 재개 필요)

## 배경

세 레포를 참고해 eimas의 `.claude/` 레이어를 현재 베스트 프랙티스에 맞게 리팩토링.

| 참고 레포 | 핵심 패턴 |
|---|---|
| [phuryn/pm-skills](https://github.com/phuryn/pm-skills) | 도메인별 스킬 묶음, `$ARGUMENTS` 지원, `marketplace.json` 카탈로그 |
| [shanraisshan/claude-code-best-practice](https://github.com/shanraisshan/claude-code-best-practice) | 에이전트 frontmatter 표준, skills preload 패턴, 컬러 코딩 |
| [Eom-TaeJun/tech-digest](https://github.com/Eom-TaeJun/tech-digest) | config-driven 아키텍처, 단계별 스크립트 분리, baseline 문서화 |

## 현재 .claude/ 상태

```
.claude/
  agents/
    eimas-runner.md         # 실행 전담 (blue)
    eimas-diagnostics.md    # 진단 전담 (red)
  skills/
    eimas-run-guide/SKILL.md       # agent preload (비공개)
    eimas-output-check/SKILL.md    # user-invocable: /eimas-output-check
    eimas-debug-guide/SKILL.md     # agent preload (비공개)
  commands/
    check.md                # /check → eimas-output-check 위임
```

## 목표 구조

```
.claude/
  agents/
    eimas-runner.md         # 현행 유지 + skills 보강
    eimas-diagnostics.md    # 현행 유지
    eimas-analyst.md        # NEW: 결과 분석/해석 전담 (purple)
  skills/
    pipeline/
      eimas-run-guide/SKILL.md       # 현행 유지
      eimas-phase-guide/SKILL.md     # NEW: 각 Phase 설명 + 입출력 스키마
    output/
      eimas-output-check/SKILL.md    # 현행 유지
      eimas-output-compare/SKILL.md  # NEW: 두 실행 결과 비교 ($ARGUMENTS)
    debug/
      eimas-debug-guide/SKILL.md     # 현행 유지
      eimas-error-patterns/SKILL.md  # NEW: 알려진 에러 패턴 카탈로그
    domain/
      eimas-schema-guide/SKILL.md    # NEW: schemas.py 핵심 필드 레퍼런스
      eimas-config-guide/SKILL.md    # NEW: configs/ 설정 옵션 레퍼런스
  commands/
    check.md          # 현행 유지: /check
    compare.md        # NEW: /compare [file1] [file2]
    phase.md          # NEW: /phase [N] → Phase N 설명
    run.md            # NEW: /run [mode] → 실행 모드 안내
  catalog.json        # NEW: pm-skills 방식 스킬 카탈로그 (선택)
```

## TODO 체크리스트

### Phase A — 스킬 디렉토리 재구성 (구조만, 내용 유지)
- [ ] `skills/` 하위에 `pipeline/`, `output/`, `debug/`, `domain/` 폴더 생성
- [ ] 기존 3개 스킬 파일 → 새 경로로 이동
- [ ] 에이전트 파일의 `skills:` 경로 업데이트

### Phase B — 신규 스킬 추가
- [ ] `pipeline/eimas-phase-guide/SKILL.md` — Phase 1~9 입출력 요약
- [ ] `output/eimas-output-compare/SKILL.md` — `$ARGUMENTS`로 두 JSON 비교
- [ ] `debug/eimas-error-patterns/SKILL.md` — 현재 debug-guide에서 패턴 분리
- [ ] `domain/eimas-schema-guide/SKILL.md` — `pipeline/schemas.py` 핵심 필드
- [ ] `domain/eimas-config-guide/SKILL.md` — `configs/default.yaml` 옵션 정리

### Phase C — 신규 에이전트 추가
- [ ] `agents/eimas-analyst.md`
  - 역할: 결과 해석, 투자 관점 코멘트, 레짐/리스크/운용 판단 요약
  - tools: Read, Bash, Glob
  - model: sonnet, color: purple
  - skills: eimas-output-check, eimas-output-compare, eimas-schema-guide

### Phase D — 신규 커맨드 추가
- [ ] `commands/compare.md` — `/compare` → eimas-output-compare 스킬 위임
- [ ] `commands/phase.md` — `/phase` → eimas-phase-guide 스킬 위임
- [ ] `commands/run.md` — `/run` → 실행 모드 안내 (Bash 실행 X, 가이드만)

### Phase E — 문서 동기화
- [ ] `CLAUDE.md`에 `.claude/` 레이어 구조 요약 추가
- [ ] `README.md`에 Claude Code 통합 섹션 추가 (에이전트/스킬/커맨드 목록)
- [ ] `CHANGELOG.md`에 v2.3.2 항목 추가

## 작업 원칙

- Python 코드 수정 없음 — AI workflow 레이어만
- 스킬 내용은 코드에서 직접 추출 (schemas.py, configs/ 등 읽어서 작성)
- pm-skills 스타일: 스킬 description에 `$ARGUMENTS` 적극 활용
- 기존 에이전트 동작 깨지지 않도록 — 경로 변경 시 반드시 agents 파일도 업데이트
