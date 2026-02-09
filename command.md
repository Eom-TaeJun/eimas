# EIMAS Command Policy (Single Entrypoint)

Last Updated: 2026-02-09

## 0) 문서 중요도 및 읽기 순서 (Priority)

### 중요도 등급

- `P0` (절대 기준, 충돌 시 우선): `command.md`
- `P1` (구조/원칙 이해): `README.md`, `ARCHITECTURE.md`, `CLAUDE.md`
- `P2` (현재 진행/실행 계획): `CURRENT_STATUS.md`, `TODO.md`, `FULL_EXECUTION_PROCESS.md`
- `P3` (세부 운영/참고): `docs/**`

### 권장 읽기 순서

1. `command.md` (`P0`)
2. `README.md` (`P1`)
3. `ARCHITECTURE.md` (`P1`)
4. `CLAUDE.md` (`P1`)
5. `CURRENT_STATUS.md` (`P2`)
6. `TODO.md` (`P2`)
7. `FULL_EXECUTION_PROCESS.md` (`P2`)
8. 필요 시 `docs/**` (`P3`)

핵심 원칙:
- 실행/진입점/명령 정책은 항상 `command.md`를 최우선 기준으로 해석한다.

## 1) 방향성 (Source of Truth)

- 통합 실행 진입점은 `main.py` 하나로 고정한다.
- 독립 기능도 모두 `python main.py --abc` 형태의 플래그로 귀속한다.
- 새 실행 래퍼(`run_*.sh`, `runner.py`)는 추가하지 않는다.

## 2) Canonical Commands

```bash
# 기본 실행
python main.py

# 빠른 실행
python main.py --short

# 풀 실행
python main.py --full

# 실시간 포함
python main.py --realtime -d 30
python main.py --full --realtime -d 30

# 포트폴리오 모듈 선택 실행
python main.py --backtest
python main.py --attribution
python main.py --stress-test

# Paper execution
python main.py --paper-auto --paper-account ra_auto
python main.py --paper-auto --paper-poll-only --paper-account ra_auto
python main.py --paper-auto --paper-backtest --paper-account ra_auto

# Quick AI validation market focus
python main.py --quick1
python main.py --quick2
```

## 3) Non-Canonical (제한)

- `cli/eimas.py run`은 `main.py` 인자 포워딩만 허용한다.
- 아래 파일은 제거되었으며 재도입하지 않는다.
  - `pipeline/runner.py`
  - `run_all_pipeline.sh`

## 4) 새 기능 추가 규칙 (`--abc` 귀속 규칙)

독립 기능을 추가할 때는 아래 순서를 따른다.

1. `main.py`에 CLI 플래그 추가 (`parser.add_argument('--abc', ...)`).
2. `run_integrated_pipeline(...)` 시그니처에 명시 파라미터 추가 (`enable_abc: bool = False`).
3. 실제 로직은 `pipeline/phases/phase*.py` 또는 `lib/*`에 구현.
4. `main.py`는 오케스트레이션만 수행하고 phase 호출만 연결.
5. 실행 관측성 유지:
   - `phase_timings`에 phase key 추가
   - 결과 메타데이터에 필요한 최소 필드 기록
6. 문서 동기화:
   - 이 파일(`command.md`)
   - `README.md` 실행 예시

## 5) 수정 범위 지정 가이드

- 명령 플래그/실행 정책 변경:
  - `main.py`
  - `command.md`
  - `README.md`
- 기능 로직 변경:
  - `pipeline/phases/*` (우선)
  - 필요 시 `lib/*`
- API 노출 변경:
  - `api/main.py`
- CLI 변경:
  - `cli/eimas.py`는 포워딩 범위에서만 수정

## 6) Do / Don't

- Do:
  - 실행 옵션은 `main.py`에만 정의
  - phase 단위로 작은 변경
  - `python -m compileall main.py cli/eimas.py api/main.py`로 최소 검증
- Don't:
  - 별도 실행 진입점 추가
  - `cli/eimas.py`에 파이프라인 비즈니스 로직 재삽입
  - 삭제된 레거시 래퍼 경로 재사용
