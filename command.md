# EIMAS Command Policy (Single Entrypoint)

Last Updated: 2026-02-19

## 0) 문서 우선순위

- `P0` (절대 기준): `command.md`
- `P1` (구조/원칙): `README.md`, `CLAUDE.md`
- `P2` (현재 상태): `CURRENT_STATUS.md`, `TODO.md`
- `P3` (세부 참고): `docs/**`

## 1) 방향성

- 실행 진입점은 `main.py` 하나로 고정
- 모든 기능은 `python main.py --abc` 형태로 귀속
- 새 실행 래퍼(`run_*.sh`, `runner.py`)는 추가하지 않음

## 2) Canonical Commands

```bash
# 시장환경 전체 분석 (Phase 1~9)
python main.py --full

# 실시간 대응 (경량 수집 → 운용 → DB 적재)
# outputs/ 최신 full 결과를 자동 로드해 운용 의사결정에 활용
python main.py --short

# Debate 제어 (--full 전용)
python main.py --full --debate-full-lookback 180 --debate-ref-lookback 45
python main.py --full --debate-skip-reference

# 실시간 스트리밍 포함
python main.py --short --realtime -d 30
python main.py --full --realtime -d 30

# 포트폴리오 모듈
python main.py --backtest
python main.py --attribution
python main.py --stress-test

# 모의주문 (--short와 함께 사용 권장)
python main.py --short --paper-auto --paper-account ra_auto
python main.py --short --paper-auto --paper-poll-only
python main.py --short --paper-auto --paper-backtest

# Quick AI 검증
python main.py --quick1   # KOSPI 포커스
python main.py --quick2   # SPX 포커스

# Pipeline profile
python main.py --full --profile us-trader-v1   # 버블/센티먼트 스킵
python main.py --full --profile legacy          # 기본 (default)
```

## 3) Non-Canonical (제한)

- `cli/eimas.py run`은 `main.py` 인자 포워딩만 허용
- 아래 파일은 제거됨, 재도입 금지:
  - `pipeline/runner.py`, `run_all_pipeline.sh`
  - `lib/path_bootstrap.py`, `lib/parallel_data_collector.py`

## 4) 새 기능 추가 규칙

1. `main.py`에 CLI 플래그 추가
2. `run_integrated_pipeline()` 시그니처에 파라미터 추가
3. 로직은 `pipeline/phases/phase*.py` 또는 `lib/*`에 구현
4. `main.py`는 오케스트레이션만 (phase 호출 연결)
5. 문서 동기화: `command.md`, `README.md`

## 5) 수정 범위 지정

- 명령/실행 정책: `main.py` + `command.md`
- 기능 로직: `pipeline/phases/*` (우선), `lib/*`
- 오케스트레이션: `pipeline/app/*`
  - Full 흐름: `run_pipeline_phases()`
  - Short 흐름: `run_short_pipeline_phases()`
- API: `api/main.py`

## 6) Do / Don't

- **Do**: 실행 옵션은 `main.py`에만 정의 / phase 단위로 작은 변경
- **Don't**: 별도 진입점 추가 / 삭제된 레거시 래퍼 재사용
