# EIMAS 현재 상태 (2026-02-19)

## 1) Snapshot

- 기준 브랜치: `main`
- Canonical entrypoints:
  - `main.py --full` : 시장환경 전체 분석
  - `main.py --short`: 실시간 대응 (full 결과 활용 → 운용 → DB 적재)
  - `api/main.py`    : FastAPI 서버
  - `cli/eimas.py`   : 포워딩 래퍼

## 2) 완료된 주요 작업

### 리팩토링 (2026-02-19)
- **Phase 1**: 미사용 파일 삭제 (`path_bootstrap`, `parallel_data_collector`, `news_event_generator`, `xai_explainer`, `explanation_generator`)
- **Phase 2**: monolith → 패키지 shim 전환 (9개 파일)
  - `bubble_detector/framework` → `lib/bubble/`
  - `event_framework` → `lib/event_framework/`
  - `report_generator` → `lib/reports/`
  - `microstructure` → `lib/microstructure/`
  - `genius_act_macro` → `lib/genius_act/`
  - `graph_clustered_portfolio` → `lib/graph_portfolio/`
  - `causality_graph/causal_network` → `lib/causality/`
- **Phase 3**: `--short` 모드 재정의
  - `run_short_pipeline_phases()` 신설 (`pipeline/app/orchestrator_steps.py`)
  - `outputs/` 최신 full 결과 자동 로드 → 운용 의사결정 컨텍스트 주입

### 이전 완료 작업
- 단일 진입점 `main.py` 통합 (Track A)
- `execution_intelligence` 도메인 분리 (Track B 부분)
- `--profile us-trader-v1`, `--paper-auto` 추가 (Track F)
- Backtest DB v2.1, Phase timing telemetry

## 3) 남은 우선순위 작업

### 🔴 Track B — 도메인 분리
- [ ] `reporting_intelligence` 신설: `lib/ai_report_generator.py`, `lib/whitening_engine.py`
- [ ] `realtime_intelligence` 신설: `pipeline/realtime.py`, `lib/binance_stream.py`
- [ ] `lib/operational_engine.py` monolith 제거 (fallback 관측 후)

### 🔴 Track C — 성능
- [ ] FULL 실행 시간: 249s → 120s 목표
- [ ] `pytest` 환경 정비

### 🟡 Track E — RA SQL Productionization
- [ ] `fi_ra` 스키마 표준 확정
- [ ] ETL: staging → mart 구조
- [ ] RA 조회용 SQL 뷰 (`v_ra_macro_regime`, `v_ra_etf_signal`)

### 🟡 Track F — 운영 리스크
- [ ] 운영 리스크 파라미터 실측 튜닝
- [ ] cron/daemon 연속 실행 + 장애 복구 검증

## 4) 세션 재시작 체크리스트

```bash
cd /home/tj/projects/autoai/eimas
git status --short
python main.py --help
python -m compileall main.py pipeline/app
```

다음 착수 권장:
1. Track E (RA SQL) — `fi_ra` 스키마 표준화
2. Track B — `reporting_intelligence` 분리
3. Track C — pytest 환경 정비
