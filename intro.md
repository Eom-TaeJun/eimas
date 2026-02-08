# 미래에셋증권 RA 인턴 지원용 EIMAS 소개 정리

기준일: 2026-02-08  
목적: 매크로/ETF 전략 RA 직무에 맞춰 EIMAS의 현재 역량, 어필 포인트, 보완 과제를 정리한다.

## 1) 채용 공고 기준 핵심 요구

- 업무: 자료조사, 조사분석자료 작성 보조, 데이터 업데이트, 세미나/유관부서 협조자료 작성 보조
- 요구: 경제/금융시장 기본 이해, 재무/회계 기본 이해, MS Office 활용, 외국어
- 우대: 6개월 이상 근무 가능, 주식 밸류에이션/기초 주식분석, SQL 활용

## 2) EIMAS 현재 파이프라인 요약 (코드 기준)

- 실행 엔트리: `main.py`의 `run_integrated_pipeline()`
- 표준 실행: `python main.py --full`
- 핵심 단계:
  - Phase 1: 거시/시장/암호화폐 + RA 기업분석 수집
  - Phase 2: 레짐/리스크/고급 정량 분석
  - Phase 3: 에이전트 토론 합의
  - Phase 4.5: 운용 의사결정/제약 점검
  - Phase 5: JSON 저장
  - Phase 7~9: 리포트/검증/아티팩트
- 최신 실행 스냅샷: `outputs/eimas_20260208_193354.json`

## 3) RA 지원 관점 어필 포인트 (증빙 기반)

### A. 매크로 + ETF 전략 관점

- 레짐/리스크 기반 투자 시그널 생성:
  - `risk_score=6.43`, `risk_level=LOW`, `final_recommendation=BULLISH`
  - `full_mode_position=BULLISH`, `reference_mode_position=BULLISH`, `modes_agree=true`
- ETF 스냅샷 자동 생성:
  - `SPY, QQQ, IWM, XLF, TLT, GLD`의 5일/20일 수익률과 모멘텀 라벨 제공
- 운용 통제 로직 내장:
  - HOLD 조건/제약 충족 여부/규칙 로그를 `operational_report`로 남김

### B. 기업 밸류에이션 + 회계 기초 대응

- `company_ra_analysis`에서 종목별 회계/밸류 지표를 구조화:
  - 예시 커버리지: `AAPL, MSFT, NVDA, JPM, XOM`
  - 밸류: trailing/forward P/E, P/B, EV/EBITDA
  - 회계: 매출, 영업이익, 순이익, 현금흐름, 자산/부채/자본
  - 비율: ROE, ROA, 마진, D/E, 유동비율
- RA 실무형 보조 섹션 포함:
  - `ra_work_support.research_tasks`
  - `seminar_material_points`
  - `cross_department_support_points`

### C. SQL 활용 역량 (우대사항 대응)

- PostgreSQL 적재 경로 연결:
  - `company_ra_analysis.postgresql.enabled=true`
  - `stored_rows=5`, `table=fi_ra.company_fundamentals`
- 재부팅 대응 런북 문서화:
  - `docs/manuals/RA_POSTGRES_REBOOT_RUNBOOK.md`

### D. RA 스타일 리포트 자동화

- RA 포맷 리포트 생성기 기본화:
  - 실행: `python3 scripts/generate_final_report.py`
  - 결과 예시: `outputs/reports/allocation_report_20260208_200759.md`
- RA 포맷 PDF 자동 생성 경로 추가:
  - 실행: `python3 scripts/generate_final_report.py --pdf`
  - 생성 순서: `RA MD -> RA HTML -> RA PDF`
  - 결과 예시:
    - `outputs/reports/allocation_report_20260208_*.md`
    - `outputs/reports/allocation_report_20260208_*.html`
    - `outputs/reports/allocation_report_20260208_*.pdf`
- IB 스타일은 옵션 유지:
  - 실행: `python3 scripts/generate_final_report.py --style ib`

### F. 추천 → 실행(모의) → DB → 백테스트 연결

- 목표가 기반 자동 모의주문 파이프라인 추가:
  - 파이프라인 옵션: `python main.py --full --paper-auto`
  - 동작: `trade_plan`의 BUY/SELL를 LIMIT 주문으로 등록 후 목표가 도달 시 자동 체결
- 실행 로그 DB 저장:
  - `data/paper_trading.db`: 계좌/주문/체결(거래) 원장
  - `data/trading.db`: `executions`(status, target_price, executed_price, external_order_id)
- 대기 주문 폴링 전용 실행:
  - `python scripts/auto_paper_execution.py --poll-only --account ra_auto`
- 백테스트 연동:
  - `python scripts/auto_paper_execution.py --run-backtest`
  - 배분비중 기반 백테스트 결과를 `backtest_runs`, `backtest_daily_nav`, `backtest_snapshots`에 저장

### E. PostgreSQL 작업 증빙 (SQL + 결과)

- 실행 DB/테이블:
  - `postgresql://postgres@127.0.0.1:55432/ra_fi`
  - `fi_ra.company_fundamentals`
- 검증 SQL:

```sql
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT ticker) AS tickers,
       MIN(as_of_date) AS min_date,
       MAX(as_of_date) AS max_date
FROM fi_ra.company_fundamentals;
```

- 검증 결과 (2026-02-08):
  - `total_rows=5`
  - `tickers=5`
  - `min_date=2026-02-08`
  - `max_date=2026-02-08`
- 샘플 조회:

```sql
SELECT as_of_date, ticker, sector, trailing_pe, revenue
FROM fi_ra.company_fundamentals
ORDER BY ticker
LIMIT 10;
```

## 4) 자기소개서에 바로 쓸 키워드

- 매크로 레짐 기반 ETF 전략 보조
- 데이터 업데이트 자동화 (거시/시장/기업 재무)
- 밸류에이션/회계 지표 정리 및 코멘트 작성
- SQL(PostgreSQL) 기반 리서치 데이터 적재/조회
- 의사결정 로그 추적 가능성 (Explainable Research Pipeline)
- 조사자료 재현성/정합성 관리 (JSON 아티팩트 기준)
- 세미나/유관부서 협조자료용 구조화 데이터 생산
- 목표가 기반 모의주문 자동화 + 체결 DB 기록 + 백테스트 루프

## 5) 자기소개서 문장 뼈대 (초안)

- “저는 거시-시장-기업 데이터를 한 파이프라인에서 연결해, RA 업무의 핵심인 데이터 업데이트와 조사자료 작성 보조를 자동화했습니다.”
- “특히 매크로/ETF 관점에서 레짐, 리스크 점수, ETF 모멘텀 스냅샷을 일관된 포맷으로 제공해 리서치 노트 작성 시간을 줄였습니다.”
- “기업 분석에서는 밸류에이션과 회계 지표를 SQL에 적재하고, 종목별 핵심 포인트를 즉시 조회 가능한 형태로 관리했습니다.”
- “실무에서는 자동화 결과를 그대로 쓰기보다, 근거 필드와 점검 로그를 함께 확인해 검증 가능한 리서치 프로세스를 유지했습니다.”

## 6) RA 관점에서 아직 부족한 부분

- ETF 전략 심화 부족:
  - 팩터 노출(가치/성장/퀄리티), 섹터·듀레이션 분해, 벤치마크 대비 추적오차 분석이 약함
- 기업분석 깊이 부족:
  - 컨센서스 EPS revision, 실적 서프라이즈, 가이던스 변화 추적이 부족함
- 산출물 형식 부족:
  - Excel/PPT 바로 제출형 포맷(표준 템플릿) 자동화가 약함
- 커버리지 확장성 부족:
  - 현재 RA 기업 커버리지가 5종목 중심으로 좁음
- 검증 체계 보강 필요:
  - 필드 단위 데이터 품질 경고(결측/비정상치)와 변경이력 리포트 고도화 필요
- 실행 체계 보강 필요:
  - 거래세션/유동성(스프레드) 반영 체결모형과 실거래 연계 전 단계 리스크 체크리스트 고도화 필요

## 7) 우선 보완 과제 (RA 맞춤)

- 1순위: ETF 전략 리포트 확장
  - 팩터/섹터/금리민감도 분해표 + 전일 대비 변화 코멘트 추가
- 2순위: 기업실적 모니터링 강화
  - EPS revision/실적 발표 캘린더/서프라이즈 요약 섹션 추가
- 3순위: SQL 스키마 확장
  - `as_of_date`, `source`, `revision_tag`, `quality_flag` 컬럼 표준화
- 4순위: 제출형 문서 자동화
  - Markdown -> Excel/PPT 템플릿 변환 파이프라인 추가
- 5순위: 영어 요약 보강
  - 세미나/외부 커뮤니케이션용 영문 1-page 요약 자동 생성

## 8) 면접/자소서에서 강조할 메시지

- “모델 정확도”보다 “리서치 실무 생산성”과 “검증 가능성”을 개선했다는 점
- SQL 기반 데이터 운영과 문서화(런북/결과 아티팩트)까지 포함해 실무 이관성을 고려했다는 점
- 매크로/ETF RA 업무를 기준으로 기능을 선택하고, 부족한 부분은 우선순위로 관리하고 있다는 점

## 9) 자기소개서 소스 문안 (요청 형식)

### 1) 현재 구현한 내용 기반 어필 포인트

- 거시-시장-기업 통합 파이프라인:
  - FRED 기반 금리/유동성, 시장 레짐/리스크, ETF/기업 지표를 하나의 파이프라인으로 묶어 데이터 업데이트와 조사 보조를 자동화했다.
- 매크로/ETF 전략 연결:
  - 레짐(BULL/BEAR/NEUTRAL), 리스크 점수, ETF 모멘텀/분해 정보(팩터·섹터·듀레이션)를 함께 제시해 거시 판단이 ETF 아이디어로 바로 이어지게 구성했다.
- 회계/재무 + 밸류에이션 구조화:
  - `company_ra_analysis`에서 매출/영업이익/순이익/현금흐름, ROE/ROA/마진/D/E와 P/E/P/B/EV/EBITDA를 종목별로 정리해 기초 기업분석 체계를 구축했다.
- SQL(PostgreSQL) 기반 데이터 운영:
  - `fi_ra.company_fundamentals` 테이블에 적재하고 SQL로 row 수, ticker 수, 기준일을 검증해 정합성과 재현성을 확보했다.
  - 재부팅 이후에도 동일하게 재가동할 수 있도록 PostgreSQL 런북(`docs/manuals/RA_POSTGRES_REBOOT_RUNBOOK.md`)을 문서화했다.
- RA 제출형 결과물 자동화:
  - 분석 결과를 RA 스타일 Markdown/HTML/PDF로 변환해 현업·인사팀이 바로 확인 가능한 문서형 증빙으로 관리했다.
- 분석-의사결정-운용 흐름 연결:
  - 추천안을 목표가 기반 모의주문으로 연결하고 실행 로그/상태를 DB에 저장하며, 백테스트 테이블과 연계해 사후검증 기반을 만들었다.

자기소개서 문장 예시:

- “저는 EIMAS를 통해 거시-ETF-기업 데이터를 하나의 리서치 파이프라인으로 통합해, RA 업무의 핵심인 데이터 업데이트와 조사자료 작성 보조를 자동화했습니다.”
- “특히 매크로 레짐과 리스크 판단을 ETF 전략 언어로 연결하고, 기업 회계·밸류에이션 지표를 PostgreSQL에 적재/검증해 분석 근거를 SQL 기반으로 재현 가능하게 관리했습니다.”
- “단순 모델 결과 제시에 그치지 않고, RA 스타일 PDF 보고서와 DB 실행 로그까지 남겨 협업과 보고 효율을 동시에 높였습니다.”

핵심 키워드:

- 매크로 레짐 분석
- ETF 전략 보조
- 기업 재무/회계 지표
- 주식 밸류에이션
- PostgreSQL
- SQL 검증 쿼리
- 리서치 자동화
- RA 스타일 PDF 리포팅
- 의사결정 로그
- 백테스트 연계

### 2) RA 업무에 맞춰 추가 구현할 내용

- ETF 전략 심화:
  - 팩터 노출(가치/성장/퀄리티), 섹터 상대강도, 듀레이션/금리 민감도를 정량 분해해 ETF 선택 근거를 더 명확히 제시.
- 기업실적 모니터링 고도화:
  - EPS revision, 실적 서프라이즈, 가이던스 변화, 캘린더 이벤트를 시계열로 추적해 선제적 코멘트 작성 지원.
- SQL 데이터 거버넌스 확장:
  - `source`, `revision_tag`, `quality_flag`, `as_of_date` 표준화와 이력 테이블 분리로 데이터 품질/변경관리 체계 강화.
- 리포트 제출 실무형 강화:
  - PDF 외에 Excel/PPT 템플릿 자동 출력과 표준 목차(요약/시장/ETF/기업/리스크/결론) 고정으로 제출 즉시 활용 가능하게 개선.
- 사후평가 및 개선 루프 정례화:
  - 추천안별 수익률, MDD, hit ratio, 예측오차를 자동 집계해 월간 리뷰와 전략 개선 프로세스에 연결.
- 협업 기능 추가:
  - 코멘트 이력, 요청사항 트래킹, 버전 비교를 붙여 RA-애널리스트-유관부서 간 커뮤니케이션 비용을 줄이는 방향으로 확장.

## 10) RA-SQL 적용영역 및 파이프라인 매핑 (자소서용)

### A. SQL 사용 영역 요약

| 사용 영역 | 설명 | SQL 예시 기능 |
|---|---|---|
| 거시지표 분석 | FRED, OECD, 한국은행 등 매크로 데이터 주기적 수집/정규화 | UPSERT, Window Functions |
| ETF/섹터 분석 | ETF 보유 종목, 섹터 비중 추적 및 기간별 수익률 비교 | JOIN, GROUP BY, ROLLUP, CTE |
| 기업 분석 | 재무제표 추이, 밸류에이션 지표 계산, 모멘텀 트래킹 | CASE, AVG OVER, LAG/LEAD |
| 퀀트 전략 백테스트 | 기간별 전략별 수익률 저장 및 비교 | INSERT INTO, MERGE(또는 UPSERT), Analytic Functions |
| 리포트 자동화 | 정량 요약 테이블/차트 생성용 데이터 추출 | VIEW, Materialized View |
| RA 분석 증빙 | 보고서에 SQL 기반 테이블 및 그래프 삽입 | EXPORT, `audit_log` 테이블 |

### B. EIMAS Phase별 SQL 통합 전략

| 대상 Phase | 통합 전략 | 예시 |
|---|---|---|
| Phase1 | 매크로 + ETF + 기업 DB 통합 테이블 설계 | `macro_series`, `etf_snapshot`, `ra_company_fundamentals` 적재 |
| Phase2 | SQL 기반 스냅샷 비교 뷰 생성 | `valuation_snapshot_mv`, `momentum_rolling_avg` |
| Phase6 | SQL 기반 전략성과 비교/저장 | `ra_backtest_runs`에 전략명, 수익률, MDD, Sharpe 저장 |
| Phase7 | SQL 기반 시각화 대시보드 반영 | `allocation_report_agent` Section 6 SQL 증빙 노출 |
| Phase9 | SQL 결과/로그 아티팩트 export | `export_sql_artifacts()` + 분석 SQL 로그 저장 |

### C. 자기소개서 문장 (요청 톤)

- EIMAS 프로젝트에서는 미국·한국 증권사 RA 업무 흐름을 기준으로, ETF 전략·거시지표·기업분석을 SQL 기반으로 자동화했습니다.
- 특히 PostgreSQL 기반 `fi_ra.company_fundamentals`와 EIMAS 내부 SQL 테이블 `ra_company_fundamentals`를 함께 운영해, PER/PBR 모니터링과 리포트 자동화를 구현했습니다.
- 또한 백테스트 성과를 `ra_backtest_runs`에 구조화해 추천-성과를 추적 가능하게 만들었고, SQL 검증 결과를 RA 스타일 PDF에 표·그래프로 직접 증빙했습니다.
