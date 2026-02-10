# US Trader Baseline (v1) - 2026-02-10

## 목적
- EIMAS를 "실거래 도움" 중심으로 재정렬한다.
- 미국 대형 트레이더 플랫폼 공통분모를 기준으로 기능 우선순위를 고정한다.
- 설명가능성(Explainability)을 1순위로 유지하되, 운영 안정성(Operational Stability)을 즉시 확보한다.

## 입력 조건 (사용자 합의)
- 목표: 실제 거래에 도움이 되는 시스템
- 자산군: 멀티에셋 + 온체인 포함
- 실행: 계속 수행 가능한 운영 형태
- 유지: AI 해석/개입
- 우선순위: 설명가능성 > 안정성
- 마감: 14일 내 v1 컷오버

## 실행 프로파일
- 신규 CLI 프로파일: `--profile us-trader-v1`
- 코드 위치:
  - `pipeline/app/profiles.py`
  - `pipeline/app/orchestrator_steps.py`
  - `main.py`
- 의도:
  - 실행/설명/리스크 가드레일 경로는 유지
  - 연구성 고비용 phase는 기본 비활성화

## 브로커 정책 (현재 구현)
- 기준 브로커: `IBKR` (현재는 paper backend 위의 IBKR 시뮬레이션 라우터)
- 구현 위치:
  - `lib/broker_execution.py`
  - `lib/auto_paper_execution.py`
  - `lib/trading_db.py` (`idempotency_key`, `order_state`, `explainability`, `broker`)
- 핵심:
  - 주문 idempotency 키로 중복 등록 방지
  - 주문 상태 머신(`created/submitted/filled/rejected/...`) 기록
  - 주문별 설명가능성 메타데이터(리스크/레짐/근거/승인요건) 저장
- 실행 환경 변수:
  - `EIMAS_EXECUTION_BROKER=ibkr`

## 멀티자산 주문정책 (현재 구현)
- 정책 버전: `us-trader-v1.1`
- 자산군 분류:
  - `us_equity`, `us_etf`, `us_bond_etf`, `us_commodity_etf`, `korea_equity`, `crypto_spot`
  - `index`, `futures`는 기본 비거래(tradable=false)
- 핵심 제약:
  - 자산군별 `min_notional`, `max_notional_pct`, `quantity_precision`, `allow_fractional`
  - 전역 `max_order_notional_pct` 캡과 자산군 캡 동시 적용
  - 자산군 비활성화 목록(`disabled_asset_classes`) 지원
- 설명가능성:
  - 주문 레코드에 `asset_class`, 요청/적용 notional, cap 적용 여부, 수량 정밀도 기록
- 실행 환경 변수:
  - `EIMAS_EXECUTION_MAX_ORDER_NOTIONAL_PCT=0.20`
  - `EIMAS_EXECUTION_DISABLED_ASSET_CLASSES=index,futures`

## Keep / Later / Remove (파일 단위)

### Keep (즉시 유지)
- `pipeline/phases/phase1_collect.py`
- `pipeline/phases/phase2_basic.py`
- `pipeline/phases/phase2_enhanced.py`
- `pipeline/phases/phase2_adjustment.py` (sentiment 유지, bubble은 profile에서 skip)
- `pipeline/phases/phase3_debate.py`
- `pipeline/phases/phase45_operational.py`
- `pipeline/phases/phase46_paper_execution.py`
- `pipeline/phases/phase5_storage.py`
- `pipeline/phases/phase7_report.py`
- `lib/operational/` (운영 제약/결정/리밸런싱)
- `lib/adapters/execution_backend.py`

### Later (v1 이후 단계적 활성화)
- `pipeline/phases/phase2_adjustment.py` 내 institutional frameworks
- `pipeline/phases/phase6_portfolio.py` 전체 (backtest/attribution/stress)
- `pipeline/phases/phase8_validation.py` (multi-LLM validation)
- `pipeline/phases/phase8_validation.py` (quick validation)
- `pipeline/phases/phase4_realtime.py` (연속 운용 모드 안정화 후 확대)

### Remove / Park (현재 목표에서 제외)
- 당장 실거래 운영과 무관한 "리서치 전용 확장" 기능
- 기준: 4주 내 체결품질/손실방지/운영안정성 개선이 불명확한 기능
- 주의: 코드 즉시 삭제보다 `profile skip + audit trail`로 먼저 격리

## v1 파이프라인 구조
1. Data Collect
2. Core Analysis (Regime/Risk/Allocation)
3. AI Interpretation & Intervention (Debate + Reasoning Chain)
4. Operational Guardrails (Constraint Repair, Failsafe, Approval)
5. Paper Execution Bridge
6. Storage + Explainable Report

## 리스크 한도 정책 (최근 금융 환경 대응형, v1 권고)
- 고정값 하드코딩보다 "시장 상태 기반 밴드" 우선
- 상태 구분:
  - High Vol: 보수적 (`turnover_cap` 축소, `max_single_asset_weight` 축소)
  - Normal Vol: 표준
  - Low Vol: 제한적 완화
- v1 기본 가드레일(권고 시작값):
  - `risk_score_high`: 65
  - `turnover_cap`: 0.20
  - `max_single_asset_weight`: 0.20
  - `crypto_max`: 0.15
- 실제 적용 전제:
  - 브로커/체결비용 구조 확정 후 파라미터 1차 재보정

## 14일 실행 계획

### Day 1-3
- `us-trader-v1` profile 고정 운영
- phase skip audit 로그 검증 (`audit_metadata.profile_skips`)
- paper execution 회귀 확인

### Day 4-7
- 브로커 우선순위 확정 (IBKR 1순위 권고)
- 주문 상태 일관성 점검 (submitted/partial/filled/canceled)
- 실패 재시도/중복방지 키 정책 고정
- 멀티자산 주문정책(v1.1) 적용 및 주문 설명 메타 확장

### Day 8-11
- 리스크 가드레일 파라미터 실측 튜닝
- 승인 게이트/킬스위치 운영 리허설
- 설명가능성 리포트 템플릿 정규화 (reasoning chain + decision rationale)

### Day 12-14
- 연속 실행(스케줄/daemon) 안정화
- 장애 복구 시나리오 테스트
- v1 컷오버 체크리스트 승인

## 남은 의사결정 (필수)
- 브로커 1순위: `IBKR` (확정)
- 체결 채널/주문유형 범위: 멀티자산 확장 기준으로 단계별 확정 필요

## 진행상황 스냅샷 (2026-02-10)
- 완료:
  - `--profile us-trader-v1` 런타임 정책 적용 + phase skip audit trail
  - IBKR-first 실행 라우터 + 주문 idempotency 적용
  - 실행 DB 확장(`broker/idempotency_key/order_state/explainability`)
  - 멀티자산 주문정책(v1.1) 및 설명 메타 확장
- 진행중:
  - 운영 리스크 파라미터 실측 튜닝
  - 연속 실행(스케줄/daemon) 장애 복구 검증

## 권장 실행 커맨드
```bash
python main.py --profile us-trader-v1
python main.py --profile us-trader-v1 --paper-auto --paper-account ra_auto
python main.py --profile us-trader-v1 --realtime -d 30
```
