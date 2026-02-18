# EIMAS — Economic Intelligence Multi-Agent System

> AI 어시스턴트가 프로젝트를 빠르게 파악하기 위한 핵심 참조 문서.
> **Version:** v2.4.0 | **Updated:** 2026-02-19

---

## 1. 프로젝트 정체성

**EIMAS**는 경제 도메인 기반 AI 에이전트 시스템입니다.

```
거시경제 데이터 수집 (FRED, yfinance, Crypto)
        ↓
정량 분석 (레짐 탐지, 리스크 점수, 포트폴리오)
        ↓
AI 에이전트 토론 → 합의 도출
        ↓
투자 추천 + 리포트 생성 (BULLISH / BEARISH / NEUTRAL)
```

---

## 2. 필수 작업 원칙

- 모든 작업 전 `command.md` 먼저 확인 (P0 문서)
- 실행 진입점은 `main.py` 단일 경로만 사용
- 새 기능은 `python main.py --abc` 형태로 귀속
- 변경 후 최소 검증: `python main.py --help`

---

## 3. 실행 모드

| 모드 | 명령 | 내용 |
|------|------|------|
| **`--short`** | `python main.py --short` | 실시간 대응 (경량 수집 → 운용 → DB 적재) |
| **`--full`** | `python main.py --full` | 시장환경 전체 분석 (Phase 1~9) |
| `--quick1/2` | `python main.py --quick1` | Quick AI 검증 (KOSPI/SPX) |
| `--profile` | `python main.py --profile us-trader-v1` | 프로파일 실행 |
| `--paper-auto` | `python main.py --paper-auto` | 모의주문 자동 실행 |
| `--backtest` | `python main.py --backtest` | 백테스트 (5년) |
| `--realtime` | `python main.py --realtime -d 60` | 실시간 스트리밍 |

### `--short` vs `--full`

```
--short: Phase 1(경량) → Phase 4(실시간) → Phase 4.5(운용) → Phase 4.6(모의주문) → Phase 5(DB)
         outputs/eimas_*.json 최신 full 결과를 자동 로드해 운용 의사결정에 활용

--full:  Phase 1~9 전체 (시장환경 분석 + AI 토론 + 리포트 + Multi-LLM 검증)
```

---

## 4. 파이프라인 구조

```
Phase 1: 데이터 수집 (FRED, Market, Crypto, Korea)
Phase 2: 정량 분석 (레짐, 리스크, 포트폴리오, ARK, ETF Flow)
Phase 3: AI 에이전트 토론 (Full 365d + Reference 90d)
Phase 4: 실시간 스트리밍 (--realtime 시)
Phase 4.5: 운용 의사결정
Phase 4.6: 모의주문 (--paper-auto 시)
Phase 5: 저장 (outputs/eimas_*.json)
Phase 7: AI 리포트 생성
Phase 8: Multi-LLM 검증 (--full 전용)
Phase 9: 아티팩트 Export
```

**핵심 파일:**

| 역할 | 파일 |
|------|------|
| 메인 오케스트레이터 | `main.py` |
| Full 파이프라인 | `pipeline/app/orchestrator_steps.py` → `run_pipeline_phases()` |
| Short 파이프라인 | `pipeline/app/orchestrator_steps.py` → `run_short_pipeline_phases()` |
| 데이터 스키마 | `pipeline/schemas.py` |
| 데이터 수집 | `pipeline/collectors.py` |
| Phase 구현체 | `pipeline/phases/phase*.py` |
| 분석 모듈 | `lib/` (패키지 + shim 구조) |
| API 서버 | `api/main.py` (FastAPI) |

---

## 5. 디렉토리 구조

```
eimas/
├── main.py                    # 단일 진입점
├── pipeline/
│   ├── app/                   # 오케스트레이터 (orchestrator_steps, runtime, profiles)
│   ├── phases/                # Phase 1~9 구현
│   ├── schemas.py             # EIMASResult
│   └── collectors.py
├── lib/                       # 기능 모듈
│   ├── bubble/                # 버블 탐지 패키지
│   ├── event_framework/       # 이벤트 탐지 패키지
│   ├── reports/               # 리포트 생성 패키지
│   ├── microstructure/        # 미세구조 분석 패키지
│   ├── genius_act/            # Genius Act 매크로 패키지
│   ├── graph_portfolio/       # 그래프 포트폴리오 패키지
│   ├── causality/             # 인과관계 분석 패키지
│   ├── adapters/              # 외부 모듈 어댑터
│   ├── operational/           # 운용 의사결정
│   └── *.py                   # 단일 모듈 (패키지 shim 포함)
├── agents/                    # AI 에이전트
├── api/                       # FastAPI 서버
├── outputs/                   # 결과물 (JSON, MD, PDF)
└── data/                      # DB (events.db, paper_trading.db 등)
```

> `lib/` 내 monolith 파일들은 동명 패키지의 shim으로 전환됨.
> 기존 `from lib.bubble_detector import BubbleDetector` 경로는 그대로 유지.

---

## 6. 핵심 경제학적 방법론

| 방법론 | 사용처 |
|--------|--------|
| GMM 3-State | 레짐 탐지 (Bull/Neutral/Bear) |
| Bekaert VIX 분해 | VIX = Uncertainty + Risk Appetite |
| Greenwood-Shleifer | 버블 탐지 (2년 100% run-up) |
| Amihud Lambda + VPIN | 미세구조 비유동성/독성 주문 |
| MST (Mantegna) + HRP | 포트폴리오 최적화 |
| Granger Causality | 유동성 인과관계 |
| Net Liquidity | Fed BS - RRP - TGA |

---

## 7. 개발자 가이드

### 새 기능 추가
1. `command.md` 확인 (P0)
2. 로직은 `pipeline/phases/*` 또는 `lib/*`에 구현
3. 진입점/플래그는 `main.py`에서만 정의
4. 문서 동기화: `command.md`, `CURRENT_STATUS.md`

### 변경 후 검증
```bash
python main.py --help
python -m compileall main.py pipeline/app
```

---

## 8. 참고 문서

| 우선순위 | 문서 | 용도 |
|----------|------|------|
| **P0** | `command.md` | 실행 명령 단일 기준 |
| **P1** | `README.md` | 프로젝트 소개 |
| **P2** | `CURRENT_STATUS.md` | 현재 상태 + 할 일 |

*Updated: 2026-02-19 KST*
