# EIMAS — Economic Intelligence Multi-Agent System

**AI 투자위원회: 거시경제 데이터 수집부터 포트폴리오 결정까지 완전 자동화**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Claude](https://img.shields.io/badge/Claude-Sonnet-D97706)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 한 줄 요약

리서치 애널리스트가 수행하는 **거시경제 진단 → ETF/섹터 분석 → 기업 커버리지 → 포트폴리오 전략** 워크플로우를 7개 AI 에이전트가 토론 방식으로 자동화한 시스템.

---

## 실제 출력 예시

> `python main.py --full` 실행 결과 (2026-02-09, 소요시간 79초)

**시장 진단**
```
레짐:        BULL (Low Volatility)
리스크 점수: 2.5 / 100  (LOW)
Fed 순유동성: $5,694B  (Abundant)
최종 권고:   BULLISH  (신뢰도 51%)
```

**ETF 모멘텀 스냅샷 (20일 수익률 기준)**
| Rank | Ticker | 역할 | 20D 수익률 | 모멘텀 |
|---:|--------|------|----------:|--------|
| 1 | XLE | 에너지 섹터 | **+14.39%** | UPTREND |
| 2 | GLD | 실물자산 헤지 | **+10.69%** | UPTREND |
| 3 | XLI | 산업재 섹터 | **+8.12%** | UPTREND |
| 8 | QQQ | 나스닥100 | -1.74% | NEUTRAL |

**기업 밸류에이션 스캔**
| Ticker | Trailing P/E | P/B | 20D | 시그널 |
|--------|------------:|----:|----:|--------|
| MSFT | 25.1x | 7.6 | -16.1% | **UNDERVALUED** |
| NVDA | 45.9x | 37.9 | +0.2% | **OVERVALUED** |
| JPM | 16.1x | 2.5 | -2.2% | FAIR |

**백테스트 성과** (EIMAS AutoPaper 전략)
```
총 수익률: +7.67%   Sharpe: 1.18   Max Drawdown: -3.69%
```

→ [전체 샘플 리포트 보기](examples/sample-run/report.md)

---

## 시스템 아키텍처

```
입력: FRED API + Yahoo Finance + Crypto APIs + Perplexity 실시간 검색
  │
  ▼
Phase 1  │ 데이터 수집      ─ FRED(금리/유동성/물가) · 시장 · 크립토 · 한국
Phase 2  │ 정량 분석        ─ GMM 레짐 · LASSO 예측 · GARCH 변동성 · HRP 최적화
Phase 3  │ AI 에이전트 토론  ─ 7개 전문 에이전트 × 2모드 (Full 365d / Ref 90d)
Phase 4  │ 실시간 스트리밍   ─ VPIN · Kyle's λ · 미세구조 (--realtime)
Phase 4.5│ 운용 의사결정    ─ 토론 합의 → 규칙 기반 포지션 결정
Phase 5  │ 저장             ─ SQLite + PostgreSQL 이중 적재
Phase 7  │ AI 리포트 생성   ─ 섹션별 자동 서술 + 그림 캡션
Phase 8  │ Multi-LLM 검증   ─ Claude × GPT-4 × Gemini 가중 투표 (--full)
Phase 9  │ 아티팩트 Export  ─ JSON / Markdown / HTML / PDF
  │
  ▼
출력: 투자 권고 + ETF 전략 + 기업 밸류에이션 + SQL 증빙 + 백테스트 성과
```

**AI 에이전트 구성**

| 에이전트 | 역할 |
|---------|------|
| MetaOrchestrator | 토론 조율 및 합의 도출 |
| AnalysisAgent | Critical Path 리스크 분석 |
| ForecastAgent | LASSO 기반 금리/시장 예측 |
| ResearchAgent | Perplexity 실시간 뉴스 검색 |
| StrategyAgent | HRP 포트폴리오 전략 수립 |
| VerificationAgent | 결과 검증 및 Fact-check |
| InterpretationDebateAgent | 경제학파 관점 토론 (Keynesian / Austrian / Monetarist) |

---

## 계량 방법론

| 분야 | 방법론 | 학술 근거 |
|------|--------|---------|
| 레짐 탐지 | GMM 3-State (Bull/Neutral/Bear) | Hamilton (1989), Engel & Hamilton (1990) |
| 이자율 예측 | LASSO + Cross-validation | Tibshirani (1996), Zou & Hastie (2005) |
| 변동성 모델링 | GARCH(1,1) + Half-life | Bollerslev (1986) |
| 포트폴리오 최적화 | Graph-Clustered HRP | de Prado (2016) |
| 미세구조 분석 | VPIN + Kyle's Lambda | Easley et al. (2011), Kyle (1985) |
| 버블 탐지 | SADF / GSADF | Phillips et al. (2015) |
| 인과관계 분석 | Granger Causality + Causal Network | Granger (1969) |
| 신용 리스크 | HY OAS + ERP (Earnings Risk Premium) | Fama & French (2004) |

---

## 빠른 시작

```bash
git clone https://github.com/Eom-TaeJun/eimas.git
cd eimas
pip install -r requirements.txt

# API 키 설정 (.env)
cp .env.example .env
# ANTHROPIC_API_KEY=sk-ant-...  (필수)
# FRED_API_KEY=...              (필수)
# OPENAI_API_KEY=sk-...         (선택, Multi-LLM 검증)
# GOOGLE_API_KEY=...            (선택, Multi-LLM 검증)
# PERPLEXITY_API_KEY=pplx-...   (선택, 실시간 뉴스)

# 전체 분석 실행
python main.py --full

# 빠른 실행 (이전 full 결과 기반 운용 의사결정)
python main.py --short
```

**실행 모드 선택**

| 명령어 | 소요시간 | 설명 |
|--------|----------|------|
| `--full` | ~80초 | Phase 1~9 전체 (Multi-LLM 검증 포함) |
| `--short` | ~10초 | 이전 full 결과 활용 → 실시간 운용 결정 |
| `--quick1` / `--quick2` | ~20초 | KOSPI / SPX Quick AI 검증 |
| `--backtest` | ~30초 | 5년 전략 백테스트 |
| `--realtime -d 60` | 연속 | 실시간 스트리밍 (60초 주기) |

---

## 기술 스택

```
언어/런타임    Python 3.10+, Node.js 18+
AI            Anthropic Claude (Sonnet), OpenAI GPT-4, Google Gemini
데이터 소스   FRED API, Yahoo Finance, Binance WebSocket, Perplexity
계량 분석     NumPy · SciPy · scikit-learn · statsmodels · PyPortfolioOpt
DB            SQLite (개발) + PostgreSQL (운영) — 이중 어댑터
백엔드        FastAPI (REST API)
프론트엔드    Next.js 대시보드
출력 포맷     JSON / Markdown / HTML / PDF (wkhtmltopdf)
```

---

## 디렉토리 구조

```
eimas/
├── main.py                        # 단일 진입점
├── pipeline/
│   ├── app/orchestrator_steps.py  # 파이프라인 오케스트레이터
│   ├── phases/                    # Phase 1~9 구현체
│   └── schemas.py                 # EIMASResult 데이터 스키마
├── agents/                        # 7개 AI 에이전트
├── lib/                           # 계량 분석 모듈 (70+)
│   ├── bubble/                    # 버블 탐지 (SADF/GSADF)
│   ├── causality/                 # Granger 인과관계
│   ├── graph_portfolio/           # HRP 포트폴리오
│   ├── microstructure/            # VPIN, Kyle's Lambda
│   └── reports/                   # AI 리포트 생성
├── core/
│   ├── db_adapter.py              # SQLite/PostgreSQL 이중 어댑터
│   └── multi_llm_debate.py        # Multi-LLM 합의 엔진
├── api/main.py                    # FastAPI 서버
├── frontend/                      # Next.js 대시보드
├── examples/sample-run/           # 실제 실행 결과 샘플
│   ├── report.md                  # 전체 분석 리포트
│   └── figures/                   # 8개 시각화 차트
└── docs/
    └── ARCHITECTURE.md            # 상세 아키텍처 문서
```

---

## 샘플 출력물

| 파일 | 내용 |
|------|------|
| [examples/sample-run/report.md](examples/sample-run/report.md) | 전체 분석 리포트 (섹션 1~10) |
| [examples/sample-run/figures/](examples/sample-run/figures/) | 리스크 분해 · 밸류에이션 맵 · ETF 모멘텀 등 8개 차트 |

---

## 관련 문서

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — 상세 아키텍처 및 데이터 플로우
- [CONTRIBUTING.md](CONTRIBUTING.md) — 개발 가이드
- [command.md](command.md) — 전체 실행 옵션 레퍼런스

---

*Built by Eom TaeJun · Quantitative Finance × Multi-Agent AI*
