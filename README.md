# EIMAS: Economic Intelligence Multi-Agent System

**AI-Native Macroeconomic Risk Analysis & Portfolio Strategy System**

EIMAS는 거시경제 데이터 수집부터 AI 에이전트 토론, 투자 추천까지 완전한 파이프라인을 제공하는 금융 리서치 시스템입니다.

[![Version](https://img.shields.io/badge/version-2.4.0-blue.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)]()

---

## 핵심 기능

- **Multi-Agent 투자위원회**: 7개 전문 AI 에이전트 (거시경제, LASSO, HRP, 학파 토론, 검증)
- **Multi-LLM 합의 엔진**: Claude + GPT-4 + Gemini 3-라운드 토론
- **정량 분석**: GMM 레짐 탐지, GARCH 변동성, VPIN 미세구조, HRP 포트폴리오
- **실시간 대응**: `--short` 모드로 full 결과 기반 실시간 운용 의사결정

---

## 빠른 시작

```bash
git clone https://github.com/Eom-TaeJun/eimas.git
cd eimas
pip install -r requirements.txt
cp .env.example .env  # API 키 설정
```

### 실행 모드

```bash
python main.py --full             # 시장환경 전체 분석 (Phase 1~9)
python main.py --short            # 실시간 대응 (full 결과 활용 → 운용 → DB 적재)
python main.py --quick1           # KOSPI AI 검증
python main.py --quick2           # SPX AI 검증
python main.py --short --paper-auto --paper-account ra_auto  # 모의주문 자동 실행
python main.py --full --profile us-trader-v1                 # US Trader 프로파일
python main.py --realtime -d 60                              # 실시간 스트리밍
```

> 모든 실행 옵션의 단일 기준: [`command.md`](./command.md)

---

## 프로젝트 구조

```
eimas/
├── main.py                 # 단일 진입점
├── pipeline/
│   ├── app/                # 오케스트레이터 (run_pipeline_phases, run_short_pipeline_phases)
│   ├── phases/             # Phase 1~9 구현
│   └── schemas.py          # EIMASResult
├── lib/                    # 기능 모듈 (패키지 + shim 구조)
│   ├── bubble/             # 버블 탐지
│   ├── event_framework/    # 이벤트 탐지
│   ├── reports/            # 리포트 생성 (AIReportGenerator, WhiteningEngine 등)
│   ├── microstructure/     # 미세구조 분석
│   ├── genius_act/         # Genius Act 매크로
│   ├── graph_portfolio/    # 그래프 포트폴리오
│   ├── causality/          # 인과관계 분석
│   ├── realtime_intelligence/  # 실시간 스트리밍
│   └── adapters/           # 외부 모듈 어댑터
├── agents/                 # AI 에이전트
├── api/                    # FastAPI 서버
├── frontend/               # Next.js 대시보드
└── outputs/                # 결과물 (JSON, MD, PDF)
```

---

## API 키 설정

```bash
ANTHROPIC_API_KEY="sk-ant-..."    # Claude (필수)
FRED_API_KEY="your-key"           # FRED 데이터 (필수)
OPENAI_API_KEY="sk-..."           # GPT-4 (선택)
GOOGLE_API_KEY="..."              # Gemini (선택)
PERPLEXITY_API_KEY="pplx-..."     # 실시간 검색 (선택)
```

---

## 문서

| 우선순위 | 문서 | 용도 |
|----------|------|------|
| **P0** | [`command.md`](./command.md) | 실행 명령 단일 기준 |
| **P1** | [`CLAUDE.md`](./CLAUDE.md) | AI 어시스턴트 참조 |
| **P2** | [`CURRENT_STATUS.md`](./CURRENT_STATUS.md) | 현재 상태 + 할 일 |
| **P2** | [`TODO.md`](./TODO.md) | 작업 트래킹 |

---

*Version 2.4.0 | Updated: 2026-02-19*
