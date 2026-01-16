# EIMAS 외부 도구 통합 가이드

> **목적**: v0, Elicit 등 최신 AI 도구를 EIMAS 워크플로우에 통합하여 생산성과 품질 향상

---

## 1. 도구 개요

### 1.1 v0 (v0.dev)

**개발사**: Vercel
**용도**: 텍스트 → React/Next.js UI 컴포넌트 자동 생성
**강점**:
- Tailwind CSS 기반 모던 디자인
- shadcn/ui 컴포넌트 라이브러리 활용
- 실시간 프리뷰 및 코드 내보내기

**EIMAS 적용 가능 영역**:
- 대시보드 UI 컴포넌트 생성
- 차트/테이블 시각화 개선
- 반응형 레이아웃 설계

### 1.2 Elicit

**개발사**: Ought
**용도**: AI 기반 학술 논문 검색 및 분석
**강점**:
- 연구 질문에 맞는 논문 자동 검색
- 논문 요약 및 핵심 발견 추출
- 메타 분석 지원

**EIMAS 적용 가능 영역**:
- 경제학 방법론 검증
- 최신 연구 트렌드 반영
- ResearchAgent 지식 베이스 확장

---

## 2. v0 통합 전략

### 2.1 현재 대시보드 문제점

| 문제 | 현재 상태 | 개선 방향 |
|------|----------|----------|
| 정적 HTML | `lib/dashboard_generator.py`가 문자열 연결 | React 컴포넌트화 |
| 스타일 일관성 | 인라인 CSS 혼재 | Tailwind CSS 통일 |
| 인터랙션 부족 | Chart.js만 사용 | 필터/드릴다운 추가 |
| 모바일 대응 | 제한적 | 완전 반응형 |

### 2.2 v0 활용 워크플로우

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 컴포넌트 명세 작성                                   │
│  - 기능 요구사항 정의                                        │
│  - 데이터 스키마 명시                                        │
│  - 스타일 가이드라인 제공                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: v0에 프롬프트 입력                                   │
│  - 상세한 UI 설명                                            │
│  - 예시 데이터 포함                                          │
│  - 다크 테마 명시                                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 생성된 코드 통합                                     │
│  - Next.js 프로젝트에 추가                                   │
│  - EIMAS API 연결                                            │
│  - 테스트 및 배포                                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 v0 프롬프트 템플릿

#### 2.3.1 LASSO 결과 대시보드

```
Create a dark-themed financial dashboard component for LASSO regression results.

Data structure:
{
  "results": [
    {
      "horizon": "VeryShort",
      "r_squared": 0.76,
      "n_observations": 131,
      "coefficients": {
        "d_Baa_Yield": 1.41,
        "Ret_Dollar_Idx": 1.03,
        "d_Spread_Baa": -0.76
      }
    },
    // Short, Long horizons...
  ]
}

Requirements:
1. Summary cards showing R², observations for each horizon
2. Horizontal bar chart for top 10 coefficients (positive=green, negative=red)
3. Collapsible table with all selected variables
4. Dark theme (#1a1a2e background, #16213e cards)
5. Responsive layout (mobile-first)
6. Use shadcn/ui components and recharts for visualization
```

#### 2.3.2 멀티에이전트 토론 뷰

```
Create a multi-agent debate visualization component.

Data structure:
{
  "opinions": [
    {
      "agent_role": "analysis",
      "topic": "market_outlook",
      "position": "BEARISH",
      "confidence": 0.75,
      "evidence": ["VIX elevated", "Credit spreads widening"]
    },
    // more opinions...
  ],
  "consensus": {
    "final_position": "CAUTIOUS",
    "agreement_score": 0.78
  }
}

Requirements:
1. Agent cards in a horizontal flex layout
2. Color-coded positions (bullish=green, bearish=red, neutral=amber)
3. Confidence meter for each agent
4. Expandable evidence list
5. Consensus summary bar at bottom
6. Animated transitions when data updates
7. Dark theme matching existing dashboard
```

#### 2.3.3 실시간 레짐 모니터

```
Create a real-time market regime monitoring component.

Features:
1. Large regime indicator (BULL/BEAR/TRANSITION/CRISIS)
2. Transition probability gauge (0-100%)
3. Historical regime timeline (last 30 days)
4. Key metrics cards (VIX, Credit Spread, Dollar Index)
5. Auto-refresh every 60 seconds
6. WebSocket-ready architecture
7. Dark theme with subtle animations
```

### 2.4 Next.js 프로젝트 구조 (제안)

```
eimas-dashboard/
├── app/
│   ├── page.tsx              # 메인 대시보드
│   ├── api/
│   │   ├── forecast/route.ts # LASSO 결과 API
│   │   ├── debate/route.ts   # 에이전트 토론 API
│   │   └── regime/route.ts   # 레짐 데이터 API
│   └── layout.tsx
├── components/
│   ├── lasso-dashboard.tsx   # v0 생성
│   ├── agent-debate.tsx      # v0 생성
│   ├── regime-monitor.tsx    # v0 생성
│   └── ui/                   # shadcn/ui
├── lib/
│   └── eimas-client.ts       # Python 백엔드 연결
└── styles/
    └── globals.css           # Tailwind 설정
```

### 2.5 Python 백엔드 연결

```python
# eimas/api/server.py (FastAPI)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js
    allow_methods=["*"],
)

@app.get("/api/forecast")
async def get_forecast():
    """LASSO 예측 결과 반환"""
    from agents.forecast_agent import ForecastAgent
    agent = ForecastAgent()
    result = await agent.execute(request)
    return result

@app.get("/api/debate")
async def get_debate():
    """에이전트 토론 결과 반환"""
    from agents.orchestrator import MetaOrchestrator
    orchestrator = MetaOrchestrator()
    result = await orchestrator.run_with_debate(query, market_data)
    return result['debate']
```

---

## 3. Elicit 통합 전략

### 3.1 활용 시나리오

#### 3.1.1 방법론 검증

```
연구 질문: "LASSO regression for macroeconomic forecasting effectiveness"

Elicit 활용:
1. 관련 논문 검색 (2020-2024)
2. 핵심 발견 추출:
   - 어떤 경제 변수가 가장 예측력 높은가?
   - 최적 정규화 파라미터 범위는?
   - Out-of-sample R² 벤치마크는?
3. EIMAS 결과와 비교 검증
```

#### 3.1.2 새로운 변수 발굴

```
연구 질문: "Novel predictors for Federal Reserve policy decisions"

Elicit 활용:
1. 최신 논문에서 사용된 예측 변수 수집
2. 현재 EIMAS에 없는 변수 식별:
   - 텍스트 기반 불확실성 지수?
   - 대안적 인플레이션 기대 측정치?
   - 글로벌 금융 조건 지수?
3. 데이터 소스 및 구현 방법 조사
```

#### 3.1.3 ResearchAgent 지식 베이스

```
┌─────────────────────────────────────────────────────────────┐
│  Elicit → 논문 수집 → 요약 추출 → ResearchAgent 컨텍스트     │
└─────────────────────────────────────────────────────────────┘

구현 방안:
1. Elicit API (비공식) 또는 수동 내보내기
2. 논문 요약을 JSON/Markdown으로 저장
3. ResearchAgent가 의견 형성 시 참조

예시 지식 베이스:
{
  "topic": "fed_policy_predictors",
  "papers": [
    {
      "title": "Forecasting Federal Funds Rate...",
      "authors": ["Smith", "Jones"],
      "year": 2023,
      "key_findings": [
        "Credit spreads outperform yield curve as predictor",
        "Optimal LASSO λ in range [0.01, 0.1]"
      ],
      "relevance_score": 0.92
    }
  ]
}
```

### 3.2 Elicit 세션 템플릿

#### 3.2.1 LASSO 방법론 검증

```
1. Elicit 접속: https://elicit.com

2. 검색 쿼리:
   "LASSO variable selection macroeconomic forecasting
    Federal Reserve interest rate prediction"

3. 필터:
   - 연도: 2018-2024
   - 인용수: 50+
   - 분야: Economics, Finance

4. 추출할 정보:
   - 사용된 종속변수
   - 선택된 주요 설명변수
   - Out-of-sample 성능 지표
   - 샘플 기간 및 데이터 빈도

5. 결과 정리:
   | 논문 | 종속변수 | 핵심변수 | R² |
   |------|---------|---------|-----|
```

#### 3.2.2 신용 스프레드 예측력 조사

```
검색 쿼리:
"credit spread prediction monetary policy
 Baa corporate bond yield forecasting"

추출 목표:
1. 어떤 신용 스프레드 측정치가 가장 효과적인가?
2. 리드-래그 관계는 어떠한가?
3. 레짐 의존성이 있는가?
```

### 3.3 지식 통합 파이프라인

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Elicit     │ ──→ │  knowledge/  │ ──→ │ ResearchAgent│
│   (수동/API) │     │  papers.json │     │   (자동참조)  │
└──────────────┘     └──────────────┘     └──────────────┘

파일 구조:
eimas/
└── knowledge/
    ├── papers.json           # Elicit 추출 논문 요약
    ├── methodologies.json    # 검증된 방법론
    └── benchmarks.json       # 성능 벤치마크
```

---

## 4. 통합 아키텍처 (전체)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           EIMAS v2.0 Architecture                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   Elicit    │    │  Perplexity │    │   Claude    │                  │
│  │  (Research) │    │   (News)    │    │  (Analysis) │                  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                  │
│         │                  │                  │                          │
│         ▼                  ▼                  ▼                          │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                    Multi-Agent Core (Python)                  │       │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │       │
│  │  │ Research   │ │ Analysis   │ │ Forecast   │ │ Strategy   │ │       │
│  │  │ Agent      │ │ Agent      │ │ Agent      │ │ Agent      │ │       │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────┘ │       │
│  │                        │                                      │       │
│  │                        ▼                                      │       │
│  │              ┌──────────────────┐                            │       │
│  │              │ MetaOrchestrator │                            │       │
│  │              └────────┬─────────┘                            │       │
│  └───────────────────────┼──────────────────────────────────────┘       │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                    FastAPI Backend                            │       │
│  │              /api/forecast  /api/debate  /api/regime          │       │
│  └───────────────────────┬──────────────────────────────────────┘       │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │                    Next.js Frontend                           │       │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐                │       │
│  │  │ LASSO      │ │ Agent      │ │ Regime     │    v0 생성     │       │
│  │  │ Dashboard  │ │ Debate     │ │ Monitor    │    컴포넌트    │       │
│  │  └────────────┘ └────────────┘ └────────────┘                │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 구현 로드맵

### Phase 1: v0 기반 UI 개선 (1주)

```
[ ] Next.js 프로젝트 초기화
[ ] v0로 LASSO 대시보드 컴포넌트 생성
[ ] v0로 에이전트 토론 뷰 생성
[ ] FastAPI 백엔드 설정
[ ] 프론트-백 연결 테스트
```

### Phase 2: Elicit 지식 통합 (1주)

```
[ ] LASSO 방법론 관련 논문 10편 수집
[ ] 핵심 발견 JSON으로 정리
[ ] ResearchAgent 지식 베이스 연결
[ ] 벤치마크 비교 기능 추가
```

### Phase 3: 실시간 기능 (2주)

```
[ ] WebSocket 기반 실시간 업데이트
[ ] 스케줄러 기반 자동 분석
[ ] 알림 시스템 (Slack/Discord)
[ ] 배포 (Vercel + Railway)
```

---

## 6. 추가 도구 후보

| 도구 | 용도 | 통합 우선순위 |
|------|------|--------------|
| **Cursor** | AI 코드 편집 | 이미 사용 중 |
| **Gemini** | 멀티모달 분석 | 높음 |
| **Anthropic Claude** | 분석/요약 | 이미 사용 중 |
| **Perplexity** | 실시간 검색 | 높음 |
| **Replit Agent** | 빠른 프로토타이핑 | 중간 |
| **Windsurf** | 코드 생성 | 낮음 |
| **Devin** | 자율 코딩 | 실험적 |

---

## 7. 참고 자료

- v0: https://v0.dev
- Elicit: https://elicit.com
- shadcn/ui: https://ui.shadcn.com
- Vercel AI SDK: https://sdk.vercel.ai
- FastAPI: https://fastapi.tiangolo.com

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-12-25 | 초기 문서 작성 |
