# EIMAS 웹 애플리케이션 가이드

**최종 업데이트**: 2026-01-11
**상태**: 프론트엔드 생성 완료, 실행 준비됨

---

## 1. 프로젝트 개요

EIMAS (Economic Intelligence Multi-Agent System)는 거시경제 데이터와 시장 데이터를 수집하여 AI 멀티에이전트 토론을 통해 시장 전망과 투자 권고를 생성하는 시스템입니다.

### 아키텍처

```
[프론트엔드]              [백엔드]               [데이터]
Next.js 14          -->  FastAPI 8000     -->  FRED API
React + Tailwind         Python 3.10+          yfinance
shadcn/ui                Claude API            SQLite DB
SWR                      Perplexity API
```

### 디렉토리 구조

```
eimas/
  frontend/          # Next.js 프론트엔드 (v0로 생성)
  api/               # FastAPI 백엔드
  lib/               # Python 분석 모듈
  agents/            # AI 에이전트
  core/              # 핵심 프레임워크
  data/              # SQLite 데이터베이스
  outputs/           # 분석 결과 JSON
  prompts/           # v0 프롬프트 파일
```

---

## 2. 상황별 가이드

### 상황 A: 처음 실행하는 경우

**필요 조건**
- Python 3.10+
- Node.js 18+
- API 키: ANTHROPIC_API_KEY, FRED_API_KEY

**실행 순서**

1. 백엔드 의존성 설치
```bash
cd /home/tj/projects/autoai/eimas
pip install -r requirements.txt
```

2. 프론트엔드 의존성 설치
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

3. 실행 (터미널 2개)
```bash
# Terminal 1: 백엔드
uvicorn api.main:app --reload --port 8000

# Terminal 2: 프론트엔드
cd frontend && npm run dev
```

4. 접속
- 프론트엔드: http://localhost:3000
- 백엔드 API: http://localhost:8000/docs

**관련 파일**
- `requirements.txt` - Python 의존성
- `frontend/package.json` - Node.js 의존성
- `RUN.md` - 상세 실행 가이드
- `run_all.sh` - 자동 실행 스크립트

---

### 상황 B: 프론트엔드만 수정하는 경우

**작업 디렉토리**: `frontend/`

**파일 구조**
```
frontend/
  app/
    page.tsx              # Dashboard (메인)
    analysis/page.tsx     # Analysis 페이지
    portfolio/page.tsx    # Portfolio 페이지
    risk/page.tsx         # Risk 페이지
    settings/page.tsx     # Settings 페이지
    layout.tsx            # 공통 레이아웃
  components/
    Navbar.tsx            # 네비게이션 바
    MetricsGrid.tsx       # 대시보드 메트릭 카드
    SignalsTable.tsx      # 시그널 테이블
    RealTimeClock.tsx     # 실시간 시계
    analysis/             # Analysis 관련 컴포넌트
    risk/                 # Risk 관련 컴포넌트
    ui/                   # shadcn/ui 컴포넌트
  lib/
    api.ts                # API 클라이언트
    types.ts              # TypeScript 타입 정의
```

**수정 후 확인**
```bash
cd frontend
npm run dev
# http://localhost:3000 에서 확인
```

---

### 상황 C: 백엔드 API 추가하는 경우

**작업 디렉토리**: `api/`

**현재 구현된 라우터**
```
api/routes/
  health.py       # GET /health
  analysis.py     # POST /analyze, GET /analyze/{id}
  regime.py       # GET /api/regime
  debate.py       # POST /debate
  report.py       # POST /report
```

**추가 필요한 라우터**
```
api/routes/
  signals.py      # GET /api/signals
  portfolio.py    # GET /api/portfolio, POST /api/paper-trade
  risk.py         # GET /api/risk, GET /api/correlation
  optimization.py # POST /api/optimize
  sectors.py      # GET /api/sectors
```

**라우터 추가 방법**
1. `api/routes/` 에 새 파일 생성
2. `api/server.py` 에 라우터 등록
3. 프론트엔드 `lib/api.ts` 에 API 함수 추가

**관련 라이브러리**
- `lib/paper_trading.py` - 페이퍼 트레이딩
- `lib/portfolio_optimizer.py` - 포트폴리오 최적화
- `lib/etf_flow_analyzer.py` - 섹터 분석

---

### 상황 D: v0로 새 컴포넌트 생성하는 경우

**사용 스크립트**: `generate_step.js`

**프롬프트 작성 규칙**
- 2000자 이하 권장 (타임아웃 방지)
- 명확한 컴포넌트 구조 명시
- API 스키마 포함
- 기술 스택 명시 (Next.js 14, shadcn/ui 등)

**새 프롬프트 생성**
```bash
# 1. 프롬프트 파일 생성
cat > prompts/step6_newpage.md << 'EOF'
# Step 6: New Page
Create a new page for...
EOF

# 2. v0 API로 생성
node generate_step.js 6

# 3. 결과 확인
ls frontend_steps/step6/
```

**관련 파일**
- `generate_step.js` - 단계별 생성 스크립트
- `prompts/` - 프롬프트 파일들
- `frontend_steps/` - 생성 결과

---

### 상황 E: v0 API 문제 해결

**문제 1: Unauthorized 에러**
```
원인: API 키 만료 또는 무효
해결:
1. https://v0.app/settings 접속
2. API 키 재발급
3. .bashrc 업데이트
   export V0_API_KEY="새로운키"
4. source ~/.bashrc
```

**문제 2: 타임아웃 (5분 초과)**
```
원인: 프롬프트가 너무 큼 (3000자 이상)
해결:
1. 프롬프트를 2000자 이하로 분할
2. 단계별로 생성 (generate_step.js 사용)
```

**문제 3: fetch failed**
```
원인: WSL 네트워크 또는 SDK 문제
해결:
1. 직접 fetch 사용 (SDK 대신)
2. 타임아웃 5분으로 설정
3. generate_step.js 사용 (이미 적용됨)
```

**API 테스트**
```bash
# 모델 목록 조회 (인증 불필요)
curl -s https://api.v0.dev/v1/models | jq

# 간단한 생성 테스트
curl -s -X POST https://api.v0.dev/v1/chats \
  -H "Authorization: Bearer $V0_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","privacy":"private","message":"Create a blue button"}' | jq -r '.id'
```

---

## 3. 파일 목록

### 실행 관련
| 파일 | 용도 |
|------|------|
| `run_all.sh` | 백엔드+프론트엔드 동시 실행 |
| `RUN.md` | 상세 실행 가이드 |

### v0 생성 관련
| 파일 | 용도 |
|------|------|
| `generate_step.js` | 단계별 프론트엔드 생성 (핵심) |
| `merge_frontend.sh` | 생성된 파일 병합 |
| `prompts/step1_dashboard.md` | Dashboard 프롬프트 |
| `prompts/step2_analysis.md` | Analysis 프롬프트 |
| `prompts/step3_portfolio.md` | Portfolio 프롬프트 |
| `prompts/step4_risk.md` | Risk 프롬프트 |
| `prompts/step5_settings.md` | Settings 프롬프트 |

### 백업 (미사용)
| 파일 | 설명 |
|------|------|
| `generate_with_v0.js` | 전체 생성 (타임아웃 문제) |
| `generate_with_v0.sh` | curl 버전 (타임아웃 문제) |
| `generate_with_v0.mjs` | ES Module (미사용) |
| `v0_prompt.md` | 원본 전체 프롬프트 (8134자) |

### 생성 결과
| 디렉토리 | 내용 |
|----------|------|
| `frontend/` | 병합된 최종 프론트엔드 |
| `frontend_steps/step1~5/` | 단계별 원본 파일 |

---

## 4. 환경 정보

### API 키 (환경변수)
```bash
# 필수
ANTHROPIC_API_KEY    # Claude API
FRED_API_KEY         # FRED 경제 데이터

# 선택
PERPLEXITY_API_KEY   # Perplexity 리서치
OPENAI_API_KEY       # OpenAI
V0_API_KEY           # v0 프론트엔드 생성
```

### 버전
- Python: 3.10+
- Node.js: v24.11.0
- v0-sdk: 0.15.3
- Next.js: 14+

### 포트
- 프론트엔드: 3000
- 백엔드: 8000

---

## 5. v0 생성 이력

| Step | 페이지 | 프롬프트 | 파일수 | Chat ID |
|------|--------|---------|--------|---------|
| 1 | Dashboard | 2021 chars | 9 | iiPxesJkBo3 |
| 2 | Analysis | 1784 chars | 9 | tNne4JiQfwu |
| 3 | Portfolio | 1721 chars | 16 | n9syVkByeq9 |
| 4 | Risk | 1527 chars | 8 | puZCXkaUx6o |
| 5 | Settings | 848 chars | 4 | b7tBdTM9VyP |

**v0.app에서 확인**: https://v0.app/chat/{Chat ID}
