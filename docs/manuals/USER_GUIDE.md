# EIMAS 사용자 가이드

> Economic Intelligence Multi-Agent System (EIMAS) v2.1

## 목차
1. [시작하기](#1-시작하기)
2. [CLI 사용법](#2-cli-사용법)
3. [웹 인터페이스 사용법](#3-웹-인터페이스-사용법)
4. [주요 기능 설명](#4-주요-기능-설명)
5. [트러블슈팅](#5-트러블슈팅)

---

## 1. 시작하기

EIMAS는 CLI(명령줄 인터페이스)와 웹 대시보드 두 가지 방식으로 사용할 수 있습니다.

### 필수 조건
- Python 3.9+
- Node.js 18+ (웹 인터페이스용)
- API 키 설정 (`.env` 파일)

### 설치
```bash
# 의존성 설치
pip install -r requirements.txt

# 프론트엔드 의존성 설치 (웹 사용 시)
cd frontend
npm install
```

---

## 2. CLI 사용법

`cli/eimas.py` 스크립트를 통해 모든 기능을 제어할 수 있습니다.

### 통합 파이프라인 실행 (`run`)
가장 핵심적인 명령어로, 데이터 수집부터 분석, 보고서 생성까지 한 번에 수행합니다.

```bash
# 기본 실행 (표준 분석)
python cli/eimas.py run

# 빠른 분석 모드 (일부 심층 분석 생략, ~15초 소요)
python cli/eimas.py run --quick

# 전체 모드 (독립 스크립트 포함, 전체 분석)
python cli/eimas.py run --full

# 실시간 모니터링 포함 (예: 60초간 VPIN 모니터링)
python cli/eimas.py run --realtime --duration 60
```

### 개별 기능 실행

#### 시그널 관리
```bash
python cli/eimas.py signal list          # 최신 시그널 목록
python cli/eimas.py signal generate      # 시그널 즉시 생성
```

#### 포트폴리오 관리
```bash
python cli/eimas.py portfolio show       # 현재 포트폴리오 상태
python cli/eimas.py portfolio optimize   # 포트폴리오 최적화
```

#### 리스크 분석
```bash
python cli/eimas.py risk check           # 리스크 점검
```

#### 트레이딩 (Paper Trading)
```bash
python cli/eimas.py trade buy SPY 10     # SPY 10주 매수
python cli/eimas.py trade sell QQQ 5     # QQQ 5주 매도
```

#### 리포트 생성
```bash
python cli/eimas.py report daily         # 일일 리포트 생성
```

---

## 3. 웹 인터페이스 사용법

웹 인터페이스는 대시보드 확인 및 심층 리서치(Elicit) 기능을 제공합니다.

### 실행 방법 (터미널 2개 사용 권장)

**Terminal 1: 백엔드 API 서버**
```bash
# 프로젝트 루트에서 실행
uvicorn api.main:app --reload --port 8000
```
- API 문서: http://localhost:8000/docs

**Terminal 2: 프론트엔드**
```bash
cd frontend
npm run dev -- --port 3000
```
- 접속: http://localhost:3000
- Deep Research (Elicit): http://localhost:3000/elicit

### Deep Research (Elicit) 사용법
1. 브라우저에서 `http://localhost:3000/elicit` 접속
2. 입력창에 리서치 주제 입력 (예: "Analyze the impact of recent Fed comments on tech stocks.")
3. "Run Research" 버튼 클릭
4. AI 에이전트들의 토론 과정과 최종 결론, 반대 의견(Devil's Advocate) 확인

---

## 4. 주요 기능 설명

### 4.1 통합 파이프라인 (Integrated Pipeline)
`eimas run` 명령으로 실행되며 다음 단계를 거칩니다:
1. **데이터 수집**: FRED(거시경제), Market(주식/ETF), Crypto, RWA
2. **분석 (Analysis)**: 레짐 탐지, 유동성 분석, Critical Path 분석, 버블 감지
3. **토론 (Debate)**: 멀티 에이전트(Full vs Reference 모드) 토론 및 합의 도출
4. **모니터링 (Realtime)**: (옵션) Binance 실시간 VPIN 모니터링
5. **저장 및 보고**: DB 저장, JSON/MD 리포트 생성

### 4.2 실시간 모니터링
암호화폐 시장의 미세구조(VPIN)를 실시간으로 분석하여 급격한 변동성을 사전에 탐지합니다.

### 4.3 Elicit Deep Research
사용자의 자연어 질문에 대해 EIMAS의 분석 엔진을 가동하여 심층적인 답변과 근거를 제시합니다.

---

## 5. 트러블슈팅

### API 서버 실행 실패
- **포트 충돌**: `lsof -i :8000`으로 점유 중인 프로세스 확인 후 종료 (`kill -9 PID`)
- **경로 문제**: 프로젝트 루트에서 실행했는지 확인 (`ls api/main.py`가 보여야 함)

### 데이터 수집 오류
- **yfinance 에러**: 일시적인 API 제한일 수 있음. 잠시 후 재시도.
- **FRED 키**: `.env` 파일에 `FRED_API_KEY`가 올바른지 확인.

### 웹 페이지 로딩 실패
- 백엔드 서버가 켜져 있는지 확인 (`curl http://localhost:8000/api/health`)
- 프론트엔드 콘솔 로그 확인
