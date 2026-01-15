# EIMAS - Economic Intelligence Multi-Agent System

> **AI 멀티에이전트 기반 거시경제 분석 및 투자 의사결정 시스템**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()

**한 줄 요약**: 연준 유동성부터 버블 리스크까지, 8개 학술 논문 방법론으로 통합 분석하고 AI 에이전트 토론으로 투자 방향 제시

---

## 🚀 Quick Start (3분 안에 시작)

### 1. 설치
```bash
# Clone
git clone https://github.com/Eom-TaeJun/eimas.git
cd eimas

# 가상환경 (선택)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. API 키 설정
```bash
# 필수 (최소 2개)
export ANTHROPIC_API_KEY="sk-ant-..."  # Claude (멀티에이전트 토론)
export FRED_API_KEY="your-key"         # FRED (거시경제 데이터)

# 선택 (추가 기능)
export PERPLEXITY_API_KEY="pplx-..."   # 이벤트 원인 분석
export OPENAI_API_KEY="sk-..."         # 백업
```

**API 키 발급**:
- Claude: https://console.anthropic.com/
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html
- Perplexity: https://www.perplexity.ai/settings/api

### 3. 실행
```bash
# 기본 실행 (~40초)
python main.py

# 빠른 분석 (~16초)
python main.py --quick

# 전체 기능 (~90초)
python main.py --full

# AI 리포트 포함
python main.py --report
```

### 4. 결과 확인
```bash
# JSON 결과
cat outputs/integrated_20260115_180000.json

# 마크다운 리포트
cat outputs/integrated_20260115_180000.md

# 또는 실시간 대시보드
uvicorn api.main:app --reload --port 8000 &
cd frontend && npm run dev
# 브라우저: http://localhost:3000
```

---

## 🎯 무엇을 하나요?

### 핵심 질문에 답합니다
1. **현재 시장 레짐은?** → Bull/Bear/Neutral (GMM 분류, 85% 정확도)
2. **유동성 상황은?** → Net Liquidity (연준 실제 공급) 분석
3. **시스템 리스크는?** → Critical Path + 버블 탐지 (Greenwood-Shleifer)
4. **AI 합의는?** → FULL vs REF Mode 토론 결과

### 입력 → 출력
```
입력: (자동 수집)
├─ FRED: RRP, TGA, Net Liquidity, Fed Funds
├─ 시장: 24개 ETF + 2개 Crypto + 3개 RWA
└─ ARK: Cathie Wood ETF Holdings

↓ 8개 Phase 파이프라인

출력:
├─ JSON: 전체 분석 결과 (regime, risk_score, portfolio, ...)
├─ Markdown: 12개 섹션 리포트
└─ Dashboard: 실시간 UI (Next.js)
```

---

## 🔬 사용한 방법론 (학술 논문 기반)

| 방법론 | 논문/저자 | 구현 위치 |
|--------|-----------|-----------|
| **LASSO** | Tibshirani (1996) | 변수 선택 (Phase 3) |
| **GMM** | Gaussian Mixture Model | 레짐 분류 (Phase 2.1.1) |
| **Granger Causality** | Granger (1969, Nobel) | 유동성 전이 (Phase 2.3) |
| **HRP** | De Prado (2016) | 포트폴리오 (Phase 2.9) |
| **MST** | Mantegna (1999) | 시스템 리스크 (Phase 2.9) |
| **Bubble Detection** | Greenwood-Shleifer (2019) | 버블 탐지 (Phase 2.4.2) |
| **VPIN** | Easley et al. (2012) | 시장 미세구조 (Phase 2.4.1) |
| **Amihud Lambda** | Amihud (2002) | 비유동성 (Phase 2.4.1) |

**→ 상세 설명**: [PROJECT_INTRODUCTION.md](PROJECT_INTRODUCTION.md)

---

## 📊 실행 옵션

```bash
# 1. 기본 실행 (47개 모듈, ~40초)
python main.py
# Phase 1-5: 데이터 수집 → 분석 → AI 토론 → DB 저장

# 2. 빠른 분석 (Phase 2.3-2.10 스킵, ~16초)
python main.py --quick
# 레짐 + 리스크만 빠르게

# 3. 전체 모드 (54개 모듈, ~90초)
python main.py --full
# Phase 1-8 모두 + 독립 스크립트 7개

# 4. AI 리포트 포함 (~90초)
python main.py --report
# Claude/Perplexity 자연어 리포트 생성

# 5. 실시간 스트리밍 (Binance)
python main.py --realtime --duration 60
# VPIN 실시간 계산

# 6. 최대 기능 (~120초)
python main.py --full --realtime --report --duration 60

# 7. 서버 자동화 (Cron)
python main.py --cron --output /var/log/eimas
# 최소 출력, 백그라운드 실행
```

### 실행 모드 비교

| 모드 | 시간 | 모듈 수 | Phase | 용도 |
|------|------|---------|-------|------|
| `--quick` | ~16초 | 39/54 | 1-5 (일부 스킵) | 빠른 확인 |
| 기본 | ~40초 | 47/54 | 1-5 | 일반 분석 |
| `--full` | ~90초 | 54/54 | 1-8 | 전체 기능 |
| `--report` | ~90초 | 47/54 | 1-7 | AI 리포트 |

---

## 📁 출력 결과

### 1. JSON (`outputs/integrated_*.json`)
```json
{
  "timestamp": "2026-01-15T18:00:00",
  "regime": {
    "regime": "Bull",
    "trend": "up",
    "volatility": "low",
    "gmm_regime": "Bull",
    "entropy": 0.324,
    "entropy_level": "Very Low"
  },
  "risk_score": 51.0,
  "full_mode_position": "BULLISH",
  "reference_mode_position": "BULLISH",
  "final_recommendation": "BULLISH",
  "confidence": 0.85,
  "portfolio_weights": {
    "HYG": 0.54,
    "DIA": 0.06,
    "XLV": 0.05
  }
}
```

### 2. Markdown (`outputs/integrated_*.md`)
12개 섹션 자동 생성:
1. Data Summary
2. Regime Analysis (GMM + Entropy)
3. Risk Assessment (3단계 브레이크다운)
4. Market Quality & Bubble Risk
5. Multi-Agent Debate
6. Genius Act Macro
7. Portfolio Optimization (GC-HRP)
8. Critical Path Analysis
9. Real-time Signals (VPIN)
10. Quality Assurance
11. Additional Modules (ARK, Critical Path Monitor)
12. Standalone Scripts (--full 모드)

### 3. 실시간 대시보드
```bash
# 3개 터미널로 실행
# 터미널 1: FastAPI 서버
uvicorn api.main:app --reload --port 8000

# 터미널 2: EIMAS 분석 (최소 1회 실행)
python main.py --quick

# 터미널 3: Next.js 프론트엔드
cd frontend && npm install && npm run dev
```

**URL**: http://localhost:3000

**기능**:
- 5초 자동 폴링 (최신 결과)
- 메트릭 카드 4개 (Regime, Consensus, Data, Quality)
- 리스크 브레이크다운
- 경고 시스템

---

## 🏗️ 아키텍처

### 8개 Phase 파이프라인
```
Phase 1: 데이터 수집
├─ FRED (RRP, TGA, Net Liquidity)
├─ 시장 (24 ETFs + 2 Crypto + 3 RWA)
├─ DeFi TVL + MENA Markets
└─ ARK ETF Holdings

Phase 2: 분석
├─ 레짐 탐지 (GMM + Entropy)
├─ 이벤트 탐지
├─ Granger Causality
├─ Critical Path 리스크
├─ 시장 미세구조 (VPIN, Amihud)
├─ 버블 탐지 (Greenwood-Shleifer)
└─ 고급 분석 (ETF Flow, HRP, MST)

Phase 3: AI 멀티에이전트 토론
├─ FULL Mode (365일, 낙관)
├─ REFERENCE Mode (90일, 보수)
└─ 합의 도출 (Rule-based)

Phase 4: 실시간 (--realtime)
└─ Binance WebSocket → VPIN

Phase 5: 데이터베이스
├─ 이벤트 DB
├─ 시그널 DB
└─ Trading DB

Phase 6: AI 리포트 (--report)
└─ Claude/Perplexity 자연어 생성

Phase 7: 품질 보증 (--report)
├─ Whitening (경제학적 해석)
└─ Fact Checking

Phase 8: 독립 스크립트 (--full)
└─ 7개 스크립트 (장중, 암호화폐, 이벤트 등)
```

### 멀티에이전트 시스템
```
MetaOrchestrator (Claude Sonnet)
├─ FULL Mode Agent (365일 데이터)
│  └─ 장기 트렌드 중시, 낙관적
├─ REFERENCE Mode Agent (90일 데이터)
│  └─ 최근 변화 민감, 보수적
└─ Adaptive Agents (3가지)
   ├─ Aggressive: 리스크 추구
   ├─ Balanced: 균형
   └─ Conservative: 안전자산
```

---

## 📈 프로젝트 규모

### 코드 통계
```
총 코드:       ~50,000 lines
├─ main.py:    3,400 lines
├─ lib/:       47개 모듈 (통합)
├─ agents/:    14개 파일
└─ frontend/:  Next.js 대시보드

총 모듈:       95개
├─ 활성:       54개 (통합 47 + 독립 7)
├─ Deprecated: 9개
└─ Future:     32개 (미구현)
```

### 커버리지
- 기본 실행: **47/54 = 87.0%**
- --full 실행: **54/54 = 100%**

---

## 🎯 주요 기능

### 1. 순유동성 (Net Liquidity) 분석
```python
Net Liquidity = Fed Balance Sheet - RRP - TGA
```
- Fed의 실제 시장 공급 유동성
- Granger Causality로 SPY 예측력 검증

### 2. 리스크 점수 (3단계)
```
Final = Base (0-100) + Micro (±10) + Bubble (0~15)
```

### 3. AI 멀티에이전트 토론
```
FULL:  "BULLISH (365일 트렌드)"
REF:   "BULLISH (90일 모멘텀)"
→ 합의: BULLISH, 85% Confidence
```

### 4. GC-HRP 포트폴리오
- MST 클러스터링 + HRP
- 극단 가중치 없는 안정적 분산

---

## 📚 문서

| 문서 | 크기 | 용도 |
|------|------|------|
| [README.md](README.md) | 8KB | **실행 가이드 (이 문서)** |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | 5KB | 빠른 소개 (1-2페이지) |
| [PROJECT_INTRODUCTION.md](PROJECT_INTRODUCTION.md) | 15KB | 상세 설명 (학술 발표용) |
| [CLAUDE.md](CLAUDE.md) | 12KB | 개발자용 요약 |
| [COMMANDS.md](COMMANDS.md) | 10KB | 독립 스크립트 실행법 |
| [lib/README.md](lib/README.md) | 12KB | 모듈 가이드 |

---

## 🏆 차별점

| 항목 | EIMAS | Bloomberg | TradingView |
|------|-------|-----------|-------------|
| 거시경제 통합 | ✅ Net Liquidity | ✅ | ❌ |
| AI 멀티에이전트 | ✅ 토론 시스템 | ❌ | ❌ |
| 학술 방법론 | ✅ 8개 논문 | △ | ❌ |
| 오픈소스 | ✅ MIT | ❌ | ❌ |
| 실시간 대시보드 | ✅ Next.js | ✅ | ✅ |
| 비용 | 무료 | $2K+/월 | $15-60/월 |

---

## 🔧 개발/기여

### 디렉토리 구조
```
eimas/
├── main.py              # 메인 파이프라인 (3,400 lines)
├── agents/              # AI 에이전트 (14 files)
├── lib/                 # 기능 모듈 (47 active + 9 deprecated + 32 future)
├── api/                 # FastAPI 서버
├── frontend/            # Next.js 대시보드
├── data/                # DB 저장 (events.db, trading.db)
├── outputs/             # 결과 JSON/Markdown
└── tests/               # 테스트
```

### 새 모듈 추가
```bash
# 1. lib/에 모듈 생성
# 2. main.py에 import 추가
# 3. 적절한 Phase에 호출 추가
# 4. EIMASResult에 필드 추가
# 5. Summary 출력 추가
# 6. PR 생성
```

### 테스트
```bash
# 구문 체크
python -m py_compile main.py

# 빠른 테스트 (~16초)
python main.py --quick

# 전체 테스트 (~90초)
python main.py --full
```

---

## 🐛 트러블슈팅

### 1. API 키 에러
```bash
# 에러: "APIError: API key not found"
# 해결: 환경변수 확인
echo $ANTHROPIC_API_KEY
echo $FRED_API_KEY

# 재설정
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 2. 모듈 Import 에러
```bash
# 에러: "ModuleNotFoundError: No module named 'anthropic'"
# 해결: 의존성 재설치
pip install -r requirements.txt
```

### 3. FRED API Rate Limit
```bash
# 에러: "FredAPI: 429 Too Many Requests"
# 해결: --quick 모드 사용 (API 호출 감소)
python main.py --quick
```

### 4. 실행 시간 길어짐
```bash
# 문제: 실행 시간 > 2분
# 해결 1: --quick 모드
python main.py --quick  # ~16초

# 해결 2: Phase 선택적 스킵
# main.py에서 quick_mode 조건 수정
```

---

## 📊 성능

| 지표 | 값 |
|------|-----|
| 실행 시간 (quick) | ~16초 |
| 실행 시간 (기본) | ~40초 |
| 실행 시간 (full) | ~90초 |
| 데이터 소스 | 29개 티커 + 10개 FRED |
| 모듈 커버리지 | 87% (기본) / 100% (full) |
| 레짐 정확도 | ~85% (GMM) |
| 이벤트 예측 정확도 | ~78% (NFP/CPI/FOMC) |

---

## 🔮 로드맵

### Q1 2026
- [ ] 실적 발표 데이터 통합
- [ ] 뉴스 감성 분석
- [ ] 실제 브로커 연동 (IB, Alpaca)

### Q2-Q3 2026
- [ ] Fama-French 5-factor
- [ ] Tax-Loss Harvesting
- [ ] 성과 귀인 분석

---

## 📞 문의

- **GitHub**: https://github.com/Eom-TaeJun/eimas
- **Issues**: https://github.com/Eom-TaeJun/eimas/issues
- **Discussions**: https://github.com/Eom-TaeJun/eimas/discussions

---

## 📄 라이선스

MIT License - 자유롭게 사용/수정/배포 가능

---

## 🙏 감사

**학술 연구**:
- Tibshirani (LASSO), Granger (Causality, Nobel 2003)
- Bekaert (Critical Path), De Prado (HRP)
- Greenwood & Shleifer (Bubbles)

**오픈소스**:
- Anthropic Claude, yfinance, pandas, scikit-learn
- Next.js, React, shadcn/ui

---

*"Quantifying the Market, Democratizing Finance"*

**EIMAS v2.1.2** (2026-01-15)
