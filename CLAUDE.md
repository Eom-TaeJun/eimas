# EIMAS - Economic Intelligence Multi-Agent System

> Claude Code가 프로젝트를 빠르게 파악하기 위한 요약 문서입니다.

**Version:** v2.2.4 (2026-02-05)
**Last Update:** 2026-02-05 01:10 KST

---

## 1. 프로젝트 개요

### 무엇인가요?

**EIMAS**는 거시경제 + 시장 데이터를 수집하고 **AI 멀티에이전트가 토론**하여 투자 권고를 생성하는 종합 퀀트 분석 시스템입니다.

**핵심 기능:**
1. **데이터 수집**: FRED(연준), yfinance(시장), 크립토/RWA 자산
2. **레짐 탐지**: Bull/Bear/Neutral 시장 상태 판단 (GMM 3-state)
3. **리스크 분석**: 유동성, 버블, 시장 미세구조 등 다차원 평가
4. **AI 토론**: Claude 기반 에이전트들의 관점 토론 → 합의 도출
5. **권고 생성**: BULLISH/BEARISH/NEUTRAL + 신뢰도 + 리스크 레벨

### 누가 사용하나요?

- 거시경제 기반 투자 의사결정이 필요한 개인/기관 투자자
- 정량적 시장 분석을 자동화하려는 퀀트 리서처
- AI 멀티에이전트 시스템을 연구하는 개발자

---

## 2. Quick Start (5분 시작)

### Step 1: 환경 설정

```bash
# 1. 의존성 설치
cd eimas
pip install -r requirements.txt

# 2. API 키 설정 (.env 파일 또는 환경변수)
export ANTHROPIC_API_KEY="sk-ant-..."      # Claude (필수)
export FRED_API_KEY="your-fred-key"        # FRED 데이터 (필수)
export PERPLEXITY_API_KEY="pplx-..."       # Perplexity (선택)

# 3. API 키 검증
python -c "from core.config import APIConfig; print(APIConfig.validate())"
```

### Step 2: 첫 실행

```bash
# 빠른 분석 (30초)
python main.py --quick

# 예상 출력:
# 📊 DATA: FRED RRP=$5B, Net Liq=$5799B, Market 24 tickers
# 📈 REGIME: Bull (Low Vol), Risk 45.2/100
# 🤖 DEBATE: FULL=BULLISH, REF=BULLISH (Agree ✓)
# 🎯 FINAL: BULLISH, Confidence 65%, Risk MEDIUM
```

### Step 3: 결과 확인

```bash
# JSON 결과
ls -la outputs/eimas_*.json | tail -1

# 마크다운 리포트
cat outputs/eimas_*.md | tail -1
```

### Step 4: 실시간 대시보드 (선택)

```bash
# 터미널 1: FastAPI 서버
uvicorn api.main:app --reload --port 8000

# 터미널 2: EIMAS 분석 (최소 1회)
python main.py --quick

# 터미널 3: 프론트엔드
cd frontend && npm install && npm run dev
# 브라우저: http://localhost:3000 (5초 자동 폴링)
```

---

## 3. 실행 모드

### 모드 비교표

| 모드 | 명령어 | 시간 | 비용 | 용도 |
|------|--------|------|------|------|
| **Short** | `python main.py --short` | 30초 | $0 | 빠른 데이터 확인 |
| **기본** | `python main.py` | 3-5분 | $0.05 | 일반 분석 + AI 리포트 |
| **Full** | `python main.py --full` | 8-10분 | $0.15 | Multi-LLM 검증 포함 |
| **Quick1** | `python main.py --quick1` | 4분 | $0.08 | KOSPI 전용 AI 검증 |
| **Quick2** | `python main.py --quick2` | 4분 | $0.08 | SPX 전용 AI 검증 |

### CLI 옵션

```bash
# 기본 실행
python main.py                    # 기본 모드
python main.py --short            # Short 모드 (버블/DTW 제외)
python main.py --quick            # --short와 동일 (alias)
python main.py --full             # Full 모드 (Multi-LLM 검증)

# Quick Mode AI Validation (2026-02-04)
python main.py --quick1           # KOSPI 전용 AI 검증
python main.py --quick2           # SPX 전용 AI 검증

# 추가 옵션
python main.py --realtime         # 실시간 스트리밍
python main.py --realtime -d 60   # 60초 스트리밍
python main.py --backtest         # 백테스팅 (5년 히스토리)
python main.py --attribution      # 성과 귀속 분석
python main.py --stress-test      # 스트레스 테스트

# 조합 예시
python main.py --full --realtime          # 전체 + 실시간
python main.py --quick2 --backtest        # SPX 검증 + 백테스트
```

### Phase별 실행 여부

```
                           기본   --short  --full  --quick1  --quick2
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Data Collection    ✅      ✅       ✅       ✅        ✅   │
│ Phase 2: Basic Analysis     ✅      ✅       ✅       ✅        ✅   │
│ Phase 2: Enhanced Analysis  ✅      ❌       ✅       ✅        ✅   │
│ Phase 3: AI Debate          ✅      ✅       ✅       ✅        ✅   │
│ Phase 7: AI Report          ✅      ❌       ✅       ✅        ✅   │
│ Phase 8: Multi-LLM Valid.   ❌      ❌       ✅       ❌        ❌   │
│ Phase 8.5: Quick Valid.     ❌      ❌       ❌       ✅        ✅   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 파이프라인 구조

### 간략 흐름도

```
Phase 1: 데이터 수집
  ├─ FRED (RRP, TGA, Net Liquidity, Fed Funds)
  ├─ Market (SPY, QQQ, TLT, GLD 등 24개)
  ├─ Crypto/RWA (BTC, ETH, USDC, ONDO, PAXG)
  └─ Korea (KOSPI, KOSDAQ, 삼성전자, SK하이닉스)

Phase 2: 분석
  ├─ 2.1 Regime Detection (GMM 3-state)
  ├─ 2.2 Event Detection (금리 변동, 뉴스)
  ├─ 2.4 Critical Path → Base Risk (0-100)
  ├─ 2.4.1 Microstructure → Liquidity Adjustment (±10)
  ├─ 2.4.2 Bubble Detector → Bubble Adjustment (+0~15)
  ├─ 2.6 Genius Act Macro (스테이블코인 + 유동성)
  ├─ 2.9 GC-HRP (MST 포트폴리오)
  ├─ 2.11 Allocation Engine (MVO, Risk Parity)
  └─ 2.12 Rebalancing Policy

Phase 3: AI 토론
  ├─ 3.1 Full Mode (365일 lookback)
  ├─ 3.2 Reference Mode (90일 lookback)
  └─ 3.3 Dual Mode Analyzer → 합의 도출

Phase 5: 저장
  ├─ JSON (eimas_*.json)
  ├─ Markdown (eimas_*.md)
  └─ Database (events.db, signals.db)

Phase 7: AI 리포트 (--report)
  ├─ AIReportGenerator (Claude/Perplexity)
  ├─ WhiteningEngine (경제학적 해석)
  └─ FactChecker (팩트체킹)

Phase 8: Multi-LLM 검증 (--full)
  └─ Claude + GPT + Perplexity 교차 검증

Phase 8.5: Quick Mode 검증 (--quick1/2)
  ├─ PortfolioValidator (포트폴리오 이론)
  ├─ AllocationReasoner (학술 논문 검색)
  ├─ MarketSentimentAgent (KOSPI/SPX 분리)
  ├─ AlternativeAssetAgent (Crypto/Gold/RWA)
  └─ FinalValidator (최종 종합)
```

### Phase별 핵심 함수

| Phase | 함수 | 파일 | 설명 |
|-------|------|------|------|
| 1 | `_collect_data()` | main.py:151 | 데이터 수집 |
| 2 | `_analyze_basic()` | main.py:183 | 기본 분석 |
| 2+ | `_analyze_enhanced()` | main.py:203 | 고급 분석 (HFT, GARCH, DTW) |
| 2+ | `_apply_extended_data_adjustment()` | main.py:352 | 리스크 조정 |
| 3 | `_run_debate()` | main.py:502 | AI 에이전트 토론 |
| 5 | `_save_results()` | main.py:678 | 결과 저장 |
| 7 | `_generate_report()` | main.py:687 | AI 리포트 생성 |
| 8.5 | `_run_quick_validation()` | main.py:750 | Quick 모드 검증 |

---

## 5. 경제학적 방법론

| 방법론 | 사용처 | 설명 |
|--------|--------|------|
| **LASSO (L1 정규화)** | ForecastAgent | 변수 선택 (sparsity), 과적합 방지 |
| **Granger Causality** | LiquidityAnalyzer | 시계열 간 인과관계 테스트 |
| **GMM 3-State** | RegimeAnalyzer | Bull/Neutral/Bear 상태 분류 |
| **Shannon Entropy** | RegimeAnalyzer | 시장 불확실성 정량화 |
| **Bekaert VIX 분해** | CriticalPath | VIX = Uncertainty + Risk Appetite |
| **Greenwood-Shleifer** | BubbleDetector | 2년 100% run-up → 버블 위험 |
| **Amihud Lambda** | Microstructure | 비유동성 측정 (가격 충격/거래량) |
| **VPIN** | Microstructure | 정보 비대칭/독성 주문 흐름 |
| **MST (Mantegna 1999)** | GraphClusteredPortfolio | 상관관계 기반 최소신장트리 |
| **HRP (De Prado)** | GraphClusteredPortfolio | 계층적 리스크 패리티 |

### 핵심 수식

```python
# 순 유동성 (Fed 유동성)
Net Liquidity = Fed Balance Sheet - RRP - TGA

# Genius Act 확장 유동성
M = B + S·B*  (순유동성 + 스테이블코인 기여도)

# 리스크 점수 (v2.2.4)
Final Risk = Base(CriticalPath) + Micro Adj(±10) + Bubble Adj(+0~15)
# Risk Score Floor: 최소 1.0 (2026-02-05 수정)

# MST 거리 공식
d(i,j) = sqrt(2 * (1 - ρ_ij))
```

---

## 6. 디렉토리 구조

```
eimas/
├── main.py              # 메인 파이프라인 (~1088줄)
├── CLAUDE.md            # 이 문서 (요약)
├── ARCHITECTURE.md      # 상세 아키텍처
├── agents/              # 에이전트 모듈 (14개)
│   ├── base_agent.py
│   ├── orchestrator.py
│   ├── analysis_agent.py
│   └── ... (forecast, research, strategy)
├── core/                # 핵심 프레임워크
│   ├── schemas.py       # 데이터 스키마
│   ├── config.py        # API 설정
│   └── debate.py        # 토론 프로토콜
├── lib/                 # 기능 모듈 (52개 활성)
│   ├── fred_collector.py
│   ├── regime_analyzer.py (GMM & Entropy)
│   ├── critical_path.py
│   ├── microstructure.py (HFT, VPIN)
│   ├── bubble_detector.py (Greenwood-Shleifer)
│   ├── graph_clustered_portfolio.py (GC-HRP + MST)
│   ├── genius_act_macro.py
│   ├── allocation_engine.py (MVO, Risk Parity, HRP)
│   ├── rebalancing_policy.py
│   ├── quick_agents/ (5개 AI 검증 에이전트)
│   └── ...
├── api/                 # FastAPI 서버
│   ├── main.py
│   └── routes/ (health, analysis, regime, debate)
├── frontend/            # Next.js 16 대시보드
│   ├── app/
│   ├── components/
│   └── lib/
├── outputs/             # 결과 JSON/MD
├── data/                # Database (events.db, signals.db)
├── archive/             # 아카이브 (53개)
│   ├── future_experimental/ (28개)
│   └── deprecated/ (25개)
└── configs/             # YAML 설정
```

---

## 7. 핵심 데이터 클래스

```python
@dataclass
class EIMASResult:
    timestamp: str

    # Phase 1: 데이터
    fred_summary: Dict
    market_data_count: int
    crypto_data_count: int

    # Phase 2: 분석
    regime: Dict                     # regime, trend, volatility
    risk_score: float                # Final Risk (1.0~100.0)
    base_risk_score: float           # CriticalPath 기본 점수
    microstructure_adjustment: float # ±10 범위 조정
    bubble_risk_adjustment: float    # 버블 리스크 가산

    market_quality: MarketQualityMetrics  # 시장 미세구조 품질
    bubble_risk: BubbleRiskMetrics        # 버블 리스크 메트릭

    genius_act_regime: str           # expansion/contraction/neutral
    portfolio_weights: Dict[str, float]   # GC-HRP 결과
    allocation_result: Dict          # MVO/Risk Parity 결과
    rebalance_decision: Dict         # 리밸런싱 결정

    # Phase 3: 토론
    full_mode_position: str          # BULLISH/BEARISH/NEUTRAL
    reference_mode_position: str
    modes_agree: bool

    # 최종 결과
    final_recommendation: str        # HOLD/BUY/SELL
    confidence: float
    risk_level: str                  # LOW/MEDIUM/HIGH
    warnings: List[str]

@dataclass
class MarketQualityMetrics:
    avg_liquidity_score: float       # 0-100 스케일
    liquidity_scores: Dict[str, float]
    high_toxicity_tickers: List[str]  # VPIN > 50%
    illiquid_tickers: List[str]       # 유동성 < 30
    data_quality: str                 # COMPLETE/PARTIAL/DEGRADED

@dataclass
class BubbleRiskMetrics:
    overall_status: str              # NONE/WATCH/WARNING/DANGER
    risk_tickers: List[Dict]         # Top 5 위험 종목
    highest_risk_ticker: str
    highest_risk_score: float
```

---

## 8. 최근 업데이트 (Changelog)

### v2.2.4 (2026-02-05) - Risk Score Fix & Documentation

**Risk Score Edge Case 수정**
- **문제**: Risk Score = 0.0 edge case 발생
- **원인**: Base risk + 음수 adjustment = 0으로 clamping
- **수정**: `main.py` line 431
  ```python
  # BEFORE: result.risk_score = max(0, ...)
  # AFTER:  result.risk_score = max(1.0, ...)
  ```
- **결과**: 최소 risk score 1.0 보장, 경제학적으로 비현실적인 0 방지

**문서 업데이트**
- CLAUDE.md 전체 재정리 (중복 제거, 구조화)
- 모드별 Phase 실행 비교표 추가
- 파이프라인 흐름도 간략화

---

### v2.2.3 (2026-02-04) - Quick Mode AI Validation

**KOSPI/SPX 분리 AI 검증 에이전트 시스템**
- `lib/quick_agents/` 패키지 신규 생성 (~3,500 lines, 8개 파일)
- 5개 전문 AI 에이전트로 Full 모드 결과 검증
- KOSPI 전용 (--quick1), SPX 전용 (--quick2) 분리 실행

**5개 검증 에이전트**:
1. **PortfolioValidator** (Claude) - 포트폴리오 이론 검증
2. **AllocationReasoner** (Perplexity) - 학술 논문 검색
3. **MarketSentimentAgent** (Claude) - KOSPI/SPX 완전 분리 분석
4. **AlternativeAssetAgent** (Perplexity) - 대체자산 판단
5. **FinalValidator** (Claude) - 최종 종합 검증

**실행 결과** (2026-02-04):
- KOSPI: NEUTRAL (30% 신뢰도), Validation FAIL
- SPX: BULLISH (80% 신뢰도), Validation CAUTION
- Market Divergence: 두 시장 강한 괴리 (STRONG)
- 성공률: 60% (5개 중 3개 성공, Perplexity API 오류)

---

### v2.2.2 (2026-02-02) - Allocation Engine & Rebalancing

**비중 산출 엔진 및 리밸런싱 정책 추가**
- `lib/allocation_engine.py` (~700 lines)
  - MVO, Risk Parity, HRP, Equal Weight, Inverse Volatility
  - Black-Litterman (views 기반)
  - AllocationConstraints (min/max weight, turnover cap)
- `lib/rebalancing_policy.py` (~550 lines)
  - Periodic (Calendar), Threshold (Drift), Hybrid
  - TradingCostModel (수수료 + 스프레드 + 시장 충격)
  - Turnover Cap 적용 (기본 30%)

---

## 9. 개발자 가이드

### 새 모듈 추가 체크리스트

1. `lib/` 에 모듈 생성
2. `if __name__ == "__main__"` 테스트 코드 포함
3. `main.py`에 import 추가 (line 45-86)
4. 적절한 Phase에 호출 코드 추가
5. `EIMASResult`에 필요한 필드 추가 (line 100-146)
6. Summary 출력에 결과 추가 (line 958-1014)
7. 이 문서(CLAUDE.md) 업데이트

### 변경 후 검증 절차 (REQUIRED)

```bash
# 1. FULL 파이프라인 테스트 (REQUIRED - ~4분 소요)
python main.py

# 2. 결과 확인
ls -la outputs/eimas_*.json | tail -1

# 3. (선택) API 서버 테스트
uvicorn api.main:app --port 8000 &
curl http://localhost:8000/health
pkill -f "uvicorn api.main"
```

**주의**: `--quick` 모드는 Phase 2.3-2.10을 스킵하므로 의존성 오류를 놓칠 수 있습니다.

### Quick Tips

**성능 최적화**:
```bash
python main.py --quick              # 30초
timeout 600 python main.py          # 5분
nohup python main.py > eimas.log 2>&1 &  # 백그라운드
```

**디버깅**:
```bash
export EIMAS_LOG_LEVEL=DEBUG
python main.py --quick

export ANTHROPIC_LOG=debug
python main.py --quick1
```

---

## 10. API 및 CLI

### API 서버

```bash
# FastAPI 서버 실행
uvicorn api.main:app --reload --port 8000

# 엔드포인트
GET  /health           # 헬스 체크
POST /analysis/run     # 분석 실행
GET  /regime/current   # 현재 레짐
POST /debate/run       # 토론 실행
GET  /latest           # 최신 JSON 반환 (대시보드용)
```

### CLI 사용법

```bash
# CLI 도움말
python -m cli.eimas --help

# 분석 실행
python -m cli.eimas analyze --quick
python -m cli.eimas analyze --report
```

### API 키 (환경변수)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."      # Claude (필수)
export FRED_API_KEY="your-fred-key"        # FRED (필수)
export PERPLEXITY_API_KEY="pplx-..."       # Perplexity (선택)
export OPENAI_API_KEY="sk-..."             # OpenAI (선택)
export GOOGLE_API_KEY="..."                # Gemini (선택)
```

---

## 11. 알려진 이슈 및 상태

### ✅ 작동 중 (Stable)

- ✅ 메인 파이프라인 (Phase 1-8 전체)
- ✅ 데이터 수집 (FRED + yfinance + Crypto/RWA)
- ✅ AI 토론 (Full + Reference mode)
- ✅ 리포트 생성 (JSON + MD + HTML)
- ✅ Portfolio Theory Modules (Allocation, Rebalancing, Backtest)
- ✅ FastAPI 서버 (/latest 엔드포인트)

### ⚠️ 알려진 이슈

**1. Perplexity API 400 오류** (우선순위: 높음)
- 증상: AllocationReasoner, AlternativeAssetAgent에서 400 Bad Request
- 영향: Quick Mode 성공률 60% (5개 중 3개만 작동)
- 해결 필요: API 키 권한 확인, 요청 형식 디버깅

**2. KOSPI 데이터 신뢰도 낮음** (우선순위: 중간)
- 증상: KOSPI 정서 신뢰도 30% (SPX 80%에 비해 낮음)
- 원인: 한국 시장 데이터 부족
- 해결 필요: Korea Exchange API 추가

**3. 대시보드 차트 미구현** (우선순위: 낮음)
- 누락: 포트폴리오 파이 차트, 상관관계 히트맵
- 현재: 텍스트 메트릭만 표시
- 필요: Recharts 통합

### 📋 다음 작업 우선순위

1. **Priority 1**: Perplexity API 오류 해결
2. **Priority 2**: Quick Mode 안정성 개선 (60% → 80%)
3. **Priority 3**: 문서화 및 사용성 개선

---

## 12. 참고 문서

| 문서 | 경로 | 용도 |
|------|------|------|
| **CLAUDE.md** | `/CLAUDE.md` | 이 문서 (전체 시스템 개요) |
| **ARCHITECTURE.md** | `/ARCHITECTURE.md` | 상세 아키텍처 |
| **README.md** | `/README.md` | 프로젝트 소개 |
| **Quick Agents README** | `/lib/quick_agents/README.md` | Quick Mode 상세 |
| **API Documentation** | `/api/README.md` | FastAPI 엔드포인트 |
| **Dashboard Guide** | `/DASHBOARD_QUICKSTART.md` | 대시보드 빠른 시작 |

---

*마지막 업데이트: 2026-02-05 01:10 KST*
*문의: EIMAS 프로젝트 담당자*
