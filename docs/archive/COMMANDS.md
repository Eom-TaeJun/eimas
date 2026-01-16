# EIMAS 명령어 가이드

> Economic Intelligence Multi-Agent System 명령어 레퍼런스

## 목차

1. [데이터 수집](#1-데이터-수집)
2. [암호화폐 24/7 모니터링](#2-암호화폐-247-모니터링)
3. [분석 실행](#3-분석-실행)
4. [이벤트 시스템](#4-이벤트-시스템)
5. [서버 & 대시보드](#5-서버--대시보드)
6. [CLI 도구](#6-cli-도구)
7. [테스트](#7-테스트)

---

## 1. 데이터 수집

### 장중 데이터 수집 (IntradayCollector)

```bash
# 어제 장중 데이터 수집 (매일 아침 실행)
python lib/intraday_collector.py

# 특정 날짜 수집
python lib/intraday_collector.py --date 2026-01-02

# 특정 티커만 수집
python lib/intraday_collector.py --tickers SPY,QQQ,GLD

# 누락된 일자 백필 (최대 7일)
python lib/intraday_collector.py --backfill
```

**수행 작업:**
- yfinance에서 1분봉 데이터 조회
- 장중 집계 계산 (시가갭, 고저시간, VWAP, 거래량분포)
- 이상 감지 (VIX 스파이크, 급락, 거래량 폭발)
- `data/stable/market.db` → 장중 집계 저장
- `data/volatile/realtime.db` → 알림/이벤트 저장

---

### 일일 데이터 수집 (DailyCollector)

```bash
# 일일 데이터 수집 (장 마감 후)
python scripts/daily_collector.py

# 특정 날짜 수집
python scripts/daily_collector.py --date 2026-01-02

# 조용히 실행 (로그 최소화)
python scripts/daily_collector.py --quiet
```

**수행 작업:**
- ETF/주식 가격 데이터 (SPY, QQQ, IWM, TLT, GLD 등)
- ARK Holdings 데이터
- 시장 지표 (VIX, Credit Spread, FX)
- FRED 거시 지표
- `data/eimas.db` 저장

**Cron 설정 (매일 오후 5시 EST):**
```bash
0 17 * * 1-5 cd /home/tj/projects/autoai/eimas && python scripts/daily_collector.py >> logs/daily.log 2>&1
```

---

### 다중 API 데이터 파이프라인 (MarketDataPipeline)

```bash
# 전체 기본 자산 수집
python lib/market_data_pipeline.py --all

# 원유 포함 수집
python lib/market_data_pipeline.py --all --with-oil

# 단일 자산 수집
python lib/market_data_pipeline.py --provider twelvedata --symbol AAPL
python lib/market_data_pipeline.py --provider cryptocompare --symbol BTC-USD --interval 1h
python lib/market_data_pipeline.py --provider yfinance --symbol CL=F

# Python에서 직접 사용
python -c "
from lib.market_data_pipeline import fetch_data, save_data
df = fetch_data('cryptocompare', 'BTC-USD', '1d', 100)
save_data(df, 'cryptocompare', 'BTC-USD', '1d')
"
```

**지원 Provider:**
| Provider | 자산 유형 | 무료 제한 | API 키 필요 |
|----------|----------|----------|------------|
| Twelve Data | 주식, FX, 원자재 | 800 calls/day, 8/min | ✅ |
| CryptoCompare | 암호화폐 | 100,000 calls/month | ❌ (선택) |
| yfinance | 전체 (백업) | 없음 (비공식) | ❌ |

**기본 수집 자산:**
| Provider | 심볼 | 설명 |
|----------|------|------|
| twelvedata | AAPL, MSFT | 미국 주식 |
| twelvedata | XAU/USD, XAG/USD | 금, 은 |
| cryptocompare | BTC-USD, ETH-USD | 암호화폐 |
| yfinance (--with-oil) | CL=F, BZ=F | WTI, 브렌트 원유 |

**저장 경로:** `data/market/{provider}_{symbol}_{interval}.csv`

**환경 변수 설정:**
```bash
# .env 파일에 추가
TWELVEDATA_API_KEY=your_key_here
CRYPTOCOMPARE_API_KEY=your_key_here  # 선택
```

---

## 2. 암호화폐 24/7 모니터링

### CryptoCollector (주말/휴일 포함)

```bash
# 현재 암호화폐 상태 조회 + 이상 탐지
python lib/crypto_collector.py --detect

# 이상 탐지 + 뉴스 원인 분석 (Perplexity API)
python lib/crypto_collector.py --detect --analyze

# 특정 코인만 모니터링
python lib/crypto_collector.py --coins BTC,ETH,SOL

# 기본 실행 (가격만 조회)
python lib/crypto_collector.py
```

**수행 작업:**
- 24시간 암호화폐 가격 수집 (BTC, ETH, SOL 등 10개)
- 이상 탐지:
  - 15분 내 ±3% 이상 변동
  - 1시간 내 ±5% 이상 변동
  - 거래량 3배 이상 폭발
  - 변동성 2.5σ 이상 급등
- Perplexity API로 이상 원인 뉴스 검색
- `data/volatile/realtime.db` → 이벤트 저장

**기본 모니터링 코인:**
| 심볼 | 이름 |
|------|------|
| BTC | Bitcoin |
| ETH | Ethereum |
| SOL | Solana |
| XRP | Ripple |
| ADA | Cardano |
| DOGE | Dogecoin |
| AVAX | Avalanche |
| DOT | Polkadot |
| LINK | Chainlink |

**주말 자동 실행 (Cron):**
```bash
# 주말(토,일) 매 시간 실행
0 * * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/crypto_collector.py --detect >> logs/crypto.log 2>&1

# 4시간마다 뉴스 분석 포함
0 */4 * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/crypto_collector.py --detect --analyze >> logs/crypto.log 2>&1
```

---

## 3. 분석 실행

### 메인 파이프라인 (main.py)

```bash
# 기본 실행
python main.py                    # 전체 분석 (~40초)
python main.py --quick            # 빠른 분석 (~16초, Phase 2.3-2.10 스킵)

# 실시간 모니터링
python main.py --realtime                  # 기본 30초
python main.py --realtime --duration 60   # 60초

# AI 리포트
python main.py --report           # Claude/Perplexity 해석 포함

# 모든 기능 실행 (권장)
python main.py --realtime --report --duration 60

# 서버/자동화
python main.py --cron             # 최소 출력 (크론용)
python main.py --output /path     # 출력 디렉토리 지정
python main.py --version          # 버전 확인
```

---

### 기능별 포함 여부

| Phase | 기능 | `main.py` | `--quick` | `--realtime` | `--report` |
|-------|------|:---------:|:---------:|:------------:|:----------:|
| 1.1 | FRED 데이터 (RRP, TGA, 유동성) | ✅ | ✅ | ✅ | ✅ |
| 1.2 | 시장 데이터 (24 tickers) | ✅ | ✅ | ✅ | ✅ |
| 1.3 | 크립토 + RWA | ✅ | ✅ | ✅ | ✅ |
| 2.1 | 레짐 탐지 (Bull/Bear/Neutral) | ✅ | ✅ | ✅ | ✅ |
| 2.1.1 | GMM & Shannon Entropy | ✅ | ❌ | ✅ | ✅ |
| 2.2 | 이벤트 탐지 | ✅ | ✅ | ✅ | ✅ |
| 2.3 | Granger Causality 분석 | ✅ | ❌ | ✅ | ✅ |
| 2.4 | CriticalPath 리스크 | ✅ | ✅ | ✅ | ✅ |
| 2.4.1 | 시장 미세구조 (Microstructure) | ✅ | ❌ | ✅ | ✅ |
| 2.4.2 | 버블 리스크 (Greenwood-Shleifer) | ✅ | ❌ | ✅ | ✅ |
| 2.5 | ETF 플로우 분석 | ✅ | ❌ | ✅ | ✅ |
| 2.6 | Genius Act Macro | ✅ | ❌ | ✅ | ✅ |
| 2.6.1 | Crypto Stress Test | ✅ | ❌ | ✅ | ✅ |
| 2.7 | 테마 ETF 분석 | ✅ | ❌ | ✅ | ✅ |
| 2.8 | 충격 전파 그래프 | ✅ | ❌ | ✅ | ✅ |
| 2.9 | GC-HRP 포트폴리오 | ✅ | ❌ | ✅ | ✅ |
| 2.10 | 통합 전략 | ✅ | ❌ | ✅ | ✅ |
| 3 | Multi-Agent 토론 | ✅ | ✅ | ✅ | ✅ |
| 4 | 실시간 VPIN 모니터링 | ❌ | ❌ | ✅ | ❌ |
| 5 | DB/JSON/MD 저장 | ✅ | ✅ | ✅ | ✅ |
| 6 | AI 리포트 생성 | ❌ | ❌ | ❌ | ✅ |
| 7 | Whitening & Fact Check | ❌ | ❌ | ❌ | ✅ |

---

### 실행 순서 (Phase Flow)

```
Phase 1: DATA COLLECTION
├─ [1.1] FRED → RRP, TGA, Net Liquidity, Fed Funds
├─ [1.2] Market Data → 24 tickers (SPY, QQQ, TLT 등)
└─ [1.3] Crypto + RWA → BTC, ETH, ONDO, PAXG, COIN

Phase 2: ANALYSIS
├─ [2.1] RegimeDetector → Bull/Bear/Neutral
├─ [2.1.1] GMMRegimeAnalyzer → 3-state + Entropy
├─ [2.2] EventDetector → 이상 이벤트 탐지
├─ [2.3] LiquidityAnalyzer → Granger Causality
├─ [2.4] CriticalPath → Base Risk Score
├─ [2.4.1] Microstructure → 유동성 품질 조정 (±10)
├─ [2.4.2] BubbleDetector → 버블 리스크 가산 (+0~15)
├─ [2.5] ETFFlowAnalyzer → 섹터 로테이션
├─ [2.6] GeniusActMacro → 스테이블코인-유동성
├─ [2.6.1] CryptoStressTest → De-peg 확률
├─ [2.7] CustomETFBuilder → 테마 ETF
├─ [2.8] ShockPropagation → 인과관계 그래프
├─ [2.9] GC-HRP → 포트폴리오 최적화
└─ [2.10] IntegratedStrategy → 통합 시그널

Phase 3: MULTI-AGENT DEBATE
├─ [3.1] FULL Mode (365일) → Position
├─ [3.2] REF Mode (90일) → Position
└─ [3.3] DualModeAnalyzer → 비교/합의

Phase 4: REALTIME (--realtime 필요)
└─ [4.1] BinanceStreamer → VPIN 실시간 계산

Phase 5: SAVE
├─ [5.1] events.db → 이벤트 저장
├─ [5.2] realtime_signals.db → 실시간 시그널
├─ [5.3] integrated_*.json → JSON 결과
└─ [5.4] integrated_*.md → 마크다운 리포트

Phase 6: AI REPORT (--report 필요)
└─ [6.1] Claude/Perplexity → 자연어 해석

Phase 7: QUALITY (--report 필요)
├─ [7.1] WhiteningEngine → 경제학적 해석
└─ [7.2] FactChecker → AI 출력 검증
```

---

### 리스크 점수 계산

```
Final Risk = Base(CriticalPath) + Micro Adj(±10) + Bubble Adj(+0~15)

예시:
- Base = 45.0 (CriticalPath 기본 점수)
- Micro Adj = -4.0 (유동성 우수)
- Bubble Adj = +10.0 (WARNING level)
- Final = 51.0/100
```

---

### 출력 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| JSON 결과 | `outputs/integrated_YYYYMMDD_HHMMSS.json` | 전체 분석 데이터 |
| MD 리포트 | `outputs/integrated_YYYYMMDD_HHMMSS.md` | 사람이 읽는 리포트 |
| 이벤트 DB | `data/events.db` | 탐지된 이벤트 저장 |
| 실시간 DB | `outputs/realtime_signals.db` | VPIN 시그널 저장 |

---

### 빠른 참조

```bash
# 일반 사용자: 전체 분석
python main.py

# 개발/테스트: 빠른 확인
python main.py --quick

# 트레이더: 실시간 모니터링
python main.py --realtime --duration 120

# 리서치: 모든 기능 + AI 해석
python main.py --realtime --report --duration 60
```

---

### 일일 분석 (DailyAnalysis)

```bash
# 전체 일일 분석 실행
python scripts/daily_analysis.py

# 리포트만 생성
python scripts/daily_analysis.py --report-only
```

**수행 작업:**
1. 시그널 수집 → DB 저장
2. 포트폴리오 후보 생성
3. 세션 분석 (전일)
4. 피드백 업데이트
5. 일일 리포트 생성

---

### 백테스트

```bash
# 백테스트 실행
python scripts/run_backtest.py

# 특정 전략 백테스트
python scripts/run_backtest.py --strategy momentum

# 기간 지정
python scripts/run_backtest.py --start 2024-01-01 --end 2024-12-31
```

---

## 4. 이벤트 시스템

### 이벤트 예측

```bash
# 이벤트 예측 실행
python -c "
from lib.event_predictor import EventPredictor
predictor = EventPredictor()
predictor.generate_report()
"
```

**수행 작업:**
- NFP, CPI, FOMC 등 경제 이벤트 예측
- 시나리오별 확률 및 수익률 계산
- 가격 목표 및 트레이딩 레벨 생성
- `outputs/event_prediction_report.md` 생성

---

### 이벤트 역추적 (Attribution)

```bash
# 이벤트 원인 분석
python -c "
from lib.event_attribution import EventAttributor
attr = EventAttributor()
report = attr.analyze_recent_events(days_back=14)
attr.generate_report(report)
"
```

**수행 작업:**
- 감지된 이벤트의 원인 분석
- Perplexity API 연동 뉴스 검색
- 크로스-에셋 상관관계 분석
- `outputs/event_attribution_report.md` 생성

---

### 이벤트 백테스트

```bash
# 이벤트 백테스트 실행
python lib/event_backtester.py
```

**수행 작업:**
- 과거 경제 이벤트 (FOMC, CPI, NFP) 분석
- 이벤트 전후 수익률 패턴 분석
- 전략 성과 평가

---

### 이상-뉴스 자동 귀인 (NewsCorrelator)

```bash
# 최근 24시간 이상 분석 + 뉴스 연결
python lib/news_correlator.py

# Python에서 직접 실행
python -c "
from lib.news_correlator import NewsCorrelator
correlator = NewsCorrelator()
attributions = correlator.process_recent_anomalies(hours_back=24)
report = correlator.generate_report(attributions)
print(report)
"

# 주말용 선물/FX 자산 수집
python -c "
from lib.news_correlator import WeekendAssetCollector
collector = WeekendAssetCollector()
anomalies = collector.collect_and_detect()
print(f'감지된 이상: {len(anomalies)}개')
"
```

**수행 작업:**
1. **이상 클러스터링**: 30분 내 발생한 이상들을 하나의 이벤트로 그룹화
2. **심각도 필터링**: 임계값(1.5) 이상만 뉴스 검색
3. **다국어 뉴스 검색**:
   - Phase 1: 영어로 글로벌 개요 검색
   - Phase 2: 관련 국가 감지 시 해당 언어(한/중/일)로 상세 검색
4. **시간 상관 분석**: 이상 발생 전 1시간 ~ 후 3시간 뉴스 검색
5. **신뢰도 계산**: 뉴스 개수, 다국어 여부, 심각도 기반
6. **DB 저장**: `data/volatile/realtime.db` → `event_attribution` 테이블

**주말 추가 자산 (일요일 저녁부터 거래):**
| 자산 | 심볼 | 거래 시작 (ET) |
|------|------|----------------|
| WTI 원유 선물 | CL=F | 일요일 18:00 |
| 금 선물 | GC=F | 일요일 18:00 |
| 은 선물 | SI=F | 일요일 18:00 |
| 달러 인덱스 | DX-Y.NYB | 일요일 17:00 |
| EUR/USD | EURUSD=X | 일요일 17:00 |
| USD/JPY | USDJPY=X | 일요일 17:00 |

**다국어 검색 전략:**
| 언어 | 트리거 키워드 |
|------|---------------|
| 한국어 | korea, samsung, kospi, hyundai, north korea |
| 중국어 | china, taiwan, xi jinping, alibaba, hong kong |
| 일본어 | japan, nikkei, yen, boj, tokyo |
| 스페인어 | venezuela, maduro, mexico, brazil, latin america |

---

## 5. 서버 & 대시보드

### FastAPI 서버

```bash
# API 서버 실행
uvicorn api.main:app --reload --port 8000

# 또는
python api/main.py
```

**엔드포인트:**
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | API 상태 |
| GET | `/api/signals` | 최신 시그널 |
| GET | `/api/portfolio` | 현재 포트폴리오 |
| GET | `/api/risk` | 리스크 지표 |
| GET | `/api/correlation` | 상관관계 행렬 |
| GET | `/api/sectors` | 섹터 로테이션 |
| GET | `/api/optimize` | 포트폴리오 최적화 |
| POST | `/api/paper-trade` | 페이퍼 트레이드 실행 |

---

### Streamlit 대시보드

```bash
# 대시보드 실행
streamlit run dashboard.py
```

**기능:**
- 실시간 시그널 모니터링
- 포트폴리오 현황
- 리스크 지표 시각화
- 상관관계 히트맵

---

### Plotly 대시보드

```bash
# Plotly 대시보드 (Dash)
python plus/dashboard_generator.py
```

---

## 6. CLI 도구

### EIMAS CLI

```bash
# 시그널 관련
python cli/eimas.py signal list          # 시그널 목록
python cli/eimas.py signal active        # 활성 시그널

# 포트폴리오 관련
python cli/eimas.py portfolio show       # 포트폴리오 조회
python cli/eimas.py portfolio optimize   # 최적화

# 리스크 관련
python cli/eimas.py risk check           # 리스크 체크
python cli/eimas.py risk exposure        # 노출도 분석

# 트레이딩
python cli/eimas.py trade buy SPY 10     # 매수
python cli/eimas.py trade sell SPY 5     # 매도
python cli/eimas.py trade status         # 상태 조회

# 리포트
python cli/eimas.py report daily         # 일일 리포트
python cli/eimas.py report weekly        # 주간 리포트
```

---

## 7. 테스트

### API 연결 테스트

```bash
# API 연결 확인 (Claude, OpenAI, Gemini, Perplexity)
python tests/test_api_connection.py
```

---

### 라이브러리 테스트

```bash
# lib 모듈 테스트
python tests/test_lib.py

# 시그널-액션 테스트
python tests/test_signal_action.py

# LASSO 예측 테스트
python tests/test_lasso_forecast.py
```

---

## 저장소 구조

```
data/
├── stable/                     # 안정 데이터 (확정, 영구 보존)
│   └── market.db
│       ├── daily_prices        # 일별 OHLCV
│       ├── intraday_summary    # 장중 집계
│       ├── economic_calendar   # 경제 이벤트
│       └── prediction_outcomes # 예측 결과
│
├── volatile/                   # 휘발성 데이터 (실시간, 이벤트)
│   └── realtime.db
│       ├── detected_events     # 감지된 이벤트
│       ├── intraday_alerts     # 장중 알림
│       ├── active_predictions  # 진행 중인 예측
│       └── market_snapshots    # 시장 스냅샷
│
├── eimas.db                    # 메인 DB (시그널, ARK, 가격)
├── trading.db                  # 트레이딩 DB
├── events.db                   # 이벤트 DB (레거시)
└── paper_trading.db            # 페이퍼 트레이딩 DB
```

---

## 일일 운영 루틴

### 평일 아침 (한국 시간 08:00)

```bash
# 1. 어제 장중 데이터 수집
python lib/intraday_collector.py

# 2. 이상-뉴스 자동 귀인 (어제 발생한 이상 분석)
python lib/news_correlator.py

# 3. 이벤트 예측 업데이트 (선택)
python -c "from lib.event_predictor import EventPredictor; EventPredictor().generate_report()"
```

### 평일 저녁 (미국 장 마감 후, 한국 시간 06:00)

```bash
# 1. 일일 데이터 수집
python scripts/daily_collector.py

# 2. 일일 분석 실행
python scripts/daily_analysis.py

# 3. 오늘 이상에 대한 뉴스 귀인
python -c "
from lib.news_correlator import NewsCorrelator
correlator = NewsCorrelator()
attrs = correlator.process_recent_anomalies(hours_back=12)
if attrs:
    print(correlator.generate_report(attrs))
"
```

### 주말 (24/7 암호화폐 + 선물 모니터링)

```bash
# 1. 암호화폐 체크 + 이상 탐지
python lib/crypto_collector.py --detect

# 2. 일요일 저녁: 선물/FX 이상 탐지
python -c "
from lib.news_correlator import WeekendAssetCollector, NewsCorrelator
collector = WeekendAssetCollector()
anomalies = collector.collect_and_detect()
if anomalies:
    correlator = NewsCorrelator()
    clusters = correlator.cluster_anomalies(anomalies)
    for cluster in clusters:
        correlator.correlate_cluster(cluster)
"

# 3. 이상 발생 시 뉴스 자동 귀인
python lib/news_correlator.py
```

**자동화 권장 (Cron):**
```bash
# 주말 매 시간 암호화폐 모니터링
0 * * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/crypto_collector.py --detect >> logs/crypto.log 2>&1

# 주말 4시간마다 뉴스 귀인 실행
0 */4 * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/news_correlator.py >> logs/correlator.log 2>&1

# 일요일 저녁 (ET 18:00 = KST 월요일 08:00) 선물 체크
0 8 * * 1 cd /home/tj/projects/autoai/eimas && python -c "from lib.news_correlator import WeekendAssetCollector; WeekendAssetCollector().collect_and_detect()" >> logs/weekend.log 2>&1
```

---

## 환경 변수

```bash
# API 키 (필수)
export ANTHROPIC_API_KEY="..."      # Claude API
export OPENAI_API_KEY="..."         # OpenAI API
export PERPLEXITY_API_KEY="..."     # Perplexity API
export GOOGLE_API_KEY="..."         # Gemini API (선택)

# 설정 (선택)
export EIMAS_DATA_DIR="/path/to/data"
export EIMAS_LOG_LEVEL="INFO"
```

---

## 문제 해결

### 데이터 없음 오류

```bash
# yfinance 캐시 정리
rm -rf ~/.cache/py-yfinance

# 7일 이상 지난 데이터는 조회 불가 (yfinance 1분봉 제한)
```

### DB 잠금 오류

```bash
# SQLite 잠금 해제
fuser -k data/eimas.db
```

### 모듈 import 오류

```bash
# PYTHONPATH 설정
export PYTHONPATH=/home/tj/projects/autoai/eimas:$PYTHONPATH
```
