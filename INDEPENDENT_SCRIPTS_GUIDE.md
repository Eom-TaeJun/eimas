# EIMAS 독립 실행 가능 스크립트 가이드

> **독립적으로 실행 가능한 Python 파일 총정리**
> 각 스크립트의 기능, 실행 방법, 출력 결과

---

## 📋 목차

1. [데이터 수집 스크립트](#1-데이터-수집-스크립트)
2. [분석 스크립트](#2-분석-스크립트)
3. [이벤트 시스템](#3-이벤트-시스템)
4. [백테스트 & 트레이딩](#4-백테스트--트레이딩)
5. [검증 & 테스트](#5-검증--테스트)
6. [유틸리티 & 도구](#6-유틸리티--도구)

---

## 1. 데이터 수집 스크립트

### 📊 intraday_collector.py

**위치**: `lib/intraday_collector.py`

**기능**:
- 전일 장중 1분봉 데이터 수집
- 장중 집계 계산 (시가갭, 고저시간, VWAP, 거래량 분포)
- 이상 탐지 (VIX 스파이크, 급락, 거래량 폭발)

**실행**:
```bash
# 어제 장중 데이터 수집 (기본)
python lib/intraday_collector.py

# 특정 날짜 수집
python lib/intraday_collector.py --date 2026-01-02

# 특정 티커만 수집
python lib/intraday_collector.py --tickers SPY,QQQ,GLD

# 누락된 일자 백필 (최대 7일)
python lib/intraday_collector.py --backfill
```

**출력**:
- DB: `data/stable/market.db` → 장중 집계 저장
- DB: `data/volatile/realtime.db` → 알림/이벤트 저장
- 콘솔: 수집 통계 및 감지된 이상 이벤트

**실행 결과** (2026-01-12):
```
✅ 수집 완료: 0/5 저장, 0 알림
(주말이라 데이터 없음)
```

---

### 📈 daily_collector.py

**위치**: `scripts/daily_collector.py`

**기능**:
- 일일 종가 데이터 수집 (장 마감 후)
- ETF/주식 가격 (SPY, QQQ, IWM, TLT 등)
- ARK Holdings 데이터
- 시장 지표 (VIX, Credit Spread, FX)
- FRED 거시 지표

**실행**:
```bash
# 일일 데이터 수집 (장 마감 후)
python scripts/daily_collector.py

# 특정 날짜 수집
python scripts/daily_collector.py --date 2026-01-02

# 조용히 실행 (로그 최소화)
python scripts/daily_collector.py --quiet
```

**출력**:
- DB: `data/eimas.db` → 가격 데이터 저장
- 콘솔: 수집 진행 상황

**Cron 설정**:
```bash
# 매일 오후 5시 EST (장 마감 후)
0 17 * * 1-5 cd /home/tj/projects/autoai/eimas && python scripts/daily_collector.py >> logs/daily.log 2>&1
```

---

### 🪙 crypto_collector.py

**위치**: `lib/crypto_collector.py`

**기능**:
- 24시간 암호화폐 가격 수집 (주말 포함)
- 이상 탐지:
  - 15분 내 ±3% 이상 변동
  - 1시간 내 ±5% 이상 변동
  - 거래량 3배 이상 폭발
  - 변동성 2.5σ 이상 급등
- Perplexity API로 이상 원인 뉴스 검색

**실행**:
```bash
# 현재 가격 + 이상 탐지
python lib/crypto_collector.py --detect

# 이상 탐지 + 뉴스 원인 분석 (Perplexity API)
python lib/crypto_collector.py --detect --analyze

# 특정 코인만 모니터링
python lib/crypto_collector.py --coins BTC,ETH,SOL

# 기본 실행 (가격만 조회)
python lib/crypto_collector.py
```

**모니터링 코인** (10개):
- BTC, ETH, SOL, XRP, ADA
- DOGE, AVAX, DOT, LINK, MATIC

**출력**:
- DB: `data/volatile/realtime.db` → 이벤트 저장
- 콘솔: 현재가, 이상 감지 리스트

**실행 결과** (2026-01-12):
```
⚠️ 총 45개 이상 감지됨
  - [15:40] BTC 거래량 3.7배 폭발
  - [16:00] ETH 거래량 7.3배 폭발
  - [15:50] ETH 변동성 4.1σ 급등
```

**Cron 설정** (주말 자동):
```bash
# 주말 매 시간 실행
0 * * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/crypto_collector.py --detect >> logs/crypto.log 2>&1

# 4시간마다 뉴스 분석 포함
0 */4 * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/crypto_collector.py --detect --analyze >> logs/crypto.log 2>&1
```

---

### 🌐 market_data_pipeline.py

**위치**: `lib/market_data_pipeline.py`

**기능**:
- 다중 API를 통한 시장 데이터 수집
- 지원 Provider:
  - Twelve Data (주식, FX, 원자재)
  - CryptoCompare (암호화폐)
  - yfinance (백업)

**실행**:
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

**지원 Provider**:
| Provider | 자산 유형 | 무료 제한 | API 키 |
|----------|----------|----------|--------|
| Twelve Data | 주식, FX, 원자재 | 800 calls/day, 8/min | 필수 |
| CryptoCompare | 암호화폐 | 100K calls/month | 선택 |
| yfinance | 전체 (백업) | 제한 없음 | 불필요 |

**기본 수집 자산**:
- twelvedata: AAPL, MSFT, XAU/USD, XAG/USD
- cryptocompare: BTC-USD, ETH-USD
- yfinance (--with-oil): CL=F, BZ=F (WTI, 브렌트)

**출력**:
- 파일: `data/market/{provider}_{symbol}_{interval}.csv`

**환경 변수**:
```bash
export TWELVEDATA_API_KEY=your_key_here
export CRYPTOCOMPARE_API_KEY=your_key_here  # 선택
```

---

### 📊 fred_collector.py

**위치**: `lib/fred_collector.py`

**기능**:
- FRED (연준) 경제 데이터 수집
- RRP (역레포), TGA (재무부 계정)
- Net Liquidity 계산
- Fed Funds Rate, Treasury Yields

**실행**:
```bash
# 단독 실행 (main.py에서 자동 호출되므로 드물게 사용)
python -c "from lib.fred_collector import FREDCollector; FREDCollector().collect()"
```

**출력**:
- Python dict: FRED 데이터 요약
- main.py Phase 1.1에서 사용됨

---

## 2. 분석 스크립트

### 📈 daily_analysis.py

**위치**: `scripts/daily_analysis.py`

**기능**:
- 일일 종합 분석 파이프라인
- 시그널 수집 → DB 저장
- 포트폴리오 후보 생성
- 세션 분석 (전일)
- 피드백 업데이트
- 일일 리포트 생성

**실행**:
```bash
# 전체 일일 분석 실행
python scripts/daily_analysis.py

# 리포트만 생성
python scripts/daily_analysis.py --report-only
```

**출력**:
- JSON: `outputs/daily_analysis_YYYY-MM-DD.json`
- 콘솔: 시그널 요약, 포트폴리오 제안

**실행 결과** (2026-01-12):
```
Signal Summary:
  Action: HEDGE
  Conviction: 52%
  Reasoning: [Path 1] WARNING: Yield Curve at -0.30...

Generated Portfolios:
  [CONSERVATIVE] ID=14
    Expected Return: 5.1%
    Expected Risk: 5.8%
    Sharpe Ratio: 0.87
```

---

### 🔮 event_predictor.py

**위치**: `lib/event_predictor.py`

**기능**:
- 경제 이벤트 예측 (NFP, CPI, FOMC 등)
- 시나리오별 확률 및 수익률 계산
- 가격 목표 및 트레이딩 레벨 생성

**실행**:
```bash
# 이벤트 예측 실행
python lib/event_predictor.py

# Python에서 직접 사용
python -c "
from lib.event_predictor import EventPredictor
predictor = EventPredictor()
# 내부적으로 분석 수행
"
```

**출력**:
- 콘솔: 이벤트 예측 리포트
- 파일: `outputs/event_prediction_report.md` (선택)

**실행 결과** (2026-01-12):
```
📅 CPI Release (2026-01-14, D+1)
  Pre-Event: +0.08%
  Post-Event Weighted: +0.04% (T+1), +0.09% (T+5)
  Recommendation: NEUTRAL - Wait for event

📅 FOMC Rate Decision (2026-01-28, D+15)
  Pre-Event: +0.12%
  Post-Event Weighted: +0.16% (T+1), +0.59% (T+5)
  Recommendation: Positive positioning
```

---

### 🔍 event_attribution.py

**위치**: `lib/event_attribution.py`

**기능**:
- 감지된 이벤트의 원인 분석
- Perplexity API 연동 뉴스 검색
- 크로스-에셋 상관관계 분석

**실행**:
```bash
# 이벤트 원인 분석
python -c "
from lib.event_attribution import EventAttributor
attr = EventAttributor()
report = attr.analyze_recent_events(days_back=14)
# report 사용
"
```

**출력**:
- Python dict: 이벤트 분석 결과
- 파일: `outputs/event_attribution_report.md` (선택)

**실행 결과** (2026-01-12):
```
[EventAttributor] Analyzing events for ['SPY', 'QQQ', 'GLD', 'TLT', 'IWM']
[EventAttributor] Found 2 events
```

---

### 📰 news_correlator.py

**위치**: `lib/news_correlator.py`

**기능**:
- 이상 탐지-뉴스 자동 귀인 시스템
- 프로세스:
  1. 이상 클러스터링 (30분 윈도우)
  2. 심각도 필터링 (> 1.5)
  3. 다국어 뉴스 검색 (영/한/중/일)
  4. 시간 상관 분석 (이상 전 1시간 ~ 후 3시간)
  5. 신뢰도 계산
  6. DB 저장

**실행**:
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

**주말 추가 자산** (일요일 저녁부터 거래):
| 자산 | 심볼 | 거래 시작 (ET) |
|------|------|----------------|
| WTI 원유 선물 | CL=F | 일요일 18:00 |
| 금 선물 | GC=F | 일요일 18:00 |
| 은 선물 | SI=F | 일요일 18:00 |
| 달러 인덱스 | DX-Y.NYB | 일요일 17:00 |
| EUR/USD | EURUSD=X | 일요일 17:00 |
| USD/JPY | USDJPY=X | 일요일 17:00 |

**다국어 검색 전략**:
| 언어 | 트리거 키워드 |
|------|---------------|
| 한국어 | korea, samsung, kospi, hyundai |
| 중국어 | china, taiwan, xi jinping, alibaba |
| 일본어 | japan, nikkei, yen, boj, tokyo |
| 스페인어 | venezuela, maduro, mexico, brazil |

**출력**:
- DB: `data/volatile/realtime.db` → `event_attribution` 테이블
- 콘솔: 발견된 뉴스 및 클러스터 정보

**실행 결과** (2026-01-12):
```
클러스터: cluster_20260103_0615
  자산: ETH, BTC
  심각도: 8.81
  뉴스: 6건
    - Ethereum $3,100-$3,150 거래 (~3-5% 랠리)
    - Bitcoin $89,810-$90,962 거래 (+0.72%)
    - 미국 베네수엘라 군사 작전
```

**Cron 설정**:
```bash
# 주말 4시간마다
0 */4 * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/news_correlator.py >> logs/correlator.log 2>&1
```

---

## 3. 이벤트 시스템

### 📊 event_tracker.py

**위치**: `lib/event_tracker.py`

**기능**:
- 이상 탐지 → 뉴스 자동 연결
- Perplexity API 기반 뉴스 검색
- main.py Phase 2.12에서 사용

**실행**:
```bash
# main.py에서 자동 호출됨
# 독립 실행 시:
python -c "
from lib.event_tracker import EventTracker
tracker = EventTracker()
results = tracker.track_and_match()
print(f'Events matched: {len(results)}')
"
```

**출력**:
- Python dict: 매칭된 이벤트 리스트
- main.py에서 통합 결과에 포함됨

---

### 🔄 event_backtester.py

**위치**: `lib/event_backtester.py`

**기능**:
- 과거 경제 이벤트 (FOMC, CPI, NFP) 분석
- 이벤트 전후 수익률 패턴 분석
- 전략 성과 평가

**실행**:
```bash
# 이벤트 백테스트 실행
python lib/event_backtester.py
```

**출력**:
- 콘솔: 백테스트 결과
- 파일: `outputs/event_backtest_results.json` (선택)

---

## 4. 백테스트 & 트레이딩

### 📊 run_backtest.py

**위치**: `scripts/run_backtest.py`

**기능**:
- 전략 백테스트 실행
- 지원 전략:
  - EIMAS_Regime (레짐 기반)
  - Multi_Factor (다중 팩터)
  - Momentum
  - Mean Reversion

**실행**:
```bash
# 백테스트 실행
python scripts/run_backtest.py

# 특정 전략 백테스트
python scripts/run_backtest.py --strategy momentum

# 기간 지정
python scripts/run_backtest.py --start 2024-01-01 --end 2024-12-31
```

**출력**:
- JSON: `outputs/backtest_results.json` (27KB)
- 콘솔: 백테스트 리포트

**실행 결과** (2026-01-12):
```
EIMAS_Regime 전략 (2020-2024):
  Total Return: +8,359.91%
  Annual Return: +143.04%
  Sharpe Ratio: 1.85
  Max Drawdown: 3.53%
  Win Rate: 39.4%
  Trades: 33개

Multi_Factor 전략 (2020-2024):
  Total Return: +338.20%
  Annual Return: +34.40%
  Sharpe Ratio: 1.10
  Win Rate: 63.6%
  Trades: 11개
```

---

### 📝 paper_trader.py

**위치**: `lib/paper_trader.py`

**기능**:
- 페이퍼 트레이딩 시뮬레이션
- 주문 실행 (매수/매도)
- 포지션 관리
- 손익 계산

**실행**:
```bash
# API 서버 통해 실행
# POST /api/paper-trade

# Python에서 직접 사용
python -c "
from lib.paper_trader import PaperTrader
trader = PaperTrader()
trader.execute_trade('SPY', 'BUY', 10)
trader.get_positions()
"
```

**출력**:
- DB: `data/paper_trading.db` → 거래 내역 저장
- Python dict: 거래 결과

---

## 5. 검증 & 테스트

### 🧪 test_api_connection.py

**위치**: `tests/test_api_connection.py`

**기능**:
- API 연결 테스트
- 지원 API: Claude, OpenAI, Gemini, Perplexity
- 간단한 Multi-AI 토론 테스트

**실행**:
```bash
# API 연결 확인
python tests/test_api_connection.py
```

**출력**:
- 콘솔: API 연결 상태 요약

**실행 결과** (2026-01-12):
```
Environment:
  ✓ Claude
  ✓ OpenAI
  ✗ Gemini (API 키 미설정)
  ✓ Perplexity

API Connections:
  ✓ Claude
  ✓ OpenAI
  ✗ Gemini
  ✗ Perplexity (Error code 400)

Debate Test: ✓ Passed

⚠ 2/4 APIs working
```

---

### 🔬 validate_methodology.py

**위치**: `scripts/validate_methodology.py`

**기능**:
- 경제학적 방법론 검증
- Claude + Perplexity API로 교차 검증
- 검증 항목:
  - Stablecoin Risk 평가
  - MST Systemic Risk 분석

**실행**:
```bash
python scripts/validate_methodology.py
```

**출력**:
- JSON: `outputs/methodology_validation_YYYYMMDD.json`
- 콘솔: 검증 결과 요약

**검증 결과** (2026-01-09):
- Stablecoin Risk: PARTIALLY_CORRECT
- MST Systemic Risk: PARTIALLY_CORRECT

---

### 🏗️ validate_integration_design.py

**위치**: `scripts/validate_integration_design.py`

**기능**:
- 아키텍처 통합 설계 검증
- Claude + Perplexity로 옵션 비교
- Risk Enhancement Layer 설계 평가

**실행**:
```bash
python scripts/validate_integration_design.py
```

**출력**:
- JSON: `outputs/integration_design_validation_YYYYMMDD.json`
- 콘솔: 설계 권장사항

**검증 결과** (2026-01-09):
- 선택: Option C (Risk Enhancement Layer)
- 이유: 리스크 통합 용이, 업계 표준 패턴

---

### 🧪 tests/test_lib.py

**위치**: `tests/test_lib.py`

**기능**:
- lib 모듈 테스트

**실행**:
```bash
python tests/test_lib.py
```

---

### 🧪 tests/test_signal_action.py

**위치**: `tests/test_signal_action.py`

**기능**:
- 시그널-액션 연결 테스트

**실행**:
```bash
python tests/test_signal_action.py
```

---

### 🧪 tests/test_lasso_forecast.py

**위치**: `tests/test_lasso_forecast.py`

**기능**:
- LASSO 예측 모델 테스트

**실행**:
```bash
python tests/test_lasso_forecast.py
```

---

## 6. 유틸리티 & 도구

### 📅 scheduler.py

**위치**: `scripts/scheduler.py`

**기능**:
- 자동화 스케줄러
- Cron 작업 관리

**실행**:
```bash
python scripts/scheduler.py
```

**출력**:
- 로그: `outputs/scheduler.log`

---

### 📊 dashboard_generator.py

**위치**: `lib/dashboard_generator.py`

**기능**:
- Plotly 대시보드 생성 (Dash)
- HTML 대시보드 생성

**실행**:
```bash
# Plotly 대시보드 (Dash)
python lib/dashboard_generator.py
```

**출력**:
- 웹 대시보드: 포트 8050 (기본)

---

### 🔄 binance_stream.py

**위치**: `lib/binance_stream.py`

**기능**:
- Binance WebSocket 실시간 스트리밍
- VPIN, OFI 계산
- main.py --realtime에서 사용

**실행**:
```bash
# main.py --realtime 옵션으로 실행
python main.py --realtime --duration 60

# 독립 실행:
python -c "
from lib.binance_stream import BinanceStreamer
streamer = BinanceStreamer()
streamer.stream(duration=60)
"
```

**출력**:
- DB: `outputs/realtime_signals.db`
- 콘솔: 실시간 VPIN/OFI 값

---

## 📊 실행 빈도 권장사항

### 매일 실행 (평일)

**아침 (한국 시간 08:00)**:
```bash
python lib/intraday_collector.py
python lib/news_correlator.py
```

**저녁 (미국 장 마감 후, 한국 시간 06:00)**:
```bash
python scripts/daily_collector.py
python scripts/daily_analysis.py
python main.py --report
```

### 주말 실행

**매 시간**:
```bash
python lib/crypto_collector.py --detect
```

**4시간마다**:
```bash
python lib/news_correlator.py
```

### 주간 실행

**매주 월요일**:
```bash
python scripts/run_backtest.py
python tests/test_api_connection.py
```

---

## 🔧 Cron 설정 예시

```bash
# 평일 장 마감 후 (매일 17:00 EST)
0 17 * * 1-5 cd /home/tj/projects/autoai/eimas && python scripts/daily_collector.py >> logs/daily.log 2>&1

# 평일 아침 (매일 08:00 KST)
0 8 * * 1-5 cd /home/tj/projects/autoai/eimas && python lib/intraday_collector.py >> logs/intraday.log 2>&1

# 주말 암호화폐 모니터링 (매 시간)
0 * * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/crypto_collector.py --detect >> logs/crypto.log 2>&1

# 주말 뉴스 귀인 (4시간마다)
0 */4 * * 6,0 cd /home/tj/projects/autoai/eimas && python lib/news_correlator.py >> logs/correlator.log 2>&1

# 일요일 저녁 선물 체크 (월요일 08:00 KST = 일요일 18:00 EST)
0 8 * * 1 cd /home/tj/projects/autoai/eimas && python -c "from lib.news_correlator import WeekendAssetCollector; WeekendAssetCollector().collect_and_detect()" >> logs/weekend.log 2>&1
```

---

## 📈 lib/ 디렉토리 주요 모듈 (96개)

### 독립 실행 가능한 모듈 확인됨 (9개)

1. **intraday_collector.py** - 장중 데이터
2. **crypto_collector.py** - 암호화폐 모니터링
3. **market_data_pipeline.py** - 다중 API 데이터
4. **event_predictor.py** - 이벤트 예측
5. **event_attribution.py** - 이벤트 역추적
6. **news_correlator.py** - 뉴스 상관관계
7. **event_backtester.py** - 이벤트 백테스트
8. **dashboard_generator.py** - Plotly 대시보드
9. **binance_stream.py** - 실시간 스트리밍

### main.py에서 호출되는 모듈 (주요)

- **fred_collector.py** - FRED 데이터
- **data_collector.py** - 시장 데이터
- **data_loader.py** - RWA 자산
- **market_indicators.py** - 시장 지표
- **enhanced_data_sources.py** - DeFi, MENA
- **regime_detector.py** - 레짐 탐지
- **regime_analyzer.py** - GMM & Entropy
- **event_framework.py** - 이벤트 탐지
- **liquidity_analysis.py** - 유동성 분석
- **critical_path.py** - 리스크 분석
- **microstructure.py** - 시장 미세구조
- **bubble_detector.py** - 버블 탐지
- **etf_flow_analyzer.py** - ETF 플로우
- **genius_act_macro.py** - Genius Act
- **custom_etf_builder.py** - 테마 ETF
- **shock_propagation_graph.py** - 충격 전파
- **graph_clustered_portfolio.py** - GC-HRP
- **integrated_strategy.py** - 통합 전략
- **event_tracker.py** - 이벤트 추적
- **dual_mode_analyzer.py** - 모드 비교
- **ai_report_generator.py** - AI 리포트
- **whitening_engine.py** - 경제학 해석
- **autonomous_agent.py** - 팩트 체킹

### 유틸리티 모듈

- **backtest.py, backtest_engine.py, backtester.py** - 백테스트 엔진
- **paper_trader.py, paper_trading.py** - 페이퍼 트레이딩
- **alert_manager.py, alerts.py** - 알림 시스템
- **notifications.py, notifier.py** - 통지
- **trade_journal.py** - 거래 일지
- **risk_manager.py** - 리스크 관리
- **portfolio_optimizer.py** - 포트폴리오 최적화

---

## 📋 요약

### 실행 가능한 스크립트 요약

| 카테고리 | 스크립트 수 | 주요 기능 |
|---------|------------|----------|
| 데이터 수집 | 5개 | intraday, daily, crypto, market_data, fred |
| 분석 | 4개 | daily_analysis, event_predictor, event_attribution, news_correlator |
| 이벤트 | 2개 | event_tracker, event_backtester |
| 백테스트 | 2개 | run_backtest, paper_trader |
| 검증 | 5개 | test_api, validate_methodology, validate_integration, test_lib, test_signal |
| 유틸리티 | 3개 | scheduler, dashboard, binance_stream |

**총 독립 실행 가능**: 21개 스크립트

---

**문서 생성일**: 2026-01-12
**작성자**: EIMAS Documentation System
**버전**: 1.0
