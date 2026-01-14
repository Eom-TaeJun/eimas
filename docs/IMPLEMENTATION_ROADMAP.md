# EIMAS Implementation Roadmap
> Economic Intelligence Multi-Agent System - 구현 로드맵

**Last Updated:** 2025-12-31

---

## 현재 시스템 상태

### 구현 완료 (35개 모듈)

| 카테고리 | 모듈 | 설명 | 상태 |
|---------|------|------|------|
| **데이터 수집** | `data_collector.py` | yfinance/FRED 기본 수집 | ✅ |
| | `fred_collector.py` | FRED API 전용 수집기 | ✅ |
| | `enhanced_data_sources.py` | FedWatch, 경제캘린더 | ✅ |
| | `unified_data_store.py` | 통합 데이터 저장소 | ✅ |
| | `ark_holdings_analyzer.py` | ARK ETF 보유종목 | ✅ |
| **분석** | `causal_network.py` | Granger Causality 네트워크 | ✅ |
| | `critical_path.py` | 리스크/불확실성 분석 | ✅ |
| | `critical_path_monitor.py` | 17개 Critical Path 모니터 | ✅ |
| | `lasso_model.py` | LASSO 회귀 분석 | ✅ |
| | `etf_flow_analyzer.py` | ETF 자금흐름 분석 | ✅ |
| | `market_indicators.py` | 밸류에이션/크레딧/VIX | ✅ |
| | `correlation_monitor.py` | 상관관계 모니터링 | ✅ |
| **시그널** | `regime_detector.py` | 시장 레짐 감지 | ✅ |
| | `etf_signal_generator.py` | ETF→시그널 변환 | ✅ |
| | `signal_pipeline.py` | 시그널 통합 파이프라인 | ✅ |
| | `leading_indicator_tester.py` | 선행지표 Granger 테스트 | ✅ |
| **포트폴리오** | `trading_db.py` | 시그널/포트폴리오 DB | ✅ |
| | `risk_manager.py` | 리스크 관리/사이징 | ✅ |
| | `session_analyzer.py` | 장중 세션 분석 | ✅ |
| | `feedback_tracker.py` | 시그널 성과 피드백 | ✅ |
| **백테스트** | `backtester.py` | 전략 백테스트 엔진 | ✅ |
| **알림/리포트** | `notifier.py` | Telegram/Slack 알림 | ✅ |
| | `alert_manager.py` | 시그널→알림 통합 | ✅ |
| | `dashboard_generator.py` | 기존 대시보드 | ✅ |
| | `report_generator.py` | HTML 리포트 생성 | ✅ |
| **포트폴리오** | `portfolio_optimizer.py` | Mean-Variance, Black-Litterman | ✅ |
| | `sector_rotation.py` | 경기 사이클 섹터 로테이션 | ✅ |
| | `paper_trader.py` | 페이퍼 트레이딩 시스템 | ✅ |
| | `performance_attribution.py` | Brinson 성과 분해 | ✅ |
| **고급 분석** | `options_flow.py` | 옵션 플로우 (Put/Call, Gamma) | ✅ |
| | `sentiment_analyzer.py` | Fear & Greed, News 센티먼트 | ✅ |
| | `factor_analyzer.py` | Fama-French 팩터 분석 | ✅ |
| **인프라** | `api/main.py` | FastAPI 대시보드 서버 | ✅ |
| | `cli/eimas.py` | CLI 도구 (8개 명령어) | ✅ |
| | `scripts/scheduler.py` | 자동화 스케줄러 (7개 태스크) | ✅ |

---

## Phase 5: 포트폴리오 최적화 (다음 구현)

### 5.1 Mean-Variance Optimizer
> 현대 포트폴리오 이론 기반 최적화

```
lib/portfolio_optimizer.py
├── MeanVarianceOptimizer
│   ├── optimize_sharpe()        # 샤프비율 최대화
│   ├── optimize_min_variance()  # 최소 분산
│   ├── optimize_risk_parity()   # 리스크 패리티
│   └── efficient_frontier()     # 효율적 프론티어
├── BlackLittermanModel
│   ├── set_views()              # 투자자 전망 설정
│   ├── posterior_returns()      # 사후 기대수익
│   └── optimize()               # BL 최적화
└── Constraints
    ├── box_constraints          # 개별 자산 한도
    ├── sector_constraints       # 섹터 한도
    └── turnover_constraints     # 회전율 제한
```

**핵심 기능:**
- 샤프비율 최대화 포트폴리오
- 최소 분산 포트폴리오
- Black-Litterman 모델 (시그널 반영)
- 제약조건 처리 (cvxpy 사용)

### 5.2 Factor Exposure Analyzer
> Fama-French 팩터 분석

```
lib/factor_analyzer.py
├── FactorExposureAnalyzer
│   ├── calculate_factor_loadings()  # 팩터 로딩
│   ├── factor_attribution()         # 성과 분해
│   ├── style_drift_detection()      # 스타일 드리프트
│   └── factor_risk_decomposition()  # 팩터 리스크
├── Factors
│   ├── market (MKT-RF)
│   ├── size (SMB)
│   ├── value (HML)
│   ├── momentum (UMD)
│   ├── quality (QMJ)
│   └── low_volatility (BAB)
└── Reports
    └── factor_exposure_report()
```

### 5.3 Tax Loss Harvester
> 세금 최적화 전략

```
lib/tax_optimizer.py
├── TaxLossHarvester
│   ├── identify_loss_candidates()   # 손실 매도 후보
│   ├── find_replacement_etf()       # 대체 ETF 찾기
│   ├── wash_sale_check()            # Wash Sale 체크
│   ├── calculate_tax_savings()      # 절세 효과
│   └── execute_harvest()            # 하베스팅 실행
└── Constants
    ├── WASH_SALE_DAYS = 30
    ├── SIMILAR_ETF_PAIRS            # 유사 ETF 쌍
    └── TAX_RATES                    # 세율 설정
```

---

## Phase 6: 실시간 모니터링 (중요)

### 6.1 Real-time Dashboard Server
> FastAPI 기반 실시간 대시보드

```
api/
├── main.py                    # FastAPI 앱
├── routes/
│   ├── signals.py             # /api/signals
│   ├── portfolio.py           # /api/portfolio
│   ├── risk.py                # /api/risk
│   └── websocket.py           # WebSocket 실시간
├── services/
│   ├── signal_service.py
│   └── portfolio_service.py
└── static/
    └── dashboard.html         # React/Vue SPA

Endpoints:
GET  /api/signals/latest       # 최신 시그널
GET  /api/portfolio/current    # 현재 포트폴리오
GET  /api/risk/metrics         # 리스크 지표
GET  /api/correlation/matrix   # 상관관계 행렬
WS   /ws/realtime              # 실시간 업데이트
```

### 6.2 CLI Tool
> 명령줄 인터페이스

```
cli/
├── eimas.py                   # 메인 CLI
├── commands/
│   ├── signal.py              # eimas signal [list|generate]
│   ├── portfolio.py           # eimas portfolio [show|optimize]
│   ├── risk.py                # eimas risk [check|report]
│   ├── backtest.py            # eimas backtest [run|compare]
│   └── report.py              # eimas report [daily|weekly]
└── utils/
    └── display.py             # Rich 기반 출력

Usage:
$ eimas signal list --today
$ eimas portfolio optimize --profile aggressive
$ eimas risk check --holdings portfolio.json
$ eimas backtest run --strategy yield_curve --period 2y
$ eimas report daily --output html
```

### 6.3 Webhook Integration
> 외부 서비스 연동

```
lib/webhook_manager.py
├── WebhookManager
│   ├── register_webhook()           # 웹훅 등록
│   ├── send_to_discord()            # Discord 알림
│   ├── send_to_email()              # 이메일 알림
│   ├── send_to_custom()             # 커스텀 웹훅
│   └── trigger_on_event()           # 이벤트 트리거
└── Templates
    ├── signal_template.json
    ├── risk_template.json
    └── daily_summary_template.json
```

---

## Phase 7: 고급 분석

### 7.1 Options Flow Analyzer
> 옵션 플로우 분석

```
lib/options_flow.py
├── OptionsFlowAnalyzer
│   ├── get_unusual_activity()       # 이상 거래량
│   ├── put_call_ratio()             # Put/Call 비율
│   ├── gamma_exposure()             # 감마 익스포저
│   ├── max_pain_level()             # Max Pain 레벨
│   └── smart_money_flow()           # 스마트 머니 추적
└── Signals
    ├── bullish_sweep_detection()
    ├── bearish_sweep_detection()
    └── institutional_flow()
```

### 7.2 Sentiment Analyzer
> 뉴스/소셜 미디어 센티먼트

```
lib/sentiment_analyzer.py
├── NewsSentiment
│   ├── fetch_financial_news()       # 금융 뉴스 수집
│   ├── analyze_sentiment()          # 센티먼트 분석
│   └── topic_extraction()           # 토픽 추출
├── SocialSentiment
│   ├── reddit_wsb_sentiment()       # WSB 센티먼트
│   ├── twitter_fintwit()            # FinTwit 분석
│   └── stocktwits_sentiment()       # StockTwits
└── AggregatedSentiment
    └── composite_score()            # 통합 점수
```

### 7.3 Sector Rotation Model
> 섹터 로테이션 전략

```
lib/sector_rotation.py
├── SectorRotationModel
│   ├── calculate_momentum()         # 섹터 모멘텀
│   ├── economic_cycle_mapping()     # 경기 사이클 매핑
│   ├── relative_strength()          # 상대 강도
│   ├── rotation_signal()            # 로테이션 시그널
│   └── optimal_sector_weights()     # 최적 섹터 비중
├── Sectors
│   ├── XLK (Technology)
│   ├── XLF (Financials)
│   ├── XLV (Healthcare)
│   ├── XLE (Energy)
│   ├── XLI (Industrials)
│   └── ... (11 SPDR Sectors)
└── CyclePhases
    ├── early_expansion
    ├── mid_expansion
    ├── late_expansion
    └── recession
```

---

## Phase 8: 실행 및 자동화

### 8.1 Paper Trading System
> 시뮬레이션 트레이딩

```
lib/paper_trader.py
├── PaperTrader
│   ├── initialize_account()         # 가상 계좌 초기화
│   ├── execute_order()              # 주문 실행 (시뮬)
│   ├── get_positions()              # 포지션 조회
│   ├── get_pnl()                    # 손익 계산
│   ├── get_transaction_history()    # 거래 내역
│   └── sync_with_signals()          # 시그널 자동 실행
├── OrderTypes
│   ├── market_order
│   ├── limit_order
│   └── stop_loss_order
└── Reports
    └── daily_pnl_report()
```

### 8.2 Scheduler & Automation
> 자동화 스케줄러

```
scripts/scheduler.py
├── EIMASScheduler
│   ├── schedule_daily_analysis()    # 일일 분석 (6:00 AM)
│   ├── schedule_signal_check()      # 시그널 체크 (매시간)
│   ├── schedule_risk_monitor()      # 리스크 모니터 (15분)
│   ├── schedule_rebalance_check()   # 리밸런스 체크 (주간)
│   └── schedule_report_generation() # 리포트 생성 (9:00 PM)
└── Triggers
    ├── on_market_open()
    ├── on_market_close()
    ├── on_signal_generated()
    └── on_risk_threshold_breach()
```

### 8.3 Performance Attribution
> Brinson 성과 분해

```
lib/performance_attribution.py
├── BrinsonAttribution
│   ├── allocation_effect()          # 배분 효과
│   ├── selection_effect()           # 종목선정 효과
│   ├── interaction_effect()         # 상호작용 효과
│   └── total_attribution()          # 전체 분해
├── RiskAttribution
│   ├── marginal_var_contribution()  # 한계 VaR
│   ├── component_var()              # 구성요소 VaR
│   └── risk_budget_analysis()       # 리스크 버짓
└── Reports
    └── attribution_report()
```

---

## Phase 9: 인프라 개선

### 9.1 Data Pipeline
> 데이터 파이프라인 개선

```
data/
├── pipeline.py
│   ├── DataPipeline
│   │   ├── extract()                # 데이터 추출
│   │   ├── transform()              # 변환
│   │   ├── validate()               # 검증
│   │   └── load()                   # 적재
│   └── IncrementalUpdate
│       └── update_since_last()      # 증분 업데이트
├── cache/
│   └── redis_cache.py               # Redis 캐싱
└── quality/
    └── data_quality_check.py        # 데이터 품질 체크
```

### 9.2 Logging & Monitoring
> 로깅 및 모니터링

```
core/
├── logging_config.py
│   ├── setup_logging()              # 로깅 설정
│   ├── StructuredLogger             # 구조화 로깅
│   └── log_to_file()                # 파일 로깅
├── metrics.py
│   ├── track_signal_latency()       # 시그널 지연
│   ├── track_api_calls()            # API 호출 수
│   └── track_error_rate()           # 에러율
└── health_check.py
    └── system_health_status()       # 시스템 상태
```

### 9.3 Configuration Management
> 설정 관리

```
configs/
├── base.yaml                        # 기본 설정
├── production.yaml                  # 프로덕션
├── development.yaml                 # 개발
└── config_loader.py
    ├── load_config()
    ├── validate_config()
    └── merge_configs()
```

---

## 구현 우선순위

### ✅ 완료됨 (Phase 5) - 2025-12-31
1. **portfolio_optimizer.py** - Mean-Variance, Black-Litterman 최적화 ✅
2. **sector_rotation.py** - 경기 사이클 기반 섹터 로테이션 ✅
3. **paper_trader.py** - 페이퍼 트레이딩 시스템 ✅

### ✅ 완료됨 (Phase 6) - 2025-12-31
4. **api/main.py** - FastAPI 실시간 대시보드 서버 ✅
5. **cli/eimas.py** - CLI 도구 (8개 명령어) ✅
6. **scripts/scheduler.py** - 자동화 스케줄러 (7개 태스크) ✅

### ✅ 완료됨 (Phase 7-8) - 2025-12-31
7. **options_flow.py** - 옵션 플로우 분석 (Put/Call, Gamma Exposure) ✅
8. **sentiment_analyzer.py** - 센티먼트 분석 (Fear & Greed, News) ✅
9. **performance_attribution.py** - Brinson 성과 분해 ✅
10. **factor_analyzer.py** - Fama-French 팩터 분석 ✅

### ✅ 완료됨 (Phase 9) - 2026-01-01
11. **data/pipeline.py** - ETL 데이터 파이프라인 ✅
12. **data/cache.py** - Redis/File 캐싱 시스템 ✅
13. **core/logging_config.py** - 구조화 로깅 (JSON) ✅
14. **core/health_check.py** - 시스템 헬스체크 ✅
15. **tests/test_lib.py** - 단위 테스트 (35개) ✅

---

## 현재 알려진 이슈

### Minor Bugs
1. `signal_pipeline.py`: Critical Path 에러 (`'int' object is not iterable`)
2. `signal_pipeline.py`: CryptoMetrics `fear_greed_level` 속성 누락
3. `signal_pipeline.py`: VIXMetrics `spot` 속성 누락
4. `risk_manager.py`: VaR 계산 스케일 오류 (% vs 절대값)

### 개선 필요
1. 에러 핸들링 강화
2. 로깅 체계화
3. 단위 테스트 추가
4. 문서화 보강

---

## 파일 구조 (목표)

```
eimas/
├── agents/                    # AI 에이전트
├── api/                       # FastAPI 서버 (NEW)
│   ├── main.py
│   ├── routes/
│   └── services/
├── cli/                       # CLI 도구 (NEW)
│   ├── eimas.py
│   └── commands/
├── configs/                   # 설정
├── core/                      # 핵심 프레임워크
├── data/                      # 데이터 파이프라인 (NEW)
├── docs/                      # 문서
├── lib/                       # 라이브러리 (25개 → 35개)
│   ├── [기존 25개 모듈]
│   ├── portfolio_optimizer.py # NEW
│   ├── factor_analyzer.py     # NEW
│   ├── sector_rotation.py     # NEW
│   ├── options_flow.py        # NEW
│   ├── sentiment_analyzer.py  # NEW
│   ├── paper_trader.py        # NEW
│   ├── tax_optimizer.py       # NEW
│   ├── performance_attribution.py # NEW
│   ├── webhook_manager.py     # NEW
│   └── data_quality.py        # NEW
├── outputs/                   # 결과물
├── scripts/                   # 스크립트
│   ├── daily_analysis.py
│   ├── run_backtest.py
│   └── scheduler.py           # NEW
├── tests/                     # 테스트 (NEW)
└── main.py
```

---

## 다음 단계

**다음 구현 항목 (Phase 10 - 향후):**
1. 실시간 WebSocket 스트리밍
2. 더 많은 백테스트 전략
3. ML 기반 시그널 예측
4. 프로덕션 Docker 배포

**현재까지 구현된 모듈: 40개**
- lib/: 32개 분석 모듈
- data/: 2개 (파이프라인, 캐싱)
- core/: 4개 (설정, 스키마, 로깅, 헬스체크)
- api/: FastAPI 서버 (17개 엔드포인트)
- cli/: CLI 도구 (8개 명령어)
- scripts/: 스케줄러 (7개 태스크)
- tests/: 단위 테스트 (35개 테스트 케이스)

**Phase 5-9 완료:**

| Phase | 모듈 | 완료일 |
|-------|------|--------|
| 5 | portfolio_optimizer, sector_rotation, paper_trader | 2025-12-31 |
| 6 | api/main.py, cli/eimas.py, scripts/scheduler.py | 2025-12-31 |
| 7-8 | options_flow, sentiment, attribution, factor | 2025-12-31 |
| 9 | pipeline, cache, logging, health_check, tests | 2026-01-01 |
