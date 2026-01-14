# EIMAS Module Status

> 마지막 업데이트: 2026-01-10

## main.py 연결 현황

### 연결됨 (31개)
```
adaptive_agents, autonomous_agent, binance_stream, bubble_detector, causal_network,
causality_graph, correlation_monitor, critical_path, custom_etf_builder,
data_collector, dual_mode_analyzer, etf_flow_analyzer, event_db,
event_framework, event_tracker, extended_data_sources, fred_collector,
genius_act_macro, graph_clustered_portfolio, integrated_strategy,
liquidity_analysis, market_indicators, microstructure, predictions_db,
realtime_pipeline, regime_analyzer, regime_detector, shock_propagation_graph,
trading_db, unified_data_store, validation_agents, volume_analyzer, whitening_engine
```

### 오늘 연결됨 (2026-01-10)
| 모듈 | Phase | 설명 |
|------|-------|------|
| `adaptive_agents` | 3.4 | 동적 리스크 에이전트 (Aggressive/Balanced/Conservative) |
| `validation_agents` | 3.4 | AI 검증 + 피드백 루프 (Claude/Perplexity/GPT/Gemini) |
| `extended_data_sources` | 1.5 | DeFiLlama (TVL, Stablecoin), MENA Markets |
| `event_tracker` | 2.12 | 거래량/가격 이상 → 시점 역추적 → 뉴스 검색 |

### 미연결 (55개)

#### 미사용 (deprecated)
| 모듈 | 설명 | 상태 |
|------|------|------|
| `risk_profile_agents` | 정적 리스크 프로파일 | adaptive_agents로 대체됨 |

#### 실행/트레이딩
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `paper_trader` | 가상 거래 실행 | HIGH |
| `paper_trading` | 페이퍼 트레이딩 유틸 | HIGH |
| `position_sizing` | 포지션 사이징 | MEDIUM |
| `trade_journal` | 거래 일지 | LOW |

#### 백테스팅
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `backtest` | 기본 백테스트 | MEDIUM |
| `backtest_engine` | 백테스트 엔진 | MEDIUM |
| `backtester` | 백테스터 클래스 | MEDIUM |
| `event_backtester` | 이벤트 기반 백테스트 | LOW |

#### 리스크 관리
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `risk_manager` | 리스크 관리자 | HIGH |
| `risk_analytics` | 리스크 분석 (VaR, CVaR) | MEDIUM |

#### 알림
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `alert_manager` | 알림 관리자 | MEDIUM |
| `alerts` | 알림 유틸 | MEDIUM |
| `notifications` | 노티피케이션 | LOW |
| `notifier` | 알림 발송 | LOW |

#### 대시보드/리포트
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `dashboard_generator` | HTML 대시보드 생성 | LOW |
| `report_generator` | 리포트 생성 | LOW |

#### 데이터 소스
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `data_loader` | RWA 자산 로더 | MEDIUM |
| `earnings` | 실적 데이터 | LOW |
| `economic_calendar` | 경제 캘린더 | LOW |
| `intraday_collector` | 인트라데이 데이터 | LOW |
| `crypto_collector` | 크립토 데이터 | LOW |

#### 대안 데이터
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `insider_trading` | SEC 내부자 거래 | LOW |
| `options_flow` | 옵션 플로우 | MEDIUM |
| `news_correlator` | 뉴스 상관관계 | LOW |
| `sentiment` | 센티먼트 기본 | LOW |
| `sentiment_analyzer` | 센티먼트 분석 | LOW |

#### 전략/최적화
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `lasso_model` | LASSO 예측 모델 | MEDIUM |
| `hrp_optimizer` | HRP 최적화 | LOW |
| `portfolio_optimizer` | 포트폴리오 최적화 | LOW |
| `factor_analyzer` | 팩터 분석 | LOW |
| `factor_exposure` | 팩터 익스포저 | LOW |
| `sector_rotation` | 섹터 로테이션 | LOW |
| `pairs_trading` | 페어 트레이딩 | LOW |
| `mean_reversion` | 평균 회귀 | LOW |
| `seasonality` | 계절성 | LOW |
| `patterns` | 패턴 인식 | LOW |
| `multi_asset` | 멀티 에셋 | LOW |

#### 기타
| 모듈 | 설명 | 우선순위 |
|------|------|----------|
| `ark_holdings_analyzer` | ARK ETF 보유 분석 | LOW |
| `asset_universe` | 자산 유니버스 | LOW |
| `critical_path_monitor` | 크리티컬 패스 모니터 | LOW |
| `debate_agent` | 토론 에이전트 | LOW |
| `enhanced_data_sources` | 확장 데이터 | LOW |
| `etf_signal_generator` | ETF 시그널 | LOW |
| `event_attribution` | 이벤트 귀인 | LOW |
| `event_predictor` | 이벤트 예측 | LOW |
| `feedback_tracker` | 피드백 추적 | LOW |
| `insight_discussion` | 인사이트 토론 | LOW |
| `leading_indicator_tester` | 선행지표 테스트 | LOW |
| `macro_analyzer` | 매크로 분석 | LOW |
| `market_data_pipeline` | 시장 데이터 파이프라인 | LOW |
| `performance_attribution` | 성과 귀인 | LOW |
| `regime_history` | 레짐 히스토리 | LOW |
| `session_analyzer` | 세션 분석 | LOW |
| `signal_pipeline` | 시그널 파이프라인 | LOW |
| `tax_optimizer` | 세금 최적화 | LOW |

---

## 미구현 기능 (파일 없음)

| 기능 | 설명 | 우선순위 |
|------|------|----------|
| **Broker API** | Alpaca/IBKR 실제 거래 연동 | HIGH |
| **Stress Testing** | Black Swan 시나리오 시뮬레이션 | MEDIUM |
| **Web UI** | Streamlit/Gradio 대시보드 | LOW |
| **P&L Tracker** | 실시간 손익 추적 + 리포트 | HIGH |

---

## 권장 연결 순서

### Phase 1: 포트폴리오 에이전트 통합
1. `adaptive_agents` → main.py Phase 3 이후 추가
2. `validation_agents` → adaptive_agents 결정 검증
3. `extended_data_sources` → Phase 1에 데이터 추가

### Phase 2: 실행 레이어
4. `paper_trader` + `paper_trading` → 가상 거래 실행
5. `risk_manager` → 리스크 한도 관리
6. P&L Tracker 신규 구현

### Phase 3: 백테스팅
7. `backtest_engine` → 전략 검증
8. Stress Testing 신규 구현

### Phase 4: 알림/대시보드
9. `alert_manager` → 실시간 알림
10. Web UI 신규 구현

---

## 실행 명령어 참고

```bash
# 기본 실행
python main.py

# 빠른 분석
python main.py --quick

# AI 리포트 포함
python main.py --report

# 실시간 스트리밍
python main.py --realtime --duration 60

# 서버/크론용
python main.py --cron --output /path/to/outputs
```

---

## Agent 목록 (2026-01-10 기준)

### lib/adaptive_agents.py
- `AggressiveAdaptiveAgent` - 동적 공격형 (base risk 70, range 25-90)
- `BalancedAdaptiveAgent` - 동적 균형형 (base risk 50, range 10-85)
- `ConservativeAdaptiveAgent` - 동적 보수형 (base risk 30, range 10-60)

### lib/validation_agents.py
- `ClaudeValidationAgent` - claude-opus-4-5-20251101
- `PerplexityValidationAgent` - sonar-pro
- `GeminiValidationAgent` - gemini-2.5-pro-exp-03-25
- `GPTValidationAgent` - o1
- `FeedbackValidationAgent` - Claude + Perplexity 피드백 검증
- `ValidationLoopManager` - 최대 3라운드 피드백 루프

### lib/extended_data_sources.py
- `DeFiLlamaCollector` - TVL, Stablecoin, Yields
- `MiddleEastMarketCollector` - KSA, UAE, TUR, QAT ETFs
- `ExtendedDataCollector` - 통합 수집기

### lib/event_tracker.py (2026-01-10 추가)
- `EventTracker` - 거래량/가격 이상 시점 역추적 + 뉴스 연결
- `TrackedEvent` - 역추적된 이벤트 데이터
- `EventTrackingResult` - 추적 결과
- Flow: Volume/Price Anomaly → Timestamp → Perplexity News Search → Event Attribution
