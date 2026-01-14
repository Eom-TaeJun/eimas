# EIMAS lib/ 모듈 가이드

> 업데이트: 2026-01-14
> 총 모듈: 54개 (Active) + 9개 (Deprecated) + 30개 (Future)

---

## 📦 모듈 분류

### ✅ 통합 모듈 (37개) - main.py에서 사용

#### Phase 1: Data Collection (5개)
| 모듈 | 기능 | main.py 위치 |
|------|------|-------------|
| `fred_collector.py` | FRED 데이터 (RRP, TGA, Net Liquidity, Fed Funds, Spreads) | Phase 1.1 (항상) |
| `data_collector.py` | 시장 데이터 (24 tickers + Crypto + RWA) | Phase 1.2 (항상) |
| `unified_data_store.py` | 통합 데이터 저장소 | Phase 1.2 |
| `market_indicators.py` | VIX, Fear & Greed Index | Phase 1.4 (not quick_mode) |
| `extended_data_sources.py` | DeFiLlama TVL, MENA Markets, On-chain signals | Phase 1.5 (not quick_mode) |

#### Phase 2: Analysis (19개)
| 모듈 | 기능 | main.py 위치 |
|------|------|-------------|
| `regime_detector.py` | 레짐 탐지 (Bull/Bear/Neutral) | Phase 2.1 (항상) |
| `regime_analyzer.py` | GMM 3-state + Shannon Entropy | Phase 2.1.1 (not quick_mode) |
| `event_framework.py` | 이벤트 탐지 (유동성, 시장) | Phase 2.2 (항상) |
| `liquidity_analysis.py` | Granger Causality 분석 | Phase 2.3 (not quick_mode) |
| `causal_network.py` | 인과관계 네트워크 (구버전) | ⚠️ deprecated 참조 |
| `critical_path.py` | Critical Path 리스크 점수 (Base) | Phase 2.4 (항상) |
| `correlation_monitor.py` | 상관관계 모니터링 | Phase 1.6 (항상) |
| `etf_flow_analyzer.py` | ETF 플로우 + 섹터 로테이션 | Phase 2.5 (not quick_mode) |
| `microstructure.py` | VPIN, Amihud Lambda, Roll Spread | Phase 2.4.1 (not quick_mode) |
| `bubble_detector.py` | Greenwood-Shleifer 버블 탐지 | Phase 2.4.2 (not quick_mode) |
| `genius_act_macro.py` | 스테이블코인-유동성 + Crypto Stress Test | Phase 2.6, 2.6.1 (not quick_mode) |
| `custom_etf_builder.py` | Theme ETF + Supply Chain Graph | Phase 2.7 (not quick_mode) |
| `causality_graph.py` | CausalityGraphEngine (고급 그래프 분석) | Phase 2.7 (not quick_mode) |
| `causality_narrative.py` | 인과관계 Narrative 자연어 생성 | Phase 2.7 (not quick_mode) |
| `shock_propagation_graph.py` | 충격 전파 그래프 (Granger 기반) | Phase 2.8 (not quick_mode) |
| `graph_clustered_portfolio.py` | GC-HRP 포트폴리오 최적화 + MST | Phase 2.9 (not quick_mode) |
| `integrated_strategy.py` | 통합 전략 엔진 (Portfolio + Causality) | Phase 2.10 (not quick_mode) |
| `volume_analyzer.py` | 거래량 이상 탐지 (Kyle 1985) | Phase 2.11 (not quick_mode) |
| `event_tracker.py` | 이상→뉴스 역추적 (Perplexity 연동) | Phase 2.12 (not quick_mode) |

#### Phase 3: Multi-Agent Debate (4개)
| 모듈 | 기능 | main.py 위치 |
|------|------|-------------|
| `agents/orchestrator.py` | MetaOrchestrator (FULL/REF Mode) | Phase 3.1, 3.2 (항상) |
| `dual_mode_analyzer.py` | 모드 비교 및 최종 권고 | Phase 3.3 (항상) |
| `adaptive_agents.py` | Adaptive Portfolio Agents (Aggressive/Balanced/Conservative) | Phase 3.4 (not quick_mode) |
| `validation_agents.py` | Validation Loop (Claude + Perplexity) | Phase 3.4.1 (not quick_mode) |

#### Phase 4: Real-time (2개)
| 모듈 | 기능 | main.py 위치 |
|------|------|-------------|
| `binance_stream.py` | Binance WebSocket 스트리밍 | Phase 4.1 (enable_realtime) |
| `realtime_pipeline.py` | VPIN 실시간 계산 + Signal DB | Phase 4.1 (enable_realtime) |

#### Phase 5: Database (3개)
| 모듈 | 기능 | main.py 위치 |
|------|------|-------------|
| `event_db.py` | Event Database 저장 | Phase 5.1 (항상) |
| `realtime_pipeline.py` | Signal Database 저장 | Phase 5.2 (항상) |
| `predictions_db.py` | Predictions Database (검증용) | Phase 5.2.1 (항상) |

#### Phase 6: AI Report (1개)
| 모듈 | 기능 | main.py 위치 |
|------|------|-------------|
| `ai_report_generator.py` | AI 리포트 생성 (Claude/Perplexity) | Phase 6.1 (generate_report) |

#### Phase 7: Quality (2개)
| 모듈 | 기능 | main.py 위치 |
|------|------|-------------|
| `whitening_engine.py` | 경제학적 해석 (Whitening) | Phase 7.1 (generate_report and not quick_mode) |
| `autonomous_agent.py` | AI 팩트체킹 (AutonomousFactChecker) | Phase 7.2 (generate_report and not quick_mode) |

---

### 🚀 독립 스크립트 (7개) - 별도 실행

이 모듈들은 main.py와 독립적으로 실행 가능하며, COMMANDS.md에 상세히 문서화되어 있습니다.

| 모듈 | 용도 | 실행 방법 |
|------|------|----------|
| `intraday_collector.py` | 1분봉 장중 데이터 수집 + 이상 탐지 | `python lib/intraday_collector.py [--date YYYY-MM-DD] [--backfill]` |
| `crypto_collector.py` | 24/7 암호화폐 모니터링 + 이상 탐지 | `python lib/crypto_collector.py --detect [--analyze]` |
| `market_data_pipeline.py` | 다중 API 데이터 수집 (Twelve Data, CryptoCompare, yfinance) | `python lib/market_data_pipeline.py --all [--with-oil]` |
| `event_predictor.py` | 경제 이벤트 예측 (NFP, CPI, FOMC) | Python 코드로 호출: `EventPredictor().generate_report()` |
| `event_attribution.py` | 이벤트 원인 분석 (Perplexity 연동) | Python 코드로 호출: `EventAttributor().analyze_recent_events()` |
| `event_backtester.py` | 이벤트 백테스트 (과거 FOMC, CPI, NFP) | `python lib/event_backtester.py` |
| `news_correlator.py` | 이상-뉴스 자동 귀인 (24시간 이상 → 뉴스 연결) | `python lib/news_correlator.py` |

**주말/일일 운영 예시:**
```bash
# 주말 암호화폐 모니터링 (Cron)
0 * * * 6,0 cd /path/to/eimas && python lib/crypto_collector.py --detect >> logs/crypto.log 2>&1

# 장중 데이터 수집 (매일 아침)
python lib/intraday_collector.py --backfill

# 이상-뉴스 귀인 (평일 저녁)
python lib/news_correlator.py
```

---

### 🛠️ 지원 모듈 (10개) - 유틸리티

main.py에서 직접 호출되지 않지만, 시스템 운영에 필요한 모듈들입니다.

| 모듈 | 용도 | 비고 |
|------|------|------|
| `ark_holdings_analyzer.py` | ARK Holdings 데이터 분석 | 별도 분석 시 사용 |
| `asset_universe.py` | 자산 유니버스 관리 | 티커 목록 관리 |
| `dashboard_generator.py` | Plotly 대시보드 생성 | `python plus/dashboard_generator.py` |
| `critical_path_monitor.py` | CriticalPath 모니터링 | critical_path.py 확장 기능 |
| `lasso_model.py` | LASSO 예측 모델 | agents/forecast_agent.py에서 사용 가능 |
| `trading_db.py` | 트레이딩 DB 스키마 | realtime_pipeline.py로 대체 중 |
| `report_generator.py` | 리포트 생성 (구버전) | ai_report_generator.py 권장 |
| `insight_discussion.py` | 인사이트 토론 | 미사용 |
| `risk_profile_agents.py` | 리스크 프로필 에이전트 | adaptive_agents.py로 대체 중 |
| `macro_analyzer.py` | 매크로 분석 | genius_act_macro.py로 통합 중 |

---

### ⚠️ 중복 가능성 (6개) - 검토 필요

다음 모듈들은 기능이 중복될 수 있으므로 향후 통합 검토가 필요합니다.

| 모듈 1 | 모듈 2 | 상태 | 권장 |
|--------|--------|------|------|
| `etf_signal_generator.py` | `etf_flow_analyzer.py` | 중복? | 통합 검토 |
| `macro_analyzer.py` | `genius_act_macro.py` | 중복? | 통합 검토 |
| `sector_rotation.py` | `etf_flow_analyzer.py` | 중복? | 통합 검토 |
| `signal_pipeline.py` | `integrated_strategy.py` | 중복? | 통합 검토 |
| `risk_profile_agents.py` | `adaptive_agents.py` | 중복? | 통합 검토 |
| `report_generator.py` | `ai_report_generator.py` | 구버전 | ai_report_generator.py 사용 |

---

## 🗂️ deprecated/ - 구버전 (9개)

main.py에서 더 이상 사용하지 않거나 다른 모듈로 대체된 모듈들입니다.

| 모듈 | 이유 | 대체 모듈 |
|------|------|----------|
| `causal_network.py` | 구버전 | `causality_graph.py` |
| `enhanced_data_sources.py` | 구버전 | `extended_data_sources.py` |
| `data_loader.py` | 통합됨 | `data_collector.py` (RWA 지원) |
| `debate_agent.py` | 구버전 | `agents/orchestrator.py` |
| `hrp_optimizer.py` | 통합됨 | `graph_clustered_portfolio.py` |
| `portfolio_optimizer.py` | 대체됨 | `graph_clustered_portfolio.py` |
| `backtest.py` | 별도 실행 | `scripts/run_backtest.py` |
| `backtest_engine.py` | 별도 실행 | `scripts/run_backtest.py` |
| `backtester.py` | 별도 실행 | `scripts/run_backtest.py` |

---

## 🔮 future/ - 미구현 (30개)

향후 구현 예정이거나 현재 사용하지 않는 기능들입니다.

### 데이터 소스 (3개)
- `earnings.py` - 실적 발표 데이터
- `economic_calendar.py` - 경제 캘린더
- `insider_trading.py` - 내부자 거래 데이터

### 분석 기능 (8개)
- `factor_analyzer.py` - 팩터 분석 (Fama-French)
- `factor_exposure.py` - 팩터 노출도
- `sentiment.py` - 감성 분석
- `sentiment_analyzer.py` - 감성 분석 v2
- `geopolitical_risk_detector.py` - 지정학적 리스크
- `leading_indicator_tester.py` - 선행지표 테스트
- `seasonality.py` - 계절성 분석
- `patterns.py` - 패턴 인식

### 전략 (3개)
- `mean_reversion.py` - 평균회귀 전략
- `pairs_trading.py` - 페어 트레이딩
- `options_flow.py` - 옵션 플로우 분석

### 포트폴리오 관리 (5개)
- `position_sizing.py` - 포지션 사이징
- `tax_optimizer.py` - 세금 최적화
- `performance_attribution.py` - 성과 귀인
- `risk_manager.py` - 리스크 관리
- `risk_analytics.py` - 리스크 분석

### 실행/관리 (7개)
- `broker_execution.py` - 실제 브로커 실행
- `paper_trader.py` - 페이퍼 트레이딩
- `paper_trading.py` - 페이퍼 트레이딩 v2
- `trade_journal.py` - 트레이드 저널
- `trading_cost_model.py` - 거래 비용 모델
- `notifications.py` - 알림 시스템
- `notifier.py` - 알림 v2
- `alerts.py` - 알림
- `alert_manager.py` - 알림 관리

### 기타 (4개)
- `session_analyzer.py` - 세션 분석
- `regime_history.py` - 레짐 히스토리
- `feedback_tracker.py` - 피드백 추적
- `multi_asset.py` - 다중 자산 분석

---

## 📚 사용 가이드

### 1. main.py 실행 모드

```bash
# 기본 실행 (분석 + 의사결정)
python main.py

# 빠른 분석 (Phase 2.3-2.10 스킵)
python main.py --quick

# 전체 분석 + AI 리포트
python main.py --report

# 전체 분석 + 실시간 VPIN (60초)
python main.py --realtime --duration 60
```

**목표:**
- `python main.py` (기본): 분석 + 리포트 + 의사결정 기능
- `python main.py --full`: 실시간에서 얻을 수 있는 모든 기능 수행

### 2. 독립 스크립트 실행

```bash
# 암호화폐 24/7 모니터링
python lib/crypto_collector.py --detect --analyze

# 장중 데이터 수집
python lib/intraday_collector.py --backfill

# 이상-뉴스 귀인
python lib/news_correlator.py
```

### 3. 새 모듈 추가 시

1. `lib/` 에 모듈 생성
2. `if __name__ == "__main__"` 테스트 코드 포함
3. main.py에 import 추가 (필요 시)
4. 이 README.md 업데이트

---

## 🔍 모듈 찾기

**키워드로 검색:**
```bash
# VPIN 관련 모듈
grep -l "VPIN" lib/*.py

# Granger Causality 관련
grep -l "Granger" lib/*.py

# Bubble Detection
ls lib/*bubble*.py
```

**Phase별 모듈:**
- Phase 1 (데이터): fred_collector, data_collector, extended_data_sources
- Phase 2 (분석): regime_*, critical_path, bubble_detector, genius_act_macro, graph_clustered_portfolio
- Phase 3 (토론): orchestrator, dual_mode_analyzer, adaptive_agents
- Phase 4 (실시간): binance_stream, realtime_pipeline
- Phase 5-7 (저장/리포트): event_db, ai_report_generator, whitening_engine

---

**마지막 업데이트:** 2026-01-14
**총 Active 모듈:** 54개 (통합 37 + 독립 7 + 지원 10)
