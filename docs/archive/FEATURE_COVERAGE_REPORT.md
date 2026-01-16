# EIMAS 기능 커버리지 분석 보고서

> 생성일: 2026-01-14
> 분석 대상: main.py vs lib/ 모듈 (94개 파일)

---

## 📊 Executive Summary

EIMAS는 **37개의 핵심 모듈**을 main.py 파이프라인에 통합하여 **7개 Phase**로 구성된 종합 분석 시스템을 구축했습니다. 추가로 **7개의 독립 스크립트**가 별도 실행 가능하며, COMMANDS.md에 문서화되어 있습니다.

**결과:**
- ✅ **통합 모듈**: 37개 (Phase 1-7)
- ✅ **독립 스크립트**: 7개 (COMMANDS.md 문서화)
- ⚠️ **미사용 모듈**: 40+개 (구버전, 미구현, 중복)

---

## 1. main.py 통합 모듈 (37개)

### Phase 1: DATA COLLECTION (5개)
| 모듈 | 기능 | 조건 |
|------|------|------|
| fred_collector.py | FRED 데이터 (RRP, TGA, Net Liquidity) | 항상 |
| data_collector.py | 시장 데이터 (24 tickers + Crypto + RWA) | 항상 |
| unified_data_store.py | 데이터 저장소 | 항상 |
| market_indicators.py | VIX, Fear & Greed | not quick_mode |
| extended_data_sources.py | DeFiLlama, MENA | not quick_mode |

### Phase 2: ANALYSIS (19개)
| 모듈 | 기능 | 조건 |
|------|------|------|
| regime_detector.py | 레짐 탐지 (Bull/Bear/Neutral) | 항상 |
| regime_analyzer.py | GMM 3-state + Shannon Entropy | not quick_mode |
| event_framework.py | 이벤트 탐지 | 항상 |
| liquidity_analysis.py | Granger Causality | not quick_mode |
| causal_network.py | 인과관계 네트워크 | not quick_mode |
| critical_path.py | Critical Path 리스크 점수 | 항상 |
| correlation_monitor.py | 상관관계 모니터링 | 항상 |
| etf_flow_analyzer.py | ETF 플로우 분석 | not quick_mode |
| microstructure.py | 시장 미세구조 (VPIN, Amihud) | not quick_mode |
| bubble_detector.py | 버블 리스크 (Greenwood-Shleifer) | not quick_mode |
| genius_act_macro.py | 스테이블코인-유동성 + Crypto Stress Test | not quick_mode |
| custom_etf_builder.py | Theme ETF + Supply Chain Graph | not quick_mode |
| causality_graph.py | CausalityGraphEngine (고급) | not quick_mode |
| causality_narrative.py | 인과관계 Narrative 생성 | not quick_mode |
| shock_propagation_graph.py | 충격 전파 그래프 | not quick_mode |
| graph_clustered_portfolio.py | GC-HRP 포트폴리오 최적화 | not quick_mode |
| integrated_strategy.py | 통합 전략 엔진 | not quick_mode |
| volume_analyzer.py | 거래량 이상 탐지 (Kyle 1985) | not quick_mode |
| event_tracker.py | 이상→뉴스 역추적 | not quick_mode |

### Phase 3: MULTI-AGENT DEBATE (4개)
| 모듈 | 기능 | 조건 |
|------|------|------|
| agents/orchestrator.py | MetaOrchestrator (FULL/REF Mode) | 항상 |
| dual_mode_analyzer.py | 모드 비교 및 최종 권고 | 항상 |
| adaptive_agents.py | Adaptive Portfolio Agents (3종) | not quick_mode |
| validation_agents.py | Validation Loop (Claude + Perplexity) | not quick_mode |

### Phase 4: REAL-TIME (2개)
| 모듈 | 기능 | 조건 |
|------|------|------|
| binance_stream.py | Binance WebSocket 스트리밍 | enable_realtime |
| realtime_pipeline.py | VPIN 실시간 계산 | enable_realtime |

### Phase 5: DATABASE (3개)
| 모듈 | 기능 | 조건 |
|------|------|------|
| event_db.py | 이벤트 DB 저장 | 항상 |
| realtime_pipeline.py | Signal DB 저장 | 항상 |
| predictions_db.py | 예측 DB 저장 | 항상 |

### Phase 6: AI REPORT (1개)
| 모듈 | 기능 | 조건 |
|------|------|------|
| ai_report_generator.py | AI 리포트 생성 | generate_report |

### Phase 7: QUALITY (2개)
| 모듈 | 기능 | 조건 |
|------|------|------|
| whitening_engine.py | 경제학적 해석 | generate_report and not quick_mode |
| autonomous_agent.py | AI 팩트체킹 | generate_report and not quick_mode |

---

## 2. 독립 실행 스크립트 (7개)

이 모듈들은 main.py와 별도로 실행 가능하며, COMMANDS.md에 상세히 문서화되어 있습니다.

| 모듈 | 용도 | 실행 방법 |
|------|------|----------|
| **intraday_collector.py** | 1분봉 장중 데이터 수집 | `python lib/intraday_collector.py [--date YYYY-MM-DD] [--backfill]` |
| **crypto_collector.py** | 24/7 암호화폐 모니터링 + 이상 탐지 | `python lib/crypto_collector.py --detect [--analyze]` |
| **market_data_pipeline.py** | 다중 API 데이터 수집 (Twelve Data, CryptoCompare) | `python lib/market_data_pipeline.py --all [--with-oil]` |
| **event_predictor.py** | 경제 이벤트 예측 (NFP, CPI, FOMC) | Python 코드로 호출 (`EventPredictor().generate_report()`) |
| **event_attribution.py** | 이벤트 원인 분석 (Perplexity 연동) | Python 코드로 호출 (`EventAttributor().analyze_recent_events()`) |
| **event_backtester.py** | 이벤트 백테스트 (과거 FOMC, CPI, NFP 분석) | `python lib/event_backtester.py` |
| **news_correlator.py** | 이상-뉴스 자동 귀인 (24시간 이상 → 뉴스 연결) | `python lib/news_correlator.py` |

**사용 예시:**
```bash
# 주말 암호화폐 모니터링 (Cron)
0 * * * 6,0 cd /path/to/eimas && python lib/crypto_collector.py --detect >> logs/crypto.log 2>&1

# 장중 데이터 수집 (매일 아침)
python lib/intraday_collector.py --backfill

# 이상-뉴스 귀인
python lib/news_correlator.py
```

---

## 3. 미사용 모듈 (40+개)

다음 모듈들은 main.py에서 사용되지 않습니다. 이유는 다음과 같습니다:

### 3.1 구버전/대체됨 (6개)
| 모듈 | 상태 | 대체 모듈 |
|------|------|----------|
| causal_network.py | 구버전 | causality_graph.py로 대체 |
| enhanced_data_sources.py | 구버전 | extended_data_sources.py로 대체 |
| data_loader.py | 통합됨 | data_collector.py에 포함 |
| debate_agent.py | 구버전 | agents/orchestrator.py로 대체 |
| hrp_optimizer.py | 통합됨 | graph_clustered_portfolio.py에 포함 |
| portfolio_optimizer.py | 대체됨 | graph_clustered_portfolio.py |

### 3.2 별도 실행 (3개)
| 모듈 | 용도 | 비고 |
|------|------|------|
| backtest.py, backtest_engine.py, backtester.py | 백테스트 엔진 | scripts/run_backtest.py에서 사용 |
| dashboard_generator.py | Plotly 대시보드 | 별도 실행 (`python plus/dashboard_generator.py`) |

### 3.3 미구현 기능 (30+개)
다음 기능들은 아직 구현되지 않았거나 main.py에 통합되지 않았습니다:

**데이터 소스:**
- earnings.py, economic_calendar.py, insider_trading.py

**분석 기능:**
- factor_analyzer.py, factor_exposure.py, seasonality.py
- sentiment.py, sentiment_analyzer.py
- geopolitical_risk_detector.py, leading_indicator_tester.py

**전략:**
- mean_reversion.py, pairs_trading.py, options_flow.py
- patterns.py

**포트폴리오:**
- position_sizing.py, tax_optimizer.py

**실행/관리:**
- broker_execution.py, paper_trader.py, paper_trading.py
- trade_journal.py, trading_cost_model.py
- notifications.py, notifier.py, alerts.py, alert_manager.py

**분석 도구:**
- performance_attribution.py, risk_manager.py, risk_analytics.py
- session_analyzer.py, regime_history.py

### 3.4 중복 가능성 (검토 필요)
| 모듈 1 | 모듈 2 | 비고 |
|--------|--------|------|
| etf_signal_generator.py | etf_flow_analyzer.py | 통합 가능? |
| macro_analyzer.py | genius_act_macro.py | 통합 가능? |
| sector_rotation.py | etf_flow_analyzer.py | 통합 가능? |
| signal_pipeline.py | integrated_strategy.py | 통합 가능? |
| risk_profile_agents.py | adaptive_agents.py | 통합 가능? |
| report_generator.py | ai_report_generator.py | 통합 가능? |
| trading_db.py | realtime_pipeline.py | SignalDatabase 사용 중 |

---

## 4. 종합 결론

### ✅ 우수한 점

1. **핵심 기능 완전 통합**
   - Phase 1-7 모든 단계에서 37개 모듈이 유기적으로 연결
   - 고급 경제학 방법론 (GMM, Entropy, Granger, LASSO, HRP, MST) 구현
   - v2.1.1 Risk Enhancement Layer (Microstructure + Bubble) 완료

2. **명확한 문서화**
   - COMMANDS.md: 독립 스크립트 7개의 실행 방법 상세 기재
   - CLAUDE.md: Phase별 실행 조건 및 출력 명확화

3. **모듈화 설계**
   - main.py는 파이프라인 조정에만 집중
   - 각 모듈은 독립 실행 가능 (`if __name__ == "__main__"`)

### ⚠️ 개선 필요

1. **코드 정리**
   - lib/ 디렉토리에 94개 파일 중 40+개가 미사용
   - 구버전 모듈 (causal_network, debate_agent 등) 제거 필요

2. **문서 보완**
   - lib/README.md 생성 필요 (각 모듈 상태 표시)
   - 중복 모듈 명확화 (etf_signal_generator vs etf_flow_analyzer)

3. **통합 검토**
   - news_correlator.py: event_tracker.py와 기능 유사 → 통합 고려
   - market_data_pipeline.py: Twelve Data API를 Phase 1.2에 추가 고려

---

## 5. 권장 조치

### Priority 1: 코드 정리 (Cleanup)
```bash
mkdir -p lib/deprecated lib/future

# 구버전 이동
mv lib/causal_network.py lib/deprecated/
mv lib/enhanced_data_sources.py lib/deprecated/
mv lib/debate_agent.py lib/deprecated/

# 미구현 기능 이동
mv lib/earnings.py lib/future/
mv lib/options_flow.py lib/future/
mv lib/sentiment*.py lib/future/
```

### Priority 2: 문서 생성
```markdown
# lib/README.md

## 모듈 상태

### ✅ 통합 모듈 (main.py 사용)
- fred_collector.py, data_collector.py, ...

### 🚀 독립 스크립트 (별도 실행)
- intraday_collector.py, crypto_collector.py, ...

### ⚠️ Deprecated (사용 중단)
- causal_network.py → causality_graph.py
- enhanced_data_sources.py → extended_data_sources.py

### 🔮 Future (미구현)
- earnings.py, options_flow.py, ...
```

### Priority 3: 통합 검토 (선택)
- [ ] news_correlator.py의 기능을 event_tracker.py와 비교
- [ ] market_data_pipeline.py를 Phase 1.2에 통합 (Twelve Data 지원)
- [ ] 중복 가능성 있는 모듈 6개 검토

---

## 6. 최종 요약표

| 분류 | 개수 | 상태 | 비고 |
|------|------|------|------|
| **통합 모듈** | 37 | ✅ 완료 | main.py Phase 1-7 |
| **독립 스크립트** | 7 | ✅ 문서화 | COMMANDS.md 참조 |
| **구버전/대체** | 6 | ⚠️ 정리 필요 | deprecated/로 이동 |
| **미구현** | 30+ | ⚠️ 정리 필요 | future/로 이동 |
| **중복 가능성** | 6 | ⚠️ 검토 필요 | 통합 또는 명확화 |

**커버리지:** 44/94 = **46.8%** (통합 + 독립)
**활용도:** 37/44 = **84.1%** (통합만 기준)

---

*이 보고서는 2026-01-14에 생성되었으며, main.py (3000+ lines)와 lib/ 모듈 (94개 파일)을 비교 분석한 결과입니다.*
