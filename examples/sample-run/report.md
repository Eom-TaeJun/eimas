# 자산배분팀 리서치 리포트
**생성 시간**: 2026-02-09T05:31:34.538000

## 1. 현재 시장 및 레짐 요약
- **레짐**: BULL (High 신뢰도)
- **변동성**: Low Vol
- **리스크 점수**: 2.5/100 (LOW)
- **Fed 순유동성**: $5694B
- **최종 권고**: BULLISH (신뢰도 51%)
- **데이터 품질**: COMPLETE

**요약**: 현재 상승 추세 (BULL 레짐), 변동성 Low Vol, 리스크 점수 2.5. 최종 권고: BULLISH.

### AI 토론/해석
- Debate recommended_action=NEUTRAL, final_recommendation=BULLISH로 판단 정합성은 불일치입니다.
- Debate 메타데이터 평균 confidence는 51.3%이며, 추론 단계는 4단계입니다.

## 2. 핵심 근거 3가지
### 근거 1: 시장 레짐 분석
- **출처**: `regime`
- **값**: Bull (Low Vol)
- **해석**: 상승 추세 레짐. Entropy 0.000으로 낮은 불확실성.

### 근거 2: Fed 유동성 상태
- **출처**: `liquidity_signal, fred_summary`
- **값**: NEUTRAL (RRP: $3B, TGA: $909B)
- **해석**: 유동성 중립. 변화 방향 관찰 필요.

### 근거 3: AI 에이전트 토론 결과
- **출처**: `debate_consensus, full_mode_position, reference_mode_position`
- **값**: Full: BULLISH, Ref: BULLISH, 합의: 예
- **해석**: 두 모드 모두 BULLISH 판단. 신뢰도 높음.

### AI 토론/해석
- 핵심 합의 포인트: Market has underlying strength.
- 주요 이견 포인트: Valuation concerns remain.

## 3. 리스크 및 반증 조건 3가지
### 리스크 1: 리스크 점수 상승
- **현재 값**: 2.5/100
- **반증 조건**: 리스크 점수가 23 이상 상승 시 뷰 재검토
- **모니터링 지표**: risk_score

### 리스크 2: 버블 리스크
- **현재 값**: NONE
- **반증 조건**: 버블 상태가 WARNING → DANGER 전환 시 즉시 비중 축소
- **모니터링 지표**: bubble_risk.overall_status

### 리스크 3: 변동성 급등
- **현재 값**: Normal
- **반증 조건**: 변동성이 High Vol로 전환 시 포지션 규모 축소 검토
- **모니터링 지표**: regime.volatility

### 리스크 신호 연계 뉴스 브리핑
- **감지 신호**: warnings=1, risk_score_warning_divergence=2.5, vix_spot=20.4, gap_signal=BEARISH
- **관련 헤드라인**: tech giants in china sold off alongside their u.s. peers last week. how to play it
- **뉴스 센티먼트**: Neutral (score=-0.10)
- **AI 연계 해석**: 갭 분석에서 risk-off 성향이 포착되어 ETF 방어 듀레이션/현금 비중 점검이 필요합니다. 검증/경고 1건이 존재해 뉴스 해석을 단일 방향으로 단정하지 않고 리스크 시나리오를 병행합니다. 뉴스 센티먼트 점수는 -0.10로 집계되었습니다. 핵심 헤드라인의 이벤트 방향이 모멘텀/변동성 지표와 같은 방향인지 교차 검증이 필요합니다.
- **외부 뉴스 API**: perplexity (ok)
- **외부 헤드라인(실시간 검색)**:
  - [Stock Market News for Feb 6, 2026](https://www.nasdaq.com/articles/stock-market-news-feb-6-2026) (Nasdaq, 2026-02-06)
    - 요약: Wall Street closed sharply lower with Dow -1.2%, Nasdaq -1.6%, S&P 500 -1.2% amid risk-off sentiment from AI spending concerns and weak labor data. VIX spiked 16.8% to 21.77, tech ETFs like XLK -1.7% and IGV -5%, signaling bearish volatility and sector rotation.
  - [Monthly Market Commentary – February 2026](https://www.parkavenuesecurities.com/monthly-market-commentary-february-2026) (Park Avenue Securities, 2026-02-01)
    - 요약: Geopolitical tensions with Greenland caused largest one-day stock drop (-2.1%) since October, boosting energy prices and volatility. Fed held rates amid inflation pressures, with expected 1-2 cuts in 2026, highlighting macro risks and risk-off shifts in US equities.
  - [Monthly Market Update - February 2026](https://madisoninvestments.com/monthly-market-update-february-2026/) (Madison Investments, 2026-02-01)
    - 요약: Sector rotation saw Energy +14.4%, Materials +8.7%, while Tech declined amid high valuations. Fed signals patience with weak jobs vs. hot inflation >2%, plus Trump naming Kevin Warsh as Fed Chair amid Powell probe, raising policy uncertainty and macro risks.
  - [Greater clarity on the main risks to the market](https://www.invesco.com/us/en/insights/greater-clarity-risks-market.html) (Invesco US, 2026-02-01)
    - 요약: Clarity emerges on key 2026 risks: Federal Reserve independence threats and potential AI bubble burst, contributing to elevated market volatility, risk score divergences, and bearish signals in US stocks.
- **출처**: `warnings + gap_analysis + extended_data.news_sentiment + sentiment_analysis.news_sentiment + external_news.perplexity.ok`

### AI 토론/해석
- 검증 점수 86.23, passed=False, warnings=1건.
- 리스크 연계 신호: warnings=1, risk_score_warning_divergence=2.5, vix_spot=20.4.
- 외부 뉴스 연계: provider=perplexity, status=ok, items=4.

## 4. 운용 관점의 액션 아이템
### 1. [HOLD] 전체 포트폴리오
- **근거**: 리밸런싱 조건 미충족
- **제약**: Turnover Cap: 30%
- **우선순위**: MEDIUM

### AI 토론/해석
- 실행 스탠스는 HOLD이며 근거는 '리밸런싱 조건 미충족'.
- AI 토론 권고(NEUTRAL)와 운용 액션 간 괴리 여부를 매 실행주기 확인합니다.

## 5. ETF 전략 분해표 (팩터/섹터/듀레이션)
| Ticker | 역할 | 팩터 노출 | 섹터/테마 | 듀레이션·금리 민감도 | 5D 수익률 | 20D 수익률 | 모멘텀 | Top Holdings(3) | Source(Q) |
|---|---|---|---|---|---:|---:|---|---|---|
| XLE | Sector Satellite | Value/Cyclicals | 에너지 섹터 | 유가 민감 | +4.31% | +14.39% | UPTREND | XOM, CVX, COP | yf_px+yf_meta (COMPLETE) |
| GLD | Real Asset Hedge | Inflation/Real Rate | 금(대체자산) | 실질금리 역민감 | +2.36% | +10.69% | UPTREND | Physical Gold Bullion | fi+yf_px+yf_meta (COMPLETE) |
| XLI | Sector Satellite | Cyclicals | 산업재 섹터 | 경기 민감 | +4.68% | +8.12% | UPTREND | GE, RTX, UNP | yf_px+yf_meta (COMPLETE) |
| IWM | Size Tilt | Small Cap | 미국 소형주 | 경기 민감 | +2.07% | +2.61% | NEUTRAL | SMCI, FTAI, CRDO | fi+yf_px+yf_meta (COMPLETE) |
| TLT | Rates Hedge | Duration | 미국 장기 국채 | 고듀레이션 | +0.85% | +0.60% | NEUTRAL | UST 20Y+, UST 25Y+, UST 30Y+ | fi+yf_px+yf_meta (COMPLETE) |
| SPY | Core | Market Beta | 미국 대형주 광범위 | 중립 | -0.20% | +0.16% | NEUTRAL | AAPL, MSFT, NVDA | fi+yf_px+yf_meta (COMPLETE) |
| XLV | Sector Satellite | Defensive | 헬스케어 섹터 | 방어적 | +1.92% | -0.26% | NEUTRAL | LLY, UNH, JNJ | yf_px+yf_meta (COMPLETE) |
| QQQ | Growth Tilt | Growth/Quality | 나스닥100(기술 비중 높음) | 상대적 고듀레이션 | -1.97% | -1.74% | NEUTRAL | AAPL, MSFT, NVDA | fi+yf_px+yf_meta (COMPLETE) |
| XLK | Sector Satellite | Quality/Growth | 기술 섹터 | 금리 민감 | -1.91% | -2.16% | NEUTRAL | AAPL, MSFT, NVDA | yf_px+yf_meta (COMPLETE) |
| XLF | Sector Satellite | Value/Cyclicals | 금융 섹터 | 장단기금리차 민감 | +1.53% | -2.93% | NEUTRAL | BRK-B, JPM, V | fi+yf_px+yf_meta (COMPLETE) |

- **출처**: `company_ra_analysis.etf_strategy_snapshot` + yfinance 가격/메타데이터 + EIMAS ETF 프로필 카탈로그(자동수집 실패 시 fallback)

### AI 토론/해석
- ETF 20D 모멘텀 기준 상위 XLE(+14.39%), 하위 XLF(-2.93%)로 상대강도 스프레드가 확인됩니다.
- 상·하위 ETF의 듀레이션/섹터 노출 변화가 레짐 해석과 정합한지 추적합니다.

## 6. 기업 커버리지 + RA 업무 지원 + SQL 증빙
| Ticker | Sector | Trailing P/E | Forward P/E | P/B | ROE | ROA | Net Margin | D/E | 5D | 20D | Signal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AAPL | Technology | 35.20 | 29.98 | 46.37 | 151.91 | 31.18 | 26.92 | 1.338 | +7.18% | +7.37% | FAIR |
| MSFT | Technology | 25.12 | 21.26 | 7.62 | 29.65 | 16.45 | 36.15 | 0.176 | -6.77% | -16.10% | UNDERVALUED |
| NVDA | Technology | 45.89 | 24.04 | 37.90 | 91.87 | 65.30 | 55.85 | 0.129 | -2.99% | +0.20% | OVERVALUED |
| JPM | Financial Services | 16.10 | 13.94 | 2.54 | 16.96 | 1.46 | 34.51 | 1.318 | +5.40% | -2.24% | FAIR |
| XOM | Energy | 22.21 | 17.87 | 2.40 | 12.77 | 7.43 | 9.93 | 0.158 | +5.41% | +21.27% | FAIR |

### 기업별 RA 코멘트
- **AAPL**: 핵심 재무지표와 이익 모멘텀의 정합성 확인 후 포지션 판단 권고.
- **MSFT**: 동종 대비 밸류 할인 + 수익성 우수. 리서치 노트 업사이드 가설 점검 권고.
- **NVDA**: 핵심 재무지표와 이익 모멘텀의 정합성 확인 후 포지션 판단 권고.
- **JPM**: 핵심 재무지표와 이익 모멘텀의 정합성 확인 후 포지션 판단 권고.
- **XOM**: 핵심 재무지표와 이익 모멘텀의 정합성 확인 후 포지션 판단 권고.

### 운영 지원 포인트 (데이터 기반)
- **Role Focus**: RA (매크로/ETF 전략)
#### 자료조사/업데이트 태스크
- 기업 재무/회계 핵심지표 업데이트: AAPL, MSFT, NVDA
- 섹터/지수 ETF 모멘텀 비교표 작성 (5일/20일 수익률)
- 밸류에이션 편차(동종 대비 P/E) 점검 및 코멘트 작성
- ETF 보유종목/섹터 비중 스냅샷 업데이트 (top holdings + sector weights)

#### 세미나/대외자료 포인트
- 최근 매크로 환경과 ETF 자금흐름 요약 슬라이드
- 기업 실적/밸류에이션 체커보드(ROE, 마진, D/E, P/E) 업데이트
- 리스크 시나리오(금리/유가/달러)별 대응 포인트 정리

#### 유관부서 협조 포인트
- 영업/운용 협조용 기업별 one-page 팩트시트 제공
- 외부 세미나용 핵심 데이터 출처 및 산식 명시
- SQL(PostgreSQL) 기반 데이터 추출 요청 대응

- **Data Update Note**: 재무제표 기반 지표는 공시 갱신 주기(분기/연간)와 시차가 있으므로 가격지표와 업데이트 주기를 구분해 운용

### PostgreSQL 증빙
- **enabled**: True
- **dsn_configured**: True
- **driver_available**: True
- **stored_rows**: 5
- **table**: fi_ra.company_fundamentals

### EIMAS Internal SQL 증빙 (SQLite)
- **enabled**: True
- **db_path**: data/ra_research.db
- **table**: ra_company_fundamentals
- **upserted_rows**: 5
- **total_rows**: 5
- **distinct_tickers**: 5
- **etf_table**: ra_etf_snapshot
- **etf_upserted_rows**: 10
- **etf_total_rows**: 10
- **etf_distinct_tickers**: 10
- **date_range**: 2026-02-09 ~ 2026-02-09
- **quality_checks**: missing_valuation=0, missing_financial=0, flagged=0
- **etf_quality_checks**: missing_returns=0, missing_holdings=0, unexpected_quality=0
- **etf_source_mix**:
  - financial_indicators,yfinance_price,yfinance_info: 6
  - yfinance_price,yfinance_info: 4
- **sql_artifacts_refreshed_at**: 2026-02-09T04:00:23.898193
- **materialized_row_counts**: valuation=5, etf_momentum=10, backtest_compare=5

```sql
SELECT COUNT(*) AS total_rows, COUNT(DISTINCT ticker) AS tickers,
       MIN(as_of_date) AS min_date, MAX(as_of_date) AS max_date
FROM fi_ra.company_fundamentals;
```

```sql
SELECT COUNT(*) AS total_rows, COUNT(DISTINCT ticker) AS distinct_tickers,
       MIN(as_of_date) AS min_date, MAX(as_of_date) AS max_date
FROM ra_company_fundamentals;
```

```sql
SELECT ticker, category, ret_20d_pct, momentum_label
FROM ra_etf_momentum_snapshot_mv
ORDER BY rank_ret_20d ASC
LIMIT 10;
```

### SQL Preview: Valuation Snapshot MV
| Ticker | Sector | Trailing P/E | P/B | 20D | Signal | PE Bucket |
|---|---|---:|---:|---:|---|---|
| JPM | Financial Services | 16.10 | 2.54 | -2.24% | FAIR | MID_PE |
| XOM | Energy | 22.21 | 2.40 | +21.27% | FAIR | MID_PE |
| MSFT | Technology | 25.12 | 7.62 | -16.10% | UNDERVALUED | MID_PE |
| AAPL | Technology | 35.20 | 46.37 | +7.37% | FAIR | HIGH_PE |
| NVDA | Technology | 45.89 | 37.90 | +0.20% | OVERVALUED | HIGH_PE |

### SQL Preview: ETF Momentum Snapshot MV
| Rank | Ticker | Category | 5D | 20D | Momentum |
|---:|---|---|---:|---:|---|
| 1 | XLE | sector | +4.31% | +14.39% | UPTREND |
| 2 | GLD | alternative | +2.36% | +10.69% | UPTREND |
| 3 | XLI | sector | +4.68% | +8.12% | UPTREND |
| 4 | IWM | market | +2.07% | +2.61% | NEUTRAL |
| 5 | TLT | bond | +0.85% | +0.60% | NEUTRAL |
| 6 | SPY | market | -0.20% | +0.16% | NEUTRAL |
| 7 | XLV | sector | +1.92% | -0.26% | NEUTRAL |
| 8 | QQQ | market | -1.97% | -1.74% | NEUTRAL |

### SQL Preview: Backtest Compare MV
| Rank | Strategy | Total Return | Ann Return | Sharpe | MaxDD |
|---:|---|---:|---:|---:|---:|
| 1 | manual_smoke | +12.00% | +8.00% | 1.10 | -12.00% |
| 2 | fast | +10.00% | +8.00% | 1.00 | -10.00% |
| 3 | EIMAS_AutoPaper_TargetPrice | +1.46% | +0.73% | 0.16 | -7.56% |
| 3 | EIMAS_AutoPaper_TargetPrice | +1.46% | +0.73% | 0.16 | -7.56% |
| 5 | EIMAS_AutoPaper_TargetPrice | +0.00% | +0.00% | 0.00 | +0.00% |

### AI 토론/해석
- 기업 커버리지 5개와 SQL 적재(rows=5, etf_rows=10)를 결합해 RA 근거를 구성했습니다.
- 백테스트 SQL 저장 runs=6를 리포트 증빙 축으로 연결했습니다.

## 7. 실행 타이밍 요약 + 운영 의사결정 추적
- **Pipeline Elapsed**: 79.588s

| Phase | Duration(s) | Status |
|---|---:|---|
| pipeline_total | 79.588 | ok |
| phase3_debate | 34.598 | ok |
| phase1_collect_data | 33.713 | ok |
| phase2_institutional_frameworks | 8.870 | ok |
| phase2_basic_analyze | 1.182 | ok |
| phase2_sentiment_bubble | 0.997 | ok |
| phase2_enhanced_analyze | 0.184 | ok |
| phase45_operational_report | 0.026 | ok |
| phase5_storage | 0.011 | ok |
| phase9_artifact_export | 0.007 | ok |
| phase2_extended_adjustment | 0.000 | ok |
| phase2_adaptive_portfolio | 0.000 | ok |
| phase4_realtime | 0.000 | ok |
| phase46_paper_execution | 0.000 | ok |
| phase6_backtest | 0.000 | ok |
| phase6_performance_attribution | 0.000 | ok |
| phase6_stress_test | 0.000 | ok |
| phase7_generate_report | 0.000 | ok |
| phase7_validate_report | 0.000 | ok |
| phase8_ai_validation | 0.000 | ok |
| phase85_quick_validation | 0.000 | ok |

### Operational Decision Summary
- **final_stance**: BULLISH
- **reason_codes**: BULL_REGIME_WITH_CONSENSUS
- **applied_rules**:
  - RULE_7: BULLISH (regime + consensus agree)
- **is_hold**: False

### AI 토론/해석
- 파이프라인 병목 구간은 pipeline_total 79.588s이며 최종 의사결정 스탠스는 BULLISH입니다.
- 운영 의사결정 규칙(reason_codes/applied_rules)을 토대로 실행 재현성을 유지합니다.

## 8. 정량 상세 지표 스냅샷
### Macro & Liquidity
| Metric | Value |
|---|---:|
| Fed Funds (%) | 3.64 |
| UST 2Y (%) | 3.47 |
| UST 10Y (%) | 4.21 |
| UST 30Y (%) | 4.85 |
| 10Y-2Y Spread (%) | 0.72 |
| HY OAS | 2.97 |
| Unemployment (%) | 4.40 |
| Initial Claims | 231000 |
| RRP ($B) | 3.10 |
| TGA ($B) | 908.80 |
| Net Liquidity ($B) | 5694.00 |
| Liquidity Regime | Abundant |

### HFT / Volatility
| Metric | Value |
|---|---:|
| Tick Rule Buy Ratio | 53.3% |
| Tick Rule Interpretation | NEUTRAL |
| Kyle's Lambda | 7.188e-11 |
| Kyle's R² | 0.701 |
| GARCH Current Vol | 11.9% |
| GARCH Forecast Avg Vol | 13.1% |
| GARCH Persistence | 0.562 |
| GARCH Half-life (days) | 1.2 |

### Information Flow / PoI
| Metric | Value |
|---|---:|
| Abnormal Volume Days | 0 |
| Abnormal Volume Ratio | 0.0% |
| CAPM QQQ Alpha | -15.04%/yr |
| CAPM QQQ Beta | 1.291 |
| PoI Index Value | 18.433 |
| PoI Mean Reversion Signal | HOLD |
| PoI Z-Score | 0.047 |
| PoI Verification | PASS |

### DTW / DBSCAN / Bubble / Sentiment
| Metric | Value |
|---|---:|
| DTW n_series | N/A |
| DTW Avg Distance | N/A |
| DTW Most Similar Pair | N/A ↔ N/A |
| DTW Lead-Lag | N/A |
| DBSCAN Outliers | N/A/N/A |
| DBSCAN Outlier Ratio | N/A |
| Bubble Status | N/A |
| Fear & Greed | 50 |
| VIX Structure | NEUTRAL |
| Put/Call Ratio | 1.19 (BEARISH/HEDGING) |

### AI 토론/해석
- 정량 신호 요약: Net Liquidity=5694.00, GARCH Vol=11.9%, PoI Signal=HOLD.
- 거시·미시구조·유동성 신호를 단일 방향으로 단정하지 않고 교차검증합니다.

## 9. 검증 / 경고 상세
| Item | Value |
|---|---|
| final_recommendation | BULLISH |
| full_mode_position | BULLISH |
| reference_mode_position | BULLISH |
| modes_agree | True |
| fact_check_grade | N/A |
| validation_final_result | N/A |
| validation_consensus_confidence | N/A |
| validation_agreement_ratio | N/A |
| verification_score | 86.23 |
| verification_passed | False |
| validation_summary | N/A |

### Warnings
- ⚠️ Extremely Low Risk (2.5/100) - Verify market conditions

### AI 토론/해석
- 검증/경고 섹션은 warnings=1건과 debate verification을 함께 제시해 오판 리스크를 통제합니다.
- 핵심 추론 하이라이트: AnalysisAgent: Risk=11.4, Regime=BULL (레짐 전환 징후: BULL → BULL_TO_BEAR (확률 55%)).

## 10. 리밸런싱 / 운용 승인 근거 + RA 코멘트 + 구현 TODO
### Rebalance Summary
| Metric | Value |
|---|---:|
| total_turnover | 0.0% |
| estimated_cost | 0.000% |
| buy_count | 0 |
| sell_count | 0 |
| hold_count | 0 |

### Trigger
- **type**: MANUAL
- **reason**: No weights available

### Approval
- **requires_human_approval**: False
- **approval_reason**: 

### RA 코멘트 (실데이터 기반)
- **commentary_source**: openai
- **commentary_model**: gpt-4o-mini
- **commentary_audit_log_id**: 23
- **macro_view**: {'regime': 'Bull (Low Vol)', 'volatility_state': 'Normal', 'risk_score': 2.53, 'fed_funds': 3.64, 'treasury_10y': 4.21, 'hy_oas': 2.97}
- **etf_view**: {'top_etfs': [{'ticker': 'XLE', 'ret_20d_pct': 14.39, 'momentum_label': 'UPTREND'}, {'ticker': 'GLD', 'ret_20d_pct': 10.69, 'momentum_label': 'UPTREND'}, {'ticker': 'XLI', 'ret_20d_pct': 8.12, 'momentum_label': 'UPTREND'}], 'bottom_etfs': [{'ticker': 'XLF', 'ret_20d_pct': -2.93, 'momentum_label': 'NEUTRAL'}, {'ticker': 'XLK', 'ret_20d_pct': -2.16, 'momentum_label': 'NEUTRAL'}, {'ticker': 'QQQ', 'ret_20d_pct': -1.74, 'momentum_label': 'NEUTRAL'}]}
- **company_view**: {'company_count': 5, 'valuation_signals': {'fair': 3, 'undervalued': 1, 'overvalued': 1}}
- **risk_view**: {'backtest_total_return': 0.077, 'backtest_sharpe': 1.18, 'backtest_mdd': -0.037}
- **execution_view**: {'final_recommendation': 'BULLISH', 'confidence': 0.51}
- **final_ra_call**: BULLISH
- **priority_actions**:
  - Consider increasing exposure to top-performing ETFs
  - Monitor underperforming ETFs for potential adjustments
  - Evaluate company valuations for rebalancing opportunities

### RA-SQL 적용 영역 매트릭스
| 사용 영역 | 설명 | SQL 예시 기능 | 상태 |
|---|---|---|---|
| 거시지표 분석 | FRED/OECD/한국은행 계열 거시 데이터 정규화 및 스냅샷 저장 | UPSERT, Window Functions | 구현 |
| ETF/섹터 분석 | ETF 구성/섹터/기간 수익률 비교를 통한 전략 분해 | JOIN, GROUP BY, ROLLUP, CTE | 구현 |
| 기업 분석 | 재무제표 추이·밸류에이션·모멘텀 결합 커버리지 관리 | CASE, AVG OVER, LAG/LEAD | 구현 |
| 퀀트 전략 백테스트 | 전략별 성과(수익률/MDD/Sharpe) 저장 및 비교 | INSERT INTO, UPSERT, Analytic Functions | 구현 |
| 리포트 자동화 | 정량 요약 테이블/차트용 결과셋 추출 | VIEW, Materialized View | 구현 |
| 종합 시그널링 | 거시/ETF/기업 신호를 통합해 단일 점수와 라벨 생성 | CTE, CASE, Weighted Composite | 구현 |
| RA 분석 증빙 | SQL 기반 지표·그래프를 PDF에 삽입하고 로그화 | EXPORT, audit_log 테이블 | 구현 |

### EIMAS Phase별 SQL 통합 전략
| 대상 Phase | 통합 전략 | 예시 | 상태 |
|---|---|---|---|
| Phase1 | 매크로+ETF+기업 DB 통합 적재 | `macro_series`, `etf_snapshot`, `ra_company_fundamentals` | 구현 |
| Phase2 | 스냅샷 비교/변화율 뷰 기반 분석 | `valuation_snapshot_mv`, `momentum_rolling_avg` | 구현 |
| Phase6 | 전략 성과 SQL 저장 및 비교 | `ra_backtest_runs`(수익률, MDD, Sharpe) | 구현 |
| Phase7 | 리포트 본문에 SQL 근거 표/코드 삽입 | `allocation_report_agent` Section 6/10 | 구현 |
| Phase9 | SQL 결과/로그 아티팩트 export | `phase9_artifacts.export_artifacts` + report artifact metadata | 구현 |

### SQL 구현 증빙 스냅샷
| Metric | Value |
|---|---:|
| PG stored_rows | 5 |
| Internal SQL upserted_rows | 5 |
| Internal SQL total_rows | 5 |
| ETF snapshot rows | 10 |
| Backtest SQL total_runs | 6 |
| Backtest price source | Market Data |
| Allocation signal rows | 1 |
| Company coverage count | 5 |
| ETF coverage count | 10 |

### RA 종합 시그널 스냅샷
| AsOf | Valuation Score | ETF Breadth Score | Macro Proxy Score | Composite Score | Signal | Companies | ETFs |
|---|---:|---:|---:|---:|---|---:|---:|
| 2026-02-08 | 0.0 | 10.0 | 3.417 | 4.183 | NEUTRAL | 5 | 10 |

### SQL 근거 요약 (데이터 기반)
- PostgreSQL `fi_ra.company_fundamentals` 저장 5건, 내부 SQL `ra_company_fundamentals` upsert 5건(총 5건) 적재 완료.
- 기업 커버리지 5종목, ETF 커버리지 10개 기준으로 `ra_valuation_snapshot_mv`/`ra_etf_momentum_snapshot_mv`/`ra_allocation_signal_mv` 미리보기 데이터 생성.
- 백테스트 SQL 저장 건수는 6이며, 가격 소스는 Market Data.
- 종합 시그널 snapshot rows=1 (signal=NEUTRAL).
- Phase 상태: Phase2=구현, Phase6=구현, Phase9=구현.

### 구현 TODO (RA 스타일 고도화)
| Priority | Task | Why | Implementation Plan | Output Artifact | Status |
|---|---|---|---|---|---|
| P1 | ETF 수익률 누락 제거(데이터 소스 이중화) | ETF 수익률 누락이 해소되어 스냅샷 완결성을 확보함 | financial_indicators + yfinance + synthetic fallback + 캐시 경로 고정으로 누락 최소화 | `ra_etf_snapshot`, `ra_etf_momentum_snapshot_mv` | done |
| P1 | 실시장 가격 기반 백테스트 재실행 체계 | 실시장 가격 백테스트가 반영되어 synthetic 의존이 해소됨 | DNS/네트워크 복구 후 `--backtest-require-market-data` 모드로 재실행 및 비교 저장 | `ra_backtest_runs`, `ra_backtest_compare_mv` | done |
| P2 | RA 코멘트 Prompt/Response 로그 저장 | 코멘트 생성 로그가 저장되어 재현성/감사추적이 가능 | 입력 스냅샷/프롬프트/응답/모델명을 `ra_commentary_audit_log`에 저장 | `ra_commentary_audit_log` (new table) | done |
| P2 | 거시-ETF-기업 종합 점수화 | 종합 시그널 뷰가 생성되어 의사결정 점수 연결 완료 (rows=1) | SQL view `vw_ra_allocation_signal` 기반으로 valuation + ETF breadth + macro proxy composite 계산 | `ra_allocation_signal_mv` (new view) | done |
| P3 | PDF 본문 시각화 캡션 자동생성 | 데이터 기반 캡션이 생성되어 시각자료 해석 일관성 확보 (figures=8) | 각 figure note를 수치 기반 인사이트 문장으로 자동 생성 | report markdown + pdf figure captions | done |

### AI 토론/해석
- 최종 RA 콜: BULLISH.
- 구현 TODO 완료 5/5건으로 운영 고도화 진행률을 명시합니다.

---
*본 리포트는 EIMAS JSON 결과를 기반으로 자동 생성되었습니다.*
*새로운 숫자나 비중은 생성되지 않았으며, 모든 값은 JSON에서 인용되었습니다.*

## 부록 A. PDF 시각자료

### Figure 1. 리스크 점수 분해
![리스크 점수 분해](figures/allocation_report_20260209_053130/risk_score_decomposition.png)
- 설명: 최종 리스크 2.5는 Base 11.4, Micro -0.9, Bubble +0.0, Extended -8.0 조합으로 산출.
- 출처: `risk_score / base_risk_score / *_adjustment`

### Figure 2. 최종 권고/신뢰도 스냅샷 (BULLISH)
![최종 권고/신뢰도 스냅샷 (BULLISH)](figures/allocation_report_20260209_053130/decision_snapshot.png)
- 설명: 최종 권고는 BULLISH, 신뢰도는 51.3% 수준.
- 출처: `final_recommendation / confidence`

### Figure 3. 거시 스냅샷(금리/신용/물가)
![거시 스냅샷(금리/신용/물가)](figures/allocation_report_20260209_053130/macro_snapshot_rates_credit.png)
- 설명: Fed Funds 3.64, 10Y 4.21, HY OAS 2.97 기준 매크로/신용 환경을 동시 점검.
- 출처: `fred_summary`

### Figure 4. 기업 밸류에이션 맵(P/E vs P/B)
![기업 밸류에이션 맵(P/E vs P/B)](figures/allocation_report_20260209_053130/company_valuation_map.png)
- 설명: 커버리지 5종목 기준 P/E 16.1~45.9, P/B 2.4~46.4 분포.
- 출처: `company_ra_analysis.companies[].valuation`

### Figure 5. ETF 모멘텀 스냅샷(20일 수익률)
![ETF 모멘텀 스냅샷(20일 수익률)](figures/allocation_report_20260209_053130/etf_momentum_snapshot.png)
- 설명: 상위 XLE +14.39%, 하위 XLF -2.93%로 모멘텀 스프레드 17.32%p.
- 출처: `company_ra_analysis.etf_strategy_snapshot`

### Figure 6. SQL 증빙 대시보드(적재 + 백테스트)
![SQL 증빙 대시보드(적재 + 백테스트)](figures/allocation_report_20260209_053130/sql_evidence_dashboard.png)
- 설명: 적재 5건, ETF 10건, Backtest 6건, Allocation MV 1건.
- 출처: `company_ra_analysis.postgresql / company_ra_analysis.internal_sql / paper_execution_backtest.ra_sql`

### Figure 7. 모니터링 대시보드 스냅샷(VIX/리스크/기회)
![모니터링 대시보드 스냅샷(VIX/리스크/기회)](figures/allocation_report_20260209_053130/monitoring_dashboard_snapshot.png)
- 설명: VIX 20.4, Market Risk 2.5, Opportunity 64.0, Pipeline Risk 2.5.
- 출처: `sentiment_analysis.vix_structure.vix_spot + risk_score + gap_analysis.confidence(%) + risk_score`

### Figure 8. 백테스트 성과 스냅샷(수익률/MDD/Sharpe/승률)
![백테스트 성과 스냅샷(수익률/MDD/Sharpe/승률)](figures/allocation_report_20260209_053130/backtest_metrics_snapshot.png)
- 설명: 총수익률 +7.67%, Sharpe 1.18, MaxDD 3.69%.
- 출처: `backtest_metrics / paper_execution_backtest.metrics`
