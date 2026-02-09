# EIMAS Module Inventory & Technical Specification
**Last Updated**: 2026-02-09
**System Version**: v2.1.2 (Real-World Agent Edition)

---

## 1. 🏗️ Core Architecture (시스템 구조)
시스템의 뼈대와 흐름을 제어하는 오케스트레이션 계층입니다.

| 모듈명 | 위치 | 주요 역할 | 핵심 기술/알고리즘 |
|---|---|---|---|
| **Main Orchestrator** | `main.py` | 전체 분석 프로세스(Phase 1~9) 순차 실행 제어 | `AsyncIO`, `Modular Pipeline` |
| **CLI Interface** | `cli/eimas.py` | 사용자 명령 처리 및 `main.py` 인자 포워딩 | `Argparse`, `Rich Console` |
| **API Server** | `api/main.py` | 외부 연동 및 프론트엔드용 REST API | `FastAPI`, `WebSocket`, `Pydantic` |
| **Config Manager** | `core/config.py` | 환경변수, API 키, 상수 중앙 관리 | `Dotenv`, `YAML Config` |
| **Database Ops** | `core/database.py` | SQLite DB 연결 및 세션 관리 | `SQLite3`, `SQLAlchemy` |

---

## 2. 🧠 Economic Analysis (경제 분석 엔진)
시장 데이터를 수학적/통계적으로 분석하여 통찰을 도출하는 핵심 두뇌입니다.

| 모듈명 | 위치 | 주요 역할 | 핵심 기술/알고리즘 |
|---|---|---|---|
| **Regime Detector** | `lib/regime_detector.py` | 시장 상태(Bull/Bear/Neutral) 판별 | `GMM (Gaussian Mixture)`, `HMM` |
| **Lasso Forecast** | `lib/lasso_model.py` | 거시경제 지표 예측 및 주요 변수 추출 | `LASSO Regression (L1 Regularization)` |
| **Liquidity Analyzer** | `lib/liquidity_analysis.py` | 유동성과 자산 가격 간 인과관계 분석 | `Granger Causality Test` |
| **Sector Rotation** | `lib/sector_rotation.py` | 경기 사이클에 따른 유망 섹터 선정 | `Relative Strength`, `Momentum Scoring` |
| **Macro Strategy** | `lib/genius_act_macro.py` | 스테이블코인 및 디지털 유동성 분석 | `Digital M2`, `Stablecoin Issuance Tracking` |
| **Causality Graph** | `lib/causal_network.py` | 지표 간 인과관계 네트워크 구축 | `PC Algorithm`, `Directed Acyclic Graph (DAG)` |

---

## 3. 🛡️ Risk Management (리스크 관리)
포트폴리오를 보호하고 시스템 리스크를 감지하는 방어 기제입니다.

| 모듈명 | 위치 | 주요 역할 | 핵심 기술/알고리즘 |
|---|---|---|---|
| **Risk Manager** | `lib/risk_manager.py` | 포트폴리오 리스크 측정 및 사이징 | `VaR (Value at Risk)`, `CVaR`, `Kelly Criterion` |
| **Microstructure** | `lib/microstructure.py` | 시장 미세구조 및 유동성 품질 분석 | `VPIN (Toxic Flow)`, `Amihud Lambda`, `Roll Spread` |
| **Bubble Detector** | `lib/bubble_detector.py` | 자산 가격 버블 형성 감지 | `Greenwood-Shleifer (Run-up & Volatility)` |
| **Shock Propagation** | `lib/shock_propagation_graph.py` | 위기 발생 시 전이 경로 시뮬레이션 | `Network Theory`, `Centrality Analysis` |
| **Critical Path** | `lib/critical_path.py` | 시장의 핵심 위험 경로 추적 | `VIX Decomposition`, `Credit Spread Monitor` |

---

## 4. 💰 Portfolio & Trading (자산 배분 및 실행)
실제 수익을 창출하기 위한 자산 배분 및 매매 실행 엔진입니다.

| 모듈명 | 위치 | 주요 역할 | 핵심 기술/알고리즘 |
|---|---|---|---|
| **Portfolio Optimizer** | `lib/portfolio_optimizer.py` | 최적 자산 배분 비중 계산 | **`HRP (Hierarchical Risk Parity)`**, **`MST (Minimum Spanning Tree)`** |
| **Paper Trader** | `lib/paper_trader.py` | 가상 매매 주문 체결 및 계좌 관리 | `Order Matching Sim`, `Slippage Model` |
| **Integrated Strategy** | `lib/integrated_strategy.py` | 다양한 시그널 종합 및 최종 판단 | `Weighted Voting`, `Signal Fusion` |
| **Correlation Monitor** | `lib/correlation_monitor.py` | 자산 간 상관관계 실시간 추적 | `Rolling Correlation`, `Diversification Ratio` |

---

## 5. 📡 Data Collection (데이터 수집)
분석의 기초가 되는 원자재(데이터)를 수집하는 파이프라인입니다.

| 모듈명 | 위치 | 주요 역할 | 데이터 소스 |
|---|---|---|---|
| **FRED Collector** | `lib/fred_collector.py` | 거시경제 지표(금리, 통화량 등) 수집 | Federal Reserve (FRED API) |
| **Data Collector** | `lib/data_collector.py` | 주식, 채권, ETF 시세 데이터 수집 | Yahoo Finance (yfinance) |
| **Crypto Collector** | `lib/crypto_collector.py` | 암호화폐 및 온체인 데이터 수집 | CoinGecko, Exchange APIs |
| **Realtime Stream** | `lib/binance_stream.py` | 초단타 분석용 실시간 시세 수신 | Binance WebSocket |
| **Market Indicators** | `lib/market_indicators.py` | 공포탐욕지수, VIX 등 심리 지표 수집 | CNN Fear&Greed, CBOE |

---

## 6. 🤖 AI & Reporting (인공지능 리포팅)
수치 데이터를 인간이 이해할 수 있는 언어로 변환하는 LLM 에이전트입니다.

| 모듈명 | 위치 | 주요 역할 | 활용 모델 |
|---|---|---|---|
| **AI Report Gen** | `lib/ai_report_generator.py` | 투자 제안서 및 시황 리포트 작성 | **Claude 3.5 Sonnet**, **Perplexity**, **GPT-4** |
| **Orchestrator** | `agents/orchestrator.py` | AI 에이전트 간 토론(Debate) 주재 | Multi-Agent Framework |
| **Whitening Engine** | `lib/whitening_engine.py` | AI 판단의 근거를 역추적(XAI) | Decision Tree Interpretation |
| **Auto Fact Check** | `lib/autonomous_agent.py` | AI 출력물의 사실 여부 교차 검증 | Web Search Grounding |

---

## 7. 🖥️ Frontend & Visualization (시각화)
사용자와 상호작용하는 대시보드 및 시각화 도구입니다.

| 모듈명 | 위치 | 주요 역할 | 기술 스택 |
|---|---|---|---|
| **Web Dashboard** | `frontend/` | 실시간 시장 모니터링 웹 UI | `Next.js 16`, `React`, `Tailwind CSS` |
| **Streamlit Dash** | `dashboard.py` | (Legacy) 데이터 분석용 간편 대시보드 | `Streamlit`, `Plotly` |
| **HTML Generator** | `lib/report_generator.py` | 정적 HTML 리포트 생성 | `Jinja2 Templates`, `Matplotlib` |

---

## 📂 디렉토리 구조 요약
```text
eimas/
├── main.py (Entry Point)
├── pipeline/ (Workflow Control)
├── lib/ (Core Logic & Algorithms)
├── agents/ (AI Persona Logic)
├── api/ (Backend Server)
├── frontend/ (Web UI)
├── data/ (DB & Cache)
├── outputs/ (Results)
└── docs/ (Documentation)
```
