# EIMAS Full Capability Report
**Generated Date**: 2026-01-19
**System Version**: v2.1.2

---

## 1. 🌐 Core Platform Capabilities (핵심 플랫폼 역량)
EIMAS는 단순한 분석 스크립트 모음이 아닌, 데이터 수집부터 의사결정까지 완결된 **AI 퀀트 플랫폼**입니다.

### 1.1 Multi-Agent Orchestration (멀티 에이전트 오케스트레이션)
- **Role-Based Debate**: Analyst(분석), skeptic(비판), Risk Manager(리스크 관리) 등 페르소나를 가진 AI들이 토론을 통해 편향 제거.
- **Dynamic Workflow**: 시장 상황(급락, 횡보)에 따라 에이전트 투입 순서와 분석 깊이를 동적으로 조절.
- **Autonomous Verification**: AI 산출물에 대한 자동 팩트체크 및 소스 검증 (Hallucination 방지).

### 1.2 Modular Pipeline Architecture (모듈형 파이프라인)
- **Phase-based Execution**: Collection → Analysis → Signal → Debate → Reporting의 5단계 파이프라인.
- **Independent Execution**: 각 모듈(리스크, 포트폴리오 등)을 CLI에서 독립적으로 호출 가능.
- **Real-time Stream**: WebSocket 기반 초단타 데이터 처리 파이프라인 (Binance 연동).

---

## 2. 🧠 Advanced Analytical Engines (고급 분석 엔진)

### 2.1 Macro & Regime Analysis (거시경제 및 레짐)
- **GMM Regime Detection**: Gaussian Mixture Model을 이용한 3-State(Bull/Bear/Neutral) 시장 국면 분류.
- **LASSO Forecasting**: L1 정규화를 통해 수백 개의 거시 변수 중 유의미한 변수만 골라내어 금리/지표 예측.
- **Shannon Entropy**: 시장의 불확실성을 엔트로피(무질서도)로 정량화.

### 2.2 Systemic Risk & Network Theory (시스템 리스크)
- **MST (Minimum Spanning Tree)**: 자산 간 상관관계 네트워크를 트리 구조로 시각화하여 리스크 허브(Hub) 식별.
- **Shock Propagation**: 특정 자산 충격 시 전이 경로 시뮬레이션 (Contagion Analysis).
- **Critical Path Monitoring**: VIX 분해, 신용 스프레드 등을 통해 위기 발생 임계 경로 추적.

### 2.3 Market Microstructure (시장 미세구조)
- **VPIN (Volume-Synchronized Probability of Informed Trading)**: 정보 기반 트레이더(세력)의 독성 주문 흐름 포착.
- **Amihud Illiquidity**: 가격 충격 대비 거래량을 측정하여 숨겨진 유동성 위기 감지.
- **Bubble Detector**: Greenwood-Shleifer 모델(가격 급등 + 변동성) 기반 버블 형성 조기 경보.

### 2.4 Crypto & Alternative Assets (크립토 및 대안자산)
- **Genius Act Macro**: 스테이블코인 발행량과 유동성을 연계한 'Digital M2' 지표 산출.
- **On-chain Analysis**: 디파이(DeFi) TVL, 스테이블코인 시가총액 등 온체인 데이터 통합 분석.
- **RWA Integration**: 토큰화된 국채(ONDO), 금(PAXG) 등 실물 연계 자산 분석.

---

## 3. 💰 Portfolio & Trading Engines (포트폴리오 및 트레이딩)

### 3.1 Optimization Algorithms (최적화 알고리즘)
- **HRP (Hierarchical Risk Parity)**: 상관관계 계층 구조를 반영한 리스크 균등 배분 (전통적 Risk Parity보다 안정적).
- **Custom ETF Builder**: 사용자 정의 테마(예: AI, 전쟁)에 맞는 종목을 자동 선별하여 ETF 구성.
- **Sector Rotation**: 경기 사이클(회복/확장/둔화/침체)에 따른 최적 섹터 로테이션 전략.

### 3.2 Execution & Management (실행 및 관리)
- **Paper Trading Engine**: 슬리피지, 수수료를 반영한 정교한 가상 매매 시스템.
- **Intraday Collector**: 장중 1분봉 실시간 수집 및 이상 징후 포착.
- **Trading DB**: 모든 거래 내역과 시그널을 SQLite에 영구 보존하여 성과 분석(Attribution) 지원.

---

## 4. 📝 AI Reporting & Explainability (리포팅 및 설명가능성)

### 4.1 Intelligent Reporting
- **Deep Research**: Perplexity API를 통해 최신 뉴스, 실적 발표, 지정학적 이슈를 실시간 반영.
- **Scenario Analysis**: Base/Bull/Bear 시나리오별 확률과 목표가 제시.
- **Comparison Engine**: 이전 리포트와 현재 상태를 비교하여 '변화(Change)' 중심의 인사이트 제공.

### 4.2 XAI (Explainable AI)
- **Whitening Engine**: 블랙박스인 AI/머신러닝 모델의 판단 근거를 해석 가능한 형태로 변환.
- **Causality Narrative**: Granger 인과관계 분석 결과를 자연어 인과관계 스토리로 변환.

---

## 5. 🛠️ Infrastructure & Tools
- **CLI Tools**: `eimas run`, `eimas trade` 등 직관적인 명령줄 도구.
- **Web Dashboard**: Next.js 기반 실시간 시장 모니터링 대시보드.
- **Cron Scheduler**: 서버 환경에서의 무인 자동화 실행 지원.
