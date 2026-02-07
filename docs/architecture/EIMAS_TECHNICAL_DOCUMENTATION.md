# EIMAS 기술 백서 (Technical Whitepaper)

**프로젝트명:** EIMAS (Economic Intelligence Multi-Agent System)  
**버전:** v2.1.2 (Modular Architecture)  
**작성일:** 2026년 1월 16일

---

## 1. 프로젝트 개요 (Overview)

### 1.1 개발 배경
현대 금융 시장은 수많은 거시경제 지표, 기업 실적, 뉴스, 그리고 시장 미세구조 데이터가 복잡하게 얽혀 있습니다. EIMAS는 **"데이터에 기반한 정량적 분석(Math)"**과 **"AI의 정성적 추론(Reasoning)"**을 결합한 **뉴로-심볼릭(Neuro-Symbolic) AI** 시스템을 지향합니다.

### 1.2 핵심 철학: 정반합(Dialectic)
단일 AI 모델의 환각(Hallucination)과 편향을 방지하기 위해, 서로 다른 관점을 가진 AI 에이전트들이 **경쟁하고 토론(Debate)**하여 최적의 합의점을 도출합니다.

---

## 2. 시스템 아키텍처 (System Architecture)

`main.py`의 `run_integrated_pipeline`가 지휘자가 되어 8단계의 전문 파이프라인을 실행합니다.

| 단계 | 모듈명 (`pipeline/`) | 역할 | 주요 기술 |
| :--- | :--- | :--- | :--- |
| **1. 수집** | `pipeline_data.py` | 데이터 수집 | FRED API, yfinance, DeFiLlama |
| **2. 분석** | `pipeline_analysis.py` | 정량 가공 | **GMM, HRP, Granger Causality** |
| **3. 토론** | `pipeline_debate.py` | **AI 추론** | **Multi-Agent Debate, Adaptive Logic** |
| **4. 실시간** | `pipeline_realtime.py` | 감시 | WebSocket, VPIN Algorithm |
| **5. 저장** | `pipeline_storage.py` | 기록 | SQLite (Events, Signals, Trading) |
| **6. 리포트** | `pipeline_report.py` | 소통 | **LLM (Claude) + Perplexity** |
| **7. 검증** | `pipeline_report.py` | 품질 보증 | **Self-Correction Loop** |
| **8. 확장** | `pipeline_standalone.py` | 심화 분석 | Intraday Anomaly Detection |

---

## 3. 정량적 분석 엔진 (Quantitative Engine)

금융공학 논문을 코드로 구현한 "수학적 두뇌"입니다.

### 3.1 시장 레짐 탐지 (GMM & Entropy)
*   **코드:** `lib/regime_detector.py`, `lib/regime_analyzer.py`
*   **기술:** Gaussian Mixture Model (비지도 학습)
*   **원리:** 시장 수익률 분포를 여러 개의 정규분포(상승/하락/횡보)로 군집화하고, **섀넌 엔트로피(Shannon Entropy)**를 통해 현재 시장 상태의 불확실성을 수치화합니다.

### 3.2 유동성 전이 분석 (Liquidity Causality)
*   **코드:** `lib/liquidity_analysis.py`
*   **기술:** Granger Causality Test
*   **원리:** 연준의 실질 유동성(Fed BS - RRP - TGA) 변화가 시차를 두고 주가(SPY) 변동을 통계적으로 유의미하게 선행하는지 검증합니다.

### 3.3 그래프 군집 포트폴리오 (GC-HRP)
*   **코드:** `lib/graph_clustered_portfolio.py`
*   **기술:** MST (Minimum Spanning Tree), Hierarchical Risk Parity
*   **원리:** 자산 간 상관관계를 네트워크 그래프로 그려 군집(Cluster)을 찾고, 각 군집의 위험 기여도가 동일하도록 비중을 배분하여 하락장 방어력을 극대화합니다.

---

## 4. AI 에이전트 및 추론 아키텍처 (Advanced AI Agents)

EIMAS의 가장 큰 특징인 "언어적 두뇌"와 "에이전트 워크플로우"에 대한 상세 기술입니다.

### 4.1 멀티 페르소나 토론 (Multi-Persona Debate)
*   **코드:** `agents/orchestrator.py`, `pipeline/pipeline_debate.py`
*   **작동 방식:**
    1.  **Full Mode Agent (Thesis):** 1년치 장기 데이터를 기반으로 펀더멘털과 추세를 중시하는 "가치 투자자" 페르소나를 가집니다. (낙관적 편향)
    2.  **Reference Mode Agent (Antithesis):** 최근 3개월 데이터를 기반으로 변동성과 모멘텀을 중시하는 "트레이더" 페르소나를 가집니다. (비관적/민감 편향)
    3.  **Synthesis (합의):** `MetaOrchestrator`가 두 에이전트의 의견 차이를 분석하고, 논리적 타당성을 평가하여 최종 결론(Consensus)을 도출합니다. 만약 의견이 극명하게 갈리면 리스크 레벨을 자동으로 상향합니다.

### 4.2 상황 적응형 에이전트 (Adaptive Agents)
*   **코드:** `lib/adaptive_agents.py`
*   **기술:** Dynamic Strategy Switching
*   **원리:** Phase 2에서 탐지된 `Regime`(시장 상황)에 따라 실행할 에이전트의 성격을 실시간으로 교체합니다.
    *   **Aggressive Agent:** Bull Market + High Liquidity 일 때 활성화. 암호화폐 및 기술주 비중 확대.
    *   **Conservative Agent:** Bear Market + Liquidity Drain 일 때 활성화. 현금 및 국채(TLT) 비중 확대.
    *   **Balanced Agent:** 불확실성(Entropy High) 구간에서 활성화. 올웨더 포트폴리오 전략 수행.

### 4.3 자기 교정 루프 (Self-Correction Loop)
*   **코드:** `lib/validation_agents.py`
*   **기술:** Chain-of-Thought (CoT) Verification
*   **원리:** 에이전트가 결정을 내린 후, 즉시 실행하지 않고 **검증자(Validator) 에이전트**에게 검토를 맡깁니다.
    *   "이 공격적인 투자가 현재 리스크 점수(80점)와 모순되지 않는가?"
    *   모순이 발견되면 의사결정을 수정(Refine)하는 루프를 최대 2회 반복합니다.

### 4.4 인과관계 추적 (Causal Discovery with Perplexity)
*   **코드:** `lib/event_tracker.py`, `lib/news_correlator.py`
*   **기술:** RAG (Retrieval-Augmented Generation)
*   **원리:**
    1.  `VolumeAnalyzer`가 특정 종목의 이상 거래량(Anomaly)을 감지합니다. (수학적 탐지)
    2.  시스템이 즉시 **Perplexity API**를 호출하여 해당 시점의 뉴스를 검색합니다.
    3.  LLM이 뉴스 내용과 주가 변동 사이의 인과관계를 분석하여 "이 거래량 급증은 실적 발표 기대감 때문임"과 같이 원인을 규명합니다.

---

## 5. 데이터베이스 및 출력 (Outputs)

### 5.1 데이터베이스 스키마
*   **`events.db`:** 시장 충격, 유동성 이벤트, 당시 시장 스냅샷 (Time-series)
*   **`trading.db`:** 생성된 포트폴리오 바스켓, 매매 시그널, 전략 근거 (Transaction)
*   **`signals.db`:** 실시간 VPIN 알림, 리스크 경보 로그 (Logs)

### 5.2 사용자 인터페이스
*   **Markdown Report:** 모든 분석 과정과 에이전트의 토론 내역이 담긴 상세 문서.
*   **Web Dashboard:** React(Next.js) 기반으로 DB와 연동되어 실시간 시장 상황과 포트폴리오를 시각화.

---

## 6. 결론

EIMAS는 기존 퀀트 시스템의 **정확성**에 LLM 에이전트의 **유연성**을 더했습니다. 수학적 모델이 "무슨 일이 일어났는지(What)"를 감지하면, AI 에이전트가 "왜 일어났는지(Why)"를 추론하고 "어떻게 대응해야 하는지(How)"를 결정하는 **완전 자동화된 투자 인텔리전스**입니다.
