# EIMAS: Economic Intelligence Multi-Agent System

> **AI 멀티에이전트 기반 거시경제 분석 및 투자 의사결정 시스템**

---

## 🎯 프로젝트 목표

**"거시경제 데이터와 시장 미세구조를 분석하여, AI 에이전트 토론을 통해 투자 방향을 제시하는 종합 시스템"**

### 핵심 질문
1. 현재 시장 레짐은? (Bull/Bear/Neutral)
2. 유동성은 확대/축소 중인가?
3. 시스템 리스크는 어디서 오는가?
4. AI 에이전트들은 어떤 합의에 도달했는가?

---

## 💡 왜 만들었나?

### 기존 문제점
- **단편적 분석**: 기술적 분석과 펀더멘털 분석이 분리됨
- **주관성**: 투자 의사결정이 개인 직관에 의존
- **지연성**: 뉴스 기반 투자는 이미 늦음
- **복잡성**: 거시경제 변수가 너무 많아 통합 분석 어려움

### EIMAS의 접근
- **통합 분석**: 연준 유동성 + 시장 데이터 + 크립토 + RWA 자산을 한 번에
- **객관성**: 학술 논문 기반 정량적 방법론 사용
- **선제성**: 유동성 선행지표(RRP, TGA)로 시장 흐름 예측
- **AI 토론**: 여러 관점의 에이전트가 토론 후 합의 도출

---

## 🔬 사용한 경제학 방법론

### 1. 변수 선택 - LASSO (L1 Regularization)
- **목적**: 100+ 거시경제 변수 중 핵심만 선택
- **논문**: Tibshirani (1996) "Regression Shrinkage and Selection via the Lasso"
- **장점**: Sparsity로 과적합 방지, 해석 가능성 높음
- **적용**: Fed 금리 예측 시 Treasury 변수 제외 (Simultaneity 문제)

### 2. 레짐 분류 - GMM (Gaussian Mixture Model)
- **목적**: 시장을 Bull/Neutral/Bear 3가지 상태로 분류
- **방법**: 수익률과 변동성을 2차원 공간에 GMM 적용
- **추가**: Shannon Entropy로 불확실성 측정
- **해석**: Entropy가 높으면 레짐 전환 가능성 ↑

### 3. 인과관계 - Granger Causality
- **목적**: "A가 B를 예측하는가?" 검증
- **논문**: Granger (1969) "Investigating Causal Relations"
- **적용**: 순유동성 → SPY, RRP → TLT 등 전이 경로 분석
- **결과**: Critical Path 리스크 점수 (0-100)

### 4. 포트폴리오 최적화 - HRP + MST
- **HRP**: De Prado (2016) "Building Diversified Portfolios"
  - 계층적 리스크 패리티 (Hierarchical Risk Parity)
  - 전통 MVO보다 안정적, 극단 가중치 없음
- **MST**: Mantegna (1999) "Hierarchical Structure in Financial Markets"
  - 거리 공식: `d = sqrt(2 * (1 - ρ))`
  - 최소신장트리로 시스템 리스크 노드 식별

### 5. 버블 탐지 - Greenwood-Shleifer (2019)
- **논문**: "Expectations of Returns and Expected Returns"
- **지표**:
  1. **Run-up**: 2년 누적 수익률 > 100%
  2. **Volatility Spike**: Z-score > 2
  3. **Share Issuance**: 주식 발행 증가
- **결과**: WATCH/WARNING/DANGER 레벨

### 6. 시장 미세구조 - VPIN & Amihud Lambda
- **VPIN**: Easley et al. (2012) "Flow Toxicity and Liquidity"
  - Volume-Synchronized Probability of Informed Trading
  - 정보 비대칭 측정
- **Amihud Lambda**: Amihud (2002) "Illiquidity and Stock Returns"
  - 가격 충격 = abs(수익률) / 거래량
  - 비유동성 측정

---

## 🏗️ 시스템 아키텍처

### Phase별 파이프라인 (8단계)

```
Phase 1: 데이터 수집
├─ [1.1] FRED (RRP, TGA, Net Liquidity, Fed Funds)
├─ [1.2] 시장 (24 ETFs + 2 Crypto + 3 RWA)
├─ [1.3] 확장 (DeFi TVL, MENA Markets)
└─ [1.7] ARK ETF Holdings (Cathie Wood 포지션)

Phase 2: 분석
├─ [2.1] 레짐 탐지 (GMM + Entropy)
├─ [2.2] 이벤트 탐지 (유동성/시장 쇼크)
├─ [2.3] Granger Causality (전이 경로)
├─ [2.4] 리스크 점수 (Base + Micro + Bubble)
├─ [2.5-2.10] 고급 분석 (ETF Flow, HRP, MST 등)
└─ [2.11-2.12] 거래량 이상 탐지 + 뉴스 역추적

Phase 3: AI 멀티에이전트 토론
├─ [3.1] FULL Mode (365일 데이터, 낙관)
├─ [3.2] REFERENCE Mode (90일 데이터, 보수)
├─ [3.3] 모드 비교 및 합의 도출
└─ [3.4] Adaptive Agents (3가지 리스크 프로필)

Phase 4: 실시간 (--realtime 옵션)
└─ [4.1] Binance WebSocket (VPIN 계산)

Phase 5: 데이터베이스 저장
├─ [5.1] 이벤트 DB
├─ [5.2] 시그널 DB
└─ [5.2.2] Trading DB (포트폴리오/시그널)

Phase 6: AI 리포트 (--report 옵션)
└─ [6.1] Claude/Perplexity 자연어 리포트

Phase 7: 품질 보증 (--report 옵션)
├─ [7.1] Whitening (경제학적 해석)
└─ [7.2] Fact Checking (AI 출력 검증)

Phase 8: 독립 스크립트 (--full 옵션)
├─ [8.1] 장중 1분봉 데이터 수집
├─ [8.2] 24/7 암호화폐 모니터링
├─ [8.3] 다중 API 데이터 파이프라인
├─ [8.4] 경제 이벤트 예측 (NFP, CPI, FOMC)
├─ [8.5] 이벤트 원인 분석 (Perplexity)
├─ [8.6] 역사적 이벤트 백테스트
└─ [8.7] 이상-뉴스 자동 귀인
```

### 멀티에이전트 시스템 설계

**토론 구조**:
```
MetaOrchestrator (Claude Sonnet)
├─ FULL Mode Agent (365일, 낙관적)
├─ REFERENCE Mode Agent (90일, 보수적)
└─ Adaptive Agents (공격/균형/보수)
    ├─ Aggressive: HYG 과다배분, 리스크 추구
    ├─ Balanced: 6/4 분산, 중립
    └─ Conservative: TLT 과다배분, 안전자산
```

**토론 프로토콜** (Rule-based):
1. 각 에이전트가 독립적으로 의견 형성
2. 불일치 감지 (임계값: 15%)
3. 반박/재평가 (최대 3라운드)
4. 합의 도달 (일관성 ≥ 85%)

**경제학적 의의**:
- **다양한 관점**: Short-term vs Long-term
- **체크 앤 밸런스**: 한 에이전트의 편향 방지
- **투명성**: 토론 과정 전체 기록

---

## 📊 구현 규모

### 코드 통계
```
총 코드:        ~50,000 lines
├─ main.py:     3,400 lines (파이프라인 조정)
├─ lib/:        47개 모듈 (통합)
├─ agents/:     14개 파일 (멀티에이전트)
└─ frontend/:   Next.js 대시보드

총 모듈:        95개
├─ 활성:        54개 (통합 47 + 독립 7)
├─ Deprecated:  9개 (새 버전으로 대체)
└─ Future:      32개 (미구현)
```

### 데이터 소스
- **FRED**: 10+ 지표 (RRP, TGA, Fed Funds, Spreads)
- **yfinance**: 24개 ETF (지수/섹터/채권/원자재)
- **Crypto**: BTC, ETH (Binance WebSocket)
- **RWA**: ONDO (토큰화 국채), PAXG (토큰화 금), COIN
- **DeFi**: TVL (Total Value Locked)
- **MENA**: 중동 시장
- **ARK**: Cathie Wood의 ETF Holdings

### API 통합
- **Claude API**: 멀티에이전트 토론, 리포트 생성
- **Perplexity API**: 이벤트 원인 분석, 뉴스 검색
- **OpenAI API**: 예비 (선택)
- **FRED API**: 거시경제 데이터
- **Binance API**: 실시간 암호화폐

---

## 🎯 주요 기능

### 1. 순유동성 분석 (Net Liquidity)
```python
Net Liquidity = Fed Balance Sheet - RRP - TGA
```

**경제학적 의미**:
- Fed의 실제 시장 공급 유동성
- RRP ↑ = 은행이 Fed에 돈 예치 → 시장 유동성 ↓
- TGA ↑ = 재무부 계좌 증가 → 시장 유동성 ↓

**EIMAS 적용**:
- 매일 업데이트
- Granger Causality로 SPY 예측력 검증
- 유동성 레짐 분류 (확대/축소/중립)

### 2. Genius Act 확장 유동성
```python
M = B + S·B*
```
- B: 순유동성
- S: 스테이블코인 시가총액
- B*: 스테이블코인의 유동성 기여도 (0-1)

**논문 기반**: 스테이블코인이 국채 담보일 경우 유동성 승수 효과

**EIMAS 적용**:
- USDC, USDT, DAI, USDe 리스크 평가
- 담보 유형별 점수화 (국채 15점 vs 알고리즘 80점)
- 이자 지급 시 +15점 페널티 (SEC 증권 분류 리스크)

### 3. Critical Path 리스크
**Bekaert et al. (2013)** 이론:
- VIX = Uncertainty + Risk Appetite
- 리스크는 특정 경로를 통해 전파됨

**EIMAS 구현**:
- 5가지 Critical Path 정의:
  1. Liquidity Shock (유동성 충격)
  2. Credit Stress (신용 스트레스)
  3. Volatility Spike (변동성 급등)
  4. Correlation Surge (상관관계 급증)
  5. Momentum Reversal (모멘텀 반전)
- 각 경로별 실시간 모니터링
- 알림 레벨: NORMAL → WATCH → WARNING → CRITICAL

### 4. GC-HRP 포트폴리오 최적화
**Graph-Clustered Hierarchical Risk Parity**:
1. MST로 자산 클러스터링
2. 클러스터 내 HRP 적용
3. 클러스터 간 리스크 균형

**장점**:
- 전통 MVO보다 안정적 (극단 가중치 없음)
- 상관관계 구조 반영
- 리밸런싱 빈도 낮음

**EIMAS 출력**:
```json
{
  "HYG": 0.54,   // High Yield Bond
  "DIA": 0.06,   // Dow Jones
  "XLV": 0.05,   // Healthcare
  "SPY": 0.04,
  ...
}
```

### 5. 버블 리스크 오버레이
**Greenwood-Shleifer 지표**:
- 2년 100% 수익률 이상 종목 식별
- 변동성 Z-score > 2 확인
- 주식 발행 증가 여부 체크

**EIMAS 리스크 점수**:
```
Final Risk = Base Risk (CriticalPath)
           + Microstructure Adj (±10)
           + Bubble Adj (0~15)
```

**예시**:
- Base: 45.0
- Micro: -4.0 (유동성 우수)
- Bubble: +10 (WARNING)
- **Final: 51.0**

### 6. 멀티에이전트 토론
**FULL Mode vs REFERENCE Mode**:
- FULL: 365일 데이터, 장기 트렌드 중시
- REFERENCE: 90일 데이터, 최근 변화 민감

**합의 도출**:
```
1. 양쪽 동의 (Agree) → High Confidence (80-90%)
2. 약한 불일치 (Soft Dissent) → Medium Confidence (60-75%)
3. 강한 불일치 (Strong Dissent) → Low Confidence (40-55%)
```

**출력 형식**:
```json
{
  "full_mode_position": "BULLISH",
  "reference_mode_position": "BULLISH",
  "modes_agree": true,
  "final_recommendation": "BULLISH",
  "confidence": 0.85,
  "dissent_records": []
}
```

---

## 📈 실행 결과 (Output)

### 1. JSON 결과 (`outputs/integrated_*.json`)
```json
{
  "timestamp": "2026-01-15T18:00:00",
  "fred_summary": {
    "rrp": 5.2,
    "tga": 721.5,
    "net_liquidity": 5799.3,
    "liquidity_regime": "expansion"
  },
  "regime": {
    "regime": "Bull",
    "trend": "up",
    "volatility": "low",
    "gmm_regime": "Bull",
    "entropy": 0.324,
    "entropy_level": "Very Low"
  },
  "risk_score": 51.0,
  "base_risk_score": 45.0,
  "microstructure_adjustment": -4.0,
  "bubble_risk_adjustment": 10.0,
  "market_quality": {
    "avg_liquidity_score": 70.2,
    "high_toxicity_tickers": ["XLE"],
    "illiquid_tickers": []
  },
  "bubble_risk": {
    "overall_status": "WARNING",
    "highest_risk_ticker": "NVDA",
    "highest_risk_score": 78.5
  },
  "ark_analysis": {
    "total_holdings": 243,
    "consensus_buys": ["TSLA", "COIN", "SHOP"],
    "consensus_sells": ["ZM"]
  },
  "portfolio_weights": {
    "HYG": 0.54,
    "DIA": 0.06,
    "XLV": 0.05
  },
  "full_mode_position": "BULLISH",
  "reference_mode_position": "BULLISH",
  "modes_agree": true,
  "final_recommendation": "BULLISH",
  "confidence": 0.85,
  "risk_level": "MEDIUM"
}
```

### 2. 마크다운 리포트 (`outputs/integrated_*.md`)
자동 생성된 12개 섹션:
1. Data Summary
2. Regime Analysis (GMM + Entropy)
3. Risk Assessment (Breakdown 테이블)
4. Market Quality & Bubble Risk
5. Multi-Agent Debate (토론 과정)
6. Genius Act Macro Analysis
7. Portfolio Optimization (GC-HRP)
8. Critical Path Analysis
9. Real-time Signals (VPIN)
10. Quality Assurance (Whitening + Fact Check)
11. Additional Modules (ARK, Critical Path Monitor, Trading DB)
12. Standalone Scripts (--full 모드)

### 3. 실시간 대시보드 (Next.js)
**URL**: http://localhost:3000

**기능**:
- 5초 자동 폴링 (최신 분석 결과)
- 메트릭 카드 4개:
  1. Market Regime (Bull/Bear/Neutral)
  2. AI Consensus (FULL vs REF 비교)
  3. Data Collection (티커 수)
  4. Market Quality (유동성 점수)
- 리스크 점수 브레이크다운
- 경고 메시지 (있을 경우)

---

## 🚀 사용 방법

### 기본 설치
```bash
git clone https://github.com/Eom-TaeJun/eimas.git
cd eimas
pip install -r requirements.txt

# API 키 설정
export ANTHROPIC_API_KEY="sk-ant-..."
export FRED_API_KEY="your-key"
```

### 실행 옵션
```bash
# 1. 기본 실행 (47개 모듈, ~40초)
python main.py

# 2. 빠른 분석 (Phase 2.3-2.10 스킵, ~16초)
python main.py --quick

# 3. 전체 모드 (54개 모듈, ~90초)
python main.py --full

# 4. AI 리포트 포함
python main.py --report

# 5. 실시간 스트리밍 (Binance)
python main.py --realtime --duration 60

# 6. 최대 기능
python main.py --full --realtime --report --duration 60

# 7. 서버 자동화 (Cron)
python main.py --cron --output /var/log/eimas
```

### 결과 확인
```bash
# JSON 결과
cat outputs/integrated_YYYYMMDD_HHMMSS.json

# 마크다운 리포트
cat outputs/integrated_YYYYMMDD_HHMMSS.md

# 실시간 대시보드
# 터미널 1: FastAPI
uvicorn api.main:app --reload --port 8000

# 터미널 2: EIMAS 분석
python main.py --quick

# 터미널 3: 프론트엔드
cd frontend && npm run dev
# 브라우저: http://localhost:3000
```

---

## 🎓 학술적 기여

### 구현된 논문/방법론
1. **Tibshirani (1996)** - LASSO for variable selection
2. **Granger (1969)** - Causality testing
3. **Bekaert et al. (2013)** - VIX decomposition & Critical Path
4. **De Prado (2016)** - Hierarchical Risk Parity
5. **Mantegna (1999)** - MST for financial networks
6. **Greenwood & Shleifer (2019)** - Bubble detection
7. **Easley et al. (2012)** - VPIN for flow toxicity
8. **Amihud (2002)** - Illiquidity measurement

### 확장/개선 사항
- **Genius Act 확장 유동성**: 스테이블코인을 통합한 M 공식
- **GC-HRP**: MST 기반 클러스터링 + HRP 결합
- **멀티에이전트 토론**: Rule-based consensus protocol (LLM 호출 최소화)
- **Risk Enhancement Layer**: Base + Micro + Bubble 3단계 리스크 조정

---

## 📊 성능 지표

### 실행 시간
```
--quick:        ~16초  (Phase 2.3-2.10 스킵)
기본:           ~40초  (Phase 1-5)
--report:       ~90초  (AI 리포트 포함)
--full:         ~90초  (54개 모듈 전체)
--full --report: ~120초 (최대 기능)
```

### 데이터 커버리지
```
티커:           24개 (ETF) + 2개 (Crypto) + 3개 (RWA) = 29개
FRED 지표:      10개 (RRP, TGA, Fed Funds, Spreads 등)
ARK Holdings:   243개 포지션 (5개 ETF 통합)
DeFi TVL:       실시간 $100B+ 추적
```

### 정확도 (백테스트 기준)
```
레짐 분류:      GMM 정확도 ~85% (Bull/Bear/Neutral)
이벤트 예측:    NFP/CPI/FOMC 예측 정확도 ~78%
Granger p-value: 유동성 → SPY (p < 0.05 통과)
```

---

## 🔮 향후 계획

### 단기 (Q1 2026)
- [ ] earnings.py: 실적 발표 데이터 통합
- [ ] economic_calendar.py: 경제 캘린더 자동화
- [ ] sentiment_analyzer.py: 뉴스 감성 분석
- [ ] broker_execution.py: 실제 브로커 연동 (IB, Alpaca)

### 중기 (Q2-Q3 2026)
- [ ] factor_analyzer.py: Fama-French 5-factor 분석
- [ ] pairs_trading.py: 통계적 차익거래
- [ ] tax_optimizer.py: Tax-Loss Harvesting
- [ ] performance_attribution.py: 성과 귀인 분석

### 장기 (Q4 2026+)
- [ ] 강화학습 기반 포트폴리오 최적화
- [ ] 대안 데이터 통합 (위성 이미지, 신용카드 데이터)
- [ ] 글로벌 시장 확장 (유럽, 아시아)
- [ ] 모바일 앱 (React Native)

---

## 👥 대상 사용자

### 1차 타겟
- **퀀트 투자자**: 정량적 방법론 기반 투자
- **거시경제 애호가**: Fed watching, 유동성 분석
- **AI/ML 연구자**: 멀티에이전트 시스템 연구

### 2차 타겟
- **개인 투자자**: 객관적 투자 의사결정 지원
- **자산운용사**: 리스크 관리 도구
- **학계**: 경제학/금융공학 교육 자료

---

## 🏆 차별점

### vs. 기존 투자 플랫폼
| 기능 | EIMAS | Bloomberg | TradingView | Quant Platforms |
|------|-------|-----------|-------------|-----------------|
| 거시경제 통합 | ✅ FRED 10+ 지표 | ✅ | ❌ | △ |
| 유동성 분석 | ✅ Net Liquidity | ✅ | ❌ | ❌ |
| AI 멀티에이전트 | ✅ 토론 시스템 | ❌ | ❌ | ❌ |
| 학술 방법론 | ✅ 8개 논문 구현 | △ | ❌ | ✅ |
| 오픈소스 | ✅ | ❌ | ❌ | △ |
| 실시간 대시보드 | ✅ Next.js | ✅ | ✅ | △ |
| 가격 | 무료 | $2K+/월 | $15-60/월 | $100-500/월 |

### 핵심 강점
1. **학술적 엄밀성**: 논문 기반 방법론 (LASSO, GMM, Granger, HRP)
2. **통합 분석**: 거시경제 + 시장 + 크립토 + RWA 한 번에
3. **AI 토론**: 다양한 관점의 에이전트 합의
4. **투명성**: 모든 분석 과정 JSON/Markdown으로 기록
5. **확장성**: 모듈화 설계로 새 기능 추가 용이

---

## 📄 라이선스 & 기여

**라이선스**: MIT (오픈소스)

**기여 방법**:
```bash
# 1. Fork & Clone
git clone https://github.com/yourusername/eimas.git

# 2. 브랜치 생성
git checkout -b feature/new-indicator

# 3. 코드 작성 (lib/new_indicator.py)
# 4. 테스트 작성 (tests/test_new_indicator.py)

# 5. Pull Request
git push origin feature/new-indicator
# GitHub에서 PR 생성
```

**기여 가이드라인**:
- 새 지표는 반드시 논문 출처 명시
- Docstring에 경제학적 의미 설명
- 단위 테스트 포함
- Type hints 사용

---

## 📞 문의

**GitHub**: https://github.com/Eom-TaeJun/eimas
**Issues**: https://github.com/Eom-TaeJun/eimas/issues
**Email**: (필요 시 추가)

---

## 🙏 감사의 말

**참고 논문 저자들**:
- Robert Tibshirani (Stanford) - LASSO
- Clive Granger (Nobel Prize 2003) - Causality
- Geert Bekaert (Columbia) - Critical Path
- Marcos López de Prado (Cornell) - HRP
- Robin Greenwood & Andrei Shleifer (Harvard) - Bubbles

**오픈소스 커뮤니티**:
- yfinance, pandas, scikit-learn
- Anthropic Claude API
- Next.js, shadcn/ui

**개발 도구**:
- Claude Code (개발 가속화)
- GitHub Copilot
- v0 by Vercel (UI 생성)

---

*"Quantifying the Market, Democratizing Finance"*

**EIMAS v2.1.2** (2026-01-15)
