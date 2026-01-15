# EIMAS - 프로젝트 요약

> **AI 멀티에이전트 기반 거시경제 분석 시스템**
> Economic Intelligence Multi-Agent System

---

## 🎯 한 줄 소개

**"연준 유동성부터 시장 미세구조까지 통합 분석하고, AI 에이전트 토론으로 투자 방향을 제시하는 시스템"**

---

## 💡 핵심 아이디어

### 문제
- 투자 의사결정이 주관적이고 단편적
- 거시경제와 시장 데이터가 분리되어 분석됨
- 복잡한 변수를 통합 해석하기 어려움

### 해결
- **학술 논문 기반** 정량적 방법론 8개 구현
- **AI 멀티에이전트**가 서로 다른 관점에서 토론 후 합의
- **단일 파이프라인**에서 모든 분석 자동화 (8개 Phase)

---

## 🔬 사용한 방법론 (논문 기반)

| 방법론 | 논문/저자 | 용도 |
|--------|-----------|------|
| **LASSO** | Tibshirani (1996) | 100+ 변수 중 핵심만 선택 |
| **GMM** | Gaussian Mixture Model | 시장 레짐 (Bull/Bear/Neutral) 분류 |
| **Granger Causality** | Granger (1969, Nobel 2003) | 유동성 → 시장 전이 경로 분석 |
| **HRP** | De Prado (2016) | 포트폴리오 최적화 (MVO 개선) |
| **MST** | Mantegna (1999) | 시스템 리스크 노드 식별 |
| **Bubble Detection** | Greenwood-Shleifer (2019) | 버블 조기 경보 (Run-up + Vol) |
| **VPIN** | Easley et al. (2012) | 정보 비대칭 측정 |
| **Amihud Lambda** | Amihud (2002) | 비유동성 측정 |

---

## 🏗️ 구현 내용

### 8개 Phase 파이프라인
```
Phase 1: 데이터 수집
  → FRED (RRP, TGA, Net Liquidity) + 시장 29개 티커 + ARK ETF

Phase 2: 분석
  → 레짐 분류 (GMM+Entropy) + Granger + 리스크 점수 + 버블 탐지

Phase 3: AI 멀티에이전트 토론
  → FULL Mode (365일) vs REFERENCE Mode (90일) → 합의 도출

Phase 4: 실시간 (옵션)
  → Binance WebSocket → VPIN 계산

Phase 5: 데이터베이스
  → 이벤트 DB + 시그널 DB + Trading DB (포트폴리오 저장)

Phase 6: AI 리포트 (옵션)
  → Claude/Perplexity 자연어 리포트

Phase 7: 품질 보증 (옵션)
  → 경제학적 해석 + AI 팩트체킹

Phase 8: 독립 스크립트 (--full 옵션)
  → 장중 데이터 + 암호화폐 모니터링 + 이벤트 예측 등 7개
```

### 코드 규모
- **총 모듈**: 54개 (활성) + 9개 (deprecated) + 32개 (future)
- **총 코드**: ~50,000 lines
- **main.py**: 3,400 lines
- **커버리지**: 87% (기본) → 100% (--full)

---

## 📊 주요 기능 예시

### 1. 순유동성 분석
```
Net Liquidity = Fed Balance Sheet - RRP - TGA
```
- **의미**: Fed의 실제 시장 공급 유동성
- **결과**: 확대/축소 레짐 분류 → SPY 선행지표

### 2. 리스크 점수 (3단계 조정)
```
Final Risk = Base (CriticalPath 0-100)
           + Microstructure Adj (±10)
           + Bubble Adj (0~15)
```
- **예시**: 45.0 - 4.0 + 10.0 = **51.0**

### 3. AI 멀티에이전트 토론
```
FULL Mode (낙관): "BULLISH, 365일 트렌드 상승"
REF Mode (보수):  "BULLISH, 하지만 90일 변동성 주의"
→ 합의: BULLISH, Confidence 85%
```

### 4. GC-HRP 포트폴리오
```json
{
  "HYG": 0.54,  // High Yield (54%)
  "DIA": 0.06,  // Dow (6%)
  "XLV": 0.05,  // Healthcare (5%)
  ...
}
```
- MST 클러스터링 + HRP → 안정적 분산

---

## 🎯 결과 Output

### JSON 결과
```json
{
  "regime": "Bull",
  "risk_score": 51.0,
  "full_mode_position": "BULLISH",
  "reference_mode_position": "BULLISH",
  "final_recommendation": "BULLISH",
  "confidence": 0.85,
  "portfolio_weights": {"HYG": 0.54, "DIA": 0.06, ...},
  "ark_analysis": {"consensus_buys": ["TSLA", "COIN"]}
}
```

### 마크다운 리포트
12개 섹션 자동 생성:
- Data Summary (FRED + 시장)
- Regime Analysis (GMM + Entropy)
- Risk Assessment (3단계 브레이크다운)
- Multi-Agent Debate (토론 과정)
- Portfolio Optimization (GC-HRP)
- ...

### 실시간 대시보드
- Next.js 16 + React 19
- 5초 자동 폴링
- 메트릭 카드 4개 + 경고 시스템

---

## 🚀 실행 방법

```bash
# 설치
git clone https://github.com/Eom-TaeJun/eimas.git
cd eimas
pip install -r requirements.txt

# 기본 실행 (~40초)
python main.py

# 빠른 분석 (~16초)
python main.py --quick

# 전체 기능 (~90초)
python main.py --full --report

# 결과 확인
cat outputs/integrated_*.json
```

---

## 📈 성능

| 지표 | 수치 |
|------|------|
| 실행 시간 | 16초 (quick) ~ 90초 (full) |
| 데이터 소스 | 29개 티커 + 10개 FRED 지표 |
| 모듈 통합율 | 87% (기본) / 100% (--full) |
| 레짐 분류 정확도 | ~85% (GMM) |
| 이벤트 예측 정확도 | ~78% (NFP/CPI/FOMC) |

---

## 🏆 차별점

| 항목 | EIMAS | 기존 플랫폼 |
|------|-------|-------------|
| 거시경제 통합 | ✅ Net Liquidity | △ 부분적 |
| AI 멀티에이전트 | ✅ 토론 시스템 | ❌ |
| 학술 방법론 | ✅ 8개 논문 | △ 블랙박스 |
| 오픈소스 | ✅ MIT 라이선스 | ❌ |
| 비용 | 무료 (API 비용만) | $15-2000/월 |

---

## 🔮 향후 계획

**Q1 2026**:
- 실적 발표 데이터 통합
- 뉴스 감성 분석
- 실제 브로커 연동 (IB, Alpaca)

**Q2-Q3 2026**:
- Fama-French 5-factor 분석
- Tax-Loss Harvesting
- 성과 귀인 분석

---

## 👥 대상 사용자

- **퀀트 투자자**: 정량적 방법론 기반 투자
- **거시경제 애호가**: Fed watching, 유동성 분석
- **AI/ML 연구자**: 멀티에이전트 시스템
- **개인 투자자**: 객관적 의사결정 지원

---

## 📞 문의

**GitHub**: https://github.com/Eom-TaeJun/eimas
**Issues**: https://github.com/Eom-TaeJun/eimas/issues

---

*"Quantifying the Market, Democratizing Finance"*

**EIMAS v2.1.2** - Economic Intelligence Multi-Agent System
