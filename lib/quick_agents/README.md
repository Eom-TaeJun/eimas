# Quick Mode AI Agents

> **Quick 모드 전용 AI 검증 에이전트 시스템**
>
> Full 모드 결과를 경제학 이론과 최신 학계 연구로 검증

---

## 개요

**Quick Mode AI Agents**는 Full 모드 (`eimas_*.json`)에서 도출한 자산배분 결과를 5개의 전문 AI 에이전트로 검증하는 시스템입니다.

### 핵심 차별점

| 특징 | Full Mode Agents | Quick Mode Agents |
|------|------------------|-------------------|
| 목적 | 시장 분석 및 진단 | Full 결과 검증 |
| 데이터 소스 | FRED, yfinance, 크립토 | Full JSON + API 리서치 |
| 실행 시간 | ~4-5분 | ~30-60초 |
| AI 모델 | Claude (토론) | Claude + Perplexity (검증) |
| KOSPI/SPX | 통합 분석 | **완전 분리 분석** |
| 학계 리서치 | 없음 | **Perplexity로 최신 논문 검색** |

---

## 에이전트 구성

### 1. Portfolio Validator (Claude API)
**포트폴리오 이론 검증 에이전트**

- **역할**: 경제학 이론 적합성 검증
- **검증 항목**:
  - Markowitz Mean-Variance Optimization (1952)
  - Black-Litterman 모델 (1992)
  - Risk Parity (Qian 2005)
  - 분산투자 적정성
- **출력**:
  ```json
  {
    "validation_result": "PASS" | "WARNING" | "FAIL",
    "theory_compliance": {
      "markowitz_mvo": "compliant",
      "risk_parity": "appropriate",
      "diversification": "adequate"
    },
    "risk_assessment": {...},
    "recommendations": [...]
  }
  ```

### 2. Allocation Reasoner (Perplexity API)
**자산배분 논리 분석 에이전트**

- **역할**: 학계 최신 연구로 배분 논리 검증
- **검색 대상**:
  - scholar.google.com (학술 논문)
  - ssrn.com (SSRN 논문)
  - arxiv.org (프리프린트)
- **검증 항목**:
  - 최근 학계 연구 (2020-2026)
  - AQR, Bridgewater, BlackRock 권고사항
  - 대안 전략 존재 여부
- **출력**:
  ```json
  {
    "reasoning_quality": "STRONG" | "MODERATE" | "WEAK",
    "academic_support": {
      "num_citations": 7,
      "key_findings": [...],
      "has_recent_research": true
    },
    "alternative_views": [...],
    "risk_factors": [...]
  }
  ```

### 3. Market Sentiment Agent (Claude API)
**시장 정서 분석 에이전트 (KOSPI + SPX 분리)**

- **역할**: 두 시장의 정서를 **독립적으로 분석**
- **KOSPI 전용 분석**:
  - FX (USDKRW) 영향
  - Samsung/SK Hynix 실적
  - 외국인 자금 흐름
  - 섹터 로테이션 (은행 vs 반도체)
- **SPX 전용 분석**:
  - Fed 통화정책
  - 빅테크 섹터 동향
  - 신용 스프레드
  - 시장 폭 (market breadth)
- **출력**:
  ```json
  {
    "kospi_sentiment": {
      "sentiment": "BULLISH",
      "confidence": 0.65,
      "key_factors": ["Strong Samsung earnings", "Weak USDKRW"],
      "sector_rotation": "Tech-led"
    },
    "spx_sentiment": {
      "sentiment": "BULLISH",
      "confidence": 0.70,
      "market_breadth": "Strong"
    },
    "comparison": {
      "divergence": "ALIGNED" | "MILD" | "STRONG",
      "implications": "...",
      "correlation_regime": "HIGH"
    }
  }
  ```

### 4. Alternative Asset Agent (Perplexity API)
**대체자산 판단 에이전트**

- **역할**: 크립토, 금, RWA 투자 판단
- **분석 대상**:
  - **Crypto**: BTC, ETH, Stablecoin 공급
  - **Commodities**: Gold (안전자산 vs 인플레이션 헤지), Oil
  - **RWA**: BlackRock BUIDL, Ondo OUSG, Franklin FOBXX
- **검색 소스**:
  - coindesk.com, theblock.co
  - bloomberg.com, ft.com
- **출력**:
  ```json
  {
    "crypto_assessment": {
      "recommendation": "BULLISH",
      "key_catalysts": ["ETF inflows", "Halving cycle"]
    },
    "commodity_assessment": {
      "gold_role": "SAFE_HAVEN",
      "diversification_benefit": "Yes"
    },
    "rwa_assessment": {
      "tokenization_trend": "ACCELERATING",
      "key_products": ["BLACKROCK BUIDL", "ONDO"]
    },
    "portfolio_role": {
      "recommended_allocation": {"crypto": "1-5%", "gold": "5-10%"},
      "correlation_benefits": "Provides diversification",
      "risk_considerations": [...]
    }
  }
  ```

### 5. Final Validator (Claude API)
**최종 검증 종합 에이전트**

- **역할**: 모든 에이전트 의견 종합 + Full vs Quick 비교
- **검증 항목**:
  - 4개 에이전트 간 합의도
  - Full 모드 vs Quick 모드 정합성
  - 최종 투자 권고 및 신뢰도
- **출력**:
  ```json
  {
    "validation_result": "PASS",
    "final_recommendation": "BULLISH",
    "confidence": 0.72,
    "agent_consensus": {
      "agreement_level": "HIGH",
      "disagreements": []
    },
    "full_vs_quick_comparison": {
      "alignment": "ALIGNED",
      "key_differences": [],
      "confidence_adjustment": "Increased"
    },
    "risk_warnings": [...],
    "action_items": [...]
  }
  ```

---

## 사용법

### 기본 실행

```python
from lib.quick_agents import QuickOrchestrator

orchestrator = QuickOrchestrator()

# 최신 Full 모드 결과 자동 로드
result = orchestrator.run_quick_validation()

print(result['final_validation']['final_recommendation'])  # BULLISH
print(result['final_validation']['confidence'])            # 0.72
```

### 특정 JSON 지정

```python
result = orchestrator.run_quick_validation(
    full_json_path="outputs/eimas_20260204_120000.json"
)
```

### CLI 실행

```bash
# Quick mode orchestrator 단독 실행
python -m lib.quick_agents.quick_orchestrator

# 결과 저장: outputs/quick_validation_YYYYMMDD_HHMMSS.json
```

---

## API 키 설정

### 필수 환경변수

```bash
# Claude API (PortfolioValidator, MarketSentimentAgent, FinalValidator)
export ANTHROPIC_API_KEY="sk-ant-..."

# Perplexity API (AllocationReasoner, AlternativeAssetAgent)
export PERPLEXITY_API_KEY="pplx-..."
```

### API 키 검증

```python
from lib.quick_agents import QuickOrchestrator

try:
    orchestrator = QuickOrchestrator()
    print("✓ All API keys valid")
except ValueError as e:
    print(f"✗ Missing API key: {e}")
```

---

## 경제학적 방법론

### 포트폴리오 이론

| 이론 | 연도 | 핵심 개념 | 검증 에이전트 |
|------|------|-----------|---------------|
| Markowitz MVO | 1952 | 평균-분산 최적화 | PortfolioValidator |
| Sharpe Ratio | 1966 | 위험 조정 수익률 | PortfolioValidator |
| Black-Litterman | 1992 | Bayesian 사전 믿음 + 시장 균형 | PortfolioValidator |
| Risk Parity | 2005 | 동일 리스크 기여도 | PortfolioValidator |

### 행동경제학

| 개념 | 학자 | 적용 분야 | 검증 에이전트 |
|------|------|-----------|---------------|
| Prospect Theory | Kahneman & Tversky | 손실 회피 | MarketSentimentAgent |
| Sentiment Indicators | Baker & Wurgler (2006) | Fear & Greed Index | MarketSentimentAgent |
| Herd Behavior | Shiller | 시장 버블 | MarketSentimentAgent |

### 대체자산

| 자산군 | 논문 | 핵심 결과 | 검증 에이전트 |
|--------|------|-----------|---------------|
| Gold | Baur & Lucey (2010) | 안전자산 역할 | AlternativeAssetAgent |
| Commodities | Gorton & Rouwenhorst (2006) | 인플레이션 헤지 | AlternativeAssetAgent |
| Stablecoin | Genius Act (TJ's framework) | 유동성-스테이블코인 연계 | AlternativeAssetAgent |

---

## 실행 흐름

```
[Start] QuickOrchestrator.run_quick_validation()
   |
   v
[Step 1] Load Full mode JSON (outputs/eimas_*.json)
   |
   v
[Step 2] Extract agent inputs
   |     - Market context (regime, risk, volatility, bubble)
   |     - Allocation result (stock/bond weights, Sharpe)
   |     - KOSPI data, SPX data (separate)
   |     - Crypto data, Commodity data
   |
   v
[Step 3] Run 4 specialist agents
   |
   ├─> [Agent 1] PortfolioValidator (Claude)
   |     ✓ Economic theory compliance
   |
   ├─> [Agent 2] AllocationReasoner (Perplexity)
   |     ✓ Academic research support
   |
   ├─> [Agent 3] MarketSentimentAgent (Claude)
   |     ✓ KOSPI sentiment (separate)
   |     ✓ SPX sentiment (separate)
   |     ✓ Divergence analysis
   |
   ├─> [Agent 4] AlternativeAssetAgent (Perplexity)
   |     ✓ Crypto outlook
   |     ✓ Gold/commodity role
   |     ✓ RWA tokenization trend
   |
   v
[Step 4] FinalValidator synthesizes all opinions
   |     - Agent consensus (4/4 agents agree?)
   |     - Full vs Quick comparison
   |     - Final recommendation + confidence
   |     - Risk warnings + action items
   |
   v
[Result] Quick validation JSON output
   |     - portfolio_validation: {...}
   |     - allocation_reasoning: {...}
   |     - market_sentiment: {...}
   |     - alternative_assets: {...}
   |     - final_validation: {...}
   |
   v
[Save] outputs/quick_validation_YYYYMMDD_HHMMSS.json
```

---

## 통합 예시

### main.py에서 Quick mode 실행

```python
from lib.quick_agents import QuickOrchestrator

async def run_quick_mode():
    # Full mode 실행 (Phase 1-2)
    full_result = await run_full_pipeline()

    # Full JSON 저장
    full_json_path = save_full_result(full_result)

    # Quick mode 검증
    orchestrator = QuickOrchestrator()
    quick_result = orchestrator.run_quick_validation(
        full_json_path=full_json_path
    )

    # 최종 권고 출력
    final_rec = quick_result['final_validation']['final_recommendation']
    confidence = quick_result['final_validation']['confidence']

    print(f"Final Recommendation: {final_rec} (Confidence: {confidence*100:.0f}%)")

    return quick_result
```

---

## 결과 해석

### 예시 출력

```json
{
  "timestamp": "2026-02-04T12:00:00",
  "portfolio_validation": {
    "validation_result": "PASS",
    "theory_compliance": {
      "markowitz_mvo": "compliant",
      "risk_parity": "appropriate"
    }
  },
  "allocation_reasoning": {
    "reasoning_quality": "STRONG",
    "academic_support": {
      "num_citations": 7,
      "has_recent_research": true
    }
  },
  "market_sentiment": {
    "kospi_sentiment": {"sentiment": "BULLISH", "confidence": 0.65},
    "spx_sentiment": {"sentiment": "BULLISH", "confidence": 0.70},
    "comparison": {"divergence": "ALIGNED"}
  },
  "alternative_assets": {
    "crypto_assessment": {"recommendation": "BULLISH"},
    "commodity_assessment": {"gold_role": "SAFE_HAVEN"}
  },
  "final_validation": {
    "validation_result": "PASS",
    "final_recommendation": "BULLISH",
    "confidence": 0.72,
    "agent_consensus": {"agreement_level": "HIGH"},
    "full_vs_quick_comparison": {"alignment": "ALIGNED"}
  },
  "execution_time_seconds": 45.3
}
```

### 해석 가이드

- **validation_result = PASS**: 모든 검증 통과
- **confidence ≥ 0.70**: 높은 신뢰도
- **agreement_level = HIGH**: 4개 에이전트 모두 동의
- **alignment = ALIGNED**: Full 모드와 일치

---

## 문제 해결

### API 키 오류

```python
ValueError: ANTHROPIC_API_KEY not found
```

**해결**: 환경변수 설정
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export PERPLEXITY_API_KEY="pplx-..."
```

### Full JSON 미발견

```python
ERROR: No Full mode JSON files found in outputs/
```

**해결**: Full 모드 먼저 실행
```bash
python main.py  # Full 모드 실행
python -m lib.quick_agents.quick_orchestrator  # Quick 모드
```

### Agent 시간 초과

```python
requests.exceptions.Timeout: Request timed out after 30s
```

**해결**: Perplexity API는 30초 타임아웃 (정상). 재시도하거나 네트워크 확인.

---

## 성능

| 항목 | 값 |
|------|-----|
| 전체 실행 시간 | ~30-60초 |
| API 호출 횟수 | 5회 (각 에이전트 1회) |
| Claude API 비용 | ~$0.02 (Sonnet 4, 2000 tokens/call) |
| Perplexity API 비용 | ~$0.01 (Sonar Large, 2000 tokens/call) |
| 총 예상 비용 | ~$0.03/run |

---

## 참고 문헌

### 포트폴리오 이론
- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*
- Sharpe, W. F. (1966). "Mutual Fund Performance." *Journal of Business*
- Black, F., & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*
- Qian, E. (2005). "Risk Parity Portfolios." *PanAgora Asset Management*

### 행동경제학
- Kahneman, D., & Tversky, A. (1979). "Prospect Theory." *Econometrica*
- Baker, M., & Wurgler, J. (2006). "Investor Sentiment and the Cross-Section of Stock Returns." *Journal of Finance*
- Shiller, R. J. (2000). *Irrational Exuberance*. Princeton University Press

### 대체자산
- Gorton, G., & Rouwenhorst, K. G. (2006). "Facts and Fantasies about Commodity Futures." *Financial Analysts Journal*
- Baur, D. G., & Lucey, B. M. (2010). "Is Gold a Hedge or a Safe Haven?" *Financial Analysts Journal*

---

*Last updated: 2026-02-04*
*Version: 1.0.0*
