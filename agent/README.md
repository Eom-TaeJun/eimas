# Economic Insight Agent

> 인과적, 설명 가능한 경제/금융 분석 에이전트

"Asset Infinity" 조건(주식, 채권, ETF, 암호화폐, 스테이블코인, RWA) 하에서 **전달 메커니즘**과 **명시적 추론 아티팩트**를 우선시하는 분석 에이전트입니다.

## Features

- **Causality-first Analysis**: 모든 분석은 인과 그래프, 메커니즘 경로, 반증 가설을 포함
- **JSON-first Output**: Pydantic 스키마 기반 구조화된 JSON 출력
- **EIMAS Integration**: 기존 EIMAS 모듈(ShockPropagation, CriticalPath, GeniusAct 등)과 완전 통합
- **Template-based Fallback**: 데이터 없이도 경제학적 템플릿 기반 분석 가능

## Quick Start

### 1. 템플릿 기반 분석

```python
from agent import EconomicInsightOrchestrator, InsightRequest

orchestrator = EconomicInsightOrchestrator()
request = InsightRequest(
    question="스테이블코인 공급 증가가 국채 수요에 미치는 영향은?"
)
report = orchestrator.run(request)

print(report.model_dump_json(indent=2))
```

### 2. CLI 실행

```bash
# 질문 직접 입력
python -m agent.cli --question "Fed 금리 인상이 시장에 미치는 영향은?"

# JSON 파일 입력
python -m agent.cli examples/request_stablecoin.json

# EIMAS 결과 활용 (outputs/ 디렉토리의 최신 결과 사용)
python -m agent.cli --with-eimas --question "현재 시장 상황 분석"

# 파일로 출력
python -m agent.cli --question "분석 질문" --output report.json
```

### 3. EIMAS 통합 분석

```python
from agent import EconomicInsightOrchestrator, InsightRequest

orchestrator = EconomicInsightOrchestrator()

# main.py 실행 결과를 활용
eimas_results = {
    'shock_propagation': {...},  # ShockPropagationGraph 결과
    'critical_path': {...},       # CriticalPathAggregator 결과
    'genius_act': {...},          # GeniusActMacroStrategy 결과
    'bubble_detector': {...},     # BubbleDetector 결과
    'portfolio': {...},           # GraphClusteredPortfolio 결과
}

request = InsightRequest(question="Fed 정책 변화의 파급 효과는?")
report = orchestrator.run_with_eimas_results(request, eimas_results)
```

## Output JSON Structure

```json
{
  "meta": {
    "request_id": "abc123",
    "timestamp": "2026-01-28T14:00:00",
    "frame": "crypto",
    "modules_used": ["genius_act_macro", "shock_propagation_graph"]
  },
  "phenomenon": "스테이블코인 공급 증가가 국채 단기물 수요를 견인하고 있다",
  "causal_graph": {
    "nodes": [...],
    "edges": [...],
    "has_cycles": false,
    "critical_path": ["Stablecoin_Supply", "Reserve_Demand", "TBill_Demand"]
  },
  "mechanisms": [
    {
      "nodes": ["Stablecoin_Supply", "Reserve_Demand", "TBill_Demand"],
      "edge_signs": ["+", "+"],
      "net_effect": "+",
      "narrative": "스테이블코인 발행 증가 → 담보 수요 증가 → 국채 매수"
    }
  ],
  "hypotheses": {
    "main": {
      "statement": "스테이블코인 성장이 국채 수요의 새로운 동인이다",
      "confidence": "high"
    },
    "rivals": [...],
    "falsification_tests": [...]
  },
  "risk": {
    "regime_shift_risks": [...],
    "data_limitations": [...]
  },
  "suggested_data": [...],
  "next_actions": [...]
}
```

## Analysis Frames

| Frame | Keywords | Template Graph |
|-------|----------|----------------|
| `macro` | Fed, 금리, inflation, GDP | Fed → Liquidity → VIX → Assets |
| `crypto` | stablecoin, BTC, DeFi | Stablecoin → Reserve → Treasury |
| `markets` | SPY, VIX, sector | Sentiment → Flows → Prices |
| `mixed` | (복합) | Macro + Crypto 결합 |

## Eval Harness

```bash
# 전체 시나리오 실행
python -m agent.evals.runner

# 특정 시나리오만
python -m agent.evals.runner --scenario S01

# JSON 출력
python -m agent.evals.runner --json

# 상세 출력
python -m agent.evals.runner --verbose
```

### 시나리오 목록 (10개)

| ID | Name | Frame |
|----|------|-------|
| S01 | Stablecoin-Treasury Channel | crypto |
| S02 | Fed Rate Policy Impact | macro |
| S03 | Liquidity Transmission Mechanism | macro |
| S04 | Crypto-Macro Correlation | mixed |
| S05 | Sector Rotation Analysis | markets |
| S06 | DeFi TVL and ETH | crypto |
| S07 | VIX Risk Transmission | markets |
| S08 | RRP Liquidity Drain | macro |
| S09 | Credit Spread Widening | markets |
| S10 | Full Macro-Crypto Integration | mixed |

## Unit Tests

```bash
# 전체 테스트
pytest agent/tests/ -v

# 스키마 테스트만
pytest agent/tests/test_schemas.py -v

# 그래프 유틸리티 테스트
pytest agent/tests/test_graph_utilities.py -v

# 오케스트레이터 통합 테스트
pytest agent/tests/test_orchestrator.py -v
```

## Directory Structure

```
agent/
├── __init__.py              # Main exports
├── cli.py                   # CLI interface
├── README.md                # This file
├── core/
│   ├── __init__.py
│   ├── adapters.py          # EIMAS → Schema adapters
│   └── orchestrator.py      # Main orchestrator
├── schemas/
│   ├── __init__.py
│   └── insight_schema.py    # Pydantic schemas
├── examples/
│   ├── request_stablecoin.json
│   ├── request_fed_policy.json
│   ├── request_market_rotation.json
│   └── request_mixed.json
├── evals/
│   ├── __init__.py
│   ├── scenarios.py         # Eval scenarios
│   └── runner.py            # Eval runner
└── tests/
    ├── __init__.py
    ├── test_schemas.py      # Schema validity tests
    ├── test_graph_utilities.py  # Graph algorithm tests
    └── test_orchestrator.py # Integration tests
```

## Economic Methodology

### Causal Graph

- **Nodes**: 경제 변수/자산/프로토콜
- **Edges**: 부호가 있는 인과관계 (+/-/?)
- **Lag**: 시차 효과 (일 단위)
- **Confidence**: p-value 기반 신뢰도 (HIGH/MEDIUM/LOW)

### Mechanism Path

- **Sign Composition**: `+ * + = +`, `+ * - = -`, `- * - = +`
- **Net Effect**: 경로 전체의 최종 효과 방향
- **Bottleneck**: 병목 노드 식별

### Hypotheses

- **Main Hypothesis**: 주요 가설 (가장 강한 메커니즘 기반)
- **Rivals**: 대안 가설들
- **Falsification Tests**: 반증 테스트 (어떤 데이터가 가설을 기각하는지)

## Integration with EIMAS Modules

| EIMAS Module | Adapter Method | Output |
|--------------|----------------|--------|
| ShockPropagationGraph | `adapt_shock_propagation()` | CausalGraph |
| CriticalPathAggregator | `adapt_critical_path()` | RegimeShiftRisk[] |
| GeniusActMacroStrategy | `adapt_genius_act()` | MechanismPath[] |
| BubbleDetector | `adapt_bubble_detector()` | RegimeShiftRisk[] |
| GraphClusteredPortfolio | `adapt_portfolio()` | NextAction[] |
| VolumeAnalyzer | `adapt_volume_analyzer()` | SuggestedDataset[], DataLimitation[] |

## Dependencies

- Python 3.10+
- pydantic >= 2.0
- (EIMAS 모듈 사용 시) yfinance, pandas, numpy

---

*Last updated: 2026-01-28*
