# Archived Agents

이 디렉토리에는 현재 활성 파이프라인(main_integrated.py)에서 사용하지 않는 에이전트들이 포함되어 있습니다.

## 보관된 에이전트 목록

| 파일 | 클래스 | 원래 용도 |
|------|--------|----------|
| `top_down_orchestrator.py` | TopDownOrchestrator | 계층적 하향식 분석 (Geo→Monetary→Asset→Sector→Individual) |
| `research_agent.py` | ResearchAgent | Perplexity API 기반 리서치 |
| `strategy_agent.py` | StrategyAgent | 포트폴리오 전략 권고 |
| `verification_agent.py` | VerificationAgent | AI 출력 검증 (Hallucination/Sycophancy) |
| `visualization_agent.py` | VisualizationAgent | 대시보드 시각화 |
| `methodology_debate.py` | MethodologyDebateAgent | 방법론 선택 토론 |
| `interpretation_debate.py` | InterpretationDebateAgent | 경제학파별 해석 토론 |
| `regime_change.py` | RegimeChangeDetectionPipeline | 레짐 변화 탐지 파이프라인 |

## 복구 방법

Archive에서 에이전트를 사용하려면 직접 import하세요:

```python
# 예시: TopDownOrchestrator 복구
from agents.archive.top_down_orchestrator import TopDownOrchestrator, TopDownResult

# 예시: ResearchAgent 복구
from agents.archive.research_agent import ResearchAgent, ResearchReport

# 예시: MethodologyDebateAgent 복구
from agents.archive.methodology_debate import MethodologyDebateAgent, MethodologyType

# 예시: InterpretationDebateAgent 복구
from agents.archive.interpretation_debate import InterpretationDebateAgent, EconomicSchool

# 예시: StrategyAgent 복구
from agents.archive.strategy_agent import StrategyAgent, PortfolioStrategy
```

## 경로 의존성 (2026-01-19 해결됨)

Archive 파일들의 import 경로가 수정되어 독립적으로 작동합니다:
- `core.debate_framework` → `core.archive.debate_framework`
- `.base_agent` → `agents.base_agent`

## 보관 이유

- 2026-01-19 기준, `main_integrated.py`는 `MetaOrchestrator`만 사용
- 이 에이전트들은 `pipeline/full_pipeline.py`에서 사용되었으나, 해당 파이프라인도 archive됨
- 코드 정리를 위해 보관하되, 향후 필요시 복구 가능하도록 유지

## 관련 Archive

- `pipeline/archive/` - full_pipeline.py (이 에이전트들의 주요 사용처)
- `core/archive/` - debate_framework.py (Multi-AI 토론 프레임워크)

## Git 롤백

전체 복구가 필요한 경우:
```bash
git checkout v2.1.2-pre-cleanup
```
