# Archived Pipeline

이 디렉토리에는 현재 활성 파이프라인에서 사용하지 않는 파이프라인 모듈이 있습니다.

## 보관된 파일

| 파일 | 설명 |
|------|------|
| `full_pipeline.py` | 전체 8단계 파이프라인 (TopDownOrchestrator, MethodologyDebateAgent 등 사용) |

## 복구 방법

```python
from pipeline.archive.full_pipeline import FullPipelineRunner, PipelineConfig
```

## 보관 이유

- `full_pipeline.py`는 archive된 에이전트들(TopDownOrchestrator, MethodologyDebateAgent 등)에 의존
- 현재 활성 파이프라인은 `main_integrated.py`로, `MetaOrchestrator`만 사용
- API의 `/analyze` 엔드포인트도 함께 비활성화됨

## 관련 Archive

- `agents/archive/` - full_pipeline.py가 사용하던 에이전트들
- `core/archive/` - DebateFramework

## Git 롤백

```bash
git checkout v2.1.2-pre-cleanup
```
