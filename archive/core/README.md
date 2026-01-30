# Archived Core Modules

이 디렉토리에는 현재 활성 파이프라인에서 사용하지 않는 핵심 모듈이 있습니다.

## 보관된 파일

| 파일 | 클래스 | 설명 |
|------|--------|------|
| `debate_framework.py` | DebateFramework | Multi-AI LLM 호출 기반 토론 프레임워크 |

## 복구 방법

```python
from core.archive.debate_framework import DebateFramework
```

## 활성 모듈 vs Archive

| 모듈 | 위치 | 특징 | 사용처 |
|------|------|------|--------|
| `DebateProtocol` | core/debate.py | Rule-based, LLM 없음, 빠름 | MetaOrchestrator |
| `DebateFramework` | core/archive/ | Multi-AI LLM 호출, 느림 | archive된 에이전트들 |

## 관련 Archive

- `agents/archive/` - DebateFramework를 사용하던 에이전트들
- `pipeline/archive/` - full_pipeline.py

## Git 롤백

```bash
git checkout v2.1.2-pre-cleanup
```
