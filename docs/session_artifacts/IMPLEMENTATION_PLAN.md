# Integrated JSON vs AI Report 통합 계획

## 현재 상황

| 파일 | 생성 모듈 | 내용 | 용도 |
|------|----------|------|------|
| `integrated_*.json` | `storage.py` | EIMASResult raw dump | 데이터 저장/재사용 |
| `integrated_*.md` | `storage.py` | EIMASResult.to_markdown() | 간단 요약 |
| `ai_report_*.json` | `report.py` | FinalReport raw dump | 리포트 데이터 |
| `ai_report_*.md` | `report.py` | AI 생성 prose (Claude/GPT) | **사람 읽기용** |
| `reports/EIMAS_*.html` | `final_report_agent.py` | 최종 HTML 리포트 | 배포용 |

## 문제점
1. **중복 저장**: 같은 데이터가 3개 형태로 저장됨
2. **혼란**: 어떤 파일을 봐야 하는지 불명확
3. **비효율**: 파일 124개 누적

---

## 통합 방안

### Option A: 단일 통합 JSON + 단일 리포트 (권장)
```
outputs/
├── eimas_YYYYMMDD_HHMMSS.json   # 모든 데이터 (EIMASResult + FinalReport)
├── eimas_YYYYMMDD_HHMMSS.md    # 읽기용 마크다운
└── reports/
    └── EIMAS_*.html             # 배포용 HTML (기존 유지)
```

**변경 사항:**
1. `EIMASResult`에 `ai_report` 필드 추가
2. `storage.py`에서 통합 저장
3. `integrated_*.json` 삭제, `ai_report_*.json` 삭제

### Option B: 현행 유지 + 정리 스크립트
- 기존 구조 유지
- 오래된 파일 자동 삭제 (30일)

---

## Proposed Changes (Option A)

### 1. pipeline/schemas.py
```python
@dataclass
class EIMASResult:
    ...
    # AI Report (통합)
    ai_report: Optional[Dict] = None  # FinalReport.to_dict()
```

### 2. main.py
```python
# Phase 6에서 ai_report 저장
if generate_report:
    ai_report = await generate_ai_report(result, market_data)
    result.ai_report = ai_report.to_dict()  # 통합
```

### 3. storage.py
```python
# 파일명 변경
output_file = output_dir / f"eimas_{timestamp_str}.json"
```

---

## 결론
Option A (통합 방식) 채택됨:
- `eimas_*.json` 형식으로 통합 저장
- `ai_report` 필드를 EIMASResult에 포함
- 기존 파일은 유지 (삭제하지 않음)
