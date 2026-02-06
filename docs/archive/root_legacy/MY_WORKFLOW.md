# 내 작업 플로우

## 표준 테스트 프로세스

### 1. 전체 실행
```bash
python main.py --full
```

### 2. 결과 확인
- 결과가 JSON 형식으로 출력됨
- 위치: `outputs/eimas_YYYYMMDD_HHMMSS.json`

### 3. 변환

#### 3.1 JSON → Markdown (스키마 기반)
```bash
python lib/json_to_md_converter.py
```

**특징:**
- 스키마 기반 렌더러 (v2.0)
- 자동 테이블 생성
- 51개 섹션 + 21개 추가 필드
- 검증 기능 내장
- 커버리지: **65.1%** ✅

**출력 통계:**
- JSON size: 96KB
- MD size: 62KB
- Sections: 51개
- Extra fields: 21개

#### 3.2 JSON → HTML (직접 변환) ⭐ 권장
```bash
python lib/json_to_html_converter.py
```

**특징:**
- MD 거치지 않고 JSON에서 직접 변환
- 정보 손실 최소화
- 완벽한 테이블 렌더링
- 다크 테마 + 반응형 디자인
- 커버리지: **88.5%** ✅

**출력 통계:**
- JSON size: 96KB
- HTML size: 85KB
- Sections: 51개

**출력:**
- 위치: `outputs/eimas_YYYYMMDD_HHMMSS.html`
- GitHub 스타일 다크 테마
- 자동 테이블, 배지, 컬러 코딩

### 4. 검증
- 변환된 MD와 HTML 파일을 확인
- 원하는 기능이 제대로 작동했는지 판단

## 주요 체크포인트
- [ ] JSON 출력 완료 (outputs/ 폴더)
- [ ] MD 변환 성공 (41%+ 커버리지)
- [ ] HTML 변환 성공 (다크 테마)
- [ ] 검증 통과 (stance 일치)
- [ ] 기능 동작 확인

## 렌더러 아키텍처 (v2.0)

```
raw_json
  → normalize(raw_json)
  → ReportModel (sections + extra_fields)
  → render_md(ReportModel)
  → markdown
  → validate(json, md)
```

### 스키마 확장 방법

새 섹션 추가 시 `lib/json_to_md_converter.py`의 `SECTION_SCHEMA`에 추가:

```python
SECTION_SCHEMA = {
    "new_section_key": {
        "priority": 25,  # 렌더링 순서
        "title": "🆕 새 섹션",
        "icon": "🆕"
    },
    # ...
}
```

## 참고사항
- 항상 `--full` 옵션을 사용하여 전체 파이프라인 실행
- 스키마 기반이므로 JSON 구조 변경에 견고함
- Unknown 필드는 자동으로 "Additional Fields"에 포함
- 모든 변환 결과물을 기반으로 종합적으로 판단
