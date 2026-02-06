[WORK_ORDER]
id: GEN-000
goal: (한 줄 목표)

context:
- (왜 필요한지 2~4줄)

scope_files:
- path/a.py
- path/b.md

tasks:
1. (정확한 수정/이동/삭제 지시)
2. (정확한 수정/이동/삭제 지시)

constraints:
- 구조 결정 금지 (설계 변경 금지)
- 지시 범위 외 파일 수정 금지
- fallback/예외 처리 패턴 유지

validation:
- python3 -m py_compile ...
- python3 -c "import ..."
- (필요한 smoke test)

deliverables:
- 변경 파일 목록
- 핵심 diff 요약
- validation 결과
[/WORK_ORDER]
