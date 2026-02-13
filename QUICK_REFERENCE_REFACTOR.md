# EIMAS 리팩토링 빠른 참조 (2026-02-13)

## ✅ 완료된 작업

| 번호 | 작업 | 파일 | 효과 |
|------|------|------|------|
| 1 | 에러 처리 개선 | `schemas.py`, `analyzers_*.py` | 0.0 대신 error_code 반환 |
| 2 | Korea Integration | `korea_integration.py` | ValuationGap 출처 추적 |
| 3 | 리스크 설정화 | `configs/risk_adjustments.yaml` | 하드코딩 제거 |
| 4 | 동적 수익률 | `lib/expected_returns.py` | 8%/4% 하드코딩 제거 |
| 5 | 포트폴리오 디버깅 | `analyzers_advanced.py` | equal-weight fallback |
| 6 | Bubble Risk 차트 | `BubbleRiskChart.tsx` | 시각화 추가 |
| 7 | Streamlit Signals | `dashboard.py` | 5개 섹션 추가 |

---

## ⚠️ 핵심 주의사항

### 1. Portfolio 검증 시 --full 모드 필수!

```bash
# ❌ 작동 안 함 (항상 빈 결과)
python main.py --quick

# ✅ 올바름
python main.py --full
```

**이유:** `--quick`은 enhanced analytics(portfolio optimization) 스킵

### 2. 설정 파일 사용법

```python
from pipeline.risk_config import get_risk_config

cfg = get_risk_config()
cfg.pcr.high_threshold          # 1.0
cfg.crypto_fng.fear_threshold   # 25
cfg.constraints.max_adjustment  # 15
```

### 3. 검증 명령

```bash
# 빠른 검증 (API 불필요)
python -c "from pipeline.risk_config import get_risk_config; print('OK')"

# 전체 검증 (API 필요)
python main.py --full
jq '.portfolio_weights | keys | length' outputs/eimas_*.json | tail -1
```

---

## 📁 주요 파일

**새로 생성:**
- `configs/risk_adjustments.yaml` - 리스크 임계값
- `lib/expected_returns.py` - 동적 수익률 계산
- `pipeline/risk_config.py` - 설정 로더
- `frontend/components/BubbleRiskChart.tsx` - 버블 리스크 차트

**주요 수정:**
- `pipeline/schemas.py` - 에러 필드 추가
- `pipeline/analyzers_advanced.py` - 포트폴리오 fallback
- `frontend_streamlit/dashboard.py` - Signals 탭

---

## 🚀 다음 작업 (선택)

- [ ] Phase 3: 병렬 실행 (`--parallel` 플래그)
- [ ] 예상 효과: 40% 시간 단축
- [ ] 현재는 Phase 1-2만으로 충분

---

**상세 문서:** `REFACTOR_LOG_20260213.md`
